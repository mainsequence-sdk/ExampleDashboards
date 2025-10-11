# dashboards/apps/floating_portfolio_analysis/pages/01_curve_and_positions.py
from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List

import streamlit as st
import QuantLib as ql

from dashboards.components.portfolio_select import sidebar_portfolio_multi_select
from dashboards.services.portfolios import PortfoliosOperations
from dashboards.services.positions import PositionOperations
from dashboards.components.date_selector import date_selector
from dashboards.core.ql import qld
import mainsequence.client as msc
import mainsequence.instruments as msi

from dashboards.core.data_nodes import get_app_data_nodes
from dashboards.components.curve_bumps_and_stats import st_curve_bumps
from mainsequence.dashboards.streamlit.scaffold import PageConfig, run_page
from dashboards.helpers.mock import build_test_portfolio as _build_test_portfolio


# Page setup
ctx = run_page(PageConfig(
    title="Curve, Stats & Positions",
    use_wide_layout=True,
    inject_theme_css=True,
))
# Ensure "render once" counter is reset on page run
st.session_state.pop("__npv_tables_rendered", None)


# --- Mock portfolio build: run ONLY on explicit click -----------------------
def _request_mock_build() -> None:
    """Set a flag; we will act on it later in this run."""
    st.session_state["_req_build_mock_portfolio"] = True

def _do_build_mock_if_requested() -> None:
    """Execute the build only if explicitly requested via the flag."""
    if not st.session_state.pop("_req_build_mock_portfolio", False):
        return
    try:
        # Lazy import to avoid any import-time side effects.
        from dashboards.helpers.mock import build_test_portfolio as _build_test_portfolio
    except Exception:
        st.error("`dashboards.helpers.mock.build_test_portfolio` is not available in this environment.")
        return
    try:
        with st.spinner("Building mock portfolio 'mock_portfolio_floating_dashboard'…"):
            _build_test_portfolio("mock_portfolio_floating_dashboard")
        st.session_state["__just_built_mock_portfolio__"] = True
        st.rerun()
    except Exception as e:
        st.exception(e)

# --- Minimal data-nodes bootstrap (needed by portfolios.py) ---
def _ensure_data_nodes() -> None:
    if st.session_state.get("_deps_bootstrapped"):
        return
    deps = get_app_data_nodes()
    try:
        deps.get("instrument_pricing_table_id")
    except KeyError:
        from dashboards.apps.bond_portfolio_analysis.settings import PRICES_TABLE_NAME
        deps.register(instrument_pricing_table_id=PRICES_TABLE_NAME)
    st.session_state["_deps_bootstrapped"] = True

_ensure_data_nodes()

def _get_cfg_from_session(ss) -> Dict[str, Any]:
    return dict(ss.get("position_cfg_mem") or {})

# ---------- Sidebar ----------
with st.sidebar:
    _ = date_selector(
        label="Valuation date",
        session_cfg_key="position_cfg_mem",
        cfg_field="valuation_date",
        key="valuation_date_input",
        help="Controls pricing date and curve construction.",
    )

    notional_value = st.number_input(
        "Portfolio notional (base currency)",
        min_value=1.0,
        step=100_000.0,
        value=float(st.session_state.get("portfolio_notional", 1_000_000.0)),
        format="%.0f",
        help="Weights will be translated into integer holdings using this notional.",
        key="portfolio_notional",
    )

    st.divider()

    st.button("Build mock portfolios", key="btn_build_mock_portfolio", use_container_width=True,
              on_click=_request_mock_build)
    _do_build_mock_if_requested()

    selected_instances = sidebar_portfolio_multi_select(
        title="Portfolios (search & multi-select)",
        key_prefix="fpa_portfolios",
        min_chars=3,
    )

    if selected_instances:
        selected_ports = [ia.reference_portfolio for ia in selected_instances]
        selected_ids = tuple(sorted(int(getattr(p, "id", p)) for p in selected_ports))
        last_ids = tuple(st.session_state.get("_active_portfolio_ids", ()))
        last_notional = float(st.session_state.get("_active_notional", 0.0) or 0.0)

        if (selected_ids != last_ids) or (float(notional_value) != last_notional):
            # Build all position templates ONCE for the current selection
            po_all = PortfoliosOperations(portfolio_list=selected_ports)
            with st.spinner("Building positions from selected portfolio(s)…"):
                ih2a_combined, positions_by_id = po_all.get_all_portfolios_as_positions(
                    portfolio_notional=float(notional_value)
                )
            if not positions_by_id:
                st.error("No positions returned from portfolio builder.")
                st.stop()

            # Persist required session state
            prev_cfg = _get_cfg_from_session(st.session_state)
            st.session_state["position_cfg_mem"] = {
                **prev_cfg,
                "portfolio_notional": float(notional_value),
            }
            st.session_state["instrument_hash_to_asset"] = ih2a_combined
            st.session_state["position_templates_mem"] = positions_by_id            # {pid -> Position template}
            st.session_state["_active_portfolio_ids"] = selected_ids
            st.session_state["_active_notional"] = float(notional_value)
            st.session_state["_ports_by_id"] = {int(p.id): p for p in selected_ports}

            # Build a merged template to drive a single curve control
            merged_lines = []
            for pos in positions_by_id.values():
                merged_lines.extend(list(pos.lines or []))
            st.session_state["_merged_template"] = msi.Position(lines=merged_lines)

            st.rerun()

# ---------- Main content ----------
cfg = _get_cfg_from_session(st.session_state)
iso_date = cfg.get("valuation_date")
if not iso_date:
    st.info("Pick a valuation date and load one or more portfolios.")
    st.stop()

valuation_date = dt.date.fromisoformat(iso_date)
ql.Settings.instance().evaluationDate = qld(valuation_date)

positions_by_id: Dict[int, msi.Position] = st.session_state.get("position_templates_mem") or {}
if not positions_by_id:
    st.info("Select one or more portfolios in the sidebar to build positions.")
    st.stop()

# 1) Single, shared curve control + chart for ALL (built from merged template)
merged_template = st.session_state.get("_merged_template")
bundle_curve = st_curve_bumps(
    position_template=merged_template,
    valuation_date=valuation_date,
    key_prefix="curve_bundle_all",
)
if not bundle_curve:
    st.stop()

base_curves = bundle_curve["base_curves"]
bumped_curves = bundle_curve["bumped_curves"]

# 2) Tabs — one per portfolio; each with stats + positions table driven by the SAME curves
ordered_ids = list(st.session_state.get("_active_portfolio_ids", positions_by_id.keys()))
id_to_name = {}
ports_by_id = st.session_state.get("_ports_by_id") or {}
for pid in ordered_ids:
    p = ports_by_id.get(int(pid))
    id_to_name[int(pid)] = getattr(p, "portfolio_name", getattr(p, "name", f"Portfolio {pid}")) if p else f"Portfolio {pid}"

tabs = st.tabs([id_to_name.get(int(pid), f"Portfolio {pid}") for pid in ordered_ids])

for pid, tab in zip(ordered_ids, tabs):
    pid = int(pid)
    pos_template = positions_by_id.get(pid)
    if pos_template is None:
        continue

    # Per-portfolio position operations using the SAME curves
    ops = PositionOperations.from_position(
        pos_template,
        base_curves_by_index=base_curves,
        valuation_date=valuation_date,
    )
    base_pos = ops.instantiate_base()
    ops.set_curves(bumped_curves_by_index=bumped_curves)
    bumped_pos = ops.instantiate_bumped()
    ops.compute_and_apply_z_spreads_from_dirty_price(
        base_position=base_pos,
        bumped_position=bumped_pos,
    )

    # Table + stats using a per-portfolio PortfoliosOperations instance (name resolution)
    one_po = PortfoliosOperations(portfolio_list=[ports_by_id.get(pid)]) if ports_by_id.get(pid) else None
    if one_po is None:
        continue
    try:
        one_po.set_portfolio_positions({pid: base_pos})
    except Exception:
        pass

    with tab:
        st.caption(f"Valuation date: {iso_date}")
        one_po.st_position_npv_table_paginated(
            position=base_pos,
            instrument_hash_to_asset=st.session_state.get("instrument_hash_to_asset"),
            bumped_position=bumped_pos,
            page_size_options=(25, 50, 100, 200),
            default_size=50,
            enable_search=True,
            key=f"npv_table_{pid}_{iso_date}",
            render_once=False,                 # table per tab
            valuation_date=valuation_date,     # show stats in the same block
            carry_days=st.session_state.get("carry_cutoff_days", 365),
        )
