# dashboards/apps/floating_portfolio_analysis/pages/01_Curve_Stats_Positions.py
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
from dashboards.helpers.mock import build_test_portfolio as _build_test_portfolio



from dashboards.components.curve_bumps_and_stats import st_curve_bumps_curve_and_stats

from mainsequence.dashboards.streamlit.scaffold import PageConfig, run_page
ctx = run_page(PageConfig(
    title="Curve, Stats & Positions",
    use_wide_layout=True,
    inject_theme_css=True,
))

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
# --- Minimal data-nodes bootstrap (needed by portfolios.py) ------------------
def _ensure_data_nodes() -> None:
    """Register the external dependency names this app expects (once per session)."""
    if st.session_state.get("_deps_bootstrapped"):
        return
    deps = get_app_data_nodes()
    # Register only if missing (avoid duplicate registration errors)
    try:
        deps.get("instrument_pricing_table_id")
    except KeyError:
        # very important step so the prices cna be extracted from the right storage
        from dashboards.apps.floating_bond_portfolio_analysis.settings import PRICES_TABLE_NAME
        deps.register(instrument_pricing_table_id=PRICES_TABLE_NAME)

    st.session_state["_deps_bootstrapped"] = True
_ensure_data_nodes()


def _get_cfg_from_session(ss) -> Dict[str, Any]:
    return dict(ss.get("position_cfg_mem") or {})


def _indices_from_position(position: msi.Position) -> List[str]:
    idxs = {
        getattr(ln.instrument, "floating_rate_index_name", None)
        for ln in (position.lines or [])
    }
    return sorted([i for i in idxs if i])


# ---------- Sidebar (kept: date, notional, portfolio picker) ----------
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
    # Quick action: build a mock portfolio for demo/testing
    st.divider()

    st.button(
        "Build mock portfolio",
        key="btn_build_mock_portfolio",
        use_container_width=True,
        on_click=_request_mock_build,  # only sets a flag
    )
    # Perform the build strictly when requested by the button callback:
    _do_build_mock_if_requested()

    selected_instances = sidebar_portfolio_multi_select(
        title="Build position from portfolio (search)",
        key_prefix="fpa_portfolios",
        min_chars=3,
    )

    if selected_instances:
        selected_portfolios = [ia.reference_portfolio for ia in selected_instances]
        active_port = selected_portfolios[0]
        portfolio_operations = PortfoliosOperations(portfolio_list=[active_port])

        active_port_id = getattr(active_port, "id", None)
        last_loaded_id = st.session_state.get("_active_portfolio_id")
        last_loaded_notional = st.session_state.get("_active_notional")

        if active_port_id != last_loaded_id or float(notional_value) != float(last_loaded_notional or 0):
            with st.spinner("Building position from selected portfolio(s)…"):
                instrument_hash_to_asset, portfolio_positions = portfolio_operations.get_all_portfolios_as_positions(
                    portfolio_notional=float(notional_value),
                )

            prev_cfg = _get_cfg_from_session(st.session_state)
            st.session_state["position_cfg_mem"] = {
                **prev_cfg,
                "portfolio_notional": float(notional_value),
            }

            if not portfolio_positions:
                st.error("No positions returned from portfolio builder.")
                st.stop()

            active_position = list(portfolio_positions.values())[0]
            active_hashes = {ln.instrument.content_hash() for ln in active_position.lines}
            ih2a_active = {h: a for h, a in (instrument_hash_to_asset or {}).items() if h in active_hashes}

            st.session_state["position_template_mem"] = active_position
            st.session_state["instrument_hash_to_asset"] = ih2a_active
            st.session_state["_active_portfolio_id"] = active_port_id
            st.session_state["_active_notional"] = float(notional_value)
            st.session_state["_portfolio_operations"] = portfolio_operations

            st.rerun()


# ---------- Main content ----------
cfg = _get_cfg_from_session(st.session_state)
iso_date = cfg.get("valuation_date")
if not iso_date:
    st.info("Pick a valuation date and load a portfolio to see curves & positions.")
    st.stop()

valuation_date = dt.date.fromisoformat(iso_date)
ql.Settings.instance().evaluationDate = qld(valuation_date)

position_template = st.session_state.get("position_template_mem")
if position_template is None:
    st.info("Select a portfolio in the sidebar to build the position.")
    st.stop()

# ✅ Single component renders the curve, the bump controllers (below the curve),
#    and the collapsible Portfolio Statistics — all together.
bundle = st_curve_bumps_curve_and_stats(
    position_template=position_template,
    valuation_date=valuation_date,
    key_prefix="curve_bundle",
)

# ---------- Positions table (unchanged; now uses positions returned by component) ----------
st.subheader("Positions — NPV (paginated)")
po = st.session_state.get("_portfolio_operations")
if po is None:
    active_id = st.session_state.get("_active_portfolio_id")
    if active_id is not None:
        active_port = msc.PortfolioIndexAsset.get_or_none(id=int(active_id))
        if active_port is not None:
            po = PortfoliosOperations(portfolio_list=[active_port])
            st.session_state["_portfolio_operations"] = po

if po is not None and bundle:
    tbl_key = f"npv_table_curve_page_{st.session_state.get('_active_portfolio_id', 'na')}_{iso_date}"

    po.st_position_npv_table_paginated(
        position=bundle["position"],
        instrument_hash_to_asset=st.session_state.get("instrument_hash_to_asset"),
        bumped_position=bundle["bumped_position"],
        page_size_options=(25, 50, 100, 200),
        default_size=50,
        enable_search=True,
        key=tbl_key,
    )
else:
    st.info("Select or rebuild a portfolio to display the positions table.")
