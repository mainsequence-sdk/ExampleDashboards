# dashboards/apps/floating_portfolio_analysis/pages/01_Curve_Stats_Positions.py
from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List, Tuple

import streamlit as st
import QuantLib as ql

from dashboards.components.curve_bump import curve_bump_controls_ex
from dashboards.services.curves import (
    KEYRATE_GRID_BY_FAMILY, curve_family_key, BumpSpec, keyrate_grid_for_index, build_curves_for_ui
)
from dashboards.components.position_yield_overlay import st_position_yield_overlay
from dashboards.plots.curves import plot_par_yield_curves_by_family
from dashboards.core.formatters import fmt_ccy
from dashboards.components.portfolio_select import sidebar_portfolio_multi_select
from dashboards.services.portfolios import PortfoliosOperations
from dashboards.services.positions import PositionOperations
from dashboards.components.date_selector import date_selector
from dashboards.core.ql import qld
import mainsequence.client as msc
from dashboards.core.data_nodes import get_app_data_nodes
from dashboards.helpers.mock import build_test_portfolio as _build_test_portfolio



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
        from dashboards.apps.floating_portfolio_analysis.settings import PRICES_TABLE_NAME
        deps.register(instrument_pricing_table_id=PRICES_TABLE_NAME)

    st.session_state["_deps_bootstrapped"] = True
_ensure_data_nodes()
# === Small helper context for this page (kept local for simplicity) ==========
def _get_cfg_from_session(ss) -> Dict[str, Any]:
    """User config persisted between reruns."""
    return dict(ss.get("position_cfg_mem") or {})

def _indices_from_template(template) -> List[str]:
    idxs = {
        getattr(ln.instrument, "floating_rate_index_name", None)
        for ln in (template.lines or [])
    }
    return sorted([i for i in idxs if i])

def _build_curve_maps(valuation_date: dt.date,
                      indices: List[str],
                      family_bumps: Dict[str, Dict[str, float]],
                     ) -> Tuple[Dict[str, ql.YieldTermStructureHandle], Dict[str, ql.YieldTermStructureHandle]]:
    base: Dict[str, ql.YieldTermStructureHandle] = {}
    bumped: Dict[str, ql.YieldTermStructureHandle] = {}
    for index_uid in indices:
        fam = curve_family_key(index_uid)
        fam_spec = family_bumps.get(fam, {}) or {}
        keyrate_bp = dict(fam_spec.get("keyrate_bp", {}))
        parallel_bp = float(fam_spec.get("parallel_bp", 0.0))
        grid_for_index = keyrate_grid_for_index(index_uid)
        spec = BumpSpec(keyrate_bp=keyrate_bp, parallel_bp=parallel_bp,
                        key_rate_grid={index_uid: tuple(grid_for_index)})
        ts_base, ts_bump, _, _ = build_curves_for_ui(qld(valuation_date), spec, index_identifier=index_uid)
        base[index_uid] = ts_base
        bumped[index_uid] = ts_bump
    return base, bumped


# ============================== PAGE =========================================
st.title("Curve, Stats & Positions")

# Surface a one-time success notice after mock build + rerun
if st.session_state.pop("__just_built_mock_portfolio__", False):
    st.success(
        "Mock portfolio **'mock_portfolio_floating_dashboard'** was created. "
        "Use the sidebar search to load it."
    )


# ---------- Sidebar ----------
with st.sidebar:
    # Valuation date — persisted in session; never silently set to 'today'
    _ = date_selector(
        label="Valuation date",
        session_cfg_key="position_cfg_mem",
        cfg_field="valuation_date",
        key="valuation_date_input",
        help="Controls pricing date and curve construction.",
    )

    # Portfolio notional
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



    # Portfolio picker (multi-select)
    selected_instances = sidebar_portfolio_multi_select(
        title="Build position from portfolio (search)",
        key_prefix="fpa_portfolios",
        min_chars=3,
    )

    # If selection exists, (re)build a position and activate it.
    if selected_instances:
        selected_portfolios = [ia.reference_portfolio for ia in selected_instances]
        active_port = selected_portfolios[0]
        portfolio_operations = PortfoliosOperations(portfolio_list=[active_port])

        active_port_id = getattr(active_port, "id", None)
        last_loaded_id = st.session_state.get("_active_portfolio_id")
        last_loaded_notional = st.session_state.get("_active_notional")

        # Only rebuild if portfolio or notional changed
        if active_port_id != last_loaded_id or float(notional_value) != float(last_loaded_notional or 0):
            try:
                with st.spinner("Building position from selected portfolio(s)…"):
                    instrument_hash_to_asset, portfolio_positions = portfolio_operations.get_all_portfolios_as_positions(
                        portfolio_notional=float(notional_value),
                    )
                # Preserve existing cfg (incl. valuation_date set by date_selector)
                prev_cfg = _get_cfg_from_session(st.session_state)
                st.session_state["position_cfg_mem"] = {
                    **prev_cfg,
                    "portfolio_notional": float(notional_value),
                }

                # Pick the first position as active
                if not portfolio_positions:
                    st.error("No positions returned from portfolio builder.")
                    st.stop()
                active_position = list(portfolio_positions.values())[0]

                # Filter the hash->asset map to hashes in the active position (safer)
                active_hashes = {ln.instrument.content_hash() for ln in active_position.lines}
                ih2a_active = {h: a for h, a in (instrument_hash_to_asset or {}).items() if h in active_hashes}

                # Persist state
                st.session_state["position_template_mem"] = active_position
                st.session_state["instrument_hash_to_asset"] = ih2a_active
                st.session_state["_active_portfolio_id"] = active_port_id
                st.session_state["_active_notional"] = float(notional_value)

                st.rerun()
            except Exception as e:
                raise e

        # Curve bump controls (per family)
        template = st.session_state.get("position_template_mem")
        if template is not None:
            present_indices = _indices_from_template(template)
            present_families = sorted({curve_family_key(i) for i in present_indices})
            st.markdown("### Curve bumps (by family)")
            prev = (st.session_state.get("curve_bump_spec_by_family")
                    or st.session_state.get("curve_bump_spec_by_index", {}))
            new_map: Dict[str, Dict[str, float]] = {}
            for fam in present_families:
                tenors = KEYRATE_GRID_BY_FAMILY[fam]
                exp = st.expander(f"{fam} — bumps", expanded=(len(present_families) == 1))
                spec, _ = curve_bump_controls_ex(
                    available_tenors=tenors,
                    default_bumps=(prev.get(fam, {}).get("keyrate_bp", {})),
                    default_parallel_bp=float(prev.get(fam, {}).get("parallel_bp", 0.0)),
                    header=f"{fam} bumps (bp)",
                    container=exp,
                    key=f"curve_bump_{fam}",
                )
                new_map[fam] = {"keyrate_bp": spec.keyrate_bp, "parallel_bp": float(spec.parallel_bp)}
            if new_map != prev:
                st.session_state["curve_bump_spec_by_family"] = new_map
                st.session_state["curve_bump_spec_by_index"] = new_map
                st.rerun()

# ---------- Main content ----------
cfg = _get_cfg_from_session(st.session_state)
iso_date = cfg.get("valuation_date")
if not iso_date:
    st.info("Pick a valuation date and load a portfolio to see curves & positions.")
    st.stop()

valuation_date = dt.date.fromisoformat(iso_date)
ql.Settings.instance().evaluationDate = qld(valuation_date)

template = st.session_state.get("position_template_mem")
if template is None:
    st.info("Select a portfolio in the sidebar to build the position.")
    st.stop()

# Curves
family_bumps = st.session_state.get("curve_bump_spec_by_family") or {}
indices = _indices_from_template(template)
base_curves, bumped_curves = _build_curve_maps(valuation_date, indices, family_bumps)

# Positions
ops = PositionOperations.from_template(template, base_curves_by_index=base_curves, valuation_date=valuation_date)
position = ops.instantiate_base()
ops.set_curves(bumped_curves_by_index=bumped_curves)
bumped_position = ops.instantiate_bumped()
ops.compute_and_apply_z_spreads_from_dirty_price(base_position=position, bumped_position=bumped_position)

# Curve + overlays
curve_slot = st.container()
overlay_traces = st_position_yield_overlay(
    position=position,
    valuation_date=valuation_date,   # ensure your overlay signature matches; use val_date=... if your function expects that name
    key="base_curve_position_overlay",
)
fig_curve = plot_par_yield_curves_by_family(
    base_curves=base_curves,
    bumped_curves=bumped_curves,
    max_years=30,
    step_months=3,
    title="Par yield curves — Base vs Bumped (families)"
)
for tr in overlay_traces:
    fig_curve.add_trace(tr)
with curve_slot:
    st.subheader("Par yield curve — Base vs Bumped")
    st.plotly_chart(fig_curve, width="stretch", key="par_curve_main")

# Stats
st.subheader("Portfolio statistics — Base vs Bumped")
default_carry_days = int(st.session_state.get("carry_cutoff_days", 365))
carry_days = st.slider(
    "Carry cutoff (days from valuation date)",
    min_value=30,
    max_value=1460,
    value=default_carry_days,
    step=30,
    key="carry_cutoff_days",
)
cutoff = valuation_date + dt.timedelta(days=carry_days)

stats = PositionOperations.portfolio_style_stats(
    base_position=position,
    bumped_position=bumped_position,
    valuation_date=valuation_date,
    cutoff=cutoff,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("NPV (base)", fmt_ccy(stats["npv_base"]), delta=fmt_ccy(stats["npv_delta"]))
c2.metric("NPV (bumped)", fmt_ccy(stats["npv_bumped"]))
c3.metric(f"Carry to {cutoff.isoformat()} (base)", fmt_ccy(stats["carry_base"]), delta=fmt_ccy(stats["carry_delta"]))
c4.metric(f"Carry to {cutoff.isoformat()} (bumped)", fmt_ccy(stats["carry_bumped"]))

# Positions table
st.subheader("Positions — NPV (paginated)")
po = st.session_state.get("_portfolio_operations")
if po is None:
    active_id = st.session_state.get("_active_portfolio_id")
    if active_id is not None:
        active_port = msc.PortfolioIndexAsset.get_or_none(id=int(active_id))
        if active_port is not None:
            po = PortfoliosOperations(portfolio_list=[active_port])
            st.session_state["_portfolio_operations"] = po


po.st_position_npv_table_paginated(
    position=position,
    instrument_hash_to_asset=st.session_state.get("instrument_hash_to_asset"),
    bumped_position=bumped_position,
    page_size_options=(25, 50, 100, 200),
    default_size=50,
    enable_search=True,
    key="npv_table_curve_page",
)
