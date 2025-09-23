from __future__ import annotations

import datetime
from typing import Dict, Optional

import streamlit as st
from mainsequence.dashboards.streamlit.core.registry import register_page
from mainsequence.instruments.utils import to_py_date



from dashboards.components.curve_bump import curve_bump_controls_ex
from dashboards.services.curves import KEYRATE_GRID_BY_FAMILY, curve_family_key
from dashboards.components.position_yield_overlay import st_position_yield_overlay
from dashboards.plots.curves import plot_par_yield_curves_by_family
from dashboards.core.formatters import fmt_ccy
from dashboards.components.portfolio_select import sidebar_portfolio_multi_select
from dashboards.services.portfolios import PortfoliosOperations
from dashboards.services.positions import PositionOperations
from dashboards.components.date_selector import date_selector
import mainsequence.client as msc

@register_page("curve_positions", "Curve, Stats & Positions", has_sidebar=True, order=0)
def render(ctx: Optional["AppContext"]):
    # ---------- Sidebar ----------
    with st.sidebar:

        # --- Valuation date (re-usable component) ---
        _ = date_selector(
            label="Valuation date",
            session_cfg_key="position_cfg_mem",
            cfg_field="valuation_date",
            key="valuation_date_input",
            help="Controls pricing date and curve construction.",
        )

        # --- Portfolio notional input (used when building from live portfolio) ---
        notional_value = st.number_input(
            "Portfolio notional (base currency)",
            min_value=1.0,
            step=100_000.0,
            value=float(st.session_state.get("portfolio_notional", 1_000_000.0)),
            format="%.0f",
            help="Weights will be translated into integer holdings using this notional.",
            key="portfolio_notional",
        )

        # --- Portfolio picker (multi-select) ---
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
            # Persist the instance so we can use it later outside this block
            st.session_state["_portfolio_operations"] = portfolio_operations
            # Identify the *first* portfolio as the active one.

            active_port_id = getattr(active_port, "id", None)
            last_loaded_id = st.session_state.get("_active_portfolio_id")
            last_loaded_notional = st.session_state.get("_active_notional")

            # Only rebuild if portfolio or notional changed (avoid expensive loops)
            if active_port_id != last_loaded_id or float(notional_value) != float(last_loaded_notional or 0):
                try:
                    with st.spinner("Building position from selected portfolio(s)…"):
                        instrument_hash_to_asset, portfolio_positions = portfolio_operations.get_all_portfolios_as_positions(
                            portfolio_notional=float(notional_value),
                        )
                    # Preserve existing cfg (incl. valuation_date set by date_selector)
                    prev_cfg = st.session_state.get("position_cfg_mem") or {}
                    st.session_state["position_cfg_mem"] = {
                                                                ** prev_cfg,
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

                    # Seed the engine to use ":memory:" on next run
                    st.session_state["position_template_mem"] = active_position
                    st.session_state["instrument_hash_to_asset"] = ih2a_active

                    # DO NOT overwrite valuation_date here — it comes from the sidebar date_selector
                    st.session_state["cfg_path"] = ":memory:"  # <-- crucial
                    st.session_state["_active_portfolio_id"] = active_port_id
                    st.session_state["_active_notional"] = float(notional_value)

                    st.rerun()
                except Exception as e:
                    raise e

            # -------------------- Curve bump controls (per index) --------------------
            if ctx is not None and getattr(ctx, "position", None):
                present_indices = sorted({
                    getattr(ln.instrument, "floating_rate_index_name", None)
                    for ln in ctx.position.lines
                    if getattr(ln.instrument, "floating_rate_index_name", None) is not None
                })
                present_families = sorted({curve_family_key(i) for i in present_indices})

                st.markdown("### Curve bumps (by family)")
                prev = (st.session_state.get("curve_bump_spec_by_family")
                        or st.session_state.get("curve_bump_spec_by_index", {}))
                new_map: Dict[str, Dict[str, float]] = {}

                for fam in present_families:
                    rep_idx = next((i for i in present_indices if curve_family_key(i) == fam), None)
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
    if ctx is None or getattr(ctx, "position", None) is None:
        st.info("Select a portfolio in the sidebar to build the position and recalc curves & tables.")
        return

    # Reserve a slot for the curve so it appears above the overlay UI
    curve_slot = st.container()

    # Calc date comes from the user's valuation_date (not from curves)
    valuation_date = ctx.valuation_date

    # Overlay UI (no chart here; it only computes/clears points and returns traces)
    overlay_traces = st_position_yield_overlay(
        position=ctx.position,
        valuation_date=valuation_date,
        key="base_curve_position_overlay",
    )

    # Build the curve ONCE, add overlay traces, render ONCE
    fig_curve = plot_par_yield_curves_by_family(
        base_curves=ctx.base_curves,
        bumped_curves=ctx.bumped_curves,
        max_years=30,
        step_months=3,
        title="Par yield curves — Base vs Bumped (families)"
    )
    for tr in overlay_traces:
        fig_curve.add_trace(tr)

    with curve_slot:
        st.subheader("Par yield curve — Base vs Bumped")
        st.plotly_chart(fig_curve,width="stretch", key="par_curve_main")

    # Stats (carry cutoff slider on the same page)
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
    cutoff = valuation_date + __import__("datetime").timedelta(days=carry_days)

    stats = PositionOperations.portfolio_style_stats(
        base_position=ctx.position,
        bumped_position= ctx.bumped_position,
        valuation_date=valuation_date,
        cutoff=cutoff,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "NPV (base)",
        fmt_ccy(stats["npv_base"],),
        delta=fmt_ccy(stats["npv_delta"],),
    )
    c2.metric("NPV (bumped)", fmt_ccy(stats["npv_bumped"],))
    c3.metric(
        f"Carry to {cutoff.isoformat()} (base)",
        fmt_ccy(stats["carry_base"], ),
        delta=fmt_ccy(stats["carry_delta"], ),
    )
    c4.metric(f"Carry to {cutoff.isoformat()} (bumped)", fmt_ccy(stats["carry_bumped"],))

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

    if po is not None and hasattr(po, "st_position_npv_table_paginated"):
        po.st_position_npv_table_paginated(
        position=ctx.position,
        instrument_hash_to_asset=st.session_state.get("instrument_hash_to_asset"),
        bumped_position=ctx.bumped_position,
        page_size_options=(25, 50, 100, 200),
        default_size=50,
        enable_search=True,
        key="npv_table_curve_page",
    )
