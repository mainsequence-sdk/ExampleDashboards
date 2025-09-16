from __future__ import annotations
import streamlit as st
from mainsequence.dashboards.streamlit.core.registry import register_page
from dashboards.fixed_income_app.context import AppContext, portfolio_stats
from dashboards.components.npv_table import st_position_npv_table_paginated
from dashboards.components.curve_bump import curve_bump_controls
from dashboards.curves.bumping import KEYRATE_GRID_TIIE
from dashboards.components.position_yield_overlay import st_position_yield_overlay
from examples.alm.ux_utils import plot_par_yield_curve


def _fmt_ccy(x: float, symbol: str, signed: bool = False) -> str:
    import math
    if x is None or not math.isfinite(float(x)):
        return "—"
    return f"{symbol}{x:{'+,.2f' if signed else ',.2f'}}"


@register_page("curve_positions", "Curve, Stats & Positions", has_sidebar=True, order=0)
def render(ctx: AppContext):
    # ---------- Sidebar ----------
    with st.sidebar:
        new_path = st.text_input("Position JSON path", value=st.session_state.get("cfg_path", ""))
        if new_path and new_path != st.session_state.get("cfg_path"):
            st.session_state["cfg_path"] = new_path
            st.rerun()

        spec = curve_bump_controls(
            available_tenors=list(KEYRATE_GRID_TIIE),
            default_bumps=st.session_state.get("curve_bump_spec", {}).get("keyrate_bp", {}),
            default_parallel_bp=float(st.session_state.get("curve_bump_spec", {}).get("parallel_bp", 0.0)),
            header="Curve bumps (bp)",
            key="global_curve_bumps",
        )
        spec_map = {"keyrate_bp": spec.keyrate_bp, "parallel_bp": float(spec.parallel_bp)}
        if spec_map != st.session_state.get("curve_bump_spec"):
            st.session_state["curve_bump_spec"] = spec_map
            st.rerun()

    # ── Reserve a slot for the curve so it appears above the overlay UI ────────
    curve_slot = st.container()

    # ── Overlay UI (no chart here; it only computes/clears points and returns traces)
    market_clean_prices = {}
    for line in ctx.position.lines:
        market_clean_prices[line.instrument.content_hash()] = 100.00

    overlay_traces = st_position_yield_overlay(
        position=ctx.position,                # always base position
        market_clean_prices=market_clean_prices,  # clean prices from market valuation
        val_date=ctx.val_date,                # valuation date for maturities
        ts_base=ctx.ts_base,                  # kept for signature compatibility
        ql_ref_date=ctx.ts_base.referenceDate(),
        nodes_base=ctx.nodes_base,
        index_hint=ctx.index_hint,
        key="base_curve_position_overlay",
    )

    # ── Build the curve ONCE, add overlay traces, render ONCE ──────────────────
    fig_curve = plot_par_yield_curve(
        ctx.ts_base, ctx.ts_bump,
        ctx.ts_base.referenceDate(),
        ctx.nodes_base, ctx.nodes_bump,
        bump_tenors={},                        # optional vertical markers
        max_years=12, step_months=3, index_hint=ctx.index_hint,
    )
    for tr in overlay_traces:
        fig_curve.add_trace(tr)

    with curve_slot:
        st.subheader("Par yield curve — Base vs Bumped")
        st.plotly_chart(fig_curve, use_container_width=True, key="par_curve_main")

    # ── Stats (carry cutoff slider on the same page) ───────────────────────────
    st.subheader("Portfolio statistics — Base vs Bumped")
    carry_days = st.slider(
        "Carry cutoff (days from valuation date)",
        min_value=30, max_value=1460,
        value=(ctx.carry_cutoff - ctx.val_date).days, step=30,
    )
    cutoff = ctx.val_date + __import__("datetime").timedelta(days=carry_days)

    stats = portfolio_stats(ctx.position, ctx.bumped_position, ctx.val_date, cutoff)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NPV (base)", _fmt_ccy(stats["npv_base"], ctx.currency_symbol),
              delta=_fmt_ccy(stats["npv_delta"], ctx.currency_symbol, signed=True))
    c2.metric("NPV (bumped)", _fmt_ccy(stats["npv_bumped"], ctx.currency_symbol))
    c3.metric(f"Carry to {cutoff.isoformat()} (base)",
              _fmt_ccy(stats["carry_base"], ctx.currency_symbol),
              delta=_fmt_ccy(stats["carry_delta"], ctx.currency_symbol, signed=True))
    c4.metric(f"Carry to {cutoff.isoformat()} (bumped)",
              _fmt_ccy(stats["carry_bumped"], ctx.currency_symbol))

    # ── Positions table ─────────────────────────────────────────────────────────
    st.subheader("Positions — NPV (paginated)")
    st_position_npv_table_paginated(
        position=ctx.position,
        currency_symbol=ctx.currency_symbol,
        bumped_position=ctx.bumped_position,
        page_size_options=(25, 50, 100, 200),
        default_size=50,
        enable_search=True,
        key="npv_table_curve_page",
    )
