# ui.dashboard_base/pages/curve_and_positions.py
from __future__ import annotations
import streamlit as st
from ui.dashboard_base.core.registry import register_page
from ui.dashboard_base.core.context import AppContext, portfolio_stats
from ui.components.npv_table import st_position_npv_table_paginated
from ui.components.curve_bump import curve_bump_controls
from ui.curves.bumping import KEYRATE_GRID_TIIE



from examples.alm.ux_utils import plot_par_yield_curve

def _fmt_ccy(x: float, symbol: str, signed: bool = False) -> str:
    import math
    if x is None or not math.isfinite(float(x)):
        return "—"
    return f"{symbol}{x:{'+,.2f' if signed else ',.2f'}}"

@register_page("curve_positions", "Curve, Stats & Positions")
def render(ctx: AppContext):
    # ---------- Sidebar (this view is the only owner) ----------
    with st.sidebar:
        # Config path
        new_path = st.text_input("Position JSON path", value=st.session_state.get("cfg_path", ""))
        if new_path and new_path != st.session_state.get("cfg_path"):
            st.session_state["cfg_path"] = new_path
            st.rerun()

        # Curve bump controls
        spec = curve_bump_controls(
            available_tenors=list(KEYRATE_GRID_TIIE),  # good default
            default_bumps=st.session_state.get("curve_bump_spec", {}).get("keyrate_bp", {}),
            default_parallel_bp=float(st.session_state.get("curve_bump_spec", {}).get("parallel_bp", 0.0)),
            header="Curve bumps (bp)",
            key="global_curve_bumps"
        )
        spec_map = {"keyrate_bp": spec.keyrate_bp, "parallel_bp": float(spec.parallel_bp)}
        if spec_map != st.session_state.get("curve_bump_spec"):
            st.session_state["curve_bump_spec"] = spec_map
            st.rerun()
    # ── Curve ───────────────────────────────────────────────────────────────────
    st.subheader("Par yield curve (base vs bumped)")
    st.plotly_chart(
        plot_par_yield_curve(
            ctx.ts_base, ctx.ts_bump,
            ctx.ts_base.referenceDate(),
            ctx.nodes_base, ctx.nodes_bump,
            bump_tenors={},  # vertical markers optional
            max_years=12, step_months=3, index_hint=ctx.index_hint
        ),
        use_container_width=True
    )

    # ── Stats (carry cutoff slider on the same page) ───────────────────────────
    st.subheader("Portfolio statistics — Base vs Bumped")
    carry_days = st.slider(
        "Carry cutoff (days from valuation date)",
        min_value=30, max_value=1460,
        value=(ctx.carry_cutoff - ctx.val_date).days, step=30
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
        key="npv_table_curve_page"
    )
