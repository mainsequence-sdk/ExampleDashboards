from __future__ import annotations
import streamlit as st
from mainsequence.dashboards.streamlit.core.registry import register_page
from dashboards.fixed_income_app.context import AppContext, portfolio_stats
from dashboards.components.npv_table import st_position_npv_table_paginated
from dashboards.components.curve_bump import curve_bump_controls
from dashboards.curves.bumping import KEYRATE_GRID_TIIE
from dashboards.components.position_yield_overlay import st_position_yield_overlay
from dashboards.ux_utils import plot_par_yield_curve
import os

from pathlib import Path
import pandas as pd


def _fmt_ccy(x: float, symbol: str, signed: bool = False) -> str:
    import math
    if x is None or not math.isfinite(float(x)):
        return "—"
    return f"{symbol}{x:{'+,.2f' if signed else ',.2f'}}"


@register_page("curve_positions", "Curve, Stats & Positions", has_sidebar=True, order=0)
def render(ctx: AppContext):
    # ---------- Sidebar ----------
    with st.sidebar:


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

        st.divider()
        st.markdown("### Build instruments from sheet")
        default_path=os.environ.get("ANALYTIC_VECTOR_PATH")
        vendor_sheet = st.text_input(
            "Valmer Vector Analytico sheet path (.xls/.xlsx)",
            value=st.session_state.get("vendor_sheet_path",default_path),
            help="Path to the Vector Analítico sheet or similar."
        )


        if st.button("Rebuilt instruments", type="primary", use_container_width=True):
            try:
                from dashboards.build_instruments import build_position_from_sheet
                with st.spinner("Building instruments and writing position.json…"):
                    _pos, _cfg, outp ,_= build_position_from_sheet(
                        vendor_sheet,
                    )
                st.session_state["vendor_sheet_path"] = vendor_sheet
                st.session_state["cfg_path"] = outp  # point app to the fresh file
                st.success(f"✓ Rebuilt instruments and saved to:\n{outp}\n✓ Build state ")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to rebuild instruments:\n{e}")

    # ── Reserve a slot for the curve so it appears above the overlay UI ────────
    curve_slot = st.container()
    main_tab, build_state_tab = st.tabs(["main", "build_state"])
    # ── Overlay UI (no chart here; it only computes/clears points and returns traces)
    with main_tab:
        # ── Reserve a slot for the curve so it appears above the overlay UI ───
        curve_slot = st.container()

        # ── Overlay UI (no chart here; it computes/clears points and returns traces)

        overlay_traces = st_position_yield_overlay(
            position=ctx.position,                # always base position
            val_date=ctx.val_date,                # valuation date for maturities
            ts_base=ctx.ts_base,                  # kept for signature compatibility
            ql_ref_date=ctx.ts_base.referenceDate(),
            nodes_base=ctx.nodes_base,
            key="base_curve_position_overlay",
        )

        # ── Build the curve ONCE, add overlay traces, render ONCE ──────────────────
        fig_curve = plot_par_yield_curve(
            ctx.ts_base, ctx.ts_bump,
            ctx.ts_base.referenceDate(),
            ctx.nodes_base, ctx.nodes_bump,
            bump_tenors={},                        # optional vertical markers
            max_years=30, step_months=3,
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

    with build_state_tab:
        st.subheader("Build state (df_out)")
        # repo root (same place position.json lives by default)
        repo_root = Path(__file__).resolve().parents[2]
        df_out_path = repo_root / "df_out.csv"
        if df_out_path.exists():
            try:
                df_out = pd.read_csv(df_out_path)
                # Boolean columns to color (support both names for coupon pass flag)
                bool_cols = [c for c in ("pass_price", "pass_coupon", "pass_coupon_count") if c in df_out.columns]

                def _truthy(v):
                    if isinstance(v, bool):
                        return v
                    s = str(v).strip().lower()
                    if s in ("true", "1", "yes", "y", "t"): return True
                    if s in ("false", "0", "no", "n", "f"): return False
                    return None  # unknown / NaN

                def _fmt_bool(v):
                    t = _truthy(v)
                    return "✓" if t is True else ("✗" if t is False else "")

                def _style_bool(v):
                    t = _truthy(v)
                    if t is True:
                        return "background-color: #58D68D; color: #0E1216; font-weight: 600; text-align: center;"
                    if t is False:
                        return "background-color: #EC7063; color: #0E1216; font-weight: 600; text-align: center;"
                    return ""

                if bool_cols:
                    styled = (
                        df_out
                        .style
                        .format({c: _fmt_bool for c in bool_cols})
                        .applymap(_style_bool, subset=bool_cols)
                    )
                    # Hide index for nicer display across pandas versions
                    try:
                        styled = styled.hide(axis="index")
                    except Exception:
                        try:
                            styled = styled.hide_index()
                        except Exception:
                            pass
                    st.dataframe(styled, use_container_width=True)
                else:
                    st.dataframe(df_out, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download df_out.csv",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name="df_out.csv",
                    mime="text/csv",
                    key="dl_df_out_csv"
                )
            except Exception as e:
                st.error(f"Failed to read {df_out_path}:\n{e}")
        else:
            st.info("No build_state found yet. Click **Rebuilt instruments** in the sidebar to generate it.")
