# dashboards/bond_real_option_analysis/views/run_analysis.py
from __future__ import annotations
import math
from typing import Tuple

import streamlit as st
import numpy as np
import pandas as pd
import QuantLib as ql
from mainsequence.dashboards.streamlit.core.registry import register_page

from dashboards.bond_real_option_analysis.context import AppContext
from dashboards.bond_real_option_analysis.engine import (
    LSMSettings, lsm_optimal_stopping, eval_sell_invest_to_horizon
)

# ---- tiny helpers ----
def _fmt_ccy(x: float, symbol: str = "MXN$") -> str:
    return "—" if x is None or not math.isfinite(float(x)) else f"{symbol}{x:,.2f}"

def _make_parallel_spread_curve(base: ql.YieldTermStructureHandle, spread_bp: float) -> ql.YieldTermStructureHandle:
    if abs(spread_bp) < 1e-12:
        return base
    spread = ql.QuoteHandle(ql.SimpleQuote(float(spread_bp) / 10_000.0))
    ts = ql.ZeroSpreadedTermStructure(base, spread, ql.Continuous, ql.Annual)
    ts.enableExtrapolation()
    return ql.YieldTermStructureHandle(ts)

def _bond_label(line) -> str:
    ins = line.instrument
    try:
        return f"{ins.content_hash()} — {getattr(ins, 'maturity_date', '')}"
    except Exception:
        return str(ins.content_hash())

@register_page("bond_option", "Bond Real Option", order=0, has_sidebar=True)
def render(ctx: AppContext):
    st.markdown("Pick an instrument from your current **Position**, set ALM wedges, and run the LSM analysis.")

    # ---------- Sidebar controls ----------
    with st.sidebar:
        # Position selector
        ids = [str(ln.instrument.content_hash()) for ln in ctx.position.lines]
        labels = [_bond_label(ln) for ln in ctx.position.lines]
        id_to_line = {ids[i]: ctx.position.lines[i] for i in range(len(ids))}
        asset_id = st.selectbox("Instrument", options=ids, format_func=lambda s: labels[ids.index(s)] if s in ids else s)

        st.divider()
        st.markdown("### LSM settings")
        n_paths = st.slider("Paths", min_value=2000, max_value=60000, value=12000, step=2000)
        seed    = st.number_input("Random seed", value=7, step=1)
        mesh_m  = st.select_slider("Decision mesh (months)", options=[1, 3, 6, 12], value=1)
        a       = st.number_input("Hull–White mean reversion (a)", value=0.03, step=0.005, format="%.4f")
        sigma   = st.number_input("Hull–White volatility (sigma)", value=0.01, step=0.002, format="%.4f")

        st.divider()
        st.markdown("### Frictions")
        bid_ask_fric = st.number_input("Bid–ask when selling (bps)", value=0.0, step=1.0, format="%.1f")
        capital_bps  = st.number_input("Capital charge while holding (bps / year)", value=0.0, step=5.0, format="%.1f")

        st.divider()
        st.markdown("### ALM wedges (parallel spread vs MARKET)")
        ftp_bp    = st.number_input("FTP curve spread (bps)", value=-25.0, step=5.0, format="%.1f")
        invest_bp = st.number_input("Invest curve spread (bps)", value=35.0, step=5.0, format="%.1f")

        run = st.button("Run analysis", type="primary", use_container_width=True)

    if not asset_id:
        st.info("Select an instrument to start.")
        st.stop()

    line = id_to_line[asset_id]
    ql.Settings.instance().evaluationDate = ql.Date(ctx.val_date.day, ctx.val_date.month, ctx.val_date.year)

    # Curves (market/FTP/invest)
    ts_market = ctx.ts_market  # from context (TIIE base curve)
    ts_ftp_eq = ts_market      # frictionless
    ts_inv_eq = ts_market

    ts_ftp_alm = _make_parallel_spread_curve(ts_market, ftp_bp)
    ts_inv_alm = _make_parallel_spread_curve(ts_market, invest_bp)

    settings = LSMSettings(
        a=float(a), sigma=float(sigma),
        n_paths=int(n_paths), seed=int(seed),
        capital_rate=float(capital_bps) / 10_000.0,
        mesh_months=int(mesh_m),
        record_diagnostics=True
    )

    if run:
        with st.spinner("Running LSM on selected bond…"):
            # Frictionless
            res_A = lsm_optimal_stopping(
                instrument=line.instrument,
                market_curve=ts_market, ftp_curve=ts_ftp_eq, invest_curve=ts_inv_eq,
                evaluation_fn=eval_sell_invest_to_horizon,
                eval_params={"bid_ask_bps": 0.0},
                settings=settings,
            )
            # ALM wedge
            res_B = lsm_optimal_stopping(
                instrument=line.instrument,
                market_curve=ts_market, ftp_curve=ts_ftp_alm, invest_curve=ts_inv_alm,
                evaluation_fn=eval_sell_invest_to_horizon,
                eval_params={"bid_ask_bps": float(bid_ask_fric)},
                settings=settings,
            )

        # ---------- Results ----------
        st.subheader("Scenario results")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Frictionless** (market = FTP = invest, no costs)")
            st.metric("LSM value @ t=0", _fmt_ccy(res_A.lsm_value),
                      delta=_fmt_ccy(res_A.lsm_value - res_A.ql_npv))
            st.caption(f"QL bond NPV (market curve): {_fmt_ccy(res_A.ql_npv)}")

        with c2:
            st.markdown("**ALM wedge** (market + FTP/invest spreads, frictions)")
            st.metric("LSM value @ t=0 (ALM PV)", _fmt_ccy(res_B.lsm_value),
                      delta=_fmt_ccy(res_B.lsm_value - res_A.lsm_value))
            st.caption(f"Δ vs Frictionless: {_fmt_ccy(res_B.lsm_value - res_A.lsm_value,)}")

        st.divider()
        st.markdown("**Exercise diagnostics (ALM)**")
        # Build a DataFrame for the exercise rate by time
        diagB = res_B.diag or {}
        times = np.array(diagB.get("exercise_rate_by_time", []), dtype=float)
        if times.size:
            # The engine stores exercise rate by *index*; build x-axis as grid times (years)
            # We don't have grid_times here, but we can infer length and ask engine? For UI,
            # show the per-step share; index ~ time order.
            df_ex = pd.DataFrame({
                "step": np.arange(len(times)),
                "exercise_share": times
            })
            st.line_chart(df_ex, x="step", y="exercise_share", use_container_width=True)
            st.caption("Share of paths that prefer selling at each decision step (higher → more likely to sell).")
        else:
            st.info("No exercise events recorded (common in frictionless runs).")

        st.divider()
        st.markdown("**Details (JSON)**")
        st.json({
            "frictionless": {"lsm_value": res_A.lsm_value, "ql_npv": res_A.ql_npv},
            "alm": {"lsm_value": res_B.lsm_value, "ql_npv": res_A.ql_npv,
                    "params": {"ftp_bp": ftp_bp, "invest_bp": invest_bp,
                               "bid_ask_bps": bid_ask_fric, "capital_bps": capital_bps}},
            "settings": dataclasses.asdict(settings) if hasattr(settings, "__dict__") else {},
        })
    else:
        st.info("Set parameters on the sidebar and click **Run analysis**.")
