# dashboards/bond_real_option_analysis/views/run_analysis.py
from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd
import streamlit as st
import QuantLib as ql
import mainsequence.instruments as msi

from mainsequence.dashboards.streamlit.core.registry import register_page
from dashboards.apps.bond_real_option_analysis.engine import (
    LSMSettings, lsm_optimal_stopping, eval_sell_invest_to_horizon
)
from dashboards.core.formatters import fmt_ccy
from dashboards.core.ql import qld
# NOTE: services path per your codebase
from dashboards.services.curves import BumpSpec, build_curves_for_ui
from mainsequence.instruments.settings import TIIE_28_UID
from dashboards.components.asset_select import sidebar_asset_single_select
from dashboards.plots.theme import apply_graph_theme ,go  # optional but nice
from plotly.subplots import make_subplots

def _fig_M_and_coupons(diag, currency_symbol: str = "$"):
    t = np.array(diag.get("grid_times", []), dtype=float)
    M = np.array(diag.get("M", []), dtype=float)
    coup = np.array(diag.get("coupon_on_grid", []), dtype=float)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=t, y=M, mode="lines+markers", name="M(t) = DF_FTP/DF_INV", secondary_y=False)
    fig.add_bar(x=t, y=coup, name="Coupon now", opacity=0.35, secondary_y=True)
    fig.update_yaxes(title_text="Multiplier (×)", secondary_y=False)
    fig.update_yaxes(title_text=f"Coupon ({currency_symbol})", secondary_y=True)
    fig.update_layout(title="ALM multiplier vs. coupon grid", hovermode="x unified", height=420)
    return fig
def _fig_exercise_rate_by_time(diag):
    t = np.array(diag.get("grid_times", []), dtype=float)
    shares = np.array(diag.get("exercise_rate_by_time", []), dtype=float)
    fig = go.Figure()
    if t.size and shares.size and t.size == shares.size:
        fig.add_scatter(x=t, y=shares, mode="lines+markers", name="Exercise share")
        fig.update_layout(
            title="Exercise intensity by decision time",
            xaxis_title="t (years)",
            yaxis_title="Share of paths selling",
            yaxis=dict(range=[0, 1]),  # always visible even if all zeros
            height=380,
            hovermode="x unified",
        )
    else:
        # Fallback empty frame with fixed axes
        fig.update_layout(
            title="Exercise intensity by decision time",
            xaxis_title="t (years)",
            yaxis_title="Share of paths selling",
            yaxis=dict(range=[0, 1]),
            height=380
        )
    return fig
def _fig_expected_stop_values(diag, bid_ask_bps: float, currency_symbol: str = "$"):
    t   = np.array(diag.get("grid_times", []), dtype=float)
    ex  = np.array(diag.get("ex_coupon_expected", []), dtype=float)        # market ex‑coupon
    M   = np.array(diag.get("M", []), dtype=float)
    stop_pv = ex * (1.0 - bid_ask_bps * 1e-4) * M
    fig = go.Figure()
    fig.add_scatter(x=t, y=ex,       mode="lines+markers", name="Expected ex‑coupon (market)")
    fig.add_scatter(x=t, y=stop_pv,  mode="lines+markers", name="Expected stop PV to horizon (ALM)")
    fig.update_layout(title="If you sell at t: expected proceeds vs. ALM stop PV", yaxis_title=f"Value ({currency_symbol})",
                      hovermode="x unified", height=420)
    return fig

def _fig_first_exercise_hist(diag):
    x = np.array(diag.get("first_exercise_times", []), dtype=float)
    if x.size == 0:
        return None
    fig = go.Figure()
    fig.add_histogram(x=x, nbinsx=min(30, max(10, len(np.unique(x)))))
    share = float(diag.get("share_exercised_at_all", 0.0))
    fig.update_layout(title=f"First exercise time distribution  •  Share exercised: {share:.1%}",
                      xaxis_title="t (years)", yaxis_title="Count", height=360)
    return fig

def _fig_short_rate_paths(diag, max_paths=20):
    t = np.array(diag.get("grid_times", []), dtype=float)
    phi = np.array(diag.get("phi", []), dtype=float)
    # We did not persist full paths (memory‑heavy). Plot φ(t) and ±1σ proxy band.
    # Use OU step std from diag['dt'],a,σ to draw an indicative envelope.
    dt = np.array(diag.get("dt", []), dtype=float)
    a = float(diag.get("a", 0.03)); sigma = float(diag.get("sigma", 0.01))
    # exact OU variance of x(t) around 0:
    var_x = np.zeros_like(t)
    if t.size:
        # cumulative: Var[x_t] = σ^2 * (1 - e^{-2at})/(2a) ; limit a→0 -> σ^2 t
        if abs(a) < 1e-8:
            var_x = (sigma**2) * t
        else:
            var_x = (sigma**2) * (1.0 - np.exp(-2.0*a*t)) / (2.0*a)
    std_r = np.sqrt(np.maximum(var_x, 0.0))  # since r = x + φ
    fig = go.Figure()
    fig.add_scatter(x=t, y=phi, mode="lines", name="φ(t)", line=dict(width=2))
    fig.add_scatter(x=t, y=phi + std_r, mode="lines", name="φ(t) + 1σ", line=dict(dash="dot"))
    fig.add_scatter(x=t, y=phi - std_r, mode="lines", name="φ(t) - 1σ", line=dict(dash="dot"))
    fig.update_layout(title="Short‑rate backdrop (φ(t) ± 1σ proxy)", xaxis_title="t (years)", yaxis_title="r",
                      height=360, hovermode="x unified")
    return fig

def _fig_value_decomposition(res_base, res_all, instrument, ts_market, ftp_bp, invest_bp, bid_ask_bps, capital_bps, settings):
    # Run isolated scenarios, keeping others at 'neutral'
    def run_case(ftp_bp_, invest_bp_, bid_ask_bps_, capital_bps_):
        ts_ftp = _make_parallel_spread_curve(ts_market, ftp_bp_)
        ts_inv = _make_parallel_spread_curve(ts_market, invest_bp_)
        stg = dataclasses.replace(settings, capital_rate=float(capital_bps_) / 10_000.0)
        r = lsm_optimal_stopping(
            instrument=instrument,
            market_curve=ts_market, ftp_curve=ts_ftp, invest_curve=ts_inv,
            evaluation_fn=eval_sell_invest_to_horizon,
            eval_params={"bid_ask_bps": float(bid_ask_bps_)},
            settings=stg,
        )
        return r.lsm_value

    v0 = float(res_base.lsm_value)
    parts = {
        "Bid–ask":    run_case(0.0, 0.0, bid_ask_bps, 0.0) - v0,
        "FTP wedge":  run_case(ftp_bp, 0.0, 0.0, 0.0)      - v0,
        "Invest wedge": run_case(0.0, invest_bp, 0.0, 0.0) - v0,
        "Capital":    run_case(0.0, 0.0, 0.0, capital_bps) - v0,
        "All combined": float(res_all.lsm_value - v0),
    }
    # Bar chart
    names = list(parts.keys())
    vals  = [parts[k] for k in names]
    fig = go.Figure(go.Bar(x=names, y=vals, text=[f"{v:+,.2f}" for v in vals], textposition="outside"))
    fig.update_layout(title="Value attribution relative to frictionless (ΔLSM)",
                      yaxis_title="ΔLSM (currency units)", height=420)
    return fig, parts
def _make_parallel_spread_curve(base: ql.YieldTermStructureHandle, spread_bp: float) -> ql.YieldTermStructureHandle:
    if abs(spread_bp) < 1e-12:
        return base
    spread = ql.QuoteHandle(ql.SimpleQuote(float(spread_bp) / 10_000.0))
    ts = ql.ZeroSpreadedTermStructure(base, spread, ql.Continuous, ql.Annual)
    ts.enableExtrapolation()
    return ql.YieldTermStructureHandle(ts)

@register_page("bond_option", "Bond Real Option", order=0, has_sidebar=True)
def render(ctx):  # ← no import of AppContext; keep it simple to avoid cycles
    st.markdown("Search an **Asset**, rebuild its instrument, set ALM wedges, and run the LSM analysis.")

    # ---------- Sidebar ----------
    with st.sidebar:
        # Valuation date (no JSON; persisted in session)
        new_val = st.date_input("Valuation date", value=ctx.val_date, key="broa_val_date")
        if new_val != ctx.val_date:
            st.session_state["valuation_date"] = new_val
            st.rerun()

        # 1) Search & select exactly one asset
        asset = sidebar_asset_single_select(title="Find asset (name or UID)", key_prefix="broa_asset")

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

        run = st.button("Run analysis", type="primary",width="stretch")

    if asset is None:
        st.info("Use the sidebar to search and load an asset.")
        return

    # ---------- Rebuild instrument (same as asset_detail.py) ----------
    pricing_detail = getattr(asset, "current_pricing_detail", None)
    if pricing_detail is None or getattr(pricing_detail, "instrument_dump", None) is None:
        st.error("Asset has no current_pricing_detail.instrument_dump; cannot rebuild instrument.")
        return

    inst_dump = pricing_detail.instrument_dump
    instrument = msi.Instrument.rebuild(inst_dump)

    # Set valuation date globally and on the instrument
    ql.Settings.instance().evaluationDate = qld(st.session_state["valuation_date"])
    instrument.set_valuation_date(st.session_state["valuation_date"])

    # ---------- Build curves ONLY NOW (depends on selected asset) ----------
    idx_uid = getattr(instrument, "floating_rate_index_name", None) or TIIE_28_UID
    ts_market, _, _, _ = build_curves_for_ui(
        qld(st.session_state["valuation_date"]),
        BumpSpec(keyrate_bp={}, parallel_bp=0.0),
        index_identifier=str(idx_uid),
    )

    # For floaters, align cashflow amounts with Market expectations
    instrument.reset_curve(ts_market)

    # FTP/Invest curves are spreads vs the selected asset's MARKET curve
    ts_ftp_eq = ts_market
    ts_inv_eq = ts_market
    ts_ftp_alm = _make_parallel_spread_curve(ts_market, float(ftp_bp))
    ts_inv_alm = _make_parallel_spread_curve(ts_market, float(invest_bp))

    settings = LSMSettings(
        a=float(a), sigma=float(sigma),
        n_paths=int(n_paths), seed=int(seed),
        capital_rate=float(capital_bps) / 10_000.0,
        mesh_months=int(mesh_m),
        record_diagnostics=True,
    )

    if run:
        with st.spinner("Running LSM on selected asset…"):
            # Frictionless baseline
            res_A = lsm_optimal_stopping(
                instrument=instrument,
                market_curve=ts_market, ftp_curve=ts_ftp_eq, invest_curve=ts_inv_eq,
                evaluation_fn=eval_sell_invest_to_horizon,
                eval_params={"bid_ask_bps": 0.0},
                settings=settings,
            )
            # ALM wedge scenario
            res_B = lsm_optimal_stopping(
                instrument=instrument,
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
            st.metric("LSM value @ t=0", fmt_ccy(res_A.lsm_value),
                      delta=fmt_ccy(res_A.lsm_value - res_A.ql_npv))
            st.caption(f"QL bond NPV (market curve): {fmt_ccy(res_A.ql_npv)}")

        with c2:
            st.markdown("**ALM wedge** (market + FTP/invest spreads, frictions)")
            st.metric("LSM value @ t=0 (ALM PV)", fmt_ccy(res_B.lsm_value),
                      delta=fmt_ccy(res_B.lsm_value - res_A.lsm_value))
            st.caption(f"Δ vs Frictionless: {fmt_ccy(res_B.lsm_value - res_A.lsm_value)}")

        st.divider()
        diagB = res_B.diag or {}

        # 3.1 ALM wedge intuition
        st.divider()
        st.markdown("### ALM wedge & coupon grid")
        fig_mc = _fig_M_and_coupons(diagB, currency_symbol="$")
        apply_graph_theme(fig_mc)
        st.plotly_chart(fig_mc,width="stretch")
        st.caption(
            "• **What it shows:** The ALM multiplier **M(t)=DF_FTP(t,H)/DF_INV(t,H)** (line) together with the coupon that pays at each grid time (bars).  \n"
            "• **How to read it:** **M(t)>1** means investing proceeds grows **less** than the FTP numéraire to H (so selling and reinvesting is less attractive); "
            "**M(t)<1** means the Invest curve grows **faster** than FTP toward H (selling becomes more attractive). Larger coupon bars at a time t boost the **hold** branch there.  \n"
            "• **Why it matters:** Explains the **ALM wedge**: differences between FTP and Invest change the economics of selling now vs holding."
        )
        # 3.2 Expected ex‑coupon vs stop PV
        fig_stop = _fig_expected_stop_values(diagB, bid_ask_bps=float(bid_ask_fric), currency_symbol="$")
        apply_graph_theme(fig_stop)
        st.plotly_chart(fig_stop, width="stretch")
        st.caption(
            "• **What it shows:** Expected **ex‑coupon sale price** under the Market model (line 1) and the corresponding **ALM stop PV to horizon** "
            "(after bid–ask and scaled by M(t), line 2).  \n"
            "• **How to read it:** Where the **stop PV** line is high, immediate liquidation is more appealing—"
            "but remember the **hold** branch also includes any coupon at t and the continuation value, so this is **intuition**, not a decision boundary.  \n"
            "• **Why it matters:** Visualizes what you’d get **if you sell now** (in FTP PV terms) versus the raw market sale price."
        )
        # --- Exercise diagnostics (ALM) ---
        st.markdown("### Exercise diagnostics (ALM)")

        # A. Exercise rate by time (uses Plotly with fixed y-range)
        fig_ex_rate = _fig_exercise_rate_by_time(diagB)
        apply_graph_theme(fig_ex_rate)
        st.plotly_chart(fig_ex_rate, width="stretch")

        # Explain this chart
        shares = np.array(diagB.get("exercise_rate_by_time", []), dtype=float)
        grid_t = np.array(diagB.get("grid_times", []), dtype=float)
        if shares.size == 0 or grid_t.size == 0:
            st.info("No diagnostics available for exercise rate. Check that `record_diagnostics=True` in LSM settings.")
        elif np.allclose(shares, 0.0):
            st.info(
                "No early liquidation occurs under current wedges/frictions—every path prefers to hold at all decision times.")
        st.caption(
            "• **What it shows:** For each decision time t, the fraction of Monte Carlo paths that would **sell** (exercise) "
            "rather than hold. \n"
            "• **How to read it:** Values closer to 1 mean a strong incentive to sell at that time; a flat line at 0 means "
            "no early exercise. Spikes around coupon dates can indicate ‘sell after receiving coupon’ patterns. \n"
            "• **Why it matters:** Highlights when the ALM wedges (FTP vs Invest), bid–ask, and capital charges make liquidation optimal."
        )

        # B. First exercise time histogram (if any path exercises)
        fig_hist = _fig_first_exercise_hist(diagB)
        if fig_hist is not None:
            apply_graph_theme(fig_hist)
            st.plotly_chart(fig_hist, width="stretch")
            st.caption(
                "• **What it shows:** The distribution of the **first time** each path chooses to sell. "
                "The subtitle shows the share of paths that exercise at least once. \n"
                "• **How to read it:** Mass near t=0 indicates a preference to sell immediately; clusters near coupon dates "
                "suggest ‘collect coupon then sell’. \n"
                "• **Why it matters:** Identifies **when** the option is being used, not just **how often**."
            )

        # 3.4 Short‑rate backdrop
        st.markdown("### Short‑rate backdrop")
        fig_r = _fig_short_rate_paths(diagB)
        apply_graph_theme(fig_r)
        st.plotly_chart(fig_r,width="stretch")
        st.caption(
            "• **What it shows:** The Hull–White drift component **φ(t)** and an indicative ±1σ envelope for the short rate.  \n"
            "• **How to read it:** Higher rates push **ex‑coupon prices lower**, often favoring **hold** (wait for coupons) unless ALM wedges flip the trade‑off.  \n"
            "• **Why it matters:** Grounds the option’s behavior in the **rate environment** implied by your Market curve (mean reversion a, volatility σ)."
        )
        # 3.5 Value attribution bars
        st.subheader("Value attribution")
        fig_attr, parts = _fig_value_decomposition(
            res_base=res_A, res_all=res_B, instrument=instrument, ts_market=ts_market,
            ftp_bp=float(ftp_bp), invest_bp=float(invest_bp),
            bid_ask_bps=float(bid_ask_fric), capital_bps=float(capital_bps), settings=settings
        )
        apply_graph_theme(fig_attr)
        st.plotly_chart(fig_attr, width="stretch")
        st.caption(
            "• **What it shows:** Change in **LSM value** versus the frictionless baseline, isolating each driver: **Bid–ask**, **FTP wedge**, **Invest wedge**, and **Capital**. "
            "The last bar (**All combined**) is the total effect of all drivers together.  \n"
            "• **How to read it:** Positive bars add option value; negative bars reduce it. "
            "Individual contributions are computed with **one driver on, others neutral**—so small non‑additivity can occur versus the combined bar.  \n"
            "• **Why it matters:** Quantifies **what’s driving** the real‑option value."
        )

        with st.expander("Contribution table"):
            st.dataframe(pd.Series(parts, name="ΔLSM").to_frame())

        st.divider()
        st.markdown("**Details (JSON)**")
        st.json({
            "asset": {
                "id": getattr(asset, "id", None),
                "unique_identifier": getattr(asset, "unique_identifier", None),
                "snapshot": getattr(getattr(asset, "current_snapshot", None), "name", None),
                "index_uid": str(idx_uid),
            },
            "frictionless": {"lsm_value": res_A.lsm_value, "ql_npv": res_A.ql_npv},
            "alm": {"lsm_value": res_B.lsm_value, "ql_npv": res_A.ql_npv,
                    "params": {"ftp_bp": ftp_bp, "invest_bp": invest_bp,
                               "bid_ask_bps": bid_ask_fric, "capital_bps": capital_bps}},
            "settings": dataclasses.asdict(settings) if hasattr(settings, "__dict__") else {},
        })

        with st.expander("Sensitivity sweeps (on demand)"):
            c1, c2 = st.columns(2)
            run_sweep = c1.button("Compute sweeps")
            rescale = c2.selectbox("Plot", ["Δ vs frictionless", "Absolute LSM"], index=0)

            if run_sweep:
                with st.spinner("Running sensitivity sweeps…"):
                    # 1D sweeps around current settings (lightweight)
                    ftp_grid = np.array(sorted(set([float(ftp_bp) + d for d in (-50, -25, 0, 25, 50)])))
                    invest_grid = np.array(sorted(set([float(invest_bp) + d for d in (-50, -25, 0, 25, 50)])))
                    bid_grid = np.array([0, 10, 25, 50, 75, 100], dtype=float)
                    cap_grid = np.array([0, 25, 50, 75, 100, 150], dtype=float)

                    def eval_case(fbp, ibp, babps, cbps):
                        ts_ftp = _make_parallel_spread_curve(ts_market, fbp)
                        ts_inv = _make_parallel_spread_curve(ts_market, ibp)
                        stg = dataclasses.replace(settings, capital_rate=float(cbps) / 10_000.0)
                        r = lsm_optimal_stopping(
                            instrument=instrument,
                            market_curve=ts_market, ftp_curve=ts_ftp, invest_curve=ts_inv,
                            evaluation_fn=eval_sell_invest_to_horizon,
                            eval_params={"bid_ask_bps": float(babps)},
                            settings=stg,
                        )
                        return r.lsm_value

                    base_val = float(res_A.lsm_value)
                    # FTP sweep
                    ftp_vals = np.array([eval_case(fbp, 0.0, 0.0, 0.0) for fbp in ftp_grid])
                    # Invest sweep
                    inv_vals = np.array([eval_case(0.0, ibp, 0.0, 0.0) for ibp in invest_grid])
                    # Bid–ask sweep
                    bid_vals = np.array([eval_case(0.0, 0.0, b, 0.0) for b in bid_grid])
                    # Capital sweep
                    cap_vals = np.array([eval_case(0.0, 0.0, 0.0, c) for c in cap_grid])

                    def plot_sweep(x, y, title, xlab):
                        yy = (y - base_val) if rescale.startswith("Δ") else y
                        fig = go.Figure(go.Scatter(x=x, y=yy, mode="lines+markers"))
                        fig.update_layout(title=title, xaxis_title=xlab,
                                          yaxis_title=("ΔLSM" if rescale.startswith("Δ") else "LSM"),
                                          height=380)
                        apply_graph_theme(fig);
                        st.plotly_chart(fig, width="stretch")

                    plot_sweep(ftp_grid, ftp_vals, "FTP spread sensitivity", "FTP spread (bp)")
                    plot_sweep(invest_grid, inv_vals, "Invest spread sensitivity", "Invest spread (bp)")
                    plot_sweep(bid_grid, bid_vals, "Bid–ask sensitivity", "Bid–ask (bp)")
                    plot_sweep(cap_grid, cap_vals, "Capital charge sensitivity", "Capital (bp/yr)")
                    st.caption(
                        "• **What it shows:** 1‑D sensitivity of the LSM value to each input while holding the others neutral.  \n"
                        "• **How to read it:** When plotting **Δ vs frictionless**, curves above 0 mean the driver increases real‑option value. "
                        "Steeper slopes indicate **higher sensitivity**.  \n"
                        "• **Why it matters:** Helps prioritize which wedges/frictions move the economics the most."
                    )

    else:
        st.info("Set parameters on the sidebar and click **Run analysis**.")
