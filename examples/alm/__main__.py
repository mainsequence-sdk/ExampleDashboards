# src/__main__.py
from __future__ import annotations

import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import QuantLib as ql

from examples.alm.utils import (
    set_eval, to_py_date,
    load_swap_nodes, apply_bumps, build_curve_from_swap_nodes,
    sample_zero_curve, price_all, portfolio_totals,
id_meta, nodes_parallel_bump, compute_lcr, compute_nsfr,
    build_liquidity_ladder, compute_repricing_gap, duration_gap_and_eve, nii_12m_shock

)
from  examples.alm.ux_utils import (
    register_theme, plot_yield_curves, plot_cashflows, plot_npvs, table_from_df,
plot_liquidity_ladder, plot_repricing_gap, plot_eve_bars, table_kpis
)

# -------------------- config --------------------
ROOT = Path(__file__).resolve().parents[0]
POSITION_JSON = ROOT / "position.json"

# Defaults if some fields are missing
DEFAULT_CASHFLOW_CUTOFF_DAYS = 365 * 3

def _read_positions() -> dict:
    with open(POSITION_JSON, "r") as fh:
        return json.load(fh)

def _abs_cutoff(valuation: dict) -> dt.date | None:
    d = dt.date.fromisoformat(valuation["valuation_date"])
    n = int(valuation.get("cashflow_cutoff_days", DEFAULT_CASHFLOW_CUTOFF_DAYS))
    return d + dt.timedelta(days=n) if n > 0 else None

# -------------------- main --------------------
def main() -> None:
    register_theme()  # ensure theme
    cfg = _read_positions()
    val_date = dt.date.fromisoformat(cfg["valuation"]["valuation_date"])
    cutoff = _abs_cutoff(cfg["valuation"])
    bumps = cfg.get("curve_bumps_bp", {})

    # 1) Settings + base/bumped curves
    ql_today = set_eval(val_date)
    base_nodes   = load_swap_nodes()
    bumped_nodes = apply_bumps(base_nodes, bumps)
    base_curve   = build_curve_from_swap_nodes(ql_today, base_nodes)
    bump_curve   = build_curve_from_swap_nodes(ql_today, bumped_nodes)

    # 2) Diagnostics + yield curve plot
    T0, Z0 = sample_zero_curve(base_curve, ql_today, max_years=12, step_months=3)
    T1, Z1 = sample_zero_curve(bump_curve, ql_today, max_years=12, step_months=3)
    fig_curve = plot_yield_curves(T0, Z0, T1, Z1, bumps)
    fig_curve.show()

    # 3) Price all instruments (bonds + swaps); also collect cashflows
    npv0, npv1, cf0, cf1, units = price_all(base_curve, bump_curve, ql_today, cfg, cutoff)

    # 4) Print portfolio totals
    totals = portfolio_totals(npv0, npv1, units)
    print("\nValuation date:", to_py_date(ql_today))
    print(totals.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    # 5) Charts
    fig_cf  = plot_cashflows(cf0[["ins_id","pay_date","amount"]],
                             cf1[["ins_id","pay_date","amount"]],
                             units_by_id=units)
    fig_cf.show()

    fig_npv = plot_npvs(
        {k: npv0[k] * units.get(k, 1.0) for k in npv0},
        {k: npv1[k] * units.get(k, 1.0) for k in npv1},
        title="Instrument NPVs (units applied)"
    )
    fig_npv.show()

    # 6) Optional summary table
    tbl = table_from_df(totals.reset_index(drop=True), title="Portfolio totals")
    tbl.show()

    meta = id_meta(cfg)
    sym = cfg.get("alm_assumptions", {}).get("currency_symbol", "USD$ ")
    hqla = cfg.get("alm_assumptions", {}).get("hqla_amount", 0.0)
    inflow_cap = cfg.get("alm_assumptions", {}).get("lcr_inflow_cap", 0.75)
    buckets_days = cfg.get("alm_assumptions", {}).get("repricing_buckets_days", [30, 90, 180, 365, 730, 1825, 36500])

    # ---- LCR / NSFR (simplified) ----
    lcr = compute_lcr(cf0, meta, to_py_date(ql_today), hqla, horizon_days=30, inflow_cap=inflow_cap)
    nsfr = compute_nsfr(meta)

    # ---- Liquidity Ladder (12M) ----
    liq = build_liquidity_ladder(cf0, to_py_date(ql_today), months=12)
    plot_liquidity_ladder(liq, currency_symbol=sym).show()

    # ---- Repricing GAP ----
    gap_df = compute_repricing_gap(cfg, base_curve, ql_today, buckets_days)
    plot_repricing_gap(gap_df, currency_symbol=sym).show()

    # ---- Duration GAP & EVE ----
    dur = duration_gap_and_eve(cfg, base_nodes, ql_today, bump_bp=1.0, eve_bp=200.0)
    plot_eve_bars(dur["EVE"], currency_symbol=sym).show()

    # ---- NII 12m sensitivity (+100bp) ----
    nii = nii_12m_shock(cfg, base_nodes, ql_today, shock_bp=100.0,
                        horizon_days=cfg.get("alm_assumptions", {}).get("nii_horizon_days", 365))

    # ---- KPI table (one place) ----
    kpis = {
        "LCR": lcr["LCR"],
        "NSFR": nsfr["NSFR"],
        "Duration Gap": dur["D_gap"],
        "Assets PV (A)": dur["A"],
        "Liabilities PV (L)": dur["L"],
        "ΔEVE +200bp": dur["EVE"]["+200bp"],
        "ΔEVE -200bp": dur["EVE"]["-200bp"],
        "NII Δ(+100bp, 12m)": nii["delta"]
    }
    table_kpis(kpis, currency_symbol=sym).show()

    # Optional: print a short console summary
    print("\n--- ALM SUMMARY ---")
    print(
        f"LCR = {lcr['LCR']:.2f}x   (HQLA {sym}{lcr['HQLA']:,.0f}, Net Outflows 30d {sym}{lcr['net_outflows_30d']:,.0f})")
    print(f"NSFR = {nsfr['NSFR']:.2f}x   (ASF {sym}{nsfr['ASF']:,.0f}, RSF {sym}{nsfr['RSF']:,.0f})")
    print(f"Duration Gap = {dur['D_gap']:.2f} y   A={sym}{dur['A']:,.0f}  L={sym}{dur['L']:,.0f}")
    print(f"ΔEVE +200bp = {sym}{dur['EVE']['+200bp']:,.0f}   ΔEVE -200bp = {sym}{dur['EVE']['-200bp']:,.0f}")
    print(f"NII 12m Δ(+100bp) = {sym}{nii['delta']:,.0f}")

if __name__ == "__main__":
    main()
