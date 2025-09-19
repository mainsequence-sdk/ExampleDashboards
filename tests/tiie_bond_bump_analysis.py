#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interest-rate swap node bump analysis → rebootstrap → bond repricing.

- Input quotes: APIDataNode('interest_rate_swaps') → deposits + swaps.
- Apply tenor bumps in bp (e.g., {"5Y": 100.0}).
- Rebuild bootstrapped curve (PiecewiseLogCubicDiscount).
- Price 4 bonds (fixed, float, fixed@par, float@par) off base vs bumped curve.
- Cashflow cutoff for plots/tables.
- Plotly dark charts:
    • Portfolio cashflows (base vs bumped + Δ)
    • NPVs (base vs bumped + Δ)
    • ZERO/YIELD CURVE (base vs bumped)  ← NEW
- Curve diagnostics: zero-rate deltas at 2Y/3Y/5Y/7Y/10Y.

Run from repo root so `src/` is importable.
Requires: QuantLib, numpy, pandas, plotly
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import QuantLib as ql

# Plotly dark
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "plotly_dark"

# Repo imports
from mainsequence.instruments.data_interface import APIDataNode
from mainsequence.instruments.utils import to_ql_date, to_py_date
from mainsequence.instruments.pricing_models.swap_pricer import add_historical_fixings

# -------------------- knobs --------------------
VALUATION_DATE   = dt.date.today()
CASHFLOW_CUTOFF  = VALUATION_DATE + dt.timedelta(days=365 * 3)   # visualize first 3y; set None for all

# Bond setup (use longer term to show effect of 5Y bump)
BOND_TERM_FIXED  = ql.Period("7Y")
BOND_TERM_FLOAT  = ql.Period("7Y")

NOTIONAL         = 100_000_000.0
UNITS_PER_BOND   = 3.0

# Fixed bond conventions
FIXED_COUPON     = 0.045   # 4.5% coupon (example)
FIXED_FREQ       = ql.Period("6M")
FIXED_DC         = ql.Thirty360(ql.Thirty360.USA)
FIXED_CAL        = ql.TARGET()
FIXED_BDC        = ql.ModifiedFollowing
SETTLEMENT_DAYS  = 2

# Floating bond conventions (USD 3M style)
FLOAT_TENOR      = ql.Period("3M")
FLOAT_SPREAD     = 0.0000
FLOAT_DC         = ql.Actual360()
FLOAT_BDC        = ql.ModifiedFollowing

# Bumps on the quoted curve (bp). Example: +100 bp at 5Y
BUMP_BP_BY_TENOR: Dict[str, float] = {
    "5Y": 100.0,
    # "3Y": 25.0,
    # "10Y": -15.0,
}

# Bootstrapping day count (same as repo’s swap builder)
BOOTSTRAP_DC     = ql.Actual365Fixed()

# -------------------- helpers --------------------
def set_eval(d: dt.date) -> ql.Date:
    qld = to_ql_date(d)
    ql.Settings.instance().evaluationDate = qld
    ql.Settings.instance().includeReferenceDateEvents = False
    ql.Settings.instance().enforceTodaysHistoricFixings = False
    return qld

def canonical_tenor(t: str) -> str:
    return t.strip().upper()

def load_swap_nodes() -> list[dict]:
    """Get a copy of interest_rate_swaps curve nodes (deposits + swaps)."""
    market = APIDataNode.get_historical_data("interest_rate_swaps", {"USD_rates": {}})
    nodes  = list(market["curve_nodes"])  # copy
    return nodes

def apply_bumps(nodes: list[dict], bumps_bp_by_tenor: Dict[str, float]) -> list[dict]:
    """Return a deep-copied nodes list with bumped 'rate' for matching tenors."""
    bump_map = {canonical_tenor(k): v for k, v in bumps_bp_by_tenor.items()}
    bumped = []
    for n in nodes:
        n2 = dict(n)
        if "tenor" in n2:
            key = canonical_tenor(n2["tenor"])
            if key in bump_map:
                bp = bump_map[key]
                n2["rate"] = float(n2["rate"]) + bp / 10_000.0
        bumped.append(n2)
    return bumped

def build_curve_from_swap_nodes(calc_date: ql.Date, nodes: list[dict]) -> ql.YieldTermStructureHandle:
    """
    Build bootstrapped curve using deposits + swaps.
    Matches your repo's build_yield_curve logic, but accepts an explicit node set.
    """
    calendar = ql.TARGET()
    day_counter = BOOTSTRAP_DC

    rate_helpers: list[ql.RateHelper] = []

    # For swap helpers we need a temporary Ibor index handle
    tmp_handle = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, 0.02, day_counter))
    ibor_index = ql.USDLibor(ql.Period("3M"), tmp_handle)

    swap_fixed_leg_frequency  = ql.Annual
    swap_fixed_leg_convention = ql.Unadjusted
    swap_fixed_leg_daycounter = ql.Thirty360(ql.Thirty360.USA)

    for node in nodes:
        t = canonical_tenor(node.get("tenor", ""))
        rate = float(node["rate"])
        quote = ql.QuoteHandle(ql.SimpleQuote(rate))
        if node["type"].lower() == "deposit":
            tenor = ql.Period(t)
            helper = ql.DepositRateHelper(quote, tenor, 2, calendar, ql.ModifiedFollowing, False, day_counter)
        elif node["type"].lower() == "swap":
            tenor = ql.Period(t)
            helper = ql.SwapRateHelper(quote, tenor, calendar,
                                       swap_fixed_leg_frequency,
                                       swap_fixed_leg_convention,
                                       swap_fixed_leg_daycounter,
                                       ibor_index)
        else:
            raise ValueError(f"Unsupported node type: {node['type']}")
        rate_helpers.append(helper)

    curve = ql.PiecewiseLogCubicDiscount(calc_date, rate_helpers, day_counter)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)

# ---------- diagnostics ----------
def print_zero_diagnostics(tag: str, ts: ql.YieldTermStructureHandle, calc_date: ql.Date):
    dc = BOOTSTRAP_DC
    comp = ql.Continuous
    freq = ql.Annual
    cal  = ql.TARGET()

    tenors = ["2Y", "3Y", "5Y", "7Y", "10Y"]
    print(f"\n{tag} zero rates:")
    for s in tenors:
        d = cal.advance(calc_date, ql.Period(s))
        z = ts.zeroRate(d, dc, comp, freq).rate()
        print(f"  {s:>3s} @ {to_py_date(d)} : {z:.6%}")

def print_zero_deltas(ts0: ql.YieldTermStructureHandle, ts1: ql.YieldTermStructureHandle, calc_date: ql.Date):
    dc = BOOTSTRAP_DC
    comp = ql.Continuous
    freq = ql.Annual
    cal  = ql.TARGET()
    tenors = ["2Y", "3Y", "5Y", "7Y", "10Y"]
    print("\nZero-rate deltas (bumped - base):")
    any_move = False
    for s in tenors:
        d = cal.advance(calc_date, ql.Period(s))
        z0 = ts0.zeroRate(d, dc, comp, freq).rate()
        z1 = ts1.zeroRate(d, dc, comp, freq).rate()
        dz_bp = (z1 - z0) * 1e4
        print(f"  {s:>3s} : {dz_bp:+.2f} bp")
        if abs(dz_bp) > 1e-4:
            any_move = True
    if not any_move:
        print("  (no change detected; check bump map & nodes)")

# ---------- ZERO/YIELD CURVE PLOT (NEW) ----------
def sample_zero_curve(ts: ql.YieldTermStructureHandle,
                      calc_date: ql.Date,
                      max_years: int = 12,
                      step_months: int = 3) -> tuple[np.ndarray, np.ndarray]:
    cal = ql.TARGET()
    dc  = BOOTSTRAP_DC
    comp = ql.Continuous
    freq = ql.Annual
    t_list: list[float] = []
    z_list: list[float] = []
    months = 1
    while months <= max_years * 12:
        d = cal.advance(calc_date, ql.Period(months, ql.Months))
        t = dc.yearFraction(calc_date, d)
        z = ts.zeroRate(d, dc, comp, freq).rate()
        t_list.append(t)
        z_list.append(z)
        months += step_months
    return np.array(t_list), np.array(z_list)

def plot_yield_curves(base_ts: ql.YieldTermStructureHandle,
                      bumped_ts: ql.YieldTermStructureHandle,
                      calc_date: ql.Date,
                      bump_tenors: Dict[str, float],
                      max_years: int = 12,
                      step_months: int = 3):
    T0, Z0 = sample_zero_curve(base_ts,   calc_date, max_years, step_months)
    T1, Z1 = sample_zero_curve(bumped_ts, calc_date, max_years, step_months)

    fig = go.Figure()
    fig.add_scatter(x=T0, y=Z0*100, mode="lines", name="Base zero curve",   line=dict(color="#5DADE2", width=2))
    fig.add_scatter(x=T1, y=Z1*100, mode="lines", name="Bumped zero curve", line=dict(color="#F5B041", width=2))

    # Mark anchor tenors we bumped
    for s, bp in bump_tenors.items():
        per = ql.Period(s)
        # Convert to years (roughly) to place a vertical marker
        yrs = per.length()
        if per.units() == ql.Months:
            yrs = per.length()/12.0
        fig.add_vline(x=yrs, line_width=1, line_dash="dash", line_color="#E74C3C",
                      annotation_text=f"{s} bump {bp:+.0f}bp", annotation_position="top right",
                      annotation_font_color="#E74C3C")

    fig.update_layout(
        title="Zero/Yield Curve — Base vs Bumped",
        xaxis_title="Maturity (years)",
        yaxis_title="Zero rate (%)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()

# ---------- instruments ----------
def make_fixed_bond(curve: ql.YieldTermStructureHandle,
                    calc_date: ql.Date,
                    notional: float,
                    coupon: float,
                    term: ql.Period) -> ql.FixedRateBond:
    start = calc_date
    end   = ql.TARGET().advance(calc_date, term)
    sched = ql.Schedule(start, end, FIXED_FREQ, FIXED_CAL,
                        FIXED_BDC, FIXED_BDC, ql.DateGeneration.Forward, False)
    bond  = ql.FixedRateBond(SETTLEMENT_DAYS, notional, sched, [coupon], FIXED_DC)
    bond.setPricingEngine(ql.DiscountingBondEngine(curve))
    return bond

def make_float_bond(curve: ql.YieldTermStructureHandle,
                    calc_date: ql.Date,
                    notional: float,
                    spread: float,
                    term: ql.Period) -> ql.FloatingRateBond:
    index = ql.USDLibor(FLOAT_TENOR, curve)  # forecasting off bootstrapped curve
    add_historical_fixings(calc_date, index) # your helper populates past fixings

    start = calc_date
    end   = ql.TARGET().advance(calc_date, term)
    sched = ql.Schedule(start, end, FLOAT_TENOR, ql.TARGET(),
                        FLOAT_BDC, FLOAT_BDC, ql.DateGeneration.Forward, False)
    bond  = ql.FloatingRateBond(SETTLEMENT_DAYS, notional, sched, index, FLOAT_DC,
                                FLOAT_BDC, 1, [1.0], [spread], [], [], False, 100.0, start)
    bond.setPricingEngine(ql.DiscountingBondEngine(curve))
    return bond

def solve_par_fixed_coupon(curve, calc_date, notional, term) -> float:
    # bisection for coupon such that clean price ~ 100
    lo, hi = -0.02, 0.15
    for _ in range(60):
        mid = 0.5*(lo+hi)
        b = make_fixed_bond(curve, calc_date, notional, mid, term)
        price = b.cleanPrice()
        if price > 100.0: hi = mid
        else:             lo = mid
    return 0.5*(lo+hi)

def solve_par_float_spread(curve, calc_date, notional, term) -> float:
    lo, hi = -0.05, 0.05
    for _ in range(60):
        mid = 0.5*(lo+hi)
        b = make_float_bond(curve, calc_date, notional, mid, term)
        price = b.cleanPrice()
        if price > 100.0: hi = mid
        else:             lo = mid
    return 0.5*(lo+hi)

# ---------- cashflows ----------
@dataclass
class CF:
    bond_id: str
    pay_date: dt.date
    kind: str
    accrual_start: dt.date | None
    accrual_end: dt.date | None
    accrual: float | None
    rate: float | None
    amount: float
    df: float
    pv: float

def cashflows_df(bond_id: str,
                 bond: ql.Bond,
                 curve: ql.YieldTermStructureHandle,
                 today: ql.Date,
                 cutoff: dt.date | None) -> pd.DataFrame:
    rows: list[CF] = []
    for cf in bond.cashflows():
        if cf.date() <= today:
            continue
        pay = to_py_date(cf.date())
        if cutoff and pay > cutoff:
            continue
        amt = cf.amount()
        df  = curve.discount(cf.date())
        c = ql.as_coupon(cf)
        if c is not None:
            kind="coupon"
            rs, re = to_py_date(c.accrualStartDate()), to_py_date(c.accrualEndDate())
            accr = c.accrualPeriod()
            try: rate = c.rate()
            except: rate = None
        else:
            kind="redemption"; rs=re=None; accr=None; rate=None
        rows.append(CF(bond_id, pay, kind, rs, re, accr, rate, amt, df, amt*df))
    return pd.DataFrame([r.__dict__ for r in rows])

# ---------- plots ----------
def plot_cashflows(base_df: pd.DataFrame, bumped_df: pd.DataFrame, units: float):
    g0 = base_df.groupby("pay_date", as_index=False)["amount"].sum().rename(columns={"amount":"amount_base"})
    g1 = bumped_df.groupby("pay_date", as_index=False)["amount"].sum().rename(columns={"amount":"amount_bumped"})
    m = g0.merge(g1, on="pay_date", how="outer").fillna(0.0)
    m["amount_base"]   *= units
    m["amount_bumped"] *= units
    m["amount_delta"]   = m["amount_bumped"] - m["amount_base"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_bar(x=m["pay_date"], y=m["amount_base"],   name="Base CF",   marker_color="#5DADE2", row=1, col=1)
    fig.add_bar(x=m["pay_date"], y=m["amount_bumped"], name="Bumped CF", marker_color="#F5B041", row=1, col=1)
    fig.add_bar(x=m["pay_date"], y=m["amount_delta"],  name="Δ Amount",   marker_color="#E74C3C", row=2, col=1)
    fig.update_layout(title="Portfolio Cashflows (Base vs Bumped) and Δ", barmode="group", height=650)
    fig.update_yaxes(title_text="Amount (MXN)", row=1, col=1)
    fig.update_yaxes(title_text="Δ Amount (MXN)", row=2, col=1)
    fig.update_xaxes(title_text="Payment date", row=2, col=1)
    fig.show()

def plot_npvs(npv0: Dict[str,float], npv1: Dict[str,float], units: float):
    keys = list(npv0.keys())
    base = np.array([npv0[k] for k in keys]) * units
    bump = np.array([npv1[k] for k in keys]) * units
    delta = bump - base

    fig = go.Figure()
    fig.add_bar(x=keys, y=base, name="Base NPV",   marker_color="#5DADE2")
    fig.add_bar(x=keys, y=bump, name="Bumped NPV", marker_color="#F5B041")
    fig.add_scatter(x=keys, y=delta, name="Δ NPV", mode="markers+text",
                    text=[f"{d:,.0f}" for d in delta], textposition="top center",
                    marker=dict(color="#E74C3C", size=10))
    fig.update_layout(title=f"Per-Portfolio NPVs (units × {units:g})",
                      barmode="group", height=420, yaxis_title="PV")
    fig.show()

# -------------------- main --------------------
def main():
    ql_today = set_eval(VALUATION_DATE)
    print(f"Valuation date: {to_py_date(ql_today)}")

    # 1) Base & bumped curves
    base_nodes   = load_swap_nodes()
    bumped_nodes = apply_bumps(base_nodes, BUMP_BP_BY_TENOR)

    base_curve   = build_curve_from_swap_nodes(ql_today, base_nodes)
    bumped_curve = build_curve_from_swap_nodes(ql_today, bumped_nodes)

    # diagnostics + CURVE PLOT (NEW)
    print_zero_diagnostics("Base", base_curve, ql_today)
    print_zero_diagnostics("Bumped", bumped_curve, ql_today)
    print_zero_deltas(base_curve, bumped_curve, ql_today)
    plot_yield_curves(base_curve, bumped_curve, ql_today, BUMP_BP_BY_TENOR, max_years=12, step_months=3)

    # 2) Instruments (base)
    fixed1  = make_fixed_bond(base_curve,  ql_today, NOTIONAL, FIXED_COUPON, BOND_TERM_FIXED)
    float1  = make_float_bond(base_curve,  ql_today, NOTIONAL, FLOAT_SPREAD, BOND_TERM_FLOAT)

    par_cpn    = solve_par_fixed_coupon(base_curve,  ql_today, NOTIONAL, BOND_TERM_FIXED)
    fixed_par  = make_fixed_bond(base_curve,  ql_today, NOTIONAL, par_cpn, BOND_TERM_FIXED)

    par_spread = solve_par_float_spread(base_curve,  ql_today, NOTIONAL, BOND_TERM_FLOAT)
    float_par  = make_float_bond(base_curve,  ql_today, NOTIONAL, par_spread, BOND_TERM_FLOAT)

    # 3) Price base
    npv_base = {
        "fixed_1":   fixed1.NPV(),
        "float_1":   float1.NPV(),
        "fixed_par": fixed_par.NPV(),
        "float_par": float_par.NPV(),
    }
    print("\nBase NPVs (per unit):")
    for k,v in npv_base.items():
        print(f"  {k:10s}: {v:,.2f}")

    # 4) Rebuild instruments on bumped curve (so forecasting AND discounting change)
    fixed1_b   = make_fixed_bond(bumped_curve, ql_today, NOTIONAL, FIXED_COUPON, BOND_TERM_FIXED)
    float1_b   = make_float_bond(bumped_curve, ql_today, NOTIONAL, FLOAT_SPREAD, BOND_TERM_FLOAT)
    fixed_par_b  = make_fixed_bond(bumped_curve, ql_today, NOTIONAL, par_cpn, BOND_TERM_FIXED)
    float_par_b  = make_float_bond(bumped_curve, ql_today, NOTIONAL, par_spread, BOND_TERM_FLOAT)

    npv_bumped = {
        "fixed_1":   fixed1_b.NPV(),
        "float_1":   float1_b.NPV(),
        "fixed_par": fixed_par_b.NPV(),
        "float_par": float_par_b.NPV(),
    }
    print("\nBumped NPVs (per unit):")
    for k,v in npv_bumped.items():
        print(f"  {k:10s}: {v:,.2f}")

    # 5) Cashflows (cutoff)
    df_base = pd.concat([
        cashflows_df("fixed_1",   fixed1,    base_curve,   ql_today, CASHFLOW_CUTOFF),
        cashflows_df("float_1",   float1,    base_curve,   ql_today, CASHFLOW_CUTOFF),
        cashflows_df("fixed_par", fixed_par, base_curve,   ql_today, CASHFLOW_CUTOFF),
        cashflows_df("float_par", float_par, base_curve,   ql_today, CASHFLOW_CUTOFF),
    ], ignore_index=True)

    df_bump = pd.concat([
        cashflows_df("fixed_1",   fixed1_b,    bumped_curve, ql_today, CASHFLOW_CUTOFF),
        cashflows_df("float_1",   float1_b,    bumped_curve, ql_today, CASHFLOW_CUTOFF),
        cashflows_df("fixed_par", fixed_par_b, bumped_curve, ql_today, CASHFLOW_CUTOFF),
        cashflows_df("float_par", float_par_b, bumped_curve, ql_today, CASHFLOW_CUTOFF),
    ], ignore_index=True)

    # 6) Portfolio totals
    units = UNITS_PER_BOND
    port_base  = {k: v * units for k, v in npv_base.items()}
    port_bump  = {k: v * units for k, v in npv_bumped.items()}

    print("\nPortfolio PV (units × {:.0f}):".format(units))
    for k in npv_base.keys():
        print(f"  {k:10s}: base={port_base[k]:,.0f}  bumped={port_bump[k]:,.0f}  Δ={port_bump[k]-port_base[k]:,.0f}")

    # 7) Visuals
    plot_cashflows(df_base, df_bump, units=units)
    plot_npvs(npv_base, npv_bumped, units=units)

    print(f"\nPar fixed coupon used:  {par_cpn:.6%}")
    print(f"Par floater spread used:{par_spread:.6%}")


if __name__ == "__main__":
    set_eval(VALUATION_DATE)
    main()
