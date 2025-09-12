#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit ALM cockpit:
- Reads position.json (assets + liabilities + ALM assumptions)
- Lets user specify curve bumps (per-tenor + parallel)
- Rebootstraps and reprices base vs bumped
- Visualizes: par yield curve (with nodes), liquidity ladder (Base vs Bumped),
  KPIs with deltas, and ΔEVE comparison.
- Streamlit totals table (currency formatted) with clickable "Details" per instrument.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st



# ---------- ensure examples/alm is importable ----------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- repo imports ----------
from examples.alm.utils import (
    set_eval, to_py_date, id_meta,
    compute_lcr, compute_nsfr, build_liquidity_ladder,
    duration_gap_and_eve, nii_12m_shock, get_bumped_position, to_ql_date,
)
from examples.alm.ux_utils import (
    register_theme, plot_par_yield_curve, plot_liquidity_ladder,
    table_kpis, plot_eve_bars_compare
)

from src.instruments.position import Position,npv_table
from src.utils import to_ql_date
from ui.curves.bumping import  build_curves_for_ui as build_bumped_curves, KEYRATE_GRID_TIIE, BumpSpec
from ui.components.curve_bump import curve_bump_controls
from ui.components.npv_table import st_position_npv_table_paginated

# ---------- Streamlit page ----------
st.set_page_config(page_title="ALM Cockpit", layout="wide")
register_theme()

# ---------- helpers ----------
DEFAULT_POS_JSON = ROOT / "position.json"

@st.cache_data(show_spinner=False)
def read_positions(path: str | Path) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def parse_manual_bumps(text: str) -> Dict[str, float]:
    """Parse lines like '5Y: 100' or '3M: -10' into {tenor: bp}."""
    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        t, v = line.split(":", 1)
        try:
            out[t.strip().upper()] = float(v.strip())
        except Exception:
            pass
    return out

def get_query_ins() -> Optional[str]:
    """Read selected instrument from query params (?ins=id)."""
    try:
        qp = st.query_params  # Streamlit >= 1.32
        if "ins" in qp and qp["ins"]:
            return qp["ins"]
    except Exception:
        qp = st.experimental_get_query_params()  # legacy
        if "ins" in qp and qp["ins"]:
            return qp["ins"][0]
    return None

def set_query_ins(ins_id: Optional[str]) -> None:
    """Set or clear ?ins=..."""
    try:
        if ins_id:
            st.query_params["ins"] = ins_id
        else:
            st.query_params.clear()
    except Exception:
        if ins_id:
            st.experimental_set_query_params(ins=ins_id)
        else:
            st.experimental_set_query_params()

def fmt_ccy(x: float, ccy: str) -> str:
    return "" if pd.isna(x) else f"{ccy}{x:,.2f}"

def fmt_units(x: float) -> str:
    return "" if pd.isna(x) else f"{x:,.2f}"

def totals_display_df_fmt(totals: pd.DataFrame, currency_symbol: str) -> pd.DataFrame:
    """
    Build a Streamlit-friendly table:
      - currency columns are preformatted as strings with thousands separators
      - 'Details' is a link to set ?ins=<instrument>
    """
    d = totals.copy()
    # Ensure required columns exist
    for col in ["instrument", "units", "base", "bumped", "delta"]:
        if col not in d.columns:
            d[col] = np.nan

    d["Instrument"] = d["instrument"].astype(str)
    d["Units"] = d["units"].apply(fmt_units)
    d["Base"] = d["base"].apply(lambda v: fmt_ccy(v, currency_symbol))
    d["Bumped"] = d["bumped"].apply(lambda v: fmt_ccy(v, currency_symbol))
    d["Δ"] = d["delta"].apply(lambda v: fmt_ccy(v, currency_symbol))

    # Details link (omit for TOTAL)
    d["Details"] = np.where(
        d["Instrument"].str.upper() == "TOTAL", "",
        d["Instrument"].map(lambda x: f"?ins={x}")
    )

    return d[["Instrument", "Units", "Base", "Bumped", "Δ", "Details"]]

def cf_for_ins(cf: pd.DataFrame, ins_id: str) -> pd.DataFrame:
    """Filter a cashflow DataFrame for an instrument id, tolerant to column name variants."""
    if cf is None or cf.empty or not isinstance(cf, pd.DataFrame):
        return pd.DataFrame()
    for k in ("ins_id", "instrument", "bond_id", "name", "id"):
        if k in cf.columns:
            return cf.loc[cf[k] == ins_id].copy()
    return pd.DataFrame()

# =========================================
#                 SIDEBAR
# =========================================
st.sidebar.title("Controls")

pos_path = st.sidebar.text_input("Position JSON path", value=str(DEFAULT_POS_JSON))
try:
    cfg = read_positions(pos_path)
    st.sidebar.success("Position file loaded.")
except Exception as e:
    st.sidebar.error(f"Failed to read {pos_path}\n{e}")
    st.stop()

val_date = dt.date.fromisoformat(cfg["valuation"]["valuation_date"])
currency_symbol = cfg.get("alm_assumptions", {}).get("currency_symbol", "USD$ ")
ccy = (cfg.get("valuation", {}).get("currency", "USD") or "USD").upper()



# Curve bumps


available_tenors =list(KEYRATE_GRID_TIIE)


spec = curve_bump_controls(
    available_tenors=available_tenors,
    default_bumps=cfg.get("curve_bumps_bp", {}),
    default_parallel_bp=0.0,
    header="Curve bumps (bp)",
    key="alm_curve_bumps"
)


ts_base, ts_bump, nodes_base, nodes_bump, index_hint = build_bumped_curves( to_ql_date(val_date), spec)

# ALM params
st.sidebar.markdown("### ALM parameters")
liq_months = st.sidebar.slider("Liquidity ladder horizon (months)", 3, 24, 12, 1)
eve_bp = st.sidebar.slider("ΔEVE shock (bp, parallel)", 50, 400, 200, 25)
nii_bp = st.sidebar.slider("NII 12m shock (bp, parallel)", -200, 400, 100, 25)
nii_days = int(cfg.get("alm_assumptions", {}).get("nii_horizon_days", 365))
hqla = float(cfg.get("alm_assumptions", {}).get("hqla_amount", 0.0))
inflow_cap = float(cfg.get("alm_assumptions", {}).get("lcr_inflow_cap", 0.75))
lcr_days = st.sidebar.slider("LCR horizon (days)", min_value=10, max_value=90, value=30, step=5)
position=Position.from_json_dict(cfg["position"])




st.sidebar.info("**units** in the position are multipliers (# of identical positions).")

# =========================================
#          REBUILD & PRICE
# =========================================
ql_today = set_eval(val_date)

bumped_position=get_bumped_position(position=position,bump_curve=ts_bump)

cutoff = to_py_date(ql_today) + dt.timedelta(days=int(cfg["valuation"].get("cashflow_cutoff_days", 3 * 365)))
# Independent, single‑responsibility calls:
cf0 = position.cashflows_by_id( cutoff, apply_units=True)
cf1 = bumped_position.cashflows_by_id( cutoff, apply_units=True)
npv0 = position.npvs_by_id( apply_units=True)
npv1 = bumped_position.npvs_by_id(apply_units=True)
units = position.units_by_id()  # used for duration gap sign split
totals_raw = npv_table(npv0, npv1, units, include_total=True)  # optional raw df for debugging

meta = id_meta(position)
units_by_id_meta = {k: v["units"] for k, v in meta.items()}
# ALM analytics (base & bumped)
base_for_risk = ts_base

lcr_base = compute_lcr(cf0, meta, to_py_date(ql_today), hqla, horizon_days=lcr_days, inflow_cap=inflow_cap)
liq_base = build_liquidity_ladder(cf0, to_py_date(ql_today), months=liq_months)
dur_base = duration_gap_and_eve(npv0, npv1, eve_bp=float(eve_bp),units=units)
nii_base = nii_12m_shock(cf0, cf1, ql_today, shock_bp=float(nii_bp), horizon_days=nii_days)
nsfr_vals = compute_nsfr(meta)

bump_for_risk = ts_bump


lcr_bump = compute_lcr(cf1, meta, to_py_date(ql_today), hqla, horizon_days=lcr_days, inflow_cap=inflow_cap)
liq_bump = build_liquidity_ladder(cf1, to_py_date(ql_today), months=liq_months)
dur_bump =  duration_gap_and_eve(npv0, npv1, eve_bp=float(eve_bp),units=units)
nii_bump = nii_12m_shock(cf0, cf1, ql_today, shock_bp=float(nii_bp), horizon_days=nii_days)

# =========================================
#              LAYOUT/SECTIONS
# =========================================
st.title("Bank ALM Cockpit")
st.caption(f"Valuation date: **{val_date.isoformat()}** — currency: **{currency_symbol.strip()}**")

# ----- curve plot + totals table (table BELOW the plot) -----
st.subheader("Par yield curve (base vs bumped)")
st.plotly_chart(
    plot_par_yield_curve(
        ts_base, ts_bump, ql_today,
        nodes_base, nodes_bump,
        bump_tenors=spec.keyrate_bp,
        max_years=12, step_months=3,
        index_hint=index_hint
    ),
    use_container_width=True
)


st.subheader("Portfolio totals (NPV)")
disp = st_position_npv_table_paginated(position, currency_symbol, bumped_position=bumped_position)
# Optional selection (alternative to clicking "Open")
ids_no_total = [i for i in disp["Instrument"].tolist() if i.upper() != "TOTAL"]
with st.expander("Select instrument (alternative to clicking 'Open')"):
    pick = st.selectbox("Instrument", options=["—"] + ids_no_total, index=0)
    if pick != "—":
        set_query_ins(pick)

# ----- instrument detail template -----
sel = get_query_ins()
if sel and sel.upper() != "TOTAL":
    st.markdown("---")
    st.subheader(f"Instrument details — **{sel}**")

    base_val = float(npv0.get(sel, np.nan)) if sel in npv0 else np.nan
    bump_val = float(npv1.get(sel, np.nan)) if sel in npv1 else np.nan

    cA, cB, cC = st.columns(3)
    cA.metric("NPV (base)", f"{currency_symbol}{base_val:,.2f}",
              delta=f"{currency_symbol}{(bump_val - base_val):+,.2f}" if np.isfinite(base_val) and np.isfinite(bump_val) else None)
    cB.metric("Units", f"{units.get(sel, 1.0):,.2f}")
    cC.button("Clear selection", on_click=lambda: set_query_ins(None))

    t1, t2, t3 = st.tabs(["Cashflows — Base", "Cashflows — Bumped", "Template / TODO"])
    with t1:
        cf_sel = cf_for_ins(cf0, sel)
        if not cf_sel.empty:
            colcfg = {
                "pay_date": st.column_config.DateColumn("Payment date", format="YYYY-MM-DD"),
                "amount": st.column_config.NumberColumn("Amount", format=f"{currency_symbol} %,.2f"),
                "pv": st.column_config.NumberColumn("PV", format=f"{currency_symbol} %,.2f"),
                "rate": st.column_config.NumberColumn("Rate", format="%.6f"),
            }
            st.dataframe(cf_sel, hide_index=True, use_container_width=True, column_config=colcfg)
        else:
            st.info("No upcoming cashflows found for this instrument in the base scenario.")
    with t2:
        cf_sel_b = cf_for_ins(cf1, sel)
        if not cf_sel_b.empty:
            colcfg = {
                "pay_date": st.column_config.DateColumn("Payment date", format="YYYY-MM-DD"),
                "amount": st.column_config.NumberColumn("Amount", format=f"{currency_symbol} %,.2f"),
                "pv": st.column_config.NumberColumn("PV", format=f"{currency_symbol} %,.2f"),
                "rate": st.column_config.NumberColumn("Rate", format="%.6f"),
            }
            st.dataframe(cf_sel_b, hide_index=True, use_container_width=True, column_config=colcfg)
        else:
            st.info("No upcoming cashflows found for this instrument in the bumped scenario.")
    with t3:
        st.markdown(
            """
            **Template ideas**
            - DV01 / PV01 by key-rate buckets
            - Coupon schedule visual
            - Sensitivities vs chosen curve tenors
            - Conventions summary
            """
        )

# ----- Liquidity ladder: Base vs Bumped -----
st.subheader("Liquidity ladder — Base vs Bumped")
cl, cr = st.columns(2)
with cl:
    st.plotly_chart(
        plot_liquidity_ladder(liq_base, currency_symbol=currency_symbol,
                              title=f"Base (next {liq_months} months)"),
        use_container_width=True
    )
with cr:
    st.plotly_chart(
        plot_liquidity_ladder(liq_bump, currency_symbol=currency_symbol,
                              title=f"Bumped (next {liq_months} months)"),
        use_container_width=True
    )

# ----- KPIs with deltas -----
st.markdown("---")
st.header("ALM Analytics — Base vs Bumped")

k = st.columns(5)
k[0].metric(
    "LCR (base)", f"{lcr_base['LCR']:.2f}×",
    delta=f"{(lcr_bump['LCR'] - lcr_base['LCR']):+.2f}×",
    help=f"Horizon {lcr_days}d | HQLA {currency_symbol}{lcr_base['HQLA']:,.0f} "
         f"| Net outflows {currency_symbol}{lcr_base['net_outflows_30d']:,.0f}"
)
k[1].metric("NSFR", f"{compute_nsfr(meta)['NSFR']:.2f}×", help="Structural funding (weights × notionals).")
k[2].metric(
    "Duration gap (base)", f"{dur_base['D_gap']:.2f}y",
    delta=f"{(dur_bump['D_gap'] - dur_base['D_gap']):+.2f}y",
    help=f"D_A={dur_base['D_A']:.2f}, D_L={dur_base['D_L']:.2f}, L/A={dur_base['k_L_over_A']:.2f}"
)
k[3].metric(
    f"ΔEVE +{int(eve_bp)}bp (base)",
    f"{currency_symbol}{dur_base['EVE'][f'+{int(eve_bp)}bp']:,.0f}",
    delta=f"{currency_symbol}{(dur_bump['EVE'][f'+{int(eve_bp)}bp'] - dur_base['EVE'][f'+{int(eve_bp)}bp']):+,.0f}"
)
k[4].metric(
    f"NII Δ(+{int(nii_bp)}bp, 12m) (base)",
    f"{currency_symbol}{nii_base['delta']:,.0f}",
    delta=f"{currency_symbol}{(nii_bump['delta'] - nii_base['delta']):+,.0f}"
)

# ----- ΔEVE grouped bars: Base vs Bumped -----
st.subheader("ΔEVE (parallel shocks) — Base vs Bumped")
st.plotly_chart(
    plot_eve_bars_compare(dur_base["EVE"], dur_bump["EVE"], currency_symbol=currency_symbol),
    use_container_width=True
)

# ----- KPI tables (optional) -----
with st.expander("Show KPI tables"):
    kpis_base = {
        "LCR (base)": lcr_base["LCR"],
        f"ΔEVE +{int(eve_bp)}bp (base)": dur_base["EVE"][f"+{int(eve_bp)}bp"],
        f"ΔEVE -{int(eve_bp)}bp (base)": dur_base["EVE"][f"-{int(eve_bp)}bp"],
        f"NII Δ(+{int(nii_bp)}bp, 12m) (base)": nii_base["delta"],
        "Duration Gap (base)": dur_base["D_gap"],
    }
    kpis_bump = {
        "LCR (bumped)": lcr_bump["LCR"],
        f"ΔEVE +{int(eve_bp)}bp (bumped)": dur_bump["EVE"][f"+{int(eve_bp)}bp"],
        f"ΔEVE -{int(eve_bp)}bp (bumped)": dur_bump["EVE"][f"-{int(eve_bp)}bp"],
        f"NII Δ(+{int(nii_bp)}bp, 12m) (bumped)": nii_bump["delta"],
        "Duration Gap (bumped)": dur_bump["D_gap"],
    }
    st.plotly_chart(table_kpis(kpis_base, currency_symbol=currency_symbol), use_container_width=True)
    st.plotly_chart(table_kpis(kpis_bump, currency_symbol=currency_symbol), use_container_width=True)

# ----- debug / raw data (optional) -----
with st.expander("Debug / raw data"):
    st.write("Totals (raw):", totals_raw)
    st.write("Cashflows (base, head):", cf0.head() if not cf0.empty else "(empty)")
    st.write("Cashflows (bumped, head):", cf1.head() if not cf1.empty else "(empty)")
