#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Bond Position viewer:
- Curve bump controls
- Par yield curve (base vs bumped)
- Paginated Position NPV table
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import datetime as dt

import numpy as np
import streamlit as st

# ---------- import path ----------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- repo imports ----------
from examples.alm.utils import set_eval, to_py_date, get_bumped_position, to_ql_date
from examples.alm.ux_utils import register_theme, plot_par_yield_curve

from src.instruments.position import Position,portfolio_stats
from src.utils import to_py_date,to_ql_date
from ui.curves.bumping import build_curves_for_ui as build_bumped_curves, KEYRATE_GRID_TIIE, BumpSpec
from ui.components.curve_bump import curve_bump_controls

# Pagination-enabled NPV table (new helper you’ll add below)
from ui.components.npv_table import st_position_npv_table_paginated
from ui.components.valuation_inputs import valuation_controls
from ui.components.position_loader import (
    position_source_input, instantiate_position
)


# ---------- Streamlit page ----------
st.set_page_config(page_title="BOND — Curve & Positions (Minimal)", layout="wide")
register_theme()

# ---------- sidebar controls ----------
DEFAULT_POS_JSON = ROOT / "position.json"

# 1) Pick file + load (cached on content)
path_str, cfg, template_position, _sig = position_source_input(DEFAULT_POS_JSON)

# 2) Valuation date (reusable component; default from cfg if present)
default_val_date = cfg.get("valuation", {}).get("valuation_date")
val_date = valuation_controls(default_date=default_val_date, key="mini_val")
ql_today = set_eval(val_date)




# Tenors list for the bump UI (TIIE grid for MXN; otherwise derive from nodes)

available_tenors = list(KEYRATE_GRID_TIIE)

# 4) Interactive bumps
spec = curve_bump_controls(
    available_tenors=available_tenors,
    default_bumps=cfg.get("curve_bumps_bp", {}),
    default_parallel_bp=0.0,
    header="Curve bumps (bp)",
    key="mini_curve_bumps"
)

# ---------- build curves with the interactive spec ----------
ts_base, ts_bump, nodes_base, nodes_bump, index_hint = build_bumped_curves( to_ql_date(val_date), spec)

# ---------- instantiate positions from the cached template (no file reload) ----------
base_position   = instantiate_position(template_position, ts_base, val_date)
bumped_position = instantiate_position(template_position, ts_bump, val_date)

# ---------- KPIs (optional) ----------
carry_days = st.sidebar.slider("Carry cutoff (days)", 30, 1460, 365, 30, key="carry_cutoff")
carry_cutoff = val_date + dt.timedelta(days=carry_days)
stats = portfolio_stats(base_position, bumped_position, val_date, carry_cutoff)

# =========================================
#                 MAIN
# =========================================
st.title("ALM — Curve & Positions (Minimal)")
currency_symbol = cfg.get("alm_assumptions", {}).get("currency_symbol", "USD$ ")
st.caption(f"Valuation date: **{val_date.isoformat()}** — currency: **{currency_symbol.strip()}**")

# --- Curve ---
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

# --- KPIs (compact) ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("NPV (base)",   f"{currency_symbol}{stats['npv_base']:,.2f}",
          delta=f"{currency_symbol}{stats['npv_delta']:+,.2f}")
c2.metric("NPV (bumped)", f"{currency_symbol}{stats['npv_bumped']:,.2f}")
c3.metric(f"Carry to {carry_cutoff.isoformat()} (base)",
          f"{currency_symbol}{stats['carry_base']:,.2f}",
          delta=f"{currency_symbol}{stats['carry_delta']:+,.2f}")
c4.metric(f"Carry to {carry_cutoff.isoformat()} (bumped)",
          f"{currency_symbol}{stats['carry_bumped']:,.2f}")

# --- Paginated NPV table ---
st.subheader("Positions — NPV (paginated)")
st_position_npv_table_paginated(
    position=base_position,
    currency_symbol=currency_symbol,
    bumped_position=bumped_position,
    page_size_options=(25, 50, 100, 200),
    default_size=50,
    enable_search=True,
    key="npv_minimal"
)
