# alm_dashboard_base/core/context.py
from __future__ import annotations

import json
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st
import QuantLib as ql

# Your existing utilities / components

from ui.curves.bumping import build_curves_for_ui as build_bumped_curves, KEYRATE_GRID_TIIE, BumpSpec
from ui.components.curve_bump import curve_bump_controls
from ui.components.npv_table import st_position_npv_table_paginated
from src.instruments.position import Position

# ---------- cacheable helpers ----------
@st.cache_data(show_spinner=False)
def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)

def qld(d: dt.date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)

# ---------- portfolio stats ----------
import pandas as pd
import numpy as np

def position_total_npv(position: Position) -> float:
    return float(sum(float(line.units) * float(line.instrument.price()) for line in position.lines))

def _agg_net_cashflows(position: Position) -> pd.DataFrame:
    rows = []
    for line in position.lines:
        inst = line.instrument
        units = float(line.units)
        s = getattr(inst, "get_net_cashflows", None)
        if callable(s):
            ser = s()
            if isinstance(ser, pd.Series):
                df = ser.to_frame("amount").reset_index()
                df = df.rename(columns={"index": "payment_date"}) if "payment_date" not in df.columns else df
                df["amount"] = df["amount"].astype(float) * units
                rows.append(df[["payment_date", "amount"]])
                continue
        g = getattr(inst, "get_cashflows", None)
        if callable(g):
            flows = g() or {}
            flat = []
            for _, items in flows.items():
                for cf in (items or []):
                    pay = cf.get("payment_date") or cf.get("date") or cf.get("pay_date") or cf.get("fixing_date")
                    amt = cf.get("amount")
                    if pay is None or amt is None:
                        continue
                    flat.append({"payment_date": pd.to_datetime(pay).date(), "amount": float(amt) * units})
            if flat:
                rows.append(pd.DataFrame(flat))
    if not rows:
        return pd.DataFrame(columns=["payment_date", "amount"])
    df_all = pd.concat(rows, ignore_index=True)
    df_all["payment_date"] = pd.to_datetime(df_all["payment_date"]).dt.date
    return df_all.groupby("payment_date", as_index=False)["amount"].sum()

def position_carry_to_cutoff(position: Position, valuation_date: dt.date, cutoff: dt.date) -> float:
    cf = _agg_net_cashflows(position)
    if cf.empty:
        return 0.0
    mask = (cf["payment_date"] > valuation_date) & (cf["payment_date"] <= cutoff)
    return float(cf.loc[mask, "amount"].sum())

def portfolio_stats(position: Position, bumped_position: Position,
                    valuation_date: dt.date, cutoff: dt.date) -> Dict[str, float]:
    npv_b = position_total_npv(position)
    npv_u = position_total_npv(bumped_position)
    carry_b = position_carry_to_cutoff(position, valuation_date, cutoff)
    carry_u = position_carry_to_cutoff(bumped_position, valuation_date, cutoff)
    return {
        "npv_base": npv_b, "npv_bumped": npv_u, "npv_delta": npv_u - npv_b,
        "carry_base": carry_b, "carry_bumped": carry_u, "carry_delta": carry_u - carry_b,
    }

# ---------- app context ----------
@dataclass
class AppContext:
    cfg: Dict[str, Any]
    val_date: dt.date
    currency_symbol: str
    index_hint: str
    ts_base: ql.YieldTermStructureHandle
    ts_bump: ql.YieldTermStructureHandle
    nodes_base: list[dict]
    nodes_bump: list[dict]
    position: Position
    bumped_position: Position
    carry_cutoff: dt.date

def build_context(config_path: str | Path, spec: BumpSpec | None = None) -> Tuple[AppContext, BumpSpec]:
    cfg = read_json(config_path)

    val_date = dt.date.fromisoformat(cfg["valuation"]["valuation_date"])
    currency_symbol = cfg.get("alm_assumptions", {}).get("currency_symbol", "USD$ ")

    # First pass (no bumps) just to discover index/nodes (UI-free)
    spec0 = BumpSpec(keyrate_bp=cfg.get("curve_bumps_bp", {}), parallel_bp=0.0)
    ts_base0, ts_bump0, nodes_base0, nodes_bump0, index_hint = build_bumped_curves(qld(val_date), spec0)

    # Final spec:
    #  1) explicit argument wins,
    #  2) otherwise whatever the curve page stored in session,
    #  3) otherwise defaults from the config file.
    if spec is None:
        ss = st.session_state.get("curve_bump_spec")
        if isinstance(ss, dict):
            spec = BumpSpec(
                keyrate_bp=ss.get("keyrate_bp", {}),
                parallel_bp=float(ss.get("parallel_bp", 0.0)),
            )
        else:
            spec = spec0

    # Curves for the chosen spec
    ts_base, ts_bump, nodes_base, nodes_bump, _ = build_bumped_curves(qld(val_date), spec)

    # Position (base)
    position = Position.from_json_dict(cfg["position"])
    for line in position.lines:
        line.instrument.valuation_date = val_date
        line.instrument.reset_curve(ts_base)

    # Bumped position
    from examples.alm.utils import get_bumped_position
    bumped_position = get_bumped_position(position=position, bump_curve=ts_bump)

    default_days = int(cfg["valuation"].get("cashflow_cutoff_days", 365))
    carry_cutoff = val_date + dt.timedelta(days=default_days)

    ctx = AppContext(
        cfg=cfg, val_date=val_date, currency_symbol=currency_symbol, index_hint=index_hint,
        ts_base=ts_base, ts_bump=ts_bump, nodes_base=nodes_base, nodes_bump=nodes_bump,
        position=position, bumped_position=bumped_position, carry_cutoff=carry_cutoff
    )
    return ctx, spec
