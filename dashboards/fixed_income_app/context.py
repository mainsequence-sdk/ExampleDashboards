# examples/alm_app/context.py
from __future__ import annotations

import json
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st
import QuantLib as ql
import pandas as pd

from dashboards.curves.bumping import build_curves_for_ui as build_bumped_curves, BumpSpec
from mainsequence.instruments.instruments.position import Position
from dashboards.components.position_loader import load_position_cached, instantiate_position

# ---------- cacheable helpers ----------
@st.cache_data(show_spinner=False)
def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)

def qld(d: dt.date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)

# ---------- portfolio stats ----------
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
            if ser.empty:
                continue
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
    ts_base: ql.YieldTermStructureHandle
    ts_bump: ql.YieldTermStructureHandle
    nodes_base: list[dict]
    nodes_bump: list[dict]
    position: Position
    bumped_position: Position
    carry_cutoff: dt.date

def build_context(config_path: str | Path | None = None, *, mode: str = "full") -> tuple[AppContext, BumpSpec]:
    cfg = read_json(config_path)

    val_date = dt.date.fromisoformat(cfg["valuation"]["valuation_date"])
    currency_symbol = "MXN$"

    ql.Settings.instance().evaluationDate = qld(val_date)
    # Final spec: session overrides file
    ss = st.session_state.get("curve_bump_spec")
    spec = BumpSpec(
        keyrate_bp=(ss or {}).get("keyrate_bp", cfg.get("curve_bumps_bp", {})),
        parallel_bp=float((ss or {}).get("parallel_bp", 0.0)),
    )

    # Build curves ONCE (internal curve caches will also help on reruns)
    ts_base, ts_bump, nodes_base, nodes_bump = build_bumped_curves(qld(val_date), spec)

    # ---- FAST position creation (cached on file content) ----
    # parse the big JSON once and reuse the template
    cfg_dict, template, _sig = load_position_cached(config_path)

    # Base position for whichever page needs it
    position = instantiate_position(template, curve=ts_base, valuation_date=val_date)

    # Build bumped position by re‑instantiating from the same template.
    # This guarantees a fresh pricing engine + discount curve for every instrument.

    if mode == "full":
        bumped_position = instantiate_position(template, curve=ts_bump, valuation_date=val_date)
    else:
        # asset_detail computes a per‑line bump locally; keep base position only
        bumped_position = position

    default_days = int(cfg["valuation"].get("cashflow_cutoff_days", 365))
    carry_cutoff = val_date + dt.timedelta(days=default_days)

    ctx = AppContext(
        cfg=cfg, val_date=val_date, currency_symbol=currency_symbol,
        ts_base=ts_base, ts_bump=ts_bump, nodes_base=nodes_base, nodes_bump=nodes_bump,
        position=position, bumped_position=bumped_position, carry_cutoff=carry_cutoff,

    )
    return ctx, spec

# The scaffold expects build_context(session_state) -> ctx.
# We wrap our build_context(file) so the example can keep session-driven behavior.
