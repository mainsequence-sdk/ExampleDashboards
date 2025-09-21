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

from dashboards.curves.bumping import build_curves_for_ui as build_bumped_curves, BumpSpec,curve_family_key
from dashboards.curves.z_spread import zspread_from_dirty_ccy,dirty_price_ccy_with_curve,make_zero_spreaded_handle
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
    base_curves: Dict[str,ql.YieldTermStructureHandle]
    bumped_curves: Dict[str,ql.YieldTermStructureHandle]
    position: Position
    bumped_position: Position
    carry_cutoff: dt.date

def build_context(config_path: str | Path | None = None, *, mode: str = "full") -> tuple[AppContext, BumpSpec]:
    # --- Read config + prepare bump spec ---
    if config_path == ":memory:":
        cfg = st.session_state.get("position_cfg_mem") or {
            "valuation": {
                "valuation_date": dt.date.today().isoformat(),
                "cashflow_cutoff_days": 365,
            }
        }
        loaded_position = st.session_state.get("position_template_mem")
        if loaded_position is None:
            raise RuntimeError("No in-memory position template found. Load a portfolio first.")
    else:
        cfg = read_json(config_path)
        # parse the big JSON once and reuse the template
        cfg_dict, loaded_position, _sig = load_position_cached(config_path)
        for line in loaded_position.lines:
            line.extra_market_info["dirty_price"]=100

    val_date = dt.date.fromisoformat(cfg["valuation"]["valuation_date"])
    currency_symbol = "MXN$"
    ql.Settings.instance().evaluationDate = qld(val_date)

    # Final spec: session overrides file
    # Final spec: session overrides (by family; accept old key for compatibility)
    by_family = (st.session_state.get("curve_bump_spec_by_family")
                 or st.session_state.get("curve_bump_spec_by_index"))

    indices_for_curves = sorted({
        getattr(l.instrument, "floating_rate_index_name", None)
        for l in loaded_position.lines
        if getattr(l.instrument, "floating_rate_index_name", None) is not None
    })

    # Helper: pick a representative *index UID* for a family that actually exists in the position
    def _rep_index_for_family(fam: str) -> str:
        for idx in indices_for_curves:
            if curve_family_key(idx) == fam:
                return idx
        # Shouldn’t happen because fams are derived from indices_for_curves, but keep a safe fallback
        return indices_for_curves[0]

    # --- Build curves once per family, then fan-out to all member indices ---
    fams = sorted({curve_family_key(i) for i in indices_for_curves})
    fam_curves: dict[str, tuple[ql.YieldTermStructureHandle, ql.YieldTermStructureHandle]] = {}

    # Choose a bump spec for the family:
    # 1) prefer explicit family key in the UI state; else
    # 2) take the first member index’s spec; else default (no bumps).
    for fam in fams:
        spec_cfg = (by_family or {}).get(fam, None)
        if spec_cfg is None:
            for idx in indices_for_curves:
                if curve_family_key(idx) == fam:
                    spec_cfg = (by_family or {}).get(idx, {})
                    break
        spec_cfg = spec_cfg or {}
        spec = BumpSpec(keyrate_bp=spec_cfg.get("keyrate_bp", {}),
                        parallel_bp=float(spec_cfg.get("parallel_bp", 0.0)))
        rep_idx = _rep_index_for_family(fam)
        ts_base, ts_bump, _, _ = build_bumped_curves(
            calc_date=qld(val_date), spec=spec, index_identifier=rep_idx
        )
        fam_curves[fam] = (ts_base, ts_bump)

        # Fan out to each concrete index UID
    base_curves, bumped_curves = {}, {}
    for idx in indices_for_curves:
        fam = curve_family_key(idx)
        base_curves[idx], bumped_curves[idx] = fam_curves[fam]

    # --- Instantiate positions from the template (no file I/O) ---
    position = instantiate_position(loaded_position, index_curve_map=base_curves, valuation_date=val_date)
    bumped_position = (instantiate_position(loaded_position, index_curve_map=bumped_curves, valuation_date=val_date)
                       if mode == "full" else position)

    # get zero spreads
    for line, bumped_line in zip(position.lines, bumped_position.lines):
        dirty_price = line.extra_market_info.get("dirty_price")
        if dirty_price is None:
            raise Exception(f"{line} does not has dirty price in extra_market_info")

        # The instrument already has the proper base curve assigned by instantiate_position
        base_curve_for_line = line.instrument.get_index_curve()

        # Force building of pricing engine if needed, then compute z-spread to match dirty price
        line.instrument.price()
        try:
            z = zspread_from_dirty_ccy(line.instrument._bond, dirty_price, discount_curve=base_curve_for_line)
        except Exception as e:
            z=0
        line.extra_market_info["z_spread"] = z
        bumped_line.extra_market_info["z_spread"] = z

    default_days = int(cfg["valuation"].get("cashflow_cutoff_days", 365))
    carry_cutoff = val_date + dt.timedelta(days=default_days)

    ctx = AppContext(
        cfg=cfg, val_date=val_date, currency_symbol=currency_symbol,
        base_curves=base_curves, bumped_curves=bumped_curves,
        position=position, bumped_position=bumped_position, carry_cutoff=carry_cutoff,
    )
    return ctx, spec


# The scaffold expects build_context(session_state) -> ctx.
# We wrap our build_context(file) so the example can keep session-driven behavior.
