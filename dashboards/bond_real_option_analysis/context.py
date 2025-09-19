# dashboards/bond_real_option_analysis/context.py
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st
import QuantLib as ql

from dashboards.components.position_loader import load_position_cached, instantiate_position
from dashboards.curves.bumping import BumpSpec, build_curves_for_ui

def qld(d: dt.date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)

@st.cache_data(show_spinner=False)
def read_json(path: str | Path) -> Dict[str, Any]:
    import json
    with open(path, "r") as fh:
        return json.load(fh)

@dataclass
class AppContext:
    cfg: Dict[str, Any]
    val_date: dt.date
    ts_market: ql.YieldTermStructureHandle   # base curve (market)
    position: Any                             # src.instruments.position.Position

def build_context(config_path: str | Path) -> Tuple[AppContext, BumpSpec]:
    cfg = read_json(config_path)
    val_date = dt.date.fromisoformat(cfg["valuation"]["valuation_date"])

    # Build the *base* (market) curve using your existing facade with zero bumps
    spec = BumpSpec(keyrate_bp={}, parallel_bp=0.0)
    ts_base, _, _, _ = build_curves_for_ui(qld(val_date), spec)

    # Load position template (cached on file content), then instantiate with base curve & date
    _cfg_dict, template, _sig = load_position_cached(config_path)
    position = instantiate_position(template, curve=ts_base, valuation_date=val_date)

    ctx = AppContext(cfg=cfg, val_date=val_date, ts_market=ts_base, position=position)
    return ctx, spec
