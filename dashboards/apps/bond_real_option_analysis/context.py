# dashboards/bond_real_option_analysis/context.py
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st
import QuantLib as ql

from dashboards.core.ql import qld
from dashboards.components.position_loader import load_position_cached, instantiate_position
from dashboards.curves.bumping import BumpSpec, build_curves_for_ui
from mainsequence.instruments.settings import TIIE_28_UID

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
    position: Any                             # mainsequence.instruments.instruments.position.Position

def build_context(config_path: str | Path) -> Tuple[AppContext, BumpSpec]:
    cfg = read_json(config_path)
    val_date = dt.date.fromisoformat(cfg["valuation"]["valuation_date"])

    # Zero bumps (market curves)
    spec = BumpSpec(keyrate_bp={}, parallel_bp=0.0)

    # Load template and discover indices present
    _cfg_dict, template, _sig = load_position_cached(config_path)
    present_indices = sorted({
        getattr(ln.instrument, "floating_rate_index_name", None)
        for ln in template.lines
        if getattr(ln.instrument, "floating_rate_index_name", None) is not None
    }) or [TIIE_28_UID]

    # Build a base curve for every index we need
    curve_map: dict[str, ql.YieldTermStructureHandle] = {}
    for idx in present_indices:
        ts_base, _, _, _ = build_curves_for_ui(qld(val_date), spec, index_identifier=idx)
        curve_map[idx] = ts_base

    # Instantiate the position with the index→curve map (new API)
    position = instantiate_position(template, index_curve_map=curve_map, valuation_date=val_date)

    # Keep one handle for backward-compat (views should prefer the instrument's own curve)
    ts_market = next(iter(curve_map.values()))

    ctx = AppContext(cfg=cfg, val_date=val_date, ts_market=ts_market, position=position)
