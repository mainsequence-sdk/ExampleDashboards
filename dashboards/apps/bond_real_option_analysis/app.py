# dashboards/bond_real_option_analysis/app.py
from __future__ import annotations
import sys, datetime as dt
from dataclasses import dataclass
from pathlib import Path
import streamlit as st

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent.parent))

from mainsequence.dashboards.streamlit.scaffold import AppConfig, run_app
from mainsequence.dashboards.streamlit.core.registry import autodiscover
from dashboards.core.theme import register_theme

register_theme()

# ---------- Minimal app context (no curves here) ----------
@dataclass
class AppContext:
    cfg: dict
    val_date: dt.date

def build_context() -> AppContext:
    """
    JSON-free, curve-free context:
      • valuation date comes from session (defaults to today)
      • NO curves are built here; views will build curves AFTER an asset is selected
    """
    val_date = st.session_state.get("valuation_date", dt.date.today())
    st.session_state.setdefault("valuation_date", val_date)
    return AppContext(cfg={}, val_date=val_date)

# ---------- Scaffold hooks ----------
def _route_selector(_qp) -> str:
    return "bond_option"

def _render_header(ctx: AppContext) -> None:
    st.title("Bond Real Option — Optimal Liquidation (LSM)")
    st.caption(f"Valuation date: **{ctx.val_date.isoformat()}**")

def _init_session(ss) -> None:
    ss.setdefault("valuation_date", dt.date.today())

def _build_context_route_aware(_session_state) -> AppContext:
    # Build minimal context without curves
    return build_context()

# Register views in this package
autodiscover("dashboards.apps.bond_real_option_analysis.views")

cfg = AppConfig(
    title="Bond Real Option Analysis",
    build_context=_build_context_route_aware,
    route_selector=_route_selector,
    render_header=_render_header,
    init_session=_init_session,
    default_page="bond_option",
)

run_app(cfg)
