# dashboards/bond_real_option_analysis/app.py
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent.parent))

from mainsequence.dashboards.streamlit.scaffold import AppConfig, run_app
from mainsequence.dashboards.streamlit.core.registry import autodiscover
from dashboards.apps.bond_real_option_analysis.context import build_context, AppContext
from dashboards.core.theme import register_theme
register_theme()

# One-page app
def _route_selector(qp) -> str:
    return "bond_option"

def _render_header(ctx: AppContext) -> None:
    st.title("Bond Real Option — Optimal Liquidation (LSM)")
    st.caption(f"Valuation date: **{ctx.val_date.isoformat()}**")

def _init_session(ss) -> None:
    default_cfg = str((ROOT.parent / "position.json").resolve())
    ss.setdefault("cfg_path", default_cfg)

def _build_context_route_aware(session_state) -> AppContext:
    cfg_path = session_state.get("cfg_path")
    ctx, _ = build_context(cfg_path)
    return ctx

# Register views in this package
autodiscover("dashboards.bond_real_option_analysis.views")

cfg = AppConfig(
    title="Bond Real Option Analysis",
    build_context=_build_context_route_aware,
    route_selector=_route_selector,
    render_header=_render_header,
    init_session=_init_session,
    default_page="bond_option",
)

run_app(cfg)
