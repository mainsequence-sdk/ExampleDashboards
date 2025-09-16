# examples/alm_app/app.py
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
from dashboards.fixed_income_app.context import  AppContext,build_context

# --- Route selection: asset detail only reachable via query param -------------
def _route_selector(qp) -> str:
    return "asset_detail" if "asset_id" in qp else "curve_positions"

# --- Header renderer (domain-aware, uses ctx) --------------------------------
def _render_header(ctx: AppContext) -> None:
    st.title("Fixed Income Dashboard — Example")
    st.caption(f"Valuation date: **{ctx.val_date.isoformat()}** — currency: **{ctx.currency_symbol.strip()}**")

# --- Session seeding (domain defaults) ---------------------------------------
def _init_session(ss) -> None:
    default_cfg = str((ROOT.parent / "position.json").resolve())
    ss.setdefault("cfg_path", default_cfg)

# Register all views in this package (side effects of @register_page)
autodiscover("ui.fixed_income_app.views")

# Build the app config and run (no logo/icon passed; scaffold supplies defaults)
cfg = AppConfig(
    title="Fixed Income Position Dashboard",
    build_context=lambda ss: _build_context_route_aware(ss),
    route_selector=_route_selector,
    render_header=_render_header,
    init_session=_init_session,
    default_page="curve_positions",
)
def _build_context_route_aware(session_state) -> AppContext:
    """Build a lighter context when hitting /?asset_id=... directly."""
    cfg_path = session_state.get("cfg_path")
    route = _route_selector(st.query_params)
    mode = "asset_detail" if route == "asset_detail" else "full"
    ctx, _ = build_context(cfg_path, mode=mode)  # <— build_context must accept mode
    return ctx


run_app(cfg)
