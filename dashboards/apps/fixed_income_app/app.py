# examples/alm_app/app.py
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import os

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent.parent))

from mainsequence.dashboards.streamlit.scaffold import AppConfig, run_app
from mainsequence.dashboards.streamlit.core.registry import autodiscover
from dashboards.apps.fixed_income_app.context import  AppContext,build_context
from dashboards.components.engine_status import (
    collect_engine_meta_from_ctx,
    render_engine_status,
)
from dashboards.apps.fixed_income_app import extensions as fi_ext  # noqa: F401
fi_ext.bootstrap()
from dashboards.core.data_nodes import get_app_data_nodes
from dashboards.core.theme import register_theme
register_theme()

# --- Route selection: asset detail only reachable via query param -------------
def _route_selector(qp) -> str:
    return "asset_detail" if "asset_id" in qp else "curve_positions"

# --- Header renderer (domain-aware, uses ctx) --------------------------------
def _render_header(ctx: AppContext) -> None:
    st.title("Fixed Income Dashboard")
    st.caption(f"Valuation date: **{ctx.val_date.isoformat()}** — currency: **{ctx.currency_symbol.strip()}**")

    deps = get_app_data_nodes()
    render_engine_status(
        collect_engine_meta_from_ctx(ctx),
        app_sections={"Data nodes": deps.as_mapping()},
        mode="sticky_bar",          # << clean sticky bar (not floating overlay)
        title="Pricing engine",
        open=False,
    )


# --- Session seeding (domain defaults) ---------------------------------------
def _init_session(ss) -> None:


    default_cfg = str((ROOT.parent.parent.parent / "data/dump_position_example.json").resolve())
    ss.setdefault("cfg_path", default_cfg)

# Register all views in this package (side effects of @register_page)
autodiscover("dashboards.apps.fixed_income_app.views")

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
