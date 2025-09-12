# ui/dashboard_base/app.py
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

# ensure pages self-register
from ui.dashboard_base.views import curve_and_positions, asset_detail  # noqa: F401
from ui.dashboard_base.core.registry import get_page
from ui.dashboard_base.core.context import build_context
from ui.dashboard_base.core.theme import inject_css_for_dark_accents

ASSETS = ROOT / "assets"
FAVICON = ASSETS / "favicon.png"
LOGO    = ASSETS / "logo.png"

st.set_page_config(page_title="Base Dashboard", page_icon=str(FAVICON), layout="wide")
st.logo(str(LOGO), icon_image=str(FAVICON))
inject_css_for_dark_accents()

# Always hide Streamlit's auto multipage nav
st.markdown("<style>[data-testid='stSidebarNav']{display:none!important}</style>", unsafe_allow_html=True)

# Session defaults
default_cfg = str((ROOT.parent / "position.json").resolve())
st.session_state.setdefault("cfg_path", default_cfg)

# Decide target page: asset detail is only reachable via query param
qp = st.query_params
target_slug = "asset_detail" if "asset_id" in qp else "curve_positions"

# For pages that don't own a sidebar, hard-hide it (container + burger)
if target_slug != "curve_positions":
    st.markdown("""
        <style>
          [data-testid="stSidebar"]{display:none!important;}
          [data-testid="stSidebarCollapseControl"]{display:none!important;}
        </style>
    """, unsafe_allow_html=True)

# Build a UI‑free context.
# If the curve view has written a bump spec to session_state, build_context will use it.
ctx, _spec = build_context(st.session_state["cfg_path"], spec=None)

# Header
st.title("ALM Dashboard — Base")
st.caption(f"Valuation date: **{ctx.val_date.isoformat()}** — currency: **{ctx.currency_symbol.strip()}**")

# Route
page = get_page(target_slug)
page.render(ctx)
