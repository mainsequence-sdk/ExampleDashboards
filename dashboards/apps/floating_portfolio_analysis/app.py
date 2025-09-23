# dashboards/apps/floating_portfolio_analysis/app.py
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

# Ensure repo root is importable (same as before)
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent.parent))

from dashboards.core.theme import register_theme

# Optional: your theme injection if needed
register_theme()

st.set_page_config(page_title="Fixed Income Position Dashboard", page_icon="📊", layout="wide")

st.title("Fixed Income Position Dashboard")
st.caption("Use the sidebar to open a page.")

# You can add quick links:
try:
    st.link_button("Open Curve, Stats & Positions", url="pages/01_Curve_Stats_Positions")
    st.link_button("Open Data Nodes", url="pages/02_Data_Nodes_Graph")
except Exception:
    # Fallback for older Streamlit (no link_button)
    st.markdown("- [Curve, Stats & Positions](pages/01_Curve_Stats_Positions)")
    st.markdown("- [Data Nodes](pages/02_Data_Nodes_Graph)")

st.info(
    "This template uses Streamlit’s native multipage routing. "
    "All pages live in the `pages/` folder; no registries or autodiscovery."
)
