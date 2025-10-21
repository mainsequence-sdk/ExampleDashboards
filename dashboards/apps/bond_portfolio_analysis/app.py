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
    st.page_link("pages/01_curve_and_positions.py", label="Open Curve, Stats & Positions")
    st.page_link("pages/02_data_nodes_dependencies.py", label="Open Data Nodes")
except Exception:
    # Fallback for older Streamlit (no page_link)
    st.markdown("- [Curve, Stats & Positions](pages/01_curve_and_positions)")
    st.markdown("- [Data Nodes](pages/02_data_nodes_dependencies)")

st.info(
    "This template uses Streamlit’s native multipage routing. "
    "All pages live in the `pages/` folder; no registries or autodiscovery."
)
