from __future__ import annotations

# =========================
# Imports (top-level only)
# =========================
import re
from datetime import datetime
from typing import Dict, Tuple, Iterable, Union, Sequence, Optional, List
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import numpy as np
import pandas as pd

import plotly.graph_objects as go
# Plot helpers are reusable from dashboards.plots; avoid app-local duplication
from dashboards.plots.heatmap import plot_serialized_correlation
from dashboards.analytics.correlation import compute_correlation, serialized_correlation_core
from mainsequence.tdag import APIDataNode
import streamlit as st





