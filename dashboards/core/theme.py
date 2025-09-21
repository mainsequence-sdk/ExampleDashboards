# dashboards/core/theme.py
from __future__ import annotations
import plotly.graph_objects as go
import plotly.io as pio

def register_theme(name: str = "ms_dark") -> None:
    if name in pio.templates:
        pio.templates.default = name
        return
    tpl = pio.templates["plotly_dark"].to_plotly_json()
    layout = tpl.setdefault("layout", {})
    layout.update({
        "font": {"family": "Inter, Segoe UI, Helvetica, Arial", "size": 13, "color": "#EAECEE"},
        "paper_bgcolor": "#0E1216",
        "plot_bgcolor": "#0E1216",
        "colorway": ["#5DADE2", "#F5B041", "#58D68D", "#AF7AC5", "#EC7063", "#F4D03F"],
        "hoverlabel": {"bgcolor": "#171C22", "bordercolor": "#2B2F36", "font": {"color": "#EAECEE"}},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "xaxis": {"showgrid": True, "gridcolor": "#2B2F36"},
        "yaxis": {"showgrid": True, "gridcolor": "#2B2F36"},
    })
    pio.templates[name] = go.layout.Template(**tpl)
    pio.templates.default = name
