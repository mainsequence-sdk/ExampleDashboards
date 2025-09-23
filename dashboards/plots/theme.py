from __future__ import annotations
from pydantic import BaseModel, ConfigDict
import plotly.graph_objects as go

class GraphTheme(BaseModel):
    """Unified look for all Plotly figures (reusable everywhere)."""
    model_config = ConfigDict(extra="forbid")

    # Layout defaults
    template: str = "plotly_dark"
    paper_bgcolor: str = "#0E1117"
    plot_bgcolor: str = "#0E1117"
    font_family: str = "Inter, Segoe UI, Helvetica, Arial, sans-serif"
    font_color: str = "#E6E9EF"

    # Accent palette
    accent: str = "#76B7FB"
    gold: str = "#D4AF37"
    silver: str = "#C0C0C0"
    bronze: str = "#CD7F32"
    gray: str = "rgba(160,160,160,0.55)"

    # Legend defaults (legend-at-bottom and isolate on click)
    legend_orientation: str = "h"
    legend_x: float = 0.0
    legend_y: float = -0.20
    legend_itemclick: str = "toggleothers"
    legend_itemdoubleclick: str = "toggle"

    # Sensible sizing if a figure didn’t specify width/height
    default_width: int = 1100
    default_height: int = 400

def apply_graph_theme(fig: go.Figure, theme: GraphTheme | None = None) -> go.Figure:
    """Apply a consistent Plotly layout theme to a figure (in‑place)."""
    t = theme or GraphTheme()
    fig.update_layout(
        template=t.template,
        paper_bgcolor=t.paper_bgcolor,
        plot_bgcolor=t.plot_bgcolor,
        font=dict(family=t.font_family, color=t.font_color),
        legend=dict(
            orientation=t.legend_orientation,
            x=t.legend_x,
            y=t.legend_y,
            itemclick=t.legend_itemclick,
            itemdoubleclick=t.legend_itemdoubleclick,
        ),
    )
    if fig.layout.width is None:
        fig.update_layout(width=t.default_width)
    if fig.layout.height is None:
        fig.update_layout(height=t.default_height)
    return fig
