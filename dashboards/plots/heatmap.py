from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dashboards.analytics.correlation import serialized_correlation_core

def plot_serialized_correlation(
    data,
    mode: str = "returns",
    asset_to_highlight: str | None = None,
    cluster_method: str = "average",
    width: int = 1200,
):
    corr_ord, Z, labels_ord = serialized_correlation_core(data, mode=mode, cluster_method=cluster_method)

    n = len(labels_ord)
    xvals = np.arange(n); yvals = np.arange(n)
    fig_heat = go.Figure(
        go.Heatmap(
            z=corr_ord.values, x=xvals, y=yvals,
            zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
            colorbar=dict(title="ρ"),
            hovertemplate="(%{y}, %{x}) • ρ=%{z:.3f}<extra></extra>",
        )
    )
    fig_heat.update_xaxes(tickmode="array", tickvals=xvals, ticktext=labels_ord, tickangle=45)
    fig_heat.update_yaxes(tickmode="array", tickvals=yvals, ticktext=labels_ord, autorange="reversed")
    fig_heat.update_layout(title="Serialized correlation (hierarchical order)",
                           width=width, height=width, hovermode="closest")

    if asset_to_highlight in labels_ord:
        idx = int(np.where(labels_ord == asset_to_highlight)[0][0])
        hl_color = "gold"
        fig_heat.add_shape(type="rect", x0=idx-0.5, x1=idx+0.5, y0=-0.5, y1=n-0.5,
                           line=dict(color=hl_color, width=2), fillcolor=hl_color, opacity=0.18, layer="above")
        fig_heat.add_shape(type="rect", x0=-0.5, x1=n-0.5, y0=idx-0.5, y1=idx+0.5,
                           line=dict(color=hl_color, width=2), fillcolor=hl_color, opacity=0.18, layer="above")
        fig_heat.add_trace(go.Scatter(
            x=[idx], y=[idx], mode="markers",
            marker=dict(size=10, symbol="x", line=dict(width=2), color=hl_color),
            name=f"★ {asset_to_highlight}", showlegend=False
        ))
        fig_heat.add_annotation(x=idx, y=-1, yref="paper", text=f"★ {asset_to_highlight}",
                                showarrow=False, yanchor="top")
        fig_heat.add_annotation(x=-0.02, xref="paper", y=idx, text=f"★ {asset_to_highlight}",
                                showarrow=False, xanchor="right")

    fig_dend = ff.create_dendrogram(
        corr_ord.values, orientation="bottom", labels=list(labels_ord),
        linkagefun=lambda _: Z
    )
    fig_dend.update_layout(title="Hierarchical clustering (dendrogram)",
                           width=width, height=340, hovermode="x unified")
    return corr_ord, Z, fig_heat, fig_dend
