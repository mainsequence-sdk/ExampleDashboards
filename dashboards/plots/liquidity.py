from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_liquidity_ladder(df: pd.DataFrame, currency_symbol: str = "$",
                          title: str = "Liquidity Ladder (next 12 months)") -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["bucket"], y=df["inflow"],  name="Inflows",  marker_color="#58D68D",
                hovertemplate=f"%{{x|%b %Y}}<br>Inflows: {currency_symbol}%{{y:,.2f}}<extra></extra>")
    fig.add_bar(x=df["bucket"], y=-df["outflow"], name="Outflows", marker_color="#EC7063",
                hovertemplate=f"%{{x|%b %Y}}<br>Outflows: {currency_symbol}%{{y:,.2f}}<extra></extra>")
    fig.add_scatter(x=df["bucket"], y=df["net"],     name="Net",        mode="lines+markers")
    fig.add_scatter(x=df["bucket"], y=df["cum_net"], name="Cumulative", mode="lines")
    fig.update_layout(barmode="relative", title=title, yaxis_title=f"Amount ({currency_symbol})", height=480)
    return fig

def plot_repricing_gap(df: pd.DataFrame, currency_symbol: str = "$") -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["bucket"], y=df["IRSA"], name="IRSA (Assets)",       marker_color="#5DADE2")
    fig.add_bar(x=df["bucket"], y=-df["IRSL"], name="IRSL (Liabilities)", marker_color="#F5B041")
    fig.add_scatter(x=df["bucket"], y=df["CUM_GAP"], name="Cumulative GAP", mode="lines+markers")
    fig.update_layout(barmode="relative", title="Repricing GAP by Bucket",
                      yaxis_title=f"Exposure ({currency_symbol})", height=480)
    return fig

def plot_eve_bars(eve: dict, currency_symbol: str = "$") -> go.Figure:
    keys = list(eve.keys())
    vals = [eve[k] for k in keys]
    fig = go.Figure()
    fig.add_bar(x=keys, y=vals, marker_color=["#EC7063" if "+" in k else "#58D68D" for k in keys])
    fig.update_layout(title="ΔEVE (parallel shocks)", yaxis_title=f"ΔPV ({currency_symbol})", height=360)
    return fig

def plot_eve_bars_compare(eve_base: dict, eve_bump: dict, currency_symbol: str = "$") -> go.Figure:
    cats = list(eve_base.keys())
    bvals = [eve_base[c] for c in cats]
    uvals = [eve_bump.get(c, np.nan) for c in cats]
    fig = go.Figure()
    fig.add_bar(x=cats, y=bvals, name="Base",  marker_color="#5DADE2")
    fig.add_bar(x=cats, y=uvals, name="Bumped", marker_color="#F5B041")
    fig.update_layout(barmode="group", title="ΔEVE (parallel shocks) — Base vs Bumped",
                      yaxis_title=f"ΔPV ({currency_symbol})", height=360)
    return fig
