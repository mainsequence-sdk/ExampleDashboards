from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dashboards.core.formatters import fmt_ccy

def table_from_df(df: pd.DataFrame,
                  title: str = "Summary",
                  currency_symbol: str = "$",
                  currency_cols: list[str] | None = None,
                  precision: int = 2) -> go.Figure:
    currency_cols = currency_cols or ["base", "bumped", "delta"]
    df2 = df.copy()
    for c in currency_cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda v: fmt_ccy(v, currency_symbol, precision))
    if "units" in df2.columns:
        df2["units"] = df2["units"].apply(lambda v: f"{v:,.2f}" if isinstance(v, (int, float, np.number)) else v)
    disp = df2.astype(str)
    cols = list(disp.columns)
    right_cols = set(currency_cols + (["units"] if "units" in disp.columns else []))
    align = ["right" if c in right_cols else "left" for c in cols]
    fig = go.Figure(data=[go.Table(
        header=dict(values=cols, fill_color="#2B2F36", align=align),
        cells=dict(values=[disp[c] for c in cols], fill_color="#111417", align=align)
    )])
    fig.update_layout(title=title, height=380)
    return fig

def table_kpis(kv: dict, currency_symbol: str = "$") -> go.Figure:
    keys = list(kv.keys())
    vals = []
    for k in keys:
        v = kv[k]
        if isinstance(v, (int, float, np.floating)):
            if "LCR" in k or "NSFR" in k:
                vals.append(f"{v:,.2f}×")
            elif "Duration" in k or "D_" in k:
                vals.append(f"{v:,.2f}y")
            else:
                vals.append(f"{currency_symbol}{v:,.2f}")
        else:
            vals.append(str(v))
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Metric", "Value"], fill_color="#2B2F36", align="left"),
        cells=dict(values=[keys, vals], fill_color="#111417", align="left")
    )])
    fig.update_layout(title="ALM KPIs", height=380)
    return fig
