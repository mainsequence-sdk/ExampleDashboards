from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_cashflows(base_cf: pd.DataFrame, bumped_cf: pd.DataFrame,
                   units_by_id: dict[str, float],
                   currency_symbol: str = "$",
                   add_time_slider: bool = True) -> go.Figure:
    def scale(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["units"] = df["ins_id"].map(units_by_id).fillna(1.0)
        df["amount"] = df["amount"] * df["units"]
        return df

    g0 = scale(base_cf).groupby("pay_date", as_index=False)["amount"].sum().rename(columns={"amount": "amount_base"})
    g1 = scale(bumped_cf).groupby("pay_date", as_index=False)["amount"].sum().rename(columns={"amount": "amount_bumped"})
    m = g0.merge(g1, on="pay_date", how="outer").fillna(0.0)
    m["amount_delta"] = m["amount_bumped"] - m["amount_base"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.65, 0.35])
    fig.add_bar(x=m["pay_date"], y=m["amount_base"], name="Base CF", marker_color="#5DADE2",
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>Base: {currency_symbol}%{{y:,.2f}}<extra></extra>", row=1, col=1)
    fig.add_bar(x=m["pay_date"], y=m["amount_bumped"], name="Bumped CF", marker_color="#F5B041",
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>Bumped: {currency_symbol}%{{y:,.2f}}<extra></extra>", row=1, col=1)
    fig.add_bar(x=m["pay_date"], y=m["amount_delta"], name="Δ Amount", marker_color="#EC7063",
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>Δ: {currency_symbol}%{{y:,.2f}}<extra></extra>", row=2, col=1)
    fig.update_yaxes(title_text=f"Amount ({currency_symbol})", tickformat=",.2f", row=1, col=1)
    fig.update_yaxes(title_text=f"Δ Amount ({currency_symbol})", tickformat=",.2f", row=2, col=1)
    fig.update_xaxes(title_text="Payment date", row=2, col=1)
    fig.update_layout(title="Portfolio Cashflows — Base vs Bumped", height=650, barmode="group")
    fig.update_xaxes(matches="x")
    if add_time_slider:
        fig.update_xaxes(row=1, col=1, rangeselector=dict(buttons=[
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=2, label="2Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]))
        fig.update_xaxes(row=2, col=1, rangeslider=dict(visible=True, bgcolor="#0B1016",
                         bordercolor="#2B2F36", borderwidth=1, thickness=0.12), type="date")
    return fig

def plot_npvs(npv_base: dict[str, float], npv_bumped: dict[str, float],
              title: str = "NPV (Base vs Bumped)") -> go.Figure:
    keys = list(npv_base.keys())
    b0 = np.array([npv_base[k] for k in keys])
    b1 = np.array([npv_bumped[k] for k in keys])
    d = b1 - b0
    fig = go.Figure()
    fig.add_bar(x=keys, y=b0, name="Base")
    fig.add_bar(x=keys, y=b1, name="Bumped")
    fig.add_scatter(x=keys, y=d, name="Δ", mode="markers+text",
                    text=[f"{x:,.0f}" for x in d], textposition="top center",
                    marker=dict(color="#EC7063", size=10))
    fig.update_layout(title=title, barmode="group", height=420, yaxis_title="PV")
    return fig

def plot_cashflows_compare(
    cfA: pd.DataFrame, cfB: pd.DataFrame,
    currency_symbol: str = "$",
    add_time_slider: bool = True
) -> go.Figure:
    """
    Compare two portfolios' net cashflows (already unit-scaled).
    Expects columns: pay_date (date/datetime), amount (float).
    """
    a = (cfA or pd.DataFrame(columns=["pay_date","amount"])).copy()
    b = (cfB or pd.DataFrame(columns=["pay_date","amount"])).copy()
    if not a.empty:
        a["pay_date"] = pd.to_datetime(a["pay_date"])
    if not b.empty:
        b["pay_date"] = pd.to_datetime(b["pay_date"])

    gA = a.groupby("pay_date", as_index=False)["amount"].sum().rename(columns={"amount": "amount_A"})
    gB = b.groupby("pay_date", as_index=False)["amount"].sum().rename(columns={"amount": "amount_B"})
    m = gA.merge(gB, on="pay_date", how="outer").fillna(0.0)
    m["amount_delta"] = m["amount_B"] - m["amount_A"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.65, 0.35])
    fig.add_bar(x=m["pay_date"], y=m["amount_A"], name="Portfolio A", marker_color="#5DADE2",
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>A: {currency_symbol}%{{y:,.2f}}<extra></extra>", row=1, col=1)
    fig.add_bar(x=m["pay_date"], y=m["amount_B"], name="Portfolio B", marker_color="#F5B041",
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>B: {currency_symbol}%{{y:,.2f}}<extra></extra>", row=1, col=1)
    fig.add_bar(x=m["pay_date"], y=m["amount_delta"], name="Δ (B − A)", marker_color="#EC7063",
                hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>Δ: {currency_symbol}%{{y:,.2f}}<extra></extra>", row=2, col=1)

    fig.update_yaxes(title_text=f"Amount ({currency_symbol})", tickformat=",.2f", row=1, col=1)
    fig.update_yaxes(title_text=f"Δ Amount ({currency_symbol})", tickformat=",.2f", row=2, col=1)
    fig.update_xaxes(title_text="Payment date", row=2, col=1)
    fig.update_layout(title="Portfolio Cashflows — A vs B", height=650, barmode="group")
    fig.update_xaxes(matches="x")
    if add_time_slider:
        fig.update_xaxes(row=1, col=1, rangeselector=dict(buttons=[
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=2, label="2Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]))
        fig.update_xaxes(row=2, col=1, rangeslider=dict(visible=True, bgcolor="#0B1016",
                         bordercolor="#2B2F36", borderwidth=1, thickness=0.12), type="date")
    return fig