# src/ux_utils.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import QuantLib as ql
from src.pricing_models.swap_pricer import make_tiie_28d_index  # add


# -------- Plot theme (consistent everywhere) --------

def register_theme(name: str = "ms_dark") -> None:
    # Start from the *full* plotly_dark template
    tpl = pio.templates["plotly_dark"].to_plotly_json()
    layout = tpl.setdefault("layout", {})

    colorway = ["#5DADE2", "#F5B041", "#58D68D", "#AF7AC5", "#EC7063", "#F4D03F"]

    # High-contrast dark palette
    layout.update({
        "font": {"family": "Inter, Segoe UI, Helvetica, Arial", "size": 13, "color": "#EAECEE"},
        "paper_bgcolor": "#0E1216",
        "plot_bgcolor":  "#0E1216",
        "colorway": colorway,
        "separators": ",.",  # 1,234.56 style
        "hoverlabel": {"bgcolor": "#171C22", "bordercolor": "#2B2F36", "font": {"color": "#EAECEE"}},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
    })

    # Axis defaults (all x/y inherit unless explicitly overridden)
    base_axis = {
        "showgrid": True,
        "gridcolor": "#2B2F36",
        "tickfont": {"color": "#EAECEE"},
        "title": {"font": {"color": "#EAECEE"}},
    }
    layout["xaxis"] = {
        **base_axis,
        # Make range *controls* readable in dark mode
        "rangeslider": {
            "visible": False,               # we will turn it on per-figure (bottom axis only)
            "bgcolor": "#0B1016",
            "bordercolor": "#2B2F36",
            "borderwidth": 1,
            "thickness": 0.12,              # fraction of axis domain
        },
        "rangeselector": {
            "bgcolor": "rgba(0,0,0,0)",     # transparent buttons background bar
            "activecolor": "#2B2F36",
            "font": {"color": "#EAECEE", "size": 12},
            # put buttons *outside* the plot area of the top subplot
            "x": 0.0, "xanchor": "left",
            "y": 1.12, "yanchor": "bottom",
        },
    }
    layout["yaxis"] = {**base_axis}

    pio.templates[name] = go.layout.Template(**tpl)
    pio.templates.default = name

# Call it at import if you like:
register_theme()
# -------- Yield curve (zero) --------
def plot_yield_curves(T0: np.ndarray, Z0: np.ndarray,
                      T1: np.ndarray, Z1: np.ndarray,
                      bump_tenors: dict[str, float] | None = None,
                      title: str = "Zero/Yield Curve — Base vs Bumped") -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=T0, y=Z0 * 100, mode="lines", name="Base zero", line=dict(width=2))
    fig.add_scatter(x=T1, y=Z1 * 100, mode="lines", name="Bumped zero", line=dict(width=2))

    if bump_tenors:
        for s, bp in bump_tenors.items():
            yrs = _tenor_to_years(s)
            fig.add_vline(x=yrs, line_width=1, line_dash="dash", line_color="#EC7063",
                          annotation_text=f"{s} {bp:+.0f}bp", annotation_position="top right",
                          annotation_font_color="#EC7063")

    fig.update_layout(title=title, xaxis_title="Maturity (years)", yaxis_title="Zero rate (%)", height=480)
    return fig


def _tenor_to_years(tenor: str) -> float:
    tenor = tenor.strip().upper()
    if tenor.endswith("Y"):
        return float(tenor[:-1])
    if tenor.endswith("M"):
        return float(tenor[:-1]) / 12.0
    if tenor.endswith("D"):
        return float(tenor[:-1]) / 365.0
    return float(tenor)  # fallback


# -------- Cashflows (portfolio) --------
def plot_cashflows(base_cf: pd.DataFrame,
                   bumped_cf: pd.DataFrame,
                   units_by_id: dict[str, float],
                   currency_symbol: str = "$",
                   add_time_slider: bool = True) -> go.Figure:
    """
    base_cf/bumped_cf must have ['ins_id','pay_date','amount'].
    """
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

    # Bars with clear hover
    fig.add_bar(
        x=m["pay_date"], y=m["amount_base"], name="Base CF", marker_color="#5DADE2",
        hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>Base: {currency_symbol}%{{y:,.2f}}<extra></extra>",
        row=1, col=1,
    )
    fig.add_bar(
        x=m["pay_date"], y=m["amount_bumped"], name="Bumped CF", marker_color="#F5B041",
        hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>Bumped: {currency_symbol}%{{y:,.2f}}<extra></extra>",
        row=1, col=1,
    )
    fig.add_bar(
        x=m["pay_date"], y=m["amount_delta"], name="Δ Amount", marker_color="#EC7063",
        hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>Δ: {currency_symbol}%{{y:,.2f}}<extra></extra>",
        row=2, col=1,
    )

    # Axis labels & readable ticks
    fig.update_yaxes(title_text=f"Amount ({currency_symbol})", tickformat=",.2f", row=1, col=1)
    fig.update_yaxes(title_text=f"Δ Amount ({currency_symbol})", tickformat=",.2f", row=2, col=1)
    fig.update_xaxes(title_text="Payment date", row=2, col=1)
    fig.update_layout(title="Portfolio Cashflows — Base vs Bumped", height=650, barmode="group")

    # Link x-axes
    fig.update_xaxes(matches="x")

    if add_time_slider:
        # Put the range *buttons* above the whole figure on the top x-axis
        fig.update_xaxes(
            row=1, col=1,
            rangeselector=dict(
                # The base styling and placement come from the template; just define the buttons.
                buttons=[
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            )
        )
        # Show the range *slider* only on the bottom axis
        fig.update_xaxes(
            row=2, col=1,
            rangeslider=dict(visible=True, bgcolor="#0B1016", bordercolor="#2B2F36", borderwidth=1, thickness=0.12),
            type="date",
        )

    return fig


# -------- NPVs (portfolio or per instrument family) --------
def plot_npvs(npv_base: dict[str, float], npv_bumped: dict[str, float], title: str = "NPV (Base vs Bumped)") -> go.Figure:
    keys = list(npv_base.keys())
    b0 = np.array([npv_base[k] for k in keys])
    b1 = np.array([npv_bumped[k] for k in keys])
    d  = b1 - b0

    fig = go.Figure()
    fig.add_bar(x=keys, y=b0, name="Base")
    fig.add_bar(x=keys, y=b1, name="Bumped")
    fig.add_scatter(x=keys, y=d, name="Δ", mode="markers+text",
                    text=[f"{x:,.0f}" for x in d], textposition="top center",
                    marker=dict(color="#EC7063", size=10))
    fig.update_layout(title=title, barmode="group", height=420, yaxis_title="PV")
    return fig


# -------- Tabular (Plotly table for a quick portfolio summary) --------
def _fmt_currency(x: float, symbol: str = "$", precision: int = 2) -> str:
    try:
        return f"{symbol}{x:,.{precision}f}"
    except Exception:
        return str(x)

def table_from_df(df: pd.DataFrame,
                  title: str = "Summary",
                  currency_symbol: str = "$",
                  currency_cols: list[str] | None = None,
                  precision: int = 2) -> go.Figure:
    """
    Build a styled table. Currency columns formatted with symbol, thousands separators, 2 decimals.
    """
    currency_cols = currency_cols or ["base", "bumped", "delta"]
    df2 = df.copy()

    # Format currency columns
    for c in currency_cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda v: _fmt_currency(v, currency_symbol, precision))

    # Format 'units' as a number with separators (no symbol)
    if "units" in df2.columns:
        df2["units"] = df2["units"].apply(lambda v: f"{v:,.2f}" if isinstance(v, (int, float, np.number)) else v)

    # Everything as string for Plotly Table
    disp = df2.astype(str)

    # Right align numeric-like columns
    cols = list(disp.columns)
    right_cols = set(currency_cols + (["units"] if "units" in disp.columns else []))
    align = ["right" if c in right_cols else "left" for c in cols]

    fig = go.Figure(data=[go.Table(
        header=dict(values=cols, fill_color="#2B2F36", align=align),
        cells=dict(values=[disp[c] for c in cols], fill_color="#111417", align=align)
    )])
    fig.update_layout(title=title, height=380)
    return fig


# --- KPI mini-table (wide) ---
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
        header=dict(values=["Metric"] + ["Value"], fill_color="#2B2F36", align="left"),
        cells=dict(values=[keys, vals], fill_color="#111417", align="left")
    )])
    fig.update_layout(title="ALM KPIs", height=380)
    return fig

# --- Liquidity ladder (inflow/outflow bars; net/cumulative lines) ---
def plot_liquidity_ladder(df: pd.DataFrame, currency_symbol: str = "$", title: str = "Liquidity Ladder (next 12 months)") -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["bucket"], y=df["inflow"],  name="Inflows",  marker_color="#58D68D",
                hovertemplate=f"%{{x|%b %Y}}<br>Inflows: {currency_symbol}%{{y:,.2f}}<extra></extra>")
    fig.add_bar(x=df["bucket"], y=-df["outflow"], name="Outflows", marker_color="#EC7063",
                hovertemplate=f"%{{x|%b %Y}}<br>Outflows: {currency_symbol}%{{y:,.2f}}<extra></extra>")
    fig.add_scatter(x=df["bucket"], y=df["net"], name="Net", mode="lines+markers")
    fig.add_scatter(x=df["bucket"], y=df["cum_net"], name="Cumulative", mode="lines")
    fig.update_layout(barmode="relative", title=title,
                      yaxis_title=f"Amount ({currency_symbol})", height=480)
    return fig


# --- Repricing GAP bars + CUM GAP line ---
def plot_repricing_gap(df: pd.DataFrame, currency_symbol: str = "$") -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["bucket"], y=df["IRSA"], name="IRSA (Assets)", marker_color="#5DADE2")
    fig.add_bar(x=df["bucket"], y=-df["IRSL"], name="IRSL (Liabilities)", marker_color="#F5B041")
    fig.add_scatter(x=df["bucket"], y=df["CUM_GAP"], name="Cumulative GAP", mode="lines+markers")
    fig.update_layout(barmode="relative", title="Repricing GAP by Bucket",
                      yaxis_title=f"Exposure ({currency_symbol})", height=480)
    return fig

# --- EVE bars for +/- shocks ---
def plot_eve_bars(eve: dict, currency_symbol: str = "$") -> go.Figure:
    keys = list(eve.keys()); vals = [eve[k] for k in keys]
    fig = go.Figure()
    fig.add_bar(x=keys, y=vals, marker_color=["#EC7063" if "+" in k else "#58D68D" for k in keys])
    fig.update_layout(title="ΔEVE (parallel shocks)", yaxis_title=f"ΔPV ({currency_symbol})", height=360)
    return fig


def plot_eve_bars_compare(eve_base: dict, eve_bump: dict, currency_symbol: str = "$") -> go.Figure:
    cats = list(eve_base.keys())  # e.g., ['+200bp', '-200bp']
    bvals = [eve_base[c] for c in cats]
    uvals = [eve_bump.get(c, np.nan) for c in cats]
    fig = go.Figure()
    fig.add_bar(x=cats, y=bvals, name="Base",   marker_color="#5DADE2")
    fig.add_bar(x=cats, y=uvals, name="Bumped", marker_color="#F5B041")
    fig.update_layout(barmode="group", title="ΔEVE (parallel shocks) — Base vs Bumped",
                      yaxis_title=f"ΔPV ({currency_symbol})", height=360)
    return fig

# ---------- helpers ----------
def _years_from_period(calc_date: ql.Date, per: ql.Period,
                       cal: ql.Calendar = ql.TARGET(),
                       dc: ql.DayCounter = ql.Actual365Fixed()) -> float:
    """Exact year fraction from calc_date to calc_date+per."""
    d = cal.advance(calc_date, per)
    return dc.yearFraction(calc_date, d)

def _nodes_to_points(nodes: list[dict], calc_date: ql.Date) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cal, dc = ql.TARGET(), ql.Actual365Fixed()
    xs, ys, labels = [], [], []
    for n in nodes:
        if "tenor" not in n:
            continue
        per = ql.Period(n["tenor"])
        xs.append(_years_from_period(calc_date, per, cal, dc))
        ys.append(float(n["rate"]))  # decimal (e.g., 0.052)
        labels.append(f'{n["type"].upper()} {n["tenor"].upper()}')
    return np.array(xs), np.array(ys), labels

def _par_yield_curve(ts: ql.YieldTermStructureHandle,
                     calc_date: ql.Date,
                     max_years: int = 12,
                     step_months: int = 3,
                     index_hint: str = "USD-LIBOR-3M") -> tuple[np.ndarray, np.ndarray]:
    cal   = ql.TARGET()
    dc    = ql.Actual365Fixed()
    # choose floating index for the synthetic par swap
    hint = (index_hint or "").upper()
    if "TIIE" in hint:
        ibor = make_tiie_28d_index(ts)
        float_step = ql.Period(28, ql.Days)
        fixed_step = ql.Period(28, ql.Days)   # simple approximation for MXN fixed leg
        fixed_dc   = ql.Actual360()
        fixed_cnv  = ql.Unadjusted
    else:
        ibor = ql.USDLibor(ql.Period("3M"), ts)
        float_step = ql.Period("3M")
        fixed_step = ql.Period("1Y")
        fixed_dc   = ql.Thirty360(ql.Thirty360.USA)
        fixed_cnv  = ql.Unadjusted

    spot  = ibor.valueDate(calc_date)
    T, Y = [], []
    m = step_months
    while m <= max_years * 12:
        end_from_today = cal.advance(calc_date, ql.Period(m, ql.Months))
        T.append(dc.yearFraction(calc_date, end_from_today))
        if m <= 12:
            df   = ts.discount(end_from_today)
            tau  = dc.yearFraction(calc_date, end_from_today)
            r_mm = (1.0 / max(df, 1e-12) - 1.0) / max(tau, 1e-12)
            Y.append(r_mm)
        else:
            end = cal.advance(spot, ql.Period(m, ql.Months))
            fixed_sched = ql.Schedule(spot, end, fixed_step, cal, fixed_cnv, fixed_cnv,
                                      ql.DateGeneration.Forward, False)
            float_sched = ql.Schedule(spot, end, float_step, cal, ql.ModifiedFollowing, ql.ModifiedFollowing,
                                      ql.DateGeneration.Forward, False)
            swap = ql.VanillaSwap(
                ql.VanillaSwap.Payer, 1.0,
                fixed_sched, 0.0, fixed_dc,
                float_sched, ibor, 0.0, ibor.dayCounter()
            )
            swap.setPricingEngine(ql.DiscountingSwapEngine(ts))
            Y.append(swap.fairRate())
        m += step_months
    return np.array(T), np.array(Y)



# ---------- NEW main plotting function ----------
def plot_par_yield_curve(base_ts: ql.YieldTermStructureHandle,
                         bumped_ts: ql.YieldTermStructureHandle,
                         calc_date: ql.Date,
                         base_nodes: list[dict],
                         bumped_nodes: list[dict],
                         bump_tenors: dict[str, float] | None = None,
                         max_years: int = 12,
                         step_months: int = 3) -> go.Figure:
    T0, Y0 = _par_yield_curve(base_ts, calc_date, max_years, step_months, index_hint=index_hint)
    T1, Y1 = _par_yield_curve(bumped_ts, calc_date, max_years, step_months, index_hint=index_hint)

    # Node markers
    xb, yb, lb_b = _nodes_to_points(base_nodes,   calc_date)
    xB, yB, lb_B = _nodes_to_points(bumped_nodes, calc_date)

    fig = go.Figure()

    # Lines
    fig.add_scatter(x=T0, y=Y0 * 100, mode="lines", name="Base par yield",
                    line=dict(color="#5DADE2", width=2))
    fig.add_scatter(x=T1, y=Y1 * 100, mode="lines", name="Bumped par yield",
                    line=dict(color="#F5B041", width=2))

    # Node markers (base = squares, bumped = diamonds)
    fig.add_scatter(
        x=xb, y=yb * 100, mode="markers", name="Base nodes",
        marker=dict(color="#5DADE2", symbol="square", size=9, line=dict(color="#0E1216", width=1)),
        text=lb_b, hovertemplate="<b>%{text}</b><br>Tenor: %{x:.2f}y<br>Rate: %{y:.3f}%<extra></extra>"
    )
    fig.add_scatter(
        x=xB, y=yB * 100, mode="markers", name="Bumped nodes",
        marker=dict(color="#F5B041", symbol="diamond", size=10, line=dict(color="#0E1216", width=1)),
        text=lb_B, hovertemplate="<b>%{text}</b><br>Tenor: %{x:.2f}y<br>Rate: %{y:.3f}%<extra></extra>"
    )

    # Annotate bumped tenors
    bump_tenors = bump_tenors or {}
    for s, bp in bump_tenors.items():
        try:
            per = ql.Period(s)
            x = _years_from_period(calc_date, per)
            fig.add_vline(
                x=x, line_width=1, line_dash="dash", line_color="#EC7063",
                annotation_text=f"{s} {bp:+.0f}bp", annotation_position="top",
                annotation_font_color="#EC7063"
            )
        except Exception:
            pass

    fig.update_layout(
        title="Par yield curve — Base vs Bumped",
        xaxis_title="Maturity (years)",
        yaxis_title="Par yield (%)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# Default theme on import
register_theme()
