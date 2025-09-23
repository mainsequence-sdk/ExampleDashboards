from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import QuantLib as ql

from dashboards.core.tenor import tenor_to_years
from dashboards.services.curves import (
    ParRateCalculator, TIIE28ParCalculator, par_curve, par_nodes_from_tenors
)
from dashboards.services.curves import curve_family_key,KEYRATE_GRID_BY_FAMILY

# --- small helper for consistent x-axis years ---
def _years_from_period(calc_date: ql.Date, per: ql.Period,
                       cal: ql.Calendar = ql.TARGET(),
                       dc: ql.DayCounter = ql.Actual365Fixed()) -> float:
    d = cal.advance(calc_date, per)
    return dc.yearFraction(calc_date, d)

def annotate_bump_lines(fig: go.Figure,
                        bump_tenors: dict[str, float] | None,
                        calc_date: ql.Date) -> None:
    for s, bp in (bump_tenors or {}).items():
        try:
            per = ql.Period(s)
            x = _years_from_period(calc_date, per)
            fig.add_vline(
                x=x, line_width=1, line_dash="dash", line_color="#EC7063",
                annotation_text=f"{s} {bp:+.0f}bp", annotation_position="top",
                annotation_font_color="#EC7063"
            )
        except Exception:
            continue

def plot_yield_curves(T0: np.ndarray, Z0: np.ndarray,
                      T1: np.ndarray, Z1: np.ndarray,
                      bump_tenors: dict[str, float] | None = None,
                      title: str = "Zero/Yield Curve — Base vs Bumped") -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=T0, y=Z0 * 100, mode="lines", name="Base zero", line=dict(width=2))
    fig.add_scatter(x=T1, y=Z1 * 100, mode="lines", name="Bumped zero", line=dict(width=2))
    if bump_tenors:
        for s, bp in bump_tenors.items():
            fig.add_vline(x=tenor_to_years(s), line_width=1, line_dash="dash", line_color="#EC7063",
                          annotation_text=f"{s} {bp:+.0f}bp", annotation_position="top right",
                          annotation_font_color="#EC7063")
    fig.update_layout(title=title, xaxis_title="Maturity (years)", yaxis_title="Zero rate (%)", height=480)
    return fig

def plot_par_yield_curve(base_ts: ql.YieldTermStructureHandle,
                         bumped_ts: ql.YieldTermStructureHandle,
                         calc_date: ql.Date,
                         base_nodes: list[dict],
                         bumped_nodes: list[dict],
                         bump_tenors: dict[str, float] | None = None,
                         max_years: int = 12,
                         step_months: int = 3,
                         par_calc: ParRateCalculator | None = None) -> go.Figure:
    par_calc = par_calc or TIIE28ParCalculator()
    T0, Y0 = par_curve(base_ts, max_years, step_months, par_calc)
    T1, Y1 = par_curve(bumped_ts, max_years, step_months, par_calc)
    xb, yb, lb_b = par_nodes_from_tenors(base_ts, base_nodes, par_calc)
    xB, yB, lb_B = par_nodes_from_tenors(bumped_ts, bumped_nodes, par_calc)

    fig = go.Figure()
    fig.add_scatter(x=T0, y=Y0 * 100, mode="lines", name="Base par yield",
                    line=dict(color="#5DADE2", width=2), legendgroup="curve")
    fig.add_scatter(x=T1, y=Y1 * 100, mode="lines", name="Bumped par yield",
                    line=dict(color="#AF7AC5", width=2, dash="dash"), opacity=0.8, legendgroup="curve")

    fig.add_scatter(x=xb, y=yb * 100, mode="markers", name="Base nodes",
                    marker=dict(color="#5DADE2", symbol="square", size=9, line=dict(color="#0E1216", width=1)),
                    text=lb_b, hovertemplate="<b>%{text}</b><br>Tenor: %{x:.2f}y<br>Rate: %{y:.3f}%<extra></extra>")
    fig.add_scatter(x=xB, y=yB * 100, mode="markers", name="Bumped nodes",
                    marker=dict(color="#F5B041", symbol="diamond", size=10, line=dict(color="#0E1216", width=1)),
                    text=lb_B, hovertemplate="<b>%{text}</b><br>Tenor: %{x:.2f}y<br>Rate: %{y:.3f}%<extra></extra>")

    annotate_bump_lines(fig, bump_tenors, calc_date)
    fig.update_layout(title="Par yield curve — Base vs Bumped",
                      xaxis_title="Maturity (years)", yaxis_title="Par yield (%)",
                      height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
def plot_par_yield_curve_single(
    base_ts: ql.YieldTermStructureHandle,
    calc_date: ql.Date,
    base_nodes: list[dict],
    *,
    max_years: int = 12,
    step_months: int = 3,
    par_calc: ParRateCalculator | None = None,
    title: str = "Par yield curve (market)"
) -> go.Figure:
    """
    Single par curve (no 'bumped' line). Useful for overlays (e.g., two portfolios' YTMs).
    """
    par_calc = par_calc or TIIE28ParCalculator()
    T, Y = par_curve(base_ts, max_years, step_months, par_calc)
    xN, yN, lb = par_nodes_from_tenors(base_ts, base_nodes, par_calc)

    fig = go.Figure()
    fig.add_scatter(x=T, y=Y * 100, mode="lines", name="Par yield", line=dict(width=2))
    fig.add_scatter(x=xN, y=yN * 100, mode="markers", name="Par nodes",
                    marker=dict(symbol="square", size=9, line=dict(color="#0E1216", width=1)),
                    text=lb,
                    hovertemplate="<b>%{text}</b><br>Tenor: %{x:.2f}y<br>Rate: %{y:.3f}%<extra></extra>")
    fig.update_layout(title=title,
                      xaxis_title="Maturity (years)", yaxis_title="Par yield (%)",
                      height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- add near the other plot helpers in plots/curves.py ---
# --- add near the other plot helpers in plots/curves.py ---
def plot_par_yield_curves_multi(
    base_curves: dict[str, ql.YieldTermStructureHandle],
    bumped_curves: dict[str, ql.YieldTermStructureHandle] | None,
    calc_date: ql.Date,
    *,
    keyrate_grid_by_index: dict[str, tuple[str, ...]] | None = None,
    max_years: int = 30,
    step_months: int = 3,
    par_calc: ParRateCalculator | None = None,
    title: str = "Par yield curves — Base vs Bumped (all)"
) -> go.Figure:
    fig = go.Figure()
    ref_lines = []

    for idx_name, ts_base in base_curves.items():
        calc_local = par_calc or TIIE28ParCalculator(index_identifier=idx_name)

        T0, Y0 = par_curve(ts_base, max_years, step_months, calc_local)
        fig.add_scatter(
            x=T0, y=Y0 * 100.0, mode="lines",
            name=f"{idx_name} — base", legendgroup=idx_name, line=dict(width=2)
        )

        ts_bump = (bumped_curves or {}).get(idx_name)
        if ts_bump is not None:
            T1, Y1 = par_curve(ts_bump, max_years, step_months, calc_local)
            fig.add_scatter(
                x=T1, y=Y1 * 100.0, mode="lines",
                name=f"{idx_name} — bumped", legendgroup=idx_name,
                line=dict(width=2, dash="dash"), opacity=0.9
            )

        tenors = tuple((keyrate_grid_by_index or {}).get(idx_name, ()))
        if tenors:
            nodes_stub = [{"tenor": t} for t in tenors]
            xN, yN, _ = par_nodes_from_tenors(ts_base, nodes_stub, calc_local)
            fig.add_scatter(
                x=xN, y=yN * 100.0, mode="markers",
                name=f"{idx_name} — nodes", legendgroup=idx_name, showlegend=False,
                marker=dict(symbol="square", size=8, line=dict(color="#0E1216", width=1)),
                text=[f"{idx_name} {t}" for t in tenors],
                hovertemplate="<b>%{text}</b><br>Tenor: %{x:.2f}y<br>Rate: %{y:.3f}%<extra></extra>"
            )

        try:
            ref_lines.append(f"{idx_name}: {ts_base.referenceDate().ISO()}")
        except Exception:
            pass

    # ⬇️ NEW: put refs as a subtitle in the title (no cropping)
    refs_text = "  •  ".join(ref_lines)
    if refs_text:
        title_text = f"{title}<br><span style='font-size:0.85em; font-weight:400;'>Refs • {refs_text}</span>"
        top_margin = 110
    else:
        title_text = title
        top_margin = 70

    fig.update_layout(
        title=dict(text=title_text, x=0, xanchor="left"),
        xaxis_title="Maturity (years)",
        yaxis_title="Par yield (%)",
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=top_margin, r=20, b=40, l=60),
    )
    return fig


def plot_par_yield_curves_by_family(
    base_curves: dict[str, ql.YieldTermStructureHandle],
    bumped_curves: dict[str, ql.YieldTermStructureHandle] | None,
    *,
    max_years: int = 30,
    step_months: int = 3,
    par_calc: ParRateCalculator | None = None,
    title: str = "Par yield curves — Base vs Bumped (families)"
) -> go.Figure:
    """
    Collapse many index UIDs into a single curve per family (TIIE, CETE, …).
    We pick the first index we see in each family as the representative for:
      - swap conventions (via ParRateCalculator(index_identifier=rep_idx))
      - key-rate node labels (via keyrate_grid_by_index[rep_idx])
    """
    # family -> representative index UID + curve handles
    fam_map: dict[str, dict] = {}
    for idx, ts_base in base_curves.items():
        fam = curve_family_key(idx)
        if fam not in fam_map:
            fam_map[fam] = {
                "rep": idx,
                "base": ts_base,
                "bump": (bumped_curves or {}).get(idx),
            }

    fig = go.Figure()
    refs = []
    for fam, info in sorted(fam_map.items()):
        rep_idx = info["rep"]
        ts_base = info["base"]
        ts_bump = info["bump"]

        calc_local = par_calc or TIIE28ParCalculator(index_identifier=rep_idx)
        T0, Y0 = par_curve(ts_base, max_years, step_months, calc_local)
        fig.add_scatter(
            x=T0, y=Y0 * 100.0, mode="lines",
            name=f"{fam} — base", legendgroup=fam, line=dict(width=2)
        )

        if ts_bump is not None:
            T1, Y1 = par_curve(ts_bump, max_years, step_months, calc_local)
            fig.add_scatter(
                x=T1, y=Y1 * 100.0, mode="lines",
                name=f"{fam} — bumped", legendgroup=fam,
                line=dict(width=2, dash="dash"), opacity=0.9
            )

        # Show par nodes for the family (use rep index's grid if available)
        tenors =KEYRATE_GRID_BY_FAMILY[fam]
        if tenors:
            nodes_stub = [{"tenor": t} for t in tenors]
            xN, yN, _ = par_nodes_from_tenors(ts_base, nodes_stub, calc_local)
            fig.add_scatter(
                x=xN, y=yN * 100.0, mode="markers",
                name=f"{fam} — nodes", legendgroup=fam, showlegend=False,
                marker=dict(symbol="square", size=8, line=dict(color="#0E1216", width=1)),
                text=[f"{fam} {t}" for t in tenors],
                hovertemplate="<b>%{text}</b><br>Tenor: %{x:.2f}y<br>Rate: %{y:.3f}%<extra></extra>"
            )

        try:
            refs.append(f"{fam}: {ts_base.referenceDate().ISO()}")
        except Exception:
            pass

    refs_text = " • ".join(refs)
    title_text = f"{title}<br><span style='font-size:0.85em; font-weight:400;'>Refs • {refs_text}</span>" if refs_text else title
    fig.update_layout(
        title=dict(text=title_text, x=0, xanchor="left"),
        xaxis_title="Maturity (years)",
        yaxis_title="Par yield (%)",
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=110 if refs_text else 70, r=20, b=40, l=60),
    )
    return fig