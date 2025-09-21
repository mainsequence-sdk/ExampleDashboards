# ui/components/position_yield_overlay.py
from __future__ import annotations

import datetime as dt
from typing import List, Dict, Any, Optional
import streamlit as st
import plotly.graph_objects as go
from typing import Callable


def _maturity_in_years(val_date: dt.date, maturity_date: dt.date) -> float:
    # Keep it simple & consistent with the curve x-axis (years)
    return (maturity_date - val_date).days / 365.0


def _compute_points(position,calculation_date) -> List[Dict[str, Any]]:
    pts: List[Dict[str, Any]] = []
    for line in position.lines:
        ins = line.instrument

        if line.extra_market_info is None:
            continue
        if "yield" not in line.extra_market_info:
            continue

        x=(ins.maturity_date-calculation_date).days/365
        ytm=line.extra_market_info["yield"]

        pts.append({
            "x": float(x),
            "y": ytm*100,                     # plot in %
            "label": str(ins.content_hash())[:3], # short label
        })
    # Sort by maturity for nicer hover/legend order
    pts.sort(key=lambda p: p["x"])
    return pts


def st_position_yield_overlay(
    *,
    position,
    val_date: dt.date,
    ts_base=None, ql_ref_date=None, nodes_base=None, index_hint=None,
    max_years: int = 12, step_months: int = 3,
    key: str = "pos_yield_overlay",
    point_extractor: Optional[Callable[[Any, dt.date], List[Dict[str, Any]]]] = None,
) -> List[go.Scatter]:
    """
    UI that computes/clears position YTMs on demand and returns Plotly traces
    to overlay on an existing curve figure. It does NOT call st.plotly_chart.
    """
    ss = st.session_state

    st.subheader("Base curve + position yields (on demand)")
    c1, c2, _ = st.columns([1, 1, 6])
    with c1:
        if st.button("Overlay position yields", key=f"{key}_compute"):
            with st.spinner("Computing yields…"):
                if point_extractor:
                    ss[f"{key}_points"] = point_extractor(position, val_date)
                else:
                    ss[f"{key}_points"] = _compute_points(position, calculation_date=val_date)
    with c2:
        if st.button("Clear points", key=f"{key}_clear"):
            ss.pop(f"{key}_points", None)

    points = ss.get(f"{key}_points", [])
    traces: List[go.Scatter] = []

    if points:
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        texts = [p["label"] for p in points]

        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name="Position YTM",
                text=texts,
                textposition="top center",
                marker=dict(size=9, symbol="circle-open"),
                hovertemplate="Maturity: %{x:.2f}y<br>YTM: %{y:.3f}%<br>Id3: %{text}<extra></extra>",
            )
        )

        with st.expander("Computed points"):
            import pandas as pd
            df = pd.DataFrame(points).rename(
                columns={"x": "maturity_years", "y": "ytm_percent", "label": "id3"}
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

    return traces
