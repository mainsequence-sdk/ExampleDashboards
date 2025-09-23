# ui/components/position_yield_overlay.py
from __future__ import annotations

import datetime as dt
from typing import List, Dict, Any, Optional
import streamlit as st
import plotly.graph_objects as go
from typing import Callable
from dashboards.services.positions import PositionOperations

def _maturity_in_years(val_date: dt.date, maturity_date: dt.date) -> float:
    # Keep it simple & consistent with the curve x-axis (years)
    return (maturity_date - val_date).days / 365.0





def st_position_yield_overlay(
    *,
    position,
    valuation_date: dt.date,
    key: str = "base_curve_position_overlay",
    point_extractor: Optional[Callable[[Any, dt.date], List[Dict[str, Any]]]] = None,
    pos_ops: Optional[PositionOperations] = None,
) -> List[go.Scatter]:
    """
    UI that computes/clears position YTMs on demand and returns Plotly traces
    to overlay on an existing curve figure. It does NOT call st.plotly_chart.
    - If `point_extractor` is provided, it is used (for compatibility).
    - Otherwise, we call PositionOperations.yield_overlay_points(...).
    """
    ss = st.session_state
    ops = pos_ops or PositionOperations()

    st.subheader("Base curve + position yields (on demand)")
    c1, c2, _ = st.columns([1, 1, 6])

    with c1:
        if st.button("Overlay position yields", key=f"{key}_compute"):
            with st.spinner("Computing yields…"):
                if callable(point_extractor):
                    ss[f"{key}_points"] = point_extractor(position, valuation_date)
                else:
                    ss[f"{key}_points"] = ops.yield_overlay_points(
                        position=position, valuation_date=valuation_date
                    )

    with c2:
        if st.button("Clear points", key=f"{key}_clear"):
            ss.pop(f"{key}_points", None)

    points = ss.get(f"{key}_points", [])
    traces: List[go.Scatter] = []

    if points:
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        texts = [p.get("label", "") for p in points]

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