# src/ui/components/curve_bump.py
from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

from dashboards.curves import BumpSpec


def _parse_manual(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        t, v = line.split(":", 1)
        try:
            out[(t or "").strip().upper()] = float(v.strip())
        except Exception:
            pass
    return out


def curve_bump_controls(*,
                        available_tenors: List[str],
                        default_bumps: Optional[Dict[str, float]] = None,
                        default_parallel_bp: float = 0.0,
                        header: str = "Curve bumps (bp)",
                        key: str = "curve_bump") -> BumpSpec:
    """
    Streamlit widget group that returns a BumpSpec for the dashboard to use.
    """
    default_bumps = default_bumps or {}
    with st.sidebar:
        st.markdown(f"### {header}")

        # Tenor selection
        default_selection = list(default_bumps.keys()) or (["5Y"] if "5Y" in available_tenors else available_tenors[:1])
        sel_tenors = st.multiselect(
            "Tenors to bump", options=available_tenors, default=default_selection, key=f"{key}_tenors"
        )

        # Per‑tenor numeric inputs
        bump_map: Dict[str, float] = {}
        for t in sel_tenors:
            bump_map[t] = st.number_input(
                f"{t} bump (bp)",
                value=float(default_bumps.get(t, 0.0)),
                step=5.0, format="%.1f", key=f"{key}_bump_{t}"
            )

        st.caption("Or paste 'tenor: bp' lines below; these override sliders if present.")
        manual_text = st.text_area(
            "Manual bumps", value="", height=80, placeholder="5Y: 100\n3Y: -10", key=f"{key}_manual"
        )
        manual = _parse_manual(manual_text)
        if manual:
            bump_map.update(manual)

        par_bp = st.slider("Parallel bump (bp)", -300.0, 300.0, float(default_parallel_bp), 1.0, key=f"{key}_par")

    return BumpSpec(keyrate_bp=bump_map, parallel_bp=par_bp)
