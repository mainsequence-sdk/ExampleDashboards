# dashboards/components/curve_bumps_and_stats.py
from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List, Tuple

import streamlit as st

from dashboards.services.curves import (
    KEYRATE_GRID_BY_FAMILY, curve_family_key, BumpSpec, keyrate_grid_for_index, build_curves_for_ui
)
from dashboards.components.position_yield_overlay import st_position_yield_overlay
from dashboards.plots.curves import plot_par_yield_curves_by_family
from dashboards.core.formatters import fmt_ccy
from dashboards.services.positions import PositionOperations
from dashboards.core.ql import qld


# ---------- helpers (local to the component) ----------
def _indices_from_position(position) -> List[str]:
    idxs = {
        getattr(ln.instrument, "floating_rate_index_name", None)
        for ln in (position.lines or [])
    }
    idxs_ref = {
        getattr(ln.instrument, "benchmark_rate_index_name", None)
        for ln in (position.lines or [])
    }
    idxs=idxs.union(idxs_ref)
    return sorted([i for i in idxs if i])

def _build_curve_maps(
    valuation_date: dt.date,
    indices: List[str],
    family_bumps: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base: Dict[str, Any] = {}
    bumped: Dict[str, Any] = {}
    for index_uid in indices:
        fam = curve_family_key(index_uid)
        fam_spec = family_bumps.get(fam, {}) or {}
        keyrate_bp = dict(fam_spec.get("keyrate_bp", {}))
        parallel_bp = float(fam_spec.get("parallel_bp", 0.0))
        grid_for_index = keyrate_grid_for_index(index_uid)
        spec = BumpSpec(
            keyrate_bp=keyrate_bp,
            parallel_bp=parallel_bp,
            key_rate_grid={index_uid: tuple(grid_for_index)},
        )
        ts_base, ts_bump, _, _ = build_curves_for_ui(qld(valuation_date), spec, index_identifier=index_uid)
        base[index_uid] = ts_base
        bumped[index_uid] = ts_bump
    return base, bumped


# ---------- the single component ----------
def st_curve_bumps(
    *,
    position_template,
    valuation_date: dt.date,
    key_prefix: str = "curve_bundle",
    title: str = "Par yield curve — Base vs Bumped",
) -> Dict[str, Any]:
    # ---- read existing bump spec ----
    prev_map = (
        st.session_state.get(f"{key_prefix}_curve_bump_spec_by_family")
        or st.session_state.get("curve_bump_spec_by_family")
        or st.session_state.get("curve_bump_spec_by_index", {})
        or {}
    )

    # ---- indices / families present ----
    indices = _indices_from_position(position_template)
    if not indices:
        st.info("No floating-rate indices found in the current position.")
        return {}

    # ---- build curves from current spec ----
    base_curves, bumped_curves = _build_curve_maps(valuation_date, indices, prev_map)

    # ---- instantiate a merged position (for overlays only) ----
    ops = PositionOperations.from_position(
        position_template,
        base_curves_by_index=base_curves,
        valuation_date=valuation_date,
    )
    base_pos = ops.instantiate_base()
    ops.set_curves(bumped_curves_by_index=bumped_curves)
    bumped_pos = ops.instantiate_bumped()
    ops.compute_and_apply_z_spreads_from_dirty_price(
        base_position=base_pos,
        bumped_position=bumped_pos,
    )

    # ===================== CHART + OVERLAYS =====================
    st.subheader(title)
    chart_slot = st.empty()
    overlay_traces = st_position_yield_overlay(
        position=base_pos,
        valuation_date=valuation_date,
        key=f"{key_prefix}_base_curve_position_overlay",
    )
    fig_curve = plot_par_yield_curves_by_family(
        base_curves=base_curves,
        bumped_curves=bumped_curves,
        max_years=30,
        step_months=3,
        title="Par yield curves — Base vs Bumped (families)",
    )
    for tr in overlay_traces:
        fig_curve.add_trace(tr)
    chart_slot.plotly_chart(fig_curve, use_container_width=True,
                            key=f"{key_prefix}_par_curve_main")

    # ===================== BUMPS (by family) =====================
    with st.expander("Curve bumps (by family)", expanded=True):
        families_per_row = 4
        present_families = sorted({curve_family_key(i) for i in indices})
        new_map: Dict[str, Dict[str, float]] = {}

        st.markdown("""
        <style>
          [data-testid="stExpander"] .stSlider, 
          [data-testid="stExpander"] .stNumberInput { margin-bottom: .25rem; }
          [data-testid="stExpander"] h5, 
          [data-testid="stExpander"] h6 { margin: .25rem 0 .25rem; }
        </style>
        """, unsafe_allow_html=True)

        for start in range(0, len(present_families), families_per_row):
            fam_slice = present_families[start:start + families_per_row]
            cols = st.columns(len(fam_slice), gap="small")
            for col, fam in zip(cols, fam_slice):
                tenors = KEYRATE_GRID_BY_FAMILY[fam]
                with col:
                    spec, _ = __render_curve_bump_controls_for_family(
                        fam=fam,
                        tenors=tenors,
                        default_bumps=(prev_map.get(fam, {}).get("keyrate_bp", {})),
                        default_parallel=float(prev_map.get(fam, {}).get("parallel_bp", 0.0)),
                        key_prefix=key_prefix,
                        container=col,
                        header=fam,
                    )
                    new_map[fam] = {
                        "keyrate_bp": spec.keyrate_bp,
                        "parallel_bp": float(spec.parallel_bp),
                    }

        if new_map and new_map != prev_map:
            st.session_state[f"{key_prefix}_curve_bump_spec_by_family"] = new_map
            st.session_state["curve_bump_spec_by_family"] = new_map
            st.session_state["curve_bump_spec_by_index"] = new_map
            st.rerun()

    # artifacts for downstream consumers
    return dict(
        base_curves=base_curves,
        bumped_curves=bumped_curves,
        position=base_pos,           # merged (for overlays)
        bumped_position=bumped_pos,  # merged (for overlays)
        fig_curve=fig_curve,
    )




# ---- internal: one-family controls (kept here to make the component self-contained) ----
def __render_curve_bump_controls_for_family(
    *,
    fam: str,
    tenors: List[str],
    default_bumps: Dict[str, float],
    default_parallel: float,
    key_prefix: str,
    container=None,
    header: str | None = None,
):
    from dashboards.components.curve_bump import curve_bump_controls_ex
    container = container or st.container()
    header = header if header is not None else f"{fam} bumps (bp)"
    spec, _ = curve_bump_controls_ex(
        available_tenors=tenors,
        default_bumps=default_bumps,
        default_parallel_bp=default_parallel,
        header=header,
        container=container,
        key=f"{key_prefix}_curve_bump_{fam}",
    )
    return spec, container

