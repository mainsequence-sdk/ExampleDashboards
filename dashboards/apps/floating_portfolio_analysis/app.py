# dashboards/apps/floating_portfolio_analysis/app.py
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import datetime as dt

import streamlit as st
import QuantLib as ql

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent.parent))

from mainsequence.dashboards.streamlit.scaffold import AppConfig, run_app
from mainsequence.dashboards.streamlit.core.registry import autodiscover


from dashboards.core.theme import register_theme
from dashboards.core.ql import qld
from dashboards.services.curves import (
    BumpSpec, build_curves_for_ui, curve_family_key, keyrate_grid_for_index
)
from dashboards.components.engine_status import (
    collect_engine_meta_from_ctx, render_engine_status, register_engine_meta_provider
)
from dashboards.services.positions import PositionOperations
import mainsequence.client as msc
from dashboards.core.data_nodes import get_app_data_nodes
register_theme()
# ─────────────────────────────────────────────────────────────────────────────
# Engine meta / data-node registration — kept inline for template clarity
# ─────────────────────────────────────────────────────────────────────────────
def _register_default_data_nodes() -> None:
    """
    Declare app-level data-node dependencies by name.
    This feeds the 'Pricing engine' status bar with a stable label for external tables.
    Adjust these registrations to reflect your environment.
    """
    deps = get_app_data_nodes()
    # Example: our valuations use Valmer's pricing vector (id/name here is arbitrary, but stable in the app)
    deps.register(instrument_pricing_table_id="vector_de_precios_valmer")

def _meta_summary_provider(ctx, meta) -> None:
    """
    Adds a 'Summary' row to the engine status:
      • valuation_date: the pricing date currently in use.
    """
    meta.add_summary(valuation_date=getattr(ctx, "valuation_date", None))

def _meta_curves_provider(ctx, meta) -> None:
    """
    Adds a 'Curves' section with each curve's reference date.
    Robust to missing curves.
    """
    try:
        refs = {k: v.referenceDate().ISO() for k, v in (getattr(ctx, "base_curves", {}) or {}).items()}
    except Exception:
        refs = {}
    if refs:
        meta.add("Curves", **refs)

def _meta_portfolio_provider(ctx, meta) -> None:
    """
    Adds runtime portfolio info if the view published it (e.g., weights_date).
    """
    wd = st.session_state.get("weights_date")
    if wd:
        meta.add("Portfolio", weights_date=wd)

def _meta_market_provider(ctx, meta) -> None:
    """
    Adds market data observation timestamp if the engine/view populated it.
    """
    asof = st.session_state.get("prices_observation_time")
    if asof:
        meta.add("Pricing data", observation_time=asof)

def _meta_build_info(ctx, meta) -> None:
    """
    Adds build/runtime info for quick diagnostics.
    """
    import os
    meta.add("Build", engine="FixedIncome/QL", backend=os.environ.get("TDAG_ENDPOINT"))

def _bootstrap_engine_meta() -> None:
    """
    Single-entry bootstrap so the template stays self-contained.
    """
    _register_default_data_nodes()
    register_engine_meta_provider(_meta_summary_provider)
    register_engine_meta_provider(_meta_curves_provider)
    register_engine_meta_provider(_meta_portfolio_provider)
    register_engine_meta_provider(_meta_market_provider)
    register_engine_meta_provider(_meta_build_info)

_bootstrap_engine_meta()
# ─────────────────────────────────────────────────────────────────────────────
#  Context (inlined here; no external registry)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AppContext:
    cfg: Dict[str, Any]
    valuation_date: dt.date
    base_curves: Dict[str, ql.YieldTermStructureHandle]
    bumped_curves: Dict[str, ql.YieldTermStructureHandle]
    position: Any
    bumped_position: Any
    carry_cutoff: dt.date

def _get_cfg_from_session(ss) -> Dict[str, Any]:
    """
    Read the user-configurable bits we persist in session (set from the sidebar).
    Keep this tiny and explicit so readers can see what drives the engine.
    """
    cfg_mem = ss.get("position_cfg_mem")
    if cfg_mem:
        return dict(cfg_mem)

    return { }


def _indices_from_template(template) -> List[str]:
    idxs = {
        getattr(ln.instrument, "floating_rate_index_name", None)
        for ln in (template.lines or [])
    }
    return sorted([i for i in idxs if i])

def _build_curve_maps(
    valuation_date: dt.date,
    indices: List[str],
    family_bumps: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, ql.YieldTermStructureHandle], Dict[str, ql.YieldTermStructureHandle]]:
    base: Dict[str, ql.YieldTermStructureHandle] = {}
    bumped: Dict[str, ql.YieldTermStructureHandle] = {}
    for index_uid in indices:
        fam = curve_family_key(index_uid)
        fam_spec = family_bumps.get(fam, {}) or {}
        keyrate_bp = dict(fam_spec.get("keyrate_bp", {}))
        parallel_bp = float(fam_spec.get("parallel_bp", 0.0))
        grid_for_index = keyrate_grid_for_index(index_uid)
        spec = BumpSpec(keyrate_bp=keyrate_bp, parallel_bp=parallel_bp,
                        key_rate_grid={index_uid: tuple(grid_for_index)})
        ts_base, ts_bump, _, _ = build_curves_for_ui(qld(valuation_date), spec, index_identifier=index_uid)
        base[index_uid] = ts_base
        bumped[index_uid] = ts_bump
    return base, bumped



def build_context(session_state) -> AppContext:
    """
    # Build everything the views need from what's in `st.session_state`:
#   1) Valuation date (set in the sidebar).
#   2) Position template (built from the portfolio picker).
#   3) Curves (base + bumped) based on the current bump spec.
#   4) Base & bumped positions, with z-spreads applied from dirty prices.
#   5) A carry cutoff date derived from a UI slider (not from config)
    """
    # 1) Config / valuation date
    cfg = _get_cfg_from_session(session_state)
    valuation_date = dt.date.fromisoformat(cfg["valuation_date"])
    ql.Settings.instance().evaluationDate = qld(valuation_date)

    # 2) Template position (required)
    template = session_state.get("position_template_mem")
    if template is None:
        raise RuntimeError("No position template loaded yet. Select a portfolio in the sidebar first.")

    # 3) Curves (per index; bumps are provided at *family* level in UI)
    family_bumps = session_state.get("curve_bump_spec_by_family") or {}
    indices = _indices_from_template(template)
    base_curves, bumped_curves = _build_curve_maps(valuation_date, indices, family_bumps)

    # 4) Instantiate base & bumped positions and apply z‑spreads from dirty prices
    ops = PositionOperations.from_template(template, base_curves_by_index=base_curves, valuation_date=valuation_date)
    position = ops.instantiate_base()
    ops.set_curves(bumped_curves_by_index=bumped_curves)
    bumped_position = ops.instantiate_bumped()
    ops.compute_and_apply_z_spreads_from_dirty_price(
        base_position=position, bumped_position=bumped_position
    )

    # 5) Carry cutoff (UI-driven; not stored in config)
    ui_carry_days = int(session_state.get("carry_cutoff_days", 365))
    carry_cutoff = valuation_date + dt.timedelta(days=ui_carry_days)
    return AppContext(
        cfg=cfg,
        valuation_date=valuation_date,
        base_curves=base_curves,
        bumped_curves=bumped_curves,
        position=position,
        bumped_position=bumped_position,
        carry_cutoff=carry_cutoff,
    )





# --- Route selection: asset detail only reachable via query param -------------
def _route_selector(qp):
    """
    Route rules:
      • If ?id= or ?unique_identifier= -> 'asset_detail'
      • Else, respect the nav param (?page= / ?view= / ?p=) if present
      • Else, fall back to the default page ('curve_positions')
    """
    # Normalize keys (qp can be dict-like or streamlit QueryParams)
    keys = set(qp.keys()) if hasattr(qp, "keys") else set(qp or [])
    if "id" in keys or "unique_identifier" in keys:
        return "asset_detail"
    # support multiple param names for safety
    for k in ("page", "view", "p"):
        v = qp.get(k) if hasattr(qp, "get") else None
        if isinstance(v, list):
            v = v[0] if v else None
        if v:
            return str(v)
    # default landing route
    return "curve_positions"

# --- Header renderer (defensive if ctx is None) ------------------------------
def _render_header(ctx: AppContext) -> None:
    # Detect current route
    route = _route_selector(st.query_params)

    if route == "asset_detail":
        # Build a clean title: "Asset Detail — <uid> · <snapshot>"
        def _qp_first(key: str):
            v = st.query_params.get(key)
            return v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else None)

        uid = None
        snap_name = None
        try:
            id_raw = _qp_first("id")
            uid_raw = _qp_first("unique_identifier")

            asset = None
            if id_raw is not None:
                asset = msc.Asset.get_or_none(id=int(id_raw))
            elif uid_raw is not None:
                asset = msc.Asset.get_or_none(unique_identifier=str(uid_raw))

            if asset is not None:
                uid = getattr(asset, "unique_identifier", None)
                snap = getattr(asset, "current_snapshot", None)
                snap_name = getattr(snap, "name", None)
        except Exception:
            # Do not crash the whole app if header fetch fails — just fall back to generic title
            pass

        title = "Asset Detail"
        if uid:
            title += f" — {uid}"
        if snap_name:
            title += f" · {snap_name}"

        st.title(title)
        # Keep the header clean for the standalone page (no engine status bar here)
        return

    # Default header for the rest of the app
    st.title("Fixed Income Dashboard")
    deps = get_app_data_nodes()
    render_engine_status(
        collect_engine_meta_from_ctx(ctx),
        app_sections={"Data nodes": deps.as_mapping()},
        mode="sticky_bar",
        title="Pricing engine",
        open=False,
    )

# --- Session seeding (NO default cfg; we wait for user to load a portfolio) --
def _init_session(ss) -> None:
    ss.setdefault("cfg_path", None)

def _build_context_route_aware(session_state) -> Optional[AppContext]:
    """
    Route-aware builder:
      • For 'asset_detail', skip heavy context (page doesn't need it).
      • Otherwise, build only if a template position exists in session.
    """
    route = _route_selector(st.query_params)
    if route == "asset_detail":
        return None
    if session_state.get("position_template_mem") is None:
        return None
    return build_context(session_state)
# Register all views in this package (side effects of @register_page)
autodiscover("dashboards.apps.floating_portfolio_analysis.views")



cfg = AppConfig(
    title="Fixed Income Position Dashboard",
    build_context=_build_context_route_aware,
    route_selector=_route_selector,
    render_header=_render_header,
    init_session=_init_session,
    default_page="curve_positions",
hide_streamlit_multipage_nav=False,use_wide_layout=True
)

run_app(cfg)
