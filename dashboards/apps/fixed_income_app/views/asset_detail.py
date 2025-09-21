# examples/alm_app/views/asset_detail.py
from __future__ import annotations
import streamlit as st
from mainsequence.dashboards.streamlit.core.registry import register_page
import mainsequence.client as msc
from dashboards.apps.fixed_income_app.context import AppContext
from dashboards.core.formatters import fmt_ccy

import json


@st.cache_data(show_spinner=False)
def _fetch_mainsequence_asset(asset_id: int):
    """Fetch a MainSequence Asset by id and return a plain dict (Pydantic v1/v2 compatible)."""
    if msc is None:
        raise ImportError("mainsequence.client is not installed/available")
    asset = msc.Asset.get(id=int(asset_id))

    return asset.model_dump()

def _instrument_json(inst) -> dict:
    """Best‑effort to get a full JSON configuration of the instrument."""
    # Prefer the instrument's own serializer if present.
    for attr in ("to_json_dict", "to_json"):
        fn = getattr(inst, attr, None)
        if callable(fn):
            try:
                data = fn()
                # Some implementations may return a nested object; ensure dict
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    # Fallback: shallow introspection
    try:
        return json.loads(json.dumps(inst, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        # Last resort: show a minimal snapshot
        return {"instrument_type": getattr(inst, "instrument_type", type(inst).__name__)}

@register_page("asset_detail", "Asset Detail", visible=False, has_sidebar=False)
def render(ctx: AppContext):
    # --- Route params ---
    qp = st.query_params
    asset_id = qp.get("asset_id", None)
    if isinstance(asset_id, list):
        asset_id = asset_id[0] if asset_id else None

    if not asset_id:
        st.info("Asset Detail is available only from the Positions table (Details link).")
        if st.button("← Back to Curve, Stats & Positions"):
            st.query_params.clear(); st.rerun()
        st.stop()

    # --- Locate instrument in the *base* position ---
    line = next((ln for ln in ctx.position.lines
                 if str(ln.instrument.content_hash()) == str(asset_id)), None)
    if line is None:
        st.error(f"Asset '{asset_id}' not found in the current position.")
        if st.button("← Back"):
            st.query_params.clear(); st.rerun()
        st.stop()

    # --- Compute base / bumped NPVs (bump this line only) ---
    units = float(line.units)
    base_pv = float(line.instrument.price()) * units

    bumped_pv = None
    # If ctx.bumped_position is a real, distinct portfolio we can reuse its price
    if getattr(ctx, "bumped_position", None) and ctx.bumped_position is not ctx.position:
        bump_line = next((ln for ln in ctx.bumped_position.lines
                          if str(ln.instrument.content_hash()) == str(asset_id)), None)
        if bump_line:
            bumped_pv = float(bump_line.instrument.price()) * units

    # Otherwise reprice this instrument against ctx.ts_bump locally (fast)
    if bumped_pv is None:
        inst_bumped = line.instrument.copy()
        idx_name = getattr(line.instrument, "floating_rate_index_name", None)
        bump_curve = (getattr(ctx, "bumped_curves", {}) or {}).get(idx_name, None)
        if bump_curve is None:
            # fallback to whatever curve the instrument already has
            bump_curve = line.instrument.get_index_curve()
        inst_bumped.reset_curve(bump_curve)
        bumped_pv = float(inst_bumped.price()) * units

    delta = bumped_pv - base_pv

    # --- Header / metrics ---
    st.subheader("Asset Detail")
    # IDs: instrument hash + optional MainSequence Asset ID
    instrument_hash = str(line.instrument.content_hash())
    # Try attribute first; if missing, try to pull from instrument JSON
    ms_asset_id = getattr(line.instrument, "main_sequence_asset_id", None)
    inst_json_full = _instrument_json(line.instrument)
    if ms_asset_id is None:
        ms_asset_id = inst_json_full.get("main_sequence_asset_id", None)

    st.caption(f"Instrument hash: `{instrument_hash}`")
    if ms_asset_id is not None:
        st.caption(f"MainSequence asset id: **{ms_asset_id}**")


    c1, c2, c3 = st.columns(3)
    c1.metric("Units", f"{units:,.2f}")
    c2.metric("NPV (base)", fmt_ccy(base_pv), delta=fmt_ccy(delta))
    c3.metric("NPV (bumped)", fmt_ccy(bumped_pv))

    st.divider()
    col_left, col_right = st.columns(2)

    # Left column: full instrument JSON configuration
    with col_left:
        st.write("**Instrument configuration (full JSON)**")
        st.json(inst_json_full)

    # Right column: MainSequence Asset JSON (if available)
    with col_right:
        st.write("**MainSequence Asset (API)**")
        if ms_asset_id is None:
            st.info("Instrument has no `main_sequence_asset_id`.")
        else:
            try:
                asset_dict = _fetch_mainsequence_asset(int(ms_asset_id))
                st.json(asset_dict)
            except Exception as e:
                st.info(f"Could not fetch MainSequence Asset id={ms_asset_id}: {e}")



    # --- Cashflows: use get_cashflows_df() exactly as requested ---
    st.divider()
    st.write("**Cashflows (base)**")
    try:
        cf_df = line.instrument.get_cashflows_df()  # must return a DataFrame
        # Display exactly what the instrument provides (no extra assumptions)
        st.dataframe(cf_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download cashflows CSV",
            data=cf_df.to_csv(index=False).encode("utf-8"),
            file_name=f"cashflows_{str(asset_id)[:12]}.csv",
            mime="text/csv",
            key="dl_cashflows_csv"
        )
    except Exception as e:
        st.error(f"get_cashflows_df() failed: {e}")

    st.divider()
    if st.button("← Back to Curve, Stats & Positions", key="asset_back_bottom"):
        st.query_params.clear(); st.rerun()
