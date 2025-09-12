# ui/dashboard_base/pages/asset_detail.py
from __future__ import annotations
import streamlit as st
from ui.dashboard_base.core.registry import register_page
from ui.dashboard_base.core.context import AppContext

@register_page("asset_detail", "Asset Detail",visible=False)
def render(ctx: AppContext):
    qp = st.query_params
    asset_id = qp.get("asset_id", None)
    if isinstance(asset_id, list):
        asset_id = asset_id[0] if asset_id else None

    # Hard gate: only reachable via Details link
    if not asset_id:
        st.info("Asset Detail is available only from the Positions table (Details link).")
        if st.button("← Back to Curve, Stats & Positions"):
            st.query_params.clear()
            st.rerun()
        st.stop()

    # Locate the instrument by its content_hash
    line = next((ln for ln in ctx.position.lines
                 if str(ln.instrument.content_hash()) == str(asset_id)), None)
    if line is None:
        st.error(f"Asset '{asset_id}' not found in the current position.")
        if st.button("← Back"):
            st.query_params.clear()
            st.rerun()
        st.stop()

    # Compute base / bumped NPVs
    base_pv = float(line.instrument.price()) * float(line.units)
    bump_line = next((ln for ln in ctx.bumped_position.lines
                      if str(ln.instrument.content_hash()) == str(asset_id)), None)
    bumped_pv = float(bump_line.instrument.price()) * float(bump_line.units) if bump_line else None
    delta = (bumped_pv - base_pv) if bumped_pv is not None else None

    st.subheader("Asset Detail")
    st.caption(f"ID: `{asset_id}`")

    def _fmt_ccy(x: float) -> str:
        import math
        return "—" if x is None or not math.isfinite(float(x)) else f"{ctx.currency_symbol}{x:,.2f}"

    c1, c2, c3 = st.columns(3)
    c1.metric("Units", f"{float(line.units):,.2f}")
    c2.metric("NPV (base)", _fmt_ccy(base_pv), delta=_fmt_ccy(delta) if delta is not None else None)
    c3.metric("NPV (bumped)", _fmt_ccy(bumped_pv) if bumped_pv is not None else "—")

    st.divider()
    st.write("**Instrument summary**")
    # Show a concise set of fields, defensively
    st.json({
        "instrument_type": getattr(line.instrument, "instrument_type", type(line.instrument).__name__),
        "issue_date": getattr(line.instrument, "issue_date", None),
        "maturity_date": getattr(line.instrument, "maturity_date", None),
        "coupon_frequency": getattr(line.instrument, "coupon_frequency", None),
        "spread": getattr(line.instrument, "spread", None),
        "face_value": getattr(line.instrument, "face_value", None),
    })

    st.divider()
    if st.button("← Back to Curve, Stats & Positions", key="asset_back_bottom"):
        st.query_params.clear()
        st.rerun()
