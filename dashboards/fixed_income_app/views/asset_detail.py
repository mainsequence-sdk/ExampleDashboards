# examples/alm_app/views/asset_detail.py
from __future__ import annotations
import streamlit as st
from mainsequence.dashboards.streamlit.core.registry import register_page
from dashboards.fixed_income_app.context import AppContext

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
        inst_bumped.reset_curve(ctx.ts_bump)
        bumped_pv = float(inst_bumped.price()) * units

    delta = bumped_pv - base_pv

    # --- Header / metrics ---
    st.subheader("Asset Detail")
    st.caption(f"ID: `{asset_id}`")

    def _fmt_ccy(x: float) -> str:
        import math
        return "—" if x is None or not math.isfinite(float(x)) else f"{ctx.currency_symbol}{x:,.2f}"

    c1, c2, c3 = st.columns(3)
    c1.metric("Units", f"{units:,.2f}")
    c2.metric("NPV (base)", _fmt_ccy(base_pv), delta=_fmt_ccy(delta))
    c3.metric("NPV (bumped)", _fmt_ccy(bumped_pv))

    st.divider()
    st.write("**Instrument summary**")
    st.json({
        "instrument_type": getattr(line.instrument, "instrument_type", type(line.instrument).__name__),
        "issue_date": getattr(line.instrument, "issue_date", None),
        "maturity_date": getattr(line.instrument, "maturity_date", None),
        "coupon_frequency": getattr(line.instrument, "coupon_frequency", None),
        "spread": getattr(line.instrument, "spread", None),
        "face_value": getattr(line.instrument, "face_value", None),
    })

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
