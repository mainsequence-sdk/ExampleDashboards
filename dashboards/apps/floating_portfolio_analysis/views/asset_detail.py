# dashboards/apps/floating_portfolio_analysis/views/asset_detail.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional
import datetime
import pytz

import streamlit as st
from mainsequence.dashboards.streamlit.core.registry import register_page
import mainsequence.client as msc
import mainsequence.instruments as msi


def _qp_first(key: str) -> Optional[str]:
    v = st.query_params.get(key, None)
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort conversion of Pydantic v1/v2 or other objects to plain dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "json"):
        return json.loads(obj.json())
    return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))


@register_page("asset_detail", "Asset Detail", visible=False, has_sidebar=False)
def render(_: Any = None) -> None:
    """
    Standalone Asset Detail:
      - Exactly one of:
          ?id=<int>                 -> msc.Asset.get_or_none(id=...)
          ?unique_identifier=<str>  -> msc.Asset.get_or_none(unique_identifier=...)
      - No context usage. No guessing. Raises on invalid/missing inputs or rebuild failures.
    """
    # ---- Strict query parsing (no mixing, no guessing) ----
    id_raw = _qp_first("id")
    uid_raw = _qp_first("unique_identifier")

    if (id_raw is None and uid_raw is None) or (id_raw is not None and uid_raw is not None):
        raise ValueError("Provide exactly one query parameter: either 'id' (int) or 'unique_identifier' (str).")

    # ---- Fetch exactly as requested ----
    if id_raw is not None:
        if not str(id_raw).isdigit():
            raise TypeError("'id' must be an integer.")
        asset = msc.Asset.get_or_none(id=int(id_raw))
        if asset is None:
            raise LookupError(f"Asset with id={id_raw} was not found.")
    else:
        uid = str(uid_raw)
        asset = msc.Asset.get_or_none(unique_identifier=uid)
        if asset is None:
            raise LookupError(f"Asset with unique_identifier='{uid}' was not found.")

    # ---- Identity chips: ONLY UID and Snapshot (as requested) ----
    uid_val = getattr(asset, "unique_identifier", None)
    snapshot = getattr(asset, "current_snapshot", None)
    snapshot_name = getattr(snapshot, "name", None)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Unique Identifier", uid_val or "—")
    with c2:
        st.metric("Snapshot", snapshot_name or "—")

    st.divider()

    # ---- Two-column main row: Asset JSON | Instrument dump ----
    left, right = st.columns(2)

    with left:
        st.subheader("Asset Detail")
        st.json(_to_dict(asset))

    pricing_detail = getattr(asset, "current_pricing_detail", None)
    if pricing_detail is None:
        raise ValueError("Asset has no current_pricing_detail; cannot rebuild instrument for cashflows.")

    inst_dump = getattr(pricing_detail, "instrument_dump", None)

    with right:
        st.subheader("Instrument dump (from platform)")
        st.json(_to_dict(inst_dump) if inst_dump is not None else {})

    # ---- Cashflows via rebuilt instrument (QuantLib) ----
    if inst_dump is None:
        raise ValueError("current_pricing_detail.instrument_dump is missing; cannot rebuild instrument.")

    instrument = msi.Instrument.rebuild(inst_dump)
    if not hasattr(instrument, "get_cashflows_df"):
        raise AttributeError("Rebuilt instrument does not expose get_cashflows_df().")

    # Use a sane valuation date for standalone inspection
    instrument.set_valuation_date(datetime.datetime.now(pytz.utc))

    cf_df = instrument.get_cashflows_df()

    st.divider()
    st.subheader("Cashflows (rebuilt instrument)")
    st.dataframe(cf_df, width="stretch", hide_index=True)

    # Download
    file_tag = f"id_{id_raw}" if id_raw is not None else f"uid_{uid_raw}"
    st.download_button(
        "Download cashflows CSV",
        data=cf_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cashflows_{file_tag}.csv",
        mime="text/csv",
        key="dl_cashflows_csv",
    )
