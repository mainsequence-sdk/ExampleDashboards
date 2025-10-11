# dashboards/apps/floating_portfolio_analysis/pages/99_Asset_Detail.py
from __future__ import annotations
import json, datetime, pytz
from typing import Any, Dict, Optional
import streamlit as st
import mainsequence.client as msc
import mainsequence.instruments as msi

st.title("Asset Detail")

def _qp_first(key: str) -> Optional[str]:
    v = st.query_params.get(key, None)
    if isinstance(v, list):
        return v[0] if v else None
    return v

def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"): return obj.model_dump()
    if hasattr(obj, "dict"): return obj.dict()
    if hasattr(obj, "json"): return json.loads(obj.json())
    return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))

# Parse query
id_raw = _qp_first("id")
uid_raw = _qp_first("unique_identifier")
if (id_raw is None and uid_raw is None) or (id_raw is not None and uid_raw is not None):
    st.warning("Provide exactly one query parameter: either 'id' (int) or 'unique_identifier' (str).")
    st.stop()

# Fetch
if id_raw is not None:
    if not str(id_raw).isdigit():
        st.error("'id' must be an integer.")
        st.stop()
    asset = msc.Asset.get_or_none(id=int(id_raw))
    if asset is None:
        st.error(f"Asset with id={id_raw} was not found.")
        st.stop()
else:
    uid = str(uid_raw)
    asset = msc.Asset.get_or_none(unique_identifier=uid)
    if asset is None:
        st.error(f"Asset with unique_identifier='{uid}' was not found.")
        st.stop()

# Identity chips
uid_val = getattr(asset, "unique_identifier", None)
snapshot_name = getattr(getattr(asset, "current_snapshot", None), "name", None)
c1, c2 = st.columns(2)
with c1: st.metric("Unique Identifier", uid_val or "—")
with c2: st.metric("Snapshot", snapshot_name or "—")
st.divider()

# Two columns: Asset JSON | Instrument dump
left, right = st.columns(2)
with left:
    st.subheader("Asset Detail")
    st.json(_to_dict(asset))

pricing_detail = getattr(asset, "current_pricing_detail", None)
if pricing_detail is None:
    st.error("Asset has no current_pricing_detail; cannot rebuild instrument for cashflows.")
    st.stop()

inst_dump = getattr(pricing_detail, "instrument_dump", None)
with right:
    st.subheader("Instrument dump (from platform)")
    st.json(_to_dict(inst_dump) if inst_dump is not None else {})

if inst_dump is None:
    st.error("current_pricing_detail.instrument_dump is missing; cannot rebuild instrument.")
    st.stop()

instrument = msi.Instrument.rebuild(inst_dump)
if not hasattr(instrument, "get_cashflows_df"):
    st.error("Rebuilt instrument does not expose get_cashflows_df().")
    st.stop()

instrument.set_valuation_date(datetime.datetime.now(pytz.utc))
cf_df = instrument.get_cashflows_df()

st.divider()
st.subheader("Cashflows (rebuilt instrument)")
st.dataframe(cf_df, width="stretch", hide_index=True)

file_tag = f"id_{id_raw}" if id_raw is not None else f"uid_{uid_raw}"
st.download_button(
    "Download cashflows CSV",
    data=cf_df.to_csv(index=False).encode("utf-8"),
    file_name=f"cashflows_{file_tag}.csv",
    mime="text/csv",
)
