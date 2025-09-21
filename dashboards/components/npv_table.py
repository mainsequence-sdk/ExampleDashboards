from __future__ import annotations

from typing import Optional, Iterable, Callable, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from mainsequence.instruments.instruments.position import Position, npv_table
from dashboards.core.formatters import fmt_ccy, fmt_units
from dashboards.curves.z_spread import zspread_from_dirty_ccy,make_zero_spreaded_handle,dirty_price_ccy_with_curve


# ---------- base (non‑paginated) builders ----------

def _fallback_npvs_by_id(position: Position, apply_units: bool = True) -> dict:
    out = {}
    for line in position.lines:
        ins = line.instrument
        ins_id = ins.content_hash()
        pv = float(ins.price())
        if apply_units:
            pv *= float(line.units)
        out[ins_id] = pv
    return out


def _fallback_units_by_id(position: Position) -> dict:
    return {line.instrument.content_hash(): float(line.units) for line in position.lines}


def _fallback_npv_table(npv_base: dict, npv_bumped: dict | None, units: dict | None,
                        include_total: bool = True) -> pd.DataFrame:
    ids = sorted(npv_base.keys())
    rows = []
    for ins_id in ids:
        base = float(npv_base.get(ins_id, np.nan))
        bumped = float(npv_bumped.get(ins_id, np.nan)) if npv_bumped is not None else np.nan
        delta = bumped - base if (npv_bumped is not None and np.isfinite(base) and np.isfinite(bumped)) else np.nan
        u = float(units.get(ins_id, np.nan)) if units else np.nan
        rows.append({"instrument": ins_id, "units": u, "base": base, "bumped": bumped, "delta": delta})
    df = pd.DataFrame(rows)
    if include_total and not df.empty:
        tot = {
            "instrument": "TOTAL", "units": np.nan,
            "base": df["base"].sum(skipna=True),
            "bumped": df["bumped"].sum(skipna=True) if npv_bumped is not None else np.nan,
            "delta": df["delta"].sum(skipna=True) if npv_bumped is not None else np.nan,
        }
        df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    return df


def build_position_npv_table(position: Position,
                             currency_symbol: str = "$",
                             bumped_position: Optional[Position] = None,
                             instrument_hash_to_asset: Optional[dict] = None,
                             *,
                             add_details_link: bool = True,
                             details_url_builder: Optional[Callable[[str], str]] = None,
                             extra_cols: Optional[Dict[str, Iterable[Any]]] = None) -> pd.DataFrame:




    base_npvs = {line.instrument.content_hash(): line.extra_market_info["dirty_price"]*line.units for line in position.lines}

    bump_npvs={}
    for line in bumped_position.lines:
        line.instrument.price()
        z=line.extra_market_info["z_spread"]
        bumped_curve=line.instrument.get_index_curve()
        h_base_z = make_zero_spreaded_handle(bumped_curve, z)
        dirty_price=dirty_price_ccy_with_curve(line.instrument._bond, h_base_z)
        bump_npvs[line.instrument.content_hash()] = dirty_price*line.units
        line.extra_market_info["dirty_price"]=dirty_price


    units = position.units_by_id()
    raw = npv_table(base_npvs, bump_npvs, units, include_total=False)

    index_by_id = {
        line.instrument.content_hash(): getattr(line.instrument, "floating_rate_index_name", "")
        for line in position.lines
    }

    d = raw.copy()
    d["Instrument"] = d["instrument"].astype(str)
    d["Index"] = d["instrument"].map(index_by_id).fillna("")
    if instrument_hash_to_asset is not None:
        hash_to_uid={k:v.unique_identifier for k,v in instrument_hash_to_asset.items()}
        d["Instrument"]=d["Instrument"].map(hash_to_uid)

    d["Units"] = d["units"].apply(lambda v: "" if pd.isna(v) else fmt_units(v))
    d["Base"] = d["base"].apply(lambda v: fmt_ccy(v, currency_symbol))
    if "bumped" in d.columns:
        d["Bumped"] = d["bumped"].apply(lambda v: "" if pd.isna(v) else fmt_ccy(v, currency_symbol))
    if "delta" in d.columns:
        d["Δ"] = d["delta"].apply(lambda v: "" if pd.isna(v) else fmt_ccy(v, currency_symbol))

    if extra_cols:
        for k, col in extra_cols.items():
            d[k] = list(col)

    cols = ["Instrument","Index", "Units", "Base"]
    if "Bumped" in d.columns: cols.append("Bumped")
    if "Δ" in d.columns: cols.append("Δ")
    if extra_cols: cols += [k for k in extra_cols.keys()]

    if add_details_link:
        builder = details_url_builder or (lambda x: f"/?asset_id={x}")
        d["Details"] = d["Instrument"].map(builder)
        cols.append("Details")

    return d[cols]


# ---------- generic pagination helper ----------

def _filter_df_contains(df: pd.DataFrame, *, query: str, cols: Iterable[str]) -> pd.DataFrame:
    if not query:
        return df
    q = query.strip().lower()
    mask = np.zeros(len(df), dtype=bool)
    for c in cols:
        if c in df.columns:
            mask |= df[c].astype(str).str.lower().str.contains(q, na=False).values
    return df.loc[mask]


def st_paginated_df(df: pd.DataFrame,
                    *,
                    key: str,
                    page_size_options=(25, 50, 100, 200),
                    default_size=50,
                    enable_search=True,
                    search_cols=("Instrument",),
                    sortable_cols=("Instrument", "Base", "Bumped", "Δ", "Units")) -> pd.DataFrame:
    """Render df with server-side pagination, light search, and single-column sorting."""
    ss = st.session_state

    cols = st.columns([1, 1, 1, 3])
    with cols[0]:
        size = st.selectbox("Rows/page", page_size_options, index=page_size_options.index(default_size)
        if default_size in page_size_options else 0, key=f"{key}_pagesize")
    with cols[1]:
        sort_col = st.selectbox("Sort by", [c for c in sortable_cols if c in df.columns], key=f"{key}_sortcol")
    with cols[2]:
        asc = st.toggle("Asc", value=False, key=f"{key}_asc")
    with cols[3]:
        query = st.text_input("Search", value="", key=f"{key}_search") if enable_search else ""

    # Filter + sort (unchanged) ...
    view = _filter_df_contains(df, query=query if enable_search else "", cols=search_cols)
    if sort_col in view.columns:
        def _to_float(s: pd.Series) -> pd.Series:
            try:
                return s.str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.replace("MXN$ ", "",
                                                                                                         regex=False).astype(
                    float)
            except Exception:
                return pd.to_numeric(s, errors="coerce")

        if view[sort_col].dtype == object:
            if sort_col in ("Base", "Bumped", "Δ"):
                view = view.assign(_sort=_to_float(view[sort_col]))
            else:
                view = view.assign(_sort=view[sort_col].astype(str))
        else:
            view = view.assign(_sort=view[sort_col])
        view = view.sort_values(by="_sort", ascending=asc, kind="mergesort").drop(columns="_sort")

    # Pagination (unchanged) ...
    total = len(view)
    pages = max(1, int(np.ceil(total / size)))
    cur = ss.get(f"{key}_page", 1)
    cur = max(1, min(cur, pages))
    start = (cur - 1) * size
    end = start + size
    page_df = view.iloc[start:end]

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("◀ Prev", disabled=(cur <= 1), key=f"{key}_prev"):
            cur = max(1, cur - 1)
    with c2:
        st.markdown(f"<div style='text-align:center'>Page <b>{cur}</b> / {pages} — {total} rows</div>",
                    unsafe_allow_html=True)
    with c3:
        if st.button("Next ▶", disabled=(cur >= pages), key=f"{key}_next"):
            cur = min(pages, cur + 1)
    ss[f"{key}_page"] = cur

    # Make 'Details' clickable when present
    col_config = {}
    if "Details" in page_df.columns:
        col_config["Details"] = st.column_config.LinkColumn(
            "Details", help="Open Asset Detail", display_text="Open"
        )

    st.dataframe(
        page_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
        key=f"{key}_table"
    )

    st.download_button("Download CSV (filtered)", data=view.to_csv(index=False).encode("utf-8"),
                       file_name="npv_table_filtered.csv", mime="text/csv", key=f"{key}_dl")

    return page_df


# ---------- paginated NPV table (Position) ----------

def st_position_npv_table_paginated(position: Position,
                                    instrument_hash_to_asset: dict,
                                    currency_symbol: str = "$",
                                    bumped_position: Optional[Position] = None,
                                    details_url_builder: Optional[Callable[[str], str]] = None,
                                    extra_cols: Optional[Dict[str, Iterable[Any]]] = None,
                                    *,
                                    page_size_options=(25, 50, 100, 200),
                                    default_size=50,
                                    enable_search=True,
                                    key: str = "npv_table") -> pd.DataFrame:
    """Build the NPV table and render it with pagination."""
    disp = build_position_npv_table(position, currency_symbol, bumped_position,
                                    instrument_hash_to_asset=instrument_hash_to_asset,
                                    add_details_link=True,
                                    details_url_builder=details_url_builder,
                                    extra_cols=extra_cols)
    return st_paginated_df(
        disp,
        key=key,
        page_size_options=page_size_options,
        default_size=default_size,
        enable_search=enable_search,
        search_cols=("Instrument",),
        sortable_cols=("Instrument", "Base", "Bumped", "Δ", "Units", "Details"),
    )
