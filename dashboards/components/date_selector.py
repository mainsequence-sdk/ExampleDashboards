# dashboards/components/date_selector.py
from __future__ import annotations

import datetime as dt
import streamlit as st


def date_selector(
    *,
    label: str = "Valuation date",
    session_cfg_key: str = "position_cfg_mem",
    cfg_field: str = "valuation_date",
    key: str = "valuation_date_input",
    help: str | None = None,
) -> dt.date:
    """
    Re-usable date selector that persists its value into st.session_state[session_cfg_key][cfg_field]
    as ISO date. Returns a `datetime.date`.

    - Reads default from session cfg if present, else today.
    - On change, updates the session cfg (no explicit st.rerun needed).
    """
    cfg = st.session_state.get(session_cfg_key) or {}
    iso_default = cfg.get(cfg_field)
    try:
        default = dt.date.fromisoformat(iso_default) if iso_default else dt.date(2025,8,23)
    except Exception:
        default = dt.date.today()

    selected = st.date_input(label, value=default, key=key, help=help, format="YYYY-MM-DD")
    if isinstance(selected, dt.datetime):  # streamlit can return datetime
        selected = selected.date()

    new_iso = selected.isoformat()
    if cfg.get(cfg_field) != new_iso:
        st.session_state[session_cfg_key] = {**cfg, cfg_field: new_iso}

    return selected
