# ui/components/valuation_inputs.py
from __future__ import annotations
import datetime as dt
import streamlit as st

def valuation_controls(default_date: dt.date | str | None = None, *, key: str = "valuation") -> dt.date:
    """
    Sidebar control that returns a valuation date.
    - default_date: dt.date | ISO 'YYYY-MM-DD' | None (defaults to today's date)
    - key: to keep widget state isolated when used multiple times
    """
    if isinstance(default_date, str):
        try:
            default = dt.date.fromisoformat(default_date)
        except Exception:
            default = dt.date.today()
    else:
        default = default_date or dt.date.today()

    with st.sidebar:
        val_date = st.date_input("Valuation date", value=default, key=f"{key}_date")
    return val_date
