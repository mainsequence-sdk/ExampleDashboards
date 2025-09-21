# dashboards/core/state.py
from __future__ import annotations
import streamlit as st

def ns(prefix: str, name: str) -> str:
    return f"{prefix}:{name}"

def get(name: str, *, prefix: str = "", default=None):
    return st.session_state.get(ns(prefix, name) if prefix else name, default)

def set(name: str, value, *, prefix: str = ""):
    st.session_state[ns(prefix, name) if prefix else name] = value

def pop(name: str, *, prefix: str = ""):
    return st.session_state.pop(ns(prefix, name) if prefix else name, None)
