# dashboards/services/contexts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Protocol

import datetime as dt
import QuantLib as ql

# ---- protocol: what a builder returns (app-defined payload) ----

class AppContextPayload(Protocol):
    """Marker protocol; concrete apps use their own dataclass payloads."""
    pass

# ---- registry ----

_Builders: Dict[str, Callable[[Dict[str, Any]], AppContextPayload]] = {}

def register_context(app_slug: str, builder: Callable[[Dict[str, Any]], AppContextPayload]) -> None:
    _Builders[app_slug] = builder

def build_context(app_slug: str, session_state: Dict[str, Any]) -> AppContextPayload:
    fn = _Builders.get(app_slug)
    if fn is None:
        raise KeyError(f"No context builder registered for '{app_slug}'.")
    return fn(session_state)
