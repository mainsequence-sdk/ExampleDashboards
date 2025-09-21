
# dashboards/core/data_nodes.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import streamlit as st


from mainsequence.tdag import APIDataNode




@dataclass
class DataNodeDeps:
    """
    App-level registry for named data-node dependencies (independent of EngineMeta).
    Example: {"instrument_pricing_id": "vector_de_precios_valmer"}
    """
    deps: Dict[str, str] = field(default_factory=dict)
    _cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def register(self, **mapping: str) -> None:
        """Add/override dependencies by name."""
        self.deps.update({k: v for k, v in mapping.items() if v})

    def as_mapping(self) -> Dict[str, str]:
        """Raw name -> identifier mapping (for display)."""
        return dict(self.deps)

    def get(self, name: str):
        """Return a live APIDataNode for a registered dependency name."""
        ident = self.deps.get(name)
        if ident is None:
            raise KeyError(f"Unknown data-node dependency: {name}")
        if name in self._cache:
            return self._cache[name]
        if APIDataNode is None:
            raise ImportError("mainsequence-sdk is not available to build APIDataNode.")
        node = APIDataNode.build_from_identifier(identifier=ident)
        self._cache[name] = node
        return node

def set_app_data_nodes(mapping: Dict[str, str]) -> None:
    """Initialize the app-level dependencies once."""
    st.session_state["_app_data_nodes"] = DataNodeDeps(mapping)

def get_app_data_nodes(default: Optional[Dict[str, str]] = None) -> DataNodeDeps:
    """Access (and lazily create) the registry anywhere in the app."""
    obj = st.session_state.get("_app_data_nodes")
    if obj is None:
        obj = DataNodeDeps(default or {})
        st.session_state["_app_data_nodes"] = obj
    return obj
