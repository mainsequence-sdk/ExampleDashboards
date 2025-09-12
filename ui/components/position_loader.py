# ui/components/position_loader.py
from __future__ import annotations

import json, hashlib
from pathlib import Path
import datetime as dt
import streamlit as st

from src.instruments.position import Position
# PositionLine import for safe instrument copying (repo layouts may differ)
try:
    from src.instruments import PositionLine
except Exception:
    from src.instruments.position import PositionLine  # fallback


# ---- utilities ----

def _file_sha1(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha1()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def _read_cfg_cached(path_str: str, content_hash: str) -> dict:
    """
    Cached by (path_str, content_hash). If content changes -> new hash -> re-read.
    """
    with open(path_str, "r") as fh:
        return json.load(fh)


@st.cache_resource(show_spinner=False)
def _build_position_template_cached(position_json_str: str) -> Position:
    """
    Build a Position from the JSON part only once.
    Use cache_resource so we don't serialize large objects repeatedly.
    """
    pos_dict = json.loads(position_json_str)
    return Position.from_json_dict(pos_dict)


def load_position_cached(path: str | Path) -> tuple[dict, Position, str]:
    """
    Reads the cfg JSON and builds a Position *template* (no curve, no valuation date).
    Returns (cfg_dict, template_position, content_hash).
    Only re-runs when file content changes.
    """
    path_str = str(path)
    sig = _file_sha1(path_str)
    cfg = _read_cfg_cached(path_str, sig)
    pos_json_str = json.dumps(cfg.get("position", {}), sort_keys=True)
    template = _build_position_template_cached(pos_json_str)
    return cfg, template, sig


def instantiate_position(template: Position,
                         curve,
                         valuation_date: dt.date) -> Position:
    """
    Create a fresh Position instance from the cached template,
    applying valuation_date and curve to each instrument without mutating the template.
    """
    new_lines = []
    for line in template.lines:
        inst = line.instrument.copy()
        inst.valuation_date = valuation_date
        inst.reset_curve(curve)
        new_lines.append(PositionLine(units=line.units, instrument=inst))
    return Position(lines=new_lines)


# ---- optional: tiny sidebar convenience wrapper ----

def position_source_input(default_path: str | Path, *, key: str = "pos_src") -> tuple[str, dict, Position, str]:
    """
    Sidebar control to pick a file path and load the position (cached-on-content).
    Returns (path_str, cfg, template_position, content_hash).
    """
    with st.sidebar:
        path_str = st.text_input("Position JSON path", value=str(default_path), key=f"{key}_path")
    try:
        cfg, template, sig = load_position_cached(path_str)
        st.sidebar.caption(f"Loaded • {len(template.lines)} lines")
        return path_str, cfg, template, sig
    except Exception as e:
        st.sidebar.error(f"Failed to read {path_str}\n{e}")
        st.stop()
