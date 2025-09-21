# dashboards/core/tenor.py
from __future__ import annotations

def tenor_to_years(tenor: str) -> float:
    t = (tenor or "").strip().upper()
    if t.endswith("Y"):
        return float(t[:-1])
    if t.endswith("M"):
        return float(t[:-1]) / 12.0
    if t.endswith("D"):
        return float(t[:-1]) / 365.0
    return float(t)
