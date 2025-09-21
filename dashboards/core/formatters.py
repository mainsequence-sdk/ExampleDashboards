# dashboards/core/formatters.py
from __future__ import annotations
import math

def fmt_ccy(x: float, symbol: str = "$", precision: int = 2) -> str:
    return "—" if x is None or not math.isfinite(float(x)) else f"{symbol}{x:,.{precision}f}"

def fmt_units(u: float, precision: int = 2) -> str:
    try:
        return f"{float(u):,.{precision}f}"
    except Exception:
        return str(u)

def fmt_bp(x: float) -> str:
    return f"{float(x):+.1f} bp"

def fmt_pct(x: float, precision: int = 3) -> str:
    return f"{float(x):.{precision}f}%"
