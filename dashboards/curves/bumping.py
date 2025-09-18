# src/curves/bumping.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import QuantLib as ql


from src.pricing_models.indices import build_zero_curve
from src.utils import to_py_date
from src.settings import TIIE_28_UID
from src.instruments.position import Position,PositionLine

KEYRATE_GRID_TIIE: Tuple[str, ...] = ("28D", "91D", "182D", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y","20Y","30Y")


@dataclass(frozen=True)
class BumpSpec:
    """A currency‑agnostic description of curve bumps."""
    keyrate_bp: Dict[str, float] = field(default_factory=dict)  # e.g. {"5Y": 50.0}
    parallel_bp: float = 0.0                                    # e.g. +10 bp
    # For TIIE bumping interpolation grid (years). Keep default unless you know what you’re doing.
    tiie_keyrate_grid: Tuple[str, ...] = KEYRATE_GRID_TIIE

    def normalized(self) -> "BumpSpec":
        """Uppercase tenors; ensure floats."""
        return BumpSpec(
            keyrate_bp={ (k or "").strip().upper(): float(v) for k, v in (self.keyrate_bp or {}).items() },
            parallel_bp=float(self.parallel_bp),
            tiie_keyrate_grid=self.tiie_keyrate_grid,
        )


def _nodes_from_ts(ts: ql.YieldTermStructureHandle, calc_date: ql.Date,
                   tenors: List[str]) -> List[dict]:
    cal, dc = ql.TARGET(), ql.Actual365Fixed()
    out: List[dict] = []
    ref = ts.referenceDate()
    for t in tenors:
        d = cal.advance(calc_date, ql.Period(t))
        if d < ref:
            d = ref
        r = ts.zeroRate(d, dc, ql.Continuous, ql.Annual).rate()
        out.append({"type": "zero", "tenor": t.upper(), "rate": float(r)})
    return out


def _bump_zero_curve(base_ts: ql.YieldTermStructureHandle,
                     calc_date: ql.Date,
                     spec: BumpSpec,
                     max_years: int = 12,
                     step_months: int = 3,
                     *,
                     dense_short_years: int = 2,
                     extra_anchor_dates: list[ql.Date] | None = None) -> ql.YieldTermStructureHandle:
    """
    Build a bumped curve by applying a time-varying *zero spread* in DISCOUNT space:

        D_bumped(t) = D_base(t) * exp( - (s_key(t) + s_par) * t )

    where s_key(t) is piecewise-linear (years) over spec.tiie_keyrate_grid (missing nodes = 0bp),
    and s_par is the constant parallel spread.

    Special cases handled exactly:
      - No bumps at all  -> return base_ts (no re-fitting, bit-for-bit equality).
      - Parallel-only    -> use QuantLib ZeroSpreadedTermStructure (no re-fitting).

    Parameters:
      dense_short_years:  add daily anchors up to this horizon to reduce short-end drift.
      extra_anchor_dates: optional list of dates to include as exact anchors (e.g., portfolio CF dates).

    Notes:
      - Uses the base term structure's day counter.
      - Reference date is pinned to base_ts.referenceDate() to avoid t=0 mismatches.
    """
    import math
    spec = spec.normalized()
    par_spread = float(spec.parallel_bp) / 10_000.0

    # 0) If no bumps at all -> return base curve unchanged
    if abs(par_spread) < 1e-12 and all(abs(v) < 1e-12 for v in (spec.keyrate_bp or {}).values()):
        return base_ts  # exact

    # Keep the reference date consistent with the base curve
    ref = base_ts.referenceDate()
    if calc_date != ref:
        calc_date = ref

    # 1) Parallel-only -> use ZeroSpreadedTermStructure (exact, no re-fitting)
    if (not spec.keyrate_bp) or all(abs(v) < 1e-12 for v in spec.keyrate_bp.values()):
        spread = ql.QuoteHandle(ql.SimpleQuote(par_spread))
        bumped = ql.ZeroSpreadedTermStructure(base_ts, spread, ql.Continuous, ql.Annual)
        bumped.enableExtrapolation()
        return ql.YieldTermStructureHandle(bumped)

    # 2) General case: key-rate bumps (possibly with parallel)
    cal = ql.TARGET()
    dc = base_ts.dayCounter()  # use the base curve's day count

    # --- Build a rich anchor grid of dates ---
    dates: list[ql.Date] = [calc_date]

    # Dense daily grid for the very front end (helps par swaps, short coupons)
    for d in range(1, dense_short_years * 365 + 1):
        dates.append(cal.advance(calc_date, ql.Period(d, ql.Days)))

    # Monthly grid thereafter — push horizon to at least the largest key‑rate tenor
    def _tenor_to_years(s: str) -> float:
        s = (s or "").strip().upper()
        if s.endswith("Y"): return float(s[:-1])
        if s.endswith("M"): return float(s[:-1]) / 12.0
        if s.endswith("D"): return float(s[:-1]) / 365.0
        return float(s)

    grid_max = 0.0 if not spec.tiie_keyrate_grid else max(_tenor_to_years(t) for t in spec.tiie_keyrate_grid)
    horizon_years = max(float(max_years), math.ceil(grid_max))  # e.g. ≥ 30y when grid has 30Y
    for m in range(dense_short_years * 12 + step_months, int(horizon_years * 12) + 1, step_months):
        dates.append(cal.advance(calc_date, ql.Period(m, ql.Months)))

    # Optional: include any extra anchors (e.g., portfolio cash-flow dates or par-maturity endpoints)
    if extra_anchor_dates:
        dates.extend(extra_anchor_dates)

    # Sort & deduplicate strictly
    dates = sorted(set(dates))

    # Time axis (years)
    t = np.array([dc.yearFraction(calc_date, d) for d in dates], dtype=float)

    # Base discounts
    D0 = np.array([base_ts.discount(d) for d in dates], dtype=float)

    # Piecewise‑linear key‑rate spread s_key(t) in decimals — strictly increasing support,
    # and 0bp outside the provided grid.
    grid_pairs = sorted(((tenor, _tenor_to_years(tenor)) for tenor in (spec.tiie_keyrate_grid or ())),
                        key=lambda p: p[1])
    xs = np.array([0.0] + [yr for _, yr in grid_pairs], dtype=float)
    ys = np.array([0.0] + [float(spec.keyrate_bp.get(tenor, 0.0)) / 10_000.0 for tenor, _ in grid_pairs],
                  dtype=float)
    s_key = np.interp(t, xs, ys, left=0.0, right=0.0)
    s_tot = s_key + par_spread

    # Apply spread in discount space. Ensure exact anchor at t=0.
    Db = D0 * np.exp(-s_tot * t)
    Db[0] = 1.0

    # Build the bumped curve from DISCOUNTS (log-linear in discounts is fine here).
    bumped = ql.DiscountCurve(dates, Db.tolist(), dc, cal)
    bumped.enableExtrapolation()
    return ql.YieldTermStructureHandle(bumped)



# --- module-local caches & helpers ---

_BASE_TS_CACHE: dict[str, ql.YieldTermStructureHandle] = {}
_BUMP_TS_CACHE: dict[tuple[str, tuple], ql.YieldTermStructureHandle] = {}

def _date_key(d: ql.Date) -> str:
    return f"{d.year():04d}-{d.month():02d}-{d.dayOfMonth():02d}"

def _spec_key(spec: BumpSpec) -> tuple:
    """
    Build a hashable, stable key from the spec:
      (parallel_bp, sorted((tenor, bp)), grid)
    Rounds to avoid float jitter.
    """
    s = spec.normalized()
    par = round(float(s.parallel_bp), 10)
    items = tuple(sorted((str(k).upper(), round(float(v), 10)) for k, v in (s.keyrate_bp or {}).items()))
    grid = tuple(s.tiie_keyrate_grid)
    return (par, items, grid)

def clear_curve_cache() -> None:
    _BASE_TS_CACHE.clear()
    _BUMP_TS_CACHE.clear()

def build_curves_for_ui(
        calc_date: ql.Date,
        spec: BumpSpec):
    """
    Facade used by dashboards (MXN/TIIE only here).

    Returns:
      base_ts, bumped_ts, nodes_base, nodes_bumped, index_hint_str
    """
    # ---- cache keys ----
    dkey = _date_key(calc_date)
    skey = _spec_key(spec)

    # ---- base curve (cache by date) ----
    base_ts = _BASE_TS_CACHE.get(dkey)
    if base_ts is None:
        base_raw = build_zero_curve(target_date=to_py_date(calc_date),index_identifier=TIIE_28_UID)
        base_ts = base_raw if isinstance(base_raw, ql.YieldTermStructureHandle) else ql.YieldTermStructureHandle(base_raw)
        base_ts.enableExtrapolation()
        _BASE_TS_CACHE[dkey] = base_ts

    # ---- bumped curve (cache by date + spec) ----
    # If there's absolutely no bump, reuse the base handle directly (exact identity)
    if (abs(spec.parallel_bp) < 1e-12) and (not spec.keyrate_bp or all(abs(v) < 1e-12 for v in spec.keyrate_bp.values())):
        bumped_ts = base_ts
    else:
        bkey = (dkey, skey)
        bumped_ts = _BUMP_TS_CACHE.get(bkey)
        if bumped_ts is None:
            bumped_ts = _bump_zero_curve(base_ts, calc_date, spec)
            _BUMP_TS_CACHE[bkey] = bumped_ts

    # ---- nodes (cheap; compute each call) ----
    tenors = list(spec.tiie_keyrate_grid)
    nodes_base = _nodes_from_ts(base_ts, calc_date, tenors)
    nodes_bump = _nodes_from_ts(bumped_ts, calc_date, tenors)

    return base_ts, bumped_ts, nodes_base, nodes_bump

def get_bumped_position(position: Position, bump_curve):
    bumped_lines = []

    for position_line in position.lines:
        base_instrument = position_line.instrument
        bumped_instrument = base_instrument.copy()
        bumped_instrument.reset_curve(bump_curve)
        bumped_line = PositionLine(units=position_line.units,
                                   instrument=bumped_instrument)
        bumped_lines.append(bumped_line)
    bumped_position = Position(lines=bumped_lines)
    return bumped_position