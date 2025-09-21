# dashboards/curves/bumping.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple,Optional

import numpy as np
import QuantLib as ql


from mainsequence.instruments.pricing_models.indices import build_zero_curve
from mainsequence.instruments.utils import to_py_date
from mainsequence.instruments.settings import TIIE_28_UID,TIIE_182_UID
from mainsequence.instruments.instruments.position import Position,PositionLine
from dashboards.core.tenor import tenor_to_years
import re

try:
    from mainsequence.instruments.settings import (
        TIIE_28_UID, TIIE_182_UID, TIIE_OVERNIGHT_UID,
        CETE_28_UID, CETE_182_UID,
    )
    _CURVE_FAMILY_OVERRIDES = {
        TIIE_28_UID: "TIIE",
        TIIE_182_UID: "TIIE",
        TIIE_OVERNIGHT_UID: "TIIE",
        CETE_28_UID: "CETE",
        CETE_182_UID: "CETE",
    }
except Exception:
    _CURVE_FAMILY_OVERRIDES = {}
def curve_family_key(index_uid: str) -> str:
    """
    Map an index UID like 'mxn:tiie-28d' or 'TIIE_182' to a canonical curve family
    (e.g., 'TIIE', 'CETE'). Uses explicit overrides first, then a robust heuristic.
    """
    if index_uid in _CURVE_FAMILY_OVERRIDES:
        return _CURVE_FAMILY_OVERRIDES[index_uid]

    u = str(index_uid).upper()
    # Drop any currency prefix like 'MXN:'.
    u = u.split(":", 1)[-1]
    # Normalize separators to underscore, then take the head token.
    u = re.sub(r"[-/]", "_", u)
    head = u.split("_", 1)[0]
    return head or u

KEYRATE_GRID = {TIIE_28_UID: ("28D", "91D", "182D", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
                TIIE_182_UID: ("28D", "91D", "182D", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
                TIIE_182_UID: ("28D", "91D", "182D", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
                CETE_28_UID: ("28D", "91D", "182D", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
                CETE_182_UID: ("28D", "91D", "182D", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),

                }



@dataclass(frozen=True)
class BumpSpec:
    """A currency‑agnostic description of curve bumps."""
    keyrate_bp: Dict[str, float] = field(default_factory=dict)  # e.g. {"5Y": 50.0}
    parallel_bp: float = 0.0                                    # e.g. +10 bp
    key_rate_grid: Dict=  field(default_factory=lambda: KEYRATE_GRID)

    def normalized(self) -> "BumpSpec":
        """Uppercase tenors; ensure floats."""
        return BumpSpec(
            keyrate_bp={ (k or "").strip().upper(): float(v) for k, v in (self.keyrate_bp or {}).items() },
            parallel_bp=float(self.parallel_bp),
            key_rate_grid=self.key_rate_grid,
        )
def bump_spec_from_dict(d: dict[str, float],  grid: Dict,parallel_bp: float = 0.0, ) -> BumpSpec:
    return BumpSpec(keyrate_bp=d or {}, parallel_bp=float(parallel_bp), key_rate_grid=grid)



def _no_bumps(spec: BumpSpec) -> bool:
    s = spec.normalized()
    if abs(float(s.parallel_bp)) >= 1e-12:
        return False
    return not any(abs(v) >= 1e-12 for v in (s.keyrate_bp or {}).values())

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
                     index_identifier: str,
                     max_years: int = 12,
                     step_months: int = 3,
                     *,
                     dense_short_years: int = 2,
                     extra_anchor_dates: list[ql.Date] | None = None) -> ql.YieldTermStructureHandle:
    import math
    spec = spec.normalized()
    par_spread = float(spec.parallel_bp) / 10_000.0

    if _no_bumps(spec):
        return base_ts

    ref = base_ts.referenceDate()
    if calc_date != ref:
        calc_date = ref

    if (not spec.keyrate_bp) or all(abs(v) < 1e-12 for v in spec.keyrate_bp.values()):
        spread = ql.QuoteHandle(ql.SimpleQuote(par_spread))
        bumped = ql.ZeroSpreadedTermStructure(base_ts, spread, ql.Continuous, ql.Annual)
        bumped.enableExtrapolation()
        return ql.YieldTermStructureHandle(bumped)

    cal = ql.TARGET()
    dc = base_ts.dayCounter()

    dates: list[ql.Date] = [calc_date]
    for d in range(1, dense_short_years * 365 + 1):
        dates.append(cal.advance(calc_date, ql.Period(d, ql.Days)))

    grid_for_index = tuple(spec.key_rate_grid.get(index_identifier, ()))
    if not grid_for_index:
        raise KeyError(f"KEYRATE_GRID has no entry for index '{index_identifier}'.")

    grid_max = max((tenor_to_years(t) for t in grid_for_index), default=0.0)
    horizon_years = max(float(max_years), math.ceil(grid_max))
    for m in range(dense_short_years * 12 + step_months, int(horizon_years * 12) + 1, step_months):
        dates.append(cal.advance(calc_date, ql.Period(m, ql.Months)))

    if extra_anchor_dates:
        dates.extend(extra_anchor_dates)
    dates = sorted(set(dates))
    t = np.array([dc.yearFraction(calc_date, d) for d in dates], dtype=float)

    D0 = np.array([base_ts.discount(d) for d in dates], dtype=float)

    grid_pairs = sorted(((tenor, tenor_to_years(tenor)) for tenor in grid_for_index), key=lambda p: p[1])
    xs = np.array([0.0] + [yr for _, yr in grid_pairs], dtype=float)
    ys = np.array([0.0] + [float(spec.keyrate_bp.get(tenor, 0.0)) / 10_000.0 for tenor, _ in grid_pairs], dtype=float)
    s_key = np.interp(t, xs, ys, left=0.0, right=0.0)
    s_tot = s_key + par_spread

    Db = D0 * np.exp(-s_tot * t)
    Db[0] = 1.0

    bumped = ql.DiscountCurve(dates, Db.tolist(), dc, cal)
    bumped.enableExtrapolation()
    return ql.YieldTermStructureHandle(bumped)




# --- module-local caches & helpers ---

_BASE_TS_CACHE: dict[tuple[str, str], ql.YieldTermStructureHandle] = {}
#keys: (date_key, index_identifier)
_BUMP_TS_CACHE: dict[tuple[str, str, tuple], ql.YieldTermStructureHandle] = {}

def _date_key(d: ql.Date) -> str:
    return f"{d.year():04d}-{d.month():02d}-{d.dayOfMonth():02d}"

def _freeze_grid(grid) -> tuple:
    """
    Convert the key‑rate grid mapping into a deterministic, hashable tuple.
    Shape: ((index_id, (TENOR1, TENOR2, ...)), ...), sorted by index_id.
    """
    if not grid:
        return ()
    try:
        return tuple(
            sorted(
                (str(k), tuple(str(t).upper() for t in (v if isinstance(v, (list, tuple)) else [v])))
                for k, v in dict(grid).items()
            )
        )
    except Exception:
        # Fallback so hashing never raises
        return (repr(grid),)

def _spec_key(spec: BumpSpec) -> tuple:
    """
    Build a hashable, stable key from the spec:
      (parallel_bp, sorted((tenor, bp)), frozen_grid)
    Rounds to avoid float jitter.
    """
    s = spec.normalized()
    par = round(float(s.parallel_bp), 10)
    items = tuple(sorted((str(k).upper(), round(float(v), 10)) for k, v in (s.keyrate_bp or {}).items()))
    frozen_grid = _freeze_grid(s.key_rate_grid)
    return (par, items, frozen_grid)


def clear_curve_cache() -> None:
    _BASE_TS_CACHE.clear()
    _BUMP_TS_CACHE.clear()

def build_curves_for_ui(calc_date: ql.Date, spec: BumpSpec, *,
                        index_identifier: str,
                        dense_short_years: int = 2,
                        step_months: int = 3,
                        extra_anchor_dates: list[ql.Date] | None = None):
    dkey = _date_key(calc_date)
    skey = _spec_key(spec)
    ikey = str(index_identifier)

    # ---- base curve (cache by date + index) ----
    base_ts = _BASE_TS_CACHE.get((dkey, ikey))
    if base_ts is None:
        base_raw = build_zero_curve(target_date=to_py_date(calc_date), index_identifier=index_identifier)
        base_ts = (base_raw if isinstance(base_raw, ql.YieldTermStructureHandle)
                   else ql.YieldTermStructureHandle(base_raw))
        base_ts.enableExtrapolation()
        _BASE_TS_CACHE[(dkey, ikey)] = base_ts

    # ---- bumped curve (cache by date + index + spec) ----
    if (abs(spec.parallel_bp) < 1e-12) and (not spec.keyrate_bp or all(abs(v) < 1e-12 for v in spec.keyrate_bp.values())):
        bumped_ts = base_ts
    else:
        bkey = (dkey, ikey, skey)
        bumped_ts = _BUMP_TS_CACHE.get(bkey)
        if bumped_ts is None:
            bumped_ts = _bump_zero_curve(
                base_ts, calc_date, spec, index_identifier,
                max_years=12, step_months=step_months,
                dense_short_years=dense_short_years,
                extra_anchor_dates=extra_anchor_dates
            )
            _BUMP_TS_CACHE[bkey] = bumped_ts

    tenors = tuple(spec.key_rate_grid.get(index_identifier, ()))
    if not tenors:
        raise KeyError(f"KEYRATE_GRID has no entry for index '{index_identifier}'.")
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