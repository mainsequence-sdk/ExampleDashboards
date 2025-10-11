from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple,Optional

import numpy as np
import QuantLib as ql


from mainsequence.instruments.pricing_models.indices import build_zero_curve
from mainsequence.instruments.utils import to_py_date
from mainsequence.instruments.instruments.position import Position,PositionLine
from dashboards.core.tenor import tenor_to_years
import re

from dataclasses import dataclass
from typing import Protocol, Tuple, List,Optional, Union
import numpy as np
import QuantLib as ql
import math

from mainsequence.instruments.pricing_models.indices import get_index
from mainsequence.instruments.pricing_models.swap_pricer import price_vanilla_swap_with_curve
from mainsequence.instruments.utils import to_py_date


from mainsequence.client import Constant as _C

_CURVE_FAMILY_OVERRIDES = {

_C.get_value(name="REFERENCE_RATE__TIIE_28"):"TIIE",
_C.get_value(name="REFERENCE_RATE__TIIE_182"):"TIIE",
_C.get_value(name="REFERENCE_RATE__TIIE_OVERNIGHT"):"TIIE",
_C.get_value(name="REFERENCE_RATE__CETE_28"):"CETE",
_C.get_value(name="REFERENCE_RATE__CETE_91"):"CETE",
_C.get_value(name="REFERENCE_RATE__CETE_182"):"CETE",
_C.get_value(name="REFERENCE_RATE__USD_SOFR"):"SOFR",

}

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

KEYRATE_GRID_BY_FAMILY = {
    "TIIE": ("28D","91D","182D","1Y","2Y","3Y","5Y","7Y","10Y","20Y","30Y"),
    "CETE": ("28D","91D","182D","1Y","2Y","3Y","5Y","7Y","10Y","20Y","30Y"),
    "UST": ("30D", "90D", "180D", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
    "SOFR": ("30D", "90D", "180D", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
}


def keyrate_grid_for_index(index_uid: str) -> tuple[str, ...]:
    fam = curve_family_key(index_uid)
    return KEYRATE_GRID_BY_FAMILY.get(fam, ())

@dataclass(frozen=True)
class BumpSpec:
    """A currency‑agnostic description of curve bumps."""
    keyrate_bp: Dict[str, float] = field(default_factory=dict)  # e.g. {"5Y": 50.0}
    parallel_bp: float = 0.0                                    # e.g. +10 bp
    key_rate_grid: Dict=  field(default_factory=lambda: KEYRATE_GRID_BY_FAMILY)

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

    tenors = tuple(KEYRATE_GRID_BY_FAMILY[_CURVE_FAMILY_OVERRIDES[index_identifier]])
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




@dataclass(frozen=True)
class SwapConventions:
    fixed_leg_tenor: ql.Period
    float_leg_tenor: ql.Period
    fixed_leg_daycount: ql.DayCounter
    fixed_leg_convention: ql.BusinessDayConvention
    float_leg_spread: float = 0.0

class ParRateCalculator(Protocol):
    def fair_rate(self, start: ql.Date, maturity: ql.Date, curve: ql.YieldTermStructureHandle) -> float: ...

class TIIE28ParCalculator:
    def __init__(self, index_identifier: str ):
        self.index_identifier = index_identifier
        self._conv = SwapConventions(
            fixed_leg_tenor=ql.Period(28, ql.Days),
            float_leg_tenor=ql.Period(28, ql.Days),
            fixed_leg_daycount=ql.Actual360(),
            fixed_leg_convention=ql.ModifiedFollowing,
            float_leg_spread=0.0,
        )

    def _index(self, ref: ql.Date):
        return get_index(target_date=to_py_date(ref), index_identifier=self.index_identifier)

    def fair_rate(self, start: ql.Date, maturity: ql.Date, curve: ql.YieldTermStructureHandle) -> float:
        ibor = self._index(start)
        swap = price_vanilla_swap_with_curve(
            calculation_date=start,
            notional=1.0,
            start_date=start,
            maturity_date=maturity,
            fixed_rate=0.0,
            fixed_leg_tenor=self._conv.fixed_leg_tenor,
            fixed_leg_convention=self._conv.fixed_leg_convention,
            fixed_leg_daycount=self._conv.fixed_leg_daycount,
            float_leg_tenor=self._conv.float_leg_tenor,
            float_leg_spread=self._conv.float_leg_spread,
            ibor_index=ibor,
            curve=curve,
        )
        return swap.fairRate()

def par_curve(ts: ql.YieldTermStructureHandle,
              max_years: int,
              step_months: int,
              calc: ParRateCalculator) -> Tuple[np.ndarray, np.ndarray]:
    spot = ts.referenceDate()
    cal = ts.calendar()
    xdc = ql.Actual360()  # x-axis year fraction (as in your current plot)
    T, Y = [], []
    m = step_months
    while m <= max_years * 12:
        mat = cal.advance(spot, ql.Period(m, ql.Months))
        T.append(xdc.yearFraction(spot, mat))
        Y.append(calc.fair_rate(spot, mat, ts))
        m += step_months
    return np.array(T), np.array(Y)

def par_nodes_from_tenors(ts: ql.YieldTermStructureHandle,
                          nodes: List[dict],
                          calc: ParRateCalculator) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    spot = ts.referenceDate()
    cal_axis = ql.TARGET()
    dc_axis = ql.Actual365Fixed()
    xs, ys, labels = [], [], []
    for n in nodes:
        tstr = (n.get("tenor") or "").upper()
        if not tstr:
            continue
        per = ql.Period(tstr)
        mat = cal_axis.advance(spot, per)
        xs.append(dc_axis.yearFraction(spot, mat))
        ys.append(calc.fair_rate(spot, mat, ts))
        labels.append(f"SWAP {tstr}")
    return np.array(xs), np.array(ys), labels




import math
import QuantLib as ql
from typing import Optional, Union, Callable

CurveLike = Union[ql.YieldTermStructure, ql.YieldTermStructureHandle]

def _extract_discount_curve_from_engine(bond: ql.Bond) -> Optional[ql.YieldTermStructureHandle]:
    try:
        eng = bond.pricingEngine()
        if eng is not None and hasattr(eng, "discountCurve"):
            h = eng.discountCurve()
            if isinstance(h, ql.YieldTermStructureHandle) and (not h.empty()):
                return h
    except Exception:
        pass
    return None

def zspread_from_dirty_ccy(
    bond: ql.Bond,
    target_dirty_ccy: float,
    discount_curve: Optional[CurveLike] = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 200
) -> float:
    """
    Compute the constant z-spread (decimal per year, e.g. 0.005 = 50 bp) to add to the
    discount curve so that the bond's DIRTY price equals `target_dirty_ccy` (currency).

    - No per-100 or face-value assumptions.
    - Uses the curve's own day counter.
    - Uses continuous compounding internally (standard for z-spread).
    - Respects Settings.includeReferenceDateEvents via cf.hasOccurred(settlement).
    """
    # 1) Discount curve handle
    if discount_curve is None:
        h = _extract_discount_curve_from_engine(bond)
        if h is None:
            raise ValueError("No DiscountingBondEngine on bond; pass `discount_curve` explicitly.")
    else:
        h = discount_curve if isinstance(discount_curve, ql.YieldTermStructureHandle) \
                           else ql.YieldTermStructureHandle(discount_curve)

    dc     = h.dayCounter()
    ref    = h.referenceDate()
    settle = bond.settlementDate()

    # 2) Future cashflows (currency amounts), honoring includeReferenceDateEvents
    flows = []
    for cf in bond.cashflows():
        if cf.hasOccurred(settle):
            continue
        flows.append((cf.date(), float(cf.amount())))
    if not flows:
        raise ValueError("No future cashflows; z-spread is undefined.")

    base_df_settle = h.discount(settle)
    t_settle       = dc.yearFraction(ref, settle)

    def pv_with_z(s: float) -> float:
        """Dirty PV in currency using base curve + constant z (continuous comp)."""
        z_settle = math.exp(-s * t_settle)
        pv = 0.0
        for d, amt in flows:
            t  = dc.yearFraction(ref, d)
            df = h.discount(d) * math.exp(-s * t)
            pv += amt * (df / (base_df_settle * z_settle))
        return pv

    def f(s: float) -> float:
        return pv_with_z(s) - float(target_dirty_ccy)

    # 3) Bracket & solve (Brent)
    a, b = -0.05, 0.05  # ±500 bp to start
    fa, fb = f(a), f(b)
    expand = 0
    while fa * fb > 0.0 and expand < 24:
        a *= 2.0; b *= 2.0
        fa, fb = f(a), f(b)
        expand += 1
    if fa * fb > 0.0:
        # Not bracketed — fail deterministically (no silent "best endpoint").
        raise RuntimeError("z-spread: could not bracket the root. Check target price and curve.")

    def _solve_brent_on_bracket(func: Callable[[float], float], lo: float, hi: float,
                                acc: float) -> float:
        # QuantLib.Brent signature: (f, accuracy, guess, step)
        guess = 0.5 * (lo + hi)
        step = 0.5 * (hi - lo)
        try:
            return float(ql.Brent().solve(lambda x: func(x), float(acc), float(guess), float(step)))
        except Exception:
            # Robust fallback: deterministic bisection on the *existing* bracket
            flo, fhi = func(lo), func(hi)
            # orient so flo <= 0 <= fhi if possible (doesn't affect convergence)
            if flo > 0 and fhi < 0:
                lo, hi, flo, fhi = hi, lo, fhi, flo
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                fmid = func(mid)
                if abs(fmid) < acc or 0.5 * (hi - lo) < acc:
                    return float(mid)
                if fmid < 0.0:
                    lo, flo = mid, fmid
                else:
                    hi, fhi = mid, fhi
            return float(0.5 * (lo + hi))

    return _solve_brent_on_bracket(f, a, b, tol)

def make_zero_spreaded_handle(base: CurveLike, z: float) -> ql.YieldTermStructureHandle:
    h = base if isinstance(base, ql.YieldTermStructureHandle) else ql.YieldTermStructureHandle(base)
    q = ql.SimpleQuote(z)
    ts = ql.ZeroSpreadedTermStructure(h, ql.QuoteHandle(q), ql.Continuous, ql.NoFrequency)
    return ql.YieldTermStructureHandle(ts)

def dirty_price_ccy_with_curve(bond: ql.Bond, discount_curve: CurveLike) -> float:
    """Dirty PV in currency using a given discount curve (no per-100 anywhere)."""
    h  = discount_curve if isinstance(discount_curve, ql.YieldTermStructureHandle) \
                        else ql.YieldTermStructureHandle(discount_curve)

    dc     = h.dayCounter()
    ref    = h.referenceDate()
    settle = bond.settlementDate()
    df_set = h.discount(settle)
    pv = 0.0
    for cf in bond.cashflows():
        if cf.hasOccurred(settle):
            continue
        d = cf.date()
        pv += float(cf.amount()) * (h.discount(d) / df_set)
    return pv