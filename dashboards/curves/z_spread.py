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