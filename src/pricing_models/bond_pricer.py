# src/pricing_models/bond_pricer.py
import QuantLib as ql
from typing import List, Dict, Any, Optional
from src.data_interface import data_interface
from src.utils import to_ql_date
import datetime
import matplotlib.pyplot as plt


def _map_daycount(dc: str) -> ql.DayCounter:
    s = (dc or '').upper()
    if s.startswith('30/360'):
        return ql.Thirty360(ql.Thirty360.USA)
    if s in ('ACT/365', 'ACT/365F', 'ACTUAL/365', 'ACTUAL/365F'):
        return ql.Actual365Fixed()
    if s in ('ACT/ACT', 'ACTUAL/ACTUAL'):
        return ql.ActualActual()
    return ql.Thirty360(ql.Thirty360.USA)


def build_discount_curve_from_bonds(calculation_date: ql.Date) -> ql.YieldTermStructureHandle:
    """
    Consume {"curve_nodes": [...]} with days_to_maturity.
      - type in {"zcb","zero","zero_coupon"}: zero-coupon bond given a zero *yield*.
        Convert yield -> clean price, then use BondHelper(quote, ZeroCouponBond).
      - type == "bond": fixed-rate bond quoted by clean price -> FixedRateBondHelper.
    """
    print("Building discount curve from 'discount_bond_curve' (days_to_maturity)...")
    market: Dict[str, Any] = data_interface.get_historical_data("discount_bond_curve", {"USD_bond_market": {}})
    curve_nodes: List[Dict[str, Any]] = market["curve_nodes"]

    calendar = ql.TARGET()
    dc365 = ql.Actual365Fixed()
    helpers: List[ql.RateHelper] = []
    settlement_days = 2

    for node in curve_nodes:
        ntype = node["type"].lower()
        days = int(node["days_to_maturity"])
        maturity = calendar.advance(calculation_date, days, ql.Days)

        if ntype in ("zcb", "zero", "zero_coupon"):
            zr = float(node["yield"])  # annual zero yield
            T = dc365.yearFraction(calculation_date, maturity)
            clean_price = 100.0 / ((1.0 + zr) ** T)  # price per 100

            # Build a ZeroCouponBond instrument and wrap it with a generic BondHelper
            zcb = ql.ZeroCouponBond(
                settlement_days, calendar, 100.0, maturity,
                ql.Following, 100.0, calculation_date
            )
            helper = ql.BondHelper(
                ql.QuoteHandle(ql.SimpleQuote(clean_price)), zcb
            )
            helpers.append(helper)

        elif ntype == "bond":
            coupon = float(node["coupon"])
            clean = float(node["clean_price"])  # per 100
            freq = ql.Period(node.get("frequency", "6M"))
            dcc = _map_daycount(node.get("day_count", "30/360"))
            issue = calculation_date  # mock issue = today

            sched = ql.Schedule(issue, maturity, freq, calendar,
                                ql.Unadjusted, ql.Unadjusted,
                                ql.DateGeneration.Forward, False)

            helper = ql.FixedRateBondHelper(
                ql.QuoteHandle(ql.SimpleQuote(clean)),
                settlement_days, 100.0, sched,
                [coupon], dcc, ql.Following, 100.0
            )
            helpers.append(helper)
        else:
            raise ValueError(f"Unsupported curve node type: {ntype}")

    curve = ql.PiecewiseLogCubicDiscount(calculation_date, helpers, dc365)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def plot_zero_coupon_curve(
        calculation_date: ql.Date | datetime.date,
        max_years: int = 30,
        step_months: int = 3,
        compounding=ql.Continuous,  # enums are ints; no type hint
        frequency=ql.Annual,  # enums are ints; no type hint
        show: bool = True,
        ax: Optional[plt.Axes] = None,
) -> tuple[list[float], list[float]]:
    """
    Plot the zero-coupon (spot) curve implied by the bond discount curve.

    Args:
        calculation_date: ql.Date or Python date used as curve 'as of' date.
        max_years: maximum maturity (in years) to sample.
        step_months: spacing of sample points in months.
        compounding: QuantLib compounding convention for zero rate extraction.
        frequency: compounding frequency (ignored for Continuous).
        show: if True, calls plt.show(). If False, just returns the data.
        ax: optional Matplotlib Axes to draw on.

    Returns:
        (tenors_in_years, zero_rates) where zero_rates are in decimals (e.g. 0.045).
    """

    # normalize date
    ql_calc_date = to_ql_date(calculation_date) if isinstance(calculation_date, datetime.date) else calculation_date
    ql.Settings.instance().evaluationDate = ql_calc_date

    # build curve
    ts_handle = build_discount_curve_from_bonds(ql_calc_date)
    ts = ts_handle.currentLink()

    calendar = ql.TARGET()
    day_count = ql.Actual365Fixed()

    # build sampling grid
    tenors_years: list[float] = []
    zero_rates: list[float] = []

    months = 0
    while months <= max_years * 12:
        months = max(months, 1)  # avoid exactly 0M
        d = calendar.advance(ql_calc_date, ql.Period(months, ql.Months))
        T = day_count.yearFraction(ql_calc_date, d)
        zr = ts_handle.zeroRate(d, day_count, compounding, frequency).rate()
        tenors_years.append(T)
        zero_rates.append(zr)
        months += step_months

    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tenors_years, [z * 100 for z in zero_rates])  # percentage, default styling
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Zero rate (%)")
    ax.set_title("Zero-Coupon Yield Curve")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if show:
        plt.show()

    return tenors_years, zero_rates


def create_fixed_rate_bond(
        calculation_date: ql.Date,
        face: float,
        issue_date: ql.Date,
        maturity_date: ql.Date,
        coupon_rate: float,
        coupon_frequency: ql.Period,
        day_count: ql.DayCounter,
        calendar: ql.Calendar = ql.TARGET(),
        business_day_convention: int = ql.Following,  # enums are ints in the Python wrapper
        settlement_days: int = 2
) -> ql.FixedRateBond:
    """Construct and engine-attach."""
    ql.Settings.instance().evaluationDate = calculation_date

    discount_curve = build_discount_curve_from_bonds(calculation_date)

    schedule = ql.Schedule(
        issue_date, maturity_date, coupon_frequency, calendar,
        business_day_convention, business_day_convention,
        ql.DateGeneration.Forward, False
    )

    bond = ql.FixedRateBond(settlement_days, face, schedule, [coupon_rate], day_count)
    bond.setPricingEngine(ql.DiscountingBondEngine(discount_curve))
    return bond


def create_floating_rate_bond_with_curve(
        *,
        calculation_date: ql.Date,
        face: float,
        issue_date: ql.Date,
        maturity_date: ql.Date,
        floating_rate_index: ql.IborIndex,
        spread: float = 0.0,
        coupon_frequency: ql.Period | None = None,
        day_count: ql.DayCounter | None = None,
        calendar: ql.Calendar | None = None,
        business_day_convention: int = ql.Following,
        settlement_days: int = 2,
        curve: ql.YieldTermStructureHandle,
        seed_past_fixings_from_curve: bool = True,
        discount_curve: Optional[ql.YieldTermStructureHandle] = None,
        schedule: Optional[ql.Schedule] = None,
) -> ql.FloatingRateBond:
    """
    Build/prices a floating-rate bond like your swap-with-curve:
      - clone index to 'curve'
      - spot-start safeguard
      - seed past/today fixings from the same curve
      - discount with the same curve
    """

    # --- evaluation settings (match swap) ---
    ql.Settings.instance().evaluationDate = calculation_date
    ql.Settings.instance().includeReferenceDateEvents = False
    ql.Settings.instance().enforceTodaysHistoricFixings = False

    if curve is None:
        raise ValueError("create_floating_rate_bond_with_curve: 'curve' is None")
    # Probe the handle by attempting a discount on calculation_date.
    # If the handle is unlinked/invalid this will raise; we convert it to a clear message.
    try:
        _ = curve.discount(calculation_date)
    except Exception as e:
        raise ValueError(
            "create_floating_rate_bond_with_curve: provided curve handle "
            "is not linked or cannot discount on calculation_date"
        ) from e

    # --- index & calendars ---
    pricing_index = floating_rate_index.clone(curve)  # forecast on the provided curve
    cal = calendar or pricing_index.fixingCalendar()
    freq = coupon_frequency or pricing_index.tenor()
    dc = day_count or pricing_index.dayCounter()

    eff_start = issue_date
    eff_end = maturity_date

    # --------- Schedule ----------
    if schedule is None:
        schedule = ql.Schedule(
            eff_start, eff_end, freq, cal,
            business_day_convention, business_day_convention,
            ql.DateGeneration.Forward, False
        )
    # else:
    #     has_periods = schedule.size() > 1
    #
    #     if not has_periods:
    #         # Redemption-only: use ZeroCouponBond
    #         zcb = ql.ZeroCouponBond(
    #             settlement_days,
    #             calendar,
    #             face,  # redemption notional per 100 original
    #             schedule.date(0) if schedule.size() == 1 else issue_date,  # maturity date
    #             payment_convention,
    #             100.0,  # redemption (% of face)
    #             issue_date
    #         )
    #         zcb.setPricingEngine(ql.DiscountingBondEngine(curve))
    #         return zcb
    #

    # --------- Instrument ----------
    bond = ql.FloatingRateBond(
        settlement_days,
        face,
        schedule,
        pricing_index,
        dc,
        business_day_convention,
        pricing_index.fixingDays(),
        [1.0],  # gearings
        [spread],  # spreads
        [], [],  # caps, floors
        False,  # inArrears
        100.0,  # redemption
        issue_date
    )

    # --------- Seed past/today fixings from this same curve ----------
    if seed_past_fixings_from_curve:
        for cf in bond.cashflows():
            cup = ql.as_floating_rate_coupon(cf)
            if cup is None:
                continue
            fix = cup.fixingDate()
            if fix <= calculation_date:
                try:
                    # If a real fixing exists already, leave it
                    _ = pricing_index.fixing(fix)
                except RuntimeError:
                    rate_found = False
                    rate_to_add = None

                    # Fallback: Loop backwards up to 10 business days to find the last available rate.
                    for days_back in range(1, 11):  # This will loop from 1 to 10
                        lookback_date = calendar.advance(fix, -days_back, ql.Days)
                        try:
                            rate_to_add = pricing_index.fixing(lookback_date)
                            # If we find a rate, store it, mark it as found, and exit the loop.
                            rate_found = True
                            break
                        except RuntimeError:
                            # If no fixing exists for this lookback_date, continue to the next day.
                            continue

                    # After the loop, check if we were successful.
                    if not rate_found:
                        # If we looped 10 times and found nothing, raise a clear error.
                        raise RuntimeError(
                            f"CRITICAL: Could not find a valid fixing for the required date {fix.ISO()} "
                            f"within the 10-business-day lookback period."
                        )

                    # If a rate was found, add it for the *original* required fixing date.
                    pricing_index.addFixing(fix, rate_to_add)

    # --------- Pricing engine ----------
    if discount_curve is not None:
        test_discount_handle = ql.YieldTermStructureHandle(discount_curve)
        test_bond_engine = ql.DiscountingBondEngine(test_discount_handle)
        bond.setPricingEngine(test_bond_engine)

    else:
        bond.setPricingEngine(ql.DiscountingBondEngine(curve))
    return bond


def create_floating_rate_bond(
        calculation_date: ql.Date,
        face: float,
        issue_date: ql.Date,
        maturity_date: ql.Date,
        floating_rate_index: ql.IborIndex,
        spread: float = 0.0,
        coupon_frequency: ql.Period = None,
        day_count: ql.DayCounter = None,
        calendar: ql.Calendar = ql.TARGET(),
        business_day_convention: int = ql.Following,
        settlement_days: int = 2
) -> ql.FloatingRateBond:
    """Construct a floating rate bond and attach pricing engine."""
    ql.Settings.instance().evaluationDate = calculation_date

    # Build discount curve for pricing
    discount_curve = build_discount_curve_from_bonds(calculation_date)

    # Link the floating rate index to the discount curve
    # Create a new index with the same characteristics but linked to our curve
    index_with_curve = floating_rate_index.clone(discount_curve)

    # Use index defaults if not specified
    if coupon_frequency is None:
        coupon_frequency = floating_rate_index.tenor()
    if day_count is None:
        day_count = floating_rate_index.dayCounter()

    schedule = ql.Schedule(
        issue_date, maturity_date, coupon_frequency, calendar,
        business_day_convention, business_day_convention,
        ql.DateGeneration.Forward, False
    )

    # Create floating rate bond with spread
    bond = ql.FloatingRateBond(
        settlement_days, face, schedule,
        index_with_curve, day_count,
        business_day_convention, settlement_days,
        [1.0],  # gearings (multiplier for the index rate)
        [spread],  # spreads
        [],  # caps
        [],  # floors
        False,  # in arrears
        100.0  # redemption
    )

    bond.setPricingEngine(ql.DiscountingBondEngine(discount_curve))
    return bond
