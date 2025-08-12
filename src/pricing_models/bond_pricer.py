# src/pricing_models/bond_pricer.py
import QuantLib as ql
from typing import List, Dict, Any
from src.data_interface import APITimeSeries
from src.utils import to_ql_date  # only used by callers; keeping import style consistent with your repo

def build_bond_discount_curve(calculation_date: ql.Date) -> ql.YieldTermStructureHandle:
    """
    Builds a discount curve from mock zero rates returned by the
    'discount_bond_curve' data table.
    """
    print("Building discount curve from 'discount_bond_curve'...")
    data = APITimeSeries.get_historical_data("discount_bond_curve", {"USD_discount_curve": {}})
    points = data["curve_points"]

    calendar = ql.TARGET()
    day_count = ql.Actual365Fixed()

    dates = [calendar.advance(calculation_date, ql.Period(p["tenor"])) for p in points]
    zero_rates = [p["zero_rate"] for p in points]

    zero_curve = ql.ZeroCurve(dates, zero_rates, day_count)
    zero_curve.enableExtrapolation()

    print("Discount curve built successfully.")
    return ql.YieldTermStructureHandle(zero_curve)


def create_fixed_rate_bond(
    calculation_date: ql.Date,
    face: float,
    issue_date: ql.Date,
    maturity_date: ql.Date,
    coupon_rate: float,
    coupon_frequency: ql.Period,
    day_count: ql.DayCounter,
    calendar: ql.Calendar = ql.TARGET(),
    business_day_convention: ql.BusinessDayConvention = ql.Following,
    settlement_days: int = 2
) -> ql.FixedRateBond:
    """
    Constructs a QuantLib FixedRateBond and attaches a DiscountingBondEngine
    using the discount curve built above.
    """
    ql.Settings.instance().evaluationDate = calculation_date

    discount_curve = build_bond_discount_curve(calculation_date)

    schedule = ql.Schedule(
        issue_date,
        maturity_date,
        coupon_frequency,
        calendar,
        business_day_convention,
        business_day_convention,
        ql.DateGeneration.Forward,
        False
    )

    bond = ql.FixedRateBond(
        settlement_days,
        face,
        schedule,
        [coupon_rate],
        day_count
    )

    engine = ql.DiscountingBondEngine(discount_curve)
    bond.setPricingEngine(engine)

    return bond
