import QuantLib as ql
from src.data_interface import APITimeSeries
from src.utils import to_py_date,to_ql_date
import datetime
from typing import List, Dict, Any


def add_historical_fixings(calculation_date: ql.Date, ibor_index: ql.IborIndex):
    """
    Fetches and adds historical fixings to an IborIndex.

    This function now calls the data interface to get a history of fixings
    and loads them into the index.

    Args:
        calculation_date (ql.Date): The date of the valuation.
        ibor_index (ql.IborIndex): The index to add fixings to.
    """
    print("Fetching and adding historical fixings...")

    # Define the historical period for which to fetch fixings.
    # A real implementation might look back further or have more complex logic.
    end_date = to_py_date(calculation_date) - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=365)  # Look back one year

    # Fetch the fixings from the new API method
    historical_fixings = APITimeSeries.get_historical_fixings(
        index_name=ibor_index.name(),
        start_date=start_date,
        end_date=end_date
    )

    if not historical_fixings:
        print("No historical fixings found in the given date range.")
        return

    # Use the more efficient addFixings (plural) method
    fixing_dates = [to_ql_date(dt) for dt in historical_fixings.keys()]
    fixing_rates = list(historical_fixings.values())

    ibor_index.addFixings(fixing_dates, fixing_rates)
    print(f"Successfully added {len(fixing_dates)} fixings for {ibor_index.name()}.")


def build_yield_curve(calculation_date: ql.Date) -> ql.YieldTermStructureHandle:
    """
    Builds a piecewise yield curve by bootstrapping over a set of market rates.
    """
    print("Building bootstrapped yield curve from market nodes...")

    rate_data = APITimeSeries.get_historical_data("interest_rates", {"USD_rates": {}})
    curve_nodes = rate_data['curve_nodes']

    calendar = ql.TARGET()
    day_counter = ql.Actual365Fixed()

    rate_helpers = []

    swap_fixed_leg_frequency = ql.Annual
    swap_fixed_leg_convention = ql.Unadjusted
    swap_fixed_leg_daycounter = ql.Thirty360(ql.Thirty360.USA)
    yield_curve_handle = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, 0.05, day_counter))
    ibor_index = ql.USDLibor(ql.Period('3M'), yield_curve_handle)

    for node in curve_nodes:
        rate = node['rate']
        tenor = ql.Period(node['tenor'])
        quote_handle = ql.QuoteHandle(ql.SimpleQuote(rate))

        if node['type'] == 'deposit':
            helper = ql.DepositRateHelper(quote_handle, tenor, 2, calendar, ql.ModifiedFollowing, False, day_counter)
            rate_helpers.append(helper)
        elif node['type'] == 'swap':
            helper = ql.SwapRateHelper(quote_handle, tenor, calendar,
                                       swap_fixed_leg_frequency,
                                       swap_fixed_leg_convention,
                                       swap_fixed_leg_daycounter,
                                       ibor_index)
            rate_helpers.append(helper)

    yield_curve = ql.PiecewiseLogCubicDiscount(calculation_date, rate_helpers, day_counter)
    yield_curve.enableExtrapolation()

    print("Yield curve built successfully.")
    return ql.YieldTermStructureHandle(yield_curve)


def price_vanilla_swap(calculation_date: ql.Date, notional: float, start_date: ql.Date,
                       maturity_date: ql.Date, fixed_rate: float, fixed_leg_tenor: ql.Period,
                       fixed_leg_convention: object, fixed_leg_daycount: ql.DayCounter,
                       float_leg_tenor: ql.Period, float_leg_spread: float,
                       ibor_index: ql.IborIndex) -> ql.VanillaSwap:
    """
    Creates a vanilla interest rate swap instrument.
    """
    # 1. Add any necessary historical fixings before building the curve
    add_historical_fixings(calculation_date, ibor_index)

    # 2. Build the yield curve for discounting
    discounting_curve = build_yield_curve(calculation_date)

    # 3. Clone the Ibor index and link it to the newly built curve for forecasting.
    pricing_ibor_index = ibor_index.clone(discounting_curve)

    # 4. Define the payment schedules for both legs
    calendar = pricing_ibor_index.fixingCalendar()

    fixed_schedule = ql.Schedule(
        start_date, maturity_date, fixed_leg_tenor, calendar,
        fixed_leg_convention, fixed_leg_convention,
        ql.DateGeneration.Forward, False
    )

    float_schedule = ql.Schedule(
        start_date, maturity_date, float_leg_tenor, calendar,
        pricing_ibor_index.businessDayConvention(), pricing_ibor_index.businessDayConvention(),
        ql.DateGeneration.Forward, False
    )

    # 5. Create the swap instrument in QuantLib
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer,  # We assume we are paying fixed, receiving float
        notional,
        fixed_schedule,
        fixed_rate,
        fixed_leg_daycount,
        float_schedule,
        pricing_ibor_index,
        float_leg_spread,
        pricing_ibor_index.dayCounter()
    )

    # 6. Create the pricing engine and attach it to the swap
    engine = ql.DiscountingSwapEngine(discounting_curve)
    swap.setPricingEngine(engine)

    return swap


def get_swap_cashflows(swap: ql.VanillaSwap) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyzes the cashflows of a swap's fixed and floating legs.
    """
    cashflows = {'fixed': [], 'floating': []}

    for cf in swap.leg(0):
        if not cf.hasOccurred():
            cashflows['fixed'].append({
                'payment_date': to_py_date(cf.date()),
                'amount': cf.amount()
            })

    for cf in swap.leg(1):
        if not cf.hasOccurred():
            coupon = ql.as_floating_rate_coupon(cf)
            cashflows['floating'].append({
                'payment_date': to_py_date(coupon.date()),
                'fixing_date': to_py_date(coupon.fixingDate()),
                'rate': coupon.rate(),
                'spread': coupon.spread(),
                'amount': coupon.amount()
            })

    return cashflows