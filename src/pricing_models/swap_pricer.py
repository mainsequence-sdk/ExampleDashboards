import QuantLib as ql
from src.data_interface import APIDataNode
from src.utils import to_py_date,to_ql_date
import datetime
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt

def add_historical_fixings(calculation_date: ql.Date, ibor_index: ql.IborIndex):
    print("Fetching and adding historical fixings...")

    end_date = to_py_date(calculation_date) - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=365)

    historical_fixings = APIDataNode.get_historical_fixings(
        index_name=ibor_index.name(),
        start_date=start_date,
        end_date=end_date
    )

    if not historical_fixings:
        print("No historical fixings found in the given date range.")
        return

    # --- NEW: keep only valid fixing dates for THIS index (Mexico calendar) and strictly in the past
    valid_qld: list[ql.Date] = []
    valid_rates: list[float] = []

    for dt_py, rate in sorted(historical_fixings.items()):
        qld = to_ql_date(dt_py)
        if qld < calculation_date and ibor_index.isValidFixingDate(qld):
            valid_qld.append(qld)
            valid_rates.append(rate)

    if not valid_qld:
        print("No valid fixing dates for the index calendar; skipping addFixings.")
        return

    # --- NEW: allow overwriting if some dates already have a fixing
    ibor_index.addFixings(valid_qld, valid_rates, True)
    print(f"Successfully added {len(valid_qld)} fixings for {ibor_index.name()}.")

def build_tiie_zero_curve_from_valmer(calculation_date: ql.Date) -> ql.YieldTermStructureHandle:
    market = APIDataNode.get_historical_data("tiie_zero_valmer", {"MXN": {}})
    nodes = market["curve_nodes"]

    cal = ql.Mexico() if hasattr(ql, "Mexico") else ql.TARGET()
    dc  = ql.Actual360()  # TIIE convention
    dates, zeros = [], []
    for n in nodes:
        d = cal.advance(calculation_date, int(n["days_to_maturity"]), ql.Days)
        dates.append(d)
        zeros.append(float(n["zero"]))

    zc = ql.ZeroCurve(dates, zeros, dc, cal)
    zc.enableExtrapolation()
    return ql.YieldTermStructureHandle(zc)


def make_ftiie_index(curve: ql.YieldTermStructureHandle,
                     settlement_days: int = 1) -> ql.OvernightIndex:
    cal = ql.Mexico() if hasattr(ql, "Mexico") else ql.TARGET()
    try:
        ccy = ql.MXNCurrency()
    except Exception:
        ccy = ql.USDCurrency()  # label only
    return ql.OvernightIndex("F-TIIE", settlement_days, ccy, cal, ql.Actual360(), curve)

def make_tiie_28d_index(
    curve: ql.YieldTermStructureHandle,
    settlement_days: int = 1,
) -> ql.IborIndex:
    """
    Construct a minimal TIIE-28D IborIndex linked to `curve`.

    Conventions used (standard for MXN TIIE 28D):
      - Tenor: 28 days
      - Settlement: T+1
      - Calendar: Mexico (fallback TARGET if not available in your wheel)
      - Business day convention: ModifiedFollowing
      - End-of-month: False
      - Day count: Actual/360
      - Currency: MXN (fallback USD if MXN class not available; does not affect math)

    Parameters
    ----------
    curve : ql.YieldTermStructureHandle
        Forwarding (and, in your setup, discounting) curve to link to the index.
    settlement_days : int
        Fixing settlement lag in business days (default 1).

    Returns
    -------
    ql.IborIndex
        An index named "TIIE-28D" ready to use in swap construction.
    """
    # Calendar & currency (graceful fallbacks if your wheel lacks Mexico/MXN)
    cal = ql.Mexico() if hasattr(ql, "Mexico") else ql.TARGET()
    try:
        ccy = ql.MXNCurrency()
    except Exception:
        ccy = ql.USDCurrency()  # label only; does not affect rates/discounting

    return ql.IborIndex(
        "TIIE-28D",
        ql.Period("28D"),
        settlement_days,  # T+1
        ccy,
        cal,
        ql.ModifiedFollowing,  # BDC
        False,  # EOM
        ql.Actual360(),  # ACT/360
        curve
    )

def build_yield_curve(calculation_date: ql.Date) -> ql.YieldTermStructureHandle:
    """
    Builds a piecewise yield curve by bootstrapping over a set of market rates.
    """
    print("Building bootstrapped yield curve from market nodes...")

    rate_data = APIDataNode.get_historical_data("interest_rate_swaps", {"USD_rates": {}})
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

def price_vanilla_swap_with_curve(
    calculation_date: ql.Date,
    notional: float,
    start_date: ql.Date,
    maturity_date: ql.Date,
    fixed_rate: float,
    fixed_leg_tenor: ql.Period,
    fixed_leg_convention: int,
    fixed_leg_daycount: ql.DayCounter,
    float_leg_tenor: ql.Period,
    float_leg_spread: float,
    ibor_index: ql.IborIndex,
    curve: ql.YieldTermStructureHandle,
    past_fixing_rate: float | None = None,
) -> ql.VanillaSwap:
    ql.Settings.instance().evaluationDate = calculation_date

    # Link the pricing index to the provided curve
    pricing_ibor_index = ibor_index.clone(curve)
    calendar = pricing_ibor_index.fixingCalendar()

    # --------- EFFECTIVE DATES (spot start safeguard) ----------
    # If user passed trade date (T) or anything <= eval, move to spot (T+settlement)
    spot_start = pricing_ibor_index.valueDate(calculation_date)
    eff_start = start_date if start_date > calculation_date else spot_start

    # Ensure termination is strictly after effective start
    eff_end = maturity_date
    if eff_end <= eff_start:
        # Minimum length: one floating period (e.g., 28D for TIIE)
        eff_end = calendar.advance(eff_start, float_leg_tenor)

    # --------- Schedules ----------
    fixed_schedule = ql.Schedule(
        eff_start, eff_end, fixed_leg_tenor, calendar,
        fixed_leg_convention, fixed_leg_convention,
        ql.DateGeneration.Forward, False
    )
    float_schedule = ql.Schedule(
        eff_start, eff_end, float_leg_tenor, calendar,
        pricing_ibor_index.businessDayConvention(), pricing_ibor_index.businessDayConvention(),
        ql.DateGeneration.Forward, False
    )

    # --------- Instrument ----------
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer,
        notional,
        fixed_schedule,
        fixed_rate,
        fixed_leg_daycount,
        float_schedule,
        pricing_ibor_index,
        float_leg_spread,
        pricing_ibor_index.dayCounter()
    )

    # --------- Past fixings fill (earliest node) ----------
    if past_fixing_rate is None:
        # derive a simple forward from the curve at +tenor (no t<0 queries)
        dc = pricing_ibor_index.dayCounter()
        probe_date = calendar.advance(calculation_date, pricing_ibor_index.tenor())
        past_fixing_rate = curve.zeroRate(probe_date, dc, ql.Continuous, ql.Annual).rate()

    for cf in swap.leg(1):
        cup = ql.as_floating_rate_coupon(cf)
        fix = cup.fixingDate()
        if fix <= calculation_date:
            try:
                _ = pricing_ibor_index.fixing(fix)
            except RuntimeError:
                pricing_ibor_index.addFixing(fix, past_fixing_rate)

    swap.setPricingEngine(ql.DiscountingSwapEngine(curve))
    return swap


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

def plot_swap_zero_curve(
    calculation_date: ql.Date | datetime.date,
    max_years: int = 30,
    step_months: int = 3,
    compounding = ql.Continuous,   # QuantLib enums are ints; don't type-hint them
    frequency  = ql.Annual,
    show: bool = False,
    ax: Optional[plt.Axes] = None,
) -> tuple[list[float], list[float]]:
    """
    Plot the zero-coupon (spot) curve implied by the swap-bootstrapped curve.

    Returns:
        (tenors_in_years, zero_rates) with zero_rates in decimals (e.g., 0.045).
    """
    # normalize date
    ql_calc = to_ql_date(calculation_date) if isinstance(calculation_date, datetime.date) else calculation_date
    ql.Settings.instance().evaluationDate = ql_calc

    # build curve from the mocked swap/deposit nodes
    ts_handle = build_yield_curve(ql_calc)

    calendar = ql.TARGET()
    day_count = ql.Actual365Fixed()

    years: list[float] = []
    zeros: list[float] = []

    months = 1
    while months <= max_years * 12:
        d = calendar.advance(ql_calc, ql.Period(months, ql.Months))
        T = day_count.yearFraction(ql_calc, d)
        z = ts_handle.zeroRate(d, day_count, compounding, frequency).rate()
        years.append(T)
        zeros.append(z)
        months += step_months

    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years, [z * 100 for z in zeros])
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Zero rate (%)")
    ax.set_title("Zero-Coupon Yield Curve (Swaps)")
    ax.grid(True, linestyle="--", alpha=0.4)

    if show:
        plt.show()

    return years, zeros

# src/pricing_models/swap_pricer.py
import QuantLib as ql

import QuantLib as ql

def price_ftiie_ois_with_curve(
    calculation_date: ql.Date,
    notional: float,
    start_date: ql.Date,
    maturity_date: ql.Date,
    fixed_rate: float,
    fixed_leg_tenor: ql.Period,          # e.g., 28D
    fixed_leg_convention: int,           # ql.ModifiedFollowing
    fixed_leg_daycount: ql.DayCounter,   # ql.Actual360()
    on_index: ql.OvernightIndex,         # FTIIE overnight index
    curve: ql.YieldTermStructureHandle,
) -> ql.OvernightIndexedSwap:
    # Consistent evaluation settings (no ‘today’ leakage)
    ql.Settings.instance().evaluationDate = calculation_date
    ql.Settings.instance().includeReferenceDateEvents = False
    ql.Settings.instance().enforceTodaysHistoricFixings = False

    cal = on_index.fixingCalendar()

    # -------- Spot start (T+1 Mexico) with tenor preservation --------
    spot_start = on_index.valueDate(calculation_date)
    start_shifted = (start_date <= calculation_date)
    eff_start = start_date if not start_shifted else spot_start

    if start_shifted:
        # shift end by the same calendar-day offset
        try:
            day_offset = int(eff_start - start_date)
        except Exception:
            day_offset = eff_start.serialNumber() - start_date.serialNumber()
        eff_end = cal.advance(maturity_date, int(day_offset), ql.Days)
    else:
        eff_end = maturity_date

    if eff_end <= eff_start:
        eff_end = cal.advance(eff_start, fixed_leg_tenor)

    fixed_sched = ql.Schedule(
        eff_start, eff_end, fixed_leg_tenor, cal,
        fixed_leg_convention, fixed_leg_convention,
        ql.DateGeneration.Forward, False
    )

    ois = ql.OvernightIndexedSwap(
        ql.OvernightIndexedSwap.Payer,
        notional,
        fixed_sched,
        fixed_rate,
        fixed_leg_daycount,
        on_index
    )
    ois.setPricingEngine(ql.DiscountingSwapEngine(curve))
    return ois

