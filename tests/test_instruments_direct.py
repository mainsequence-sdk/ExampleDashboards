import math
import datetime as dt
from pathlib import Path
import sys
import pytest

# Make sure project root is importable so `src/...` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Skip if QuantLib isn't available in your env
try:
    import QuantLib as ql  # noqa: F401
except Exception:
    pytest.skip("QuantLib not available; skipping instrument tests.", allow_module_level=True)

from src.instruments.european_option import EuropeanOption
from src.instruments.interest_rate_swap import InterestRateSwap
from src.instruments.fixed_rate_bond import FixedRateBond
from src.instruments.vanilla_fx_option import VanillaFXOption
from src.instruments.knockout_fx_option import KnockOutFXOption


from src.pricing_models.bond_pricer import plot_zero_coupon_curve
from src.pricing_models.swap_pricer import plot_swap_zero_curve

def _is_finite_number(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def test_interest_rate_swap_direct():
    import QuantLib as ql

    notional = 10_000_000
    start_date = dt.date(2025, 7, 23)
    maturity_date = dt.date(2030, 7, 23)
    fixed_rate = .0780098200
    valuation_date = dt.date(2026, 1, 15)

    # Simple flat curve + USDLibor(3M) for float leg
    ql.Settings.instance().evaluationDate = ql.Date(15, 1, 2026)
    flat_yts = ql.YieldTermStructureHandle(
        ql.FlatForward(ql.Date(21, 7, 2025), 0.05, ql.Actual365Fixed())
    )
    ibor_index = ql.USDLibor(ql.Period("3M"), flat_yts)

    swap = InterestRateSwap(
        notional=notional,
        start_date=start_date,
        maturity_date=maturity_date,
        fixed_rate=fixed_rate,
        fixed_leg_tenor=ql.Period("6M"),
        fixed_leg_convention=ql.Unadjusted,
        fixed_leg_daycount=ql.Thirty360(ql.Thirty360.USA),
        float_leg_tenor=ql.Period("3M"),
        float_leg_spread=0.0,
        float_leg_ibor_index=ibor_index,
        valuation_date=valuation_date,
    )

    npv = swap.price()
    assert _is_finite_number(npv)


def test_tiie_swap():

    from src.instruments.interest_rate_swap import TIIESwap
    from src.pricing_models.swap_pricer import debug_tiie_trade,build_tiie_zero_curve_from_valmer,make_tiie_28d_index,price_vanilla_swap_with_curve
    from src.utils import to_ql_date
    trade_date = dt.date(2025, 9, 6)  # valuation date (T)
    swap = TIIESwap(
        notional=100_000_000,
        start_date=trade_date,  # we’ll move to spot internally
        maturity_date=trade_date,  # ignored when tenor is provided
        tenor=ql.Period("364W"),  #
        fixed_rate=.0780098200,  # 7.624865% in DECIMAL
        valuation_date=trade_date ,
        float_leg_spread=0.0,
        # legs default to 28D / ACT/360 / ModFollowing / Mexico in TIIESwap
    )

    ql.Settings.instance().evaluationDate = to_ql_date(swap.valuation_date)

    npv = swap.price()
    fair = swap._swap.fairRate()
    print("NPV:", npv, "Fair:", fair)

    npv = swap.price()
    fixed_pv = swap._swap.fixedLegNPV()
    float_pv = swap._swap.floatingLegNPV()

    # build the exact curve and index used by TIIESwap
    ql_val = to_ql_date(swap.valuation_date)
    curve = build_tiie_zero_curve_from_valmer(ql_val)
    tiie28 = make_tiie_28d_index(curve)
    debug_tiie_trade(ql_val, swap._swap, curve, tiie28)

    def prove_curve_anchor(curve: ql.YieldTermStructureHandle, ibor_index: ql.IborIndex | None = None):
        link = curve.currentLink()
        ref = link.referenceDate()
        evald = ql.Settings.instance().evaluationDate
        print("\n[CURVE ANCHOR PROOF]")
        print(
            f"referenceDate: {ref.year():04d}-{ref.month():02d}-{ref.dayOfMonth():02d}  DF(ref)={curve.discount(ref):.8f}")
        print(
            f"evaluationDate: {evald.year():04d}-{evald.month():02d}-{evald.dayOfMonth():02d}  DF(eval)={curve.discount(evald):.8f}")
        if ibor_index is not None:
            cal = ibor_index.fixingCalendar()
            fixing = cal.adjust(evald, ql.Following)
            while not ibor_index.isValidFixingDate(fixing):
                fixing = cal.advance(fixing, 1, ql.Days)
            spot = ibor_index.valueDate(fixing)
            print(
                f"spot(T+1): {spot.year():04d}-{spot.month():02d}-{spot.dayOfMonth():02d}  DF(spot)={curve.discount(spot):.8f}")

    prove_curve_anchor(curve, tiie28)
    print("NPV:", npv)
    print("Fixed PV:", fixed_pv)
    print("Float PV:", float_pv)

    # Par checks: fixed leg PV ~ notional; total NPV ~ 0
    assert abs(fixed_pv - swap.notional) / swap.notional < 5e-3
    assert abs(npv) < 1e-2 * swap.notional

def test_fixed_rate_bond_direct():
    import QuantLib as ql

    bond = FixedRateBond(
        face_value=1_000_000,
        coupon_rate=0.0425,
        issue_date=dt.date(2024, 8, 12),
        maturity_date=dt.date(2026, 8, 12),
        coupon_frequency=ql.Period("6M"),
        day_count=ql.Thirty360(ql.Thirty360.USA),
        valuation_date=dt.date(2025, 8, 12),
    )

    npv = bond.price()
    analytics = bond.analytics()

    assert _is_finite_number(npv)
    assert {"clean_price", "dirty_price", "accrued_amount"} <= analytics.keys()
    assert _is_finite_number(analytics["clean_price"])
    assert _is_finite_number(analytics["dirty_price"])
    assert _is_finite_number(analytics["accrued_amount"])


def test_european_option_direct():
    option = EuropeanOption(
        underlying="SPY",
        strike=180.0,
        maturity=dt.date(2025, 12, 31),
        option_type="call",
    )

    price = option.price()
    greeks = option.get_greeks()

    assert _is_finite_number(price)
    for k in ("delta", "gamma", "vega", "theta", "rho"):
        assert k in greeks and _is_finite_number(greeks[k])


def test_plot_zero_curve_from_bonds():
    import matplotlib.pyplot as plt
    calc_date = dt.date(2025, 8, 12)
    fig, ax = plt.subplots()
    years, zeros = plot_zero_coupon_curve(calc_date, max_years=12, step_months=3, show=False, ax=ax)
    assert len(years) > 4 and len(zeros) == len(years)

    plt.show()

def test_plot_zero_curve_from_swaps():
    import matplotlib.pyplot as plt
    calc_date = dt.date(2025, 8, 12)
    fig, ax = plt.subplots()
    years, zeros = plot_swap_zero_curve(calc_date, max_years=12, step_months=3, show=False, ax=ax)
    assert len(years) > 4 and len(zeros) == len(years)
    plt.show()


def test_fx_option():
    """Test the Vanilla FX Option implementation."""
    print("Testing Vanilla FX Option implementation...")

    try:
        # Create a EUR/USD call option
        fx_option = VanillaFXOption(
            currency_pair="EURUSD",
            strike=1.10,
            maturity=datetime.date(2026, 6, 15),
            option_type="call",
            notional=1000000,
            calculation_date=datetime.date.today()
        )

        print(f"Created FX Option: {fx_option.currency_pair} {fx_option.option_type}")
        print(f"Strike: {fx_option.strike}")
        print(f"Notional: {fx_option.notional:,.0f}")
        print(f"Maturity: {fx_option.maturity}")

        # Get market info
        market_info = fx_option.get_market_info()
        print(f"\nMarket Data:")
        print(f"Spot FX Rate: {market_info['spot_fx_rate']:.4f}")
        print(f"Volatility: {market_info['volatility']:.2%}")
        print(f"Domestic Rate ({market_info['domestic_currency']}): {market_info['domestic_rate']:.2%}")
        print(f"Foreign Rate ({market_info['foreign_currency']}): {market_info['foreign_rate']:.2%}")

        # Price the option
        price = fx_option.price()
        print(f"\nOption Price: {price:,.2f} USD")

        # Get Greeks
        greeks = fx_option.get_greeks()
        print(f"\nGreeks:")
        print(f"Delta: {greeks['delta']:,.4f}")
        print(f"Gamma: {greeks['gamma']:,.6f}")
        print(f"Vega: {greeks['vega']:,.4f}")
        print(f"Theta: {greeks['theta']:,.4f}")
        print(f"Rho (Domestic): {greeks['rho_domestic']:,.4f}")

        print("\n✓ FX Option test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ FX Option test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knockout_fx_option():
    """Test the Knock-out FX Option implementation."""
    print("Testing Knock-out FX Option implementation...")

    try:
        # Test 1: Up-and-out EUR/USD call option
        print("\n--- Test 1: Up-and-out EUR/USD Call ---")
        knockout_option_up = KnockOutFXOption(
            currency_pair="EURUSD",
            strike=1.10,
            barrier=1.15,  # Barrier above current spot
            maturity=dt.date(2026, 6, 15),
            option_type="call",
            barrier_type="up_and_out",
            notional=1000000,
            rebate=1000.0,
            calculation_date=dt.date.today()
        )

        print(f"Created Up-and-Out FX Option: {knockout_option_up.currency_pair} {knockout_option_up.option_type}")
        print(f"Strike: {knockout_option_up.strike}")
        print(f"Barrier: {knockout_option_up.barrier} ({knockout_option_up.barrier_type})")
        print(f"Notional: {knockout_option_up.notional:,.0f}")
        print(f"Rebate: {knockout_option_up.rebate:,.0f}")

        # Get market info
        market_info = knockout_option_up.get_market_info()
        print(f"\nMarket Data:")
        print(f"Spot FX Rate: {market_info['spot_fx_rate']:.4f}")
        print(f"Volatility: {market_info['volatility']:.2%}")
        print(f"Domestic Rate: {market_info['domestic_rate']:.2%}")
        print(f"Foreign Rate: {market_info['foreign_rate']:.2%}")

        # Get barrier info
        barrier_info = knockout_option_up.get_barrier_info()
        print(f"\nBarrier Information:")
        print(f"Barrier Status: {barrier_info['barrier_status']}")
        print(f"Distance to Barrier: {barrier_info['distance_to_barrier_pct']:.2f}%")

        # Price the option
        price = knockout_option_up.price()
        print(f"\nOption Price: {price:,.2f} USD")
        assert _is_finite_number(price), "Price should be a finite number"
        assert price >= 0, "Knock-out option price should be non-negative"

        # Get Greeks
        greeks = knockout_option_up.get_greeks()
        print(f"\nGreeks:")
        print(f"Delta: {greeks['delta']:,.4f}")
        print(f"Gamma: {greeks['gamma']:,.6f}")
        print(f"Vega: {greeks['vega']:,.4f}")
        print(f"Theta: {greeks['theta']:,.4f}")
        print(f"Rho (Domestic): {greeks['rho_domestic']:,.4f}")

        # Validate Greeks are finite
        for greek_name, greek_value in greeks.items():
            assert _is_finite_number(greek_value), f"{greek_name} should be a finite number"

        # Test 2: Down-and-out EUR/USD put option
        print("\n--- Test 2: Down-and-out EUR/USD Put ---")
        knockout_option_down = KnockOutFXOption(
            currency_pair="EURUSD",
            strike=1.08,
            barrier=1.05,  # Barrier below current spot
            maturity=dt.date(2026, 6, 15),
            option_type="put",
            barrier_type="down_and_out",
            notional=500000,
            rebate=500.0,
            calculation_date=dt.date.today()
        )

        print(f"Created Down-and-Out FX Option: {knockout_option_down.currency_pair} {knockout_option_down.option_type}")
        print(f"Strike: {knockout_option_down.strike}")
        print(f"Barrier: {knockout_option_down.barrier} ({knockout_option_down.barrier_type})")

        # Price the option
        price_down = knockout_option_down.price()
        print(f"Option Price: {price_down:,.2f} USD")
        assert _is_finite_number(price_down), "Price should be a finite number"
        assert price_down >= 0, "Knock-out option price should be non-negative"

        # Get barrier info
        barrier_info_down = knockout_option_down.get_barrier_info()
        print(f"Barrier Status: {barrier_info_down['barrier_status']}")
        print(f"Distance to Barrier: {barrier_info_down['distance_to_barrier_pct']:.2f}%")

        # Test 3: Compare with vanilla option (knock-out should be cheaper)
        print("\n--- Test 3: Comparison with Vanilla Option ---")
        vanilla_option = VanillaFXOption(
            currency_pair="EURUSD",
            strike=1.10,
            maturity=dt.date(2026, 6, 15),
            option_type="call",
            notional=1000000,
            calculation_date=dt.date.today()
        )

        vanilla_price = vanilla_option.price()
        knockout_price = knockout_option_up.price()

        print(f"Vanilla Option Price: {vanilla_price:,.2f} USD")
        print(f"Knock-out Option Price: {knockout_price:,.2f} USD")
        print(f"Price Difference: {vanilla_price - knockout_price:,.2f} USD")

        # Knock-out option should be cheaper than vanilla (barrier reduces value)
        assert knockout_price <= vanilla_price, "Knock-out option should be cheaper than vanilla option"

        print("\n✓ Knock-out FX Option tests completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Knock-out FX Option test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


# test_european_option_direct()
# test_fixed_rate_bond_direct()
# test_interest_rate_swap_direct()

# test_plot_zero_curve_from_swaps()
# test_plot_zero_curve_from_bonds()
# test_knockout_fx_option()
test_tiie_swap()
