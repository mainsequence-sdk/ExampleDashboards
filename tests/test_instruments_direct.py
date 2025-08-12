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
    fixed_rate = 0.762
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

    swap = TIIESwap(
        notional=50_000_000,
        start_date=dt.date(2025, 8, 12),
        maturity_date=dt.date(2028, 8, 12),
        fixed_rate=0.10,  # 10% fixed
        fixed_leg_convention=ql.Unadjusted,
        valuation_date=dt.date(2025, 8, 12),
        float_leg_spread=0.0,
        # float_leg_ibor_index=None  -> will auto-use TIIE(28D) on Valmer curve
    )

    print("NPV:", swap.price())

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

# test_european_option_direct()
# test_fixed_rate_bond_direct()
# test_interest_rate_swap_direct()

# test_plot_zero_curve_from_swaps()
# test_plot_zero_curve_from_bonds()
test_tiie_swap()