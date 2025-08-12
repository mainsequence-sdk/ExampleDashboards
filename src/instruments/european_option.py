import datetime
import QuantLib as ql
from src.instruments.base_instrument import DerivativeInstrument
from src.data_interface import APITimeSeries, DateInfo
from src.pricing_models.black_scholes import create_bsm_model
from src.utils import to_ql_date


class EuropeanOption(DerivativeInstrument):
    """
    Represents a European-style option and provides methods to price it and get its Greeks.
    """

    def __init__(self, underlying: str, strike: float, maturity: datetime.date, option_type: str):
        """
        Initializes the European Option.

        Args:
            underlying: The ticker or identifier of the underlying asset.
            strike: The strike price of the option.
            maturity: The expiration date of the option (as a datetime.date object).
            option_type: The type of the option, either 'call' or 'put'.
        """
        self.underlying = underlying
        self.strike = strike
        self.maturity = maturity

        if option_type.lower() not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'.")

        self.option_type = ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put

        # Set the calculation date to today
        self.calculation_date = datetime.date.today()

        # This will be populated after pricing
        self._option = None
        self._engine = None

    def _setup_pricing_components(self):
        """
        Private helper method to set up all QuantLib components needed for pricing.
        This avoids duplicating code between price() and get_greeks().
        """
        # 1. Fetch necessary market data
        asset_range_map = {self.underlying: DateInfo(start_date=self.calculation_date)}
        market_data = APITimeSeries.get_historical_data("equities_daily", asset_range_map)

        spot_price = market_data['spot_price']
        volatility = market_data['volatility']
        risk_free_rate = market_data['risk_free_rate']
        dividend_yield = market_data['dividend_yield']

        # 2. Convert dates
        ql_calculation_date = to_ql_date(self.calculation_date)
        ql_maturity_date = to_ql_date(self.maturity)
        ql.Settings.instance().evaluationDate = ql_calculation_date

        # 3. Set up the BSM process
        bsm_process = create_bsm_model(
            calculation_date=ql_calculation_date,
            spot_price=spot_price,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield
        )

        # 4. Define the instrument and engine
        payoff = ql.PlainVanillaPayoff(self.option_type, self.strike)
        exercise = ql.EuropeanExercise(ql_maturity_date)
        self._option = ql.VanillaOption(payoff, exercise)

        self._engine = ql.AnalyticEuropeanEngine(bsm_process)
        self._option.setPricingEngine(self._engine)

    def price(self) -> float:
        """
        Prices the European option using an Analytic Black-Scholes engine.

        Returns:
            The Net Present Value (NPV) of the option.
        """
        print(f"Pricing European {'Call' if self.option_type == ql.Option.Call else 'Put'} Option on {self.underlying}")
        print(f"Strike: {self.strike}, Maturity: {self.maturity}")
        print("--------------------------------------------------")

        if not self._option:
            self._setup_pricing_components()

        npv = self._option.NPV()
        print(f"Successfully priced option. NPV = {npv:.4f}")
        return npv

    def get_greeks(self) -> dict:
        """
        Calculates the standard Greeks for the European option.

        Returns:
            A dictionary containing the values for delta, gamma, vega, theta, and rho.
        """
        if not self._option:
            # Ensure pricing components are set up if price() hasn't been called
            self._setup_pricing_components()
            # We need to calculate the NPV to populate the greeks
            self._option.NPV()

        greeks = {
            "delta": self._option.delta(),
            "gamma": self._option.gamma(),
            "vega": self._option.vega() / 100,  # Typically vega is for a 1% change in vol
            "theta": self._option.theta() / 365,  # Typically theta is per day
            "rho": self._option.rho() / 100  # Typically rho is for a 1% change in rate
        }

        print("\nCalculating Greeks...")
        return greeks