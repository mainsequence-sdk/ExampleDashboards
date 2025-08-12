import datetime
import QuantLib as ql
from src.instruments.base_instrument import DerivativeInstrument
from src.pricing_models.swap_pricer import price_vanilla_swap, get_swap_cashflows
from src.utils import to_ql_date
from typing import List, Dict, Any


class InterestRateSwap(DerivativeInstrument):
    """
    Represents a standard vanilla Interest Rate Swap (IRS).
    """

    def __init__(self, notional: float, start_date: datetime.date, maturity_date: datetime.date,
                 fixed_rate: float, fixed_leg_tenor: ql.Period, fixed_leg_convention: object,
                 fixed_leg_daycount: ql.DayCounter, float_leg_tenor: ql.Period,
                 float_leg_spread: float, float_leg_ibor_index: ql.IborIndex,
                 valuation_date: datetime.date):
        """
        Initializes the Interest Rate Swap.

        Args:
            notional (float): The notional amount of the swap.
            start_date (datetime.date): The effective date of the swap.
            maturity_date (datetime.date): The termination date of the swap.
            fixed_rate (float): The fixed rate paid by the fixed leg.
            fixed_leg_tenor (ql.Period): The frequency of payments for the fixed leg.
            fixed_leg_convention (object): Business day convention for the fixed leg.
            fixed_leg_daycount (ql.DayCounter): Day count convention for the fixed leg.
            float_leg_tenor (ql.Period): The frequency of payments for the floating leg.
            float_leg_spread (float): The spread over the floating rate index.
            float_leg_ibor_index (ql.IborIndex): The floating rate index (e.g., SOFR, Euribor).
            valuation_date (datetime.date): The date for which the instrument is being valued.
        """
        self.notional = notional
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.fixed_rate = fixed_rate
        self.fixed_leg_tenor = fixed_leg_tenor
        self.fixed_leg_convention = fixed_leg_convention
        self.fixed_leg_daycount = fixed_leg_daycount
        self.float_leg_tenor = float_leg_tenor
        self.float_leg_spread = float_leg_spread
        self.float_leg_ibor_index = float_leg_ibor_index
        self.valuation_date = valuation_date
        self._swap = None  # To hold the QuantLib instrument

    def _setup_pricer(self):
        """
        Private helper to create the QuantLib swap object if it doesn't exist.
        """
        if self._swap is None:
            ql_valuation_date = to_ql_date(self.valuation_date)
            ql.Settings.instance().evaluationDate = ql_valuation_date

            self._swap = price_vanilla_swap(
                calculation_date=ql_valuation_date,
                notional=self.notional,
                start_date=to_ql_date(self.start_date),
                maturity_date=to_ql_date(self.maturity_date),
                fixed_rate=self.fixed_rate,
                fixed_leg_tenor=self.fixed_leg_tenor,
                fixed_leg_convention=self.fixed_leg_convention,
                fixed_leg_daycount=self.fixed_leg_daycount,
                float_leg_tenor=self.float_leg_tenor,
                float_leg_spread=self.float_leg_spread,
                ibor_index=self.float_leg_ibor_index
            )

    def price(self) -> float:
        """
        Prices the Interest Rate Swap.
        """
        print(f"Pricing Interest Rate Swap as of {self.valuation_date}")
        print(f"Notional: {self.notional:,.2f}, Fixed Rate: {self.fixed_rate:.2%}")
        print(f"Start: {self.start_date}, Maturity: {self.maturity_date}")
        print("--------------------------------------------------")

        self._setup_pricer()
        npv = self._swap.NPV()

        print(f"Successfully priced swap. NPV = {npv:,.2f}")
        return npv

    def get_cashflows(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieves the projected cashflows for both legs of the swap.
        """
        print("\nAnalyzing Swap Cashflows...")
        self._setup_pricer()
        return get_swap_cashflows(self._swap)