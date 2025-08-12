# src/instruments/fixed_rate_bond.py
import datetime
import QuantLib as ql
from src.instruments.base_instrument import DerivativeInstrument
from src.pricing_models.bond_pricer import create_fixed_rate_bond
from src.utils import to_ql_date

class FixedRateBond(DerivativeInstrument):
    """
    Plain-vanilla fixed-rate bond priced off a discount curve.
    """

    def __init__(
        self,
        face_value: float,
        coupon_rate: float,
        issue_date: datetime.date,
        maturity_date: datetime.date,
        coupon_frequency: ql.Period,
        day_count: ql.DayCounter,
        calendar: ql.Calendar = ql.TARGET(),
        business_day_convention: ql.BusinessDayConvention = ql.Following,
        settlement_days: int = 2,
        valuation_date: datetime.date | None = None
    ):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.coupon_frequency = coupon_frequency
        self.day_count = day_count
        self.calendar = calendar
        self.business_day_convention = business_day_convention
        self.settlement_days = settlement_days
        self.valuation_date = valuation_date or datetime.date.today()

        self._bond = None  # QuantLib bond object

    def _setup_pricer(self):
        if self._bond is None:
            ql_calc_date = to_ql_date(self.valuation_date)
            self._bond = create_fixed_rate_bond(
                calculation_date=ql_calc_date,
                face=self.face_value,
                issue_date=to_ql_date(self.issue_date),
                maturity_date=to_ql_date(self.maturity_date),
                coupon_rate=self.coupon_rate,
                coupon_frequency=self.coupon_frequency,
                day_count=self.day_count,
                calendar=self.calendar,
                business_day_convention=self.business_day_convention,
                settlement_days=self.settlement_days
            )

    def price(self) -> float:
        """
        Returns the NPV (currency units) of the bond.
        """
        print(f"Pricing Fixed Rate Bond as of {self.valuation_date}")
        print(f"Face: {self.face_value:,.2f}, Coupon: {self.coupon_rate:.2%}")
        print(f"Issue: {self.issue_date}, Maturity: {self.maturity_date}")
        print("--------------------------------------------------")

        self._setup_pricer()
        npv = self._bond.NPV()

        print(f"Successfully priced bond. NPV = {npv:,.4f}")
        return npv

    def analytics(self) -> dict:
        """
        Useful bond stats besides NPV.
        """
        self._setup_pricer()
        # Trigger calculation to ensure prices/accrual up-to-date
        _ = self._bond.NPV()
        return {
            "clean_price": self._bond.cleanPrice(),    # price per 100 face
            "dirty_price": self._bond.dirtyPrice(),    # includes accrued
            "accrued_amount": self._bond.accruedAmount()
        }
