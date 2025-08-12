import datetime
from typing import Optional

import QuantLib as ql
from pydantic import BaseModel, Field, PrivateAttr

from src.pricing_models.bond_pricer import create_fixed_rate_bond
from src.utils import to_ql_date


class FixedRateBond(BaseModel):
    """Plain-vanilla fixed-rate bond."""

    face_value: float = Field(
        ..., description="Face (par) amount of the bond (currency units)."
    )
    coupon_rate: float = Field(
        ..., description="Annual coupon rate as a decimal (e.g., 0.045 for 4.5%)."
    )
    issue_date: datetime.date = Field(
        ..., description="Bond issue date (schedule start)."
    )
    maturity_date: datetime.date = Field(
        ..., description="Bond redemption date (schedule end)."
    )
    coupon_frequency: ql.Period = Field(
        ..., description="Coupon interval, e.g., ql.Period('6M') for semiannual."
    )
    day_count: ql.DayCounter = Field(
        ..., description="Day-count basis, e.g., ql.Thirty360(ql.Thirty360.USA)."
    )

    calendar: ql.Calendar = Field(
        default_factory=ql.TARGET,
        description="Holiday calendar used for the schedule."
    )
    business_day_convention: int = Field(
        default=ql.Following,
        description="Date roll convention (QuantLib enum int)."
    )
    settlement_days: int = Field(
        default=2,
        description="Settlement lag in business days."
    )
    valuation_date: datetime.date = Field(
        default_factory=datetime.date.today,
        description="Valuation date (QuantLib evaluation date)."
    )

    model_config = {"arbitrary_types_allowed": True}

    _bond: Optional[ql.FixedRateBond] = PrivateAttr(default=None)

    def _setup_pricer(self) -> None:
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
        self._setup_pricer()
        return float(self._bond.NPV())

    def analytics(self) -> dict:
        self._setup_pricer()
        _ = self._bond.NPV()
        return {
            "clean_price": self._bond.cleanPrice(),
            "dirty_price": self._bond.dirtyPrice(),
            "accrued_amount": self._bond.accruedAmount(),
        }
