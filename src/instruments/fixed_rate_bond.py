import datetime
from typing import Optional, Dict, Any,List

import QuantLib as ql
from pydantic import BaseModel, Field, PrivateAttr

from src.pricing_models.bond_pricer import create_fixed_rate_bond
from src.utils import to_ql_date,to_py_date

from pydantic import BaseModel, Field, PrivateAttr, field_serializer, field_validator
from .base_instrument import InstrumentModel


from .ql_fields import (
    QuantLibPeriod as QPeriod,
    QuantLibDayCounter as QDayCounter,
    QuantLibCalendar as QCalendar,
    QuantLibBDC as QBDC,
)


class FixedRateBond(InstrumentModel):
    """Plain-vanilla fixed-rate bond."""

    face_value: float = Field(...)
    coupon_rate: float = Field(...)
    issue_date: datetime.date = Field(...)
    maturity_date: datetime.date = Field(...)
    coupon_frequency: QPeriod = Field(...)
    day_count: QDayCounter = Field(...)
    calendar: QCalendar = Field(default_factory=ql.TARGET)
    business_day_convention: QBDC = Field(default=ql.Following)
    settlement_days: int = Field(default=2)

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

    def get_cashflows(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Future cashflows of a fixed-rate bond, grouped like swap legs.
        Returns:
          {
            "fixed":      [{"payment_date": date, "amount": float}, ...],
            "redemption": [{"payment_date": date, "amount": float}, ...]
          }
        """
        self._setup_pricer()
        # Ensure hasOccurred() uses this model's valuation date
        ql.Settings.instance().evaluationDate = to_ql_date(self.valuation_date)

        out: Dict[str, List[Dict[str, Any]]] = {"fixed": [], "redemption": []}

        for cf in self._bond.cashflows():
            if cf.hasOccurred():
                continue

            cpn = ql.as_fixed_rate_coupon(cf)
            if cpn is not None:
                out["fixed"].append({
                    "payment_date": to_py_date(cpn.date()),
                    "amo  unt": float(cpn.amount()),
                })
            else:
                # principal repayment(s) – e.g., a Redemption/SimpleCashFlow at maturity
                out["redemption"].append({
                    "payment_date": to_py_date(cf.date()),
                    "amount": float(cf.amount()),
                })

        return out

    def get_net_cashflows(self):
        import pandas as pd
        cashflows= self.get_cashflows()

        df_cupons=pd.DataFrame(cashflows["floating"]).set_index("payment_date")
        df_redemption = pd.DataFrame(cashflows["redemption"]).set_index("payment_date")

        joint_index = df_cupons.index.union(df_redemption.index)
        df_cupons = df_cupons.reindex(joint_index).fillna(0.0)
        df_redemption = df_redemption.reindex(joint_index).fillna(0.0)

        net_cashflow = df_cupons["amount"] + df_redemption["amount"]
        net_cashflow.name = "net_cashflow"
        return net_cashflow
