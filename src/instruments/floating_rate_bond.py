import datetime
from typing import Optional, Dict, Any,List

import QuantLib as ql
from pydantic import BaseModel, Field, PrivateAttr, field_serializer, field_validator

from src.pricing_models.bond_pricer import (
    create_floating_rate_bond,
    create_floating_rate_bond_with_curve,
)
from src.utils import to_ql_date,to_py_date
from .json_codec import (
    JSONMixin,
    period_to_json, period_from_json,
    daycount_to_json, daycount_from_json,
    calendar_to_json, calendar_from_json,
    ibor_to_json, ibor_from_json,
)


class FloatingRateBond(BaseModel, JSONMixin):
    """Floating-rate bond with specified floating rate index."""

    face_value: float = Field(
        ..., description="Face (par) amount of the bond (currency units)."
    )
    floating_rate_index: ql.IborIndex = Field(
        ..., description="Floating rate index (e.g., ql.USDLibor(ql.Period('3M'), curve))."
    )
    spread: float = Field(
        default=0.0, description="Spread (decimal) added to the floating index."
    )
    issue_date: datetime.date = Field(
        ..., description="Bond issue date (schedule start)."
    )
    maturity_date: datetime.date = Field(
        ..., description="Bond redemption date (schedule end)."
    )
    coupon_frequency: ql.Period = Field(
        ..., description="Coupon interval, e.g., ql.Period('3M') for quarterly."
    )
    day_count: ql.DayCounter = Field(
        ..., description="Day-count basis, e.g., ql.Actual360()."
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

    _bond: Optional[ql.FloatingRateBond] = PrivateAttr(default=None)

    # ---------- JSON field serializers ----------
    @field_serializer("coupon_frequency", when_used="json")
    def _ser_coupon_frequency(self, v: ql.Period) -> str:
        return period_to_json(v)

    @field_serializer("day_count", when_used="json")
    def _ser_day_count(self, v: ql.DayCounter) -> str:
        return daycount_to_json(v)

    @field_serializer("calendar", when_used="json")
    def _ser_calendar(self, v: ql.Calendar) -> Dict[str, Any]:
        return calendar_to_json(v)

    @field_serializer("floating_rate_index", when_used="json")
    def _ser_floating_rate_index(self, v: ql.IborIndex) -> Dict[str, Any]:
        return ibor_to_json(v)

    # ---------- JSON field validators (decode) ----------
    @field_validator("coupon_frequency", mode="before")
    @classmethod
    def _val_coupon_frequency(cls, v):
        return period_from_json(v)

    @field_validator("day_count", mode="before")
    @classmethod
    def _val_day_count(cls, v):
        return daycount_from_json(v)

    @field_validator("calendar", mode="before")
    @classmethod
    def _val_calendar(cls, v):
        return calendar_from_json(v)

    @field_validator("floating_rate_index", mode="before")
    @classmethod
    def _val_floating_rate_index(cls, v):
        return ibor_from_json(v)


    @property
    def bond(self) -> ql.FloatingRateBond:
        """
        Provides lazy-loaded, on-demand access to the underlying QuantLib
        FloatingRateBond object.
        """
        if self._bond is None:
            self._bond = self._build_ql_bond()
        return self._bond

    def reset_curve(self, curve: ql.YieldTermStructure) -> None:
        """Re-prices the swap using a new, user-provided curve."""
        # Call the common swap construction logic with the new curve.
        self._build_bond(curve)

    def _build_bond(self,curve):
        ql_calc_date = to_ql_date(self.valuation_date)
        if curve is not None:

            self._bond = create_floating_rate_bond_with_curve(
                calculation_date=ql_calc_date,
                face=self.face_value,
                issue_date=to_ql_date(self.issue_date),
                maturity_date=to_ql_date(self.maturity_date),
                floating_rate_index=self.floating_rate_index,
                spread=self.spread,
                coupon_frequency=self.coupon_frequency,
                day_count=self.day_count,
                calendar=self.calendar,
                business_day_convention=self.business_day_convention,
                settlement_days=self.settlement_days,
                curve=curve,
                seed_past_fixings_from_curve=True,
            )
        else:
            # Legacy path (no forwarding curve on the index) — your existing behavior
            self._bond = create_floating_rate_bond(
                calculation_date=ql_calc_date,
                face=self.face_value,
                issue_date=to_ql_date(self.issue_date),
                maturity_date=to_ql_date(self.maturity_date),
                floating_rate_index=self.floating_rate_index,
                spread=self.spread,
                coupon_frequency=self.coupon_frequency,
                day_count=self.day_count,
                calendar=self.calendar,
                business_day_convention=self.business_day_convention,
                settlement_days=self.settlement_days
            )

    def _setup_pricer(self) -> None:
        if self._bond is None:
            ql_calc_date = to_ql_date(self.valuation_date)
            # Align swap behavior: forecast “today”, no historic fixing required
            ql.Settings.instance().evaluationDate = ql_calc_date
            ql.Settings.instance().includeReferenceDateEvents = False
            ql.Settings.instance().enforceTodaysHistoricFixings = False

            # If the index is already linked to a forwarding curve, price like the swap
            try:
                curve = self.floating_rate_index.forwardingTermStructure()
            except Exception as e:
                raise e
            self._build_bond(curve)



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


    def get_cashflows(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Future cashflows of a floating-rate bond, grouped like swap legs.
        Returns:
          {
            "floating":   [{"payment_date": date, "fixing_date": date, "rate": float, "spread": float, "amount": float}, ...],
            "redemption": [{"payment_date": date, "amount": float}, ...]
          }
        """
        self._setup_pricer()
        # Ensure hasOccurred() uses this model's valuation date
        ql.Settings.instance().evaluationDate = to_ql_date(self.valuation_date)

        out: Dict[str, List[Dict[str, Any]]] = {"floating": [], "redemption": []}

        for cf in self._bond.cashflows():
            if cf.hasOccurred():
                continue

            cpn = ql.as_floating_rate_coupon(cf)
            if cpn is not None:
                out["floating"].append({
                    "payment_date": to_py_date(cpn.date()),
                    "fixing_date": to_py_date(cpn.fixingDate()),
                    "rate": float(cpn.rate()),
                    "spread": float(cpn.spread()),
                    "amount": float(cpn.amount()),
                })
            else:
                # principal repayment(s)
                out["redemption"].append({
                    "payment_date": to_py_date(cf.date()),
                    "amount": float(cf.amount()),
                })

        return out