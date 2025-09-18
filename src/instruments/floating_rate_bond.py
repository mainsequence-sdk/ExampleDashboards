import datetime
from typing import Optional, Dict, Any,List

import QuantLib as ql
from pydantic import BaseModel, Field, PrivateAttr, field_serializer, field_validator

from src.pricing_models.bond_pricer import (
    create_floating_rate_bond,
    create_floating_rate_bond_with_curve,
)
from src.pricing_models.indices import get_index

from src.utils import to_ql_date,to_py_date
from .json_codec import (
    JSONMixin,
    period_to_json, period_from_json,
    daycount_to_json, daycount_from_json,
    calendar_to_json, calendar_from_json,
    ibor_to_json, ibor_from_json,
schedule_to_json, schedule_from_json
)

from .ql_fields import (
    QuantLibPeriod as QPeriod,
    QuantLibDayCounter as QDayCounter,
    QuantLibCalendar as QCalendar,
    QuantLibBDC as QBDC,
    QuantLibSchedule as QSchedule,
)

from .base_instrument import InstrumentModel

class FloatingRateBond(InstrumentModel):
    """Floating-rate bond with specified floating rate index."""

    face_value: float = Field(...)
    floating_rate_index_name: str = Field(
        ...,

    )
    spread: float = Field(default=0.0)
    issue_date: datetime.date = Field(...)
    maturity_date: datetime.date = Field(...)
    coupon_frequency: QPeriod = Field(...)
    day_count: QDayCounter = Field(...)
    calendar: QCalendar = Field(default_factory=ql.TARGET)
    business_day_convention: QBDC = Field(default=ql.Following)
    settlement_days: int = Field(default=2)
    schedule: Optional[QSchedule] = Field(None)

    model_config = {"arbitrary_types_allowed": True}
    _bond: Optional[ql.FloatingRateBond] = PrivateAttr(default=None)
    _index: Optional[ql.IborIndex] = PrivateAttr(default=None)
    _with_yield: Optional[float] = PrivateAttr(default=None)



    # ---------- lifecycle ----------

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        if self.valuation_date is None:
            raise ValueError("Set valuation_date before pricing: set_valuation_date(dt).")
        self._index = get_index(
            self.floating_rate_index_name,
            target_date=self.valuation_date,
            hydrate_fixings=True,
        )

    def _on_valuation_date_set(self) -> None:
        self._index = None
        self._bond = None
        self._with_yield = None

    def reset_curve(self, curve: ql.YieldTermStructureHandle) -> None:
        """Optional: re-link a custom curve to this index and rebuild."""
        if self.valuation_date is None:
            raise ValueError("Set valuation_date before reset_curve().")
        self._index = get_index(
            self.floating_rate_index_name,
            target_date=self.valuation_date,
            forwarding_curve=curve,
            hydrate_fixings=True,
        )
        self._bond = None

        # ---------- build / price ----------

    def _build_bond(self, curve: ql.YieldTermStructure, with_yield: Optional[float] = None) -> None:
        ql_calc_date = to_ql_date(self.valuation_date)

        discount_curve = None
        if with_yield is not None:
            discount_curve = ql.FlatForward(
                ql_calc_date, with_yield, self.day_count, ql.Compounded, ql.Annual
            )

        self._bond = create_floating_rate_bond_with_curve(
            calculation_date=ql_calc_date,
            face=self.face_value,
            issue_date=to_ql_date(self.issue_date),
            maturity_date=to_ql_date(self.maturity_date),
            floating_rate_index=self._index,  # runtime object
            spread=self.spread,
            coupon_frequency=self.coupon_frequency,
            day_count=self.day_count,
            calendar=self.calendar,
            business_day_convention=self.business_day_convention,
            settlement_days=self.settlement_days,
            curve=curve,
            discount_curve=discount_curve,
            seed_past_fixings_from_curve=True,
            schedule=self.schedule
        )
        self._with_yield = with_yield

    def _setup_pricer(self, with_yield: Optional[float] = None) -> None:
        self._ensure_index()

        ql_calc_date = to_ql_date(self.valuation_date)
        ql.Settings.instance().evaluationDate = ql_calc_date
        ql.Settings.instance().includeReferenceDateEvents = False
        ql.Settings.instance().enforceTodaysHistoricFixings = False

        curve = self._index.forwardingTermStructure()
        if self._bond is None:
            self._build_bond(curve, with_yield=with_yield)
        else:
            if self._with_yield != with_yield:
                self._bond = None
                self._build_bond(curve, with_yield=with_yield)

    def price(self, with_yield: Optional[float] = None) -> float:
        self._setup_pricer(with_yield=with_yield)
        return float(self._bond.NPV())




    def price(self,with_yield:Optional[float]=None) -> float:
        self._setup_pricer(with_yield=with_yield)
        return float(self._bond.NPV())

    def analytics(self,with_yield:Optional[float]=None) -> dict:
        self._setup_pricer(with_yield=with_yield)
        _ = self._bond.NPV()
        return {
            "clean_price": self._bond.cleanPrice(),
            "dirty_price": self._bond.dirtyPrice(),
            "accrued_amount": self._bond.accruedAmount(),
        }
    def get_net_cashflows(self):
        import pandas as pd
        cashflows= self.get_cashflows()

        df_cupons,df_redemption=None,None
        try:
            if len(cashflows["floating"]) != 0:
                df_cupons=pd.DataFrame(cashflows["floating"]).set_index("payment_date")
            if len(cashflows["redemption"]) != 0:
                df_redemption = pd.DataFrame(cashflows["redemption"]).set_index("payment_date")
        except Exception as e:
            raise e
        if df_cupons is None and df_redemption is None:
            return pd.DataFrame()
        elif df_cupons is None and df_redemption is not None:
            net_cashflow = df_redemption["amount"]
        elif df_redemption is None and df_cupons is not None:
            net_cashflow = df_cupons["amount"]
        else:
            joint_index = df_cupons.index.union(df_redemption.index)
            df_cupons = df_cupons.reindex(joint_index).fillna(0.0)
            df_redemption = df_redemption.reindex(joint_index).fillna(0.0)

            net_cashflow = df_cupons["amount"] + df_redemption["amount"]
            net_cashflow.name = "net_cashflow"
        return net_cashflow
    def get_cashflows_df(self):
        import pandas as pd
        cashflows= self.get_cashflows()

        df_cupons=pd.DataFrame(cashflows["floating"]).set_index("payment_date")
        df_redemption = pd.DataFrame(cashflows["redemption"]).set_index("payment_date")

        joint_index = df_cupons.index.union(df_redemption.index)
        df_cupons = df_cupons.reindex(joint_index).fillna(0.0)
        df_redemption = df_redemption.reindex(joint_index).fillna(0.0)

        df_cupons["redemption"] = df_redemption["amount"]

        return df_cupons

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

    def get_yield(self,override_clean_price:Optional[float]=None) -> float:
        self._setup_pricer()
        # Make sure we evaluate on this object's valuation_date
        ql.Settings.instance().evaluationDate = to_ql_date(self.valuation_date)

        clean_price=override_clean_price
        if clean_price is None:
            clean_price = self._bond.cleanPrice()  # price per 100 nominal
        freq: ql.Frequency = self.coupon_frequency.frequency()  # convert Period -> Frequency
        settlement: ql.Date = self._bond.settlementDate()

        # Use the overload: yield(Real, DayCounter, Compounding, Frequency, Date)
        ytm = self._bond.bondYield(
            clean_price,
            self.day_count,
            ql.Compounded,
            freq,
            settlement
        )
        return float(ytm)