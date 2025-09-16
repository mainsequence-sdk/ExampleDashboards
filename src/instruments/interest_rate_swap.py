import datetime
from typing import Optional, List, Dict, Any

import QuantLib as ql
from pydantic import BaseModel, Field, PrivateAttr, field_serializer, field_validator

from src.data_interface import data_interface
from src.pricing_models.swap_pricer import (price_vanilla_swap, get_swap_cashflows,
                                            price_vanilla_swap_with_curve,make_ftiie_index,
price_ftiie_ois_with_curve

                                            )
from src.pricing_models.indices import (build_tiie_zero_curve_from_valmer,get_index)
from src.utils import to_ql_date,to_py_date
from .json_codec import (
    JSONMixin,
    period_to_json, period_from_json,
    daycount_to_json, daycount_from_json,
    ibor_to_json, ibor_from_json,
)
import pandas as pd
from .base_instrument import InstrumentModel


class InterestRateSwap(InstrumentModel):
    """Plain-vanilla fixed-for-floating interest rate swap."""

    notional: float = Field(
        ..., description="Contract notional (currency units)."
    )
    start_date: datetime.date = Field(
        ..., description="Effective date when the swap begins accruing."
    )
    maturity_date: datetime.date = Field(
        ..., description="Termination date of the swap."
    )
    fixed_rate: float = Field(
        ..., description="Annual fixed rate on the fixed leg (decimal, e.g., 0.055)."
    )

    fixed_leg_tenor: ql.Period = Field(
        ..., description="Fixed-leg coupon period, e.g., ql.Period('6M')."
    )
    fixed_leg_convention: int = Field(
        ..., description="Business-day convention for fixed leg (QuantLib enum int, e.g., ql.Unadjusted)."
    )
    fixed_leg_daycount: ql.DayCounter = Field(
        ..., description="Day-count basis for fixed leg, e.g., ql.Thirty360(ql.Thirty360.USA)."
    )

    float_leg_tenor: ql.Period = Field(
        ..., description="Floating-leg coupon period, e.g., ql.Period('3M')."
    )
    float_leg_spread: float = Field(
        ..., description="Spread (decimal) added to the floating index."
    )
    float_leg_ibor_index: ql.IborIndex = Field(
        ..., description="Floating index object (e.g., ql.USDLibor(ql.Period('3M'), curve))."
    )

    valuation_date: datetime.datetime = Field(
        ..., description="Valuation date (sets QuantLib evaluation date)."
    )

    model_config = {"arbitrary_types_allowed": True}

    _swap: Optional[ql.VanillaSwap] = PrivateAttr(default=None)

    # ---------- JSON field serializers ----------
    @field_serializer("fixed_leg_tenor", "float_leg_tenor", when_used="json")
    def _ser_tenors(self, v: ql.Period, _info) -> str:
        return period_to_json(v)

    @field_serializer("fixed_leg_daycount", when_used="json")
    def _ser_fixed_daycount(self, v: ql.DayCounter) -> str:
        return daycount_to_json(v)

    @field_serializer("float_leg_ibor_index", when_used="json")
    def _ser_ibor(self, v: Optional[ql.IborIndex]) -> Optional[Dict[str, Any]]:
        return ibor_to_json(v)

    # ---------- JSON field validators (decode) ----------
    @field_validator("fixed_leg_tenor", "float_leg_tenor", mode="before")
    @classmethod
    def _val_periods(cls, v):
        return period_from_json(v)

    @field_validator("fixed_leg_daycount", mode="before")
    @classmethod
    def _val_fixed_daycount(cls, v):
        return daycount_from_json(v)

    @field_validator("float_leg_ibor_index", mode="before")
    @classmethod
    def _val_ibor(cls, v):
        # Rebuild a basic index shell (curve linking happens in pricing code).
        return ibor_from_json(v)



    def _setup_pricer(self) -> None:
        if self._swap is None:
            ql_val = to_ql_date(self.valuation_date)
            ql.Settings.instance().evaluationDate = ql_val
            self._swap = price_vanilla_swap(
                calculation_date=ql_val,
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
        self._setup_pricer()
        return float(self._swap.NPV())

    def get_cashflows(self) -> Dict[str, List[Dict[str, Any]]]:
        self._setup_pricer()
        return get_swap_cashflows(self._swap)
    def get_net_cashflows(self)->pd.Series:
        cashflows=self.get_cashflows()
        fixed_df=pd.DataFrame(cashflows["fixed"]).set_index("payment_date")
        float_df = pd.DataFrame(cashflows["floating"]).set_index("payment_date")
        joint_index=fixed_df.index.union(float_df.index)
        fixed_df=fixed_df.reindex(joint_index).fillna(0.0)
        float_df=float_df.reindex(joint_index).fillna(0.0)

        net_cashflow=fixed_df["amount"]-float_df["amount"]
        net_cashflow.name="net_cashflow"
        return net_cashflow


class TIIESwap(InterestRateSwap):
    """
    Mexican peso fixed-for-floating TIIE(28D) swap.
    Inherits the IR swap model and only changes:
      - leg defaults (28D, ACT/360),
      - pricing hookup (uses Valmer TIIE zero curve).
    """
    tenor: Optional[ql.Period] = Field(
        default=None,
        description="If set (e.g. ql.Period('156W')), maturity is start + tenor using spot start (T+1)."
    )
    # Override some defaults for MXN/TIIE
    fixed_leg_tenor: ql.Period = Field(default=ql.Period("28D"), description="Fixed leg frequency (28D).")
    fixed_leg_convention: int = Field(default=ql.ModifiedFollowing, description="BDC for fixed leg (ModFollowing).")
    fixed_leg_daycount: ql.DayCounter = Field(default=ql.Actual360(), description="Fixed leg day count (ACT/360).")
    float_leg_tenor: ql.Period = Field(default=ql.Period("28D"), description="Floating leg frequency (28D).")
    float_leg_ibor_index: Optional[ql.IborIndex] = Field(
        default=None,
        description="If None, a TIIE-28D index linked to the Valmer zero curve will be created."
    )

    model_config = {"arbitrary_types_allowed": True}
    _swap: Optional[ql.VanillaSwap] = PrivateAttr(default=None)

    # Serializer/validator for the optional 'tenor'
    @field_serializer("tenor", when_used="json")
    def _ser_tenor(self, v: Optional[ql.Period]) -> Optional[str]:
        return period_to_json(v)

    @field_validator("tenor", mode="before")
    @classmethod
    def _val_tenor(cls, v):
        return period_from_json(v)

    def _build_swap(self, curve: ql.YieldTermStructure) -> None:
        """
        Private helper method to construct the QuantLib swap object.
        This contains the common logic previously duplicated in _setup_pricer and reset_curve.
        """
        ql_val = to_ql_date(self.valuation_date)
        ql.Settings.instance().evaluationDate = ql_val
        ql.Settings.instance().includeReferenceDateEvents = False
        ql.Settings.instance().enforceTodaysHistoricFixings = False

        # 1) TIIE-28D index on the provided curve
        tiie = get_index(
            "TIIE", tenor="28D",
            forwarding_curve=curve,
            target_date=self.valuation_date,
            hydrate_fixings=True
        )
        cal = tiie.fixingCalendar()

        # 2) Effective start = T+1 FROM TRADE (your start_date), not valueDate()
        trade = to_ql_date(self.start_date)
        eff_start = cal.adjust(trade, ql.Following)

        # 3) Effective end
        if self.tenor is not None:
            eff_end = cal.advance(eff_start, self.tenor)
        else:
            eff_end = to_ql_date(self.maturity_date)

        # 4) Price vanilla IRS using the schedule and the provided curve
        self._swap = price_vanilla_swap_with_curve(
            calculation_date=ql_val,
            notional=self.notional,
            start_date=eff_start,
            maturity_date=eff_end,
            fixed_rate=self.fixed_rate,
            fixed_leg_tenor=self.fixed_leg_tenor,
            fixed_leg_convention=self.fixed_leg_convention,
            fixed_leg_daycount=self.fixed_leg_daycount,
            float_leg_tenor=self.float_leg_tenor,
            float_leg_spread=self.float_leg_spread,
            ibor_index=tiie,
            curve=curve,
        )

    def reset_curve(self, curve: ql.YieldTermStructure) -> None:
        """Re-prices the swap using a new, user-provided curve."""
        # Call the common swap construction logic with the new curve.
        self._build_swap(curve)

    def _setup_pricer(self) -> None:
        """Sets up the initial pricer using the default Valmer curve."""
        if self._swap is not None:
            return

        # Build the default TIIE curve.
        default_curve = build_tiie_zero_curve_from_valmer(self.valuation_date)

        # Call the common swap construction logic.
        self._build_swap(default_curve)

