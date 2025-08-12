import datetime
from typing import Optional, List, Dict, Any

import QuantLib as ql
from pydantic import BaseModel, Field, PrivateAttr

from src.pricing_models.swap_pricer import (price_vanilla_swap, get_swap_cashflows,
                                            build_tiie_zero_curve_from_valmer,
                                            price_vanilla_swap_with_curve,

                                            )
from src.utils import to_ql_date


class InterestRateSwap(BaseModel):
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

    valuation_date: datetime.date = Field(
        ..., description="Valuation date (sets QuantLib evaluation date)."
    )

    model_config = {"arbitrary_types_allowed": True}

    _swap: Optional[ql.VanillaSwap] = PrivateAttr(default=None)

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


class TIIESwap(InterestRateSwap):
    """
    Mexican peso fixed-for-floating TIIE(28D) swap.
    Inherits the IR swap model and only changes:
      - leg defaults (28D, ACT/360),
      - pricing hookup (uses Valmer TIIE zero curve).
    """

    # Override some defaults for MXN/TIIE
    fixed_leg_tenor: ql.Period = Field(
        default=ql.Period("28D"),
        description="Fixed-leg coupon period (default: 28D)."
    )
    fixed_leg_daycount: ql.DayCounter = Field(
        default=ql.Actual360(),
        description="Fixed-leg day-count basis (default: ACT/360)."
    )
    float_leg_tenor: ql.Period = Field(
        default=ql.Period("28D"),
        description="Floating-leg coupon period (TIIE 28D)."
    )
    # Allow omitting the index; we’ll create TIIE(28D) from the Valmer curve if missing.
    float_leg_ibor_index: Optional[ql.IborIndex] = Field(
        default=None,
        description="Floating index. If omitted, uses ql.TIIE(28D) linked to the Valmer zero curve."
    )

    model_config = {"arbitrary_types_allowed": True}
    _swap: Optional[ql.VanillaSwap] = PrivateAttr(default=None)

    def _setup_pricer(self) -> None:
        if self._swap is not None:
            return

        ql_val = to_ql_date(self.valuation_date)
        ql.Settings.instance().evaluationDate = ql_val

        # 1) Build MXN TIIE zero curve (Valmer)
        curve = build_tiie_zero_curve_from_valmer(ql_val)

        # 2) Create/validate the TIIE index
        if self.float_leg_ibor_index is not None:
            ibor = self.float_leg_ibor_index
        else:
            if hasattr(ql, "TIIE"):
                ibor = ql.TIIE(ql.Period("28D"), curve)
            else:
                # Pragmatic fallback if ql.TIIE isn't available in your wheel:
                ibor = ql.USDLibor(ql.Period("1M"), curve)

        # 3) Price using the provided curve (no USD bootstrapping)
        self._swap = price_vanilla_swap_with_curve(
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
            ibor_index=ibor,
            curve=curve,  # <- key: use the Valmer zero curve
        )