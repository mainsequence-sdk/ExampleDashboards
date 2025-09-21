# dashboards/curves/rataes.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple, List,Optional, Union
import numpy as np
import QuantLib as ql
import math

from mainsequence.instruments.pricing_models.indices import get_index
from mainsequence.instruments.pricing_models.swap_pricer import price_vanilla_swap_with_curve
from mainsequence.instruments.utils import to_py_date
from mainsequence.instruments.settings import TIIE_28_UID

@dataclass(frozen=True)
class SwapConventions:
    fixed_leg_tenor: ql.Period
    float_leg_tenor: ql.Period
    fixed_leg_daycount: ql.DayCounter
    fixed_leg_convention: ql.BusinessDayConvention
    float_leg_spread: float = 0.0

class ParRateCalculator(Protocol):
    def fair_rate(self, start: ql.Date, maturity: ql.Date, curve: ql.YieldTermStructureHandle) -> float: ...

class TIIE28ParCalculator:
    def __init__(self, index_identifier: str ):
        self.index_identifier = index_identifier
        self._conv = SwapConventions(
            fixed_leg_tenor=ql.Period(28, ql.Days),
            float_leg_tenor=ql.Period(28, ql.Days),
            fixed_leg_daycount=ql.Actual360(),
            fixed_leg_convention=ql.ModifiedFollowing,
            float_leg_spread=0.0,
        )

    def _index(self, ref: ql.Date):
        return get_index(target_date=to_py_date(ref), index_identifier=self.index_identifier)

    def fair_rate(self, start: ql.Date, maturity: ql.Date, curve: ql.YieldTermStructureHandle) -> float:
        ibor = self._index(start)
        swap = price_vanilla_swap_with_curve(
            calculation_date=start,
            notional=1.0,
            start_date=start,
            maturity_date=maturity,
            fixed_rate=0.0,
            fixed_leg_tenor=self._conv.fixed_leg_tenor,
            fixed_leg_convention=self._conv.fixed_leg_convention,
            fixed_leg_daycount=self._conv.fixed_leg_daycount,
            float_leg_tenor=self._conv.float_leg_tenor,
            float_leg_spread=self._conv.float_leg_spread,
            ibor_index=ibor,
            curve=curve,
        )
        return swap.fairRate()

def par_curve(ts: ql.YieldTermStructureHandle,
              max_years: int,
              step_months: int,
              calc: ParRateCalculator) -> Tuple[np.ndarray, np.ndarray]:
    spot = ts.referenceDate()
    cal = ts.calendar()
    xdc = ql.Actual360()  # x-axis year fraction (as in your current plot)
    T, Y = [], []
    m = step_months
    while m <= max_years * 12:
        mat = cal.advance(spot, ql.Period(m, ql.Months))
        T.append(xdc.yearFraction(spot, mat))
        Y.append(calc.fair_rate(spot, mat, ts))
        m += step_months
    return np.array(T), np.array(Y)

def par_nodes_from_tenors(ts: ql.YieldTermStructureHandle,
                          nodes: List[dict],
                          calc: ParRateCalculator) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    spot = ts.referenceDate()
    cal_axis = ql.TARGET()
    dc_axis = ql.Actual365Fixed()
    xs, ys, labels = [], [], []
    for n in nodes:
        tstr = (n.get("tenor") or "").upper()
        if not tstr:
            continue
        per = ql.Period(tstr)
        mat = cal_axis.advance(spot, per)
        xs.append(dc_axis.yearFraction(spot, mat))
        ys.append(calc.fair_rate(spot, mat, ts))
        labels.append(f"SWAP {tstr}")
    return np.array(xs), np.array(ys), labels




