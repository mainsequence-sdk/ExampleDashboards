from __future__ import annotations

import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from .base_instrument import Instrument  # runtime_checkable Protocol: requires .price() -> float

from typing import Type, Mapping
from .european_option import EuropeanOption
from .vanilla_fx_option import VanillaFXOption
from .knockout_fx_option import KnockOutFXOption
from .fixed_rate_bond import FixedRateBond
from .floating_rate_bond import FloatingRateBond
from .interest_rate_swap import InterestRateSwap, TIIESwap


@dataclass(frozen=True)
class PositionLine:
    """
    A single position: an instrument and the number of units held.
    Units may be negative for short positions.
    """
    instrument: Instrument
    units: float = 1.0

    def unit_price(self) -> float:
        return float(self.instrument.price())

    def market_value(self) -> float:
        return self.units * self.unit_price()


class Position(BaseModel):
    """
    A collection of instrument positions with convenient aggregations.

    - Each line is an (instrument, units) pair.
    - `price()` returns the sum of units * instrument.price().
    - `get_cashflows(aggregate=...)` merges cashflows from instruments that expose `get_cashflows()`.
      * Expects each instrument's `get_cashflows()` to return a dict[str, list[dict]], like the swap.
      * Amounts are scaled by `units`. Unknown structures are passed through best-effort.
    - `get_greeks()` sums greeks from instruments that expose `get_greeks()`.
    """

    lines: List[PositionLine] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_json_dict(
            cls,
            data: Dict[str, Any],
            registry: Optional[Mapping[str, Type]] = None
    ) -> "Position":
        # default registry with your known instruments
        reg = dict(registry or {
            "EuropeanOption": EuropeanOption,
            "VanillaFXOption": VanillaFXOption,
            "KnockOutFXOption": KnockOutFXOption,
            "FixedRateBond": FixedRateBond,
            "FloatingRateBond": FloatingRateBond,
            "InterestRateSwap": InterestRateSwap,
            "TIIESwap": TIIESwap,
        })
        lines: List[PositionLine] = []
        for item in data.get("lines", []):
            t = item.get("instrument_type")
            payload = item.get("instrument", {})
            units = float(item.get("units", 1.0))
            cls_ = reg.get(t)
            if cls_ is None or not hasattr(cls_, "from_json"):
                raise ValueError(f"Unknown or non-JSONable instrument type: {t}")
            inst = cls_.from_json(payload)  # <-- use JSONMixin.from_json (now accepts dict)
            lines.append(PositionLine(instrument=inst, units=units))
        return cls(lines=lines)

    # ---- validation ---------------------------------------------------------
    @field_validator("lines")
    @classmethod
    def _validate_lines(cls, v: List[PositionLine]) -> List[PositionLine]:
        for i, line in enumerate(v):
            inst = line.instrument
            # Accept anything implementing the Instrument Protocol (price() -> float)
            if not hasattr(inst, "price") or not callable(getattr(inst, "price")):
                raise TypeError(
                    f"lines[{i}].instrument must implement price() -> float; got {type(inst).__name__}"
                )
        return v

    # ---- mutation helpers ---------------------------------------------------
    def add(self, instrument: Instrument, units: float = 1.0) -> None:
        """Append a new position line."""
        self.lines.append(PositionLine(instrument=instrument, units=units))

    def extend(self, items: Iterable[Tuple[Instrument, float]]) -> None:
        """Append many (instrument, units) items."""
        for inst, qty in items:
            self.add(inst, qty)

    # ---- pricing ------------------------------------------------------------
    def price(self) -> float:
        """Total market value: Σ units * instrument.price()."""
        return float(sum(line.market_value() for line in self.lines))

    def price_breakdown(self) -> List[Dict[str, Any]]:
        """
        Line-by-line price decomposition.
        Returns: [{instrument, units, unit_price, market_value}, ...]
        """
        out: List[Dict[str, Any]] = []
        for line in self.lines:
            out.append(
                {
                    "instrument": type(line.instrument).__name__,
                    "units": line.units,
                    "unit_price": line.unit_price(),
                    "market_value": line.market_value(),
                }
            )
        return out

    # ---- cashflows ----------------------------------------------------------
    def get_cashflows(self, aggregate: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Merge cashflows from all instruments that implement `get_cashflows()`.

        Returns a dict keyed by leg/label (e.g., "fixed", "floating") with lists of cashflow dicts.
        Each cashflow's 'amount' is scaled by position units. Original fields are preserved;
        metadata 'instrument' and 'units' are added for traceability.

        If aggregate=True, amounts are summed by payment date within each leg.
        """
        combined: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for idx, line in enumerate(self.lines):
            inst = line.instrument
            if not hasattr(inst, "get_cashflows"):
                continue  # silently skip instruments without cashflows
            flows = inst.get_cashflows()  # type: ignore[attr-defined]
            if not isinstance(flows, dict):
                continue

            for leg, items in flows.items():
                if not isinstance(items, (list, tuple)):
                    continue
                for cf in items:
                    if not isinstance(cf, dict):
                        continue
                    scaled = dict(cf)  # shallow copy
                    # scale common amount field if present
                    if "amount" in scaled and isinstance(scaled["amount"], (int, float)):
                        scaled["amount"] = float(scaled["amount"]) * line.units
                    # annotate
                    scaled.setdefault("instrument", type(inst).__name__)
                    scaled.setdefault("units", line.units)
                    scaled.setdefault("position_index", idx)
                    combined[leg].append(scaled)

        if not aggregate:
            return dict(combined)

        # Aggregate amounts by payment date (fallback to 'date' or 'fixing_date' if needed)
        aggregated: Dict[str, List[Dict[str, Any]]] = {}
        for leg, items in combined.items():
            buckets: Dict[datetime.date, float] = defaultdict(float)
            exemplars: Dict[datetime.date, Dict[str, Any]] = {}

            for cf in items:
                # identify a date field
                dt = (
                    cf.get("payment_date")
                    or cf.get("date")
                    or cf.get("fixing_date")
                )
                if isinstance(dt, datetime.date):
                    amount = float(cf.get("amount", 0.0))
                    buckets[dt] += amount
                    # keep exemplar fields for output ordering/context
                    if dt not in exemplars:
                        exemplars[dt] = {k: v for k, v in cf.items() if k not in {"amount", "units", "position_index"}}
                # if no usable date, just pass through (unaggregated)
                else:
                    buckets_key = None  # sentinel
                    # Collect undated flows under today's key to avoid loss
                    buckets[datetime.date.today()] += float(cf.get("amount", 0.0))

            # build sorted list
            leg_rows: List[Dict[str, Any]] = []
            for dt, amt in sorted(buckets.items(), key=lambda kv: kv[0]):
                row = {"payment_date": dt, "amount": amt}
                # attach exemplar metadata if any
                ex = exemplars.get(dt)
                if ex:
                    row.update({k: v for k, v in ex.items() if k in ("leg", "rate", "spread")})
                leg_rows.append(row)
            aggregated[leg] = leg_rows

        return aggregated

    # ---- greeks (optional) --------------------------------------------------
    def get_greeks(self) -> Dict[str, float]:
        """
        Aggregate greeks from instruments that implement `get_greeks()`.

        For each instrument i with dictionary Gi and units ui, returns Σ ui * Gi[key].
        Keys not common across all instruments are included on a best-effort basis.
        """
        totals: Dict[str, float] = defaultdict(float)
        for line in self.lines:
            inst = line.instrument
            getg = getattr(inst, "get_greeks", None)
            if callable(getg):
                g = getg()
                if isinstance(g, dict):
                    for k, v in g.items():
                        if isinstance(v, (int, float)):
                            totals[k] += line.units * float(v)
        return dict(totals)

    # ---- convenience constructors -------------------------------------------
    @classmethod
    def from_single(cls, instrument: Instrument, units: float = 1.0) -> "Position":
        return cls(lines=[PositionLine(instrument=instrument, units=units)])
