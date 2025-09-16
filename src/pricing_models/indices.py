# pricing_models/indices.py
# -*- coding: utf-8 -*-
"""
Index factory for QuantLib.

Usage
-----
>>> from pricing_models.indices import get_index
>>> idx1 = get_index("TIIE_28", target_date=date(2024, 6, 14))  # TIIE 28-day as of target_date
>>> idx2 = get_index("EURIBOR")                                 # Defaults to 3M Euribor
>>> idx3 = get_index("EURIBOR_6M")
>>> idx4 = get_index("SOFOR")                                   # Robust aliasing -> SOFR
>>> idx5 = get_index("USD_LIBOR_3M")
>>> idx6 = get_index("SONIA")

You can also supply a forwarding curve handle (date-aware registry still works):
>>> h = ql.RelinkableYieldTermStructureHandle()
>>> idx = get_index("EURIBOR_1M", forwarding_curve=h)

Extensibility
-------------
- Register your own index builders:
>>> from pricing_models.indices import register_index
>>> register_index("MXN_TIIE_91", lambda curve, *, target_date=None, settlement_days=None: MyTiie91(curve))

- Discover what’s available:
>>> from pricing_models.indices import list_registered
>>> list_registered()

Notes
-----
- Some indices (e.g., TIIE) may or may not be present depending on your QuantLib build.
- Name parsing is forgiving: spaces, hyphens, colons, and case are ignored;
  'SOFOR' is treated as 'SOFR'; 'EURIBOR' defaults to 'EURIBOR_3M'.
- NEW: The registry is date-aware. Pass `target_date` to get curve state as of that date.
"""

from __future__ import annotations

import datetime
import re
from typing import Callable, Dict, Optional, Tuple, Union, Any

import QuantLib as ql
from functools import lru_cache

from src.data_interface import data_interface
from src.utils import to_py_date, to_ql_date
import os


# ----------------------------- Normalization helpers ----------------------------- #

def _normalize_name(name: str) -> str:
    s = name.strip().upper()
    s = s.replace("€", "E")
    s = s.replace("O/N", "ON")
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    s = s.strip("_")
    return s


def _split_tokens(key: str) -> Tuple[str, Tuple[str, ...]]:
    parts = tuple(t for t in key.split("_") if t)
    if not parts:
        raise ValueError("Empty index name")
    return parts[0], parts[1:]


def _normalize_tenor(maybe_tenor: Optional[str]) -> Optional[str]:
    if maybe_tenor is None:
        return None
    t = maybe_tenor.strip().upper()
    if re.fullmatch(r"\d+", t):
        t = f"{t}D"
    if re.fullmatch(r"\d+(D|W|M|Y)", t):
        return t
    if t in {"ON", "O/N", "OVERNIGHT"}:
        return "ON"
    if t in {"SN", "TN"}:
        return t
    raise ValueError(f"Unrecognized tenor: {maybe_tenor!r}")


def _ensure_py_date(
    d: Optional[Union[datetime.date, datetime.datetime, ql.Date]]
) -> datetime.date:
    """Return a Python date. If None, prefer QuantLib evaluation date, else today()."""
    if d is None:
        try:
            qld = ql.Settings.instance().evaluationDate
            if qld != ql.Date():
                return to_py_date(qld)
        except Exception:
            pass
        return datetime.date.today()
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    # ql.Date
    return to_py_date(d)


# ----------------------------- Registry core ----------------------------- #

# Public registry that maps canonical keys -> builder(curve, *, target_date=None, settlement_days=None) -> QuantLib index
Builder = Callable[..., ql.Index]
_REGISTRY: Dict[str, Builder] = {}

# Aliases (normalized) -> canonical key (normalized)
_ALIASES: Dict[str, str] = {
    "SOFOR": "SOFR",
    "EON": "EONIA",
    "ESTER": "ESTR",
    "E_STR": "ESTR",
}
_ALIASES.update({
    "TIIE28D": "TIIE_28D",
    "TIIE-28D": "TIIE_28D",
    "TIIE91D": "TIIE_91D",
    "TIIE-91D": "TIIE_91D",
    "MXN_TIIE": "TIIE",
})


@lru_cache(maxsize=64)
def _default_tiie_curve_cached(date_key: datetime.date) -> ql.YieldTermStructureHandle:
    # Build a curve for the canonical (date-only) key and cache it.
    target_dt = datetime.datetime.combine(date_key, datetime.time())
    return build_tiie_zero_curve_from_valmer(target_dt)


def _default_tiie_curve(target_date: Optional[Union[datetime.date, datetime.datetime, ql.Date]]) -> ql.YieldTermStructureHandle:
    # Wrapper that normalizes and dispatches into the cached function
    dk = _ensure_py_date(target_date)
    return _default_tiie_curve_cached(dk)


def register_index(name: str, builder: Builder) -> None:
    """
    Register a builder under a normalized name so that `get_index(name)` finds it.

    The builder may be one of:
      - def builder(curve) -> Index
      - def builder(curve, *, target_date=None) -> Index
      - def builder(curve, *, target_date=None, settlement_days=None) -> Index
    """
    key = _normalize_name(name)
    _REGISTRY[key] = builder


def list_registered() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY.keys()))


def _create_if_exists(class_name: str, *args):
    cls = getattr(ql, class_name, None)
    return None if cls is None else cls(*args)


# ----------------------------- Built-in builders ----------------------------- #

def _euribor_builder(tenor: str) -> Builder:
    tenor = _normalize_tenor(tenor or "3M")

    def _b(curve: ql.YieldTermStructureHandle, *, target_date=None, settlement_days=None) -> ql.Index:
        return ql.Euribor(ql.Period(tenor), curve)

    return _b


def _libor_builder(currency: str, tenor: str) -> Builder:
    ccy = currency.upper()
    tenor = _normalize_tenor(tenor or "3M")
    class_name = f"{ccy}Libor"

    def _b(curve: ql.YieldTermStructureHandle, *, target_date=None, settlement_days=None) -> ql.Index:
        cls = getattr(ql, class_name, None)
        if cls is None:
            raise KeyError(
                f"{class_name} is not available in your QuantLib build. "
                f"Register a custom builder for '{ccy}_LIBOR_{tenor}' via register_index()."
            )
        return cls(ql.Period(tenor), curve)

    return _b


def _overnight_simple_builder(ql_class_name: str) -> Builder:
    def _b(curve: ql.YieldTermStructureHandle, *, target_date=None, settlement_days=None) -> ql.Index:
        cls = getattr(ql, ql_class_name, None)
        if cls is None:
            raise KeyError(f"{ql_class_name} is not available in your QuantLib build.")
        return cls(curve)
    return _b


def _tiie_registry_builder(tenor: str) -> Builder:
    """
    Registry builder for TIIE that is date-aware; it picks a cached Valmer curve for the given target_date
    unless a non-empty forwarding curve is supplied.
    """
    tenor = _normalize_tenor(tenor or "28D")

    def _b(curve: ql.YieldTermStructureHandle, *, target_date=None, settlement_days=None) -> ql.Index:
        try:
            use_curve = curve if (curve is not None and not curve.empty()) else _default_tiie_curve(target_date)
        except Exception:
            # As a last resort, fall back to today's curve
            use_curve = _default_tiie_curve(target_date)
        return make_tiie_index(use_curve, tenor, (settlement_days or 1))

    return _b


# Pre-register common overnight indices and well-known families
def _bootstrap_registry() -> None:
    # Overnight families (simple constructors)
    for name, ql_class in [
        ("SOFR", "Sofr"),
        ("SONIA", "Sonia"),
        ("EONIA", "Eonia"),
        ("ESTR", "Estr"),
        ("TONAR", "Tonar"),
        ("FEDFUNDS", "FedFunds"),
        ("SARON", "Saron"),
    ]:
        register_index(name, _overnight_simple_builder(ql_class))

    # Euribor family
    for t in ("1W", "2W", "1M", "3M", "6M", "12M"):
        register_index(f"EURIBOR_{t}", _euribor_builder(t))
    register_index("EURIBOR", _euribor_builder("3M"))

    # LIBOR (legacy)
    _libor_ccys = ("USD", "GBP", "JPY", "CHF", "EUR", "CAD", "AUD")
    _libor_tenors = ("1W", "1M", "3M", "6M", "12M")
    for c in _libor_ccys:
        for t in _libor_tenors:
            register_index(f"{c}_LIBOR_{t}", _libor_builder(c, t))
        register_index(f"LIBOR_{c}", _libor_builder(c, "3M"))

    # TIIE defaults (date-aware)
    register_index("TIIE_28D", _tiie_registry_builder("28D"))
    register_index("TIIE_28", _tiie_registry_builder("28D"))
    register_index("TIIE_91D", _tiie_registry_builder("91D"))
    register_index("TIIE_91", _tiie_registry_builder("91D"))
    register_index("TIIE", _tiie_registry_builder("28D"))


_bootstrap_registry()


# ----------------------------- Factory API ----------------------------- #

def add_historical_fixings(calculation_date: ql.Date, ibor_index: ql.IborIndex):
    from src.data_interface import data_interface
    import datetime
    from src.utils import to_py_date, to_ql_date

    print("Fetching and adding historical fixings...")

    end_date = to_py_date(calculation_date) - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=365)

    historical_fixings = data_interface.get_historical_fixings(
        index_name=ibor_index.name(),
        start_date=start_date,
        end_date=end_date
    )

    if not historical_fixings:
        print("No historical fixings found in the given date range.")
        return

    valid_qld: list[ql.Date] = []
    valid_rates: list[float] = []

    for dt_py, rate in sorted(historical_fixings.items()):
        qld = to_ql_date(dt_py)
        if qld < calculation_date and ibor_index.isValidFixingDate(qld):
            valid_qld.append(qld)
            valid_rates.append(rate)

    if not valid_qld:
        print("No valid fixing dates for the index calendar; skipping addFixings.")
        return

    ibor_index.addFixings(valid_qld, valid_rates, True)
    print(f"Successfully added {len(valid_qld)} fixings for {ibor_index.name()}.")


def build_tiie_zero_curve_from_valmer(target_date: Union[datetime.date, datetime.datetime]) -> ql.YieldTermStructureHandle:
    from src.utils import to_ql_date
    from src.settings import TIIE_28_ZERO_CURVE
    import datetime

    nodes = data_interface.get_historical_discount_curve(TIIE_28_ZERO_CURVE, target_date)

    ql_date = to_ql_date(target_date)
    base = ql_date
    base_py = target_date if isinstance(target_date, datetime.datetime) else datetime.datetime.combine(target_date, datetime.time())

    cal = ql.Mexico() if hasattr(ql, "Mexico") else ql.TARGET()
    dc = ql.Actual360()

    dates = [base]
    discounts = [1.0]
    seen = {base.serialNumber()}

    for n in sorted(nodes, key=lambda n: int(n["days_to_maturity"])):
        days = int(n["days_to_maturity"])
        if days < 0:
            continue
        d = base_py + datetime.timedelta(days=days)
        d = to_ql_date(d)

        sn = d.serialNumber()
        if sn in seen:
            continue

        z = n.get("zero", n.get("zero_rate", n.get("rate")))
        z = float(z)
        if z > 1.0:
            z *= 0.01

        T = dc.yearFraction(base, d)
        df = 1.0 / (1.0 + z * T)  # Valmer zero is simple ACT/360

        dates.append(d)
        discounts.append(df)
        seen.add(sn)

    ts = ql.DiscountCurve(dates, discounts, dc, cal)
    ts.enableExtrapolation()
    return ql.YieldTermStructureHandle(ts)


def make_tiie_index(
        curve: ql.YieldTermStructureHandle,
        tenor: str = "28D",
        settlement_days: int = 1,
) -> ql.IborIndex:
    """
    Construct a TIIE IborIndex linked to `curve` with a given day-based tenor (e.g., '28D', '91D').
    Conventions (MXN TIIE standard):
      - Settlement: T+1
      - Calendar: Mexico (TARGET fallback)
      - BDC: ModifiedFollowing
      - EOM: False
      - Day count: Actual/360
      - Name: 'TIIE-<tenor>' (e.g., TIIE-28D)
    """
    from src.data_interface import data_interface
    import datetime

    tenor = _normalize_tenor(tenor or "28D")
    if tenor in {"ON", "SN", "TN"}:
        raise ValueError("TIIE does not support overnight/SN/TN tenors; use a fixed-day tenor like '28D' or '91D'.")

    cal = ql.Mexico() if hasattr(ql, "Mexico") else ql.TARGET()
    try:
        ccy = ql.MXNCurrency()
    except Exception:
        ccy = ql.USDCurrency()

    index = ql.IborIndex(
        f"TIIE-{tenor}",
        ql.Period(tenor),
        settlement_days,
        ccy,
        cal,
        ql.ModifiedFollowing,
        False,
        ql.Actual360(),
        curve
    )

    fixings = data_interface.get_historical_fixings(
        index_name="TIIE28",
        start_date=datetime.date(1990, 1, 1),
        end_date=to_py_date(curve.referenceDate())
    )

    dates, values = [], []
    for py_d, v in sorted(fixings.items()):
        ql_d = to_ql_date(py_d)
        if index.isValidFixingDate(ql_d):
            dates.append(ql_d)
            values.append(float(v))

    if dates:
        index.addFixings(dates, values, True)

    return index


def make_tiie_28d_index(
        curve: ql.YieldTermStructureHandle,
        settlement_days: int = 1,
) -> ql.IborIndex:
    return make_tiie_index(curve, "28D", settlement_days)


def _apply_alias(key: str) -> str:
    return _ALIASES.get(key, key)


def _find_tenor_in_tokens(tokens: Tuple[str, ...]) -> Optional[str]:
    for t in tokens[::-1]:
        if re.fullmatch(r"\d+(D|W|M|Y)", t) or t in {"ON", "SN", "TN"}:
            return t
        if re.fullmatch(r"\d+", t):
            return f"{t}D"
    return None


def _invoke_builder(builder: Builder,
                    curve: ql.YieldTermStructureHandle,
                    *,
                    target_date: Optional[Union[datetime.date, datetime.datetime, ql.Date]] = None,
                    settlement_days: Optional[int] = None) -> ql.Index:
    """Call builders with flexible signatures while preserving backward compatibility."""
    try:
        return builder(curve, target_date=target_date, settlement_days=settlement_days)
    except TypeError:
        try:
            return builder(curve, target_date=target_date)
        except TypeError:
            try:
                return builder(curve, settlement_days=settlement_days)
            except TypeError:
                return builder(curve)


def get_index(
        name: str,
        target_date: datetime.datetime,
        *,
        tenor: Optional[str] = None,
        forwarding_curve: Optional[ql.YieldTermStructureHandle] = None,
        hydrate_fixings: bool = False,
        settlement_days: Optional[int] = None
) -> ql.Index:
    """
    Return a QuantLib index instance based on a human-friendly name.

    Parameters
    ----------
    name : str
        Examples: "TIIE_28", "EURIBOR", "EURIBOR_6M", "SOFOR", "USD_LIBOR_3M".
    tenor : Optional[str]
        Optional tenor override like "3M", "6M", "28D".
    forwarding_curve : Optional[ql.YieldTermStructureHandle]
        Forwarding curve handle. If empty/None, some families (e.g., TIIE) will source a curve by `target_date`.
    target_date : Optional[date|datetime|ql.Date]
        **New.** The as-of date used to source/default the curve from the registry when a curve is not supplied.
        If omitted, uses QuantLib evaluation date (if set) or today.
    calculation_date : Optional[ql.Date]
        Used only for optional historical fixings hydration. If omitted and `hydrate_fixings=True`,
        we’ll use `to_ql_date(target_date)` instead.
    hydrate_fixings : bool
        If True and the index is Ibor-like, we’ll backfill fixings up to `calculation_date`.
    settlement_days : Optional[int]
        Optional override (used by TIIE builders; defaults to 1 if omitted).

    Returns
    -------
    ql.Index
    """
    key0 = _normalize_name(name)
    key = _apply_alias(key0)
    family, rest = _split_tokens(key)

    curve = forwarding_curve or ql.YieldTermStructureHandle()

    # If an exact registry entry exists, prefer it (date-aware call)
    if key in _REGISTRY:
        idx = _invoke_builder(_REGISTRY[key], curve, target_date=target_date, settlement_days=settlement_days)
        # Optional fixings hydration
        if hydrate_fixings and isinstance(idx, ql.IborIndex):
            calc_dt = to_ql_date(target_date)
            add_historical_fixings(calc_dt, idx)
        return idx

    # Heuristics for families
    # 1) EURIBOR
    if family == "EURIBOR":
        t = _normalize_tenor(tenor) if tenor else _find_tenor_in_tokens(rest) or "3M"
        idx = _invoke_builder(_euribor_builder(t), curve, target_date=target_date_py, settlement_days=settlement_days)
        if hydrate_fixings and isinstance(idx, ql.IborIndex):
            calc_dt = calculation_date or to_ql_date(target_date_py)
            add_historical_fixings(calc_dt, idx)
        return idx

    # 2) TIIE (incl. MXN_TIIE forms)
    if family in {"TIIE", "MXN", "MXN_TIIE"} or "TIIE" in rest:
        t = _normalize_tenor(tenor) if tenor else _find_tenor_in_tokens((family,) + rest) or "28D"
        idx = _invoke_builder(_tiie_registry_builder(t), curve, target_date=target_date_py, settlement_days=settlement_days)
        if hydrate_fixings and isinstance(idx, ql.IborIndex):
            calc_dt = calculation_date or to_ql_date(target_date_py)
            add_historical_fixings(calc_dt, idx)
        return idx

    # 3) <CCY>_LIBOR_<tenor> or LIBOR_<CCY>[_<tenor>]
    if family == "LIBOR" and rest:
        ccy = rest[0]
        t = _normalize_tenor(tenor) if tenor else _find_tenor_in_tokens(rest[1:]) or "3M"
        idx = _invoke_builder(_libor_builder(ccy, t), curve, target_date=target_date_py, settlement_days=settlement_days)
        if hydrate_fixings and isinstance(idx, ql.IborIndex):
            calc_dt = calculation_date or to_ql_date(target_date_py)
            add_historical_fixings(calc_dt, idx)
        return idx

    if family in {"USD", "GBP", "JPY", "CHF", "EUR", "CAD", "AUD"} and (len(rest) and rest[0] == "LIBOR"):
        ccy = family
        t = _normalize_tenor(tenor) if tenor else _find_tenor_in_tokens(rest[1:]) or "3M"
        idx = _invoke_builder(_libor_builder(ccy, t), curve, target_date=target_date_py, settlement_days=settlement_days)
        if hydrate_fixings and isinstance(idx, ql.IborIndex):
            calc_dt = calculation_date or to_ql_date(target_date_py)
            add_historical_fixings(calc_dt, idx)
        return idx

    # 4) Overnight simple classes by bare name
    if family in {"SOFR", "SONIA", "EONIA", "ESTR", "TONAR", "FEDFUNDS", "SARON"}:
        idx = _invoke_builder(_overnight_simple_builder(family.title()), curve, target_date=target_date_py, settlement_days=settlement_days)
        if hydrate_fixings and isinstance(idx, ql.IborIndex):
            calc_dt = calculation_date or to_ql_date(target_date_py)
            add_historical_fixings(calc_dt, idx)
        return idx

    # 5) Fallback: user-registered under base family
    if family in _REGISTRY:
        idx = _invoke_builder(_REGISTRY[family], curve, target_date=target_date_py, settlement_days=settlement_days)
        if hydrate_fixings and isinstance(idx, ql.IborIndex):
            calc_dt = calculation_date or to_ql_date(target_date_py)
            add_historical_fixings(calc_dt, idx)
        return idx

    suggestions = ", ".join(
        s for s in list_registered()
        if s.startswith(family) or family in s
    ) or "no close matches"
    raise KeyError(
        f"Unrecognized index name: {name!r} (normalized: {key0!r}). "
        f"Close matches/registered: {suggestions}."
    )


# ----------------------------- Convenience alias ----------------------------- #

index_by_name = get_index
