# pricing_models/indices.py
# -*- coding: utf-8 -*-
"""
Index factory for QuantLib.

Usage
-----
>>> from pricing_models.indices import get_index
>>> idx1 = get_index("TIIE_28")              # TIIE 28-day (if your QuantLib build provides it)
>>> idx2 = get_index("EURIBOR")              # Defaults to 3M Euribor
>>> idx3 = get_index("EURIBOR_6M")           # 6M Euribor
>>> idx4 = get_index("SOFOR")                # Robust aliasing -> SOFR
>>> idx5 = get_index("USD_LIBOR_3M")         # 3M USD LIBOR (legacy index)
>>> idx6 = get_index("SONIA")                # GBP overnight

You can also supply a forwarding curve handle:
>>> h = ql.RelinkableYieldTermStructureHandle()
>>> idx = get_index("EURIBOR_1M", forwarding_curve=h)

Extensibility
-------------
- Register your own index builders:
>>> from pricing_models.indices import register_index
>>> register_index("MXN_TIIE_91", lambda curve: MyTiie91(curve))

- Discover what’s available:
>>> from pricing_models.indices import list_registered
>>> list_registered()

Notes
-----
- Some indices (e.g., TIIE) may or may not be present depending on your QuantLib build.
  If your build lacks a specific class, the factory will tell you how to register one.
- Name parsing is forgiving: spaces, hyphens, colons, and case are ignored;
  'SOFOR' is treated as 'SOFR'; 'EURIBOR' defaults to 'EURIBOR_3M'.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Optional, Tuple

import QuantLib as ql
from functools import lru_cache


# ----------------------------- Normalization helpers ----------------------------- #

def _normalize_name(name: str) -> str:
    """
    Upper-cases, strips whitespace, replaces non-alphanumerics with underscores,
    normalizes common variants (e.g., O/N -> ON, €STR -> ESTR).
    """
    s = name.strip().upper()
    s = s.replace("€", "E")
    s = s.replace("O/N", "ON")
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    s = s.strip("_")
    return s


def _split_tokens(key: str) -> Tuple[str, Tuple[str, ...]]:
    """
    Splits a normalized key into first token (family) + rest (params),
    e.g., "EURIBOR_6M" -> ("EURIBOR", ("6M",))
          "USD_LIBOR_3M" -> ("USD", ("LIBOR", "3M"))
    """
    parts = tuple(t for t in key.split("_") if t)
    if not parts:
        raise ValueError("Empty index name")
    return parts[0], parts[1:]


def _normalize_tenor(maybe_tenor: Optional[str]) -> Optional[str]:
    if maybe_tenor is None:
        return None
    t = maybe_tenor.strip().upper()
    # Bare digits -> assume days
    if re.fullmatch(r"\d+", t):
        t = f"{t}D"
    # Valid tenor tokens like 1W/2W/1M/3M/6M/12M/28D/90D/1Y
    if re.fullmatch(r"\d+(D|W|M|Y)", t):
        return t
    # Overnight synonyms
    if t in {"ON", "O/N", "OVERNIGHT"}:
        return "ON"
    # Spot-next / tomorrow-next (not universally supported by all families)
    if t in {"SN", "TN"}:
        return t
    raise ValueError(f"Unrecognized tenor: {maybe_tenor!r}")


# ----------------------------- Registry core ----------------------------- #

# Public registry that maps canonical keys -> builder(curve) -> QuantLib index
_REGISTRY: Dict[str, Callable[[ql.YieldTermStructureHandle], ql.Index]] = {}

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

@lru_cache(maxsize=1)
def _default_tiie_curve() -> ql.YieldTermStructureHandle:
    # Use your own builder; cached so we don’t rebuild from CSV every call.
    return build_tiie_zero_curve_from_valmer(None)


def register_index(name: str, builder: Callable[[ql.YieldTermStructureHandle], ql.Index]) -> None:
    """
    Register a builder under a normalized name so that `get_index(name)` finds it.
    """
    key = _normalize_name(name)
    _REGISTRY[key] = builder


def list_registered() -> Tuple[str, ...]:
    """
    Returns currently registered canonical keys (sorted).
    """
    return tuple(sorted(_REGISTRY.keys()))


# ----------------------------- Built-in builders ----------------------------- #

def _create_if_exists(class_name: str, *args):
    cls = getattr(ql, class_name, None)
    return None if cls is None else cls(*args)

def _euribor_builder(tenor: str) -> Callable[[ql.YieldTermStructureHandle], ql.Index]:
    tenor = _normalize_tenor(tenor or "3M")
    def _b(curve: ql.YieldTermStructureHandle) -> ql.Index:
        # Prefer generic constructor to avoid relying on convenience subclasses
        return ql.Euribor(ql.Period(tenor), curve)
    return _b

def _libor_builder(currency: str, tenor: str) -> Callable[[ql.YieldTermStructureHandle], ql.Index]:
    ccy = currency.upper()
    tenor = _normalize_tenor(tenor or "3M")
    class_name = f"{ccy}Libor"  # e.g., USDLibor, GBPLibor, JPYLibor, CHFLibor, CADLibor, EURLibor
    def _b(curve: ql.YieldTermStructureHandle) -> ql.Index:
        cls = getattr(ql, class_name, None)
        if cls is None:
            raise KeyError(
                f"{class_name} is not available in your QuantLib build. "
                f"Register a custom builder for '{ccy}_LIBOR_{tenor}' via register_index()."
            )
        return cls(ql.Period(tenor), curve)
    return _b

def _overnight_simple_builder(ql_class_name: str) -> Callable[[ql.YieldTermStructureHandle], ql.Index]:
    def _b(curve: ql.YieldTermStructureHandle) -> ql.Index:
        cls = getattr(ql, ql_class_name, None)
        if cls is None:
            raise KeyError(f"{ql_class_name} is not available in your QuantLib build.")
        return cls(curve)
    return _b


def _tiie_registry_builder(tenor: str) -> Callable[[ql.YieldTermStructureHandle], ql.Index]:
    def _b(curve: ql.YieldTermStructureHandle) -> ql.Index:
        # Honor a provided curve; otherwise fall back to the cached Valmer curve.
        try:
            use_curve = curve if (curve is not None and not curve.empty()) else _default_tiie_curve()
        except Exception:
            use_curve = _default_tiie_curve()
        return make_tiie_index(use_curve, tenor)
    return _b

# Pre-register common overnight indices and well-known families
def _bootstrap_registry() -> None:
    # Overnight families (simple constructors)
    for name, ql_class in [
        ("SOFR", "Sofr"),
        ("SONIA", "Sonia"),
        ("EONIA", "Eonia"),
        ("ESTR", "Estr"),     # if unavailable, user can alias/register to their build
        ("TONAR", "Tonar"),   # JPY TONA (often spelled TONAR in QL)
        ("FEDFUNDS", "FedFunds"),
        ("SARON", "Saron"),
    ]:
        register_index(name, _overnight_simple_builder(ql_class))

    # Euribor family: default tenor = 3M if omitted
    for t in ("1W", "2W", "1M", "3M", "6M", "12M"):
        register_index(f"EURIBOR_{t}", _euribor_builder(t))
    register_index("EURIBOR", _euribor_builder("3M"))

    # LIBOR (legacy) — add common currencies; extend as needed
    _libor_ccys = ("USD", "GBP", "JPY", "CHF", "EUR", "CAD", "AUD")
    _libor_tenors = ("1W", "1M", "3M", "6M", "12M")
    for c in _libor_ccys:
        for t in _libor_tenors:
            register_index(f"{c}_LIBOR_{t}", _libor_builder(c, t))
        # helpful alternates like LIBOR_USD_3M or USDLIBOR3M
        register_index(f"LIBOR_{c}", _libor_builder(c, "3M"))

    # # TIIE defaults to 28 days; also register common 91D
    register_index("TIIE_28D", _tiie_registry_builder("28D"))
    register_index("TIIE_28",  _tiie_registry_builder("28D"))
    register_index("TIIE_91D", _tiie_registry_builder("91D"))
    register_index("TIIE_91",  _tiie_registry_builder("91D"))
    register_index("TIIE",     _tiie_registry_builder("28D"))


_bootstrap_registry()


# ----------------------------- Factory API ----------------------------- #
def add_historical_fixings(calculation_date: ql.Date, ibor_index: ql.IborIndex):
    from src.data_interface import APIDataNode
    import datetime
    from src.utils import to_py_date,to_ql_date

    print("Fetching and adding historical fixings...")

    end_date = to_py_date(calculation_date) - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=365)

    historical_fixings = APIDataNode.get_historical_fixings(
        index_name=ibor_index.name(),
        start_date=start_date,
        end_date=end_date
    )

    if not historical_fixings:
        print("No historical fixings found in the given date range.")
        return

    # --- NEW: keep only valid fixing dates for THIS index (Mexico calendar) and strictly in the past
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

    # --- NEW: allow overwriting if some dates already have a fixing
    ibor_index.addFixings(valid_qld, valid_rates, True)
    print(f"Successfully added {len(valid_qld)} fixings for {ibor_index.name()}.")
def build_tiie_zero_curve_from_valmer(_: ql.Date | None = None) -> ql.YieldTermStructureHandle:
    from src.data_interface import APIDataNode
    from src.utils import to_ql_date
    import datetime

    market = APIDataNode.get_historical_data("tiie_zero_valmer", {"MXN": {}})
    nodes  = market["curve_nodes"]
    base_py = market["base_date"]           # Python date from CSV
    base = to_ql_date(base_py)

    cal = ql.Mexico() if hasattr(ql, "Mexico") else ql.TARGET()
    dc  = ql.Actual360()

    # Reference date is the CSV base_date; discount( base_date ) := 1.0
    dates = [base]
    discounts = [1.0]
    seen = {base.serialNumber()}

    for n in sorted(nodes, key=lambda n: int(n["days_to_maturity"])):
        days = int(n["days_to_maturity"])
        if days < 0:
            continue
        d=base_py+datetime.timedelta(days=days)
        d=to_ql_date(d)

        sn = d.serialNumber()
        if sn in seen:
            continue

        z = n.get("zero", n.get("zero_rate", n.get("rate")))
        z = float(z)
        if z > 1.0:  # CSV percent -> decimal
            z *= 0.01

        T = dc.yearFraction(base, d)
        df = 1.0 / (1.0 + z * T)   # Valmer zero is simple ACT/360

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
    tenor = _normalize_tenor(tenor or "28D")
    if tenor in {"ON", "SN", "TN"}:
        raise ValueError("TIIE does not support overnight/SN/TN tenors; use a fixed-day tenor like '28D' or '91D'.")

    cal = ql.Mexico() if hasattr(ql, "Mexico") else ql.TARGET()
    try:
        ccy = ql.MXNCurrency()
    except Exception:
        ccy = ql.USDCurrency()  # label only; doesn’t affect pricing math

    return ql.IborIndex(
        f"TIIE-{tenor}",
        ql.Period(tenor),
        settlement_days,
        ccy,
        cal,
        ql.ModifiedFollowing,
        False,                # end-of-month
        ql.Actual360(),
        curve
    )

def make_tiie_28d_index(
    curve: ql.YieldTermStructureHandle,
    settlement_days: int = 1,
) -> ql.IborIndex:
    # Back-compat shim
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

def get_index(
    name: str,
    *,
    tenor: Optional[str] = None,
    forwarding_curve: Optional[ql.YieldTermStructureHandle] = None,
        calculation_date: Optional[ql.Date] = None,
        hydrate_fixings: bool = False,
        settlement_days: Optional[int] = None
) -> ql.Index:
    """
    Return a QuantLib index instance based on a human-friendly name.

    Parameters
    ----------
    name : str
        Examples: "TIIE_28", "EURIBOR", "EURIBOR_6M", "SOFOR", "USD_LIBOR_3M".
        Delimiters (space, hyphen, colon, underscore) are ignored.
    tenor : Optional[str]
        Optional tenor override like "3M", "6M", "28D".
        If 'name' already contains a tenor, it wins unless you explicitly set this.
    forwarding_curve : Optional[ql.YieldTermStructureHandle]
        Forwarding curve handle. Defaults to an empty handle.

    Returns
    -------
    ql.Index
        A configured QuantLib Index (IborIndex or OvernightIndex subclass).

    Raises
    ------
    KeyError, ValueError
        If no suitable builder is registered/available or the tenor is invalid.
    """
    key0 = _normalize_name(name)
    key = _apply_alias(key0)
    family, rest = _split_tokens(key)

    curve = forwarding_curve or ql.YieldTermStructureHandle()

    # If the exact key is registered, use it.
    if key in _REGISTRY:
        return _REGISTRY[key](curve)

    # Heuristics for common patterns (EURIBOR_<tenor>, <CCY>_LIBOR_<tenor>, LIBOR_<CCY>[_<tenor>], TIIE_<tenor>)
    # 1) EURIBOR
    if family == "EURIBOR":
        t = _normalize_tenor(tenor) if tenor else _find_tenor_in_tokens(rest) or "3M"
        return _euribor_builder(t)(curve)

    # 2) TIIE
    if family in {"TIIE", "MXN", "MXN_TIIE"} or "TIIE" in rest:
        t = _normalize_tenor(tenor) if tenor else _find_tenor_in_tokens((family,) + rest) or "28D"

        # Use provided curve if present; otherwise cached Valmer curve
        use_curve = forwarding_curve
        try:
            if use_curve is None or use_curve.empty():
                use_curve = _default_tiie_curve()
        except Exception:
            use_curve = _default_tiie_curve()

        idx = make_tiie_index(use_curve, t, settlement_days or 1)

        # Optional fixings hydration via this module (keeps everything index-related here)
        if hydrate_fixings and calculation_date is not None and isinstance(idx, ql.IborIndex):
            add_historical_fixings(calculation_date, idx)

        return idx

    # 3) <CCY>_LIBOR_<tenor> or LIBOR_<CCY>[_<tenor>]
    if family == "LIBOR" and rest:
        ccy = rest[0]
        t = _normalize_tenor(tenor) if tenor else _find_tenor_in_tokens(rest[1:]) or "3M"
        return _libor_builder(ccy, t)(curve)
    if family in {"USD", "GBP", "JPY", "CHF", "EUR", "CAD", "AUD"} and (len(rest) and rest[0] == "LIBOR"):
        ccy = family
        t = _normalize_tenor(tenor) if tenor else _find_tenor_in_tokens(rest[1:]) or "3M"
        return _libor_builder(ccy, t)(curve)

    # 4) Overnight simple classes by bare name (SOFR, SONIA, EONIA, ESTR, TONAR, SARON)
    if family in {"SOFR", "SONIA", "EONIA", "ESTR", "TONAR", "FEDFUNDS", "SARON"}:
        return _overnight_simple_builder(family.title())(curve)

    # 5) Last-chance: if user registered a builder under just the family name (e.g., "MYINDEX")
    if family in _REGISTRY:
        return _REGISTRY[family](curve)

    # Nothing matched
    suggestions = ", ".join(
        s for s in list_registered()
        if s.startswith(family) or family in s
    ) or "no close matches"
    raise KeyError(
        f"Unrecognized index name: {name!r} (normalized: {key0!r}). "
        f"Close matches/registered: {suggestions}."
    )


# ----------------------------- Convenience alias ----------------------------- #

# A short alias if you prefer
index_by_name = get_index
