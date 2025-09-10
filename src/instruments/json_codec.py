# src/instruments/json_codec.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union
import QuantLib as ql
from src.pricing_models.indices import get_index as _index_by_name
import hashlib
# ----------------------------- ql.Period -------------------------------------

_UNITS_TO_SHORT = {
    ql.Days: "D",
    ql.Weeks: "W",
    ql.Months: "M",
    ql.Years: "Y",
}

def period_to_json(p: Optional[Union[str, ql.Period]]) -> Optional[str]:
    """
    Encode a QuantLib Period as a compact string like '28D', '3M', '6M', '2Y'.
    Accepts strings and passes them through (idempotent).
    """
    if p is None:
        return None
    if isinstance(p, ql.Period):
        return f"{p.length()}{_UNITS_TO_SHORT[p.units()]}"
    return str(p)

def period_from_json(v: Optional[Union[str, ql.Period]]) -> Optional[ql.Period]:
    """Decode strings like '28D', '3M' into ql.Period; pass ql.Period through."""
    if v is None or isinstance(v, ql.Period):
        return v
    return ql.Period(str(v))


# ----------------------------- ql.DayCounter ---------------------------------

# Prefer explicit enumerations you actually use in this codebase.
_DAYCOUNT_FACTORIES = {
    "Actual360": lambda: ql.Actual360(),
    "Actual365Fixed": lambda: ql.Actual365Fixed(),
    # Default to USA when we can't introspect Thirty360 convention via SWIG.
    "Thirty360": lambda: ql.Thirty360(ql.Thirty360.USA),
    "Thirty360_USA": lambda: ql.Thirty360(ql.Thirty360.USA),
    "Thirty360_BondBasis": lambda: ql.Thirty360(ql.Thirty360.BondBasis),
    "Thirty360_European": lambda: ql.Thirty360(ql.Thirty360.European),
    "Thirty360_ISMA": lambda: ql.Thirty360(ql.Thirty360.ISMA),
    "Thirty360_ISDA": lambda: ql.Thirty360(ql.Thirty360.ISDA),
}

def daycount_to_json(dc: ql.DayCounter) -> str:
    """Encode common DayCounters to a stable string token."""
    if isinstance(dc, ql.Actual360):
        return "Actual360"
    if isinstance(dc, ql.Actual365Fixed):
        return "Actual365Fixed"
    if isinstance(dc, ql.Thirty360):
        # SWIG doesn't expose convention reliably; default to USA in JSON.
        return "Thirty360_USA"
    # Fallback to class name (caller should ensure a known value)
    return dc.__class__.__name__

def daycount_from_json(v: Union[str, ql.DayCounter]) -> ql.DayCounter:
    """Decode from a string token back to a DayCounter instance."""
    if isinstance(v, ql.DayCounter):
        return v
    key = str(v)
    factory = _DAYCOUNT_FACTORIES.get(key)
    if not factory and key == "Thirty360":
        factory = _DAYCOUNT_FACTORIES["Thirty360"]
    if not factory:
        raise ValueError(f"Unsupported day_count '{key}'")
    return factory()


# ----------------------------- ql.Calendar -----------------------------------

# Keep a small, explicit map of calendars you use; easy to extend later.
_CALENDAR_CTORS = {
    "TARGET": ql.TARGET,
    "NullCalendar": ql.NullCalendar,
    "UnitedStates": ql.UnitedStates,   # needs .Market enum
    "Mexico": ql.Mexico,               # needs .Market enum
}

def calendar_to_json(cal: ql.Calendar) -> Dict[str, Any]:
    """
    Encode a Calendar as {"name": "TARGET"} or {"name": "UnitedStates", "market": 0}
    (market is an int of the Enum value).
    """
    name = cal.__class__.__name__
    out: Dict[str, Any] = {"name": name}
    # Some calendars expose a .market() accessor in SWIG; if so, include it.
    try:
        out["market"] = int(cal.market())
    except Exception:
        pass
    return out

def calendar_from_json(v: Union[Dict[str, Any], str, ql.Calendar]) -> ql.Calendar:
    """Decode dict or string into a Calendar instance."""
    if isinstance(v, ql.Calendar):
        return v
    if isinstance(v, str):
        ctor = _CALENDAR_CTORS.get(v)
        if not ctor:
            raise ValueError(f"Unsupported calendar '{v}'")
        return ctor()
    if isinstance(v, dict):
        name = v.get("name", "TARGET")
        ctor = _CALENDAR_CTORS.get(name)
        if not ctor:
            raise ValueError(f"Unsupported calendar '{name}'")
        market = v.get("market", None)
        if market is None:
            return ctor()
        # e.g., ql.UnitedStates(ql.UnitedStates.Market(market))
        enum_cls = getattr(ql, name)
        return ctor(enum_cls.Market(int(market)))
    raise TypeError(f"Cannot decode calendar from {type(v).__name__}")


# ----------------------------- ql.IborIndex ----------------------------------

def ibor_to_json(idx: Optional[ql.IborIndex]) -> Optional[Dict[str, Any]]:
    """
    Encode an IborIndex without trying to serialize the curve handle:
    {"family": "USDLibor", "tenor": "3M"}  or {"family":"Euribor","tenor":"6M"}.
    """
    if idx is None:
        return None
    name_upper = idx.name().upper()
    if "TIIE" in name_upper or "MXNTIIE" in name_upper:
        return {"family": "TIIE-28D", "tenor": "28D"}

    family = getattr(idx, "familyName", lambda: None)() or idx.name()
    try:
        ten = period_to_json(idx.tenor())
    except Exception:
        ten = None
    out = {"family": str(family)}
    if ten:
        out["tenor"] = ten
    return out

def _construct_ibor(family: str, tenor: str) -> ql.IborIndex:
    p = ql.Period(tenor)
    # Common families—extend as needed
    if hasattr(ql, "USDLibor") and family == "USDLibor":
        return ql.USDLibor(p, ql.YieldTermStructureHandle())
    if hasattr(ql, "Euribor") and family == "Euribor":
        return ql.Euribor(p, ql.YieldTermStructureHandle())
    # Generic fallback if QuantLib exposes the family by name
    ctor = getattr(ql, family, None)
    if ctor:
        try:
            return ctor(p, ql.YieldTermStructureHandle())
        except TypeError:
            return ctor(p)
    # TIIE is not a built-in IborIndex; TIIE swaps build their own index later.
    raise ValueError(f"Unsupported Ibor index family '{family}'")

def ibor_from_json(v: Union[None, str, Dict[str, Any], ql.IborIndex]) -> Optional[ql.IborIndex]:
    """
    Decode from JSON into a ql.IborIndex, delegating to the central factory when possible.
    Falls back to legacy parsing for 'USDLibor3M' / 'Euribor6M' styles.
    NOTE: TIIE for swaps remains handled in TIIESwap; this function does not change that flow.
    """
    if v is None or isinstance(v, ql.IborIndex):
        return v

    # 1) String form: try the factory first (supports: 'EURIBOR_6M', 'USD_LIBOR_3M', 'SOFOR'→'SOFR', etc.)
    if isinstance(v, str):
        if _index_by_name is not None:
            try:
                idx = _index_by_name(v)
                # For instruments here we expect an IborIndex; ignore overnight-only results.
                if isinstance(idx, ql.IborIndex):
                    return idx
            except Exception:
                pass  # fall back to legacy parser
        # Legacy fallback: 'USDLibor3M' / 'Euribor6M' / 'USDLibor' (defaults 3M)
        name = v
        tenor = "3M"
        for t in ("1M", "3M", "6M", "12M", "1Y", "28D"):
            if name.endswith(t):
                tenor = t
                family = name[:-len(t)]
                break
        else:
            family = name
        return _construct_ibor(family, tenor)

    # 2) Dict form: try the factory if we have family/tenor; else fallback
    if isinstance(v, dict):
        family = v.get("family") or v.get("name")
        tenor = v.get("tenor", "3M")
        if not family:
            return None
        if _index_by_name is not None:
            try:
                # Accept either {'family':'Euribor','tenor':'6M'} or {'name':'USD_LIBOR','tenor':'3M'}
                candidate = f"{family}_{tenor}" if tenor else family
                idx = _index_by_name(candidate)
                if isinstance(idx, ql.IborIndex):
                    return idx
            except Exception as e:
                raise e
        return _construct_ibor(family, tenor)

    raise TypeError(f"Cannot decode IborIndex from {type(v).__name__}")


# ----------------------------- Generic mixin ---------------------------------

class JSONMixin:
    """
    Mixin to give Pydantic models convenient JSON round-trip helpers.
    Uses Pydantic's JSON mode (so field_serializers are honored).
    """
    def to_json_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, **json_kwargs: Any) -> str:
        return json.dumps(self.to_json_dict(), default=str, **json_kwargs)

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]):
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, payload: Union[str, bytes, Dict[str, Any]]):  # <-- broadened
        if isinstance(payload, dict):
            return cls.from_json_dict(payload)
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        return cls.from_json_dict(json.loads(payload))


    def to_canonical_json(self) -> str:
        """
        Canonical JSON used for hashing:
        - keys sorted
        - no extra whitespace
        - UTF-8 friendly (no ASCII escaping)
        """
        data = self.to_json_dict()
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def content_hash(self, algorithm: str = "sha256") -> str:
        """
        Hash of the canonical JSON representation.
        `algorithm` must be a hashlib-supported name (e.g., 'sha256', 'sha1', 'md5', 'blake2b').
        """
        s = self.to_canonical_json().encode("utf-8")
        h = hashlib.new(algorithm)
        h.update(s)
        return h.hexdigest()

    @classmethod
    def hash_payload(cls, payload: Union[str, bytes, Dict[str, Any]], algorithm: str = "sha256") -> str:
        """
        Hash an arbitrary JSON payload (str/bytes/dict) using the same canonicalization.
        Useful if you have serialized JSON already and want the same digest.
        """
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            obj = json.loads(payload)
        elif isinstance(payload, dict):
            obj = payload
        else:
            raise TypeError(f"Unsupported payload type: {type(payload).__name__}")
        s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        h = hashlib.new(algorithm)
        h.update(s)
        return h.hexdigest()