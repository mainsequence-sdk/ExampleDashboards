from __future__ import annotations

from typing import Optional, Dict, Any
from typing_extensions import Annotated
from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema
import QuantLib as ql

# Reuse your existing codec helpers
from src.instruments.json_codec import (
    # Period
    period_to_json, period_from_json,
    # DayCounter
    daycount_to_json, daycount_from_json,
    # Calendar
    calendar_to_json, calendar_from_json,
    # IborIndex
    ibor_to_json, ibor_from_json,
    # Schedule
    schedule_to_json, schedule_from_json,
)

# ---------- Period -----------------------------------------------------------
# ql.Period <-> "28D" | "3M" | "2Y"
QuantLibPeriod = Annotated[
    ql.Period,
    BeforeValidator(period_from_json),
    PlainSerializer(period_to_json, return_type=str),
]

# ---------- DayCounter -------------------------------------------------------
# ql.DayCounter <-> "Actual360" | "Actual365Fixed" | "Thirty360_USA"
QuantLibDayCounter = Annotated[
    ql.DayCounter,
    BeforeValidator(daycount_from_json),
    PlainSerializer(daycount_to_json, return_type=str),
]

# ---------- Calendar ---------------------------------------------------------
# ql.Calendar <-> {"name":"Mexico","market":0} | {"name":"TARGET"}
QuantLibCalendar = Annotated[
    ql.Calendar,
    BeforeValidator(calendar_from_json),
    PlainSerializer(calendar_to_json, return_type=Dict[str, Any]),
]

# ---------- IborIndex --------------------------------------------------------
# ql.IborIndex <-> {"family":"TIIE-28D","tenor":"28D"} | {"family":"Euribor","tenor":"6M"} | "USD_LIBOR_3M"
QuantLibIborIndex = Annotated[
    Optional[ql.IborIndex],
    BeforeValidator(ibor_from_json),
    PlainSerializer(ibor_to_json, return_type=Optional[Dict[str, Any]]),
]

# ---------- Schedule ---------------------------------------------------------
# ql.Schedule <-> {"dates":[...], "calendar":{...}, "business_day_convention":"Following", ...}
QuantLibSchedule = Annotated[
    Optional[ql.Schedule],
    BeforeValidator(schedule_from_json),
    PlainSerializer(schedule_to_json, return_type=Optional[Dict[str, Any]]),
    # Optional: give API users a stable JSON schema hint
    WithJsonSchema(
        {
            "type": ["object", "null"],
            "properties": {
                "dates": {"type": "array", "items": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}},
                "calendar": {"type": "object"},
                "business_day_convention": {"type": ["string", "integer"]},
                "termination_business_day_convention": {"type": ["string", "integer"]},
                "end_of_month": {"type": "boolean"},
                "tenor": {"type": "string"},
                "rule": {"type": ["string", "integer"]},
            },
            "required": ["dates"],
            "additionalProperties": True,
        },
        mode="serialization",
    ),
]
