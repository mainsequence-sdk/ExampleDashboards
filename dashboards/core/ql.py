# dashboards/core/ql.py
from __future__ import annotations
import datetime as dt
import QuantLib as ql

def qld(d: dt.date | ql.Date) -> ql.Date:
    if isinstance(d, ql.Date):
        return d
    return ql.Date(d.day, d.month, d.year)

def as_handle(ts) -> ql.YieldTermStructureHandle:
    return ts if isinstance(ts, ql.YieldTermStructureHandle) else ql.YieldTermStructureHandle(ts)

def spot_from_index(calc_date: ql.Date, index: ql.IborIndex) -> ql.Date:
    cal = index.fixingCalendar()
    fixing = cal.adjust(calc_date, ql.Following)
    while not index.isValidFixingDate(fixing):
        fixing = cal.advance(fixing, 1, ql.Days)
    return index.valueDate(fixing)
