# -*- coding: utf-8 -*-
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import QuantLib as ql

from src.instruments import PositionLine, Position
from src.pricing_models.indices import build_tiie_zero_curve_from_valmer,get_index,make_tiie_index
from src.instruments.floating_rate_bond import FloatingRateBond
import re
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Minimal floating floater construction with discounting
# ---------------------------------------------------------------------------
@dataclass
class BuiltBond:
    row_ix: int
    emisora: str
    serie: str
    bond: FloatingRateBond
    eval_date: dt.date


# ---------------------------------------------------------------------------
# Optional: if your project exposes these, we'll use them; else we fall back.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_val_date(v) -> dt.date:
    """Handle integer like 20240903 or string '2024-09-03'."""
    if pd.isna(v):
        raise ValueError("FECHA is required")
    s = str(int(v)) if isinstance(v, (int, np.integer)) else str(v)
    try:
        # Try yyyymmdd (e.g., 20240903)
        if len(s) == 8 and s.isdigit():
            return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        # Fall back to pandas parsing
        return pd.to_datetime(s).date()
    except Exception:
        return pd.to_datetime(v).date()

def parse_iso_date(v) -> dt.date:
    if pd.isna(v):
        raise ValueError("Missing date")
    return pd.to_datetime(v).date()

def qld(d: dt.date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)

def pyd(q: ql.Date) -> dt.date:
    return dt.date(q.year(), int(q.month()), q.dayOfMonth())

# ---------------------------------------------------------------------------
# Minimal floating floater construction with discounting
# ---------------------------------------------------------------------------

def _parse_coupon_period(freq_val, default_days: int = 28) -> ql.Period:
    """
    Parse 'FREC. CPN' strings like '30Dias', '184Dias', '31dias', '32 Días', etc.
    Falls back to default_days if missing or unparsable.
    """
    if pd.isna(freq_val):
        return ql.Period(default_days, ql.Days)
    s = str(freq_val).strip().lower()
    # extract the first integer we see
    m = re.search(r"(\d+)", s)
    if not m:
        return ql.Period(default_days, ql.Days)
    days = int(m.group(1))
    if days <= 0:
        days = default_days
    return ql.Period(days, ql.Days)


def build_qll_floater_from_row_new(
    row: pd.Series,
    curve,  # ql.YieldTermStructure or Handle
    *,
    calendar: ql.Calendar = ql.Mexico(),
    dc: ql.DayCounter = ql.Actual360(),
    bdc: int = ql.Following,
    settlement_days: int = 1,          # MXN standard
    SPREAD_IS_PERCENT: bool = True,
) -> BuiltBond:
    """
    Exact construction: explicit schedule forced from the sheet.
    - Next pay date = FECHA + (FREC - DIAS_TRANSC), adjusted.
    - Remaining coupons = CUPONES X COBRAR exactly (last date = FECHA VCTO).
    - Explicit date list -> ql.Schedule -> ql.FloatingRateBond.
    """

    # ---- helpers ----------------------------------------------------------


    def _adjust(d: dt.date) -> dt.date:
        return pyd(calendar.adjust(qld(d), bdc))

    def _parse_coupon_days(freq_val, default_days: int = 28) -> int:
        if pd.isna(freq_val):
            return default_days
        s = str(freq_val).lower()
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else default_days

    # ---- parse sheet fields ----------------------------------------------
    eval_date      = parse_val_date(row["FECHA"])
    maturity_date  = parse_iso_date(row["FECHA VCTO"])
    face           = float(row["VALOR NOMINAL"])

    freq_days      = _parse_coupon_days(row.get("FREC. CPN"), default_days=28)
    dias_trans     = int(row["DIAS TRANSC. CPN"]) if pd.notna(row.get("DIAS TRANSC. CPN")) else 0
    coupons_left   = int(row["CUPONES X COBRAR"]) if pd.notna(row.get("CUPONES X COBRAR")) else None

    cupon_actual_pct = float(row["CUPON ACTUAL"]) if not pd.isna(row["CUPON ACTUAL"]) else np.nan
    sraw              = 0.0 if pd.isna(row["SOBRETASA"]) else float(row["SOBRETASA"])
    spread_decimal    = (sraw / 100.0) if SPREAD_IS_PERCENT else sraw

    emisora = str(row.get("EMISORA", ""))
    serie   = str(row.get("SERIE", ""))

    # ---- global settings --------------------------------------------------
    ql.Settings.instance().evaluationDate = qld(eval_date)
    ql.Settings.instance().includeReferenceDateEvents = False
    ql.Settings.instance().enforceTodaysHistoricFixings = False

    # ---- index with your forwarding curve (and hydrated fixings) ----------
    tiie_index = make_tiie_index( curve)

    # ---- build explicit schedule dates -----------------------------------
    # previous (accrual start) = FECHA - DIAS_TRANSC, payment dates in business days
    prev_pay_unadj = eval_date - dt.timedelta(days=max(0, dias_trans))
    prev_pay       = _adjust(prev_pay_unadj)  # payment dates are business days

    # next payment, then forward by freq_days
    next_pay_unadj = prev_pay + dt.timedelta(days=freq_days)
    next_pay       = _adjust(next_pay_unadj)

    # use adjusted maturity so it's on the same convention as other dates
    maturity_pay = _adjust(maturity_date)

    # If the sheet gives remaining coupons, enforce that exact count (last = maturity),
    # but don't duplicate maturity if the ladder already hits it.
    pay_dates: List[dt.date] = []
    if coupons_left is not None and coupons_left > 0:
        target = max(0, coupons_left - 1)  # we will append maturity at the end
        d = next_pay
        while len(pay_dates) < target and d < maturity_pay:
            pay_dates.append(d)
            d = _adjust(d + dt.timedelta(days=freq_days))
    else:
        d = next_pay
        while d < maturity_pay:
            pay_dates.append(d)
            d = _adjust(d + dt.timedelta(days=freq_days))

    # always end exactly at maturity once
    if not pay_dates or pay_dates[-1] != maturity_pay:
        pay_dates.append(maturity_pay)

    # Compose the explicit date list expected by QuantLib Schedule:
    all_dates_py = [prev_pay] + pay_dates

    # --- sanitize: strictly increasing / dedupe identical neighbors ---
    _sanitized = []
    for d in all_dates_py:
        if not _sanitized or d > _sanitized[-1]:
            _sanitized.append(d)
        elif d == _sanitized[-1]:
            # drop duplicate (prevents "null accrual period")
            continue
        else:
            # This should never happen; protects against out-of-order dates.
            raise ValueError(f"Non-increasing schedule: {_sanitized[-1]} -> {d}")
    all_dates_py = _sanitized

    # Explicit schedule
    date_vec = ql.DateVector()
    for d in all_dates_py:
        date_vec.push_back(qld(d))
    schedule = ql.Schedule(date_vec, calendar, bdc)

    # ---- build the QuantLib floater with that schedule --------------------
    # fixingDays=2, gearings=[1], spreads=[spread], no caps/floors, redemption=100
    ql_bond = ql.FloatingRateBond(
        settlement_days,
        face,
        schedule,
        tiie_index,
        dc,
        bdc,
        1,
        [1.0],
        [spread_decimal],
        [], [],  # caps, floors
        False,   # inArrears
        100.0,
        qld(all_dates_py[0])  # use schedule start as issueDate to keep accrual consistent
    )

    # Discounting engine on your curve
    ql_bond.setPricingEngine(ql.DiscountingBondEngine(curve))

    # ---- enforce running coupon = 'CUPON ACTUAL' --------------------------
    # if not np.isnan(cupon_actual_pct):
    #     target_rate = cupon_actual_pct / 100.0  # decimal
    #     for cf in ql_bond.cashflows():
    #         cpn = ql.as_floating_rate_coupon(cf)
    #         break
    #     cpn.rate()==target_rate#!!!

    try:
        _ = ql_bond.NPV()  # re-touch pricing after inserting the fixing
    except Exception as e:
        from src.utils import to_py_date
        def debug_bond_cashflows(ql_bond: ql.Bond, curve_handle: ql.YieldTermStructureHandle) -> pd.DataFrame:
            """
            Analyzes a bond's cash flows against a curve to diagnose "negative time" errors.
            This function creates a DataFrame detailing each future cash flow, its fixing
            date, and the time difference relative to the curve's reference date,
            pinpointing any problematic cash flows.

            Args:
                ql_bond: The QuantLib Bond object to analyze.
                curve_handle: The YieldTermStructureHandle used for pricing.

            Returns:
                A pandas DataFrame with detailed diagnostic information for each cash flow.
            """
            curve_ref_date_ql = curve_handle.referenceDate()
            curve_ref_date_py = to_py_date(curve_ref_date_ql)

            # It's good practice to set this explicitly for the check
            ql.Settings.instance().includeReferenceDateEvents = False
            evaluation_date = ql.Settings.instance().evaluationDate

            diagnostic_data = []

            for cf in ql_bond.cashflows():
                # --- THIS IS THE CORRECTED LINE ---
                # The check now uses the correct number of arguments.
                if cf.hasOccurred(evaluation_date):
                    continue

                payment_date = to_py_date(cf.date())
                time_to_payment = (payment_date - curve_ref_date_py).days

                cpn = ql.as_floating_rate_coupon(cf)
                if cpn:
                    fixing_date = to_py_date(cpn.fixingDate())
                    time_to_fixing = (fixing_date - curve_ref_date_py).days
                    status = "ERROR: Fixing Date is before Curve Reference Date" if time_to_fixing < 0 else "OK"

                    diagnostic_data.append({
                        "cash_flow_type": "Floating Coupon",
                        "payment_date": payment_date,
                        "fixing_date": fixing_date,
                        "curve_ref_date": curve_ref_date_py,
                        "time_to_payment_days": time_to_payment,
                        "time_to_fixing_days": time_to_fixing,
                        "status": status,
                    })
                else:
                    diagnostic_data.append({
                        "cash_flow_type": "Redemption",
                        "payment_date": payment_date,
                        "fixing_date": None,
                        "curve_ref_date": curve_ref_date_py,
                        "time_to_payment_days": time_to_payment,
                        "time_to_fixing_days": None,
                        "status": "OK",
                    })

            if not diagnostic_data:
                print("No future cash flows found to analyze.")
                return pd.DataFrame()

            return pd.DataFrame(diagnostic_data)

        forecasting_curve_handle = tiie_index.forwardingTermStructure()

        diagnostic_df = debug_bond_cashflows(ql_bond, forecasting_curve_handle)
        print("Cash Flow Sanity Check:")
        print(diagnostic_df)

        ts = tiie_index.timeSeries()
        dates = [d.to_date() for d in ts.dates()]
        rates = list(ts.values())
        df = pd.DataFrame({'fixing_rate': rates}, index=pd.to_datetime(dates))
        df.index.name = 'date'

        raise e
    # ---- return wrapper used by run_price_check ---------------------------
    return BuiltBond(
        row_ix=int(row.name) if row.name is not None else -1,
        emisora=emisora,
        serie=serie,
        bond=ql_bond,     # underlying ql.FloatingRateBond (your loop uses .dirtyPrice())
        eval_date=eval_date,
    )


def build_qll_floater_from_row(
    row: pd.Series,
    curve,
    *,
    calendar: ql.Calendar = ql.Mexico(),
    dc: ql.DayCounter = ql.Actual360(),
    bdc: int = ql.Following,
    settlement_days: int = 1,
    SPREAD_IS_PERCENT: bool = True,
) -> FloatingRateBond:
    """Create a QuantLib FloatingRateBond for a TIIE-28 floater row."""
    # --- read inputs (Spanish columns) ---
    eval_date = parse_val_date(row["FECHA"])
    issue_date = parse_iso_date(row["FECHA EMISION"])
    maturity_date = parse_iso_date(row["FECHA VCTO"])
    face = float(row["VALOR NOMINAL"])
    sobretasa_raw = 0.0 if pd.isna(row["SOBRETASA"]) else float(row["SOBRETASA"])
    spread_decimal = (sobretasa_raw / 100.0) if SPREAD_IS_PERCENT else sobretasa_raw

    emisora = str(row.get("EMISORA", ""))
    serie = str(row.get("SERIE", ""))

    # --- settings ---
    ql.Settings.instance().evaluationDate = qld(eval_date)
    ql.Settings.instance().includeReferenceDateEvents = False
    ql.Settings.instance().enforceTodaysHistoricFixings = False

    # --- coupon frequency from 'FREC. CPN' ---
    coupon_freq = _parse_coupon_period(row.get("FREC. CPN"), default_days=28)

    # --- index and schedule ---
    tiie_index = make_tiie_index(
        curve=curve,

    )

    # --- floater with constant spread over the index ---

    frb = FloatingRateBond(
        face_value=face,
        floating_rate_index=tiie_index,
        spread=spread_decimal,
        issue_date=issue_date,
        maturity_date=maturity_date,
        coupon_frequency=coupon_freq,
        day_count=dc,
        calendar=calendar,
        business_day_convention=bdc,
        settlement_days=settlement_days,
        valuation_date=eval_date,
    )

    analytics = frb.analytics()  # force construction now



    return  BuiltBond(
        row_ix=int(row.name) if row.name is not None else -1,
        emisora=emisora,
        serie=serie,
        bond=frb,                 # ql.FloatingRateBond for pricing in run_price_check
        eval_date=eval_date,
    )

# ---------------------------------------------------------------------------
# Cashflow extraction (future only) — similar shape to your model's get_cashflows
# ---------------------------------------------------------------------------
def extract_future_cashflows(built: BuiltBond) -> Dict[str, List[Dict[str, Any]]]:
    ql_bond=built.bond.bond
    ql.Settings.instance().evaluationDate = qld(built.eval_date)

    out: Dict[str, List[Dict[str, Any]]] = {"floating": [], "redemption": []}
    for cf in ql_bond.cashflows():
        if cf.hasOccurred():
            continue
        cpn = ql.as_floating_rate_coupon(cf)
        if cpn is not None:
            out["floating"].append({
                "payment_date": pyd(cpn.date()),
                "fixing_date": pyd(cpn.fixingDate()),
                "rate": float(cpn.rate()),
                "spread": float(cpn.spread()),
                "amount": float(cpn.amount()),
            })
        else:
            out["redemption"].append({
                "payment_date": pyd(cf.date()),
                "amount": float(cf.amount()),
            })
    return out




# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run_price_check(
    TIIE_BONDS: pd.DataFrame,
    *,
    SPREAD_IS_PERCENT: bool = True,
    price_tol_bp: float = 2.0,   # allowable dirty-price diff in basis points of price (per 100)
    coupon_tol_bp: float = 1.0,  # allowable running coupon diff in bp (annualized)

) -> pd.DataFrame:
    """
    Loop the DF, build each TIIE-28 floater, compare dirty price vs 'PRECIO SUCIO'
    and the future coupon count vs 'CUPONES X COBRAR'. Returns a result DataFrame.
    """
    results = []
    instrument_map = {}
    # Use each row's FECHA as valuation; we’ll also build (or reuse) a curve for that date
    # If you have a proper Valmer curve builder, use it per date; else one flat per date.
    curve_cache: Dict[dt.date, ql.YieldTermStructure] = {}

    for ix, row in tqdm(TIIE_BONDS.iterrows(),desc="bulding indstruments"):
        eval_date = parse_val_date(row["FECHA"])

        if eval_date not in curve_cache:

            curve = build_tiie_zero_curve_from_valmer(qld(eval_date))  # expected to return ql.YieldTermStructure

            curve_cache[eval_date] = curve
        else:
            curve = curve_cache[eval_date]

        # Build bond
        built = build_qll_floater_from_row(
            row, curve, SPREAD_IS_PERCENT=SPREAD_IS_PERCENT
        )
        ql_bond=built.bond.bond
        # Model analytics
        face = float(row["VALOR NOMINAL ACTUALIZADO"])
        model_dirty = float(ql_bond.dirtyPrice())  *face/100    # per 100 nominal
        model_clean = float(ql_bond.cleanPrice())*face/100      # per 100 nominal
        model_accr = model_dirty-model_clean    # currency units

        # accrued per 100 to compare against sheet accrued (if needed)
        model_accr_per100 = 100.0 * (model_accr / face)

        # Market sheet dirty/clean (per 100)
        mkt_dirty = float(row["PRECIO SUCIO"])
        mkt_clean = float(row["PRECIO LIMPIO"])
        if mkt_dirty ==0:
            continue

        # Running coupon sanity (compare to CUPON ACTUAL)
        running_coupon_model = np.nan
        for cf in ql_bond.cashflows():
            cpn = ql.as_floating_rate_coupon(cf)
            if cpn is None:
                continue
            if (cpn.accrualStartDate() <= qld(eval_date)) and (qld(eval_date) < cpn.accrualEndDate()):
                running_coupon_model = 100.0 * float(cpn.rate())
                break

        # Future cashflows
        cfs = extract_future_cashflows(built)
        future_cpn_count = len(cfs["floating"])
        expected_count = int(row["CUPONES X COBRAR"]) if not pd.isna(row["CUPONES X COBRAR"]) else np.nan

        # Diffs
        price_diff_bp = 100.0 * (model_dirty - mkt_dirty)/mkt_dirty  # "price bps" since price is per 100
        coupon_diff_bp = (running_coupon_model - float(row["CUPON ACTUAL"])) * 100.0 if not np.isnan(running_coupon_model) else np.nan
        pass_price = abs(price_diff_bp) <= price_tol_bp
        pass_cpn_count = (np.isnan(expected_count) or (future_cpn_count == expected_count))

        results.append({
            "instrument_hash" :built.bond.content_hash(),
            "row": int(ix),
            "EMISORA": row.get("EMISORA", ""),
            "SERIE": row.get("SERIE", ""),
            "FECHA": eval_date,
            "VALOR NOMINAL": face,
            "SOBRETASA_in": float(row["SOBRETASA"]),
            "SOBRETASA_decimal": (float(row["SOBRETASA"]) / 100.0) if SPREAD_IS_PERCENT and not pd.isna(row["SOBRETASA"]) else float(row["SOBRETASA"] or 0.0),
            "CUPON ACTUAL (sheet) %": float(row["CUPON ACTUAL"]),
            "CUPON ACTUAL (model) %": running_coupon_model,
            "coupon_diff_bp": coupon_diff_bp,
            "PRECIO SUCIO (sheet)": mkt_dirty,
            "PRECIO SUCIO (model)": model_dirty,
            "price_diff_bp": price_diff_bp,
            "PRECIO LIMPIO (sheet)": mkt_clean,
            "PRECIO LIMPIO (model)": model_clean,
            "accrued_per_100 (model)": model_accr_per100,
            "CUPONES X COBRAR (sheet)": expected_count,
            "CUPONES FUTUROS (model)": future_cpn_count,
            "pass_price": pass_price,
            "pass_coupon_count": pass_cpn_count,

        })

        instrument_map[built.bond.content_hash()]=built.bond

    return pd.DataFrame(results),instrument_map

# ---------------------------------------------------------------------------
# OPTIONAL: build your Pydantic FloatingRateBond for persistence/JSON.
# This block runs only if your model class is importable.
# ---------------------------------------------------------------------------
def try_build_pydantic_model_instances(TIIE_BONDS: pd.DataFrame, curve: ql.YieldTermStructure) -> List[Any]:
    try:
        # Import your model from your codebase
        from your_project.models.floating_rate_bond import FloatingRateBond as FRBModel  # <-- change import
    except Exception:
        return []  # silently skip if not available

    instances = []
    for _, row in TIIE_BONDS.iterrows():
        eval_date = parse_val_date(row["FECHA"])
        issue_date = parse_iso_date(row["FECHA EMISION"])
        maturity_date = parse_iso_date(row["FECHA VCTO"])
        face = float(row["VALOR NOMINAL"])
        sraw = 0.0 if pd.isna(row["SOBRETASA"]) else float(row["SOBRETASA"])
        spread_decimal = sraw / 100.0  # flip to False if your file is already decimal

        ql.Settings.instance().evaluationDate = qld(eval_date)
        tiie_index = make_tiie_28d_index(curve)

        mdl = FRBModel(
            face_value=face,
            floating_rate_index=tiie_index,
            spread=spread_decimal,
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_frequency=ql.Period(28, ql.Days),
            day_count=ql.Actual360(),
            calendar=ql.Mexico(),
            business_day_convention=ql.Following,
            settlement_days=2,
            valuation_date=eval_date,
        )

        # If your model supports injecting a curve, do it so pricing works immediately
        try:
            mdl.reset_curve(curve)  # calls your _build_bond() path with the curve
        except Exception:
            pass

        instances.append(mdl)
    return instances
if __name__ == "__main__":
    # Minimal sample DF — replace with your real TIIE_BONDS DataFrame.
    df = pd.read_excel("/home/jose/Downloads/VectorAnalitico24h_2025-08-27.xls")

    fixed_income = df[df["CUPON ACTUAL"] != 0.0]
    floating_tiie = df[df["SUBYACENTE"] == "TIIE28"]

    # If you want to enforce that SOBRETASA is already decimal (e.g., 0.00213229), set to False
    SPREAD_IS_PERCENT = True

    df_out,instrument_map = run_price_check(floating_tiie, SPREAD_IS_PERCENT=SPREAD_IS_PERCENT)
    pd.set_option("display.float_format", lambda x: f"{x:,.6f}")


    #get subset for position
    best_pricing=df_out.sort_values("price_diff_bp",ascending=True).iloc[:100]
    position_instruments={k:v for k,v in instrument_map.items() if k in df_out.instrument_hash.to_list()}

    position_lines=[]
    for k,v in position_instruments.items():
        line=PositionLine(instrument=v,units=int(100_000_000/v.price()))
        position_lines.append(line)
    position=Position(lines=position_lines)
    dump=position.to_json_dict()
    a=5


