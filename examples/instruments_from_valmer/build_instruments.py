# -*- coding: utf-8 -*-
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import QuantLib as ql
from src.pricing_models.indices import build_tiie_zero_curve_from_valmer,get_index,make_tiie_index
from src.instruments.floating_rate_bond import FloatingRateBond
import re


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
    cupon_actual_pct = float(row["CUPON ACTUAL"]) if not pd.isna(row["CUPON ACTUAL"]) else np.nan
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

    _ = frb.analytics()  # force construction now

    if not np.isnan(cupon_actual_pct):
        current_coupon_rate = cupon_actual_pct / 100.0  # decimal
        ql_bond = frb.bond  # underlying ql.FloatingRateBond
        eval_qld = qld(eval_date)

        for cf in ql_bond.cashflows():
            cpn = ql.as_floating_rate_coupon(cf)
            if cpn is None:
                continue
            if cpn.accrualStartDate() <= eval_qld and eval_qld < cpn.accrualEndDate():
                fixing_dt = cpn.fixingDate()
                # index fixing = running coupon - spread
                tiie_fixing = max(0.0, current_coupon_rate - spread_decimal)
                tiie_index.addFixing(fixing_dt, tiie_fixing, True)  # overwrite if exists
                break
        _ = frb.analytics()


    return  BuiltBond(
        row_ix=int(row.name) if row.name is not None else -1,
        emisora=emisora,
        serie=serie,
        bond=frb.bond,                 # ql.FloatingRateBond for pricing in run_price_check
        eval_date=eval_date,
    )

# ---------------------------------------------------------------------------
# Cashflow extraction (future only) — similar shape to your model's get_cashflows
# ---------------------------------------------------------------------------
def extract_future_cashflows(built: BuiltBond) -> Dict[str, List[Dict[str, Any]]]:
    ql.Settings.instance().evaluationDate = qld(built.eval_date)

    out: Dict[str, List[Dict[str, Any]]] = {"floating": [], "redemption": []}
    for cf in built.bond.cashflows():
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

    # Use each row's FECHA as valuation; we’ll also build (or reuse) a curve for that date
    # If you have a proper Valmer curve builder, use it per date; else one flat per date.
    curve_cache: Dict[dt.date, ql.YieldTermStructure] = {}

    for ix, row in TIIE_BONDS.iterrows():
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

        # Model analytics
        model_dirty = float(built.bond.dirtyPrice())      # per 100 nominal
        model_clean = float(built.bond.cleanPrice())      # per 100 nominal
        model_accr = float(built.bond.accruedAmount())    # currency units
        face = float(row["VALOR NOMINAL"])
        # accrued per 100 to compare against sheet accrued (if needed)
        model_accr_per100 = 100.0 * (model_accr / face)

        # Market sheet dirty/clean (per 100)
        mkt_dirty = float(row["PRECIO SUCIO"])
        mkt_clean = float(row["PRECIO LIMPIO"])

        # Running coupon sanity (compare to CUPON ACTUAL)
        running_coupon_model = np.nan
        for cf in built.bond.cashflows():
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
        price_diff_bp = 100.0 * (model_dirty - mkt_dirty)  # "price bps" since price is per 100
        coupon_diff_bp = (running_coupon_model - float(row["CUPON ACTUAL"])) * 100.0 if not np.isnan(running_coupon_model) else np.nan
        pass_price = abs(price_diff_bp) <= price_tol_bp
        pass_cpn_count = (np.isnan(expected_count) or (future_cpn_count == expected_count))

        results.append({
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

    return pd.DataFrame(results)

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
    df = pd.read_excel("/home/jose/Downloads/OneDrive_1_9-10-2025/VectorAnalitico24h_2024-09-03.xls")

    fixed_income = df[df["CUPON ACTUAL"] != 0.0]
    floating_tiie = df[df["SUBYACENTE"] == "TIIE28"]

    # If you want to enforce that SOBRETASA is already decimal (e.g., 0.00213229), set to False
    SPREAD_IS_PERCENT = True

    df_out = run_price_check(floating_tiie, SPREAD_IS_PERCENT=SPREAD_IS_PERCENT)
    pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
    print(df_out)

    # Optional: build your Pydantic model instances (if importable) for persistence/JSON
    # If you *do* have a Valmer curve builder, you probably want to call it here for the eval date.
    eval_date = parse_val_date(floating_tiie.loc[0, "FECHA"])
    curve_for_models = build_tiie_zero_curve_from_valmer(qld(eval_date))

    _instances = try_build_pydantic_model_instances(floating_tiie, curve_for_models)
    if _instances:
        print(f"\nBuilt {len(_instances)} FloatingRateBond model instance(s).")
        # Example: show analytics of the first one
        try:
            print("Model[0] analytics:", _instances[0].analytics())
            print("Next 3 cashflows:", _instances[0].get_cashflows()["floating"][:3])
        except Exception as e:
            print("Analytics on model instance failed (likely helper funcs missing):", e)