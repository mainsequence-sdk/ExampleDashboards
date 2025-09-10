# src/utils.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import QuantLib as ql

from src.data_interface import APIDataNode
from src.instruments import PositionLine, Position
from src.pricing_models.indices import build_tiie_zero_curve_from_valmer


# -------------------- existing conversions --------------------
def to_ql_date(d: dt.date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)


def to_py_date(qld: ql.Date) -> dt.date:
    return dt.date(qld.year(), qld.month(), qld.dayOfMonth())


# -------------------- global settings --------------------
def set_eval(d: dt.date) -> ql.Date:
    qld = to_ql_date(d)
    ql.Settings.instance().evaluationDate = qld
    ql.Settings.instance().includeReferenceDateEvents = False
    ql.Settings.instance().enforceTodaysHistoricFixings = False
    return qld


# -------------------- bootstrapping (deposits + swaps) --------------------
def canonical_tenor(t: str) -> str:
    return (t or "").strip().upper()


def load_swap_nodes() -> List[dict]:
    market = APIDataNode.get_historical_data("interest_rate_swaps", {"USD_rates": {}})
    return list(market["curve_nodes"])  # copy


def apply_bumps(nodes: List[dict], bumps_bp_by_tenor: Dict[str, float]) -> List[dict]:
    bump_map = {canonical_tenor(k): v for k, v in bumps_bp_by_tenor.items()}
    bumped = []
    for n in nodes:
        n2 = dict(n)
        t = canonical_tenor(n2.get("tenor", ""))
        if t in bump_map:
            n2["rate"] = float(n2["rate"]) + bump_map[t] / 10_000.0
        bumped.append(n2)
    return bumped


def build_curve_from_swap_nodes(calc_date: ql.Date, nodes: List[dict]) -> ql.YieldTermStructureHandle:
    calendar = ql.TARGET()
    day_counter = ql.Actual365Fixed()

    tmp_handle = ql.YieldTermStructureHandle(ql.FlatForward(calc_date, 0.02, day_counter))
    ibor_index = ql.USDLibor(ql.Period("3M"), tmp_handle)

    swap_fixed_leg_frequency = ql.Annual
    swap_fixed_leg_convention = ql.Unadjusted
    swap_fixed_leg_daycounter = ql.Thirty360(ql.Thirty360.USA)

    helpers: List[ql.RateHelper] = []
    for node in nodes:
        t = canonical_tenor(node.get("tenor", ""))
        rate = float(node["rate"])
        quote = ql.QuoteHandle(ql.SimpleQuote(rate))
        if node["type"].lower() == "deposit":
            helper = ql.DepositRateHelper(quote, ql.Period(t), 2, calendar, ql.ModifiedFollowing, False, day_counter)
        elif node["type"].lower() == "swap":
            helper = ql.SwapRateHelper(quote, ql.Period(t), calendar,
                                       swap_fixed_leg_frequency, swap_fixed_leg_convention,
                                       swap_fixed_leg_daycounter, ibor_index)
        else:
            raise ValueError(f"Unsupported node type: {node['type']}")
        helpers.append(helper)

    curve = ql.PiecewiseLogCubicDiscount(calc_date, helpers, day_counter)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


# -------------------- mapping helpers --------------------
def calendar_from_str(name: str) -> ql.Calendar:
    n = (name or "TARGET").strip().upper()
    if n == "TARGET": return ql.TARGET()
    if n in ("US", "UNITEDSTATES", "UNITED_STATES"): return ql.UnitedStates()
    if n in ("MEXICO", "MX"): return ql.Mexico() if hasattr(ql, "Mexico") else ql.TARGET()
    return ql.TARGET()


def bdc_from_str(name: str) -> int:
    n = (name or "").strip().lower()
    return {
        "following": ql.Following,
        "modifiedfollowing": ql.ModifiedFollowing,
        "preceding": ql.Preceding,
        "unadjusted": ql.Unadjusted,
    }.get(n, ql.ModifiedFollowing)


def dcc_from_str(s: str) -> ql.DayCounter:
    n = (s or "30/360").strip().upper()
    if n.startswith("30/360"): return ql.Thirty360(ql.Thirty360.USA)
    if n in ("ACT/360", "ACTUAL/360"): return ql.Actual360()
    if n in ("ACT/365", "ACT/365F", "ACTUAL/365", "ACTUAL/365F"): return ql.Actual365Fixed()
    if n in ("ACT/ACT", "ACTUAL/ACTUAL"): return ql.ActualActual()
    return ql.Thirty360(ql.Thirty360.USA)


def make_index(index_str: str, curve: ql.YieldTermStructureHandle) -> ql.IborIndex:
    s = (index_str or "").strip().upper()
    if s in ("MXN-TIIE-28D", "MXN-TIIE-28", "TIIE-28D", "TIIE28", "TIIE"):
        return make_tiie_28d_index(curve)
    # Default: parse USD-LIBOR-<tenor>
    ten = "3M"
    if "-" in s:
        last = s.split("-")[-1]
        if last.endswith(("M", "Y")):
            ten = last
    return ql.USDLibor(ql.Period(ten), curve)


# -------------------- cashflow extraction --------------------
@dataclass
class CFRow:
    ins_id: str
    leg: str
    pay_date: dt.date
    kind: str
    amount: float
    df: float
    pv: float
    accrual_start: dt.date | None = None
    accrual_end: dt.date | None = None
    accrual: float | None = None
    rate: float | None = None


def swap_cashflows_df(ins_id: str, swap: ql.VanillaSwap, curve: ql.YieldTermStructureHandle,
                      today: ql.Date, cutoff: dt.date | None) -> pd.DataFrame:
    rows: List[CFRow] = []
    for leg_idx, leg_name in enumerate(["fixed", "float"]):
        for c in swap.leg(leg_idx):
            if c.date() <= today: continue
            pay = to_py_date(c.date())
            if cutoff and pay > cutoff: continue
            amt = float(c.amount())
            df = float(curve.discount(c.date()))
            cup = ql.as_coupon(c)
            if cup is not None:
                rows.append(CFRow(ins_id, leg_name, pay, "coupon", amt, df, amt * df,
                                  to_py_date(cup.accrualStartDate()), to_py_date(cup.accrualEndDate()),
                                  cup.accrualPeriod(), _safe_rate(cup)))
            else:
                rows.append(CFRow(ins_id, leg_name, pay, "other", amt, df, amt * df))
    return pd.DataFrame([r.__dict__ for r in rows])


def _safe_rate(cup: ql.Coupon) -> float | None:
    try:
        return float(cup.rate())
    except Exception:
        return None


# -------------------- zero sampling for curve plots --------------------
def sample_zero_curve(ts: ql.YieldTermStructureHandle, calc_date: ql.Date,
                      max_years: int = 12, step_months: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    cal = ql.TARGET()
    dc = ql.Actual365Fixed()
    comp, freq = ql.Continuous, ql.Annual
    t_list: List[float] = []
    z_list: List[float] = []
    m = 1
    while m <= max_years * 12:
        d = cal.advance(calc_date, ql.Period(m, ql.Months))
        t = dc.yearFraction(calc_date, d)
        z = ts.zeroRate(d, dc, comp, freq).rate()
        t_list.append(t);
        z_list.append(z)
        m += step_months
    return np.array(t_list), np.array(z_list)


# -------------------- portfolio NPV impact --------------------

def get_bumped_position(position: Position, bump_curve):
    bumped_lines = []

    for position_line in position.lines:
        base_instrument = position_line.instrument
        bumped_instrument = base_instrument.copy()
        bumped_instrument.reset_curve(bump_curve)
        bumped_line = PositionLine(units=position_line.units,
                                   instrument=bumped_instrument)
        bumped_lines.append(bumped_line)
    bumped_position = Position(lines=bumped_lines)
    return bumped_position


def price_all(
        positions: Position, bumped_position: Position,
        cutoff: dt.date | None) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, dict]:
    npv0: Dict[str, float] = {}
    npv1: Dict[str, float] = {}
    units: Dict[str, float] = {}
    cf0_list: List[pd.DataFrame] = []
    cf1_list: List[pd.DataFrame] = []

    for position_line, bumped_line in zip(positions.lines, positions.lines):
        base_instrument = position_line.instrument
        bumped_instrument = bumped_line.instrument

        original_hash = base_instrument.content_hash()

        npv0[original_hash] = position_line.units * float(base_instrument.price())
        npv1[original_hash] = position_line.units * float(bumped_instrument.price())

        df0 = (base_instrument.get_net_cashflows() * position_line.units).to_frame("amount")
        df0.loc[:, "ins_id"] = original_hash
        df1 = (bumped_instrument.get_net_cashflows() * position_line.units).to_frame("amount")
        df1.loc[:, "ins_id"] = original_hash

        cf0_list.append(df0.reset_index())
        cf1_list.append(df1.reset_index())

        units[original_hash] = position_line.units
    cf0 = pd.concat(cf0_list, ignore_index=True) if cf0_list else pd.DataFrame()
    cf1 = pd.concat(cf1_list, ignore_index=True) if cf1_list else pd.DataFrame()
    return npv0, npv1, cf0, cf1, units


# ---- Helpers for per-ID metadata ----
def id_meta(positions: Position) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for position_line in positions.lines:
        id = position_line.instrument.content_hash()
        instrument = position_line.instrument
        meta[id] = {
            "units": position_line.units,
            "type": instrument.__class__.__name__,
            # "lcr_inflow_rate": float(b.get("lcr_inflow_rate", 0.0)),
            # "lcr_outflow_rate": float(b.get("lcr_outflow_rate", 0.0)),
            # "nsfr_asf_weight": float(b.get("nsfr_asf_weight", 0.0)),
            # "nsfr_rsf_weight": float(b.get("nsfr_rsf_weight", 0.0)),
            # "kind": b.get("kind","fixed").lower()
        }

    return meta


# ---- LCR (simplified; 30d horizon; inflow cap) ----
def compute_lcr(cf_base: pd.DataFrame,
                meta_by_id: dict[str, dict],
                valuation_date: dt.date,
                hqla_amount: float,
                horizon_days: int = 30,
                inflow_cap: float = 0.75) -> dict:
    if cf_base.empty:
        return {"LCR": np.inf, "outflows_30d": 0.0, "inflows_30d": 0.0, "net_outflows_30d": 0.0, "HQLA": hqla_amount}
    end = valuation_date + dt.timedelta(days=horizon_days)
    c = cf_base[(cf_base["payment_date"] > valuation_date) & (cf_base["payment_date"] <= end)].copy()

    # Merge rates per ID
    c["outflow_rate"] = c["ins_id"].map(lambda i: meta_by_id.get(i, {}).get("lcr_outflow_rate", 0.0))
    c["inflow_rate"] = c["ins_id"].map(lambda i: meta_by_id.get(i, {}).get("lcr_inflow_rate", 0.0))

    outflows = (c["amount"].clip(upper=0).abs() * c["outflow_rate"]).sum()
    inflows = (c["amount"].clip(lower=0) * c["inflow_rate"]).sum()

    # Apply 75% inflow cap (min(inflows, cap * outflows))
    capped_inflows = min(inflows, inflow_cap * outflows)
    net_outflows = max(outflows - capped_inflows, 0.0)
    lcr = (hqla_amount / net_outflows) if net_outflows > 0 else np.inf

    return {
        "HQLA": hqla_amount,
        "outflows_30d": float(outflows),
        "inflows_30d": float(inflows),
        "capped_inflows_30d": float(capped_inflows),
        "net_outflows_30d": float(net_outflows),
        "LCR": float(lcr)
    }


# ---- NSFR (simplified weights × notionals) ----
def compute_nsfr(meta_by_id: dict[str, dict]) -> dict:
    ASF = 0.0;
    RSF = 0.0
    for k, m in meta_by_id.items():
        notion = m.get("notional", 0.0) * m.get("units", 1.0)
        if m.get("side") == "liability":
            ASF += notion * m.get("nsfr_asf_weight", 0.0)
        else:
            RSF += notion * m.get("nsfr_rsf_weight", 0.0)
    ratio = (ASF / RSF) if RSF > 0 else np.inf
    return {"ASF": ASF, "RSF": RSF, "NSFR": ratio}


# ---- Liquidity ladder: monthly buckets for next N months ----
def build_liquidity_ladder(cf_base: pd.DataFrame,
                           start_date: dt.date,
                           months: int = 12) -> pd.DataFrame:
    if cf_base.empty:
        return pd.DataFrame(columns=["bucket", "inflow", "outflow", "net", "cum_net"])
    # take only horizon cashflows
    end = start_date + dt.timedelta(days=int(months * 30.44))
    d = cf_base[(cf_base["payment_date"] > start_date) & (cf_base["payment_date"] <= end)].copy()
    d["month"] = pd.to_datetime(d["payment_date"]).dt.to_period("M").dt.to_timestamp("M")
    g = d.groupby("month")["amount"].sum().reset_index()
    g["inflow"] = g["amount"].clip(lower=0.0)
    g["outflow"] = -g["amount"].clip(upper=0.0)
    g["net"] = g["inflow"] - g["outflow"]
    g["cum_net"] = g["net"].cumsum()
    g = g.drop(columns=["amount"]).rename(columns={"month": "bucket"})
    return g


# ---- Repricing gap by buckets (based on next reset for float, maturity for fixed) ----
def compute_repricing_gap(positions: dict,
                          base_curve: ql.YieldTermStructureHandle,
                          calc_date: ql.Date,
                          buckets_days: List[int]) -> pd.DataFrame:
    """Return DataFrame with columns: bucket, IRSA, IRSL, GAP, CUM_GAP (using notionals × units)."""
    cal = ql.TARGET()
    bucket_edges = [0] + buckets_days
    labels = []
    for i in range(len(bucket_edges) - 1):
        a, b = bucket_edges[i], bucket_edges[i + 1]
        labels.append(f"{a + 1}-{b}d" if b < 36500 else f">{a}d")

    vals = np.zeros((len(labels), 2))  # [IRSA, IRSL]

    def bucket_index(days: int) -> int:
        for i in range(len(bucket_edges) - 1):
            if bucket_edges[i] < days <= bucket_edges[i + 1]:
                return i
        return len(labels) - 1

    # Bonds
    for b in positions.get("bonds", []):
        notional = float(b["notional"]) * float(b.get("units", 1.0))
        side = b.get("side", "asset").lower()
        tenor = ql.Period(b.get("tenor", "1Y"))
        cald = cal.advance(calc_date, tenor)
        # repricing date: fixed → maturity; float → next coupon (approx = today + float_tenor)
        if b.get("kind", "fixed").lower().startswith("float"):
            ftenor = ql.Period(b.get("float_tenor", "3M"))
            repr_date = cal.advance(calc_date, ftenor)
        else:
            repr_date = cald
        days = int(repr_date - calc_date)
        idx = bucket_index(days)
        if side == "liability":
            vals[idx, 1] += notional
        else:
            vals[idx, 0] += notional

    # Swaps → treat notional as rate-sensitive (reprices with float tenor)
    for s in positions.get("swaps", []):
        notional = float(s["notional"]) * float(s.get("units", 1.0))
        side = s.get("side", "asset").lower()
        ftenor = ql.Period(s.get("float_leg_tenor", "3M"))
        repr_date = cal.advance(calc_date, ftenor)
        days = int(repr_date - calc_date)
        idx = bucket_index(days)
        if side == "liability":
            vals[idx, 1] += notional
        else:
            vals[idx, 0] += notional

    df = pd.DataFrame({"bucket": labels, "IRSA": vals[:, 0], "IRSL": vals[:, 1]})
    df["GAP"] = df["IRSA"] - df["IRSL"]
    df["CUM_GAP"] = df["GAP"].cumsum()
    return df


# ---- Duration gap & EVE (finite-difference duration using +1bp curve) ----
def duration_gap_and_eve(npv0: Dict,
                         npv1: Dict,
                         units: Dict,
                         bump_bp=.01,
                         eve_bp: float = 200.0) -> dict:
    dY = bump_bp / 10_000.0

    A_pv = 0.0;
    L_pv = 0.0;
    A_num = 0.0;
    L_num = 0.0  # PV sums

    for i_id in npv0.keys():
        PV0 = npv0[i_id]
        PV1 = npv1[i_id]
        u = units[i_id]

        dur = - (PV1 - PV0) / (PV0 * dY) if PV0 != 0 else 0.0
        if u < 0:
            L_pv += PV0;
            L_num += dur * PV0
        else:
            A_pv += PV0;
            A_num += dur * PV0

    D_A = (A_num / A_pv) if abs(A_pv) > 1e-12 else 0.0
    D_L = (L_num / L_pv) if abs(L_pv) > 1e-12 else 0.0
    k = (L_pv / A_pv) if abs(A_pv) > 1e-12 else 0.0
    D_gap = D_A - k * D_L

    # EVE shocks via duration linearization
    dy = eve_bp / 10_000.0
    dEVE_up = - (D_A * A_pv - D_L * L_pv) * dy
    dEVE_down = + (D_A * A_pv - D_L * L_pv) * dy  # symmetric approx

    return {
        "A": A_pv, "L": L_pv, "k_L_over_A": k,
        "D_A": D_A, "D_L": D_L, "D_gap": D_gap,
        "EVE": {f"+{int(eve_bp)}bp": dEVE_up, f"-{int(eve_bp)}bp": dEVE_down}
    }


# ---- NII 12m shock: sum projected cash amounts in horizon under +bp ----
def nii_12m_shock(cf0,
                  cf1,
                  calc_date: ql.Date,
                  shock_bp: float = 100.0,
                  horizon_days: int = 365) -> dict:
    cutoff = to_py_date(calc_date) + dt.timedelta(days=horizon_days)

    # NII uses *amounts* in horizon (not PV). Values already signed by side.
    base_amt = cf0["amount"].sum() if not cf0.empty else 0.0
    sh_amt = cf1["amount"].sum() if not cf1.empty else 0.0
    return {"base_amount_12m": base_amt,
            "delta": sh_amt - base_amt}


def portfolio_totals(npv0: dict, npv1: dict, units) -> pd.DataFrame:
    rows = []
    for ins_id in npv0.keys():
        base = float(npv0[ins_id])
        bump = float(npv1[ins_id])
        u = units[ins_id]
        delta = bump - base
        rows.append({"instrument": ins_id, "units": u, "base": base, "bumped": bump, "delta": delta})
    df = pd.DataFrame(rows)

    total_row = {
        "instrument": "TOTAL",
        "units": np.nan,  # keep column numeric
        "base": df["base"].sum(),
        "bumped": df["bumped"].sum(),
        "delta": df["delta"].sum(),
    }
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)


def nodes_parallel_bump(nodes: List[dict], bp: float) -> List[dict]:
    bump = float(bp) / 10_000.0
    out = []
    for n in nodes:
        m = dict(n)
        m["rate"] = float(m["rate"]) + bump
        out.append(m)
    return out


# ---- Build curves for UI (USD via nodes; MXN via Valmer TIIE) ----
def _nodes_from_ts(ts: ql.YieldTermStructureHandle, calc_date: ql.Date,
                   tenors: list[str]) -> list[dict]:
    cal, dc = ql.TARGET(), ql.Actual365Fixed()
    out = []
    ref = ts.referenceDate()
    for t in tenors:
        d = cal.advance(calc_date, ql.Period(t))
        if d < ref:
            d = ref
        try:
            r = ts.zeroRate(d, dc, ql.Continuous, ql.Annual).rate()
        except Exception as e:
            raise e
        out.append({"type": "zero", "tenor": t.upper(), "rate": float(r)})
    return out


def _bump_curve_ts(base_ts: ql.YieldTermStructureHandle, calc_date: ql.Date,
                   tenor_bumps_bp: dict[str, float] | None,
                   parallel_bp: float = 0.0,
                   max_years: int = 12, step_months: int = 3) -> ql.YieldTermStructureHandle:
    """Create a bumped zero curve by sampling base_ts and adding a piecewise-linear zero spread."""
    tenor_bumps_bp = {(k or "").upper(): float(v) for k, v in (tenor_bumps_bp or {}).items()}
    cal, dc = ql.TARGET(), ql.Actual365Fixed()

    # sample grid — include short-end anchors so 28D/91D don't sit left of the first knot
    short_anchors = [cal.advance(calc_date, ql.Period(p)) for p in ("28D", "1M", "2M")]

    monthly = [cal.advance(calc_date, ql.Period(m, ql.Months))
               for m in range(step_months, max_years * 12 + 1, step_months)]
    dates = sorted(set(short_anchors + monthly))
    # strictly increasing and unique

    z0 = [base_ts.zeroRate(d, dc, ql.Continuous, ql.Annual).rate() for d in dates]

    # build bump function (piecewise linear in years)
    def tenor_to_years(s: str) -> float:
        s = s.strip().upper()
        if s.endswith("Y"): return float(s[:-1])
        if s.endswith("M"): return float(s[:-1]) / 12.0
        if s.endswith("D"): return float(s[:-1]) / 365.0
        return float(s)

    if tenor_bumps_bp:
        xs = np.array(sorted(tenor_to_years(k) for k in tenor_bumps_bp.keys()))
        ys = np.array([tenor_bumps_bp[k] / 10_000.0 for k in sorted(tenor_bumps_bp.keys(), key=tenor_to_years)])
    else:
        xs = np.array([0.0, max_years])
        ys = np.array([0.0, 0.0])

    t_grid = np.array([dc.yearFraction(calc_date, d) for d in dates])
    keyrate_spread = np.interp(t_grid, xs, ys, left=ys[0], right=ys[-1])
    par_spread = float(parallel_bp) / 10_000.0

    z_bumped = [max(z + par_spread + s, -0.20) for z, s in zip(z0, keyrate_spread)]  # guard
    # --- Anchor reference date at calc_date by PREPENDING calc_date to the grid ---
    # Interpolate the key-rate spread at t=0 as well (use left endpoint if needed).
    if tenor_bumps_bp:
        k0 = float(np.interp(0.0, xs, ys, left=ys[0], right=ys[-1]))
    else:
        k0 = 0.0
    z0_at_ref = base_ts.zeroRate(calc_date, dc, ql.Continuous, ql.Annual).rate()
    z0_at_ref_bumped = max(z0_at_ref + par_spread + k0, -0.20)
    dates2 = [calc_date] + dates
    z_bumped2 = [z0_at_ref_bumped] + z_bumped
    # Use the overload: ZeroCurve(dates, zeroRates, dayCounter[, calendar])
    bumped = ql.ZeroCurve(dates2, z_bumped2, dc, cal)

    bumped.enableExtrapolation()
    return ql.YieldTermStructureHandle(bumped)


def build_curves_for_ui(cfg: dict, calc_date: ql.Date,
                        tenor_bumps_bp: dict[str, float],
                        parallel_bp: float):
    cur = (cfg.get("valuation", {}).get("currency", "USD") or "USD").upper()
    curve_source = (cfg.get("alm_assumptions", {}).get("curve_source", "") or "").upper()

    if cur == "MXN" or "TIIE" in curve_source:
        # Base TIIE curve from Valmer (your helper)
        base_ts = build_tiie_zero_curve_from_valmer(calc_date)
        if not isinstance(base_ts, ql.YieldTermStructureHandle):
            base_ts = ql.YieldTermStructureHandle(base_ts)
        bump_ts = _bump_curve_ts(base_ts, calc_date, tenor_bumps_bp, parallel_bp)

        # Node markers for plotting
        tenors = ["28D", "91D", "182D", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]
        nodes_base = _nodes_from_ts(base_ts, calc_date, tenors)
        nodes_bump = _nodes_from_ts(bump_ts, calc_date, tenors)
        return base_ts, bump_ts, nodes_base, nodes_bump, "MXN-TIIE-28D"

    # USD path (unchanged)
    nodes_base = load_swap_nodes()
    nodes_tenor = apply_bumps(nodes_base, tenor_bumps_bp) if tenor_bumps_bp else list(nodes_base)
    nodes_final = nodes_parallel_bump(nodes_tenor, parallel_bp) if abs(parallel_bp) > 1e-12 else nodes_tenor
    base_ts = build_curve_from_swap_nodes(calc_date, nodes_base)
    bump_ts = build_curve_from_swap_nodes(calc_date, nodes_final)
    return base_ts, bump_ts, nodes_base, nodes_final, "USD-LIBOR-3M"
