# dashboards/bond_real_option_analysis/engine.py
# ------------------------------------------------------------
# Generalized LSM optimal stopping engine for bond "hold vs sell & invest".
# - Works with QuantLib bonds (Fixed or Floating). For floaters, future coupon
#   amounts are taken from the current pricer (i.e., curve expectations).
# - No use of HullWhite.dynamics()/stateProcess() (portable across builds).
# - Market curve drives HW dynamics + sale price; FTP curve is PV numéraire;
#   Invest curve applies to the stop branch (invest-to-horizon multiplier).
# - Evaluation function pluggable via 'evaluation_fn'.
# ------------------------------------------------------------

from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List, Any

import numpy as np
import QuantLib as ql


# -------------------------- helpers --------------------------

def as_handle(ts) -> ql.YieldTermStructureHandle:
    return ts if isinstance(ts, ql.YieldTermStructureHandle) else ql.YieldTermStructureHandle(ts)

def discount(ts_or_handle, d_or_t):
    return ts_or_handle.discount(d_or_t) if isinstance(ts_or_handle, ql.YieldTermStructureHandle) \
           else ts_or_handle.discount(d_or_t)

def qld(d) -> ql.Date:
    # Accept datetime.date or ql.Date
    if isinstance(d, ql.Date):
        return d
    return ql.Date(d.day, d.month, d.year)

def _inst_forward_at_date(ts_or_handle, date: ql.Date, dc: ql.DayCounter, h_days: int = 1) -> float:
    """Approx f(0,t) using two nearby Dates to avoid daycount mismatches."""
    d0 = date
    dp = ql.Date(d0.serialNumber() + h_days)
    dm = ql.Date(max(d0.serialNumber() - h_days, ql.Settings.instance().evaluationDate.serialNumber()))
    if dm < ql.Settings.instance().evaluationDate:
        dm = ql.Settings.instance().evaluationDate
    Dp = discount(ts_or_handle, dp)
    Dm = discount(ts_or_handle, dm)
    t_p = dc.yearFraction(ql.Settings.instance().evaluationDate, dp)
    t_m = dc.yearFraction(ql.Settings.instance().evaluationDate, dm)
    h = t_p - t_m if t_p != t_m else 1e-8
    return -(np.log(Dp) - np.log(Dm)) / h

def _phi_grid_from_curve(ts_or_handle, a: float, sigma: float,
                         grid_dates: List[ql.Date],
                         grid_times: np.ndarray,
                         dc: ql.DayCounter) -> np.ndarray:
    """phi(t) = f(0,t) + sigma^2/(2a^2) * (1 - e^{-a t})^2 on the provided grid."""
    out = np.empty(len(grid_times))
    for i, (d, t) in enumerate(zip(grid_dates, grid_times)):
        f0t = _inst_forward_at_date(ts_or_handle, d, dc)
        out[i] = f0t + (sigma**2) / (2.0 * a * a) * (1.0 - np.exp(-a * t))**2
    return out

def _get_ql_bond(obj) -> ql.Bond:
    """Extract underlying QuantLib bond from various wrappers."""
    if isinstance(obj, ql.Bond):
        return obj
    if hasattr(obj, "_bond"):
        return getattr(obj, "_bond")
    if hasattr(obj, "bond"):
        return getattr(obj, "bond")
    raise TypeError("Cannot extract ql.Bond from instrument")

def _cashflows_time_amount(bond: ql.Bond, eval_date: ql.Date, dc: ql.DayCounter) -> List[Tuple[float, float]]:
    """Future cashflows (>= eval_date), merged by time (in year fractions)."""
    cfs: Dict[float, float] = {}
    for cf in bond.cashflows():
        if cf.date() < eval_date:
            continue  # keep same-day flows
        t = dc.yearFraction(eval_date, cf.date())
        cfs[t] = cfs.get(t, 0.0) + float(cf.amount())
    return sorted(cfs.items(), key=lambda x: x[0])

def _grid_from_bond(bond: ql.Bond, eval_date: ql.Date, dc: ql.DayCounter,
                    mesh_months: int = 1) -> Tuple[List[ql.Date], np.ndarray]:
    """Decision grid: eval_date + all CF dates + a monthly mesh between them."""
    # Gather CF dates on/after eval_date
    dates: List[ql.Date] = [eval_date]
    for cf in bond.cashflows():
        if cf.date() >= eval_date:
            dates.append(cf.date())
    # add monthly mesh
    cal = ql.TARGET()
    uniq = sorted(set(dates), key=lambda d: d.serialNumber())
    grid: List[ql.Date] = []
    for i in range(len(uniq)-1):
        d0, d1 = uniq[i], uniq[i+1]
        grid.append(d0)
        d = cal.advance(d0, ql.Period(mesh_months, ql.Months))
        while d < d1:
            grid.append(d)
            d = cal.advance(d, ql.Period(mesh_months, ql.Months))
    grid.append(uniq[-1])
    grid = sorted(set(grid), key=lambda d: d.serialNumber())
    times = np.array([dc.yearFraction(eval_date, d) for d in grid], dtype=float)
    assert abs(times[0]) < 1e-12
    return grid, times


# -------------------------- default evaluation function -----------------------

def eval_sell_invest_to_horizon(i: int, t_i: float, r_i: np.ndarray,
                                coupon_now: float, cont_est: np.ndarray,
                                ex_coupon_core: np.ndarray, multiplier_Mi: float,
                                params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Stop = (sale proceeds net) * DF_FTP(t,H)/DF_INV(t,H); Hold = coupon_now + continuation."""
    bid_ask_bps = float(params.get("bid_ask_bps", 0.0))
    stop_val = ex_coupon_core * (1.0 - bid_ask_bps * 1e-4) * float(multiplier_Mi)
    hold_now = coupon_now + cont_est
    return stop_val, hold_now


# -------------------------- main engine --------------------------------------

@dataclass
class LSMSettings:
    a: float = 0.03
    sigma: float = 0.01
    n_paths: int = 10000
    seed: int = 42
    capital_rate: float = 0.0   # continuous per annum while holding
    mesh_months: int = 1        # decision mesh granularity
    record_diagnostics: bool = True

@dataclass
class LSMResult:
    lsm_value: float
    ql_npv: float
    settings: LSMSettings
    diag: Dict[str, Any] = dataclasses.field(default_factory=dict)

def lsm_optimal_stopping(
    instrument,                          # ql.Bond or wrapper with ._bond / .bond
    market_curve, ftp_curve, invest_curve,
    evaluation_fn: Callable = eval_sell_invest_to_horizon,
    eval_params: Dict[str, Any] | None = None,
    *,
    settings: LSMSettings = LSMSettings(),
    day_count: ql.DayCounter = ql.Actual365Fixed(),
) -> LSMResult:
    """
    MARKET curve: HW dynamics + ex-coupon pricing.
    FTP curve: continuation discount (PV numéraire).
    INVEST curve: stop-branch invest-to-horizon multiplier.
    """
    eval_params = eval_params or {}

    # Core objects & grids
    bond = _get_ql_bond(instrument)
    eval_date = ql.Settings.instance().evaluationDate
    grid_dates, grid_times = _grid_from_bond(bond, eval_date, day_count, settings.mesh_months)

    # Cashflow map onto grid (deterministic amounts from pricer; for floaters this is an approximation)
    cf_times_amts = _cashflows_time_amount(bond, eval_date, day_count)
    t2i = {round(t, 10): i for i, t in enumerate(np.round(grid_times, 10))}
    coupon_on_grid = np.zeros(len(grid_times))
    for t, a in cf_times_amts:
        rt = round(t, 10)
        if rt not in t2i:
            raise RuntimeError(f"Cashflow time t={t} not on decision grid.")
        coupon_on_grid[t2i[rt]] += a

    # Simulate HW short rates WITHOUT .dynamics(): r_t = x_t + phi(t)
    n_steps = len(grid_times) - 1
    dt = np.diff(grid_times)
    a, sigma = settings.a, settings.sigma
    dc = day_count

    # OU exact step params
    decays = np.exp(-a * dt)
    stds = sigma * np.sqrt((1.0 - np.exp(-2.0*a*dt)) / (2.0*a))

    phi_vals = _phi_grid_from_curve(market_curve, a, sigma, grid_dates, grid_times, dc)
    # r0 ~ f(0,0+)
    f00 = _inst_forward_at_date(market_curve, eval_date, dc)
    x0 = f00 - phi_vals[0]

    rng = np.random.default_rng(settings.seed)
    Z = rng.standard_normal(size=(settings.n_paths, n_steps))
    rates = np.zeros((settings.n_paths, len(grid_times)))
    x = np.full(settings.n_paths, x0, dtype=float)
    rates[:, 0] = x + phi_vals[0]
    for i in range(n_steps):
        x = x * decays[i] + stds[i] * Z[:, i]
        rates[:, i+1] = x + phi_vals[i+1]

    # Deterministic FTP step DFs + capital charge
    df_ftp = np.array([discount(ftp_curve, d) for d in grid_dates])
    df_inv = np.array([discount(invest_curve, d) for d in grid_dates])
    step_df_ftp = df_ftp[1:] / df_ftp[:-1]
    step_disc_hold = step_df_ftp * np.exp(-settings.capital_rate * dt)

    # Horizon multiplier M_i = DF_FTP(t_i,H)/DF_INV(t_i,H)
    H_idx = len(grid_times) - 1
    M = (df_ftp[H_idx] / df_ftp) / (df_inv[H_idx] / df_inv)

    # Hull–White analytical zero-bond for pricing ex-coupon (market curve)
    hw = ql.HullWhite(as_handle(market_curve), a, sigma)

    def ex_coupon_price_at(t_i: float, r_i: float) -> float:
        price = 0.0
        for (T, C) in cf_times_amts:
            if T > t_i + 1e-12:
                price += C * hw.discountBond(t_i, T, r_i)
        return price

    # Basis for regression
    def basis(xv: np.ndarray) -> np.ndarray:
        return np.vstack([np.ones_like(xv), xv, xv**2]).T

    # Backward LSM with optional diagnostics
    V = np.zeros((settings.n_paths, len(grid_times)))
    V[:, -1] = coupon_on_grid[-1]

    first_ex_idx = np.full(settings.n_paths, fill_value=-1, dtype=int)
    exercise_rate = np.zeros(len(grid_times))

    for i in range(len(grid_times)-2, -1, -1):
        t_i = grid_times[i]
        r_i = rates[:, i]
        cont_disc = step_disc_hold[i] * V[:, i+1]

        X = basis(r_i)
        beta, *_ = np.linalg.lstsq(X, cont_disc, rcond=None)
        cont_est = X @ beta

        # Vectorized ex-coupon pricing per path
        ex_core = np.fromiter((ex_coupon_price_at(t_i, r) for r in r_i), dtype=float, count=len(r_i))
        stop_val, hold_now = evaluation_fn(i, t_i, r_i, float(coupon_on_grid[i]), cont_est, ex_core, M[i], eval_params)

        exercise = stop_val > hold_now
        V[:, i] = np.where(exercise, stop_val, coupon_on_grid[i] + cont_disc)

        if settings.record_diagnostics:
            exercise_rate[i] = float(np.mean(exercise))
            # mark first exercise time
            newly = (first_ex_idx < 0) & exercise
            first_ex_idx[newly] = i

    lsm_value = float(np.mean(V[:, 0]))

    # Frictionless benchmark (market curve)
    bond.setPricingEngine(ql.DiscountingBondEngine(as_handle(market_curve)))
    ql_npv = float(bond.NPV())

    diag: Dict[str, Any] = {}
    if settings.record_diagnostics:
        times = grid_times.tolist()
        diag["exercise_rate_by_time"] = exercise_rate.tolist()
        # Convert first-ex idx to times where set
        first_ex_times = [times[j] for j in first_ex_idx if j >= 0]
        diag["first_exercise_times"] = first_ex_times
        diag["share_exercised_at_all"] = float(np.mean(first_ex_idx >= 0))

    return LSMResult(lsm_value=lsm_value, ql_npv=ql_npv, settings=settings, diag=diag)
