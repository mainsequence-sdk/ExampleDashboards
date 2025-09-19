# alm_lsm_engine.py
# ------------------------------------------------------------
# Generalized LSM optimal stopping engine for a bond under HW1F.
# - Inputs to scenarios: (market_rate, ftp_rate, invest_rate, bond)
# - ALM framing: market curve drives pricing & dynamics; FTP curve is PV numeraire;
#   invest curve applies to the stop branch (sell & invest).
# - No use of HullWhite.dynamics()/stateProcess(); portable across builds.
# - Pass a custom `evaluation_fn` to implement your economics for stop vs continue.
# ------------------------------------------------------------

import QuantLib as ql
import numpy as np

# -------------------------- Global setup --------------------------
evaluation_date = ql.Date(18, 9, 2025)
ql.Settings.instance().evaluationDate = evaluation_date

calendar    = ql.TARGET()
day_count   = ql.Actual365Fixed()
settle_days = 2

# HW model params (change if needed)
DEFAULT_A     = 0.03
DEFAULT_SIGMA = 0.01

# -------------------------- Utilities --------------------------
def as_handle(ts_or_handle):
    return ts_or_handle if isinstance(ts_or_handle, ql.YieldTermStructureHandle) \
                        else ql.YieldTermStructureHandle(ts_or_handle)

def discount(ts_or_handle, d_or_t):
    # Works for both ql.YieldTermStructure and Handle
    return ts_or_handle.discount(d_or_t) if isinstance(ts_or_handle, ql.YieldTermStructureHandle) \
           else ts_or_handle.discount(d_or_t)

def unique_sorted_dates(dates):
    seen, out = set(), []
    for d in dates:
        s = d.serialNumber()
        if s not in seen:
            seen.add(s); out.append(d)
    out.sort(key=lambda dd: dd.serialNumber())
    return out

def inst_forward_at_date(ts_or_handle, date, h_days=1):
    """Approx f(0,t) via finite difference on Dates (robust across wrappers)."""
    d0 = date
    dp = ql.Date(d0.serialNumber() + h_days)
    dm = ql.Date(max(d0.serialNumber() - h_days, evaluation_date.serialNumber()))
    if dm < evaluation_date:
        dm = evaluation_date
    Dp = discount(ts_or_handle, dp)
    Dm = discount(ts_or_handle, dm)
    tau_p = day_count.yearFraction(evaluation_date, dp)
    tau_m = day_count.yearFraction(evaluation_date, dm)
    h = tau_p - tau_m if tau_p != tau_m else 1e-8
    return -(np.log(Dp) - np.log(Dm)) / h

def phi_grid_from_curve(ts_or_handle, a, sigma, grid_dates, grid_times):
    """phi(t) = f(0,t) + (sigma^2/(2 a^2)) * (1 - e^{-a t})^2 on the provided grid."""
    phi_vals = np.empty(len(grid_times))
    for i, (d, t) in enumerate(zip(grid_dates, grid_times)):
        f0t = inst_forward_at_date(ts_or_handle, d)
        phi_vals[i] = f0t + (sigma**2)/(2.0*a*a) * (1.0 - np.exp(-a*t))**2
    return phi_vals

# -------------------------- Bond helpers --------------------------
def extract_cashflows(bond):
    """Return list of (time, amount) for cashflows on/after evaluation_date, merged by time."""
    cfs = []
    for cf in bond.cashflows():
        if cf.date() < evaluation_date:   # keep same-day flows
            continue
        t = day_count.yearFraction(evaluation_date, cf.date())
        cfs.append((t, cf.amount()))
    # merge coincident times (coupon + redemption)
    agg = {}
    for t, a in cfs:
        agg[t] = agg.get(t, 0.0) + a
    return sorted(agg.items(), key=lambda x: x[0])

def build_grid_for_bond(bond, mesh_months=1):
    """Decision grid: evaluation_date + all future CF dates + monthly mesh between CFs."""
    # start with eval date and all future cashflow dates
    cf_dates = [evaluation_date]
    for cf in bond.cashflows():
        if cf.date() >= evaluation_date:
            cf_dates.append(cf.date())
    cf_dates = unique_sorted_dates(cf_dates)
    # add mesh between consecutive CF dates
    grid_dates = []
    for i in range(len(cf_dates)-1):
        d0, d1 = cf_dates[i], cf_dates[i+1]
        grid_dates.append(d0)
        d = calendar.advance(d0, ql.Period(mesh_months, ql.Months))
        while d < d1:
            grid_dates.append(d)
            d = calendar.advance(d, ql.Period(mesh_months, ql.Months))
    grid_dates.append(cf_dates[-1])
    grid_dates = unique_sorted_dates(grid_dates)

    grid_times = np.array([day_count.yearFraction(evaluation_date, d) for d in grid_dates], dtype=float)
    assert abs(grid_times[0]) < 1e-12, "first grid time must be 0.0"

    # map cashflows onto grid
    cf_times_amts = extract_cashflows(bond)
    time_to_idx = {round(t, 10): i for i, t in enumerate(np.round(grid_times, 10))}
    coupon_on_grid = np.zeros(len(grid_times))
    for t, a in cf_times_amts:
        rt = round(t, 10)
        if rt not in time_to_idx:
            raise RuntimeError(f"Cashflow time t={t} not on decision grid.")
        coupon_on_grid[time_to_idx[rt]] += a

    return grid_dates, grid_times, coupon_on_grid, cf_times_amts

# -------------------------- HW simulator (no .dynamics() needed) -------------
def simulate_HW_rates(n_paths, grid_dates, grid_times, a, sigma, market_curve, seed=42):
    """Simulate r_t = x_t + phi(t) with exact OU step for x_t under the MARKET curve."""
    np.random.seed(seed)
    n_steps = len(grid_times) - 1
    dt      = np.diff(grid_times)
    decays  = np.exp(-a * dt)
    stds    = sigma * np.sqrt((1.0 - np.exp(-2.0*a*dt))/(2.0*a))

    phi_vals = phi_grid_from_curve(market_curve, a, sigma, grid_dates, grid_times)

    # r0 approx f(0,0+)
    r0  = inst_forward_at_date(market_curve, evaluation_date)
    x0  = r0 - phi_vals[0]  # ~0

    rates = np.zeros((n_paths, len(grid_times)))
    x     = np.full(n_paths, x0, dtype=float)
    rates[:, 0] = x + phi_vals[0]

    Z = np.random.standard_normal(size=(n_paths, n_steps))
    for i in range(n_steps):
        x = x * decays[i] + stds[i] * Z[:, i]
        rates[:, i+1] = x + phi_vals[i+1]
    return rates

# -------------------------- Default evaluation function ----------------------
def eval_sell_invest_to_horizon(i, t_i, r_i, coupon_now, cont_est, ex_coupon_core, multiplier_Mi, params):
    """
    Generic evaluation function for the stop/continue decision.

    Arguments:
      i:               time index
      t_i:             current time (year fraction)
      r_i:             np.array of short rates on each path at t_i
      coupon_now:      scalar cashflow at t_i (0 if none)
      cont_est:        np.array, regression estimate of E[ discounted V_{i+1} | r_i ]
      ex_coupon_core:  np.array of ex-coupon model prices at (t_i, r_i)
      multiplier_Mi:   scalar = DF_FTP(t_i,H) / DF_INV(t_i,H)
      params:          dict e.g. {'bid_ask_bps': 5.0}

    Returns:
      stop_val, hold_now  (both np.array shape (n_paths,))
    """
    bid_ask_bps = float(params.get('bid_ask_bps', 0.0))
    stop_val = ex_coupon_core * (1.0 - bid_ask_bps * 1e-4) * float(multiplier_Mi)
    hold_now = coupon_now + cont_est
    return stop_val, hold_now

# -------------------------- LSM engine (generalized) -------------------------
def lsm_optimal_stopping(
    bond,
    market_curve,
    ftp_curve,
    invest_curve,
    *,
    a=DEFAULT_A,
    sigma=DEFAULT_SIGMA,
    n_paths=10000,
    seed=42,
    capital_rate=0.0,
    mesh_months=1,
    basis_fn=None,
    evaluation_fn=eval_sell_invest_to_horizon,
    eval_params=None,
):
    """
    Generalized LSM engine.
      - Uses MARKET curve for: HW dynamics & ex-coupon pricing
      - Uses FTP curve for: continuation discounting (PV numeraire)
      - Uses INVEST curve for: stop-branch invest-to-horizon multiplier

    Returns: dict with 'lsm_value', 'ql_npv', 'settings'
    """
    eval_params = eval_params or {}

    # Build grid & cashflows
    grid_dates, grid_times, coupon_on_grid, cf_times_amts = build_grid_for_bond(bond, mesh_months=mesh_months)

    # Simulate short rates under MARKET curve
    rates = simulate_HW_rates(n_paths, grid_dates, grid_times, a, sigma, market_curve, seed)

    # Analytic ex-coupon via HW discountBond
    hw = ql.HullWhite(as_handle(market_curve), a, sigma)

    # FTP step discount (exact, deterministic) + capital charge
    df_ftp = np.array([discount(ftp_curve, d) for d in grid_dates])
    step_df_ftp = df_ftp[1:] / df_ftp[:-1]
    dt = np.diff(grid_times)
    step_disc_hold = step_df_ftp * np.exp(-capital_rate * dt)

    # Invest-to-horizon multiplier M_i = DF_FTP(t_i,H) / DF_INV(t_i,H)
    df_inv = np.array([discount(invest_curve, d) for d in grid_dates])
    H_idx = len(grid_times) - 1
    M = (df_ftp[H_idx] / df_ftp) / (df_inv[H_idx] / df_inv)   # shape (N_times,)

    # Basis for regression (low-order polynomial by default)
    def default_basis(x):
        return np.vstack([np.ones_like(x), x, x**2]).T
    basis = basis_fn or default_basis

    # ex-coupon pricing helper (using model times)
    def ex_coupon_price_at(t_i, r_i):
        price = 0.0
        for (T, C) in cf_times_amts:
            if T > t_i + 1e-12:
                price += C * hw.discountBond(t_i, T, r_i)
        return price

    # Backward induction
    V = np.zeros((n_paths, len(grid_times)))
    V[:, -1] = coupon_on_grid[-1]

    for i in range(len(grid_times)-2, -1, -1):
        t_i = grid_times[i]
        r_i = rates[:, i]
        coupon_now = float(coupon_on_grid[i])

        # Discounted continuation (FTP step df + capital charge already embedded)
        cont_disc = step_disc_hold[i] * V[:, i+1]

        # Regress E[cont_disc | r_i]
        X = basis(r_i)
        coeff, *_ = np.linalg.lstsq(X, cont_disc, rcond=None)
        cont_est = X @ coeff

        # Ex-coupon price along each path (vectorized through Python loop; robust & readable)
        ex_core = np.fromiter((ex_coupon_price_at(t_i, r) for r in r_i), dtype=float, count=len(r_i))

        # Use the evaluation function to get stop vs. hold values
        stop_val, hold_now = evaluation_fn(
            i, t_i, r_i, coupon_now, cont_est, ex_core, M[i], eval_params
        )

        # Optimal decision
        exercise = stop_val > hold_now
        V[:, i] = np.where(exercise, stop_val, coupon_on_grid[i] + cont_disc)

    # LSM value at t=0 (FTP PV measure in general)
    lsm_value = float(np.mean(V[:, 0]))

    # Frictionless benchmark NPV under MARKET curve
    bond.setPricingEngine(ql.DiscountingBondEngine(as_handle(market_curve)))
    ql_npv = bond.NPV()

    return {
        "lsm_value": lsm_value,
        "ql_npv": ql_npv,
        "settings": {
            "paths": n_paths,
            "mesh_months": mesh_months,
            "a": a,
            "sigma": sigma,
            "capital_rate": capital_rate,
            "seed": seed,
        },
    }

# -------------------------- Scenario runner (3 rates + bond) -----------------
def analyze_scenarios(
    market_rate: float,
    ftp_rate: float,
    invest_rate: float,
    bond: ql.Bond,
    *,
    a=DEFAULT_A,
    sigma=DEFAULT_SIGMA,
    n_paths=12000,
    seed=7,
    bid_ask_bps_frictionless=0.0,
    bid_ask_bps_alm=5.0,
    capital_rate_frictionless=0.0,
    capital_rate_alm=0.005,
    mesh_months=1,
    evaluation_fn=eval_sell_invest_to_horizon,
):
    """Build flat curves from the three rates and run both scenarios."""

    # Build flat curves (you can swap these with real curves if you have them)
    market_curve  = ql.FlatForward(evaluation_date, market_rate, day_count)
    ftp_curve_eq  = market_curve  # frictionless uses the same curve
    invest_curve_eq = market_curve

    ftp_curve_alm    = ql.FlatForward(evaluation_date, ftp_rate, day_count)
    invest_curve_alm = ql.FlatForward(evaluation_date, invest_rate, day_count)

    # --- Scenario A: Frictionless (all curves equal) ---
    res_A = lsm_optimal_stopping(
        bond, market_curve, ftp_curve_eq, invest_curve_eq,
        a=a, sigma=sigma, n_paths=n_paths, seed=seed,
        capital_rate=capital_rate_frictionless, mesh_months=mesh_months,
        evaluation_fn=evaluation_fn,
        eval_params={"bid_ask_bps": bid_ask_bps_frictionless},
    )

    # --- Scenario B: ALM wedge (different FTP & invest) ---
    res_B = lsm_optimal_stopping(
        bond, market_curve, ftp_curve_alm, invest_curve_alm,
        a=a, sigma=sigma, n_paths=n_paths, seed=seed+1,
        capital_rate=capital_rate_alm, mesh_months=mesh_months,
        evaluation_fn=evaluation_fn,
        eval_params={"bid_ask_bps": bid_ask_bps_alm},
    )

    # Print summary
    print("\n=== Scenario A: Frictionless (market=FTP=invest) ===")
    print(f"LSM optimal liquidation value @ t=0: {res_A['lsm_value']:,.6f}")
    print(f"QuantLib bond NPV (market curve):     {res_A['ql_npv']:,.6f}")
    print(f"Settings: {res_A['settings']}")

    print("\n=== Scenario B: ALM wedge (market vs FTP vs invest) ===")
    print(f"LSM optimal liquidation value @ t=0 (ALM PV): {res_B['lsm_value']:,.6f}")
    print(f"Plain QuantLib bond NPV (market curve):       {res_B['ql_npv']:,.6f}")
    print(f"Settings: {res_B['settings']}")
    print(f"Curves: market={market_rate*100:.2f}%, FTP={ftp_rate*100:.2f}%, invest={invest_rate*100:.2f}%")
    print(f"Frictions: frictionless bid-ask={bid_ask_bps_frictionless} bps, ALM bid-ask={bid_ask_bps_alm} bps")
    print(f"Capital charges: frictionless={capital_rate_frictionless*1e2:.1f} bps, ALM={capital_rate_alm*1e2:.1f} bps")

    return res_A, res_B

# -------------------------- Example usage ------------------------------------
if __name__ == "__main__":
    # Build an example bond (replace with your own ql.Bond if desired)
    issue    = evaluation_date
    maturity = calendar.advance(issue, ql.Period(5, ql.Years))
    coupon   = 0.05
    freq     = ql.Semiannual
    face     = 100.0

    schedule = ql.Schedule(issue, maturity, ql.Period(freq), calendar,
                           ql.Following, ql.Following, ql.DateGeneration.Backward, False)
    example_bond = ql.FixedRateBond(settle_days, face, schedule, [coupon], day_count)

    # Run analysis with 3 input rates (market / FTP / invest)
    analyze_scenarios(
        market_rate=0.030,
        ftp_rate=0.0275,
        invest_rate=0.0335,
        bond=example_bond,
        a=0.03, sigma=0.01,
        n_paths=10000, seed=123,
        bid_ask_bps_frictionless=0.0,
        bid_ask_bps_alm=5.0,
        capital_rate_frictionless=0.0,
        capital_rate_alm=0.005,
        mesh_months=1,
        evaluation_fn=eval_sell_invest_to_horizon,  # you can pass your own
    )
