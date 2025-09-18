# hold_option_hw_lsm.py
# ------------------------------------------------------------
# Optimal "hold vs sell-and-invest" decision for a bond
# using Longstaff–Schwartz (LSM) on Hull–White short-rate paths.
#
# Dependencies: QuantLib-python, numpy
# ------------------------------------------------------------

import QuantLib as ql
import numpy as np

# ---------- Inputs you can change ----------
evaluation_date = ql.Date(18, 9, 2025)
ql.Settings.instance().evaluationDate = evaluation_date

calendar    = ql.TARGET()
day_count   = ql.Actual365Fixed()
settle_days = 2

# Funding curve (OIS-like). Flat 3.0% for illustration.
flat_funding_rate = 0.03
funding_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(evaluation_date, flat_funding_rate, day_count, ql.Continuous, ql.NoFrequency)
)

# Hull–White parameters (use your calibrated ones if available)
a     = 0.03   # mean reversion
sigma = 0.01   # volatility

# Bond example (replace with your own cashflows if desired)
issue    = evaluation_date
maturity = calendar.advance(issue, ql.Period(5, ql.Years))
coupon   = 0.05      # 5% annual coupon
freq     = ql.Semiannual
face     = 100.0

schedule = ql.Schedule(issue, maturity, ql.Period(freq), calendar,
                       ql.Following, ql.Following, ql.DateGeneration.Backward, False)
bond = ql.FixedRateBond(settle_days, face, schedule, [coupon], day_count)

# ---------- Decision grid: coupon dates + monthly mesh in between ----------
def unique_sorted_dates(dates):
    seen = set()
    out = []
    for d in dates:
        s = d.serialNumber()
        if s not in seen:
            seen.add(s)
            out.append(d)
    # Sort by serialNumber to be robust across bindings
    out.sort(key=lambda dd: dd.serialNumber())
    return out

grid_dates = list(schedule)  # includes issue/maturity and coupon dates
# Add monthly mesh between each consecutive schedule date
for i in range(len(schedule)-1):
    d0, d1 = schedule[i], schedule[i+1]
    d = calendar.advance(d0, ql.Period(1, ql.Months))
    while d < d1:
        grid_dates.append(d)
        d = calendar.advance(d, ql.Period(1, ql.Months))

grid_dates = unique_sorted_dates(grid_dates)
grid_times = np.array([day_count.yearFraction(evaluation_date, d) for d in grid_dates], dtype=float)
assert abs(grid_times[0] - 0.0) < 1e-12, "First grid time should be t=0"

# ---------- LSM/MC parameters ----------
n_paths  = 20000
rng_seed = 42

# Frictions / wedges (set to zero for frictionless baseline)
bid_ask_bps  = 5.0   # cost to sell, in price bps (5 bps = 0.0005 * price)
capital_rate = 0.00  # continuous running capital charge when holding (per annum)

# ---------- Extract bond cashflows as (time, amount) ----------
def cashflows_time_amount(bond_obj):
    # Filter out cashflows strictly before the evaluation date.
    # Keep same-day flows (equivalent to hasOccurred(refDate, includeRefDate=False)).
    cfs = []
    for cf in bond_obj.cashflows():
        if cf.date() < evaluation_date:
            continue
        t = day_count.yearFraction(evaluation_date, cf.date())
        amt = cf.amount()
        cfs.append((t, amt))
    # Combine any that happen at the same time (coupon + redemption at maturity)
    time_to_amt = {}
    for t, a in cfs:
        time_to_amt[t] = time_to_amt.get(t, 0.0) + a
    cfs_unique = sorted(time_to_amt.items(), key=lambda x: x[0])
    return cfs_unique

cf_times_amts = cashflows_time_amount(bond)
cf_times = np.array([t for t, _ in cf_times_amts], dtype=float)
cf_amts  = np.array([a for _, a in cf_times_amts], dtype=float)

# Map coupon/redemption cash at grid times
time_to_index = {round(t, 10): i for i, t in enumerate(np.round(grid_times, 10))}
coupon_on_grid = np.zeros(len(grid_times))
for t, a in cf_times_amts:
    rt = round(t, 10)
    if rt in time_to_index:
        coupon_on_grid[time_to_index[rt]] += a
    else:
        # ensure all cashflow times are on the grid
        raise RuntimeError(f"Cashflow time t={t} not on decision grid.")

# ---------- Build model & process ----------
hw  = ql.HullWhite(funding_ts, a, sigma)
dyn = hw.dynamics()               # ShortRateDynamics
process = dyn.process()           # StochasticProcess1D for the state x_t

# Time grid for simulation
time_grid = ql.TimeGrid(list(grid_times))

# RNG & path generator
urng = ql.MersenneTwisterUniformRng(rng_seed)
usg  = ql.UniformRandomSequenceGenerator(len(time_grid)-1, urng)
grsg = ql.GaussianRandomSequenceGenerator(usg)
pg   = ql.GaussianPathGenerator(process, time_grid, grsg, False)

# Helper: ex‑coupon model price at time t_i for a given short rate r
# Uses Hull–White analytic zero‑bond pricing P(t,T|r). Ex‑coupon excludes CF at t_i.
def ex_coupon_price_at(hw_model, t_i, r_i):
    price = 0.0
    for (T, C) in cf_times_amts:
        if T > t_i + 1e-12:
            price += C * hw_model.discountBond(t_i, T, r_i)
    return price

# ---------- Simulate short-rate paths (x_t -> r_t) ----------
rates = np.zeros((n_paths, len(grid_times)), dtype=float)
for p in range(n_paths):
    path = pg.next().value()  # path of the state x_t
    for i in range(len(grid_times)):
        t = grid_times[i]
        x = path[i]
        rates[p, i] = dyn.shortRate(t, x)  # map to short rate r_t

# Per-step discount factors while holding (include optional capital charge)
dt = np.diff(grid_times)                                        # shape (n_steps,)
disc_hold = np.exp(-(rates[:, :-1] + capital_rate) * dt)        # shape (n_paths, n_steps)

# ---------- Longstaff–Schwartz backward induction ----------
V = np.zeros((n_paths, len(grid_times)), dtype=float)
V[:, -1] = coupon_on_grid[-1]  # value right before final cashflow = final cashflow

def basis(x):
    # Stable, low-order polynomial basis for regression
    return np.vstack([np.ones_like(x), x, x**2]).T

def sell_value(price):
    # Apply transaction cost (bid-ask)
    return price * (1.0 - bid_ask_bps * 1e-4)

# Backward loop over decision times
for i in range(len(grid_times)-2, -1, -1):
    t_i = grid_times[i]
    r_i = rates[:, i]

    # Discounted continuation if we hold through (i -> i+1)
    cont_disc = disc_hold[:, i] * V[:, i+1]

    # Regress E[cont_disc | r_i] using simple polynomial basis
    X = basis(r_i)
    coeff, *_ = np.linalg.lstsq(X, cont_disc, rcond=None)
    cont_est = X @ coeff

    # Immediate liquidation value: ex‑coupon price at (t_i, r_i), net of costs
    stop_val = np.array([sell_value(ex_coupon_price_at(hw, t_i, r)) for r in r_i])

    # Continue value includes any coupon at t_i (ex‑coupon sale excludes it)
    hold_now = coupon_on_grid[i] + cont_est

    # Optimal decision
    exercise = stop_val > hold_now
    V[:, i] = np.where(exercise, stop_val, coupon_on_grid[i] + cont_disc)

# The time-0 value is the average across paths
lsm_value = float(np.mean(V[:, 0]))

# ---------- Frictionless model value for comparison ----------
bond.setPricingEngine(ql.DiscountingBondEngine(funding_ts))
ql_price = bond.NPV()  # currency units for notional=100.0

print("------------------------------------------------------------")
print(f"LSM optimal liquidation value @ t=0: {lsm_value:,.6f}")
print(f"QuantLib bond NPV (frictionless):     {ql_price:,.6f}")
print("Settings:")
print(f"  paths={n_paths}, mesh points={len(grid_times)}, bid_ask_bps={bid_ask_bps:.1f}, capital_rate={capital_rate:.4f}")
print("------------------------------------------------------------")
