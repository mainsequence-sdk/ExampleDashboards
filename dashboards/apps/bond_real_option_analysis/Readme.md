# Bond Real‑Option: Optimal **Hold vs Sell & Invest** (LSM on Hull–White 1F)

This module prices the right to **liquidate** a bond (sell ex‑coupon) at a set of decision dates
and invest the proceeds to a fixed horizon. The choice at each decision time is:

- **Continue (Hold):** receive any coupon due *now* and carry on, or
- **Stop (Sell & Invest):** sell the bond **ex‑coupon**, pay bid–ask, invest proceeds until horizon on an “Invest” curve,
  and measure value using an **FTP** valuation curve.

Valuation uses **Longstaff–Schwartz (LSM)** on simulated **Hull–White 1‑factor (HW1F)** short‑rate paths.

- **Market curve**: drives HW dynamics and the **ex‑coupon sale price**.
- **FTP curve**: **numéraire** for continuation PVs.
- **Invest curve**: accrual on the stop branch.
- **Frictions**: bid–ask when selling; **capital charge** while holding.

> In a single‑curve, frictionless world (**Market = FTP = Invest**, bid–ask = 0, capital charge = 0),
> the option value is **exactly 0**.

---

## Contents

1. [Problem statement](#problem-statement)  
2. [Notation](#notation)  
3. [Assumptions](#assumptions)  
4. [Zero‑value (option = 0) cases](#zero-value-option--0-cases)  
5. [ALM wedges (Market vs FTP vs Invest)](#alm-wedges-market-vs-ftp-vs-invest)  
6. [Pricing equations](#pricing-equations)  
7. [LSM algorithm](#lsm-algorithm)  
8. [Hull–White model & simulation details](#hullwhite-model--simulation-details)  
9. [Ex‑coupon sale price formula](#excoupon-sale-price-formula)  
10. [Floating cashflows vs fixed](#floating-cashflows-vs-fixed)  
11. [Convergence checks & recommendations](#convergence-checks--recommendations)

---

## Problem statement

Given future cashflows \(\{(T_j, C_j)\}\) and a decision grid
\(0 = t_0 < t_1 < \cdots < t_N = H\), at each \(t_i\) one can:

- **Hold:** receive \(\text{coupon}_i\) (if any) and continue, or
- **Sell:** liquidate **ex‑coupon** at \(t_i\), pay bid–ask, invest to horizon \(H\).

Goal: **maximize FTP PV** at \(t=0\).

---

## Notation

- Curves and discount factors:
  - \(\mathcal{C}^{\text{mkt}}, \mathcal{C}^{\text{ftp}}, \mathcal{C}^{\text{inv}}\)
  - \(D^X(t)\): discount from 0 to \(t\) on curve \(X\);  
    \(\mathrm{DF}_X(t,s) \equiv D^X(s) / D^X(t)\) is the discount from \(t\) to \(s\).
- Bid–ask cost: \(k\) (price **bps**; proceeds are multiplied by \(1-k\cdot10^{-4}\)).
- Capital charge: \(c\) (continuous per‑annum factor applied while holding).
- HW1F short rate: \(r_t = x_t + \phi(t)\), with
  \[
    dx_t = -a\,x_t\,dt + \sigma\,dW_t, \qquad
    \phi(t) = f(0,t) + \frac{\sigma^2}{2a^2}\bigl(1-e^{-a t}\bigr)^2 .
  \]

---

## Assumptions

1. **Market‑consistent liquidation.** Ex‑coupon sale price is computed under the **Market** curve & HW model.
2. **FTP objective.** Continuation values are discounted on the **FTP** curve. Stop proceeds are converted to FTP PV via
   \[
     M_i \equiv \frac{\mathrm{DF}_{\text{FTP}}(t_i,H)}{\mathrm{DF}_{\text{INV}}(t_i,H)} .
   \]
3. **Frictions.** Selling incurs bid–ask \(k\) (bps of price).
4. **Capital charge.** While holding, continuation is attenuated by \(e^{-c\,\Delta t_i}\).
5. **Decision grid.** Exercise is allowed only on the grid (coupon dates plus optional monthly mesh).
6. **Floaters.** Future coupons follow the **current pricer** (“frozen coefficients”). See caveats.

---

## Zero‑value (option = 0) cases

The option is worth **zero** when these hold simultaneously:

- **Curves equal:** \(\mathcal{C}^{\text{mkt}}=\mathcal{C}^{\text{ftp}}=\mathcal{C}^{\text{inv}}\),
- **No frictions:** \(k=0\), \(c=0\).

Sketch: with single curve \(B_t\) as numéraire and ex‑coupon price \(P_t\),
\(\frac{P_t}{B_t}\) is a **martingale**. For any stopping time \(\tau\),
\(\mathbb{E}[B_\tau^{-1} P_\tau] = B_0^{-1} P_0\).
Thus **early liquidation adds no value**, and LSM matches deterministic NPV (up to MC noise).

---

## ALM wedges (Market vs FTP vs Invest)

If **FTP** and **Invest** differ from **Market**:

- **Stop branch** at \(t_i\):  
  ex‑coupon price \(P_{\text{ex}}(t_i,r_i)\) (market) × \((1-k)\) ×
  \[
    M_i = \frac{\mathrm{DF}_{\text{FTP}}(t_i,H)}{\mathrm{DF}_{\text{INV}}(t_i,H)} .
  \]
- **Hold branch**: discount continuation with **FTP** step DFs and apply capital charge.

These wedges make exercise regions non‑trivial and the option typically **positive**.

---

## Pricing equations

Let \(V_i(r_i)\) be value at grid time \(t_i\). Then:

- **Ex‑coupon sale (market model):**
  \[
    P_{\text{ex}}(t_i,r_i)=\sum_{j:\,T_j>t_i} C_j\,P^{\text{HW}}(t_i,T_j\mid r_i).
  \]
- **Stop value (ALM):** \(\text{Stop}_i(r_i)=(1-k)\,P_{\text{ex}}(t_i,r_i)\,M_i\).
- **Continuation value:**
  \[
    \text{Hold}_i(r_i) = \text{coupon}_i + \mathbb{E}\!\left[
      \mathrm{DF}_{\text{FTP}}(t_i,t_{i+1})\,e^{-c\,\Delta t_i}\,V_{i+1}(r_{i+1})
      \mid r_i \right].
  \]
- **Bellman recursion:** \(V_i(r_i)=\max\{\text{Stop}_i(r_i),\text{Hold}_i(r_i)\},\quad V_N=\text{coupon}_N.\)

---

## LSM algorithm

1. **Simulate** HW1F short‑rate paths \(r_0,\dots,r_N\).
2. **Terminal payoff:** \(V_N=\text{coupon}_N\).
3. **Backward** for \(i=N-1\to 0\):
   - Discount continuation: \(Y=\mathrm{DF}_{\text{FTP}}(t_i,t_{i+1})\,e^{-c\Delta t_i}\,V_{i+1}\).
   - **Regress** \(Y\) on basis \(\{1, r_i, r_i^2\}\) to estimate \(\widehat{\mathbb{E}}[Y \mid r_i]\).
   - Compute stop \(S=(1-k)\,P_{\text{ex}}(t_i,r_i)\,M_i\).
   - Set \(V_i=\max\{S, \ \text{coupon}_i+\widehat{\mathbb{E}}[Y\mid r_i]\}\).
4. **Estimate** \(V_0=\frac{1}{P}\sum_p V^{(p)}_0\).

---

## Hull–White model & simulation details

Exact OU step for \(x_t\):
\[
x_{i+1}=x_i\,e^{-a\Delta t_i}+\sigma\sqrt{\tfrac{1-e^{-2a\Delta t_i}}{2a}}\,Z_i,\quad Z_i\sim\mathcal{N}(0,1).
\]
Short rate: \(r_i = x_i + \phi(t_i)\) where
\[
\phi(t)= f(0,t) + \frac{\sigma^2}{2a^2}(1-e^{-a t})^2.
\]
The instantaneous forward \(f(0,t)\) is computed from the **Market** curve via
finite differences on **Dates** (robust to day‑count choices).

> **Numerics:** avoid \(a\) extremely close to 0; for \(a \lesssim 10^{-6}\) use the limit  
> \(\phi(t) \rightarrow f(0,t) + \tfrac{\sigma^2}{2}t^2\) and OU step \(\text{std} \rightarrow \sigma\sqrt{\Delta t}\).

---

## Ex‑coupon sale price formula

Under HW1F:
\[
P^{\text{HW}}(t,T\mid r_t)=A(t,T)\,e^{-B(t,T)\,r_t}, \quad B(t,T)=\frac{1-e^{-a(T-t)}}{a},
\]
with \(A(t,T)\) tied to \((a,\sigma)\) and the initial Market curve. **Ex‑coupon** excludes the coupon
that pays at \(t_i\). That coupon appears only in the **Hold** branch.

---

## Floating cashflows vs fixed

- **Fixed bonds:** exact; cashflows \(\{C_j\}\) are known.
- **Floaters:** future coupons depend on future fixings. Default approximation uses the **current pricer**
  (today’s forwards). If full path consistency is needed, extend the engine to **re‑forecast coupons**
  along the simulation grid.

---

## Convergence checks & recommendations

- **Frictionless sanity:** set \(k=0, c=0\) and Market = FTP = Invest ⇒ LSM ≈ QuantLib NPV.
- **Paths / mesh:** 10k–50k paths and monthly mesh are good starting points.
- **Regression basis:** quadratic in \(r\) is stable; expand cautiously.
- **Diagnostics:** inspect exercise share vs time and first exercise time distribution.
