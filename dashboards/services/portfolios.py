# dashboards/markets/portfolios.py
# -----------------------------------------------------------------------------
# PortfoliosOperations — single place for portfolio data engineering.
#
# Design goals:
# - Fail fast (raise on misuse; no UI stops).
# - All core data (signals, prices, Valmer panels) are internal cached
#   calculations that any method can reuse.
# - Pure(ish) data: the only UI helper kept is the optional Streamlit table
#   for token weights (to avoid breaking your view). Everything else returns
#   DataFrames/arrays/figures.
# -----------------------------------------------------------------------------

from __future__ import annotations

from functools import cached_property
from typing import List, Tuple, Any, Dict, Optional, Iterable, Sequence, Union, Mapping

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import streamlit as st
import math
import mainsequence.client as msc
from mainsequence.tdag import APIDataNode
import mainsequence.instruments as msi
import pytz
from dashboards.core.data_nodes import get_app_data_nodes
from mainsequence.instruments.instruments.position import Position

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (pure; no side effects)
# ─────────────────────────────────────────────────────────────────────────────

def ytd_returns(
    portfolio_prices: pd.DataFrame,
    all_signals: pd.DataFrame,
    *,
    as_percent: bool = False,
    round_to: int = 4,
) -> pd.Series:
    """
    Compute YTD returns for portfolios listed in all_signals.index['portfolio_name'].
    Uses first valid price on/after Jan 1 and the last valid price available.
    """
    cols = all_signals.index.get_level_values("portfolio_name").unique().tolist()
    cols = [c for c in cols if c in portfolio_prices.columns]
    if not cols:
        return pd.Series(dtype=float, name="ytd_return")

    df = portfolio_prices[cols].sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    end = df.index.max()
    start = pd.Timestamp(end.year, 1, 1)
    ytd = df.loc[df.index >= start]
    if ytd.empty:
        return pd.Series(index=cols, dtype=float, name="ytd_return")

    first = ytd.bfill().iloc[0]
    last = ytd.ffill().iloc[-1]
    ret = (last / first - 1).rename("ytd_return").round(round_to)
    return (ret * 100).round(round_to) if as_percent else ret

def _make_weighted_lines(
    assets: List[Any],
    weights: Dict[str, float],
    portfolio_notional: float,
) -> List[Dict[str, Any]]:
    """
    Turn (assets, weights, notional) into Position JSON lines with integer units.
    Allocation rule: floor to avoid exceeding the total notional.
    """

    ##Now because we are using Bonds and pricing with market reference curve this doesnt guarantee the right price
    # The super trick is to  get the  yield at which the bonds are actually trading and transform to an spread so when we bump the curves
    #then we can see the true impact
    deps = get_app_data_nodes()
    data_node = deps.get("instrument_pricing_table_id")

    #get las tobservation workflow move to main node  This should match the curve date!!!

    last_observation=data_node.get_last_observation(asset_list=assets)
    if last_observation.empty:
        raise Exception("No price for portfolio assets")
    else:
        # observation_time=last_observation.index.get_level_values("max")
        # publish_engine_meta("Pricing data", observation_time=getattr(asof, "date", lambda: asof)())
        dirty_price_map=last_observation.reset_index("time_index")["close"].to_dict()


    lines: List[Dict[str, Any]] = []
    for a in assets or []:
        uid = getattr(a, "unique_identifier", None)
        if uid is None:
            continue
        w = float((weights or {}).get(uid, 0.0))
        det = a.current_pricing_detail
        if det is None:
            continue
        instrument_dump=det.instrument_dump
        itype = instrument_dump["instrument_type"]
        inst_payload =instrument_dump["instrument"]
        if not itype or not isinstance(inst_payload, dict):
            # Malformed or unexpected shape; skip safely.
            continue

        target_nominal = w * float(portfolio_notional)
        raw_units = target_nominal / dirty_price_map[uid]
        units = int(math.floor(raw_units + 1e-12))  # integer holdings, deterministic


        lines.append({
            "instrument_type": itype,
            "instrument": inst_payload,
            "units": units,
            "extra_market_info":{"dirty_price":dirty_price_map[uid]}

        })
    return lines

def latest_snapshot_weights_core(
    all_signals: pd.DataFrame,
    asset_col: str = "unique_identifier",
    weight_col: str = "signal_weight",
    normalize: Optional[str] = None,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Timestamp], pd.Timestamp]:
    """
    Convert the last signal snapshot (per portfolio) into weight Series per portfolio.
    Returns:
      - dict: portfolio -> Series(index=unique_identifier, values=weights)
      - dict: portfolio -> last snapshot timestamp
      - common_end_date: min(last snapshot timestamp across portfolios)
    """
    df = all_signals.reset_index().copy()
    df["time_index"] = pd.to_datetime(df["time_index"], utc=True)

    last_ts = df.groupby("portfolio_name")["time_index"].max()
    df = df.merge(last_ts.rename("last_ts"), left_on="portfolio_name", right_index=True)
    latest = df[df["time_index"] == df["last_ts"]].copy()

    weights_by_port: Dict[str, pd.Series] = {}
    last_date_by_port: Dict[str, pd.Timestamp] = {}

    for p, g in latest.groupby("portfolio_name"):
        w = g.groupby(asset_col)[weight_col].sum().astype(float)
        if normalize == "sum":
            s = w.sum()
            if s != 0:
                w = w / s
        elif normalize == "abs":
            s = w.abs().sum()
            if s > 0:
                w = w / s
        weights_by_port[p] = w.sort_index()
        last_date_by_port[p] = g["time_index"].max()

    common_end_date = min(last_date_by_port.values())
    return weights_by_port, last_date_by_port, common_end_date


def _simulate_portfolios_core(
    all_signals: pd.DataFrame,
    valmer_prices: pd.DataFrame,             # wide DF: columns = unique_identifier
    lead_returns: Dict[str, float],          # horizon simple-return targets per portfolio
    window: int = 180,                       # trading days for covariance
    N_months: float = 1.0,                   # horizon in months (~21 trading days/month)
    samples: int = 20000,
    normalize_weights: Optional[str] = "sum",# 'sum', 'abs', or None
    min_obs_per_asset: int = 120,
    regularize_assets: float = 1e-6,         # ridge for asset Σ
    regularize_ports: float = 1e-10,         # tiny ridge for portfolio Σ (summary only)
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str], List[str]]:
    """
    Core engine that performs the joint simulation in **dataframe/array space only**.
    Returns: summary_df, rank_df, sim_ports, assets_final, portfolios
    """
    # ---------- Weights ----------
    weights_by_port, last_date_by_port, common_end_date = latest_snapshot_weights_core(
        all_signals, normalize=normalize_weights
    )
    portfolios = sorted(weights_by_port.keys())
    P = len(portfolios)
    if P == 0:
        raise ValueError("No portfolios in latest snapshot.")

    # ---------- Prices & returns window ----------
    prices = valmer_prices.sort_index()
    union_assets = sorted(set().union(*[set(w.index) for w in weights_by_port.values()]))
    assets_in_prices = [a for a in union_assets if a in prices.columns]
    if not assets_in_prices:
        raise ValueError("None of the snapshot assets are present in `valmer_prices` columns.")

    rets = prices[assets_in_prices].pct_change().loc[:common_end_date]
    rets = rets.tail(max(window * 3, window))
    enough = rets.notna().sum() >= min_obs_per_asset
    rets = rets.loc[:, enough.index[enough]]
    lr = rets.dropna().tail(window)
    if lr.shape[0] < max(30, min_obs_per_asset // 2):
        raise ValueError(f"Not enough clean rows for covariance (have {lr.shape[0]}).")

    assets_final = lr.columns.tolist()
    A = len(assets_final)

    # ---------- Σ_assets (daily -> horizon) ----------
    Sigma_assets_daily = lr.cov().astype(float).values

    # Ridge regularization for numerical stability
    if regularize_assets and regularize_assets > 0:
        Sigma_assets_daily += np.eye(A) * (regularize_assets * np.trace(Sigma_assets_daily) / A)

    N_days = int(round(21 * float(N_months)))
    Sigma_assets_N = Sigma_assets_daily * N_days

    # PSD guard
    def _to_psd(C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        C = 0.5 * (C + C.T)
        w, v = np.linalg.eigh(C)
        w = np.clip(w, eps, None)
        return (v * w) @ v.T
    Sigma_assets_N = _to_psd(Sigma_assets_N)

    # ---------- Build W over assets_final; drop empty ports ----------
    W_rows, valid_ports = [], []
    for p in portfolios:
        w = weights_by_port[p].reindex(assets_final).fillna(0.0).astype(float)
        if np.allclose(w.values, 0):
            continue
        if normalize_weights == "sum":
            s = w.sum()
            if s != 0:
                w = w / s
        elif normalize_weights == "abs":
            s = w.abs().sum()
            if s > 0:
                w = w / s
        W_rows.append(w.values)
        valid_ports.append(p)

    if not valid_ports:
        raise ValueError("All portfolios lost all holdings after alignment with covariance window.")
    portfolios, W = valid_ports, np.vstack(W_rows)  # (P x A)
    P = W.shape[0]

    # ---------- Σ_ports (for summary) ----------
    Sigma_ports_daily = W @ Sigma_assets_daily @ W.T
    Sigma_ports_N     = W @ Sigma_assets_N     @ W.T
    if regularize_ports and regularize_ports > 0:
        Sigma_ports_N += np.eye(P) * regularize_ports
    sigma_daily = np.sqrt(np.clip(np.diag(Sigma_ports_daily), 0, None))
    sigma_N     = np.sqrt(np.clip(np.diag(Sigma_ports_N),     0, None))

    # ---------- Mean calibration & asset-space simulation ----------
    mu_ports  = np.array([lead_returns.get(p, 0.0) for p in portfolios], dtype=float)  # horizon means
    mu_assets = np.linalg.pinv(W, rcond=1e-12) @ mu_ports
    rng = np.random.default_rng(random_seed)
    sim_assets = rng.multivariate_normal(mean=mu_assets, cov=Sigma_assets_N, size=samples)  # (S x A)
    sim_ports  = sim_assets @ W.T                                                            # (S x P)
    # Enforce exact portfolio centering on μ_ports
    sim_ports += (mu_ports - sim_ports.mean(axis=0))

    # ---------- Rank probabilities (absolute returns) ----------
    order = np.argsort(-sim_ports, axis=1)      # desc
    ranks = order.argsort(axis=1) + 1
    rank_probs = np.zeros((P, P))
    for k in range(1, P + 1):
        rank_probs[:, k - 1] = (ranks == k).mean(axis=0)
    rank_df = pd.DataFrame(rank_probs, index=portfolios, columns=[f"Rank {k}" for k in range(1, P + 1)])

    # ---------- Summary ----------
    def q(x, p): return float(np.quantile(x, p))
    rows = []
    for j, p in enumerate(portfolios):
        rs = sim_ports[:, j]
        rows.append({
            "portfolio": p,
            "n_holdings_used": int((W[j, :] != 0).sum()),
            "sigma_daily": float(sigma_daily[j]),
            "sigma_N": float(sigma_N[j]),
            "lead_return": float(mu_ports[j]),
            "mean_horizon_return": float(rs.mean()),
            "median_horizon_return": float(np.median(rs)),
            "p05_horizon_return": q(rs, 0.05),
            "p95_horizon_return": q(rs, 0.95),
        })
    summary_df = pd.DataFrame(rows).set_index("portfolio")

    return summary_df, rank_df, sim_ports, assets_final, portfolios


def _numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    ex = set(exclude or [])
    return [c for c in df.columns if c not in ex and pd.api.types.is_numeric_dtype(df[c])]


def _format_to_percent_strings(series: pd.Series, decimals: int) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").mul(100.0).round(decimals)
    return vals.map(lambda x: f"{x:.{decimals}f}%")


def _format_to_numeric_strings(series: pd.Series, decimals: int) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").round(decimals)
    return vals.map(lambda x: f"{x:.{decimals}f}")


def match_mask(
    series_uid: pd.Series,
    token: str,
    *,
    case_insensitive: bool = True,
    match_mode: str = "contains",
) -> pd.Series:
    """
    Build a boolean mask over `series_uid` to indicate which rows match `token`.
    match_mode: "contains" | "delimited" | "prefix" | "regex"
    """
    s = series_uid.astype(str)
    token_cmp = str(token)
    if case_insensitive and match_mode != "regex":
        s = s.str.lower()
        token_cmp = token_cmp.lower()

    if match_mode == "contains":
        pattern = re.escape(token_cmp)
    elif match_mode == "delimited":
        pattern = r"(?:^|_)" + re.escape(token_cmp) + r"(?:_|$)"
    elif match_mode == "prefix":
        pattern = r"^" + re.escape(token_cmp)
    elif match_mode == "regex":
        # honor caller's regex case preference
        return s.str.contains(token_cmp, regex=True, na=False, case=not case_insensitive)
    else:
        raise ValueError("match_mode must be one of: contains | delimited | prefix | regex")

    return s.str.contains(pattern, regex=True, na=False)


def bucketize_days(days: pd.Series, bucket_days: Sequence[int]) -> tuple[pd.Series, List[str]]:
    """
    Convert time-to-maturity in DAYS to labeled categorical buckets.

    Example:
      bucket_days=[365, 730] -> labels: ["0–365d", "365–730d", ">730d"]
    """
    edges = [0] + sorted(set(int(d) for d in bucket_days))
    bins = [-np.inf] + edges[1:] + [np.inf]
    labels = [f"{edges[i]}–{edges[i+1]}d" for i in range(len(edges) - 1)] + [f">{edges[-1]}d"]
    cats = pd.cut(days.clip(lower=0), bins=bins, labels=labels, right=True, include_lowest=True)
    return cats.astype("category"), labels


def weighted_duration_by_maturity_bucket_df(
    all_signals: pd.DataFrame,
    maturity_series: pd.Series,
    duration_series: pd.Series,
    unique_identifier_filter: Iterable[str],
    maturity_bucket_days: Sequence[int],
    *,
    asof: Union[pd.Timestamp, datetime.datetime, None] = None,
    case_insensitive: bool = True,
    match_mode: str = "contains",        # 'contains' catches LF_BONDES in LF_BONDESF
    lowercase_token_index: bool = True,
    weighted_mode: str = "contribution", # 'contribution' or 'average'
) -> pd.DataFrame:
    """
    Core computation for maturity-bucketed weighted duration.
    Returns a MultiIndex DataFrame indexed by (token, maturity_bucket) with a column 'weighted_duration'.
    """
    df = all_signals.reset_index()
    if "unique_identifier" not in df.columns or "signal_weight" not in df.columns:
        raise KeyError("`all_signals` must contain 'unique_identifier' and 'signal_weight'.")

    # Instrument info mapping
    m = pd.Series(maturity_series)
    d = pd.Series(duration_series)
    info = pd.DataFrame({"unique_identifier": m.index.astype(str)})

    # maturity as datetime (accept Unix seconds or datetime-like)
    if np.issubdtype(m.dtype, np.number):
        info["maturity_dt"] = pd.to_datetime(m.values.astype("float64"), unit="s", utc=True, errors="coerce")
    else:
        info["maturity_dt"] = pd.to_datetime(m.values, utc=True, errors="coerce")
    info["duration"] = d.reindex(m.index).values

    # as-of date
    if asof is None:
        asof_candidate = m.name if isinstance(m.name, (pd.Timestamp, datetime.datetime)) else None
        if asof_candidate is None and "time_index" in df.columns:
            asof_candidate = pd.to_datetime(df["time_index"]).max()
        if asof_candidate is None:
            asof_candidate = pd.Timestamp.utcnow()
        asof = pd.to_datetime(asof_candidate, utc=True)

    # days to maturity
    info["days_to_mty"] = (info["maturity_dt"] - asof).dt.total_seconds() / (24 * 3600)

    # Merge and precompute contributions
    merged = df.merge(info, on="unique_identifier", how="left")
    merged["duration_contrib"] = merged["signal_weight"] * merged["duration"]

    # Bucketize
    merged["maturity_bucket"], ordered_labels = bucketize_days(merged["days_to_mty"], maturity_bucket_days)

    # Compute per token
    s_uid = merged["unique_identifier"]
    results = []
    for tok in unique_identifier_filter:
        token_lbl = str(tok).lower() if lowercase_token_index else str(tok)
        mask = match_mask(s_uid, tok, case_insensitive=case_insensitive, match_mode=match_mode)
        sub = merged[mask]

        if weighted_mode == "contribution":
            val = sub.groupby("maturity_bucket", dropna=False, observed=False)["duration_contrib"].sum()
            val = val.reindex(ordered_labels, fill_value=0.0).rename("weighted_duration")
        elif weighted_mode == "average":
            w = sub.groupby("maturity_bucket", dropna=False, observed=False)["signal_weight"].sum().reindex(ordered_labels, fill_value=0.0)
            wd = sub.groupby("maturity_bucket", dropna=False, observed=False)["duration_contrib"].sum().reindex(ordered_labels, fill_value=0.0)
            val = pd.Series(np.where(w.values != 0, wd.values / w.values, 0.0),
                            index=w.index, name="weighted_duration")
        else:
            raise ValueError("weighted_mode must be 'contribution' or 'average'")

        block = val.to_frame()
        block.insert(0, "token", token_lbl)
        block = block.reset_index().set_index(["token", "maturity_bucket"])
        results.append(block)

    result = pd.concat(results).sort_index() if results else (
        pd.DataFrame(columns=["token", "maturity_bucket", "weighted_duration"])
          .set_index(["token", "maturity_bucket"])
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class PortfoliosOperations:
    """
    Cohesive engine for portfolio analytics. Everything you need lives here:
      - signals (latest snapshot by portfolio)
      - prices (wide: time x portfolio)
      - Valmer panels: prices, duration, maturity (time x unique_identifier)
      - ready-to-use analytics (weights by token, maturity buckets, simulation)

    All heavy state is exposed as cached properties to make downstream calls simple.
    """

    def __init__(self, portfolio_list: List[msc.Portfolio], *, valmer_node_identifier: str = "vector_de_precios_valmer"):
        self.portfolio_list = portfolio_list
        self._valmer_node_identifier = valmer_node_identifier

    def set_portfolio_positions(
            self,
            positions_by_id: Mapping[int, "Position"],


    ) -> None:
        """

        """
        self.portfolio_positions =positions_by_id



    # ---------- Raw signals loader (all rows) ----------
    @cached_property
    def _signals_all(self) -> pd.DataFrame:
        """All available signal rows (not just latest), one frame for all portfolios."""
        parts = []
        for p in self.portfolio_list:
            df = p.signal_local_time_serie.get_data_between_dates_from_api()
            if df is None or df.empty:
                continue
            df = df.copy()
            df["time_index"] = pd.to_datetime(df["time_index"], utc=True, errors="raise")
            df["portfolio_name"] = p.portfolio_name
            parts.append(df[["time_index", "portfolio_name", "unique_identifier", "signal_weight"]])

        if not parts:
            raise RuntimeError("No signals found for selected portfolios.")
        return pd.concat(parts, axis=0, ignore_index=True)

    # ---------- Public signals snapshot (latest per portfolio) ----------
    @cached_property
    def signals(self) -> pd.DataFrame:
        """
        Latest holdings snapshot for each portfolio.
        Index: ['time_index','portfolio_name'] (UTC-naive), columns: ['unique_identifier','signal_weight'].
        """
        all_signals = self._signals_all
        max_per_port = all_signals.groupby("portfolio_name")["time_index"].transform("max")
        snap = (
            all_signals.loc[all_signals["time_index"].eq(max_per_port)]
                       .sort_values(["portfolio_name", "time_index"])
                       .reset_index(drop=True)
        )
        return snap.set_index(["time_index", "portfolio_name"])

    @cached_property
    def min_date_signals(self) -> pd.Timestamp:
        """Earliest timestamp available across *all* signals (pre-latest filter)."""
        return pd.to_datetime(self._signals_all["time_index"]).min()

    @cached_property
    def unique_identifiers(self) -> List[str]:
        """Universe of assets from the latest snapshots."""
        return sorted(self.signals["unique_identifier"].astype(str).unique().tolist())

    # ---------- Prices (portfolio NAV/close) ----------
    @cached_property
    def prices(self) -> pd.DataFrame:
        """
        Wide price frame across the selected portfolios.
        Index: UTC-naive datetime (ms), Columns: portfolio_name.
        """
        frames: List[pd.Series] = []
        for p in self.portfolio_list:
            df = p.local_time_serie.get_data_between_dates_from_api()
            if df is None or len(df) == 0:
                continue
            df = df.copy()
            df["time_index"] = pd.to_datetime(df["time_index"], utc=True, errors="raise")
            s = (
                df.dropna(subset=["time_index"])
                  .sort_values("time_index")
                  .set_index("time_index")["close"]
                  .rename(p.portfolio_name)
            )
            frames.append(s)

        if not frames:
            raise RuntimeError("No portfolio price history available.")
        prices = pd.concat(frames, axis=1).sort_index().ffill()

        # Ensure Plotly-safe index (drop tz, floor to ms)
        idx = pd.to_datetime(prices.index, utc=True, errors="raise").tz_convert("UTC").tz_localize(None).floor("ms")
        prices = prices.copy()
        prices.index = idx
        return prices

    # Keep for backward-compat usage in the view (returns the cached 'prices')
    def build_portfolio_prices_from_instances(self) -> pd.DataFrame:
        return self.prices

    # ---------- Valmer panels (assets) ----------
    @cached_property
    def _valmer_panels(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch Valmer panels for all unique_identifiers starting at min_date_signals.
        Returns: (prices, duration, maturity)
        """
        if not self.unique_identifiers:
            raise RuntimeError("Valmer panels requested but no unique identifiers discovered.")
        node = APIDataNode.build_from_identifier(identifier=self._valmer_node_identifier)
        rd = {uid: {"start_date": self.min_date_signals, "start_date_operand": ">="} for uid in self.unique_identifiers}
        info = node.get_ranged_data_per_asset(range_descriptor=rd)
        piv = info.reset_index().pivot
        prices = piv(index="time_index", columns="unique_identifier", values="close")
        duration = piv(index="time_index", columns="unique_identifier", values="duracion")
        maturity = piv(index="time_index", columns="unique_identifier", values="fechavcto")
        return prices, duration, maturity

    @cached_property
    def valmer_prices_panel(self) -> pd.DataFrame:
        return self._valmer_panels[0]

    @cached_property
    def valmer_duration_panel(self) -> pd.DataFrame:
        return self._valmer_panels[1]

    @cached_property
    def valmer_maturity_panel(self) -> pd.DataFrame:
        return self._valmer_panels[2]

    # ---------- Convenience snapshots from panels ----------
    def maturity_snapshot(self, asof: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        Snapshot of Valmer 'fechavcto' as of `asof` (UTC now if None).
        """
        asof = pd.Timestamp.utcnow().tz_localize("UTC") if asof is None else pd.to_datetime(asof, utc=True)
        df2 = self.valmer_maturity_panel.copy()
        df2.index = pd.to_datetime(df2.index, utc=True, errors="raise")
        sel = df2.loc[df2.index <= asof]
        return sel.iloc[-1].dropna() if not sel.empty else pd.Series(dtype=float)

    def duration_snapshot(self, asof: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        Snapshot of Valmer 'duracion' as of `asof` (UTC now if None).
        """
        asof = pd.Timestamp.utcnow().tz_localize("UTC") if asof is None else pd.to_datetime(asof, utc=True)
        df2 = self.valmer_duration_panel.copy()
        df2.index = pd.to_datetime(df2.index, utc=True, errors="raise")
        sel = df2.loc[df2.index <= asof]
        return sel.iloc[-1].dropna() if not sel.empty else pd.Series(dtype=float)

    # ---------- Small utility getters to preserve existing call sites ----------
    def get_min_date_in_signals(self) -> pd.Timestamp:
        return self.min_date_signals

    def get_unique_indetifiers_in_portfolios(self) -> List[str]:
        # Keep misspelling for backward compatibility
        return self.unique_identifiers

    # ---------- Token weights ----------
    @staticmethod
    def sum_signal_weights_by_identifier(
        all_signals: pd.DataFrame,
        unique_identifier_filter: Iterable[str],
        *,
        keep_time_index: bool = False,
        case_insensitive: bool = True,
        match_mode: str = "contains",  # 'contains' | 'delimited' | 'prefix' | 'regex'
        lowercase_col_names: bool = True,
    ) -> pd.DataFrame:
        """
        Build a wide table (per portfolio_name, or per [time_index, portfolio_name])
        summing `signal_weight` where `unique_identifier` matches each token.

        Columns that do not match any token are grouped into an "Others" column.
        """
        required = {"unique_identifier", "signal_weight"}
        missing = required - set(all_signals.columns)
        if missing:
            raise KeyError(f"`all_signals` missing required columns: {sorted(missing)}")

        df = all_signals.reset_index()

        needed = ["portfolio_name"] + (["time_index"] if keep_time_index else [])
        for col in needed:
            if col not in df.columns:
                raise KeyError(f"`all_signals` must have '{col}' in the index or columns.")

        groupers = ["portfolio_name"] if not keep_time_index else ["time_index", "portfolio_name"]
        base_index = df[groupers].drop_duplicates().set_index(groupers).index

        if not unique_identifier_filter:
            # If no filter, everything is "Others"
            out = df.groupby(groupers, dropna=False, observed=False)["signal_weight"].sum().to_frame("Others")
            return out.sort_index()

        uid = df["unique_identifier"].astype(str)

        out_cols = []
        for tok in unique_identifier_filter:
            col_name = str(tok).lower() if lowercase_col_names else str(tok)
            mask = match_mask(uid, tok, case_insensitive=case_insensitive, match_mode=match_mode)
            s = (
                df.loc[mask]
                  .groupby(groupers, dropna=False, observed=False)["signal_weight"]
                  .sum()
                  .reindex(base_index, fill_value=0.0)
                  .rename(col_name)
            )
            out_cols.append(s)

        out = pd.concat(out_cols, axis=1)

        # Add "Others"
        total_weights = df.groupby(groupers, dropna=False)["signal_weight"].sum()
        matched_weights = out.sum(axis=1)
        out["Others"] = total_weights.subtract(matched_weights, fill_value=0)

        desired_order = [str(tok).lower() if lowercase_col_names else str(tok) for tok in unique_identifier_filter]
        if "Others" not in desired_order:
            desired_order.append("Others")

        return out.reindex(columns=desired_order).sort_index()

    def sum_signal_weights_table(
        self,
        unique_identifier_filter: Iterable[str],
        *,
        keep_time_index: bool = False,
        case_insensitive: bool = True,
        match_mode: str = "contains",
        lowercase_col_names: bool = True,
        percent: bool = True,   # display numeric columns as % by default
        decimals: int = 2,
        title: Optional[str] = None,
        render: bool = True,    # if True, render with Streamlit styling
        return_df: bool = False,# if True, return the aggregated DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Build and (optionally) render a Streamlit table of token weight sums.
        Returns the aggregated df if `return_df=True`, otherwise None.
        """
        agg_df = self.sum_signal_weights_by_identifier(
            self.signals,
            unique_identifier_filter,
            keep_time_index=keep_time_index,
            case_insensitive=case_insensitive,
            match_mode=match_mode,
            lowercase_col_names=lowercase_col_names,
        )
        disp = agg_df.reset_index(drop=False)

        # Format numeric cols
        num_cols = _numeric_columns(disp, exclude=["time_index", "portfolio_name"])
        if percent:
            for c in num_cols:
                disp[c] = _format_to_percent_strings(disp[c], decimals)
        else:
            for c in num_cols:
                disp[c] = _format_to_numeric_strings(disp[c], decimals)

        if render:
            if title:
                st.subheader(title)

            cols_to_convert = disp.columns.drop('portfolio_name')
            for col in cols_to_convert:
                disp[col] = disp[col].str.replace('%', '', regex=False).astype(float)

            st.dataframe(
                disp.style.background_gradient(
                    cmap='plasma',
                    subset=cols_to_convert
                ).format("{:.2f}%", subset=cols_to_convert)
            )

        return agg_df if return_df else None

    # ---------- Maturity buckets ----------
    def weighted_duration_by_maturity_bucket_table(
        self,
        maturity_series: Optional[pd.Series],
        duration_series: Optional[pd.Series],
        unique_identifier_filter: Iterable[str],
        maturity_bucket_days: Sequence[int],
        *,
        signals: Optional[pd.DataFrame] = None,  # allow per-portfolio slice
        case_insensitive: bool = True,
        match_mode: str = "contains",
        lowercase_token_index: bool = True,
        weighted_mode: str = "contribution",
        percent: bool = True,  # display as % share by bucket (default)
        decimals: int = 2,
        title: Optional[str] = None,
        render: bool = True,  # set False to only get the df back
        return_df: bool = False,  # if True, return the RAW df from the builder
    ) -> Optional[pd.DataFrame]:
        """
        Build and (optionally) render a Streamlit table of weighted durations by maturity bucket.
        - If `percent=True`, shows share of weighted_duration per group across buckets.
          Grouping priority: ['portfolio_name', 'token'] if present; else ['token']; else all rows.
        - Returns the RAW df from the builder when `return_df=True`.
        """
        # Default to internal snapshots if not provided
        if maturity_series is None:
            maturity_series = self.maturity_snapshot()
        if duration_series is None:
            duration_series = self.duration_snapshot()

        raw = weighted_duration_by_maturity_bucket_df(
            all_signals=(self.signals if signals is None else signals),
            maturity_series=maturity_series,
            duration_series=duration_series,
            unique_identifier_filter=unique_identifier_filter,
            maturity_bucket_days=maturity_bucket_days,
            case_insensitive=case_insensitive,
            match_mode=match_mode,
            lowercase_token_index=lowercase_token_index,
            weighted_mode=weighted_mode,
        )

        disp = raw.reset_index(drop=False).copy()

        # If nothing to show, render empty df gracefully
        if "weighted_duration" not in disp.columns or disp.empty:
            if render:
                if title:
                    st.subheader(title)
                st.dataframe(disp, use_container_width=True)
            return raw if return_df else None

        if percent:
            group_cols: List[str] = []
            if "portfolio_name" in disp.columns:
                group_cols.append("portfolio_name")
            if "token" in disp.columns:
                group_cols.append("token")

            if group_cols:
                denom = disp.groupby(group_cols)["weighted_duration"].transform("sum").replace(0, pd.NA)
            else:
                total = disp["weighted_duration"].sum()
                denom = pd.Series([total] * len(disp), index=disp.index).replace(0, pd.NA)

            disp["weighted_duration"] = (disp["weighted_duration"] / denom).astype(float)
            disp.rename(columns={"weighted_duration": "weighted_duration_share"}, inplace=True)
            disp["weighted_duration_share"] = _format_to_percent_strings(disp["weighted_duration_share"], decimals)
        else:
            disp["weighted_duration"] = _format_to_numeric_strings(disp["weighted_duration"], decimals)

        if render:
            if title:
                st.subheader(title)
            st.dataframe(disp, use_container_width=True)

        return raw if return_df else None

    # ---------- Simulation & Rolling leaders ----------
    def simulate_portfolios_centered_on_targets(
        self,
        valmer_prices: Optional[pd.DataFrame],  # if None, uses internal Valmer prices panel
        lead_returns: Dict[str, float],  # horizon simple-return targets per portfolio
        window: int = 180,  # trading days for covariance
        N_months: float = 1.0,  # horizon in months (~21 * months trading days)
        samples: int = 20000,
        normalize_weights: Optional[str] = "sum",  # 'sum', 'abs', or None
        min_obs_per_asset: int = 120,
        regularize_assets: float = 1e-6,  # ridge for asset Σ
        regularize_ports: float = 1e-10,  # tiny ridge for portfolio Σ (summary only)
        random_seed: int = 42,
    ):
        """
        Simulate portfolios jointly in asset space using a covariance estimated from history.
        Portfolio returns are built as R_p = W·R_assets. The distribution used for ranking
        and IQR visuals is the **absolute simulated portfolio returns**.

        Returns
        -------
        summary_df, rank_df, sim_ports, assets_final, portfolios, fig_iqr, fig_rank
        """
        vp = self.valmer_prices_panel if valmer_prices is None else valmer_prices
        summary_df, rank_df, sim_ports, assets_final, portfolios = _simulate_portfolios_core(
            all_signals=self.signals,
            valmer_prices=vp,
            lead_returns=lead_returns,
            window=window,
            N_months=N_months,
            samples=samples,
            normalize_weights=normalize_weights,
            min_obs_per_asset=min_obs_per_asset,
            regularize_assets=regularize_assets,
            regularize_ports=regularize_ports,
            random_seed=random_seed,
        )

        # ============================
        # Plotly figures (neutral styling; view can apply theme)
        # ============================
        gold, silver, bronze = "#D4AF37", "#C0C0C0", "#CD7F32"
        gray = "rgba(160,160,160,0.55)"
        P = len(portfolios)

        # ---- IQR chart (absolute simulated returns) ----
        sim = sim_ports  # absolute (not excess) returns
        q1s = np.quantile(sim, 0.25, axis=0)
        meds = np.quantile(sim, 0.50, axis=0)
        q3s = np.quantile(sim, 0.75, axis=0)
        mu_ports = np.array([lead_returns.get(p, 0.0) for p in portfolios], dtype=float)

        fig_iqr = go.Figure()
        fig_iqr.add_trace(go.Scatter(
            x=portfolios, y=meds, mode="markers",
            marker=dict(size=9, color="#FFFFFF"),
            error_y=dict(
                type="data", symmetric=False,
                array=(q3s - meds), arrayminus=(meds - q1s),
                color="#76B7FB", thickness=8, width=0
            ),
            customdata=np.stack([q1s, q3s], axis=1),
            hovertemplate=(
                "Portfolio: %{x}"
                "<br>Q1: %{customdata[0]:.2%}"
                "<br>Median: %{y:.2%}"
                "<br>Q3: %{customdata[1]:.2%}<extra></extra>"
            ),
            name="IQR (Q1–Q3) & median"
        ))
        fig_iqr.add_trace(go.Scatter(
            x=portfolios, y=mu_ports, mode="markers",
            marker=dict(size=10, symbol="diamond", color="#FFD166", line=dict(width=0)),
            hovertemplate="Portfolio: %{x}<br>Target: %{y:.2%}<extra></extra>",
            name="Target (lead_return)"
        ))
        fig_iqr.update_layout(
            title=f"IQR of simulated portfolio returns (absolute) — N_months={N_months}, window={window}",
            xaxis_title="Portfolio",
            yaxis_title="Horizon simple return",
            yaxis=dict(tickformat=".1%", zeroline=True, zerolinewidth=1, zerolinecolor="rgba(255,255,255,0.25)"),
            margin=dict(l=60, r=40, t=60, b=100),
            width=max(560, min(1600, 32 * P + 260)),
            hovermode="x",
            legend=dict(orientation="h", y=1.02, x=0)
        )
        fig_iqr.update_xaxes(
            tickangle=90,
            type="category",
            categoryorder="array",
            categoryarray=portfolios,
            tickmode="array",
            tickvals=portfolios,
            ticktext=portfolios
        )

        # ---- Rank probabilities ----
        rank_probs = rank_df.to_numpy()
        fig_rank = go.Figure()
        for k in range(P):
            color_k = gold if k == 0 else silver if k == 1 else bronze if k == 2 else gray
            fig_rank.add_trace(
                go.Bar(
                    x=portfolios,
                    y=rank_probs[:, k],
                    name=f"Rank {k + 1}",
                    marker=dict(color=color_k),
                    hovertemplate=f"Portfolio: %{{x}}<br>P(Rank {k + 1}): %{{y:.1%}}<extra></extra>",
                )
            )
        fig_rank.update_layout(
            title=f"Probability of finishing at each rank (N_months={N_months})",
            xaxis_title="Portfolio",
            yaxis_title="Probability",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            barmode="stack",
            legend=dict(orientation="h", y=1.02, x=0),
            margin=dict(l=60, r=40, t=60, b=60)
        )

        return summary_df, rank_df, sim_ports, assets_final, portfolios, fig_iqr, fig_rank

    def plot_competitive_leaders(
        self,
        portfolio_prices: pd.DataFrame,
        windows={"3M": 63, "6M": 126, "9M": 189},  # ~21 trading days/month
        always_show=None,  # list of tickers to always plot in light blue
        title_prefix="Latest rolling window (base = 100 at window start)",
        legend_at_bottom: bool = True,
        isolate_on_click: bool = True,  # single click isolates; double click toggles
        show_baseline: bool = True
    ):
        """
        Plot only the MOST RECENT window per horizon, normalizing the left edge to 100.
        - Highlight Top 1/2/3 in gold/silver/bronze
        - Always show selected tickers in light blue
        - Legend at the bottom; single-click isolate (configurable)
        Returns: (figs: dict[label->Figure], latest_top3: dict[label->Series])
        """
        pp = portfolio_prices.copy()
        pp.index = pd.to_datetime(pp.index)
        pp = pp.sort_index().dropna(axis=1, how="all")
        if pp.empty or pp.shape[1] == 0:
            raise ValueError("No price data after cleaning.")

        always_set = set(always_show or [])
        missing = sorted(list(always_set - set(pp.columns)))
        if missing:
            print(f"[info] always_show tickers not found and will be ignored: {missing}")

        figs: Dict[str, go.Figure] = {}
        latest_top3: Dict[str, pd.Series] = {}

        gold, silver, bronze = "#D4AF37", "#C0C0C0", "#CD7F32"
        grey, light_blue = "rgba(160,160,160,0.55)", "#76B7FB"
        top_colors = {1: gold, 2: silver, 3: bronze}

        for wl, n in windows.items():
            if n <= 0:
                continue
            window_len = min(len(pp), n + 1)
            if window_len < 2:
                continue

            block = pp.tail(window_len).copy()

            # Ensure a non-NaN base per series; fall back to first valid in window
            base = block.iloc[0].copy()
            for col in block.columns:
                if pd.isna(base[col]):
                    fv = block[col].first_valid_index()
                    if fv is not None:
                        base_val = block.at[fv, col]
                        base[col] = base_val
                        block.iat[0, block.columns.get_loc(col)] = base_val

            valid_cols = base.dropna().index
            if len(valid_cols) == 0:
                continue
            block = block[valid_cols].ffill()
            base = base[valid_cols]

            # Normalize to 100 at the start of the window
            norm = (block / base) * 100.0

            # Current leaders by last value
            last_vals = norm.iloc[-1].dropna()
            ranked = last_vals.sort_values(ascending=False)
            top_names = ranked.index[:3].tolist()
            latest_top3[wl] = ranked.head(3)

            def line_style(col: str):
                is_top = col in top_names
                in_always = col in always_set
                if in_always:
                    color = light_blue
                    width = 3 if is_top else 2.2
                    rank_pos = top_names.index(col) + 1 if is_top else None
                    return color, width, rank_pos
                rank_pos = top_names.index(col) + 1 if is_top else None
                color = top_colors.get(rank_pos, grey)
                width = 3 if is_top else 1.2
                return color, width, rank_pos

            fig = go.Figure()
            last_date = norm.index.max()
            pad_days = max(3, n // 20)  # right-edge label room
            x_range = [norm.index.min(), last_date + pd.Timedelta(days=pad_days)]

            # Right-edge labels that hide/show with their line (legendgroup)
            y_last = norm.iloc[-1]
            order = y_last.dropna().sort_values().index.tolist()
            jitter_map = {name: i for i, name in enumerate(order)}
            jitter_scale = 0.04 * (np.nanmax(y_last) - np.nanmin(y_last) + 1e-9)

            for col in norm.columns:
                series = norm[col]
                if series.dropna().empty:
                    continue
                color, width, rank_pos = line_style(col)
                fig.add_trace(go.Scatter(
                    x=series.index, y=series.values, name=col,
                    legendgroup=col, mode="lines",
                    line=dict(width=width, color=color),
                    hovertemplate=(
                        f"{col}"
                        "<br>Date: %{x|%Y-%m-%d}"
                        "<br>Index (base=100 at window start): %{y:.2f}<extra></extra>"
                    ),
                    showlegend=True
                ))
                yv = y_last.get(col)
                if pd.isna(yv):
                    continue
                fig.add_trace(go.Scatter(
                    x=[last_date + pd.Timedelta(days=pad_days * 0.95)],
                    y=[yv + jitter_scale * (jitter_map[col] % 3 - 1)],
                    text=[f"{col}" + (f"  ({rank_pos})" if rank_pos else "")],
                    mode="text", textposition="middle left",
                    textfont=dict(size=12, color=color),
                    hoverinfo="skip", legendgroup=col, showlegend=False,
                    cliponaxis=False
                ))

            if show_baseline:
                fig.add_hline(y=100, line_width=1, line_dash="dot", opacity=0.6)

            legend_kwargs = dict(
                orientation="h", xanchor="left", x=0.0,
                yanchor="top", y=-0.20,  # below x-axis
                traceorder="normal",
            )
            if isolate_on_click:
                legend_kwargs.update(itemclick="toggleothers", itemdoubleclick="toggle")

            fig.update_layout(
                title=f"{title_prefix} — {wl}",
                xaxis_title="Date", yaxis_title="Index (100 = price at window start)",
                xaxis=dict(range=x_range, showgrid=False), yaxis=dict(zeroline=False),
                hovermode="x unified",
                margin=dict(l=60, r=160, t=60, b=140 if legend_at_bottom else 60),
                legend=legend_kwargs if legend_at_bottom else dict(),
            )
            figs[wl] = fig

        return figs, latest_top3

    # ---------- Small convenience ----------
    def get_leading_portfolios(self, portfolio_prices: pd.DataFrame) -> Dict[str, float]:
        return ytd_returns(portfolio_prices, self.signals).to_dict()

    def _build_position_from_portfolio(self,
            portfolio:msc.Portfolio,
            portfolio_notional: float,
    ) -> Tuple[Any, Dict[str, Any], str, Dict[str, Any]]:
        """
        Build a Position/config from a MainSequence portfolio.
        Returns: (position, cfg_dict, cfg_path, meta)
          - position is None (we instantiate later via existing context builder)
          - cfg_path is ':memory:' (signals “use session template”)
        """
        import pytz

        latest_weights,weights_date= portfolio.get_latest_weights()
        uids = list(latest_weights.keys())
        if not uids:
            raise ValueError("Selected portfolio has no latest weights.")
        assets = msc.Asset.filter(unique_identifier__in=uids)
        if not assets:
            raise ValueError("Could not resolve assets for latest weights.")

        total_notional = float(portfolio_notional)
        lines = _make_weighted_lines(assets, latest_weights, total_notional)
        if not lines:
            raise ValueError("No valid instrument lines could be built from portfolio assets.")

        pos_obj = msi.Position.from_json_dict({"lines": lines})
        pos_obj.position_date=weights_date
        aid_to_asset = {a.id: a for a in assets}
        instrument_hash_to_asset = {
            l.instrument.content_hash(): aid_to_asset[l.instrument.main_sequence_asset_id] for l in pos_obj.lines
        }



        return instrument_hash_to_asset,pos_obj

    def get_all_portfolios_as_positions(self,portfolio_notional):
        all_instrument_hash={}
        all_positions={}
        with st.spinner("Building position from selected portfolio(s)…"):
            for p in self.portfolio_list:
                instrument_hash_to_asset, portfolio_position =self._build_position_from_portfolio(p, portfolio_notional)
                all_instrument_hash.update(instrument_hash_to_asset)
                all_positions[p.id] = portfolio_position

            st.success(f"✓ Loaded position from portfolio(s) (in‑memory template)")

        return  all_instrument_hash,all_positions

        # ---------------------------------------------------------------------
        # Positions table — SAME table as before + extra column 'portfolio_name'
        # No bumping logic here; if you pass a bumped_position, it's only used
        # to show the bumped/Δ columns. This class does not compute bumps.
        # ---------------------------------------------------------------------

    def st_position_npv_table_paginated(
            self,
            *,
            position: Position,
            instrument_hash_to_asset: Mapping[object, msc.Asset] | None,
            bumped_position: Position | None = None,
            page_size_options: tuple[int, ...] = (25, 50, 100, 200),
            default_size: int = 50,
            enable_search: bool = True,
            key: str = "npv_table",
    ) -> None:
        import pandas as pd
        import streamlit as st
        import math

        # --- resolve portfolio_name for this position (via stored map) ---
        def _portfolio_name_for_position() -> str:
            pid = None
            for k, v in getattr(self, "portfolio_positions", {}).items():
                if v is position:
                    pid = k
                    break
            if pid is not None:
                for p in self.portfolio_list:
                    if getattr(p, "id", None) == pid:
                        # prefer 'portfolio_name' if present (as used elsewhere in your file)
                        return getattr(p, "portfolio_name", getattr(p, "name", f"Portfolio {pid}"))
                return f"Portfolio {pid}"
            # fallback: single active portfolio, if any
            if self.portfolio_list:
                p0 = self.portfolio_list[0]
                return getattr(p0, "portfolio_name", getattr(p0, "name", "Portfolio"))
            return "Portfolio"

        portfolio_name = _portfolio_name_for_position()

        # --- index bumped lines by content hash (string) if provided ---
        bump_idx: dict[str, Any] = {}
        if bumped_position is not None:
            for bl in bumped_position.lines:
                try:
                    bump_idx[str(bl.instrument.content_hash())] = bl
                except Exception:
                    pass

        # --- build rows ---
        rows: list[dict] = []
        for ln in position.lines:
            inst = ln.instrument
            # key used for asset map lookup: KEEP the original hash object (not str)
            try:
                hash_key = inst.content_hash()
                hash_str = str(hash_key)
            except Exception:
                hash_key, hash_str = None, None

            units = float(getattr(ln, "units", 0.0) or 0.0)

            # base price / npv
            try:
                price_base = float(inst.price())
            except Exception:
                price_base = float("nan")
            npv_base = units * price_base if price_base == price_base else float("nan")

            # bumped (only if provided; no computation here)
            price_bumped = None
            npv_bumped = None
            npv_delta = None
            if hash_str and hash_str in bump_idx:
                try:
                    price_bumped = float(bump_idx[hash_str].instrument.price())
                    npv_bumped = units * price_bumped
                    npv_delta = npv_bumped - npv_base
                except Exception:
                    pass

            # unique_identifier comes from the provided asset map (no JSON fallbacks)
            uid = None
            if instrument_hash_to_asset is not None and hash_key in instrument_hash_to_asset:
                try:
                    uid = getattr(instrument_hash_to_asset[hash_key], "unique_identifier", None)
                except Exception:
                    uid = None

            ms_asset_id = getattr(inst, "main_sequence_asset_id", None)
            inst_type = getattr(inst, "instrument_type", type(inst).__name__)

            # 'Details' link: route recognizes ?asset_id=
            details_url = f"?asset_id={hash_str or (ms_asset_id if ms_asset_id is not None else '')}"

            rows.append(
                {
                    "portfolio_name": portfolio_name,
                    "instrument_type": inst_type,
                    "unique_identifier": uid,
                    "content_hash": hash_str,
                    "ms_asset_id": ms_asset_id,
                    "units": units,
                    "price_base": price_base,
                    "price_bumped": price_bumped,
                    "npv_base": npv_base,
                    "npv_bumped": npv_bumped,
                    "npv_delta": npv_delta,
                    "details": details_url,
                }
            )

        if not rows:
            st.info("Positions table is empty.")
            return

        df = pd.DataFrame(rows)

        # --- search ---
        if enable_search:
            q = st.text_input("Search", value="", key=f"{key}_search", placeholder="Filter by portfolio, UID, type…")
            if q:
                ql = q.strip().lower()
                mask = (
                        df["portfolio_name"].fillna("").str.lower().str.contains(ql)
                        | df["instrument_type"].fillna("").str.lower().str.contains(ql)
                        | df["unique_identifier"].fillna("").astype(str).str.lower().str.contains(ql)
                        | df["content_hash"].fillna("").astype(str).str.lower().str.contains(ql)
                        | df["ms_asset_id"].fillna("").astype(str).str.lower().str.contains(ql)
                )
                df = df.loc[mask].reset_index(drop=True)

        # --- pagination ---
        page_size_options = tuple(page_size_options) if page_size_options else (25, 50, 100, 200)
        if default_size not in page_size_options:
            default_size = page_size_options[0]
        page_size = st.selectbox("Rows per page", page_size_options,
                                 index=page_size_options.index(default_size), key=f"{key}_pagesize")

        total = len(df)
        pages = max(1, math.ceil(total / page_size))
        curr_page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key=f"{key}_page")
        start = (curr_page - 1) * page_size
        end = start + page_size
        page_df = df.iloc[start:end].reset_index(drop=True)
        currency_symbol=None
        # --- render ---
        st.data_editor(
            page_df,
            use_container_width=True,
            hide_index=True,
            disabled=True,
            column_config={
                "portfolio_name": st.column_config.TextColumn("Portfolio"),
                "instrument_type": st.column_config.TextColumn("Type"),
                "unique_identifier": st.column_config.TextColumn("Unique ID"),
                "content_hash": st.column_config.TextColumn("Content hash"),
                "ms_asset_id": st.column_config.NumberColumn("MS Asset ID"),
                "units": st.column_config.NumberColumn("Units", format="%.2f"),
                "price_base": st.column_config.NumberColumn("Price (base)", format="%.6f"),
                "price_bumped": st.column_config.NumberColumn("Price (bumped)", format="%.6f"),
                "npv_base": st.column_config.NumberColumn(
                    f"NPV (base){' ' + currency_symbol if currency_symbol else ''}", format="%,.2f"
                ),
                "npv_bumped": st.column_config.NumberColumn(
                    f"NPV (bumped){' ' + currency_symbol if currency_symbol else ''}", format="%,.2f"
                ),
                "npv_delta": st.column_config.NumberColumn(
                    f"ΔNPV{' ' + currency_symbol if currency_symbol else ''}", format="%,.2f"
                ),
                "details": st.column_config.LinkColumn("Details"),
            },
            column_order=[
                "portfolio_name",
                "instrument_type",
                "unique_identifier",
                "content_hash",
                "ms_asset_id",
                "units",
                "price_base",
                "price_bumped",
                "npv_base",
                "npv_bumped",
                "npv_delta",
                "details",
            ],
            key=f"{key}_editor",
        )

        # CSV (filtered, without 'details' link)
        csv_cols = [c for c in df.columns if c != "details"]
        st.download_button(
            "Download CSV",
            data=df[csv_cols].to_csv(index=False).encode("utf-8"),
            file_name="positions_npv.csv",
            mime="text/csv",
            key=f"{key}_csv",
        )