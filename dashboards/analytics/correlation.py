from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list

def compute_correlation(data: pd.DataFrame, mode: str = "returns", min_periods: int = 20) -> pd.DataFrame:
    if mode == "corr":
        corr = data.copy()
        cols = corr.columns
        return corr.loc[cols, cols]

    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.sort_index().replace([np.inf, -np.inf], np.nan)

    if mode == "prices":
        df = np.log(df).diff()

    corr = df.corr(min_periods=min_periods)
    np.fill_diagonal(corr.values, 1.0)
    return corr.dropna(how="all", axis=0).dropna(how="all", axis=1)

def serialized_correlation_core(
    data: pd.DataFrame,
    mode: str = "returns",
    cluster_method: str = "average",
):
    corr = compute_correlation(data, mode=mode).fillna(0)
    if corr.shape[0] < 2:
        raise ValueError("Need at least 2 assets to cluster/plot.")

    dist = np.sqrt(0.5 * (1 - corr.clip(-1, 1)))
    Z = linkage(squareform(dist.values, checks=False),
                method=cluster_method, optimal_ordering=True)

    order = leaves_list(Z)
    labels = corr.index.to_numpy()
    labels_ord = labels[order]
    corr_ord = corr.loc[labels_ord, labels_ord]
    return corr_ord, Z, labels_ord
