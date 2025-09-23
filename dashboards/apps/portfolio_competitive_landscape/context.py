from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Dict
import pandas as pd
import streamlit as st

# MainSequence SDK imports
import mainsequence.client as ms_client
from dashboards.plots.theme import GraphTheme
from dashboards.services.portfolios import PortfoliosOperations


# ─────────────────────────────────────────────────────────────────────────────
#  Application Context Dataclass
# ─────────────────────────────────────────────────────────────────────────────



@dataclass
class AppContext:
    """Minimal payload used by the view."""
    cfg: Dict[str, Any]
    portfolio_operations: PortfoliosOperations
    always_show_final: List[str] = field(default_factory=list)
    sections: Dict[str, Any] = field(default_factory=dict)
    simulation_params: Dict[str, Any] = field(default_factory=dict)
    graph_theme: GraphTheme = field(default_factory=GraphTheme)

# ─────────────────────────────────────────────────────────────────────────────
#  Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────
def init_session_for_scaffold(ss) -> None:
    """Sets default values for all UI widgets in the session state."""
    ss.setdefault("peers_reload_token", 0)
    ss.setdefault("tipo", "EN INSTRUMENTOS DE DEUDA")
    ss.setdefault("clas", "MEDIANO PLAZO")
    ss.setdefault("tokens_csv", "91,M_BONOS,S_UDIBONO,LF_BONDES")
    ss.setdefault("match_mode", "contains")
    ss.setdefault("case_insensitive", True)
    ss.setdefault("buckets_csv", "365,730,1825")
    ss.setdefault("w3", 63)
    ss.setdefault("w6", 126)
    ss.setdefault("w9", 189)
    # Sections
    ss.setdefault("section_weight_sums", True)
    ss.setdefault("section_maturity_buckets", True)
    ss.setdefault("section_competitive_leaders", True)
    ss.setdefault("section_serialized_correlation", True)
    ss.setdefault("section_simulation", True)
    ss.setdefault("section_status_overview", True)
    # Sim params
    ss.setdefault("sim_window", 180)
    ss.setdefault("sim_N_months", 3)
    ss.setdefault("sim_samples", 20000)
    ss.setdefault("sim_min_obs", 120)
    ss.setdefault("sim_norm", "sum")
    ss.setdefault("sim_excess", True)

# ─────────────────────────────────────────────────────────────────────────────
#  Cached Data Loading Functions
# ─────────────────────────────────────────────────────────────────────────────






@st.cache_data(show_spinner="Fetching latest holdings...")
def _build_all_signals_from_latest(portfolios_index_assets: List[ms_client.PortfolioIndexAsset]) -> Tuple[pd.DataFrame, pd.Timestamp]:




    parts = []
    for ia in portfolios_index_assets:
        p=ia.reference_portfolio
        df = p.signal_local_time_serie.get_data_between_dates_from_api()
        if df is None or df.empty:
            continue
        df = df.copy()
        df["time_index"] = pd.to_datetime(df["time_index"], errors="coerce")
        df["portfolio_name"] = p.portfolio_name
        parts.append(df[["time_index", "portfolio_name", "unique_identifier", "signal_weight"]])

    if not parts:
        return pd.DataFrame(), pd.Timestamp.utcnow()
    all_signals = pd.concat(parts, axis=0)
    min_date = all_signals["time_index"].min()
    all_signals = (
        all_signals.sort_values(["portfolio_name","time_index"])
                   .loc[lambda d: d["time_index"].eq(d.groupby("portfolio_name")["time_index"].transform("max"))]
                   .sort_values(["portfolio_name","time_index"])
                   .reset_index(drop=True)
    )
    return all_signals, pd.to_datetime(min_date)







def _snapshot_series_asof(wide_df: pd.DataFrame, asof_ts: pd.Timestamp) -> pd.Series:
    if wide_df is None or wide_df.empty:
        return pd.Series(dtype=float)
    df2 = wide_df.copy()
    df2.index = pd.to_datetime(df2.index, utc=True, errors="coerce")
    sel = df2.loc[df2.index <= asof_ts]
    return sel.iloc[-1].dropna() if not sel.empty else pd.Series(dtype=float)



# ─────────────────────────────────────────────────────────────────────────────
#  Context Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_context(ss: dict) -> AppContext:
    """
    Main data pipeline. Reads config from session_state, runs computations,
    and returns an AppContext instance.
    """


    selected_portfolio_index_assets: List[Any] = ss.get("selected_portfolio_instances", []) or []

    if not selected_portfolio_index_assets:
        raise RuntimeError("No portfolios have been selected/loaded.")
    portfolio_list = [ia.reference_portfolio for ia in selected_portfolio_index_assets]
    portfolio_operations=    PortfoliosOperations(portfolio_list)


    always_show_final = [p.portfolio_name for p in portfolio_list]

    # Sections and sim params for the view
    sections = {k.replace("section_", ""): v for k, v in ss.items() if k.startswith("section_")}
    simulation_params = {k.replace("sim_", ""): v for k, v in ss.items() if k.startswith("sim_")}

    return AppContext(
        cfg={},
        portfolio_operations=portfolio_operations,

        always_show_final=always_show_final,
        sections=sections,
        simulation_params=simulation_params,
        graph_theme=GraphTheme(),
    )


def build_context_for_scaffold(session_state) -> Optional[AppContext]:
    """Scaffold hook: run pipeline on demand when portfolios are loaded."""
    if st.session_state.get("_run_pipeline", False) and st.session_state.get("selected_portfolio_instances"):
        # Reset trigger; compute fresh
        st.session_state["_run_pipeline"] = False
        return build_context(session_state)
    return None
