from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go  # charts only
from dashboards.components.portfolio_select import sidebar_portfolio_multi_select
import pytz
import datetime

from dashboards.apps.portfolio_competitive_landscape.context import AppContext
from dashboards.plots.theme import apply_graph_theme
from mainsequence.dashboards.streamlit.core.registry import register_page
from dashboards.plots.heatmap import plot_serialized_correlation

# Placeholders for empty inputs
TOKENS_PLACEHOLDER = "e.g., 91,M_BONOS,S_UDIBONO,LF_BONDES"
BUCKETS_PLACEHOLDER = "e.g., 365,730,1825"


@register_page("main_analysis", "Competition Analysis", has_sidebar=True)
def render(ctx: Optional[AppContext]):
    """Render the app: global sidebar (portfolio search + as‑of) + per-tab controls."""

    # ── Sidebar: only *global* controls ──────────────────────────────────────
    with st.sidebar:
        st.header("Select portfolios")
        loaded = sidebar_portfolio_multi_select(
            title="Peer portfolios (search & select)",
            key_prefix="pcl_portfolios",
            min_chars=3,
        )
        if loaded is not None:
            st.session_state["selected_portfolio_instances"] = loaded
            st.session_state["_run_pipeline"] = True
            st.success(f"Loaded {len(loaded)} portfolio(s).")
            st.rerun()

        st.divider()

    # ── If data/context not ready yet ────────────────────────────────────────
    if ctx is None:
        st.info("Select data source and *As‑of* in the sidebar, then change **Tipo/Clas** or click **Refresh state** to load data.")
        return

    # Use internal, sanitized prices (UTC-naive ms) from PortfoliosOperations
    prices_for_plots = ctx.portfolio_operations.prices

    # ── Tabs (always visible; each tab has its own first-row settings) ───────
    tab_names = [
        "Weights by token",
        "Maturity buckets",
        "Rolling leaders",
        "Serialized correlation",
        "Simulation",
        "Status",
    ]
    tabs = st.tabs(tab_names)

    # ─────────────────────────────────────────────────────────────────────────
    #  Tab 0: Weights by token
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Weights by token")
        st.markdown(
            "Use tokens to group instruments (ticker prefixes/labels) and display the **share of portfolio weight** "
            "captured by each token in the latest holdings snapshot."
        )

        # Defaults from init_session_for_scaffold
        default_tokens_csv = "91,M_BONOS,S_UDIBONO,LF_BONDES"
        default_match_mode = "contains"
        default_case_ins = True

        # First row: tab-specific controls with defaults + placeholders
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.text_input(
                "Identifier tokens (comma‑sep)",
                key="tokens_csv",
                value=st.session_state.get("tokens_csv", default_tokens_csv),
                placeholder=TOKENS_PLACEHOLDER,
            )
        with c2:
            MODE_OPTS = ["contains", "delimited", "prefix", "regex"]
            st.selectbox(
                "Match mode",
                MODE_OPTS,
                index=MODE_OPTS.index(st.session_state.get("match_mode", default_match_mode)),
                key="match_mode",
            )
        with c3:
            st.checkbox(
                "Case insensitive",
                value=st.session_state.get("case_insensitive", default_case_ins),
                key="case_insensitive",
            )

        tokens_csv_val = st.session_state.get("tokens_csv", default_tokens_csv)
        tokens = [t.strip() for t in tokens_csv_val.split(",") if t.strip()]

        # Render Streamlit table; also get the aggregated df back (kept for compatibility)
        agg_df = ctx.portfolio_operations.sum_signal_weights_table(
            tokens,
            keep_time_index=False,
            case_insensitive=st.session_state.get("case_insensitive", default_case_ins),
            match_mode=st.session_state.get("match_mode", default_match_mode),
            percent=True,
            decimals=2,
            title=None,
            render=True,
            return_df=True,
        )
        with st.expander("Show data (raw)"):
            st.dataframe(agg_df.reset_index(drop=False), width="stretch")

    # ─────────────────────────────────────────────────────────────────────────
    #  Tab 1: Maturity buckets
    # ─────────────────────────────────────────────────────────────────────────
    # Use internal Valmer panels + snapshots
    valmer_prices = ctx.portfolio_operations.valmer_prices_panel
    maturity_s = ctx.portfolio_operations.maturity_snapshot(datetime.datetime.now(pytz.utc))
    duration_s = ctx.portfolio_operations.duration_snapshot(datetime.datetime.now(pytz.utc))

    with tabs[1]:
        st.subheader("Weighted duration by maturity bucket")

        # Disclosure: show snapshot date used vs requested As‑of
        snap_m = maturity_s.name
        snap_d = duration_s.name
        snap_m_utc = pd.Timestamp(snap_m).tz_convert("UTC") if snap_m is not None else None
        snap_d_utc = pd.Timestamp(snap_d).tz_convert("UTC") if snap_d is not None else None
        snap_used_utc = None
        if snap_m_utc is not None or snap_d_utc is not None:
            candidates = [d for d in [snap_m_utc, snap_d_utc] if d is not None]
            snap_used_utc = max(candidates)
            if snap_m_utc is not None and snap_d_utc is not None and snap_m_utc.date() != snap_d_utc.date():
                st.caption(
                    f"Using Valmer snapshots — maturity **{snap_m_utc.strftime('%Y-%m-%d')} UTC**, "
                    f"duration **{snap_d_utc.strftime('%Y-%m-%d')} UTC**; "
                    f"analysis date = **{snap_used_utc.strftime('%Y-%m-%d')} UTC** "
                )
            else:
                st.caption(
                    f"Using Valmer snapshot **{snap_used_utc.strftime('%Y-%m-%d')} UTC** "
                )

        # First row: tab-specific controls with defaults + placeholders
        default_buckets_csv = "365,730,1825"
        default_weighted_mode = "contribution"

        c1, c2 = st.columns([2, 1])
        with c1:
            st.text_input(
                "Maturity bucket edges (days, comma‑sep)",
                key="buckets_csv",
                value=st.session_state.get("buckets_csv", default_buckets_csv),
                placeholder=BUCKETS_PLACEHOLDER,
            )
        with c2:
            WM_OPTS = ["contribution", "average"]
            st.selectbox(
                "Weighted duration mode",
                WM_OPTS,
                index=WM_OPTS.index(st.session_state.get("weighted_mode", default_weighted_mode)),
                key="weighted_mode",
            )

        # Parse tab settings
        buckets_csv_val = st.session_state.get("buckets_csv", default_buckets_csv)
        try:
            maturity_buckets = [int(x.strip()) for x in buckets_csv_val.split(",") if x.strip()]
        except Exception:
            maturity_buckets = []
        if not maturity_buckets:
            maturity_buckets = [int(x) for x in default_buckets_csv.split(",")]

        tokens_csv_val = st.session_state.get("tokens_csv", "91,M_BONOS,S_UDIBONO,LF_BONDES")
        tokens = [t.strip() for t in tokens_csv_val.split(",") if t.strip()]

        # Build per-portfolio raw tables (no rendering), then combine
        all_bucket_dfs: List[pd.DataFrame] = []
        portfolios = ctx.portfolio_operations.signals.index.get_level_values("portfolio_name").unique()

        for portfolio in portfolios:
            signals_for_portfolio = ctx.portfolio_operations.signals.loc[pd.IndexSlice[:, portfolio], :]
            if signals_for_portfolio.empty:
                continue

            raw_df = ctx.portfolio_operations.weighted_duration_by_maturity_bucket_table(
                maturity_series=maturity_s,
                duration_series=duration_s,
                unique_identifier_filter=tokens,
                maturity_bucket_days=maturity_buckets,
                signals=signals_for_portfolio,  # << use slice per portfolio
                case_insensitive=st.session_state.get("case_insensitive", True),
                match_mode=st.session_state.get("match_mode", "contains"),
                lowercase_token_index=True,
                weighted_mode=st.session_state.get("weighted_mode", default_weighted_mode),
                percent=False,      # get raw values; we will normalize after concatenation
                decimals=2,
                title=None,
                render=False,
                return_df=True,
            )

            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                part = raw_df.reset_index(drop=False).copy()
                part["portfolio_name"] = portfolio
                all_bucket_dfs.append(part)

        if all_bucket_dfs:
            combined_bkt_df = pd.concat(all_bucket_dfs, axis=0, ignore_index=True)

            # Attach the actual analysis (snapshot) date to each row for transparency
            analysis_date_str = snap_used_utc.strftime("%Y-%m-%d") if snap_used_utc is not None else None
            combined_bkt_df["snapshot_date"] = analysis_date_str

            # Normalize to PERCENTAGE within (portfolio_name, token) across buckets (safe division)
            wd = pd.to_numeric(combined_bkt_df["weighted_duration"], errors="coerce")
            denom = combined_bkt_df.groupby(["portfolio_name", "token"])["weighted_duration"].transform("sum")
            den = pd.to_numeric(denom, errors="coerce").replace(0, np.nan)
            pct = (wd / den).replace([np.inf, -np.inf], np.nan)
            combined_bkt_df["weighted_duration_pct"] = pct * 100.0  # float with NaNs

            # Order and render
            cols_disp = ["portfolio_name", "snapshot_date", "token", "maturity_bucket", "weighted_duration_pct"]
            cols_disp = [c for c in cols_disp if c in combined_bkt_df.columns]
            disp = combined_bkt_df[cols_disp].copy()

            st.dataframe(
                disp,
                width="stretch",
                hide_index=True,
                column_config={
                    "snapshot_date": "Snapshot",
                    "weighted_duration_pct": st.column_config.NumberColumn("Weighted duration (%)", format="%.2f"),
                },
            )
            with st.expander("Show data (raw)"):
                st.dataframe(combined_bkt_df, width="stretch")
        else:
            st.info("No maturity bucket data to display for the selected funds.")

    # ─────────────────────────────────────────────────────────────────────────
    #  Tab 2: Rolling leaders
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Rolling leaders — latest windows")

        # Defaults
        default_w3, default_w6, default_w9 = 63, 126, 189

        # First row: tab-specific controls with defaults
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input(
                "3M window (days)", min_value=10, max_value=400, step=1,
                value=int(st.session_state.get("w3", default_w3)), key="w3"
            )
        with c2:
            st.number_input(
                "6M window (days)", min_value=20, max_value=600, step=1,
                value=int(st.session_state.get("w6", default_w6)), key="w6"
            )
        with c3:
            st.number_input(
                "9M window (days)", min_value=30, max_value=800, step=1,
                value=int(st.session_state.get("w9", default_w9)), key="w9"
            )

        if prices_for_plots.shape[1] == 0:
            st.info("No portfolio prices available.")
        else:
            windows = {
                "3M": int(st.session_state.get("w3", default_w3)),
                "6M": int(st.session_state.get("w6", default_w6)),
                "9M": int(st.session_state.get("w9", default_w9)),
            }
            figs_leaders, latest_top3 = ctx.portfolio_operations.plot_competitive_leaders(
                prices_for_plots,
                windows=windows,
                always_show=ctx.always_show_final,
                title_prefix="Latest rolling window (base = 100)",
                legend_at_bottom=True,
                isolate_on_click=True,
                show_baseline=True,
            )
            for _, fig in figs_leaders.items():
                apply_graph_theme(fig, ctx.graph_theme)
                st.plotly_chart(fig, width="stretch")

            with st.expander("Latest Top‑3 per window"):
                for wl, s in latest_top3.items():
                    st.write(f"**{wl}**")
                    st.dataframe(s.rename("last_value").to_frame(), width="stretch")

    # ─────────────────────────────────────────────────────────────────────────
    #  Tab 3: Serialized correlation
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Serialized correlation (portfolio level)")
        if prices_for_plots.shape[1] <= 1:
            st.info("Need at least two portfolios with price history.")
        else:
            _, _, fig_heat, fig_dend = plot_serialized_correlation(prices_for_plots, mode="returns")
            apply_graph_theme(fig_heat, ctx.graph_theme)
            apply_graph_theme(fig_dend, ctx.graph_theme)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_heat, width="stretch")
            with c2:
                st.plotly_chart(fig_dend, width="stretch")

    # ─────────────────────────────────────────────────────────────────────────
    #  Tab 4: Simulation
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Simulation (notebook parity)")

        # Defaults
        default_sim_window = 180
        default_sim_min_obs = 120
        default_sim_months = 3
        default_sim_samples = 20000
        default_sim_norm = "sum"
        default_sim_excess = True

        # First row: tab-specific controls with defaults
        st.markdown("**Simulation settings**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input(
                "Lookback window (days)",
                min_value=60, max_value=1500, step=5,
                value=int(st.session_state.get("sim_window", default_sim_window)),
                key="sim_window",
            )
            st.number_input(
                "Min obs per asset",
                min_value=60, max_value=2000, step=10,
                value=int(st.session_state.get("sim_min_obs", default_sim_min_obs)),
                key="sim_min_obs",
            )
        with c2:
            st.number_input(
                "Horizon (months)",
                min_value=1, max_value=24, step=1,
                value=int(st.session_state.get("sim_N_months", default_sim_months)),
                key="sim_N_months",
            )
            NORM_OPTS = ["sum", "none"]
            st.selectbox(
                "Normalize weights",
                NORM_OPTS,
                index=NORM_OPTS.index(st.session_state.get("sim_norm", default_sim_norm)),
                key="sim_norm",
            )
        with c3:
            st.number_input(
                "Number of simulations",
                min_value=1000, max_value=200000, step=1000,
                value=int(st.session_state.get("sim_samples", default_sim_samples)),
                key="sim_samples",
            )
            st.checkbox(
                "Plot excess over target (center at 0)",
                value=st.session_state.get("sim_excess", default_sim_excess),
                key="sim_excess",
            )

        # Compute lead returns (YTD by default)
        lead = ctx.portfolio_operations.get_leading_portfolios(prices_for_plots)

        try:
            summary_df, rank_df, sim_ports, _, _, fig_iqr, fig_rank = ctx.portfolio_operations.simulate_portfolios_centered_on_targets(
                valmer_prices=valmer_prices,  # internal Valmer panel
                lead_returns=lead,
                window=int(st.session_state.get("sim_window", default_sim_window)),
                N_months=int(st.session_state.get("sim_N_months", default_sim_months)),
                samples=int(st.session_state.get("sim_samples", default_sim_samples)),
                normalize_weights=st.session_state.get("sim_norm", default_sim_norm),
                min_obs_per_asset=int(st.session_state.get("sim_min_obs", default_sim_min_obs)),
            )

            # Streamlit tables for summaries (no Plotly tables)
            if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                sum_disp = summary_df.copy()
                pct_keys = ("ret", "mean", "p", "prob", "quant", "median")
                for c in sum_disp.columns:
                    if any(k in c.lower() for k in pct_keys) and pd.api.types.is_numeric_dtype(sum_disp[c]):
                        sum_disp[c] = sum_disp[c] * 100.0
                st.subheader("Simulation Summary")
                st.dataframe(sum_disp, width="stretch")

            if isinstance(rank_df, pd.DataFrame) and not rank_df.empty:
                rank_disp = rank_df.copy()
                for c in rank_df.columns:
                    if any(k in c.lower() for k in ("rank", "prob", "p")) and pd.api.types.is_numeric_dtype(rank_df[c]):
                        rank_df[c] = rank_df[c] * 100.0
                st.subheader("Rank Probabilities")
                st.dataframe(rank_df, width="stretch")

            # Charts
            if fig_iqr:
                apply_graph_theme(fig_iqr, getattr(getattr(ctx, "theme", None), "graph", None))
                st.plotly_chart(fig_iqr, width="stretch")
            if fig_rank:
                apply_graph_theme(fig_rank, getattr(getattr(ctx, "theme", None), "graph", None))
                st.plotly_chart(fig_rank, width="stretch")

        except Exception as e:
            st.error(f"Simulation failed: {e}")

    st.caption("Scope: TARGET_COMB = ")
