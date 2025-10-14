# dashboards/apps/floating_portfolio_analysis/pages/01_curve_and_positions.py
from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List

import streamlit as st
import QuantLib as ql

from dashboards.components.portfolio_select import sidebar_portfolio_multi_select
from dashboards.services.portfolios import PortfoliosOperations
from dashboards.services.positions import PositionOperations
from dashboards.components.date_selector import date_selector
from dashboards.core.ql import qld
import mainsequence.client as msc
import mainsequence.instruments as msi

from dashboards.core.data_nodes import get_app_data_nodes
from dashboards.components.curve_bumps_and_stats import st_curve_bumps
from dashboards.plots.heatmap import plot_serialized_correlation

from mainsequence.dashboards.streamlit.scaffold import PageConfig, run_page
import dashboards.components.portfolio_select as portsel
import pandas as pd
import numpy as np
# Page setup
ctx = run_page(PageConfig(
    title="Portfolio Competition Analysis",
    use_wide_layout=True,
    inject_theme_css=True,
))
# Ensure "render once" counter is reset on page run
st.session_state.pop("__npv_tables_rendered", None)


# --- Mock portfolio build: run ONLY on explicit click -----------------------
def _request_mock_build() -> None:
    """Set a flag; we will act on it later in this run."""
    st.session_state["_req_build_mock_portfolio"] = True

def _do_build_mock_if_requested() -> None:
    """Execute the build only if explicitly requested via the flag."""
    if not st.session_state.pop("_req_build_mock_portfolio", False):
        return
    try:
        # Lazy import to avoid any import-time side effects.
        from dashboards.helpers.mock import build_test_portfolio as _build_test_portfolio
    except Exception:
        st.error("`dashboards.helpers.mock.build_test_portfolio` is not available in this environment.")
        return
    try:
        with st.spinner("Building mock portfolio 'mock_portfolio_floating_dashboard'…"):
            _build_test_portfolio("mock_portfolio_floating_dashboard")
        st.session_state["__just_built_mock_portfolio__"] = True
        st.rerun()
    except Exception as e:
        st.exception(e)

def _dedupe_portfolios(port_list: List[Any]) -> List[Any]:
    """Keep the last occurrence per id; ignore items without a valid id."""
    seen: Dict[int, Any] = {}
    for p in port_list:
        pid_raw = getattr(p, "id", None)
        if pid_raw is None:
            continue
        try_id = int(pid_raw) if str(pid_raw).lstrip("-").isdigit() else None
        if try_id is None:
            continue
        seen[try_id] = p
    return list(seen.values())

def _flatten_group_portfolios(group_obj: Any) -> List[Any]:
    """
    PortfolioGroup.portfolios may contain ints or Portfolio objects.
    Resolve ints via PortfolioIndexAsset; keep objects as-is.
    """
    out: List[Any] = []
    plist = getattr(group_obj, "portfolios", []) or []
    for entry in plist:
        if hasattr(entry, "id"):  # already a Portfolio-like object
            out.append(entry)
        else:
            s = str(entry)
            if s.lstrip("-").isdigit():
                p = msc.PortfolioIndexAsset.get_or_none(id=int(s))
                if p is not None:
                    out.append(p)
    return out

# --- Minimal data-nodes bootstrap (needed by portfolios.py) ---
def _ensure_data_nodes() -> None:
    if st.session_state.get("_deps_bootstrapped"):
        return
    deps = get_app_data_nodes()
    try:
        deps.get("instrument_pricing_table_id")
    except KeyError:
        from dashboards.apps.bond_portfolio_analysis.settings import PRICES_TABLE_NAME
        deps.register(instrument_pricing_table_id=PRICES_TABLE_NAME)
    st.session_state["_deps_bootstrapped"] = True

_ensure_data_nodes()

def _get_cfg_from_session(ss) -> Dict[str, Any]:
    return dict(ss.get("position_cfg_mem") or {})

# ---------- Sidebar ----------
with st.sidebar:
    _ = date_selector(
        label="Valuation date",
        session_cfg_key="position_cfg_mem",
        cfg_field="valuation_date",
        key="valuation_date_input",
        help="Controls pricing date and curve construction.",
    )

    notional_value = st.number_input(
        "Portfolio notional (base currency)",
        min_value=1.0,
        step=100_000.0,
        value=float(st.session_state.get("portfolio_notional", 1_000_000.0)),
        format="%.0f",
        help="Weights will be translated into integer holdings using this notional.",
        key="portfolio_notional",
    )

    st.divider()

    st.button("Build mock portfolios", key="btn_build_mock_portfolio", use_container_width=True,
              on_click=_request_mock_build)
    _do_build_mock_if_requested()

    selected_instances = sidebar_portfolio_multi_select(
        title="Portfolios (search)",
        key_prefix="fpa_portfolios",
        min_chars=3,
    )

    # Optional: also search/select portfolio GROUPS, then flatten to portfolios.
    include_groups = st.toggle("Include portfolio groups", value=True, key="fpa_include_groups")
    selected_group_instances = []
    if include_groups and hasattr(portsel, "sidebar_portfolio_group_multi_select"):
        selected_group_instances = portsel.sidebar_portfolio_group_multi_select(
            title="Portfolio groups (search & multi-select)",
            key_prefix="fpa_groups",
            min_chars=3,
        )

    if selected_instances or selected_group_instances:
        # 1) direct portfolios from the portfolio picker
        selected_ports: List[Any] = []
        for ia in (selected_instances or []):
            refp = getattr(ia, "reference_portfolio", None)
            if refp is not None:
                selected_ports.append(refp)

        # 2) portfolios coming from selected groups
        for gi in (selected_group_instances or []):
            grp = getattr(gi, "reference_group", None) or gi  # accept raw group too
            selected_ports.extend(_flatten_group_portfolios(grp))

        # 3) dedupe by id
        selected_ports = _dedupe_portfolios(selected_ports)
        if not selected_ports:
            st.info("Select one or more portfolios or groups to continue.")
            st.stop()

        selected_ids = tuple(sorted(int(getattr(p, "id", 0)) for p in selected_ports))
        last_ids = tuple(st.session_state.get("_active_portfolio_ids", ()))
        last_notional = float(st.session_state.get("_active_notional", 0.0) or 0.0)

        if (selected_ids != last_ids) or (float(notional_value) != last_notional):
            # Build all position templates ONCE for the current selection
            po_all = PortfoliosOperations(portfolio_list=selected_ports)
            with st.spinner("Building positions from selected portfolio(s)…"):
                ih2a_combined, positions_by_id = po_all.get_all_portfolios_as_positions(
                    portfolio_notional=float(notional_value)
                )
            if not positions_by_id:
                st.error("No positions returned from portfolio builder.")
                st.stop()



            # Persist required session state
            prev_cfg = _get_cfg_from_session(st.session_state)
            st.session_state["position_cfg_mem"] = {
                **prev_cfg,
                "portfolio_notional": float(notional_value),
            }
            st.session_state["instrument_hash_to_asset"] = ih2a_combined
            st.session_state["position_templates_mem"] = positions_by_id            # {pid -> Position template}
            st.session_state["_active_portfolio_ids"] = selected_ids
            st.session_state["_active_notional"] = float(notional_value)
            st.session_state["_ports_by_id"] = {int(p.id): p for p in selected_ports}

            # Build a merged template to drive a single curve control
            merged_lines = []
            for pos in positions_by_id.values():
                merged_lines.extend(list(pos.lines or []))
            st.session_state["_merged_template"] = msi.Position(lines=merged_lines)

            st.rerun()

# ---------- Main content ----------
cfg = _get_cfg_from_session(st.session_state)
iso_date = cfg.get("valuation_date")
if not iso_date:
    st.info("Pick a valuation date and load one or more portfolios.")
    st.stop()

valuation_date = dt.date.fromisoformat(iso_date)
ql.Settings.instance().evaluationDate = qld(valuation_date)

positions_by_id: Dict[int, msi.Position] = st.session_state.get("position_templates_mem") or {}
if not positions_by_id:
    st.info("Select one or more portfolios in the sidebar to build positions.")
    st.stop()
ordered_ids = list(st.session_state.get("_active_portfolio_ids", positions_by_id.keys()))
ports_by_id = st.session_state.get("_ports_by_id") or {}
all_portfolios = [ports_by_id.get(int(pid)) for pid in ordered_ids if ports_by_id.get(int(pid))]
po_all = PortfoliosOperations(portfolio_list=all_portfolios) if all_portfolios else None

# ── Top-level tabs: Curve Controller | Portfolios Impact | Competition tabs ──
tab_labels = [
    "Curve Controller",
    "Portfolios Impact",
    "Weights by token",
    "Maturity buckets",
    "Rolling leaders",
    "Serialized correlation",
    "Simulation",
    "Status",
]
(curve_tab, impact_tab,
 tab_weights, tab_maturity, tab_leaders,
 tab_corr, tab_sim, tab_status) = st.tabs(tab_labels)

# Optional: if your competition pipeline provides an AppContext, expose it via session
# so these tabs can consume it without extra imports/wiring.
comp_ctx = st.session_state.get("competition_ctx", None)

# 1) ── Curve Controller (unchanged logic; shared curves for all portfolios)
merged_template = st.session_state.get("_merged_template")
bundle_curve = None
with curve_tab:
    st.caption(f"Valuation date: {iso_date}")
    bundle_curve = st_curve_bumps(
        position_template=merged_template,
        valuation_date=valuation_date,
        key_prefix="curve_bundle_all",
    )

# 2) ── Portfolios Impact (driven by the SAME curves as above)
with impact_tab:
    if not bundle_curve:
        st.info("Adjust the curve bump settings in **Curve Controller** to build base and bumped curves.")
    else:
        base_curves = bundle_curve["base_curves"]
        bumped_curves = bundle_curve["bumped_curves"]

        id_to_name = {}
        for pid in ordered_ids:
            p = ports_by_id.get(int(pid))
            id_to_name[int(pid)] = getattr(p, "portfolio_name", getattr(p, "name", f"Portfolio {pid}")) if p else f"Portfolio {pid}"

        tabs_imp = st.tabs([id_to_name.get(int(pid), f"Portfolio {pid}") for pid in ordered_ids])

        for pid, tab in zip(ordered_ids, tabs_imp):
            pid = int(pid)
            pos_template = positions_by_id.get(pid)
            if pos_template is None:
                continue

            # Per-portfolio position operations using the SAME curves
            ops = PositionOperations.from_position(
                pos_template,
                base_curves_by_index=base_curves,
                valuation_date=valuation_date,
            )
            base_pos = ops.instantiate_base()
            ops.set_curves(bumped_curves_by_index=bumped_curves)
            bumped_pos = ops.instantiate_bumped()
            ops.compute_and_apply_z_spreads_from_dirty_price(
                base_position=base_pos,
                bumped_position=bumped_pos,
            )

            one_po = PortfoliosOperations(portfolio_list=[ports_by_id.get(pid)]) if ports_by_id.get(pid) else None
            if one_po is None:
                continue
            one_po.set_portfolio_positions({pid: base_pos})

            with tab:
                st.caption(f"Valuation date: {iso_date}")
                one_po.st_position_npv_table_paginated(
                    position=base_pos,
                    instrument_hash_to_asset=st.session_state.get("instrument_hash_to_asset"),
                    bumped_position=bumped_pos,
                    page_size_options=(25, 50, 100, 200),
                    default_size=50,
                    enable_search=True,
                    key=f"npv_table_{pid}_{iso_date}",
                    render_once=False,
                    valuation_date=valuation_date,
                    carry_days=st.session_state.get("carry_cutoff_days", 365),
                )

# 3) ── Weights by token  (competition_view.py: tabs[0] content)
with tab_weights:

    st.subheader("Weights by token")
    st.markdown(
        "Use tokens to group instruments (ticker prefixes/labels) and display the **share of portfolio weight** "
        "captured by each token in the latest holdings snapshot."
    )

    # Defaults (kept local to avoid sidebar dependency)
    default_tokens_csv = "91,M_BONOS,S_UDIBONO,LF_BONDES"
    default_match_mode = "contains"
    default_case_ins = True

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.text_input(
            "Identifier tokens (comma-sep)",
            key="tokens_csv",
            value=st.session_state.get("tokens_csv", default_tokens_csv),
            placeholder="e.g., 91,M_BONOS,S_UDIBONO,LF_BONDES",
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
    if not po_all:
        st.info("Select one or more portfolios to compute token weights.")
    else:

        agg_df = po_all.sum_signal_weights_table(
            unique_identifier_filter=tokens,
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
        if agg_df is not None:
            st.info("Signals does not found in portfolios")
        if po_all and agg_df is not None:
            st.dataframe(agg_df.reset_index(drop=False), width="stretch")

# 4) ── Maturity buckets  (competition_view.py: tabs[1] content)
with tab_maturity:

    if not po_all:
        st.info("Select one or more portfolios to compute maturity buckets.")
    else:
        st.subheader("Weighted duration by maturity bucket")
        # Snapshot disclosure based on valuation_date
        asof_utc = pd.Timestamp(valuation_date).tz_localize("UTC")
        snap_m = po_all.maturity_snapshot(asof=valuation_date).name
        snap_d = po_all.duration_snapshot(asof=valuation_date).name

        snap_m_utc = pd.Timestamp(snap_m).tz_convert("UTC") if snap_m is not None else None
        snap_d_utc = pd.Timestamp(snap_d).tz_convert("UTC") if snap_d is not None else None
        snap_used_utc = None
        if snap_m_utc is not None or snap_d_utc is not None:
            candidates = [d for d in [snap_m_utc, snap_d_utc] if d is not None]
            snap_used_utc = max(candidates)
            if snap_m_utc is not None and snap_d_utc is not None and snap_m_utc.date() != snap_d_utc.date():
                st.caption(
                    f"maturity **{snap_m_utc.strftime('%Y-%m-%d')} UTC**, "
                    f"duration **{snap_d_utc.strftime('%Y-%m-%d')} UTC**; "
                    f"analysis date = **{snap_used_utc.strftime('%Y-%m-%d')} UTC** "
                    f"(last available ≤ Valuation **{asof_utc.strftime('%Y-%m-%d')} UTC**)."
                )
            else:
                st.caption(
                    f"Using snapshot **{snap_used_utc.strftime('%Y-%m-%d')} UTC** "
                    f"(last available <= As-of **{asof_utc.strftime('%Y-%m-%d')} UTC**)."
                )

        default_buckets_csv = "365,730,1825"
        default_weighted_mode = "contribution"
        c1, c2 = st.columns([2, 1])
        with c1:
            st.text_input(
                "Maturity bucket edges (days, comma-sep)",
                key="buckets_csv",
                value=st.session_state.get("buckets_csv", default_buckets_csv),
                placeholder="e.g., 365,730,1825",
            )
        with c2:
            WM_OPTS = ["contribution", "average"]
            st.selectbox(
                "Weighted duration mode",
                WM_OPTS,
                index=WM_OPTS.index(st.session_state.get("weighted_mode", default_weighted_mode)),
                key="weighted_mode",
            )

        maturity_s = po_all.maturity_snapshot(asof=valuation_date)
        duration_s = po_all.duration_snapshot(asof=valuation_date)
        if maturity_s.empty or duration_s.empty:
            st.info("Maturity/duration snapshots not available at selected As-of.")
        else:
            buckets_csv_val = st.session_state.get("buckets_csv", default_buckets_csv)
            try:
                maturity_buckets = [int(x.strip()) for x in buckets_csv_val.split(",") if x.strip()]
            except Exception:
                maturity_buckets = [int(x) for x in default_buckets_csv.split(",")]

            tokens_csv_val = st.session_state.get("tokens_csv", "91,M_BONOS,S_UDIBONO,LF_BONDES")
            tokens = [t.strip() for t in tokens_csv_val.split(",") if t.strip()]

            all_bucket_dfs: List[pd.DataFrame] = []
            portfolios = po_all.signals.index.get_level_values("portfolio_name").unique()
            for portfolio in portfolios:
                signals_for_portfolio = po_all.signals.loc[pd.IndexSlice[:, portfolio], :]
                if signals_for_portfolio.empty:
                    continue
                raw_df = po_all.weighted_duration_by_maturity_bucket_table(
                    maturity_series=maturity_s,
                    duration_series=duration_s,
                    unique_identifier_filter=tokens,
                    maturity_bucket_days=maturity_buckets,
                    signals=signals_for_portfolio,
                    case_insensitive=st.session_state.get("case_insensitive", True),
                    match_mode=st.session_state.get("match_mode", "contains"),
                    lowercase_token_index=True,
                    weighted_mode=st.session_state.get("weighted_mode", default_weighted_mode),
                    percent=False, decimals=2, title=None, render=False, return_df=True)

                if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                    part = raw_df.reset_index(drop=False).copy()
                    part["portfolio_name"] = portfolio
                    all_bucket_dfs.append(part)

            if all_bucket_dfs:
                combined_bkt_df = pd.concat(all_bucket_dfs, axis=0, ignore_index=True)
                analysis_date_str = snap_used_utc.strftime("%Y-%m-%d") if snap_used_utc is not None else None
                combined_bkt_df["snapshot_date"] = analysis_date_str

                wd = pd.to_numeric(combined_bkt_df["weighted_duration"], errors="coerce")
                denom = combined_bkt_df.groupby(["portfolio_name", "token"])["weighted_duration"].transform("sum")
                den = pd.to_numeric(denom, errors="coerce").replace(0, np.nan)
                pct = (wd / den).replace([np.inf, -np.inf], np.nan)
                combined_bkt_df["weighted_duration_pct"] = pct * 100.0

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

# 5) ── Rolling leaders  (competition_view.py: tabs[2] content)
with tab_leaders:
    if not po_all:
        st.info("Select one or more portfolios to plot leaders.")
    else:
        st.subheader("Rolling leaders — latest windows")
        default_w3, default_w6, default_w9 = 63, 126, 189
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("3M window (days)", min_value=10, max_value=400, step=1,
                            value=int(st.session_state.get("w3", default_w3)), key="w3")
        with c2:
            st.number_input("6M window (days)", min_value=20, max_value=600, step=1,
                            value=int(st.session_state.get("w6", default_w6)), key="w6")
        with c3:
            st.number_input("9M window (days)", min_value=30, max_value=800, step=1,
                            value=int(st.session_state.get("w9", default_w9)), key="w9")

        if po_all.prices.shape[1] == 0:
            st.info("No portfolio prices available.")
        else:
            windows = {
                "3M": int(st.session_state.get("w3", default_w3)),
                "6M": int(st.session_state.get("w6", default_w6)),
                "9M": int(st.session_state.get("w9", default_w9)),
            }
            figs_leaders, latest_top3 = po_all.plot_competitive_leaders(
                po_all.prices, windows=windows, always_show=None,
                title_prefix="Latest rolling window (base = 100)",
                legend_at_bottom=True, isolate_on_click=True, show_baseline=True)
            for _, fig in figs_leaders.items():
                st.plotly_chart(fig, width="stretch")
            with st.expander("Latest Top-3 per window"):
                for wl, s in latest_top3.items():
                    st.write(f"**{wl}**")
                    st.dataframe(s.rename("last_value").to_frame(), width="stretch")

# 6) ── Serialized correlation  (competition_view.py: tabs[3] content)
with tab_corr:
    if not po_all:
        st.info("Select at least two portfolios to compute correlation.")
    else:
        st.subheader("Serialized correlation (portfolio level)")
        if po_all.prices.shape[1] <= 1:
            st.info("Need at least two portfolios with price history.")
        else:
            prices_for_plots = comp_ctx.portfolio_prices
            _, _, fig_heat, fig_dend = plot_serialized_correlation(po_all.prices, mode="returns")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_heat, width="stretch")
            with c2:
                st.plotly_chart(fig_dend, width="stretch")

# 7) ── Simulation  (competition_view.py: tabs[4] content)
with tab_sim:
    if not po_all:
        st.info("Select one or more portfolios to run the simulation.")
    else:
        st.subheader("Simulation (notebook parity)")
        default_sim_window = 180
        default_sim_min_obs = 120
        default_sim_months = 3
        default_sim_samples = 20000
        default_sim_norm = "sum"
        default_sim_excess = True

        st.markdown("**Simulation settings**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.number_input("Lookback window (days)", min_value=60, max_value=1500, step=5,
                            value=int(st.session_state.get("sim_window", default_sim_window)), key="sim_window")
            st.number_input("Min obs per asset", min_value=60, max_value=2000, step=10,
                            value=int(st.session_state.get("sim_min_obs", default_sim_min_obs)), key="sim_min_obs")
        with c2:
            st.number_input("Horizon (months)", min_value=1, max_value=24, step=1,
                            value=int(st.session_state.get("sim_N_months", default_sim_months)), key="sim_N_months")
            NORM_OPTS = ["sum", "none"]
            st.selectbox("Normalize weights", NORM_OPTS,
                         index=NORM_OPTS.index(st.session_state.get("sim_norm", default_sim_norm)), key="sim_norm")
        with c3:
            st.number_input("Number of simulations", min_value=1000, max_value=200000, step=1000,
                            value=int(st.session_state.get("sim_samples", default_sim_samples)), key="sim_samples")
            st.checkbox("Plot excess over target (center at 0)",
                        value=st.session_state.get("sim_excess", default_sim_excess), key="sim_excess")

        lead = {}
        try:
            lead = po_all.get_leading_portfolios(po_all.prices)
        except Exception as e:
            st.error(f"Failed to compute lead (target) returns: {e}")

        if not lead:
            st.info("Lead/target returns not available; cannot run simulation.")
        else:
            try:
                summary_df, rank_df, sim_ports, _, _, fig_iqr, fig_rank = po_all.simulate_portfolios_centered_on_targets(
                    valmer_prices=None,
                    lead_returns=lead,
                    window=int(st.session_state.get("sim_window", default_sim_window)),
                    N_months=int(st.session_state.get("sim_N_months", default_sim_months)),
                    samples=int(st.session_state.get("sim_samples", default_sim_samples)),
                    normalize_weights=st.session_state.get("sim_norm", default_sim_norm),
                    min_obs_per_asset=int(st.session_state.get("sim_min_obs", default_sim_min_obs)),
                )
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
                    for c in rank_disp.columns:
                        if any(k in c.lower() for k in ("rank", "prob", "p")) and pd.api.types.is_numeric_dtype(rank_disp[c]):
                            rank_disp[c] = rank_disp[c] * 100.0
                    st.subheader("Rank Probabilities")
                    st.dataframe(rank_disp, width="stretch")

                if fig_iqr:
                    st.plotly_chart(fig_iqr, width="stretch")
                if fig_rank:
                    st.plotly_chart(fig_rank, width="stretch")
            except Exception as e:
                st.error(f"Simulation failed: {e}")

# 8) ── Status  (competition_view.py: tabs[5] content)
with tab_status:
    if comp_ctx is None:
        st.info("Competition context not available. Provide `competition_ctx` in `st.session_state`.")
    else:
        st.subheader("Status Overview")
        if comp_ctx.all_signals_mi.empty:
            st.warning("No holdings data found for the selected peer group. Cannot display status.")
        else:
            status_df = comp_ctx.peers_df.copy()
            loaded_funds = comp_ctx.all_signals_mi.index.get_level_values("portfolio_name").unique().tolist()
            status_df_filtered = status_df[status_df["Fondo"].isin(loaded_funds)] if "Fondo" in status_df.columns else status_df

            latest_holdings = (
                comp_ctx.all_signals_mi.reset_index()
                .groupby("portfolio_name")
                .agg(latest_holdings_date=("time_index", "first"), total_weight=("signal_weight", "sum"))
            )
            merged_df = pd.merge(
                status_df_filtered,
                latest_holdings,
                left_on=getattr(comp_ctx, "COL_FONDO", "Fondo"),
                right_index=True,
                how="left",
            )

            date_cols = ["last_run_time", "pdf_last_update", "fund_price_last_update"]
            for col in date_cols:
                if col in merged_df.columns:
                    merged_df[col] = pd.to_datetime(merged_df[col], errors="coerce")
            for col in merged_df.select_dtypes(include=["datetimetz"]).columns:
                merged_df[col] = merged_df[col].dt.tz_localize(None)
            if "latest_holdings_date" in merged_df.columns and pd.api.types.is_datetime64_any_dtype(merged_df["latest_holdings_date"]):
                if getattr(merged_df["latest_holdings_date"].dt, "tz", None) is not None:
                    merged_df["latest_holdings_date"] = merged_df["latest_holdings_date"].dt.tz_localize(None)

            display_cols = {
                "Operadora": "Operadora",
                "Fondo": "Fondo",
                "last_run_time": "Last Scrape",
                "pdf_last_update": "PDF Date",
                "fund_price_last_update": "Price Date",
                "latest_holdings_date": "Holdings Date",
                "Link Cartera Semanal": "Weekly Report",
            }
            existing_display_cols_keys = [k for k in display_cols if k in merged_df.columns]
            display_df = merged_df[existing_display_cols_keys].rename(columns=display_cols)

            def _hl_nan(row):
                is_na = row.isnull().any()
                return ["background-color: #e62e2e; color: white" if is_na else "" for _ in row]

            st.dataframe(
                display_df.style.apply(_hl_nan, axis=1),
                width="stretch",
                hide_index=True,
                column_config={
                    "Weekly Report": st.column_config.LinkColumn("Link", display_text="🔗 Document Link"),
                    "Last Scrape": st.column_config.DatetimeColumn("Last Scrape", format="YYYY-MM-DD HH:mm"),
                    "PDF Date": st.column_config.DatetimeColumn("PDF Date", format="YYYY-MM-DD"),
                    "Price Date": st.column_config.DatetimeColumn("Price Date", format="YYYY-MM-DD"),
                    "Holdings Date": st.column_config.DatetimeColumn("Holdings Date", format="YYYY-MM-DD"),
                },
            )