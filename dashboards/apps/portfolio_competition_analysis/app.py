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
        from dashboards.apps.portfolio_competition_analysis.settings import PRICES_TABLE_NAME
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

            st.session_state.setdefault("_base_positions_by_id", {})[pid] = base_pos
            st.session_state.setdefault("_id_to_name", {})[pid] = id_to_name[pid]

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
    st.subheader("Weighted duration by filter × maturity bucket")

    # Filters input
    default_wd_filters_csv = "FIXED,FLOAT"
    filters_csv = st.text_input(
        "Instrument filters (UID contains, comma‑separated)",
        key="wd_filters_csv",
        value=st.session_state.get("wd_filters_csv", default_wd_filters_csv),
        placeholder="e.g., FIXED,FLOAT,91",
    )
    # Empty input => function aggregates ALL assets under a single 'All' row
    filters = tuple(t.strip() for t in (filters_csv or "").split(",") if t.strip())

    # Use the positions built in Portfolios Impact
    base_positions_by_id = st.session_state.get("_base_positions_by_id", {})
    id_to_name = st.session_state.get("_id_to_name", {int(pid): getattr(ports_by_id[int(pid)], "portfolio_name", f"Portfolio {pid}") for pid in ordered_ids})

    if not base_positions_by_id:
        st.info("Base positions are not available yet. Open the **Portfolios Impact** tab first.")
    else:
        maturity_tabs = st.tabs([id_to_name[int(pid)] for pid in ordered_ids])

        for pid, tab in zip(ordered_ids, maturity_tabs):
            pid = int(pid)
            with tab:
                base_pos = base_positions_by_id[pid]  # <-- reuse, no re-instantiation
                po_all.st_weighted_duration_buckets_by_type(
                    position=base_pos,
                    portfolio_name=id_to_name[pid],
                    valuation_date=valuation_date,
                    identifier_filters=filters,                      # drives the 2nd index
                    maturity_bucket_days=(365, 730, 1825),
                    decimals=2,
                    instrument_hash_to_asset=st.session_state["instrument_hash_to_asset"],
                    key=f"wd_buckets_{pid}_{iso_date}",
                )

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
            prices_for_plots = po_all.prices
            _, _, fig_heat, fig_dend = plot_serialized_correlation(prices_for_plots, mode="returns")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_heat, width="stretch")
            with c2:
                st.plotly_chart(fig_dend, width="stretch")

