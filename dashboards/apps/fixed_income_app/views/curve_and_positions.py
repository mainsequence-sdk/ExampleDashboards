from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from mainsequence.dashboards.streamlit.core.registry import register_page
from dashboards.apps.fixed_income_app.context import AppContext, portfolio_stats
from dashboards.components.npv_table import st_position_npv_table_paginated

from dashboards.components.curve_bump import curve_bump_controls_ex
from dashboards.curves.bumping import KEYRATE_GRID,curve_family_key
from dashboards.components.position_yield_overlay import st_position_yield_overlay
from dashboards.plots.curves import plot_par_yield_curves_by_family
import mainsequence.instruments as msi
from dashboards.core.formatters import fmt_ccy
import datetime
import math
import pandas as pd
from dashboards.core.data_nodes import get_app_data_nodes
from dashboards.components.engine_status import publish_engine_meta
from dashboards.core.tenor import tenor_to_years

import mainsequence.client as msc


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

def _find_dump_position_example() -> Path:
    """
    Locate 'data/dump_position_example.json' by walking upwards from this file.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "data" / "dump_position_example.json"
        if candidate.exists():
            return candidate
    # Fallback: sometimes examples are nested one level deeper
    for parent in [here, *here.parents]:
        candidate = parent / "examples" / "data" / "dump_position_example.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate 'data/dump_position_example.json' from views/curve_and_positions.py"
    )


def _safe_portfolio_id(p: Any) -> Any:
    try:
        if hasattr(p, "model_dump"):
            d = p.model_dump()
            return d.get("id")
        if isinstance(p, dict):
            return p.get("id")
        return getattr(p, "id", None)
    except Exception:  # pragma: no cover
        return None


def build_position_from_portfolio(
    portfolio_index_asset: Any,
portfolio_notional:float,
) -> Tuple[Any, Dict[str, Any], str, Dict[str, Any]]:
    """
    Build a position/config from a MainSequence portfolio.

    Only the 'mock_position=True' path is implemented: it ignores the portfolio and
    returns the path to 'data/dump_position_example.json', so the app can load/price
    immediately via the existing context builder.

    Returns: (position, cfg_dict, outp_path, meta)
    """
    # 1) Collect assets from the latest portfolio weights
    import pytz

    latest_weights,weights_date = portfolio_index_asset.reference_portfolio.get_latest_weights(), datetime.datetime.now(pytz.utc)
    publish_engine_meta("Portfolio", weights_date=weights_date)

    uids = list((latest_weights or {}).keys())
    if not uids:
        raise ValueError("Selected portfolio has no latest weights.")
    assets = msc.Asset.filter(unique_identifier__in=uids)

    if not assets:
        raise ValueError("Could not resolve assets for latest weights.")

        # 2) Determine portfolio notional (sidebar default is 1_000_000)
    total_notional = float(
        portfolio_notional if portfolio_notional is not None
        else st.session_state.get("portfolio_notional", 1_000_000.0)
    )

    # 3) Translate weights -> integer holdings (units)
    aid_to_asset={a.id:a for a in assets}
    lines = _make_weighted_lines(assets, latest_weights, total_notional)
    if not lines:
        raise ValueError("No valid instrument lines could be built from portfolio assets.")

    # 4) Normalize through Position to lock the exact shape our loader expects.
    pos_obj = msi.Position.from_json_dict({"lines": lines})

    instrument_hash_to_asset={l.instrument.content_hash():aid_to_asset[l.instrument.main_sequence_asset_id] for l in pos_obj.lines}

    # 4) Stash template + minimal app config in memory for build_context(':memory:')
    st.session_state["position_template_mem"] = pos_obj
    st.session_state["instrument_hash_to_asset"] = instrument_hash_to_asset
    st.session_state["position_cfg_mem"] = {
        # Only fields used by build_context are required here.
        "valuation": {
            "valuation_date": datetime.date.today().isoformat(),
            "cashflow_cutoff_days": 365,
        },
        # Curve bumps actually come from st.session_state["curve_bump_spec"]
        "curve_bumps_bp": {},
        "portfolio_notional": total_notional,
    }

    meta = {
        "source": "live_portfolio",
        "portfolio_id": _safe_portfolio_id(portfolio_index_asset),
    }
    # Keep API compatible with callers: position=None, cfg_dict (for completeness),
    # cfg_path=':memory:' signals the in-memory path, meta for logging.
    return None, st.session_state["position_cfg_mem"], ":memory:", meta



@register_page("curve_positions", "Curve, Stats & Positions", has_sidebar=True, order=0)
def render(ctx: AppContext):
    # ---------- Sidebar ----------
    with st.sidebar:
        # -------------------- Portfolio search / ajax-like select --------------------
        st.markdown("### Build position from portfolio (search)")
        st.caption(
            "Type at least **3** characters to search portfolios "
        )

        # --- Portfolio notional input (used when building from live portfolio) ---
        notional_value = st.number_input(
            "Portfolio notional (base currency)",
            min_value=1.0,
            step=100_000.0,
            value=1_000_000.0,
            format="%.0f",
            help="Weights will be translated into integer holdings using this notional.",
            key="portfolio_notional",
        )

        @st.cache_data(show_spinner=False)
        def _search_portfolios(q: str) -> List[Dict[str, Any]]:
            if not q or len(q.strip()) < 3:
                return []
            if msc is None:
                # No client available; show empty result so UI stays clean
                return []
            # Call the endpoint; normalize results to a common dict shape
            results = msc.PortfolioIndexAsset.filter(current_snapshot__name__contains=q.strip())
            out: List[Dict[str, Any]] = []
            for p in (results or []):
                pid = p.id
                pname = p.current_snapshot.name
                out.append({"id": pid, "name": pname, "instance": p})
            return out

        q = st.text_input(
            "Search portfolios",
            placeholder="e.g. MXN ALM Desk …",
            key="portfolio_search_q",
        )



        portfolios = _search_portfolios(q) if q and len(q.strip()) >= 3 else []
        if q and len(q.strip()) < 3:
            st.write(" ")
            st.caption("Keep typing… need **3+** characters to search.")

        if portfolios:
            labels = [f"{p['name']} (id={p['id']})" for p in portfolios]
            sel = st.selectbox(
                "Select a portfolio",
                options=list(range(len(portfolios))),
                format_func=lambda i: labels[i],
                key="portfolio_select_idx",
            )
            col_load, col_clear = st.columns([3, 2])
            with col_load:
                if st.button(
                    "Load selected portfolio",
                    type="primary",
                    use_container_width=True,
                    key="btn_build_from_portfolio",
                ):
                    try:
                        with st.spinner("Building position from selected portfolio…"):
                            _pos, _cfg, outp, meta = build_position_from_portfolio(
                                portfolios[sel]["instance"],
                                portfolio_notional=float(st.session_state.get("portfolio_notional", 1_000_000.0)),
                            )
                        st.session_state["cfg_path"] = outp  # ':memory:' or file path
                        if outp == ":memory:":
                            st.success(f"✓ Loaded position for {labels[sel]} (in-memory template)")
                        else:
                            st.success(f"✓ Loaded mock position for {labels[sel]}\nUsing: `{outp}`")
                        st.rerun()
                    except Exception as e:  # pragma: no cover
                        st.error(f"Failed to build from portfolio:\n{e}")
            with col_clear:
                if st.button(
                    "Clear selection", use_container_width=True, key="btn_clear_portfolio"
                ):
                    st.session_state.pop("portfolio_select_idx", None)
                    st.session_state.pop("portfolio_search_q", None)
                    st.rerun()

        st.divider()

        # -------------------- Curve bump controls (per index) --------------------
        # Which indices do we actually have in the loaded position?
        present_indices = sorted({
            getattr(ln.instrument, "floating_rate_index_name", None)
            for ln in ctx.position.lines
            if getattr(ln.instrument, "floating_rate_index_name", None) is not None
        })
        present_families = sorted({curve_family_key(i) for i in present_indices})

        with st.sidebar:
            st.markdown("### Curve bumps (by family)")
            # read current per-index state (if any)
            prev = (st.session_state.get("curve_bump_spec_by_family")
                     or st.session_state.get("curve_bump_spec_by_index", {}))
            new_map = {}

            for fam in present_families:
                rep_idx = next((i for i in present_indices if curve_family_key(i) == fam), None)
                tenors = list(
                    KEYRATE_GRID.get(rep_idx, next((g for k, g in KEYRATE_GRID.items()
                                                    if curve_family_key(k) == fam), ()))
                )
                exp = st.expander(f"{fam} — bumps", expanded=(len(present_families) == 1))
                spec, _ = curve_bump_controls_ex(
                    available_tenors=tenors,
                    default_bumps=(prev.get(fam, {}).get("keyrate_bp", {})),
                    default_parallel_bp=float(prev.get(fam, {}).get("parallel_bp", 0.0)),
                    header=f"{fam} bumps (bp)",
                    container=exp,
                    key=f"curve_bump_{fam}",
                )
                new_map[fam] = {"keyrate_bp": spec.keyrate_bp, "parallel_bp": float(spec.parallel_bp)}

            if new_map != prev:
                st.session_state["curve_bump_spec_by_family"] = new_map
                # keep old key in sync for safety (can be removed later)
                st.session_state["curve_bump_spec_by_index"] = new_map
                st.rerun()

    # ---------- Main content ----------
    # Reserve a slot for the curve so it appears above the overlay UI
    curve_slot = st.container()



    # Overlay UI (no chart here; it only computes/clears points and returns traces)
    overlay_traces = st_position_yield_overlay(
        position=ctx.position,
        val_date=ctx.val_date,
        ts_base=None, ql_ref_date=None, nodes_base=None,
        key="base_curve_position_overlay",
    )
    try:
        any_ts = next(iter(ctx.base_curves.values()))
        calc_date = any_ts.referenceDate()
    except StopIteration:
        st.warning("No curves available to plot.")
        st.stop()
    # Build the curve ONCE, add overlay traces, render ONCE
    fig_curve = plot_par_yield_curves_by_family(
        base_curves=ctx.base_curves,
        bumped_curves=ctx.bumped_curves,
        calc_date=calc_date,
        keyrate_grid_by_index=KEYRATE_GRID,
        max_years=30,
        step_months=3,
        title="Par yield curves — Base vs Bumped (families)"
    )
    for tr in overlay_traces:
        fig_curve.add_trace(tr)

    with curve_slot:
        st.subheader("Par yield curve — Base vs Bumped")
        st.plotly_chart(fig_curve, use_container_width=True, key="par_curve_main")

    # Stats (carry cutoff slider on the same page)
    st.subheader("Portfolio statistics — Base vs Bumped")
    carry_days = st.slider(
        "Carry cutoff (days from valuation date)",
        min_value=30,
        max_value=1460,
        value=(ctx.carry_cutoff - ctx.val_date).days,
        step=30,
    )
    cutoff = ctx.val_date + __import__("datetime").timedelta(days=carry_days)

    stats = portfolio_stats(ctx.position, ctx.bumped_position, ctx.val_date, cutoff)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "NPV (base)",
        fmt_ccy(stats["npv_base"], ctx.currency_symbol),
        delta=fmt_ccy(stats["npv_delta"], ctx.currency_symbol, ),
    )
    c2.metric("NPV (bumped)", fmt_ccy(stats["npv_bumped"], ctx.currency_symbol))
    c3.metric(
        f"Carry to {cutoff.isoformat()} (base)",
        fmt_ccy(stats["carry_base"], ctx.currency_symbol),
        delta=fmt_ccy(stats["carry_delta"], ctx.currency_symbol, ),
    )
    c4.metric(f"Carry to {cutoff.isoformat()} (bumped)", fmt_ccy(stats["carry_bumped"], ctx.currency_symbol))

    # Positions table
    st.subheader("Positions — NPV (paginated)")
    st_position_npv_table_paginated(
        position=ctx.position,
        instrument_hash_to_asset=st.session_state.get("instrument_hash_to_asset"),
        currency_symbol=ctx.currency_symbol,
        bumped_position=ctx.bumped_position,
        page_size_options=(25, 50, 100, 200),
        default_size=50,
        enable_search=True,
        key="npv_table_curve_page",
    )
