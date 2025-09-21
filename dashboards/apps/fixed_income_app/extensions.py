# dashboards/apps/fixed_income_app/extensions.py
from __future__ import annotations
import streamlit as st

from dashboards.core.data_nodes import get_app_data_nodes
from dashboards.components.engine_status import register_engine_meta_provider

def _register_default_data_nodes():
    # App-level deps by name (override elsewhere if needed)
    deps = get_app_data_nodes()
    deps.register(instrument_pricing_table_id="vector_de_precios_valmer")

# --- Providers (push into meta; NO fixed field assumptions) -------------------

def _summary_provider(ctx, meta):
    # Summary row: valuation & currency
    meta.add_summary(
        valuation_date=getattr(ctx, "val_date", None),
        currency=getattr(ctx, "currency_symbol", None),
    )

    # Section with per-curve reference dates
    try:
        refs = {k: v.referenceDate().ISO() for k, v in (getattr(ctx, "base_curves", {}) or {}).items()}
    except Exception:
        refs = {}
    if refs:
        meta.add("Curves", **refs)

def _portfolio_provider(ctx, meta):
    # Whatever your engine/view published at runtime (e.g., weights_date)
    wd = st.session_state.get("weights_date")
    if wd:
        meta.add("Portfolio", weights_date=wd)

def _market_provider(ctx, meta):
    asof = st.session_state.get("prices_observation_time")
    if asof:
        meta.add("Pricing data", observation_time=asof)

def _build_info(ctx, meta):
    env = st.session_state.get("env", "dev")
    meta.add("Build", engine="FixedIncome/QL", env=env)

def bootstrap():
    _register_default_data_nodes()
    register_engine_meta_provider(_summary_provider)
    register_engine_meta_provider(_portfolio_provider)
    register_engine_meta_provider(_market_provider)
    register_engine_meta_provider(_build_info)
