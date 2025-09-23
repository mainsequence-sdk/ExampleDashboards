from __future__ import annotations

from mainsequence.dashboards.streamlit.scaffold import AppConfig, run_app
from mainsequence.dashboards.streamlit.core.registry import autodiscover





# Import the context builder and session initializer from our new context module
from dashboards.apps.portfolio_competitive_landscape.context import (
    build_context_for_scaffold,
    init_session_for_scaffold,
)

# Discover all pages decorated with @register_page in the specified package
autodiscover("dashboards.apps.portfolio_competitive_landscape.views")

# Configure the application using the AppConfig contract from the framework
cfg = AppConfig(
    title="Competition Analysis — Interactive",
    build_context=build_context_for_scaffold,
    init_session=init_session_for_scaffold,
    default_page="main_analysis",  # Slug for the main view
)

# Run the app
if __name__ == "__main__":
    run_app(cfg)