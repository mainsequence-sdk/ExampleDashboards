# alm_dashboard_base/core/registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List

@dataclass(frozen=True)
class Page:
    slug: str
    title: str
    render: Callable          # signature: (ctx) -> None
    visible: bool = True      # 👈 NEW: controls presence in the nav

_PAGES: Dict[str, Page] = {}

def register_page(slug: str, title: str, *, visible: bool = True):
    """Decorator to register a page function."""
    def _wrap(fn: Callable) -> Callable:
        if slug in _PAGES:
            raise ValueError(f"Duplicate page slug '{slug}'")
        _PAGES[slug] = Page(slug=slug, title=title, render=fn, visible=visible)
        return fn
    return _wrap

def list_pages(*, visible_only: bool = False) -> List[Page]:
    pages = list(_PAGES.values())
    return [p for p in pages if p.visible] if visible_only else pages

def get_page(slug: str) -> Page | None:
    return _PAGES.get(slug)
