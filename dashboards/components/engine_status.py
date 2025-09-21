# dashboards/components/engine_status.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Callable, List, Optional, Iterable, Tuple
import datetime as dt
import streamlit as st

# -------------------------- Runtime container (dynamic) ------------------------

@dataclass
class EngineMeta:
    """
    Dynamic container for the Pricing Engine HUD.
    - No predefined attributes.
    - Sections are free-form dicts: {section_name: {key: value, ...}}
    - 'summary' is an optional small dict rendered in the chip's top grid.
    """
    summary: Dict[str, Any] = field(default_factory=dict)
    sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)  # optional section ordering

    def add(self, section: str, **mapping: Any) -> "EngineMeta":
        if not mapping:
            return self
        sec = self.sections.setdefault(section, {})
        for k, v in mapping.items():
            if v not in (None, "", []):
                sec[k] = v
        return self

    def add_pairs(self, section: str, pairs: Iterable[Tuple[str, Any]]) -> "EngineMeta":
        sec = self.sections.setdefault(section, {})
        for k, v in pairs:
            if v not in (None, "", []):
                sec[k] = v
        return self

    def add_summary(self, **mapping: Any) -> "EngineMeta":
        for k, v in mapping.items():
            if v not in (None, "", []):
                self.summary[k] = v
        return self

    def set_order(self, *section_names: str) -> "EngineMeta":
        self.order = list(section_names)
        return self

# -------------------------- Providers (mutate `EngineMeta`) --------------------

# Provider signature: fn(ctx, meta) -> None
_MetaProvider = Callable[[Any, EngineMeta], None]
_META_PROVIDERS: List[_MetaProvider] = []

def register_engine_meta_provider(fn: _MetaProvider) -> None:
    _META_PROVIDERS.append(fn)

# -------------------------- Publishing API (no provider needed) ----------------

def publish_engine_meta(section: str, **mapping: Any) -> None:
    """
    Ad-hoc publishing. Call from any view/engine code:
        publish_engine_meta("Portfolio", weights_date=...)
    """
    store = st.session_state.setdefault("_engine_meta_published", {})
    sec = store.setdefault(section, {})
    for k, v in mapping.items():
        if v not in (None, "", []):
            sec[k] = v

def publish_engine_meta_summary(**mapping: Any) -> None:
    """
    Publish items to the summary row of the chip.
        publish_engine_meta_summary(reference_date=..., currency=...)
    """
    store = st.session_state.setdefault("_engine_meta_published_summary", {})
    for k, v in mapping.items():
        if v not in (None, "", []):
            store[k] = v

def clear_published_engine_meta(section: Optional[str] = None) -> None:
    if section is None:
        st.session_state.pop("_engine_meta_published", None)
        st.session_state.pop("_engine_meta_published_summary", None)
    else:
        store = st.session_state.get("_engine_meta_published", {})
        store.pop(section, None)

# -------------------------- Collector -----------------------------------------

def collect_engine_meta_from_ctx(ctx: Any) -> EngineMeta:
    """
    Build the HUD payload purely from providers + published state.
    No static fields; providers decide what to expose, using `ctx` if they want.
    """
    meta = EngineMeta()

    # 1) Providers mutate `meta`
    for fn in _META_PROVIDERS:
        try:
            fn(ctx, meta)
        except Exception:
            # Keep HUD resilient even if a provider fails
            continue

    # 2) Merge published (ad-hoc) sections and summary last (to override providers)
    pub_sections = st.session_state.get("_engine_meta_published", {})
    for sec_name, mapping in dict(pub_sections).items():
        meta.add(sec_name, **mapping)

    pub_summary = st.session_state.get("_engine_meta_published_summary", {})
    if pub_summary:
        meta.add_summary(**pub_summary)

    return meta

# -------------------------- Renderer ------------------------------------------

def render_engine_status(
    meta: EngineMeta | Mapping[str, Any] | Iterable[Tuple[str, Any]],
    *,
    app_sections: Optional[Dict[str, Mapping[str, Any]]] = None,
    title: str = "Pricing engine",
    mode: str = "sticky_bar",          # "sticky_bar" (default) or "floating_chip"
    position: str = "top-right",       # only used for floating mode
    open: bool = False,
) -> None:
    # Build content (summary + sections)
    summary_pairs, section_blocks = _normalize_for_render(meta)
    app_blocks = []
    if app_sections:
        for name, mapping in app_sections.items():
            app_blocks.append(_render_section(name, mapping))

    # ------- MODE A: STICKY BAR (recommended) -------
    if mode == "sticky_bar":
        # Always-on mini bar aligned to the right, non-intrusive
        bar_style = (
            "position: sticky; top: 0; z-index: 9999; width: 100%; "
            "margin: 0; padding: 0;"
        )
        # Right-aligned inner row
        row_style = (
            "display: flex; justify-content: flex-end; align-items: center; "
            "gap: 8px; padding: 6px 12px 2px 12px;"
        )
        # Compact chip
        chip_style = (
            "cursor: pointer; padding: .25rem .5rem; border-radius: 8px; "
            "border: 1px solid rgba(140,149,159,.3); "
            "background: rgba(30,30,30,.65); color: inherit; "
            "backdrop-filter: blur(3px); font-size: .80rem; user-select: none;"
        )
        grid_style = "display:grid; grid-template-columns:auto auto; gap:.25rem .75rem; margin-top:.5rem;"
        row_label = "font-weight:600; opacity:.75; margin-right:.5rem;"
        row_value = "font-variant-numeric:tabular-nums; overflow:hidden; text-overflow:ellipsis;"
        section_style = "margin-top:.5rem; border-top:1px dashed rgba(140,149,159,.35); padding-top:.5rem;"

        # Summary line (single compact string)
        summary_text = " · ".join(f"{k}: {_fmt(v)}" for k, v in summary_pairs) or title
        details_html = "".join(section_blocks + app_blocks)
        # If there are no sections, we still show the summary in the chip
        body_html = (
            f"<div style='{grid_style}'>"
            + "".join(
                f"<div style='white-space:nowrap;'><span style='{row_label}'>{k}</span>"
                f"<span style='{row_value}'>{_fmt(v)}</span></div>"
                for k, v in summary_pairs
            )
            + "</div>"
            + (f"<div style='{section_style}'>{details_html}</div>" if details_html else "")
        )

        html = f"""
        <div style="{bar_style}">
          <div style="{row_style}">
            <details style="{chip_style}"{' open' if open else ''}>
              <summary>ⓘ {summary_text}</summary>
              {body_html}
            </details>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        return

    # ------- MODE B: FLOATING CHIP (kept for completeness) -------
    top, lr = (8, 12)
    if position == "top-right":
        pos_css = f"top:{top}px; right:{lr}px;"
    elif position == "top-left":
        pos_css = f"top:{top}px; left:{lr}px;"
    elif position == "bottom-right":
        pos_css = f"bottom:{top}px; right:{lr}px;"
    else:
        pos_css = f"bottom:{top}px; left:{lr}px;"

    box_style = f"position:fixed; z-index:99999; {pos_css} max-width:min(92vw,560px);"
    chip_style = (
        "cursor:pointer; padding:.5rem .75rem; border-radius:10px; "
        "border:1px solid rgba(140,149,159,.3); background:rgba(30,30,30,.85); "
        "backdrop-filter:blur(4px); font-size:.85rem; user-select:none;"
    )
    grid_style = "display:grid; grid-template-columns:auto auto; gap:.25rem .75rem; margin-top:.5rem;"
    row_label = "font-weight:600; opacity:.7; margin-right:.5rem;"
    row_value = "font-variant-numeric:tabular-nums; overflow:hidden; text-overflow:ellipsis;"
    section_style = "margin-top:.5rem; border-top:1px dashed rgba(140,149,159,.35); padding-top:.5rem;"

    summary_html = "".join(
        f"<div style='white-space:nowrap;'><span style='{row_label}'>{k}</span>"
        f"<span style='{row_value}'>{_fmt(v)}</span></div>"
        for k, v in summary_pairs
    )
    sections_html = "".join(section_blocks + app_blocks)

    html = f"""
    <div style="{box_style}">
      <details style="{chip_style}"{' open' if open else ''}>
        <summary>{title}</summary>
        <div style="{grid_style}">{summary_html}</div>
        <div style="{section_style}">{sections_html}</div>
      </details>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
# -------------------------- Internals -----------------------------------------

_HUD_CSS = """
<style>
.engine-status { position: fixed; z-index: 1000; }
.es-chip {
  cursor: pointer;
  padding: .5rem .75rem;
  border-radius: 10px;
  border: 1px solid rgba(140, 149, 159, .3);
  background: rgba(255,255,255,.92);
  backdrop-filter: blur(4px);
  box-shadow: 0 2px 8px rgba(0,0,0,.06);
  font-size: .85rem;
  user-select: none;
  max-width: min(92vw, 560px);
}
.es-chip summary { list-style: none; }
.es-chip summary::-webkit-details-marker { display: none; }
.es-chip summary::after { content: " ⓘ"; opacity: .7; font-weight: 600; }
.es-grid { display: grid; grid-template-columns: auto auto; gap: .25rem .75rem; margin-top: .5rem; }
.es-row { white-space: nowrap; }
.es-label { font-weight: 600; opacity: .7; margin-right: .5rem; }
.es-value { font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; }
.es-section { margin-top: .5rem; border-top: 1px dashed rgba(140,149,159,.4); padding-top: .5rem; }
.es-section h4 { margin: 0 0 .25rem 0; font-size: .8rem; opacity: .8; }
@media (prefers-color-scheme: dark) {
  .es-chip { background: rgba(30,30,30,.85); color: #e6e6e6; border-color: rgba(140,149,159,.35); }
}
</style>
"""

def _fmt(v: Any) -> str:
    try:
        iso = getattr(v, "ISO", None)  # QuantLib.Date
        if callable(iso):
            return str(iso())
    except Exception:
        pass
    if isinstance(v, dt.datetime):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, dt.date):
        return v.isoformat()
    return str(v)

def _normalize_for_render(meta: EngineMeta | Mapping[str, Any] | Iterable[Tuple[str, Any]]):
    if isinstance(meta, EngineMeta):
        summary_pairs = list(meta.summary.items())
        # Respect caller-defined section order if provided
        names = meta.order or sorted(meta.sections.keys())
        blocks = [_render_section(name, meta.sections[name]) for name in names if meta.sections.get(name)]
        return summary_pairs, blocks

    # Mapping or pairs -> summary only (no sections)
    mapping = dict(meta) if isinstance(meta, Mapping) else dict(meta)
    return list(mapping.items()), []

def _render_section(name: str, mapping: Mapping[str, Any]) -> str:
    rows = "".join(
        f"<div class='es-row'><span class='es-label'>{str(k)}</span>"
        f"<span class='es-value'>{_fmt(v)}</span></div>"
        for k, v in dict(mapping).items() if v not in (None, "", [])
    )
    return f"<div class='es-section'><h4>{name}</h4><div class='es-grid'>{rows}</div></div>"
