from __future__ import annotations
from typing import Any, Dict, List, Optional
import streamlit as st
import mainsequence.client as msc






@st.cache_data(show_spinner=False)
def _search_portfolios_cache(q: str) -> List[Dict[str, Any]]:
    """
    JSON-safe cache: returns [{'id': int, 'name': str, 'instance_json': str}, ...]
    """
    if not q or len(q.strip()) < 3:
        return []
    if msc is None:
        return []

    results = msc.PortfolioIndexAsset.filter(current_snapshot__name__contains=q.strip())
    out: List[Dict[str, Any]] = []
    for p in (results or []):
        try:
            # p is a Pydantic model (v2 in your repo). Dump to JSON string.
            out.append({
                "id": int(p.id),
                "name": getattr(p.current_snapshot, "name", str(p.id)),
                "instance_json": p.model_dump_json(),
            })
        except Exception:
            # fall back to minimal info; the wrapper will refetch by id if needed
            out.append({"id": int(getattr(p, "id", 0)), "name": getattr(getattr(p, "current_snapshot", None), "name", "—")})
    return out


# 2) Public function (same name as before) rehydrates instances for the caller.
#    No @st.cache_data here — it uses the cached JSON above and builds objects.
def _search_portfolios(q: str) -> List[Dict[str, Any]]:
    """
    Search portfolios and return [{'id','name','instance'}] where 'instance' is a
    real PortfolioIndexAsset. Uses a JSON-only cached layer to satisfy Streamlit.
    """
    raw = _search_portfolios_cache(q)

    # Rehydrate PortfolioIndexAsset instances from the cached JSON
    out: List[Dict[str, Any]] = []
    for rec in raw:
        pid = rec.get("id")
        name = rec.get("name")

        inst = None
        js = rec.get("instance_json")
        if js:
            # Try Pydantic v2, then v1, then dict, finally a DB fetch by id.
            try:
                inst = msc.PortfolioIndexAsset.model_validate_json(js)  # pydantic v2
            except Exception:
                try:
                    inst = msc.PortfolioIndexAsset.parse_raw(js)         # pydantic v1
                except Exception:
                    try:
                        import json as _json
                        inst = msc.PortfolioIndexAsset.model_validate(_json.loads(js))  # v2 dict
                    except Exception:
                        pass

        if inst is None and pid is not None:
            try:
                inst = msc.PortfolioIndexAsset.get_or_none(id=int(pid))  # last resort
            except Exception:
                inst = None

        if inst is not None:
            out.append({"id": pid, "name": name, "instance": inst})
        else:
            # If we couldn't rebuild, skip the row to avoid breaking the UI.
            # (or keep it with instance=None if you prefer)
            continue

    return out


def sidebar_portfolio_multi_select(
    *,
    title: str = "Build position from portfolio (search)",
    key_prefix: str = "portfolio_select",
    min_chars: int = 3,
) -> Optional[List[Any]]:
    """
    Streamlit sidebar widget:
      - Text search (cached)
      - Multi-select of results
      - 'Load selected' returns list of PortfolioIndexAsset instances (and persists them)
      - 'Clear selection' resets both current UI and persisted loaded selection
    Returns the list only when 'Load selected' is clicked; otherwise None.
    Persisted selection is available in:
      • st.session_state[f"{key_prefix}_loaded_ids"]         -> Tuple[int, ...]
      • st.session_state[f"{key_prefix}_loaded_instances"]   -> List[PortfolioIndexAsset]
    """
    loaded_ids_key = f"{key_prefix}_loaded_ids"
    loaded_insts_key = f"{key_prefix}_loaded_instances"
    results_key = f"{key_prefix}_results"
    select_key = f"{key_prefix}_ids"
    query_key = f"{key_prefix}_q"

    with st.sidebar:
        st.markdown(f"### {title}")
        st.caption(f"Type at least **{min_chars}** characters to search portfolios")

        # Callback: run search when user presses Enter
        def _do_search_from_state(prefix: str, min_chars_local: int) -> None:
            q_local = st.session_state.get(f"{prefix}_q", "")
            if q_local and len(q_local.strip()) >= min_chars_local:
                st.session_state[f"{prefix}_results"] = _search_portfolios(q_local.strip())
            else:
                # Clear stale results if user erased the query
                st.session_state.pop(f"{prefix}_results", None)

        q = st.text_input(
            "Search portfolios",
            placeholder="e.g. MXN ALM Desk …",
            key=query_key,
            on_change=_do_search_from_state,
            args=(key_prefix, min_chars),
        )

        # Manual trigger still available
        trigger = st.button("Search", key=f"{key_prefix}_btn")
        if trigger and q and len(q.strip()) >= min_chars:
            st.session_state[results_key] = _search_portfolios(q.strip())

        portfolios = st.session_state.get(results_key, [])
        if q and len(q.strip()) < min_chars:
            st.write(" ")
            st.caption(f"Keep typing… need **{min_chars}+** characters to search.")

        # If we have results, show a multi-select to build a *candidate* selection.
        if portfolios:
            options = [p["id"] for p in portfolios]
            label_of = {p["id"]: f"{p['name']} (id={p['id']})" for p in portfolios}
            selected_ids = st.multiselect(
                "Select portfolios",
                options=options,
                format_func=lambda pid: label_of.get(pid, str(pid)),
                key=select_key,
            )

            col_load, col_clear = st.columns([3, 2])
            with col_load:
                if st.button("Load selected", type="primary", use_container_width=True, key=f"{key_prefix}_load"):
                    # Build instances *from the displayed results* for the chosen IDs
                    id2inst = {p["id"]: p["instance"] for p in portfolios}
                    try:
                        instances = [id2inst[pid] for pid in selected_ids]
                    except KeyError:
                        instances = []

                    # Persist the exact loaded selection (by id **and** by object)
                    st.session_state[loaded_ids_key] = tuple(sorted(int(i) for i in selected_ids))
                    st.session_state[loaded_insts_key] = instances

                    # Return only on explicit Load
                    return instances

            with col_clear:
                if st.button("Clear selection", use_container_width=True, key=f"{key_prefix}_clear"):
                    # Clear UI state AND the persisted loaded selection
                    st.session_state.pop(select_key, None)
                    st.session_state.pop(results_key, None)
                    st.session_state.pop(loaded_ids_key, None)
                    st.session_state.pop(loaded_insts_key, None)
                    st.rerun()

        # If no current results, still show what’s *already* loaded (informational only).
        loaded_ids = st.session_state.get(loaded_ids_key)
        if loaded_ids:
            st.caption(f"Loaded selection: {len(loaded_ids)} portfolio(s) kept in memory")

    loaded_instances = st.session_state.get(loaded_insts_key)
    if loaded_instances:
        return loaded_instances
    return None


@st.cache_data(show_spinner=False)
def _search_portfolio_groups(q: str) -> List[Dict[str, Any]]:
    """
    Search MainSequence PortfolioGroup by display_name / unique_identifier (contains).
    Returns a list of {'id', 'display_name', 'unique_identifier', 'instance'} dicts.
    """
    if not q or len(q.strip()) < 3:
        return []
    if msc is None:
        return []

    q = q.strip()
    results_by_id: Dict[int, Any] = {}

    # Search by display_name
    by_name = msc.PortfolioGroup.filter(display_name__contains=q)
    for g in (by_name or []):
        results_by_id[g.id] = g

    # Search by unique_identifier
    by_uid = msc.PortfolioGroup.filter(unique_identifier__contains=q)
    for g in (by_uid or []):
        results_by_id[g.id] = g


    out: List[Dict[str, Any]] = []
    for g in results_by_id.values():
        gid=g.id
        uid = g.unique_identifier
        name = g.display_name
        out.append({
            "id": gid,
            "display_name": name,
            "unique_identifier": uid,
            "instance": g,
        })

    # Nice ordering by display name (fallback to uid)
    out.sort(key=lambda r: r.get("display_name").lower())
    return out


def sidebar_portfolio_group_multi_select(
    *,
    title: str = "Portfolio groups (search & select)",
    key_prefix: str = "portfolio_group_select",
    min_chars: int = 3,
) -> Optional[List[Any]]:
    """
    Streamlit sidebar widget for selecting PortfolioGroup(s).

    UX:
      • Text search (cached)
      • Multi-select on results
      • "Load selected" returns a list of PortfolioGroup instances (and persists them)
      • "Clear selection" resets both UI and persisted selection

    Returns:
      - list[PortfolioGroup] only when "Load selected" is pressed; otherwise None.

    Session keys (customized by key_prefix):
      • st.session_state[f"{key_prefix}_loaded_ids"]         -> Tuple[int, ...]
      • st.session_state[f"{key_prefix}_loaded_instances"]   -> List[PortfolioGroup]
    """
    loaded_ids_key = f"{key_prefix}_loaded_ids"
    loaded_insts_key = f"{key_prefix}_loaded_instances"
    results_key = f"{key_prefix}_results"
    select_key = f"{key_prefix}_ids"
    query_key = f"{key_prefix}_q"

    with st.sidebar:
        st.markdown(f"### {title}")
        st.caption(f"Type at least **{min_chars}** characters to search by display name or UID")

        # Run search when user presses Enter in the input
        def _do_search_from_state(prefix: str, min_chars_local: int) -> None:
            q_local = st.session_state.get(f"{prefix}_q", "")
            if q_local and len(q_local.strip()) >= min_chars_local:
                st.session_state[f"{prefix}_results"] = _search_portfolio_groups(q_local.strip())
            else:
                st.session_state.pop(f"{prefix}_results", None)

        q = st.text_input(
            "Search groups",
            placeholder="e.g. ALM, Rates, MXN…",
            key=query_key,
            on_change=_do_search_from_state,
            args=(key_prefix, min_chars),
        )

        # Manual trigger (optional)
        trigger = st.button("Search", key=f"{key_prefix}_btn")
        if trigger and q and len(q.strip()) >= min_chars:
            st.session_state[results_key] = _search_portfolio_groups(q.strip())

        groups = st.session_state.get(results_key, [])
        if q and len(q.strip()) < min_chars:
            st.write(" ")
            st.caption(f"Keep typing… need **{min_chars}+** characters to search.")

        # Results → multiselect
        if groups:
            options = [g["id"] for g in groups if g.get("id") is not None]
            id_to_label = {
                g["id"]: f"{g.get('display_name') or g.get('unique_identifier') or g['id']}  ({g.get('unique_identifier') or '—'})"
                for g in groups if g.get("id") is not None
            }
            selected_ids = st.multiselect(
                "Select groups",
                options=options,
                format_func=lambda gid: id_to_label.get(gid, str(gid)),
                key=select_key,
            )

            col_load, col_clear = st.columns([3, 2])
            with col_load:
                if st.button("Load selected", type="primary", use_container_width=True, key=f"{key_prefix}_load"):
                    id2inst = {g["id"]: g["instance"] for g in groups if g.get("id") is not None}
                    instances = [id2inst[gid] for gid in selected_ids if gid in id2inst]

                    # Persist the exact loaded selection (by id **and** by object)
                    st.session_state[loaded_ids_key] = tuple(sorted(int(i) for i in selected_ids if i is not None))
                    st.session_state[loaded_insts_key] = instances

                    # Return only on explicit Load
                    return instances

            with col_clear:
                if st.button("Clear selection", use_container_width=True, key=f"{key_prefix}_clear"):
                    # Clear UI state AND the persisted loaded selection
                    st.session_state.pop(select_key, None)
                    st.session_state.pop(results_key, None)
                    st.session_state.pop(loaded_ids_key, None)
                    st.session_state.pop(loaded_insts_key, None)
                    st.rerun()

        # If no current results, still show what’s *already* loaded (informational only).
        loaded_ids = st.session_state.get(loaded_ids_key)
        if loaded_ids:
            st.caption(f"Loaded groups: {len(loaded_ids)} kept in memory")

    loaded_instances = st.session_state.get(loaded_insts_key)
    if loaded_instances:
        return loaded_instances
    return None
