# dashboards/apps/floating_portfolio_analysis/pages/02_Data_Nodes_Graph.py
from __future__ import annotations
from typing import Any, Dict
import re, json
import streamlit as st

st.title("Data Nodes — Dependencies")

def _sanitize_id(x: Any) -> str:
    s = f"{x}"
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    if not re.match(r"^[A-Za-z_]", s):
        s = f"n_{s}"
    return s

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = (hex_color or "").strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)
    try:
        r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
        return r, g, b
    except Exception:
        return 238, 238, 238  # fallback #EEE

def _ideal_text_color(bg_hex: str) -> str:
    r, g, b = _hex_to_rgb(bg_hex or "#EEEEEE")
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    return "#000000" if luminance > 145 else "#FFFFFF"

def build_mermaid_from_graph(graph: Dict[str, Any], direction: str = "TD") -> str:
    nodes = graph.get("nodes", []); edges = graph.get("edges", [])
    id_map = {n["id"]: _sanitize_id(n["id"]) for n in nodes}
    lines = ["%%{init: {'theme':'neutral'}}%%", f"graph {direction}"]
    for n in nodes:
        nid = id_map[n["id"]]
        title = n.get("card_title") or str(n["id"])
        subtitle = n.get("card_subtitle") or ""
        label = title if not subtitle else f"{title}\\n{subtitle}"
        lines.append(f'  {nid}["{label}"]')
    for e in edges:
        s = id_map.get(e["source"], _sanitize_id(e["source"]))
        t = id_map.get(e["target"], _sanitize_id(e["target"]))
        lines.append(f"  {s} --> {t}")
    for n in nodes:
        nid = id_map[n["id"]]
        fill = n.get("color") or "#EEEEEE"; stroke = "#666666"; text = _ideal_text_color(fill)
        lines.append(f"  style {nid} fill:{fill},stroke:{stroke},stroke-width:1px,color:{text}")
    return "\n".join(lines)

def _mock_fetch_dependencies(_: Any = None) -> Dict[str, Any]:
    return {
        'edges': [{'source': 66, 'target': 'API_26'},
                  {'source': 66, 'target': 'API_41'},
                  {'source': 737, 'target': 66}],
        'nodes': [
            {'id': 737,'update_hash':'interpolatedprices_01b6381e96aaf7ccdb182a44088ed97a','card_title':'InterpolatedPrices','depth':0,'color':'#9E9E9E','background_color':'#9E9E9E','icon':'\uf2f2',
             'properties': {'update_hash':'interpolatedprices_01b6381e96aaf7ccdb182a44088ed97a','human_readable':'interpolatedprices_b976a9c6374e1c755c638a9beb925862','local_time_serie_id':737,'remote_table_hash_id':'interpolatedprices_b976a9c6374e1c755c638a9beb925862','remote_table_id':28,'data_source_id':2,'error_on_last_update':False,'last_update':'2010-01-01T00:00:00Z','next_update':'2010-01-01T00:01:00Z'},
             'card_subtitle':'','badges':[]},
            {'id': 66,'update_hash':'wrapperdatanode_28e4106768c376d879cb0b10fad12ee3','card_title':'WrapperDataNode','depth':1,'color':'#EEEEEE','background_color':'#EEEEEE','icon':'\uf5fd',
             'properties': {'update_hash':'wrapperdatanode_28e4106768c376d879cb0b10fad12ee3','human_readable':'wrapperdatanode_83bf812e3e71b8c39c9ea6b751935611','local_time_serie_id':66,'remote_table_hash_id':'wrapperdatanode_83bf812e3e71b8c39c9ea6b751935611','remote_table_id':27,'data_source_id':2,'error_on_last_update':False,'last_update':'2010-01-01T00:00:00Z','next_update':'2010-01-01T00:01:00Z'},
             'card_subtitle':'','badges':[]},
            {'id': 'API_26','update_hash':'API_26','card_title':'AlpacaEquityBars','depth':2,'color':'#6200EE','background_color':'#EEEEEE','icon':'\uf6ff',
             'properties': {'update_hash':'None','human_readable':'alpaca_1d_bars','local_time_serie_id':None,'remote_table_hash_id':'alpacaequitybars_6f05109da045164afc0d9143dcb986f4','remote_table_id':26,'data_source_id':2,'is_api':True},
             'card_subtitle':'API remote update','badges':[]},
            {'id': 'API_41','update_hash':'API_41','card_title':'BinanceHistoricalBars','depth':2,'color':'#6200EE','background_color':'#EEEEEE','icon':'\uf6ff',
             'properties': {'update_hash':'None','human_readable':'binance_1d_bars','local_time_serie_id':None,'remote_table_hash_id':'binancehistoricalbars_29460104b75552b77f1191f3f5dfa5be','remote_table_id':41,'data_source_id':2,'is_api':True},
             'card_subtitle':'API remote update','badges':[]}
        ],
        'groups': []
    }

col1, col2 = st.columns([3, 1])
with col1:
    direction = st.selectbox("Layout", ["TD", "LR", "BT", "RL"], index=0)
with col2:
    show_raw = st.toggle("Show raw payload", value=False)

graph = _mock_fetch_dependencies()
mermaid_text = build_mermaid_from_graph(graph, direction=direction)

if hasattr(st, "mermaid"):
    st.mermaid(mermaid_text)
else:
    st.markdown(f"```mermaid\n{mermaid_text}\n```")

if show_raw:
    try:
        st.json(graph)
    except Exception:
        st.code(json.dumps(graph, indent=2))

st.download_button(
    "Download .mmd",
    data=mermaid_text.encode("utf-8"),
    file_name="data_nodes_dependencies.mmd",
    mime="text/plain",
)
