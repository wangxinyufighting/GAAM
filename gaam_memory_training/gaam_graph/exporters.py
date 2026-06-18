from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import networkx as nx

from .utils import write_json


def graph_to_json(g: nx.MultiDiGraph) -> Dict[str, Any]:
    return {
        "graph": dict(g.graph),
        "nodes": [
            {"id": nid, **attrs}
            for nid, attrs in g.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, "key": k, **attrs}
            for u, v, k, attrs in g.edges(keys=True, data=True)
        ],
    }


def export_graph(g: nx.MultiDiGraph, output_prefix: str | Path, evidence_pack: Dict[str, Any]) -> None:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(output_prefix) + ".graph.json", graph_to_json(g))
    write_json(str(output_prefix) + ".evidence_pack.json", evidence_pack)
    graphml_g = _graphml_safe(g)
    nx.write_graphml(graphml_g, str(output_prefix) + ".graphml")


def _graphml_safe(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    out = nx.MultiDiGraph()
    out.graph.update({k: _safe_value(v) for k, v in g.graph.items()})
    for nid, attrs in g.nodes(data=True):
        out.add_node(nid, **{k: _safe_value(v) for k, v in attrs.items()})
    for u, v, k, attrs in g.edges(keys=True, data=True):
        out.add_edge(u, v, key=k, **{kk: _safe_value(vv) for kk, vv in attrs.items()})
    return out


def _safe_value(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)
