from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .utils import normalize_text


DEFAULT_TYPE_CHAR_LIMITS = {
    "event": 0,
    "fact": 0,
    "abstract_memory": 0,
    "entity": 0,
    "topic": 0,
    "session": 0,
}

INFORMATIVE_NODE_WEIGHTS = {
    "fact": 6.0,
    "abstract_memory": 5.0,
    "event": 2.4,
    "entity": 1.6,
    "topic": 0.7,
    "session": 0.2,
}

STRUCTURAL_NODE_TYPES = {"topic", "session"}
EVIDENCE_NODE_TYPES = {"fact", "abstract_memory"}
EVIDENCE_EDGE_TYPES = {"EXTRACTED_AS", "SUPPORTS", "FACT_SUBJECT", "FACT_OBJECT"}
GENERIC_ENTITY_NAMES = {"user", "assistant", "speaker", "unknown"}


@dataclass(frozen=True)
class EdgeRef:
    source: str
    target: str
    key: int
    attrs: Dict[str, Any]
    traversal: str


class GraphWalker:
    """Sample readable trajectories from exported GAAM graph JSON."""

    def __init__(
        self,
        graph_json: Dict[str, Any],
        *,
        seed: Optional[int] = None,
        directed: bool = False,
        max_node_chars: int = 0,
        type_char_limits: Optional[Dict[str, int]] = None,
    ):
        self.graph_json = graph_json
        self.directed = directed
        self.max_node_chars = max_node_chars
        self.type_char_limits = dict(DEFAULT_TYPE_CHAR_LIMITS)
        if type_char_limits:
            self.type_char_limits.update(type_char_limits)
        self.rng = random.Random(seed)
        self.nodes_by_id = {node["id"]: node for node in graph_json.get("nodes", [])}
        self.adjacency: Dict[str, List[EdgeRef]] = {node_id: [] for node_id in self.nodes_by_id}
        self._build_adjacency()
        self.sessions_by_node = self._build_node_sessions()

    def sample_walks(
        self,
        num_walks: int,
        *,
        walk_length: int = 5,
        start_node_types: Optional[List[str]] = None,
        min_nodes: int = 2,
        max_structural_ratio: float = 0.35,
        require_evidence_node: bool = True,
        max_consecutive_next: int = 1,
        require_multi_session: bool = False,
        min_sessions: int = 2,
    ) -> List[Dict[str, Any]]:
        start_node_types = start_node_types or ["abstract_memory", "fact"]
        candidates = [
            node_id
            for node_id, node in self.nodes_by_id.items()
            if node.get("type") in start_node_types and self.adjacency.get(node_id)
        ]
        if not candidates:
            candidates = [node_id for node_id in self.nodes_by_id if self.adjacency.get(node_id)]

        walks: List[Dict[str, Any]] = []
        seen = set()
        attempts = 0
        max_attempts = max(num_walks * (200 if require_multi_session else 30), 30)
        while len(walks) < num_walks and attempts < max_attempts and candidates:
            attempts += 1
            start_id = self.rng.choice(candidates)
            walk = self._walk(start_id, walk_length=walk_length, max_consecutive_next=max_consecutive_next)
            if len(walk["nodes"]) < min_nodes:
                continue
            if require_evidence_node and not self._has_evidence_node(walk["nodes"]):
                continue
            if require_multi_session and len(walk.get("session_ids", [])) < min_sessions:
                continue
            if self._structural_ratio(walk["nodes"]) > max_structural_ratio:
                continue
            signature = tuple(node["id"] for node in walk["nodes"])
            if signature in seen:
                continue
            seen.add(signature)
            walk["trajectory_id"] = f"traj_{len(walks) + 1:04d}"
            walks.append(walk)
        return walks

    def _build_adjacency(self) -> None:
        for edge in self.graph_json.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source not in self.nodes_by_id or target not in self.nodes_by_id:
                continue
            key = int(edge.get("key", 0))
            attrs = {k: v for k, v in edge.items() if k not in {"source", "target", "key"}}
            self.adjacency.setdefault(source, []).append(EdgeRef(source, target, key, attrs, "out"))
            if not self.directed:
                self.adjacency.setdefault(target, []).append(EdgeRef(source, target, key, attrs, "in"))

    def _walk(self, start_id: str, *, walk_length: int, max_consecutive_next: int) -> Dict[str, Any]:
        node_ids = [start_id]
        edge_refs: List[EdgeRef] = []
        previous_id: Optional[str] = None
        current_id = start_id
        consecutive_next = 0
        for _ in range(max(walk_length - 1, 0)):
            options = self._candidate_edges(current_id, previous_id, consecutive_next, max_consecutive_next)
            if not options:
                break
            edge = self._weighted_choice(options)
            next_id = edge.target if edge.traversal == "out" else edge.source
            node_ids.append(next_id)
            edge_refs.append(edge)
            previous_id = current_id
            current_id = next_id
            if edge.attrs.get("type") == "NEXT":
                consecutive_next += 1
            else:
                consecutive_next = 0
        return {
            "nodes": [self._node_view(node_id) for node_id in node_ids],
            "edges": [self._edge_view(edge) for edge in edge_refs],
            "session_ids": self._walk_session_ids(node_ids),
            "text": self.format_walk_text(node_ids, edge_refs),
        }

    def _build_node_sessions(self) -> Dict[str, List[str]]:
        sessions = {node_id: set() for node_id in self.nodes_by_id}

        for node_id, node in self.nodes_by_id.items():
            session_id = node.get("session_id")
            if node.get("type") in {"session", "event"} and session_id:
                sessions[node_id].add(str(session_id))
            source_event_id = node.get("source_event_id")
            if node.get("type") == "fact" and source_event_id in self.nodes_by_id:
                event_session = self.nodes_by_id[source_event_id].get("session_id")
                if event_session:
                    sessions[node_id].add(str(event_session))

        for edge in self.graph_json.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("type")
            if source not in sessions or target not in sessions:
                continue
            if edge_type == "EXTRACTED_AS":
                sessions[target].update(sessions[source])

        for edge in self.graph_json.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("type")
            if source not in sessions or target not in sessions:
                continue
            if edge_type == "SUPPORTS":
                sessions[target].update(sessions[source])
            elif edge_type in {"FACT_SUBJECT", "FACT_OBJECT", "MENTIONS"}:
                sessions[target].update(sessions[source])

        return {node_id: sorted(values) for node_id, values in sessions.items()}

    def _candidate_edges(
        self,
        current_id: str,
        previous_id: Optional[str],
        consecutive_next: int,
        max_consecutive_next: int,
    ) -> List[EdgeRef]:
        options = self.adjacency.get(current_id, [])
        options = [edge for edge in options if not self._uses_generic_entity_bridge(edge)]
        if previous_id is None:
            no_backtrack = options
        else:
            no_backtrack = [
                edge
                for edge in options
                if (edge.target if edge.traversal == "out" else edge.source) != previous_id
            ]
        if max_consecutive_next < 0:
            return no_backtrack
        return [
            edge
            for edge in no_backtrack
            if not (edge.attrs.get("type") == "NEXT" and consecutive_next >= max_consecutive_next)
        ]

    def _uses_generic_entity_bridge(self, edge: EdgeRef) -> bool:
        if edge.attrs.get("type") not in {"FACT_SUBJECT", "FACT_OBJECT", "MENTIONS"}:
            return False
        source = self.nodes_by_id.get(edge.source, {})
        target = self.nodes_by_id.get(edge.target, {})
        return self._is_generic_entity(source) or self._is_generic_entity(target)

    def _weighted_choice(self, options: List[EdgeRef]) -> EdgeRef:
        weights = []
        for edge in options:
            next_id = edge.target if edge.traversal == "out" else edge.source
            next_type = self.nodes_by_id.get(next_id, {}).get("type", "unknown")
            edge_type = edge.attrs.get("type", "")
            weight = INFORMATIVE_NODE_WEIGHTS.get(next_type, 1.0)
            if edge_type in EVIDENCE_EDGE_TYPES:
                weight *= 2.0
            elif edge_type in {"BELONGS_TO", "BELONGS_TO_TOPIC", "ABSTRACTS"}:
                weight *= 0.5
            elif edge_type == "NEXT":
                weight *= 0.55
            weights.append(weight)
        return self.rng.choices(options, weights=weights, k=1)[0]

    def _has_evidence_node(self, nodes: List[Dict[str, Any]]) -> bool:
        return any(node.get("type") in EVIDENCE_NODE_TYPES for node in nodes)

    def _is_generic_entity(self, node: Dict[str, Any]) -> bool:
        if node.get("type") != "entity":
            return False
        name = normalize_text(str(node.get("name", ""))).lower()
        return name in GENERIC_ENTITY_NAMES

    def _structural_ratio(self, nodes: List[Dict[str, Any]]) -> float:
        if not nodes:
            return 1.0
        structural = sum(1 for node in nodes if node.get("type") in STRUCTURAL_NODE_TYPES)
        return structural / len(nodes)

    def _walk_session_ids(self, node_ids: List[str]) -> List[str]:
        session_ids = set()
        for node_id in node_ids:
            session_ids.update(self.sessions_by_node.get(node_id, []))
        return sorted(session_ids)

    def _node_view(self, node_id: str) -> Dict[str, Any]:
        node = self.nodes_by_id[node_id]
        node_type = node.get("type", "unknown")
        label, truncated, source = self._node_label(node)
        view: Dict[str, Any] = {
            "id": node_id,
            "type": node_type,
            "label": label,
            "label_source": source,
            "label_truncated": truncated,
        }
        session_ids = self.sessions_by_node.get(node_id, [])
        if session_ids:
            view["session_ids"] = session_ids
            if len(session_ids) == 1:
                view["source_session_id"] = session_ids[0]
        for key in ["speaker", "timestamp", "subject", "predicate", "object", "status", "scope", "confidence"]:
            if key in node:
                view[key] = node[key]
        return view

    def _edge_view(self, edge: EdgeRef) -> Dict[str, Any]:
        return {
            "source": edge.source,
            "target": edge.target,
            "type": edge.attrs.get("type", "RELATED_TO"),
            "traversal": edge.traversal,
        }

    def _node_label(self, node: Dict[str, Any]) -> tuple[str, bool, str]:
        node_type = node.get("type")
        if node_type == "event":
            speaker = node.get("speaker") or "unknown"
            return (*self._clip(f"{speaker}: {node.get('text', '')}", node_type=node_type), "speaker:text")
        if node_type == "fact":
            text = node.get("text") or f"{node.get('subject', '')} {node.get('predicate', '')} {node.get('object', '')}"
            return (*self._clip(text, node_type=node_type), "text")
        if node_type == "abstract_memory":
            return (*self._clip(node.get("summary", ""), node_type=node_type), "summary")
        if node_type == "entity":
            return (*self._clip(node.get("name", ""), node_type=node_type), "name")
        if node_type == "topic":
            return (*self._clip(node.get("name", ""), node_type=node_type), "name")
        if node_type == "session":
            return (*self._clip(node.get("session_id", ""), node_type=node_type), "session_id")
        return (*self._clip(str(node), node_type=node_type), "node")

    def _edge_label(self, edge: EdgeRef) -> str:
        edge_type = edge.attrs.get("type", "RELATED_TO")
        arrow = "->" if edge.traversal == "out" else "<-"
        return f"{arrow} {edge_type} {arrow}"

    def _clip(self, text: str, *, node_type: str) -> tuple[str, bool]:
        text = normalize_text(str(text))
        limit = self.type_char_limits.get(node_type, self.max_node_chars)
        if limit <= 0 or len(text) <= limit:
            return text, False
        return text[: limit - 3].rstrip() + "...", True

    def format_walk_text(self, node_ids: List[str], edge_refs: List[EdgeRef]) -> str:
        first_node = self.nodes_by_id[node_ids[0]]
        first_label, _, _ = self._node_label(first_node)
        parts = [f"[{first_node.get('type')}] {first_label}"]
        for edge, node_id in zip(edge_refs, node_ids[1:]):
            node = self.nodes_by_id[node_id]
            label, _, _ = self._node_label(node)
            parts.append(self._edge_label(edge))
            parts.append(f"[{node.get('type')}] {label}")
        return " ".join(parts)
