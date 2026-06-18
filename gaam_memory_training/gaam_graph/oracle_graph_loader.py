from __future__ import annotations

import json
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

# Forbidden keys that indicate leakage of evaluation targets
FORBIDDEN_ARTIFACT_KEYS = {
    "question",
    "answer",
    "gold_answer",
    "question_type",
    "query",
    "target",
}

ALLOWED_TARGET_PATHS = {"edges[].target", "edges.target"}


class OracleGraphLeakageError(ValueError):
    """Raised when forbidden evaluation keys are found in oracle graph artifacts."""
    pass


class OracleGraphFormatError(ValueError):
    """Raised when oracle graph structure is malformed."""
    pass


class OracleGraphLoader:
    """
    Read-only access layer for oracle graph artifacts.

    Loads and validates oracle graph JSON files produced by build_lme_graph.py.
    Provides indexes, session mappings, and evidence context formatting.

    Never builds graphs, modifies artifacts, or calls LLMs.
    """

    def __init__(
        self,
        graph_path: str | Path,
        evidence_pack_path: str | Path | None = None,
        *,
        strict: bool = True,
        sanitize: bool = False,
    ) -> None:
        """
        Initialize loader with oracle graph artifact.

        Args:
            graph_path: Path to .graph.json file
            evidence_pack_path: Optional path to .evidence_pack.json file
            strict: If True (default), raise on leakage or malformed structure
            sanitize: If True, remove forbidden keys (debug only, not for training)

        Raises:
            ValueError: If both strict and sanitize are True
            OracleGraphLeakageError: If forbidden keys found in strict mode
            OracleGraphFormatError: If graph structure is invalid in strict mode
        """
        if strict and sanitize:
            raise ValueError("strict and sanitize cannot both be true")

        self.graph_path = Path(graph_path)
        self.evidence_pack_path = Path(evidence_pack_path) if evidence_pack_path else None
        self.strict = strict
        self.sanitize = sanitize

        # Load and validate graph
        self._graph_data = self._load_json(self.graph_path)
        self._sanitized_keys: List[str] = []

        if self.sanitize:
            self._sanitized_keys = self._sanitize_artifact(self._graph_data, "graph")

        if self.strict:
            self.validate_no_leakage()
            self.validate_integrity()

        # Load optional evidence pack
        self._evidence_data: Optional[Dict[str, Any]] = None
        if self.evidence_pack_path:
            self._evidence_data = self._load_json(self.evidence_pack_path)
            if self.sanitize:
                self._sanitized_keys.extend(
                    self._sanitize_artifact(self._evidence_data, "evidence_pack")
                )
            if self.strict:
                self._validate_no_leakage_in_dict(
                    self._evidence_data,
                    f"{self.evidence_pack_path.name}",
                    ""
                )

        # Build indexes
        self._build_indexes()
        self._leakage_checked = self.strict
        self._integrity_checked = self.strict

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _sanitize_artifact(self, data: Any, artifact_name: str) -> List[str]:
        """Remove forbidden keys from artifact (debug mode only)."""
        removed_keys = []
        self._recursive_sanitize(data, removed_keys, artifact_name, "")
        return removed_keys

    def _recursive_sanitize(
        self,
        obj: Any,
        removed_keys: List[str],
        artifact_name: str,
        path: str
    ) -> None:
        """Recursively remove forbidden keys."""
        if isinstance(obj, dict):
            keys_to_remove = []
            for key in obj.keys():
                json_path = f"{path}.{key}" if path else key
                if self._is_forbidden_key_at_path(key, json_path):
                    keys_to_remove.append(key)
                    removed_keys.append(f"{artifact_name}:{json_path}")

            for key in keys_to_remove:
                del obj[key]

            for key, value in obj.items():
                json_path = f"{path}.{key}" if path else key
                self._recursive_sanitize(value, removed_keys, artifact_name, json_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                json_path = f"{path}[{i}]"
                self._recursive_sanitize(item, removed_keys, artifact_name, json_path)

    def _build_indexes(self) -> None:
        """Build internal indexes for fast access."""
        nodes_list = self._graph_data.get("nodes", [])
        edges_list = self._graph_data.get("edges", [])

        # Node indexes
        self._nodes_by_id: Dict[str, Dict[str, Any]] = {}
        self._nodes_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for node in nodes_list:
            node_id = node.get("id")
            if node_id:
                self._nodes_by_id[node_id] = node
                node_type = node.get("type")
                if node_type:
                    self._nodes_by_type[node_type].append(node)

        # Edge indexes
        self._edges: List[Dict[str, Any]] = edges_list
        self._edges_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._out_edges_by_node: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._in_edges_by_node: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._edge_by_tuple: Dict[tuple, Dict[str, Any]] = {}

        for edge in edges_list:
            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("type")
            key = str(edge.get("key", 0))

            if edge_type:
                self._edges_by_type[edge_type].append(edge)

            if source:
                self._out_edges_by_node[source].append(edge)

            if target:
                self._in_edges_by_node[target].append(edge)

            if source and target:
                self._edge_by_tuple[(source, target, key)] = edge

        # Session indexes
        self._sessions_by_node: Dict[str, List[str]] = {}
        self._nodes_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._infer_node_sessions()

    def _infer_node_sessions(self) -> None:
        """Infer session IDs for each node based on graph structure.

        This uses fixed-point propagation so inference does not depend on the
        order of nodes in the JSON artifact.
        """
        session_sets: Dict[str, Set[str]] = {
            node_id: set() for node_id in self._nodes_by_id
        }

        # Seed direct session-bearing nodes first.
        for node_id, node in self._nodes_by_id.items():
            node_type = node.get("type")
            if node_type in {"session", "event"}:
                session_id = node.get("session_id")
                if session_id:
                    session_sets[node_id].add(str(session_id))

        # Propagate through explicit source_event_id links and graph edges until
        # no node gains new session IDs.
        changed = True
        while changed:
            changed = False
            for node_id, node in self._nodes_by_id.items():
                before = len(session_sets[node_id])
                node_type = node.get("type")

                if node_type == "fact":
                    source_event_id = node.get("source_event_id")
                    if source_event_id in session_sets:
                        session_sets[node_id].update(session_sets[source_event_id])

                for edge in self._in_edges_by_node.get(node_id, []):
                    edge_type = edge.get("type")
                    source = edge.get("source")
                    if source not in session_sets:
                        continue

                    if (
                        (node_type == "fact" and edge_type == "EXTRACTED_AS")
                        or (node_type == "abstract_memory" and edge_type == "SUPPORTS")
                        or (node_type == "entity" and edge_type in {"MENTIONS", "FACT_SUBJECT", "FACT_OBJECT"})
                        or (node_type == "topic" and edge_type == "BELONGS_TO_TOPIC")
                    ):
                        session_sets[node_id].update(session_sets[source])

                if len(session_sets[node_id]) != before:
                    changed = True

        for node_id, sessions in session_sets.items():
            node = self._nodes_by_id[node_id]
            session_list = sorted(sessions)
            self._sessions_by_node[node_id] = session_list
            for session_id in session_list:
                self._nodes_by_session[session_id].append(node)

    @property
    def record_id(self) -> str:
        """Get record ID from graph metadata."""
        return self._graph_data.get("graph", {}).get("record_id", "")

    @property
    def graph_meta(self) -> Dict[str, Any]:
        """Get graph metadata."""
        return deepcopy(self._graph_data.get("graph", {}))

    @property
    def node_count(self) -> int:
        """Get total node count."""
        return len(self._nodes_by_id)

    @property
    def edge_count(self) -> int:
        """Get total edge count."""
        return len(self._edges)

    def get_node(self, node_id: str) -> Dict[str, Any]:
        """
        Get node by ID.

        Raises:
            KeyError: If node not found
        """
        return deepcopy(self._nodes_by_id[node_id])

    def get_node_or_none(self, node_id: str) -> Dict[str, Any] | None:
        """Get node by ID, or None if not found."""
        node = self._nodes_by_id.get(node_id)
        return deepcopy(node) if node is not None else None

    def get_edge(
        self,
        source: str,
        target: str,
        key: int | str = 0
    ) -> Dict[str, Any]:
        """
        Get edge by source, target, and key.

        Raises:
            KeyError: If edge not found
        """
        return deepcopy(self._edge_by_tuple[(source, target, str(key))])

    def iter_nodes(
        self,
        node_type: str | None = None
    ) -> Iterable[Dict[str, Any]]:
        """Iterate over nodes, optionally filtered by type."""
        if node_type is None:
            for node in self._nodes_by_id.values():
                yield deepcopy(node)
        else:
            for node in self._nodes_by_type.get(node_type, []):
                yield deepcopy(node)

    def iter_edges(
        self,
        edge_type: str | None = None
    ) -> Iterable[Dict[str, Any]]:
        """Iterate over edges, optionally filtered by type."""
        if edge_type is None:
            for edge in self._edges:
                yield deepcopy(edge)
        else:
            for edge in self._edges_by_type.get(edge_type, []):
                yield deepcopy(edge)

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get all nodes of a specific type."""
        return deepcopy(list(self._nodes_by_type.get(node_type, [])))

    def get_edges_by_type(self, edge_type: str) -> List[Dict[str, Any]]:
        """Get all edges of a specific type."""
        return deepcopy(list(self._edges_by_type.get(edge_type, [])))

    def get_neighbors(
        self,
        node_id: str,
        *,
        direction: str = "both",
        edge_types: Set[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring nodes.

        Args:
            node_id: Source node ID
            direction: "out", "in", or "both"
            edge_types: Optional set of edge types to filter

        Returns:
            List of neighbor node dicts
        """
        neighbors = []
        neighbor_ids = set()

        if direction in {"out", "both"}:
            for edge in self._out_edges_by_node.get(node_id, []):
                if edge_types is None or edge.get("type") in edge_types:
                    target = edge.get("target")
                    if target and target not in neighbor_ids and target in self._nodes_by_id:
                        neighbor_ids.add(target)
                        neighbors.append(deepcopy(self._nodes_by_id[target]))

        if direction in {"in", "both"}:
            for edge in self._in_edges_by_node.get(node_id, []):
                if edge_types is None or edge.get("type") in edge_types:
                    source = edge.get("source")
                    if source and source not in neighbor_ids and source in self._nodes_by_id:
                        neighbor_ids.add(source)
                        neighbors.append(deepcopy(self._nodes_by_id[source]))

        return neighbors

    def get_session_ids_for_node(self, node_id: str) -> List[str]:
        """Get session IDs associated with a node."""
        return list(self._sessions_by_node.get(node_id, []))

    def get_session_ids_for_nodes(self, node_ids: List[str]) -> List[str]:
        """Get unique session IDs associated with multiple nodes."""
        sessions: Set[str] = set()
        for node_id in node_ids:
            sessions.update(self._sessions_by_node.get(node_id, []))
        return sorted(sessions)

    def get_nodes_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all nodes associated with a session."""
        return deepcopy(list(self._nodes_by_session.get(session_id, [])))

    def build_evidence_context(
        self,
        node_ids: List[str],
        *,
        include_edges: bool = True,
        max_chars: int | None = None,
    ) -> str:
        """
        Build readable evidence context string from node IDs.

        Args:
            node_ids: List of node IDs to include
            include_edges: If True, include edges between nodes
            max_chars: Optional character limit for output (truncates result)

        Returns:
            Formatted evidence context string
        """
        lines = []
        node_set = set(node_ids)

        # Format nodes
        for node_id in node_ids:
            node = self.get_node_or_none(node_id)
            if not node:
                continue

            lines.append(f"[Node {node_id}]")
            node_type = node.get("type", "unknown")
            lines.append(f"type: {node_type}")

            # Session info
            sessions = self.get_session_ids_for_node(node_id)
            if sessions:
                lines.append(f"sessions: {', '.join(sessions)}")

            # Type-specific fields
            if node_type == "event":
                for field in ["speaker", "timestamp", "text"]:
                    if field in node:
                        lines.append(f"{field}: {node[field]}")

            elif node_type == "fact":
                for field in ["text", "subject", "predicate", "object", "temporal_scope", "status", "confidence"]:
                    if field in node:
                        lines.append(f"{field}: {node[field]}")

            elif node_type == "abstract_memory":
                for field in ["summary", "scope", "supporting_fact_ids", "confidence"]:
                    if field in node:
                        lines.append(f"{field}: {node[field]}")

            elif node_type == "entity":
                for field in ["name", "entity_type", "aliases", "confidence"]:
                    if field in node:
                        lines.append(f"{field}: {node[field]}")

            elif node_type == "session":
                for field in ["session_id", "timestamp"]:
                    if field in node:
                        lines.append(f"{field}: {node[field]}")

            elif node_type == "topic":
                for field in ["name", "description"]:
                    if field in node:
                        lines.append(f"{field}: {node[field]}")

            lines.append("")

        # Format edges between these nodes
        if include_edges:
            edge_lines = []
            for node_id in node_ids:
                for edge in self._out_edges_by_node.get(node_id, []):
                    target = edge.get("target")
                    if target in node_set:
                        edge_type = edge.get("type", "UNKNOWN")
                        edge_lines.append(f"{node_id} --{edge_type}--> {target}")

            if edge_lines:
                lines.append("[Edges]")
                lines.extend(edge_lines)
                lines.append("")

        result = "\n".join(lines)

        # Truncate if needed
        if max_chars is not None and len(result) > max_chars:
            result = result[:max_chars]

        return result

    def validate_no_leakage(self) -> None:
        """
        Validate that no forbidden keys exist in the artifact.

        Raises:
            OracleGraphLeakageError: If forbidden keys found
        """
        self._validate_no_leakage_in_dict(
            self._graph_data,
            self.graph_path.name,
            ""
        )
        self._leakage_checked = True

    def _validate_no_leakage_in_dict(
        self,
        obj: Any,
        artifact_name: str,
        path: str
    ) -> None:
        """Recursively check for forbidden keys."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                json_path = f"{path}.{key}" if path else key

                if self._is_forbidden_key_at_path(key, json_path):
                    raise OracleGraphLeakageError(
                        f"Forbidden key '{key}' found at {json_path} in {artifact_name}"
                    )

                self._validate_no_leakage_in_dict(value, artifact_name, json_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                json_path = f"{path}[{i}]"
                self._validate_no_leakage_in_dict(item, artifact_name, json_path)

    def _is_forbidden_key_at_path(self, key: str, path: str) -> bool:
        """Return whether a key is forbidden at a JSON path.

        The graph format legitimately uses edge target fields, so `target` is
        allowed only for edge endpoints.
        """
        if key not in FORBIDDEN_ARTIFACT_KEYS:
            return False
        if key != "target":
            return True
        normalized = path.replace("[0]", "[]")
        # Generalize all list indexes without regex overhead.
        parts = []
        for part in normalized.split("."):
            if "[" in part:
                part = part.split("[", 1)[0] + "[]"
            parts.append(part)
        normalized = ".".join(parts)
        return not (
            normalized in ALLOWED_TARGET_PATHS
            or any(normalized.endswith(f".{allowed}") for allowed in ALLOWED_TARGET_PATHS)
        )

    def validate_integrity(self) -> None:
        """
        Validate graph structure integrity.

        Raises:
            OracleGraphFormatError: If structure is invalid
        """
        # Check root structure
        if not isinstance(self._graph_data, dict):
            raise OracleGraphFormatError("Graph root must be a dict")

        graph_meta = self._graph_data.get("graph")
        if not isinstance(graph_meta, dict):
            raise OracleGraphFormatError("'graph' field must be a dict")

        if not graph_meta.get("record_id"):
            raise OracleGraphFormatError("'graph.record_id' is required")

        nodes_list = self._graph_data.get("nodes")
        if not isinstance(nodes_list, list):
            raise OracleGraphFormatError("'nodes' field must be a list")

        edges_list = self._graph_data.get("edges")
        if not isinstance(edges_list, list):
            raise OracleGraphFormatError("'edges' field must be a list")

        # Check node structure
        node_ids = set()
        for i, node in enumerate(nodes_list):
            if not isinstance(node, dict):
                raise OracleGraphFormatError(f"Node at index {i} must be a dict")

            node_id = node.get("id")
            if not node_id:
                raise OracleGraphFormatError(f"Node at index {i} missing 'id'")

            if node_id in node_ids:
                raise OracleGraphFormatError(f"Duplicate node ID: {node_id}")
            node_ids.add(node_id)

            if not node.get("type"):
                raise OracleGraphFormatError(f"Node {node_id} missing 'type'")

        # Check edge structure
        for i, edge in enumerate(edges_list):
            if not isinstance(edge, dict):
                raise OracleGraphFormatError(f"Edge at index {i} must be a dict")

            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("type")

            if not source:
                raise OracleGraphFormatError(f"Edge at index {i} missing 'source'")
            if not target:
                raise OracleGraphFormatError(f"Edge at index {i} missing 'target'")
            if not edge_type:
                raise OracleGraphFormatError(f"Edge at index {i} missing 'type'")

            if source not in node_ids:
                raise OracleGraphFormatError(
                    f"Edge at index {i} references unknown source: {source}"
                )
            if target not in node_ids:
                raise OracleGraphFormatError(
                    f"Edge at index {i} references unknown target: {target}"
                )

        self._integrity_checked = True

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        node_type_counts = {
            node_type: len(nodes)
            for node_type, nodes in self._nodes_by_type.items()
        }

        edge_type_counts = {
            edge_type: len(edges)
            for edge_type, edges in self._edges_by_type.items()
        }

        session_count = len(self._nodes_by_type.get("session", []))

        return {
            "record_id": self.record_id,
            "graph_path": str(self.graph_path),
            "evidence_pack_path": str(self.evidence_pack_path) if self.evidence_pack_path else None,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "node_type_counts": node_type_counts,
            "edge_type_counts": edge_type_counts,
            "session_count": session_count,
            "leakage_checked": self._leakage_checked,
            "integrity_checked": self._integrity_checked,
            "sanitized_keys": self._sanitized_keys,
        }
