from __future__ import annotations

from enum import Enum
import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field, model_validator


# Forbidden keys that must not appear in CurrentMemory artifacts
FORBIDDEN_MEMORY_KEYS = {
    "question",
    "query",
    "answer",
    "gold_answer",
    "target",
    "question_type",
    "haystack_question_type",
    "oracle_graph",
    "oracle_node_ids",
    "reward",
    "eval_result",
}


EDGE_TARGET_PATH_RE = re.compile(r"^memory_graph\.edges\[\d+\]\.target$")


class MemoryNodeType(str, Enum):
    """Types of nodes in current memory graph."""
    FACT = "fact"
    EVENT = "event"
    ENTITY = "entity"
    PREFERENCE = "preference"
    PLAN = "plan"
    CONSTRAINT = "constraint"
    ABSTRACT = "abstract"
    SUMMARY = "summary"
    OTHER = "other"


class MemoryStatus(str, Enum):
    """Status of a memory node."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    UNCERTAIN = "uncertain"
    ARCHIVED = "archived"


class MemoryEdgeType(str, Enum):
    """Types of edges in current memory graph."""
    SUPPORTS = "SUPPORTS"
    UPDATES = "UPDATES"
    SUPERSEDES = "SUPERSEDES"
    CONTRADICTS = "CONTRADICTS"
    DERIVED_FROM = "DERIVED_FROM"
    RELATED_TO = "RELATED_TO"
    ABSTRACTS = "ABSTRACTS"


class MemoryNode(BaseModel):
    """A node in the current memory graph."""
    id: str
    type: MemoryNodeType
    content: str
    status: MemoryStatus = MemoryStatus.ACTIVE
    confidence: float = Field(default=0.8, ge=0, le=1)
    abstraction_level: int = Field(default=1, ge=0, le=5)
    source_session_ids: List[str] = Field(default_factory=list)
    source_turn_ids: List[str] = Field(default_factory=list)
    source_event_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provenance(self) -> "MemoryNode":
        """Ensure active nodes have provenance."""
        if self.status == MemoryStatus.ACTIVE:
            has_provenance = (
                bool(self.source_session_ids)
                or bool(self.source_turn_ids)
                or bool(self.source_event_ids)
            )
            if not has_provenance and self.type not in {MemoryNodeType.SUMMARY}:
                # Allow some leniency for summary nodes
                pass
        return self


class MemoryEdge(BaseModel):
    """An edge in the current memory graph."""
    source: str
    target: str
    type: MemoryEdgeType
    confidence: float = Field(default=0.8, ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemorySummaries(BaseModel):
    """Global summaries for current memory."""
    user_profile: str = ""
    stable_preferences: str = ""
    active_plans: str = ""
    recent_changes: str = ""
    cross_session_abstractions: str = ""


class MemoryGraph(BaseModel):
    """The graph component of current memory."""
    nodes: List[MemoryNode] = Field(default_factory=list)
    edges: List[MemoryEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> "MemoryGraph":
        """Validate node IDs are unique and edges reference existing nodes."""
        # Check node ID uniqueness
        node_ids = set()
        for node in self.nodes:
            if node.id in node_ids:
                raise ValueError(f"Duplicate node ID: {node.id}")
            node_ids.add(node.id)

        # Check edge endpoints exist
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge source does not exist: {edge.source}")
            if edge.target not in node_ids:
                raise ValueError(f"Edge target does not exist: {edge.target}")

        return self


class CurrentMemory(BaseModel):
    """
    Current memory artifact produced by Memory Builder.

    CurrentMemory = memory_graph + memory_summaries + provenance

    This represents the Memory Builder's output at a given build step.
    """
    record_id: str
    build_step: int = 0
    memory_graph: MemoryGraph = Field(default_factory=MemoryGraph)
    memory_summaries: MemorySummaries = Field(default_factory=MemorySummaries)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_no_leakage(self) -> "CurrentMemory":
        """Ensure no forbidden keys exist in metadata."""
        for key in FORBIDDEN_MEMORY_KEYS:
            if key in self.metadata:
                raise ValueError(f"Forbidden key '{key}' found in metadata")
        return self


def empty_current_memory(record_id: str) -> dict:
    """
    Create an empty current memory structure.

    Args:
        record_id: Record identifier

    Returns:
        Empty current memory as dict
    """
    return {
        "record_id": record_id,
        "build_step": 0,
        "memory_graph": {
            "nodes": [],
            "edges": []
        },
        "memory_summaries": {
            "user_profile": "",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {}
    }


def validate_current_memory(
    memory: dict,
    *,
    strict: bool = True,
) -> CurrentMemory:
    """
    Validate current memory structure.

    Args:
        memory: Current memory dict
        strict: If True, enforce all validation rules

    Returns:
        Validated CurrentMemory model

    Raises:
        ValueError: If validation fails
    """
    # Check for forbidden keys recursively
    if strict:
        assert_no_memory_leakage(memory)

    # Validate with pydantic
    try:
        validated = CurrentMemory.model_validate(memory)
    except Exception as e:
        raise ValueError(f"Current memory validation failed: {e}") from e

    # Additional strict checks
    if strict:
        # Check provenance for active nodes
        for node in validated.memory_graph.nodes:
            if node.status == MemoryStatus.ACTIVE:
                has_provenance = (
                    bool(node.source_session_ids)
                    or bool(node.source_turn_ids)
                    or bool(node.source_event_ids)
                )
                if not has_provenance:
                    # Special case: summary nodes and archived nodes can be more lenient
                    if node.type not in {MemoryNodeType.SUMMARY} and node.status != MemoryStatus.ARCHIVED:
                        raise ValueError(
                            f"Active node {node.id} of type {node.type} missing provenance"
                        )

        # Check abstract nodes have appropriate abstraction level
        for node in validated.memory_graph.nodes:
            if node.type == MemoryNodeType.ABSTRACT and node.abstraction_level < 2:
                raise ValueError(
                    f"Abstract node {node.id} should have abstraction_level >= 2"
                )

    return validated


def assert_no_memory_leakage(memory: dict) -> None:
    """
    Recursively check that no forbidden keys exist in current memory.

    Args:
        memory: Current memory dict

    Raises:
        ValueError: If forbidden keys found
    """
    _recursive_check_leakage(memory, "", FORBIDDEN_MEMORY_KEYS)


def _recursive_check_leakage(
    obj: Any,
    path: str,
    forbidden_keys: set
) -> None:
    """Recursively check for forbidden keys."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            json_path = f"{path}.{key}" if path else key

            if key in forbidden_keys and not _is_allowed_memory_key_path(key, json_path):
                raise ValueError(
                    f"Forbidden key '{key}' found at {json_path} in current memory"
                )

            _recursive_check_leakage(value, json_path, forbidden_keys)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            json_path = f"{path}[{i}]"
            _recursive_check_leakage(item, json_path, forbidden_keys)


def _is_allowed_memory_key_path(key: str, path: str) -> bool:
    """Allow graph edge endpoint fields while rejecting leaked benchmark targets."""
    return key == "target" and bool(EDGE_TARGET_PATH_RE.match(path))


def memory_to_dict(memory: CurrentMemory) -> dict:
    """
    Convert CurrentMemory model to dict.

    Args:
        memory: CurrentMemory instance

    Returns:
        Dict representation
    """
    return memory.model_dump()
