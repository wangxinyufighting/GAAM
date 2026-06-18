"""
Tests for OracleGraphLoader.

Run with: pytest gaam_memory_training/tests/test_oracle_graph_loader.py
"""

import json
import tempfile
from pathlib import Path

import pytest

from gaam_graph.oracle_graph_loader import (
    FORBIDDEN_ARTIFACT_KEYS,
    OracleGraphFormatError,
    OracleGraphLeakageError,
    OracleGraphLoader,
)


@pytest.fixture
def safe_graph_data():
    """Safe graph artifact without leakage."""
    return {
        "graph": {
            "record_id": "test_record",
            "root_topic_id": "topic_root"
        },
        "nodes": [
            {
                "id": "sess_001",
                "type": "session",
                "session_id": "sess_001",
                "timestamp": "2024-01-01T10:00:00"
            },
            {
                "id": "evt_001",
                "type": "event",
                "session_id": "sess_001",
                "turn_id": 0,
                "speaker": "user",
                "text": "I graduated with a degree in Business Administration.",
                "timestamp": "2024-01-01T10:00:00"
            },
            {
                "id": "fact_001",
                "type": "fact",
                "subject": "user",
                "predicate": "has_attribute",
                "object": "Business Administration",
                "text": "The user graduated with a degree in Business Administration.",
                "fact_type": "education",
                "temporal_scope": "historical",
                "status": "active",
                "confidence": 0.9,
                "source_event_id": "evt_001",
                "timestamp": "2024-01-01T10:00:00"
            },
            {
                "id": "ent_user",
                "type": "entity",
                "name": "user",
                "entity_type": "person",
                "aliases": ["user"],
                "confidence": 0.9
            },
            {
                "id": "abs_001",
                "type": "abstract_memory",
                "summary": "The user has an education background in Business Administration.",
                "scope": "user_profile",
                "confidence": 0.85,
                "status": "active"
            }
        ],
        "edges": [
            {"source": "evt_001", "target": "sess_001", "key": 0, "type": "BELONGS_TO"},
            {"source": "evt_001", "target": "fact_001", "key": 0, "type": "EXTRACTED_AS", "confidence": 0.9},
            {"source": "fact_001", "target": "ent_user", "key": 0, "type": "FACT_SUBJECT"},
            {"source": "fact_001", "target": "abs_001", "key": 0, "type": "SUPPORTS", "confidence": 0.85}
        ]
    }


@pytest.fixture
def leaky_graph_data():
    """Graph with forbidden keys (leakage)."""
    return {
        "graph": {
            "record_id": "test_record",
            "question": "What degree does the user have?",
            "gold_answer": "Business Administration",
            "question_type": "education"
        },
        "nodes": [
            {
                "id": "evt_001",
                "type": "event",
                "session_id": "sess_001",
                "turn_id": 0,
                "speaker": "user",
                "text": "I graduated with a degree in Business Administration."
            }
        ],
        "edges": []
    }


@pytest.fixture
def malformed_graph_data():
    """Graph with integrity issues."""
    return {
        "graph": {
            "record_id": "test_record"
        },
        "nodes": [
            {
                "id": "evt_001",
                "type": "event",
                "text": "Some event"
            },
            {
                "id": "evt_002",
                "type": "event",
                "text": "Another event"
            }
        ],
        "edges": [
            # Edge pointing to non-existent node
            {"source": "evt_001", "target": "evt_999", "type": "NEXT"}
        ]
    }


def test_load_safe_graph(safe_graph_data, tmp_path):
    """Test loading a safe graph artifact."""
    graph_path = tmp_path / "safe.graph.json"
    graph_path.write_text(json.dumps(safe_graph_data))

    loader = OracleGraphLoader(graph_path, strict=True)

    assert loader.record_id == "test_record"
    assert loader.node_count == 5
    assert loader.edge_count == 4

    # Validate passes
    loader.validate_no_leakage()
    loader.validate_integrity()

    summary = loader.summary()
    assert summary["record_id"] == "test_record"
    assert summary["leakage_checked"]
    assert summary["integrity_checked"]
    assert summary["node_count"] == 5
    assert summary["edge_count"] == 4


def test_reject_leaky_graph(leaky_graph_data, tmp_path):
    """Test that strict mode rejects graphs with forbidden keys."""
    graph_path = tmp_path / "leaky.graph.json"
    graph_path.write_text(json.dumps(leaky_graph_data))

    with pytest.raises(OracleGraphLeakageError) as exc_info:
        OracleGraphLoader(graph_path, strict=True)

    assert "question" in str(exc_info.value)


def test_sanitize_mode(leaky_graph_data, tmp_path):
    """Test sanitize mode removes forbidden keys."""
    graph_path = tmp_path / "leaky.graph.json"
    graph_path.write_text(json.dumps(leaky_graph_data))

    loader = OracleGraphLoader(graph_path, strict=False, sanitize=True)

    # Should load successfully
    assert loader.record_id == "test_record"

    # Check sanitized keys reported
    summary = loader.summary()
    assert len(summary["sanitized_keys"]) > 0
    assert any("question" in key for key in summary["sanitized_keys"])

    # Graph meta should not contain forbidden keys
    for key in FORBIDDEN_ARTIFACT_KEYS:
        assert key not in loader.graph_meta


def test_reject_malformed_graph(malformed_graph_data, tmp_path):
    """Test that strict mode rejects malformed graphs."""
    graph_path = tmp_path / "malformed.graph.json"
    graph_path.write_text(json.dumps(malformed_graph_data))

    with pytest.raises(OracleGraphFormatError) as exc_info:
        OracleGraphLoader(graph_path, strict=True)

    assert "evt_999" in str(exc_info.value)


def test_session_inference(safe_graph_data, tmp_path):
    """Test session inference for different node types."""
    graph_path = tmp_path / "safe.graph.json"
    graph_path.write_text(json.dumps(safe_graph_data))

    loader = OracleGraphLoader(graph_path, strict=True)

    # Event node should map to its session_id
    evt_sessions = loader.get_session_ids_for_node("evt_001")
    assert "sess_001" in evt_sessions

    # Fact node should inherit from source event
    fact_sessions = loader.get_session_ids_for_node("fact_001")
    assert "sess_001" in fact_sessions

    # Abstract memory should aggregate from supporting facts
    abs_sessions = loader.get_session_ids_for_node("abs_001")
    assert "sess_001" in abs_sessions


def test_evidence_context(safe_graph_data, tmp_path):
    """Test evidence context building."""
    graph_path = tmp_path / "safe.graph.json"
    graph_path.write_text(json.dumps(safe_graph_data))

    loader = OracleGraphLoader(graph_path, strict=True)

    # Build context for event and fact
    context = loader.build_evidence_context(
        ["evt_001", "fact_001"],
        include_edges=True
    )

    # Check that context contains key information
    assert "evt_001" in context
    assert "fact_001" in context
    assert "type: event" in context
    assert "type: fact" in context
    assert "speaker: user" in context
    assert "Business Administration" in context
    assert "EXTRACTED_AS" in context

    # Check no forbidden keys in context
    for key in FORBIDDEN_ARTIFACT_KEYS:
        assert key not in context


def test_node_access(safe_graph_data, tmp_path):
    """Test node access methods."""
    graph_path = tmp_path / "safe.graph.json"
    graph_path.write_text(json.dumps(safe_graph_data))

    loader = OracleGraphLoader(graph_path, strict=True)

    # Get node by ID
    node = loader.get_node("evt_001")
    assert node["type"] == "event"

    # Get node or none
    assert loader.get_node_or_none("evt_001") is not None
    assert loader.get_node_or_none("nonexistent") is None

    # Get nodes by type
    events = loader.get_nodes_by_type("event")
    assert len(events) == 1
    assert events[0]["id"] == "evt_001"

    facts = loader.get_nodes_by_type("fact")
    assert len(facts) == 1


def test_public_access_returns_copies(safe_graph_data, tmp_path):
    """Test public accessors do not expose mutable internal state."""
    graph_path = tmp_path / "safe.graph.json"
    graph_path.write_text(json.dumps(safe_graph_data))

    loader = OracleGraphLoader(graph_path, strict=True)

    node = loader.get_node("evt_001")
    node["type"] = "mutated"
    assert loader.get_node("evt_001")["type"] == "event"

    edge = loader.get_edge("evt_001", "sess_001", 0)
    edge["type"] = "mutated"
    assert loader.get_edge("evt_001", "sess_001", 0)["type"] == "BELONGS_TO"

    nodes = loader.get_nodes_by_type("event")
    nodes[0]["type"] = "mutated"
    assert loader.get_node("evt_001")["type"] == "event"

    meta = loader.graph_meta
    meta["record_id"] = "mutated"
    assert loader.record_id == "test_record"


def test_edge_access(safe_graph_data, tmp_path):
    """Test edge access methods."""
    graph_path = tmp_path / "safe.graph.json"
    graph_path.write_text(json.dumps(safe_graph_data))

    loader = OracleGraphLoader(graph_path, strict=True)

    # Get edge by endpoints
    edge = loader.get_edge("evt_001", "sess_001", 0)
    assert edge["type"] == "BELONGS_TO"

    # Get edges by type
    extracted_edges = loader.get_edges_by_type("EXTRACTED_AS")
    assert len(extracted_edges) == 1


def test_neighbors(safe_graph_data, tmp_path):
    """Test neighbor queries."""
    graph_path = tmp_path / "safe.graph.json"
    graph_path.write_text(json.dumps(safe_graph_data))

    loader = OracleGraphLoader(graph_path, strict=True)

    # Out neighbors
    out_neighbors = loader.get_neighbors("evt_001", direction="out")
    out_ids = {n["id"] for n in out_neighbors}
    assert "sess_001" in out_ids
    assert "fact_001" in out_ids

    # In neighbors
    in_neighbors = loader.get_neighbors("sess_001", direction="in")
    in_ids = {n["id"] for n in in_neighbors}
    assert "evt_001" in in_ids

    # Filter by edge type
    extracted_neighbors = loader.get_neighbors(
        "evt_001",
        direction="out",
        edge_types={"EXTRACTED_AS"}
    )
    assert len(extracted_neighbors) == 1
    assert extracted_neighbors[0]["id"] == "fact_001"


def test_strict_and_sanitize_mutual_exclusion():
    """Test that strict and sanitize cannot both be True."""
    with pytest.raises(ValueError) as exc_info:
        # Create a dummy temp file just for this test
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json.dumps({"graph": {"record_id": "test"}, "nodes": [], "edges": []}))
            temp_path = f.name

        try:
            OracleGraphLoader(temp_path, strict=True, sanitize=True)
        finally:
            Path(temp_path).unlink()

    assert "strict and sanitize cannot both be true" in str(exc_info.value)


def test_missing_node_id(tmp_path):
    """Test validation catches missing node IDs."""
    bad_data = {
        "graph": {"record_id": "test"},
        "nodes": [
            {"type": "event", "text": "no id field"}
        ],
        "edges": []
    }

    graph_path = tmp_path / "bad.graph.json"
    graph_path.write_text(json.dumps(bad_data))

    with pytest.raises(OracleGraphFormatError) as exc_info:
        OracleGraphLoader(graph_path, strict=True)

    assert "missing 'id'" in str(exc_info.value)


def test_duplicate_node_id(tmp_path):
    """Test validation catches duplicate node IDs."""
    bad_data = {
        "graph": {"record_id": "test"},
        "nodes": [
            {"id": "evt_001", "type": "event"},
            {"id": "evt_001", "type": "event"}
        ],
        "edges": []
    }

    graph_path = tmp_path / "bad.graph.json"
    graph_path.write_text(json.dumps(bad_data))

    with pytest.raises(OracleGraphFormatError) as exc_info:
        OracleGraphLoader(graph_path, strict=True)

    assert "Duplicate node ID" in str(exc_info.value)


def test_evidence_pack_loading(safe_graph_data, tmp_path):
    """Test loading with evidence pack."""
    graph_path = tmp_path / "safe.graph.json"
    graph_path.write_text(json.dumps(safe_graph_data))

    evidence_data = {
        "record_id": "test_record",
        "nodes": [
            {"id": "fact_001", "type": "fact", "text": "Some fact"}
        ],
        "edges": []
    }
    evidence_path = tmp_path / "safe.evidence_pack.json"
    evidence_path.write_text(json.dumps(evidence_data))

    loader = OracleGraphLoader(
        graph_path,
        evidence_pack_path=evidence_path,
        strict=True
    )

    summary = loader.summary()
    assert summary["evidence_pack_path"] == str(evidence_path)


def test_leaky_evidence_pack(safe_graph_data, tmp_path):
    """Test that leaky evidence pack is rejected in strict mode."""
    graph_path = tmp_path / "safe.graph.json"
    graph_path.write_text(json.dumps(safe_graph_data))

    leaky_evidence = {
        "record_id": "test_record",
        "question": "leaked question",
        "nodes": [],
        "edges": []
    }
    evidence_path = tmp_path / "leaky.evidence_pack.json"
    evidence_path.write_text(json.dumps(leaky_evidence))

    with pytest.raises(OracleGraphLeakageError):
        OracleGraphLoader(
            graph_path,
            evidence_pack_path=evidence_path,
            strict=True
        )


def test_target_allowed_only_for_edges(tmp_path):
    """Test target leakage is rejected except for edge endpoint fields."""
    graph_with_edge_target = {
        "graph": {"record_id": "test"},
        "nodes": [
            {"id": "a", "type": "event"},
            {"id": "b", "type": "event"},
        ],
        "edges": [
            {"source": "a", "target": "b", "type": "NEXT"},
        ],
    }
    graph_path = tmp_path / "edge_target.graph.json"
    graph_path.write_text(json.dumps(graph_with_edge_target))
    OracleGraphLoader(graph_path, strict=True)

    graph_with_top_target = {
        "target": "leaked target answer",
        "graph": {"record_id": "test"},
        "nodes": [],
        "edges": [],
    }
    graph_path = tmp_path / "top_target.graph.json"
    graph_path.write_text(json.dumps(graph_with_top_target))
    with pytest.raises(OracleGraphLeakageError) as exc_info:
        OracleGraphLoader(graph_path, strict=True)
    assert "target" in str(exc_info.value)

    graph_with_node_target = {
        "graph": {"record_id": "test"},
        "nodes": [
            {"id": "a", "type": "event", "target": "leaked target answer"},
        ],
        "edges": [],
    }
    graph_path = tmp_path / "node_target.graph.json"
    graph_path.write_text(json.dumps(graph_with_node_target))
    with pytest.raises(OracleGraphLeakageError) as exc_info:
        OracleGraphLoader(graph_path, strict=True)
    assert "nodes[0].target" in str(exc_info.value)


def test_session_inference_independent_of_node_order(tmp_path):
    """Test session inference does not depend on node order in JSON."""
    graph = {
        "graph": {"record_id": "test"},
        "nodes": [
            {
                "id": "abs_001",
                "type": "abstract_memory",
                "summary": "The user has an education background.",
            },
            {
                "id": "fact_001",
                "type": "fact",
                "text": "The user graduated with a degree.",
                "source_event_id": "evt_001",
            },
            {
                "id": "evt_001",
                "type": "event",
                "session_id": "sess_001",
                "turn_id": 0,
                "speaker": "user",
                "text": "I graduated with a degree.",
            },
            {
                "id": "sess_001",
                "type": "session",
                "session_id": "sess_001",
            },
        ],
        "edges": [
            {"source": "evt_001", "target": "fact_001", "key": 0, "type": "EXTRACTED_AS"},
            {"source": "fact_001", "target": "abs_001", "key": 0, "type": "SUPPORTS"},
        ],
    }
    graph_path = tmp_path / "reordered.graph.json"
    graph_path.write_text(json.dumps(graph))

    loader = OracleGraphLoader(graph_path, strict=True)

    assert loader.get_session_ids_for_node("evt_001") == ["sess_001"]
    assert loader.get_session_ids_for_node("fact_001") == ["sess_001"]
    assert loader.get_session_ids_for_node("abs_001") == ["sess_001"]
