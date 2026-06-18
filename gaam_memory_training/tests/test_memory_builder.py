"""
Tests for BaselineMemoryBuilder.

Run with: pytest gaam_memory_training/tests/test_memory_builder.py
"""

import json

import pytest

from gaam_graph.lme_loader import LMEEvent, LMERecord
from gaam_graph.memory_builder import BaselineMemoryBuilder, build_current_memory_from_record
from gaam_graph.memory_schema import MemoryNodeType, validate_current_memory
from gaam_graph.raw_history import raw_history_from_lme_record
from gaam_graph.utils import normalize_text


@pytest.fixture
def sample_lme_record():
    """Create a sample LMERecord with user turns."""
    events = [
        LMEEvent(
            event_id="evt_001",
            session_id="sess_01",
            turn_id=0,
            speaker="user",
            text="I graduated with a degree in Business Administration.",
            timestamp="2024-01-01T10:00:00",
        ),
        LMEEvent(
            event_id="evt_002",
            session_id="sess_01",
            turn_id=1,
            speaker="user",
            text="I prefer Python for data analysis projects.",
            timestamp="2024-01-01T10:01:00",
        ),
        LMEEvent(
            event_id="evt_003",
            session_id="sess_02",
            turn_id=0,
            speaker="user",
            text="I plan to start a machine learning course next month.",
            timestamp="2024-01-02T10:00:00",
        ),
    ]

    return LMERecord(
        record_id="test_record",
        events=events,
        question="What degree does the user have?",
        answer="Business Administration",
        question_type="education",
        raw={},
    )


def test_baseline_builder_produces_valid_memory(sample_lme_record):
    """Test that baseline builder produces valid CurrentMemory."""
    builder = BaselineMemoryBuilder()
    history = raw_history_from_lme_record(sample_lme_record)

    memory = builder.build_case_memory(history)

    # Should validate successfully
    validated = validate_current_memory(memory, strict=True)

    assert validated.record_id == "test_record"
    assert len(validated.memory_graph.nodes) > 0
    assert validated.memory_summaries is not None


def test_builder_creates_nodes_and_edges(sample_lme_record):
    """Test that builder creates expected node types."""
    builder = BaselineMemoryBuilder()
    memory = build_current_memory_from_record(sample_lme_record, builder=builder)

    nodes = memory["memory_graph"]["nodes"]
    edges = memory["memory_graph"]["edges"]

    # Should have event and fact nodes
    event_nodes = [n for n in nodes if n["type"] == MemoryNodeType.EVENT.value]
    fact_nodes = [n for n in nodes if n["type"] == MemoryNodeType.FACT.value]

    assert len(event_nodes) > 0
    assert len(fact_nodes) > 0

    # Should have edges
    assert len(edges) > 0


def test_builder_has_provenance(sample_lme_record):
    """Test that all active nodes have provenance."""
    builder = BaselineMemoryBuilder()
    memory = build_current_memory_from_record(sample_lme_record, builder=builder)

    nodes = memory["memory_graph"]["nodes"]

    for node in nodes:
        if node["status"] == "active":
            has_provenance = (
                bool(node.get("source_session_ids"))
                or bool(node.get("source_turn_ids"))
                or bool(node.get("source_event_ids"))
            )
            # Allow leniency for summary nodes
            if node["type"] != MemoryNodeType.SUMMARY.value:
                assert has_provenance, f"Node {node['id']} missing provenance"


def test_builder_does_not_use_questions():
    """
    Test that two records with identical events but different questions
    produce identical memory.
    """
    events = [
        LMEEvent(
            event_id="evt_001",
            session_id="sess_01",
            turn_id=0,
            speaker="user",
            text="I like machine learning.",
            timestamp="2024-01-01T10:00:00",
        ),
    ]

    record_a = LMERecord(
        record_id="test",
        events=events,
        question="LEAK_QUESTION_A what should never be stored?",
        answer="LEAK_ANSWER_A",
        question_type="LEAK_TYPE_A",
        raw={},
    )

    record_b = LMERecord(
        record_id="test",
        events=events,
        question="LEAK_QUESTION_B different question entirely",
        answer="LEAK_ANSWER_B",
        question_type="LEAK_TYPE_B",
        raw={},
    )

    builder = BaselineMemoryBuilder()

    memory_a = build_current_memory_from_record(record_a, builder=builder)
    memory_b = build_current_memory_from_record(record_b, builder=builder)

    assert memory_a == memory_b

    serialized = json.dumps(memory_a, ensure_ascii=False)
    for leaked_text in [
        "LEAK_QUESTION_A",
        "LEAK_ANSWER_A",
        "LEAK_TYPE_A",
        "LEAK_QUESTION_B",
        "LEAK_ANSWER_B",
        "LEAK_TYPE_B",
    ]:
        assert leaked_text not in serialized


def test_builder_does_not_mutate_previous_memory():
    """Test that update_memory does not mutate the input."""
    from gaam_graph.memory_schema import empty_current_memory
    from gaam_graph.raw_history import RawTurn

    builder = BaselineMemoryBuilder()

    previous = empty_current_memory("test")
    original_node_count = len(previous["memory_graph"]["nodes"])

    turn = RawTurn(
        event_id="evt_001",
        session_id="sess_01",
        turn_id=0,
        speaker="user",
        text="I like Python programming and data science.",
        timestamp="2024-01-01T10:00:00",
    )

    updated = builder.update_memory(previous, turn)

    # Previous should remain unchanged
    assert len(previous["memory_graph"]["nodes"]) == original_node_count

    # Updated should have new nodes
    assert len(updated["memory_graph"]["nodes"]) > original_node_count


def test_abstract_node_support(sample_lme_record):
    """Test that abstract nodes are created for multi-turn sessions."""
    builder = BaselineMemoryBuilder()
    memory = build_current_memory_from_record(sample_lme_record, builder=builder)

    nodes = memory["memory_graph"]["nodes"]
    edges = memory["memory_graph"]["edges"]

    # Should have at least one abstract node
    abstract_nodes = [n for n in nodes if n["type"] == MemoryNodeType.ABSTRACT.value]
    assert len(abstract_nodes) > 0

    # Abstract nodes should have abstraction_level >= 2
    for node in abstract_nodes:
        assert node["abstraction_level"] >= 2

    # Abstract nodes should have provenance
    for node in abstract_nodes:
        assert len(node["source_session_ids"]) > 0

    # Should have ABSTRACTS edges
    abstracts_edges = [e for e in edges if e["type"] == "ABSTRACTS"]
    assert len(abstracts_edges) > 0


def test_min_text_chars_filter():
    """Test that short user turns are filtered out."""
    events = [
        LMEEvent(
            event_id="evt_001",
            session_id="sess_01",
            turn_id=0,
            speaker="user",
            text="Hi",  # Too short
            timestamp="2024-01-01T10:00:00",
        ),
        LMEEvent(
            event_id="evt_002",
            session_id="sess_01",
            turn_id=1,
            speaker="user",
            text="I have a background in software engineering with 5 years of experience.",
            timestamp="2024-01-01T10:01:00",
        ),
    ]

    record = LMERecord(
        record_id="test",
        events=events,
        question=None,
        answer=None,
        question_type=None,
        raw={},
    )

    builder = BaselineMemoryBuilder(min_user_text_chars=20)
    memory = build_current_memory_from_record(record, builder=builder)

    event_nodes = [
        n for n in memory["memory_graph"]["nodes"]
        if n["type"] == MemoryNodeType.EVENT.value
    ]

    # Should only have 1 event node (the long one)
    assert len(event_nodes) == 1
    assert "software engineering" in event_nodes[0]["content"]


def test_fact_node_stores_full_text_without_truncation():
    """Test that fact nodes preserve full normalized turn text."""
    long_text = (
        "This is a detailed biographical note about the user's work, preferences, "
        "constraints, and future plans. "
        + "important detail " * 40
        + "final detail should remain visible."
    )
    events = [
        LMEEvent(
            event_id="evt_001",
            session_id="sess_01",
            turn_id=0,
            speaker="user",
            text=long_text,
            timestamp="2024-01-01T10:00:00",
        )
    ]
    record = LMERecord(
        record_id="test",
        events=events,
        question=None,
        answer=None,
        question_type=None,
        raw={},
    )

    memory = build_current_memory_from_record(record, builder=BaselineMemoryBuilder())
    fact_nodes = [
        n for n in memory["memory_graph"]["nodes"]
        if n["type"] == MemoryNodeType.FACT.value
    ]

    assert len(fact_nodes) == 1
    assert fact_nodes[0]["content"] == f"The user mentioned: {normalize_text(long_text)}"
    assert "final detail should remain visible." in fact_nodes[0]["content"]


def test_global_summaries(sample_lme_record):
    """Test that global summaries are populated."""
    builder = BaselineMemoryBuilder()
    memory = build_current_memory_from_record(sample_lme_record, builder=builder)

    summaries = memory["memory_summaries"]

    # recent_changes should be populated
    assert summaries["recent_changes"] != ""
    assert "sessions" in summaries["recent_changes"]


def test_preference_detection(sample_lme_record):
    """Test that preference markers are detected."""
    builder = BaselineMemoryBuilder()
    memory = build_current_memory_from_record(sample_lme_record, builder=builder)

    summaries = memory["memory_summaries"]

    # Should detect "prefer" in the text
    assert summaries["stable_preferences"] != ""


def test_plan_detection(sample_lme_record):
    """Test that plan markers are detected."""
    builder = BaselineMemoryBuilder()
    memory = build_current_memory_from_record(sample_lme_record, builder=builder)

    summaries = memory["memory_summaries"]

    # Should detect "plan" in the text
    assert summaries["active_plans"] != ""


def test_empty_record():
    """Test handling of empty record."""
    record = LMERecord(
        record_id="empty",
        events=[],
        question=None,
        answer=None,
        question_type=None,
        raw={},
    )

    builder = BaselineMemoryBuilder()
    memory = build_current_memory_from_record(record, builder=builder)

    # Should be valid but empty
    validated = validate_current_memory(memory, strict=True)
    assert validated.record_id == "empty"
    assert len(validated.memory_graph.nodes) == 0


def test_max_event_nodes_cap():
    """Test that max_event_nodes cap is respected."""
    events = [
        LMEEvent(
            event_id=f"evt_{i:03d}",
            session_id="sess_01",
            turn_id=i,
            speaker="user",
            text=f"This is user message number {i} with sufficient length.",
            timestamp="2024-01-01T10:00:00",
        )
        for i in range(10)
    ]

    record = LMERecord(
        record_id="test",
        events=events,
        question=None,
        answer=None,
        question_type=None,
        raw={},
    )

    builder = BaselineMemoryBuilder(max_event_nodes=5)
    memory = build_current_memory_from_record(record, builder=builder)

    event_nodes = [
        n for n in memory["memory_graph"]["nodes"]
        if n["type"] == MemoryNodeType.EVENT.value
    ]

    # Should be capped at 5
    assert len(event_nodes) == 5


def test_session_summary_and_abstract_created():
    """Test that session with 2+ retained turns gets summary and abstract."""
    events = [
        LMEEvent(
            event_id="evt_001",
            session_id="sess_01",
            turn_id=0,
            speaker="user",
            text="I work as a software engineer at a tech company.",
            timestamp="2024-01-01T10:00:00",
        ),
        LMEEvent(
            event_id="evt_002",
            session_id="sess_01",
            turn_id=1,
            speaker="user",
            text="I specialize in backend development and database design.",
            timestamp="2024-01-01T10:01:00",
        ),
    ]

    record = LMERecord(
        record_id="test",
        events=events,
        question=None,
        answer=None,
        question_type=None,
        raw={},
    )

    builder = BaselineMemoryBuilder()
    memory = build_current_memory_from_record(record, builder=builder)

    nodes = memory["memory_graph"]["nodes"]

    # Should have summary node
    summary_nodes = [n for n in nodes if n["type"] == MemoryNodeType.SUMMARY.value]
    assert len(summary_nodes) > 0

    # Should have abstract node
    abstract_nodes = [n for n in nodes if n["type"] == MemoryNodeType.ABSTRACT.value]
    assert len(abstract_nodes) > 0
