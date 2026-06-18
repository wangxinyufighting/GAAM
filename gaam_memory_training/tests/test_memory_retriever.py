"""
Tests for CurrentMemoryRetriever.

Run with: pytest gaam_memory_training/tests/test_memory_retriever.py
"""

import pytest

from gaam_graph.memory_retriever import CurrentMemoryRetriever
from gaam_graph.memory_schema import empty_current_memory


@pytest.fixture
def sample_memory():
    """Create sample current memory."""
    memory = empty_current_memory("test_record")

    # Add some memory nodes
    memory["memory_graph"]["nodes"] = [
        {
            "id": "mem_fact_001",
            "type": "fact",
            "content": "The user likes Python programming for data analysis.",
            "status": "active",
            "confidence": 0.9,
            "source_session_ids": ["sess_01"],
            "source_turn_ids": ["1"],
            "source_event_ids": ["evt_001"],
        },
        {
            "id": "mem_fact_002",
            "type": "fact",
            "content": "The user graduated with a degree in Computer Science.",
            "status": "active",
            "confidence": 0.85,
            "source_session_ids": ["sess_02"],
            "source_turn_ids": ["3"],
            "source_event_ids": ["evt_003"],
        },
        {
            "id": "mem_abs_001",
            "type": "abstract",
            "content": "The user has technical background in programming.",
            "status": "active",
            "confidence": 0.8,
            "abstraction_level": 2,
            "source_session_ids": ["sess_01", "sess_02"],
            "source_event_ids": ["evt_001", "evt_003"],
        },
        {
            "id": "mem_archived_001",
            "type": "fact",
            "content": "Archived old information.",
            "status": "archived",
            "confidence": 0.5,
            "source_session_ids": ["sess_00"],
        },
    ]

    memory["memory_summaries"]["user_profile"] = "Technical user with programming background"

    return memory


def test_retrieves_relevant_fact_node(sample_memory):
    """Test that relevant fact nodes are retrieved."""
    retriever = CurrentMemoryRetriever(top_k=3)

    evidence = retriever.retrieve(
        current_memory=sample_memory,
        question="What programming language does the user prefer?",
    )

    assert len(evidence) > 0

    # Should retrieve the Python fact
    memory_ids = {ev.memory_id for ev in evidence}
    assert "mem_fact_001" in memory_ids


def test_retrieves_abstract_node(sample_memory):
    """Test that abstract nodes are retrieved."""
    retriever = CurrentMemoryRetriever(top_k=3)

    evidence = retriever.retrieve(
        current_memory=sample_memory,
        question="What is the user's technical background?",
    )

    memory_ids = {ev.memory_id for ev in evidence}
    assert "mem_abs_001" in memory_ids


def test_includes_summaries_when_enabled(sample_memory):
    """Test that summaries are included when enabled."""
    retriever = CurrentMemoryRetriever(top_k=3, include_summaries=True)

    evidence = retriever.retrieve(
        current_memory=sample_memory,
        question="What do you know about the user?",
    )

    # Should include summary evidence
    has_summary = any(ev.memory_type == "summary" for ev in evidence)
    assert has_summary


def test_excludes_archived_nodes(sample_memory):
    """Test that archived nodes are excluded by default."""
    retriever = CurrentMemoryRetriever(top_k=10)

    evidence = retriever.retrieve(
        current_memory=sample_memory,
        question="What information do you have?",
    )

    # Should not include archived node
    memory_ids = {ev.memory_id for ev in evidence}
    assert "mem_archived_001" not in memory_ids


def test_preserves_source_session_ids(sample_memory):
    """Test that source session IDs are preserved."""
    retriever = CurrentMemoryRetriever(top_k=3)

    evidence = retriever.retrieve(
        current_memory=sample_memory,
        question="What programming language does the user prefer?",
    )

    # Find Python fact evidence
    python_evidence = next(
        (ev for ev in evidence if ev.memory_id == "mem_fact_001"),
        None
    )

    assert python_evidence is not None
    assert "sess_01" in python_evidence.source_session_ids


def test_deterministic_ordering_for_tied_scores(sample_memory):
    """Test that ordering is deterministic."""
    retriever = CurrentMemoryRetriever(top_k=3)

    evidence1 = retriever.retrieve(
        current_memory=sample_memory,
        question="Tell me about the user",
    )

    evidence2 = retriever.retrieve(
        current_memory=sample_memory,
        question="Tell me about the user",
    )

    # Should return same order
    ids1 = [ev.memory_id for ev in evidence1]
    ids2 = [ev.memory_id for ev in evidence2]

    assert ids1 == ids2


def test_empty_memory_returns_empty_evidence():
    """Test handling of empty memory."""
    memory = empty_current_memory("test")
    retriever = CurrentMemoryRetriever(top_k=3)

    evidence = retriever.retrieve(
        current_memory=memory,
        question="What do you know?",
    )

    # Should have no node evidence (may have empty summaries)
    node_evidence = [ev for ev in evidence if ev.memory_type != "summary"]
    assert len(node_evidence) == 0


def test_top_k_limit_respected(sample_memory):
    """Test that top_k limit is respected."""
    retriever = CurrentMemoryRetriever(top_k=1, include_summaries=False)

    evidence = retriever.retrieve(
        current_memory=sample_memory,
        question="What do you know about the user?",
    )

    # Should return at most top_k nodes
    node_evidence = [ev for ev in evidence if ev.memory_type != "summary"]
    assert len(node_evidence) <= 1
