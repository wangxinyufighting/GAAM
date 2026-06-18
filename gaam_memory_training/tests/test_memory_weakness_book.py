"""
Tests for memory weakness book.

Run with: pytest gaam_memory_training/tests/test_memory_weakness_book.py
"""

import tempfile
from pathlib import Path

import pytest

from gaam_graph.memory_weakness_book import (
    MemoryWeaknessBook,
    WeaknessEntry,
    WeaknessStatus,
    stable_weakness_id,
)
from gaam_graph.reward_schema import (
    FailureType,
    MemoryRewardReport,
    QuestionRewardItem,
    RewardTarget,
)


def test_empty_book():
    """Test creating empty weakness book."""
    book = MemoryWeaknessBook.empty("test_record")

    assert book.record_id == "test_record"
    assert len(book.weaknesses) == 0
    assert book.version == 1


def test_stable_weakness_id():
    """Test stable weakness ID generation."""
    id1 = stable_weakness_id(
        record_id="test",
        question_id="q_001",
        failure_type="missing_fact",
        oracle_node_ids=["fact_001", "fact_002"],
    )

    id2 = stable_weakness_id(
        record_id="test",
        question_id="q_001",
        failure_type="missing_fact",
        oracle_node_ids=["fact_002", "fact_001"],  # Different order
    )

    # Should be stable regardless of order
    assert id1 == id2
    assert id1.startswith("weak_")


def test_update_from_one_failure():
    """Test updating from one failure."""
    book = MemoryWeaknessBook.empty("test")

    report = MemoryRewardReport(
        record_id="test",
        target=RewardTarget.MEMORY_BUILDER,
        total_reward=0.5,
        update_reward=0.5,
        monitoring_score=0.5,
        components=[],
        question_items=[
            QuestionRewardItem(
                question_id="q_001",
                record_id="test",
                question_type="fact",
                expected_answer="Python",
                prediction="",
                correctness=0.0,
                evidence_support=0.0,
                citation_quality=0.0,
                diagnostic_value=1.0,
                failure_types=[FailureType.MISSING_FACT],
                supporting_oracle_node_ids=["fact_001"],
                related_session_ids=["s1"],
            )
        ],
    )

    updated = book.update_from_reward_report(report, round_id=1)

    assert len(updated) == 1
    assert len(book.weaknesses) == 1
    assert book.weaknesses[0].failure_type == FailureType.MISSING_FACT
    assert book.weaknesses[0].frequency == 1


def test_repeated_failure_increments_frequency():
    """Test that repeated failure increments frequency."""
    book = MemoryWeaknessBook.empty("test")

    report1 = MemoryRewardReport(
        record_id="test",
        target=RewardTarget.MEMORY_BUILDER,
        total_reward=0.5,
        update_reward=0.5,
        monitoring_score=0.5,
        components=[],
        question_items=[
            QuestionRewardItem(
                question_id="q_001",
                record_id="test",
                correctness=0.0,
                evidence_support=0.0,
                citation_quality=0.0,
                diagnostic_value=1.0,
                failure_types=[FailureType.MISSING_FACT],
                supporting_oracle_node_ids=["fact_001"],
            )
        ],
    )

    book.update_from_reward_report(report1, round_id=1)
    assert book.weaknesses[0].frequency == 1

    # Same failure again
    book.update_from_reward_report(report1, round_id=2)
    assert book.weaknesses[0].frequency == 2


def test_repeated_oracle_nodes_merge():
    """Test that repeated exact oracle nodes merge into same weakness."""
    book = MemoryWeaknessBook.empty("test")

    item1 = QuestionRewardItem(
        question_id="q_001",
        record_id="test",
        correctness=0.0,
        evidence_support=0.0,
        citation_quality=0.0,
        diagnostic_value=1.0,
        failure_types=[FailureType.MISSING_FACT],
        supporting_oracle_node_ids=["fact_001"],
    )

    report1 = MemoryRewardReport(
        record_id="test",
        target=RewardTarget.MEMORY_BUILDER,
        total_reward=0.5,
        update_reward=0.5,
        monitoring_score=0.5,
        components=[],
        question_items=[item1],
    )

    book.update_from_reward_report(report1, round_id=1)

    # Same question, same failure type, same oracle nodes -> should merge
    book.update_from_reward_report(report1, round_id=2)

    assert len(book.weaknesses) == 1
    assert book.weaknesses[0].frequency == 2


def test_overlapping_oracle_nodes_merge_across_questions():
    """Test overlapping oracle nodes merge across different questions."""
    book = MemoryWeaknessBook.empty("test")

    report1 = MemoryRewardReport(
        record_id="test",
        target=RewardTarget.MEMORY_BUILDER,
        total_reward=0.5,
        update_reward=0.5,
        monitoring_score=0.5,
        components=[],
        question_items=[
            QuestionRewardItem(
                question_id="q_001",
                record_id="test",
                correctness=0.0,
                evidence_support=0.0,
                citation_quality=0.0,
                diagnostic_value=1.0,
                failure_types=[FailureType.MISSING_FACT],
                supporting_oracle_node_ids=["fact_001", "fact_002"],
            )
        ],
    )
    report2 = MemoryRewardReport(
        record_id="test",
        target=RewardTarget.MEMORY_BUILDER,
        total_reward=0.5,
        update_reward=0.5,
        monitoring_score=0.5,
        components=[],
        question_items=[
            QuestionRewardItem(
                question_id="q_002",
                record_id="test",
                correctness=0.2,
                evidence_support=0.0,
                citation_quality=0.0,
                diagnostic_value=1.0,
                failure_types=[FailureType.MISSING_FACT],
                supporting_oracle_node_ids=["fact_002", "fact_003"],
            )
        ],
    )

    book.update_from_reward_report(report1, round_id=1)
    book.update_from_reward_report(report2, round_id=2)

    assert len(book.weaknesses) == 1
    assert book.weaknesses[0].frequency == 2
    assert book.weaknesses[0].missing_oracle_node_ids == ["fact_001", "fact_002", "fact_003"]


def test_different_failure_type_creates_separate_weakness():
    """Test that different failure type creates separate weakness."""
    book = MemoryWeaknessBook.empty("test")

    report1 = MemoryRewardReport(
        record_id="test",
        target=RewardTarget.MEMORY_BUILDER,
        total_reward=0.5,
        update_reward=0.5,
        monitoring_score=0.5,
        components=[],
        question_items=[
            QuestionRewardItem(
                question_id="q_001",
                record_id="test",
                correctness=0.0,
                evidence_support=0.0,
                citation_quality=0.0,
                diagnostic_value=1.0,
                failure_types=[FailureType.MISSING_FACT],
                supporting_oracle_node_ids=["fact_001"],
            )
        ],
    )

    report2 = MemoryRewardReport(
        record_id="test",
        target=RewardTarget.MEMORY_BUILDER,
        total_reward=0.5,
        update_reward=0.5,
        monitoring_score=0.5,
        components=[],
        question_items=[
            QuestionRewardItem(
                question_id="q_001",
                record_id="test",
                correctness=0.0,
                evidence_support=0.0,
                citation_quality=0.0,
                diagnostic_value=1.0,
                failure_types=[FailureType.WRONG_FACT],  # Different type
                supporting_oracle_node_ids=["fact_001"],
            )
        ],
    )

    book.update_from_reward_report(report1, round_id=1)
    book.update_from_reward_report(report2, round_id=2)

    assert len(book.weaknesses) == 2


def test_mark_mastered():
    """Test marking weakness as mastered."""
    book = MemoryWeaknessBook.empty("test")

    report = MemoryRewardReport(
        record_id="test",
        target=RewardTarget.MEMORY_BUILDER,
        total_reward=0.5,
        update_reward=0.5,
        monitoring_score=0.5,
        components=[],
        question_items=[
            QuestionRewardItem(
                question_id="q_001",
                record_id="test",
                correctness=0.0,
                evidence_support=0.0,
                citation_quality=0.0,
                diagnostic_value=1.0,
                failure_types=[FailureType.MISSING_FACT],
                supporting_oracle_node_ids=["fact_001"],
            )
        ],
    )

    book.update_from_reward_report(report, round_id=1)

    assert book.weaknesses[0].status == WeaknessStatus.ACTIVE

    # Mark mastered
    book.mark_mastered(question_id="q_001", round_id=2)

    assert book.weaknesses[0].status == WeaknessStatus.MASTERED


def test_save_load_round_trip():
    """Test save and load round trip."""
    book = MemoryWeaknessBook.empty("test")

    report = MemoryRewardReport(
        record_id="test",
        target=RewardTarget.MEMORY_BUILDER,
        total_reward=0.5,
        update_reward=0.5,
        monitoring_score=0.5,
        components=[],
        question_items=[
            QuestionRewardItem(
                question_id="q_001",
                record_id="test",
                correctness=0.0,
                evidence_support=0.0,
                citation_quality=0.0,
                diagnostic_value=1.0,
                failure_types=[FailureType.MISSING_FACT],
                supporting_oracle_node_ids=["fact_001"],
            )
        ],
    )

    book.update_from_reward_report(report, round_id=1)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "weakness_book.json"
        book.save(path)

        # Load
        loaded_book = MemoryWeaknessBook.load(path, record_id="test")

        assert loaded_book.record_id == "test"
        assert len(loaded_book.weaknesses) == 1
        assert loaded_book.weaknesses[0].failure_type == FailureType.MISSING_FACT


def test_question_agent_context_export():
    """Test exporting context for Question Agent."""
    book = MemoryWeaknessBook.empty("test")

    report = MemoryRewardReport(
        record_id="test",
        target=RewardTarget.MEMORY_BUILDER,
        total_reward=0.5,
        update_reward=0.5,
        monitoring_score=0.5,
        components=[],
        question_items=[
            QuestionRewardItem(
                question_id="q_001",
                record_id="test",
                question_type="multi_session",
                correctness=0.0,
                evidence_support=0.0,
                citation_quality=0.0,
                diagnostic_value=1.0,
                failure_types=[FailureType.MISSING_MULTI_SESSION_LINK],
                supporting_oracle_node_ids=["fact_001"],
                related_session_ids=["s1", "s2"],
            )
        ],
    )

    book.update_from_reward_report(report, round_id=1)

    context = book.to_question_agent_context(max_entries=10, include_oracle_ids=True)

    assert "active_weaknesses" in context
    assert len(context["active_weaknesses"]) == 1
    assert context["active_weaknesses"][0]["failure_type"] == "missing_multi_session_link"
    assert "missing_oracle_node_ids" in context["active_weaknesses"][0]
