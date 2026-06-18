"""
Tests for reward schema.

Run with: pytest gaam_memory_training/tests/test_reward_schema.py
"""

import pytest

from gaam_graph.reward_schema import (
    FailureType,
    MemoryRewardReport,
    QuestionAgentRewardReport,
    QuestionRewardItem,
    RewardComponent,
    RewardTarget,
    assert_reward_report_no_policy_leakage,
    validate_memory_reward_report,
    validate_question_agent_reward_report,
)


def test_valid_reward_component():
    """Test valid reward component."""
    component = RewardComponent(
        name="answer_utility",
        score=0.75,
        weight=0.25,
        weighted_score=0.1875,
        rationale="Good answers",
    )

    assert component.name == "answer_utility"
    assert component.score == 0.75
    assert component.weight == 0.25


def test_invalid_score_below_zero():
    """Test that score below 0 is rejected."""
    with pytest.raises(Exception):
        RewardComponent(
            name="test",
            score=-0.1,
            weight=0.5,
            weighted_score=0.0,
        )


def test_invalid_score_above_one():
    """Test that score above 1 is rejected."""
    with pytest.raises(Exception):
        RewardComponent(
            name="test",
            score=1.5,
            weight=0.5,
            weighted_score=0.0,
        )


def test_valid_question_reward_item():
    """Test valid question reward item."""
    item = QuestionRewardItem(
        question_id="q_001",
        record_id="test_record",
        question_type="fact",
        expected_answer="Python",
        prediction="Python",
        correctness=1.0,
        evidence_support=0.9,
        citation_quality=1.0,
        diagnostic_value=0.0,
        failure_types=[],
    )

    assert item.question_id == "q_001"
    assert item.correctness == 1.0
    assert len(item.failure_types) == 0


def test_valid_memory_reward_report():
    """Test valid memory reward report."""
    report = MemoryRewardReport(
        record_id="test_record",
        target=RewardTarget.MEMORY_BUILDER,
        total_reward=0.65,
        update_reward=0.65,
        monitoring_score=0.60,
        components=[
            RewardComponent(
                name="answer_utility",
                score=0.7,
                weight=0.25,
                weighted_score=0.175,
            )
        ],
        question_items=[],
        failure_summary={},
    )

    assert report.record_id == "test_record"
    assert report.target == RewardTarget.MEMORY_BUILDER
    assert report.total_reward == 0.65


def test_failure_type_enum():
    """Test all failure types are valid."""
    failure_types = [
        FailureType.MISSING_FACT,
        FailureType.WRONG_FACT,
        FailureType.MISSING_ABSTRACTION,
        FailureType.WRONG_ABSTRACTION,
        FailureType.MISSING_MULTI_SESSION_LINK,
        FailureType.TEMPORAL_ERROR,
        FailureType.CONTRADICTION_UPDATE_ERROR,
        FailureType.OVER_COMPRESSION,
        FailureType.REDUNDANCY_NOISE,
        FailureType.UNSUPPORTED_ANSWER,
        FailureType.ANSWERER_INSUFFICIENT_EVIDENCE,
        FailureType.FORMAT_SAFETY_ERROR,
        FailureType.LEAKAGE_RISK,
        FailureType.UNKNOWN,
    ]

    for ft in failure_types:
        assert isinstance(ft.value, str)


def test_report_json_round_trip():
    """Test report serialization."""
    report = MemoryRewardReport(
        record_id="test",
        total_reward=0.5,
        update_reward=0.5,
        monitoring_score=0.5,
        components=[],
        question_items=[],
    )

    report_dict = report.model_dump()
    assert report_dict["record_id"] == "test"

    # Validate round trip
    validated = validate_memory_reward_report(report_dict, strict=False)
    assert validated.record_id == "test"


def test_assert_no_policy_leakage():
    """Test leakage detection."""
    # Valid report
    valid_report = {
        "record_id": "test",
        "target": "memory_builder",
        "total_reward": 0.5,
        "update_reward": 0.5,
        "monitoring_score": 0.5,
        "components": [],
        "question_items": [],
    }

    assert_reward_report_no_policy_leakage(valid_report)  # Should not raise

    # Missing target
    invalid_report_1 = {
        "record_id": "test",
        "total_reward": 0.5,
    }

    with pytest.raises(ValueError) as exc_info:
        assert_reward_report_no_policy_leakage(invalid_report_1)
    assert "target" in str(exc_info.value)

    # Contains memory_graph
    invalid_report_2 = {
        "record_id": "test",
        "target": "memory_builder",
        "memory_graph": {},
    }

    with pytest.raises(ValueError) as exc_info:
        assert_reward_report_no_policy_leakage(invalid_report_2)
    assert "memory_graph" in str(exc_info.value)


def test_question_agent_reward_report():
    """Test question agent reward report."""
    report = QuestionAgentRewardReport(
        record_id="test",
        target=RewardTarget.QUESTION_AGENT,
        total_reward=0.7,
        components=[],
        per_question_scores=[],
        coverage_gain=0.3,
        adversarial_success_rate=0.5,
        diversity_score=0.6,
    )

    assert report.target == RewardTarget.QUESTION_AGENT
    assert report.coverage_gain == 0.3

