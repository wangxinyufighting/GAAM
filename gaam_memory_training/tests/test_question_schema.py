"""
Tests for QuestionSchema.

Run with: pytest gaam_memory_training/tests/test_question_schema.py
"""

import pytest

from gaam_graph.question_schema import (
    FORBIDDEN_QUESTION_ARTIFACT_KEYS,
    GeneratedQuestion,
    OracleValidityReport,
    OracleValidityVerdict,
    QuestionStatus,
    QuestionType,
    assert_no_question_artifact_leakage,
    validate_generated_question,
    validate_question_set,
)


def test_valid_generated_question():
    """Test validation of a valid generated question."""
    question = {
        "question_id": "q_001",
        "record_id": "test_record",
        "question": "What degree does the user have?",
        "answer": "Business Administration",
        "question_type": "personal_fact",
        "status": "candidate",
        "supporting_trajectory_ids": ["traj_001"],
        "supporting_node_ids": ["fact_001"],
        "supporting_session_ids": ["sess_01"],
        "reason": "Generated from oracle evidence",
        "metadata": {},
    }

    validated = validate_generated_question(question, strict=True)
    assert validated.question_id == "q_001"
    assert validated.question == "What degree does the user have?"
    assert validated.answer == "Business Administration"


def test_accepted_question_requires_evidence():
    """Test that accepted questions must have evidence IDs."""
    question = {
        "question_id": "q_001",
        "record_id": "test_record",
        "question": "What degree does the user have?",
        "answer": "Business Administration",
        "question_type": "personal_fact",
        "status": "accepted",
        # Missing supporting IDs
        "supporting_trajectory_ids": [],
        "supporting_node_ids": [],
    }

    with pytest.raises(ValueError) as exc_info:
        validate_generated_question(question, strict=True)

    assert "supporting_node_ids" in str(exc_info.value) or "supporting_trajectory_ids" in str(exc_info.value)


def test_reject_empty_question():
    """Test that empty question text is rejected."""
    question = {
        "question_id": "q_001",
        "record_id": "test_record",
        "question": "",  # Empty
        "answer": "Some answer",
        "status": "candidate",
    }

    with pytest.raises(ValueError):
        validate_generated_question(question, strict=True)


def test_allow_normal_question_and_answer_fields():
    """Test that normal 'question' and 'answer' fields are allowed."""
    question = {
        "question_id": "q_001",
        "record_id": "test_record",
        "question": "What is the user's preference?",
        "answer": "Python programming",
        "status": "candidate",
        "supporting_trajectory_ids": ["traj_001"],
        "supporting_node_ids": ["fact_001"],
    }

    # Should not raise - normal question/answer fields are OK
    validated = validate_generated_question(question, strict=True)
    assert validated.question == "What is the user's preference?"
    assert validated.answer == "Python programming"


def test_reject_benchmark_leakage():
    """Test that benchmark-specific keys are rejected."""
    question = {
        "question_id": "q_001",
        "record_id": "test_record",
        "question": "What is the user's preference?",
        "answer": "Python programming",
        "gold_answer": "Leaked gold answer",  # Forbidden
        "status": "candidate",
    }

    with pytest.raises(ValueError) as exc_info:
        validate_generated_question(question, strict=True)

    assert "gold_answer" in str(exc_info.value)


def test_no_leakage_validation():
    """Test leakage detection in question artifacts."""
    # Safe artifact
    safe_artifact = {
        "record_id": "test",
        "questions": [
            {
                "question_id": "q_001",
                "question": "What?",
                "answer": "Something",
            }
        ]
    }

    assert_no_question_artifact_leakage(safe_artifact)  # Should not raise

    # Leaky artifact
    leaky_artifact = {
        "record_id": "test",
        "benchmark_question": "Leaked question",  # Forbidden
        "questions": []
    }

    with pytest.raises(ValueError) as exc_info:
        assert_no_question_artifact_leakage(leaky_artifact)

    assert "benchmark_question" in str(exc_info.value)


def test_oracle_validity_report():
    """Test OracleValidityReport validation."""
    report = OracleValidityReport(
        question_id="q_001",
        record_id="test_record",
        verdict=OracleValidityVerdict.ACCEPT,
        answerability=1.0,
        grounding=0.9,
        citation_validity=1.0,
        leakage_risk=0.0,
        issues=[],
        supporting_node_ids=["fact_001"],
        supporting_trajectory_ids=["traj_001"],
        rationale="Question is oracle-valid",
    )

    assert report.verdict == OracleValidityVerdict.ACCEPT
    assert report.answerability == 1.0
    assert report.grounding == 0.9


def test_question_set_validation():
    """Test QuestionSet validation."""
    question_set = {
        "record_id": "test_record",
        "graph_path": "/path/to/graph.json",
        "generation_config": {},
        "questions": [
            {
                "question_id": "q_001",
                "record_id": "test_record",
                "question": "What?",
                "answer": "Something",
                "status": "accepted",
                "supporting_trajectory_ids": ["traj_001"],
                "supporting_node_ids": ["fact_001"],
            }
        ],
        "validity_reports": [],
        "metadata": {},
    }

    validated = validate_question_set(question_set, strict=True)
    assert validated.record_id == "test_record"
    assert len(validated.questions) == 1


def test_all_question_types():
    """Test all question types are accepted."""
    for qt in QuestionType:
        question = {
            "question_id": "q_001",
            "record_id": "test_record",
            "question": "What?",
            "answer": "Something",
            "question_type": qt.value,
            "status": "candidate",
            "supporting_trajectory_ids": ["traj_001"],
            "supporting_node_ids": ["fact_001"],
        }

        validated = validate_generated_question(question, strict=True)
        assert validated.question_type == qt


def test_all_question_statuses():
    """Test all question statuses are accepted."""
    for status in QuestionStatus:
        question = {
            "question_id": "q_001",
            "record_id": "test_record",
            "question": "What?",
            "answer": "Something",
            "status": status.value,
            "supporting_trajectory_ids": ["traj_001"] if status == QuestionStatus.ACCEPTED else [],
            "supporting_node_ids": ["fact_001"] if status == QuestionStatus.ACCEPTED else [],
        }

        validated = validate_generated_question(question, strict=False)
        assert validated.status == status
