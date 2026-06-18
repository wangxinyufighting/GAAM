"""
Tests for AnswerSchema.

Run with: pytest gaam_memory_training/tests/test_answer_schema.py
"""

import pytest

from gaam_graph.answer_schema import (
    FORBIDDEN_ANSWER_INPUT_KEYS,
    AnswerReport,
    AnswerRequest,
    AnswerStatus,
    answer_request_from_generated_question,
    assert_no_answer_input_leakage,
    validate_answer_report,
    validate_answer_request,
    validate_supporting_memory_ids,
)


def test_valid_answer_request():
    """Test validation of a valid answer request."""
    request = {
        "record_id": "test_record",
        "question_id": "q_001",
        "question": "What does the user prefer?",
        "current_memory_path": "/path/to/memory.json",
        "metadata": {},
    }

    validated = validate_answer_request(request, strict=True)
    assert validated.record_id == "test_record"
    assert validated.question_id == "q_001"
    assert validated.question == "What does the user prefer?"


def test_valid_answer_report():
    """Test validation of a valid answer report."""
    report = AnswerReport(
        record_id="test_record",
        question_id="q_001",
        question="What does the user prefer?",
        prediction="Python programming",
        answer_status=AnswerStatus.ANSWERED,
        confidence=0.85,
        supporting_memory_ids=["mem_fact_001"],
        rationale="Found in memory",
        model_info={"model": "test_model"},
    )

    assert report.record_id == "test_record"
    assert report.prediction == "Python programming"
    assert report.answer_status == AnswerStatus.ANSWERED


def test_reject_empty_question():
    """Test that empty question text is rejected."""
    request = {
        "record_id": "test_record",
        "question_id": "q_001",
        "question": "",  # Empty
    }

    with pytest.raises(ValueError):
        validate_answer_request(request, strict=True)


def test_reject_forbidden_gold_answer():
    """Test that gold_answer is rejected in request."""
    request = {
        "record_id": "test_record",
        "question_id": "q_001",
        "question": "What?",
        "gold_answer": "Leaked answer",  # Forbidden
    }

    with pytest.raises(ValueError) as exc_info:
        validate_answer_request(request, strict=True)

    assert "gold_answer" in str(exc_info.value)


def test_reject_forbidden_supporting_node_ids():
    """Test that oracle supporting_node_ids are rejected in request metadata."""
    request = {
        "record_id": "test_record",
        "question_id": "q_001",
        "question": "What?",
        "metadata": {
            "supporting_node_ids": ["oracle_fact_001"],  # Forbidden (oracle IDs)
        },
    }

    with pytest.raises(ValueError) as exc_info:
        validate_answer_request(request, strict=True)

    assert "supporting_node_ids" in str(exc_info.value)


def test_reject_forbidden_request_metadata_on_model_creation():
    """Test AnswerRequest rejects leaked metadata even without helper validation."""
    with pytest.raises(ValueError) as exc_info:
        AnswerRequest(
            record_id="test_record",
            question_id="q_001",
            question="What?",
            metadata={"gold_answer": "secret"},
        )

    assert "gold_answer" in str(exc_info.value)


def test_allow_supporting_memory_ids_in_report():
    """Test that supporting_memory_ids are allowed in report."""
    report = AnswerReport(
        record_id="test_record",
        question_id="q_001",
        question="What?",
        prediction="Something",
        answer_status=AnswerStatus.ANSWERED,
        confidence=0.8,
        supporting_memory_ids=["mem_fact_001", "mem_fact_002"],  # CurrentMemory IDs - OK
    )

    assert len(report.supporting_memory_ids) == 2


def test_reject_forbidden_report_metadata():
    """Test AnswerReport rejects leaked evaluator metadata."""
    with pytest.raises(ValueError) as exc_info:
        validate_answer_report(
            {
                "record_id": "test_record",
                "question_id": "q_001",
                "question": "What?",
                "prediction": "Something",
                "answer_status": "answered",
                "confidence": 0.8,
                "supporting_memory_ids": ["mem_fact_001"],
                "metadata": {"oracle_answer": "secret"},
            },
            strict=True,
        )

    assert "oracle_answer" in str(exc_info.value)


def test_validate_supporting_memory_ids_rejects_unknown_ids():
    """Test supporting_memory_ids must come from retrieved CurrentMemory evidence."""
    with pytest.raises(ValueError) as exc_info:
        validate_supporting_memory_ids(
            ["mem_fact_001", "oracle_fact_001"],
            allowed_memory_ids={"mem_fact_001"},
        )

    assert "oracle_fact_001" in str(exc_info.value)


def test_answer_request_from_generated_question():
    """Test that sanitizer drops expected answer and oracle IDs."""
    question = {
        "question_id": "q_001",
        "record_id": "test_record",
        "question": "What does the user prefer?",
        "answer": "Python programming",  # Should be dropped
        "supporting_node_ids": ["oracle_fact_001"],  # Should be dropped
        "supporting_trajectory_ids": ["traj_001"],  # Should be dropped
        "supporting_session_ids": ["sess_01"],  # Should be dropped
        "reason": "Some reason",  # Should be dropped
        "metadata": {"some": "data"},  # Should be dropped
    }

    request = answer_request_from_generated_question(question)

    assert request.question_id == "q_001"
    assert request.record_id == "test_record"
    assert request.question == "What does the user prefer?"

    # Check that sensitive fields were dropped
    request_dict = request.model_dump()
    assert "answer" not in request_dict
    assert "supporting_node_ids" not in request_dict
    assert "supporting_trajectory_ids" not in request_dict
    assert "reason" not in request_dict
    assert request_dict["metadata"] == {}


def test_no_leakage_validation():
    """Test leakage detection in answer input."""
    # Safe request
    safe_request = {
        "record_id": "test",
        "question_id": "q_001",
        "question": "What?",
    }

    assert_no_answer_input_leakage(safe_request)  # Should not raise

    # Leaky request
    leaky_request = {
        "record_id": "test",
        "question_id": "q_001",
        "question": "What?",
        "oracle_graph": {},  # Forbidden
    }

    with pytest.raises(ValueError) as exc_info:
        assert_no_answer_input_leakage(leaky_request)

    assert "oracle_graph" in str(exc_info.value)


def test_all_answer_statuses():
    """Test all answer statuses are valid."""
    for status in AnswerStatus:
        report = AnswerReport(
            record_id="test_record",
            question_id="q_001",
            question="What?",
            prediction="Something" if status == AnswerStatus.ANSWERED else "",
            answer_status=status,
            confidence=0.5,
        )

        assert report.answer_status == status
