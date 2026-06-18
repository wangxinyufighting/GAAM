"""
Tests for FrozenAnswerer.

Run with: pytest gaam_memory_training/tests/test_answer_agent.py
"""

import pytest

from gaam_graph.answer_agent import FrozenAnswerer
from gaam_graph.answer_schema import AnswerRequest, AnswerStatus, answer_request_from_generated_question
from gaam_graph.memory_schema import empty_current_memory


class DummyLLM:
    """Minimal LLM stub for FrozenAnswerer tests."""

    def __init__(self, response):
        self.response = response

    def chat_json(self, system: str, user: str):
        assert "Frozen Answerer" in system
        assert "retrieved_memory" in user
        return self.response


@pytest.fixture
def sample_memory():
    """Create sample current memory."""
    memory = empty_current_memory("test_record")

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
            "content": "The user prefers working with pandas and numpy libraries.",
            "status": "active",
            "confidence": 0.85,
            "source_session_ids": ["sess_01"],
            "source_turn_ids": ["2"],
            "source_event_ids": ["evt_002"],
        },
    ]

    return memory


def test_no_llm_answer_returns_evidence_backed_report(sample_memory):
    """Test that no-LLM mode returns evidence-backed report."""
    answerer = FrozenAnswerer(no_llm=True)

    request = AnswerRequest(
        record_id="test_record",
        question_id="q_001",
        question="What programming language does the user prefer?",
    )

    report = answerer.answer(
        current_memory=sample_memory,
        request=request,
    )

    assert report.record_id == "test_record"
    assert report.question_id == "q_001"
    assert report.answer_status == AnswerStatus.ANSWERED
    assert report.prediction != ""
    assert len(report.supporting_memory_ids) > 0


def test_insufficient_evidence_returns_status(sample_memory):
    """Test that insufficient evidence returns appropriate status."""
    # Empty memory
    empty_mem = empty_current_memory("test_record")

    answerer = FrozenAnswerer(no_llm=True)

    request = AnswerRequest(
        record_id="test_record",
        question_id="q_001",
        question="What is the user's favorite color?",
    )

    report = answerer.answer(
        current_memory=empty_mem,
        request=request,
    )

    assert report.answer_status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert report.prediction == ""
    assert report.confidence == 0.0


def test_generated_question_sanitizer_drops_expected_answer():
    """Test that sanitizer drops expected answer."""
    question = {
        "question_id": "q_001",
        "record_id": "test_record",
        "question": "What language does the user prefer?",
        "answer": "Python programming",  # Should be dropped
        "supporting_node_ids": ["oracle_fact_001"],  # Should be dropped
    }

    request = answer_request_from_generated_question(question)

    # Check that answer was dropped
    request_dict = request.model_dump()
    assert "answer" not in request_dict


def test_generated_question_sanitizer_drops_oracle_ids():
    """Test that sanitizer drops oracle supporting IDs."""
    question = {
        "question_id": "q_001",
        "record_id": "test_record",
        "question": "What?",
        "answer": "Something",
        "supporting_node_ids": ["oracle_fact_001"],  # Oracle IDs
        "supporting_trajectory_ids": ["traj_001"],
    }

    request = answer_request_from_generated_question(question)

    request_dict = request.model_dump()
    assert "supporting_node_ids" not in request_dict
    assert "supporting_trajectory_ids" not in request_dict


def test_batch_answer(sample_memory):
    """Test batch answering."""
    answerer = FrozenAnswerer(no_llm=True)

    requests = [
        AnswerRequest(
            record_id="test_record",
            question_id=f"q_{i:03d}",
            question=f"Question {i}?",
        )
        for i in range(3)
    ]

    reports = answerer.batch_answer(
        current_memory=sample_memory,
        requests=requests,
    )

    assert len(reports) == 3
    for i, report in enumerate(reports):
        assert report.question_id == f"q_{i:03d}"


def test_supporting_memory_ids_are_current_memory_ids(sample_memory):
    """Test that supporting IDs are CurrentMemory IDs, not oracle IDs."""
    answerer = FrozenAnswerer(no_llm=True)

    request = AnswerRequest(
        record_id="test_record",
        question_id="q_001",
        question="What programming language does the user prefer?",
    )

    report = answerer.answer(
        current_memory=sample_memory,
        request=request,
    )

    # Supporting IDs should be from CurrentMemory
    for mem_id in report.supporting_memory_ids:
        assert mem_id.startswith("mem_") or mem_id.startswith("summary_")


def test_answer_includes_supporting_evidence(sample_memory):
    """Test that answer includes supporting evidence."""
    answerer = FrozenAnswerer(no_llm=True)

    request = AnswerRequest(
        record_id="test_record",
        question_id="q_001",
        question="What does the user prefer?",
    )

    report = answerer.answer(
        current_memory=sample_memory,
        request=request,
    )

    if report.answer_status == AnswerStatus.ANSWERED:
        assert len(report.supporting_evidence) > 0
        for ev in report.supporting_evidence:
            assert ev.memory_id != ""
            assert ev.memory_type != ""
            assert ev.content != ""


def test_llm_answer_rejects_unknown_supporting_memory_id(sample_memory):
    """Test LLM mode rejects invented or oracle-looking support IDs."""
    answerer = FrozenAnswerer(
        llm=DummyLLM(
            {
                "prediction": "Python programming",
                "answer_status": "answered",
                "confidence": 0.9,
                "supporting_memory_ids": ["oracle_fact_001"],
                "rationale": "Bad citation",
            }
        )
    )

    request = AnswerRequest(
        record_id="test_record",
        question_id="q_001",
        question="What programming language does the user prefer?",
    )

    report = answerer.answer(current_memory=sample_memory, request=request)

    assert report.answer_status == AnswerStatus.INVALID_OUTPUT
    assert report.supporting_memory_ids == []
    assert "Invalid supporting_memory_ids" in report.rationale


def test_llm_answer_accepts_retrieved_supporting_memory_id(sample_memory):
    """Test LLM mode accepts IDs that came from retrieved CurrentMemory evidence."""
    answerer = FrozenAnswerer(
        llm=DummyLLM(
            {
                "prediction": "Python programming",
                "answer_status": "answered",
                "confidence": 0.9,
                "supporting_memory_ids": ["mem_fact_001"],
                "rationale": "Supported by memory",
            }
        ),
        model_id="dummy_answerer",
    )

    request = AnswerRequest(
        record_id="test_record",
        question_id="q_001",
        question="What programming language does the user prefer?",
    )

    report = answerer.answer(current_memory=sample_memory, request=request)

    assert report.answer_status == AnswerStatus.ANSWERED
    assert report.prediction == "Python programming"
    assert report.supporting_memory_ids == ["mem_fact_001"]
    assert [ev.memory_id for ev in report.supporting_evidence] == ["mem_fact_001"]
    assert report.model_info["model"] == "dummy_answerer"


def test_answerer_revalidates_constructed_request_metadata(sample_memory):
    """Test answer() rejects leaked metadata even if a request bypassed pydantic init."""
    request = AnswerRequest.model_construct(
        record_id="test_record",
        question_id="q_001",
        question="What?",
        current_memory_path="",
        metadata={"gold_answer": "secret"},
    )

    answerer = FrozenAnswerer(no_llm=True)

    with pytest.raises(ValueError) as exc_info:
        answerer.answer(current_memory=sample_memory, request=request)

    assert "gold_answer" in str(exc_info.value)
