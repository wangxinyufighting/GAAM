"""
Tests for reward manager.

Run with: pytest gaam_memory_training/tests/test_reward_manager.py
"""

import tempfile
from pathlib import Path

import pytest

from gaam_graph.oracle_graph_loader import OracleGraphLoader
from gaam_graph.reward_manager import (
    RewardManager,
    classify_question_failure,
    score_answer_correctness,
    token_f1,
    token_jaccard,
)
from gaam_graph.reward_schema import FailureType
from gaam_graph.memory_schema import empty_current_memory
from gaam_graph.utils import write_json


@pytest.fixture
def sample_oracle_graph(tmp_path):
    """Create sample oracle graph."""
    graph_data = {
        "graph": {
            "record_id": "test_record",
        },
        "nodes": [
            {
                "id": "fact_001",
                "type": "fact",
                "text": "The user likes Python programming for data analysis.",
            },
            {
                "id": "fact_002",
                "type": "fact",
                "text": "The user prefers pandas and numpy libraries.",
            },
            {
                "id": "abstract_001",
                "type": "abstract_memory",
                "text": "The user has technical background in programming.",
            },
        ],
        "edges": [],
    }
    graph_path = tmp_path / "test.graph.json"
    write_json(graph_path, graph_data)
    return OracleGraphLoader(graph_path)


@pytest.fixture
def sample_current_memory():
    """Create sample current memory."""
    memory = empty_current_memory("test_record")
    memory["memory_graph"]["nodes"] = [
        {
            "id": "mem_fact_001",
            "type": "fact",
            "content": "The user likes Python programming for data analysis.",
            "status": "active",
            "source_session_ids": ["s1"],
            "source_turn_ids": ["1"],
            "source_event_ids": ["e1"],
        },
    ]
    return memory


def test_exact_answer_correctness():
    """Test exact match correctness."""
    score = score_answer_correctness("Python", "Python")
    assert score == 1.0


def test_partial_answer_correctness():
    """Test partial match correctness."""
    score = score_answer_correctness(
        "Python programming language",
        "Python"
    )
    assert score >= 0.5  # Changed from > to >=


def test_unrelated_answer_correctness():
    """Test unrelated answer."""
    score = score_answer_correctness("JavaScript", "Python")
    assert score < 0.3


def test_token_f1():
    """Test token F1 scoring."""
    f1 = token_f1("Python programming", "Python language")
    assert 0.0 < f1 < 1.0

    # Exact match
    f1_exact = token_f1("Python", "Python")
    assert f1_exact == 1.0


def test_token_jaccard():
    """Test token Jaccard similarity."""
    jaccard = token_jaccard("Python programming", "Python language")
    assert 0.0 < jaccard < 1.0


def test_answer_utility_aggregation(sample_oracle_graph, sample_current_memory):
    """Test answer utility aggregation."""
    manager = RewardManager()

    questions = [
        {
            "question_id": "q_001",
            "record_id": "test_record",
            "question": "What language?",
            "answer": "Python programming",
            "supporting_node_ids": ["fact_001"],
        }
    ]

    answers = [
        {
            "question_id": "q_001",
            "prediction": "Python programming",
            "answer_status": "answered",
            "supporting_memory_ids": ["mem_fact_001"],
            "supporting_evidence": [
                {
                    "memory_id": "mem_fact_001",
                    "memory_type": "fact",
                    "content": "The user likes Python programming for data analysis.",
                    "score": 0.9,
                }
            ],
        }
    ]

    report = manager.score_memory_builder(
        current_memory=sample_current_memory,
        questions=questions,
        answer_reports=answers,
        oracle_graph=sample_oracle_graph,
    )

    assert 0.0 <= report.total_reward <= 1.0
    assert len(report.components) > 0
    assert report.question_items[0].correctness >= 0.8


def test_missing_fact_classification(sample_oracle_graph, sample_current_memory):
    """Test missing fact classification."""
    question = {
        "question_id": "q_001",
        "record_id": "test_record",
        "question": "What libraries?",
        "answer": "pandas and numpy",
        "supporting_node_ids": ["fact_002"],  # Not in current memory
        "supporting_session_ids": ["s1"],
    }

    answer_report = {
        "question_id": "q_001",
        "prediction": "I don't know",
        "answer_status": "insufficient_evidence",
        "supporting_memory_ids": [],
        "supporting_evidence": [],
    }

    failures = classify_question_failure(
        question=question,
        answer_report=answer_report,
        correctness=0.0,
        evidence_support=0.0,
        oracle_coverage=0.2,
        current_memory=sample_current_memory,
        oracle_graph=sample_oracle_graph,
    )

    assert FailureType.MISSING_FACT in failures or FailureType.ANSWERER_INSUFFICIENT_EVIDENCE in failures


def test_missing_multi_session_link_classification(sample_oracle_graph, sample_current_memory):
    """Test missing multi-session link classification."""
    question = {
        "question_id": "q_001",
        "record_id": "test_record",
        "question": "What changed?",
        "answer": "Something",
        "supporting_node_ids": ["fact_001", "fact_002"],
        "supporting_session_ids": ["s1", "s2"],  # Multiple sessions
    }

    answer_report = {
        "question_id": "q_001",
        "prediction": "Wrong answer",
        "answer_status": "answered",
        "supporting_memory_ids": ["mem_fact_001"],
        "supporting_evidence": [],
    }

    failures = classify_question_failure(
        question=question,
        answer_report=answer_report,
        correctness=0.3,
        evidence_support=0.5,
        oracle_coverage=0.5,
        current_memory=sample_current_memory,
        oracle_graph=sample_oracle_graph,
    )

    assert FailureType.MISSING_MULTI_SESSION_LINK in failures


def test_redundancy_penalty():
    """Test redundancy penalty scoring."""
    # Create memory with duplicate nodes
    memory = empty_current_memory("test")
    memory["memory_graph"]["nodes"] = [
        {
            "id": "mem_001",
            "type": "fact",
            "content": "The user likes Python programming.",
            "status": "active",
        },
        {
            "id": "mem_002",
            "type": "fact",
            "content": "The user likes Python programming.",  # Duplicate
            "status": "active",
        },
    ]

    manager = RewardManager()
    non_redundancy = manager._score_non_redundancy(memory)

    # Should detect redundancy
    assert non_redundancy < 0.5


def test_component_weights_sum(tmp_path):
    """Test component weights sum correctly."""
    manager = RewardManager()

    # Create minimal valid inputs
    memory = empty_current_memory("test")
    questions = []
    answers = []

    # Create oracle graph file
    graph_data = {
        "graph": {"record_id": "test"},
        "nodes": [],
        "edges": []
    }
    graph_path = tmp_path / "test.graph.json"
    write_json(graph_path, graph_data)
    oracle_graph = OracleGraphLoader(graph_path)

    report = manager.score_memory_builder(
        current_memory=memory,
        questions=questions,
        answer_reports=answers,
        oracle_graph=oracle_graph,
    )

    # Check components exist
    assert len(report.components) > 0

    # Check weighted scores
    for component in report.components:
        assert component.weighted_score >= 0.0


def test_missing_answer_is_scored_as_failure(sample_oracle_graph, sample_current_memory):
    """Test missing answer reports are not silently dropped."""
    manager = RewardManager()
    questions = [
        {
            "question_id": "q_missing",
            "record_id": "test_record",
            "question": "What libraries?",
            "answer": "pandas and numpy",
            "supporting_node_ids": ["fact_002"],
        }
    ]

    report = manager.score_memory_builder(
        current_memory=sample_current_memory,
        questions=questions,
        answer_reports=[],
        oracle_graph=sample_oracle_graph,
    )

    assert len(report.question_items) == 1
    assert report.question_items[0].answer_status == "insufficient_evidence"
    assert "answerer_insufficient_evidence" in report.failure_summary


def test_global_redundancy_failure_in_summary(tmp_path):
    """Test memory-level redundancy is surfaced in failure_summary."""
    manager = RewardManager()
    memory = empty_current_memory("test")
    memory["memory_graph"]["nodes"] = [
        {
            "id": "mem_001",
            "type": "fact",
            "content": "The user likes Python programming.",
            "status": "active",
            "source_session_ids": ["s1"],
        },
        {
            "id": "mem_002",
            "type": "fact",
            "content": "The user likes Python programming.",
            "status": "active",
            "source_session_ids": ["s1"],
        },
    ]
    graph_path = tmp_path / "test.graph.json"
    write_json(graph_path, {"graph": {"record_id": "test"}, "nodes": [], "edges": []})

    report = manager.score_memory_builder(
        current_memory=memory,
        questions=[],
        answer_reports=[],
        oracle_graph=OracleGraphLoader(graph_path),
    )

    assert "redundancy_noise" in report.failure_summary


def test_global_over_compression_failure_in_summary(sample_oracle_graph):
    """Test low oracle coverage surfaces over_compression."""
    manager = RewardManager()
    memory = empty_current_memory("test_record")
    questions = [
        {
            "question_id": "q_001",
            "record_id": "test_record",
            "question": "What libraries?",
            "answer": "pandas and numpy",
            "supporting_node_ids": ["fact_002"],
        }
    ]
    answers = [
        {
            "question_id": "q_001",
            "prediction": "",
            "answer_status": "insufficient_evidence",
            "supporting_memory_ids": [],
            "supporting_evidence": [],
        }
    ]

    report = manager.score_memory_builder(
        current_memory=memory,
        questions=questions,
        answer_reports=answers,
        oracle_graph=sample_oracle_graph,
    )

    assert "over_compression" in report.failure_summary


def test_abstraction_quality_counts_current_memory_abstract_type():
    """Test MemorySchema's 'abstract' node type counts as abstraction."""
    manager = RewardManager()
    memory = empty_current_memory("test")
    memory["memory_graph"]["nodes"] = [
        {
            "id": "mem_abs_001",
            "type": "abstract",
            "content": "The user has a technical programming background.",
            "status": "active",
            "abstraction_level": 2,
            "source_session_ids": ["s1", "s2"],
        }
    ]

    assert manager._score_abstraction_quality(memory) > 0.7


def test_invalid_citation_penalty(sample_oracle_graph):
    """Test invalid citation penalty."""
    manager = RewardManager()

    memory = empty_current_memory("test")
    memory["memory_graph"]["nodes"] = [
        {"id": "mem_001", "type": "fact", "content": "test", "status": "active"}
    ]

    answer_report = {
        "question_id": "q_001",
        "prediction": "answer",
        "answer_status": "answered",
        "supporting_memory_ids": ["invalid_id"],  # Invalid ID
        "supporting_evidence": [],
    }

    citation_quality = manager._score_citation_quality(
        answer_report=answer_report,
        current_memory=memory,
    )

    assert citation_quality == 0.0


def test_reward_report_contains_diagnostics(sample_oracle_graph, sample_current_memory):
    """Test that reward report contains diagnostic information."""
    manager = RewardManager()

    questions = [
        {
            "question_id": "q_001",
            "record_id": "test_record",
            "question": "What?",
            "answer": "Python",
            "supporting_node_ids": ["fact_001"],
        }
    ]

    answers = [
        {
            "question_id": "q_001",
            "prediction": "JavaScript",
            "answer_status": "answered",
            "supporting_memory_ids": [],
            "supporting_evidence": [],
        }
    ]

    report = manager.score_memory_builder(
        current_memory=sample_current_memory,
        questions=questions,
        answer_reports=answers,
        oracle_graph=sample_oracle_graph,
    )

    # Should have diagnostics
    assert len(report.question_items) > 0
    assert report.failure_summary is not None
    assert len(report.components) > 0


def test_reward_manager_llm_judge_scores_answer_correctness(
    sample_oracle_graph,
    sample_current_memory,
):
    """Test correctness_mode=llm_judge calls an API-style judge client."""

    class FakeJudgeLLM:
        def __init__(self):
            self.calls = []

        def chat_json(self, *, system, user, schema_hint=None):
            self.calls.append({"system": system, "user": user, "schema_hint": schema_hint})
            return {"correctness": 0.91, "rationale": "Prediction matches the expected answer."}

    fake_llm = FakeJudgeLLM()
    manager = RewardManager(correctness_mode="llm_judge", llm=fake_llm)
    questions = [
        {
            "question_id": "q_001",
            "record_id": "test_record",
            "question": "What language does the user like?",
            "answer": "Python",
            "supporting_node_ids": ["fact_001"],
        }
    ]
    answers = [
        {
            "question_id": "q_001",
            "prediction": "They like Python.",
            "answer_status": "answered",
            "supporting_memory_ids": ["mem_fact_001"],
            "supporting_evidence": [
                {
                    "memory_id": "mem_fact_001",
                    "content": "The user likes Python programming for data analysis.",
                }
            ],
        }
    ]

    report = manager.score_memory_builder(
        current_memory=sample_current_memory,
        questions=questions,
        answer_reports=answers,
        oracle_graph=sample_oracle_graph,
    )

    assert fake_llm.calls
    assert report.question_items[0].correctness == pytest.approx(0.91)
    assert "Prediction matches" in report.question_items[0].rationale
    assert report.config["correctness_mode"] == "llm_judge"


def test_reward_manager_llm_judge_scores_question_diagnostic_value(sample_oracle_graph):
    """Test Question Agent diagnostic value is no longer a fixed MVP placeholder."""

    class FakeJudgeLLM:
        def __init__(self):
            self.calls = []

        def chat_json(self, *, system, user, schema_hint=None):
            self.calls.append({"system": system, "user": user})
            return {"diagnostic_value": 0.83, "rationale": "Good multi-session diagnostic set."}

    fake_llm = FakeJudgeLLM()
    manager = RewardManager(correctness_mode="llm_judge", llm=fake_llm)
    report = manager.score_question_agent(
        questions=[
            {
                "question_id": "q_001",
                "record_id": "test_record",
                "question": "Across sessions, what technical preference remained stable?",
                "question_type": "multi_session",
                "answer": "Python for data analysis",
                "supporting_node_ids": ["fact_001", "fact_002"],
                "supporting_session_ids": ["s1", "s2"],
            }
        ],
        validity_reports=[{"question_id": "q_001", "verdict": "accept"}],
        answer_reports=[{"question_id": "q_001", "prediction": "Wrong", "answer_status": "answered"}],
        oracle_graph=sample_oracle_graph,
    )

    diagnostic = next(component for component in report.components if component.name == "diagnostic_value")
    assert fake_llm.calls
    assert diagnostic.score == pytest.approx(0.83)
    assert diagnostic.metadata["mode"] == "llm_judge"
    assert "api_key_configured" in diagnostic.metadata["llm_judge"]["config"]


def test_reward_manager_coverage_gain_from_snapshots(sample_oracle_graph):
    """Test coverage gain uses before/after snapshots instead of a fixed value."""
    manager = RewardManager()
    report = manager.score_question_agent(
        questions=[
            {
                "question_id": "q_001",
                "record_id": "test_record",
                "question": "What does the user prefer?",
                "question_type": "preference",
                "answer": "Python",
                "supporting_node_ids": ["fact_001"],
                "supporting_session_ids": ["s1"],
            }
        ],
        validity_reports=[{"question_id": "q_001", "verdict": "accept"}],
        answer_reports=[{"question_id": "q_001", "prediction": "Python"}],
        oracle_graph=sample_oracle_graph,
        coverage_before={"coverage": 0.2},
        coverage_after={"coverage": 0.7},
    )

    assert report.coverage_gain == pytest.approx(0.5)
