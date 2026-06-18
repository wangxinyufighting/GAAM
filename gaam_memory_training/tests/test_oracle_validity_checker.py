"""
Tests for OracleValidityChecker.

Run with: pytest gaam_memory_training/tests/test_oracle_validity_checker.py
"""

import json
import tempfile
from pathlib import Path

import pytest

from gaam_graph.oracle_graph_loader import OracleGraphLoader
from gaam_graph.oracle_validity_checker import OracleValidityChecker
from gaam_graph.question_schema import GeneratedQuestion, OracleValidityVerdict


@pytest.fixture
def sample_graph_data():
    """Create sample graph data."""
    return {
        "graph": {"record_id": "test_record"},
        "nodes": [
            {
                "id": "sess_01",
                "type": "session",
                "session_id": "sess_01",
            },
            {
                "id": "evt_01",
                "type": "event",
                "session_id": "sess_01",
                "speaker": "user",
                "text": "I graduated with a degree in Business Administration.",
            },
            {
                "id": "fact_01",
                "type": "fact",
                "text": "The user graduated with a degree in Business Administration.",
                "label": "User has Business Administration degree",
            },
        ],
        "edges": [
            {"source": "evt_01", "target": "sess_01", "type": "BELONGS_TO"},
            {"source": "evt_01", "target": "fact_01", "type": "EXTRACTED_AS"},
        ],
    }


@pytest.fixture
def oracle_loader(sample_graph_data, tmp_path):
    """Create an OracleGraphLoader."""
    graph_path = tmp_path / "test.graph.json"
    graph_path.write_text(json.dumps(sample_graph_data))
    return OracleGraphLoader(graph_path, strict=True)


def test_accept_grounded_answer(oracle_loader):
    """Test that grounded answer with valid citations is accepted."""
    checker = OracleValidityChecker(oracle_loader)

    question = GeneratedQuestion(
        question_id="q_001",
        record_id="test_record",
        question="What degree does the user have?",
        answer="Business Administration",
        status="candidate",
        supporting_trajectory_ids=["traj_001"],
        supporting_node_ids=["fact_01"],
    )

    trajectories = [
        {
            "trajectory_id": "traj_001",
            "nodes": [
                {
                    "id": "fact_01",
                    "type": "fact",
                    "label": "User has Business Administration degree",
                }
            ],
            "session_ids": ["sess_01"],
        }
    ]

    report = checker.validate_question(question, trajectories=trajectories)

    assert report.verdict == OracleValidityVerdict.ACCEPT
    assert len(report.issues) == 0


def test_reject_missing_cited_trajectory(oracle_loader):
    """Test rejection when cited trajectory doesn't exist."""
    checker = OracleValidityChecker(oracle_loader)

    question = GeneratedQuestion(
        question_id="q_001",
        record_id="test_record",
        question="What degree does the user have?",
        answer="Business Administration",
        status="candidate",
        supporting_trajectory_ids=["traj_999"],  # Doesn't exist
        supporting_node_ids=["fact_01"],
    )

    trajectories = [
        {
            "trajectory_id": "traj_001",
            "nodes": [{"id": "fact_01", "type": "fact", "label": "Some fact"}],
        }
    ]

    report = checker.validate_question(question, trajectories=trajectories)

    assert report.verdict == OracleValidityVerdict.REJECT
    assert any("traj_999" in issue for issue in report.issues)


def test_reject_missing_cited_node(oracle_loader):
    """Test rejection when cited node doesn't exist."""
    checker = OracleValidityChecker(oracle_loader)

    question = GeneratedQuestion(
        question_id="q_001",
        record_id="test_record",
        question="What degree does the user have?",
        answer="Business Administration",
        status="candidate",
        supporting_trajectory_ids=["traj_001"],
        supporting_node_ids=["fact_999"],  # Doesn't exist
    )

    trajectories = [
        {
            "trajectory_id": "traj_001",
            "nodes": [{"id": "fact_01", "type": "fact", "label": "Some fact"}],
        }
    ]

    report = checker.validate_question(question, trajectories=trajectories)

    assert report.verdict == OracleValidityVerdict.REJECT
    assert any("fact_999" in issue for issue in report.issues)


def test_reject_answer_not_supported_by_evidence(oracle_loader):
    """Test rejection when answer is not supported by evidence."""
    checker = OracleValidityChecker(oracle_loader)

    question = GeneratedQuestion(
        question_id="q_001",
        record_id="test_record",
        question="What degree does the user have?",
        answer="Computer Science",  # Not in evidence
        status="candidate",
        supporting_trajectory_ids=["traj_001"],
        supporting_node_ids=["fact_01"],
    )

    trajectories = [
        {
            "trajectory_id": "traj_001",
            "nodes": [
                {
                    "id": "fact_01",
                    "type": "fact",
                    "label": "User has Business Administration degree",
                }
            ],
        }
    ]

    report = checker.validate_question(question, trajectories=trajectories)

    # Should be REVISE or REJECT due to grounding
    assert report.verdict in {OracleValidityVerdict.REVISE, OracleValidityVerdict.REJECT}
    assert report.grounding < 0.6


def test_reject_placeholder_question_by_default(oracle_loader):
    """Test that placeholder questions are rejected by default."""
    checker = OracleValidityChecker(oracle_loader, allow_placeholder_questions=False)

    question = GeneratedQuestion(
        question_id="q_001",
        record_id="test_record",
        question="What user information is supported by the cited evidence?",  # Placeholder
        answer="Some evidence",
        status="candidate",
        supporting_trajectory_ids=["traj_001"],
        supporting_node_ids=["fact_01"],
        reason="Generated in no-LLM mode for pipeline testing.",
    )

    trajectories = [
        {
            "trajectory_id": "traj_001",
            "nodes": [
                {
                    "id": "fact_01",
                    "type": "fact",
                    "label": "Some evidence",
                }
            ],
        }
    ]

    report = checker.validate_question(question, trajectories=trajectories)

    assert report.verdict == OracleValidityVerdict.REJECT
    assert any("placeholder" in issue.lower() for issue in report.issues)


def test_reject_question_artifact_leakage(oracle_loader):
    """Test oracle-validity hard gate rejects leaked benchmark metadata."""
    checker = OracleValidityChecker(oracle_loader)

    question = GeneratedQuestion(
        question_id="q_001",
        record_id="test_record",
        question="What degree does the user have?",
        answer="Business Administration",
        status="candidate",
        supporting_trajectory_ids=["traj_001"],
        supporting_node_ids=["fact_01"],
        metadata={"gold_answer": "Business Administration"},
    )

    trajectories = [
        {
            "trajectory_id": "traj_001",
            "nodes": [
                {
                    "id": "fact_01",
                    "type": "fact",
                    "label": "User has Business Administration degree",
                }
            ],
        }
    ]

    report = checker.validate_question(question, trajectories=trajectories)

    assert report.verdict == OracleValidityVerdict.REJECT
    assert report.leakage_risk == 1.0
    assert any("gold_answer" in issue for issue in report.issues)


def test_allow_placeholder_with_flag(oracle_loader):
    """Test that placeholder questions are allowed with flag."""
    checker = OracleValidityChecker(oracle_loader, allow_placeholder_questions=True)

    question = GeneratedQuestion(
        question_id="q_001",
        record_id="test_record",
        question="What user information is supported by the cited evidence?",
        answer="Some evidence",
        status="candidate",
        supporting_trajectory_ids=["traj_001"],
        supporting_node_ids=["fact_01"],
        metadata={"mode": "no_llm"},
    )

    trajectories = [
        {
            "trajectory_id": "traj_001",
            "nodes": [
                {
                    "id": "fact_01",
                    "type": "fact",
                    "label": "Some evidence",
                }
            ],
        }
    ]

    report = checker.validate_question(question, trajectories=trajectories)

    # Should not be rejected for being placeholder
    placeholder_issues = [i for i in report.issues if "placeholder" in i.lower()]
    assert len(placeholder_issues) == 0


def test_reject_single_session_when_multi_session_required(oracle_loader):
    """Test rejection when multi-session required but only one session cited."""
    checker = OracleValidityChecker(oracle_loader)

    question = GeneratedQuestion(
        question_id="q_001",
        record_id="test_record",
        question="What?",
        answer="Something",
        status="candidate",
        supporting_trajectory_ids=["traj_001"],
        supporting_node_ids=["fact_01"],
    )

    trajectories = [
        {
            "trajectory_id": "traj_001",
            "nodes": [
                {
                    "id": "fact_01",
                    "type": "fact",
                    "label": "Some fact",
                }
            ],
            "session_ids": ["sess_01"],
        }
    ]

    report = checker.validate_question(
        question,
        trajectories=trajectories,
        require_multi_session=True,
        min_sessions=2,
    )

    assert report.verdict == OracleValidityVerdict.REJECT
    assert any("multi-session" in issue.lower() for issue in report.issues)


def test_validate_multiple_questions(oracle_loader):
    """Test validating multiple questions at once."""
    checker = OracleValidityChecker(oracle_loader)

    questions = [
        GeneratedQuestion(
            question_id=f"q_{i:03d}",
            record_id="test_record",
            question="What?",
            answer="Business Administration",
            status="candidate",
            supporting_trajectory_ids=["traj_001"],
            supporting_node_ids=["fact_01"],
        )
        for i in range(3)
    ]

    trajectories = [
        {
            "trajectory_id": "traj_001",
            "nodes": [
                {
                    "id": "fact_01",
                    "type": "fact",
                    "label": "User has Business Administration degree",
                }
            ],
        }
    ]

    reports = checker.validate_questions(questions, trajectories=trajectories)

    assert len(reports) == 3
    for report in reports:
        assert report.question_id in {f"q_{i:03d}" for i in range(3)}
