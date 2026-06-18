"""
Tests for QuestionAgent.

Run with: pytest gaam_memory_training/tests/test_question_agent.py
"""

import pytest

from gaam_graph.question_agent import QuestionAgent
from gaam_graph.question_schema import QuestionType


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    return {
        "graph": {
            "record_id": "test_record",
        },
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
                "text": "I like Python programming.",
            },
            {
                "id": "fact_01",
                "type": "fact",
                "text": "The user likes Python programming.",
                "label": "User prefers Python",
            },
        ],
        "edges": [
            {"source": "evt_01", "target": "sess_01", "type": "BELONGS_TO"},
            {"source": "evt_01", "target": "fact_01", "type": "EXTRACTED_AS"},
        ],
    }


def test_no_llm_agent_returns_deterministic_candidates(sample_graph):
    """Test that no-LLM agent returns deterministic candidates."""
    agent = QuestionAgent(no_llm=True)

    # Sample trajectories (mock)
    trajectories = [
        {
            "trajectory_id": "traj_001",
            "nodes": [
                {
                    "id": "fact_01",
                    "type": "fact",
                    "label": "User prefers Python",
                }
            ],
            "session_ids": ["sess_01"],
        }
    ]

    candidates = agent.generate_candidates(
        record_id="test_record",
        graph_path="/path/to/graph.json",
        trajectories=trajectories,
        num_questions=1,
    )

    assert len(candidates) == 1
    question = candidates[0]

    assert question.question_id is not None
    assert question.record_id == "test_record"
    assert question.question != ""
    assert question.answer != ""
    assert len(question.supporting_trajectory_ids) > 0
    assert len(question.supporting_node_ids) > 0


def test_all_candidates_have_ids(sample_graph):
    """Test that all generated candidates have IDs."""
    agent = QuestionAgent(no_llm=True)

    trajectories = [
        {
            "trajectory_id": f"traj_{i:03d}",
            "nodes": [
                {
                    "id": f"fact_{i:02d}",
                    "type": "fact",
                    "label": f"Fact {i}",
                }
            ],
            "session_ids": ["sess_01"],
        }
        for i in range(3)
    ]

    candidates = agent.generate_candidates(
        record_id="test_record",
        graph_path="/path/to/graph.json",
        trajectories=trajectories,
        num_questions=3,
    )

    assert len(candidates) == 3

    for candidate in candidates:
        assert candidate.question_id is not None
        assert candidate.question_id != ""


def test_supporting_node_ids_exist_in_trajectories():
    """Test that supporting node IDs exist in trajectories."""
    agent = QuestionAgent(no_llm=True)

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

    candidates = agent.generate_candidates(
        record_id="test_record",
        graph_path="/path/to/graph.json",
        trajectories=trajectories,
        num_questions=1,
    )

    assert len(candidates) == 1
    question = candidates[0]

    # Check that cited nodes exist in trajectories
    trajectory_node_ids = set()
    for traj in trajectories:
        for node in traj.get("nodes", []):
            trajectory_node_ids.add(node["id"])

    for node_id in question.supporting_node_ids:
        assert node_id in trajectory_node_ids


def test_select_best_evidence_node():
    """Test that best evidence node is selected correctly."""
    agent = QuestionAgent(no_llm=True)

    trajectory = {
        "nodes": [
            {"id": "evt_01", "type": "event", "label": "Event"},
            {"id": "fact_01", "type": "fact", "label": "Fact"},  # Should be selected
            {"id": "entity_01", "type": "entity", "label": "Entity"},
        ]
    }

    best = agent._select_best_evidence_node(trajectory)
    assert best is not None
    assert best["type"] == "fact"


def test_abstract_memory_preferred_over_event():
    """Test that abstract_memory is preferred over event."""
    agent = QuestionAgent(no_llm=True)

    trajectory = {
        "nodes": [
            {"id": "evt_01", "type": "event", "label": "Event"},
            {"id": "abs_01", "type": "abstract_memory", "label": "Abstract"},  # Should be selected
        ]
    }

    best = agent._select_best_evidence_node(trajectory)
    assert best is not None
    assert best["type"] == "abstract_memory"


def test_fact_preferred_over_abstract():
    """Test that fact is preferred over abstract_memory."""
    agent = QuestionAgent(no_llm=True)

    trajectory = {
        "nodes": [
            {"id": "abs_01", "type": "abstract_memory", "label": "Abstract"},
            {"id": "fact_01", "type": "fact", "label": "Fact"},  # Should be selected
        ]
    }

    best = agent._select_best_evidence_node(trajectory)
    assert best is not None
    assert best["type"] == "fact"
