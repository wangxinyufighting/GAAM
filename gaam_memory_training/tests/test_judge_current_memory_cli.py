"""
Tests for judge_current_memory CLI.

Run with: pytest gaam_memory_training/tests/test_judge_current_memory_cli.py
"""

import json
import sys
from pathlib import Path

from gaam_graph.memory_schema import empty_current_memory
from gaam_graph.utils import write_json


def _write_cli_artifacts(
    tmp_path,
    *,
    graph_record_id="test_record",
    extra_answer=False,
    prediction="Python programming",
):
    """Write minimal CLI artifacts and return their paths."""
    # Create oracle graph
    graph_data = {
        "graph": {"record_id": graph_record_id},
        "nodes": [
            {
                "id": "fact_001",
                "type": "fact",
                "text": "The user likes Python programming.",
            }
        ],
        "edges": [],
    }
    graph_path = tmp_path / "test.graph.json"
    write_json(graph_path, graph_data)

    # Create current memory
    memory = empty_current_memory("test_record")
    memory["memory_graph"]["nodes"] = [
        {
            "id": "mem_001",
            "type": "fact",
            "content": "The user likes Python programming.",
            "status": "active",
            "source_session_ids": ["s1"],
        }
    ]
    memory_path = tmp_path / "test.current_memory.json"
    write_json(memory_path, memory)

    # Create questions
    questions = {
        "record_id": "test_record",
        "questions": [
            {
                "question_id": "q_001",
                "record_id": "test_record",
                "question": "What does the user like?",
                "answer": "Python programming",
                "question_type": "preference",
                "status": "accepted",
                "supporting_node_ids": ["fact_001"],
                "supporting_trajectory_ids": ["traj_001"],
                "supporting_session_ids": ["s1"],
            }
        ],
        "validity_reports": [
            {
                "question_id": "q_001",
                "record_id": "test_record",
                "verdict": "accept",
                "answerability": 1.0,
                "grounding": 1.0,
                "citation_validity": 1.0,
                "leakage_risk": 0.0,
                "issues": [],
                "supporting_node_ids": ["fact_001"],
                "supporting_trajectory_ids": ["traj_001"],
                "rationale": "test",
            }
        ],
    }
    questions_path = tmp_path / "test.questions.json"
    write_json(questions_path, questions)

    # Create answers
    answers = {
        "record_id": "test_record",
        "answers": [
            {
                "question_id": "q_001",
                "prediction": prediction,
                "answer_status": "answered",
                "supporting_memory_ids": ["mem_001"],
                "supporting_evidence": [
                    {
                        "memory_id": "mem_001",
                        "memory_type": "fact",
                        "content": "The user likes Python programming.",
                        "score": 0.9,
                    }
                ],
            }
        ],
    }
    if extra_answer:
        answers["answers"].append(
            {
                "question_id": "q_extra",
                "prediction": "extra",
                "answer_status": "answered",
                "supporting_memory_ids": ["mem_001"],
                "supporting_evidence": [],
            }
        )
    answers_path = tmp_path / "test.answers.json"
    write_json(answers_path, answers)

    return graph_path, memory_path, questions_path, answers_path


def test_cli_smoke(tmp_path):
    """Test CLI smoke path with minimal inputs."""
    graph_path, memory_path, questions_path, answers_path = _write_cli_artifacts(tmp_path)

    # Run CLI
    import subprocess

    project_root = Path(__file__).parent.parent.parent
    builder_root = project_root / "gaam_memory_training"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/judge_current_memory.py",
            "--graph", str(graph_path),
            "--current_memory", str(memory_path),
            "--questions", str(questions_path),
            "--answers", str(answers_path),
            "--output_dir", str(tmp_path / "outputs"),
            "--write_question_agent_reward",
            "--overwrite",
        ],
        cwd=str(builder_root),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0

    # Check outputs exist
    output_dir = tmp_path / "outputs"
    assert (output_dir / "test_record.reward_report.json").exists()
    assert (output_dir / "test_record.question_agent_reward_report.json").exists()
    assert (output_dir / "test_record.weakness_book.json").exists()
    assert (output_dir / "manifest.jsonl").exists()
    assert (output_dir / "summary.json").exists()

    reward_report = json.loads((output_dir / "test_record.reward_report.json").read_text())
    assert reward_report["weakness_updates"] == []


def test_cli_rejects_graph_record_id_mismatch(tmp_path):
    """Test CLI fails when oracle graph record_id differs from memory."""
    graph_path, memory_path, questions_path, answers_path = _write_cli_artifacts(
        tmp_path,
        graph_record_id="wrong_record",
    )

    import subprocess

    builder_root = Path(__file__).parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "scripts/judge_current_memory.py",
            "--graph", str(graph_path),
            "--current_memory", str(memory_path),
            "--questions", str(questions_path),
            "--answers", str(answers_path),
            "--output_dir", str(tmp_path / "outputs"),
            "--overwrite",
        ],
        cwd=str(builder_root),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "graph" in result.stderr


def test_cli_rejects_extra_answer_ids(tmp_path):
    """Test CLI fails when answers contain IDs not present in questions."""
    graph_path, memory_path, questions_path, answers_path = _write_cli_artifacts(
        tmp_path,
        extra_answer=True,
    )

    import subprocess

    builder_root = Path(__file__).parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "scripts/judge_current_memory.py",
            "--graph", str(graph_path),
            "--current_memory", str(memory_path),
            "--questions", str(questions_path),
            "--answers", str(answers_path),
            "--output_dir", str(tmp_path / "outputs"),
            "--overwrite",
        ],
        cwd=str(builder_root),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Extra answers" in result.stderr


def test_cli_writes_weakness_updates_into_reward_report(tmp_path):
    """Test reward report includes weakness deltas after book update."""
    graph_path, memory_path, questions_path, answers_path = _write_cli_artifacts(
        tmp_path,
        prediction="JavaScript",
    )

    import subprocess

    builder_root = Path(__file__).parent.parent
    result = subprocess.run(
        [
            sys.executable,
            "scripts/judge_current_memory.py",
            "--graph", str(graph_path),
            "--current_memory", str(memory_path),
            "--questions", str(questions_path),
            "--answers", str(answers_path),
            "--output_dir", str(tmp_path / "outputs"),
            "--overwrite",
        ],
        cwd=str(builder_root),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    reward_report = json.loads(
        (tmp_path / "outputs" / "test_record.reward_report.json").read_text()
    )
    assert reward_report["weakness_updates"]
