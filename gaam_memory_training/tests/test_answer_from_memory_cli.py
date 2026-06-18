"""
CLI smoke tests for answer_from_memory.py.

Run with: pytest gaam_memory_training/tests/test_answer_from_memory_cli.py
"""

import json
import os
from pathlib import Path
import subprocess
import sys


def _write_current_memory(path: Path) -> None:
    memory = {
        "record_id": "record_answer_cli_001",
        "build_step": 0,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_fact_001",
                    "type": "fact",
                    "content": "The user prefers Python for memory experiments.",
                    "status": "active",
                    "confidence": 0.9,
                    "source_session_ids": ["sess_01"],
                    "source_turn_ids": ["1"],
                    "source_event_ids": ["evt_001"],
                }
            ],
            "edges": [],
        },
        "memory_summaries": {
            "user_profile": "The user works on memory experiments.",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": "",
        },
        "metadata": {},
    }
    path.write_text(json.dumps(memory), encoding="utf-8")


def _write_question_set(path: Path) -> None:
    question_set = {
        "record_id": "record_answer_cli_001",
        "graph_path": "oracle.graph.json",
        "questions": [
            {
                "question_id": "q_001",
                "record_id": "record_answer_cli_001",
                "question": "What does the user prefer for memory experiments?",
                "answer": "Python",
                "question_type": "preference",
                "status": "accepted",
                "supporting_trajectory_ids": ["traj_001"],
                "supporting_node_ids": ["fact_oracle_001"],
                "supporting_session_ids": ["sess_01"],
                "reason": "Test question",
                "metadata": {"mode": "test"},
            }
        ],
        "validity_reports": [],
    }
    path.write_text(json.dumps(question_set), encoding="utf-8")


def test_answer_from_memory_cli_no_llm_question_file_smoke(tmp_path):
    """Test no-LLM CLI writes answer output for an accepted question file."""
    repo_root = Path(__file__).resolve().parents[1]
    memory_path = tmp_path / "record.current_memory.json"
    question_path = tmp_path / "record.accepted_questions.json"
    output_path = tmp_path / "record.answers.json"
    _write_current_memory(memory_path)
    _write_question_set(question_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/answer_from_memory.py",
            "--current_memory",
            str(memory_path),
            "--question_file",
            str(question_path),
            "--output",
            str(output_path),
            "--no_llm",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["summary"]["num_questions"] == 1
    assert output["answers"][0]["question_id"] == "q_001"
    assert output["answers"][0]["supporting_memory_ids"] == ["mem_fact_001", "summary_user_profile"]
    assert "answer" not in output["answers"][0]


def test_answer_from_memory_cli_rejects_wrong_question_file_shape(tmp_path):
    """Test CLI fails instead of silently answering zero questions."""
    repo_root = Path(__file__).resolve().parents[1]
    memory_path = tmp_path / "record.current_memory.json"
    wrong_question_path = tmp_path / "record.validity_reports.json"
    _write_current_memory(memory_path)
    wrong_question_path.write_text(json.dumps({"validity_reports": []}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/answer_from_memory.py",
            "--current_memory",
            str(memory_path),
            "--question_file",
            str(wrong_question_path),
            "--output",
            str(tmp_path / "answers.json"),
            "--no_llm",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "questions" in result.stderr


def test_answer_from_memory_cli_default_llm_without_key_fails_cleanly(tmp_path):
    """Test default LLM mode creates a client and fails cleanly without API keys."""
    repo_root = Path(__file__).resolve().parents[1]
    memory_path = tmp_path / "record.current_memory.json"
    output_path = tmp_path / "record.answers.json"
    _write_current_memory(memory_path)

    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/answer_from_memory.py",
            "--current_memory",
            str(memory_path),
            "--question",
            "What does the user prefer for memory experiments?",
            "--question_id",
            "manual_q_001",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "API_KEY" in result.stderr
    assert "Traceback" not in result.stderr
