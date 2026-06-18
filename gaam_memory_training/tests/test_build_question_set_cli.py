"""
CLI smoke tests for build_question_set.py.

Run with: pytest gaam_memory_training/tests/test_build_question_set_cli.py
"""

import json
import os
from pathlib import Path
import subprocess
import sys

from gaam_graph.question_schema import validate_question_set


def _write_graph(path: Path) -> None:
    graph = {
        "graph": {"record_id": "record_cli_001"},
        "nodes": [
            {
                "id": "evt_001",
                "type": "event",
                "session_id": "sess_01",
                "speaker": "user",
                "text": "I prefer Python for memory research experiments.",
            },
            {
                "id": "fact_001",
                "type": "fact",
                "text": "The user prefers Python for memory research experiments.",
                "source_event_id": "evt_001",
            },
        ],
        "edges": [
            {"source": "evt_001", "target": "fact_001", "type": "EXTRACTED_AS"},
        ],
    }
    path.write_text(json.dumps(graph), encoding="utf-8")


def test_build_question_set_cli_smoke(tmp_path):
    """Test no-LLM CLI writes valid accepted question artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    graph_path = tmp_path / "record_cli_001.graph.json"
    output_dir = tmp_path / "question_sets"
    _write_graph(graph_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_question_set.py",
            "--graph",
            str(graph_path),
            "--output_dir",
            str(output_dir),
            "--num_walks",
            "1",
            "--walk_length",
            "2",
            "--num_questions",
            "1",
            "--no_llm",
            "--allow_placeholder_questions",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    accepted_path = output_dir / "record_cli_001.accepted_questions.json"
    assert accepted_path.exists()

    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    validate_question_set(
        {
            "record_id": accepted["record_id"],
            "graph_path": accepted["graph_path"],
            "questions": accepted["questions"],
            "validity_reports": accepted["validity_reports"],
        },
        strict=True,
    )
    assert accepted["summary"]["accepted_count"] == 1
    assert accepted["questions"][0]["status"] == "accepted"


def test_build_question_set_cli_returns_nonzero_on_missing_graph(tmp_path):
    """Test CLI returns non-zero when a graph cannot be processed."""
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_question_set.py",
            "--graph",
            str(tmp_path / "missing.graph.json"),
            "--output_dir",
            str(tmp_path / "question_sets"),
            "--num_walks",
            "1",
            "--num_questions",
            "1",
            "--no_llm",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1


def test_build_question_set_cli_default_llm_without_key_fails(tmp_path):
    """Test default LLM mode fails loudly when no API key is configured."""
    repo_root = Path(__file__).resolve().parents[1]
    graph_path = tmp_path / "record_cli_001.graph.json"
    _write_graph(graph_path)

    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_question_set.py",
            "--graph",
            str(graph_path),
            "--output_dir",
            str(tmp_path / "question_sets"),
            "--num_walks",
            "1",
            "--walk_length",
            "2",
            "--num_questions",
            "1",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "API_KEY" in result.stderr
