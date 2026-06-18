"""
Tests for training loop.

Run with: pytest gaam_memory_training/tests/test_training_loop.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from gaam_graph.rollout_schema import StageStatus
from gaam_graph.training_loop import OneRoundTrainingConfig, OneRoundTrainingLoop
from gaam_graph.utils import read_json
from gaam_graph.utils import write_json


INPUT_PATH = Path("data/longmemeval/longmemeval_s_cleaned.json")
GRAPH_DIR = Path("outputs/no_leak_smoke_test")
RECORD_ID = "e47becba"


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_one_round_no_llm_smoke(temp_output_dir):
    """Test one round end-to-end with no-LLM mode."""
    config = OneRoundTrainingConfig(
        input_path=INPUT_PATH,
        graph_dir=GRAPH_DIR,
        output_dir=temp_output_dir,
        round_id=0,
        max_records=1,
        overwrite=True,
        no_llm_question=True,
        no_llm_answer=True,
        allow_placeholder_questions=True,
    )

    loop = OneRoundTrainingLoop(config)
    summary = loop.run()

    # Check summary
    assert summary.round_id == 0
    assert summary.attempted_records == 1
    assert summary.succeeded_records == 1
    assert summary.failed_records == 0
    assert summary.skipped_records == 0

    # Check manifest exists
    manifest_path = temp_output_dir / "manifest.jsonl"
    assert manifest_path.exists()

    # Check summary file
    summary_path = temp_output_dir / "summary.json"
    assert summary_path.exists()

    expected_paths = [
        temp_output_dir / "current_memory" / f"{RECORD_ID}.current_memory.json",
        temp_output_dir / "question_sets" / f"{RECORD_ID}.candidate_questions.json",
        temp_output_dir / "question_sets" / f"{RECORD_ID}.validity_reports.json",
        temp_output_dir / "question_sets" / f"{RECORD_ID}.accepted_questions.json",
        temp_output_dir / "answers" / f"{RECORD_ID}.answers.json",
        temp_output_dir / "rewards" / f"{RECORD_ID}.reward_report.json",
        temp_output_dir / "rewards" / f"{RECORD_ID}.question_agent_reward_report.json",
        temp_output_dir / "weakness_books" / f"{RECORD_ID}.weakness_book.json",
        temp_output_dir / "traces" / f"{RECORD_ID}.rollout_trace.json",
    ]
    for path in expected_paths:
        assert path.exists(), path

    accepted = read_json(temp_output_dir / "question_sets" / f"{RECORD_ID}.accepted_questions.json")
    assert "questions" in accepted
    assert "validity_reports" in accepted
    assert accepted["summary"]["accepted_count"] == len(accepted["questions"])

    trace = read_json(temp_output_dir / "traces" / f"{RECORD_ID}.rollout_trace.json")
    assert trace["status"] == "ok"
    assert trace["artifacts"]["rollout_trace_path"].endswith(f"{RECORD_ID}.rollout_trace.json")
    assert [stage["name"] for stage in trace["stages"]] == [
        "load_oracle_graph",
        "build_current_memory",
        "generate_questions",
        "oracle_validity",
        "answer_questions",
        "score_rewards",
        "write_trace",
    ]


def test_missing_graph_writes_failed_trace(temp_output_dir):
    """Test that missing graph writes failed trace."""
    graph_dir = temp_output_dir / "empty_graphs"
    graph_dir.mkdir()
    output_dir = temp_output_dir / "round"
    config = OneRoundTrainingConfig(
        input_path=INPUT_PATH,
        graph_dir=graph_dir,
        output_dir=output_dir,
        round_id=0,
        max_records=1,
        overwrite=True,
        no_llm_question=True,
        no_llm_answer=True,
    )

    loop = OneRoundTrainingLoop(config)
    summary = loop.run()

    assert summary.succeeded_records == 0
    assert summary.failed_records == 1
    trace_path = output_dir / "traces" / f"{RECORD_ID}.rollout_trace.json"
    assert trace_path.exists()
    trace = read_json(trace_path)
    assert trace["status"] == "failed"
    assert "Graph file not found" in trace["error"]
    assert trace["artifacts"]["rollout_trace_path"] == str(trace_path)


def test_record_id_consistency_is_enforced(temp_output_dir):
    """Test that record ID consistency is enforced."""
    graph_dir = temp_output_dir / "graphs"
    graph_dir.mkdir()
    graph = read_json(GRAPH_DIR / f"{RECORD_ID}.graph.json")
    graph["graph"]["record_id"] = "wrong_record_id"
    write_json(graph_dir / f"{RECORD_ID}.graph.json", graph)

    output_dir = temp_output_dir / "round"
    config = OneRoundTrainingConfig(
        input_path=INPUT_PATH,
        graph_dir=graph_dir,
        output_dir=output_dir,
        round_id=0,
        max_records=1,
        overwrite=True,
        no_llm_question=True,
        no_llm_answer=True,
        allow_placeholder_questions=True,
    )

    summary = OneRoundTrainingLoop(config).run()

    assert summary.succeeded_records == 0
    assert summary.failed_records == 1
    trace = read_json(output_dir / "traces" / f"{RECORD_ID}.rollout_trace.json")
    assert trace["status"] == "failed"
    assert "does not match record" in trace["error"]


def test_rollout_trace_contains_artifact_paths_and_metrics(temp_output_dir):
    """Test that rollout trace contains expected fields."""
    config = OneRoundTrainingConfig(
        input_path=INPUT_PATH,
        graph_dir=GRAPH_DIR,
        output_dir=temp_output_dir,
        round_id=0,
        max_records=1,
        overwrite=True,
        no_llm_question=True,
        no_llm_answer=True,
        allow_placeholder_questions=True,
    )

    loop = OneRoundTrainingLoop(config)
    summary = loop.run()

    assert summary.succeeded_records == 1
    trace = read_json(temp_output_dir / "traces" / f"{RECORD_ID}.rollout_trace.json")

    # Check required fields
    assert "round_id" in trace
    assert "record_id" in trace
    assert "status" in trace
    assert "config" in trace
    assert "artifacts" in trace
    assert "stages" in trace
    assert "metrics" in trace

    # Check artifact paths
    artifacts = trace["artifacts"]
    assert artifacts["graph_path"].endswith(f"{RECORD_ID}.graph.json")
    assert artifacts["current_memory_path"].endswith(f"{RECORD_ID}.current_memory.json")
    assert artifacts["accepted_questions_path"].endswith(f"{RECORD_ID}.accepted_questions.json")
    assert artifacts["answers_path"].endswith(f"{RECORD_ID}.answers.json")
    assert artifacts["reward_report_path"].endswith(f"{RECORD_ID}.reward_report.json")

    # Check metrics
    metrics = trace["metrics"]
    assert "memory_update_reward" in metrics
    assert "question_agent_total_reward" in metrics
    assert metrics["accepted_question_count"] > 0


def test_run_adversarial_memory_training_cli_no_llm(temp_output_dir):
    """Test the CLI can run from gaam_memory_training without external PYTHONPATH."""
    output_dir = temp_output_dir / "cli_round"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_adversarial_memory_training.py",
            "--input",
            str(INPUT_PATH),
            "--graph_dir",
            str(GRAPH_DIR),
            "--output_dir",
            str(output_dir),
            "--max_records",
            "1",
            "--round_id",
            "0",
            "--no_llm_question",
            "--no_llm_answer",
            "--allow_placeholder_questions",
            "--overwrite",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    summary = read_json(output_dir / "summary.json")
    assert summary["succeeded_records"] == 1
    assert (output_dir / "manifest.jsonl").exists()
