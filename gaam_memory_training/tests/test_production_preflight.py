"""Tests for production E2E preflight checks."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from gaam_graph.production_preflight import ProductionPreflightConfig, run_production_preflight


ROOT_DIR = Path(__file__).resolve().parents[1]


def _write_model_dir(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def _write_case_and_graphs(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_path = tmp_path / "records.json"
    graph_dir = tmp_path / "graphs"
    split_path = tmp_path / "split.json"
    graph_dir.mkdir()
    records = [
        {"id": "case_train", "sessions": [{"session_id": "s1", "messages": []}]},
        {"id": "case_test", "sessions": [{"session_id": "s2", "messages": []}]},
    ]
    input_path.write_text(json.dumps(records), encoding="utf-8")
    for record_id in ["case_train", "case_test"]:
        (graph_dir / f"{record_id}.graph.json").write_text(
            json.dumps({"record_id": record_id, "nodes": [], "edges": []}),
            encoding="utf-8",
        )
    split_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "case_train",
                        "split": "train",
                        "oracle_graph_path": str(graph_dir / "case_train.graph.json"),
                    },
                    {
                        "record_id": "case_test",
                        "split": "test",
                        "oracle_graph_path": str(graph_dir / "case_test.graph.json"),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return input_path, graph_dir, split_path


def _base_config(tmp_path: Path) -> ProductionPreflightConfig:
    input_path, graph_dir, split_path = _write_case_and_graphs(tmp_path)
    memory_model = tmp_path / "memory_model"
    question_model = tmp_path / "question_model"
    _write_model_dir(memory_model)
    _write_model_dir(question_model)
    code_a1_root = tmp_path / "Code-A1"
    (code_a1_root / "verl" / "verl").mkdir(parents=True)
    (code_a1_root / "verl" / "trainer").mkdir(parents=True)
    (code_a1_root / "verl" / "trainer" / "main_ppo.py").write_text("# fake", encoding="utf-8")
    return ProductionPreflightConfig(
        input_path=input_path,
        oracle_graph_dir=graph_dir,
        split_manifest=split_path,
        memory_model_path=memory_model,
        question_model_path=question_model,
        code_a1_root=code_a1_root,
        training_output_dir=tmp_path / "train_out",
        eval_output_dir=tmp_path / "eval_out",
        answer_backend="no_llm",
        judge_backend="heuristic",
        require_cuda=False,
        check_imports=False,
        require_verl_import=False,
        train_batch_size=8,
        questions_per_case=4,
    )


def test_production_preflight_succeeds_for_minimal_real_split(tmp_path: Path):
    report = run_production_preflight(_base_config(tmp_path))

    assert report["status"] == "succeeded"
    assert report["checks"]["split"]["split_counts"]["train"] == 1
    assert report["checks"]["split"]["split_counts"]["test"] == 1
    assert report["checks"]["cuda"]["status"] == "skipped"
    assert any("train_batch_size=8 is larger" in warning for warning in report["warnings"])


def test_production_preflight_fails_when_api_evaluation_key_is_missing(tmp_path: Path):
    config = _base_config(tmp_path)
    config.answer_backend = "api"
    config.judge_backend = "api"

    report = run_production_preflight(config)

    assert report["status"] == "failed"
    assert any("ANSWER_BACKEND=api requires" in error for error in report["errors"])
    assert any("JUDGE_BACKEND=api requires" in error for error in report["errors"])


def test_production_preflight_fails_on_missing_oracle_graph(tmp_path: Path):
    config = _base_config(tmp_path)
    (config.oracle_graph_dir / "case_test.graph.json").unlink()

    report = run_production_preflight(config)

    assert report["status"] == "failed"
    assert report["checks"]["split"]["missing_graph_examples"] == ["case_test"]


def test_production_preflight_cli_outputs_json(tmp_path: Path):
    config = _base_config(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_production_e2e_preflight.py",
            "--input",
            str(config.input_path),
            "--oracle_graph_dir",
            str(config.oracle_graph_dir),
            "--split_manifest",
            str(config.split_manifest),
            "--memory_model_path",
            str(config.memory_model_path),
            "--question_model_path",
            str(config.question_model_path),
            "--code_a1_root",
            str(config.code_a1_root),
            "--training_output_dir",
            str(config.training_output_dir),
            "--eval_output_dir",
            str(config.eval_output_dir),
            "--answer_backend",
            "no_llm",
            "--judge_backend",
            "heuristic",
            "--require_cuda",
            "0",
            "--check_imports",
            "0",
            "--require_verl_import",
            "0",
        ],
        cwd=ROOT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "succeeded"
