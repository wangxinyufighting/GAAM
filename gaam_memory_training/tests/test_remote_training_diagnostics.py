"""Tests for remote training diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from gaam_graph.production_preflight import ProductionPreflightConfig
from gaam_graph.remote_training_diagnostics import (
    RemoteTrainingDiagnosticsConfig,
    collect_remote_training_diagnostics,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def _write_model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def _write_minimal_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    input_path = tmp_path / "records.json"
    graph_dir = tmp_path / "graphs"
    split_path = tmp_path / "split.json"
    memory_model = tmp_path / "memory_model"
    question_model = tmp_path / "question_model"
    code_a1_root = tmp_path / "Code-A1"
    graph_dir.mkdir()
    input_path.write_text(json.dumps([{"id": "case_train", "sessions": []}, {"id": "case_test", "sessions": []}]), encoding="utf-8")
    for record_id in ["case_train", "case_test"]:
        (graph_dir / f"{record_id}.graph.json").write_text(json.dumps({"record_id": record_id}), encoding="utf-8")
    split_path.write_text(
        json.dumps(
            {
                "records": [
                    {"record_id": "case_train", "split": "train", "oracle_graph_path": str(graph_dir / "case_train.graph.json")},
                    {"record_id": "case_test", "split": "test", "oracle_graph_path": str(graph_dir / "case_test.graph.json")},
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_model(memory_model)
    _write_model(question_model)
    (code_a1_root / "verl" / "verl" / "trainer").mkdir(parents=True)
    (code_a1_root / "verl" / "verl" / "trainer" / "main_ppo.py").write_text("# fake", encoding="utf-8")
    return input_path, graph_dir, split_path, memory_model, question_model, code_a1_root


def _config(tmp_path: Path) -> RemoteTrainingDiagnosticsConfig:
    input_path, graph_dir, split_path, memory_model, question_model, code_a1_root = _write_minimal_inputs(tmp_path)
    return RemoteTrainingDiagnosticsConfig(
        preflight_config=ProductionPreflightConfig(
            input_path=input_path,
            oracle_graph_dir=graph_dir,
            split_manifest=split_path,
            memory_model_path=memory_model,
            question_model_path=question_model,
            code_a1_root=code_a1_root,
            training_output_dir=tmp_path / "training",
            eval_output_dir=tmp_path / "eval",
            answer_backend="no_llm",
            judge_backend="heuristic",
            require_cuda=False,
            check_imports=False,
            require_verl_import=False,
        ),
        output_path=tmp_path / "diagnostics.json",
        include_nvidia_smi=False,
        env_keys=["DEEPSEEK_API_KEY", "ROUNDS"],
    )


def test_collect_remote_training_diagnostics_writes_report_and_redacts_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("ROUNDS", "1")

    report = collect_remote_training_diagnostics(_config(tmp_path))

    assert report["status"] == "succeeded"
    assert report["preflight"]["status"] == "succeeded"
    assert report["env"]["DEEPSEEK_API_KEY"] == "<redacted>"
    assert report["env"]["ROUNDS"] == "1"
    assert report["nvidia_smi"]["status"] == "skipped"
    assert (tmp_path / "diagnostics.json").exists()


def test_collect_remote_training_diagnostics_cli_outputs_json(tmp_path: Path):
    input_path, graph_dir, split_path, memory_model, question_model, code_a1_root = _write_minimal_inputs(tmp_path)
    output_path = tmp_path / "cli_diagnostics.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_remote_training_diagnostics.py",
            "--input",
            str(input_path),
            "--oracle_graph_dir",
            str(graph_dir),
            "--split_manifest",
            str(split_path),
            "--memory_model_path",
            str(memory_model),
            "--question_model_path",
            str(question_model),
            "--code_a1_root",
            str(code_a1_root),
            "--training_output_dir",
            str(tmp_path / "training"),
            "--eval_output_dir",
            str(tmp_path / "eval"),
            "--output",
            str(output_path),
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
            "--skip_nvidia_smi",
            "--json",
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
    assert output_path.exists()


def test_collect_remote_training_diagnostics_cli_falls_back_to_deepseek_key(
    tmp_path: Path,
    monkeypatch,
):
    input_path, graph_dir, split_path, memory_model, question_model, code_a1_root = _write_minimal_inputs(tmp_path)
    output_path = tmp_path / "cli_diagnostics_api.json"
    monkeypatch.setenv("MEMORY_API_KEY", "")
    monkeypatch.setenv("GAAM_MEMORY_BUILDER_API_KEY", "")
    monkeypatch.setenv("ANSWER_API_KEY", "")
    monkeypatch.setenv("GAAM_ANSWERER_API_KEY", "")
    monkeypatch.setenv("JUDGE_API_KEY", "")
    monkeypatch.setenv("GAAM_EVAL_JUDGE_API_KEY", "")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_API_KEY", "")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_ENABLED", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-fallback-key")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_remote_training_diagnostics.py",
            "--input",
            str(input_path),
            "--oracle_graph_dir",
            str(graph_dir),
            "--split_manifest",
            str(split_path),
            "--memory_model_path",
            str(memory_model),
            "--question_model_path",
            str(question_model),
            "--code_a1_root",
            str(code_a1_root),
            "--training_output_dir",
            str(tmp_path / "training"),
            "--eval_output_dir",
            str(tmp_path / "eval"),
            "--output",
            str(output_path),
            "--memory_backend",
            "stateful_api",
            "--answer_backend",
            "api",
            "--judge_backend",
            "api",
            "--require_cuda",
            "0",
            "--check_imports",
            "0",
            "--require_verl_import",
            "0",
            "--skip_nvidia_smi",
            "--json",
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
    configured = report["preflight"]["checks"]["api_keys"]["api_key_configured"]
    assert configured["reward_judge"] is True
    assert configured["memory"] is True
    assert configured["answer"] is True
    assert configured["judge"] is True
    assert report["env"]["DEEPSEEK_API_KEY"] == "<redacted>"
