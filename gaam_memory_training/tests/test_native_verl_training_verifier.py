"""Tests for native VERL training output verification."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from gaam_graph.native_verl_training_verifier import (
    resolve_native_verl_checkpoint_paths,
    verify_native_verl_training_output,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_successful_run(output_dir: Path) -> Path:
    hf_dir = output_dir / "rounds/round_000/question_agent/checkpoints/global_step_1/huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")

    dataset_dir = output_dir / "rounds/round_000/question_agent/dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs/round_000.question_agent.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo "
        "actor_rollout_ref.model.lora_rank=0\n",
        encoding="utf-8",
    )

    manifest = {
        "manifest_version": "gaam_native_verl_dual_cotraining_v1",
        "trainer": "GAAMDualRayTrainer",
        "status": "succeeded",
        "config": {
            "dry_run": False,
            "save_hf_model": True,
            "require_hf_checkpoint": True,
        },
        "round_records": [
            {
                "round_index": 0,
                "status": "succeeded",
                "actor_records": [
                    {
                        "round_index": 0,
                        "actor_role": "question_agent",
                        "status": "succeeded",
                        "dry_run": False,
                        "returncode": 0,
                        "dataset_dir": str(dataset_dir),
                        "checkpoint_dir": str(output_dir / "rounds/round_000/question_agent/checkpoints"),
                        "log_path": str(log_path),
                        "latest_hf_model_path": str(hf_dir),
                    }
                ],
            }
        ],
        "final_model_paths": {"question_agent": str(hf_dir)},
    }
    _write_json(output_dir / "dual_cotraining_manifest.json", manifest)
    return hf_dir


def test_native_verl_training_verifier_passes_real_checkpoint_shape(tmp_path: Path):
    output_dir = tmp_path / "run"
    _make_successful_run(output_dir)

    report = verify_native_verl_training_output(output_dir)

    assert report.status == "passed"
    assert report.metrics["num_actor_steps"] == 1
    assert report.metrics["num_actor_steps_with_hf_checkpoint"] == 1
    assert report.issues == []


def test_native_verl_training_verifier_fails_missing_checkpoint(tmp_path: Path):
    output_dir = tmp_path / "run"
    _make_successful_run(output_dir)
    manifest_path = output_dir / "dual_cotraining_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["round_records"][0]["actor_records"][0]["latest_hf_model_path"] = None
    manifest["final_model_paths"] = {"question_agent": str(output_dir / "missing_model")}
    _write_json(manifest_path, manifest)

    report = verify_native_verl_training_output(output_dir)

    assert report.status == "failed"
    assert any("latest_hf_model_path is missing" in issue for issue in report.issues)
    assert any("Final model path" in issue for issue in report.issues)


def test_resolve_native_verl_checkpoint_paths(tmp_path: Path):
    output_dir = tmp_path / "run"
    hf_dir = _make_successful_run(output_dir)
    manifest_path = output_dir / "dual_cotraining_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_model_paths"]["memory_builder"] = str(hf_dir)
    _write_json(manifest_path, manifest)

    resolved = resolve_native_verl_checkpoint_paths(output_dir)

    assert resolved["memory_builder_checkpoint"] == str(hf_dir)
    assert resolved["question_agent_checkpoint"] == str(hf_dir)
    assert resolved["status"] == "succeeded"


def test_resolve_native_verl_checkpoints_cli_shell_output(tmp_path: Path):
    output_dir = tmp_path / "run"
    hf_dir = _make_successful_run(output_dir)
    manifest_path = output_dir / "dual_cotraining_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_model_paths"]["memory_builder"] = str(hf_dir)
    _write_json(manifest_path, manifest)

    result = subprocess.run(
        [
            "python",
            "scripts/resolve_native_verl_checkpoints.py",
            "--output_dir",
            str(output_dir),
            "--format",
            "shell",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert f"MEMORY_MODEL={hf_dir}" in result.stdout
    assert f"MEMORY_MODEL_PATH={hf_dir}" in result.stdout
    assert f"QUESTION_MODEL_PATH={hf_dir}" in result.stdout
