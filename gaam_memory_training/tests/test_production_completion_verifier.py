"""Tests for final production E2E completion verification."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from gaam_graph.production_completion_verifier import (
    ProductionCompletionVerificationConfig,
    verify_production_e2e_completion,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def _write_complete_evidence(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    training_dir = tmp_path / "training"
    eval_dir = tmp_path / "eval"
    bundle_dir = tmp_path / "bundle"
    split_path = tmp_path / "split.json"
    model_dir = tmp_path / "model"
    training_dir.mkdir()
    eval_dir.mkdir()
    bundle_dir.mkdir()
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    split_path.write_text(json.dumps({"records": []}), encoding="utf-8")

    (training_dir / "dual_cotraining_manifest.json").write_text(
        json.dumps(
            {
                "trainer": "GAAMDualRayTrainer",
                "status": "succeeded",
                "config": {"dry_run": False},
                "final_model_paths": {
                    "memory_builder": str(model_dir),
                    "question_agent": str(model_dir),
                },
            }
        ),
        encoding="utf-8",
    )
    (training_dir / "native_verl_training_verification.json").write_text(
        json.dumps({"status": "passed", "metrics": {"num_actor_steps": 2}, "warnings": []}),
        encoding="utf-8",
    )
    (eval_dir / "case_evaluation_manifest.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "split_manifest_path": str(split_path),
                "evaluation_split": "test",
                "num_records": 1,
                "num_judged": 1,
                "accuracy": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "case_evaluation_verification.json").write_text(
        json.dumps({"status": "passed", "warnings": []}),
        encoding="utf-8",
    )
    (eval_dir / "evaluation_summary.json").write_text(
        json.dumps({"status": "succeeded", "accuracy": 1.0}),
        encoding="utf-8",
    )
    (eval_dir / "production_run_audit.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "training_output_dir": str(training_dir),
                "expected_evaluation_split": "test",
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    tar_path = tmp_path / "bundle.tar.gz"
    tar_path.write_text("tar", encoding="utf-8")
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps({"status": "succeeded", "tar_path": str(tar_path)}),
        encoding="utf-8",
    )
    (bundle_dir / "production_artifact_bundle_verification.json").write_text(
        json.dumps({"status": "passed", "warnings": []}),
        encoding="utf-8",
    )
    return training_dir, eval_dir, bundle_dir, split_path


def test_verify_production_e2e_completion_passes_clean_evidence(tmp_path: Path):
    training_dir, eval_dir, bundle_dir, split_path = _write_complete_evidence(tmp_path)

    report = verify_production_e2e_completion(
        ProductionCompletionVerificationConfig(
            training_output_dir=training_dir,
            evaluation_output_dir=eval_dir,
            artifact_bundle_dir=bundle_dir,
            split_manifest_path=split_path,
        )
    )

    assert report["status"] == "passed"
    assert report["checks"]["training"]["verification_status"] == "passed"
    assert report["checks"]["evaluation"]["accuracy"] == 1.0
    assert (eval_dir / "production_e2e_completion_verification.json").exists()


def test_verify_production_e2e_completion_fails_on_dry_run(tmp_path: Path):
    training_dir, eval_dir, bundle_dir, split_path = _write_complete_evidence(tmp_path)
    manifest_path = training_dir / "dual_cotraining_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["dry_run"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_production_e2e_completion(
        ProductionCompletionVerificationConfig(
            training_output_dir=training_dir,
            evaluation_output_dir=eval_dir,
            artifact_bundle_dir=bundle_dir,
            split_manifest_path=split_path,
        )
    )

    assert report["status"] == "failed"
    assert any("dry_run" in error for error in report["errors"])


def test_verify_production_e2e_completion_cli_outputs_json(tmp_path: Path):
    training_dir, eval_dir, bundle_dir, split_path = _write_complete_evidence(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_production_e2e_completion.py",
            "--training_output_dir",
            str(training_dir),
            "--evaluation_output_dir",
            str(eval_dir),
            "--artifact_bundle_dir",
            str(bundle_dir),
            "--split_manifest",
            str(split_path),
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
    assert report["status"] == "passed"
