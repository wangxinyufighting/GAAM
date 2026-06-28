"""Tests for compact production artifact bundle export."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tarfile

from gaam_graph.production_artifact_bundle import (
    ProductionArtifactBundleConfig,
    export_production_artifact_bundle,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    training_dir = tmp_path / "training"
    eval_dir = tmp_path / "eval"
    split_path = tmp_path / "split.json"
    training_dir.mkdir()
    eval_dir.mkdir()
    (training_dir / "dual_cotraining_manifest.json").write_text(
        json.dumps({"status": "succeeded", "final_model_paths": {"memory_builder": "m", "question_agent": "q"}}),
        encoding="utf-8",
    )
    (training_dir / "native_verl_training_verification.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    (eval_dir / "case_evaluation_manifest.json").write_text(
        json.dumps({"status": "succeeded", "accuracy": 1.0}),
        encoding="utf-8",
    )
    (eval_dir / "case_evaluation_verification.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    (eval_dir / "evaluation_summary.json").write_text(
        json.dumps({"status": "succeeded", "accuracy": 1.0}),
        encoding="utf-8",
    )
    (eval_dir / "production_run_audit.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    split_path.write_text(json.dumps({"records": []}), encoding="utf-8")
    return training_dir, eval_dir, split_path


def test_export_production_artifact_bundle_copies_evidence_and_redacts_env(tmp_path: Path):
    training_dir, eval_dir, split_path = _write_artifacts(tmp_path)
    bundle_dir = tmp_path / "bundle"

    report = export_production_artifact_bundle(
        ProductionArtifactBundleConfig(
            training_output_dir=training_dir,
            evaluation_output_dir=eval_dir,
            split_manifest_path=split_path,
            bundle_output_dir=bundle_dir,
            run_id="test_run",
            env_snapshot={"DEEPSEEK_API_KEY": "secret", "ROUNDS": "1"},
        )
    )

    assert report["status"] == "succeeded"
    assert report["run_id"] == "test_run"
    assert report["env_snapshot"]["DEEPSEEK_API_KEY"] == "<redacted>"
    assert report["env_snapshot"]["ROUNDS"] == "1"
    assert (bundle_dir / "training" / "dual_cotraining_manifest.json").exists()
    assert (bundle_dir / "evaluation" / "production_run_audit.json").exists()
    assert (bundle_dir / "split" / "split_manifest.json").exists()
    assert (bundle_dir / "bundle_manifest.json").exists()
    assert Path(report["tar_path"]).exists()
    with tarfile.open(report["tar_path"], "r:gz") as tar:
        names = set(tar.getnames())
    assert any(name.endswith("bundle_manifest.json") for name in names)
    assert all("secret" not in json.dumps(item, ensure_ascii=False) for item in report["files"])


def test_export_production_artifact_bundle_cli_outputs_json(tmp_path: Path):
    training_dir, eval_dir, split_path = _write_artifacts(tmp_path)
    bundle_dir = tmp_path / "bundle_cli"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_production_artifact_bundle.py",
            "--training_output_dir",
            str(training_dir),
            "--evaluation_output_dir",
            str(eval_dir),
            "--split_manifest",
            str(split_path),
            "--bundle_output_dir",
            str(bundle_dir),
            "--run_id",
            "cli_run",
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
    assert report["run_id"] == "cli_run"
    assert (bundle_dir / "bundle_manifest.json").exists()
