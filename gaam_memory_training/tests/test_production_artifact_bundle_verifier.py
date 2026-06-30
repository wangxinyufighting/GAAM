"""Tests for production artifact bundle verification."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from gaam_graph.production_artifact_bundle import (
    ProductionArtifactBundleConfig,
    export_production_artifact_bundle,
)
from gaam_graph.production_artifact_bundle_verifier import (
    ProductionArtifactBundleVerificationConfig,
    verify_production_artifact_bundle,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def _write_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    training_dir = tmp_path / "training"
    eval_dir = tmp_path / "eval"
    split_path = tmp_path / "split.json"
    training_dir.mkdir()
    eval_dir.mkdir()
    (training_dir / "dual_cotraining_manifest.json").write_text(
        json.dumps({"status": "succeeded"}),
        encoding="utf-8",
    )
    (training_dir / "native_verl_training_verification.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    (eval_dir / "case_evaluation_manifest.json").write_text(
        json.dumps({"status": "succeeded"}),
        encoding="utf-8",
    )
    (eval_dir / "case_evaluation_verification.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    (eval_dir / "evaluation_summary.json").write_text(
        json.dumps({"status": "succeeded"}),
        encoding="utf-8",
    )
    (eval_dir / "production_run_audit.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    split_path.write_text(json.dumps({"records": []}), encoding="utf-8")
    return training_dir, eval_dir, split_path


def test_verify_production_artifact_bundle_passes_clean_bundle(tmp_path: Path):
    training_dir, eval_dir, split_path = _write_evidence(tmp_path)
    bundle_dir = tmp_path / "bundle"
    export_production_artifact_bundle(
        ProductionArtifactBundleConfig(
            training_output_dir=training_dir,
            evaluation_output_dir=eval_dir,
            split_manifest_path=split_path,
            bundle_output_dir=bundle_dir,
        )
    )

    report = verify_production_artifact_bundle(
        ProductionArtifactBundleVerificationConfig(bundle_output_dir=bundle_dir)
    )

    assert report["status"] == "passed"
    assert report["metrics"]["num_missing_labels"] == 0
    assert (bundle_dir / "production_artifact_bundle_verification.json").exists()


def test_verify_production_artifact_bundle_fails_on_missing_required_file(tmp_path: Path):
    training_dir, eval_dir, split_path = _write_evidence(tmp_path)
    bundle_dir = tmp_path / "bundle"
    export_production_artifact_bundle(
        ProductionArtifactBundleConfig(
            training_output_dir=training_dir,
            evaluation_output_dir=eval_dir,
            split_manifest_path=split_path,
            bundle_output_dir=bundle_dir,
            create_tar=False,
        )
    )
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        item for item in manifest["files"]
        if item["label"] != "evaluation/evaluation_summary.json"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_production_artifact_bundle(
        ProductionArtifactBundleVerificationConfig(
            bundle_output_dir=bundle_dir,
            require_tar=False,
        )
    )

    assert report["status"] == "failed"
    assert any("missing required labels" in error.lower() for error in report["errors"])


def test_verify_production_artifact_bundle_cli_outputs_json(tmp_path: Path):
    training_dir, eval_dir, split_path = _write_evidence(tmp_path)
    bundle_dir = tmp_path / "bundle_cli"
    export_production_artifact_bundle(
        ProductionArtifactBundleConfig(
            training_output_dir=training_dir,
            evaluation_output_dir=eval_dir,
            split_manifest_path=split_path,
            bundle_output_dir=bundle_dir,
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_production_artifact_bundle.py",
            "--bundle_output_dir",
            str(bundle_dir),
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
