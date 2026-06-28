"""Tests for production E2E run audit."""

from __future__ import annotations

import json
from pathlib import Path

from gaam_graph.production_run_audit import ProductionRunAuditConfig, audit_production_run


def _write_training_artifacts(training_dir: Path, model_dir: Path) -> None:
    training_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    manifest = {
        "manifest_version": "gaam_native_verl_dual_cotraining_v1",
        "trainer": "GAAMDualRayTrainer",
        "status": "succeeded",
        "round_records": [{"round_index": 0, "status": "succeeded", "actor_records": []}],
        "final_model_paths": {
            "memory_builder": str(model_dir),
            "question_agent": str(model_dir),
        },
    }
    verification = {
        "status": "passed",
        "metrics": {"num_actor_steps": 2},
        "warnings": [],
    }
    (training_dir / "dual_cotraining_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    (training_dir / "native_verl_training_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_evaluation_artifacts(eval_dir: Path, split_path: Path, *, evaluation_split: str = "test") -> None:
    eval_dir.mkdir(parents=True)
    manifest = {
        "manifest_version": "gaam_case_evaluation_v1",
        "status": "succeeded",
        "input_path": "records.json",
        "split_manifest_path": str(split_path),
        "evaluation_split": evaluation_split,
        "selected_record_ids": ["case_test"],
        "num_records": 1,
        "num_judged": 1,
        "accuracy": 1.0,
        "reports": [],
    }
    verification = {
        "manifest_version": "gaam_case_evaluation_verification_v1",
        "status": "passed",
        "warnings": [],
        "errors": [],
    }
    (eval_dir / "case_evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    (eval_dir / "case_evaluation_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False),
        encoding="utf-8",
    )


def test_production_run_audit_passes_clean_e2e_artifacts(tmp_path: Path):
    training_dir = tmp_path / "training"
    eval_dir = tmp_path / "eval"
    model_dir = tmp_path / "model"
    split_path = tmp_path / "split.json"
    split_path.write_text('{"records":[]}', encoding="utf-8")
    _write_training_artifacts(training_dir, model_dir)
    _write_evaluation_artifacts(eval_dir, split_path)

    report = audit_production_run(
        ProductionRunAuditConfig(
            training_output_dir=training_dir,
            evaluation_output_dir=eval_dir,
            split_manifest_path=split_path,
            expected_evaluation_split="test",
        )
    )

    assert report["status"] == "passed"
    assert report["training"]["verification_status"] == "passed"
    assert report["evaluation"]["accuracy"] == 1.0
    assert (eval_dir / "production_run_audit.json").exists()


def test_production_run_audit_fails_on_wrong_evaluation_split(tmp_path: Path):
    training_dir = tmp_path / "training"
    eval_dir = tmp_path / "eval"
    model_dir = tmp_path / "model"
    split_path = tmp_path / "split.json"
    split_path.write_text('{"records":[]}', encoding="utf-8")
    _write_training_artifacts(training_dir, model_dir)
    _write_evaluation_artifacts(eval_dir, split_path, evaluation_split="train")

    report = audit_production_run(
        ProductionRunAuditConfig(
            training_output_dir=training_dir,
            evaluation_output_dir=eval_dir,
            split_manifest_path=split_path,
            expected_evaluation_split="test",
        )
    )

    assert report["status"] == "failed"
    assert any("evaluation split mismatch" in error.lower() for error in report["errors"])
