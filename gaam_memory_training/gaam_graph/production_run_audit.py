"""Production end-to-end run audit utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from gaam_graph.utils import write_json


@dataclass(frozen=True)
class ProductionRunAuditConfig:
    training_output_dir: Path
    evaluation_output_dir: Path | None = None
    split_manifest_path: Path | None = None
    expected_evaluation_split: str | None = "test"
    write_report: bool = True


def audit_production_run(config: ProductionRunAuditConfig) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    training = _audit_training(config.training_output_dir, errors, warnings)
    evaluation = _audit_evaluation(
        config.evaluation_output_dir,
        config.split_manifest_path,
        config.expected_evaluation_split,
        errors,
        warnings,
    )

    report = {
        "manifest_version": "gaam_production_run_audit_v1",
        "status": "passed" if not errors else "failed",
        "training_output_dir": str(config.training_output_dir),
        "evaluation_output_dir": str(config.evaluation_output_dir) if config.evaluation_output_dir else None,
        "split_manifest_path": str(config.split_manifest_path) if config.split_manifest_path else None,
        "expected_evaluation_split": config.expected_evaluation_split,
        "training": training,
        "evaluation": evaluation,
        "errors": errors,
        "warnings": warnings,
    }

    if config.write_report:
        output_dir = config.evaluation_output_dir or config.training_output_dir
        write_json(output_dir / "production_run_audit.json", report)
    return report


def _audit_training(
    training_output_dir: Path,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    manifest_path = training_output_dir / "dual_cotraining_manifest.json"
    verification_path = training_output_dir / "native_verl_training_verification.json"
    training: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "verification_path": str(verification_path),
        "manifest_status": None,
        "verification_status": None,
        "final_model_paths": {},
        "num_rounds": None,
        "num_actor_steps": None,
    }

    manifest = _read_json_if_exists(manifest_path, errors, "training manifest")
    if manifest:
        training["manifest_status"] = manifest.get("status")
        training["final_model_paths"] = manifest.get("final_model_paths", {})
        round_records = manifest.get("round_records", [])
        training["num_rounds"] = len(round_records) if isinstance(round_records, list) else None
        if manifest.get("status") != "succeeded":
            errors.append(f"Training manifest status is not succeeded: {manifest.get('status')}")
        final_model_paths = manifest.get("final_model_paths", {})
        if not isinstance(final_model_paths, dict) or not final_model_paths:
            errors.append("Training manifest has no final_model_paths.")
        else:
            for actor_role in ("memory_builder", "question_agent"):
                path_value = final_model_paths.get(actor_role)
                if not path_value:
                    errors.append(f"Missing final model path for {actor_role}.")
                elif not Path(str(path_value)).exists():
                    errors.append(f"Final model path for {actor_role} does not exist: {path_value}")

    verification = _read_json_if_exists(verification_path, errors, "training verification")
    if verification:
        training["verification_status"] = verification.get("status")
        metrics = verification.get("metrics", {})
        training["num_actor_steps"] = metrics.get("num_actor_steps") if isinstance(metrics, dict) else None
        if verification.get("status") != "passed":
            errors.append(f"Training verification status is not passed: {verification.get('status')}")
        warnings.extend(str(item) for item in verification.get("warnings", []) if item)

    return training


def _audit_evaluation(
    evaluation_output_dir: Path | None,
    split_manifest_path: Path | None,
    expected_evaluation_split: str | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any] | None:
    if evaluation_output_dir is None:
        warnings.append("No evaluation_output_dir was provided; skipping evaluation audit.")
        return None

    manifest_path = evaluation_output_dir / "case_evaluation_manifest.json"
    verification_path = evaluation_output_dir / "case_evaluation_verification.json"
    evaluation: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "verification_path": str(verification_path),
        "manifest_status": None,
        "verification_status": None,
        "evaluation_split": None,
        "num_records": None,
        "num_judged": None,
        "accuracy": None,
        "selected_record_ids": [],
    }

    manifest = _read_json_if_exists(manifest_path, errors, "case evaluation manifest")
    if manifest:
        evaluation["manifest_status"] = manifest.get("status")
        evaluation["evaluation_split"] = manifest.get("evaluation_split")
        evaluation["num_records"] = manifest.get("num_records")
        evaluation["num_judged"] = manifest.get("num_judged")
        evaluation["accuracy"] = manifest.get("accuracy")
        evaluation["selected_record_ids"] = manifest.get("selected_record_ids", [])
        if manifest.get("status") != "succeeded":
            errors.append(f"Case evaluation manifest status is not succeeded: {manifest.get('status')}")
        if expected_evaluation_split and manifest.get("evaluation_split") != expected_evaluation_split:
            errors.append(
                "Case evaluation split mismatch: "
                f"expected {expected_evaluation_split}, got {manifest.get('evaluation_split')}"
            )
        if split_manifest_path and str(manifest.get("split_manifest_path")) != str(split_manifest_path):
            errors.append(
                "Case evaluation split manifest mismatch: "
                f"expected {split_manifest_path}, got {manifest.get('split_manifest_path')}"
            )
        if not manifest.get("selected_record_ids"):
            errors.append("Case evaluation selected_record_ids is empty.")
        if manifest.get("num_records") != manifest.get("num_judged"):
            errors.append(
                f"Case evaluation has unjudged records: num_records={manifest.get('num_records')} "
                f"num_judged={manifest.get('num_judged')}"
            )

    verification = _read_json_if_exists(verification_path, errors, "case evaluation verification")
    if verification:
        evaluation["verification_status"] = verification.get("status")
        if verification.get("status") != "passed":
            errors.append(f"Case evaluation verification status is not passed: {verification.get('status')}")
        warnings.extend(str(item) for item in verification.get("warnings", []) if item)

    return evaluation


def _read_json_if_exists(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(f"Missing {label}: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Failed to parse {label}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} is not a JSON object: {path}")
        return None
    return value
