"""Final completion verifier for GAAM production E2E runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from gaam_graph.utils import write_json


@dataclass(frozen=True)
class ProductionCompletionVerificationConfig:
    training_output_dir: Path
    evaluation_output_dir: Path
    artifact_bundle_dir: Path
    expected_evaluation_split: str = "test"
    split_manifest_path: Path | None = None
    write_report: bool = True
    report_output_dir: Path | None = None


def verify_production_e2e_completion(
    config: ProductionCompletionVerificationConfig,
) -> dict[str, Any]:
    """Verify all final evidence files from a real production E2E run."""
    errors: list[str] = []
    warnings: list[str] = []

    training = _check_training(config.training_output_dir, errors, warnings)
    evaluation = _check_evaluation(
        config.evaluation_output_dir,
        config.expected_evaluation_split,
        config.split_manifest_path,
        errors,
        warnings,
    )
    audit = _check_audit(
        config.evaluation_output_dir,
        config.training_output_dir,
        config.expected_evaluation_split,
        errors,
        warnings,
    )
    bundle = _check_bundle(config.artifact_bundle_dir, errors, warnings)

    report = {
        "manifest_version": "gaam_production_e2e_completion_verification_v1",
        "status": "passed" if not errors else "failed",
        "training_output_dir": str(config.training_output_dir),
        "evaluation_output_dir": str(config.evaluation_output_dir),
        "artifact_bundle_dir": str(config.artifact_bundle_dir),
        "expected_evaluation_split": config.expected_evaluation_split,
        "split_manifest_path": str(config.split_manifest_path) if config.split_manifest_path else None,
        "checks": {
            "training": training,
            "evaluation": evaluation,
            "audit": audit,
            "artifact_bundle": bundle,
        },
        "errors": errors,
        "warnings": warnings,
    }
    if config.write_report:
        output_dir = config.report_output_dir or config.evaluation_output_dir
        write_json(output_dir / "production_e2e_completion_verification.json", report)
    return report


def _check_training(
    training_output_dir: Path,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    manifest_path = training_output_dir / "dual_cotraining_manifest.json"
    verification_path = training_output_dir / "native_verl_training_verification.json"
    manifest = _read_json(manifest_path, errors, "training manifest")
    verification = _read_json(verification_path, errors, "training verification")

    result: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "verification_path": str(verification_path),
        "manifest_status": manifest.get("status") if manifest else None,
        "verification_status": verification.get("status") if verification else None,
        "final_model_paths": manifest.get("final_model_paths", {}) if manifest else {},
        "metrics": verification.get("metrics", {}) if verification else {},
    }
    if manifest and manifest.get("status") != "succeeded":
        errors.append(f"Training manifest status is not succeeded: {manifest.get('status')}")
    if manifest and manifest.get("config", {}).get("dry_run") is True:
        errors.append("Training manifest reports dry_run=True; this is not real training.")
    if verification and verification.get("status") != "passed":
        errors.append(f"Training verification status is not passed: {verification.get('status')}")
    if verification:
        warnings.extend(str(item) for item in verification.get("warnings", []) if item)
    final_model_paths = result["final_model_paths"]
    if not isinstance(final_model_paths, dict) or not final_model_paths:
        errors.append("Training manifest has no final_model_paths.")
    else:
        for actor_role in ("memory_builder", "question_agent"):
            value = final_model_paths.get(actor_role)
            if not value:
                errors.append(f"Missing final model path for {actor_role}.")
            elif not Path(str(value)).exists():
                errors.append(f"Final model path for {actor_role} does not exist: {value}")
    return result


def _check_evaluation(
    evaluation_output_dir: Path,
    expected_evaluation_split: str,
    split_manifest_path: Path | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    manifest_path = evaluation_output_dir / "case_evaluation_manifest.json"
    verification_path = evaluation_output_dir / "case_evaluation_verification.json"
    summary_path = evaluation_output_dir / "evaluation_summary.json"
    manifest = _read_json(manifest_path, errors, "case evaluation manifest")
    verification = _read_json(verification_path, errors, "case evaluation verification")
    summary = _read_json(summary_path, errors, "evaluation summary")

    result = {
        "manifest_path": str(manifest_path),
        "verification_path": str(verification_path),
        "summary_path": str(summary_path),
        "manifest_status": manifest.get("status") if manifest else None,
        "verification_status": verification.get("status") if verification else None,
        "summary_status": summary.get("status") if summary else None,
        "evaluation_split": manifest.get("evaluation_split") if manifest else None,
        "num_records": manifest.get("num_records") if manifest else None,
        "num_judged": manifest.get("num_judged") if manifest else None,
        "accuracy": manifest.get("accuracy") if manifest else None,
    }
    if manifest and manifest.get("status") != "succeeded":
        errors.append(f"Case evaluation manifest status is not succeeded: {manifest.get('status')}")
    if verification and verification.get("status") != "passed":
        errors.append(f"Case evaluation verification status is not passed: {verification.get('status')}")
    if summary and summary.get("status") != "succeeded":
        errors.append(f"Evaluation summary status is not succeeded: {summary.get('status')}")
    if manifest and manifest.get("evaluation_split") != expected_evaluation_split:
        errors.append(
            "Case evaluation split mismatch: "
            f"expected {expected_evaluation_split}, got {manifest.get('evaluation_split')}"
        )
    if split_manifest_path and manifest and str(manifest.get("split_manifest_path")) != str(split_manifest_path):
        errors.append(
            "Case evaluation split manifest mismatch: "
            f"expected {split_manifest_path}, got {manifest.get('split_manifest_path')}"
        )
    if manifest and manifest.get("num_records") != manifest.get("num_judged"):
        errors.append(
            f"Case evaluation has unjudged records: num_records={manifest.get('num_records')} "
            f"num_judged={manifest.get('num_judged')}"
        )
    if verification:
        warnings.extend(str(item) for item in verification.get("warnings", []) if item)
    return result


def _check_audit(
    evaluation_output_dir: Path,
    training_output_dir: Path,
    expected_evaluation_split: str,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    audit_path = evaluation_output_dir / "production_run_audit.json"
    audit = _read_json(audit_path, errors, "production run audit")
    result = {
        "path": str(audit_path),
        "status": audit.get("status") if audit else None,
    }
    if audit and audit.get("status") != "passed":
        errors.append(f"Production run audit status is not passed: {audit.get('status')}")
    if audit and str(audit.get("training_output_dir")) != str(training_output_dir):
        errors.append(
            "Production run audit training_output_dir mismatch: "
            f"expected {training_output_dir}, got {audit.get('training_output_dir')}"
        )
    if audit and audit.get("expected_evaluation_split") != expected_evaluation_split:
        errors.append(
            "Production run audit expected_evaluation_split mismatch: "
            f"expected {expected_evaluation_split}, got {audit.get('expected_evaluation_split')}"
        )
    if audit:
        warnings.extend(str(item) for item in audit.get("warnings", []) if item)
    return result


def _check_bundle(
    artifact_bundle_dir: Path,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    manifest_path = artifact_bundle_dir / "bundle_manifest.json"
    verification_path = artifact_bundle_dir / "production_artifact_bundle_verification.json"
    manifest = _read_json(manifest_path, errors, "artifact bundle manifest")
    verification = _read_json(verification_path, errors, "artifact bundle verification")
    result = {
        "manifest_path": str(manifest_path),
        "verification_path": str(verification_path),
        "manifest_status": manifest.get("status") if manifest else None,
        "verification_status": verification.get("status") if verification else None,
        "tar_path": manifest.get("tar_path") if manifest else None,
    }
    if manifest and manifest.get("status") != "succeeded":
        errors.append(f"Artifact bundle status is not succeeded: {manifest.get('status')}")
    if verification and verification.get("status") != "passed":
        errors.append(f"Artifact bundle verification status is not passed: {verification.get('status')}")
    if manifest and manifest.get("tar_path") and not Path(str(manifest["tar_path"])).exists():
        errors.append(f"Artifact bundle tarball does not exist: {manifest['tar_path']}")
    if verification:
        warnings.extend(str(item) for item in verification.get("warnings", []) if item)
    return result


def _read_json(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
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
