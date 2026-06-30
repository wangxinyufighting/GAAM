"""Verification utilities for compact GAAM production evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from gaam_graph.utils import write_json


REQUIRED_BUNDLE_LABELS = {
    "training/dual_cotraining_manifest.json",
    "training/native_verl_training_verification.json",
    "split/split_manifest.json",
    "evaluation/case_evaluation_manifest.json",
    "evaluation/case_evaluation_verification.json",
    "evaluation/evaluation_summary.json",
    "evaluation/production_run_audit.json",
}


@dataclass(frozen=True)
class ProductionArtifactBundleVerificationConfig:
    bundle_output_dir: Path
    require_evaluation: bool = True
    require_tar: bool = True
    write_report: bool = True


def verify_production_artifact_bundle(
    config: ProductionArtifactBundleVerificationConfig,
) -> dict[str, Any]:
    """Verify bundle manifest, copied files, checksums, and optional tarball."""
    bundle_dir = config.bundle_output_dir
    manifest_path = bundle_dir / "bundle_manifest.json"
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        report = _report(config, manifest_path, errors=[f"Missing bundle manifest: {manifest_path}"], warnings=warnings)
        return _maybe_write(report, config)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report = _report(
            config,
            manifest_path,
            errors=[f"Failed to parse bundle manifest: {type(exc).__name__}: {exc}"],
            warnings=warnings,
        )
        return _maybe_write(report, config)

    if manifest.get("manifest_version") != "gaam_production_artifact_bundle_v1":
        errors.append(f"Unexpected bundle manifest version: {manifest.get('manifest_version')}")
    if manifest.get("status") != "succeeded":
        errors.append(f"Bundle status is not succeeded: {manifest.get('status')}")

    files = manifest.get("files", [])
    if not isinstance(files, list) or not files:
        errors.append("Bundle manifest has no copied files.")
        files = []

    labels = {str(item.get("label", "")) for item in files if isinstance(item, dict)}
    required_labels = set(REQUIRED_BUNDLE_LABELS)
    if not config.require_evaluation:
        required_labels = {label for label in required_labels if not label.startswith("evaluation/")}
    missing_labels = sorted(required_labels - labels)
    if missing_labels:
        errors.append(f"Bundle is missing required labels: {missing_labels}")

    checksum_failures = []
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            errors.append(f"Bundle file entry #{index} is not an object.")
            continue
        bundle_path = Path(str(item.get("bundle_path", "")))
        if not bundle_path.is_absolute():
            bundle_path = bundle_dir / bundle_path.relative_to(bundle_dir) if _is_under(bundle_path, bundle_dir) else bundle_path
        if not bundle_path.exists():
            errors.append(f"Bundled file does not exist: {bundle_path}")
            continue
        expected_sha = str(item.get("sha256", ""))
        actual_sha = _sha256_file(bundle_path)
        if expected_sha and expected_sha != actual_sha:
            checksum_failures.append(str(bundle_path))
    if checksum_failures:
        errors.append(f"Bundle checksum mismatch for files: {checksum_failures}")

    missing = manifest.get("missing", [])
    if missing:
        errors.append(f"Bundle manifest reports missing source files: {missing}")

    tar_path = manifest.get("tar_path")
    if config.require_tar:
        if not tar_path:
            errors.append("Bundle manifest has no tar_path.")
        elif not Path(str(tar_path)).exists():
            errors.append(f"Bundle tarball does not exist: {tar_path}")
        else:
            expected_tar_sha = str(manifest.get("tar_sha256", ""))
            actual_tar_sha = _sha256_file(Path(str(tar_path)))
            if expected_tar_sha and expected_tar_sha != actual_tar_sha:
                errors.append(f"Bundle tarball checksum mismatch: {tar_path}")
    elif not tar_path:
        warnings.append("Bundle tarball was not required and is not present.")

    report = _report(
        config,
        manifest_path,
        errors=errors,
        warnings=warnings,
        metrics={
            "num_files": len(files),
            "num_required_labels": len(required_labels),
            "num_missing_labels": len(missing_labels),
        },
    )
    return _maybe_write(report, config)


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report(
    config: ProductionArtifactBundleVerificationConfig,
    manifest_path: Path,
    *,
    errors: list[str],
    warnings: list[str],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "manifest_version": "gaam_production_artifact_bundle_verification_v1",
        "status": "passed" if not errors else "failed",
        "bundle_output_dir": str(config.bundle_output_dir),
        "manifest_path": str(manifest_path),
        "require_evaluation": config.require_evaluation,
        "require_tar": config.require_tar,
        "metrics": metrics or {},
        "errors": errors,
        "warnings": warnings,
    }


def _maybe_write(
    report: dict[str, Any],
    config: ProductionArtifactBundleVerificationConfig,
) -> dict[str, Any]:
    if config.write_report:
        config.bundle_output_dir.mkdir(parents=True, exist_ok=True)
        write_json(config.bundle_output_dir / "production_artifact_bundle_verification.json", report)
    return report
