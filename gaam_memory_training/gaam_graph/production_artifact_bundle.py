"""Export compact evidence bundles for GAAM production runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from typing import Any

from gaam_graph.utils import write_json


SECRET_TOKENS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


@dataclass(frozen=True)
class ProductionArtifactBundleConfig:
    """Configuration for a compact, checkpoint-free production evidence bundle."""

    training_output_dir: Path
    bundle_output_dir: Path
    evaluation_output_dir: Path | None = None
    split_manifest_path: Path | None = None
    extra_files: list[Path] = field(default_factory=list)
    create_tar: bool = True
    run_id: str | None = None
    env_snapshot: dict[str, str] | None = None


def export_production_artifact_bundle(config: ProductionArtifactBundleConfig) -> dict[str, Any]:
    """Copy small audit/evaluation artifacts and write checksums."""
    bundle_dir = config.bundle_output_dir
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, Any]] = []
    missing: list[str] = []
    warnings: list[str] = []

    candidates = _candidate_files(config)
    for label, source in candidates:
        if source is None:
            continue
        if not source.exists() or not source.is_file():
            missing.append(f"{label}: {source}")
            continue
        destination = bundle_dir / _bundle_relative_path(label, source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "label": label,
                "source": str(source),
                "bundle_path": str(destination),
                "sha256": _sha256_file(destination),
                "bytes": destination.stat().st_size,
            }
        )

    env_snapshot = _redact_env(config.env_snapshot or {})
    manifest = {
        "manifest_version": "gaam_production_artifact_bundle_v1",
        "status": "succeeded" if copied else "failed",
        "run_id": config.run_id,
        "training_output_dir": str(config.training_output_dir),
        "evaluation_output_dir": str(config.evaluation_output_dir) if config.evaluation_output_dir else None,
        "split_manifest_path": str(config.split_manifest_path) if config.split_manifest_path else None,
        "bundle_output_dir": str(bundle_dir),
        "files": copied,
        "missing": missing,
        "warnings": warnings,
        "env_snapshot": env_snapshot,
    }
    manifest_path = bundle_dir / "bundle_manifest.json"
    write_json(manifest_path, manifest)

    if config.create_tar:
        tar_path = bundle_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(bundle_dir, arcname=bundle_dir.name)
        manifest["tar_path"] = str(tar_path)
        manifest["tar_sha256"] = _sha256_file(tar_path)
        write_json(manifest_path, manifest)

    return manifest


def _candidate_files(config: ProductionArtifactBundleConfig) -> list[tuple[str, Path | None]]:
    files: list[tuple[str, Path | None]] = [
        ("training/dual_cotraining_manifest.json", config.training_output_dir / "dual_cotraining_manifest.json"),
        ("training/native_verl_training_verification.json", config.training_output_dir / "native_verl_training_verification.json"),
        ("split/split_manifest.json", config.split_manifest_path),
    ]
    if config.evaluation_output_dir is not None:
        files.extend(
            [
                ("evaluation/case_evaluation_manifest.json", config.evaluation_output_dir / "case_evaluation_manifest.json"),
                ("evaluation/case_evaluation_verification.json", config.evaluation_output_dir / "case_evaluation_verification.json"),
                ("evaluation/evaluation_summary.json", config.evaluation_output_dir / "evaluation_summary.json"),
                ("evaluation/production_run_audit.json", config.evaluation_output_dir / "production_run_audit.json"),
            ]
        )
    for index, path in enumerate(config.extra_files):
        files.append((f"extra/{index:03d}_{path.name}", path))
    return files


def _bundle_relative_path(label: str, source: Path) -> Path:
    suffix = source.suffix
    if label.endswith(source.name):
        return Path(label)
    return Path(label).with_suffix(suffix)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_env(env: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in sorted(env.items()):
        if any(token in key.upper() for token in SECRET_TOKENS):
            redacted[key] = "<redacted>" if value else ""
        else:
            redacted[key] = str(value)
    return redacted
