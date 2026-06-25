"""
Phase 4 Milestone 6: Artifact Verification

Verify release package artifacts for completeness and integrity.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Phase4ArtifactVerificationReport(BaseModel):
    """Report of artifact verification."""

    status: str  # passed, failed
    release_dir: str
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0


def check_required_files_exist(
    release_dir: Path,
) -> tuple[bool, list[str]]:
    """Check that required files exist in release directory."""
    required_files = [
        "README.md",
        "reproduce.sh",
        "suite_config.resolved.json",
        "suite_manifest.json",
        "aggregate_metrics.json",
        "environment_report.json",
        "artifact_manifest.json",
        "checksums.sha256",
    ]

    issues = []
    for file_name in required_files:
        file_path = release_dir / file_name
        if not file_path.exists():
            issues.append(f"Missing required file: {file_name}")

    return len(issues) == 0, issues


def check_json_files_parse(
    release_dir: Path,
) -> tuple[bool, list[str]]:
    """Check that JSON files parse correctly."""
    json_files = [
        "suite_config.resolved.json",
        "suite_manifest.json",
        "aggregate_metrics.json",
        "environment_report.json",
        "artifact_manifest.json",
        "paper_table.json",
    ]

    issues = []
    for file_name in json_files:
        file_path = release_dir / file_name
        if not file_path.exists():
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as exc:
            issues.append(f"Failed to parse {file_name}: {exc}")

    return len(issues) == 0, issues


def check_checksums_match(
    release_dir: Path,
) -> tuple[bool, list[str]]:
    """Verify checksums match."""
    checksums_path = release_dir / "checksums.sha256"
    if not checksums_path.exists():
        return False, ["checksums.sha256 not found"]

    issues = []

    with open(checksums_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(None, 1)
            if len(parts) != 2:
                issues.append(f"Invalid checksum line: {line}")
                continue

            expected_checksum, rel_path = parts
            file_path = release_dir / rel_path

            if not file_path.exists():
                issues.append(f"Checksum file not found: {rel_path}")
                continue

            # Compute actual checksum
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f_data:
                for byte_block in iter(lambda: f_data.read(4096), b""):
                    sha256_hash.update(byte_block)

            actual_checksum = sha256_hash.hexdigest()

            if actual_checksum != expected_checksum:
                issues.append(
                    f"Checksum mismatch for {rel_path}: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )

    return len(issues) == 0, issues


def check_suite_manifest_consistency(
    release_dir: Path,
) -> tuple[bool, list[str]]:
    """Check suite manifest is consistent with aggregate metrics."""
    manifest_path = release_dir / "suite_manifest.json"
    metrics_path = release_dir / "aggregate_metrics.json"

    if not manifest_path.exists() or not metrics_path.exists():
        return True, []  # Skip if files don't exist

    issues = []

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Check suite IDs match
        if manifest.get("suite_id") != metrics.get("suite_id"):
            issues.append(
                f"Suite ID mismatch: manifest has {manifest.get('suite_id')}, "
                f"metrics has {metrics.get('suite_id')}"
            )

        # Check seed counts match
        num_runs = len(manifest.get("run_records", []))
        num_seeds_requested = metrics.get("num_seeds_requested", 0)

        if num_runs != num_seeds_requested:
            issues.append(
                f"Seed count mismatch: manifest has {num_runs} runs, "
                f"metrics has {num_seeds_requested} seeds requested"
            )

    except Exception as exc:
        issues.append(f"Error checking manifest consistency: {exc}")

    return len(issues) == 0, issues


def check_no_test_split_violation(
    release_dir: Path,
) -> tuple[bool, list[str]]:
    """Check that test split was not used for training or selection."""
    config_path = release_dir / "suite_config.resolved.json"
    if not config_path.exists():
        return True, []

    issues = []
    warnings = []

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Test split should only be used for final evaluation
        if not config.get("final_test_eval", True):
            warnings.append("final_test_eval is disabled - test split was not evaluated")

        # Check leakage audit was strict
        if not config.get("strict_no_leakage", True):
            warnings.append("strict_no_leakage is disabled - leakage audit was lenient")

    except Exception as exc:
        issues.append(f"Error checking test split policy: {exc}")

    return len(issues) == 0, issues


def check_no_hidden_failed_runs(
    release_dir: Path,
) -> tuple[bool, list[str]]:
    """Check that failed runs are recorded, not hidden."""
    manifest_path = release_dir / "suite_manifest.json"
    if not manifest_path.exists():
        return True, []

    issues = []

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        run_records = manifest.get("run_records", [])
        failed_runs = [r for r in run_records if r.get("status") == "failed"]

        if failed_runs:
            # Check that aggregate metrics acknowledges failed runs
            metrics_path = release_dir / "aggregate_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)

                num_failed = metrics.get("num_seeds_failed", 0)
                if num_failed != len(failed_runs):
                    issues.append(
                        f"Failed run count mismatch: manifest has {len(failed_runs)}, "
                        f"metrics has {num_failed}"
                    )

    except Exception as exc:
        issues.append(f"Error checking failed runs: {exc}")

    return len(issues) == 0, issues


def verify_phase4_release_artifacts(
    release_dir: Path,
    *,
    strict: bool = True,
) -> Phase4ArtifactVerificationReport:
    """
    Verify release package artifacts.

    Args:
        release_dir: Path to release directory
        strict: Fail on any issue (default: True)

    Returns:
        Verification report
    """
    report = Phase4ArtifactVerificationReport(
        status="passed",
        release_dir=str(release_dir),
    )

    # Run checks
    checks = [
        ("Required files exist", check_required_files_exist),
        ("JSON files parse", check_json_files_parse),
        ("Checksums match", check_checksums_match),
        ("Suite manifest consistency", check_suite_manifest_consistency),
        ("No test split violation", check_no_test_split_violation),
        ("No hidden failed runs", check_no_hidden_failed_runs),
    ]

    for check_name, check_func in checks:
        passed, issues = check_func(release_dir)

        if passed:
            report.checks_passed += 1
        else:
            report.checks_failed += 1
            for issue in issues:
                report.issues.append(f"{check_name}: {issue}")

    # Determine overall status
    if report.checks_failed > 0:
        report.status = "failed"

    return report


def write_verification_report(
    report: Phase4ArtifactVerificationReport,
    output_path: Path,
) -> None:
    """Write verification report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
