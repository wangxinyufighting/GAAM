"""
Phase 4 Milestone 3: Trainer Step Inspection

Inspects and validates completed trainer steps.

Key functions:
- inspect_phase4_trainer_step: validate trainer step completeness
- check_trainer_step_artifacts: verify expected files exist
- check_trainer_step_no_leakage: validate no-leakage status
- check_trainer_step_metrics: validate finite metrics

Design principle:
Honest reporting: if something is missing or invalid, report it clearly.
"""

import json
from pathlib import Path
from typing import Any

from gaam_graph.dataset_split_schema import DatasetSplitIssue, DatasetSplitName
from gaam_graph.phase4_trainer_schema import (
    Phase4TrainerBackend,
    Phase4TrainerStepManifest,
)


def load_trainer_step_manifest(run_dir: Path) -> Phase4TrainerStepManifest:
    """Load trainer step manifest from completed run."""
    manifest_path = run_dir / "trainer_step_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Trainer step manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Phase4TrainerStepManifest.model_validate(data)


def check_trainer_step_artifacts(
    manifest: Phase4TrainerStepManifest,
    run_dir: Path,
) -> list[DatasetSplitIssue]:
    """
    Check that expected artifacts exist.

    Checks:
    - Manifest exists (already loaded)
    - Actor reports exist for requested actors
    - Input batch paths exist
    - Checkpoint paths exist for successful real updates
    - Dry-run reports do not claim weights were written
    - Trainer-ready batch files exist
    """
    issues: list[DatasetSplitIssue] = []

    for report in manifest.actor_reports:
        actor_role = report.actor_role.value

        # Check input batch path
        if report.input_batch_path:
            batch_path = Path(report.input_batch_path)
            if not batch_path.exists():
                issues.append(
                    DatasetSplitIssue(
                        severity="error",
                        code="missing_artifact",
                        message=f"[{actor_role}] Input batch not found: {batch_path}",
                    )
                )

        # Check trainer report
        if report.output_dir:
            report_path = Path(report.output_dir) / "trainer_report.json"
            if not report_path.exists():
                issues.append(
                    DatasetSplitIssue(
                        severity="warning",
                        code="missing_artifact",
                        message=f"[{actor_role}] Trainer report not found: {report_path}",
                    )
                )

        # Check trainer-ready batch
        if report.trainer_ready_batch_path:
            trainer_ready_path = Path(report.trainer_ready_batch_path)
            if not trainer_ready_path.exists():
                issues.append(
                    DatasetSplitIssue(
                        severity="warning",
                        code="missing_artifact",
                        message=f"[{actor_role}] Trainer-ready batch not found: {trainer_ready_path}",
                    )
                )

        # Check checkpoint for successful updates
        if report.status == "succeeded" and report.written_checkpoint_path:
            checkpoint_path = Path(report.written_checkpoint_path)
            if not checkpoint_path.exists():
                issues.append(
                    DatasetSplitIssue(
                        severity="error",
                        code="missing_artifact",
                        message=f"[{actor_role}] Checkpoint directory not found: {checkpoint_path}",
                    )
                )
            else:
                # Check checkpoint metadata
                metadata_path = checkpoint_path / "checkpoint_metadata.json"
                if not metadata_path.exists():
                    issues.append(
                        DatasetSplitIssue(
                            severity="warning",
                            code="missing_artifact",
                            message=f"[{actor_role}] Checkpoint metadata not found: {metadata_path}",
                        )
                    )

        # Check dry-run consistency
        if manifest.backend == Phase4TrainerBackend.DRY_RUN:
            if report.written_checkpoint_path:
                checkpoint_path = Path(report.written_checkpoint_path)
                metadata_path = checkpoint_path / "checkpoint_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    if metadata.get("weights_written", False):
                        issues.append(
                            DatasetSplitIssue(
                                severity="error",
                                code="invalid_state",
                                message=f"[{actor_role}] Dry-run checkpoint claims weights_written=True",
                            )
                        )

    return issues


def check_trainer_step_no_leakage(
    manifest: Phase4TrainerStepManifest,
) -> list[DatasetSplitIssue]:
    """
    Check no-leakage status.

    Rules:
    - All actor reports must have no_leakage_passed=True
    """
    issues: list[DatasetSplitIssue] = []

    for report in manifest.actor_reports:
        if not report.no_leakage_passed:
            issues.append(
                DatasetSplitIssue(
                    severity="error",
                    code="leakage_risk",
                    message=f"[{report.actor_role.value}] No-leakage check failed",
                )
            )

    return issues


def check_trainer_step_metrics(
    manifest: Phase4TrainerStepManifest,
) -> list[DatasetSplitIssue]:
    """
    Check that metrics are finite.

    Rules:
    - NaN/Infinity metrics are invalid
    """
    import math

    issues: list[DatasetSplitIssue] = []

    # Check manifest-level metrics
    for key, value in manifest.metrics.items():
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                issues.append(
                    DatasetSplitIssue(
                        severity="error",
                        code="invalid_metric",
                        message=f"Manifest metric '{key}' is not finite: {value}",
                    )
                )

    # Check actor-level metrics
    for report in manifest.actor_reports:
        actor_role = report.actor_role.value

        for field_name in [
            "reward_mean",
            "advantage_mean",
            "loss",
            "approx_kl",
            "clip_fraction",
            "entropy",
            "learning_rate",
            "grad_norm",
            "update_seconds",
        ]:
            value = getattr(report, field_name, None)
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    issues.append(
                        DatasetSplitIssue(
                            severity="error",
                            code="invalid_metric",
                            message=f"[{actor_role}] Metric '{field_name}' is not finite: {value}",
                        )
                    )

    return issues


def check_trainer_step_split_policy(
    manifest: Phase4TrainerStepManifest,
) -> list[DatasetSplitIssue]:
    """
    Check split policy compliance.

    Rules:
    - Train split should have model updates
    - Dev/test splits should not update models (unless explicitly overridden)
    """
    issues: list[DatasetSplitIssue] = []

    if manifest.split == DatasetSplitName.TRAIN:
        # Train split should have updates
        succeeded_updates = [
            r for r in manifest.actor_reports if r.status == "succeeded"
        ]

        if not succeeded_updates:
            issues.append(
                DatasetSplitIssue(
                    severity="warning",
                    code="unexpected_state",
                    message="Train split has no successful updates",
                )
            )

    else:
        # Dev/test splits should not update models
        for report in manifest.actor_reports:
            if report.status == "succeeded" and report.written_checkpoint_path:
                checkpoint_path = Path(report.written_checkpoint_path)
                metadata_path = checkpoint_path / "checkpoint_metadata.json"

                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    if metadata.get("weights_written", False):
                        issues.append(
                            DatasetSplitIssue(
                                severity="error",
                                code="leakage_risk",
                                message=f"[{report.actor_role.value}] Dev/test split should not update model weights",
                            )
                        )

    return issues


def inspect_phase4_trainer_step(
    run_dir: Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Inspect and validate Phase 4 trainer step.

    Checks:
    - Manifest exists and is valid
    - Expected artifacts exist
    - No-leakage status
    - Finite metrics
    - Split policy compliance

    Args:
        run_dir: Trainer step output directory
        strict: If True, treat warnings as errors

    Returns:
        Inspection report dict with:
        - passed: bool
        - issues: list of DatasetSplitIssue
        - manifest: trainer step manifest dict
    """
    issues: list[DatasetSplitIssue] = []

    # Load manifest
    try:
        manifest = load_trainer_step_manifest(run_dir)
    except Exception as e:
        issues.append(
            DatasetSplitIssue(
                severity="error",
                code="missing_artifact",
                message=f"Failed to load trainer step manifest: {e}",
            )
        )
        return {
            "passed": False,
            "issues": [issue.model_dump() for issue in issues],
            "manifest": None,
        }

    # Run checks
    issues.extend(check_trainer_step_artifacts(manifest, run_dir))
    issues.extend(check_trainer_step_no_leakage(manifest))
    issues.extend(check_trainer_step_metrics(manifest))
    issues.extend(check_trainer_step_split_policy(manifest))

    # Determine pass/fail
    has_errors = any(issue.severity == "error" for issue in issues)
    has_warnings = any(issue.severity == "warning" for issue in issues)

    passed = not has_errors
    if strict and has_warnings:
        passed = False

    return {
        "passed": passed,
        "issues": [issue.model_dump() for issue in issues],
        "manifest": manifest.model_dump(),
        "num_errors": sum(1 for issue in issues if issue.severity == "error"),
        "num_warnings": sum(1 for issue in issues if issue.severity == "warning"),
    }


def write_trainer_step_inspection_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Write trainer step inspection report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
