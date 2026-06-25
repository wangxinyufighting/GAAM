"""
Phase 4 Milestone 4: Co-Training Inspection

Validation and completeness checking for co-training experiments.

Key functions:
- inspect_phase4_cotraining: full experiment validation
- check_cotraining_artifacts: verify expected files exist
- check_cotraining_split_policy: validate train/dev/test compliance
- check_cotraining_checkpoint_consistency: validate checkpoint carryover

Design principle:
Strict validation with optional strict mode that treats warnings as errors.
"""

import json
from pathlib import Path
from typing import Any

from gaam_graph.dataset_split_schema import DatasetSplitIssue, DatasetSplitName
from gaam_graph.phase4_cotraining_schema import (
    Phase4CotrainingManifest,
    Phase4RoundReport,
    Phase4RoundStatus,
)
from gaam_graph.phase4_rollout_schema import MultiCaseRolloutManifest
from gaam_graph.phase4_trainer_schema import Phase4TrainerStepManifest


def load_cotraining_manifest_for_inspection(run_dir: Path) -> Phase4CotrainingManifest:
    """Load co-training manifest from completed experiment."""
    manifest_path = run_dir / "cotraining_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Co-training manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Phase4CotrainingManifest.model_validate(data)


def check_cotraining_artifacts(
    manifest: Phase4CotrainingManifest,
    run_dir: Path,
) -> list[DatasetSplitIssue]:
    """
    Check that expected artifacts exist for each completed round.

    Checks:
    - Round plan exists
    - Train rollout manifest exists
    - Trainer step manifest exists
    - Dev rollout manifest exists
    - Checkpoint selection exists
    - Round summary exists
    """
    issues: list[DatasetSplitIssue] = []

    for report in manifest.round_reports:
        round_id = report.round_id
        round_dir = Path(report.output_dir)

        # Check round plan
        round_plan_path = round_dir / "round_plan.json"
        if not round_plan_path.exists():
            issues.append(
                DatasetSplitIssue(
                    severity="warning",
                    code="missing_artifact",
                    message=f"[Round {round_id}] Round plan not found: {round_plan_path}",
                )
            )

        # Check train rollout
        if report.train_rollout_manifest_path:
            train_manifest_path = Path(report.train_rollout_manifest_path)
            if not train_manifest_path.exists():
                issues.append(
                    DatasetSplitIssue(
                        severity="error",
                        code="missing_artifact",
                        message=f"[Round {round_id}] Train rollout manifest not found: {train_manifest_path}",
                    )
                )

        # Check trainer step
        if report.trainer_step_manifest_path:
            trainer_manifest_path = Path(report.trainer_step_manifest_path)
            if not trainer_manifest_path.exists():
                issues.append(
                    DatasetSplitIssue(
                        severity="error",
                        code="missing_artifact",
                        message=f"[Round {round_id}] Trainer step manifest not found: {trainer_manifest_path}",
                    )
                )

        # Check dev rollout
        if report.dev_rollout_manifest_path:
            dev_manifest_path = Path(report.dev_rollout_manifest_path)
            if not dev_manifest_path.exists():
                issues.append(
                    DatasetSplitIssue(
                        severity="error",
                        code="missing_artifact",
                        message=f"[Round {round_id}] Dev rollout manifest not found: {dev_manifest_path}",
                    )
                )

        # Check checkpoint selection
        if report.checkpoint_selection_path:
            selection_path = Path(report.checkpoint_selection_path)
            if not selection_path.exists():
                issues.append(
                    DatasetSplitIssue(
                        severity="warning",
                        code="missing_artifact",
                        message=f"[Round {round_id}] Checkpoint selection not found: {selection_path}",
                    )
                )

        # Check round summary
        round_summary_path = round_dir / "round_summary.json"
        if not round_summary_path.exists():
            issues.append(
                DatasetSplitIssue(
                    severity="warning",
                    code="missing_artifact",
                    message=f"[Round {round_id}] Round summary not found: {round_summary_path}",
                )
            )

    return issues


def check_cotraining_split_policy(
    manifest: Phase4CotrainingManifest,
) -> list[DatasetSplitIssue]:
    """
    Check split policy compliance.

    Rules:
    - Train rollouts should be on TRAIN split
    - Dev rollouts should be on DEV split
    - Dev/test splits should not update models
    """
    issues: list[DatasetSplitIssue] = []

    for report in manifest.round_reports:
        round_id = report.round_id

        # Check train rollout split
        if report.train_rollout_manifest_path:
            try:
                with open(report.train_rollout_manifest_path, "r", encoding="utf-8") as f:
                    train_data = json.load(f)
                train_manifest = MultiCaseRolloutManifest.model_validate(train_data)

                if train_manifest.split != DatasetSplitName.TRAIN:
                    issues.append(
                        DatasetSplitIssue(
                            severity="error",
                            code="split_policy_violation",
                            message=f"[Round {round_id}] Train rollout is not on TRAIN split: {train_manifest.split.value}",
                        )
                    )
            except Exception as e:
                issues.append(
                    DatasetSplitIssue(
                        severity="warning",
                        code="validation_error",
                        message=f"[Round {round_id}] Failed to validate train rollout manifest: {e}",
                    )
                )

        # Check dev rollout split
        if report.dev_rollout_manifest_path:
            try:
                with open(report.dev_rollout_manifest_path, "r", encoding="utf-8") as f:
                    dev_data = json.load(f)
                dev_manifest = MultiCaseRolloutManifest.model_validate(dev_data)

                if dev_manifest.split != DatasetSplitName.DEV:
                    issues.append(
                        DatasetSplitIssue(
                            severity="error",
                            code="split_policy_violation",
                            message=f"[Round {round_id}] Dev rollout is not on DEV split: {dev_manifest.split.value}",
                        )
                    )
            except Exception as e:
                issues.append(
                    DatasetSplitIssue(
                        severity="warning",
                        code="validation_error",
                        message=f"[Round {round_id}] Failed to validate dev rollout manifest: {e}",
                    )
                )

        # Check trainer step split
        if report.trainer_step_manifest_path:
            try:
                with open(report.trainer_step_manifest_path, "r", encoding="utf-8") as f:
                    trainer_data = json.load(f)
                trainer_manifest = Phase4TrainerStepManifest.model_validate(trainer_data)

                if trainer_manifest.split != DatasetSplitName.TRAIN:
                    issues.append(
                        DatasetSplitIssue(
                            severity="error",
                            code="split_policy_violation",
                            message=f"[Round {round_id}] Trainer step is not on TRAIN split: {trainer_manifest.split.value}",
                        )
                    )
            except Exception as e:
                issues.append(
                    DatasetSplitIssue(
                        severity="warning",
                        code="validation_error",
                        message=f"[Round {round_id}] Failed to validate trainer step manifest: {e}",
                    )
                )

    return issues


def check_cotraining_no_leakage(
    manifest: Phase4CotrainingManifest,
    min_pass_rate: float = 0.8,
) -> list[DatasetSplitIssue]:
    """
    Check no-leakage pass rates meet threshold.

    Args:
        manifest: co-training manifest
        min_pass_rate: minimum acceptable pass rate (default 0.8)
    """
    issues: list[DatasetSplitIssue] = []

    for report in manifest.round_reports:
        round_id = report.round_id

        # Check train no-leakage
        if report.train_no_leakage_pass_rate is not None:
            if report.train_no_leakage_pass_rate < min_pass_rate:
                issues.append(
                    DatasetSplitIssue(
                        severity="error",
                        code="leakage_risk",
                        message=f"[Round {round_id}] Train no-leakage pass rate too low: {report.train_no_leakage_pass_rate:.2f} < {min_pass_rate}",
                    )
                )

        # Check dev no-leakage
        if report.dev_no_leakage_pass_rate is not None:
            if report.dev_no_leakage_pass_rate < min_pass_rate:
                issues.append(
                    DatasetSplitIssue(
                        severity="error",
                        code="leakage_risk",
                        message=f"[Round {round_id}] Dev no-leakage pass rate too low: {report.dev_no_leakage_pass_rate:.2f} < {min_pass_rate}",
                    )
                )

    return issues


def check_cotraining_checkpoint_consistency(
    manifest: Phase4CotrainingManifest,
) -> list[DatasetSplitIssue]:
    """
    Check checkpoint carryover consistency.

    Rules:
    - Round t+1 input checkpoints should match round t selected checkpoints
    - If no checkpoints selected in round t, round t+1 should use same as round t input
    """
    issues: list[DatasetSplitIssue] = []

    for i in range(1, len(manifest.round_reports)):
        prev_report = manifest.round_reports[i - 1]
        curr_report = manifest.round_reports[i]

        # Check Memory Builder checkpoint carryover
        expected_mb = (
            prev_report.selected_memory_builder_checkpoint_id
            or prev_report.input_memory_builder_checkpoint_id
        )
        actual_mb = curr_report.input_memory_builder_checkpoint_id

        if expected_mb != actual_mb:
            issues.append(
                DatasetSplitIssue(
                    severity="warning",
                    code="checkpoint_inconsistency",
                    message=f"[Round {curr_report.round_id}] Memory Builder checkpoint mismatch: expected {expected_mb}, got {actual_mb}",
                )
            )

        # Check Question Agent checkpoint carryover
        expected_qa = (
            prev_report.selected_question_agent_checkpoint_id
            or prev_report.input_question_agent_checkpoint_id
        )
        actual_qa = curr_report.input_question_agent_checkpoint_id

        if expected_qa != actual_qa:
            issues.append(
                DatasetSplitIssue(
                    severity="warning",
                    code="checkpoint_inconsistency",
                    message=f"[Round {curr_report.round_id}] Question Agent checkpoint mismatch: expected {expected_qa}, got {actual_qa}",
                )
            )

    return issues


def check_cotraining_metrics(
    manifest: Phase4CotrainingManifest,
) -> list[DatasetSplitIssue]:
    """
    Check that metrics are finite (no NaN/Infinity).
    """
    import math

    issues: list[DatasetSplitIssue] = []

    for report in manifest.round_reports:
        round_id = report.round_id

        # Check reward metrics
        for field_name in [
            "train_memory_reward_mean",
            "train_question_reward_mean",
            "dev_memory_reward_mean",
            "dev_question_reward_mean",
        ]:
            value = getattr(report, field_name, None)
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    issues.append(
                        DatasetSplitIssue(
                            severity="error",
                            code="invalid_metric",
                            message=f"[Round {round_id}] Metric '{field_name}' is not finite: {value}",
                        )
                    )

        # Check no-leakage rates
        for field_name in ["train_no_leakage_pass_rate", "dev_no_leakage_pass_rate"]:
            value = getattr(report, field_name, None)
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    issues.append(
                        DatasetSplitIssue(
                            severity="error",
                            code="invalid_metric",
                            message=f"[Round {round_id}] Metric '{field_name}' is not finite: {value}",
                        )
                    )
                elif not (0.0 <= value <= 1.0):
                    issues.append(
                        DatasetSplitIssue(
                            severity="error",
                            code="invalid_metric",
                            message=f"[Round {round_id}] Metric '{field_name}' out of range [0,1]: {value}",
                        )
                    )

    return issues


def inspect_phase4_cotraining(
    run_dir: Path,
    *,
    strict: bool = False,
    min_no_leakage_pass_rate: float = 0.8,
) -> dict[str, Any]:
    """
    Inspect and validate Phase 4 co-training experiment.

    Checks:
    - Manifest exists and is valid
    - Expected artifacts exist per round
    - Split policy compliance
    - No-leakage thresholds
    - Checkpoint consistency
    - Finite metrics

    Args:
        run_dir: Co-training experiment output directory
        strict: If True, treat warnings as errors
        min_no_leakage_pass_rate: Minimum acceptable no-leakage pass rate

    Returns:
        Inspection report dict with:
        - passed: bool
        - issues: list of DatasetSplitIssue
        - manifest: co-training manifest dict
    """
    issues: list[DatasetSplitIssue] = []

    # Load manifest
    try:
        manifest = load_cotraining_manifest_for_inspection(run_dir)
    except Exception as e:
        issues.append(
            DatasetSplitIssue(
                severity="error",
                code="missing_artifact",
                message=f"Failed to load co-training manifest: {e}",
            )
        )
        return {
            "passed": False,
            "issues": [issue.model_dump() for issue in issues],
            "manifest": None,
        }

    # Run checks
    issues.extend(check_cotraining_artifacts(manifest, run_dir))
    issues.extend(check_cotraining_split_policy(manifest))
    issues.extend(check_cotraining_no_leakage(manifest, min_no_leakage_pass_rate))
    issues.extend(check_cotraining_checkpoint_consistency(manifest))
    issues.extend(check_cotraining_metrics(manifest))

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


def write_cotraining_inspection_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Write co-training inspection report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
