"""
Phase 4 Milestone 3: Checkpoint Selection

Selects checkpoints from trainer step results, optionally gated by dev evaluation.

Key functions:
- select_latest_successful_checkpoints: simple latest-successful policy
- select_checkpoints_with_optional_dev_metrics: dev-gated selection
- write_checkpoint_selection_report: write selection metadata

Design principle:
Never select checkpoints based on test split. Test split is final reporting only.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from gaam_graph.phase4_rollout_schema import MultiCaseRolloutManifest
from gaam_graph.phase4_trainer_schema import (
    CheckpointSelectionReport,
    Phase4TrainerStepManifest,
)
from gaam_graph.grpo_schema import ActorRole


def select_latest_successful_checkpoints(
    trainer_manifest: Phase4TrainerStepManifest,
) -> dict[ActorRole, str]:
    """
    Select latest successful checkpoints from trainer step.

    Policy:
    - Select checkpoint from succeeded actor reports
    - Fail if no successful updates

    Returns:
        Dict mapping ActorRole to checkpoint_id
    """
    selected: dict[ActorRole, str] = {}

    for report in trainer_manifest.actor_reports:
        if report.status == "succeeded" and report.written_checkpoint_id:
            selected[report.actor_role] = report.written_checkpoint_id

    return selected


def select_checkpoints_with_optional_dev_metrics(
    trainer_manifest: Phase4TrainerStepManifest,
    dev_rollout_manifest: MultiCaseRolloutManifest | None = None,
    *,
    max_dev_regression: float = 0.02,
) -> CheckpointSelectionReport:
    """
    Select checkpoints with optional dev evaluation gate.

    Policy:
    1. If no dev run supplied, select latest successful train checkpoints
    2. If dev run supplied, require:
       - dev no-leakage pass rate acceptable
       - dev memory reward does not regress beyond max_dev_regression
       - dev status is succeeded or partial with enough successful records
    3. Never select checkpoints based on test split

    Args:
        trainer_manifest: Trainer step manifest
        dev_rollout_manifest: Optional dev rollout for gating
        max_dev_regression: Maximum allowed dev reward regression

    Returns:
        CheckpointSelectionReport with selection results
    """
    selection_id = f"checkpoint_selection_round{trainer_manifest.round_id:03d}_step{trainer_manifest.step_id:03d}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    report = CheckpointSelectionReport(
        selection_id=selection_id,
        round_id=trainer_manifest.round_id,
        step_id=trainer_manifest.step_id,
        train_trainer_step_manifest_path=str(
            Path(trainer_manifest.output_dir) / "trainer_step_manifest.json"
        ),
        dev_rollout_manifest_path=None,
        selection_policy="latest_successful",
    )

    # Select latest successful checkpoints from train
    selected = select_latest_successful_checkpoints(trainer_manifest)

    if not selected:
        report.warnings.append("No successful checkpoints found in trainer step")
        return report

    # If no dev evaluation, accept train checkpoints
    if dev_rollout_manifest is None:
        report.selected_memory_builder_checkpoint_id = selected.get(
            ActorRole.MEMORY_BUILDER
        )
        report.selected_question_agent_checkpoint_id = selected.get(
            ActorRole.QUESTION_AGENT
        )
        report.metrics["selection_policy"] = "latest_successful_no_dev"
        return report

    # Dev evaluation provided: apply gate
    report.dev_rollout_manifest_path = str(
        Path(dev_rollout_manifest.output_dir) / "multi_case_rollout_manifest.json"
    )
    report.selection_policy = "latest_successful_with_dev_gate"

    # Check dev status
    if dev_rollout_manifest.status not in ["succeeded", "partial"]:
        report.warnings.append(
            f"Dev rollout status is {dev_rollout_manifest.status}, rejecting checkpoints"
        )
        return report

    if not dev_rollout_manifest.selected_record_ids:
        report.selected_memory_builder_checkpoint_id = selected.get(
            ActorRole.MEMORY_BUILDER
        )
        report.selected_question_agent_checkpoint_id = selected.get(
            ActorRole.QUESTION_AGENT
        )
        report.selection_policy = "latest_successful_empty_dev"
        report.metrics["selection_policy"] = "latest_successful_empty_dev"
        report.metrics["dev_num_records"] = 0
        return report

    # Check dev no-leakage pass rate
    dev_metrics = dev_rollout_manifest.metrics
    dev_no_leakage_pass_rate = dev_metrics.get("no_leakage_pass_rate", 0.0)

    if dev_no_leakage_pass_rate < 0.8:
        report.warnings.append(
            f"Dev no-leakage pass rate too low: {dev_no_leakage_pass_rate:.2%}"
        )
        return report

    # Check dev memory reward regression
    dev_memory_reward_mean = dev_metrics.get("memory_reward_mean")
    train_memory_reward_mean = trainer_manifest.metrics.get("memory_builder_reward_mean")

    if (
        dev_memory_reward_mean is not None
        and train_memory_reward_mean is not None
    ):
        regression = train_memory_reward_mean - dev_memory_reward_mean

        if regression > max_dev_regression:
            report.warnings.append(
                f"Dev memory reward regressed by {regression:.4f} (max allowed: {max_dev_regression:.4f})"
            )
            return report

        report.metrics["dev_memory_reward_regression"] = regression

    # All gates passed, select checkpoints
    report.selected_memory_builder_checkpoint_id = selected.get(
        ActorRole.MEMORY_BUILDER
    )
    report.selected_question_agent_checkpoint_id = selected.get(
        ActorRole.QUESTION_AGENT
    )

    report.metrics["dev_no_leakage_pass_rate"] = dev_no_leakage_pass_rate
    report.metrics["dev_memory_reward_mean"] = dev_memory_reward_mean
    report.metrics["train_memory_reward_mean"] = train_memory_reward_mean

    return report


def write_checkpoint_selection_report(
    report: CheckpointSelectionReport,
    output_path: Path,
) -> None:
    """Write checkpoint selection report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)


def load_checkpoint_selection_report(path: Path) -> CheckpointSelectionReport:
    """Load checkpoint selection report from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return CheckpointSelectionReport.model_validate(data)
