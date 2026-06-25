"""
Phase 4 Milestone 4: Multi-Round Adversarial Co-Training Schema

Data contracts for multi-round co-training loop.

Key types:
- Phase4RoundStatus: round execution status
- Phase4CotrainingConfig: top-level experiment configuration
- Phase4RoundPlan: what will run in one round
- Phase4RoundReport: what happened in one round
- Phase4CotrainingManifest: full experiment artifact
- Phase4EarlyStopReport: early stopping decision

Design principle:
Honest reporting: every field must reflect what actually happened,
not what was requested or what we hoped would happen.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from gaam_graph.dataset_split_schema import DatasetSplitName
from gaam_graph.phase4_trainer_schema import Phase4TrainerBackend


class Phase4RoundStatus(str, Enum):
    """Status of a training round."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class Phase4CotrainingConfig(BaseModel):
    """Configuration for multi-round co-training experiment."""

    # Required
    split_manifest_path: str
    output_dir: str
    num_rounds: int = 1

    # Record selection
    train_records_per_round: int | None = None
    dev_records_per_round: int | None = None
    train_record_ids: list[str] | None = None
    dev_record_ids: list[str] | None = None
    record_schedule_policy: str = "deterministic_window"

    # Backends
    rollout_backend: str = "local_fallback"
    trainer_backend: Phase4TrainerBackend = Phase4TrainerBackend.DRY_RUN
    trainer_mode: str = "dry_run"

    # Model paths
    memory_builder_model_path: str | None = None
    question_agent_model_path: str | None = None
    answerer_model_path: str | None = None

    # Initial checkpoints
    initial_memory_builder_checkpoint_id: str | None = None
    initial_question_agent_checkpoint_id: str | None = None
    checkpoint_registry_path: str | None = None

    # Actor update control
    update_memory_builder: bool = True
    update_question_agent: bool = True

    # Dev evaluation and checkpoint selection
    max_dev_regression: float = 0.02
    min_dev_no_leakage_pass_rate: float = 0.8

    # Early stopping
    early_stop_patience: int | None = None
    early_stop_metric: str = "dev_memory_reward_mean"
    early_stop_min_delta: float = 0.0

    # Failure handling
    continue_on_round_failure: bool = False
    continue_on_record_failure: bool = True
    allow_empty_batch: bool = False
    strict_no_leakage: bool = True

    # Execution control
    seed: int = 0
    resume: bool = False
    overwrite: bool = False


class Phase4RoundPlan(BaseModel):
    """Plan for one training round."""

    round_id: int
    train_record_ids: list[str]
    dev_record_ids: list[str]
    memory_builder_checkpoint_id: str | None = None
    question_agent_checkpoint_id: str | None = None
    memory_builder_checkpoint_path: str | None = None
    question_agent_checkpoint_path: str | None = None
    checkpoint_registry_path: str | None = None
    output_dir: str
    seed: int


class Phase4RoundReport(BaseModel):
    """Report of what happened in one training round."""

    round_id: int
    status: Phase4RoundStatus
    output_dir: str

    # Artifact paths
    train_rollout_dir: str | None = None
    train_rollout_manifest_path: str | None = None
    trainer_step_dir: str | None = None
    trainer_step_manifest_path: str | None = None
    dev_rollout_dir: str | None = None
    dev_rollout_manifest_path: str | None = None
    checkpoint_selection_path: str | None = None

    # Checkpoint tracking
    input_memory_builder_checkpoint_id: str | None = None
    input_question_agent_checkpoint_id: str | None = None
    selected_memory_builder_checkpoint_id: str | None = None
    selected_question_agent_checkpoint_id: str | None = None

    # Key metrics
    train_memory_reward_mean: float | None = None
    train_question_reward_mean: float | None = None
    dev_memory_reward_mean: float | None = None
    dev_question_reward_mean: float | None = None
    train_no_leakage_pass_rate: float | None = None
    dev_no_leakage_pass_rate: float | None = None

    # Additional metrics and diagnostics
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Phase4CotrainingManifest(BaseModel):
    """Top-level experiment manifest."""

    manifest_version: str = "phase4_cotraining_loop_v1"
    run_id: str
    split_manifest_path: str
    output_dir: str

    num_rounds_requested: int
    num_rounds_completed: int = 0
    status: Phase4RoundStatus

    started_at: str
    finished_at: str | None = None

    config: dict[str, Any]
    round_reports: list[Phase4RoundReport] = Field(default_factory=list)

    final_memory_builder_checkpoint_id: str | None = None
    final_question_agent_checkpoint_id: str | None = None
    checkpoint_registry_path: str | None = None

    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Phase4EarlyStopReport(BaseModel):
    """Early stopping decision report."""

    should_stop: bool
    reason: str | None = None
    metric_name: str
    best_value: float | None = None
    latest_value: float | None = None
    best_round_id: int | None = None
    patience: int | None = None
    rounds_without_improvement: int = 0


def aggregate_round_status(reports: list[Phase4RoundReport]) -> Phase4RoundStatus:
    """
    Aggregate status across rounds.

    Rules:
    - If any failed, return failed
    - If all succeeded, return succeeded
    - If any partial, return partial
    - Otherwise pending or running
    """
    if not reports:
        return Phase4RoundStatus.PENDING

    statuses = [r.status for r in reports]

    if Phase4RoundStatus.FAILED in statuses:
        return Phase4RoundStatus.FAILED
    if Phase4RoundStatus.RUNNING in statuses:
        return Phase4RoundStatus.RUNNING
    if all(s == Phase4RoundStatus.SUCCEEDED for s in statuses):
        return Phase4RoundStatus.SUCCEEDED
    if Phase4RoundStatus.PARTIAL in statuses:
        return Phase4RoundStatus.PARTIAL

    return Phase4RoundStatus.PENDING


def compute_experiment_metrics(
    manifest: Phase4CotrainingManifest,
) -> dict[str, Any]:
    """
    Compute top-level experiment metrics.

    Returns:
        Dictionary with:
        - num_rounds_succeeded
        - num_rounds_failed
        - num_rounds_partial
        - best_dev_memory_reward_mean
        - best_dev_round_id
        - early_stopped
    """
    metrics: dict[str, Any] = {}

    succeeded = [
        r for r in manifest.round_reports if r.status == Phase4RoundStatus.SUCCEEDED
    ]
    failed = [
        r for r in manifest.round_reports if r.status == Phase4RoundStatus.FAILED
    ]
    partial = [
        r for r in manifest.round_reports if r.status == Phase4RoundStatus.PARTIAL
    ]

    metrics["num_rounds_succeeded"] = len(succeeded)
    metrics["num_rounds_failed"] = len(failed)
    metrics["num_rounds_partial"] = len(partial)

    # Best dev memory reward
    dev_rewards = [
        (r.round_id, r.dev_memory_reward_mean)
        for r in manifest.round_reports
        if r.dev_memory_reward_mean is not None
    ]

    if dev_rewards:
        best_round_id, best_reward = max(dev_rewards, key=lambda x: x[1])
        metrics["best_dev_memory_reward_mean"] = best_reward
        metrics["best_dev_round_id"] = best_round_id
    else:
        metrics["best_dev_memory_reward_mean"] = None
        metrics["best_dev_round_id"] = None

    # Check if early stopped
    metrics["early_stopped"] = (
        manifest.num_rounds_completed < manifest.num_rounds_requested
        and manifest.status != Phase4RoundStatus.FAILED
    )

    return metrics


def generate_cotraining_run_id(*, seed: int, timestamp: str | None = None) -> str:
    """
    Generate unique run ID for co-training experiment.

    Format: cotraining_seed{seed}_{timestamp}
    """
    if timestamp is None:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    return f"cotraining_seed{seed}_{timestamp}"
