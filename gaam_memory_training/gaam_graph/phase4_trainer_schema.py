"""
Phase 4 Milestone 3: Trainer Step Schema

Defines data contracts for GRPO trainer step execution over aggregated multi-case batches.

Key types:
- Phase4TrainerBackend: execution backend (dry_run/local_grpo/verl)
- Phase4TrainerStepConfig: trainer step configuration
- Phase4ActorTrainerInput: per-actor input metadata
- Phase4ActorTrainerReport: per-actor update result
- Phase4TrainerStepManifest: top-level audit artifact
- CheckpointSelectionReport: checkpoint selection metadata

Design principle:
Narrow trainer boundary: safe aggregated batches in → validated actor-specific
model updates → auditable checkpoints out.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from gaam_graph.dataset_split_schema import DatasetSplitName
from gaam_graph.grpo_schema import ActorRole


class Phase4TrainerBackend(str, Enum):
    """Trainer backend for Phase 4 trainer step."""

    DRY_RUN = "dry_run"
    LOCAL_GRPO = "local_grpo"
    VERL = "verl"


class Phase4TrainerStepConfig(BaseModel):
    """Configuration for one Phase 4 trainer step."""

    source_multi_case_run_dir: str
    output_dir: str
    backend: Phase4TrainerBackend = Phase4TrainerBackend.DRY_RUN
    round_id: int = 0
    step_id: int = 0
    actors: list[ActorRole] = Field(
        default_factory=lambda: [
            ActorRole.MEMORY_BUILDER,
            ActorRole.QUESTION_AGENT,
        ]
    )
    memory_builder_model_path: str | None = None
    question_agent_model_path: str | None = None
    memory_builder_checkpoint_id: str | None = None
    question_agent_checkpoint_id: str | None = None
    checkpoint_registry_path: str | None = None
    learning_rate: float = 1e-6
    max_grad_norm: float = 1.0
    clip_ratio: float = 0.2
    kl_coef: float = 0.01
    entropy_coef: float = 0.0
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    require_non_empty_batch: bool = True
    allow_empty_batch: bool = False
    strict_no_leakage: bool = True
    overwrite: bool = False
    resume: bool = False
    seed: int = 0

    model_config = {"extra": "forbid"}


class Phase4ActorTrainerInput(BaseModel):
    """Input metadata for one actor's trainer step."""

    actor_role: ActorRole
    batch_path: str
    batch_manifest_path: str | None = None
    source_multi_case_run_dir: str
    source_multi_case_manifest_path: str
    split: DatasetSplitName
    round_id: int
    step_id: int
    num_samples: int
    num_groups: int
    no_leakage_passed: bool
    selected_record_ids: list[str]

    model_config = {"extra": "forbid"}


class Phase4ActorTrainerReport(BaseModel):
    """Result report for one actor's trainer step."""

    actor_role: ActorRole
    status: str  # succeeded, skipped, failed, partial
    backend: Phase4TrainerBackend
    input_batch_path: str
    input_batch_manifest_path: str | None = None
    output_dir: str
    loaded_checkpoint_id: str | None = None
    loaded_checkpoint_path: str | None = None
    written_checkpoint_id: str | None = None
    written_checkpoint_path: str | None = None
    trainer_ready_batch_path: str | None = None
    tensor_payload_path: str | None = None
    num_input_samples: int = 0
    num_selected_samples: int = 0
    reward_mean: float | None = None
    advantage_mean: float | None = None
    loss: float | None = None
    approx_kl: float | None = None
    clip_fraction: float | None = None
    entropy: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    update_seconds: float | None = None
    no_leakage_passed: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class Phase4TrainerStepManifest(BaseModel):
    """Top-level audit artifact for one trainer step."""

    manifest_version: str = "phase4_trainer_step_v1"
    run_id: str
    source_multi_case_run_dir: str
    source_multi_case_manifest_path: str
    output_dir: str
    split: DatasetSplitName
    round_id: int
    step_id: int
    backend: Phase4TrainerBackend
    status: str
    started_at: str
    finished_at: str | None = None
    actor_reports: list[Phase4ActorTrainerReport] = Field(default_factory=list)
    checkpoint_registry_path: str | None = None
    dev_eval_path: str | None = None
    selected_for_next_round: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class CheckpointSelectionReport(BaseModel):
    """Metadata for checkpoint selection from trainer step and optional dev eval."""

    manifest_version: str = "phase4_checkpoint_selection_v1"
    selection_id: str
    round_id: int
    step_id: int
    train_trainer_step_manifest_path: str
    dev_rollout_manifest_path: str | None = None
    selected_memory_builder_checkpoint_id: str | None = None
    selected_question_agent_checkpoint_id: str | None = None
    selection_policy: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


def aggregate_actor_trainer_status(
    actor_reports: list[Phase4ActorTrainerReport],
) -> str:
    """
    Aggregate per-actor trainer statuses into a trainer-step-level status.

    Rules:
    - all succeeded -> succeeded
    - all skipped -> skipped
    - all failed -> failed
    - mix -> partial
    """
    if not actor_reports:
        return "pending"

    statuses = {r.status for r in actor_reports}

    if statuses == {"succeeded"}:
        return "succeeded"

    if statuses == {"skipped"}:
        return "skipped"

    if statuses == {"failed"}:
        return "failed"

    # Mix of succeeded/failed/skipped/partial
    return "partial"


def compute_trainer_step_metrics(
    actor_reports: list[Phase4ActorTrainerReport],
) -> dict[str, Any]:
    """Compute aggregate metrics across actor updates."""
    succeeded = [r for r in actor_reports if r.status == "succeeded"]

    metrics: dict[str, Any] = {
        "num_actors_requested": len(actor_reports),
        "num_actors_succeeded": len(succeeded),
        "num_actors_failed": sum(1 for r in actor_reports if r.status == "failed"),
        "num_actors_skipped": sum(1 for r in actor_reports if r.status == "skipped"),
    }

    if succeeded:
        # Aggregate loss
        losses = [r.loss for r in succeeded if r.loss is not None]
        if losses:
            metrics["mean_loss"] = sum(losses) / len(losses)
            metrics["min_loss"] = min(losses)
            metrics["max_loss"] = max(losses)

        # Aggregate KL
        kls = [r.approx_kl for r in succeeded if r.approx_kl is not None]
        if kls:
            metrics["mean_approx_kl"] = sum(kls) / len(kls)

        # Aggregate reward
        rewards = [r.reward_mean for r in succeeded if r.reward_mean is not None]
        if rewards:
            metrics["mean_reward"] = sum(rewards) / len(rewards)

        # Per-actor metrics
        for report in succeeded:
            role_key = report.actor_role.value
            if report.loss is not None:
                metrics[f"{role_key}_loss"] = report.loss
            if report.reward_mean is not None:
                metrics[f"{role_key}_reward_mean"] = report.reward_mean
            if report.approx_kl is not None:
                metrics[f"{role_key}_approx_kl"] = report.approx_kl

    return metrics


def generate_trainer_step_run_id(
    split: DatasetSplitName, round_id: int, step_id: int
) -> str:
    """Generate a unique run ID for a trainer step."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"trainer_{split.value}_round{round_id:03d}_step{step_id:03d}_{timestamp}"


def sanitize_metric_value(value: Any) -> Any:
    """
    Sanitize metric values for JSON serialization.

    Convert NaN/Infinity to None.
    """
    import math

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Sanitize all metric values in a dictionary."""
    return {k: sanitize_metric_value(v) for k, v in metrics.items()}
