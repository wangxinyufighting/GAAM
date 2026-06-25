"""
Phase 4 Milestone 2: Multi-Case Rollout Schema

Defines data contracts for multi-case GRPO rollout execution and aggregation.

Key types:
- MultiCaseRolloutStatus: rollout execution status
- MultiCaseRecordRun: per-record execution result
- MultiCaseRolloutManifest: top-level audit artifact
- AggregatedReplayManifest: split-level replay buffer metadata
- ActorBatchManifest: actor-specific update batch metadata

Design principle:
Multi-case aggregation is a first-class training artifact, not merely
a folder of independent per-record results.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gaam_graph.dataset_split_schema import DatasetSplitIssue, DatasetSplitName


class MultiCaseRolloutStatus(str, Enum):
    """Status of multi-case rollout execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class MultiCaseRecordRun(BaseModel):
    """Per-record execution result in a multi-case rollout."""

    record_id: str
    split: DatasetSplitName
    status: MultiCaseRolloutStatus
    output_dir: str
    final_report_path: str | None = None
    replay_item_path: str | None = None
    memory_update_batch_path: str | None = None
    question_update_batch_path: str | None = None
    dataproto_dir: str | None = None
    no_leakage_passed: bool = False
    memory_reward_mean: float | None = None
    question_reward_mean: float | None = None
    replay_items: int = 0
    weakness_updates: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class MultiCaseRolloutManifest(BaseModel):
    """Top-level audit artifact for a multi-case rollout run."""

    manifest_version: str = "phase4_multi_case_rollout_v1"
    run_id: str
    split_manifest_path: str
    split: DatasetSplitName
    selected_record_ids: list[str]
    round_id: int
    step_id: int
    backend: str
    trainer_mode: str
    output_dir: str
    started_at: str
    finished_at: str | None = None
    status: MultiCaseRolloutStatus
    records: list[MultiCaseRecordRun] = Field(default_factory=list)
    aggregate_paths: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    issues: list[DatasetSplitIssue] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class AggregatedReplayManifest(BaseModel):
    """Metadata for split-level replay buffer aggregation."""

    manifest_version: str = "phase4_aggregated_replay_v1"
    source_rollout_manifest_path: str
    replay_buffer_path: str
    selected_record_ids: list[str]
    num_source_records: int
    num_replay_items: int
    num_train_items: int
    num_eval_only_items: int
    no_leakage_pass_count: int
    skipped_records: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class ActorBatchManifest(BaseModel):
    """Metadata for aggregated actor update batches."""

    manifest_version: str = "phase4_actor_batch_v1"
    source_rollout_manifest_path: str
    actor_role: str  # "memory_builder" or "question_agent"
    split: DatasetSplitName
    batch_path: str
    record_ids: list[str]
    num_groups: int
    num_samples: int
    num_selected_samples: int
    reward_mean: float | None = None
    reward_std: float | None = None
    no_leakage_passed: bool
    missing_records: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


def aggregate_record_status(
    record_runs: list[MultiCaseRecordRun],
) -> MultiCaseRolloutStatus:
    """
    Aggregate per-record statuses into a split-level status.

    Rules:
    - all succeeded -> succeeded
    - all failed -> failed
    - mix -> partial
    - any running -> running
    - all pending -> pending
    """
    if not record_runs:
        return MultiCaseRolloutStatus.PENDING

    statuses = {r.status for r in record_runs}

    if MultiCaseRolloutStatus.RUNNING in statuses:
        return MultiCaseRolloutStatus.RUNNING

    if statuses == {MultiCaseRolloutStatus.SUCCEEDED}:
        return MultiCaseRolloutStatus.SUCCEEDED

    if statuses == {MultiCaseRolloutStatus.FAILED}:
        return MultiCaseRolloutStatus.FAILED

    if statuses == {MultiCaseRolloutStatus.PENDING}:
        return MultiCaseRolloutStatus.PENDING

    # Mix of succeeded/failed/partial
    return MultiCaseRolloutStatus.PARTIAL


def split_counts(
    record_runs: list[MultiCaseRecordRun],
) -> dict[MultiCaseRolloutStatus, int]:
    """Count records by status."""
    counts: dict[MultiCaseRolloutStatus, int] = {
        MultiCaseRolloutStatus.PENDING: 0,
        MultiCaseRolloutStatus.RUNNING: 0,
        MultiCaseRolloutStatus.SUCCEEDED: 0,
        MultiCaseRolloutStatus.PARTIAL: 0,
        MultiCaseRolloutStatus.FAILED: 0,
    }
    for r in record_runs:
        counts[r.status] += 1
    return counts


def compute_aggregate_metrics(
    record_runs: list[MultiCaseRecordRun],
) -> dict[str, Any]:
    """Compute aggregate metrics across records."""
    succeeded = [r for r in record_runs if r.status == MultiCaseRolloutStatus.SUCCEEDED]

    if not succeeded:
        return {
            "num_records": len(record_runs),
            "num_succeeded": 0,
            "num_failed": sum(
                1
                for r in record_runs
                if r.status == MultiCaseRolloutStatus.FAILED
            ),
            "num_partial": sum(
                1
                for r in record_runs
                if r.status == MultiCaseRolloutStatus.PARTIAL
            ),
        }

    memory_rewards = [r.memory_reward_mean for r in succeeded if r.memory_reward_mean is not None]
    question_rewards = [r.question_reward_mean for r in succeeded if r.question_reward_mean is not None]

    metrics: dict[str, Any] = {
        "num_records": len(record_runs),
        "num_succeeded": len(succeeded),
        "num_failed": sum(
            1 for r in record_runs if r.status == MultiCaseRolloutStatus.FAILED
        ),
        "num_partial": sum(
            1 for r in record_runs if r.status == MultiCaseRolloutStatus.PARTIAL
        ),
        "total_replay_items": sum(r.replay_items for r in succeeded),
        "total_weakness_updates": sum(r.weakness_updates for r in succeeded),
        "no_leakage_pass_rate": sum(1 for r in succeeded if r.no_leakage_passed)
        / len(succeeded),
    }

    if memory_rewards:
        metrics["memory_reward_mean"] = sum(memory_rewards) / len(memory_rewards)
        metrics["memory_reward_min"] = min(memory_rewards)
        metrics["memory_reward_max"] = max(memory_rewards)

    if question_rewards:
        metrics["question_reward_mean"] = sum(question_rewards) / len(question_rewards)
        metrics["question_reward_min"] = min(question_rewards)
        metrics["question_reward_max"] = max(question_rewards)

    return metrics


def generate_run_id(split: DatasetSplitName, round_id: int, step_id: int) -> str:
    """Generate a unique run ID for a multi-case rollout."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{split.value}_round{round_id:03d}_step{step_id:03d}_{timestamp}"
