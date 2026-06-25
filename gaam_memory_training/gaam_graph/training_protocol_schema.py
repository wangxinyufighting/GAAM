"""
Phase 4 Milestone 1: Training Protocol Schema

Data contracts for multi-case training execution with split-aware policies.

Key principles:
- Explicit input policies for each actor (Memory Builder, Question Agent, Answerer)
- Train/dev/test actor update policies
- No leakage of benchmark questions into Memory Builder
- Aggregated split-level run reports
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from gaam_graph.dataset_split_schema import DatasetSplitIssue, DatasetSplitName


class TrainingProtocolPolicy(BaseModel):
    """Policy specification for what each actor can see and do.

    This is the audit trail for leakage-safe training.
    """

    memory_builder_allowed_inputs: list[str] = Field(
        default_factory=lambda: ["raw_history", "previous_current_memory", "checkpoint"]
    )
    memory_builder_forbidden_inputs: list[str] = Field(
        default_factory=lambda: [
            "benchmark_target_question",
            "benchmark_answer",
            "oracle_graph",
            "generated_training_questions",
            "question_type",
        ]
    )
    question_agent_allowed_inputs: list[str] = Field(
        default_factory=lambda: [
            "oracle_graph",
            "graph_walks",
            "weakness_book",
            "previous_question_reward",
        ]
    )
    reward_allowed_inputs: list[str] = Field(
        default_factory=lambda: [
            "answers",
            "generated_questions",
            "oracle_validity_metadata",
            "memory_quality_metrics",
            "oracle_graph",
        ]
    )
    benchmark_question_policy: str = "available_only_after_memory_construction"
    actor_update_policy: dict[str, bool] = Field(
        default_factory=lambda: {
            "memory_builder": True,
            "question_agent": True,
        }
    )


class TrainingProtocolManifest(BaseModel):
    """Protocol manifest for each Phase 4 split-level run.

    Written before rollout starts as audit trail.
    """

    protocol_version: str = "phase4_training_protocol_v1"
    run_id: str
    split_manifest_path: str
    active_split: DatasetSplitName
    selected_record_ids: list[str]
    max_records: int | None = None
    memory_builder_input_policy: str
    question_agent_input_policy: str
    answerer_input_policy: str
    reward_input_policy: str
    no_leakage_policy: str
    backend: str
    trainer_mode: str
    output_dir: str
    created_at: str
    policy: TrainingProtocolPolicy = Field(default_factory=TrainingProtocolPolicy)


class SplitRunRecordReport(BaseModel):
    """Per-record execution report within a split run."""

    record_id: str
    split: DatasetSplitName
    status: str
    backend: str
    no_leakage_passed: bool
    remote_gpu_ready: bool | None = None
    code_a1_verl_ready: bool | None = None
    memory_reward_mean: float | None = None
    question_reward_mean: float | None = None
    replay_items: int = 0
    weakness_updates: int = 0
    output_dir: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SplitRunReport(BaseModel):
    """Aggregated report for split-level run across multiple records."""

    run_id: str
    split_manifest_path: str
    split: DatasetSplitName
    status: str
    selected_record_ids: list[str]
    records: list[SplitRunRecordReport]
    metrics: dict[str, Any] = Field(default_factory=dict)
    issues: list[DatasetSplitIssue] = Field(default_factory=list)
    protocol_manifest_path: str | None = None
    started_at: str
    finished_at: str | None = None


# ============================================================================
# Default Policies
# ============================================================================


def get_default_policy_for_split(split: DatasetSplitName) -> TrainingProtocolPolicy:
    """Get default actor update policy for split.

    Train: update both Memory Builder and Question Agent
    Dev: no actor updates (checkpoint selection only)
    Test: no actor updates (final evaluation only)
    """
    policy = TrainingProtocolPolicy()

    if split == DatasetSplitName.TRAIN:
        policy.actor_update_policy = {
            "memory_builder": True,
            "question_agent": True,
        }
        policy.benchmark_question_policy = "forbidden_during_memory_construction"
    elif split == DatasetSplitName.DEV:
        policy.actor_update_policy = {
            "memory_builder": False,
            "question_agent": False,
        }
        policy.benchmark_question_policy = (
            "allowed_after_memory_construction_for_eval"
        )
    elif split == DatasetSplitName.TEST:
        policy.actor_update_policy = {
            "memory_builder": False,
            "question_agent": False,
        }
        policy.benchmark_question_policy = (
            "allowed_after_memory_construction_for_official_eval"
        )

    return policy
