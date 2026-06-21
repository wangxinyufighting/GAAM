"""
Phase 3 Milestone 5: Distributed Reward Schema

Data contracts for distributed reward service, weakness book deltas,
and replay buffer items.

Key features:
- RewardWorkerInput: file-path contract for reward computation
- RewardWorkerReport: terminal status report
- WeaknessBookDelta: mergeable weakness updates
- ReplayBufferItem: trainer-safe replay item
- All schemas are JSON-serializable and stable
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# Reward Worker Status
# ============================================================================


class RewardWorkerStatus(str, Enum):
    """Status of reward worker execution."""

    SUCCEEDED = "succeeded"  # Full reward computation completed
    PARTIAL = "partial"  # Useful diagnostics but incomplete reward
    SKIPPED = "skipped"  # No work to do (missing inputs)
    FAILED = "failed"  # Malformed inputs or exceptions


# ============================================================================
# Reward Worker Input
# ============================================================================


class RewardWorkerInput(BaseModel):
    """Input contract for reward worker (file-path based)."""

    job_id: str
    record_id: str
    round_id: int = 0
    step_id: int = 0

    # Input artifact paths
    rollout_job_dir: str | None = None
    oracle_graph_path: str
    current_memory_path: str | None = None
    question_set_path: str | None = None
    answer_report_path: str | None = None
    trainer_adapter_report_path: str | None = None
    weakness_book_path: str | None = None

    # Output
    output_dir: str

    # Execution options
    reward_mode: str = "heuristic"  # heuristic or llm_judge
    compute_memory_builder_reward: bool = True
    compute_question_agent_reward: bool = True
    update_weakness_book: bool = True
    write_replay_item: bool = True

    # Optional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Reward Worker Report
# ============================================================================


class RewardWorkerReport(BaseModel):
    """Report from reward worker execution."""

    report_id: str
    status: RewardWorkerStatus
    record_id: str
    job_id: str
    round_id: int
    step_id: int
    output_dir: str

    # Output artifact paths
    memory_reward_path: str | None = None
    question_reward_path: str | None = None
    weakness_book_path: str | None = None
    weakness_delta_path: str | None = None
    replay_item_path: str | None = None

    # Scalar rewards
    memory_update_reward: float | None = None
    memory_monitoring_score: float | None = None
    question_agent_reward: float | None = None

    # Weakness updates
    weakness_update_count: int = 0
    failure_type_counts: dict[str, int] = Field(default_factory=dict)

    # Diagnostics
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    # Timestamps
    started_at: str | None = None
    finished_at: str | None = None


# ============================================================================
# Weakness Book Delta
# ============================================================================


class WeaknessBookDelta(BaseModel):
    """Mergeable delta for distributed Weakness Book updates."""

    delta_id: str
    record_id: str
    source_report_id: str
    round_id: int
    step_id: int

    # Weakness ID changes
    added_weakness_ids: list[str] = Field(default_factory=list)
    updated_weakness_ids: list[str] = Field(default_factory=list)
    mastered_weakness_ids: list[str] = Field(default_factory=list)

    # Weakness updates (must not contain oracle answers)
    weakness_updates: list[dict[str, Any]] = Field(default_factory=list)

    # Metadata
    created_at: str | None = None


# ============================================================================
# Replay Buffer Item
# ============================================================================


class ReplayBufferItem(BaseModel):
    """Trainer-safe replay buffer item."""

    replay_id: str
    record_id: str
    job_id: str
    round_id: int
    step_id: int

    # Checkpoint IDs
    memory_builder_checkpoint_id: str | None = None
    question_agent_checkpoint_id: str | None = None

    # Scalar rewards
    memory_reward: float | None = None
    question_reward: float | None = None
    monitoring_score: float | None = None

    # Artifact paths (references, not embeddings)
    memory_reward_report_path: str | None = None
    question_reward_report_path: str | None = None
    trainer_ready_batch_paths: dict[str, str] = Field(default_factory=dict)
    weakness_delta_path: str | None = None

    # Diagnostics
    diagnostic_tags: list[str] = Field(default_factory=list)
    no_leakage_passed: bool = True
    selected_for_training: bool = True

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


# ============================================================================
# Replay Buffer Manifest
# ============================================================================


class ReplayBufferManifestEntry(BaseModel):
    """Manifest entry for replay buffer."""

    replay_id: str
    record_id: str
    job_id: str
    round_id: int
    step_id: int
    status: str
    memory_reward: float | None = None
    question_reward: float | None = None
    selected_for_training: bool = True
    created_at: str | None = None


# ============================================================================
# Distributed Reward Summary
# ============================================================================


class DistributedRewardSummary(BaseModel):
    """Summary of distributed reward service execution."""

    summary_id: str
    backend: str  # local_fallback or ray
    reward_mode: str
    num_jobs: int
    succeeded: int = 0
    partial: int = 0
    skipped: int = 0
    failed: int = 0
    total_weakness_updates: int = 0
    total_replay_items: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
