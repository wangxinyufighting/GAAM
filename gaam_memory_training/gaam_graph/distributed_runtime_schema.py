"""
Phase 3 Milestone 1: Distributed Runtime Schema.

This module defines all Phase 3 runtime contracts without introducing
actual distributed dependencies (Ray, verl, etc.).

Design principles:
1. Define explicit worker boundaries even for local fake-distributed execution
2. Keep Phase 2 artifact contracts as source of truth
3. Make checkpoint and job assignment explicit for resume
4. Prepare for future Ray/verl integration without requiring them now
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gaam_graph.grpo_schema import ActorRole
from gaam_graph.code_a1_alignment import AlignmentIssue


# ============================================================================
# Phase 3 Enums
# ============================================================================

class Phase3WorkerRole(str, Enum):
    """Worker roles in Phase 3 distributed runtime."""
    MEMORY_BUILDER = "memory_builder"
    QUESTION_AGENT = "question_agent"
    ANSWERER = "answerer"
    REWARD = "reward"
    ORACLE_VALIDITY = "oracle_validity"
    PLANNER = "planner"


class Phase3Backend(str, Enum):
    """Backend execution mode for Phase 3 runtime."""
    LOCAL_FAKE_DISTRIBUTED = "local_fake_distributed"
    RAY = "ray"
    VERL = "verl"
    CODE_A1_VERL = "code_a1_verl"


class Phase3JobStatus(str, Enum):
    """Status of a Phase 3 rollout job."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# Actor Runtime Config
# ============================================================================

class ActorRuntimeConfig(BaseModel):
    """Runtime configuration for a trainable actor."""
    role: ActorRole
    policy_id: str
    model_path: str | None = None
    tokenizer_path: str | None = None
    checkpoint_id: str | None = None
    checkpoint_path: str | None = None
    reference_checkpoint_id: str | None = None
    reference_checkpoint_path: str | None = None
    generation_config: dict[str, Any] = Field(default_factory=dict)
    update_config: dict[str, Any] = Field(default_factory=dict)
    resource_config: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Phase 3 Runtime Config
# ============================================================================

class Phase3RuntimeConfig(BaseModel):
    """
    Top-level configuration for Phase 3 distributed runtime.

    This config defines all inputs, outputs, worker counts, and actor configs
    needed to plan and execute a distributed rollout.
    """
    runtime_id: str
    backend: Phase3Backend = Phase3Backend.LOCAL_FAKE_DISTRIBUTED
    input_records_path: Path
    oracle_graph_dir: Path
    output_dir: Path
    checkpoint_registry_path: Path | None = None
    record_id_filter: str | None = None
    max_records: int | None = None
    round_id: int = 0
    step_id: int = 0
    seed: int = 0
    num_rollout_workers: int = 1
    num_reward_workers: int = 1
    memory_builder: ActorRuntimeConfig
    question_agent: ActorRuntimeConfig
    answerer_config: dict[str, Any] = Field(default_factory=dict)
    reward_config: dict[str, Any] = Field(default_factory=dict)
    strict_no_leakage: bool = True


# ============================================================================
# Job and Worker Specs
# ============================================================================

class Phase3RolloutJob(BaseModel):
    """
    A single rollout job assigned to workers.

    Each job processes one record through the adversarial rollout pipeline.
    """
    job_id: str
    record_id: str
    round_id: int
    step_id: int
    seed: int
    memory_checkpoint_id: str | None = None
    question_checkpoint_id: str | None = None
    assigned_rollout_worker: str
    assigned_reward_worker: str
    status: Phase3JobStatus = Phase3JobStatus.PENDING


class Phase3WorkerSpec(BaseModel):
    """Specification for a logical or physical worker."""
    worker_id: str
    role: Phase3WorkerRole
    backend: Phase3Backend
    resource_config: dict[str, Any] = Field(default_factory=dict)


class Phase3WorkerPlan(BaseModel):
    """
    Complete worker and job assignment plan.

    This plan is written by the runtime planner and consumed by workers.
    """
    runtime_id: str
    backend: Phase3Backend
    workers: list[Phase3WorkerSpec]
    jobs: list[Phase3RolloutJob]


# ============================================================================
# Runtime Trace
# ============================================================================

class Phase3RuntimeTrace(BaseModel):
    """
    Trace of a single job execution in Phase 3 runtime.

    This trace links a job to its Phase 2-compatible output artifacts.
    """
    trace_id: str
    runtime_id: str
    job_id: str
    record_id: str
    status: Phase3JobStatus
    rollout_trace_path: str | None = None
    memory_update_batch_path: str | None = None
    question_update_batch_path: str | None = None
    reward_matrix_path: str | None = None
    issues: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Readiness Report
# ============================================================================

class Phase3ReadinessReport(BaseModel):
    """
    Phase 3 runtime readiness validation report.

    This report validates that all inputs, checkpoints, and configs are
    ready for distributed execution.
    """
    report_id: str
    runtime_id: str
    status: str
    backend: Phase3Backend
    issues: list[AlignmentIssue] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
