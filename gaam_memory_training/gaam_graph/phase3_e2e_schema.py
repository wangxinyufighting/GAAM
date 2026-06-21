"""
Phase 3 Milestone 6: End-to-End Distributed GRPO Schema

Data contracts for orchestrating the complete Phase 3 training pipeline.

Key features:
- Run config for one-command E2E execution
- Stage reports for each pipeline stage
- Round reports for multi-round training
- Final run report with aggregated metrics
- Support for local_fallback, Ray, vendored verl, Code-A1/verl backends
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Phase3E2EBackend(str, Enum):
    """Execution backend for Phase 3 E2E pipeline."""

    LOCAL_FALLBACK = "local_fallback"
    RAY = "ray"
    VENDORED_VERL = "vendored_verl"
    CODE_A1_VERL = "code_a1_verl"


class Phase3E2EStageName(str, Enum):
    """Pipeline stage names."""

    PREFLIGHT = "preflight"
    RUNTIME_PLAN = "runtime_plan"
    ROLLOUT = "rollout"
    REWARD = "reward"
    REPLAY = "replay"
    TRAINER = "trainer"
    CHECKPOINT = "checkpoint"
    INSPECTION = "inspection"
    FINALIZE = "finalize"


class Phase3E2EStageStatus(str, Enum):
    """Stage execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


class Phase3E2ERunConfig(BaseModel):
    """
    Configuration for one Phase 3 E2E GRPO run.

    MVP defaults:
    - record_id = "e47becba"
    - backend = local_fallback
    - trainer_mode = dry_run
    - round_id = 0, step_id = 0
    """

    run_id: str
    input_records_path: str
    oracle_graph_dir: str
    output_dir: str
    record_id: str = "e47becba"
    round_id: int = 0
    step_id: int = 0
    max_records: int = 1
    seed: int = 0
    backend: Phase3E2EBackend = Phase3E2EBackend.LOCAL_FALLBACK
    trainer_mode: str = "dry_run"
    reward_mode: str = "heuristic"
    num_rollout_workers: int = 1
    num_reward_workers: int = 1
    strict: bool = True
    overwrite: bool = False
    resume: bool = False
    require_real_case: bool = True
    require_memory_reward: bool = True
    require_question_reward: bool = False
    update_memory_builder: bool = True
    update_question_agent: bool = True
    write_verl_dataproto: bool = False
    use_vendored_verl_if_available: bool = False
    memory_builder_model_path: str | None = None
    question_agent_model_path: str | None = None
    answerer_model_path: str | None = None
    checkpoint_registry_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("max_records")
    @classmethod
    def validate_max_records(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_records must be at least 1")
        return v

    @model_validator(mode="after")
    def validate_exclusive_flags(self) -> "Phase3E2ERunConfig":
        if self.overwrite and self.resume:
            raise ValueError("overwrite and resume cannot both be true")
        return self

    @field_validator("metadata")
    @classmethod
    def validate_no_leakage_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure metadata doesn't contain forbidden oracle/benchmark fields."""
        from gaam_graph.code_a1_alignment import check_for_forbidden_fields

        forbidden = check_for_forbidden_fields(v)
        if forbidden:
            raise ValueError(f"metadata contains forbidden fields: {forbidden}")
        return v


class Phase3E2EStageReport(BaseModel):
    """Report for one pipeline stage execution."""

    stage: Phase3E2EStageName
    status: Phase3E2EStageStatus
    started_at: str
    finished_at: str | None = None
    output_dir: str | None = None
    primary_artifact_path: str | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Phase3E2ERoundReport(BaseModel):
    """Report for one training round (round_id, step_id)."""

    round_id: int
    step_id: int
    record_ids: list[str]
    status: Phase3E2EStageStatus
    rollout_report_path: str | None = None
    reward_summary_path: str | None = None
    replay_buffer_path: str | None = None
    trainer_report_path: str | None = None
    checkpoint_registry_path: str | None = None
    memory_reward_mean: float | None = None
    question_reward_mean: float | None = None
    num_replay_items: int = 0
    num_weakness_updates: int = 0
    no_leakage_passed: bool = False
    stage_reports: list[Phase3E2EStageReport] = Field(default_factory=list)


class Phase3E2ERunReport(BaseModel):
    """Final report for complete E2E run."""

    run_id: str
    status: Phase3E2EStageStatus
    backend: Phase3E2EBackend
    output_dir: str
    record_id: str
    round_reports: list[Phase3E2ERoundReport]
    final_checkpoint_registry_path: str | None = None
    final_replay_buffer_path: str | None = None
    final_weakness_book_path: str | None = None
    final_summary_path: str | None = None
    no_leakage_passed: bool
    remote_gpu_ready: bool = False
    code_a1_verl_ready: bool = False
    remote_gpu_readiness_gaps: list[str] = Field(default_factory=list)
    code_a1_verl_readiness_gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: str
    finished_at: str | None = None


def create_stage_report(
    stage: Phase3E2EStageName,
    status: Phase3E2EStageStatus,
    started_at: str | None = None,
) -> Phase3E2EStageReport:
    """Helper to create a stage report with current timestamp."""
    if started_at is None:
        started_at = datetime.utcnow().isoformat()

    return Phase3E2EStageReport(
        stage=stage,
        status=status,
        started_at=started_at,
    )


def finalize_stage_report(
    report: Phase3E2EStageReport,
    status: Phase3E2EStageStatus | None = None,
) -> Phase3E2EStageReport:
    """Helper to finalize a stage report with current timestamp."""
    report.finished_at = datetime.utcnow().isoformat()
    if status is not None:
        report.status = status
    return report
