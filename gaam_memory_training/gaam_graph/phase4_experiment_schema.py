"""
Phase 4 Milestone 5: Production Experiment Schema

Data contracts for production-grade experiments with held-out evaluation.

Key types:
- Phase4ProductionExperimentConfig: top-level experiment configuration
- Phase4ProductionExperimentManifest: full experiment artifact
- Phase4FinalEvaluationConfig: test evaluation configuration
- Phase4FinalEvaluationManifest: test evaluation results

Design principle:
Honest production readiness - never imply real training occurred when it didn't.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from gaam_graph.dataset_split_schema import DatasetSplitName
from gaam_graph.phase4_cotraining_schema import Phase4RoundStatus
from gaam_graph.phase4_trainer_schema import Phase4TrainerBackend


class Phase4ProductionExperimentConfig(BaseModel):
    """Configuration for production experiment with held-out evaluation."""

    # Required
    split_manifest_path: str
    output_dir: str
    num_rounds: int

    # Record selection
    train_records_per_round: int | None = None
    dev_records_per_round: int | None = None
    test_records_limit: int | None = None

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

    # Final evaluation
    final_test_eval: bool = True
    run_baselines: bool = True
    baseline_names: list[str] = Field(
        default_factory=lambda: ["no_llm_memory", "initial_checkpoint"]
    )

    # Execution control
    seed: int = 0
    overwrite: bool = False
    resume: bool = False
    strict_no_leakage: bool = True

    @field_validator("num_rounds")
    @classmethod
    def validate_num_rounds(cls, value: int) -> int:
        if value < 1:
            raise ValueError("num_rounds must be at least 1")
        return value

    @field_validator(
        "train_records_per_round",
        "dev_records_per_round",
        "test_records_limit",
    )
    @classmethod
    def validate_non_negative_limits(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("record limits must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_exclusive_flags(self) -> "Phase4ProductionExperimentConfig":
        if self.overwrite and self.resume:
            raise ValueError("overwrite and resume cannot both be true")
        return self


class Phase4ProductionExperimentManifest(BaseModel):
    """Top-level production experiment manifest."""

    manifest_version: str = "phase4_production_experiment_v1"
    experiment_id: str
    status: str
    started_at: str
    finished_at: str | None = None

    split_manifest_path: str
    split_manifest_hash: str | None = None
    output_dir: str

    cotraining_dir: str | None = None
    cotraining_manifest_path: str | None = None

    final_evaluation_dir: str | None = None
    final_evaluation_manifest_path: str | None = None

    final_memory_builder_checkpoint_id: str | None = None
    final_question_agent_checkpoint_id: str | None = None
    final_checkpoint_registry_path: str | None = None

    reports_dir: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Phase4FinalEvaluationConfig(BaseModel):
    """Configuration for held-out test evaluation."""

    split_manifest_path: str
    output_dir: str
    checkpoint_registry_path: str | None = None
    memory_builder_checkpoint_id: str | None = None
    question_agent_checkpoint_id: str | None = None
    memory_builder_model_path: str | None = None
    question_agent_model_path: str | None = None
    answerer_model_path: str | None = None
    test_records_limit: int | None = None
    baseline_name: str | None = None
    seed: int = 0
    overwrite: bool = False

    @field_validator("test_records_limit")
    @classmethod
    def validate_test_records_limit(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("test_records_limit must be non-negative")
        return value


class Phase4FinalEvaluationManifest(BaseModel):
    """Manifest for held-out test evaluation."""

    manifest_version: str = "phase4_final_evaluation_v1"
    eval_id: str
    split: DatasetSplitName = DatasetSplitName.TEST
    status: str
    started_at: str
    finished_at: str | None = None

    selected_record_ids: list[str]
    test_rollout_dir: str
    test_rollout_manifest_path: str | None = None

    memory_builder_checkpoint_id: str | None = None
    question_agent_checkpoint_id: str | None = None
    checkpoint_registry_path: str | None = None

    baseline_name: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    leakage_audit_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def generate_experiment_id(*, seed: int, timestamp: str | None = None) -> str:
    """
    Generate unique experiment ID.

    Format: prod_exp_seed{seed}_{timestamp}
    """
    if timestamp is None:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    return f"prod_exp_seed{seed}_{timestamp}"


def generate_eval_id(*, baseline: str | None, timestamp: str | None = None) -> str:
    """
    Generate unique evaluation ID.

    Format: final_eval_{baseline}_{timestamp} or final_eval_{timestamp}
    """
    if timestamp is None:
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if baseline:
        return f"final_eval_{baseline}_{timestamp}"
    return f"final_eval_{timestamp}"
