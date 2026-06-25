"""
Phase 4 Milestone 6: Benchmark Suite Schema

Data contracts for benchmark suite execution, aggregation, and release.
"""

import hashlib
import time
from typing import Any

from pydantic import BaseModel, Field


class Phase4BenchmarkSuiteConfig(BaseModel):
    """Configuration for a benchmark suite."""

    suite_name: str
    split_manifest_path: str
    output_dir: str
    seeds: list[int]

    num_rounds: int
    train_records_per_round: int | None = None
    dev_records_per_round: int | None = None
    test_records_limit: int | None = None

    rollout_backend: str = "local_fallback"
    trainer_backend: str = "dry_run"
    trainer_mode: str = "dry_run"

    memory_builder_model_path: str | None = None
    question_agent_model_path: str | None = None
    answerer_model_path: str | None = None

    final_test_eval: bool = True
    run_baselines: bool = True
    baseline_names: list[str] = Field(
        default_factory=lambda: ["no_llm_memory", "initial_checkpoint"]
    )

    strict_no_leakage: bool = True
    require_cuda: bool = False

    resume: bool = False
    overwrite: bool = False
    force_rerun_succeeded: bool = False
    allow_empty_batch: bool = False

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if not self.suite_name:
            issues.append("suite_name must be non-empty")

        if not self.seeds:
            issues.append("seeds must be non-empty")

        if len(self.seeds) != len(set(self.seeds)):
            issues.append("seeds must be unique")

        if self.num_rounds < 1:
            issues.append("num_rounds must be >= 1")

        if self.train_records_per_round is not None and self.train_records_per_round < 0:
            issues.append("train_records_per_round must be non-negative")

        if self.dev_records_per_round is not None and self.dev_records_per_round < 0:
            issues.append("dev_records_per_round must be non-negative")

        if self.test_records_limit is not None and self.test_records_limit < 0:
            issues.append("test_records_limit must be non-negative")

        if self.overwrite and self.resume:
            issues.append("overwrite and resume cannot both be true")

        return issues


class Phase4SuiteRunRecord(BaseModel):
    """Record of one seed run within a benchmark suite."""

    seed: int
    status: str
    output_dir: str
    experiment_manifest_path: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    final_memory_builder_checkpoint_id: str | None = None
    final_question_agent_checkpoint_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Phase4BenchmarkSuiteManifest(BaseModel):
    """Manifest tracking benchmark suite execution."""

    manifest_version: str = "phase4_benchmark_suite_v1"
    suite_id: str
    suite_name: str
    status: str
    started_at: str
    finished_at: str | None = None

    suite_config_path: str | None = None
    resolved_config_path: str
    output_dir: str

    run_records: list[Phase4SuiteRunRecord] = Field(default_factory=list)
    aggregate_dir: str | None = None
    release_dir: str | None = None

    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Phase4AggregateMetrics(BaseModel):
    """Aggregated metrics across multiple seeds."""

    suite_id: str
    num_seeds_requested: int
    num_seeds_succeeded: int
    num_seeds_partial: int
    num_seeds_failed: int

    final_metrics_by_method: list[dict[str, Any]] = Field(default_factory=list)
    aggregate_by_method: dict[str, dict[str, Any]] = Field(default_factory=dict)
    statistical_tests: dict[str, Any] = Field(default_factory=dict)
    leakage_summary: dict[str, Any] = Field(default_factory=dict)


def aggregate_suite_status(run_records: list[Phase4SuiteRunRecord]) -> str:
    """
    Aggregate suite status from run records.

    Rules:
    - all succeeded -> succeeded
    - some succeeded and some partial/failed -> partial
    - all failed -> failed
    - interrupted but resumable -> partial
    """
    if not run_records:
        return "pending"

    statuses = [r.status for r in run_records]

    if all(s == "succeeded" for s in statuses):
        return "succeeded"

    if all(s == "failed" for s in statuses):
        return "failed"

    return "partial"


def generate_suite_id(
    *,
    suite_name: str,
    timestamp: str | None = None,
) -> str:
    """
    Generate unique suite ID.

    Format: {suite_name}_{timestamp_hash}
    """
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    # Use short hash of timestamp for uniqueness
    hash_suffix = hashlib.sha256(timestamp.encode()).hexdigest()[:8]

    return f"{suite_name}_{hash_suffix}"
