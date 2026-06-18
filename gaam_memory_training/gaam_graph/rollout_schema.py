"""
Rollout schema module.

Defines data contracts for training rollout traces.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class StageStatus(str, Enum):
    """Status of a rollout stage."""
    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


class StageTrace(BaseModel):
    """Trace for a single rollout stage."""
    name: str
    status: StageStatus
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ArtifactPaths(BaseModel):
    """Paths to rollout artifacts."""
    graph_path: str
    current_memory_path: str | None = None
    candidate_questions_path: str | None = None
    validity_reports_path: str | None = None
    accepted_questions_path: str | None = None
    answers_path: str | None = None
    reward_report_path: str | None = None
    question_agent_reward_report_path: str | None = None
    weakness_book_path: str | None = None
    rollout_trace_path: str | None = None


class OneRoundTrace(BaseModel):
    """Complete trace for one training round."""
    round_id: int
    record_id: str
    status: StageStatus
    config: Dict[str, Any]
    artifacts: ArtifactPaths
    stages: List[StageTrace]
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class TrainingRoundSummary(BaseModel):
    """Summary for a training round."""
    round_id: int
    input_path: str
    graph_dir: str
    output_dir: str
    attempted_records: int
    succeeded_records: int
    failed_records: int
    skipped_records: int
    average_memory_update_reward: float | None = None
    average_question_agent_reward: float | None = None
    manifest_path: str
