"""
GRPO Data Contracts for Phase 2 Adversarial Co-Training.

This module defines JSON-safe schemas for GRPO-style policy optimization:
- Actor separation (Memory Builder vs Question Agent)
- Group-relative advantage computation
- Rollout traces and actor update batches
- Anti-leakage by construction

All models are Pydantic-based and must round-trip through JSON serialization.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field, model_validator


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty")


def _require_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


class ActorRole(str, Enum):
    """Actor role in adversarial co-training."""
    MEMORY_BUILDER = "memory_builder"
    QUESTION_AGENT = "question_agent"


class SampleStatus(str, Enum):
    """Status of a policy sample."""
    CANDIDATE = "candidate"  # generated but not yet evaluated
    ACCEPTED = "accepted"    # usable for reward / advantage
    REJECTED = "rejected"    # invalid or filtered
    FAILED = "failed"        # generation or validation failed
    SKIPPED = "skipped"      # skipped due to zero signal or config


class GroupStatus(str, Enum):
    """Status of a grouped rollout after advantage computation."""
    OK = "ok"
    ZERO_VARIANCE = "zero_variance"
    TOO_FEW_SAMPLES = "too_few_samples"
    ALL_FAILED = "all_failed"
    SKIPPED = "skipped"


class PolicySample(BaseModel):
    """
    One sampled output from a trainable actor.

    Role-specific artifact examples:
    - Memory Builder: {"current_memory": {...}, "current_memory_path": "..."}
    - Question Agent: {"candidate_questions": [...], "validity_reports": [...], "accepted_questions": [...]}
    """
    sample_id: str
    group_id: str
    record_id: str
    role: ActorRole
    status: SampleStatus = SampleStatus.CANDIDATE
    prompt: str = ""
    response: str = ""
    artifact: Dict[str, Any] = Field(default_factory=dict)
    model_id: str = ""
    temperature: float | None = None
    logprob_sum: float | None = None
    token_count: int | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fields(self) -> PolicySample:
        _require_non_empty(self.sample_id, "sample_id")
        _require_non_empty(self.group_id, "group_id")
        _require_non_empty(self.record_id, "record_id")
        if self.token_count is not None and self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.logprob_sum is not None:
            _require_finite(self.logprob_sum, "logprob_sum")
        return self


class GRPORewardItem(BaseModel):
    """Scalar reward for one sample."""
    sample_id: str
    group_id: str
    record_id: str
    role: ActorRole
    reward: float
    reward_components: Dict[str, float] = Field(default_factory=dict)
    source_report_path: str | None = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fields(self) -> GRPORewardItem:
        _require_non_empty(self.sample_id, "sample_id")
        _require_non_empty(self.group_id, "group_id")
        _require_non_empty(self.record_id, "record_id")
        _require_finite(self.reward, "reward")

        for key, value in self.reward_components.items():
            _require_finite(value, f"reward_components[{key}]")

        return self


class GRPOAdvantageItem(BaseModel):
    """Computed advantage for one sample."""
    sample_id: str
    group_id: str
    record_id: str
    role: ActorRole
    reward: float
    group_mean: float
    group_std: float
    advantage: float
    normalized: bool = True
    selected_for_update: bool = True
    rank_in_group: int | None = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fields(self) -> GRPOAdvantageItem:
        _require_non_empty(self.sample_id, "sample_id")
        _require_non_empty(self.group_id, "group_id")
        _require_non_empty(self.record_id, "record_id")
        _require_finite(self.reward, "reward")
        _require_finite(self.group_mean, "group_mean")
        _require_finite(self.group_std, "group_std")
        if self.group_std < 0:
            raise ValueError("group_std must be non-negative")
        _require_finite(self.advantage, "advantage")
        if self.rank_in_group is not None and self.rank_in_group < 1:
            raise ValueError("rank_in_group must be >= 1")

        return self


class GroupedRollout(BaseModel):
    """A group of samples and their rewards/advantages."""
    group_id: str
    record_id: str
    role: ActorRole
    status: GroupStatus = GroupStatus.OK
    samples: List[PolicySample] = Field(default_factory=list)
    rewards: List[GRPORewardItem] = Field(default_factory=list)
    advantages: List[GRPOAdvantageItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_consistency(self) -> GroupedRollout:
        _require_non_empty(self.group_id, "group_id")
        _require_non_empty(self.record_id, "record_id")

        # Check samples consistency
        for sample in self.samples:
            if sample.group_id != self.group_id:
                raise ValueError(
                    f"Sample {sample.sample_id} has group_id {sample.group_id}, "
                    f"expected {self.group_id}"
                )
            if sample.record_id != self.record_id:
                raise ValueError(
                    f"Sample {sample.sample_id} has record_id {sample.record_id}, "
                    f"expected {self.record_id}"
                )
            if sample.role != self.role:
                raise ValueError(
                    f"Sample {sample.sample_id} has role {sample.role}, "
                    f"expected {self.role}"
                )

        # Check sample_id uniqueness
        sample_ids = [s.sample_id for s in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_ids must be unique within samples")

        reward_ids = [r.sample_id for r in self.rewards]
        if len(reward_ids) != len(set(reward_ids)):
            raise ValueError("reward sample_ids must be unique within rewards")

        advantage_ids = [a.sample_id for a in self.advantages]
        if len(advantage_ids) != len(set(advantage_ids)):
            raise ValueError("advantage sample_ids must be unique within advantages")

        # Check rewards consistency
        for reward in self.rewards:
            if reward.group_id != self.group_id:
                raise ValueError(
                    f"Reward for {reward.sample_id} has group_id {reward.group_id}, "
                    f"expected {self.group_id}"
                )
            if reward.record_id != self.record_id:
                raise ValueError(
                    f"Reward for {reward.sample_id} has record_id {reward.record_id}, "
                    f"expected {self.record_id}"
                )
            if reward.role != self.role:
                raise ValueError(
                    f"Reward for {reward.sample_id} has role {reward.role}, "
                    f"expected {self.role}"
                )

        # Reward sample ids should be subset of sample ids if samples present
        if self.samples:
            reward_sample_ids = {r.sample_id for r in self.rewards}
            valid_sample_ids = set(sample_ids)
            if not reward_sample_ids.issubset(valid_sample_ids):
                invalid = reward_sample_ids - valid_sample_ids
                raise ValueError(
                    f"Reward sample_ids {invalid} not found in samples"
                )

        # Check advantages consistency
        for adv in self.advantages:
            if adv.group_id != self.group_id:
                raise ValueError(
                    f"Advantage for {adv.sample_id} has group_id {adv.group_id}, "
                    f"expected {self.group_id}"
                )
            if adv.record_id != self.record_id:
                raise ValueError(
                    f"Advantage for {adv.sample_id} has record_id {adv.record_id}, "
                    f"expected {self.record_id}"
                )
            if adv.role != self.role:
                raise ValueError(
                    f"Advantage for {adv.sample_id} has role {adv.role}, "
                    f"expected {self.role}"
                )

        # Advantage sample ids should be subset of reward ids if rewards present
        if self.rewards:
            advantage_sample_ids = {a.sample_id for a in self.advantages}
            reward_sample_ids = {r.sample_id for r in self.rewards}
            if not advantage_sample_ids.issubset(reward_sample_ids):
                invalid = advantage_sample_ids - reward_sample_ids
                raise ValueError(
                    f"Advantage sample_ids {invalid} not found in rewards"
                )

        return self


class ActorUpdateItem(BaseModel):
    """
    Sanitized update item for one actor sample.

    Should NOT contain:
    - oracle graph
    - expected answers
    - oracle supporting node ids
    - full reward report
    - benchmark target question
    - benchmark gold answer
    """
    sample_id: str
    group_id: str
    record_id: str
    role: ActorRole
    prompt: str
    response: str
    reward: float
    advantage: float
    selected_for_update: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fields(self) -> ActorUpdateItem:
        _require_non_empty(self.sample_id, "sample_id")
        _require_non_empty(self.group_id, "group_id")
        _require_non_empty(self.record_id, "record_id")
        _require_finite(self.reward, "reward")
        _require_finite(self.advantage, "advantage")
        return self


class ActorUpdateBatch(BaseModel):
    """Batch of actor update items for one training step."""
    role: ActorRole
    batch_id: str
    round_id: int
    step_id: int
    items: List[ActorUpdateItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fields(self) -> ActorUpdateBatch:
        _require_non_empty(self.batch_id, "batch_id")
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")

        item_ids = [item.sample_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("item sample_ids must be unique within items")

        # All items must have the same role as the batch
        for item in self.items:
            if item.role != self.role:
                raise ValueError(
                    f"Item {item.sample_id} has role {item.role}, "
                    f"expected batch role {self.role}"
                )

        return self


class GRPOTrainingTrace(BaseModel):
    """Full trace for one local GRPO step."""
    trace_id: str
    round_id: int
    step_id: int
    status: str
    config: Dict[str, Any] = Field(default_factory=dict)
    memory_groups: List[GroupedRollout] = Field(default_factory=list)
    question_groups: List[GroupedRollout] = Field(default_factory=list)
    memory_update_batch_path: str | None = None
    question_update_batch_path: str | None = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> GRPOTrainingTrace:
        _require_non_empty(self.trace_id, "trace_id")
        _require_non_empty(self.status, "status")
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if self.step_id < 0:
            raise ValueError("step_id must be non-negative")

        for group in self.memory_groups:
            if group.role != ActorRole.MEMORY_BUILDER:
                raise ValueError(
                    f"memory_groups contains {group.group_id} with role {group.role}, "
                    f"expected {ActorRole.MEMORY_BUILDER}"
                )
        for group in self.question_groups:
            if group.role != ActorRole.QUESTION_AGENT:
                raise ValueError(
                    f"question_groups contains {group.group_id} with role {group.role}, "
                    f"expected {ActorRole.QUESTION_AGENT}"
                )

        return self


class GRPOTrainingSummary(BaseModel):
    """Summary for one GRPO training round."""
    round_id: int
    total_steps: int
    succeeded_steps: int
    failed_steps: int
    skipped_steps: int
    average_memory_reward: float | None = None
    average_question_reward: float | None = None
    average_memory_advantage_std: float | None = None
    average_question_advantage_std: float | None = None
    trace_dir: str
    checkpoint_dir: str | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> GRPOTrainingSummary:
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if self.total_steps < 0:
            raise ValueError("total_steps must be non-negative")
        if self.succeeded_steps < 0:
            raise ValueError("succeeded_steps must be non-negative")
        if self.failed_steps < 0:
            raise ValueError("failed_steps must be non-negative")
        if self.skipped_steps < 0:
            raise ValueError("skipped_steps must be non-negative")
        if not self.trace_dir:
            raise ValueError("trace_dir cannot be empty")

        total = self.succeeded_steps + self.failed_steps + self.skipped_steps
        if total > self.total_steps:
            raise ValueError(
                f"succeeded ({self.succeeded_steps}) + failed ({self.failed_steps}) + "
                f"skipped ({self.skipped_steps}) = {total} > total_steps ({self.total_steps})"
            )

        optional_metrics = {
            "average_memory_reward": self.average_memory_reward,
            "average_question_reward": self.average_question_reward,
            "average_memory_advantage_std": self.average_memory_advantage_std,
            "average_question_advantage_std": self.average_question_advantage_std,
        }
        for field_name, value in optional_metrics.items():
            if value is not None:
                _require_finite(value, field_name)

        return self
