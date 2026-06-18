"""
Reward schema module.

Defines data contracts for reward reports and diagnostics.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RewardTarget(str, Enum):
    """Target agent for reward."""
    MEMORY_BUILDER = "memory_builder"
    QUESTION_AGENT = "question_agent"


class FailureType(str, Enum):
    """Types of memory failures."""
    MISSING_FACT = "missing_fact"
    WRONG_FACT = "wrong_fact"
    MISSING_ABSTRACTION = "missing_abstraction"
    WRONG_ABSTRACTION = "wrong_abstraction"
    MISSING_MULTI_SESSION_LINK = "missing_multi_session_link"
    TEMPORAL_ERROR = "temporal_error"
    CONTRADICTION_UPDATE_ERROR = "contradiction_update_error"
    OVER_COMPRESSION = "over_compression"
    REDUNDANCY_NOISE = "redundancy_noise"
    UNSUPPORTED_ANSWER = "unsupported_answer"
    ANSWERER_INSUFFICIENT_EVIDENCE = "answerer_insufficient_evidence"
    FORMAT_SAFETY_ERROR = "format_safety_error"
    LEAKAGE_RISK = "leakage_risk"
    UNKNOWN = "unknown"


class RewardComponent(BaseModel):
    """Individual reward component with weight."""
    name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    weighted_score: float = Field(ge=0.0)
    rationale: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuestionRewardItem(BaseModel):
    """Reward item for a single question."""
    question_id: str
    record_id: str
    question_type: str = "other"
    expected_answer: str = ""
    prediction: str = ""
    answer_status: str = ""
    correctness: float = Field(ge=0.0, le=1.0)
    evidence_support: float = Field(ge=0.0, le=1.0)
    citation_quality: float = Field(ge=0.0, le=1.0)
    diagnostic_value: float = Field(ge=0.0, le=1.0)
    failure_types: List[FailureType] = Field(default_factory=list)
    supporting_oracle_node_ids: List[str] = Field(default_factory=list)
    supporting_memory_ids: List[str] = Field(default_factory=list)
    related_session_ids: List[str] = Field(default_factory=list)
    rationale: str = ""


class MemoryRewardReport(BaseModel):
    """Reward report for Memory Builder."""
    record_id: str
    target: RewardTarget = RewardTarget.MEMORY_BUILDER
    total_reward: float = Field(ge=0.0, le=1.0)
    update_reward: float = Field(ge=0.0, le=1.0)
    monitoring_score: float = Field(ge=0.0, le=1.0)
    components: List[RewardComponent]
    question_items: List[QuestionRewardItem]
    failure_summary: Dict[str, int] = Field(default_factory=dict)
    weakness_updates: List[Dict[str, Any]] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuestionAgentRewardReport(BaseModel):
    """Reward report for Question Agent."""
    record_id: str
    target: RewardTarget = RewardTarget.QUESTION_AGENT
    total_reward: float = Field(ge=0.0, le=1.0)
    components: List[RewardComponent]
    per_question_scores: List[Dict[str, Any]]
    coverage_gain: float = Field(ge=0.0, le=1.0)
    adversarial_success_rate: float = Field(ge=0.0, le=1.0)
    diversity_score: float = Field(ge=0.0, le=1.0)
    config: Dict[str, Any] = Field(default_factory=dict)


def validate_memory_reward_report(
    report: dict,
    *,
    strict: bool = True,
) -> MemoryRewardReport:
    """
    Validate a memory reward report dict.

    Args:
        report: Report dict to validate
        strict: If True, enforce strict validation

    Returns:
        Validated MemoryRewardReport model

    Raises:
        ValueError: If validation fails
    """
    try:
        validated = MemoryRewardReport.model_validate(report)
    except Exception as e:
        raise ValueError(f"Memory reward report validation failed: {e}") from e

    if strict:
        assert_reward_report_no_policy_leakage(report)

    return validated


def validate_question_agent_reward_report(
    report: dict,
    *,
    strict: bool = True,
) -> QuestionAgentRewardReport:
    """
    Validate a question agent reward report dict.

    Args:
        report: Report dict to validate
        strict: If True, enforce strict validation

    Returns:
        Validated QuestionAgentRewardReport model

    Raises:
        ValueError: If validation fails
    """
    try:
        validated = QuestionAgentRewardReport.model_validate(report)
    except Exception as e:
        raise ValueError(f"Question agent reward report validation failed: {e}") from e

    return validated


def assert_reward_report_no_policy_leakage(report: dict) -> None:
    """
    Assert that reward report is not shaped like CurrentMemory.

    Reward reports may contain expected_answer and oracle fields.
    They must not be accepted as CurrentMemory artifacts.

    Args:
        report: Report dict to check

    Raises:
        ValueError: If report looks like policy input
    """
    # Reward reports must have explicit target field
    if "target" not in report:
        raise ValueError("Reward report missing required 'target' field")

    # Reward reports should not have memory_graph field
    if "memory_graph" in report:
        raise ValueError("Reward report should not contain 'memory_graph' field")

    # Reward reports should not have memory_summaries field
    if "memory_summaries" in report:
        raise ValueError("Reward report should not contain 'memory_summaries' field")

