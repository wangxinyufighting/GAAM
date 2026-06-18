"""
Question schema module.

Defines data contracts for questions and validation reports.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field, model_validator


# Forbidden keys in question artifacts
FORBIDDEN_QUESTION_ARTIFACT_KEYS = {
    "benchmark_question",
    "target_question",
    "gold_answer",
    "benchmark_answer",
    "question_type_from_benchmark",
    "oracle_gold_answer",
    "reward",
    "eval_result",
}


class QuestionType(str, Enum):
    """Types of generated questions."""
    PERSONAL_FACT = "personal_fact"
    PREFERENCE = "preference"
    INTENTION = "intention"
    ACTION = "action"
    TEMPORAL = "temporal"
    MULTI_HOP = "multi_hop"
    MULTI_SESSION = "multi_session"
    SUMMARY = "summary"
    CONTRADICTION_UPDATE = "contradiction_update"
    OTHER = "other"


class QuestionStatus(str, Enum):
    """Status of a generated question."""
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVISED = "revised"


class GeneratedQuestion(BaseModel):
    """A question generated from oracle graph evidence."""
    question_id: str
    record_id: str
    question: str
    answer: str
    question_type: QuestionType = QuestionType.OTHER
    status: QuestionStatus = QuestionStatus.CANDIDATE
    supporting_trajectory_ids: List[str] = Field(default_factory=list)
    supporting_node_ids: List[str] = Field(default_factory=list)
    supporting_session_ids: List[str] = Field(default_factory=list)
    reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_question_format(self) -> "GeneratedQuestion":
        """Validate question format."""
        # Question and answer must be non-empty
        if not self.question or not self.question.strip():
            raise ValueError(f"Question {self.question_id} has empty question text")
        if not self.answer or not self.answer.strip():
            raise ValueError(f"Question {self.question_id} has empty answer text")

        if self.status == QuestionStatus.ACCEPTED:
            # Accepted questions must have evidence
            if not self.supporting_node_ids:
                raise ValueError(f"Accepted question {self.question_id} missing supporting_node_ids")
            if not self.supporting_trajectory_ids:
                raise ValueError(f"Accepted question {self.question_id} missing supporting_trajectory_ids")

        # Question should end with ? unless it's a summary
        if self.question_type != QuestionType.SUMMARY and not self.question.rstrip().endswith("?"):
            # Allow but don't enforce for non-summary questions
            pass

        return self


class OracleValidityVerdict(str, Enum):
    """Verdict from oracle validity checking."""
    ACCEPT = "accept"
    REJECT = "reject"
    REVISE = "revise"


class OracleValidityReport(BaseModel):
    """Validity report for a generated question."""
    question_id: str
    record_id: str
    verdict: OracleValidityVerdict
    answerability: float = Field(ge=0, le=1)
    grounding: float = Field(ge=0, le=1)
    citation_validity: float = Field(ge=0, le=1)
    leakage_risk: float = Field(ge=0, le=1, default=0.0)
    issues: List[str] = Field(default_factory=list)
    supporting_node_ids: List[str] = Field(default_factory=list)
    supporting_trajectory_ids: List[str] = Field(default_factory=list)
    rationale: str = ""


class QuestionSet(BaseModel):
    """A set of generated questions with validity reports."""
    record_id: str
    graph_path: str
    generation_config: Dict[str, Any] = Field(default_factory=dict)
    questions: List[GeneratedQuestion] = Field(default_factory=list)
    validity_reports: List[OracleValidityReport] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def validate_generated_question(
    question: dict,
    *,
    strict: bool = True,
) -> GeneratedQuestion:
    """
    Validate a generated question dict.

    Args:
        question: Question dict to validate
        strict: If True, enforce all validation rules

    Returns:
        Validated GeneratedQuestion model

    Raises:
        ValueError: If validation fails
    """
    if strict:
        # Check for forbidden keys
        assert_no_question_artifact_leakage(question)

    # Validate with pydantic
    try:
        validated = GeneratedQuestion.model_validate(question)
    except Exception as e:
        raise ValueError(f"Question validation failed: {e}") from e

    return validated


def validate_question_set(
    question_set: dict,
    *,
    strict: bool = True,
) -> QuestionSet:
    """
    Validate a question set dict.

    Args:
        question_set: QuestionSet dict to validate
        strict: If True, enforce all validation rules

    Returns:
        Validated QuestionSet model

    Raises:
        ValueError: If validation fails
    """
    if strict:
        # Check for forbidden keys
        assert_no_question_artifact_leakage(question_set)

    # Validate with pydantic
    try:
        validated = QuestionSet.model_validate(question_set)
    except Exception as e:
        raise ValueError(f"QuestionSet validation failed: {e}") from e

    return validated


def assert_no_question_artifact_leakage(artifact: dict) -> None:
    """
    Assert that no forbidden keys exist in question artifact.

    Args:
        artifact: Dict to check

    Raises:
        ValueError: If forbidden keys found
    """
    _recursive_check_leakage(artifact, "", FORBIDDEN_QUESTION_ARTIFACT_KEYS)


def _recursive_check_leakage(
    obj: Any,
    path: str,
    forbidden_keys: set
) -> None:
    """Recursively check for forbidden keys."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            json_path = f"{path}.{key}" if path else key

            if key in forbidden_keys:
                raise ValueError(
                    f"Forbidden key '{key}' found at {json_path} in question artifact"
                )

            _recursive_check_leakage(value, json_path, forbidden_keys)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            json_path = f"{path}[{i}]"
            _recursive_check_leakage(item, json_path, forbidden_keys)
