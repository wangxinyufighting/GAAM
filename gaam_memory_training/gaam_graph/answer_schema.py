"""
Answer schema module.

Defines data contracts for answer requests and reports.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, List, Set

from pydantic import BaseModel, Field, model_validator


# Forbidden keys in answer input
FORBIDDEN_ANSWER_INPUT_KEYS = {
    "gold_answer",
    "answer",
    "benchmark_answer",
    "expected_answer",
    "oracle_answer",
    "oracle_graph",
    "oracle_node_ids",
    "oracle_supporting_node_ids",
    "supporting_node_ids",  # Oracle supporting node IDs (not CurrentMemory IDs)
    "supporting_trajectory_ids",
    "supporting_session_ids",
    "validity_reports",
    "question_type",
    "haystack_question_type",
    "reward",
    "eval_result",
    "raw_history",
}


class AnswerStatus(str, Enum):
    """Status of an answer."""
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REFUSED_UNSUPPORTED = "refused_unsupported"
    INVALID_OUTPUT = "invalid_output"


class AnswerRequest(BaseModel):
    """Request to answer a question from current memory."""
    record_id: str
    question_id: str
    question: str
    current_memory_path: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fields(self) -> "AnswerRequest":
        """Validate request fields."""
        if not self.record_id or not self.record_id.strip():
            raise ValueError("record_id cannot be empty")
        if not self.question_id or not self.question_id.strip():
            raise ValueError("question_id cannot be empty")
        if not self.question or not self.question.strip():
            raise ValueError("question cannot be empty")
        assert_no_answer_input_leakage(self.model_dump())
        return self


class RetrievedMemoryEvidence(BaseModel):
    """Evidence retrieved from current memory."""
    memory_id: str
    memory_type: str
    content: str
    score: float = Field(ge=0, le=1)
    source_session_ids: List[str] = Field(default_factory=list)
    source_turn_ids: List[str] = Field(default_factory=list)
    source_event_ids: List[str] = Field(default_factory=list)


class AnswerReport(BaseModel):
    """Report for an answered question."""
    record_id: str
    question_id: str
    question: str
    prediction: str
    answer_status: AnswerStatus
    confidence: float = Field(ge=0, le=1)
    supporting_memory_ids: List[str] = Field(default_factory=list)
    supporting_evidence: List[RetrievedMemoryEvidence] = Field(default_factory=list)
    rationale: str = ""
    model_info: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_report_fields(self) -> "AnswerReport":
        """Validate report fields and guard against leaked evaluator fields."""
        assert_no_answer_input_leakage(self.model_dump())
        return self


def validate_answer_request(
    request: dict,
    *,
    strict: bool = True,
) -> AnswerRequest:
    """
    Validate an answer request dict.

    Args:
        request: Request dict to validate
        strict: If True, check for forbidden keys

    Returns:
        Validated AnswerRequest model

    Raises:
        ValueError: If validation fails
    """
    if strict:
        assert_no_answer_input_leakage(request)

    try:
        validated = AnswerRequest.model_validate(request)
    except Exception as e:
        raise ValueError(f"Answer request validation failed: {e}") from e

    return validated


def validate_answer_report(
    report: dict,
    *,
    strict: bool = True,
) -> AnswerReport:
    """
    Validate an answer report dict.

    Args:
        report: Report dict to validate
        strict: If True, enforce all validation rules

    Returns:
        Validated AnswerReport model

    Raises:
        ValueError: If validation fails
    """
    if strict:
        assert_no_answer_input_leakage(report)

    try:
        validated = AnswerReport.model_validate(report)
    except Exception as e:
        raise ValueError(f"Answer report validation failed: {e}") from e

    return validated


def validate_supporting_memory_ids(
    supporting_memory_ids: Iterable[str],
    *,
    allowed_memory_ids: Set[str],
) -> List[str]:
    """
    Validate supporting memory IDs returned by an Answerer.

    Answer reports may only cite memory IDs that came from the retrieved
    CurrentMemory evidence. This prevents LLM mode from inventing IDs or
    leaking oracle node IDs into downstream reward attribution.
    """
    validated_ids = []
    invalid_ids = []

    for memory_id in supporting_memory_ids:
        if not isinstance(memory_id, str) or not memory_id.strip():
            invalid_ids.append(str(memory_id))
            continue

        if memory_id not in allowed_memory_ids:
            invalid_ids.append(memory_id)
            continue

        validated_ids.append(memory_id)

    if invalid_ids:
        invalid_display = ", ".join(invalid_ids)
        raise ValueError(f"Invalid supporting_memory_ids: {invalid_display}")

    return validated_ids


def assert_no_answer_input_leakage(payload: dict) -> None:
    """
    Assert that no forbidden keys exist in answer input.

    Args:
        payload: Dict to check

    Raises:
        ValueError: If forbidden keys found
    """
    _recursive_check_leakage(payload, "", FORBIDDEN_ANSWER_INPUT_KEYS)


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
                    f"Forbidden key '{key}' found at {json_path} in answer input"
                )

            _recursive_check_leakage(value, json_path, forbidden_keys)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            json_path = f"{path}[{i}]"
            _recursive_check_leakage(item, json_path, forbidden_keys)


def answer_request_from_generated_question(question: dict) -> AnswerRequest:
    """
    Create safe AnswerRequest from generated question dict.

    This function intentionally drops:
    - answer
    - supporting_node_ids (oracle IDs)
    - supporting_trajectory_ids
    - supporting_session_ids
    - reason
    - Most metadata

    Args:
        question: Generated question dict

    Returns:
        Safe AnswerRequest with only question text
    """
    return AnswerRequest(
        record_id=question["record_id"],
        question_id=question["question_id"],
        question=question["question"],
        metadata={},
    )
