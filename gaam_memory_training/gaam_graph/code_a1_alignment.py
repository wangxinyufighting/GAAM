"""
Code-A1 alignment layer for Phase 2 Milestone 5.

This module defines role mapping between GAAM actors and Code-A1 dual-actor training,
compatibility checks, and alignment reports.

Design principles:
1. Map GAAM's Memory Builder and Question Agent to Code-A1's Code LLM and Test LLM roles
2. Validate that actor update batches contain no forbidden teacher fields
3. Generate alignment reports without introducing Code-A1 runtime dependencies
4. Prepare for Phase 3 distributed training handoff
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any
import json
import uuid

from pydantic import BaseModel, Field

from gaam_graph.grpo_schema import ActorRole, ActorUpdateBatch


# ============================================================================
# Code-A1 Role Mapping
# ============================================================================

class CodeA1Role(str, Enum):
    """Code-A1 dual-actor role types."""
    CODE_LLM = "code_llm"
    TEST_LLM = "test_llm"


class GAAMActorMapping(BaseModel):
    """Mapping between GAAM actor and Code-A1 role."""
    gaam_role: ActorRole
    code_a1_role: CodeA1Role
    actor_name: str
    objective_summary: str
    allowed_inputs: list[str]
    forbidden_inputs: list[str]
    update_batch_role: ActorRole


class AlignmentIssueSeverity(str, Enum):
    """Severity levels for alignment issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AlignmentIssue(BaseModel):
    """An issue found during alignment validation."""
    severity: AlignmentIssueSeverity
    code: str
    message: str
    artifact_path: str | None = None
    record_id: str | None = None
    role: ActorRole | None = None


class CodeA1AlignmentReport(BaseModel):
    """Alignment report for GAAM rollout/trainer artifacts."""
    report_id: str
    rollout_dir: str
    trainer_dir: str | None = None
    status: str
    actor_mappings: list[GAAMActorMapping]
    issues: list[AlignmentIssue] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Default Actor Mappings
# ============================================================================

def build_default_actor_mappings() -> list[GAAMActorMapping]:
    """
    Build default GAAM to Code-A1 role mappings.

    Returns:
        List of actor mappings
    """
    return [
        GAAMActorMapping(
            gaam_role=ActorRole.MEMORY_BUILDER,
            code_a1_role=CodeA1Role.CODE_LLM,
            actor_name="memory_builder",
            objective_summary="Construct compact, useful, non-leaky current memory from raw conversation history",
            allowed_inputs=[
                "raw_history_derived_prompt",
                "previous_current_memory_if_iterative",
                "sanitized_reward_advantage_in_update_batch",
            ],
            forbidden_inputs=[
                "benchmark_target_question",
                "gold_answer",
                "oracle_graph",
                "oracle_supporting_node_ids",
                "reward_reports",
                "answer_reports",
            ],
            update_batch_role=ActorRole.MEMORY_BUILDER,
        ),
        GAAMActorMapping(
            gaam_role=ActorRole.QUESTION_AGENT,
            code_a1_role=CodeA1Role.TEST_LLM,
            actor_name="question_agent",
            objective_summary="Generate oracle-valid, broad, diagnostic questions that test memory quality",
            allowed_inputs=[
                "oracle_graph_trajectories",
                "sanitized_weakness_context_from_weakness_book",
                "oracle_validity_feedback_through_reward_advantage",
            ],
            forbidden_inputs=[
                "benchmark_target_question",
                "gold_answer",
                "answerer_hidden_state",
                "unsanitized_reward_reports",
            ],
            update_batch_role=ActorRole.QUESTION_AGENT,
        ),
    ]


# ============================================================================
# Forbidden Field Detection
# ============================================================================

FORBIDDEN_FIELDS_FOR_ANY_ACTOR = [
    "oracle_graph",
    "supporting_node_ids",
    "oracle_supporting_node_ids",
    "supporting_trajectory_ids",
    "gold_answer",
    "benchmark_answer",
    "benchmark_question",
    "raw_history",
    "answer_reports",
    "reward_report",
    "validity_reports",
    "expected_answer",
]


def _parse_json_like_string(value: str) -> Any | None:
    """Return parsed JSON for JSON-like strings, otherwise None."""
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _check_forbidden_fields_recursive(data: Any, path: str) -> list[str]:
    """Recursively find forbidden teacher fields in dicts, lists, and JSON strings."""
    found = []

    if isinstance(data, dict):
        for key, value in data.items():
            child_path = f"{path}.{key}" if path else key
            if key in FORBIDDEN_FIELDS_FOR_ANY_ACTOR and value is not None:
                found.append(child_path)
            found.extend(_check_forbidden_fields_recursive(value, child_path))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            found.extend(_check_forbidden_fields_recursive(item, f"{path}[{idx}]"))
    elif isinstance(data, str):
        parsed = _parse_json_like_string(data)
        if isinstance(parsed, (dict, list)):
            found.extend(_check_forbidden_fields_recursive(parsed, path))

    return found


def check_for_forbidden_fields(data: dict | ActorUpdateBatch) -> list[str]:
    """
    Check if actor update batch contains forbidden teacher fields.

    Args:
        data: Raw dict or ActorUpdateBatch to check

    Returns:
        List of forbidden field names found
    """
    # Convert to dict if needed
    if isinstance(data, ActorUpdateBatch):
        batch_dict = data.model_dump()
    else:
        batch_dict = data

    return _check_forbidden_fields_recursive(batch_dict, "batch")


def _iter_manifest_records(manifest_path: Path, max_records: int | None = None):
    """Yield manifest records with 1-based line numbers, honoring an optional limit."""
    seen = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if max_records is not None and seen >= max_records:
                break
            if not line.strip():
                continue
            seen += 1
            yield line_num, line


def _artifact_exists(path_value: str, base_dir: Path) -> bool:
    """Check absolute paths directly and relative paths against the trainer dir."""
    path = Path(path_value)
    if path.is_absolute():
        return path.exists()
    return path.exists() or (base_dir / path).exists()


# ============================================================================
# Rollout Alignment Inspection
# ============================================================================

def inspect_rollout_alignment(
    rollout_dir: Path,
    *,
    max_records: int | None = None,
) -> list[AlignmentIssue]:
    """
    Inspect rollout directory for alignment issues.

    Args:
        rollout_dir: Path to rollout directory

    Returns:
        List of alignment issues found
    """
    issues = []

    # Check manifest exists
    manifest_path = rollout_dir / "manifest.jsonl"
    if not manifest_path.exists():
        issues.append(AlignmentIssue(
            severity=AlignmentIssueSeverity.ERROR,
            code="missing_manifest",
            message="Rollout manifest.jsonl not found",
            artifact_path=str(manifest_path),
        ))
        return issues

    # Check summary exists
    summary_path = rollout_dir / "summary.json"
    if not summary_path.exists():
        issues.append(AlignmentIssue(
            severity=AlignmentIssueSeverity.WARNING,
            code="missing_summary",
            message="Rollout summary.json not found",
            artifact_path=str(summary_path),
        ))

    # Read manifest and check each record
    for line_num, line in _iter_manifest_records(manifest_path, max_records):
            try:
                record = json.loads(line.strip())
                record_id = record.get("record_id")

                # Check groups exist
                groups_dir = rollout_dir / "groups"
                if not groups_dir.exists():
                    issues.append(AlignmentIssue(
                        severity=AlignmentIssueSeverity.ERROR,
                        code="missing_groups_dir",
                        message="Groups directory not found",
                        artifact_path=str(groups_dir),
                        record_id=record_id,
                    ))
                    continue

                # Check for memory group
                memory_group_path = groups_dir / f"{record_id}.memory_group.json"
                if not memory_group_path.exists():
                    issues.append(AlignmentIssue(
                        severity=AlignmentIssueSeverity.WARNING,
                        code="missing_memory_group",
                        message=f"Memory group not found for record {record_id}",
                        artifact_path=str(memory_group_path),
                        record_id=record_id,
                        role=ActorRole.MEMORY_BUILDER,
                    ))

                # Check for question group
                question_group_path = groups_dir / f"{record_id}.question_group.json"
                if not question_group_path.exists():
                    issues.append(AlignmentIssue(
                        severity=AlignmentIssueSeverity.WARNING,
                        code="missing_question_group",
                        message=f"Question group not found for record {record_id}",
                        artifact_path=str(question_group_path),
                        record_id=record_id,
                        role=ActorRole.QUESTION_AGENT,
                    ))

                # Check update batches
                update_batches_dir = rollout_dir / "update_batches"
                if update_batches_dir.exists():
                    # Check memory batch
                    memory_batch_path = update_batches_dir / f"{record_id}.memory_update_batch.json"
                    if memory_batch_path.exists():
                        with open(memory_batch_path, "r", encoding="utf-8") as bf:
                            batch_data = json.load(bf)

                            # Check for forbidden fields in raw JSON first
                            forbidden = check_for_forbidden_fields(batch_data)
                            if forbidden:
                                issues.append(AlignmentIssue(
                                    severity=AlignmentIssueSeverity.ERROR,
                                    code="unsafe_actor_batch",
                                    message=f"Memory batch contains forbidden fields: {', '.join(forbidden)}",
                                    artifact_path=str(memory_batch_path),
                                    record_id=record_id,
                                    role=ActorRole.MEMORY_BUILDER,
                                ))

                    # Check question batch
                    question_batch_path = update_batches_dir / f"{record_id}.question_update_batch.json"
                    if question_batch_path.exists():
                        with open(question_batch_path, "r", encoding="utf-8") as bf:
                            batch_data = json.load(bf)

                            # Check for forbidden fields in raw JSON first
                            forbidden = check_for_forbidden_fields(batch_data)
                            if forbidden:
                                issues.append(AlignmentIssue(
                                    severity=AlignmentIssueSeverity.ERROR,
                                    code="unsafe_actor_batch",
                                    message=f"Question batch contains forbidden fields: {', '.join(forbidden)}",
                                    artifact_path=str(question_batch_path),
                                    record_id=record_id,
                                    role=ActorRole.QUESTION_AGENT,
                                ))

            except json.JSONDecodeError:
                issues.append(AlignmentIssue(
                    severity=AlignmentIssueSeverity.ERROR,
                    code="invalid_manifest_line",
                    message=f"Invalid JSON at manifest line {line_num}",
                    artifact_path=str(manifest_path),
                ))
            except Exception as e:
                issues.append(AlignmentIssue(
                    severity=AlignmentIssueSeverity.ERROR,
                    code="manifest_processing_error",
                    message=f"Error processing manifest line {line_num}: {str(e)}",
                    artifact_path=str(manifest_path),
                ))

    return issues


# ============================================================================
# Trainer Alignment Inspection
# ============================================================================

def inspect_trainer_alignment(
    trainer_dir: Path,
    *,
    max_records: int | None = None,
) -> list[AlignmentIssue]:
    """
    Inspect trainer directory for alignment issues.

    Args:
        trainer_dir: Path to trainer directory

    Returns:
        List of alignment issues found
    """
    issues = []

    # Check manifest exists
    manifest_path = trainer_dir / "manifest.jsonl"
    if not manifest_path.exists():
        issues.append(AlignmentIssue(
            severity=AlignmentIssueSeverity.ERROR,
            code="missing_trainer_manifest",
            message="Trainer manifest.jsonl not found",
            artifact_path=str(manifest_path),
        ))
        return issues

    # Check traces directory
    traces_dir = trainer_dir / "traces"
    if not traces_dir.exists():
        issues.append(AlignmentIssue(
            severity=AlignmentIssueSeverity.WARNING,
            code="missing_traces_dir",
            message="Trainer traces directory not found",
            artifact_path=str(traces_dir),
        ))

    # Check checkpoints directory structure
    checkpoints_dir = trainer_dir / "checkpoints"
    if not checkpoints_dir.exists():
        issues.append(AlignmentIssue(
            severity=AlignmentIssueSeverity.INFO,
            code="missing_checkpoints_dir",
            message="Checkpoints directory not found (may be dry-run)",
            artifact_path=str(checkpoints_dir),
        ))
        return issues

    # Read manifest and check each record
    for line_num, line in _iter_manifest_records(manifest_path, max_records):
            try:
                record = json.loads(line.strip())
                record_id = record.get("record_id")
                metrics = record.get("metrics") or {}

                # Check if policy was updated
                memory_updated = record.get("memory_updated", metrics.get("memory_updated", False))
                question_updated = record.get("question_updated", metrics.get("question_updated", False))

                # If updated, check for checkpoint
                if memory_updated:
                    memory_checkpoint_path = record.get("memory_checkpoint_path")
                    if not memory_checkpoint_path:
                        issues.append(AlignmentIssue(
                            severity=AlignmentIssueSeverity.ERROR,
                            code="missing_checkpoint_after_update",
                            message=f"Memory Builder updated but no checkpoint path in manifest",
                            artifact_path=str(manifest_path),
                            record_id=record_id,
                            role=ActorRole.MEMORY_BUILDER,
                        ))
                    elif not _artifact_exists(memory_checkpoint_path, trainer_dir):
                        issues.append(AlignmentIssue(
                            severity=AlignmentIssueSeverity.ERROR,
                            code="checkpoint_path_not_found",
                            message=f"Memory Builder checkpoint path does not exist: {memory_checkpoint_path}",
                            artifact_path=memory_checkpoint_path,
                            record_id=record_id,
                            role=ActorRole.MEMORY_BUILDER,
                        ))

                if question_updated:
                    question_checkpoint_path = record.get("question_checkpoint_path")
                    if not question_checkpoint_path:
                        issues.append(AlignmentIssue(
                            severity=AlignmentIssueSeverity.ERROR,
                            code="missing_checkpoint_after_update",
                            message=f"Question Agent updated but no checkpoint path in manifest",
                            artifact_path=str(manifest_path),
                            record_id=record_id,
                            role=ActorRole.QUESTION_AGENT,
                        ))
                    elif not _artifact_exists(question_checkpoint_path, trainer_dir):
                        issues.append(AlignmentIssue(
                            severity=AlignmentIssueSeverity.ERROR,
                            code="checkpoint_path_not_found",
                            message=f"Question Agent checkpoint path does not exist: {question_checkpoint_path}",
                            artifact_path=question_checkpoint_path,
                            record_id=record_id,
                            role=ActorRole.QUESTION_AGENT,
                        ))

            except json.JSONDecodeError:
                issues.append(AlignmentIssue(
                    severity=AlignmentIssueSeverity.ERROR,
                    code="invalid_trainer_manifest_line",
                    message=f"Invalid JSON at trainer manifest line {line_num}",
                    artifact_path=str(manifest_path),
                ))
            except Exception as e:
                issues.append(AlignmentIssue(
                    severity=AlignmentIssueSeverity.ERROR,
                    code="trainer_manifest_processing_error",
                    message=f"Error processing trainer manifest line {line_num}: {str(e)}",
                    artifact_path=str(manifest_path),
                ))

    return issues


# ============================================================================
# Alignment Report Builder
# ============================================================================

def build_alignment_report(
    *,
    rollout_dir: Path,
    trainer_dir: Path | None = None,
    max_records: int | None = None,
) -> CodeA1AlignmentReport:
    """
    Build comprehensive alignment report for rollout and trainer artifacts.

    Args:
        rollout_dir: Path to rollout directory
        trainer_dir: Optional path to trainer directory

    Returns:
        Alignment report with issues and metrics
    """
    report_id = str(uuid.uuid4())[:8]
    issues = []

    # Check rollout
    rollout_issues = inspect_rollout_alignment(rollout_dir, max_records=max_records)
    issues.extend(rollout_issues)

    # Check trainer if provided
    if trainer_dir:
        trainer_issues = inspect_trainer_alignment(trainer_dir, max_records=max_records)
        issues.extend(trainer_issues)
    else:
        issues.append(AlignmentIssue(
            severity=AlignmentIssueSeverity.INFO,
            code="no_trainer_dir",
            message="Trainer directory not provided, skipping trainer validation",
        ))

    # Determine status
    error_count = sum(1 for issue in issues if issue.severity == AlignmentIssueSeverity.ERROR)
    warning_count = sum(1 for issue in issues if issue.severity == AlignmentIssueSeverity.WARNING)
    info_count = sum(1 for issue in issues if issue.severity == AlignmentIssueSeverity.INFO)

    if error_count > 0:
        status = "failed"
    elif warning_count > 0:
        status = "warnings"
    else:
        status = "passed"

    # Build metrics
    metrics = {
        "total_issues": len(issues),
        "error_count": error_count,
        "warning_count": warning_count,
        "info_count": info_count,
    }

    return CodeA1AlignmentReport(
        report_id=report_id,
        rollout_dir=str(rollout_dir),
        trainer_dir=str(trainer_dir) if trainer_dir else None,
        status=status,
        actor_mappings=build_default_actor_mappings(),
        issues=issues,
        metrics=metrics,
    )
