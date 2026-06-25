"""
Phase 4 Milestone 1: Dataset Split Schema

Data contracts for train/dev/test split creation, validation, and inspection.

Key principles:
- Deterministic split assignment from seed
- Explicit split files for reproducibility
- No leakage of benchmark questions or answers into split metadata
- Self-contained split records with validation metadata
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DatasetSplitName(str, Enum):
    """Split names for train/dev/test protocol."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class DatasetSplitIssueSeverity(str, Enum):
    """Issue severity levels for split validation."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DatasetSplitIssue(BaseModel):
    """Validation issue discovered during split creation or inspection."""

    severity: DatasetSplitIssueSeverity
    code: str
    message: str
    record_id: str | None = None
    path: str | None = None


FORBIDDEN_METADATA_KEYS = {
    "target_question",
    "benchmark_question",
    "gold_answer",
    "benchmark_answer",
    "target_answer",
    "question_text",
    "answer_text",
    "supporting_node_content",
    "oracle_answer",
}

_NORMALIZED_FORBIDDEN_METADATA_KEYS = {
    re.sub(r"[^a-z0-9]", "", key.lower()) for key in FORBIDDEN_METADATA_KEYS
}


def _normalize_metadata_key(key: Any) -> str:
    """Normalize metadata keys for leakage-sensitive key matching."""
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def find_leakage_sensitive_metadata_paths(
    value: Any,
    *,
    path: str = "metadata",
) -> list[str]:
    """Find leakage-sensitive metadata keys recursively.

    The split manifest is an audit artifact, so this check intentionally
    catches nested and case/style variants such as TargetQuestion,
    target-question, and nested.target_question.
    """
    found: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _normalize_metadata_key(key) in _NORMALIZED_FORBIDDEN_METADATA_KEYS:
                found.append(child_path)
            found.extend(find_leakage_sensitive_metadata_paths(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                find_leakage_sensitive_metadata_paths(
                    child,
                    path=f"{path}[{index}]",
                )
            )

    return found


class DatasetSplitRecord(BaseModel):
    """Single record assignment in a dataset split.

    Important: This must not store benchmark question text, answer text,
    or any other leakage-sensitive content. Only metadata and paths.
    """

    record_id: str
    split: DatasetSplitName
    source_index: int | None = None
    raw_record_path: str
    oracle_graph_path: str
    has_benchmark_question: bool = False
    has_benchmark_answer: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_no_leakage_in_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure metadata does not contain leakage-sensitive fields."""
        found = find_leakage_sensitive_metadata_paths(v)
        if found:
            raise ValueError(
                "Split record metadata contains forbidden leakage-sensitive "
                f"keys at: {found}"
            )
        return v


class DatasetSplitManifest(BaseModel):
    """Complete dataset split specification.

    This is the reusable source of truth for train/dev/test splits.
    Can be shared across machines and runs for reproducibility.
    """

    manifest_version: str = "phase4_dataset_split_v1"
    dataset_name: str = "longmemeval"
    source_records_path: str
    oracle_graph_dir: str
    seed: int
    split_ratios: dict[DatasetSplitName, float]
    created_at: str
    records: list[DatasetSplitRecord]
    counts: dict[DatasetSplitName, int]
    record_id_hash: str
    issues: list[DatasetSplitIssue] = Field(default_factory=list)

    @field_validator("split_ratios")
    @classmethod
    def validate_ratios_sum(cls, v: dict[DatasetSplitName, float]) -> dict[DatasetSplitName, float]:
        """Validate that split ratios approximately sum to 1.0.

        Allow 0.0 sum for empty manifests (e.g., when all records filtered out).
        """
        total = sum(v.values())
        if total > 0.0 and not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Split ratios must sum to approximately 1.0, got {total:.3f}"
            )
        return v


# ============================================================================
# Helper Functions
# ============================================================================


def split_counts(records: list[DatasetSplitRecord]) -> dict[DatasetSplitName, int]:
    """Compute split counts from records."""
    counts = {split: 0 for split in DatasetSplitName}
    for record in records:
        counts[record.split] += 1
    return counts


def compute_split_assignment_hash(records: list[DatasetSplitRecord]) -> str:
    """Compute deterministic hash of split assignments.

    Hash is computed over sorted (split, record_id) pairs to detect
    split drift between local and remote runs.
    """
    lines = []
    sorted_records = sorted(records, key=lambda r: (r.split.value, r.record_id))
    for record in sorted_records:
        lines.append(f"{record.split.value}:{record.record_id}")
    content = "\n".join(lines) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ============================================================================
# Issue Code Registry
# ============================================================================

# Standard issue codes for split validation
ISSUE_MISSING_RAW_RECORD = "missing_raw_record"
ISSUE_MISSING_ORACLE_GRAPH = "missing_oracle_graph"
ISSUE_DUPLICATE_RECORD_ID = "duplicate_record_id"
ISSUE_DUPLICATE_SPLIT_ASSIGNMENT = "duplicate_split_assignment"
ISSUE_EMPTY_SPLIT = "empty_split"
ISSUE_INVALID_RATIO = "invalid_ratio"
ISSUE_RECORD_ID_NOT_FOUND = "record_id_not_found"
ISSUE_ORACLE_GRAPH_RECORD_ID_MISMATCH = "oracle_graph_record_id_mismatch"
ISSUE_TARGET_QUESTION_PRESENT_BUT_PROTECTED = "target_question_present_but_protected"
ISSUE_LEAKAGE_SENSITIVE_FIELD_IN_METADATA = "leakage_sensitive_field_in_metadata"
