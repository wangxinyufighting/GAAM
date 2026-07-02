"""Schemas for adversarial memory refactoring patches.

The refactor policy does not write memory directly. It emits a structured
MemoryPatch that can be scored in a sandbox and committed only after a hard
gate passes.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MemoryRefactorAction(str, Enum):
    """Allowed actions for a memory refactoring policy."""

    ADD = "ADD"
    REFACTOR = "REFACTOR"
    REVISE = "REVISE"
    SPLIT = "SPLIT"
    LINK_ONLY = "LINK_ONLY"
    NO_OP = "NO_OP"
    DEPRECATE = "DEPRECATE"


class MemoryChunkStatus(str, Enum):
    """Lifecycle state for refactor-policy chunks."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    UNCERTAIN = "uncertain"
    ARCHIVED = "archived"


class QuestionEdgeRole(str, Enum):
    """How strongly a question depends on a chunk."""

    CORE = "core"
    SUPPORT = "support"
    NEAR = "near"
    HISTORICAL = "historical"


class AtomicFact(BaseModel):
    """One evidence-backed fact inside a memory chunk."""

    fact_id: str = ""
    text: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source_span: str = ""

    @model_validator(mode="after")
    def validate_fact(self) -> "AtomicFact":
        if not self.text.strip():
            raise ValueError("atomic fact text cannot be empty")
        return self


class ValidityScope(BaseModel):
    """Temporal, entity, or conditional scope for a chunk."""

    time_range: str | None = None
    entity_scope: list[str] = Field(default_factory=list)
    condition: str | None = None


class MemoryProvenance(BaseModel):
    """Audit information for generated or updated chunks."""

    operation: MemoryRefactorAction | str = MemoryRefactorAction.ADD
    attack_id: str = ""
    source_evidence_id: str = ""
    generated_by_policy: str = ""


class QuestionEdge(BaseModel):
    """Weighted question-memory edge used for targeted regression tests."""

    question_id: str
    role: QuestionEdgeRole = QuestionEdgeRole.SUPPORT
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = ""
    last_verified_at: str | None = None
    pass_count: int = Field(default=0, ge=0)
    fail_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_question_id(self) -> "QuestionEdge":
        if not self.question_id.strip():
            raise ValueError("question_id cannot be empty")
        return self


class MemoryChunk(BaseModel):
    """Chunk representation for refactor-policy sandboxing."""

    id: str = ""
    content: str
    atomic_facts: list[AtomicFact] = Field(default_factory=list)
    embedding: Any | None = None
    status: MemoryChunkStatus = MemoryChunkStatus.ACTIVE
    version: int = Field(default=1, ge=1)
    created_at: str | None = None
    updated_at: str | None = None
    parent_chunks: list[str] = Field(default_factory=list)
    provenance: MemoryProvenance = Field(default_factory=MemoryProvenance)
    validity_scope: ValidityScope = Field(default_factory=ValidityScope)
    question_edges: list[QuestionEdge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_chunk(self) -> "MemoryChunk":
        if not self.content.strip():
            raise ValueError("chunk content cannot be empty")
        if "linked_questions" in self.metadata:
            raise ValueError("linked_questions is deprecated; use weighted question_edges")
        return self


class MemoryChunkUpdate(BaseModel):
    """Updated form of an existing chunk."""

    id: str
    content: str | None = None
    atomic_facts: list[AtomicFact] | None = None
    status: MemoryChunkStatus | None = None
    version: int | None = Field(default=None, ge=1)
    parent_chunks: list[str] | None = None
    provenance: MemoryProvenance | None = None
    validity_scope: ValidityScope | None = None
    question_edges: list[QuestionEdge] | None = None
    change_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_update(self) -> "MemoryChunkUpdate":
        if not self.id.strip():
            raise ValueError("updated chunk id cannot be empty")
        if self.content is not None and not self.content.strip():
            raise ValueError("updated chunk content cannot be empty")
        return self


class QuestionEdgeUpdate(BaseModel):
    """A patch-level update to one question-memory edge."""

    question_id: str
    chunk_id: str
    role: QuestionEdgeRole = QuestionEdgeRole.SUPPORT
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "policy_patch"
    last_verified_at: str | None = None
    pass_count: int = Field(default=0, ge=0)
    fail_count: int = Field(default=0, ge=0)


class MemoryPatch(BaseModel):
    """Policy output: a structured memory modification proposal."""

    patch_id: str = ""
    action_type: MemoryRefactorAction
    touched_chunk_ids: list[str] = Field(default_factory=list)
    new_chunks: list[MemoryChunk] = Field(default_factory=list)
    updated_chunks: list[MemoryChunkUpdate] = Field(default_factory=list)
    deprecated_chunk_ids: list[str] = Field(default_factory=list)
    question_edge_updates: list[QuestionEdgeUpdate] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    brief_rationale: str = ""

    @model_validator(mode="after")
    def validate_patch_semantics(self) -> "MemoryPatch":
        touched = set(self.touched_chunk_ids)
        deprecated = set(self.deprecated_chunk_ids)
        updated = {chunk.id for chunk in self.updated_chunks}
        overlap = updated & deprecated
        if overlap:
            raise ValueError(f"chunks cannot be both updated and deprecated: {sorted(overlap)}")

        if self.action_type == MemoryRefactorAction.LINK_ONLY:
            if self.new_chunks or self.updated_chunks or self.deprecated_chunk_ids:
                raise ValueError("LINK_ONLY patches may only update question edges")
            if not self.question_edge_updates:
                raise ValueError("LINK_ONLY requires at least one question_edge_update")

        if self.action_type == MemoryRefactorAction.NO_OP:
            if (
                self.new_chunks
                or self.updated_chunks
                or self.deprecated_chunk_ids
                or self.question_edge_updates
            ):
                raise ValueError("NO_OP patches must not mutate memory")

        if self.action_type in {
            MemoryRefactorAction.REFACTOR,
            MemoryRefactorAction.REVISE,
            MemoryRefactorAction.SPLIT,
            MemoryRefactorAction.DEPRECATE,
        } and not touched:
            raise ValueError(f"{self.action_type.value} requires touched_chunk_ids")

        if self.action_type == MemoryRefactorAction.REFACTOR:
            affected = len(self.new_chunks) + len(self.updated_chunks) + len(self.deprecated_chunk_ids)
            if affected == 0:
                raise ValueError("REFACTOR must create, update, or deprecate at least one chunk")

        return self


class PatchEvalResult(BaseModel):
    """Sandbox evaluation result for a MemoryPatch."""

    valid_json: bool = True
    schema_pass: bool = True
    current_correct: bool = False
    current_evidence_supported: bool = False
    local_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    near_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    anchor_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    conflict_count: int = Field(default=0, ge=0)
    unsupported_fact_count: int = Field(default=0, ge=0)
    redundancy_count: int = Field(default=0, ge=0)
    over_compression_flag: bool = False
    memory_growth: int = 0
    reward: float = 0.0
    commit_allowed: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reward(self) -> "PatchEvalResult":
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        return self


class CommitGateConfig(BaseModel):
    """Hard-gate thresholds for committing sandboxed patches."""

    min_local_regression_acc: float = Field(default=0.90, ge=0.0, le=1.0)
    min_anchor_acc: float = Field(default=0.95, ge=0.0, le=1.0)


class WeightedRewardConfig(BaseModel):
    """Weights from the v0.1 implementation plan."""

    current_weight: float = 3.0
    retrieval_weight: float = 1.0
    local_weight: float = 2.0
    near_weight: float = 1.0
    anchor_weight: float = 2.0
    structure_weight: float = 1.0
    penalty_weight: float = 1.0


def parse_memory_patch(solution_str: str) -> MemoryPatch:
    """Parse and validate a model-produced MemoryPatch JSON string."""
    try:
        raw = json.loads(solution_str)
    except Exception as exc:
        raise ValueError(f"invalid MemoryPatch JSON: {exc}") from exc
    try:
        return MemoryPatch.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"MemoryPatch schema violation: {exc}") from exc

