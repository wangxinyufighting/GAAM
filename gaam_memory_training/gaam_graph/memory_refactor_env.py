"""Sandbox environment for MemoryPatch reward and commit gating."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from gaam_graph.memory_refactor_schema import (
    CommitGateConfig,
    MemoryChunk,
    MemoryChunkStatus,
    MemoryPatch,
    MemoryRefactorAction,
    PatchEvalResult,
    QuestionEdge,
    QuestionEdgeRole,
    WeightedRewardConfig,
    parse_memory_patch,
)


WORD_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text or "")}


def _overlap_score(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def compute_weighted_reward(
    result: PatchEvalResult,
    *,
    config: WeightedRewardConfig | None = None,
) -> float:
    """Compute GRPO reward from sandbox metrics.

    Reward and commit are intentionally separate: a patch can receive a useful
    training signal while still failing the hard commit gate.
    """
    weights = config or WeightedRewardConfig()
    current = 1.0 if result.current_correct and result.current_evidence_supported else 0.3
    if not result.current_correct:
        current = -1.0

    retrieval = result.retrieval_precision + result.retrieval_recall
    near = result.near_pass_rate
    structure = 1.0
    structure -= 0.25 * min(result.unsupported_fact_count, 4)
    structure -= 0.25 * min(result.conflict_count, 4)
    structure -= 0.15 * min(result.redundancy_count, 4)
    if result.over_compression_flag:
        structure -= 1.0

    penalty = 0.0
    if not result.valid_json:
        penalty += 3.0
    if not result.schema_pass:
        penalty += 2.0
    penalty += 0.5 * result.unsupported_fact_count
    penalty += 0.5 * result.conflict_count
    penalty += 0.2 * max(result.memory_growth - 5, 0)
    if result.over_compression_flag:
        penalty += 1.0

    return (
        weights.current_weight * current
        + weights.retrieval_weight * retrieval
        + weights.local_weight * result.local_pass_rate
        + weights.near_weight * near
        + weights.anchor_weight * result.anchor_pass_rate
        + weights.structure_weight * structure
        - weights.penalty_weight * penalty
    )


def commit_gate_allows(
    result: PatchEvalResult,
    *,
    config: CommitGateConfig | None = None,
) -> bool:
    """Return whether a sandbox result is safe to commit."""
    gate = config or CommitGateConfig()
    return (
        result.valid_json
        and result.schema_pass
        and result.current_correct
        and result.current_evidence_supported
        and result.local_pass_rate >= gate.min_local_regression_acc
        and result.anchor_pass_rate >= gate.min_anchor_acc
        and result.conflict_count == 0
        and result.unsupported_fact_count == 0
        and not result.over_compression_flag
    )


@dataclass(frozen=True)
class CommitCandidate:
    """A patch paired with its sandbox evaluation."""

    patch: MemoryPatch
    eval_result: PatchEvalResult


@dataclass(frozen=True)
class CommitDecision:
    """Result of choosing whether to commit a group of candidates."""

    action: str
    selected_patch_id: str | None = None
    reason: str = ""
    reward: float | None = None


class MemoryEnv:
    """In-memory sandbox for refactor patches.

    This is deliberately lightweight: it can load JSON snapshots, apply patches
    to a copied environment, and compute deterministic structural/retrieval
    scores for GRPO reward functions without mutating the main memory store.
    """

    def __init__(
        self,
        chunks: list[MemoryChunk] | None = None,
        *,
        historical_questions: dict[str, str] | None = None,
    ) -> None:
        self.chunks = list(chunks or [])
        self.historical_questions = dict(historical_questions or {})

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "MemoryEnv":
        chunks = [MemoryChunk.model_validate(chunk) for chunk in snapshot.get("chunks", [])]
        return cls(chunks, historical_questions=snapshot.get("historical_questions", {}))

    @classmethod
    def load_snapshot(cls, snapshot_id: str) -> "MemoryEnv":
        path = Path(snapshot_id)
        if not path.exists():
            raise FileNotFoundError(f"memory snapshot does not exist: {snapshot_id}")
        return cls.from_snapshot(json.loads(path.read_text(encoding="utf-8")))

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "chunks": [chunk.model_dump(mode="json") for chunk in self.chunks],
            "historical_questions": dict(self.historical_questions),
        }

    def retrieve(self, query: str, top_k: int = 8) -> list[MemoryChunk]:
        scored = [
            (_overlap_score(query, chunk.content), chunk)
            for chunk in self.chunks
            if chunk.status == MemoryChunkStatus.ACTIVE
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for score, chunk in scored[:top_k] if score > 0.0]

    def apply_patch_sandbox(self, patch: MemoryPatch) -> "MemoryEnv":
        env = deepcopy(self)
        env._apply_patch_in_place(patch)
        return env

    def evaluate_patch(
        self,
        patch: MemoryPatch | str,
        *,
        question: str,
        gold_answer: str,
        gold_evidence: str,
        local_test_ids: list[str] | None = None,
        near_test_ids: list[str] | None = None,
        anchor_test_ids: list[str] | None = None,
    ) -> PatchEvalResult:
        if isinstance(patch, str):
            try:
                patch = parse_memory_patch(patch)
            except ValueError as exc:
                result = PatchEvalResult(
                    valid_json="invalid MemoryPatch JSON" not in str(exc),
                    schema_pass=False,
                    diagnostics={"error": str(exc)},
                )
                result.reward = compute_weighted_reward(result)
                return result

        before_count = len(self.chunks)
        sandbox = self.apply_patch_sandbox(patch)
        retrieved = sandbox.retrieve(question)
        retrieved_text = "\n".join(chunk.content for chunk in retrieved)
        current_correct = bool(gold_answer) and gold_answer.lower() in retrieved_text.lower()
        evidence_supported = _overlap_score(gold_evidence, retrieved_text) >= 0.15
        relevant_chunk_ids = {
            chunk.id
            for chunk in retrieved
            if _overlap_score(gold_evidence, chunk.content) >= 0.15
        }
        retrieved_ids = {chunk.id for chunk in retrieved}
        retrieval_precision = len(relevant_chunk_ids) / max(len(retrieved_ids), 1)
        retrieval_recall = 1.0 if relevant_chunk_ids else 0.0

        local_pass_rate = sandbox._question_pass_rate(local_test_ids or [])
        near_pass_rate = sandbox._question_pass_rate(near_test_ids or [])
        anchor_pass_rate = sandbox._question_pass_rate(anchor_test_ids or [])
        if not local_test_ids:
            local_pass_rate = 1.0
        if not near_test_ids:
            near_pass_rate = 1.0
        if not anchor_test_ids:
            anchor_pass_rate = 1.0

        unsupported = sandbox._unsupported_fact_count(patch, gold_evidence)
        over_compression = _detect_over_compression(patch)
        redundancy = sandbox._redundancy_count()
        result = PatchEvalResult(
            valid_json=True,
            schema_pass=True,
            current_correct=current_correct,
            current_evidence_supported=evidence_supported,
            local_pass_rate=local_pass_rate,
            near_pass_rate=near_pass_rate,
            anchor_pass_rate=anchor_pass_rate,
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            conflict_count=_count_conflict_flags(patch),
            unsupported_fact_count=unsupported,
            redundancy_count=redundancy,
            over_compression_flag=over_compression,
            memory_growth=len(sandbox.chunks) - before_count,
            diagnostics={
                "retrieved_chunk_ids": list(retrieved_ids),
                "relevant_chunk_ids": list(relevant_chunk_ids),
            },
        )
        result.reward = compute_weighted_reward(result)
        result.commit_allowed = commit_gate_allows(result)
        return result

    def _apply_patch_in_place(self, patch: MemoryPatch) -> None:
        by_id = {chunk.id: chunk for chunk in self.chunks if chunk.id}
        next_index = len(self.chunks)

        for update in patch.updated_chunks:
            if update.id not in by_id:
                raise ValueError(f"updated chunk not found: {update.id}")
            chunk = by_id[update.id]
            data = chunk.model_dump()
            for field in (
                "content",
                "atomic_facts",
                "status",
                "version",
                "parent_chunks",
                "provenance",
                "validity_scope",
                "question_edges",
            ):
                value = getattr(update, field)
                if value is not None:
                    data[field] = value
            data["version"] = max(int(data.get("version") or 1), chunk.version + 1)
            by_id[update.id] = MemoryChunk.model_validate(data)

        for chunk_id in patch.deprecated_chunk_ids:
            if chunk_id in by_id:
                data = by_id[chunk_id].model_dump()
                data["status"] = MemoryChunkStatus.DEPRECATED
                data["version"] = by_id[chunk_id].version + 1
                by_id[chunk_id] = MemoryChunk.model_validate(data)

        for chunk in patch.new_chunks:
            data = chunk.model_dump()
            if not data.get("id"):
                data["id"] = f"new_chunk_{next_index}"
                next_index += 1
            by_id[data["id"]] = MemoryChunk.model_validate(data)

        for edge_update in patch.question_edge_updates:
            chunk_id = edge_update.chunk_id
            if chunk_id.startswith("new_chunk_") and chunk_id not in by_id:
                ordered_new = [chunk for chunk in by_id if chunk.startswith("new_chunk_")]
                if ordered_new:
                    chunk_id = ordered_new[0]
            if chunk_id not in by_id:
                continue
            chunk = by_id[chunk_id]
            edges = [edge for edge in chunk.question_edges if edge.question_id != edge_update.question_id]
            edges.append(
                QuestionEdge(
                    question_id=edge_update.question_id,
                    role=edge_update.role,
                    weight=edge_update.weight,
                    source=edge_update.source,
                    last_verified_at=edge_update.last_verified_at,
                    pass_count=edge_update.pass_count,
                    fail_count=edge_update.fail_count,
                )
            )
            data = chunk.model_dump()
            data["question_edges"] = edges
            by_id[chunk_id] = MemoryChunk.model_validate(data)

        self.chunks = list(by_id.values())

    def _question_pass_rate(self, question_ids: list[str]) -> float:
        if not question_ids:
            return 1.0
        passes = 0
        for question_id in question_ids:
            question = self.historical_questions.get(question_id, question_id)
            retrieved = self.retrieve(question)
            if any(
                edge.question_id == question_id
                and edge.role in {QuestionEdgeRole.CORE, QuestionEdgeRole.SUPPORT}
                and edge.weight >= 0.5
                and chunk.status == MemoryChunkStatus.ACTIVE
                for chunk in retrieved
                for edge in chunk.question_edges
            ):
                passes += 1
        return passes / len(question_ids)

    def _unsupported_fact_count(self, patch: MemoryPatch, evidence: str) -> int:
        count = 0
        for chunk in patch.new_chunks:
            facts = chunk.atomic_facts or []
            for fact in facts:
                if _overlap_score(fact.text, evidence) < 0.1:
                    count += 1
        for update in patch.updated_chunks:
            existing_text = ""
            for chunk in self.chunks:
                if chunk.id == update.id:
                    existing_text = chunk.content
                    break
            for fact in update.atomic_facts or []:
                supported_by_new_evidence = _overlap_score(fact.text, evidence) >= 0.1
                preserved_from_existing_chunk = _overlap_score(fact.text, existing_text) >= 0.1
                if not supported_by_new_evidence and not preserved_from_existing_chunk:
                    count += 1
        return count

    def _redundancy_count(self) -> int:
        active = [chunk for chunk in self.chunks if chunk.status == MemoryChunkStatus.ACTIVE]
        count = 0
        for index, chunk in enumerate(active):
            for other in active[index + 1 :]:
                if _overlap_score(chunk.content, other.content) > 0.9:
                    count += 1
        return count


class CommitManager:
    """Select and commit the best hard-gate-passing patch."""

    def __init__(self, env: MemoryEnv, *, gate_config: CommitGateConfig | None = None) -> None:
        self.env = env
        self.gate_config = gate_config or CommitGateConfig()

    def maybe_commit_best_patch(self, candidates: list[CommitCandidate]) -> CommitDecision:
        valid = [
            candidate
            for candidate in candidates
            if commit_gate_allows(candidate.eval_result, config=self.gate_config)
        ]
        if not valid:
            return CommitDecision(action="rollback", reason="no candidate passed commit gate")
        best = max(valid, key=lambda candidate: candidate.eval_result.reward)
        self.env = self.env.apply_patch_sandbox(best.patch)
        return CommitDecision(
            action="commit",
            selected_patch_id=best.patch.patch_id,
            reason="best passing candidate committed",
            reward=best.eval_result.reward,
        )


def _detect_over_compression(patch: MemoryPatch) -> bool:
    if patch.action_type != MemoryRefactorAction.REFACTOR:
        return False
    if len(patch.touched_chunk_ids) <= 1 or len(patch.new_chunks) != 1:
        return False
    chunk = patch.new_chunks[0]
    fact_count = len(chunk.atomic_facts)
    return fact_count >= 3 or len(chunk.content) > 900


def _count_conflict_flags(patch: MemoryPatch) -> int:
    return sum(1 for flag in patch.risk_flags if "conflict" in flag.lower() or "contradict" in flag.lower())
