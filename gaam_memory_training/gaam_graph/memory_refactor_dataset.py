"""Dataset and prompt helpers for Memory Refactoring Policy training."""

from __future__ import annotations

import json
from typing import Any

from gaam_graph.memory_refactor_schema import MemoryRefactorAction, MemoryChunk


ALLOWED_ACTIONS = [action.value for action in MemoryRefactorAction]


def build_refactor_policy_prompt(
    *,
    question: str,
    gold_evidence: str,
    gold_answer: str,
    retrieved_chunks: list[MemoryChunk | dict[str, Any]],
    relation_hints: list[str] | None = None,
) -> str:
    """Build the single-turn JSON-patch prompt recommended by the plan."""
    safe_chunks = [
        chunk.model_dump(mode="json") if isinstance(chunk, MemoryChunk) else chunk
        for chunk in retrieved_chunks
    ]
    payload = {
        "role": "Memory Refactoring Policy",
        "task": "Return one valid MemoryPatch JSON object. Do not answer the question.",
        "question": question,
        "gold_evidence": gold_evidence,
        "gold_answer": gold_answer,
        "retrieved_chunks": safe_chunks,
        "relation_hints": relation_hints or [],
        "allowed_actions": ALLOWED_ACTIONS,
        "requirements": [
            "Use ADD, REFACTOR, REVISE, SPLIT, LINK_ONLY, NO_OP, or DEPRECATE.",
            "REFACTOR is many-to-many: old chunks plus evidence may produce zero, one, or many chunks.",
            "Do not force independent facts into one chunk.",
            "Every new or revised atomic fact must be supported by gold_evidence.",
            "Preserve time ranges, conditions, exceptions, provenance, and parent_chunks.",
            "Use question_edge_updates with explicit role and weight.",
            "Return JSON only.",
        ],
        "output_schema_keys": [
            "patch_id",
            "action_type",
            "touched_chunk_ids",
            "new_chunks",
            "updated_chunks",
            "deprecated_chunk_ids",
            "question_edge_updates",
            "risk_flags",
            "brief_rationale",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_verl_refactor_row(
    *,
    prompt: str | list[dict[str, str]],
    attack_id: str,
    memory_snapshot_id: str,
    question: str,
    gold_answer: str,
    gold_evidence: str,
    retrieved_chunk_ids: list[str],
    local_test_ids: list[str] | None = None,
    near_test_ids: list[str] | None = None,
    anchor_test_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Create one VERL-compatible row for offline GRPO refactor training."""
    ground_truth = {
        "question": question,
        "gold_answer": gold_answer,
        "gold_evidence": gold_evidence,
        "memory_snapshot_id": memory_snapshot_id,
    }
    return {
        "data_source": "adversarial_memory_refactor",
        "prompt": prompt,
        "ability": "memory_refactor",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "attack_id": attack_id,
            "memory_snapshot_id": memory_snapshot_id,
            "question": question,
            "gold_answer": gold_answer,
            "gold_evidence": gold_evidence,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "local_test_ids": local_test_ids or [],
            "near_test_ids": near_test_ids or [],
            "anchor_test_ids": anchor_test_ids or [],
        },
    }
