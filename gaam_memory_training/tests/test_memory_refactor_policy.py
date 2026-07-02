"""Regression tests for Memory Refactoring Policy contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaam_graph.memory_refactor_dataset import build_refactor_policy_prompt, build_verl_refactor_row
from gaam_graph.memory_refactor_env import CommitCandidate, CommitManager, MemoryEnv
from gaam_graph.memory_refactor_schema import (
    AtomicFact,
    MemoryChunk,
    MemoryPatch,
    MemoryRefactorAction,
    QuestionEdge,
    QuestionEdgeRole,
)
from gaam_graph.verl_gaam_reward import compute_score


def _base_chunk() -> MemoryChunk:
    return MemoryChunk(
        id="mem_editor_old",
        content="The user previously used VSCode as a code editor.",
        atomic_facts=[
            AtomicFact(
                fact_id="fact_old_editor",
                text="The user previously used VSCode as a code editor.",
                confidence=0.9,
            )
        ],
        question_edges=[
            QuestionEdge(
                question_id="q_editor_history",
                role=QuestionEdgeRole.CORE,
                weight=0.92,
                pass_count=3,
            )
        ],
    )


def _good_refactor_patch() -> MemoryPatch:
    return MemoryPatch(
        patch_id="patch_cursor_refactor",
        action_type=MemoryRefactorAction.REFACTOR,
        touched_chunk_ids=["mem_editor_old"],
        updated_chunks=[
            {
                "id": "mem_editor_old",
                "content": (
                    "The user previously used VSCode as a code editor, and now primarily "
                    "uses Cursor as a code editor."
                ),
                "atomic_facts": [
                    {
                        "fact_id": "fact_old_editor",
                        "text": "The user previously used VSCode as a code editor.",
                        "confidence": 0.9,
                    },
                    {
                        "fact_id": "fact_new_editor",
                        "text": "The user now primarily uses Cursor.",
                        "confidence": 0.91,
                    },
                ],
                "change_summary": "Added the temporal update without deleting the older fact.",
            }
        ],
        question_edge_updates=[
            {
                "question_id": "q_current",
                "chunk_id": "mem_editor_old",
                "role": "core",
                "weight": 0.95,
                "pass_count": 1,
            }
        ],
    )


def test_memory_patch_refactor_is_many_to_many_and_uses_weighted_edges() -> None:
    patch = MemoryPatch(
        patch_id="patch_many_to_many",
        action_type=MemoryRefactorAction.REFACTOR,
        touched_chunk_ids=["mem_a", "mem_b"],
        new_chunks=[
            {
                "id": "mem_new_a",
                "content": "The user now primarily uses Cursor.",
                "atomic_facts": [{"text": "The user now primarily uses Cursor."}],
                "parent_chunks": ["mem_a"],
            },
            {
                "id": "mem_new_b",
                "content": "The user's earlier VSCode usage is historical.",
                "atomic_facts": [{"text": "The user's earlier VSCode usage is historical."}],
                "parent_chunks": ["mem_b"],
            },
        ],
        deprecated_chunk_ids=["mem_b"],
        question_edge_updates=[
            {
                "question_id": "q_current",
                "chunk_id": "mem_new_a",
                "role": "core",
                "weight": 0.95,
            }
        ],
    )

    assert len(patch.new_chunks) == 2
    assert patch.question_edge_updates[0].role == QuestionEdgeRole.CORE
    assert patch.question_edge_updates[0].weight == pytest.approx(0.95)


def test_link_only_rejects_chunk_mutation() -> None:
    with pytest.raises(ValueError, match="LINK_ONLY"):
        MemoryPatch(
            action_type=MemoryRefactorAction.LINK_ONLY,
            new_chunks=[{"content": "This should not be created."}],
            question_edge_updates=[
                {
                    "question_id": "q_current",
                    "chunk_id": "mem_existing",
                    "role": "core",
                    "weight": 0.9,
                }
            ],
        )


def test_memory_env_scores_patch_and_commit_gate_separates_reward_from_commit() -> None:
    env = MemoryEnv(
        [_base_chunk()],
        historical_questions={
            "q_editor_history": "What editor did the user use before?",
            "q_current": "What editor does the user primarily use now?",
        },
    )
    patch = _good_refactor_patch()

    result = env.evaluate_patch(
        patch,
        question="What editor does the user primarily use now?",
        gold_answer="Cursor",
        gold_evidence="The user now primarily uses Cursor as their code editor.",
        local_test_ids=["q_editor_history"],
        anchor_test_ids=[],
    )

    assert result.reward > 0
    assert result.commit_allowed is True

    blocked = result.model_copy(update={"anchor_pass_rate": 0.5, "commit_allowed": False})
    manager = CommitManager(env)
    decision = manager.maybe_commit_best_patch([CommitCandidate(patch=patch, eval_result=blocked)])

    assert blocked.reward > 0
    assert decision.action == "rollback"


def test_over_compressed_refactor_gets_reward_signal_but_cannot_commit() -> None:
    env = MemoryEnv([_base_chunk()])
    patch = MemoryPatch(
        patch_id="patch_overmerge",
        action_type=MemoryRefactorAction.REFACTOR,
        touched_chunk_ids=["mem_a", "mem_b"],
        new_chunks=[
            {
                "id": "mem_overmerged",
                "content": (
                    "The user now primarily uses Cursor. The user writes academic papers. "
                    "The user prefers Python for data work."
                ),
                "atomic_facts": [
                    {"text": "The user now primarily uses Cursor."},
                    {"text": "The user writes academic papers."},
                    {"text": "The user prefers Python for data work."},
                ],
                "parent_chunks": ["mem_a", "mem_b"],
            }
        ],
    )

    result = env.evaluate_patch(
        patch,
        question="What editor does the user primarily use now?",
        gold_answer="Cursor",
        gold_evidence="The user now primarily uses Cursor as their code editor.",
    )

    assert result.over_compression_flag is True
    assert result.commit_allowed is False


def test_verl_reward_scores_memory_refactor_patch_without_committing(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    env = MemoryEnv([_base_chunk()], historical_questions={"q_editor_history": "editor history"})
    snapshot_path.write_text(json.dumps(env.to_snapshot(), ensure_ascii=False), encoding="utf-8")

    result = compute_score(
        "adversarial_memory_refactor",
        _good_refactor_patch().model_dump_json(),
        {
            "question": "What editor does the user primarily use now?",
            "gold_answer": "Cursor",
            "gold_evidence": "The user now primarily uses Cursor as their code editor.",
            "memory_snapshot_id": str(snapshot_path),
        },
        {"local_test_ids": ["q_editor_history"]},
    )

    assert result["score"] > 0
    assert result["commit_allowed"] is True
    assert len(env.chunks) == 1


def test_verl_reward_rejects_invalid_patch_json() -> None:
    result = compute_score("adversarial_memory_refactor", "{not json", {}, {})

    assert result["score"] == pytest.approx(-3.0)
    assert result["valid_json"] is False
    assert result["schema_pass"] is False


def test_dataset_row_keeps_prompt_and_reward_metadata_separate() -> None:
    prompt = build_refactor_policy_prompt(
        question="What editor does the user use now?",
        gold_evidence="The user now primarily uses Cursor.",
        gold_answer="Cursor",
        retrieved_chunks=[_base_chunk()],
    )
    row = build_verl_refactor_row(
        prompt=prompt,
        attack_id="atk_1",
        memory_snapshot_id="snapshot_1",
        question="What editor does the user use now?",
        gold_answer="Cursor",
        gold_evidence="The user now primarily uses Cursor.",
        retrieved_chunk_ids=["mem_editor_old"],
        local_test_ids=["q_editor_history"],
        near_test_ids=["q_near"],
        anchor_test_ids=["q_anchor"],
    )

    assert row["data_source"] == "adversarial_memory_refactor"
    assert row["ability"] == "memory_refactor"
    assert row["reward_model"]["ground_truth"]["memory_snapshot_id"] == "snapshot_1"
    assert row["extra_info"]["memory_snapshot_id"] == "snapshot_1"
    assert "linked_questions" not in prompt
