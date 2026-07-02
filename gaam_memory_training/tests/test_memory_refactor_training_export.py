"""Tests for production Memory Refactor GRPO dataset export."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from gaam_graph.memory_refactor_training import (
    MemoryRefactorExportConfig,
    export_memory_refactor_grpo_dataset,
)
from gaam_graph.memory_refactor_schema import MemoryPatch, MemoryRefactorAction
from gaam_graph.verl_gaam_reward import compute_score


def _sample(split: str = "train") -> dict:
    return {
        "attack_id": f"atk_editor_{split}",
        "split": split,
        "question": "What code editor does the user primarily use now?",
        "gold_answer": "Cursor",
        "gold_evidence": "The user now primarily uses Cursor as their code editor.",
        "memory_snapshot": {
            "chunks": [
                {
                    "id": "mem_editor_old",
                    "content": "The user previously used VSCode as a code editor.",
                    "atomic_facts": [
                        {
                            "fact_id": "fact_old_editor",
                            "text": "The user previously used VSCode as a code editor.",
                            "confidence": 0.9,
                        }
                    ],
                    "question_edges": [
                        {
                            "question_id": "q_editor_history",
                            "role": "core",
                            "weight": 0.92,
                            "pass_count": 3,
                        }
                    ],
                }
            ],
            "historical_questions": {
                "q_editor_history": "What code editor did the user use previously?"
            },
        },
        "relation_hints": ["temporal_update"],
        "local_test_ids": ["q_editor_history"],
        "near_test_ids": [],
        "anchor_test_ids": [],
    }


def test_export_memory_refactor_grpo_dataset_writes_verl_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "samples.json"
    output_dir = tmp_path / "dataset"
    input_path.write_text(
        json.dumps({"samples": [_sample("train"), _sample("dev")]}, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = export_memory_refactor_grpo_dataset(
        MemoryRefactorExportConfig(input_path=input_path, output_dir=output_dir)
    )

    assert manifest["num_train_rows"] == 1
    assert manifest["num_val_rows"] == 1

    row = pd.read_parquet(manifest["train_path"]).to_dict(orient="records")[0]
    assert row["data_source"] == "adversarial_memory_refactor"
    assert row["ability"] == "memory_refactor"
    assert row["reward_model"]["ground_truth"]["gold_answer"] == "Cursor"
    assert Path(row["extra_info"]["memory_snapshot_id"]).exists()
    assert row["extra_info"]["local_test_ids"] == ["q_editor_history"]

    patch = MemoryPatch(
        patch_id="patch_export_check",
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
                    {"text": "The user previously used VSCode as a code editor."},
                    {"text": "The user now primarily uses Cursor."},
                ],
            }
        ],
        question_edge_updates=[
            {
                "question_id": "q_current",
                "chunk_id": "mem_editor_old",
                "role": "core",
                "weight": 0.95,
            }
        ],
    )
    reward = compute_score(
        row["data_source"],
        patch.model_dump_json(),
        row["reward_model"]["ground_truth"],
        row["extra_info"],
    )
    assert reward["commit_allowed"] is True


def test_export_memory_refactor_grpo_dataset_cli(tmp_path: Path) -> None:
    input_path = tmp_path / "samples.jsonl"
    output_dir = tmp_path / "dataset"
    input_path.write_text(
        "\n".join(json.dumps(_sample(split), ensure_ascii=False) for split in ("train", "dev")),
        encoding="utf-8",
    )
    root_dir = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_memory_refactor_grpo_dataset.py",
            "--input",
            str(input_path),
            "--output_dir",
            str(output_dir),
        ],
        cwd=root_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Memory Refactor GRPO dataset exported" in result.stdout
    assert (output_dir / "memory_refactor.train.parquet").exists()
    assert (output_dir / "memory_refactor.val.parquet").exists()
