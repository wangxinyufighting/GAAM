"""Production dataset export for Memory Refactoring Policy GRPO training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from gaam_graph.memory_refactor_dataset import (
    build_refactor_policy_prompt,
    build_verl_refactor_row,
)
from gaam_graph.memory_refactor_env import MemoryEnv
from gaam_graph.memory_refactor_schema import MemoryChunk
from gaam_graph.utils import stable_id, write_json


@dataclass(frozen=True)
class MemoryRefactorExportConfig:
    input_path: Path
    output_dir: Path
    max_records_per_split: int | None = None
    val_split_preference: tuple[str, ...] = ("dev", "test", "train")


def export_memory_refactor_grpo_dataset(config: MemoryRefactorExportConfig) -> dict[str, Any]:
    """Export MemoryPatch training samples to native VERL parquet files."""
    samples = _load_samples(config.input_path)
    split_samples = _split_samples(samples, config.max_records_per_split)
    train_samples = split_samples.get("train", [])
    val_samples = _choose_val_samples(split_samples, config.val_split_preference)
    if not train_samples:
        raise ValueError("No train samples available for memory refactor export.")
    if not val_samples:
        val_samples = train_samples[:1]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _build_rows(train_samples, config.output_dir, split="train")
    val_rows = _build_rows(val_samples, config.output_dir, split="val")

    train_path = config.output_dir / "memory_refactor.train.parquet"
    val_path = config.output_dir / "memory_refactor.val.parquet"
    _write_parquet(train_rows, train_path)
    _write_parquet(val_rows, val_path)

    manifest = {
        "manifest_version": "gaam_memory_refactor_grpo_dataset_v1",
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "train_attack_ids": [row["extra_info"]["attack_id"] for row in train_rows],
        "val_attack_ids": [row["extra_info"]["attack_id"] for row in val_rows],
        "data_source": "adversarial_memory_refactor",
        "ability": "memory_refactor",
    }
    manifest_path = config.output_dir / "memory_refactor.dataset_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def _load_samples(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return list(payload.get("samples", []))


def _split_samples(
    samples: list[dict[str, Any]],
    max_records_per_split: int | None,
) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for sample in samples:
        splits.setdefault(str(sample.get("split", "train")), []).append(sample)
    if max_records_per_split is None:
        return splits
    return {split: items[:max_records_per_split] for split, items in splits.items()}


def _choose_val_samples(
    split_samples: dict[str, list[dict[str, Any]]],
    preference: tuple[str, ...],
) -> list[dict[str, Any]]:
    for split in preference:
        if split_samples.get(split):
            return split_samples[split]
    return []


def _build_rows(
    samples: list[dict[str, Any]],
    output_dir: Path,
    *,
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    snapshot_dir = output_dir / "snapshots"
    for index, sample in enumerate(samples):
        attack_id = str(sample.get("attack_id") or stable_id("atk", split, index))
        env, snapshot_path = _prepare_snapshot(sample, snapshot_dir, attack_id)
        question = str(sample["question"])
        gold_evidence = str(sample["gold_evidence"])
        gold_answer = str(sample["gold_answer"])
        retrieved_chunks = _retrieved_chunks(sample, env, question)
        prompt_payload = build_refactor_policy_prompt(
            question=question,
            gold_evidence=gold_evidence,
            gold_answer=gold_answer,
            retrieved_chunks=retrieved_chunks,
            relation_hints=list(sample.get("relation_hints", [])),
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are the GAAM Memory Refactoring Policy. "
                    "Return only one valid MemoryPatch JSON object."
                ),
            },
            {"role": "user", "content": prompt_payload},
        ]
        row = build_verl_refactor_row(
            prompt=prompt,
            attack_id=attack_id,
            memory_snapshot_id=str(snapshot_path),
            question=question,
            gold_answer=gold_answer,
            gold_evidence=gold_evidence,
            retrieved_chunk_ids=[
                chunk.id for chunk in retrieved_chunks if isinstance(chunk, MemoryChunk) and chunk.id
            ],
            local_test_ids=list(sample.get("local_test_ids", [])),
            near_test_ids=list(sample.get("near_test_ids", [])),
            anchor_test_ids=list(sample.get("anchor_test_ids", [])),
        )
        row["extra_info"].update(
            {
                "split": split,
                "index": index,
                "row_id": stable_id("memory_refactor_row", split, attack_id, index),
            }
        )
        rows.append(row)
    return rows


def _prepare_snapshot(
    sample: dict[str, Any],
    snapshot_dir: Path,
    attack_id: str,
) -> tuple[MemoryEnv, Path]:
    if sample.get("memory_snapshot_path"):
        snapshot_path = Path(sample["memory_snapshot_path"])
        return MemoryEnv.load_snapshot(str(snapshot_path)), snapshot_path
    snapshot = sample["memory_snapshot"]
    env = MemoryEnv.from_snapshot(snapshot)
    snapshot_path = snapshot_dir / f"{attack_id}.snapshot.json"
    write_json(snapshot_path, env.to_snapshot())
    return env, snapshot_path


def _retrieved_chunks(sample: dict[str, Any], env: MemoryEnv, question: str) -> list[MemoryChunk]:
    if sample.get("retrieved_chunks"):
        return [MemoryChunk.model_validate(chunk) for chunk in sample["retrieved_chunks"]]
    retrieved = env.retrieve(question)
    return retrieved or list(env.chunks)


def _write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)

