#!/usr/bin/env python3
"""Validate GAAM production native VERL training inputs before launch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


FORBIDDEN_SPLIT_KEYS = {
    "question",
    "answer",
    "gold_answer",
    "benchmark_question",
    "benchmark_answer",
    "target_question",
    "target_answer",
    "oracle_answer",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether GAAM native VERL training inputs are production-ready."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--oracle_graph_dir", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, required=True)
    parser.add_argument("--memory_model_path", type=Path, required=True)
    parser.add_argument("--question_model_path", type=Path, required=True)
    parser.add_argument("--code_a1_root", type=Path, required=True)
    parser.add_argument("--memory_input_mode", default="full")
    parser.add_argument("--allow_static_incremental_scaffold", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--questions_per_case", type=int, default=None)
    parser.add_argument("--reward_judge_enabled", default=os.getenv("GAAM_REWARD_JUDGE_ENABLED", "0"))
    parser.add_argument("--reward_judge_api_key", default=os.getenv("GAAM_REWARD_JUDGE_API_KEY", ""))
    parser.add_argument("--max_reported_issues", type=int, default=30)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    _check_file(args.input, "input", errors)
    _check_dir(args.oracle_graph_dir, "oracle_graph_dir", errors)
    _check_file(args.split_manifest, "split_manifest", errors)
    _check_dir(args.memory_model_path, "memory_model_path", errors)
    _check_dir(args.question_model_path, "question_model_path", errors)
    _check_dir(args.code_a1_root / "verl" / "verl", "Code-A1 vendored VERL package", errors)

    if args.memory_input_mode == "incremental" and not args.allow_static_incremental_scaffold:
        errors.append(
            "memory_input_mode=incremental is blocked for production native parquet training. "
            "Use stateful rollout/evaluation, or explicitly allow static scaffold only for smoke tests."
        )

    if args.questions_per_case is not None and args.questions_per_case < 1:
        errors.append(f"questions_per_case must be >= 1, got {args.questions_per_case}.")

    if args.train_batch_size is not None and args.train_batch_size < 1:
        errors.append(f"train_batch_size must be >= 1, got {args.train_batch_size}.")

    if _truthy(args.reward_judge_enabled) and not args.reward_judge_api_key:
        errors.append(
            "GAAM_REWARD_JUDGE_ENABLED=1 but GAAM_REWARD_JUDGE_API_KEY is empty."
        )

    split_records = []
    split_counts: dict[str, int] = {}
    if args.split_manifest.exists():
        try:
            split_data = _read_json(args.split_manifest)
            forbidden_paths = _find_forbidden_keys(split_data)
            if forbidden_paths:
                errors.append(
                    "split_manifest contains forbidden benchmark target fields: "
                    + ", ".join(forbidden_paths[: args.max_reported_issues])
                )
            split_records = _extract_split_records(split_data)
            if not split_records:
                errors.append("split_manifest contains no records.")
            missing_graphs = []
            for item in split_records:
                split = str(item.get("split", ""))
                record_id = str(item.get("record_id", ""))
                split_counts[split] = split_counts.get(split, 0) + 1
                if record_id and not (args.oracle_graph_dir / f"{record_id}.graph.json").exists():
                    missing_graphs.append(record_id)
            if missing_graphs:
                errors.append(
                    f"{len(missing_graphs)} split records are missing oracle graphs. "
                    f"Examples: {missing_graphs[: args.max_reported_issues]}"
                )
            if split_counts.get("train", 0) < 1:
                errors.append("split_manifest has no train records.")
            if split_counts.get("dev", 0) < 1:
                warnings.append("split_manifest has no dev records; validation will reuse fallback data.")
            if split_counts.get("test", 0) < 1:
                warnings.append("split_manifest has no test records; final held-out evaluation is not possible.")
            if args.train_batch_size is not None:
                train_count = split_counts.get("train", 0)
                if 0 < train_count < args.train_batch_size:
                    warnings.append(
                        f"train_batch_size={args.train_batch_size} is larger than graph-backed "
                        f"train records={train_count}; native script will cap the batch size to "
                        "avoid an empty VERL dataloader."
                    )
        except Exception as exc:
            errors.append(f"Failed to inspect split_manifest: {type(exc).__name__}: {exc}")

    report = {
        "status": "failed" if errors else "succeeded",
        "input": str(args.input),
        "oracle_graph_dir": str(args.oracle_graph_dir),
        "split_manifest": str(args.split_manifest),
        "memory_model_path": str(args.memory_model_path),
        "question_model_path": str(args.question_model_path),
        "code_a1_root": str(args.code_a1_root),
        "memory_input_mode": args.memory_input_mode,
        "num_split_records": len(split_records),
        "split_counts": split_counts,
        "train_batch_size": args.train_batch_size,
        "questions_per_case": args.questions_per_case,
        "errors": errors,
        "warnings": warnings,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


def _check_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists() or not path.is_file():
        errors.append(f"{label} does not exist or is not a file: {path}")


def _check_dir(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists() or not path.is_dir():
        errors.append(f"{label} does not exist or is not a directory: {path}")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_split_records(split_data: Any) -> list[dict[str, Any]]:
    if isinstance(split_data, dict) and isinstance(split_data.get("records"), list):
        return [item for item in split_data["records"] if isinstance(item, dict)]
    if isinstance(split_data, list):
        return [item for item in split_data if isinstance(item, dict)]
    return []


def _find_forbidden_keys(obj: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else key
            if key in FORBIDDEN_SPLIT_KEYS:
                hits.append(child_path)
            hits.extend(_find_forbidden_keys(value, child_path))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            hits.extend(_find_forbidden_keys(value, f"{path}[{index}]"))
    return hits


def _truthy(value: str) -> bool:
    return value in {"1", "true", "True", "yes", "YES"}


if __name__ == "__main__":
    raise SystemExit(main())
