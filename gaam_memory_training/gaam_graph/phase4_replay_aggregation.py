"""
Phase 4 Milestone 2: Replay Buffer Aggregation

Aggregates per-record replay items into split-level replay buffer.

Key functions:
- collect_record_replay_items: gather replay items from per-record outputs
- build_replay_item_key: stable key for duplicate detection
- aggregate_replay_buffer: merge items with filtering and validation

Design principle:
Replay aggregation preserves no-leakage guarantees established in earlier phases.
Dev/test replay items are marked eval_only and not mixed into train buffers.
"""

import json
from pathlib import Path
from typing import Any

from gaam_graph.dataset_split_schema import DatasetSplitName
from gaam_graph.phase4_rollout_schema import (
    AggregatedReplayManifest,
    MultiCaseRecordRun,
    MultiCaseRolloutStatus,
)


def build_replay_item_key(item: dict[str, Any]) -> str:
    """
    Build a stable key for replay item deduplication.

    Key format: {record_id}:{round_id}:{step_id}:{memory_checkpoint_id}:{question_checkpoint_id}
    """
    record_id = item.get("record_id", "unknown")
    round_id = item.get("round_id", 0)
    step_id = item.get("step_id", 0)
    memory_checkpoint = (
        item.get("memory_builder_checkpoint_id")
        or item.get("memory_checkpoint_id")
        or "none"
    )
    question_checkpoint = (
        item.get("question_agent_checkpoint_id")
        or item.get("question_checkpoint_id")
        or "none"
    )

    return f"{record_id}:{round_id}:{step_id}:{memory_checkpoint}:{question_checkpoint}"


def collect_record_replay_items(
    record_run_dirs: list[Path],
) -> list[dict[str, Any]]:
    """
    Collect replay items from per-record output directories.

    Searches for replay item files in standard Phase 3 locations:
    - rounds/round_NNN_step_NNN/reward/jobs/{record_id}/*.replay_item.json
    - rounds/round_NNN_step_NNN/reward/replay_buffer.jsonl
    - rounds/round_NNN_step_NNN/rollout/replay_item.json (legacy fallback)
    - replay_item.json (legacy fallback)

    Returns:
        List of replay item dictionaries.
    """
    items: list[dict[str, Any]] = []

    for run_dir in record_run_dirs:
        if not run_dir.exists():
            continue

        rounds_dir = run_dir / "rounds"
        if rounds_dir.exists():
            for round_dir in rounds_dir.iterdir():
                if not round_dir.is_dir():
                    continue

                reward_dir = round_dir / "reward"
                if reward_dir.exists():
                    for replay_file in sorted(reward_dir.glob("jobs/**/*.replay_item.json")):
                        try:
                            with open(replay_file, "r", encoding="utf-8") as f:
                                item = json.load(f)
                            items.append(item)
                        except Exception:
                            pass

                    replay_buffer = reward_dir / "replay_buffer.jsonl"
                    if replay_buffer.exists():
                        try:
                            with open(replay_buffer, "r", encoding="utf-8") as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        items.append(json.loads(line))
                        except Exception:
                            pass

                rollout_dir = round_dir / "rollout"
                replay_file = rollout_dir / "replay_item.json"

                if replay_file.exists():
                    try:
                        with open(replay_file, "r", encoding="utf-8") as f:
                            item = json.load(f)
                        items.append(item)
                    except Exception:
                        # Skip malformed files
                        pass

        # Fallback: check root replay_item.json
        root_replay = run_dir / "replay_item.json"
        if root_replay.exists():
            try:
                with open(root_replay, "r", encoding="utf-8") as f:
                    item = json.load(f)
                items.append(item)
            except Exception:
                pass

    return items


def filter_replay_items_for_split(
    items: list[dict[str, Any]],
    split: DatasetSplitName,
    succeeded_record_ids: set[str],
) -> list[dict[str, Any]]:
    """
    Filter replay items based on split and record success status.

    Rules for train split:
    - Include if no_leakage_passed == true
    - Include if selected_for_training == true
    - Include if record_id in succeeded_record_ids

    Rules for dev/test splits:
    - Mark as eval_only
    - Include if no_leakage_passed == true
    """
    filtered: list[dict[str, Any]] = []

    for item in items:
        record_id = item.get("record_id", "unknown")
        no_leakage_passed = item.get("no_leakage_passed", False)
        selected_for_training = item.get("selected_for_training", True)

        if split == DatasetSplitName.TRAIN:
            # Train filter: strict
            if (
                no_leakage_passed
                and selected_for_training
                and record_id in succeeded_record_ids
            ):
                filtered.append(item)
        else:
            # Dev/test filter: mark as eval_only
            if no_leakage_passed:
                item_copy = item.copy()
                item_copy["eval_only"] = True
                item_copy["selected_for_training"] = False
                filtered.append(item_copy)

    return filtered


def detect_duplicate_replay_keys(
    items: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Detect duplicate replay item keys.

    Returns:
        Dict mapping duplicate keys to their occurrence count.
    """
    key_counts: dict[str, int] = {}

    for item in items:
        key = build_replay_item_key(item)
        key_counts[key] = key_counts.get(key, 0) + 1

    # Return only duplicates
    return {k: v for k, v in key_counts.items() if v > 1}


def aggregate_replay_buffer(
    record_runs: list[MultiCaseRecordRun],
    output_path: Path,
    *,
    split: DatasetSplitName,
    strict: bool = False,
) -> AggregatedReplayManifest:
    """
    Aggregate replay items from multiple records into a split-level buffer.

    Args:
        record_runs: List of per-record run results
        output_path: Path to write aggregated replay buffer (JSONL)
        split: Active split (train/dev/test)
        strict: If True, fail on duplicates; if False, keep first and warn

    Returns:
        AggregatedReplayManifest with aggregation metadata
    """
    # Collect succeeded record IDs
    succeeded_records = [
        r for r in record_runs if r.status == MultiCaseRolloutStatus.SUCCEEDED
    ]
    succeeded_record_ids = {r.record_id for r in succeeded_records}
    skipped_records = [
        r.record_id
        for r in record_runs
        if r.status != MultiCaseRolloutStatus.SUCCEEDED
    ]

    # Collect replay items from succeeded records
    run_dirs = [Path(r.output_dir) for r in succeeded_records]
    all_items = collect_record_replay_items(run_dirs)

    # Filter by split policy
    filtered_items = filter_replay_items_for_split(
        all_items, split, succeeded_record_ids
    )

    # Detect duplicates
    duplicates = detect_duplicate_replay_keys(filtered_items)

    if duplicates and strict:
        raise ValueError(
            f"Duplicate replay keys found in strict mode: {list(duplicates.keys())[:5]}"
        )

    # Deduplicate: keep first occurrence
    seen_keys: set[str] = set()
    deduped_items: list[dict[str, Any]] = []

    for item in filtered_items:
        key = build_replay_item_key(item)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_items.append(item)

    # Write aggregated replay buffer (JSONL format)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in deduped_items:
            f.write(json.dumps(item) + "\n")

    # Count train vs eval_only items
    num_train_items = sum(
        1 for item in deduped_items if not item.get("eval_only", False)
    )
    num_eval_only_items = sum(
        1 for item in deduped_items if item.get("eval_only", False)
    )

    # Count no_leakage passes
    no_leakage_pass_count = sum(
        1 for item in deduped_items if item.get("no_leakage_passed", False)
    )

    # Build manifest
    manifest = AggregatedReplayManifest(
        source_rollout_manifest_path="",  # Will be set by caller
        replay_buffer_path=str(output_path),
        selected_record_ids=list(succeeded_record_ids),
        num_source_records=len(succeeded_records),
        num_replay_items=len(deduped_items),
        num_train_items=num_train_items,
        num_eval_only_items=num_eval_only_items,
        no_leakage_pass_count=no_leakage_pass_count,
        skipped_records=skipped_records,
        metrics={
            "num_duplicates_removed": len(filtered_items) - len(deduped_items),
            "duplicate_keys": list(duplicates.keys())[:10] if duplicates else [],
        },
    )

    return manifest


def read_replay_buffer(replay_path: Path) -> list[dict[str, Any]]:
    """Read aggregated replay buffer from JSONL file."""
    items: list[dict[str, Any]] = []

    if not replay_path.exists():
        return items

    with open(replay_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    return items
