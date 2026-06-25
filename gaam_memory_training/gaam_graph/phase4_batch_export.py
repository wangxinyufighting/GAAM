"""
Phase 4 Milestone 2: Actor Batch Aggregation and Export

Aggregates actor-specific update batches across multiple records.

Key functions:
- collect_actor_update_batches: gather per-record actor batches
- merge_actor_update_batches: combine batches preserving GRPO groups
- write_actor_batch_manifest: write metadata
- export_phase4_grpo_batches: export training-ready batches

Design principle:
Do not flatten away group identity. GRPO depends on comparing samples within a group.
Each record contributes independent groups to the merged batch.
"""

import json
from pathlib import Path
from typing import Any

from gaam_graph.dataset_split_schema import DatasetSplitName
from gaam_graph.grpo_schema import ActorRole
from gaam_graph.phase4_rollout_schema import (
    ActorBatchManifest,
    MultiCaseRecordRun,
    MultiCaseRolloutStatus,
)


ACTOR_ROLE_ALIASES = {
    "memory_builder": {"memory_builder", ActorRole.MEMORY_BUILDER.value},
    "question_agent": {"question_agent", ActorRole.QUESTION_AGENT.value},
}


def _actor_role_matches(value: Any, actor_role: str) -> bool:
    """Return whether a serialized role value matches the requested actor."""
    if value is None:
        return True
    return str(value) in ACTOR_ROLE_ALIASES.get(actor_role, {actor_role})


def _load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _resolve_artifact_path(path_value: str | Path, run_dir: Path) -> Path:
    """Resolve artifact paths that may be absolute or relative to a run ancestor."""
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path

    for base_dir in (run_dir, *run_dir.parents):
        candidate = base_dir / path
        if candidate.exists():
            return candidate

    return path


def _candidate_batch_paths(record_run: MultiCaseRecordRun, actor_role: str) -> list[Path]:
    """Discover candidate actor batch paths from real and legacy artifacts."""
    run_dir = Path(record_run.output_dir)
    paths: list[Path] = []

    explicit = (
        record_run.memory_update_batch_path
        if actor_role == "memory_builder"
        else record_run.question_update_batch_path
    )
    if explicit:
        paths.append(_resolve_artifact_path(explicit, run_dir))

    if run_dir.exists():
        batch_suffix = (
            "memory_update_batch"
            if actor_role == "memory_builder"
            else "question_update_batch"
        )
        paths.extend(sorted(run_dir.glob(f"rounds/*/trainer/{actor_role}/update_batch.json")))
        paths.extend(sorted(run_dir.glob(f"rounds/*/trainer/{actor_role}/trainer_ready_batch.json")))
        paths.extend(sorted(run_dir.glob(f"rounds/*/rollout/update_batches/*.{actor_role}.json")))
        paths.extend(sorted(run_dir.glob(f"rounds/*/rollout/update_batches/*{actor_role}*.json")))
        paths.extend(sorted(run_dir.glob(f"rounds/*/rollout/jobs/*/update_batches/*.{batch_suffix}.json")))
        paths.extend(sorted(run_dir.glob(f"jobs/*/update_batches/*.{batch_suffix}.json")))
        paths.extend(sorted(run_dir.glob(f"{actor_role}.update_batch.json")))

        for replay_path in sorted(run_dir.glob("rounds/*/reward/jobs/**/*.replay_item.json")):
            replay_item = _load_json_file(replay_path)
            if not replay_item:
                continue
            batch_path = replay_item.get("trainer_ready_batch_paths", {}).get(actor_role)
            if batch_path:
                paths.append(_resolve_artifact_path(batch_path, run_dir))

        for replay_buffer in sorted(run_dir.glob("rounds/*/reward/replay_buffer.jsonl")):
            try:
                with open(replay_buffer, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        replay_item = json.loads(line)
                        batch_path = replay_item.get("trainer_ready_batch_paths", {}).get(actor_role)
                        if batch_path:
                            paths.append(_resolve_artifact_path(batch_path, run_dir))
            except Exception:
                pass

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def collect_actor_update_batches(
    record_runs: list[MultiCaseRecordRun],
    actor_role: str,
) -> list[dict[str, Any]]:
    """
    Collect actor update batches from per-record outputs.

    Searches for:
    - rounds/round_NNN_step_NNN/trainer/{actor_role}/update_batch.json
    - {actor_role}.update_batch.json (fallback)

    Args:
        record_runs: List of per-record run results
        actor_role: "memory_builder" or "question_agent"

    Returns:
        List of actor update batch dictionaries
    """
    batches: list[dict[str, Any]] = []

    for record_run in record_runs:
        if record_run.status != MultiCaseRolloutStatus.SUCCEEDED:
            continue

        for batch_path in _candidate_batch_paths(record_run, actor_role):
            if not batch_path.exists():
                continue
            batch = _load_json_file(batch_path)
            if not batch:
                continue
            batch["_source_record_id"] = record_run.record_id
            batch["_source_batch_path"] = str(batch_path)
            batches.append(batch)

    return batches


def merge_actor_update_batches(
    batches: list[dict[str, Any]],
    actor_role: str,
) -> dict[str, Any]:
    """
    Merge actor update batches while preserving GRPO group structure.

    Each record contributes independent groups. Group IDs are prefixed with
    record_id to ensure uniqueness:
    - round{R}_step{S}_{record_id}_memory
    - round{R}_step{S}_{record_id}_question

    Args:
        batches: List of per-record actor batches
        actor_role: "memory_builder" or "question_agent"

    Returns:
        Merged actor update batch
    """
    if not batches:
        return {
            "actor_role": actor_role,
            "groups": [],
            "items": [],
            "num_groups": 0,
            "num_samples": 0,
            "num_selected_samples": 0,
        }

    merged_groups: list[dict[str, Any]] = []
    merged_items: list[dict[str, Any]] = []
    round_id = 0
    step_id = 0

    for batch in batches:
        # Verify actor role matches
        batch_role = batch.get("actor_role") or batch.get("role") or actor_role
        if not _actor_role_matches(batch_role, actor_role):
            continue

        source_record_id = batch.get("_source_record_id", "unknown")
        round_id = int(batch.get("round_id", round_id) or 0)
        step_id = int(batch.get("step_id", step_id) or 0)

        # Current GAAM ActorUpdateBatch schema uses items.
        for item in batch.get("items", []):
            item_role = item.get("role", actor_role)
            if not _actor_role_matches(item_role, actor_role):
                continue
            item_copy = item.copy()
            original_group_id = item_copy.get("group_id", "")
            item_copy["group_id"] = f"{source_record_id}_{original_group_id}"
            item_copy["source_record_id"] = source_record_id
            merged_items.append(item_copy)

        # Local dataproto exports use top-level samples instead of items/groups.
        for sample in batch.get("samples", []):
            sample_role = sample.get("role", actor_role)
            if not _actor_role_matches(sample_role, actor_role):
                continue
            sample_copy = sample.copy()
            original_group_id = sample_copy.get("group_id") or batch.get(
                "batch_id",
                "samples",
            )
            sample_copy["group_id"] = f"{source_record_id}_{original_group_id}"
            sample_copy["source_record_id"] = source_record_id
            merged_items.append(sample_copy)

        # Legacy grouped schema used by earlier tests/prototypes.
        groups = batch.get("groups", [])

        for group in groups:
            # Prefix group_id with record_id to ensure uniqueness
            original_group_id = group.get("group_id", "")
            prefixed_group_id = f"{source_record_id}_{original_group_id}"

            group_copy = group.copy()
            group_copy["group_id"] = prefixed_group_id
            group_copy["source_record_id"] = source_record_id

            merged_groups.append(group_copy)

    # Compute aggregate stats
    num_samples = sum(len(g.get("samples", [])) for g in merged_groups)
    num_samples += len(merged_items)
    num_selected = sum(
        sum(1 for s in g.get("samples", []) if s.get("selected_for_update", False))
        for g in merged_groups
    )
    num_selected += sum(
        1 for item in merged_items if item.get("selected_for_update", True)
    )

    merged_batch = {
        "actor_role": actor_role,
        "role": actor_role,
        "batch_id": f"phase4_merged_{actor_role}",
        "round_id": round_id,
        "step_id": step_id,
        "groups": merged_groups,
        "items": merged_items,
        "num_groups": len(merged_groups),
        "num_samples": num_samples,
        "num_selected_samples": num_selected,
    }

    return merged_batch


def compute_batch_reward_stats(
    batch: dict[str, Any],
) -> tuple[float | None, float | None]:
    """
    Compute mean and std of rewards in an actor batch.

    Returns:
        (reward_mean, reward_std) or (None, None) if no rewards
    """
    rewards: list[float] = []

    for group in batch.get("groups", []):
        for sample in group.get("samples", []):
            reward = sample.get("reward")
            if reward is not None and isinstance(reward, (int, float)):
                rewards.append(float(reward))
    for item in batch.get("items", []):
        reward = item.get("reward", item.get("reward_score", item.get("score")))
        if reward is not None and isinstance(reward, (int, float)):
            rewards.append(float(reward))

    if not rewards:
        return None, None

    mean_reward = sum(rewards) / len(rewards)

    if len(rewards) > 1:
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = variance**0.5
    else:
        std_reward = 0.0

    return mean_reward, std_reward


def check_actor_batch_no_leakage(
    batch: dict[str, Any],
    actor_role: str,
) -> tuple[bool, list[str]]:
    """
    Check actor batch for forbidden fields based on actor role.

    Memory Builder stricter than Question Agent.

    Returns:
        (passed, violations)
    """
    violations: list[str] = []

    # Forbidden fields for Memory Builder
    memory_builder_forbidden = [
        "benchmark_target_question",
        "benchmark_answer",
        "oracle_graph",
        "oracle_node",
        "oracle_edge",
        "generated_training_question",
        "question_type",
        "reward_report",
    ]

    # Forbidden fields for Question Agent
    question_agent_forbidden = [
        "benchmark_target_question",
        "benchmark_answer",
        "test_label",
    ]

    forbidden = (
        memory_builder_forbidden
        if actor_role == "memory_builder"
        else question_agent_forbidden
    )

    # Check batch-level metadata
    for key in forbidden:
        if key in batch:
            violations.append(f"batch.{key}")

    # Check group-level metadata
    for i, group in enumerate(batch.get("groups", [])):
        for key in forbidden:
            if key in group:
                violations.append(f"groups[{i}].{key}")

        # Check sample-level
        for j, sample in enumerate(group.get("samples", [])):
            for key in forbidden:
                if key in sample:
                    violations.append(f"groups[{i}].samples[{j}].{key}")

            # Check nested context
            context = sample.get("context", {})
            for key in forbidden:
                if key in context:
                    violations.append(f"groups[{i}].samples[{j}].context.{key}")

    for i, item in enumerate(batch.get("items", [])):
        for key in forbidden:
            if key in item:
                violations.append(f"items[{i}].{key}")
        metadata = item.get("metadata", {})
        if isinstance(metadata, dict):
            for key in forbidden:
                if key in metadata:
                    violations.append(f"items[{i}].metadata.{key}")

    passed = len(violations) == 0
    return passed, violations


def write_actor_batch_manifest(
    batch: dict[str, Any],
    batch_path: Path,
    source_rollout_manifest_path: str,
    split: DatasetSplitName,
    record_ids: list[str],
    missing_records: list[str],
) -> ActorBatchManifest:
    """
    Write actor batch manifest with metadata.

    Args:
        batch: Merged actor update batch
        batch_path: Path to batch JSON file
        source_rollout_manifest_path: Path to source rollout manifest
        split: Active split
        record_ids: Successfully merged record IDs
        missing_records: Records that failed or were skipped

    Returns:
        ActorBatchManifest
    """
    actor_role = batch["actor_role"]

    # Compute reward stats
    reward_mean, reward_std = compute_batch_reward_stats(batch)

    # Check no-leakage
    no_leakage_passed, violations = check_actor_batch_no_leakage(batch, actor_role)

    manifest = ActorBatchManifest(
        source_rollout_manifest_path=source_rollout_manifest_path,
        actor_role=actor_role,
        split=split,
        batch_path=str(batch_path),
        record_ids=record_ids,
        num_groups=batch["num_groups"],
        num_samples=batch["num_samples"],
        num_selected_samples=batch["num_selected_samples"],
        reward_mean=reward_mean,
        reward_std=reward_std,
        no_leakage_passed=no_leakage_passed,
        missing_records=missing_records,
    )

    return manifest


def export_phase4_grpo_batches(
    record_runs: list[MultiCaseRecordRun],
    output_dir: Path,
    actors: list[str],
    source_rollout_manifest_path: str,
    split: DatasetSplitName,
) -> dict[str, ActorBatchManifest]:
    """
    Export GRPO-ready actor batches from multi-case rollout.

    Args:
        record_runs: List of per-record run results
        output_dir: Output directory for actor batches
        actors: List of actor roles to export (e.g., ["memory_builder", "question_agent"])
        source_rollout_manifest_path: Path to source rollout manifest
        split: Active split

    Returns:
        Dict mapping actor_role to ActorBatchManifest
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, ActorBatchManifest] = {}

    succeeded_record_ids = [
        r.record_id
        for r in record_runs
        if r.status == MultiCaseRolloutStatus.SUCCEEDED
    ]
    missing_record_ids = [
        r.record_id
        for r in record_runs
        if r.status != MultiCaseRolloutStatus.SUCCEEDED
    ]

    for actor_role in actors:
        # Collect per-record batches
        batches = collect_actor_update_batches(record_runs, actor_role)

        # Merge batches
        merged_batch = merge_actor_update_batches(batches, actor_role)

        # Write batch JSON
        batch_path = output_dir / f"{actor_role}.update_batch.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(merged_batch, f, indent=2)

        # Write manifest
        manifest = write_actor_batch_manifest(
            merged_batch,
            batch_path,
            source_rollout_manifest_path,
            split,
            succeeded_record_ids,
            missing_record_ids,
        )

        manifest_path = output_dir / f"{actor_role}.batch_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

        manifests[actor_role] = manifest

    return manifests
