"""
Phase 4 Milestone 3: Trainer Input Loading and Validation

Loads and validates aggregated actor batches from Phase 4 Milestone 2 multi-case rollout.

Key functions:
- load_multi_case_rollout_manifest: load completed M2 rollout
- locate_actor_batch_paths: find actor-specific update batches
- load_actor_batch: read and normalize batch format
- validate_actor_batch_for_update: enforce split/update/no-leakage policies
- summarize_actor_trainer_input: metadata summary

Design principle:
Validate all safety constraints before any model loading or update.
"""

import json
from pathlib import Path
from typing import Any

from gaam_graph.dataset_split_schema import DatasetSplitName
from gaam_graph.phase4_batch_export import check_actor_batch_no_leakage
from gaam_graph.phase4_rollout_schema import MultiCaseRolloutManifest
from gaam_graph.phase4_trainer_schema import Phase4ActorTrainerInput
from gaam_graph.grpo_schema import ActorRole


def load_multi_case_rollout_manifest(run_dir: Path) -> MultiCaseRolloutManifest:
    """Load multi-case rollout manifest from completed Phase 4 M2 run."""
    manifest_path = run_dir / "multi_case_rollout_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Multi-case rollout manifest not found: {manifest_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return MultiCaseRolloutManifest.model_validate(data)


def locate_actor_batch_paths(
    run_dir: Path,
    actor_roles: list[ActorRole],
) -> dict[ActorRole, tuple[Path, Path | None]]:
    """
    Locate actor batch paths and manifests in aggregated output.

    Returns:
        Dict mapping ActorRole to (batch_path, manifest_path).
    """
    aggregate_dir = run_dir / "aggregate" / "actor_batches"

    if not aggregate_dir.exists():
        raise FileNotFoundError(
            f"Aggregate actor batches directory not found: {aggregate_dir}"
        )

    result: dict[ActorRole, tuple[Path, Path | None]] = {}

    for actor_role in actor_roles:
        role_name = actor_role.value
        batch_path = aggregate_dir / f"{role_name}.update_batch.json"
        manifest_path = aggregate_dir / f"{role_name}.batch_manifest.json"

        if not batch_path.exists():
            raise FileNotFoundError(f"Actor batch not found: {batch_path}")

        result[actor_role] = (
            batch_path,
            manifest_path if manifest_path.exists() else None,
        )

    return result


def load_actor_batch(path: Path) -> dict[str, Any]:
    """Load actor update batch from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_actor_batch_manifest(path: Path | None) -> dict[str, Any] | None:
    """Load an actor batch manifest when it exists."""
    if path is None or not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, dict) else None


def normalize_actor_batch_samples(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Normalize batch samples from various formats into a flat list.

    Supports:
    - batch["items"]
    - batch["groups"][*]["samples"]
    - batch["samples"]

    Returns:
        Flat list of all samples.
    """
    samples: list[dict[str, Any]] = []

    # Format 1: items
    if "items" in batch:
        samples.extend(batch["items"])

    # Format 2: groups[*].samples
    if "groups" in batch:
        for group in batch["groups"]:
            if "samples" in group:
                samples.extend(group["samples"])

    # Format 3: samples
    if "samples" in batch:
        samples.extend(batch["samples"])

    return samples


def validate_actor_batch_for_update(
    batch: dict[str, Any],
    actor_role: ActorRole,
    *,
    split: DatasetSplitName,
    require_non_empty: bool,
    strict_no_leakage: bool = True,
    manifest_no_leakage_passed: bool | None = None,
) -> list[str]:
    """
    Validate actor batch for model update.

    Rules:
    1. split == train is required for model updates
    2. Dev/test batches must be rejected for update mode
    3. no_leakage_passed == true is required from batch manifest
    4. Actor-specific forbidden fields must not be present
    5. Empty batches are allowed only if require_non_empty=False

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    # Rule 1 & 2: Split policy
    if split != DatasetSplitName.TRAIN:
        errors.append(
            f"Model updates only allowed on train split, got: {split.value}"
        )

    # Rule 3: No-leakage check
    no_leakage_passed, violations = check_actor_batch_no_leakage(
        batch, actor_role.value
    )

    if strict_no_leakage and manifest_no_leakage_passed is False:
        errors.append(
            f"Actor batch manifest reports no_leakage_passed=False for {actor_role.value}"
        )

    if strict_no_leakage and not no_leakage_passed:
        errors.append(
            f"No-leakage check failed for {actor_role.value}: {violations}"
        )

    # Rule 5: Empty batch check
    samples = normalize_actor_batch_samples(batch)

    if require_non_empty and len(samples) == 0:
        errors.append(
            f"Empty batch for {actor_role.value} (use --allow_empty_batch to override)"
        )

    return errors


def summarize_actor_trainer_input(
    actor_role: ActorRole,
    batch_path: Path,
    batch_manifest_path: Path | None,
    multi_case_manifest: MultiCaseRolloutManifest,
    multi_case_manifest_path: Path | None = None,
) -> Phase4ActorTrainerInput:
    """
    Summarize actor trainer input metadata.

    Reads batch and manifest to extract:
    - num_samples, num_groups
    - no_leakage_passed
    - selected_record_ids
    """
    batch = load_actor_batch(batch_path)

    # Count samples
    samples = normalize_actor_batch_samples(batch)
    num_samples = len(samples)

    # Count groups
    num_groups = len(batch.get("groups", []))
    if num_groups == 0 and num_samples > 0:
        # If no groups structure, treat as one implicit group
        num_groups = 1

    # Load manifest if available. If it is missing, fall back to a direct batch
    # check so old/debug artifacts can still be inspected honestly.
    manifest_data = load_actor_batch_manifest(batch_manifest_path)
    batch_no_leakage_passed, _ = check_actor_batch_no_leakage(
        batch, actor_role.value
    )
    no_leakage_passed = (
        bool(manifest_data.get("no_leakage_passed", False))
        if manifest_data is not None
        else batch_no_leakage_passed
    )

    # Extract selected record IDs from multi-case manifest
    selected_record_ids = multi_case_manifest.selected_record_ids

    return Phase4ActorTrainerInput(
        actor_role=actor_role,
        batch_path=str(batch_path),
        batch_manifest_path=str(batch_manifest_path) if batch_manifest_path else None,
        source_multi_case_run_dir=multi_case_manifest.output_dir,
        source_multi_case_manifest_path=str(
            multi_case_manifest_path
            or Path(multi_case_manifest.output_dir) / "multi_case_rollout_manifest.json"
        ),
        split=multi_case_manifest.split,
        round_id=multi_case_manifest.round_id,
        step_id=multi_case_manifest.step_id,
        num_samples=num_samples,
        num_groups=num_groups,
        no_leakage_passed=no_leakage_passed,
        selected_record_ids=selected_record_ids,
    )


def load_and_validate_actor_inputs(
    run_dir: Path,
    actor_roles: list[ActorRole],
    *,
    require_non_empty: bool = True,
    strict_no_leakage: bool = True,
) -> tuple[MultiCaseRolloutManifest, dict[ActorRole, Phase4ActorTrainerInput], list[str]]:
    """
    Load multi-case rollout and validate all actor inputs.

    Returns:
        (multi_case_manifest, actor_inputs, errors)
    """
    errors: list[str] = []

    # Load multi-case manifest
    multi_case_manifest_path = run_dir / "multi_case_rollout_manifest.json"
    try:
        multi_case_manifest = load_multi_case_rollout_manifest(run_dir)
    except Exception as e:
        errors.append(f"Failed to load multi-case manifest: {e}")
        return None, {}, errors  # type: ignore

    # Locate actor batch paths
    try:
        actor_batch_paths = locate_actor_batch_paths(run_dir, actor_roles)
    except Exception as e:
        errors.append(f"Failed to locate actor batch paths: {e}")
        return multi_case_manifest, {}, errors

    # Load and validate each actor batch
    actor_inputs: dict[ActorRole, Phase4ActorTrainerInput] = {}

    for actor_role, (batch_path, manifest_path) in actor_batch_paths.items():
        try:
            # Load batch
            batch = load_actor_batch(batch_path)
            batch_manifest = load_actor_batch_manifest(manifest_path)

            # Validate
            validation_errors = validate_actor_batch_for_update(
                batch,
                actor_role,
                split=multi_case_manifest.split,
                require_non_empty=require_non_empty,
                strict_no_leakage=strict_no_leakage,
                manifest_no_leakage_passed=(
                    None
                    if batch_manifest is None
                    else bool(batch_manifest.get("no_leakage_passed", False))
                ),
            )

            if validation_errors:
                errors.extend(
                    [f"[{actor_role.value}] {e}" for e in validation_errors]
                )

            # Summarize input
            actor_input = summarize_actor_trainer_input(
                actor_role,
                batch_path,
                manifest_path,
                multi_case_manifest,
                multi_case_manifest_path,
            )

            actor_inputs[actor_role] = actor_input

        except Exception as e:
            errors.append(f"Failed to load {actor_role.value} batch: {e}")

    return multi_case_manifest, actor_inputs, errors
