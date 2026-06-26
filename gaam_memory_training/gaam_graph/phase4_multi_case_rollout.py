"""
Phase 4 Milestone 2: Multi-Case Rollout Execution

Runs multiple records from a dataset split and aggregates results.

Key functions:
- select_records_for_multi_case_rollout: select records from split manifest
- build_phase3_config_for_record: build Phase 3 config for one record
- run_record_rollout: execute Phase 3 E2E for one record
- run_multi_case_rollout: main orchestration loop
- write_multi_case_summary_markdown: human-readable summary

Design principle:
Thin orchestration layer that calls Phase 3 E2E once per record.
Supports resume, continue-on-failure, and split-aware policies.
"""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from gaam_graph.dataset_split_schema import (
    DatasetSplitManifest,
    DatasetSplitName,
)
from gaam_graph.dataset_splitter import validate_split_manifest
from gaam_graph.phase4_batch_export import export_phase4_grpo_batches
from gaam_graph.phase4_replay_aggregation import aggregate_replay_buffer
from gaam_graph.phase4_rollout_schema import (
    MultiCaseRecordRun,
    MultiCaseRolloutManifest,
    MultiCaseRolloutStatus,
    aggregate_record_status,
    compute_aggregate_metrics,
    generate_run_id,
    split_counts,
)
from gaam_graph.phase3_e2e_schema import (
    Phase3E2EBackend,
    Phase3E2ERunConfig,
    Phase3E2EStageStatus,
)
from gaam_graph.training_protocol_schema import get_default_policy_for_split


def select_records_for_multi_case_rollout(
    split_manifest: DatasetSplitManifest,
    split: DatasetSplitName,
    max_records: int | None = None,
    record_ids: list[str] | None = None,
) -> list[str]:
    """
    Select records from split manifest for rollout.

    Args:
        split_manifest: Dataset split manifest
        split: Target split (train/dev/test)
        max_records: Maximum number of records to select
        record_ids: Explicit record IDs to select (overrides max_records)

    Returns:
        List of selected record IDs
    """
    # Filter records by split
    split_records = [r for r in split_manifest.records if r.split == split]

    # If explicit IDs are provided, use exactly that request, including [].
    if record_ids is not None:
        split_record_ids = {r.record_id for r in split_records}
        selected = [rid for rid in record_ids if rid in split_record_ids]
        return selected

    # Otherwise select up to max_records
    selected = [r.record_id for r in split_records]

    if max_records is not None:
        if max_records <= 0:
            return []
        selected = selected[:max_records]

    return selected


def build_phase3_config_for_record(
    record_id: str,
    split_manifest: DatasetSplitManifest,
    output_dir: Path,
    backend: str,
    trainer_mode: str,
    round_id: int,
    step_id: int,
    **kwargs: Any,
) -> Phase3E2ERunConfig:
    """
    Build Phase 3 E2E config for a single record.

    Args:
        record_id: Record ID to run
        split_manifest: Dataset split manifest
        output_dir: Output directory for this record
        backend: Backend name (local_fallback, ray, etc.)
        trainer_mode: Trainer mode (dry_run, local_model, etc.)
        round_id: Training round ID
        step_id: Training step ID
        **kwargs: Additional Phase 3 config parameters

    Returns:
        Phase 3 E2E run config
    """
    # Find record in manifest
    record = next(
        (r for r in split_manifest.records if r.record_id == record_id), None
    )

    if record is None:
        raise ValueError(f"Record {record_id} not found in split manifest")

    # Get split-aware training protocol policy
    policy = get_default_policy_for_split(record.split)

    update_memory_builder = kwargs.pop(
        "update_memory_builder",
        policy.actor_update_policy.get("memory_builder", False),
    )
    update_question_agent = kwargs.pop(
        "update_question_agent",
        policy.actor_update_policy.get("question_agent", False),
    )
    seed = int(kwargs.pop("seed", 0) or 0)
    strict = bool(kwargs.pop("strict", False))
    overwrite = bool(kwargs.pop("overwrite", True))
    resume = bool(kwargs.pop("resume", False))
    write_verl_dataproto = bool(kwargs.pop("write_verl_dataproto", False))
    use_vendored_verl_if_available = bool(
        kwargs.pop("use_vendored_verl_if_available", False)
    )

    return Phase3E2ERunConfig(
        run_id=f"phase4_{record.split.value}_{record_id}_round{round_id:03d}_step{step_id:03d}",
        input_records_path=record.raw_record_path,
        oracle_graph_dir=str(Path(record.oracle_graph_path).parent),
        record_id=record_id,
        output_dir=str(output_dir),
        backend=Phase3E2EBackend(backend),
        trainer_mode=trainer_mode,
        round_id=round_id,
        step_id=step_id,
        seed=seed,
        max_records=1,
        strict=strict,
        overwrite=overwrite,
        resume=resume,
        require_real_case=True,
        update_memory_builder=update_memory_builder,
        update_question_agent=update_question_agent,
        write_verl_dataproto=write_verl_dataproto,
        use_vendored_verl_if_available=use_vendored_verl_if_available,
        memory_builder_model_path=kwargs.pop("memory_builder_model_path", None),
        question_agent_model_path=kwargs.pop("question_agent_model_path", None),
        answerer_model_path=kwargs.pop("answerer_model_path", None),
        memory_builder_checkpoint_id=kwargs.pop(
            "memory_builder_checkpoint_id", None
        ),
        question_agent_checkpoint_id=kwargs.pop(
            "question_agent_checkpoint_id", None
        ),
        checkpoint_registry_path=kwargs.pop("checkpoint_registry_path", None),
        metadata={
            "phase": "phase4",
            "split": record.split.value,
            "policy_actor_updates": {
                "memory_builder": update_memory_builder,
                "question_agent": update_question_agent,
            },
        },
    )


def run_record_rollout(
    record_id: str,
    config: Phase3E2ERunConfig,
    output_dir: Path,
) -> MultiCaseRecordRun:
    """
    Execute Phase 3 E2E rollout for a single record.

    Args:
        record_id: Record ID
        config: Phase 3 config
        output_dir: Output directory for this record

    Returns:
        MultiCaseRecordRun with execution result
    """
    split = DatasetSplitName(config.metadata.get("split", "train"))

    record_run = MultiCaseRecordRun(
        record_id=record_id,
        split=split,
        status=MultiCaseRolloutStatus.RUNNING,
        output_dir=str(output_dir),
    )

    try:
        from gaam_graph.phase3_e2e_orchestrator import run_phase3_end_to_end_grpo

        # Run Phase 3 E2E
        final_report = run_phase3_end_to_end_grpo(config)

        # Extract metrics from final report
        if final_report.status == Phase3E2EStageStatus.SUCCEEDED:
            record_run.status = MultiCaseRolloutStatus.SUCCEEDED
        elif final_report.status == Phase3E2EStageStatus.PARTIAL:
            record_run.status = MultiCaseRolloutStatus.PARTIAL
        else:
            record_run.status = MultiCaseRolloutStatus.FAILED

        record_run.final_report_path = str(output_dir / "final_report.json")
        record_run.no_leakage_passed = final_report.no_leakage_passed
        record_run.errors.extend(final_report.errors)
        record_run.warnings.extend(final_report.warnings)

        if final_report.round_reports:
            round_report = final_report.round_reports[0]
            record_run.memory_reward_mean = round_report.memory_reward_mean
            record_run.question_reward_mean = round_report.question_reward_mean
            record_run.replay_items = round_report.num_replay_items
            record_run.weakness_updates = round_report.num_weakness_updates
            record_run.dataproto_dir = None

            if round_report.replay_buffer_path:
                record_run.replay_item_path = round_report.replay_buffer_path

            for stage_report in round_report.stage_reports:
                if stage_report.artifact_paths.get("memory_builder_trainer_ready_batch"):
                    record_run.memory_update_batch_path = stage_report.artifact_paths[
                        "memory_builder_trainer_ready_batch"
                    ]
                if stage_report.artifact_paths.get("question_agent_trainer_ready_batch"):
                    record_run.question_update_batch_path = stage_report.artifact_paths[
                        "question_agent_trainer_ready_batch"
                    ]

        if final_report.status == Phase3E2EStageStatus.FAILED:
            record_run.status = MultiCaseRolloutStatus.FAILED

    except Exception as e:
        record_run.status = MultiCaseRolloutStatus.FAILED
        record_run.errors.append(str(e))
        record_run.errors.append(traceback.format_exc())

    return record_run


def run_multi_case_rollout(
    split_manifest_path: Path,
    split: DatasetSplitName,
    output_dir: Path,
    backend: str = "local_fallback",
    trainer_mode: str = "dry_run",
    max_records: int | None = None,
    record_ids: list[str] | None = None,
    round_id: int = 0,
    step_id: int = 0,
    resume: bool = False,
    overwrite: bool = False,
    continue_on_record_failure: bool = False,
    write_dataproto: bool = False,
    allow_empty_batch: bool = False,
    **kwargs: Any,
) -> MultiCaseRolloutManifest:
    """
    Run multi-case rollout across selected records from a split.

    Args:
        split_manifest_path: Path to dataset split manifest
        split: Target split (train/dev/test)
        output_dir: Output directory for multi-case run
        backend: Backend name
        trainer_mode: Trainer mode
        max_records: Maximum records to select
        record_ids: Explicit record IDs to run
        round_id: Training round ID
        step_id: Training step ID
        resume: Skip already-succeeded records
        overwrite: Overwrite output directory
        continue_on_record_failure: Continue after record failure
        write_dataproto: Export GRPO batches
        allow_empty_batch: Allow an empty selected-record set for smoke tests
        **kwargs: Additional Phase 3 config parameters

    Returns:
        MultiCaseRolloutManifest
    """
    # Load split manifest
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        split_manifest_dict = json.load(f)
    split_manifest = DatasetSplitManifest.model_validate(split_manifest_dict)

    # Validate split manifest
    validation_issues = validate_split_manifest(split_manifest)
    if any(issue.severity == "error" for issue in validation_issues):
        raise ValueError(f"Split manifest has errors: {validation_issues}")

    # Select records
    selected_record_ids = select_records_for_multi_case_rollout(
        split_manifest, split, max_records, record_ids
    )

    if not selected_record_ids and not allow_empty_batch:
        raise ValueError(f"No records selected from split {split}")

    # Generate run ID
    run_id = generate_run_id(split, round_id, step_id)

    # Check output directory policy
    if output_dir.exists():
        if not overwrite and not resume:
            raise FileExistsError(
                f"Output directory exists: {output_dir}. Use --overwrite or --resume."
            )
        if overwrite and not resume:
            # Clear output directory
            import shutil

            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize rollout manifest
    manifest_path = output_dir / "multi_case_rollout_manifest.json"
    manifest = MultiCaseRolloutManifest(
        run_id=run_id,
        split_manifest_path=str(split_manifest_path),
        split=split,
        selected_record_ids=selected_record_ids,
        round_id=round_id,
        step_id=step_id,
        backend=backend,
        trainer_mode=trainer_mode,
        output_dir=str(output_dir),
        started_at=datetime.utcnow().isoformat(),
        status=MultiCaseRolloutStatus.RUNNING,
    )

    if not selected_record_ids:
        manifest.finished_at = datetime.utcnow().isoformat()
        manifest.status = MultiCaseRolloutStatus.SUCCEEDED
        manifest.metrics = compute_aggregate_metrics(manifest.records)

        aggregate_dir = output_dir / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)

        replay_buffer_path = aggregate_dir / "replay_buffer.jsonl"
        replay_manifest = aggregate_replay_buffer(
            manifest.records,
            replay_buffer_path,
            split=split,
            strict=False,
        )
        replay_manifest.source_rollout_manifest_path = str(manifest_path)

        replay_manifest_path = aggregate_dir / "aggregated_replay_manifest.json"
        with open(replay_manifest_path, "w", encoding="utf-8") as f:
            json.dump(replay_manifest.model_dump(), f, indent=2)

        manifest.aggregate_paths["replay_buffer"] = str(replay_buffer_path)
        manifest.aggregate_paths["replay_manifest"] = str(replay_manifest_path)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

        summary_path = output_dir / "summary.md"
        write_multi_case_summary_markdown(manifest, summary_path)

        return manifest

    # Load existing records if resuming
    existing_records: dict[str, MultiCaseRecordRun] = {}
    if resume and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing_manifest_dict = json.load(f)
        existing_manifest = MultiCaseRolloutManifest.model_validate(
            existing_manifest_dict
        )
        existing_records = {r.record_id: r for r in existing_manifest.records}

    # Run each record
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    total_selected = len(selected_record_ids)
    for record_index, record_id in enumerate(selected_record_ids, start=1):
        # Check if already succeeded (resume mode)
        if resume and record_id in existing_records:
            existing_run = existing_records[record_id]
            if existing_run.status == MultiCaseRolloutStatus.SUCCEEDED:
                print(
                    f"[Multi-case rollout] {split.value} {record_index}/{total_selected} "
                    f"{record_id}: skipped existing succeeded run",
                    flush=True,
                )
                manifest.records.append(existing_run)
                continue

        # Build config
        record_output_dir = records_dir / record_id
        record_output_dir.mkdir(parents=True, exist_ok=True)

        config = build_phase3_config_for_record(
            record_id,
            split_manifest,
            record_output_dir,
            backend,
            trainer_mode,
            round_id,
            step_id,
            overwrite=True,
            resume=False,
            write_verl_dataproto=write_dataproto,
            **kwargs,
        )

        print(
            f"[Multi-case rollout] {split.value} {record_index}/{total_selected} "
            f"{record_id}: starting",
            flush=True,
        )
        record_run = run_record_rollout(record_id, config, record_output_dir)
        print(
            f"[Multi-case rollout] {split.value} {record_index}/{total_selected} "
            f"{record_id}: {record_run.status.value}",
            flush=True,
        )
        manifest.records.append(record_run)

        # Update manifest after each record
        manifest.status = aggregate_record_status(manifest.records)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

        # Check failure policy
        if (
            record_run.status == MultiCaseRolloutStatus.FAILED
            and not continue_on_record_failure
        ):
            break

    # Finalize manifest
    manifest.finished_at = datetime.utcnow().isoformat()
    manifest.status = aggregate_record_status(manifest.records)
    manifest.metrics = compute_aggregate_metrics(manifest.records)

    # Aggregate replay buffer
    aggregate_dir = output_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    replay_buffer_path = aggregate_dir / "replay_buffer.jsonl"
    replay_manifest = aggregate_replay_buffer(
        manifest.records,
        replay_buffer_path,
        split=split,
        strict=False,
    )
    replay_manifest.source_rollout_manifest_path = str(manifest_path)

    # Write replay manifest
    replay_manifest_path = aggregate_dir / "aggregated_replay_manifest.json"
    with open(replay_manifest_path, "w", encoding="utf-8") as f:
        json.dump(replay_manifest.model_dump(), f, indent=2)

    manifest.aggregate_paths["replay_buffer"] = str(replay_buffer_path)
    manifest.aggregate_paths["replay_manifest"] = str(replay_manifest_path)

    # Export actor batches (train split only by default)
    if split == DatasetSplitName.TRAIN or write_dataproto:
        actor_batches_dir = aggregate_dir / "actor_batches"
        actor_manifests = export_phase4_grpo_batches(
            manifest.records,
            actor_batches_dir,
            ["memory_builder", "question_agent"],
            str(manifest_path),
            split,
        )

        manifest.aggregate_paths["actor_batches"] = str(actor_batches_dir)
        for actor_role, actor_manifest in actor_manifests.items():
            manifest.aggregate_paths[f"{actor_role}_batch"] = actor_manifest.batch_path
            manifest.aggregate_paths[f"{actor_role}_manifest"] = str(
                actor_batches_dir / f"{actor_role}.batch_manifest.json"
            )

    # Write final manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)

    # Write summary
    summary_path = output_dir / "summary.md"
    write_multi_case_summary_markdown(manifest, summary_path)

    return manifest


def write_multi_case_summary_markdown(
    manifest: MultiCaseRolloutManifest,
    output_path: Path,
) -> None:
    """Write human-readable summary for multi-case rollout."""
    counts = split_counts(manifest.records)

    lines = [
        f"# Phase 4 Multi-Case Rollout: {manifest.split}",
        "",
        f"**Run ID:** {manifest.run_id}",
        f"**Split:** {manifest.split}",
        f"**Round:** {manifest.round_id}",
        f"**Step:** {manifest.step_id}",
        f"**Backend:** {manifest.backend}",
        f"**Trainer Mode:** {manifest.trainer_mode}",
        f"**Status:** {manifest.status}",
        "",
        "## Records",
        "",
        f"- Selected: {len(manifest.selected_record_ids)}",
        f"- Succeeded: {counts[MultiCaseRolloutStatus.SUCCEEDED]}",
        f"- Failed: {counts[MultiCaseRolloutStatus.FAILED]}",
        f"- Partial: {counts[MultiCaseRolloutStatus.PARTIAL]}",
        "",
        "## Metrics",
        "",
    ]

    for key, value in manifest.metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Aggregate Artifacts",
            "",
        ]
    )

    for key, path in manifest.aggregate_paths.items():
        lines.append(f"- {key}: `{path}`")

    lines.extend(
        [
            "",
            "## Timestamps",
            "",
            f"- Started: {manifest.started_at}",
            f"- Finished: {manifest.finished_at or 'N/A'}",
            "",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
