"""
Phase 4 Milestone 1: Training Protocol

Core logic for split-level multi-case training execution.

Key principles:
- Reuse Phase 3 E2E orchestrator for per-record execution
- Aggregate per-record reports into split-level report
- Enforce split-aware actor update policies
- Continue on record failure when requested
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from gaam_graph.dataset_split_schema import (
    DatasetSplitIssue,
    DatasetSplitIssueSeverity,
    DatasetSplitManifest,
    DatasetSplitName,
)
from gaam_graph.phase3_e2e_orchestrator import run_phase3_end_to_end_grpo
from gaam_graph.phase3_e2e_schema import Phase3E2EBackend, Phase3E2ERunConfig
from gaam_graph.training_protocol_schema import (
    SplitRunRecordReport,
    SplitRunReport,
    TrainingProtocolManifest,
    get_default_policy_for_split,
)


def select_split_records(
    manifest: DatasetSplitManifest,
    split: DatasetSplitName,
    max_records: int | None = None,
) -> list[str]:
    """Select record IDs from split manifest for execution.

    Args:
        manifest: Dataset split manifest
        split: Which split to select from
        max_records: Optional limit on number of records

    Returns:
        List of record IDs sorted for deterministic execution
    """
    selected = [r.record_id for r in manifest.records if r.split == split]
    selected.sort()  # Deterministic order

    if max_records is not None and len(selected) > max_records:
        selected = selected[:max_records]

    return selected


def build_training_protocol_manifest(
    run_id: str,
    split_manifest_path: Path,
    split: DatasetSplitName,
    selected_record_ids: list[str],
    backend: str,
    trainer_mode: str,
    output_dir: Path,
    max_records: int | None = None,
) -> TrainingProtocolManifest:
    """Build training protocol manifest before execution.

    This manifest serves as the audit trail for what the run was allowed to use.
    """
    policy = get_default_policy_for_split(split)

    return TrainingProtocolManifest(
        run_id=run_id,
        split_manifest_path=str(split_manifest_path),
        active_split=split,
        selected_record_ids=selected_record_ids,
        max_records=max_records,
        memory_builder_input_policy=",".join(policy.memory_builder_allowed_inputs),
        question_agent_input_policy=",".join(policy.question_agent_allowed_inputs),
        answerer_input_policy="current_memory,generated_questions",
        reward_input_policy=",".join(policy.reward_allowed_inputs),
        no_leakage_policy=policy.benchmark_question_policy,
        backend=backend,
        trainer_mode=trainer_mode,
        output_dir=str(output_dir),
        created_at=datetime.utcnow().isoformat(),
        policy=policy,
    )


def run_split_smoke(
    split_manifest: DatasetSplitManifest,
    split: DatasetSplitName,
    output_dir: Path,
    backend: Phase3E2EBackend,
    trainer_mode: str,
    split_manifest_path: Path | None = None,
    max_records: int | None = None,
    round_id: int = 0,
    step_id: int = 0,
    seed: int = 0,
    memory_builder_model_path: str | None = None,
    question_agent_model_path: str | None = None,
    answerer_model_path: str | None = None,
    update_memory_builder: bool | None = None,
    update_question_agent: bool | None = None,
    overwrite: bool = False,
    continue_on_record_failure: bool = False,
    strict: bool = False,
) -> SplitRunReport:
    """Run Phase 4 split smoke across selected records.

    Args:
        split_manifest: Dataset split manifest
        split: Which split to run (train/dev/test)
        output_dir: Output directory for split run
        backend: Phase 3 backend (local_fallback, ray, etc.)
        trainer_mode: Trainer mode (dry_run, local_model, vendored_verl)
        split_manifest_path: Original split manifest path for audit reports
        max_records: Optional limit on records
        round_id: Round ID for this run
        step_id: Step ID for this run
        seed: Random seed
        memory_builder_model_path: Optional model path for Memory Builder
        question_agent_model_path: Optional model path for Question Agent
        answerer_model_path: Optional model path for Answerer
        update_memory_builder: Override actor update policy for Memory Builder
        update_question_agent: Override actor update policy for Question Agent
        overwrite: Overwrite existing output directory
        continue_on_record_failure: Continue on per-record failure
        strict: Strict mode for validation

    Returns:
        SplitRunReport with aggregated per-record results
    """
    run_id = f"phase4_{split.value}_{split_manifest.dataset_name}_round{round_id:03d}_step{step_id:03d}"
    started_at = datetime.utcnow().isoformat()
    split_manifest_path_str = str(split_manifest_path) if split_manifest_path else ""

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=overwrite)

    # Select records
    selected_record_ids = select_split_records(split_manifest, split, max_records)

    if not selected_record_ids:
        # Return empty report with issue
        return SplitRunReport(
            run_id=run_id,
            split_manifest_path=split_manifest_path_str,
            split=split,
            status="failed",
            selected_record_ids=[],
            records=[],
            metrics={"num_selected": 0, "num_succeeded": 0, "num_failed": 0},
            issues=[
                DatasetSplitIssue(
                    severity=DatasetSplitIssueSeverity.ERROR,
                    code="no_records_selected",
                    message=f"No records found for {split.value} split",
                )
            ],
            started_at=started_at,
            finished_at=datetime.utcnow().isoformat(),
        )

    # Build training protocol manifest
    protocol_manifest = build_training_protocol_manifest(
        run_id=run_id,
        split_manifest_path=Path(split_manifest_path_str) if split_manifest_path_str else Path(""),
        split=split,
        selected_record_ids=selected_record_ids,
        backend=backend.value,
        trainer_mode=trainer_mode,
        output_dir=output_dir,
        max_records=max_records,
    )

    # Write protocol manifest
    protocol_manifest_path = output_dir / "training_protocol_manifest.json"
    with open(protocol_manifest_path, "w", encoding="utf-8") as f:
        json.dump(protocol_manifest.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    # Determine actor update policy
    policy = get_default_policy_for_split(split)
    if update_memory_builder is None:
        update_memory_builder = policy.actor_update_policy["memory_builder"]
    if update_question_agent is None:
        update_question_agent = policy.actor_update_policy["question_agent"]

    # Run per-record Phase 3 E2E
    record_reports = []
    records_dir = output_dir / "records"
    records_dir.mkdir(exist_ok=True)

    for record_id in selected_record_ids:
        record_output_dir = records_dir / record_id

        # Build Phase 3 config
        phase3_config = Phase3E2ERunConfig(
            run_id=f"{run_id}_{record_id}",
            input_records_path=split_manifest.source_records_path,
            oracle_graph_dir=split_manifest.oracle_graph_dir,
            record_id=record_id,
            output_dir=str(record_output_dir),
            backend=backend,
            trainer_mode=trainer_mode,
            round_id=round_id,
            step_id=step_id,
            seed=seed,
            max_records=1,
            memory_builder_model_path=memory_builder_model_path,
            question_agent_model_path=question_agent_model_path,
            answerer_model_path=answerer_model_path,
            update_memory_builder=update_memory_builder,
            update_question_agent=update_question_agent,
            overwrite=overwrite,
            strict=strict,
        )

        try:
            # Run Phase 3 E2E for this record
            phase3_report = run_phase3_end_to_end_grpo(phase3_config)

            # Extract metrics
            memory_reward = None
            question_reward = None
            replay_items = 0
            weakness_updates = 0

            if phase3_report.round_reports:
                round_report = phase3_report.round_reports[0]
                memory_reward = round_report.memory_reward_mean
                question_reward = round_report.question_reward_mean
                replay_items = round_report.num_replay_items
                weakness_updates = round_report.num_weakness_updates

            record_reports.append(
                SplitRunRecordReport(
                    record_id=record_id,
                    split=split,
                    status=phase3_report.status.value,
                    backend=phase3_report.backend.value,
                    no_leakage_passed=phase3_report.no_leakage_passed,
                    remote_gpu_ready=phase3_report.remote_gpu_ready,
                    code_a1_verl_ready=phase3_report.code_a1_verl_ready,
                    memory_reward_mean=memory_reward,
                    question_reward_mean=question_reward,
                    replay_items=replay_items,
                    weakness_updates=weakness_updates,
                    output_dir=str(record_output_dir),
                    errors=phase3_report.errors,
                    warnings=phase3_report.warnings,
                )
            )

        except Exception as e:
            record_reports.append(
                SplitRunRecordReport(
                    record_id=record_id,
                    split=split,
                    status="failed",
                    backend=backend.value,
                    no_leakage_passed=False,
                    output_dir=str(record_output_dir),
                    errors=[f"Phase 3 execution failed: {type(e).__name__}: {e}"],
                )
            )

            if not continue_on_record_failure:
                break

    # Aggregate metrics
    num_succeeded = sum(1 for r in record_reports if r.status == "succeeded")
    num_failed = sum(1 for r in record_reports if r.status == "failed")
    num_partial = sum(1 for r in record_reports if r.status == "partial")

    memory_rewards = [r.memory_reward_mean for r in record_reports if r.memory_reward_mean is not None]
    question_rewards = [r.question_reward_mean for r in record_reports if r.question_reward_mean is not None]

    metrics = {
        "num_selected": len(selected_record_ids),
        "num_executed": len(record_reports),
        "num_succeeded": num_succeeded,
        "num_failed": num_failed,
        "num_partial": num_partial,
        "mean_memory_reward": sum(memory_rewards) / len(memory_rewards) if memory_rewards else None,
        "mean_question_reward": sum(question_rewards) / len(question_rewards) if question_rewards else None,
        "total_replay_items": sum(r.replay_items for r in record_reports),
        "total_weakness_updates": sum(r.weakness_updates for r in record_reports),
        "no_leakage_pass_count": sum(1 for r in record_reports if r.no_leakage_passed),
        "code_a1_verl_ready_count": sum(
            1 for r in record_reports if r.code_a1_verl_ready
        ),
    }

    # Determine overall status
    if num_failed == 0 and num_succeeded == len(record_reports):
        status = "succeeded"
    elif num_succeeded > 0:
        status = "partial"
    else:
        status = "failed"

    finished_at = datetime.utcnow().isoformat()

    split_run_report = SplitRunReport(
        run_id=run_id,
        split_manifest_path=split_manifest_path_str,
        split=split,
        status=status,
        selected_record_ids=selected_record_ids,
        records=record_reports,
        metrics=metrics,
        protocol_manifest_path=str(protocol_manifest_path),
        started_at=started_at,
        finished_at=finished_at,
    )

    # Write split run report
    split_report_path = output_dir / "split_run_report.json"
    with open(split_report_path, "w", encoding="utf-8") as f:
        json.dump(split_run_report.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    return split_run_report


def aggregate_split_reports(reports: list[SplitRunRecordReport]) -> dict:
    """Aggregate per-record reports into summary statistics."""
    if not reports:
        return {}

    return {
        "total_records": len(reports),
        "succeeded": sum(1 for r in reports if r.status == "succeeded"),
        "failed": sum(1 for r in reports if r.status == "failed"),
        "partial": sum(1 for r in reports if r.status == "partial"),
        "no_leakage_passed": sum(1 for r in reports if r.no_leakage_passed),
        "total_replay_items": sum(r.replay_items for r in reports),
        "total_weakness_updates": sum(r.weakness_updates for r in reports),
    }


def write_split_summary_markdown(report: SplitRunReport, output_path: Path) -> None:
    """Write human-readable summary markdown for split run."""
    lines = []
    lines.append(f"# Phase 4 Split Run: {report.split.value}\n")
    lines.append(f"- **Run ID:** {report.run_id}")
    lines.append(f"- **Split:** {report.split.value}")
    lines.append(f"- **Status:** {report.status}")
    lines.append(f"- **Selected records:** {len(report.selected_record_ids)}")
    lines.append(f"- **Executed:** {report.metrics.get('num_executed', 0)}")
    lines.append(f"- **Succeeded:** {report.metrics.get('num_succeeded', 0)}")
    lines.append(f"- **Failed:** {report.metrics.get('num_failed', 0)}")
    lines.append(f"- **Partial:** {report.metrics.get('num_partial', 0)}")
    lines.append("")

    if report.metrics.get("mean_memory_reward") is not None:
        lines.append(f"- **Mean memory reward:** {report.metrics['mean_memory_reward']:.3f}")
    if report.metrics.get("mean_question_reward") is not None:
        lines.append(f"- **Mean question reward:** {report.metrics['mean_question_reward']:.3f}")

    lines.append(f"- **Total replay items:** {report.metrics.get('total_replay_items', 0)}")
    lines.append(f"- **Total weakness updates:** {report.metrics.get('total_weakness_updates', 0)}")
    lines.append(f"- **No-leakage passed:** {report.metrics.get('no_leakage_pass_count', 0)}/{len(report.records)}")
    lines.append("")

    lines.append("## Per-Record Results\n")
    lines.append("| Record ID | Status | Memory Reward | Question Reward | Replay Items |")
    lines.append("| --- | --- | --- | --- | --- |")

    for record in report.records:
        memory_reward = f"{record.memory_reward_mean:.3f}" if record.memory_reward_mean is not None else "—"
        question_reward = f"{record.question_reward_mean:.3f}" if record.question_reward_mean is not None else "—"
        lines.append(
            f"| {record.record_id} | {record.status} | {memory_reward} | {question_reward} | {record.replay_items} |"
        )

    lines.append("")

    if report.issues:
        lines.append("## Issues\n")
        for issue in report.issues:
            lines.append(f"- **{issue.severity.value.upper()}**: {issue.message}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
