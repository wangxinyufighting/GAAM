"""
Phase 4 Milestone 4: Co-Training Loop Runner

Main orchestration for multi-round adversarial co-training.

Key functions:
- run_phase4_cotraining_loop: top-level loop
- run_phase4_cotraining_round: one round execution
- evaluate_early_stopping: early stopping decision

Design principle:
Honest orchestration: if something fails, record it and decide whether to continue.
Never pretend success when there was failure.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from gaam_graph.dataset_split_schema import DatasetSplitName
from gaam_graph.dataset_splitter import load_dataset_split_manifest
from gaam_graph.phase4_checkpoint_selection import (
    select_checkpoints_with_optional_dev_metrics,
    write_checkpoint_selection_report,
)
from gaam_graph.phase4_cotraining_schema import (
    Phase4CotrainingConfig,
    Phase4CotrainingManifest,
    Phase4EarlyStopReport,
    Phase4RoundPlan,
    Phase4RoundReport,
    Phase4RoundStatus,
    aggregate_round_status,
    compute_experiment_metrics,
    generate_cotraining_run_id,
)
from gaam_graph.phase4_grpo_step import run_phase4_grpo_trainer_step
from gaam_graph.phase4_multi_case_rollout import run_multi_case_rollout
from gaam_graph.phase4_rollout_schema import MultiCaseRolloutManifest
from gaam_graph.phase4_round_scheduler import (
    build_round_plan,
    get_previous_round_report,
)
from gaam_graph.phase4_trainer_schema import (
    Phase4TrainerStepConfig,
    Phase4TrainerStepManifest,
)
from gaam_graph.grpo_schema import ActorRole


def run_phase4_cotraining_loop(
    config: Phase4CotrainingConfig,
) -> Phase4CotrainingManifest:
    """
    Run multi-round adversarial co-training loop.

    Pipeline per round:
    1. Train rollout on train split
    2. Trainer step updates actors
    3. Dev rollout using provisional checkpoints
    4. Checkpoint selection
    5. Carry selected checkpoints to next round

    Args:
        config: experiment configuration

    Returns:
        Top-level co-training manifest
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split manifest
    split_manifest = load_dataset_split_manifest(Path(config.split_manifest_path))

    # Check for resume
    manifest_path = output_dir / "cotraining_manifest.json"
    if config.resume and manifest_path.exists():
        manifest = load_cotraining_manifest(manifest_path)
        print(f"Resuming from round {manifest.num_rounds_completed}")
    else:
        # Create new manifest
        if manifest_path.exists() and not config.overwrite:
            raise FileExistsError(
                f"Manifest already exists: {manifest_path}. Use --overwrite or --resume."
            )

        run_id = generate_cotraining_run_id(
            seed=config.seed,
            timestamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        )

        manifest = Phase4CotrainingManifest(
            run_id=run_id,
            split_manifest_path=config.split_manifest_path,
            output_dir=str(output_dir),
            num_rounds_requested=config.num_rounds,
            status=Phase4RoundStatus.RUNNING,
            started_at=datetime.now(timezone.utc).isoformat(),
            config=config.model_dump(),
        )

        # Write initial manifest
        write_cotraining_manifest(manifest, manifest_path)

    # Run rounds
    for round_id in range(manifest.num_rounds_completed, config.num_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_id}/{config.num_rounds - 1}")
        print(f"{'='*60}")

        # Get previous round for checkpoint carryover
        previous_round = get_previous_round_report(manifest.round_reports, round_id)

        # Build round plan
        round_plan = build_round_plan(config, split_manifest, round_id, previous_round)

        # Write round plan
        round_plan_path = Path(round_plan.output_dir) / "round_plan.json"
        round_plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(round_plan_path, "w", encoding="utf-8") as f:
            json.dump(round_plan.model_dump(), f, indent=2)

        # Run round
        round_report = run_phase4_cotraining_round(config, round_plan)

        # Update manifest
        manifest.round_reports.append(round_report)
        manifest.num_rounds_completed = round_id + 1

        # Update final checkpoint IDs
        if round_report.selected_memory_builder_checkpoint_id:
            manifest.final_memory_builder_checkpoint_id = (
                round_report.selected_memory_builder_checkpoint_id
            )
        if round_report.selected_question_agent_checkpoint_id:
            manifest.final_question_agent_checkpoint_id = (
                round_report.selected_question_agent_checkpoint_id
            )

        # Write manifest after each round
        write_cotraining_manifest(manifest, manifest_path)

        if (
            round_report.status == Phase4RoundStatus.FAILED
            and not config.continue_on_round_failure
        ):
            manifest.status = Phase4RoundStatus.FAILED
            manifest.finished_at = datetime.now(timezone.utc).isoformat()
            manifest.metrics.update(compute_experiment_metrics(manifest))
            write_cotraining_manifest(manifest, manifest_path)
            write_cotraining_summary(manifest, output_dir)
            return manifest

        # Check early stopping
        if config.early_stop_patience is not None:
            early_stop_report = evaluate_early_stopping(
                manifest.round_reports,
                metric_name=config.early_stop_metric,
                patience=config.early_stop_patience,
                min_delta=config.early_stop_min_delta,
            )

            if early_stop_report.should_stop:
                print(f"\nEarly stopping: {early_stop_report.reason}")
                early_stop_path = output_dir / "early_stop_report.json"
                with open(early_stop_path, "w", encoding="utf-8") as f:
                    json.dump(early_stop_report.model_dump(), f, indent=2)

                manifest.status = Phase4RoundStatus.SUCCEEDED
                manifest.finished_at = datetime.now(timezone.utc).isoformat()
                manifest.metrics["early_stopped"] = True
                manifest.metrics["early_stop_reason"] = early_stop_report.reason
                write_cotraining_manifest(manifest, manifest_path)
                write_cotraining_summary(manifest, output_dir)
                return manifest

    # All rounds completed
    manifest.status = aggregate_round_status(manifest.round_reports)
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    manifest.metrics.update(compute_experiment_metrics(manifest))

    write_cotraining_manifest(manifest, manifest_path)
    write_cotraining_summary(manifest, output_dir)

    return manifest


def run_phase4_cotraining_round(
    config: Phase4CotrainingConfig,
    round_plan: Phase4RoundPlan,
) -> Phase4RoundReport:
    """
    Run one training round.

    Stages:
    1. Train rollout
    2. Trainer step
    3. Dev rollout
    4. Checkpoint selection

    Args:
        config: experiment configuration
        round_plan: round execution plan

    Returns:
        Round report
    """
    round_id = round_plan.round_id
    round_dir = Path(round_plan.output_dir)
    round_dir.mkdir(parents=True, exist_ok=True)

    report = Phase4RoundReport(
        round_id=round_id,
        status=Phase4RoundStatus.RUNNING,
        output_dir=str(round_dir),
        input_memory_builder_checkpoint_id=round_plan.memory_builder_checkpoint_id,
        input_question_agent_checkpoint_id=round_plan.question_agent_checkpoint_id,
    )

    try:
        # Stage 1: Train rollout
        print(f"\n[Round {round_id}] Stage 1: Train rollout")
        train_rollout_dir = round_dir / "train_rollout"
        train_manifest = run_multi_case_rollout(
            split_manifest_path=Path(config.split_manifest_path),
            split=DatasetSplitName.TRAIN,
            output_dir=train_rollout_dir,
            backend=config.rollout_backend,
            trainer_mode=config.trainer_mode,
            record_ids=round_plan.train_record_ids,
            round_id=round_id,
            step_id=0,
            checkpoint_registry_path=round_plan.checkpoint_registry_path,
            memory_builder_checkpoint_id=round_plan.memory_builder_checkpoint_id,
            question_agent_checkpoint_id=round_plan.question_agent_checkpoint_id,
            memory_builder_model_path=config.memory_builder_model_path,
            question_agent_model_path=config.question_agent_model_path,
            answerer_model_path=config.answerer_model_path,
            update_memory_builder=config.update_memory_builder,
            update_question_agent=config.update_question_agent,
            continue_on_record_failure=config.continue_on_record_failure,
            allow_empty_batch=config.allow_empty_batch,
            overwrite=True,
        )

        report.train_rollout_dir = str(train_rollout_dir)
        report.train_rollout_manifest_path = str(
            train_rollout_dir / "multi_case_rollout_manifest.json"
        )
        report.train_memory_reward_mean = train_manifest.metrics.get(
            "memory_reward_mean"
        )
        report.train_question_reward_mean = train_manifest.metrics.get(
            "question_reward_mean"
        )
        report.train_no_leakage_pass_rate = train_manifest.metrics.get(
            "no_leakage_pass_rate"
        )

        # Stage 2: Trainer step
        print(f"\n[Round {round_id}] Stage 2: Trainer step")
        trainer_step_dir = round_dir / "trainer_step"
        trainer_actors: list[ActorRole] = []
        if config.update_memory_builder:
            trainer_actors.append(ActorRole.MEMORY_BUILDER)
        if config.update_question_agent:
            trainer_actors.append(ActorRole.QUESTION_AGENT)

        trainer_config = Phase4TrainerStepConfig(
            source_multi_case_run_dir=str(train_rollout_dir),
            output_dir=str(trainer_step_dir),
            backend=config.trainer_backend,
            round_id=round_id,
            step_id=0,
            actors=trainer_actors,
            memory_builder_model_path=config.memory_builder_model_path,
            question_agent_model_path=config.question_agent_model_path,
            memory_builder_checkpoint_id=round_plan.memory_builder_checkpoint_id,
            question_agent_checkpoint_id=round_plan.question_agent_checkpoint_id,
            checkpoint_registry_path=str(
                trainer_step_dir / "checkpoint_registry.json"
            ),
            allow_empty_batch=config.allow_empty_batch,
            strict_no_leakage=config.strict_no_leakage,
            overwrite=True,
        )

        trainer_manifest = run_phase4_grpo_trainer_step(trainer_config)

        report.trainer_step_dir = str(trainer_step_dir)
        report.trainer_step_manifest_path = str(
            trainer_step_dir / "trainer_step_manifest.json"
        )

        # Stage 3: Dev rollout
        print(f"\n[Round {round_id}] Stage 3: Dev rollout")
        dev_rollout_dir = round_dir / "dev_rollout"

        # Dev rollout uses provisional checkpoints from trainer step
        # (evaluation only, no model updates)
        dev_manifest = run_multi_case_rollout(
            split_manifest_path=Path(config.split_manifest_path),
            split=DatasetSplitName.DEV,
            output_dir=dev_rollout_dir,
            backend=config.rollout_backend,
            trainer_mode="dry_run",
            record_ids=round_plan.dev_record_ids,
            round_id=round_id,
            step_id=0,
            checkpoint_registry_path=trainer_manifest.checkpoint_registry_path,
            memory_builder_model_path=config.memory_builder_model_path,
            question_agent_model_path=config.question_agent_model_path,
            answerer_model_path=config.answerer_model_path,
            update_memory_builder=False,
            update_question_agent=False,
            continue_on_record_failure=config.continue_on_record_failure,
            allow_empty_batch=config.allow_empty_batch,
            overwrite=True,
        )

        report.dev_rollout_dir = str(dev_rollout_dir)
        report.dev_rollout_manifest_path = str(
            dev_rollout_dir / "multi_case_rollout_manifest.json"
        )
        report.dev_memory_reward_mean = dev_manifest.metrics.get("memory_reward_mean")
        report.dev_question_reward_mean = dev_manifest.metrics.get(
            "question_reward_mean"
        )
        report.dev_no_leakage_pass_rate = dev_manifest.metrics.get(
            "no_leakage_pass_rate"
        )

        # Stage 4: Checkpoint selection
        print(f"\n[Round {round_id}] Stage 4: Checkpoint selection")
        selection_report = select_checkpoints_with_optional_dev_metrics(
            trainer_manifest,
            dev_rollout_manifest=dev_manifest,
            max_dev_regression=config.max_dev_regression,
        )

        checkpoint_selection_path = round_dir / "checkpoint_selection.json"
        write_checkpoint_selection_report(selection_report, checkpoint_selection_path)

        report.checkpoint_selection_path = str(checkpoint_selection_path)
        report.selected_memory_builder_checkpoint_id = (
            selection_report.selected_memory_builder_checkpoint_id
            or report.input_memory_builder_checkpoint_id
        )
        report.selected_question_agent_checkpoint_id = (
            selection_report.selected_question_agent_checkpoint_id
            or report.input_question_agent_checkpoint_id
        )
        report.metrics["checkpoint_registry_path"] = (
            trainer_manifest.checkpoint_registry_path
        )
        report.metrics["trainer_status"] = trainer_manifest.status
        report.metrics["selection_policy"] = selection_report.selection_policy

        # Collect warnings from selection
        if selection_report.warnings:
            report.warnings.extend(selection_report.warnings)

        # Determine round status
        if trainer_manifest.status == "succeeded":
            if selection_report.warnings:
                report.status = Phase4RoundStatus.PARTIAL
            else:
                report.status = Phase4RoundStatus.SUCCEEDED
        elif trainer_manifest.status == "partial":
            report.status = Phase4RoundStatus.PARTIAL
        else:
            report.status = Phase4RoundStatus.FAILED

    except Exception as e:
        report.status = Phase4RoundStatus.FAILED
        report.errors.append(str(e))

    finally:
        # Write round summary
        write_round_summary(report, round_dir / "round_summary.json")

    return report


def evaluate_early_stopping(
    round_reports: list[Phase4RoundReport],
    *,
    metric_name: str,
    patience: int | None,
    min_delta: float,
) -> Phase4EarlyStopReport:
    """
    Evaluate early stopping condition.

    Policy:
    - Track best dev metric
    - If no improvement > min_delta for patience rounds, stop

    Args:
        round_reports: all round reports so far
        metric_name: metric to track (e.g., "dev_memory_reward_mean")
        patience: number of rounds without improvement before stopping
        min_delta: minimum improvement to count as progress

    Returns:
        Early stopping decision report
    """
    if patience is None:
        return Phase4EarlyStopReport(
            should_stop=False,
            metric_name=metric_name,
            patience=None,
        )

    # Extract metric values (larger is better)
    values = []
    for r in round_reports:
        if metric_name in r.model_dump():
            value = getattr(r, metric_name)
            if value is not None:
                values.append((r.round_id, value))

    if not values:
        # No values yet, don't stop
        return Phase4EarlyStopReport(
            should_stop=False,
            metric_name=metric_name,
            patience=patience,
        )

    best_round_id, best_value = values[0]
    rounds_without_improvement = 0

    for round_id, value in values[1:]:
        if value > best_value + min_delta:
            best_round_id = round_id
            best_value = value
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

    latest_round_id, latest_value = values[-1]

    should_stop = rounds_without_improvement >= patience

    reason = None
    if should_stop:
        reason = f"No improvement in {metric_name} for {patience} rounds (best: {best_value:.4f} at round {best_round_id}, latest: {latest_value:.4f})"

    return Phase4EarlyStopReport(
        should_stop=should_stop,
        reason=reason,
        metric_name=metric_name,
        best_value=best_value,
        latest_value=latest_value,
        best_round_id=best_round_id,
        patience=patience,
        rounds_without_improvement=rounds_without_improvement,
    )


def write_cotraining_manifest(
    manifest: Phase4CotrainingManifest,
    output_path: Path,
) -> None:
    """Write co-training manifest to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)


def load_cotraining_manifest(manifest_path: Path) -> Phase4CotrainingManifest:
    """Load co-training manifest from JSON."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Phase4CotrainingManifest.model_validate(data)


def write_round_summary(
    report: Phase4RoundReport,
    output_path: Path,
) -> None:
    """Write round summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)


def write_cotraining_summary(
    manifest: Phase4CotrainingManifest,
    output_dir: Path,
) -> None:
    """Write human-readable summary markdown."""
    summary_path = output_dir / "summary.md"

    lines = [
        f"# Phase 4 Co-Training Experiment: {manifest.run_id}",
        "",
        f"**Status:** {manifest.status.value}",
        f"**Rounds completed:** {manifest.num_rounds_completed}/{manifest.num_rounds_requested}",
        f"**Started:** {manifest.started_at}",
        f"**Finished:** {manifest.finished_at or 'running'}",
        "",
        "## Configuration",
        "",
        f"- Split manifest: `{manifest.split_manifest_path}`",
        f"- Rollout backend: {manifest.config.get('rollout_backend')}",
        f"- Trainer backend: {manifest.config.get('trainer_backend')}",
        f"- Seed: {manifest.config.get('seed')}",
        "",
        "## Round Summary",
        "",
        "| Round | Status | Train Memory | Train Question | Dev Memory | Dev Question | Selected Checkpoints |",
        "|-------|--------|--------------|----------------|------------|--------------|----------------------|",
    ]

    for r in manifest.round_reports:
        train_mem = (
            f"{r.train_memory_reward_mean:.3f}"
            if r.train_memory_reward_mean is not None
            else "-"
        )
        train_q = (
            f"{r.train_question_reward_mean:.3f}"
            if r.train_question_reward_mean is not None
            else "-"
        )
        dev_mem = (
            f"{r.dev_memory_reward_mean:.3f}"
            if r.dev_memory_reward_mean is not None
            else "-"
        )
        dev_q = (
            f"{r.dev_question_reward_mean:.3f}"
            if r.dev_question_reward_mean is not None
            else "-"
        )

        checkpoints = []
        if r.selected_memory_builder_checkpoint_id:
            checkpoints.append(f"MB:{r.selected_memory_builder_checkpoint_id[:8]}")
        if r.selected_question_agent_checkpoint_id:
            checkpoints.append(f"QA:{r.selected_question_agent_checkpoint_id[:8]}")
        checkpoint_str = ", ".join(checkpoints) if checkpoints else "-"

        lines.append(
            f"| {r.round_id} | {r.status.value} | {train_mem} | {train_q} | {dev_mem} | {dev_q} | {checkpoint_str} |"
        )

    lines.extend([
        "",
        "## Final Metrics",
        "",
    ])

    metrics = manifest.metrics
    if metrics.get("best_dev_memory_reward_mean") is not None:
        lines.append(
            f"- Best dev memory reward: {metrics['best_dev_memory_reward_mean']:.4f} "
            f"(round {metrics.get('best_dev_round_id')})"
        )
    lines.append(f"- Rounds succeeded: {metrics.get('num_rounds_succeeded', 0)}")
    lines.append(f"- Rounds failed: {metrics.get('num_rounds_failed', 0)}")
    lines.append(f"- Rounds partial: {metrics.get('num_rounds_partial', 0)}")

    if metrics.get("early_stopped"):
        lines.append(f"- Early stopped: Yes ({metrics.get('early_stop_reason')})")

    lines.extend([
        "",
        "## Final Checkpoints",
        "",
        f"- Memory Builder: `{manifest.final_memory_builder_checkpoint_id or 'none'}`",
        f"- Question Agent: `{manifest.final_question_agent_checkpoint_id or 'none'}`",
        "",
    ])

    if manifest.warnings:
        lines.extend([
            "## Warnings",
            "",
        ])
        for w in manifest.warnings:
            lines.append(f"- {w}")
        lines.append("")

    if manifest.errors:
        lines.extend([
            "## Errors",
            "",
        ])
        for e in manifest.errors:
            lines.append(f"- {e}")
        lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
