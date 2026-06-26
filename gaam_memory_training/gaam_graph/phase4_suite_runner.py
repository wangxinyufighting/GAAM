"""
Phase 4 Milestone 6: Benchmark Suite Runner

Run multiple Phase 4 production experiments as a benchmark suite.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from gaam_graph.phase4_baseline_eval import run_phase4_baseline_evaluations
from gaam_graph.phase4_cotraining_loop import run_phase4_cotraining_loop
from gaam_graph.phase4_cotraining_schema import (
    Phase4CotrainingConfig,
    Phase4RoundStatus,
)
from gaam_graph.phase4_experiment_report import export_experiment_reports
from gaam_graph.phase4_experiment_schema import (
    Phase4FinalEvaluationConfig,
    Phase4ProductionExperimentConfig,
    Phase4ProductionExperimentManifest,
    generate_experiment_id,
)
from gaam_graph.phase4_final_evaluation import (
    run_phase4_final_evaluation,
    write_final_evaluation_summary,
)
from gaam_graph.phase4_leakage_audit import (
    run_phase4_leakage_audit,
    write_leakage_audit_report,
)
from gaam_graph.phase4_suite_schema import (
    Phase4BenchmarkSuiteConfig,
    Phase4BenchmarkSuiteManifest,
    Phase4SuiteRunRecord,
    aggregate_suite_status,
    generate_suite_id,
)


def build_experiment_config_for_seed(
    suite_config: Phase4BenchmarkSuiteConfig,
    seed: int,
    output_dir: Path,
) -> Phase4ProductionExperimentConfig:
    """Build production experiment config for one seed."""
    from gaam_graph.phase4_trainer_schema import Phase4TrainerBackend

    # Map string to enum
    trainer_backend = Phase4TrainerBackend(suite_config.trainer_backend)

    return Phase4ProductionExperimentConfig(
        split_manifest_path=suite_config.split_manifest_path,
        output_dir=str(output_dir),
        num_rounds=suite_config.num_rounds,
        train_records_per_round=suite_config.train_records_per_round,
        dev_records_per_round=suite_config.dev_records_per_round,
        test_records_limit=suite_config.test_records_limit,
        rollout_backend=suite_config.rollout_backend,
        trainer_backend=trainer_backend,
        trainer_mode=suite_config.trainer_mode,
        memory_builder_model_path=suite_config.memory_builder_model_path,
        question_agent_model_path=suite_config.question_agent_model_path,
        answerer_model_path=suite_config.answerer_model_path,
        final_test_eval=suite_config.final_test_eval,
        run_baselines=suite_config.run_baselines,
        baseline_names=suite_config.baseline_names,
        strict_no_leakage=suite_config.strict_no_leakage,
        seed=seed,
        overwrite=suite_config.overwrite,
        resume=suite_config.resume,
    )


def should_run_seed(
    run_record: Phase4SuiteRunRecord | None,
    force_rerun_succeeded: bool,
) -> bool:
    """Determine if a seed should be run based on its previous status."""
    if run_record is None:
        return True

    if run_record.status == "succeeded":
        return force_rerun_succeeded

    # Rerun partial or failed
    return True


def run_seed_experiment(
    suite_config: Phase4BenchmarkSuiteConfig,
    seed: int,
    output_dir: Path,
) -> Phase4SuiteRunRecord:
    """
    Run production experiment for one seed.

    Returns run record with status and metrics.
    """
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        # Build experiment config
        exp_config = build_experiment_config_for_seed(suite_config, seed, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
            json.dump(exp_config.model_dump(mode="json"), f, indent=2)

        experiment_manifest = Phase4ProductionExperimentManifest(
            experiment_id=generate_experiment_id(
                seed=seed,
                timestamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            ),
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            split_manifest_path=exp_config.split_manifest_path,
            output_dir=str(output_dir),
        )

        # Run co-training loop
        cotraining_dir = output_dir / "cotraining"
        cotraining_config = Phase4CotrainingConfig(
            split_manifest_path=exp_config.split_manifest_path,
            output_dir=str(cotraining_dir),
            num_rounds=exp_config.num_rounds,
            train_records_per_round=exp_config.train_records_per_round,
            dev_records_per_round=exp_config.dev_records_per_round,
            rollout_backend=exp_config.rollout_backend,
            trainer_backend=exp_config.trainer_backend,
            trainer_mode=exp_config.trainer_mode,
            memory_builder_model_path=exp_config.memory_builder_model_path,
            question_agent_model_path=exp_config.question_agent_model_path,
            answerer_model_path=exp_config.answerer_model_path,
            seed=seed,
            overwrite=exp_config.overwrite,
            resume=exp_config.resume,
            strict_no_leakage=exp_config.strict_no_leakage,
            allow_empty_batch=suite_config.allow_empty_batch,
        )
        cotraining_manifest = run_phase4_cotraining_loop(cotraining_config)
        cotraining_errors = [
            f"round_{round_report.round_id}: {error}"
            for round_report in cotraining_manifest.round_reports
            for error in round_report.errors
        ]
        cotraining_warnings = [
            f"round_{round_report.round_id}: {warning}"
            for round_report in cotraining_manifest.round_reports
            for warning in round_report.warnings
        ]

        experiment_manifest.cotraining_dir = str(cotraining_dir)
        experiment_manifest.cotraining_manifest_path = str(
            cotraining_dir / "cotraining_manifest.json"
        )
        if cotraining_manifest.status == Phase4RoundStatus.FAILED:
            experiment_manifest.errors.append("Co-training loop failed.")
        experiment_manifest.errors.extend(cotraining_errors)
        experiment_manifest.warnings.extend(cotraining_warnings)

        final_round = (
            cotraining_manifest.round_reports[-1]
            if cotraining_manifest.round_reports
            else None
        )
        if final_round is not None:
            experiment_manifest.final_memory_builder_checkpoint_id = (
                final_round.selected_memory_builder_checkpoint_id
            )
            experiment_manifest.final_question_agent_checkpoint_id = (
                final_round.selected_question_agent_checkpoint_id
            )
            experiment_manifest.final_checkpoint_registry_path = final_round.metrics.get(
                "checkpoint_registry_path"
            )

        # Run final test evaluation if requested
        final_eval_manifest = None
        if (
            suite_config.final_test_eval
            and cotraining_manifest.status == Phase4RoundStatus.SUCCEEDED
            and final_round is not None
            and final_round.selected_memory_builder_checkpoint_id
        ):
            eval_config = Phase4FinalEvaluationConfig(
                split_manifest_path=exp_config.split_manifest_path,
                output_dir=str(output_dir / "final_evaluation"),
                checkpoint_registry_path=experiment_manifest.final_checkpoint_registry_path,
                memory_builder_checkpoint_id=final_round.selected_memory_builder_checkpoint_id,
                question_agent_checkpoint_id=final_round.selected_question_agent_checkpoint_id,
                memory_builder_model_path=exp_config.memory_builder_model_path,
                question_agent_model_path=exp_config.question_agent_model_path,
                answerer_model_path=exp_config.answerer_model_path,
                test_records_limit=exp_config.test_records_limit,
                seed=seed,
                overwrite=exp_config.overwrite,
            )

            final_eval_manifest = run_phase4_final_evaluation(eval_config)
            write_final_evaluation_summary(final_eval_manifest, output_dir / "final_evaluation")
            experiment_manifest.final_evaluation_dir = str(output_dir / "final_evaluation")
            experiment_manifest.final_evaluation_manifest_path = str(
                output_dir / "final_evaluation" / "final_evaluation_manifest.json"
            )
        elif suite_config.final_test_eval:
            experiment_manifest.warnings.append(
                "Final test evaluation requested but no selected final checkpoint was available."
            )

        if suite_config.run_baselines:
            run_phase4_baseline_evaluations(
                split_manifest_path=exp_config.split_manifest_path,
                output_dir=output_dir / "final_evaluation" / "baselines",
                baselines=exp_config.baseline_names,
                initial_memory_builder_checkpoint_id=exp_config.initial_memory_builder_checkpoint_id,
                initial_question_agent_checkpoint_id=exp_config.initial_question_agent_checkpoint_id,
                final_memory_builder_checkpoint_id=experiment_manifest.final_memory_builder_checkpoint_id,
                final_question_agent_checkpoint_id=experiment_manifest.final_question_agent_checkpoint_id,
                checkpoint_registry_path=experiment_manifest.final_checkpoint_registry_path,
                memory_builder_model_path=exp_config.memory_builder_model_path,
                question_agent_model_path=exp_config.question_agent_model_path,
                answerer_model_path=exp_config.answerer_model_path,
                test_records_limit=exp_config.test_records_limit,
                seed=seed,
                overwrite=exp_config.overwrite,
            )

        if exp_config.strict_no_leakage:
            audit_report = run_phase4_leakage_audit(output_dir, strict=True)
            audit_path = output_dir / "reports" / "leakage_audit.json"
            write_leakage_audit_report(audit_report, audit_path)
            if audit_report.status == "failed":
                experiment_manifest.errors.append(
                    f"Leakage audit failed with {len(audit_report.issues)} issues"
                )

        export_experiment_reports(output_dir)
        experiment_manifest.reports_dir = str(output_dir / "reports")

        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Determine overall status
        if experiment_manifest.errors:
            status = "failed"
        elif cotraining_manifest.status == Phase4RoundStatus.SUCCEEDED:
            if final_eval_manifest is None or final_eval_manifest.status == "succeeded":
                status = "succeeded"
            else:
                status = "partial"
        else:
            status = cotraining_manifest.status.value

        # Collect metrics
        metrics = {}
        if final_eval_manifest and final_eval_manifest.metrics:
            metrics.update(final_eval_manifest.metrics)

        experiment_manifest.status = status
        experiment_manifest.finished_at = datetime.now(timezone.utc).isoformat()
        experiment_manifest.metrics.update(metrics)
        with open(output_dir / "experiment_manifest.json", "w", encoding="utf-8") as f:
            json.dump(experiment_manifest.model_dump(mode="json"), f, indent=2)

        return Phase4SuiteRunRecord(
            seed=seed,
            status=status,
            output_dir=str(output_dir),
            experiment_manifest_path=str(output_dir / "experiment_manifest.json"),
            started_at=started_at,
            finished_at=finished_at,
            final_memory_builder_checkpoint_id=(
                final_round.selected_memory_builder_checkpoint_id
                if cotraining_manifest.round_reports
                else None
            ),
            final_question_agent_checkpoint_id=(
                final_round.selected_question_agent_checkpoint_id
                if cotraining_manifest.round_reports
                else None
            ),
            metrics=metrics,
            warnings=experiment_manifest.warnings,
            errors=experiment_manifest.errors,
        )

    except Exception as exc:
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return Phase4SuiteRunRecord(
            seed=seed,
            status="failed",
            output_dir=str(output_dir),
            started_at=started_at,
            finished_at=finished_at,
            errors=[str(exc)],
        )


def write_suite_manifest(
    manifest: Phase4BenchmarkSuiteManifest,
    output_path: Path,
) -> None:
    """Write suite manifest to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)


def load_suite_manifest(manifest_path: Path) -> Phase4BenchmarkSuiteManifest:
    """Load suite manifest from JSON file."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Phase4BenchmarkSuiteManifest.model_validate(data)


def run_phase4_benchmark_suite(
    config: Phase4BenchmarkSuiteConfig,
) -> Phase4BenchmarkSuiteManifest:
    """
    Run benchmark suite with multiple seeds.

    Executes one Phase 4 production experiment per seed.
    Writes suite manifest after each seed for resume capability.

    Returns:
        Suite manifest with run records for all seeds
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate suite ID
    suite_id = generate_suite_id(suite_name=config.suite_name)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Check for existing manifest if resuming
    manifest_path = output_dir / "suite_manifest.json"
    existing_run_records = {}

    if config.resume and manifest_path.exists():
        existing_manifest = load_suite_manifest(manifest_path)
        suite_id = existing_manifest.suite_id  # Preserve suite ID
        existing_run_records = {r.seed: r for r in existing_manifest.run_records}

    # Initialize manifest
    manifest = Phase4BenchmarkSuiteManifest(
        suite_id=suite_id,
        suite_name=config.suite_name,
        status="running",
        started_at=started_at,
        suite_config_path=None,  # Set externally if needed
        resolved_config_path=str(output_dir / "suite_config.resolved.json"),
        output_dir=str(output_dir),
        run_records=[],
    )

    # Run each seed
    for seed in config.seeds:
        seed_output_dir = output_dir / "runs" / f"seed_{seed}"

        # Check if we should run this seed
        existing_record = existing_run_records.get(seed)
        if not should_run_seed(existing_record, config.force_rerun_succeeded):
            # Skip this seed, keep existing record
            manifest.run_records.append(existing_record)
            continue

        # Run seed experiment
        run_record = run_seed_experiment(config, seed, seed_output_dir)
        manifest.run_records.append(run_record)

        # Update suite status
        manifest.status = aggregate_suite_status(manifest.run_records)

        # Write manifest after each seed for resume capability
        write_suite_manifest(manifest, manifest_path)

    # Finalize manifest
    manifest.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest.status = aggregate_suite_status(manifest.run_records)

    # Compute suite-level metrics
    num_succeeded = sum(1 for r in manifest.run_records if r.status == "succeeded")
    num_partial = sum(1 for r in manifest.run_records if r.status == "partial")
    num_failed = sum(1 for r in manifest.run_records if r.status == "failed")

    manifest.metrics = {
        "num_seeds_requested": len(config.seeds),
        "num_seeds_succeeded": num_succeeded,
        "num_seeds_partial": num_partial,
        "num_seeds_failed": num_failed,
    }

    # Write final manifest
    write_suite_manifest(manifest, manifest_path)

    return manifest
