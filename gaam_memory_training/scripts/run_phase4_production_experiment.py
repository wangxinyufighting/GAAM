#!/usr/bin/env python3
"""
Run Phase 4 production experiment with held-out evaluation.

Usage:
    python scripts/run_phase4_production_experiment.py \
      --split_manifest outputs/splits/split.json \
      --output_dir outputs/phase4_experiments/exp001 \
      --num_rounds 3 \
      --train_records_per_round 8 \
      --dev_records_per_round 4

Exit codes:
    0 = succeeded
    1 = failed
    2 = partial
    3 = validation error
    4 = backend not ready
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from gaam_graph.phase4_backend_readiness import check_phase4_backend_readiness
from gaam_graph.phase4_baseline_eval import run_phase4_baseline_evaluations
from gaam_graph.phase4_cotraining_loop import run_phase4_cotraining_loop
from gaam_graph.phase4_cotraining_schema import (
    Phase4CotrainingConfig,
    Phase4RoundStatus,
)
from gaam_graph.phase4_experiment_report import export_experiment_reports
from gaam_graph.phase4_experiment_schema import (
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
from gaam_graph.phase4_trainer_schema import Phase4TrainerBackend


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 4 production experiment with held-out evaluation"
    )

    # Required
    parser.add_argument(
        "--split_manifest",
        type=str,
        required=True,
        help="Path to dataset split manifest",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for experiment",
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        required=True,
        help="Number of training rounds",
    )

    # Record selection
    parser.add_argument(
        "--train_records_per_round",
        type=int,
        help="Train records per round",
    )

    parser.add_argument(
        "--dev_records_per_round",
        type=int,
        help="Dev records per round",
    )

    parser.add_argument(
        "--test_records_limit",
        type=int,
        help="Limit number of test records",
    )

    # Backends
    parser.add_argument(
        "--rollout_backend",
        type=str,
        default="local_fallback",
        help="Rollout backend",
    )

    parser.add_argument(
        "--trainer_backend",
        type=str,
        default="dry_run",
        choices=["dry_run", "local_grpo", "verl"],
        help="Trainer backend",
    )

    parser.add_argument(
        "--trainer_mode",
        type=str,
        default="dry_run",
        help="Trainer mode",
    )

    # Model paths
    parser.add_argument(
        "--memory_builder_model_path",
        type=str,
        help="Memory Builder model path",
    )

    parser.add_argument(
        "--question_agent_model_path",
        type=str,
        help="Question Agent model path",
    )

    parser.add_argument(
        "--answerer_model_path",
        type=str,
        help="Answerer model path",
    )

    # Initial checkpoints
    parser.add_argument(
        "--checkpoint_registry_path",
        type=str,
        help="Initial checkpoint registry path",
    )

    # Evaluation control
    parser.add_argument(
        "--final_test_eval",
        action="store_true",
        default=True,
        help="Run final test evaluation",
    )

    parser.add_argument(
        "--no_final_test_eval",
        action="store_false",
        dest="final_test_eval",
        help="Skip final test evaluation",
    )

    parser.add_argument(
        "--run_baselines",
        action="store_true",
        default=True,
        help="Run baseline evaluations",
    )

    parser.add_argument(
        "--no_run_baselines",
        action="store_false",
        dest="run_baselines",
        help="Skip baseline evaluations",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        action="append",
        dest="baseline_names",
        help="Baseline names to run (can specify multiple)",
    )

    # Execution control
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from partial experiment",
    )

    parser.add_argument(
        "--strict_no_leakage",
        action="store_true",
        default=True,
        help="Strict no-leakage validation",
    )

    parser.add_argument(
        "--no_strict_no_leakage",
        action="store_false",
        dest="strict_no_leakage",
        help="Disable strict no-leakage validation",
    )

    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Require CUDA availability",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format",
    )

    args = parser.parse_args()

    # Parse backend
    trainer_backend = Phase4TrainerBackend(args.trainer_backend)

    # Create experiment config
    baseline_names = args.baseline_names or ["no_llm_memory", "initial_checkpoint"]

    exp_config = Phase4ProductionExperimentConfig(
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        num_rounds=args.num_rounds,
        train_records_per_round=args.train_records_per_round,
        dev_records_per_round=args.dev_records_per_round,
        test_records_limit=args.test_records_limit,
        rollout_backend=args.rollout_backend,
        trainer_backend=trainer_backend,
        trainer_mode=args.trainer_mode,
        memory_builder_model_path=args.memory_builder_model_path,
        question_agent_model_path=args.question_agent_model_path,
        answerer_model_path=args.answerer_model_path,
        checkpoint_registry_path=args.checkpoint_registry_path,
        final_test_eval=args.final_test_eval,
        run_baselines=args.run_baselines,
        baseline_names=baseline_names,
        seed=args.seed,
        overwrite=args.overwrite,
        resume=args.resume,
        strict_no_leakage=args.strict_no_leakage,
    )

    output_dir = Path(args.output_dir)

    # Generate experiment ID
    experiment_id = generate_experiment_id(
        seed=args.seed,
        timestamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )

    # Create experiment manifest
    manifest = Phase4ProductionExperimentManifest(
        experiment_id=experiment_id,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "experiment_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(exp_config.model_dump(mode="json"), f, indent=2)

        # Step 1: Backend readiness check
        print("\n=== Step 1: Backend Readiness Check ===")
        readiness = check_phase4_backend_readiness(
            trainer_backend=trainer_backend,
            memory_builder_model_path=args.memory_builder_model_path,
            question_agent_model_path=args.question_agent_model_path,
            answerer_model_path=args.answerer_model_path,
            output_dir=args.output_dir,
            require_cuda=args.require_cuda,
        )

        if not readiness.ready:
            print("\nBackend not ready. Issues:")
            for issue in readiness.issues:
                print(f"  - {issue}")
            manifest.status = "failed"
            manifest.errors.extend(readiness.issues)
            manifest.metrics["backend_readiness"] = readiness.model_dump(mode="json")
            sys.exit(4)

        print("Backend ready.")
        manifest.metrics["backend_readiness"] = readiness.model_dump(mode="json")

        # Step 2: Co-training loop
        print("\n=== Step 2: Multi-Round Co-Training ===")
        cotraining_dir = output_dir / "cotraining"

        cotraining_config = Phase4CotrainingConfig(
            split_manifest_path=args.split_manifest,
            output_dir=str(cotraining_dir),
            num_rounds=args.num_rounds,
            train_records_per_round=args.train_records_per_round,
            dev_records_per_round=args.dev_records_per_round,
            rollout_backend=args.rollout_backend,
            trainer_backend=trainer_backend,
            trainer_mode=args.trainer_mode,
            memory_builder_model_path=args.memory_builder_model_path,
            question_agent_model_path=args.question_agent_model_path,
            answerer_model_path=args.answerer_model_path,
            checkpoint_registry_path=args.checkpoint_registry_path,
            seed=args.seed,
            overwrite=args.overwrite,
            resume=args.resume,
            strict_no_leakage=args.strict_no_leakage,
        )

        cotraining_manifest = run_phase4_cotraining_loop(cotraining_config)

        manifest.cotraining_dir = str(cotraining_dir)
        manifest.cotraining_manifest_path = str(
            cotraining_dir / "cotraining_manifest.json"
        )

        # Extract final checkpoints
        if cotraining_manifest.round_reports:
            last_round = cotraining_manifest.round_reports[-1]
            manifest.final_memory_builder_checkpoint_id = (
                last_round.selected_memory_builder_checkpoint_id
            )
            manifest.final_question_agent_checkpoint_id = (
                last_round.selected_question_agent_checkpoint_id
            )
            manifest.final_checkpoint_registry_path = last_round.metrics.get(
                "checkpoint_registry_path"
            )

        print(f"Co-training completed: {cotraining_manifest.status}")

        # Step 3: Final test evaluation
        if args.final_test_eval and manifest.final_memory_builder_checkpoint_id:
            print("\n=== Step 3: Final Test Evaluation ===")
            final_eval_dir = output_dir / "final_evaluation"

            from gaam_graph.phase4_experiment_schema import Phase4FinalEvaluationConfig

            eval_config = Phase4FinalEvaluationConfig(
                split_manifest_path=args.split_manifest,
                output_dir=str(final_eval_dir),
                checkpoint_registry_path=manifest.final_checkpoint_registry_path,
                memory_builder_checkpoint_id=manifest.final_memory_builder_checkpoint_id,
                question_agent_checkpoint_id=manifest.final_question_agent_checkpoint_id,
                memory_builder_model_path=args.memory_builder_model_path,
                question_agent_model_path=args.question_agent_model_path,
                answerer_model_path=args.answerer_model_path,
                test_records_limit=args.test_records_limit,
                seed=args.seed,
                overwrite=args.overwrite,
            )

            eval_manifest = run_phase4_final_evaluation(eval_config)
            write_final_evaluation_summary(eval_manifest, final_eval_dir)

            manifest.final_evaluation_dir = str(final_eval_dir)
            manifest.final_evaluation_manifest_path = str(
                final_eval_dir / "final_evaluation_manifest.json"
            )

            print(f"Final evaluation completed: {eval_manifest.status}")
        elif args.final_test_eval:
            warning = (
                "Final test evaluation requested but no final Memory Builder "
                "checkpoint was selected; skipping final evaluation."
            )
            print(f"\n[Warning] {warning}")
            manifest.warnings.append(warning)

        # Step 4: Baseline evaluations
        if args.run_baselines:
            print("\n=== Step 4: Baseline Evaluations ===")
            baselines_dir = output_dir / "final_evaluation" / "baselines"

            baseline_manifests = run_phase4_baseline_evaluations(
                split_manifest_path=args.split_manifest,
                output_dir=baselines_dir,
                baselines=baseline_names,
                final_memory_builder_checkpoint_id=manifest.final_memory_builder_checkpoint_id,
                final_question_agent_checkpoint_id=manifest.final_question_agent_checkpoint_id,
                checkpoint_registry_path=manifest.final_checkpoint_registry_path,
                memory_builder_model_path=args.memory_builder_model_path,
                question_agent_model_path=args.question_agent_model_path,
                answerer_model_path=args.answerer_model_path,
                test_records_limit=args.test_records_limit,
                seed=args.seed,
                overwrite=args.overwrite,
            )

            print(f"Baselines completed: {len(baseline_manifests)}")

        # Step 5: Leakage audit
        if args.strict_no_leakage:
            print("\n=== Step 5: Leakage Audit ===")
            audit_report = run_phase4_leakage_audit(output_dir, strict=True)

            audit_path = output_dir / "reports" / "leakage_audit.json"
            write_leakage_audit_report(audit_report, audit_path)

            if audit_report.status == "failed":
                print(f"Leakage audit failed: {len(audit_report.issues)} issues")
                manifest.status = "failed"
                manifest.errors.append(f"Leakage audit failed with {len(audit_report.issues)} issues")
            else:
                print("Leakage audit passed.")

        # Step 6: Export reports
        print("\n=== Step 6: Export Reports ===")
        export_experiment_reports(output_dir)

        manifest.reports_dir = str(output_dir / "reports")

        # Final status
        if manifest.status == "running":
            if cotraining_manifest.status == Phase4RoundStatus.SUCCEEDED:
                manifest.status = "succeeded"
            elif cotraining_manifest.status == Phase4RoundStatus.PARTIAL:
                manifest.status = "partial"
            else:
                manifest.status = "failed"

    except Exception as e:
        manifest.status = "failed"
        manifest.errors.append(str(e))
        print(f"\nExperiment failed: {e}", file=sys.stderr)

    finally:
        manifest.finished_at = datetime.now(timezone.utc).isoformat()

        # Write experiment manifest
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "experiment_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

    # Output results
    if args.json:
        print(json.dumps(manifest.model_dump(), indent=2))
    else:
        print(f"\n=== Experiment Complete ===")
        print(f"Status: {manifest.status}")
        print(f"Experiment ID: {manifest.experiment_id}")
        print(f"Output: {output_dir}")

    # Exit code
    if manifest.status == "succeeded":
        sys.exit(0)
    elif manifest.status == "partial":
        sys.exit(2)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
