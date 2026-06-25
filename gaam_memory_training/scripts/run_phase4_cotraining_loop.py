#!/usr/bin/env python3
"""
Phase 4 Milestone 4: Multi-Round Adversarial Co-Training Loop CLI

Run multi-round adversarial co-training experiment.

Usage:
    python scripts/run_phase4_cotraining_loop.py \
      --split_manifest outputs/splits/longmemeval_s.phase4.seed0.json \
      --output_dir outputs/phase4_cotraining/exp001 \
      --num_rounds 3 \
      --train_records_per_round 8 \
      --dev_records_per_round 4 \
      --overwrite

Exit codes:
    0 = succeeded
    1 = failed
    2 = partial
    3 = validation error
"""

import argparse
import sys
from pathlib import Path

from gaam_graph.phase4_cotraining_loop import run_phase4_cotraining_loop
from gaam_graph.phase4_cotraining_schema import (
    Phase4CotrainingConfig,
    Phase4RoundStatus,
)
from gaam_graph.phase4_trainer_schema import Phase4TrainerBackend


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 4 multi-round adversarial co-training loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic dry-run
  python scripts/run_phase4_cotraining_loop.py \
    --split_manifest outputs/splits/longmemeval_s.phase4.seed0.json \
    --output_dir outputs/phase4_cotraining/exp001 \
    --num_rounds 3 \
    --train_records_per_round 8 \
    --dev_records_per_round 4 \
    --overwrite

  # Debug single record
  python scripts/run_phase4_cotraining_loop.py \
    --split_manifest outputs/splits/e47becba.debug_split.json \
    --output_dir outputs/phase4_cotraining/e47becba_debug \
    --num_rounds 2 \
    --train_record_id e47becba \
    --dev_record_id e47becba \
    --allow_empty_batch \
    --overwrite

  # With early stopping
  python scripts/run_phase4_cotraining_loop.py \
    --split_manifest outputs/splits/longmemeval_s.phase4.seed0.json \
    --output_dir outputs/phase4_cotraining/exp002 \
    --num_rounds 10 \
    --train_records_per_round 16 \
    --dev_records_per_round 8 \
    --early_stop_patience 3 \
    --early_stop_metric dev_memory_reward_mean \
    --overwrite
        """,
    )

    # Required arguments
    parser.add_argument(
        "--split_manifest",
        required=True,
        help="Path to dataset split manifest JSON",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for co-training experiment",
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
        help="Max train records per round",
    )
    parser.add_argument(
        "--dev_records_per_round",
        type=int,
        help="Max dev records per round",
    )
    parser.add_argument(
        "--train_record_id",
        action="append",
        dest="train_record_ids",
        help="Explicit train record ID (can specify multiple)",
    )
    parser.add_argument(
        "--dev_record_id",
        action="append",
        dest="dev_record_ids",
        help="Explicit dev record ID (can specify multiple)",
    )
    parser.add_argument(
        "--record_schedule_policy",
        choices=["deterministic_window", "fixed_subset"],
        default="deterministic_window",
        help="Record scheduling policy",
    )

    # Backends
    parser.add_argument(
        "--rollout_backend",
        default="local_fallback",
        help="Rollout backend (default: local_fallback)",
    )
    parser.add_argument(
        "--trainer_backend",
        choices=["dry_run", "local_grpo", "verl"],
        default="dry_run",
        help="Trainer backend (default: dry_run)",
    )
    parser.add_argument(
        "--trainer_mode",
        default="dry_run",
        help="Trainer mode (default: dry_run)",
    )

    # Model paths
    parser.add_argument(
        "--memory_builder_model_path",
        help="Memory Builder model path",
    )
    parser.add_argument(
        "--question_agent_model_path",
        help="Question Agent model path",
    )
    parser.add_argument(
        "--answerer_model_path",
        help="Answerer model path",
    )

    # Initial checkpoints
    parser.add_argument(
        "--initial_memory_builder_checkpoint_id",
        help="Initial Memory Builder checkpoint ID",
    )
    parser.add_argument(
        "--initial_question_agent_checkpoint_id",
        help="Initial Question Agent checkpoint ID",
    )
    parser.add_argument(
        "--checkpoint_registry_path",
        help="Checkpoint registry path",
    )

    # Actor update control
    parser.add_argument(
        "--no_update_memory_builder",
        action="store_true",
        help="Disable Memory Builder updates",
    )
    parser.add_argument(
        "--no_update_question_agent",
        action="store_true",
        help="Disable Question Agent updates",
    )

    # Dev evaluation
    parser.add_argument(
        "--max_dev_regression",
        type=float,
        default=0.02,
        help="Max dev reward regression threshold (default: 0.02)",
    )
    parser.add_argument(
        "--min_dev_no_leakage_pass_rate",
        type=float,
        default=0.8,
        help="Min dev no-leakage pass rate (default: 0.8)",
    )

    # Early stopping
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        help="Early stop patience (number of rounds without improvement)",
    )
    parser.add_argument(
        "--early_stop_metric",
        default="dev_memory_reward_mean",
        help="Early stop metric (default: dev_memory_reward_mean)",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Early stop min delta (default: 0.0)",
    )

    # Failure handling
    parser.add_argument(
        "--continue_on_round_failure",
        action="store_true",
        help="Continue to next round even if current round fails",
    )
    parser.add_argument(
        "--continue_on_record_failure",
        action="store_true",
        default=True,
        help="Continue rollout even if individual records fail (default: True)",
    )
    parser.add_argument(
        "--no_continue_on_record_failure",
        action="store_false",
        dest="continue_on_record_failure",
        help="Stop rollout if any record fails",
    )
    parser.add_argument(
        "--allow_empty_batch",
        action="store_true",
        help="Allow empty actor batches (for debugging)",
    )
    parser.add_argument(
        "--strict_no_leakage",
        action="store_true",
        default=True,
        help="Strict no-leakage validation (default: True)",
    )
    parser.add_argument(
        "--no_strict_no_leakage",
        action="store_false",
        dest="strict_no_leakage",
        help="Disable strict no-leakage validation",
    )

    # Execution control
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing experiment",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing experiment",
    )

    # Output format
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output manifest as JSON",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.train_record_ids and args.train_records_per_round:
        print(
            "Warning: Both --train_record_id and --train_records_per_round specified. "
            "Explicit record IDs take precedence.",
            file=sys.stderr,
        )

    if args.dev_record_ids and args.dev_records_per_round:
        print(
            "Warning: Both --dev_record_id and --dev_records_per_round specified. "
            "Explicit record IDs take precedence.",
            file=sys.stderr,
        )

    # Build config
    config = Phase4CotrainingConfig(
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        num_rounds=args.num_rounds,
        train_records_per_round=args.train_records_per_round,
        dev_records_per_round=args.dev_records_per_round,
        train_record_ids=args.train_record_ids,
        dev_record_ids=args.dev_record_ids,
        record_schedule_policy=args.record_schedule_policy,
        rollout_backend=args.rollout_backend,
        trainer_backend=Phase4TrainerBackend(args.trainer_backend),
        trainer_mode=args.trainer_mode,
        memory_builder_model_path=args.memory_builder_model_path,
        question_agent_model_path=args.question_agent_model_path,
        answerer_model_path=args.answerer_model_path,
        initial_memory_builder_checkpoint_id=args.initial_memory_builder_checkpoint_id,
        initial_question_agent_checkpoint_id=args.initial_question_agent_checkpoint_id,
        checkpoint_registry_path=args.checkpoint_registry_path,
        update_memory_builder=not args.no_update_memory_builder,
        update_question_agent=not args.no_update_question_agent,
        max_dev_regression=args.max_dev_regression,
        min_dev_no_leakage_pass_rate=args.min_dev_no_leakage_pass_rate,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
        early_stop_min_delta=args.early_stop_min_delta,
        continue_on_round_failure=args.continue_on_round_failure,
        continue_on_record_failure=args.continue_on_record_failure,
        allow_empty_batch=args.allow_empty_batch,
        strict_no_leakage=args.strict_no_leakage,
        seed=args.seed,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    # Run co-training loop
    try:
        manifest = run_phase4_cotraining_loop(config)

        if args.json:
            import json

            print(json.dumps(manifest.model_dump(), indent=2))
        else:
            print(f"\nCo-training experiment complete: {manifest.run_id}")
            print(f"Status: {manifest.status.value}")
            print(
                f"Rounds completed: {manifest.num_rounds_completed}/{manifest.num_rounds_requested}"
            )
            print(f"Output: {manifest.output_dir}")

            if manifest.final_memory_builder_checkpoint_id:
                print(
                    f"Final Memory Builder checkpoint: {manifest.final_memory_builder_checkpoint_id}"
                )
            if manifest.final_question_agent_checkpoint_id:
                print(
                    f"Final Question Agent checkpoint: {manifest.final_question_agent_checkpoint_id}"
                )

        # Exit code based on status
        if manifest.status == Phase4RoundStatus.SUCCEEDED:
            sys.exit(0)
        elif manifest.status == Phase4RoundStatus.PARTIAL:
            sys.exit(2)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
