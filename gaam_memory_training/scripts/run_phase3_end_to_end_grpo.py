#!/usr/bin/env python3
"""
Phase 3 Milestone 6: End-to-End GRPO CLI

One-command end-to-end execution of the complete Phase 3 distributed GRPO pipeline.

Usage:
    # Local smoke run
    python scripts/run_phase3_end_to_end_grpo.py \
      --input data/longmemeval/longmemeval_s_cleaned.json \
      --oracle_graph_dir outputs/longmemeval_s_graph \
      --record_id e47becba \
      --output_dir outputs/phase3_e2e_smoke \
      --backend local_fallback \
      --trainer_mode dry_run \
      --overwrite

    # Ray-oriented local run
    python scripts/run_phase3_end_to_end_grpo.py \
      --input data/longmemeval/longmemeval_s_cleaned.json \
      --oracle_graph_dir outputs/longmemeval_s_graph \
      --record_id e47becba \
      --output_dir outputs/phase3_e2e_ray \
      --backend ray \
      --num_rollout_workers 2 \
      --overwrite
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.phase3_e2e_orchestrator import run_phase3_end_to_end_grpo
from gaam_graph.phase3_e2e_schema import Phase3E2EBackend, Phase3E2ERunConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 E2E distributed GRPO pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input records JSON (e.g., longmemeval_s_cleaned.json)",
    )
    parser.add_argument(
        "--oracle_graph_dir",
        required=True,
        help="Directory containing oracle graphs",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for E2E run",
    )

    # Common arguments
    parser.add_argument(
        "--record_id",
        default="e47becba",
        help="Record ID to process (default: e47becba)",
    )
    parser.add_argument(
        "--round_id",
        type=int,
        default=0,
        help="Training round ID (default: 0)",
    )
    parser.add_argument(
        "--step_id",
        type=int,
        default=0,
        help="Training step ID (default: 0)",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=1,
        help="Maximum number of records to process (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        choices=["local_fallback", "ray", "vendored_verl", "code_a1_verl"],
        default="local_fallback",
        help="Execution backend (default: local_fallback)",
    )
    parser.add_argument(
        "--trainer_mode",
        choices=["dry_run", "local_model", "vendored_verl"],
        default="dry_run",
        help="Trainer execution mode (default: dry_run)",
    )
    parser.add_argument(
        "--reward_mode",
        choices=["heuristic", "learned"],
        default="heuristic",
        help="Reward computation mode (default: heuristic)",
    )

    # Worker counts
    parser.add_argument(
        "--num_rollout_workers",
        type=int,
        default=1,
        help="Number of rollout workers (default: 1)",
    )
    parser.add_argument(
        "--num_reward_workers",
        type=int,
        default=1,
        help="Number of reward workers (default: 1)",
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
    parser.add_argument(
        "--checkpoint_registry",
        help="Checkpoint registry path",
    )

    # Flags
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail on any stage failure (default: True)",
    )
    parser.add_argument(
        "--no_strict",
        action="store_false",
        dest="strict",
        help="Continue on stage failures when possible",
    )
    parser.add_argument(
        "--require_memory_reward",
        action="store_true",
        default=True,
        help="Require memory reward (default: True)",
    )
    parser.add_argument(
        "--no_require_memory_reward",
        action="store_false",
        dest="require_memory_reward",
        help="Allow missing memory reward",
    )
    parser.add_argument(
        "--require_question_reward",
        action="store_true",
        default=False,
        help="Require question reward (default: False)",
    )
    parser.add_argument(
        "--write_verl_dataproto",
        action="store_true",
        help="Write verl DataProto format",
    )
    parser.add_argument(
        "--use_vendored_verl_if_available",
        action="store_true",
        help="Use vendored verl if available",
    )
    parser.add_argument(
        "--no_update_memory_builder",
        action="store_false",
        dest="update_memory_builder",
        help="Skip Memory Builder updates",
    )
    parser.add_argument(
        "--no_update_question_agent",
        action="store_false",
        dest="update_question_agent",
        help="Skip Question Agent updates",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.overwrite and args.resume:
        logger.error("Cannot use both --overwrite and --resume")
        return 1

    # Build run ID
    run_id = f"phase3_e2e_{args.record_id}_round{args.round_id:03d}_step{args.step_id:03d}"

    # Build config
    try:
        config = Phase3E2ERunConfig(
            run_id=run_id,
            input_records_path=args.input,
            oracle_graph_dir=args.oracle_graph_dir,
            output_dir=args.output_dir,
            record_id=args.record_id,
            round_id=args.round_id,
            step_id=args.step_id,
            max_records=args.max_records,
            seed=args.seed,
            backend=Phase3E2EBackend(args.backend),
            trainer_mode=args.trainer_mode,
            reward_mode=args.reward_mode,
            num_rollout_workers=args.num_rollout_workers,
            num_reward_workers=args.num_reward_workers,
            strict=args.strict,
            overwrite=args.overwrite,
            resume=args.resume,
            require_memory_reward=args.require_memory_reward,
            require_question_reward=args.require_question_reward,
            update_memory_builder=args.update_memory_builder if hasattr(args, 'update_memory_builder') else True,
            update_question_agent=args.update_question_agent if hasattr(args, 'update_question_agent') else True,
            write_verl_dataproto=args.write_verl_dataproto,
            use_vendored_verl_if_available=args.use_vendored_verl_if_available,
            memory_builder_model_path=args.memory_builder_model_path,
            question_agent_model_path=args.question_agent_model_path,
            answerer_model_path=args.answerer_model_path,
            checkpoint_registry_path=args.checkpoint_registry,
        )
    except Exception as e:
        logger.error(f"Invalid configuration: {e}")
        return 1

    # Run E2E pipeline
    logger.info(f"Starting Phase 3 E2E GRPO run: {run_id}")
    logger.info(f"  Record ID: {args.record_id}")
    logger.info(f"  Backend: {args.backend}")
    logger.info(f"  Trainer mode: {args.trainer_mode}")
    logger.info(f"  Output dir: {args.output_dir}")

    try:
        report = run_phase3_end_to_end_grpo(config)

        # Print summary
        print("\n" + "=" * 80)
        print("Phase 3 E2E GRPO Run Complete")
        print("=" * 80)
        print(f"Run ID: {report.run_id}")
        print(f"Status: {report.status.value}")
        print(f"Record ID: {report.record_id}")
        print(f"Backend: {report.backend.value}")
        print(f"No-leakage: {'passed' if report.no_leakage_passed else 'FAILED'}")
        print(f"Remote GPU ready: {'yes' if report.remote_gpu_ready else 'no'}")
        if report.remote_gpu_readiness_gaps:
            print("Remote GPU readiness gaps:")
            for gap in report.remote_gpu_readiness_gaps:
                print(f"  - {gap}")
        print(f"Code-A1/verl ready: {'yes' if report.code_a1_verl_ready else 'no'}")
        if report.code_a1_verl_readiness_gaps:
            print("Code-A1/verl readiness gaps:")
            for gap in report.code_a1_verl_readiness_gaps:
                print(f"  - {gap}")

        if report.round_reports:
            round_report = report.round_reports[0]
            print(f"\nRound {round_report.round_id}, Step {round_report.step_id}:")
            if round_report.memory_reward_mean is not None:
                print(f"  Memory reward: {round_report.memory_reward_mean:.3f}")
            if round_report.question_reward_mean is not None:
                print(f"  Question reward: {round_report.question_reward_mean:.3f}")
            print(f"  Weakness updates: {round_report.num_weakness_updates}")
            print(f"  Replay items: {round_report.num_replay_items}")

        print(f"\nOutput directory: {report.output_dir}")
        print(f"Final report: {Path(report.output_dir) / 'final_report.json'}")
        print(f"Summary: {report.final_summary_path}")

        if report.warnings:
            print(f"\nWarnings ({len(report.warnings)}):")
            for warning in report.warnings[:5]:
                print(f"  - {warning}")
            if len(report.warnings) > 5:
                print(f"  ... and {len(report.warnings) - 5} more")

        if report.errors:
            print(f"\nErrors ({len(report.errors)}):")
            for error in report.errors[:5]:
                print(f"  - {error}")
            if len(report.errors) > 5:
                print(f"  ... and {len(report.errors) - 5} more")

        print("=" * 80)

        # Return appropriate exit code
        if report.status.value == "failed":
            return 1
        elif report.status.value == "partial":
            return 2 if args.strict else 0
        else:
            return 0

    except Exception as e:
        logger.error(f"E2E pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
