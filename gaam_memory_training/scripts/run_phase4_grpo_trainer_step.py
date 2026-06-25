#!/usr/bin/env python3
"""
Phase 4 Milestone 3: GRPO Trainer Step CLI

Runs one GRPO trainer step over aggregated multi-case batches from Phase 4 Milestone 2.

Usage:
    python scripts/run_phase4_grpo_trainer_step.py \
      --multi_case_run_dir outputs/phase4_multi_case_rollout/train_round000_step000 \
      --output_dir outputs/phase4_trainer_steps/train_round000_step000 \
      --backend dry_run \
      --round_id 0 \
      --step_id 0 \
      --actors memory_builder question_agent \
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

from gaam_graph.phase4_grpo_step import run_phase4_grpo_trainer_step
from gaam_graph.phase4_trainer_schema import (
    Phase4TrainerBackend,
    Phase4TrainerStepConfig,
)
from gaam_graph.grpo_schema import ActorRole


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Phase 4 GRPO trainer step over aggregated multi-case batches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--multi_case_run_dir",
        type=str,
        required=True,
        help="Path to completed Phase 4 M2 multi-case rollout directory",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for trainer step results",
    )

    # Backend configuration
    parser.add_argument(
        "--backend",
        type=str,
        choices=["dry_run", "local_grpo", "verl"],
        default="dry_run",
        help="Trainer backend (default: dry_run)",
    )

    # Training configuration
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
        "--actors",
        type=str,
        nargs="+",
        choices=["memory_builder", "question_agent"],
        default=["memory_builder", "question_agent"],
        help="Actors to update (default: both)",
    )

    # Model/checkpoint paths
    parser.add_argument(
        "--memory_builder_model_path",
        type=str,
        help="Memory Builder model path (for real updates)",
    )

    parser.add_argument(
        "--question_agent_model_path",
        type=str,
        help="Question Agent model path (for real updates)",
    )

    parser.add_argument(
        "--memory_builder_checkpoint_id",
        type=str,
        help="Memory Builder checkpoint ID to load",
    )

    parser.add_argument(
        "--question_agent_checkpoint_id",
        type=str,
        help="Question Agent checkpoint ID to load",
    )

    parser.add_argument(
        "--checkpoint_registry_path",
        type=str,
        help="Checkpoint registry path",
    )

    # Hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate (default: 1e-6)",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm (default: 1.0)",
    )

    parser.add_argument(
        "--clip_ratio",
        type=float,
        default=0.2,
        help="PPO clip ratio (default: 0.2)",
    )

    parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.01,
        help="KL divergence coefficient (default: 0.01)",
    )

    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.0,
        help="Entropy coefficient (default: 0.0)",
    )

    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=1,
        help="Mini-batch size (default: 1)",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )

    # Validation flags
    parser.add_argument(
        "--allow_empty_batch",
        action="store_true",
        help="Allow empty actor batches (default: False)",
    )

    parser.add_argument(
        "--strict_no_leakage",
        action="store_true",
        default=True,
        help="Strict no-leakage validation (default: True)",
    )

    # Execution flags
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output directory",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    # Output format
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format (for automation)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entrypoint."""
    args = parse_args()

    # Parse backend
    backend = Phase4TrainerBackend(args.backend)

    # Parse actors
    actors = [ActorRole(actor) for actor in args.actors]

    # Build config
    config = Phase4TrainerStepConfig(
        source_multi_case_run_dir=args.multi_case_run_dir,
        output_dir=args.output_dir,
        backend=backend,
        round_id=args.round_id,
        step_id=args.step_id,
        actors=actors,
        memory_builder_model_path=args.memory_builder_model_path,
        question_agent_model_path=args.question_agent_model_path,
        memory_builder_checkpoint_id=args.memory_builder_checkpoint_id,
        question_agent_checkpoint_id=args.question_agent_checkpoint_id,
        checkpoint_registry_path=args.checkpoint_registry_path,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        clip_ratio=args.clip_ratio,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        require_non_empty_batch=not args.allow_empty_batch,
        allow_empty_batch=args.allow_empty_batch,
        strict_no_leakage=args.strict_no_leakage,
        overwrite=args.overwrite,
        resume=args.resume,
        seed=args.seed,
    )

    try:
        # Run trainer step
        manifest = run_phase4_grpo_trainer_step(config)

        # Output format
        if args.json:
            import json
            print(json.dumps(manifest.model_dump(), indent=2))
        else:
            print(f"✅ Trainer step completed: {manifest.status}")
            print(f"   Output: {manifest.output_dir}")
            print(f"   Run ID: {manifest.run_id}")
            print(f"   Actors: {len(manifest.actor_reports)}")

            for report in manifest.actor_reports:
                status_emoji = "✅" if report.status == "succeeded" else "❌"
                print(
                    f"   {status_emoji} {report.actor_role.value}: {report.status}"
                )

                if report.written_checkpoint_id:
                    print(f"      Checkpoint: {report.written_checkpoint_id}")

            if manifest.metrics:
                print(f"   Metrics: {len(manifest.metrics)} computed")

        # Exit code based on status
        if manifest.status == "succeeded":
            return 0
        elif manifest.status == "partial":
            return 2
        else:
            return 1

    except FileExistsError as e:
        print(f"❌ Output directory exists: {e}", file=sys.stderr)
        print("   Use --overwrite or --resume", file=sys.stderr)
        return 3

    except ValueError as e:
        print(f"❌ Validation error: {e}", file=sys.stderr)
        return 3

    except Exception as e:
        print(f"❌ Trainer step failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
