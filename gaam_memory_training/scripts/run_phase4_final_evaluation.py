#!/usr/bin/env python3
"""
Run Phase 4 final held-out evaluation.

Usage:
    python scripts/run_phase4_final_evaluation.py \
      --split_manifest outputs/splits/split.json \
      --output_dir outputs/final_eval \
      --memory_builder_checkpoint_id mb_ckpt_001 \
      --question_agent_checkpoint_id qa_ckpt_001

Exit codes:
    0 = succeeded
    1 = failed
    2 = partial
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.phase4_experiment_schema import Phase4FinalEvaluationConfig
from gaam_graph.phase4_final_evaluation import (
    run_phase4_final_evaluation,
    write_final_evaluation_summary,
)


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 final held-out evaluation")

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
        help="Output directory for evaluation results",
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoint_registry_path",
        type=str,
        help="Checkpoint registry path",
    )

    parser.add_argument(
        "--memory_builder_checkpoint_id",
        type=str,
        help="Memory Builder checkpoint ID",
    )

    parser.add_argument(
        "--question_agent_checkpoint_id",
        type=str,
        help="Question Agent checkpoint ID",
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

    # Evaluation control
    parser.add_argument(
        "--test_records_limit",
        type=int,
        help="Limit number of test records",
    )

    parser.add_argument(
        "--baseline_name",
        type=str,
        help="Baseline name (for comparison runs)",
    )

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
        "--json",
        action="store_true",
        help="Output JSON format",
    )

    args = parser.parse_args()

    # Create config
    config = Phase4FinalEvaluationConfig(
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        checkpoint_registry_path=args.checkpoint_registry_path,
        memory_builder_checkpoint_id=args.memory_builder_checkpoint_id,
        question_agent_checkpoint_id=args.question_agent_checkpoint_id,
        memory_builder_model_path=args.memory_builder_model_path,
        question_agent_model_path=args.question_agent_model_path,
        answerer_model_path=args.answerer_model_path,
        test_records_limit=args.test_records_limit,
        baseline_name=args.baseline_name,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    try:
        # Run evaluation
        manifest = run_phase4_final_evaluation(config)

        # Write summary
        write_final_evaluation_summary(manifest, Path(args.output_dir))

        # Output results
        if args.json:
            print(json.dumps(manifest.model_dump(), indent=2))
        else:
            print(f"\n=== Final Evaluation Complete ===")
            print(f"Status: {manifest.status}")
            print(f"Test records: {len(manifest.selected_record_ids)}")
            print(f"Memory reward: {manifest.metrics.get('test_memory_reward_mean')}")
            print(f"Question reward: {manifest.metrics.get('test_question_reward_mean')}")
            print(f"No-leakage: {manifest.metrics.get('test_no_leakage_pass_rate')}")
            print(f"\nResults written to: {args.output_dir}")

        # Exit code
        if manifest.status == "succeeded":
            sys.exit(0)
        elif manifest.status == "partial":
            sys.exit(2)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
