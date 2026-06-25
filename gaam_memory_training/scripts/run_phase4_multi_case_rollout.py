#!/usr/bin/env python3
"""
Phase 4 Milestone 2: Multi-Case Rollout CLI

Runs Phase 4 multi-case rollout across selected records from a dataset split.

Example usage:
    python scripts/run_phase4_multi_case_rollout.py \
      --split_manifest outputs/splits/longmemeval_s.phase4.seed0.json \
      --split train \
      --output_dir outputs/phase4_multi_case_rollout/train_round000_step000 \
      --backend local_fallback \
      --trainer_mode dry_run \
      --max_records 8 \
      --round_id 0 \
      --step_id 0 \
      --overwrite
"""

import argparse
import sys
from pathlib import Path

from gaam_graph.dataset_split_schema import DatasetSplitName
from gaam_graph.phase4_multi_case_rollout import run_multi_case_rollout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 multi-case rollout across dataset split"
    )

    # Required arguments
    parser.add_argument(
        "--split_manifest",
        type=Path,
        required=True,
        help="Path to dataset split manifest JSON",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "dev", "test"],
        help="Target split to run",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for multi-case run",
    )

    # Execution parameters
    parser.add_argument(
        "--backend",
        type=str,
        default="local_fallback",
        choices=["local_fallback", "ray", "vendored_verl", "code_a1_verl"],
        help="Backend for Phase 3 execution",
    )
    parser.add_argument(
        "--trainer_mode",
        type=str,
        default="dry_run",
        choices=["dry_run", "local_model", "vendored_verl"],
        help="Trainer mode",
    )

    # Record selection
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum number of records to select",
    )
    parser.add_argument(
        "--record_id",
        type=str,
        action="append",
        dest="record_ids",
        help="Explicit record ID to run (repeatable)",
    )

    # Training loop parameters
    parser.add_argument(
        "--round_id",
        type=int,
        default=0,
        help="Training round ID",
    )
    parser.add_argument(
        "--step_id",
        type=int,
        default=0,
        help="Training step ID",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    # Model paths (optional)
    parser.add_argument(
        "--memory_builder_model_path",
        type=str,
        default=None,
        help="Memory Builder model path",
    )
    parser.add_argument(
        "--question_agent_model_path",
        type=str,
        default=None,
        help="Question Agent model path",
    )
    parser.add_argument(
        "--answerer_model_path",
        type=str,
        default=None,
        help="Answerer model path",
    )

    # Execution control
    parser.add_argument(
        "--num_rollout_workers",
        type=int,
        default=1,
        help="Number of rollout workers (future use)",
    )
    parser.add_argument(
        "--continue_on_record_failure",
        action="store_true",
        help="Continue after record failure",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-succeeded records",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings",
    )
    parser.add_argument(
        "--write_dataproto",
        action="store_true",
        help="Export GRPO batches",
    )

    args = parser.parse_args()

    # Validate split manifest exists
    if not args.split_manifest.exists():
        print(f"Error: Split manifest not found: {args.split_manifest}", file=sys.stderr)
        return 1

    # Build kwargs for Phase 3 config
    phase3_kwargs = {}
    if args.memory_builder_model_path:
        phase3_kwargs["memory_builder_model_path"] = args.memory_builder_model_path
    if args.question_agent_model_path:
        phase3_kwargs["question_agent_model_path"] = args.question_agent_model_path
    if args.answerer_model_path:
        phase3_kwargs["answerer_model_path"] = args.answerer_model_path
    if args.seed is not None:
        phase3_kwargs["seed"] = args.seed

    try:
        # Run multi-case rollout
        manifest = run_multi_case_rollout(
            split_manifest_path=args.split_manifest,
            split=DatasetSplitName(args.split),
            output_dir=args.output_dir,
            backend=args.backend,
            trainer_mode=args.trainer_mode,
            max_records=args.max_records,
            record_ids=args.record_ids,
            round_id=args.round_id,
            step_id=args.step_id,
            resume=args.resume,
            overwrite=args.overwrite,
            continue_on_record_failure=args.continue_on_record_failure,
            write_dataproto=args.write_dataproto,
            **phase3_kwargs,
        )

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"Multi-Case Rollout Complete")
        print(f"{'=' * 60}")
        print(f"Run ID: {manifest.run_id}")
        print(f"Status: {manifest.status}")
        print(f"Split: {manifest.split}")
        print(f"Records selected: {len(manifest.selected_record_ids)}")
        print(f"Records succeeded: {sum(1 for r in manifest.records if r.status == 'succeeded')}")
        print(f"Records failed: {sum(1 for r in manifest.records if r.status == 'failed')}")
        print(f"\nOutput: {args.output_dir}")
        print(f"Manifest: {args.output_dir / 'multi_case_rollout_manifest.json'}")
        print(f"Summary: {args.output_dir / 'summary.md'}")

        # Check status for exit code
        if manifest.status == "failed":
            return 1
        elif manifest.status == "partial" and args.strict:
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
