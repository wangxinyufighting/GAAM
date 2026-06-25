#!/usr/bin/env python3
"""
Run Phase 4 Split Smoke Test

Runs Phase 3 E2E pipeline across selected records from a dataset split.

Usage:
    # Local smoke test
    python scripts/run_phase4_split_smoke.py \\
      --split_manifest outputs/splits/longmemeval_s.phase4_split.seed0.json \\
      --split train \\
      --output_dir outputs/phase4_split_smoke/train \\
      --backend local_fallback \\
      --trainer_mode dry_run \\
      --max_records 2 \\
      --overwrite

    # Debug single case
    python scripts/run_phase4_split_smoke.py \\
      --split_manifest outputs/splits/e47becba.debug_split.json \\
      --split train \\
      --output_dir outputs/phase4_split_smoke/e47becba \\
      --backend local_fallback \\
      --trainer_mode dry_run \\
      --max_records 1 \\
      --overwrite

    # Remote GPU with Code-A1/verl
    python scripts/run_phase4_split_smoke.py \\
      --split_manifest outputs/splits/e47becba.debug_split.json \\
      --split train \\
      --output_dir outputs/phase4_split_smoke/e47becba_code_a1_verl \\
      --backend code_a1_verl \\
      --trainer_mode vendored_verl \\
      --memory_builder_model_path /mnt/local2/wxy/models/Qwen3-0.6B \\
      --question_agent_model_path /mnt/local2/wxy/models/Qwen3-0.6B \\
      --answerer_model_path /mnt/local2/wxy/models/Qwen3-0.6B \\
      --max_records 1 \\
      --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.dataset_split_schema import DatasetSplitManifest, DatasetSplitName
from gaam_graph.phase3_e2e_schema import Phase3E2EBackend
from gaam_graph.training_protocol import run_split_smoke, write_split_summary_markdown


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 4 split smoke test across selected records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--split_manifest",
        required=True,
        type=Path,
        help="Path to split manifest JSON",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "dev", "test"],
        help="Which split to run",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Output directory for split run",
    )

    # Backend and trainer
    parser.add_argument(
        "--backend",
        default="local_fallback",
        choices=["local_fallback", "ray", "vendored_verl", "code_a1_verl"],
        help="Phase 3 backend (default: local_fallback)",
    )
    parser.add_argument(
        "--trainer_mode",
        default="dry_run",
        choices=["dry_run", "local_model", "vendored_verl"],
        help="Trainer mode (default: dry_run)",
    )

    # Optional arguments
    parser.add_argument(
        "--max_records",
        type=int,
        help="Maximum number of records to process",
    )
    parser.add_argument(
        "--round_id",
        type=int,
        default=0,
        help="Round ID (default: 0)",
    )
    parser.add_argument(
        "--step_id",
        type=int,
        default=0,
        help="Step ID (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    # Model paths
    parser.add_argument(
        "--memory_builder_model_path",
        help="Model path for Memory Builder",
    )
    parser.add_argument(
        "--question_agent_model_path",
        help="Model path for Question Agent",
    )
    parser.add_argument(
        "--answerer_model_path",
        help="Model path for Answerer",
    )

    # Actor update policy overrides
    parser.add_argument(
        "--update_memory_builder",
        action="store_true",
        default=None,
        help="Override: update Memory Builder (default: True for train, False for dev/test)",
    )
    parser.add_argument(
        "--no_update_memory_builder",
        dest="update_memory_builder",
        action="store_false",
        help="Override: do not update Memory Builder",
    )
    parser.add_argument(
        "--update_question_agent",
        action="store_true",
        default=None,
        help="Override: update Question Agent (default: True for train, False for dev/test)",
    )
    parser.add_argument(
        "--no_update_question_agent",
        dest="update_question_agent",
        action="store_false",
        help="Override: do not update Question Agent",
    )

    # Execution policy
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )
    parser.add_argument(
        "--continue_on_record_failure",
        action="store_true",
        help="Continue execution when a record fails",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode for validation",
    )

    args = parser.parse_args()

    # Load split manifest
    if not args.split_manifest.exists():
        print(f"Error: Split manifest not found: {args.split_manifest}", file=sys.stderr)
        return 1

    try:
        with open(args.split_manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest = DatasetSplitManifest.model_validate(manifest_data)
    except Exception as e:
        print(f"Error: Failed to load manifest: {e}", file=sys.stderr)
        return 1

    # Parse split
    split = DatasetSplitName(args.split)

    # Parse backend
    backend = Phase3E2EBackend(args.backend)

    print("=" * 60)
    print(f"Phase 4 Split Smoke: {split.value}")
    print("=" * 60)
    print(f"Split manifest: {args.split_manifest}")
    print(f"Dataset: {manifest.dataset_name}")
    print(f"Split: {split.value}")
    print(f"Backend: {backend.value}")
    print(f"Trainer mode: {args.trainer_mode}")
    print(f"Max records: {args.max_records or 'all'}")
    print(f"Output dir: {args.output_dir}")
    print("")

    # Run split smoke
    try:
        split_report = run_split_smoke(
            split_manifest=manifest,
            split=split,
            output_dir=args.output_dir,
            backend=backend,
            trainer_mode=args.trainer_mode,
            split_manifest_path=args.split_manifest,
            max_records=args.max_records,
            round_id=args.round_id,
            step_id=args.step_id,
            seed=args.seed,
            memory_builder_model_path=args.memory_builder_model_path,
            question_agent_model_path=args.question_agent_model_path,
            answerer_model_path=args.answerer_model_path,
            update_memory_builder=args.update_memory_builder,
            update_question_agent=args.update_question_agent,
            overwrite=args.overwrite,
            continue_on_record_failure=args.continue_on_record_failure,
            strict=args.strict,
        )
    except Exception as e:
        print(f"\nError: Split smoke failed: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Write summary markdown
    summary_path = args.output_dir / "summary.md"
    write_split_summary_markdown(split_report, summary_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Split Run Summary")
    print("=" * 60)
    print(f"Run ID: {split_report.run_id}")
    print(f"Status: {split_report.status}")
    print(f"Selected records: {len(split_report.selected_record_ids)}")
    print(f"Executed: {split_report.metrics.get('num_executed', 0)}")
    print(f"Succeeded: {split_report.metrics.get('num_succeeded', 0)}")
    print(f"Failed: {split_report.metrics.get('num_failed', 0)}")
    print(f"Partial: {split_report.metrics.get('num_partial', 0)}")
    print("")

    if split_report.metrics.get("mean_memory_reward") is not None:
        print(f"Mean memory reward: {split_report.metrics['mean_memory_reward']:.3f}")
    if split_report.metrics.get("mean_question_reward") is not None:
        print(f"Mean question reward: {split_report.metrics['mean_question_reward']:.3f}")

    print(f"Total replay items: {split_report.metrics.get('total_replay_items', 0)}")
    print(f"Total weakness updates: {split_report.metrics.get('total_weakness_updates', 0)}")
    print(f"No-leakage passed: {split_report.metrics.get('no_leakage_pass_count', 0)}/{len(split_report.records)}")
    print("")

    print("Per-record results:")
    for record in split_report.records:
        status_icon = "✓" if record.status == "succeeded" else "✗"
        print(f"  {status_icon} {record.record_id}: {record.status}")
        if record.errors:
            for error in record.errors[:3]:  # Show first 3 errors
                print(f"      Error: {error}")

    print("")
    print(f"Summary written to: {summary_path}")
    print(f"Full report: {args.output_dir / 'split_run_report.json'}")

    # Determine exit code
    if split_report.status == "failed":
        return 1
    elif split_report.status == "partial" and args.strict:
        return 2
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
