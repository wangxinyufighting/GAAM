#!/usr/bin/env python3
"""
Create Dataset Split for Phase 4 Multi-Case Training

Creates deterministic train/dev/test splits from LongMemEval records.
Supports both random seeded assignment and explicit ID lists.

Usage:
    # Random split
    python scripts/create_dataset_split.py \\
      --input data/longmemeval/longmemeval_s_cleaned.json \\
      --oracle_graph_dir outputs/longmemeval_s_graph \\
      --output outputs/splits/longmemeval_s.phase4_split.seed0.json \\
      --train_ratio 0.8 \\
      --dev_ratio 0.1 \\
      --test_ratio 0.1 \\
      --seed 0 \\
      --overwrite

    # Debug single-case
    python scripts/create_dataset_split.py \\
      --input data/longmemeval/longmemeval_s_cleaned.json \\
      --oracle_graph_dir outputs/longmemeval_s_graph \\
      --output outputs/splits/e47becba.debug_split.json \\
      --record_id e47becba \\
      --debug_single_case \\
      --overwrite

    # Explicit ID assignment
    python scripts/create_dataset_split.py \\
      --input data/longmemeval/longmemeval_s_cleaned.json \\
      --oracle_graph_dir outputs/longmemeval_s_graph \\
      --output outputs/splits/longmemeval_s.manual.json \\
      --train_ids train_ids.txt \\
      --dev_ids dev_ids.txt \\
      --test_ids test_ids.txt \\
      --overwrite
"""

import argparse
import sys
from pathlib import Path

from gaam_graph.dataset_split_schema import DatasetSplitIssueSeverity
from gaam_graph.dataset_splitter import (
    create_explicit_split,
    create_random_split,
    read_record_id_file,
    write_split_manifest,
)


def main():
    parser = argparse.ArgumentParser(
        description="Create train/dev/test split for Phase 4 multi-case training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to LongMemEval input JSON",
    )
    parser.add_argument(
        "--oracle_graph_dir",
        required=True,
        type=Path,
        help="Directory containing oracle graph files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for split manifest JSON",
    )

    # Split mode: random or explicit
    split_mode = parser.add_mutually_exclusive_group()
    split_mode.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )
    split_mode.add_argument(
        "--train_ids",
        type=Path,
        help="Path to file with train record IDs (one per line)",
    )

    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=0.1,
        help="Dev split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--dev_ids",
        type=Path,
        help="Path to file with dev record IDs (one per line)",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test_ids",
        type=Path,
        help="Path to file with test record IDs (one per line)",
    )

    # Optional arguments
    parser.add_argument(
        "--dataset_name",
        default="longmemeval",
        help="Dataset name (default: longmemeval)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic split (default: 0)",
    )
    parser.add_argument(
        "--record_id",
        action="append",
        help="Filter to specific record ID (can be repeated)",
    )
    parser.add_argument(
        "--require_oracle_graph",
        action="store_true",
        default=True,
        help="Fail if oracle graph missing (default: True)",
    )
    parser.add_argument(
        "--no_require_oracle_graph",
        dest="require_oracle_graph",
        action="store_false",
        help="Allow missing oracle graphs",
    )
    parser.add_argument(
        "--debug_single_case",
        action="store_true",
        help="Allow single-record train-only split for debugging",
    )
    parser.add_argument(
        "--allow_empty_split",
        action="store_true",
        help="Allow empty dev/test splits",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )

    args = parser.parse_args()

    # Validate output path
    if args.output.exists() and not args.overwrite:
        print(f"Error: Output file exists: {args.output}", file=sys.stderr)
        print("Use --overwrite to overwrite", file=sys.stderr)
        return 1

    # Determine split mode
    explicit_mode = bool(args.train_ids or args.dev_ids or args.test_ids)

    if explicit_mode:
        # Explicit ID assignment mode
        if not all([args.train_ids, args.dev_ids, args.test_ids]):
            print(
                "Error: Explicit mode requires all of --train_ids, --dev_ids, --test_ids",
                file=sys.stderr,
            )
            return 1

        print("Creating explicit split from ID files...")
        train_ids = read_record_id_file(args.train_ids)
        dev_ids = read_record_id_file(args.dev_ids)
        test_ids = read_record_id_file(args.test_ids)

        print(f"  Train IDs: {len(train_ids)}")
        print(f"  Dev IDs: {len(dev_ids)}")
        print(f"  Test IDs: {len(test_ids)}")

        manifest = create_explicit_split(
            input_path=args.input,
            oracle_graph_dir=args.oracle_graph_dir,
            dataset_name=args.dataset_name,
            train_ids=train_ids,
            dev_ids=dev_ids,
            test_ids=test_ids,
            require_oracle_graph=args.require_oracle_graph,
        )

    else:
        # Random seeded assignment mode
        print(f"Creating random split with seed {args.seed}...")
        print(f"  Train ratio: {args.train_ratio}")
        print(f"  Dev ratio: {args.dev_ratio}")
        print(f"  Test ratio: {args.test_ratio}")

        manifest = create_random_split(
            input_path=args.input,
            oracle_graph_dir=args.oracle_graph_dir,
            dataset_name=args.dataset_name,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            record_id_filter=args.record_id,
            require_oracle_graph=args.require_oracle_graph,
            debug_single_case=args.debug_single_case,
            allow_empty_split=args.allow_empty_split,
        )

    # Write manifest
    write_split_manifest(manifest, args.output)
    print(f"\nSplit manifest written to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Split Summary")
    print("=" * 60)
    print(f"Dataset: {manifest.dataset_name}")
    print(f"Source records: {manifest.source_records_path}")
    print(f"Oracle graph dir: {manifest.oracle_graph_dir}")
    print(f"Seed: {manifest.seed}")
    print(f"Record hash: {manifest.record_id_hash}")
    print("")
    print("Counts:")
    for split_name, count in manifest.counts.items():
        print(f"  {split_name.value}: {count}")
    print("")

    # Report issues
    if manifest.issues:
        print("Issues:")
        error_count = sum(
            1 for issue in manifest.issues if issue.severity == DatasetSplitIssueSeverity.ERROR
        )
        warning_count = sum(
            1 for issue in manifest.issues if issue.severity == DatasetSplitIssueSeverity.WARNING
        )
        info_count = sum(
            1 for issue in manifest.issues if issue.severity == DatasetSplitIssueSeverity.INFO
        )

        print(f"  Errors: {error_count}")
        print(f"  Warnings: {warning_count}")
        print(f"  Info: {info_count}")
        print("")

        for issue in manifest.issues:
            prefix = f"[{issue.severity.value.upper()}]"
            if issue.record_id:
                print(f"  {prefix} {issue.record_id}: {issue.message}")
            else:
                print(f"  {prefix} {issue.message}")

        # Return error code if errors present
        if error_count > 0:
            print("\nStatus: FAILED (errors present)")
            return 1
        elif warning_count > 0:
            print("\nStatus: SUCCEEDED (with warnings)")
            return 0
        else:
            print("\nStatus: SUCCEEDED")
            return 0
    else:
        print("Status: SUCCEEDED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
