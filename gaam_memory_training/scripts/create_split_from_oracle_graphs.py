#!/usr/bin/env python3
"""
Create a train/dev/test split from existing oracle graph files only.

This script scans an oracle graph directory for ``*.graph.json`` files, extracts
their record IDs, and creates a split manifest containing only those cases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.dataset_split_schema import DatasetSplitIssueSeverity  # noqa: E402
from gaam_graph.dataset_splitter import create_random_split, write_split_manifest  # noqa: E402


def discover_graph_record_ids(oracle_graph_dir: Path) -> list[str]:
    """Discover record IDs from ``*.graph.json`` files."""
    if not oracle_graph_dir.exists():
        raise FileNotFoundError(f"Oracle graph directory not found: {oracle_graph_dir}")
    return sorted(
        path.name.removesuffix(".graph.json")
        for path in oracle_graph_dir.glob("*.graph.json")
        if path.is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a split manifest from available oracle graph files only."
    )
    parser.add_argument("--input", type=Path, required=True, help="LongMemEval input JSON")
    parser.add_argument("--oracle_graph_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset_name", default="longmemeval")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_empty_split", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Optional cap on number of discovered graph cases to include after sorting.",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"Error: output exists: {args.output}", file=sys.stderr)
        print("Use --overwrite to replace it.", file=sys.stderr)
        return 1

    record_ids = discover_graph_record_ids(args.oracle_graph_dir)
    if args.max_records is not None:
        record_ids = record_ids[: args.max_records]

    print("Discovered oracle graph cases:")
    print(f"  oracle_graph_dir: {args.oracle_graph_dir}")
    print(f"  num_cases: {len(record_ids)}")
    if record_ids:
        preview = ", ".join(record_ids[:20])
        suffix = " ..." if len(record_ids) > 20 else ""
        print(f"  record_ids: {preview}{suffix}")

    if not record_ids:
        print("Error: no *.graph.json files found.", file=sys.stderr)
        return 1

    debug_single_case = len(record_ids) == 1
    allow_empty_split = args.allow_empty_split or debug_single_case

    manifest = create_random_split(
        input_path=args.input,
        oracle_graph_dir=args.oracle_graph_dir,
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        record_id_filter=record_ids,
        require_oracle_graph=True,
        debug_single_case=debug_single_case,
        allow_empty_split=allow_empty_split,
    )

    write_split_manifest(manifest, args.output)
    print(f"\nSplit manifest written to: {args.output}")

    print("\n" + "=" * 60)
    print("Oracle Graph Split Summary")
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

    if manifest.issues:
        error_count = sum(
            1 for issue in manifest.issues if issue.severity == DatasetSplitIssueSeverity.ERROR
        )
        warning_count = sum(
            1 for issue in manifest.issues if issue.severity == DatasetSplitIssueSeverity.WARNING
        )
        print("")
        print("Issues:")
        print(f"  Errors: {error_count}")
        print(f"  Warnings: {warning_count}")
        for issue in manifest.issues:
            prefix = f"[{issue.severity.value.upper()}]"
            if issue.record_id:
                print(f"  {prefix} {issue.record_id}: {issue.message}")
            else:
                print(f"  {prefix} {issue.message}")
        if error_count:
            print("\nStatus: FAILED")
            return 1

    print("\nStatus: SUCCEEDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
