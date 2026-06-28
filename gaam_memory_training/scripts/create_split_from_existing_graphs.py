#!/usr/bin/env python3
"""Create a train/dev/test split from oracle graphs that already exist."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.dataset_split_schema import DatasetSplitIssueSeverity  # noqa: E402
from gaam_graph.dataset_splitter import create_random_split, write_split_manifest  # noqa: E402
from gaam_graph.lme_loader import LongMemEvalLoader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic train/dev/test split using only records that "
            "have prebuilt oracle graph files."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("data/longmemeval/longmemeval_s_cleaned.json"))
    parser.add_argument("--oracle_graph_dir", type=Path, default=Path("outputs/longmemeval_s_graph"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset_name", default="longmemeval")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_empty_split", action="store_true")
    parser.add_argument("--debug_single_case", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Optional cap after sorting detected graph-backed record IDs.",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"Error: output exists: {args.output}", file=sys.stderr)
        print("Use --overwrite to replace it.", file=sys.stderr)
        return 1

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1
    if not args.oracle_graph_dir.exists():
        print(f"Error: oracle graph directory not found: {args.oracle_graph_dir}", file=sys.stderr)
        return 1

    record_ids = _graph_backed_record_ids(args.input, args.oracle_graph_dir)
    if args.max_cases is not None:
        record_ids = record_ids[: args.max_cases]

    if not record_ids:
        print(
            f"Error: no input records have matching oracle graphs in {args.oracle_graph_dir}",
            file=sys.stderr,
        )
        return 1

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
        debug_single_case=args.debug_single_case,
        allow_empty_split=args.allow_empty_split,
    )
    write_split_manifest(manifest, args.output)

    error_count = sum(
        1 for issue in manifest.issues if issue.severity == DatasetSplitIssueSeverity.ERROR
    )
    warning_count = sum(
        1 for issue in manifest.issues if issue.severity == DatasetSplitIssueSeverity.WARNING
    )

    print("== Split from existing oracle graphs ==")
    print(f"Input records: {args.input}")
    print(f"Oracle graph dir: {args.oracle_graph_dir}")
    print(f"Detected graph-backed records: {len(record_ids)}")
    print(f"Output: {args.output}")
    print("Counts:")
    for split_name, count in manifest.counts.items():
        print(f"  {split_name.value}: {count}")
    print(f"Issues: errors={error_count}, warnings={warning_count}")
    if manifest.issues:
        for issue in manifest.issues[:20]:
            print(f"  [{issue.severity.value.upper()}] {issue.record_id or '-'}: {issue.message}")
    return 1 if error_count else 0


def _graph_backed_record_ids(input_path: Path, oracle_graph_dir: Path) -> list[str]:
    graph_ids = {
        path.name.removesuffix(".graph.json")
        for path in oracle_graph_dir.glob("*.graph.json")
        if path.is_file() and path.name.endswith(".graph.json")
    }
    records = LongMemEvalLoader(str(input_path)).load()
    input_ids = {record.record_id for record in records}
    return sorted(graph_ids & input_ids)


if __name__ == "__main__":
    raise SystemExit(main())
