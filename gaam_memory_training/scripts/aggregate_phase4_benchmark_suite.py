#!/usr/bin/env python3
"""
Aggregate Phase 4 Benchmark Suite

Aggregate metrics across multiple seeds and generate reports.
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.phase4_statistical_analysis import (
    compute_phase4_statistical_tests,
    write_statistical_tests_json,
)
from gaam_graph.phase4_suite_aggregation import (
    aggregate_phase4_benchmark_suite,
    collect_seed_level_metrics,
)
from gaam_graph.phase4_suite_reporting import generate_suite_reports
from gaam_graph.phase4_suite_runner import load_suite_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Phase 4 benchmark suite metrics"
    )

    parser.add_argument(
        "--suite_dir",
        type=str,
        required=True,
        help="Path to benchmark suite directory",
    )

    parser.add_argument(
        "--include_failed",
        action="store_true",
        help="Include failed runs in aggregation",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory (default: suite_dir/aggregate)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only",
    )

    args = parser.parse_args()

    try:
        suite_dir = Path(args.suite_dir)

        if not suite_dir.exists():
            print(f"Suite directory not found: {suite_dir}", file=sys.stderr)
            sys.exit(1)

        if not args.json:
            print(f"Aggregating benchmark suite: {suite_dir}")
            print()

        aggregate_dir = Path(args.output_dir) if args.output_dir else suite_dir / "aggregate"

        # Aggregate metrics
        aggregate_metrics = aggregate_phase4_benchmark_suite(
            suite_dir,
            include_failed=args.include_failed,
            output_dir=aggregate_dir,
        )

        # Load suite manifest for seed metrics
        manifest = load_suite_manifest(suite_dir / "suite_manifest.json")
        seed_metrics = collect_seed_level_metrics(suite_dir, manifest.run_records)

        # Compute statistical tests
        statistical_tests = compute_phase4_statistical_tests(seed_metrics)

        write_statistical_tests_json(statistical_tests, aggregate_dir / "statistical_tests.json")

        # Generate reports
        generate_suite_reports(
            suite_dir,
            aggregate_metrics.aggregate_by_method,
            formats=["markdown", "latex", "json"],
            output_dir=aggregate_dir,
        )

        if args.json:
            print(json.dumps(aggregate_metrics.model_dump(), indent=2))
        else:
            print()
            print("=" * 80)
            print("Aggregation Complete")
            print("=" * 80)
            print(f"Suite ID: {aggregate_metrics.suite_id}")
            print(f"Seeds requested: {aggregate_metrics.num_seeds_requested}")
            print(f"Seeds succeeded: {aggregate_metrics.num_seeds_succeeded}")
            print(f"Seeds partial: {aggregate_metrics.num_seeds_partial}")
            print(f"Seeds failed: {aggregate_metrics.num_seeds_failed}")
            print()
            print("Output files:")
            print(f"  - {aggregate_dir / 'aggregate_metrics.json'}")
            print(f"  - {aggregate_dir / 'aggregate_metrics.csv'}")
            print(f"  - {aggregate_dir / 'seed_level_metrics.jsonl'}")
            print(f"  - {aggregate_dir / 'statistical_tests.json'}")
            print(f"  - {aggregate_dir / 'paper_table.md'}")
            print(f"  - {aggregate_dir / 'paper_table.tex'}")
            print(f"  - {aggregate_dir / 'learning_curves.json'}")
            print(f"  - {aggregate_dir / 'figure_data.json'}")

        sys.exit(0)

    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
