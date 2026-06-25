#!/usr/bin/env python3
"""
Phase 4 Milestone 4: Metrics Export CLI

Export co-training metrics to CSV or JSONL for analysis.

Usage:
    python scripts/export_phase4_cotraining_metrics.py \
      --run_dir outputs/phase4_cotraining/exp001 \
      --output metrics.csv

Exit codes:
    0 = success
    1 = error
"""

import argparse
import sys
from pathlib import Path

from gaam_graph.phase4_cotraining_loop import load_cotraining_manifest
from gaam_graph.phase4_metrics_export import export_cotraining_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Export Phase 4 co-training metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to CSV
  python scripts/export_phase4_cotraining_metrics.py \
    --run_dir outputs/phase4_cotraining/exp001 \
    --output metrics.csv

  # Export to JSONL
  python scripts/export_phase4_cotraining_metrics.py \
    --run_dir outputs/phase4_cotraining/exp001 \
    --output metrics.jsonl \
    --format jsonl
        """,
    )

    parser.add_argument(
        "--run_dir",
        required=True,
        help="Co-training experiment directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default="csv",
        help="Output format (default: csv)",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_path = Path(args.output)

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Load manifest
    try:
        manifest_path = run_dir / "cotraining_manifest.json"
        manifest = load_cotraining_manifest(manifest_path)

        # Auto-detect format from extension if not specified
        if args.format == "csv" and output_path.suffix == ".jsonl":
            format = "jsonl"
        elif args.format == "jsonl" and output_path.suffix == ".csv":
            format = "csv"
        else:
            format = args.format

        # Export metrics
        export_cotraining_metrics(manifest, output_path, format=format)

        print(f"Metrics exported to: {output_path}")
        print(f"Format: {format}")
        print(f"Rounds: {len(manifest.round_reports)}")

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
