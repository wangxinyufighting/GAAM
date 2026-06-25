#!/usr/bin/env python3
"""
Export Phase 4 experiment reports.

Usage:
    python scripts/export_phase4_experiment_report.py \
      --run_dir outputs/phase4_experiments/exp001 \
      --formats csv jsonl json markdown

Exit codes:
    0 = succeeded
    1 = failed
"""

import argparse
import sys
from pathlib import Path

from gaam_graph.phase4_experiment_report import export_experiment_reports


def main():
    parser = argparse.ArgumentParser(description="Export Phase 4 experiment reports")

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Experiment run directory",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (defaults to run_dir/reports)",
    )

    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["csv", "jsonl", "json", "markdown"],
        choices=["csv", "jsonl", "json", "markdown"],
        help="Report formats to generate",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Use output_dir if specified, otherwise use run_dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_dir

    try:
        # Export reports
        export_experiment_reports(output_dir, formats=args.formats)

        print(f"\n=== Reports Exported ===")
        print(f"Formats: {', '.join(args.formats)}")
        print(f"Location: {output_dir / 'reports'}")

        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
