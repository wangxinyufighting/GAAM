#!/usr/bin/env python3
"""
Phase 4 Milestone 4: Co-Training Inspection CLI

Inspect and validate completed co-training experiment.

Usage:
    python scripts/inspect_phase4_cotraining_loop.py \
      --run_dir outputs/phase4_cotraining/exp001 \
      --strict

Exit codes:
    0 = passed
    1 = failed
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.phase4_cotraining_inspection import (
    inspect_phase4_cotraining,
    write_cotraining_inspection_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Phase 4 co-training experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inspection
  python scripts/inspect_phase4_cotraining_loop.py \
    --run_dir outputs/phase4_cotraining/exp001

  # Strict mode (warnings treated as errors)
  python scripts/inspect_phase4_cotraining_loop.py \
    --run_dir outputs/phase4_cotraining/exp001 \
    --strict

  # JSON output
  python scripts/inspect_phase4_cotraining_loop.py \
    --run_dir outputs/phase4_cotraining/exp001 \
    --json > inspection.json
        """,
    )

    parser.add_argument(
        "--run_dir",
        required=True,
        help="Co-training experiment directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--min_no_leakage_pass_rate",
        type=float,
        default=0.8,
        help="Minimum acceptable no-leakage pass rate (default: 0.8)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--output",
        help="Write inspection report to file",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Run inspection
    try:
        report = inspect_phase4_cotraining(
            run_dir,
            strict=args.strict,
            min_no_leakage_pass_rate=args.min_no_leakage_pass_rate,
        )

        # Write report to file if requested
        if args.output:
            output_path = Path(args.output)
            write_cotraining_inspection_report(report, output_path)
            if not args.json:
                print(f"Inspection report written to: {output_path}")

        # Output
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            # Human-readable output
            print(f"\nCo-Training Inspection: {run_dir}")
            print(f"{'='*60}")
            print(f"Status: {'PASSED' if report['passed'] else 'FAILED'}")
            print(f"Errors: {report['num_errors']}")
            print(f"Warnings: {report['num_warnings']}")

            if report["manifest"]:
                manifest = report["manifest"]
                print(f"\nExperiment: {manifest['run_id']}")
                print(
                    f"Rounds: {manifest['num_rounds_completed']}/{manifest['num_rounds_requested']}"
                )
                print(f"Status: {manifest['status']}")

            if report["issues"]:
                print(f"\nIssues:")
                for issue in report["issues"]:
                    severity = issue["severity"].upper()
                    code = issue["code"]
                    message = issue["message"]
                    print(f"  [{severity}] {code}: {message}")
            else:
                print("\nNo issues found.")

        # Exit code
        sys.exit(0 if report["passed"] else 1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
