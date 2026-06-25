#!/usr/bin/env python3
"""
Phase 4 Milestone 3: Trainer Step Inspection CLI

Inspects and validates completed trainer steps.

Usage:
    python scripts/inspect_phase4_grpo_trainer_step.py \
      --run_dir outputs/phase4_trainer_steps/train_round000_step000 \
      --strict

Exit codes:
    0 = passed
    1 = failed (errors found)
    2 = warnings found (strict mode only)
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.phase4_trainer_inspection import inspect_phase4_trainer_step


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect and validate Phase 4 trainer step",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Trainer step output directory to inspect",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format (for automation)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Write inspection report to file",
    )

    return parser.parse_args()


def main() -> int:
    """Main entrypoint."""
    args = parse_args()

    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    try:
        # Run inspection
        report = inspect_phase4_trainer_step(run_dir, strict=args.strict)

        # Write to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            print(f"Inspection report written to: {output_path}")

        # Output format
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            # Human-readable output
            passed = report["passed"]
            num_errors = report["num_errors"]
            num_warnings = report["num_warnings"]

            if passed:
                print(f"✅ Trainer step passed inspection")
            else:
                print(f"❌ Trainer step failed inspection")

            if num_errors > 0:
                print(f"   Errors: {num_errors}")

            if num_warnings > 0:
                print(f"   Warnings: {num_warnings}")

            # Print issues
            if report["issues"]:
                print(f"\nIssues found:")

                for issue in report["issues"]:
                    severity = issue["severity"]
                    code = issue.get("code", issue.get("category", "unknown"))
                    message = issue["message"]

                    emoji = "❌" if severity == "error" else "⚠️"
                    print(f"  {emoji} [{code}] {message}")

            # Print manifest summary
            manifest = report.get("manifest")
            if manifest:
                print(f"\nTrainer Step Summary:")
                print(f"  Run ID: {manifest['run_id']}")
                print(f"  Status: {manifest['status']}")
                print(f"  Split: {manifest['split']}")
                print(f"  Backend: {manifest['backend']}")
                print(f"  Actors: {len(manifest['actor_reports'])}")

                for actor_report in manifest["actor_reports"]:
                    actor_role = actor_report["actor_role"]
                    status = actor_report["status"]
                    no_leakage = actor_report["no_leakage_passed"]

                    status_emoji = "✅" if status == "succeeded" else "❌"
                    leakage_emoji = "✅" if no_leakage else "❌"

                    print(f"    {status_emoji} {actor_role}: {status}")
                    print(f"       No-leakage: {leakage_emoji}")

                    if actor_report.get("written_checkpoint_id"):
                        print(
                            f"       Checkpoint: {actor_report['written_checkpoint_id']}"
                        )

        # Exit code
        if report["passed"]:
            return 0
        elif args.strict and num_warnings > 0:
            return 2
        else:
            return 1

    except Exception as e:
        print(f"❌ Inspection failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
