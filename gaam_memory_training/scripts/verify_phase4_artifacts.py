#!/usr/bin/env python3
"""
Verify Phase 4 Release Artifacts

Verify release package integrity and completeness.
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.phase4_artifact_verification import (
    verify_phase4_release_artifacts,
    write_verification_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Verify Phase 4 release artifacts"
    )

    parser.add_argument(
        "--release_dir",
        type=str,
        required=True,
        help="Path to release directory",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any issue",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only",
    )

    args = parser.parse_args()

    try:
        release_dir = Path(args.release_dir)

        if not release_dir.exists():
            print(f"Release directory not found: {release_dir}", file=sys.stderr)
            sys.exit(1)

        if not args.json:
            print(f"Verifying release artifacts: {release_dir}")
            print()

        # Verify artifacts
        report = verify_phase4_release_artifacts(release_dir, strict=args.strict)

        # Write report
        write_verification_report(report, release_dir / "verification_report.json")

        if args.json:
            print(json.dumps(report.model_dump(), indent=2))
        else:
            print(f"Checks passed: {report.checks_passed}")
            print(f"Checks failed: {report.checks_failed}")
            print()

            if report.issues:
                print("Issues:")
                for issue in report.issues:
                    print(f"  ❌ {issue}")
                print()

            if report.warnings:
                print("Warnings:")
                for warning in report.warnings:
                    print(f"  ⚠️  {warning}")
                print()

            if report.status == "passed":
                print("✅ Verification passed")
            else:
                print("❌ Verification failed")

        sys.exit(0 if report.status == "passed" else 1)

    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
