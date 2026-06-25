#!/usr/bin/env python3
"""
Build Phase 4 Release Package

Build reproducible release package from benchmark suite.
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.phase4_release_package import build_phase4_release_package


def main():
    parser = argparse.ArgumentParser(
        description="Build Phase 4 release package"
    )

    parser.add_argument(
        "--suite_dir",
        type=str,
        required=True,
        help="Path to benchmark suite directory",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory (default: suite_dir/release)",
    )

    parser.add_argument(
        "--include_checkpoints",
        action="store_true",
        help="Include model checkpoints in release package",
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

        output_dir = Path(args.output_dir) if args.output_dir else None

        if not args.json:
            print(f"Building release package from: {suite_dir}")
            if output_dir:
                print(f"Output directory: {output_dir}")
            else:
                print(f"Output directory: {suite_dir / 'release'}")
            print()

        # Build release package
        release_dir = build_phase4_release_package(
            suite_dir,
            output_dir=output_dir,
            include_checkpoints=args.include_checkpoints,
        )

        if args.json:
            result = {
                "status": "succeeded",
                "release_dir": str(release_dir),
            }
            print(json.dumps(result, indent=2))
        else:
            print()
            print("=" * 80)
            print("Release Package Complete")
            print("=" * 80)
            print(f"Release directory: {release_dir}")
            print()
            print("Contents:")
            print("  - README.md")
            print("  - reproduce.sh")
            print("  - suite_config.resolved.json")
            print("  - suite_manifest.json")
            print("  - aggregate_metrics.json")
            print("  - aggregate_metrics.csv")
            print("  - paper_table.md")
            print("  - paper_table.tex")
            print("  - learning_curves.json")
            print("  - environment_report.json")
            print("  - artifact_manifest.json")
            print("  - checksums.sha256")
            print()
            print("To reproduce:")
            print(f"  cd {release_dir}")
            print("  bash reproduce.sh")

        sys.exit(0)

    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
