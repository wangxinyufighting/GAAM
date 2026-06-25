#!/usr/bin/env python3
"""
Inspect Phase 4 Benchmark Suite

Validate benchmark suite artifacts and completeness.
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.phase4_suite_runner import load_suite_manifest


def inspect_suite_artifacts(suite_dir: Path) -> tuple[bool, list[str]]:
    """Check that all expected artifacts exist."""
    issues = []

    # Check suite manifest
    if not (suite_dir / "suite_manifest.json").exists():
        issues.append("Missing suite_manifest.json")

    # Check resolved config
    if not (suite_dir / "suite_config.resolved.json").exists():
        issues.append("Missing suite_config.resolved.json")

    # Check aggregate directory
    aggregate_dir = suite_dir / "aggregate"
    if aggregate_dir.exists():
        expected_files = [
            "aggregate_metrics.json",
            "aggregate_metrics.csv",
            "seed_level_metrics.jsonl",
        ]

        for file_name in expected_files:
            if not (aggregate_dir / file_name).exists():
                issues.append(f"Missing aggregate file: {file_name}")

    return len(issues) == 0, issues


def inspect_per_seed_runs(suite_dir: Path, manifest) -> tuple[bool, list[str]]:
    """Check per-seed experiment outputs."""
    issues = []

    for run_record in manifest.run_records:
        seed = run_record.seed
        run_dir = Path(run_record.output_dir)

        if not run_dir.exists():
            issues.append(f"Seed {seed}: output directory not found")
            continue

        # Check cotraining
        if not (run_dir / "cotraining" / "cotraining_manifest.json").exists():
            issues.append(f"Seed {seed}: missing cotraining manifest")

        # Check final evaluation (if status is succeeded)
        if run_record.status == "succeeded":
            if not (run_dir / "final_evaluation" / "final_evaluation_manifest.json").exists():
                issues.append(f"Seed {seed}: missing final evaluation manifest")

    return len(issues) == 0, issues


def inspect_suite(suite_dir: Path, *, strict: bool) -> tuple[bool, list[str], list[str]]:
    """
    Inspect benchmark suite.

    Returns:
        Tuple of (passed, issues, warnings)
    """
    issues = []
    warnings = []

    # Load suite manifest
    try:
        manifest = load_suite_manifest(suite_dir / "suite_manifest.json")
    except Exception as exc:
        issues.append(f"Failed to load suite manifest: {exc}")
        return False, issues, warnings

    # Check artifacts
    artifacts_ok, artifacts_issues = inspect_suite_artifacts(suite_dir)
    issues.extend(artifacts_issues)

    # Check per-seed runs
    runs_ok, runs_issues = inspect_per_seed_runs(suite_dir, manifest)
    issues.extend(runs_issues)

    # Check for failed runs
    failed_runs = [r for r in manifest.run_records if r.status == "failed"]
    if failed_runs:
        warnings.append(f"{len(failed_runs)} seed(s) failed")

    # Check suite status
    if manifest.status != "succeeded":
        if strict:
            issues.append(f"Suite status is {manifest.status}, not succeeded")
        else:
            warnings.append(f"Suite status is {manifest.status}")

    passed = len(issues) == 0
    return passed, issues, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Phase 4 benchmark suite"
    )

    parser.add_argument(
        "--suite_dir",
        type=str,
        required=True,
        help="Path to benchmark suite directory",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings",
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

        # Inspect suite
        passed, issues, warnings = inspect_suite(suite_dir, strict=args.strict)

        if args.json:
            result = {
                "status": "passed" if passed else "failed",
                "suite_dir": str(suite_dir),
                "issues": issues,
                "warnings": warnings,
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"Inspecting benchmark suite: {suite_dir}")
            print()

            if issues:
                print("Issues:")
                for issue in issues:
                    print(f"  ❌ {issue}")
                print()

            if warnings:
                print("Warnings:")
                for warning in warnings:
                    print(f"  ⚠️  {warning}")
                print()

            if passed:
                print("✅ Inspection passed")
            else:
                print("❌ Inspection failed")

        sys.exit(0 if passed else 1)

    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
