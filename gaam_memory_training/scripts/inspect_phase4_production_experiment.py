#!/usr/bin/env python3
"""
Inspect Phase 4 production experiment.

Usage:
    python scripts/inspect_phase4_production_experiment.py \
      --run_dir outputs/phase4_experiments/exp001 \
      --strict

Exit codes:
    0 = passed
    1 = failed
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.dataset_split_schema import DatasetSplitIssue
from gaam_graph.phase4_cotraining_inspection import inspect_phase4_cotraining
from gaam_graph.phase4_experiment_schema import Phase4ProductionExperimentManifest
from gaam_graph.phase4_leakage_audit import (
    load_leakage_audit_report,
    run_phase4_leakage_audit,
)


def inspect_phase4_production_experiment(
    run_dir: Path,
    *,
    strict: bool = False,
) -> dict:
    """
    Inspect production experiment for completeness and correctness.

    Args:
        run_dir: Experiment directory
        strict: Treat warnings as errors

    Returns:
        Inspection report
    """
    issues: list[DatasetSplitIssue] = []
    warnings: list[str] = []

    # Load experiment manifest
    manifest_path = run_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        issues.append(
            DatasetSplitIssue(
                severity="error",
                code="missing_artifact",
                message=f"Experiment manifest not found: {manifest_path}",
            )
        )
        return {
            "passed": False,
            "issues": [issue.model_dump() for issue in issues],
            "warnings": warnings,
        }

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest = Phase4ProductionExperimentManifest.model_validate(manifest_data)

    # Check co-training artifacts
    if manifest.cotraining_manifest_path:
        cotraining_path = Path(manifest.cotraining_manifest_path)
        if not cotraining_path.exists():
            issues.append(
                DatasetSplitIssue(
                    severity="error",
                    code="missing_artifact",
                    message=f"Co-training manifest not found: {cotraining_path}",
                )
            )
        else:
            # Inspect co-training loop
            cotraining_dir = Path(manifest.cotraining_dir)
            cotraining_report = inspect_phase4_cotraining(
                cotraining_dir, strict=strict
            )

            if not cotraining_report["passed"]:
                for issue_dict in cotraining_report["issues"]:
                    issues.append(DatasetSplitIssue.model_validate(issue_dict))

    # Check final evaluation artifacts
    if manifest.final_evaluation_manifest_path:
        eval_path = Path(manifest.final_evaluation_manifest_path)
        if not eval_path.exists():
            issues.append(
                DatasetSplitIssue(
                    severity="error",
                    code="missing_artifact",
                    message=f"Final evaluation manifest not found: {eval_path}",
                )
            )

    # Check final checkpoints
    if manifest.final_memory_builder_checkpoint_id is None:
        warnings.append("No final Memory Builder checkpoint selected")

    if manifest.final_question_agent_checkpoint_id is None:
        warnings.append("No final Question Agent checkpoint selected")

    # Check leakage audit
    leakage_audit_path = run_dir / "reports" / "leakage_audit.json"
    if leakage_audit_path.exists():
        try:
            audit_report = load_leakage_audit_report(leakage_audit_path)
            if audit_report.status == "failed":
                issues.append(
                    DatasetSplitIssue(
                        severity="error",
                        code="leakage_risk",
                        message=f"Leakage audit failed with {len(audit_report.issues)} issues",
                    )
                )
        except Exception as e:
            warnings.append(f"Failed to load leakage audit report: {e}")
    else:
        warnings.append("Leakage audit report not found")

    # Check reports directory
    if manifest.reports_dir:
        reports_dir = Path(manifest.reports_dir)
        if not reports_dir.exists():
            warnings.append(f"Reports directory not found: {reports_dir}")

    # Determine pass/fail
    has_errors = any(issue.severity == "error" for issue in issues)
    has_warnings = len(warnings) > 0 or any(
        issue.severity == "warning" for issue in issues
    )

    passed = not has_errors
    if strict and has_warnings:
        passed = False

    return {
        "passed": passed,
        "issues": [issue.model_dump() for issue in issues],
        "warnings": warnings,
        "manifest": manifest.model_dump(),
        "num_errors": sum(1 for issue in issues if issue.severity == "error"),
        "num_warnings": len(warnings)
        + sum(1 for issue in issues if issue.severity == "warning"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Phase 4 production experiment"
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Experiment run directory",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Run inspection
    report = inspect_phase4_production_experiment(run_dir, strict=args.strict)

    # Output results
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n=== Production Experiment Inspection ===")
        print(f"Passed: {report['passed']}")
        print(f"Errors: {report['num_errors']}")
        print(f"Warnings: {report['num_warnings']}")

        if report["issues"]:
            print(f"\nIssues:")
            for issue in report["issues"]:
                severity = issue["severity"]
                message = issue["message"]
                print(f"  [{severity}] {message}")

        if report["warnings"]:
            print(f"\nWarnings:")
            for warning in report["warnings"]:
                print(f"  - {warning}")

    # Exit code
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
