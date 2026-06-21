#!/usr/bin/env python3
"""
CLI for Phase 2 Milestone 5: Inspect alignment of rollout/trainer artifacts.

This script inspects Phase 2 rollout and trainer outputs for Code-A1/verl compatibility:
- Validates actor update batches are safe (no forbidden teacher fields)
- Exports verl-like JSON batches for future tensorization
- Builds checkpoint registry for distributed training handoff
- Generates alignment report with issues and metrics

Usage examples:

    # Basic alignment check
    python scripts/inspect_phase2_alignment.py \
      --rollout_dir outputs/phase2_grpo_rollout \
      --output_dir outputs/phase2_alignment_report

    # Include trainer validation and checkpoint registry
    python scripts/inspect_phase2_alignment.py \
      --rollout_dir outputs/phase2_grpo_rollout \
      --trainer_dir outputs/phase2_grpo_trainer_local_hf \
      --output_dir outputs/phase2_alignment_report \
      --export_verl_like_batches \
      --build_checkpoint_registry

    # Strict mode (fail on errors)
    python scripts/inspect_phase2_alignment.py \
      --rollout_dir outputs/phase2_grpo_rollout \
      --trainer_dir outputs/phase2_grpo_trainer_local_hf \
      --output_dir outputs/phase2_alignment_report \
      --strict

    # Include unselected samples in verl-like batches
    python scripts/inspect_phase2_alignment.py \
      --rollout_dir outputs/phase2_grpo_rollout \
      --output_dir outputs/phase2_alignment_report \
      --export_verl_like_batches \
      --include_unselected_samples
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.code_a1_alignment import (
    build_alignment_report,
    AlignmentIssueSeverity,
)
from gaam_graph.verl_batch_adapter import export_verl_like_batches
from gaam_graph.checkpoint_registry import write_checkpoint_registry


def write_summary_text(report, output_path: Path):
    """Write human-readable summary to text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 2 Milestone 5: Code-A1/verl Alignment Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Report ID: {report.report_id}\n")
        f.write(f"Status: {report.status}\n")
        f.write(f"Rollout dir: {report.rollout_dir}\n")
        f.write(f"Trainer dir: {report.trainer_dir or '(not provided)'}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Actor Mappings (GAAM -> Code-A1)\n")
        f.write("-" * 80 + "\n\n")

        for mapping in report.actor_mappings:
            f.write(f"GAAM Role: {mapping.gaam_role.value}\n")
            f.write(f"Code-A1 Role: {mapping.code_a1_role.value}\n")
            f.write(f"Actor Name: {mapping.actor_name}\n")
            f.write(f"Objective: {mapping.objective_summary}\n")
            f.write(f"Allowed Inputs: {', '.join(mapping.allowed_inputs)}\n")
            f.write(f"Forbidden Inputs: {', '.join(mapping.forbidden_inputs)}\n")
            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("Issues Summary\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"Total issues: {report.metrics['total_issues']}\n")
        f.write(f"  Errors: {report.metrics['error_count']}\n")
        f.write(f"  Warnings: {report.metrics['warning_count']}\n")
        f.write(f"  Info: {report.metrics['info_count']}\n\n")

        if report.issues:
            f.write("-" * 80 + "\n")
            f.write("Issue Details\n")
            f.write("-" * 80 + "\n\n")

            for issue in report.issues:
                f.write(f"[{issue.severity.value.upper()}] {issue.code}\n")
                f.write(f"  Message: {issue.message}\n")
                if issue.artifact_path:
                    f.write(f"  Artifact: {issue.artifact_path}\n")
                if issue.record_id:
                    f.write(f"  Record ID: {issue.record_id}\n")
                if issue.role:
                    f.write(f"  Role: {issue.role.value}\n")
                f.write("\n")
        else:
            f.write("No issues found.\n\n")

        f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Phase 2 alignment for Code-A1/verl compatibility"
    )

    # Input/output
    parser.add_argument("--rollout_dir", type=Path, required=True, help="Rollout directory from Milestone 2")
    parser.add_argument("--trainer_dir", type=Path, help="Optional trainer directory from Milestone 3/4")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for alignment report")

    # Export options
    parser.add_argument("--export_verl_like_batches", action="store_true", default=True, help="Export verl-like batches (default)")
    parser.add_argument("--no_export_verl_like_batches", dest="export_verl_like_batches", action="store_false")
    parser.add_argument("--build_checkpoint_registry", action="store_true", default=True, help="Build checkpoint registry (default)")
    parser.add_argument("--no_build_checkpoint_registry", dest="build_checkpoint_registry", action="store_false")

    # Sample filtering
    parser.add_argument("--include_unselected_samples", action="store_true", help="Include unselected samples in verl-like batches")

    # Validation mode
    parser.add_argument("--strict", action="store_true", help="Fail on ERROR issues")

    # Record filtering
    parser.add_argument("--max_records", type=int, help="Maximum number of records to process")

    args = parser.parse_args()

    # Validate inputs
    if not args.rollout_dir.exists():
        print(f"Error: Rollout directory does not exist: {args.rollout_dir}", file=sys.stderr)
        return 1

    if args.trainer_dir and not args.trainer_dir.exists():
        print(f"Error: Trainer directory does not exist: {args.trainer_dir}", file=sys.stderr)
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 2 Milestone 5: Inspect Code-A1/verl Alignment")
    print("=" * 80)
    print(f"Rollout dir: {args.rollout_dir}")
    print(f"Trainer dir: {args.trainer_dir or '(not provided)'}")
    print(f"Output dir: {args.output_dir}")
    print(f"Export verl-like batches: {args.export_verl_like_batches}")
    print(f"Build checkpoint registry: {args.build_checkpoint_registry}")
    print(f"Include unselected samples: {args.include_unselected_samples}")
    print(f"Max records: {args.max_records or '(all)'}")
    print(f"Strict mode: {args.strict}")
    print("=" * 80)
    print()

    # Build alignment report
    print("Building alignment report...")
    report = build_alignment_report(
        rollout_dir=args.rollout_dir,
        trainer_dir=args.trainer_dir,
        max_records=args.max_records,
    )

    # Write alignment report JSON
    report_path = args.output_dir / "alignment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        import json
        json.dump(report.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    print(f"✓ Alignment report: {report_path}")

    # Write summary text
    summary_path = args.output_dir / "summary.txt"
    write_summary_text(report, summary_path)
    print(f"✓ Summary: {summary_path}")

    # Export verl-like batches
    if args.export_verl_like_batches:
        print("\nExporting verl-like batches...")
        try:
            exported_paths = export_verl_like_batches(
                rollout_dir=args.rollout_dir,
                output_dir=args.output_dir,
                include_unselected=args.include_unselected_samples,
                tokenizer=None,  # No tokenizer provided in Milestone 5
                max_records=args.max_records,
            )
            print(f"✓ Exported {len(exported_paths)} verl-like batches")
            manifest_path = args.output_dir / "verl_like_batches" / "manifest.jsonl"
            print(f"✓ Manifest: {manifest_path}")
        except Exception as e:
            print(f"✗ Failed to export verl-like batches: {e}", file=sys.stderr)
            if args.strict:
                return 1

    # Build checkpoint registry
    if args.build_checkpoint_registry and args.trainer_dir:
        print("\nBuilding checkpoint registry...")
        try:
            registry_path = write_checkpoint_registry(
                trainer_dir=args.trainer_dir,
                output_path=args.output_dir / "checkpoint_registry.json",
            )
            print(f"✓ Checkpoint registry: {registry_path}")
        except Exception as e:
            print(f"✗ Failed to build checkpoint registry: {e}", file=sys.stderr)
            if args.strict:
                return 1

    # Print issue summary
    print("\n" + "=" * 80)
    print("Issue Summary")
    print("=" * 80)
    print(f"Total issues: {report.metrics['total_issues']}")
    print(f"  Errors: {report.metrics['error_count']}")
    print(f"  Warnings: {report.metrics['warning_count']}")
    print(f"  Info: {report.metrics['info_count']}")
    print(f"Status: {report.status}")

    # Print issues by severity
    if report.issues:
        errors = [i for i in report.issues if i.severity == AlignmentIssueSeverity.ERROR]
        warnings = [i for i in report.issues if i.severity == AlignmentIssueSeverity.WARNING]
        info = [i for i in report.issues if i.severity == AlignmentIssueSeverity.INFO]

        if errors:
            print("\nErrors:")
            for issue in errors:
                print(f"  [{issue.code}] {issue.message}")
                if issue.artifact_path:
                    print(f"    Artifact: {issue.artifact_path}")

        if warnings:
            print("\nWarnings:")
            for issue in warnings:
                print(f"  [{issue.code}] {issue.message}")
                if issue.artifact_path:
                    print(f"    Artifact: {issue.artifact_path}")

        if info:
            print("\nInfo:")
            for issue in info:
                print(f"  [{issue.code}] {issue.message}")

    print("=" * 80)

    # Determine exit code
    if args.strict and report.metrics["error_count"] > 0:
        print("\n✗ Alignment check FAILED (strict mode, errors found)")
        return 1
    else:
        print("\n✓ Alignment check complete")
        return 0


if __name__ == "__main__":
    exit(main())
