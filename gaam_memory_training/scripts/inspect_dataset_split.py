#!/usr/bin/env python3
"""
Inspect Dataset Split for Phase 4

Validates and inspects dataset split manifests.

Usage:
    python scripts/inspect_dataset_split.py \\
      --split_manifest outputs/splits/longmemeval_s.phase4_split.seed0.json \\
      --strict

    python scripts/inspect_dataset_split.py \\
      --split_manifest outputs/splits/e47becba.debug_split.json \\
      --json
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.dataset_split_schema import (
    DatasetSplitIssueSeverity,
    DatasetSplitManifest,
)
from gaam_graph.dataset_splitter import validate_split_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and validate dataset split manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--split_manifest",
        required=True,
        type=Path,
        help="Path to split manifest JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on errors (exit code 1)",
    )
    parser.add_argument(
        "--fail_on_warning",
        action="store_true",
        help="Fail on warnings too (exit code 2)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON",
    )

    args = parser.parse_args()

    # Load manifest
    if not args.split_manifest.exists():
        print(f"Error: Split manifest not found: {args.split_manifest}", file=sys.stderr)
        return 1

    try:
        with open(args.split_manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest = DatasetSplitManifest.model_validate(manifest_data)
    except Exception as e:
        print(f"Error: Failed to load manifest: {e}", file=sys.stderr)
        return 1

    # Re-validate manifest
    validation_issues = validate_split_manifest(manifest)
    all_issues = manifest.issues + validation_issues

    # Count issues by severity
    error_count = sum(
        1 for issue in all_issues if issue.severity == DatasetSplitIssueSeverity.ERROR
    )
    warning_count = sum(
        1 for issue in all_issues if issue.severity == DatasetSplitIssueSeverity.WARNING
    )
    info_count = sum(
        1 for issue in all_issues if issue.severity == DatasetSplitIssueSeverity.INFO
    )

    # Check for duplicate record IDs
    record_ids = [r.record_id for r in manifest.records]
    duplicate_ids = [rid for rid in set(record_ids) if record_ids.count(rid) > 1]

    # Check for cross-split duplicates (should be impossible but check anyway)
    split_assignments = {}
    for record in manifest.records:
        if record.record_id not in split_assignments:
            split_assignments[record.record_id] = []
        split_assignments[record.record_id].append(record.split.value)
    cross_split_duplicates = {
        rid: splits for rid, splits in split_assignments.items() if len(splits) > 1
    }

    # JSON output
    if args.json:
        output = {
            "manifest_path": str(args.split_manifest),
            "dataset": manifest.dataset_name,
            "seed": manifest.seed,
            "record_hash": manifest.record_id_hash,
            "counts": {k.value: v for k, v in manifest.counts.items()},
            "validation": {
                "error_count": error_count,
                "warning_count": warning_count,
                "info_count": info_count,
                "duplicate_record_ids": len(duplicate_ids),
                "cross_split_duplicates": len(cross_split_duplicates),
            },
            "issues": [issue.model_dump(mode="json") for issue in all_issues],
            "status": "PASSED" if error_count == 0 else "FAILED",
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print("=" * 60)
        print("Dataset Split Inspection")
        print("=" * 60)
        print(f"Manifest: {args.split_manifest}")
        print(f"Dataset: {manifest.dataset_name}")
        print(f"Source records: {manifest.source_records_path}")
        print(f"Oracle graph dir: {manifest.oracle_graph_dir}")
        print(f"Seed: {manifest.seed}")
        print(f"Record hash: {manifest.record_id_hash}")
        print("")

        print("Counts:")
        for split_name, count in manifest.counts.items():
            print(f"  {split_name.value}: {count}")
        print("")

        print("Validation:")
        print(f"  Missing oracle graphs: {sum(1 for i in all_issues if i.code == 'missing_oracle_graph')}")
        print(f"  Duplicate record IDs: {len(duplicate_ids)}")
        print(f"  Cross-split duplicates: {len(cross_split_duplicates)}")
        print(f"  Graph record_id mismatches: {sum(1 for i in all_issues if i.code == 'oracle_graph_record_id_mismatch')}")
        print(f"  Leakage-sensitive fields stored: {sum(1 for i in all_issues if 'leakage' in i.code)}")
        print("")

        if all_issues:
            print("Issues:")
            print(f"  Errors: {error_count}")
            print(f"  Warnings: {warning_count}")
            print(f"  Info: {info_count}")
            print("")

            for issue in all_issues:
                prefix = f"[{issue.severity.value.upper()}]"
                if issue.record_id:
                    print(f"  {prefix} {issue.record_id}: {issue.message}")
                else:
                    print(f"  {prefix} {issue.message}")

        print("")
        if error_count > 0:
            print("Status: FAILED")
        elif warning_count > 0:
            print("Status: PASSED (with warnings)")
        else:
            print("Status: PASSED")

    # Determine exit code
    if error_count > 0:
        if args.strict:
            return 1
    if warning_count > 0:
        if args.fail_on_warning:
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
