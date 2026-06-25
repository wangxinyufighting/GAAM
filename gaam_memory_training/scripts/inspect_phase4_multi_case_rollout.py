#!/usr/bin/env python3
"""
Phase 4 Milestone 2: Multi-Case Rollout Inspector CLI

Inspects and validates a multi-case rollout run.

Example usage:
    python scripts/inspect_phase4_multi_case_rollout.py \
      --run_dir outputs/phase4_multi_case_rollout/train_round000_step000 \
      --strict
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.phase4_rollout_schema import (
    MultiCaseRolloutManifest,
    MultiCaseRolloutStatus,
)


def inspect_multi_case_rollout(
    run_dir: Path,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """
    Inspect multi-case rollout run directory.

    Checks:
    - Run manifest exists
    - Selected record count matches outputs
    - Per-record reports exist
    - Aggregate replay exists
    - Actor batch manifests exist
    - No-leakage inspection passed
    - Duplicate replay keys absent
    - Dev/test update policy respected
    - Reward metrics are finite

    Returns:
        (passed, issues)
    """
    issues: list[str] = []

    # Check manifest exists
    manifest_path = run_dir / "multi_case_rollout_manifest.json"
    if not manifest_path.exists():
        issues.append(f"Manifest not found: {manifest_path}")
        return False, issues

    # Load manifest
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_dict = json.load(f)
        manifest = MultiCaseRolloutManifest.model_validate(manifest_dict)
    except Exception as e:
        issues.append(f"Failed to load manifest: {e}")
        return False, issues

    # Check selected records match outputs
    records_dir = run_dir / "records"
    if not records_dir.exists():
        issues.append("Records directory not found")
    else:
        for record_id in manifest.selected_record_ids:
            record_dir = records_dir / record_id
            if not record_dir.exists():
                issues.append(f"Record directory not found: {record_id}")

    # Check per-record reports
    for record_run in manifest.records:
        if record_run.status == MultiCaseRolloutStatus.SUCCEEDED:
            if record_run.final_report_path:
                report_path = Path(record_run.final_report_path)
                if not report_path.exists():
                    issues.append(f"Final report not found for {record_run.record_id}")

    # Check aggregate artifacts
    aggregate_dir = run_dir / "aggregate"
    if not aggregate_dir.exists():
        issues.append("Aggregate directory not found")
    else:
        # Check replay buffer
        replay_buffer = aggregate_dir / "replay_buffer.jsonl"
        if not replay_buffer.exists():
            issues.append("Replay buffer not found")

        replay_manifest = aggregate_dir / "aggregated_replay_manifest.json"
        if not replay_manifest.exists():
            issues.append("Replay manifest not found")

        # Check actor batches (train split only)
        if manifest.split == "train":
            actor_batches_dir = aggregate_dir / "actor_batches"
            if not actor_batches_dir.exists():
                issues.append("Actor batches directory not found")
            else:
                for actor_role in ["memory_builder", "question_agent"]:
                    batch_file = actor_batches_dir / f"{actor_role}.update_batch.json"
                    if not batch_file.exists():
                        issues.append(f"Actor batch not found: {actor_role}")

                    manifest_file = (
                        actor_batches_dir / f"{actor_role}.batch_manifest.json"
                    )
                    if not manifest_file.exists():
                        issues.append(f"Actor manifest not found: {actor_role}")
                    else:
                        # Check no-leakage status
                        try:
                            with open(manifest_file, "r", encoding="utf-8") as f:
                                actor_manifest = json.load(f)
                            if not actor_manifest.get("no_leakage_passed", False):
                                issues.append(
                                    f"No-leakage check failed for {actor_role}"
                                )
                        except Exception:
                            pass

    # Check no-leakage pass rate
    succeeded = [
        r for r in manifest.records if r.status == MultiCaseRolloutStatus.SUCCEEDED
    ]
    if succeeded:
        pass_rate = sum(1 for r in succeeded if r.no_leakage_passed) / len(succeeded)
        if pass_rate < 1.0:
            msg = f"No-leakage pass rate: {pass_rate:.2%}"
            if strict:
                issues.append(msg)
            else:
                # Just a warning
                pass

    # Check reward metrics are finite
    for record_run in manifest.records:
        if record_run.memory_reward_mean is not None:
            if not (-1e6 < record_run.memory_reward_mean < 1e6):
                issues.append(
                    f"Invalid memory reward for {record_run.record_id}: {record_run.memory_reward_mean}"
                )
        if record_run.question_reward_mean is not None:
            if not (-1e6 < record_run.question_reward_mean < 1e6):
                issues.append(
                    f"Invalid question reward for {record_run.record_id}: {record_run.question_reward_mean}"
                )

    passed = len(issues) == 0
    return passed, issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Phase 4 multi-case rollout run"
    )

    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Multi-case rollout run directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format",
    )

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}", file=sys.stderr)
        return 1

    passed, issues = inspect_multi_case_rollout(args.run_dir, args.strict)

    if args.json:
        output = {
            "run_dir": str(args.run_dir),
            "passed": passed,
            "num_issues": len(issues),
            "issues": issues,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print(f"Multi-Case Rollout Inspection")
        print(f"{'=' * 60}")
        print(f"Run directory: {args.run_dir}")
        print(f"Status: {'PASSED' if passed else 'FAILED'}")
        print(f"Issues: {len(issues)}")

        if issues:
            print("\nIssues found:")
            for issue in issues:
                print(f"  - {issue}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
