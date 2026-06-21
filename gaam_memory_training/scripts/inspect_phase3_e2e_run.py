#!/usr/bin/env python3
"""
Phase 3 Milestone 6: E2E Run Inspector CLI

Inspects and validates completed Phase 3 E2E GRPO runs.

Usage:
    python scripts/inspect_phase3_e2e_run.py \
      --run_dir outputs/phase3_e2e_smoke \
      --strict
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.phase3_e2e_schema import Phase3E2ERunReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def inspect_phase3_e2e_run(run_dir: Path, strict: bool = False) -> int:
    """
    Inspect Phase 3 E2E run directory.

    Returns:
        0 if run succeeded and all checks pass
        1 if run failed or critical artifacts missing
        2 if run partial or warnings present
    """
    # Load final report
    final_report_path = run_dir / "final_report.json"
    if not final_report_path.exists():
        logger.error(f"Final report not found: {final_report_path}")
        return 1

    with open(final_report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    try:
        report = Phase3E2ERunReport.model_validate(report_data)
    except Exception as e:
        logger.error(f"Failed to parse final report: {e}")
        return 1

    # Print summary
    print("\n" + "=" * 80)
    print("Phase 3 E2E Run Inspection")
    print("=" * 80)
    print(f"Run ID: {report.run_id}")
    print(f"Status: {report.status.value}")
    print(f"Record: {report.record_id}")
    print(f"Backend: {report.backend.value}")
    print(f"Started: {report.started_at}")
    print(f"Finished: {report.finished_at or 'N/A'}")

    # Check stage statuses
    if report.round_reports:
        round_report = report.round_reports[0]
        succeeded = sum(1 for r in round_report.stage_reports if r.status.value == "succeeded")
        partial = sum(1 for r in round_report.stage_reports if r.status.value == "partial")
        failed = sum(1 for r in round_report.stage_reports if r.status.value == "failed")
        skipped = sum(1 for r in round_report.stage_reports if r.status.value == "skipped")

        print(f"\nStages: {succeeded} succeeded, {partial} partial, {failed} failed, {skipped} skipped")

        # Print stage details
        print("\nStage Results:")
        for stage_report in round_report.stage_reports:
            status_symbol = {
                "succeeded": "✓",
                "partial": "⚠",
                "failed": "✗",
                "skipped": "−",
            }.get(stage_report.status.value, "?")

            print(f"  {status_symbol} {stage_report.stage.value:15s} {stage_report.status.value:10s}")

            if stage_report.errors:
                for error in stage_report.errors[:2]:
                    print(f"      Error: {error}")
            if stage_report.warnings:
                for warning in stage_report.warnings[:2]:
                    print(f"      Warning: {warning}")

    # Check replay items
    print("\nReplay Items:")
    if report.round_reports:
        round_report = report.round_reports[0]
        print(f"  Total: {round_report.num_replay_items}")
        print(f"  Weakness updates: {round_report.num_weakness_updates}")

    # Check rewards
    print("\nRewards:")
    if report.round_reports:
        round_report = report.round_reports[0]
        if round_report.memory_reward_mean is not None:
            print(f"  Memory reward: {round_report.memory_reward_mean:.3f}")
        else:
            print("  Memory reward: N/A")

        if round_report.question_reward_mean is not None:
            print(f"  Question reward: {round_report.question_reward_mean:.3f}")
        else:
            print("  Question reward: N/A")

    # Check no-leakage
    print("\nNo-Leakage Validation:")
    if report.no_leakage_passed:
        print("  ✓ Passed")
    else:
        print("  ✗ FAILED")

    # Check remote GPU readiness
    print("\nRemote GPU Readiness:")
    if report.remote_gpu_ready:
        print("  ✓ Ready")
    else:
        print("  − Not ready")
        for gap in report.remote_gpu_readiness_gaps:
            print(f"    Gap: {gap}")

    print("\nCode-A1/verl Readiness:")
    if report.code_a1_verl_ready:
        print("  ✓ Ready")
    else:
        print("  − Not ready")
        for gap in report.code_a1_verl_readiness_gaps:
            print(f"    Gap: {gap}")

    # Check artifacts
    print("\nKey Artifacts:")
    artifacts_exist = 0
    artifacts_total = 0

    artifact_checks = [
        ("Final report", final_report_path),
        ("Summary", run_dir / "summary.md"),
        ("Stage manifest", run_dir / "stage_manifest.jsonl"),
    ]

    if report.final_checkpoint_registry_path:
        artifact_checks.append(
            ("Checkpoint registry", Path(report.final_checkpoint_registry_path))
        )

    if report.final_replay_buffer_path:
        artifact_checks.append(("Replay buffer", Path(report.final_replay_buffer_path)))

    for name, path in artifact_checks:
        artifacts_total += 1
        if path.exists():
            artifacts_exist += 1
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} (missing)")

    print(f"\n  {artifacts_exist}/{artifacts_total} artifacts present")

    # Print warnings/errors summary
    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for warning in report.warnings[:5]:
            print(f"  - {warning}")
        if len(report.warnings) > 5:
            print(f"  ... and {len(report.warnings) - 5} more")

    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for error in report.errors[:5]:
            print(f"  - {error}")
        if len(report.errors) > 5:
            print(f"  ... and {len(report.errors) - 5} more")

    print("=" * 80 + "\n")

    # Determine exit code
    if report.status.value == "failed":
        logger.error("Run status: FAILED")
        return 1

    if not report.no_leakage_passed:
        logger.error("No-leakage validation: FAILED")
        return 1

    if artifacts_exist < artifacts_total:
        logger.error(f"Missing artifacts: {artifacts_total - artifacts_exist}")
        if strict:
            return 1

    if report.errors:
        logger.error(f"Run completed with {len(report.errors)} errors")
        if strict:
            return 1

    if report.status.value == "partial":
        logger.warning("Run status: PARTIAL")
        return 2 if strict else 0

    if report.warnings:
        logger.warning(f"Run completed with {len(report.warnings)} warnings")
        return 2 if strict else 0

    logger.info("Run inspection: PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Phase 3 E2E GRPO run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to E2E run directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero on any warning or missing artifact",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return 1

    return inspect_phase3_e2e_run(run_dir, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
