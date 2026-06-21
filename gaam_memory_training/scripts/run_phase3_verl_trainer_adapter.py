#!/usr/bin/env python3
"""
CLI for Phase 3 Milestone 4: verl Trainer Adapter

This script converts local DataProto artifacts (from Milestone 2/3)
to trainer-ready batches with logprobs, returns, and optional verl conversion.

Usage:
    python scripts/run_phase3_verl_trainer_adapter.py \
      --dataproto_dir outputs/phase3_ray_prototype/jobs/e47becba/dataproto \
      --output_dir outputs/phase3_verl_trainer_adapter \
      --actor_role memory_builder \
      --mode dry_run \
      --allow_stub_logprobs
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 3 verl trainer adapter"
    )

    # Input/output
    parser.add_argument(
        "--dataproto_dir",
        type=Path,
        required=True,
        help="Input directory with LocalDataProto artifacts",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for trainer artifacts",
    )

    # Actor selection
    parser.add_argument(
        "--actor_role",
        type=str,
        required=True,
        choices=["memory_builder", "question_agent", "all"],
        help="Actor role to process (or 'all' for both)",
    )

    # Execution mode
    parser.add_argument(
        "--mode",
        type=str,
        default="dry_run",
        choices=["dry_run", "local_model", "vendored_verl"],
        help="Adapter mode",
    )

    # Logprob handling
    parser.add_argument(
        "--allow_stub_logprobs",
        action="store_true",
        default=False,
        help="Allow stub logprobs in dry-run mode",
    )
    parser.add_argument(
        "--disallow_stub_logprobs",
        action="store_true",
        help="Fail if real model logprobs unavailable",
    )

    # Returns strategy
    parser.add_argument(
        "--returns_strategy",
        type=str,
        default="advantages_as_returns",
        choices=["advantages_as_returns", "rewards_plus_advantages"],
        help="Returns computation strategy",
    )

    # Checkpoint registry
    parser.add_argument(
        "--checkpoint_registry",
        type=Path,
        help="Checkpoint registry path",
    )

    # Model paths (for local_model mode)
    parser.add_argument(
        "--memory_model_path",
        type=str,
        help="Memory Builder model path",
    )
    parser.add_argument(
        "--question_model_path",
        type=str,
        help="Question Agent model path",
    )
    parser.add_argument(
        "--reference_model_path",
        type=str,
        help="Reference model path",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Tokenizer path",
    )

    # verl conversion
    parser.add_argument(
        "--write_verl_dataproto",
        action="store_true",
        help="Write vendored verl DataProto",
    )

    # Validation
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any error",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )

    args = parser.parse_args()

    # Delay project imports until after argparse has handled --help. Importing
    # the trainer adapter imports torch, which can initialize OpenMP.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

    from gaam_graph.grpo_schema import ActorRole
    from gaam_graph.verl_trainer_adapter import (
        VerlTrainerAdapterMode,
        run_verl_trainer_adapter_step,
    )

    # Validate inputs
    if not args.dataproto_dir.exists():
        print(f"Error: DataProto directory not found: {args.dataproto_dir}", file=sys.stderr)
        return 1

    # Check output directory
    if args.output_dir.exists() and not args.overwrite:
        print(f"Error: Output directory exists: {args.output_dir}", file=sys.stderr)
        print("Use --overwrite to proceed", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Handle allow_stub_logprobs vs disallow_stub_logprobs
    if args.disallow_stub_logprobs:
        allow_stub_logprobs = False
    else:
        allow_stub_logprobs = True

    # If no explicit flag, default to True for dry_run, False for local_model
    if not args.allow_stub_logprobs and not args.disallow_stub_logprobs:
        if args.mode == "dry_run":
            allow_stub_logprobs = True
        elif args.mode == "local_model":
            # For Milestone 4 MVP, local_model still requires stub logprobs
            allow_stub_logprobs = True

    # Parse mode
    mode = VerlTrainerAdapterMode(args.mode)

    # Determine which actors to process
    if args.actor_role == "all":
        actor_roles = [ActorRole.MEMORY_BUILDER, ActorRole.QUESTION_AGENT]
    else:
        actor_roles = [ActorRole(args.actor_role)]

    print("=" * 80)
    print("Phase 3 Milestone 4: verl Trainer Adapter")
    print("=" * 80)
    print(f"DataProto dir: {args.dataproto_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mode: {mode.value}")
    print(f"Actor roles: {[r.value for r in actor_roles]}")
    print(f"Allow stub logprobs: {allow_stub_logprobs}")
    print(f"Returns strategy: {args.returns_strategy}")
    if args.write_verl_dataproto:
        print("verl DataProto conversion: enabled")
    print("=" * 80)
    print()

    # Run adapter for each actor
    reports = []
    manifest_entries: list[dict] = []

    for actor_role in actor_roles:
        print(f"Processing {actor_role.value}...")

        # Create actor-specific output directory
        actor_output_dir = args.output_dir / actor_role.value
        actor_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            report = run_verl_trainer_adapter_step(
                dataproto_dir=args.dataproto_dir,
                output_dir=actor_output_dir,
                actor_role=actor_role,
                mode=mode,
                checkpoint_registry_path=args.checkpoint_registry,
                allow_stub_logprobs=allow_stub_logprobs,
                returns_strategy=args.returns_strategy,
                write_verl_dataproto=args.write_verl_dataproto,
            )

            reports.append(report)

            # Add to manifest
            manifest_entries.append(
                {
                    "actor_role": actor_role.value,
                    "status": report.status,
                    "report_path": f"{actor_role.value}/trainer_step_report.json",
                    "trainer_ready_batch_path": report.trainer_ready_batch_path,
                    "metrics": report.metrics,
                }
            )

            # Print result
            status_label = (
                "[OK]"
                if report.status == "succeeded"
                else "[PARTIAL]"
                if report.status == "partial"
                else "[SKIPPED]"
                if report.status == "skipped"
                else "[FAILED]"
            )
            print(f"  {actor_role.value}: {status_label} {report.status}")

            if report.warnings:
                for warning in report.warnings:
                    print(f"    warning: {warning}")

            if report.errors:
                for error in report.errors:
                    print(f"    error: {error}")

        except Exception as e:
            print(f"  {actor_role.value}: [FAILED] failed - {e}")
            if args.strict:
                return 1
            continue

    print()

    # Write manifest
    manifest_path = args.output_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    # Write summary
    succeeded = sum(1 for r in reports if r.status == "succeeded")
    partial = sum(1 for r in reports if r.status == "partial")
    failed = sum(1 for r in reports if r.status == "failed")
    skipped = sum(1 for r in reports if r.status == "skipped")

    summary = {
        "mode": mode.value,
        "dataproto_dir": str(args.dataproto_dir),
        "output_dir": str(args.output_dir),
        "actor_roles": [r.value for r in actor_roles],
        "num_actors": len(actor_roles),
        "succeeded": succeeded,
        "partial": partial,
        "failed": failed,
        "skipped": skipped,
        "allow_stub_logprobs": allow_stub_logprobs,
        "returns_strategy": args.returns_strategy,
        "write_verl_dataproto": args.write_verl_dataproto,
    }

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write adapter report
    adapter_report = {
        "reports": [r.model_dump(mode="json") for r in reports],
        "summary": summary,
    }

    adapter_report_path = args.output_dir / "adapter_report.json"
    with open(adapter_report_path, "w", encoding="utf-8") as f:
        json.dump(adapter_report, f, indent=2)

    print("Wrote:")
    print(f"  manifest.jsonl")
    print(f"  summary.json")
    print(f"  adapter_report.json")
    for actor_role in actor_roles:
        actor_dir = actor_role.value
        print(f"  {actor_dir}/trainer_step_report.json")
        if (args.output_dir / actor_dir / "trainer_ready_batch.json").exists():
            print(f"  {actor_dir}/trainer_ready_batch.json")
            print(f"  {actor_dir}/trainer_tensor_payload.pt")
        if args.write_verl_dataproto and (args.output_dir / actor_dir / "verl_dataproto.pt").exists():
            print(f"  {actor_dir}/verl_dataproto.pt")
    print()

    # Check for failures in strict mode
    if args.strict and (failed > 0):
        print(f"Error: {failed} actors failed (strict mode)", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    exit(main())
