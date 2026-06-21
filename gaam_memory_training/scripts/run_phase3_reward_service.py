#!/usr/bin/env python3
"""
Phase 3 Milestone 5: Distributed Reward Service CLI

Run distributed reward service for Phase 3 rollout jobs.

Usage:
    # Single job from directory
    python scripts/run_phase3_reward_service.py \
      --job_dir outputs/phase3_ray_prototype/jobs/e47becba \
      --oracle_graph_dir outputs/longmemeval_s_graph \
      --output_dir outputs/phase3_reward_service \
      --record_id e47becba \
      --backend local_fallback

    # Explicit artifact paths
    python scripts/run_phase3_reward_service.py \
      --oracle_graph outputs/longmemeval_s_graph/e47becba.graph.json \
      --current_memory outputs/current_memory_test/e47becba.current_memory.json \
      --questions outputs/question_sets_smoke/e47becba.accepted_questions.json \
      --answers outputs/answers_smoke/e47becba.answers.json \
      --output_dir outputs/phase3_reward_service \
      --record_id e47becba

    # Batch jobs
    python scripts/run_phase3_reward_service.py \
      --jobs_dir outputs/phase3_ray_prototype/jobs \
      --oracle_graph_dir outputs/longmemeval_s_graph \
      --output_dir outputs/phase3_reward_service \
      --max_records 10
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.distributed_reward_schema import (
    DistributedRewardSummary,
    ReplayBufferManifestEntry,
)
from gaam_graph.distributed_reward_service import (
    RewardWorkerInput,
    build_reward_worker_input_from_job_dir,
    run_reward_worker_job,
    run_reward_worker_job_local,
    run_reward_worker_job_ray,
)
from gaam_graph.replay_buffer import ReplayBufferWriter


def run_single_job(
    *,
    input_data: RewardWorkerInput,
    backend: str,
) -> None:
    """Run reward service for a single job."""
    print(f"Running reward worker for {input_data.record_id}...")

    if backend == "local_fallback":
        report = run_reward_worker_job_local(input_data)
    elif backend == "ray":
        report = run_reward_worker_job_ray(input_data)
    else:  # auto
        report = run_reward_worker_job(input_data)

    print(f"  Status: {report.status}")
    if report.memory_update_reward is not None:
        print(f"  Memory reward: {report.memory_update_reward:.3f}")
    if report.question_agent_reward is not None:
        print(f"  Question reward: {report.question_agent_reward:.3f}")
    if report.weakness_update_count > 0:
        print(f"  Weakness updates: {report.weakness_update_count}")

    if report.warnings:
        for warning in report.warnings:
            print(f"  Warning: {warning}")

    if report.errors:
        for error in report.errors:
            print(f"  Error: {error}")


def run_batch_jobs(
    *,
    jobs_dir: Path,
    oracle_graph_dir: Path,
    output_dir: Path,
    max_records: int | None,
    backend: str,
    reward_mode: str,
    no_question_agent_reward: bool,
    no_weakness_update: bool,
    no_replay_item: bool,
    strict: bool,
) -> DistributedRewardSummary:
    """Run reward service for batch of jobs."""
    started_at = datetime.utcnow().isoformat()
    summary_id = f"reward_service_{started_at}"

    # Discover job directories
    job_dirs = [d for d in jobs_dir.iterdir() if d.is_dir()]

    if max_records:
        job_dirs = job_dirs[:max_records]

    print(f"Processing {len(job_dirs)} jobs...")

    succeeded = 0
    partial = 0
    skipped = 0
    failed = 0
    total_weakness_updates = 0
    total_replay_items = 0
    warnings: list[str] = []
    errors: list[str] = []

    # Create replay buffer writer
    replay_buffer_path = output_dir / "replay_buffer.jsonl"
    replay_writer = ReplayBufferWriter(replay_buffer_path)

    manifest_entries: list[ReplayBufferManifestEntry] = []

    for job_dir in job_dirs:
        record_id = job_dir.name

        try:
            # Build input from job directory
            job_output_dir = output_dir / "jobs" / record_id
            input_data = build_reward_worker_input_from_job_dir(
                job_dir=job_dir,
                oracle_graph_dir=oracle_graph_dir,
                output_dir=job_output_dir,
                record_id=record_id,
                reward_mode=reward_mode,
                compute_memory_builder_reward=True,
                compute_question_agent_reward=not no_question_agent_reward,
                update_weakness_book=not no_weakness_update,
                write_replay_item=not no_replay_item,
            )

            # Run reward worker
            if backend == "local_fallback":
                report = run_reward_worker_job_local(input_data)
            elif backend == "ray":
                report = run_reward_worker_job_ray(input_data)
            else:  # auto
                report = run_reward_worker_job(input_data)

            # Update counts
            if report.status == "succeeded":
                succeeded += 1
            elif report.status == "partial":
                partial += 1
            elif report.status == "skipped":
                skipped += 1
            elif report.status == "failed":
                failed += 1

            total_weakness_updates += report.weakness_update_count

            if report.replay_item_path:
                total_replay_items += 1

                # Load and append replay item
                with open(report.replay_item_path, "r", encoding="utf-8") as f:
                    from gaam_graph.distributed_reward_schema import ReplayBufferItem
                    item_data = json.load(f)
                    item = ReplayBufferItem.model_validate(item_data)
                    replay_writer.append(item)

                    # Add manifest entry
                    manifest_entries.append(
                        ReplayBufferManifestEntry(
                            replay_id=item.replay_id,
                            record_id=item.record_id,
                            job_id=item.job_id,
                            round_id=item.round_id,
                            step_id=item.step_id,
                            status=report.status,
                            memory_reward=item.memory_reward,
                            question_reward=item.question_reward,
                            selected_for_training=item.selected_for_training,
                            created_at=item.created_at,
                        )
                    )

            warnings.extend(report.warnings)
            errors.extend(report.errors)

            print(f"  {record_id}: {report.status}")

        except Exception as e:
            failed += 1
            error_msg = f"{record_id}: {type(e).__name__}: {e}"
            errors.append(error_msg)
            print(f"  {error_msg}")

            if strict:
                raise

    # Write manifest
    replay_writer.write_manifest(manifest_entries)

    # Build summary
    finished_at = datetime.utcnow().isoformat()

    summary = DistributedRewardSummary(
        summary_id=summary_id,
        backend=backend,
        reward_mode=reward_mode,
        num_jobs=len(job_dirs),
        succeeded=succeeded,
        partial=partial,
        skipped=skipped,
        failed=failed,
        total_weakness_updates=total_weakness_updates,
        total_replay_items=total_replay_items,
        warnings=warnings[:100],  # Limit warnings
        errors=errors[:100],  # Limit errors
        started_at=started_at,
        finished_at=finished_at,
    )

    # Write summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.model_dump(mode="json"), f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 3 distributed reward service"
    )

    # Input mode 1: Job directory
    parser.add_argument(
        "--job_dir",
        type=Path,
        help="Single Phase 3 job directory",
    )

    # Input mode 2: Batch jobs
    parser.add_argument(
        "--jobs_dir",
        type=Path,
        help="Directory containing multiple job directories",
    )

    # Input mode 3: Explicit artifacts
    parser.add_argument(
        "--oracle_graph",
        type=Path,
        help="Path to oracle graph JSON",
    )
    parser.add_argument(
        "--current_memory",
        type=Path,
        help="Path to current memory JSON",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        help="Path to accepted questions JSON",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        help="Path to answer report JSON",
    )

    # Common inputs
    parser.add_argument(
        "--oracle_graph_dir",
        type=Path,
        help="Directory containing oracle graphs (for job_dir/jobs_dir modes)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for reward service results",
    )
    parser.add_argument(
        "--record_id",
        type=str,
        help="Record ID (required for explicit artifact mode)",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        help="Maximum number of records to process (for jobs_dir mode)",
    )

    # Execution options
    parser.add_argument(
        "--backend",
        choices=["local_fallback", "ray", "auto"],
        default="local_fallback",
        help="Execution backend",
    )
    parser.add_argument(
        "--reward_mode",
        choices=["heuristic", "llm_judge"],
        default="heuristic",
        help="Reward computation mode",
    )
    parser.add_argument(
        "--no_question_agent_reward",
        action="store_true",
        help="Skip Question Agent reward computation",
    )
    parser.add_argument(
        "--no_weakness_update",
        action="store_true",
        help="Skip Weakness Book updates",
    )
    parser.add_argument(
        "--no_replay_item",
        action="store_true",
        help="Skip replay item generation",
    )

    # Control flags
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit on first error",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )

    args = parser.parse_args()

    # Validate input modes
    if args.job_dir:
        input_mode = "job_dir"
        if not args.oracle_graph_dir:
            print("Error: --oracle_graph_dir required with --job_dir", file=sys.stderr)
            sys.exit(1)
    elif args.jobs_dir:
        input_mode = "jobs_dir"
        if not args.oracle_graph_dir:
            print("Error: --oracle_graph_dir required with --jobs_dir", file=sys.stderr)
            sys.exit(1)
    elif args.oracle_graph:
        input_mode = "explicit"
        if not args.record_id:
            print("Error: --record_id required with explicit artifact paths", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: Must specify --job_dir, --jobs_dir, or --oracle_graph", file=sys.stderr)
        sys.exit(1)

    # Check output directory
    if args.output_dir.exists() and not args.overwrite:
        print(f"Error: Output directory exists: {args.output_dir}", file=sys.stderr)
        print("Use --overwrite to replace", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run based on input mode
    if input_mode == "job_dir":
        # Single job from directory
        job_output_dir = args.output_dir / "jobs" / args.job_dir.name
        input_data = build_reward_worker_input_from_job_dir(
            job_dir=args.job_dir,
            oracle_graph_dir=args.oracle_graph_dir,
            output_dir=job_output_dir,
            record_id=args.job_dir.name if not args.record_id else args.record_id,
            reward_mode=args.reward_mode,
            compute_memory_builder_reward=True,
            compute_question_agent_reward=not args.no_question_agent_reward,
            update_weakness_book=not args.no_weakness_update,
            write_replay_item=not args.no_replay_item,
        )

        run_single_job(input_data=input_data, backend=args.backend)

        print("Done!")

    elif input_mode == "jobs_dir":
        # Batch jobs
        summary = run_batch_jobs(
            jobs_dir=args.jobs_dir,
            oracle_graph_dir=args.oracle_graph_dir,
            output_dir=args.output_dir,
            max_records=args.max_records,
            backend=args.backend,
            reward_mode=args.reward_mode,
            no_question_agent_reward=args.no_question_agent_reward,
            no_weakness_update=args.no_weakness_update,
            no_replay_item=args.no_replay_item,
            strict=args.strict,
        )

        print(f"\nSummary:")
        print(f"  Total jobs: {summary.num_jobs}")
        print(f"  Succeeded: {summary.succeeded}")
        print(f"  Partial: {summary.partial}")
        print(f"  Skipped: {summary.skipped}")
        print(f"  Failed: {summary.failed}")
        print(f"  Weakness updates: {summary.total_weakness_updates}")
        print(f"  Replay items: {summary.total_replay_items}")

        if summary.failed > 0 and args.strict:
            sys.exit(1)

        print("Done!")

    elif input_mode == "explicit":
        # Explicit artifacts
        job_output_dir = args.output_dir / "jobs" / args.record_id

        input_data = RewardWorkerInput(
            job_id=f"{args.record_id}_round0_step0",
            record_id=args.record_id,
            round_id=0,
            step_id=0,
            rollout_job_dir=None,
            oracle_graph_path=str(args.oracle_graph),
            current_memory_path=str(args.current_memory) if args.current_memory else None,
            question_set_path=str(args.questions) if args.questions else None,
            answer_report_path=str(args.answers) if args.answers else None,
            trainer_adapter_report_path=None,
            weakness_book_path=None,
            output_dir=str(job_output_dir),
            reward_mode=args.reward_mode,
            compute_memory_builder_reward=True,
            compute_question_agent_reward=not args.no_question_agent_reward,
            update_weakness_book=not args.no_weakness_update,
            write_replay_item=not args.no_replay_item,
        )

        run_single_job(input_data=input_data, backend=args.backend)

        print("Done!")


if __name__ == "__main__":
    main()
