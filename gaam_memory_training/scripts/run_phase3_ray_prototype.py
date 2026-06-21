#!/usr/bin/env python3
"""
CLI for Phase 3 Milestone 3: Ray Worker Prototype

This script runs distributed rollout jobs using Ray workers (if available)
or a local fallback executor.

Usage:
    python scripts/run_phase3_ray_prototype.py \
      --input_records data/longmemeval/longmemeval_s_cleaned.json \
      --oracle_graph_dir outputs/longmemeval_s_graph \
      --output_dir outputs/phase3_ray_prototype \
      --record_id e47becba \
      --max_records 1 \
      --num_rollout_workers 2 \
      --num_reward_workers 1 \
      --no_llm
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.distributed_runtime import Phase3RuntimePlanner
from gaam_graph.distributed_runtime_schema import (
    ActorRuntimeConfig,
    Phase3Backend,
    Phase3RuntimeConfig,
)
from gaam_graph.grpo_schema import ActorRole
from gaam_graph.ray_runtime import (
    RayRuntimeConfig,
    RayRuntimeExecutor,
    is_ray_available,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 3 Ray worker prototype"
    )

    # Input sources
    parser.add_argument(
        "--input_records",
        type=Path,
        required=True,
        help="LongMemEval records JSON",
    )
    parser.add_argument(
        "--oracle_graph_dir",
        type=Path,
        required=True,
        help="Oracle graph directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory",
    )

    # Record filtering
    parser.add_argument(
        "--record_id",
        type=str,
        help="Single record ID to process",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        help="Maximum number of records",
    )

    # Worker configuration
    parser.add_argument(
        "--num_rollout_workers",
        type=int,
        default=2,
        help="Number of rollout workers",
    )
    parser.add_argument(
        "--num_reward_workers",
        type=int,
        default=1,
        help="Number of reward workers",
    )

    # Execution mode
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "ray", "local_fallback"],
        default="auto",
        help="Execution backend",
    )
    parser.add_argument(
        "--require_ray",
        action="store_true",
        help="Fail if Ray is unavailable",
    )
    parser.add_argument(
        "--ray_address",
        type=str,
        help="Ray cluster address",
    )
    parser.add_argument(
        "--ray_local_mode",
        action="store_true",
        help="Run Ray in local mode",
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        help="Number of CPUs for Ray",
    )
    parser.add_argument(
        "--num_gpus",
        type=float,
        help="Number of GPUs for Ray",
    )
    parser.add_argument(
        "--job_timeout_seconds",
        type=int,
        help="Job timeout in seconds",
    )

    # Policy configuration
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Dry-run without LLM calls",
    )

    # Tokenizer options
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="HuggingFace tokenizer path",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        help="Maximum prompt tokens",
    )
    parser.add_argument(
        "--max_response_length",
        type=int,
        help="Maximum response tokens",
    )
    parser.add_argument(
        "--truncation",
        type=str,
        default="error",
        choices=["error", "left", "right"],
        help="Truncation strategy",
    )

    # Export options
    parser.add_argument(
        "--write_torch_tensors",
        action="store_true",
        help="Write PyTorch tensor files",
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

    # Validate inputs
    if not args.input_records.exists():
        print(f"Error: Input records not found: {args.input_records}", file=sys.stderr)
        return 1

    if not args.oracle_graph_dir.exists():
        print(f"Error: Oracle graph directory not found: {args.oracle_graph_dir}", file=sys.stderr)
        return 1

    # Check output directory
    if args.output_dir.exists() and not args.overwrite:
        print(f"Error: Output directory exists: {args.output_dir}", file=sys.stderr)
        print("Use --overwrite to proceed", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine backend
    backend = args.backend
    if backend == "auto":
        backend = "ray" if is_ray_available() else "local_fallback"
    elif backend == "ray" and not is_ray_available():
        if args.require_ray:
            print("Error: Ray is required but not available", file=sys.stderr)
            print("Install with: pip install ray>=2.9", file=sys.stderr)
            return 1
        else:
            backend = "local_fallback"

    # Explicitly set require_ray=False for local_fallback
    if backend == "local_fallback":
        args.require_ray = False

    # Build runtime config
    runtime_config = Phase3RuntimeConfig(
        runtime_id=f"phase3_ray_{args.output_dir.name}",
        backend=Phase3Backend.RAY if backend == "ray" else Phase3Backend.LOCAL_FAKE_DISTRIBUTED,
        input_records_path=args.input_records,
        oracle_graph_dir=args.oracle_graph_dir,
        output_dir=args.output_dir,
        checkpoint_registry_path=None,
        memory_builder=ActorRuntimeConfig(
            role=ActorRole.MEMORY_BUILDER,
            policy_id="memory_builder_policy",
            model_path="placeholder",
            checkpoint_id=None,
        ),
        question_agent=ActorRuntimeConfig(
            role=ActorRole.QUESTION_AGENT,
            policy_id="question_agent_policy",
            model_path="placeholder",
            checkpoint_id=None,
        ),
        num_rollout_workers=args.num_rollout_workers,
        num_reward_workers=args.num_reward_workers,
        record_id_filter=args.record_id,
        max_records=args.max_records,
    )

    # Write runtime config
    config_path = args.output_dir / "runtime_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(runtime_config.model_dump(mode="json"), f, indent=2)

    # Build worker plan
    print("=" * 80)
    print("Phase 3 Milestone 3: Ray Worker Prototype")
    print("=" * 80)
    print(f"Runtime ID: {runtime_config.runtime_id}")
    print(f"Backend: {backend}")
    if backend == "ray":
        print(f"Ray available: {is_ray_available()}")
        if args.ray_address:
            print(f"Ray address: {args.ray_address}")
        else:
            print("Ray address: local")
    print(f"Records: {args.max_records or 'all'}")
    if args.record_id:
        print(f"Record ID filter: {args.record_id}")
    print(f"Rollout workers: {args.num_rollout_workers}")
    print(f"Reward workers: {args.num_reward_workers}")
    print("=" * 80)
    print()

    print("Planning jobs...")
    planner = Phase3RuntimePlanner(runtime_config)
    worker_plan = planner.build_worker_plan()

    # Write worker plan
    plan_path = args.output_dir / "worker_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(worker_plan.model_dump(mode="json"), f, indent=2)

    print(f"Planned {len(worker_plan.jobs)} jobs")
    print()

    # Build Ray config
    ray_config = RayRuntimeConfig(
        require_ray=args.require_ray,
        ray_address=args.ray_address,
        local_mode=args.ray_local_mode,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        job_timeout_seconds=args.job_timeout_seconds,
    )

    # Execute jobs
    print("Dispatching jobs...")
    executor = RayRuntimeExecutor(
        config=runtime_config,
        worker_plan=worker_plan,
        ray_config=ray_config,
        tokenizer_path=args.tokenizer_path,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        truncation=args.truncation,
        write_torch_tensors=args.write_torch_tensors,
    )

    results = executor.run()

    # Print results
    for result in results:
        status_symbol = "✓" if result.status.value == "succeeded" else "✗"
        print(f"  {result.record_id} -> {result.worker_id}: {status_symbol} {result.status.value}")

    print()
    print("Wrote:")
    print(f"  runtime_config.json")
    print(f"  worker_plan.json")
    print(f"  ray_runtime_manifest.jsonl")
    print(f"  ray_runtime_summary.json")
    print(f"  ray_runtime_trace.jsonl")
    print()

    # Check for failures
    failed = [r for r in results if r.status.value == "failed"]
    if failed and args.strict:
        print(f"Error: {len(failed)} jobs failed (strict mode)", file=sys.stderr)
        for result in failed:
            print(f"  {result.record_id}: {result.error}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    exit(main())
