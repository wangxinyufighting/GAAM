#!/usr/bin/env python3
"""
CLI for Phase 3 Milestone 1: Inspect Phase 3 runtime readiness.

This script validates that Phase 3 distributed runtime inputs, checkpoints,
and configurations are ready for execution.

Usage examples:

    # Basic readiness check
    python scripts/inspect_phase3_runtime_readiness.py \
      --input_records data/longmemeval/longmemeval_s_cleaned.json \
      --oracle_graph_dir outputs/longmemeval_s_graph \
      --memory_model_path Qwen/Qwen2.5-0.5B \
      --question_model_path Qwen/Qwen2.5-0.5B \
      --output_dir outputs/phase3_runtime_readiness

    # With checkpoint registry
    python scripts/inspect_phase3_runtime_readiness.py \
      --input_records data/longmemeval/longmemeval_s_cleaned.json \
      --oracle_graph_dir outputs/longmemeval_s_graph \
      --checkpoint_registry outputs/phase2_alignment_report/checkpoint_registry.json \
      --memory_model_path Qwen/Qwen2.5-0.5B \
      --question_model_path Qwen/Qwen2.5-0.5B \
      --output_dir outputs/phase3_runtime_readiness \
      --max_records 5

    # Write Code-A1 draft config
    python scripts/inspect_phase3_runtime_readiness.py \
      --input_records data/longmemeval/longmemeval_s_cleaned.json \
      --oracle_graph_dir outputs/longmemeval_s_graph \
      --memory_model_path Qwen/Qwen2.5-0.5B \
      --question_model_path Qwen/Qwen2.5-0.5B \
      --output_dir outputs/phase3_runtime_readiness \
      --write_code_a1_draft_config

    # Strict mode
    python scripts/inspect_phase3_runtime_readiness.py \
      --input_records data/longmemeval/longmemeval_s_cleaned.json \
      --oracle_graph_dir outputs/longmemeval_s_graph \
      --memory_model_path Qwen/Qwen2.5-0.5B \
      --question_model_path Qwen/Qwen2.5-0.5B \
      --output_dir outputs/phase3_runtime_readiness \
      --strict
"""

import argparse
import hashlib
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

from gaam_graph.distributed_runtime_schema import (
    Phase3RuntimeConfig,
    ActorRuntimeConfig,
    Phase3Backend,
)
from gaam_graph.distributed_runtime import LocalFakeDistributedRuntime
from gaam_graph.code_a1_runtime_mapping import draft_code_a1_omegaconf
from gaam_graph.code_a1_alignment import AlignmentIssueSeverity
from gaam_graph.grpo_schema import ActorRole


def write_summary_text(report, output_path: Path):
    """Write human-readable summary to text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Phase 3 Milestone 1: Distributed Runtime Readiness Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Report ID: {report.report_id}\n")
        f.write(f"Runtime ID: {report.runtime_id}\n")
        f.write(f"Status: {report.status}\n")
        f.write(f"Backend: {report.backend.value}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Checkpoint Resolution\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"Memory Builder checkpoint resolved: {report.metrics.get('memory_checkpoint_resolved', False)}\n")
        f.write(f"Question Agent checkpoint resolved: {report.metrics.get('question_checkpoint_resolved', False)}\n")
        f.write(f"Memory Builder reference resolved: {report.metrics.get('memory_reference_resolved', False)}\n")
        f.write(f"Question Agent reference resolved: {report.metrics.get('question_reference_resolved', False)}\n")
        f.write(f"Cold-start actors: {report.metrics.get('cold_start_actor_count', 0)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("Worker Plan\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"Number of jobs: {report.metrics.get('num_jobs', 0)}\n")
        f.write(f"Rollout workers: {report.metrics.get('num_rollout_workers', 0)}\n")
        f.write(f"Reward workers: {report.metrics.get('num_reward_workers', 0)}\n\n")

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
        description="Inspect Phase 3 distributed runtime readiness"
    )

    # Input/output
    parser.add_argument("--input_records", type=Path, required=True, help="Input records JSON file")
    parser.add_argument("--oracle_graph_dir", type=Path, required=True, help="Directory with oracle graphs")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for readiness report")

    # Checkpoint resolution
    parser.add_argument("--checkpoint_registry", type=Path, help="Optional checkpoint registry JSON")
    parser.add_argument("--memory_checkpoint_id", type=str, help="Explicit Memory Builder checkpoint ID")
    parser.add_argument("--question_checkpoint_id", type=str, help="Explicit Question Agent checkpoint ID")

    # Model paths (for cold start)
    parser.add_argument("--memory_model_path", type=str, required=True, help="Memory Builder model path")
    parser.add_argument("--question_model_path", type=str, required=True, help="Question Agent model path")

    # Runtime config
    parser.add_argument("--max_records", type=int, help="Maximum number of records to process")
    parser.add_argument("--num_rollout_workers", type=int, default=2, help="Number of rollout workers")
    parser.add_argument("--num_reward_workers", type=int, default=2, help="Number of reward workers")
    parser.add_argument("--backend", type=str, default="local_fake_distributed",
                        choices=["local_fake_distributed", "ray", "verl", "code_a1_verl"],
                        help="Runtime backend")
    parser.add_argument("--round_id", type=int, default=0, help="GRPO round ID")
    parser.add_argument("--step_id", type=int, default=0, help="GRPO step ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Validation mode
    parser.add_argument("--strict", action="store_true", help="Fail on ERROR issues")

    # Export options
    parser.add_argument("--write_code_a1_draft_config", action="store_true",
                        help="Write Code-A1 draft OmegaConf config")

    args = parser.parse_args()

    # Validate inputs
    if not args.input_records.exists():
        print(f"Error: Input records file does not exist: {args.input_records}", file=sys.stderr)
        return 1

    if not args.oracle_graph_dir.exists():
        print(f"Error: Oracle graph directory does not exist: {args.oracle_graph_dir}", file=sys.stderr)
        return 1

    if args.checkpoint_registry and not args.checkpoint_registry.exists():
        print(f"Error: Checkpoint registry does not exist: {args.checkpoint_registry}", file=sys.stderr)
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 3 Milestone 1: Inspect Distributed Runtime Readiness")
    print("=" * 80)
    print(f"Input records: {args.input_records}")
    print(f"Oracle graph dir: {args.oracle_graph_dir}")
    print(f"Checkpoint registry: {args.checkpoint_registry or '(not provided)'}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max records: {args.max_records or '(all)'}")
    print(f"Rollout workers: {args.num_rollout_workers}")
    print(f"Reward workers: {args.num_reward_workers}")
    print(f"Backend: {args.backend}")
    print(f"Strict mode: {args.strict}")
    print(f"Write Code-A1 draft config: {args.write_code_a1_draft_config}")
    print("=" * 80)
    print()

    # Build runtime config
    runtime_id = hashlib.sha256(str(args.output_dir).encode()).hexdigest()[:8]

    runtime_config = Phase3RuntimeConfig(
        runtime_id=runtime_id,
        backend=Phase3Backend(args.backend),
        input_records_path=args.input_records,
        oracle_graph_dir=args.oracle_graph_dir,
        output_dir=args.output_dir,
        checkpoint_registry_path=args.checkpoint_registry,
        max_records=args.max_records,
        round_id=args.round_id,
        step_id=args.step_id,
        seed=args.seed,
        num_rollout_workers=args.num_rollout_workers,
        num_reward_workers=args.num_reward_workers,
        memory_builder=ActorRuntimeConfig(
            role=ActorRole.MEMORY_BUILDER,
            policy_id="memory_builder_policy",
            model_path=args.memory_model_path,
            checkpoint_id=args.memory_checkpoint_id,
        ),
        question_agent=ActorRuntimeConfig(
            role=ActorRole.QUESTION_AGENT,
            policy_id="question_agent_policy",
            model_path=args.question_model_path,
            checkpoint_id=args.question_checkpoint_id,
        ),
        strict_no_leakage=True,
    )

    # Run local fake-distributed runtime
    print("Building runtime plan and validating readiness...")
    runtime = LocalFakeDistributedRuntime(runtime_config)
    report = runtime.run()

    print(f"✓ Runtime config: {args.output_dir / 'runtime_config.json'}")
    print(f"✓ Worker plan: {args.output_dir / 'worker_plan.json'}")
    print(f"✓ Case assignments: {args.output_dir / 'case_assignments.jsonl'}")
    print(f"✓ Runtime trace: {args.output_dir / 'local_runtime_trace.jsonl'}")
    print(f"✓ Readiness report: {args.output_dir / 'readiness_report.json'}")

    # Write summary text
    summary_path = args.output_dir / "summary.txt"
    write_summary_text(report, summary_path)
    print(f"✓ Summary: {summary_path}")

    # Write Code-A1 draft config if requested
    if args.write_code_a1_draft_config:
        print("\nGenerating Code-A1 draft config...")
        try:
            draft_config = draft_code_a1_omegaconf(runtime_config)
            draft_config_path = args.output_dir / "code_a1_draft_config.json"
            with open(draft_config_path, "w", encoding="utf-8") as f:
                json.dump(draft_config, f, ensure_ascii=False, indent=2)
            print(f"✓ Code-A1 draft config: {draft_config_path}")
        except Exception as e:
            print(f"✗ Failed to generate Code-A1 draft config: {e}", file=sys.stderr)
            if args.strict:
                return 1

    # Print issue summary
    print("\n" + "=" * 80)
    print("Readiness Summary")
    print("=" * 80)
    print(f"Status: {report.status}")
    print(f"Total issues: {report.metrics['total_issues']}")
    print(f"  Errors: {report.metrics['error_count']}")
    print(f"  Warnings: {report.metrics['warning_count']}")
    print(f"  Info: {report.metrics['info_count']}")

    # Print checkpoint resolution
    print("\nCheckpoint Resolution:")
    print(f"  Memory Builder: {'✓' if report.metrics.get('memory_checkpoint_resolved') else '✗ (cold start)'}")
    print(f"  Question Agent: {'✓' if report.metrics.get('question_checkpoint_resolved') else '✗ (cold start)'}")

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
        print("\n✗ Readiness check FAILED (strict mode, errors found)")
        return 1
    else:
        print("\n✓ Readiness check complete")
        return 0


if __name__ == "__main__":
    exit(main())
