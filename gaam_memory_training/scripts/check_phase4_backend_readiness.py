#!/usr/bin/env python3
"""
Check Phase 4 backend readiness.

Usage:
    python scripts/check_phase4_backend_readiness.py \
      --trainer_backend verl \
      --memory_builder_model_path /path/to/model \
      --require_cuda

Exit codes:
    0 = ready
    1 = not ready
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.phase4_backend_readiness import check_phase4_backend_readiness
from gaam_graph.phase4_trainer_schema import Phase4TrainerBackend


def main():
    parser = argparse.ArgumentParser(description="Check Phase 4 backend readiness")

    parser.add_argument(
        "--trainer_backend",
        type=str,
        default="dry_run",
        choices=["dry_run", "local_grpo", "verl"],
        help="Trainer backend to check",
    )

    parser.add_argument(
        "--memory_builder_model_path",
        type=str,
        help="Memory Builder model path",
    )

    parser.add_argument(
        "--question_agent_model_path",
        type=str,
        help="Question Agent model path",
    )

    parser.add_argument(
        "--answerer_model_path",
        type=str,
        help="Answerer model path",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to check writability",
    )

    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="Require CUDA availability",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format",
    )

    args = parser.parse_args()

    # Parse backend
    backend = Phase4TrainerBackend(args.trainer_backend)

    # Run readiness check
    report = check_phase4_backend_readiness(
        trainer_backend=backend,
        memory_builder_model_path=args.memory_builder_model_path,
        question_agent_model_path=args.question_agent_model_path,
        answerer_model_path=args.answerer_model_path,
        output_dir=args.output_dir,
        require_cuda=args.require_cuda,
    )

    # Output results
    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
    else:
        print(f"\n=== Backend Readiness Check: {backend.value} ===\n")
        print(f"Python version: {report.python_version}")
        print(f"PyTorch available: {report.torch_available}")
        print(f"CUDA available: {report.cuda_available}")
        print(f"Transformers available: {report.transformers_available}")
        print(f"VERL available: {report.verl_available}")
        print(f"Model paths valid: {report.model_paths_valid}")
        print(f"Output dir writable: {report.output_dir_writable}")

        if report.warnings:
            print(f"\nWarnings ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"  - {warning}")

        if report.issues:
            print(f"\nIssues ({len(report.issues)}):")
            for issue in report.issues:
                print(f"  - {issue}")

        print(f"\nReady: {report.ready}")

    # Exit code
    sys.exit(0 if report.ready else 1)


if __name__ == "__main__":
    main()
