#!/usr/bin/env python3
"""
Run Phase 4 Benchmark Suite

Execute multiple Phase 4 production experiments as a benchmark suite.
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.phase4_suite_config import (
    load_and_resolve_suite_config,
    write_resolved_config,
)
from gaam_graph.phase4_suite_runner import run_phase4_benchmark_suite


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 4 benchmark suite with multiple seeds"
    )

    parser.add_argument(
        "--suite_config",
        type=str,
        required=True,
        help="Path to suite config file (JSON or YAML)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Override seeds",
    )

    parser.add_argument(
        "--trainer_backend",
        type=str,
        choices=["dry_run", "local_grpo", "verl"],
        help="Override trainer backend",
    )

    parser.add_argument(
        "--memory_builder_model_path",
        type=str,
        help="Override Memory Builder model path",
    )

    parser.add_argument(
        "--question_agent_model_path",
        type=str,
        help="Override Question Agent model path",
    )

    parser.add_argument(
        "--answerer_model_path",
        type=str,
        help="Override Answerer model path",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing suite manifest",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing suite",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only",
    )

    args = parser.parse_args()

    # Build CLI overrides
    cli_overrides = {}
    if args.output_dir:
        cli_overrides["output_dir"] = args.output_dir
    if args.seeds:
        cli_overrides["seeds"] = args.seeds
    if args.trainer_backend:
        cli_overrides["trainer_backend"] = args.trainer_backend
    if args.memory_builder_model_path:
        cli_overrides["memory_builder_model_path"] = args.memory_builder_model_path
    if args.question_agent_model_path:
        cli_overrides["question_agent_model_path"] = args.question_agent_model_path
    if args.answerer_model_path:
        cli_overrides["answerer_model_path"] = args.answerer_model_path
    if args.resume:
        cli_overrides["resume"] = True
    if args.overwrite:
        cli_overrides["overwrite"] = True

    try:
        # Load and resolve config
        config = load_and_resolve_suite_config(
            config_path=Path(args.suite_config),
            cli_overrides=cli_overrides,
        )

        # Validate config
        issues = config.validate_config()
        if issues:
            if not args.json:
                print("Configuration validation failed:", file=sys.stderr)
                for issue in issues:
                    print(f"  - {issue}", file=sys.stderr)
            else:
                print(json.dumps({"status": "validation_error", "issues": issues}))
            sys.exit(3)

        # Write resolved config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path = output_dir / "suite_config.resolved.json"
        write_resolved_config(config, resolved_config_path)

        if not args.json:
            print(f"Running benchmark suite: {config.suite_name}")
            print(f"Output directory: {config.output_dir}")
            print(f"Seeds: {config.seeds}")
            print()

        # Run suite
        manifest = run_phase4_benchmark_suite(config)

        if args.json:
            print(json.dumps(manifest.model_dump(), indent=2))
        else:
            print()
            print("=" * 80)
            print("Benchmark Suite Complete")
            print("=" * 80)
            print(f"Suite ID: {manifest.suite_id}")
            print(f"Status: {manifest.status}")
            print(f"Seeds requested: {len(config.seeds)}")
            print(f"Seeds succeeded: {manifest.metrics.get('num_seeds_succeeded', 0)}")
            print(f"Seeds partial: {manifest.metrics.get('num_seeds_partial', 0)}")
            print(f"Seeds failed: {manifest.metrics.get('num_seeds_failed', 0)}")
            print()
            print(f"Manifest: {output_dir / 'suite_manifest.json'}")

        # Exit code based on status
        if manifest.status == "succeeded":
            sys.exit(0)
        elif manifest.status == "partial":
            sys.exit(2)
        else:
            sys.exit(1)

    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "error", "error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
