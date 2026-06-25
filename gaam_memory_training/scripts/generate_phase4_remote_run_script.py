#!/usr/bin/env python3
"""
Generate Phase 4 Remote Run Script

Generate bash script for remote GPU suite execution.
"""

import argparse
import sys
from pathlib import Path

from gaam_graph.phase4_suite_config import load_suite_config
from gaam_graph.phase4_remote_scripts import (
    generate_remote_run_script,
    generate_slurm_script,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 4 remote run script"
    )

    parser.add_argument(
        "--suite_config",
        type=str,
        required=True,
        help="Path to suite config file",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for generated script",
    )

    parser.add_argument(
        "--conda_env",
        type=str,
        help="Conda environment name",
    )

    parser.add_argument(
        "--project_dir",
        type=str,
        help="Project directory path",
    )

    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Generate SLURM batch script",
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="gpu",
        help="SLURM partition (default: gpu)",
    )

    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=1,
        help="GPUs per node (default: 1)",
    )

    parser.add_argument(
        "--time_limit",
        type=str,
        default="24:00:00",
        help="SLURM time limit (default: 24:00:00)",
    )

    args = parser.parse_args()

    try:
        # Load suite config
        config_path = Path(args.suite_config)
        if not config_path.exists():
            print(f"Suite config not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        suite_config = load_suite_config(config_path)

        output_path = Path(args.output)

        if args.slurm:
            # Generate SLURM script
            generate_slurm_script(
                suite_config,
                job_name=suite_config.get("suite_name"),
                partition=args.partition,
                gpus_per_node=args.gpus_per_node,
                time_limit=args.time_limit,
                conda_env=args.conda_env,
                project_dir=args.project_dir,
                output_path=output_path,
            )
            print(f"Generated SLURM script: {output_path}")
        else:
            # Generate bash script
            generate_remote_run_script(
                suite_config,
                suite_config_path=str(config_path),
                conda_env=args.conda_env,
                project_dir=args.project_dir,
                output_path=output_path,
            )
            print(f"Generated remote run script: {output_path}")

        print()
        print("To execute:")
        print(f"  bash {output_path}")

        sys.exit(0)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
