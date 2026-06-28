#!/usr/bin/env python3
"""Export a compact GAAM production evidence bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.production_artifact_bundle import (  # noqa: E402
    ProductionArtifactBundleConfig,
    export_production_artifact_bundle,
)


SNAPSHOT_ENV_KEYS = [
    "MEMORY_MODEL_PATH",
    "QUESTION_MODEL_PATH",
    "CODE_A1_ROOT",
    "ROUNDS",
    "ORDER",
    "MEMORY_NUM_GPUS",
    "QUESTION_NUM_GPUS",
    "ROLLOUT_N",
    "TRAIN_BATCH_SIZE",
    "QUESTIONS_PER_CASE",
    "EVALUATION_SPLIT",
    "MEMORY_BACKEND",
    "ANSWER_BACKEND",
    "JUDGE_BACKEND",
    "GAAM_REWARD_JUDGE_ENABLED",
    "GAAM_REWARD_JUDGE_BASE_URL",
    "GAAM_REWARD_JUDGE_MODEL",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy compact production evidence artifacts into one bundle. "
            "Large checkpoint directories are referenced by manifest paths, not copied."
        )
    )
    parser.add_argument("--training_output_dir", type=Path, required=True)
    parser.add_argument("--evaluation_output_dir", type=Path, default=None)
    parser.add_argument("--split_manifest", type=Path, default=None)
    parser.add_argument("--bundle_output_dir", type=Path, required=True)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--extra_file", type=Path, action="append", default=[])
    parser.add_argument("--no_tar", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    env_snapshot = {key: os.getenv(key, "") for key in SNAPSHOT_ENV_KEYS}
    report = export_production_artifact_bundle(
        ProductionArtifactBundleConfig(
            training_output_dir=args.training_output_dir,
            evaluation_output_dir=args.evaluation_output_dir,
            split_manifest_path=args.split_manifest,
            bundle_output_dir=args.bundle_output_dir,
            extra_files=args.extra_file,
            create_tar=not args.no_tar,
            run_id=args.run_id,
            env_snapshot=env_snapshot,
        )
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("\n=== GAAM Production Artifact Bundle ===")
        print(f"Status: {report['status']}")
        print(f"Bundle dir: {report['bundle_output_dir']}")
        print(f"Files copied: {len(report['files'])}")
        if report.get("tar_path"):
            print(f"Tarball: {report['tar_path']}")
        if report.get("missing"):
            print("\nMissing:")
            for item in report["missing"]:
                print(f"  - {item}")

    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
