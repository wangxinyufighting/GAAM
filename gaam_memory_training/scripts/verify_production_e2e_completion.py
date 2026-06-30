#!/usr/bin/env python3
"""Verify final GAAM production E2E completion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.production_completion_verifier import (  # noqa: E402
    ProductionCompletionVerificationConfig,
    verify_production_e2e_completion,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the final evidence for a GAAM production E2E run: native VERL "
            "training, held-out evaluation, production audit, and artifact bundle."
        )
    )
    parser.add_argument("--training_output_dir", type=Path, required=True)
    parser.add_argument("--evaluation_output_dir", type=Path, required=True)
    parser.add_argument("--artifact_bundle_dir", type=Path, required=True)
    parser.add_argument("--expected_evaluation_split", default="test")
    parser.add_argument("--split_manifest", type=Path, default=None)
    parser.add_argument("--report_output_dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = verify_production_e2e_completion(
        ProductionCompletionVerificationConfig(
            training_output_dir=args.training_output_dir,
            evaluation_output_dir=args.evaluation_output_dir,
            artifact_bundle_dir=args.artifact_bundle_dir,
            expected_evaluation_split=args.expected_evaluation_split,
            split_manifest_path=args.split_manifest,
            report_output_dir=args.report_output_dir,
        )
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("\n=== GAAM Production E2E Completion Verification ===")
        print(f"Status: {report['status']}")
        print(f"Training output: {report['training_output_dir']}")
        print(f"Evaluation output: {report['evaluation_output_dir']}")
        print(f"Artifact bundle: {report['artifact_bundle_dir']}")
        if report.get("warnings"):
            print("\nWarnings:")
            for warning in report["warnings"]:
                print(f"  - {warning}")
        if report.get("errors"):
            print("\nErrors:")
            for error in report["errors"]:
                print(f"  - {error}")

    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
