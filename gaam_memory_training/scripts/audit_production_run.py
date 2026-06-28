#!/usr/bin/env python3
"""Audit a GAAM production E2E run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.production_run_audit import (  # noqa: E402
    ProductionRunAuditConfig,
    audit_production_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit production GAAM training and evaluation artifacts."
    )
    parser.add_argument("--training_output_dir", type=Path, required=True)
    parser.add_argument("--evaluation_output_dir", type=Path, default=None)
    parser.add_argument("--split_manifest", type=Path, default=None)
    parser.add_argument("--expected_evaluation_split", default="test")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = audit_production_run(
        ProductionRunAuditConfig(
            training_output_dir=args.training_output_dir,
            evaluation_output_dir=args.evaluation_output_dir,
            split_manifest_path=args.split_manifest,
            expected_evaluation_split=args.expected_evaluation_split,
        )
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("\n=== GAAM Production Run Audit ===")
        print(f"Status: {report['status']}")
        print(f"Training output: {report['training_output_dir']}")
        print(f"Evaluation output: {report['evaluation_output_dir']}")
        training = report.get("training") or {}
        evaluation = report.get("evaluation") or {}
        print(f"Training manifest: {training.get('manifest_status')}")
        print(f"Training verification: {training.get('verification_status')}")
        if evaluation:
            print(f"Evaluation manifest: {evaluation.get('manifest_status')}")
            print(f"Evaluation verification: {evaluation.get('verification_status')}")
            print(f"Evaluation split: {evaluation.get('evaluation_split')}")
            print(f"Accuracy: {evaluation.get('accuracy')}")
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
