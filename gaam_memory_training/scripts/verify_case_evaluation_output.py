#!/usr/bin/env python3
"""Verify GAAM case-evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.case_evaluation_verifier import (  # noqa: E402
    CaseEvaluationVerificationConfig,
    verify_case_evaluation_output,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify GAAM case-evaluation output artifacts.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--allow_unjudged", action="store_true")
    parser.add_argument("--strict_answer_leakage", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = verify_case_evaluation_output(
        CaseEvaluationVerificationConfig(
            output_dir=args.output_dir,
            input_path=args.input,
            allow_unjudged=args.allow_unjudged,
            strict_answer_leakage=args.strict_answer_leakage,
        )
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("\n=== Case Evaluation Verification ===")
        print(f"Status: {report['status']}")
        print(f"Output dir: {report['output_dir']}")
        metrics = report.get("metrics", {})
        if metrics:
            print(f"Reports: {metrics.get('num_reports')}")
            print(f"Judged: {metrics.get('num_judged')}")
            print(f"Scanned memory files: {metrics.get('scanned_memory_files')}")
            print(f"Accuracy: {metrics.get('accuracy')}")
        if report.get("warnings"):
            print("\nWarnings:")
            for warning in report["warnings"]:
                print(f"  - {warning}")
        if report.get("errors"):
            print("\nErrors:")
            for error in report["errors"]:
                print(f"  - {error}")
        print(f"\nReport: {args.output_dir / 'case_evaluation_verification.json'}")

    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
