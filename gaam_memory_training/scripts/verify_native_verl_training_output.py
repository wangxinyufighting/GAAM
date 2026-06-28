#!/usr/bin/env python3
"""Verify GAAM native dual Code-A1/VERL training outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.native_verl_training_verifier import (  # noqa: E402
    verify_native_verl_training_output,
    write_native_verl_training_verification_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that GAAM native dual VERL training produced real checkpoints."
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--report_path",
        type=Path,
        default=None,
        help="Where to write JSON verification report. Defaults to <output_dir>/native_verl_training_verification.json.",
    )
    parser.add_argument(
        "--no_require_hf_checkpoint",
        action="store_true",
        help="Do not fail when HF checkpoints are missing. Use only for dry-run/smoke diagnostics.",
    )
    parser.add_argument(
        "--no_require_grpo_log_markers",
        action="store_true",
        help="Do not require actor logs to contain native GRPO launch markers.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")

    args = parser.parse_args()
    report = verify_native_verl_training_output(
        args.output_dir,
        require_hf_checkpoint=not args.no_require_hf_checkpoint,
        require_grpo_log_markers=not args.no_require_grpo_log_markers,
    )
    report_path = args.report_path or (args.output_dir / "native_verl_training_verification.json")
    write_native_verl_training_verification_report(report, report_path)

    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("== GAAM native VERL training verification ==")
        print(f"status: {payload['status']}")
        print(f"output_dir: {payload['output_dir']}")
        print(f"manifest: {payload['manifest_path']}")
        print(f"checks_passed: {payload['checks_passed']}")
        print(f"checks_failed: {payload['checks_failed']}")
        print(f"report: {report_path}")
        if payload["issues"]:
            print("\nIssues:")
            for issue in payload["issues"]:
                print(f"  - {issue}")
        if payload["warnings"]:
            print("\nWarnings:")
            for warning in payload["warnings"]:
                print(f"  - {warning}")

    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
