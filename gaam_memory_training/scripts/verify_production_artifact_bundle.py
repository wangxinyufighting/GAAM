#!/usr/bin/env python3
"""Verify a compact GAAM production evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.production_artifact_bundle_verifier import (  # noqa: E402
    ProductionArtifactBundleVerificationConfig,
    verify_production_artifact_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify GAAM production artifact bundle files, hashes, and tarball."
    )
    parser.add_argument("--bundle_output_dir", type=Path, required=True)
    parser.add_argument("--no_require_evaluation", action="store_true")
    parser.add_argument("--no_require_tar", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = verify_production_artifact_bundle(
        ProductionArtifactBundleVerificationConfig(
            bundle_output_dir=args.bundle_output_dir,
            require_evaluation=not args.no_require_evaluation,
            require_tar=not args.no_require_tar,
        )
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("\n=== GAAM Production Artifact Bundle Verification ===")
        print(f"Status: {report['status']}")
        print(f"Bundle output: {report['bundle_output_dir']}")
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
