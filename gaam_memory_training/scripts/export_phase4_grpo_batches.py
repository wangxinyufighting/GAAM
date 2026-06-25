#!/usr/bin/env python3
"""
Phase 4 Milestone 2: GRPO Batch Export CLI

Exports GRPO-ready actor batches from a completed multi-case rollout.

Example usage:
    python scripts/export_phase4_grpo_batches.py \
      --multi_case_run_dir outputs/phase4_multi_case_rollout/train_round000_step000 \
      --output_dir outputs/phase4_multi_case_rollout/train_round000_step000/aggregate/dataproto \
      --actors memory_builder question_agent \
      --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

from gaam_graph.dataset_split_schema import DatasetSplitName
from gaam_graph.phase4_batch_export import export_phase4_grpo_batches
from gaam_graph.phase4_rollout_schema import MultiCaseRolloutManifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export GRPO batches from multi-case rollout"
    )

    parser.add_argument(
        "--multi_case_run_dir",
        type=Path,
        required=True,
        help="Multi-case rollout run directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for GRPO batches",
    )
    parser.add_argument(
        "--actors",
        nargs="+",
        default=["memory_builder", "question_agent"],
        help="Actor roles to export",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory",
    )

    args = parser.parse_args()

    # Check run directory exists
    if not args.multi_case_run_dir.exists():
        print(
            f"Error: Run directory not found: {args.multi_case_run_dir}",
            file=sys.stderr,
        )
        return 1

    # Load rollout manifest
    manifest_path = args.multi_case_run_dir / "multi_case_rollout_manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_dict = json.load(f)
        manifest = MultiCaseRolloutManifest.model_validate(manifest_dict)
    except Exception as e:
        print(f"Error loading manifest: {e}", file=sys.stderr)
        return 1

    # Check output directory policy
    if args.output_dir.exists() and not args.overwrite:
        print(
            f"Error: Output directory exists: {args.output_dir}. Use --overwrite.",
            file=sys.stderr,
        )
        return 1

    try:
        # Export batches
        actor_manifests = export_phase4_grpo_batches(
            manifest.records,
            args.output_dir,
            args.actors,
            str(manifest_path),
            manifest.split,
        )

        # Write export manifest
        export_manifest = {
            "export_version": "phase4_grpo_batch_export_v1",
            "source_rollout_run_id": manifest.run_id,
            "source_manifest_path": str(manifest_path),
            "split": manifest.split,
            "actors": list(actor_manifests.keys()),
            "actor_manifests": {
                actor_role: str(args.output_dir / f"{actor_role}.batch_manifest.json")
                for actor_role in actor_manifests
            },
        }

        export_manifest_path = args.output_dir / "export_manifest.json"
        with open(export_manifest_path, "w", encoding="utf-8") as f:
            json.dump(export_manifest, f, indent=2)

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"GRPO Batch Export Complete")
        print(f"{'=' * 60}")
        print(f"Run ID: {manifest.run_id}")
        print(f"Split: {manifest.split}")
        print(f"Actors exported: {len(actor_manifests)}")

        for actor_role, actor_manifest in actor_manifests.items():
            print(f"\n{actor_role}:")
            print(f"  Groups: {actor_manifest.num_groups}")
            print(f"  Samples: {actor_manifest.num_samples}")
            print(f"  Selected: {actor_manifest.num_selected_samples}")
            print(f"  No-leakage: {'PASSED' if actor_manifest.no_leakage_passed else 'FAILED'}")
            if actor_manifest.reward_mean is not None:
                print(f"  Reward mean: {actor_manifest.reward_mean:.4f}")

        print(f"\nOutput: {args.output_dir}")
        print(f"Export manifest: {export_manifest_path}")

        # Check no-leakage status
        all_passed = all(m.no_leakage_passed for m in actor_manifests.values())
        if not all_passed:
            print("\nWarning: Some batches failed no-leakage checks", file=sys.stderr)
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
