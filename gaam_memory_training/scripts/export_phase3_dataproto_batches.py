#!/usr/bin/env python3
"""
CLI for Phase 3 Milestone 2: Export Phase 3 DataProto batches.

This script converts Phase 2 actor update batches or verl-like batches into
tensor-ready local DataProto format for future verl integration.

Usage examples:

    # From Phase 2 rollout directory
    python scripts/export_phase3_dataproto_batches.py \
      --rollout_dir outputs/phase2_grpo_rollout \
      --output_dir outputs/phase3_dataproto_batches \
      --record_id e47becba

    # From Phase 2 verl-like batches
    python scripts/export_phase3_dataproto_batches.py \
      --verl_like_dir outputs/phase2_alignment_report/verl_like_batches \
      --output_dir outputs/phase3_dataproto_batches \
      --record_id e47becba

    # With HuggingFace tokenizer
    python scripts/export_phase3_dataproto_batches.py \
      --rollout_dir outputs/phase2_grpo_rollout \
      --output_dir outputs/phase3_dataproto_batches \
      --tokenizer_path Qwen/Qwen2.5-0.5B \
      --max_prompt_length 4096 \
      --max_response_length 2048

    # Write PyTorch tensors
    python scripts/export_phase3_dataproto_batches.py \
      --rollout_dir outputs/phase2_grpo_rollout \
      --output_dir outputs/phase3_dataproto_batches \
      --write_torch_tensors

    # Include unselected samples
    python scripts/export_phase3_dataproto_batches.py \
      --rollout_dir outputs/phase2_grpo_rollout \
      --output_dir outputs/phase3_dataproto_batches \
      --include_unselected
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.local_dataproto_adapter import export_local_dataproto_batches


def main():
    parser = argparse.ArgumentParser(
        description="Export Phase 3 local DataProto batches"
    )

    # Input source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--rollout_dir", type=Path, help="Phase 2 rollout directory")
    source_group.add_argument("--verl_like_dir", type=Path, help="Phase 2 verl-like batches directory")

    # Output
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")

    # Filtering
    parser.add_argument("--record_id", type=str, help="Optional single record ID to export")
    parser.add_argument("--max_records", type=int, help="Maximum number of records to process")
    parser.add_argument("--include_unselected", action="store_true", help="Include unselected samples")

    # Tokenizer options
    parser.add_argument("--tokenizer_path", type=str, help="HuggingFace tokenizer path")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for tokenizer")
    parser.add_argument("--max_prompt_length", type=int, help="Maximum prompt tokens")
    parser.add_argument("--max_response_length", type=int, help="Maximum response tokens")
    parser.add_argument("--truncation", type=str, default="error",
                        choices=["error", "left", "right"], help="Truncation strategy")

    # Export options
    parser.add_argument("--write_torch_tensors", action="store_true", help="Write PyTorch tensor files")

    # Validation
    parser.add_argument("--strict", action="store_true", help="Fail on any error")

    args = parser.parse_args()

    # Validate input
    if args.rollout_dir and not args.rollout_dir.exists():
        print(f"Error: Rollout directory does not exist: {args.rollout_dir}", file=sys.stderr)
        return 1

    if args.verl_like_dir and not args.verl_like_dir.exists():
        print(f"Error: verl-like directory does not exist: {args.verl_like_dir}", file=sys.stderr)
        return 1

    # Load tokenizer if specified
    tokenizer = None
    if args.tokenizer_path:
        try:
            from transformers import AutoTokenizer
            print(f"Loading tokenizer from {args.tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path,
                trust_remote_code=args.trust_remote_code,
            )
            print(f"✓ Loaded tokenizer: {tokenizer.__class__.__name__}")
        except ImportError:
            print("Warning: transformers not installed, using fallback tokenizer", file=sys.stderr)
            tokenizer = None
        except Exception as e:
            print(f"Error loading tokenizer: {e}", file=sys.stderr)
            if args.strict:
                return 1
            tokenizer = None

    # Print configuration
    print("=" * 80)
    print("Phase 3 Milestone 2: Export Local DataProto Batches")
    print("=" * 80)
    if args.rollout_dir:
        print(f"Source: Rollout directory ({args.rollout_dir})")
    else:
        print(f"Source: verl-like directory ({args.verl_like_dir})")
    print(f"Output: {args.output_dir}")
    print(f"Record ID filter: {args.record_id or '(all)'}")
    print(f"Max records: {args.max_records or '(all)'}")
    print(f"Include unselected: {args.include_unselected}")
    print(f"Tokenizer: {args.tokenizer_path or '(simple fallback)'}")
    if args.max_prompt_length:
        print(f"Max prompt length: {args.max_prompt_length}")
    if args.max_response_length:
        print(f"Max response length: {args.max_response_length}")
    print(f"Truncation: {args.truncation}")
    print(f"Write PyTorch tensors: {args.write_torch_tensors}")
    print("=" * 80)
    print()

    # Export batches
    try:
        print("Exporting local DataProto batches...")
        exported_paths = export_local_dataproto_batches(
            rollout_dir=args.rollout_dir,
            verl_like_dir=args.verl_like_dir,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            record_id=args.record_id,
            max_records=args.max_records,
            include_unselected=args.include_unselected,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            truncation=args.truncation,
            write_torch_tensors=args.write_torch_tensors,
        )

        print(f"\n✓ Exported {len(exported_paths)} local DataProto batches")

        # Print sample paths
        if exported_paths:
            print("\nExported files:")
            for i, path in enumerate(exported_paths[:5]):
                print(f"  {path.name}")
            if len(exported_paths) > 5:
                print(f"  ... and {len(exported_paths) - 5} more")

        # Print manifest
        manifest_path = args.output_dir / "manifest.jsonl"
        print(f"\n✓ Manifest: {manifest_path}")

        # Print summary
        summary_path = args.output_dir / "summary.json"
        print(f"✓ Summary: {summary_path}")

        # Print missing fields
        print("\nMissing fields for full verl compatibility:")
        print("  - old_log_probs (requires active actor model)")
        print("  - ref_log_probs (requires reference policy)")
        print("  - token_level_logprobs (requires model inference)")
        print("  - returns (requires advantage-to-return conversion)")

        print("\n" + "=" * 80)
        print("Export complete!")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n✗ Export failed: {e}", file=sys.stderr)
        if args.strict:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
