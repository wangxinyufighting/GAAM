#!/usr/bin/env python3
"""
Build current memory artifacts from LongMemEval-style raw history.

Usage:
    python scripts/build_current_memory.py \\
      --input data/longmemeval/longmemeval_s_cleaned.json \\
      --output_dir outputs/current_memory \\
      --max_records 10
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.lme_loader import LongMemEvalLoader
from gaam_graph.memory_builder import BaselineMemoryBuilder, build_current_memory_from_record
from gaam_graph.memory_schema import validate_current_memory
from gaam_graph.utils import write_json


def main():
    parser = argparse.ArgumentParser(
        description="Build current memory artifacts from LongMemEval-style input"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="LongMemEval-style input JSON"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory for current memory outputs"
    )
    parser.add_argument(
        "--max_records",
        type=int,
        help="Optional limit on number of records to process"
    )
    parser.add_argument(
        "--record_id",
        type=str,
        help="Optional single record filter"
    )
    parser.add_argument(
        "--include_assistant_turns",
        action="store_true",
        help="Include assistant turns as event nodes"
    )
    parser.add_argument(
        "--min_user_text_chars",
        type=int,
        default=20,
        help="Minimum text length for user turns (default: 20)"
    )
    parser.add_argument(
        "--max_event_nodes",
        type=int,
        help="Optional cap on event nodes per record (for smoke tests)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs"
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        help="Optional summary output path (default: output_dir/summary.json)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize builder
    builder = BaselineMemoryBuilder(
        include_assistant_turns=args.include_assistant_turns,
        min_user_text_chars=args.min_user_text_chars,
        max_event_nodes=args.max_event_nodes,
    )

    # Load records
    print(f"Loading records from {args.input}")
    loader = LongMemEvalLoader(str(args.input))
    records = loader.load()  # Returns list of LMERecord

    # Filter by record_id if specified
    if args.record_id:
        records = [r for r in records if r.record_id == args.record_id]
        if not records:
            print(f"Error: Record {args.record_id} not found", file=sys.stderr)
            return 1

    # Limit records if specified
    if args.max_records:
        records = records[:args.max_records]

    print(f"Processing {len(records)} records")

    # Process records
    manifest_path = args.output_dir / "manifest.jsonl"
    manifest_file = open(manifest_path, "w", encoding="utf-8")

    results = []
    processed_count = 0
    failed_count = 0
    total_nodes = 0
    total_edges = 0

    for i, record in enumerate(records, 1):
        print(f"[{i}/{len(records)}] Processing {record.record_id}")

        output_path = args.output_dir / f"{record.record_id}.current_memory.json"

        # Check if exists and not overwriting
        if output_path.exists() and not args.overwrite:
            print(f"  Skipping (already exists, use --overwrite to replace)")
            continue

        try:
            # Build current memory
            memory = build_current_memory_from_record(record, builder=builder)

            # Validate
            validated = validate_current_memory(memory, strict=True)

            # Write output
            write_json(output_path, memory)

            # Collect stats
            node_count = len(memory["memory_graph"]["nodes"])
            edge_count = len(memory["memory_graph"]["edges"])
            session_ids = set()
            for node in memory["memory_graph"]["nodes"]:
                session_ids.update(node.get("source_session_ids", []))
            session_count = len(session_ids)

            retained_turn_count = sum(
                1 for n in memory["memory_graph"]["nodes"]
                if n["type"] == "event"
            )

            # Write manifest row
            manifest_row = {
                "record_id": record.record_id,
                "output_path": str(output_path),
                "node_count": node_count,
                "edge_count": edge_count,
                "session_count": session_count,
                "retained_turn_count": retained_turn_count,
                "status": "ok",
            }
            manifest_file.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
            manifest_file.flush()

            # Update totals
            processed_count += 1
            total_nodes += node_count
            total_edges += edge_count

            print(f"  ✓ nodes={node_count}, edges={edge_count}, sessions={session_count}")

        except Exception as e:
            print(f"  ✗ Failed: {e}", file=sys.stderr)

            # Write manifest row for failure
            manifest_row = {
                "record_id": record.record_id,
                "output_path": str(output_path),
                "node_count": 0,
                "edge_count": 0,
                "session_count": 0,
                "retained_turn_count": 0,
                "status": "failed",
                "error": str(e),
            }
            manifest_file.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
            manifest_file.flush()

            failed_count += 1
            continue

    manifest_file.close()

    # Write summary
    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "processed_records": processed_count,
        "failed_records": failed_count,
        "total_memory_nodes": total_nodes,
        "total_memory_edges": total_edges,
    }

    summary_path = args.summary_json or (args.output_dir / "summary.json")
    write_json(summary_path, summary)

    print(f"\nSummary:")
    print(f"  Processed: {processed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total edges: {total_edges}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Summary: {summary_path}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
