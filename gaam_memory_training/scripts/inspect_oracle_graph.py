#!/usr/bin/env python3
"""
Inspect oracle graph artifacts.

Usage:
    python scripts/inspect_oracle_graph.py \\
      --graph outputs/no_leak_smoke_test/e47becba.graph.json \\
      --evidence_pack outputs/no_leak_smoke_test/e47becba.evidence_pack.json \\
      --strict

    python scripts/inspect_oracle_graph.py \\
      --graph outputs/no_leak_smoke_test/e47becba.graph.json \\
      --node evt_1bf02d4060e8312e \\
      --context \\
      --strict
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.oracle_graph_loader import (
    OracleGraphFormatError,
    OracleGraphLeakageError,
    OracleGraphLoader,
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect oracle graph artifacts and validate no-leakage constraints"
    )
    parser.add_argument(
        "--graph",
        required=True,
        type=Path,
        help="Path to .graph.json file"
    )
    parser.add_argument(
        "--evidence_pack",
        type=Path,
        help="Optional path to .evidence_pack.json file"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation (raise on leakage or malformed structure)"
    )
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Debug mode: remove forbidden keys (not for training)"
    )
    parser.add_argument(
        "--node",
        type=str,
        help="Node ID to inspect"
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Print evidence context for the node"
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        help="Maximum characters for evidence context"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.graph.exists():
        print(json.dumps({"error": f"Graph file not found: {args.graph}"}), file=sys.stderr)
        return 1

    if args.evidence_pack and not args.evidence_pack.exists():
        print(json.dumps({"error": f"Evidence pack not found: {args.evidence_pack}"}), file=sys.stderr)
        return 1

    # Load oracle graph
    try:
        loader = OracleGraphLoader(
            graph_path=args.graph,
            evidence_pack_path=args.evidence_pack,
            strict=args.strict,
            sanitize=args.sanitize,
        )
    except OracleGraphLeakageError as e:
        print(json.dumps({
            "error": "leakage_detected",
            "message": str(e),
            "graph": str(args.graph),
        }), file=sys.stderr)
        return 1
    except OracleGraphFormatError as e:
        print(json.dumps({
            "error": "format_error",
            "message": str(e),
            "graph": str(args.graph),
        }), file=sys.stderr)
        return 1
    except Exception as e:
        print(json.dumps({
            "error": "load_failed",
            "message": str(e),
            "graph": str(args.graph),
        }), file=sys.stderr)
        return 1

    # Get summary
    summary = loader.summary()

    # Print summary
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # If node inspection requested
    if args.node:
        print("\n" + "="*80)
        print(f"Node: {args.node}")
        print("="*80 + "\n")

        node = loader.get_node_or_none(args.node)
        if node is None:
            print(f"Node not found: {args.node}", file=sys.stderr)
            return 1

        print(json.dumps(node, indent=2, ensure_ascii=False))

        if args.context:
            print("\n" + "="*80)
            print("Evidence Context")
            print("="*80 + "\n")

            context = loader.build_evidence_context(
                [args.node],
                include_edges=True,
                max_chars=args.max_chars
            )
            print(context)

    return 0


if __name__ == "__main__":
    sys.exit(main())
