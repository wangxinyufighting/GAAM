#!/usr/bin/env python3
"""Run stateful incremental Current Memory building through API or local HF model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.stateful_incremental_memory import (  # noqa: E402
    StatefulIncrementalMemoryConfig,
    build_stateful_incremental_memories,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Current Memory with real stateful incremental rollouts."
    )
    parser.add_argument("--input", type=Path, default=Path("data/longmemeval/longmemeval_s_cleaned.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/stateful_incremental_memory"))
    parser.add_argument("--record_id", default=None)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--session_chunk_size", type=int, default=4)
    parser.add_argument("--max_chunk_chars", type=int, default=12000)
    parser.add_argument("--max_previous_memory_chars", type=int, default=12000)
    parser.add_argument("--llm_backend", choices=["api", "local_hf"], default="api")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--torch_dtype", default=None)

    args = parser.parse_args()
    defaults = StatefulIncrementalMemoryConfig(
        input_path=args.input,
        output_dir=args.output_dir,
    )
    manifest = build_stateful_incremental_memories(
        StatefulIncrementalMemoryConfig(
            input_path=args.input,
            output_dir=args.output_dir,
            record_id=args.record_id,
            max_records=args.max_records,
            session_chunk_size=args.session_chunk_size,
            max_chunk_chars=args.max_chunk_chars,
            max_previous_memory_chars=args.max_previous_memory_chars,
            llm_backend=args.llm_backend,
            model=args.model or defaults.model,
            base_url=args.base_url or defaults.base_url,
            api_key=args.api_key or defaults.api_key,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens or defaults.max_new_tokens,
            device_map=args.device_map or defaults.device_map,
            torch_dtype=args.torch_dtype or defaults.torch_dtype,
        )
    )
    print("== Stateful incremental memory building complete ==")
    print(f"status: {manifest['status']}")
    print(f"records: {len(manifest['records'])}")
    print(f"manifest: {args.output_dir / 'stateful_incremental_memory_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
