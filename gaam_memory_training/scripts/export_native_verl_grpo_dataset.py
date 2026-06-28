#!/usr/bin/env python3
"""Export GAAM data to native VERL GRPO parquet format."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.native_verl_grpo import NativeVerlExportConfig, export_native_verl_grpo_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export GAAM actor data to native VERL GRPO parquet files."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--oracle_graph_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--actor_role",
        choices=["memory_builder", "question_agent"],
        required=True,
    )
    parser.add_argument("--split_manifest", type=Path, default=None)
    parser.add_argument("--max_history_chars", type=int, default=16000)
    parser.add_argument("--max_oracle_chars", type=int, default=12000)
    parser.add_argument(
        "--questions_per_case",
        type=int,
        default=8,
        help="Target number of questions generated per case for question_agent prompts.",
    )
    parser.add_argument(
        "--memory_input_mode",
        choices=["full", "incremental"],
        default="full",
        help="Memory Builder input mode. 'incremental' exports one row per session chunk.",
    )
    parser.add_argument(
        "--memory_session_chunk_size",
        type=int,
        default=4,
        help="Number of sessions per incremental Memory Builder chunk. Use <=0 for one full chunk.",
    )
    parser.add_argument(
        "--max_memory_chunk_chars",
        type=int,
        default=None,
        help="Prompt character budget for each incremental raw-history chunk.",
    )
    parser.add_argument(
        "--max_previous_memory_chars",
        type=int,
        default=6000,
        help="Prompt character budget for previous Current Memory in incremental mode.",
    )
    parser.add_argument(
        "--allow_static_incremental_scaffold",
        action="store_true",
        help=(
            "Allow non-stateful incremental prompt-format export. "
            "Use only for smoke tests; production training should use a stateful rollout worker."
        ),
    )
    parser.add_argument("--max_records_per_split", type=int, default=None)

    args = parser.parse_args()
    manifest = export_native_verl_grpo_dataset(
        NativeVerlExportConfig(
            input_path=args.input,
            oracle_graph_dir=args.oracle_graph_dir,
            output_dir=args.output_dir,
            actor_role=args.actor_role,
            split_manifest_path=args.split_manifest,
            max_history_chars=args.max_history_chars,
            max_oracle_chars=args.max_oracle_chars,
            questions_per_case=args.questions_per_case,
            memory_input_mode=args.memory_input_mode,
            memory_session_chunk_size=args.memory_session_chunk_size,
            max_memory_chunk_chars=args.max_memory_chunk_chars,
            max_previous_memory_chars=args.max_previous_memory_chars,
            allow_static_incremental_scaffold=args.allow_static_incremental_scaffold,
            max_records_per_split=args.max_records_per_split,
        )
    )

    print("== Native VERL GRPO dataset exported ==")
    print(f"actor_role: {manifest['actor_role']}")
    print(f"train_path: {manifest['train_path']}")
    print(f"val_path: {manifest['val_path']}")
    print(f"num_train_rows: {manifest['num_train_rows']}")
    print(f"num_val_rows: {manifest['num_val_rows']}")
    print(f"manifest: {args.output_dir / (args.actor_role + '.dataset_manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
