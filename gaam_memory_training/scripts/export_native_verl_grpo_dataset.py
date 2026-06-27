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
