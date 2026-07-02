#!/usr/bin/env python3
"""Export Memory Refactoring Policy samples to native VERL GRPO parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.memory_refactor_training import (
    MemoryRefactorExportConfig,
    export_memory_refactor_grpo_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Memory Refactoring Policy training samples to VERL parquet."
    )
    parser.add_argument("--input", type=Path, required=True, help="JSON/JSONL attack samples")
    parser.add_argument("--output_dir", type=Path, required=True, help="Dataset output directory")
    parser.add_argument("--max_records_per_split", type=int, default=None)
    args = parser.parse_args()

    manifest = export_memory_refactor_grpo_dataset(
        MemoryRefactorExportConfig(
            input_path=args.input,
            output_dir=args.output_dir,
            max_records_per_split=args.max_records_per_split,
        )
    )
    print("== Memory Refactor GRPO dataset exported ==")
    print(f"train_path: {manifest['train_path']}")
    print(f"val_path: {manifest['val_path']}")
    print(f"num_train_rows: {manifest['num_train_rows']}")
    print(f"num_val_rows: {manifest['num_val_rows']}")
    print(f"manifest: {args.output_dir / 'memory_refactor.dataset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

