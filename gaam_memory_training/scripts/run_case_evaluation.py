#!/usr/bin/env python3
"""Run GAAM case-level evaluation: sessions -> memory -> answer -> judge."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.case_evaluation import CaseEvaluationConfig, run_case_evaluation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate GAAM memory on benchmark case questions."
    )
    parser.add_argument("--input", type=Path, default=Path("data/longmemeval/longmemeval_s_cleaned.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/case_evaluation"))
    parser.add_argument("--split_manifest", type=Path, default=None)
    parser.add_argument("--evaluation_split", choices=["train", "dev", "test", "all"], default=None)
    parser.add_argument("--record_id", default=None)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--memory_backend", choices=["baseline", "stateful_api", "stateful_local_hf"], default="baseline")
    parser.add_argument("--answer_backend", choices=["api", "local_hf", "no_llm"], default="api")
    parser.add_argument("--judge_backend", choices=["api", "heuristic"], default="api")
    parser.add_argument("--session_chunk_size", type=int, default=4)
    parser.add_argument("--max_chunk_chars", type=int, default=12000)
    parser.add_argument("--max_previous_memory_chars", type=int, default=12000)
    parser.add_argument("--memory_model", default=None)
    parser.add_argument("--memory_base_url", default=None)
    parser.add_argument("--memory_api_key", default=None)
    parser.add_argument("--memory_max_new_tokens", type=int, default=None)
    parser.add_argument("--memory_device_map", default=None)
    parser.add_argument("--memory_torch_dtype", default=None)
    parser.add_argument("--answer_model", default=None)
    parser.add_argument("--answer_base_url", default=None)
    parser.add_argument("--answer_api_key", default=None)
    parser.add_argument("--answer_max_new_tokens", type=int, default=None)
    parser.add_argument("--answer_device_map", default=None)
    parser.add_argument("--answer_torch_dtype", default=None)
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--judge_base_url", default=None)
    parser.add_argument("--judge_api_key", default=None)

    args = parser.parse_args()
    defaults = CaseEvaluationConfig(input_path=args.input, output_dir=args.output_dir)
    manifest = run_case_evaluation(
        CaseEvaluationConfig(
            input_path=args.input,
            output_dir=args.output_dir,
            split_manifest_path=args.split_manifest,
            evaluation_split=args.evaluation_split,
            record_id=args.record_id,
            max_records=args.max_records,
            memory_backend=args.memory_backend,
            answer_backend=args.answer_backend,
            judge_backend=args.judge_backend,
            session_chunk_size=args.session_chunk_size,
            max_chunk_chars=args.max_chunk_chars,
            max_previous_memory_chars=args.max_previous_memory_chars,
            memory_model=args.memory_model or defaults.memory_model,
            memory_base_url=args.memory_base_url or defaults.memory_base_url,
            memory_api_key=args.memory_api_key or defaults.memory_api_key,
            memory_max_new_tokens=args.memory_max_new_tokens or defaults.memory_max_new_tokens,
            memory_device_map=args.memory_device_map or defaults.memory_device_map,
            memory_torch_dtype=args.memory_torch_dtype or defaults.memory_torch_dtype,
            answer_model=args.answer_model or defaults.answer_model,
            answer_base_url=args.answer_base_url or defaults.answer_base_url,
            answer_api_key=args.answer_api_key or defaults.answer_api_key,
            answer_max_new_tokens=args.answer_max_new_tokens or defaults.answer_max_new_tokens,
            answer_device_map=args.answer_device_map or defaults.answer_device_map,
            answer_torch_dtype=args.answer_torch_dtype or defaults.answer_torch_dtype,
            judge_model=args.judge_model or defaults.judge_model,
            judge_base_url=args.judge_base_url or defaults.judge_base_url,
            judge_api_key=args.judge_api_key or defaults.judge_api_key,
        )
    )
    print("== GAAM case evaluation complete ==")
    print(f"status: {manifest['status']}")
    print(f"num_records: {manifest['num_records']}")
    print(f"num_judged: {manifest['num_judged']}")
    print(f"accuracy: {manifest['accuracy']}")
    print(f"manifest: {args.output_dir / 'case_evaluation_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
