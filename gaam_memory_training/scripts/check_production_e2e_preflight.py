#!/usr/bin/env python3
"""Run GAAM production E2E preflight checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaam_graph.production_preflight import ProductionPreflightConfig, run_production_preflight, truthy


def _env_first_nonempty(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value)
    return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the full GAAM production training/evaluation environment before launch."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--oracle_graph_dir", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, required=True)
    parser.add_argument("--memory_model_path", type=Path, required=True)
    parser.add_argument("--question_model_path", type=Path, required=True)
    parser.add_argument("--code_a1_root", type=Path, required=True)
    parser.add_argument("--training_output_dir", type=Path, required=True)
    parser.add_argument("--eval_output_dir", type=Path, default=None)
    parser.add_argument("--run_evaluation", default=os.getenv("RUN_EVALUATION", "1"))
    parser.add_argument("--evaluation_split", default=os.getenv("EVALUATION_SPLIT", "test"))
    parser.add_argument("--memory_backend", default=os.getenv("MEMORY_BACKEND", "stateful_local_hf"))
    parser.add_argument("--answer_backend", default=os.getenv("ANSWER_BACKEND", "api"))
    parser.add_argument("--judge_backend", default=os.getenv("JUDGE_BACKEND", "api"))
    parser.add_argument(
        "--memory_api_key",
        default=_env_first_nonempty(
            "MEMORY_API_KEY",
            "GAAM_MEMORY_BUILDER_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            default="",
        ),
    )
    parser.add_argument(
        "--answer_api_key",
        default=_env_first_nonempty(
            "ANSWER_API_KEY",
            "GAAM_ANSWERER_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            default="",
        ),
    )
    parser.add_argument(
        "--judge_api_key",
        default=_env_first_nonempty(
            "JUDGE_API_KEY",
            "GAAM_EVAL_JUDGE_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            default="",
        ),
    )
    parser.add_argument("--reward_judge_enabled", default=os.getenv("GAAM_REWARD_JUDGE_ENABLED", "0"))
    parser.add_argument(
        "--reward_judge_api_key",
        default=_env_first_nonempty(
            "GAAM_REWARD_JUDGE_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            default="",
        ),
    )
    parser.add_argument("--require_cuda", default=os.getenv("REQUIRE_CUDA", "1"))
    parser.add_argument("--check_imports", default=os.getenv("CHECK_IMPORTS", "1"))
    parser.add_argument("--require_verl_import", default=os.getenv("REQUIRE_VERL_IMPORT", "1"))
    parser.add_argument("--memory_num_gpus", type=int, default=int(os.getenv("MEMORY_NUM_GPUS", os.getenv("NUM_GPUS", "1"))))
    parser.add_argument("--question_num_gpus", type=int, default=int(os.getenv("QUESTION_NUM_GPUS", os.getenv("NUM_GPUS", "1"))))
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--questions_per_case", type=int, default=None)
    parser.add_argument("--max_reported_issues", type=int, default=30)
    args = parser.parse_args()

    report = run_production_preflight(
        ProductionPreflightConfig(
            input_path=args.input,
            oracle_graph_dir=args.oracle_graph_dir,
            split_manifest=args.split_manifest,
            memory_model_path=args.memory_model_path,
            question_model_path=args.question_model_path,
            code_a1_root=args.code_a1_root,
            training_output_dir=args.training_output_dir,
            eval_output_dir=args.eval_output_dir,
            run_evaluation=truthy(args.run_evaluation),
            evaluation_split=args.evaluation_split,
            memory_backend=args.memory_backend,
            answer_backend=args.answer_backend,
            judge_backend=args.judge_backend,
            memory_api_key=args.memory_api_key,
            answer_api_key=args.answer_api_key,
            judge_api_key=args.judge_api_key,
            reward_judge_enabled=truthy(args.reward_judge_enabled),
            reward_judge_api_key=args.reward_judge_api_key,
            require_cuda=truthy(args.require_cuda),
            check_imports=truthy(args.check_imports),
            require_verl_import=truthy(args.require_verl_import),
            memory_num_gpus=args.memory_num_gpus,
            question_num_gpus=args.question_num_gpus,
            train_batch_size=args.train_batch_size,
            questions_per_case=args.questions_per_case,
            max_reported_issues=args.max_reported_issues,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
