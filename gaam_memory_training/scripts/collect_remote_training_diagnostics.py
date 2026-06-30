#!/usr/bin/env python3
"""Collect GAAM remote training diagnostics without launching training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.production_preflight import ProductionPreflightConfig, truthy  # noqa: E402
from gaam_graph.remote_training_diagnostics import (  # noqa: E402
    RemoteTrainingDiagnosticsConfig,
    collect_remote_training_diagnostics,
)


DIAGNOSTIC_ENV_KEYS = [
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "MEMORY_MODEL_PATH",
    "QUESTION_MODEL_PATH",
    "CODE_A1_ROOT",
    "SPLIT_MANIFEST",
    "RUN_EVALUATION",
    "EVALUATION_SPLIT",
    "MEMORY_BACKEND",
    "ANSWER_BACKEND",
    "JUDGE_BACKEND",
    "GAAM_REWARD_JUDGE_ENABLED",
    "GAAM_REWARD_JUDGE_BASE_URL",
    "GAAM_REWARD_JUDGE_MODEL",
    "GAAM_REWARD_JUDGE_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "MEMORY_API_KEY",
    "GAAM_MEMORY_BUILDER_API_KEY",
    "ANSWER_API_KEY",
    "GAAM_ANSWERER_API_KEY",
    "JUDGE_API_KEY",
    "GAAM_EVAL_JUDGE_API_KEY",
]


def _env_first_nonempty(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value)
    return default


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a JSON diagnostics report for a remote GAAM production run. "
            "This runs preflight checks and records package/CUDA versions without starting training."
        )
    )
    parser.add_argument("--input", type=Path, default=Path(os.getenv("INPUT_PATH", "data/longmemeval/longmemeval_s_cleaned.json")))
    parser.add_argument("--oracle_graph_dir", type=Path, default=Path(os.getenv("ORACLE_GRAPH_DIR", "outputs/longmemeval_s_graph")))
    parser.add_argument("--split_manifest", type=Path, default=Path(os.getenv("SPLIT_MANIFEST", "outputs/splits/longmemeval_s.existing_graphs.seed0.json")))
    parser.add_argument("--memory_model_path", type=Path, default=Path(os.getenv("MEMORY_MODEL_PATH", "")))
    parser.add_argument("--question_model_path", type=Path, default=Path(os.getenv("QUESTION_MODEL_PATH", "")))
    parser.add_argument("--code_a1_root", type=Path, default=Path(os.getenv("CODE_A1_ROOT", str(ROOT_DIR.parent / "Code-A1" / "Code-A1"))))
    parser.add_argument("--training_output_dir", type=Path, default=Path(os.getenv("TRAINING_OUTPUT_DIR", "outputs/production_native_verl_dual_cotraining")))
    parser.add_argument("--eval_output_dir", type=Path, default=Path(os.getenv("EVAL_OUTPUT_DIR", "outputs/production_post_training_case_evaluation")))
    parser.add_argument("--output", type=Path, default=Path(os.getenv("DIAGNOSTICS_OUTPUT", "outputs/remote_training_diagnostics.json")))
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
    parser.add_argument("--train_batch_size", type=int, default=int(os.getenv("TRAIN_BATCH_SIZE", "8")))
    parser.add_argument("--questions_per_case", type=int, default=int(os.getenv("QUESTIONS_PER_CASE", "8")))
    parser.add_argument("--skip_nvidia_smi", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = collect_remote_training_diagnostics(
        RemoteTrainingDiagnosticsConfig(
            preflight_config=ProductionPreflightConfig(
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
            ),
            output_path=args.output,
            include_nvidia_smi=not args.skip_nvidia_smi,
            env_keys=DIAGNOSTIC_ENV_KEYS,
        )
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("\n=== GAAM Remote Training Diagnostics ===")
        print(f"Status: {report['status']}")
        print(f"Output: {args.output}")
        print(f"Python: {report['python']['executable']}")
        torch_info = report.get("torch", {})
        print(f"Torch: {torch_info.get('version')} CUDA={torch_info.get('cuda_version')} available={torch_info.get('cuda_available')}")
        if report.get("preflight", {}).get("errors"):
            print("\nPreflight errors:")
            for error in report["preflight"]["errors"]:
                print(f"  - {error}")

    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
