#!/usr/bin/env python3
"""Run GAAM dual-agent co-training with native Code-A1/VERL GRPO kernels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaam_graph.native_verl_dual_trainer import (  # noqa: E402
    ACTOR_MEMORY_BUILDER,
    ACTOR_QUESTION_AGENT,
    GAAMDualRayTrainer,
    GAAMDualRayTrainerConfig,
    NativeVerlActorConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GAAM dual-agent native VERL GRPO co-training."
    )
    parser.add_argument("--root_dir", type=Path, default=ROOT_DIR)
    parser.add_argument("--code_a1_root", type=Path, required=True)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--oracle_graph_dir", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--order",
        default="question_agent,memory_builder",
        help="Comma-separated actor order per round.",
    )
    parser.add_argument("--memory_model_path", type=Path, required=True)
    parser.add_argument("--question_model_path", type=Path, required=True)
    parser.add_argument("--memory_num_gpus", type=int, default=1)
    parser.add_argument("--question_num_gpus", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--rollout_n", type=int, default=4)
    parser.add_argument("--memory_rollout_n", type=int, default=None)
    parser.add_argument("--question_rollout_n", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--memory_train_batch_size", type=int, default=None)
    parser.add_argument("--question_train_batch_size", type=int, default=None)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=4)
    parser.add_argument("--ppo_micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--memory_max_prompt_length", type=int, default=None)
    parser.add_argument("--question_max_prompt_length", type=int, default=None)
    parser.add_argument("--max_response_length", type=int, default=2048)
    parser.add_argument("--memory_max_response_length", type=int, default=None)
    parser.add_argument("--question_max_response_length", type=int, default=None)
    parser.add_argument("--max_history_chars", type=int, default=16000)
    parser.add_argument("--max_oracle_chars", type=int, default=12000)
    parser.add_argument(
        "--questions_per_case",
        type=int,
        default=8,
        help="Target number of questions generated per case by the Question Agent.",
    )
    parser.add_argument(
        "--memory_input_mode",
        choices=["full", "incremental"],
        default="full",
        help="Memory Builder input mode. 'incremental' exports one row per session chunk.",
    )
    parser.add_argument("--memory_session_chunk_size", type=int, default=4)
    parser.add_argument("--max_memory_chunk_chars", type=int, default=None)
    parser.add_argument("--max_previous_memory_chars", type=int, default=6000)
    parser.add_argument(
        "--allow_static_incremental_scaffold",
        action="store_true",
        help=(
            "Allow non-stateful incremental prompt-format export. "
            "Use only for smoke tests; production training should use a stateful rollout worker."
        ),
    )
    parser.add_argument("--max_records_per_split", type=int, default=None)
    parser.add_argument("--total_epochs_per_actor_step", type=int, default=1)
    parser.add_argument("--total_training_steps_per_actor_step", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--test_freq", type=int, default=1)
    parser.add_argument("--lr", default="1e-6")
    parser.add_argument("--rollout_tp_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", default="0.6")
    parser.add_argument("--logger", default="console")
    parser.add_argument("--no_save_hf_model", action="store_true")
    parser.add_argument(
        "--no_require_hf_checkpoint",
        action="store_true",
        help=(
            "Do not fail an actor step when native VERL exits without an HF checkpoint. "
            "Use only for smoke tests; production full-model training should require checkpoints."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()
    order = tuple(actor.strip() for actor in args.order.split(",") if actor.strip())

    memory_config = NativeVerlActorConfig(
        actor_role=ACTOR_MEMORY_BUILDER,
        initial_model_path=args.memory_model_path,
        num_gpus=args.memory_num_gpus,
        rollout_n=args.memory_rollout_n or args.rollout_n,
        train_batch_size=args.memory_train_batch_size or args.train_batch_size,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=args.ppo_micro_batch_size_per_gpu,
        max_prompt_length=args.memory_max_prompt_length or args.max_prompt_length,
        max_response_length=args.memory_max_response_length or args.max_response_length,
        lr=args.lr,
        rollout_tp_size=args.rollout_tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    question_config = NativeVerlActorConfig(
        actor_role=ACTOR_QUESTION_AGENT,
        initial_model_path=args.question_model_path,
        num_gpus=args.question_num_gpus,
        rollout_n=args.question_rollout_n or args.rollout_n,
        train_batch_size=args.question_train_batch_size or args.train_batch_size,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        ppo_micro_batch_size_per_gpu=args.ppo_micro_batch_size_per_gpu,
        max_prompt_length=args.question_max_prompt_length or args.max_prompt_length,
        max_response_length=args.question_max_response_length or args.max_response_length,
        lr=args.lr,
        rollout_tp_size=args.rollout_tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    trainer = GAAMDualRayTrainer(
        GAAMDualRayTrainerConfig(
            root_dir=args.root_dir,
            code_a1_root=args.code_a1_root,
            python_bin=args.python_bin,
            input_path=args.input,
            oracle_graph_dir=args.oracle_graph_dir,
            split_manifest=args.split_manifest,
            output_dir=args.output_dir,
            rounds=args.rounds,
            order=order,
            memory_builder=memory_config,
            question_agent=question_config,
            nnodes=args.nnodes,
            total_epochs_per_actor_step=args.total_epochs_per_actor_step,
            total_training_steps_per_actor_step=args.total_training_steps_per_actor_step,
            save_freq=args.save_freq,
            test_freq=args.test_freq,
            max_history_chars=args.max_history_chars,
            max_oracle_chars=args.max_oracle_chars,
            questions_per_case=args.questions_per_case,
            memory_input_mode=args.memory_input_mode,
            memory_session_chunk_size=args.memory_session_chunk_size,
            max_memory_chunk_chars=args.max_memory_chunk_chars,
            max_previous_memory_chars=args.max_previous_memory_chars,
            allow_static_incremental_scaffold=args.allow_static_incremental_scaffold,
            max_records_per_split=args.max_records_per_split,
            logger=args.logger,
            save_hf_model=not args.no_save_hf_model,
            require_hf_checkpoint=not args.no_require_hf_checkpoint,
            dry_run=args.dry_run,
        )
    )
    manifest = trainer.fit()
    return 0 if manifest["status"] in {"succeeded", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
