#!/usr/bin/env python3
"""
CLI for Phase 2 Milestone 3/4: Local GRPO Trainer.

This script runs the local GRPO trainer that consumes Milestone 2 rollout artifacts
and executes policy update APIs.

Milestone 3: Stub policy clients
Milestone 4: Real trainable local model backends

Usage (stub backend):
    python scripts/run_grpo_adversarial_training.py \
      --rollout_dir outputs/phase2_grpo_rollout_smoke \
      --output_dir outputs/phase2_grpo_trainer_smoke \
      --dry_run \
      --max_records 1

Usage (local HF backend):
    python scripts/run_grpo_adversarial_training.py \
      --rollout_dir outputs/phase2_grpo_rollout \
      --output_dir outputs/phase2_grpo_trainer_local_hf \
      --policy_backend local_hf \
      --memory_model_path /path/to/memory-builder-model \
      --question_model_path /path/to/question-agent-model \
      --device cpu \
      --no_dry_run \
      --max_records 1
"""

import argparse
import sys
from pathlib import Path

# Add project root to path when running as `python scripts/...`
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.grpo_schema import ActorRole
from gaam_graph.grpo_trainer import GRPOTrainerConfig, LocalGRPOTrainer
from gaam_graph.policy_factory import build_policy_client


def main():
    parser = argparse.ArgumentParser(
        description="Run local GRPO trainer for adversarial co-training"
    )

    # Input/output
    parser.add_argument("--rollout_dir", type=Path, required=True, help="Rollout directory from Milestone 2")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for trainer artifacts")

    # Round/step IDs
    parser.add_argument("--round_id", type=int, default=0, help="Training round ID")
    parser.add_argument("--step_id", type=int, default=0, help="Training step ID")

    # Record filtering
    parser.add_argument("--record_id", type=str, help="Process only this record ID")
    parser.add_argument("--max_records", type=int, help="Maximum number of records to process")

    # Training mode
    parser.add_argument("--dry_run", action="store_true", default=True, help="Dry run mode (default)")
    parser.add_argument("--no_dry_run", dest="dry_run", action="store_false", help="Real update mode")

    # Execution flags
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first error")
    parser.add_argument("--require_both_actor_batches", action="store_true", help="Fail if either actor batch missing")

    # Batch reconstruction
    parser.add_argument("--reconstruct_missing_batches", action="store_true", default=True, help="Reconstruct batches from groups (default)")
    parser.add_argument("--no_reconstruct_missing_batches", dest="reconstruct_missing_batches", action="store_false")

    # Item selection
    parser.add_argument("--selected_only", action="store_true", default=True, help="Use only selected items (default)")
    parser.add_argument("--all_items", dest="selected_only", action="store_false", help="Use all items")

    # Policy IDs
    parser.add_argument("--memory_policy_id", type=str, default="stub_memory_builder", help="Memory Builder policy ID")
    parser.add_argument("--question_policy_id", type=str, default="stub_question_agent", help="Question Agent policy ID")

    # Checkpoint flags
    parser.add_argument("--write_checkpoints", action="store_true", default=True, help="Write checkpoint metadata (default)")
    parser.add_argument("--no_write_checkpoints", dest="write_checkpoints", action="store_false")

    # Safety flags
    parser.add_argument("--allow_teacher_fields_for_stub", action="store_true", help="Allow teacher fields in dry-run mode (for diagnostics only)")

    # Backend selection (Milestone 4)
    parser.add_argument("--policy_backend", type=str, default="stub", choices=["stub", "local_hf"], help="Policy backend type")
    parser.add_argument("--memory_policy_backend", type=str, choices=["stub", "local_hf"], help="Memory Builder backend (overrides --policy_backend)")
    parser.add_argument("--question_policy_backend", type=str, choices=["stub", "local_hf"], help="Question Agent backend (overrides --policy_backend)")

    # Local HF backend configuration
    parser.add_argument("--memory_model_path", type=str, help="Path to Memory Builder model (required for local_hf backend)")
    parser.add_argument("--question_model_path", type=str, help="Path to Question Agent model (required for local_hf backend)")
    parser.add_argument("--memory_tokenizer_path", type=str, help="Path to Memory Builder tokenizer (defaults to model path)")
    parser.add_argument("--question_tokenizer_path", type=str, help="Path to Question Agent tokenizer (defaults to model path)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow custom HuggingFace model code for local_hf backends")

    # Training hyperparameters
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Micro batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--advantage_clip", type=float, default=5.0, help="Advantage clipping value")

    # LoRA configuration
    parser.add_argument("--train_lora", action="store_true", help="Train with LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, help="Comma-separated LoRA target modules")

    args = parser.parse_args()

    # Determine backends
    memory_backend = args.memory_policy_backend or args.policy_backend
    question_backend = args.question_policy_backend or args.policy_backend

    # Validate local_hf requirements
    if memory_backend == "local_hf" and not args.memory_model_path:
        parser.error("--memory_model_path is required when using local_hf backend for Memory Builder")

    if question_backend == "local_hf" and not args.question_model_path:
        parser.error("--question_model_path is required when using local_hf backend for Question Agent")

    # Build local HF configs
    memory_hf_config = None
    question_hf_config = None

    if memory_backend == "local_hf":
        memory_hf_config = {
            "model_path": args.memory_model_path,
            "tokenizer_path": args.memory_tokenizer_path,
            "device": args.device,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_seq_length": args.max_seq_length,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "advantage_clip": args.advantage_clip,
            "train_lora": args.train_lora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "trust_remote_code": args.trust_remote_code,
        }
        if args.lora_target_modules:
            memory_hf_config["lora_target_modules"] = args.lora_target_modules.split(",")

    if question_backend == "local_hf":
        question_hf_config = {
            "model_path": args.question_model_path,
            "tokenizer_path": args.question_tokenizer_path,
            "device": args.device,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_seq_length": args.max_seq_length,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "advantage_clip": args.advantage_clip,
            "train_lora": args.train_lora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "trust_remote_code": args.trust_remote_code,
        }
        if args.lora_target_modules:
            question_hf_config["lora_target_modules"] = args.lora_target_modules.split(",")

    # Build policy clients
    memory_policy = build_policy_client(
        role=ActorRole.MEMORY_BUILDER,
        backend=memory_backend,
        policy_id=args.memory_policy_id,
        local_hf_config=memory_hf_config,
    )

    question_policy = build_policy_client(
        role=ActorRole.QUESTION_AGENT,
        backend=question_backend,
        policy_id=args.question_policy_id,
        local_hf_config=question_hf_config,
    )

    # Create config
    config = GRPOTrainerConfig(
        rollout_dir=args.rollout_dir,
        output_dir=args.output_dir,
        round_id=args.round_id,
        step_id=args.step_id,
        record_id=args.record_id,
        max_records=args.max_records,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
        require_both_actor_batches=args.require_both_actor_batches,
        reconstruct_missing_batches=args.reconstruct_missing_batches,
        selected_only=args.selected_only,
        memory_policy_id=args.memory_policy_id,
        question_policy_id=args.question_policy_id,
        write_checkpoints=args.write_checkpoints,
        allow_teacher_fields_for_stub=args.allow_teacher_fields_for_stub,
    )

    # Run trainer
    print("=" * 80)
    print("Phase 2 Milestone 3/4: Local GRPO Trainer")
    print("=" * 80)
    print(f"Rollout dir: {config.rollout_dir}")
    print(f"Output dir: {config.output_dir}")
    print(f"Dry run: {config.dry_run}")
    print(f"Memory policy backend: {memory_backend}")
    print(f"Question policy backend: {question_backend}")
    print(f"Memory policy ID: {config.memory_policy_id}")
    print(f"Question policy ID: {config.question_policy_id}")
    if memory_backend == "local_hf":
        print(f"Memory model: {args.memory_model_path}")
        print(f"Device: {args.device}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Train LoRA: {args.train_lora}")
    if question_backend == "local_hf":
        print(f"Question model: {args.question_model_path}")
    print(f"Reconstruct missing batches: {config.reconstruct_missing_batches}")
    print(f"Selected only: {config.selected_only}")
    print("=" * 80)

    trainer = LocalGRPOTrainer(config, memory_policy=memory_policy, question_policy=question_policy)
    summary = trainer.run()

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total records: {summary.total_records}")
    print(f"  Succeeded: {summary.succeeded_records}")
    print(f"  Partial: {summary.partial_records}")
    print(f"  Skipped: {summary.skipped_records}")
    print(f"  Failed: {summary.failed_records}")
    if summary.average_memory_reward is not None:
        print(f"  Avg memory reward: {summary.average_memory_reward:.3f}")
    if summary.average_question_reward is not None:
        print(f"  Avg question reward: {summary.average_question_reward:.3f}")
    if summary.average_memory_advantage is not None:
        print(f"  Avg memory advantage: {summary.average_memory_advantage:.3f}")
    if summary.average_question_advantage is not None:
        print(f"  Avg question advantage: {summary.average_question_advantage:.3f}")
    print(f"  Manifest: {summary.manifest_path}")
    print("=" * 80)

    # Return exit code
    if summary.failed_records > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit(main())
