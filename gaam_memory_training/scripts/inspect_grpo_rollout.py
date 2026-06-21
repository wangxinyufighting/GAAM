#!/usr/bin/env python3
"""
CLI for Phase 2 Milestone 2: Local Adversarial Rollout.

This script runs the local adversarial rollout pipeline:
1. Sample K memory candidates per case
2. Sample M question-set candidates per case
3. Evaluate all valid memory × question pairs
4. Compute GRPO advantages
5. Write rollout traces and optional actor update batches
"""

import argparse
from pathlib import Path

from gaam_graph.adversarial_rollout import (
    AdversarialRolloutConfig,
    LocalAdversarialRolloutRunner,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run local adversarial rollout for GRPO-style adversarial co-training"
    )

    # Input/output
    parser.add_argument("--input", type=Path, required=True, help="Path to LongMemEval JSON")
    parser.add_argument("--graph_dir", type=Path, required=True, help="Directory with oracle graphs")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for rollout artifacts")

    # Record filtering
    parser.add_argument("--record_id", type=str, help="Process only this record ID")
    parser.add_argument("--max_records", type=int, help="Maximum number of records to process")

    # Round/step IDs
    parser.add_argument("--round_id", type=int, default=0, help="Training round ID")
    parser.add_argument("--step_id", type=int, default=0, help="Training step ID")

    # Sampling configuration
    parser.add_argument("--num_memory_samples", type=int, default=3, help="Number of memory samples per case")
    parser.add_argument("--num_question_samples", type=int, default=3, help="Number of question samples per case")
    parser.add_argument("--base_seed", type=int, default=42, help="Base random seed")

    # LLM flags
    parser.add_argument("--no_llm_memory", action="store_true", default=True, help="Use baseline memory builder (no LLM)")
    parser.add_argument("--no_llm_question", action="store_true", help="Use no-LLM question generation")
    parser.add_argument("--no_llm_answer", action="store_true", help="Use no-LLM answerer")

    # Answer configuration
    parser.add_argument("--answer_top_k", type=int, default=8, help="Top-k for answer retrieval")
    parser.add_argument("--answer_model", type=str, help="Model for answer generation")
    parser.add_argument("--answer_temperature", type=float, default=0.0, help="Temperature for answer generation")
    parser.add_argument("--max_answer_chars", type=int, default=500, help="Max characters for answer")

    # Reward configuration
    parser.add_argument("--correctness_mode", type=str, default="heuristic", help="Correctness evaluation mode")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")

    # Weakness book
    parser.add_argument("--update_weakness_book", action="store_true", help="Update weakness book after scoring")
    parser.add_argument("--no_update_weakness_book", dest="update_weakness_book", action="store_false")
    parser.set_defaults(update_weakness_book=True)
    parser.add_argument("--weakness_book_dir", type=Path, help="Directory for weakness books")

    # GRPO configuration
    parser.add_argument("--normalize_advantages", action="store_true", default=True, help="Normalize advantages by std")
    parser.add_argument("--no_normalize_advantages", dest="normalize_advantages", action="store_false")
    parser.add_argument("--skip_zero_variance", action="store_true", default=True, help="Skip zero-variance groups")
    parser.add_argument("--no_skip_zero_variance", dest="skip_zero_variance", action="store_false")

    # Output configuration
    parser.add_argument("--write_actor_update_batches", action="store_true", default=True, help="Write actor update batches")
    parser.add_argument("--no_write_actor_update_batches", dest="write_actor_update_batches", action="store_false")

    # Execution flags
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first error")

    args = parser.parse_args()

    # Check overwrite
    if args.output_dir.exists() and not args.overwrite:
        print(f"Error: Output directory exists: {args.output_dir}")
        print("Use --overwrite to overwrite")
        return 1

    # Create config
    config = AdversarialRolloutConfig(
        input_path=args.input,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        round_id=args.round_id,
        step_id=args.step_id,
        record_id=args.record_id,
        max_records=args.max_records,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
        num_memory_samples=args.num_memory_samples,
        num_question_samples=args.num_question_samples,
        base_seed=args.base_seed,
        no_llm_memory=args.no_llm_memory,
        no_llm_question=args.no_llm_question,
        no_llm_answer=args.no_llm_answer,
        answer_top_k=args.answer_top_k,
        answer_model=args.answer_model,
        answer_temperature=args.answer_temperature,
        max_answer_chars=args.max_answer_chars,
        correctness_mode=args.correctness_mode,
        split=args.split,
        update_weakness_book=args.update_weakness_book,
        weakness_book_dir=args.weakness_book_dir,
        normalize_advantages=args.normalize_advantages,
        skip_zero_variance=args.skip_zero_variance,
        write_actor_update_batches=args.write_actor_update_batches,
    )

    # Run rollout
    print("=" * 80)
    print("Phase 2 Milestone 2: Local Adversarial Rollout")
    print("=" * 80)
    print(f"Input: {config.input_path}")
    print(f"Graph dir: {config.graph_dir}")
    print(f"Output dir: {config.output_dir}")
    print(f"Memory samples: {config.num_memory_samples}")
    print(f"Question samples: {config.num_question_samples}")
    print(f"No-LLM modes: memory={config.no_llm_memory}, question={config.no_llm_question}, answer={config.no_llm_answer}")
    print("=" * 80)

    runner = LocalAdversarialRolloutRunner(config)
    summary = runner.run()

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total steps: {summary.total_steps}")
    print(f"  Succeeded: {summary.succeeded_steps}")
    print(f"  Failed: {summary.failed_steps}")
    print(f"  Skipped: {summary.skipped_steps}")
    print(f"  Trace dir: {summary.trace_dir}")
    print("=" * 80)

    # Return exit code
    if summary.failed_steps > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit(main())
