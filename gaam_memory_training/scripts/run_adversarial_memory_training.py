#!/usr/bin/env python3
"""
Run adversarial memory training - one round.

Usage:
    python scripts/run_adversarial_memory_training.py \
      --input data/longmemeval/longmemeval_s_cleaned.json \
      --graph_dir outputs/no_leak_smoke_test \
      --output_dir outputs/adversarial_training/round_000 \
      --max_records 1 \
      --round_id 0 \
      --no_llm_question \
      --no_llm_answer \
      --allow_placeholder_questions \
      --overwrite
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path so this CLI works when run as
# `python scripts/run_adversarial_memory_training.py` from gaam_memory_training.
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.training_loop import OneRoundTrainingConfig, OneRoundTrainingLoop


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run one-round adversarial memory training")

    # Required inputs
    parser.add_argument("--input", type=Path, required=True, help="Input LongMemEval JSON file")
    parser.add_argument("--graph_dir", type=Path, required=True, help="Directory containing oracle graphs")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for round artifacts")

    # Round config
    parser.add_argument("--round_id", type=int, default=0, help="Round ID")
    parser.add_argument("--record_id", type=str, help="Process single record ID only")
    parser.add_argument("--max_records", type=int, help="Maximum records to process")

    # Execution control
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first record failure")

    # Memory builder config
    parser.add_argument("--include_assistant_turns", action="store_true", help="Include assistant turns")
    parser.add_argument("--min_user_text_chars", type=int, default=20, help="Min user text chars")
    parser.add_argument("--max_event_nodes", type=int, help="Max event nodes")

    # Question generation config
    parser.add_argument("--num_walks", type=int, default=8, help="Number of graph walks")
    parser.add_argument("--walk_length", type=int, default=5, help="Walk length")
    parser.add_argument("--num_questions", type=int, default=5, help="Number of questions to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--directed", action="store_true", help="Use directed walks")
    parser.add_argument("--require_multi_session", action="store_true", help="Require multi-session trajectories")
    parser.add_argument("--min_sessions", type=int, default=2, help="Min sessions for multi-session")
    parser.add_argument("--max_structural_ratio", type=float, default=0.35, help="Max structural edge ratio")
    parser.add_argument("--max_consecutive_next", type=int, default=1, help="Max consecutive NEXT edges")
    parser.add_argument("--question_types", nargs="+", help="Question types to generate")
    parser.add_argument("--accept_revise", action="store_true", help="Accept REVISE verdict questions")
    parser.add_argument("--allow_placeholder_questions", action="store_true", help="Allow placeholder questions")
    parser.add_argument("--allow_empty_question_set", action="store_true", help="Allow empty question set")

    # LLM mode
    parser.add_argument("--no_llm_question", action="store_true", help="No-LLM mode for question generation")
    parser.add_argument("--no_llm_answer", action="store_true", help="No-LLM mode for answering")

    # Answerer config
    parser.add_argument("--answer_top_k", type=int, default=8, help="Top K retrieval")
    parser.add_argument("--answer_model", type=str, help="Answer model ID")
    parser.add_argument("--answer_temperature", type=float, default=0.0, help="Answer temperature")
    parser.add_argument("--max_answer_chars", type=int, default=500, help="Max answer chars")

    # Reward config
    parser.add_argument("--correctness_mode", type=str, default="heuristic", help="Correctness mode")
    parser.add_argument("--split", type=str, default="train", help="Question split")
    parser.add_argument("--weakness_book_dir", type=Path, help="Existing weakness book directory")
    parser.add_argument("--write_question_agent_reward", action="store_true", default=True, help="Write QA reward")

    args = parser.parse_args()

    # Create config
    config = OneRoundTrainingConfig(
        input_path=args.input,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        round_id=args.round_id,
        record_id=args.record_id,
        max_records=args.max_records,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
        include_assistant_turns=args.include_assistant_turns,
        min_user_text_chars=args.min_user_text_chars,
        max_event_nodes=args.max_event_nodes,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        num_questions=args.num_questions,
        seed=args.seed,
        directed=args.directed,
        require_multi_session=args.require_multi_session,
        min_sessions=args.min_sessions,
        max_structural_ratio=args.max_structural_ratio,
        max_consecutive_next=args.max_consecutive_next,
        question_types=args.question_types,
        accept_revise=args.accept_revise,
        allow_placeholder_questions=args.allow_placeholder_questions,
        allow_empty_question_set=args.allow_empty_question_set,
        no_llm_question=args.no_llm_question,
        no_llm_answer=args.no_llm_answer,
        answer_top_k=args.answer_top_k,
        answer_model=args.answer_model,
        answer_temperature=args.answer_temperature,
        max_answer_chars=args.max_answer_chars,
        correctness_mode=args.correctness_mode,
        split=args.split,
        weakness_book_dir=args.weakness_book_dir,
        write_question_agent_reward=args.write_question_agent_reward,
    )

    # Run training loop
    try:
        loop = OneRoundTrainingLoop(config)
        summary = loop.run()

        print(f"\n{'='*60}")
        print(f"Training Round {summary.round_id} Complete")
        print(f"{'='*60}")
        print(f"Attempted: {summary.attempted_records}")
        print(f"Succeeded: {summary.succeeded_records}")
        print(f"Failed: {summary.failed_records}")
        print(f"Skipped: {summary.skipped_records}")
        if summary.average_memory_update_reward is not None:
            print(f"Average Memory Update Reward: {summary.average_memory_update_reward:.3f}")
        if summary.average_question_agent_reward is not None:
            print(f"Average Question Agent Reward: {summary.average_question_agent_reward:.3f}")
        print(f"\nOutput: {args.output_dir}")
        print(f"Manifest: {summary.manifest_path}")
        print(f"{'='*60}\n")

        return 0 if summary.failed_records == 0 else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
