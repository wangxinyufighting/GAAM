#!/usr/bin/env python3
"""
Judge Current Memory CLI.

Reads existing artifacts and writes reward reports and weakness book.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path so the CLI works when run as
# `python scripts/judge_current_memory.py` from gaam_memory_training.
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.memory_weakness_book import MemoryWeaknessBook
from gaam_graph.oracle_graph_loader import OracleGraphLoader
from gaam_graph.reward_manager import RewardManager
from gaam_graph.utils import read_json, write_json


def main():
    parser = argparse.ArgumentParser(
        description="Judge current memory and produce reward reports"
    )

    # Required inputs
    parser.add_argument(
        "--graph",
        required=True,
        help="Path to oracle graph JSON"
    )
    parser.add_argument(
        "--current_memory",
        required=True,
        help="Path to current memory JSON"
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to accepted questions JSON"
    )
    parser.add_argument(
        "--answers",
        required=True,
        help="Path to answers JSON"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for reward reports"
    )

    # Optional inputs
    parser.add_argument(
        "--weakness_book",
        help="Path to existing weakness book JSON"
    )
    parser.add_argument(
        "--round_id",
        type=int,
        default=0,
        help="Current round ID"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "heldout", "diagnostic"],
        help="Question split"
    )
    parser.add_argument(
        "--correctness_mode",
        default="heuristic",
        choices=["heuristic", "llm_judge"],
        help="Correctness scoring mode"
    )
    parser.add_argument(
        "--write_question_agent_reward",
        action="store_true",
        help="Also write question agent reward report"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs"
    )

    args = parser.parse_args()

    # Validate inputs exist
    graph_path = Path(args.graph)
    memory_path = Path(args.current_memory)
    questions_path = Path(args.questions)
    answers_path = Path(args.answers)

    if not graph_path.exists():
        print(f"Error: Graph file not found: {graph_path}", file=sys.stderr)
        return 1

    if not memory_path.exists():
        print(f"Error: Current memory file not found: {memory_path}", file=sys.stderr)
        return 1

    if not questions_path.exists():
        print(f"Error: Questions file not found: {questions_path}", file=sys.stderr)
        return 1

    if not answers_path.exists():
        print(f"Error: Answers file not found: {answers_path}", file=sys.stderr)
        return 1

    # Load artifacts
    print(f"Loading oracle graph: {graph_path}")
    try:
        oracle_graph = OracleGraphLoader(graph_path)
    except Exception as exc:
        print(f"Error: Failed to load oracle graph: {exc}", file=sys.stderr)
        return 1

    print(f"Loading current memory: {memory_path}")
    current_memory = read_json(memory_path)

    print(f"Loading questions: {questions_path}")
    questions_data = read_json(questions_path)

    print(f"Loading answers: {answers_path}")
    answers_data = read_json(answers_path)

    # Extract record ID
    record_id = current_memory.get("record_id")
    if not record_id:
        print("Error: Current memory missing record_id", file=sys.stderr)
        return 1

    # Validate record ID consistency
    if questions_data.get("record_id") != record_id:
        print(
            f"Error: Record ID mismatch - memory: {record_id}, questions: {questions_data.get('record_id')}",
            file=sys.stderr
        )
        return 1

    if answers_data.get("record_id") != record_id:
        print(
            f"Error: Record ID mismatch - memory: {record_id}, answers: {answers_data.get('record_id')}",
            file=sys.stderr
        )
        return 1

    if oracle_graph.record_id != record_id:
        print(
            f"Error: Record ID mismatch - memory: {record_id}, graph: {oracle_graph.record_id}",
            file=sys.stderr
        )
        return 1

    # Extract questions and answers lists
    questions = questions_data.get("questions", [])
    answers = answers_data.get("answers", [])

    if not questions:
        print("Error: No questions found in questions file", file=sys.stderr)
        return 1

    if not answers:
        print("Error: No answers found in answers file", file=sys.stderr)
        return 1

    # Validate question IDs match
    question_ids = {q["question_id"] for q in questions}
    answer_ids = {a["question_id"] for a in answers}

    if question_ids != answer_ids:
        missing_in_answers = question_ids - answer_ids
        extra_in_answers = answer_ids - question_ids

        if missing_in_answers:
            print(
                f"Error: Questions missing answers: {missing_in_answers}",
                file=sys.stderr
            )
            return 1

        if extra_in_answers:
            print(
                f"Error: Extra answers without questions: {extra_in_answers}",
                file=sys.stderr
            )
            return 1

    # Load or create weakness book
    weakness_book = None
    if args.weakness_book:
        weakness_book_path = Path(args.weakness_book)
        if weakness_book_path.exists():
            print(f"Loading weakness book: {weakness_book_path}")
            weakness_book = MemoryWeaknessBook.load(
                weakness_book_path,
                record_id=record_id
            )
        else:
            print(f"Warning: Weakness book not found, creating new: {weakness_book_path}")
            weakness_book = MemoryWeaknessBook.empty(record_id)
    else:
        weakness_book = MemoryWeaknessBook.empty(record_id)

    # Create RewardManager
    reward_manager = RewardManager(
        correctness_mode=args.correctness_mode,
        weakness_book=weakness_book,
    )

    # Score Memory Builder
    print(f"\nScoring Memory Builder for {record_id}...")
    memory_reward = reward_manager.score_memory_builder(
        current_memory=current_memory,
        questions=questions,
        answer_reports=answers,
        oracle_graph=oracle_graph,
        split=args.split,
    )

    print(f"  Total reward: {memory_reward.total_reward:.3f}")
    print(f"  Update reward: {memory_reward.update_reward:.3f}")
    print(f"  Monitoring score: {memory_reward.monitoring_score:.3f}")
    print(f"  Questions: {len(memory_reward.question_items)}")
    print(f"  Failures: {sum(memory_reward.failure_summary.values())}")

    # Update weakness book
    print("\nUpdating weakness book...")
    weakness_updates = weakness_book.update_from_reward_report(
        memory_reward,
        round_id=args.round_id,
    )
    memory_reward.weakness_updates = [
        weakness.model_dump(mode="json") for weakness in weakness_updates
    ]
    print(f"  Updated weaknesses: {len(weakness_updates)}")
    print(f"  Active weaknesses: {len(weakness_book.get_active())}")

    question_agent_reward = None
    if args.write_question_agent_reward:
        validity_reports = questions_data.get("validity_reports", [])
        if not validity_reports:
            print(
                "Error: --write_question_agent_reward requires validity_reports in the questions file",
                file=sys.stderr,
            )
            return 1

        print("\nScoring Question Agent...")
        question_agent_reward = reward_manager.score_question_agent(
            questions=questions,
            validity_reports=validity_reports,
            answer_reports=answers,
            oracle_graph=oracle_graph,
        )
        print(f"  Question Agent reward: {question_agent_reward.total_reward:.3f}")

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    reward_report_path = output_dir / f"{record_id}.reward_report.json"
    question_agent_reward_path = output_dir / f"{record_id}.question_agent_reward_report.json"
    weakness_book_path = output_dir / f"{record_id}.weakness_book.json"
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"

    # Check overwrite
    if not args.overwrite:
        if reward_report_path.exists():
            print(f"Error: Output exists (use --overwrite): {reward_report_path}", file=sys.stderr)
            return 1
        if question_agent_reward and question_agent_reward_path.exists():
            print(f"Error: Output exists (use --overwrite): {question_agent_reward_path}", file=sys.stderr)
            return 1

    # Write reward report
    print(f"\nWriting reward report: {reward_report_path}")
    reward_report_dict = memory_reward.model_dump(mode="json")
    write_json(reward_report_path, reward_report_dict)

    if question_agent_reward:
        print(f"Writing question agent reward report: {question_agent_reward_path}")
        write_json(question_agent_reward_path, question_agent_reward.model_dump(mode="json"))

    # Write weakness book
    print(f"Writing weakness book: {weakness_book_path}")
    weakness_book.save(weakness_book_path)

    # Write manifest entry
    manifest_entry = {
        "record_id": record_id,
        "graph_path": str(graph_path),
        "current_memory_path": str(memory_path),
        "questions_path": str(questions_path),
        "answers_path": str(answers_path),
        "reward_report_path": str(reward_report_path),
        "question_agent_reward_report_path": str(question_agent_reward_path) if question_agent_reward else "",
        "weakness_book_path": str(weakness_book_path),
        "status": "ok",
    }

    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

    print(f"Appended to manifest: {manifest_path}")

    # Update summary
    summary = {
        "num_records": 1,
        "succeeded": 1,
        "failed": 0,
        "average_update_reward": memory_reward.update_reward,
        "average_answer_utility": next(
            (c.score for c in memory_reward.components if c.name == "answer_utility"),
            0.0
        ),
        "average_question_agent_reward": question_agent_reward.total_reward if question_agent_reward else 0.0,
        "failure_type_counts": memory_reward.failure_summary,
    }

    # Merge with existing summary if exists
    if summary_path.exists():
        existing_summary = read_json(summary_path)
        summary["num_records"] = existing_summary.get("num_records", 0) + 1
        summary["succeeded"] = existing_summary.get("succeeded", 0) + 1

        # Update averages
        n = summary["num_records"]
        old_reward = existing_summary.get("average_update_reward", 0.0)
        summary["average_update_reward"] = (
            (old_reward * (n - 1) + memory_reward.update_reward) / n
        )
        old_qa_reward = existing_summary.get("average_question_agent_reward", 0.0)
        summary["average_question_agent_reward"] = (
            (old_qa_reward * (n - 1) + summary["average_question_agent_reward"]) / n
        )

        # Merge failure counts
        for ft, count in memory_reward.failure_summary.items():
            summary["failure_type_counts"][ft] = (
                existing_summary.get("failure_type_counts", {}).get(ft, 0) + count
            )

    write_json(summary_path, summary)
    print(f"Updated summary: {summary_path}")

    print("\n✅ Judgment complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
