#!/usr/bin/env python3
"""
Answer questions from current memory.

Usage:
    python scripts/answer_from_memory.py \\
      --current_memory outputs/current_memory/e47becba.current_memory.json \\
      --question "What does the user prefer?" \\
      --question_id manual_q_001 \\
      --output outputs/answers/e47becba.answers.json \\
      --no_llm
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.answer_agent import FrozenAnswerer
from gaam_graph.answer_schema import AnswerRequest, answer_request_from_generated_question
from gaam_graph.llm import LLMError, OpenAICompatibleLLM
from gaam_graph.memory_retriever import CurrentMemoryRetriever
from gaam_graph.utils import read_json, write_json


def main():
    parser = argparse.ArgumentParser(
        description="Answer questions from current memory"
    )

    # Input
    parser.add_argument(
        "--current_memory",
        required=True,
        type=Path,
        help="Path to current memory JSON"
    )

    # Question input (one of these)
    question_group = parser.add_mutually_exclusive_group(required=True)
    question_group.add_argument(
        "--question",
        type=str,
        help="Single question text"
    )
    question_group.add_argument(
        "--question_file",
        type=Path,
        help="Path to accepted questions JSON"
    )

    parser.add_argument(
        "--question_id",
        type=str,
        help="Question ID (required if --question is used)"
    )

    # Output
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for answers JSON"
    )

    # Answerer config
    parser.add_argument("--no_llm", action="store_true", help="Use no-LLM mode")
    parser.add_argument("--top_k", type=int, default=8, help="Top K retrieval (default: 8)")
    parser.add_argument("--include_summaries", action="store_true", default=True, help="Include summaries")
    parser.add_argument("--model", type=str, help="Model identifier")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (default: 0.0)")
    parser.add_argument("--max_answer_chars", type=int, help="Max answer length")

    # Other
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")

    args = parser.parse_args()

    # Validate inputs
    if not args.current_memory.exists():
        print(f"Error: Current memory not found: {args.current_memory}", file=sys.stderr)
        return 1

    if args.question and not args.question_id:
        print("Error: --question_id is required when using --question", file=sys.stderr)
        return 1

    if args.question_file and not args.question_file.exists():
        print(f"Error: Question file not found: {args.question_file}", file=sys.stderr)
        return 1

    if args.output.exists() and not args.overwrite:
        print(f"Error: Output exists (use --overwrite): {args.output}", file=sys.stderr)
        return 1

    # Load current memory
    print(f"Loading current memory from {args.current_memory}")
    current_memory = read_json(args.current_memory)
    record_id = current_memory["record_id"]

    # Create requests
    requests = []

    if args.question:
        # Single manual question
        request = AnswerRequest(
            record_id=record_id,
            question_id=args.question_id,
            question=args.question,
            current_memory_path=str(args.current_memory),
        )
        requests.append(request)

    else:
        # Load question file
        question_data = read_json(args.question_file)
        if not isinstance(question_data, dict) or "questions" not in question_data:
            print(
                f"Error: Question file must be an object with a 'questions' list: {args.question_file}",
                file=sys.stderr,
            )
            return 1

        questions = question_data["questions"]
        if not isinstance(questions, list):
            print(f"Error: 'questions' must be a list: {args.question_file}", file=sys.stderr)
            return 1
        if not questions:
            print(f"Error: Question file contains no questions: {args.question_file}", file=sys.stderr)
            return 1

        print(f"Loaded {len(questions)} questions from {args.question_file}")

        for i, q in enumerate(questions):
            # Sanitize: drop answer and oracle supporting IDs
            try:
                request = answer_request_from_generated_question(q)
            except Exception as exc:
                print(f"Error: Invalid question at index {i}: {exc}", file=sys.stderr)
                return 1
            request.current_memory_path = str(args.current_memory)
            requests.append(request)

    print(f"Answering {len(requests)} questions...")

    # Initialize answerer
    retriever = CurrentMemoryRetriever(
        top_k=args.top_k,
        include_summaries=args.include_summaries,
    )
    llm = None
    if not args.no_llm:
        llm = OpenAICompatibleLLM(temperature=args.temperature)
        if args.model:
            llm.model = args.model

    answerer = FrozenAnswerer(
        llm=llm,
        no_llm=args.no_llm,
        retriever=retriever,
        model_id=args.model or ("no_llm_baseline" if args.no_llm else llm.model),
        max_answer_chars=args.max_answer_chars or 500,
    )

    # Answer questions
    try:
        reports = answerer.batch_answer(
            current_memory=current_memory,
            requests=requests,
        )
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: Failed to answer questions: {exc}", file=sys.stderr)
        return 1

    # Compute summary statistics
    answered_count = sum(1 for r in reports if r.answer_status == "answered")
    insufficient_count = sum(1 for r in reports if r.answer_status == "insufficient_evidence")
    avg_confidence = (
        sum(r.confidence for r in reports) / len(reports)
        if reports else 0.0
    )

    # Prepare output
    output_data = {
        "record_id": record_id,
        "current_memory_path": str(args.current_memory),
        "answerer_config": {
            "backend": "no_llm" if args.no_llm else "llm",
            "model": args.model or ("no_llm_baseline" if args.no_llm else llm.model),
            "temperature": args.temperature,
            "top_k": args.top_k,
            "include_summaries": args.include_summaries,
        },
        "answers": [r.model_dump(mode="json") for r in reports],
        "summary": {
            "num_questions": len(reports),
            "answered_count": answered_count,
            "insufficient_evidence_count": insufficient_count,
            "average_confidence": round(avg_confidence, 3),
        },
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output_data)

    print(f"\nSummary:")
    print(f"  Questions: {len(reports)}")
    print(f"  Answered: {answered_count}")
    print(f"  Insufficient evidence: {insufficient_count}")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  Output: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
