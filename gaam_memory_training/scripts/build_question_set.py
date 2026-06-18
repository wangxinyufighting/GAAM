#!/usr/bin/env python3
"""
Build question sets from oracle graphs.

Usage:
    python scripts/build_question_set.py \\
      --graph outputs/no_leak_smoke_test/e47becba.graph.json \\
      --output_dir outputs/question_sets \\
      --num_walks 8 \\
      --walk_length 5 \\
      --num_questions 5 \\
      --no_llm
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gaam_graph.oracle_graph_loader import OracleGraphLoader
from gaam_graph.oracle_validity_checker import OracleValidityChecker
from gaam_graph.question_agent import QuestionAgent
from gaam_graph.llm import OpenAICompatibleLLM
from gaam_graph.question_schema import (
    OracleValidityVerdict,
    QuestionStatus,
)
from gaam_graph.utils import read_json, write_json


def process_graph(graph_path: Path, args) -> dict:
    """Process a single graph file."""
    print(f"Processing {graph_path.name}")

    # Load oracle graph
    loader = OracleGraphLoader(graph_path, strict=True)
    record_id = loader.record_id

    # Load graph dict for walker
    graph_dict = read_json(graph_path)

    # Initialize agent
    llm = None if args.no_llm else OpenAICompatibleLLM()
    agent = QuestionAgent(llm=llm, no_llm=args.no_llm)

    # Sample trajectories
    print(f"  Sampling {args.num_walks} trajectories...")
    trajectories = agent.sample_trajectories(
        graph=graph_dict,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        seed=args.seed,
        directed=args.directed,
        require_multi_session=args.require_multi_session,
        min_sessions=args.min_sessions,
        max_structural_ratio=args.max_structural_ratio,
        max_consecutive_next=args.max_consecutive_next,
    )

    print(f"  Generated {len(trajectories)} trajectories")

    # Generate candidate questions
    print(f"  Generating {args.num_questions} candidate questions...")
    desired_types = args.question_types.split(",") if args.question_types else None

    candidates = agent.generate_candidates(
        record_id=record_id,
        graph_path=str(graph_path),
        trajectories=trajectories,
        num_questions=args.num_questions,
        require_multi_session=args.require_multi_session,
        min_sessions=args.min_sessions,
        desired_question_types=desired_types,
    )

    print(f"  Generated {len(candidates)} candidates")

    # Initialize validity checker
    checker = OracleValidityChecker(
        oracle_loader=loader,
        allow_placeholder_questions=args.allow_placeholder_questions,
    )

    # Validate questions
    print(f"  Validating questions...")
    reports = checker.validate_questions(
        questions=candidates,
        trajectories=trajectories,
        require_multi_session=args.require_multi_session,
        min_sessions=args.min_sessions,
    )

    # Select accepted questions
    accepted = []
    for question, report in zip(candidates, reports):
        if report.verdict == OracleValidityVerdict.ACCEPT:
            question.status = QuestionStatus.ACCEPTED
            accepted.append(question)
        elif report.verdict == OracleValidityVerdict.REVISE and args.accept_revise:
            question.status = QuestionStatus.REVISED
            accepted.append(question)
        else:
            question.status = QuestionStatus.REJECTED

    print(f"  Accepted: {len(accepted)}, Rejected: {len(candidates) - len(accepted)}")

    # Write outputs
    output_dir = args.output_dir / (args.output_subdir or "")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Candidate file
    candidate_path = output_dir / f"{record_id}.candidate_questions.json"
    candidate_data = {
        "record_id": record_id,
        "graph_path": str(graph_path),
        "generation_config": {
            "num_walks": args.num_walks,
            "walk_length": args.walk_length,
            "num_questions": args.num_questions,
            "require_multi_session": args.require_multi_session,
            "min_sessions": args.min_sessions,
            "no_llm": args.no_llm,
        },
        "trajectories": trajectories,
        "questions": [q.model_dump(mode="json") for q in candidates],
    }
    write_json(candidate_path, candidate_data)

    # Validity file
    validity_path = output_dir / f"{record_id}.validity_reports.json"
    validity_data = {
        "record_id": record_id,
        "validity_config": {
            "allow_placeholder_questions": args.allow_placeholder_questions,
        },
        "reports": [r.model_dump(mode="json") for r in reports],
    }
    write_json(validity_path, validity_data)

    # Accepted file
    accepted_path = output_dir / f"{record_id}.accepted_questions.json"

    # Count question types
    question_type_counts = {}
    for q in accepted:
        qt = q.question_type.value
        question_type_counts[qt] = question_type_counts.get(qt, 0) + 1
    accepted_question_ids = {q.question_id for q in accepted}

    accepted_data = {
        "record_id": record_id,
        "graph_path": str(graph_path),
        "questions": [q.model_dump(mode="json") for q in accepted],
        "validity_reports": [
            r.model_dump(mode="json") for r, q in zip(reports, candidates)
            if q.question_id in accepted_question_ids
        ],
        "summary": {
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "rejected_count": len(candidates) - len(accepted),
            "question_type_counts": question_type_counts,
        },
    }
    write_json(accepted_path, accepted_data)

    return {
        "record_id": record_id,
        "graph_path": str(graph_path),
        "candidate_path": str(candidate_path),
        "validity_path": str(validity_path),
        "accepted_path": str(accepted_path),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build question sets from oracle graphs"
    )

    # Input/output
    graph_group = parser.add_mutually_exclusive_group(required=True)
    graph_group.add_argument("--graph", type=Path, help="Path to single .graph.json file")
    graph_group.add_argument("--graph_dir", type=Path, help="Directory containing .graph.json files")

    parser.add_argument("--output_dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--output_subdir", type=str, help="Optional subdirectory name")

    # Trajectory sampling
    parser.add_argument("--num_walks", type=int, default=8, help="Number of walks (default: 8)")
    parser.add_argument("--walk_length", type=int, default=5, help="Walk length (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--directed", action="store_true", help="Use directed walks")

    # Question generation
    parser.add_argument("--num_questions", type=int, default=5, help="Number of questions (default: 5)")
    parser.add_argument("--no_llm", action="store_true", help="Use no-LLM mode (deterministic)")
    parser.add_argument("--question_types", type=str, help="Comma-separated desired question types")

    # Multi-session
    parser.add_argument("--require_multi_session", action="store_true", help="Require multi-session evidence")
    parser.add_argument("--min_sessions", type=int, default=2, help="Min sessions for multi-session (default: 2)")

    # Walker config
    parser.add_argument("--max_structural_ratio", type=float, default=0.35, help="Max structural ratio (default: 0.35)")
    parser.add_argument("--max_consecutive_next", type=int, default=1, help="Max consecutive NEXT edges (default: 1)")

    # Validity
    parser.add_argument("--accept_revise", action="store_true", help="Accept questions with 'revise' verdict")
    parser.add_argument("--allow_placeholder_questions", action="store_true", help="Allow no-LLM placeholder questions")

    # Processing
    parser.add_argument("--max_records", type=int, help="Max records to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    args = parser.parse_args()

    # Collect graph files
    if args.graph:
        graph_files = [args.graph]
    else:
        graph_files = sorted(args.graph_dir.glob("*.graph.json"))

    if args.max_records:
        graph_files = graph_files[:args.max_records]

    print(f"Processing {len(graph_files)} graphs")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process graphs
    manifest_path = args.output_dir / "manifest.jsonl"
    results = []
    failed_results = []
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        for i, graph_path in enumerate(graph_files, 1):
            print(f"\n[{i}/{len(graph_files)}] {graph_path.name}")

            try:
                result = process_graph(graph_path, args)
                manifest_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                manifest_file.flush()
                results.append(result)

            except Exception as e:
                print(f"  ✗ Failed: {e}", file=sys.stderr)
                result = {
                    "record_id": graph_path.stem.replace(".graph", ""),
                    "graph_path": str(graph_path),
                    "status": "failed",
                    "error": str(e),
                }
                manifest_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                manifest_file.flush()
                failed_results.append(result)

    # Write summary
    total_candidates = sum(r.get("candidate_count", 0) for r in results)
    total_accepted = sum(r.get("accepted_count", 0) for r in results)

    # Convert args to dict with Path objects converted to strings
    config_dict = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
        else:
            config_dict[key] = value

    summary = {
        "processed_graphs": len(results),
        "failed_graphs": len(failed_results),
        "total_candidates": total_candidates,
        "total_accepted": total_accepted,
        "config": config_dict,
    }

    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)

    print(f"\nSummary:")
    print(f"  Processed: {len(results)} graphs")
    print(f"  Failed: {len(failed_results)} graphs")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Total accepted: {total_accepted}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Summary: {summary_path}")

    return 0 if not failed_results else 1


if __name__ == "__main__":
    sys.exit(main())
