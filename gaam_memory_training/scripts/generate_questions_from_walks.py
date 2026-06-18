#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaam_graph.graph_walker import GraphWalker
from gaam_graph.llm import OpenAICompatibleLLM
from gaam_graph.utils import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample graph-walk trajectories and generate questions with an LLM.")
    parser.add_argument("--graph", required=True, help="Path to an exported <record_id>.graph.json file.")
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults to <graph>.generated_questions.json.")
    parser.add_argument("--num_walks", type=int, default=8, help="Number of graph-walk trajectories to sample.")
    parser.add_argument("--walk_length", type=int, default=5, help="Maximum number of nodes in each trajectory.")
    parser.add_argument("--num_questions", type=int, default=5, help="Target number of generated questions.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducible walks.")
    parser.add_argument("--directed", action="store_true", help="Respect edge direction while walking. Default walks are undirected.")
    parser.add_argument("--max_node_chars", type=int, default=0, help="Fallback maximum characters kept for unknown node labels. 0 means no truncation.")
    parser.add_argument("--max_event_chars", type=int, default=0, help="Maximum characters kept for event node labels. 0 means no truncation.")
    parser.add_argument("--max_fact_chars", type=int, default=0, help="Maximum characters kept for fact node labels. 0 means no truncation.")
    parser.add_argument("--max_abstract_chars", type=int, default=0, help="Maximum characters kept for abstract memory node labels. 0 means no truncation.")
    parser.add_argument(
        "--max_structural_ratio",
        type=float,
        default=0.35,
        help="Maximum allowed ratio of topic/session nodes in each sampled trajectory.",
    )
    parser.add_argument(
        "--max_consecutive_next",
        type=int,
        default=1,
        help="Maximum allowed consecutive NEXT edges in a trajectory. Use -1 to disable this limit.",
    )
    parser.add_argument(
        "--allow_no_evidence_node",
        action="store_true",
        help="Allow trajectories that contain no fact or abstract_memory nodes.",
    )
    parser.add_argument(
        "--require_multi_session",
        action="store_true",
        help="Only keep trajectories whose evidence comes from at least --min_sessions distinct sessions.",
    )
    parser.add_argument(
        "--min_sessions",
        type=int,
        default=2,
        help="Minimum distinct source sessions required when --require_multi_session is enabled.",
    )
    parser.add_argument("--no-llm", action="store_true", help="Only sample trajectories and emit simple placeholder questions.")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    graph_json = read_json(graph_path)
    walker = GraphWalker(
        graph_json,
        seed=args.seed,
        directed=args.directed,
        max_node_chars=args.max_node_chars,
        type_char_limits={
            "event": args.max_event_chars,
            "fact": args.max_fact_chars,
            "abstract_memory": args.max_abstract_chars,
        },
    )
    trajectories = walker.sample_walks(
        args.num_walks,
        walk_length=args.walk_length,
        max_structural_ratio=args.max_structural_ratio,
        require_evidence_node=not args.allow_no_evidence_node,
        max_consecutive_next=args.max_consecutive_next,
        require_multi_session=args.require_multi_session,
        min_sessions=args.min_sessions,
    )
    payload = {
        "record_id": graph_json.get("graph", {}).get("record_id"),
        "num_questions": args.num_questions,
        "require_multi_session": args.require_multi_session,
        "min_sessions": args.min_sessions,
        "trajectories": trajectories,
    }

    if args.no_llm:
        llm_output = {"questions": _placeholder_questions(trajectories, args.num_questions)}
    else:
        prompt_path = ROOT / "gaam_graph" / "prompts" / "question_generation_from_walks.txt"
        system = prompt_path.read_text(encoding="utf-8")
        user = json.dumps(payload, ensure_ascii=False, indent=2)
        llm_output = OpenAICompatibleLLM().chat_json(system=system, user=user)

    questions = _normalize_questions(llm_output)
    output = {
        "graph_path": str(graph_path),
        "record_id": payload["record_id"],
        "num_walks": len(trajectories),
        "walk_length": args.walk_length,
        "max_structural_ratio": args.max_structural_ratio,
        "max_consecutive_next": args.max_consecutive_next,
        "require_evidence_node": not args.allow_no_evidence_node,
        "require_multi_session": args.require_multi_session,
        "min_sessions": args.min_sessions,
        "char_limits": {
            "event": args.max_event_chars,
            "fact": args.max_fact_chars,
            "abstract_memory": args.max_abstract_chars,
            "fallback": args.max_node_chars,
        },
        "trajectories": trajectories,
        "questions": questions,
        "raw_llm_output": llm_output,
    }
    output_path = Path(args.output) if args.output else _default_output_path(graph_path)
    write_json(output_path, output)
    print(f"Wrote {len(questions)} questions from {len(trajectories)} trajectories to {output_path}")


def _normalize_questions(llm_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    questions = llm_output.get("questions", [])
    if not isinstance(questions, list):
        return []
    normalized = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question or not answer:
            continue
        normalized.append({
            "question": question,
            "answer": answer,
            "question_type": str(item.get("question_type") or "other"),
            "supporting_trajectory_ids": _string_list(item.get("supporting_trajectory_ids")),
            "supporting_node_ids": _string_list(item.get("supporting_node_ids")),
            "reason": str(item.get("reason") or "").strip(),
        })
    return normalized


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _default_output_path(graph_path: Path) -> Path:
    stem = graph_path.stem
    if stem.endswith(".graph"):
        stem = stem[: -len(".graph")]
    return graph_path.with_name(f"{stem}.generated_questions.json")


def _placeholder_questions(trajectories: List[Dict[str, Any]], num_questions: int) -> List[Dict[str, Any]]:
    questions = []
    for traj in trajectories[:num_questions]:
        nodes = traj.get("nodes", [])
        evidence_node = _best_evidence_node(nodes)
        if not evidence_node:
            continue
        questions.append({
            "question": "What information is supported by this graph trajectory?",
            "answer": evidence_node.get("label", ""),
            "question_type": "other",
            "supporting_trajectory_ids": [traj.get("trajectory_id", "")],
            "supporting_node_ids": [evidence_node.get("id", "")],
            "reason": "Placeholder generated without an LLM for trajectory inspection.",
        })
    return questions


def _best_evidence_node(nodes: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for node_type in ["fact", "abstract_memory", "event"]:
        for node in nodes:
            if node.get("type") == node_type and node.get("label"):
                return node
    return nodes[0] if nodes else None


if __name__ == "__main__":
    main()
