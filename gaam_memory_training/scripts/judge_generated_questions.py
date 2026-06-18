#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaam_graph.llm import OpenAICompatibleLLM
from gaam_graph.utils import read_json, write_json


SCORE_KEYS = [
    "answerability",
    "evidence_grounding",
    "specificity",
    "memory_value",
    "reasoning_depth",
    "clarity",
    "multi_session_value",
]

ISSUE_VALUES = {
    "unsupported_answer",
    "vague_question",
    "generic_template",
    "trivial_copy",
    "answer_too_long",
    "answer_too_short",
    "ambiguous_reference",
    "depends_on_outside_knowledge",
    "mostly_assistant_advice",
    "weak_multi_session_use",
    "cited_nodes_missing",
    "cited_trajectories_missing",
}

VERDICTS = {"accept", "revise", "reject"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge generated question quality with an LLM-as-judge.")
    parser.add_argument("--input", required=True, help="Path to a *.generated_questions*.json file.")
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults to <input>.question_judgments.json.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of questions judged per LLM request.")
    parser.add_argument("--accept_threshold", type=float, default=4.0, help="Overall score needed for accepted_questions.")
    parser.add_argument("--max_trajectory_chars", type=int, default=4000, help="Maximum chars retained per trajectory text in judge prompt.")
    parser.add_argument("--no-llm", action="store_true", help="Run a deterministic heuristic judge for smoke tests.")
    args = parser.parse_args()

    input_path = Path(args.input)
    generated = read_json(input_path)
    questions = _questions_with_ids(generated.get("questions", []))
    trajectories_by_id = _trajectories_by_id(generated.get("trajectories", []))

    if args.no_llm:
        judgments = [_heuristic_judgment(question, trajectories_by_id) for question in questions]
        raw_outputs: List[Dict[str, Any]] = []
        judge_mode = "heuristic"
    else:
        judgments, raw_outputs = _judge_with_llm(
            generated=generated,
            questions=questions,
            trajectories_by_id=trajectories_by_id,
            batch_size=max(args.batch_size, 1),
            max_trajectory_chars=args.max_trajectory_chars,
        )
        judge_mode = "llm"

    judgments = _normalize_judgments(judgments, questions)
    summary = _summarize(judgments, accept_threshold=args.accept_threshold)
    output = {
        "input_path": str(input_path),
        "record_id": generated.get("record_id"),
        "judge_mode": judge_mode,
        "accept_threshold": args.accept_threshold,
        "num_questions": len(questions),
        "summary": summary,
        "judgments": judgments,
        "raw_judge_outputs": raw_outputs,
    }
    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    write_json(output_path, output)
    print(
        "Judged "
        f"{len(judgments)} questions: "
        f"{summary['accepted_count']} accept, "
        f"{summary['revise_count']} revise, "
        f"{summary['rejected_count']} reject. "
        f"Wrote {output_path}"
    )


def _judge_with_llm(
    *,
    generated: Dict[str, Any],
    questions: List[Dict[str, Any]],
    trajectories_by_id: Dict[str, Dict[str, Any]],
    batch_size: int,
    max_trajectory_chars: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt_path = ROOT / "gaam_graph" / "prompts" / "question_value_judge.txt"
    system = prompt_path.read_text(encoding="utf-8")
    llm = OpenAICompatibleLLM()
    judgments: List[Dict[str, Any]] = []
    raw_outputs: List[Dict[str, Any]] = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        payload = {
            "record_id": generated.get("record_id"),
            "require_multi_session": generated.get("require_multi_session", False),
            "min_sessions": generated.get("min_sessions", 1),
            "questions": [
                _question_payload(question, trajectories_by_id, max_trajectory_chars=max_trajectory_chars)
                for question in batch
            ],
        }
        raw = llm.chat_json(system=system, user=json.dumps(payload, ensure_ascii=False, indent=2))
        raw_outputs.append(raw)
        batch_judgments = raw.get("judgments", [])
        if isinstance(batch_judgments, list):
            judgments.extend(item for item in batch_judgments if isinstance(item, dict))
    return judgments, raw_outputs


def _questions_with_ids(questions: Any) -> List[Dict[str, Any]]:
    if not isinstance(questions, list):
        return []
    result = []
    for idx, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            continue
        question = dict(item)
        question.setdefault("question_id", f"q_{idx:04d}")
        result.append(question)
    return result


def _trajectories_by_id(trajectories: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(trajectories, list):
        return {}
    result = {}
    for item in trajectories:
        if isinstance(item, dict) and item.get("trajectory_id"):
            result[str(item["trajectory_id"])] = item
    return result


def _question_payload(
    question: Dict[str, Any],
    trajectories_by_id: Dict[str, Dict[str, Any]],
    *,
    max_trajectory_chars: int,
) -> Dict[str, Any]:
    supporting_trajectory_ids = _string_list(question.get("supporting_trajectory_ids"))
    evidence = []
    missing_trajectory_ids = []
    for trajectory_id in supporting_trajectory_ids:
        trajectory = trajectories_by_id.get(trajectory_id)
        if not trajectory:
            missing_trajectory_ids.append(trajectory_id)
            continue
        evidence.append(_compact_trajectory(trajectory, max_chars=max_trajectory_chars))
    return {
        "question_id": question.get("question_id"),
        "question": question.get("question", ""),
        "answer": question.get("answer", ""),
        "question_type": question.get("question_type", "other"),
        "supporting_trajectory_ids": supporting_trajectory_ids,
        "supporting_node_ids": _string_list(question.get("supporting_node_ids")),
        "generator_reason": question.get("reason", ""),
        "missing_trajectory_ids": missing_trajectory_ids,
        "evidence_trajectories": evidence,
    }


def _compact_trajectory(trajectory: Dict[str, Any], *, max_chars: int) -> Dict[str, Any]:
    text = str(trajectory.get("text") or "")
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return {
        "trajectory_id": trajectory.get("trajectory_id"),
        "session_ids": trajectory.get("session_ids", []),
        "text": text,
        "nodes": [
            {
                "id": node.get("id"),
                "type": node.get("type"),
                "label": node.get("label"),
                "session_ids": node.get("session_ids", []),
                "source_session_id": node.get("source_session_id"),
            }
            for node in trajectory.get("nodes", [])
            if isinstance(node, dict)
        ],
    }


def _normalize_judgments(judgments: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {str(item.get("question_id", "")): item for item in judgments if isinstance(item, dict)}
    normalized = []
    for question in questions:
        question_id = str(question.get("question_id"))
        item = by_id.get(question_id, {})
        scores = _normalize_scores(item.get("scores", {}))
        overall_score = _score_value(item.get("overall_score"), default=_default_overall(scores))
        verdict = str(item.get("verdict") or _verdict_for_score(overall_score)).strip().lower()
        if verdict not in VERDICTS:
            verdict = _verdict_for_score(overall_score)
        issues = [issue for issue in _string_list(item.get("issues")) if issue in ISSUE_VALUES]
        if not by_id.get(question_id):
            issues.append("cited_trajectories_missing")
        normalized.append({
            "question_id": question_id,
            "question": str(question.get("question") or ""),
            "answer": str(question.get("answer") or ""),
            "question_type": str(question.get("question_type") or "other"),
            "supporting_trajectory_ids": _string_list(question.get("supporting_trajectory_ids")),
            "supporting_node_ids": _string_list(question.get("supporting_node_ids")),
            "overall_score": overall_score,
            "scores": scores,
            "verdict": verdict,
            "issues": sorted(set(issues)),
            "rationale": str(item.get("rationale") or "").strip(),
            "improvement_suggestion": str(item.get("improvement_suggestion") or "").strip(),
        })
    return normalized


def _normalize_scores(value: Any) -> Dict[str, int]:
    value = value if isinstance(value, dict) else {}
    return {key: _score_value(value.get(key), default=1) for key in SCORE_KEYS}


def _score_value(value: Any, *, default: int | float) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = int(default)
    return max(1, min(score, 5))


def _default_overall(scores: Dict[str, int]) -> int:
    if not scores:
        return 1
    return _score_value(sum(scores.values()) / len(scores), default=1)


def _verdict_for_score(score: int) -> str:
    if score >= 4:
        return "accept"
    if score == 3:
        return "revise"
    return "reject"


def _summarize(judgments: List[Dict[str, Any]], *, accept_threshold: float) -> Dict[str, Any]:
    scores = [float(item.get("overall_score", 0)) for item in judgments]
    accepted = [item for item in judgments if item.get("verdict") == "accept" and item.get("overall_score", 0) >= accept_threshold]
    revise = [item for item in judgments if item.get("verdict") == "revise"]
    rejected = [item for item in judgments if item.get("verdict") == "reject"]
    issue_counts: Dict[str, int] = {}
    for item in judgments:
        for issue in item.get("issues", []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    return {
        "average_overall_score": round(statistics.mean(scores), 3) if scores else 0.0,
        "accepted_count": len(accepted),
        "revise_count": len(revise),
        "rejected_count": len(rejected),
        "accepted_question_ids": [item["question_id"] for item in accepted],
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def _heuristic_judgment(question: Dict[str, Any], trajectories_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    text = str(question.get("question") or "").strip()
    answer = str(question.get("answer") or "").strip()
    supporting_trajectory_ids = _string_list(question.get("supporting_trajectory_ids"))
    supporting_node_ids = _string_list(question.get("supporting_node_ids"))
    evidence_sessions = set()
    for trajectory_id in supporting_trajectory_ids:
        evidence_sessions.update(trajectories_by_id.get(trajectory_id, {}).get("session_ids", []))

    issues = []
    if not supporting_trajectory_ids or any(tid not in trajectories_by_id for tid in supporting_trajectory_ids):
        issues.append("cited_trajectories_missing")
    if not supporting_node_ids:
        issues.append("cited_nodes_missing")
    if _looks_generic(text):
        issues.append("generic_template")
    if len(answer) > 220:
        issues.append("answer_too_long")
    if len(answer) < 2:
        issues.append("answer_too_short")
    if len(evidence_sessions) < 2:
        issues.append("weak_multi_session_use")

    scores = {
        "answerability": 4 if supporting_trajectory_ids and answer else 2,
        "evidence_grounding": 4 if supporting_node_ids and "cited_trajectories_missing" not in issues else 2,
        "specificity": 2 if "generic_template" in issues else 4,
        "memory_value": 4 if not _looks_generic(text) else 2,
        "reasoning_depth": 4 if len(evidence_sessions) >= 2 else 2,
        "clarity": 4 if text.endswith("?") and len(text.split()) >= 5 else 3,
        "multi_session_value": 4 if len(evidence_sessions) >= 2 else 2,
    }
    overall_score = _default_overall(scores)
    if "generic_template" in issues:
        overall_score = min(overall_score, 2)
    return {
        "question_id": question.get("question_id"),
        "overall_score": overall_score,
        "scores": scores,
        "verdict": _verdict_for_score(overall_score),
        "issues": issues,
        "rationale": "Heuristic smoke-test judgment based on citations, specificity, answer length, and session coverage.",
        "improvement_suggestion": "" if overall_score >= 4 else "Make the question more specific and ensure the answer is directly supported by cited evidence.",
    }


def _looks_generic(text: str) -> bool:
    lowered = text.lower()
    generic_phrases = [
        "what information is supported",
        "this graph trajectory",
        "what does the trajectory",
        "what is mentioned",
        "what information",
    ]
    return any(phrase in lowered for phrase in generic_phrases)


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith(".generated_questions"):
        stem = stem[: -len(".generated_questions")]
    return input_path.with_name(f"{stem}.question_judgments.json")


if __name__ == "__main__":
    main()
