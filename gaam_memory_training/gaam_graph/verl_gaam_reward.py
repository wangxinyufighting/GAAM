"""
GAAM custom reward function for native VERL GRPO.

VERL's naive reward manager calls:
    compute_score(data_source, solution_str, ground_truth, extra_info=None)

The reward here is intentionally deterministic and multiprocessing-safe so it
can run inside native VERL/Ray workers.
"""

from __future__ import annotations

import json
import re
from typing import Any


FORBIDDEN_LEAKAGE_TERMS = {
    "benchmark question",
    "target question",
    "gold answer",
    "oracle answer",
    "haystack_question_type",
}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a scalar GAAM reward for VERL native GRPO."""
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except Exception:
            ground_truth = {"required_terms": []}
    if not isinstance(ground_truth, dict):
        ground_truth = {"required_terms": []}

    actor_role = ground_truth.get("actor_role") or (extra_info or {}).get("actor_role")
    if data_source == "gaam_memory_builder" or actor_role == "memory_builder":
        details = _score_memory_builder(solution_str, ground_truth)
    elif data_source == "gaam_question_agent" or actor_role == "question_agent":
        details = _score_question_agent(solution_str, ground_truth)
    else:
        details = {"score": 0.0, "unknown_data_source": data_source}

    details["score"] = float(max(0.0, min(1.0, details.get("score", 0.0))))
    details["data_source"] = data_source
    return details


def _score_memory_builder(solution: str, ground_truth: dict[str, Any]) -> dict[str, Any]:
    text = solution.strip()
    lower = text.lower()
    required_terms = _required_terms(ground_truth)

    parse_score = 0.0
    parsed = _try_parse_json(text)
    if isinstance(parsed, dict):
        parse_score = 0.2
        has_graph = isinstance(parsed.get("memory_graph"), dict)
        has_summaries = isinstance(parsed.get("memory_summaries"), dict)
        parse_score += 0.15 if has_graph else 0.0
        parse_score += 0.15 if has_summaries else 0.0
    else:
        has_graph = "memory_graph" in lower or "nodes" in lower
        has_summaries = "memory_summaries" in lower or "summary" in lower
        parse_score += 0.1 if has_graph else 0.0
        parse_score += 0.1 if has_summaries else 0.0

    coverage = _coverage_score(lower, required_terms)
    abstraction = _keyword_fraction(
        lower,
        [
            "summary",
            "abstraction",
            "preference",
            "update",
            "session",
            "evidence",
            "provenance",
        ],
    )
    compression = _length_band_score(len(text), low=600, target_low=1200, target_high=8000, high=16000)
    leakage_penalty = _leakage_penalty(lower)

    score = (
        parse_score
        + 0.30 * coverage
        + 0.15 * abstraction
        + 0.10 * compression
        - leakage_penalty
    )
    return {
        "score": score,
        "parse_score": parse_score,
        "coverage_score": coverage,
        "abstraction_score": abstraction,
        "compression_score": compression,
        "leakage_penalty": leakage_penalty,
        "num_required_terms": len(required_terms),
    }


def _score_question_agent(solution: str, ground_truth: dict[str, Any]) -> dict[str, Any]:
    text = solution.strip()
    lower = text.lower()
    required_terms = _required_terms(ground_truth)
    questions = _extract_questions(text)

    count_score = min(1.0, len(questions) / 8.0)
    diversity_score = _question_diversity_score(lower)
    coverage = _coverage_score(lower, required_terms)
    difficulty = _keyword_fraction(
        lower,
        [
            "multi-hop",
            "multi session",
            "multi-session",
            "temporal",
            "preference",
            "contradiction",
            "update",
            "summary",
            "abstraction",
        ],
    )
    redundancy_penalty = _question_redundancy_penalty(questions)
    leakage_penalty = _leakage_penalty(lower)

    score = (
        0.25 * count_score
        + 0.25 * diversity_score
        + 0.25 * coverage
        + 0.15 * difficulty
        - redundancy_penalty
        - leakage_penalty
    )
    return {
        "score": score,
        "question_count_score": count_score,
        "diversity_score": diversity_score,
        "coverage_score": coverage,
        "difficulty_score": difficulty,
        "redundancy_penalty": redundancy_penalty,
        "leakage_penalty": leakage_penalty,
        "num_questions": len(questions),
        "num_required_terms": len(required_terms),
    }


def _try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _required_terms(ground_truth: dict[str, Any]) -> list[str]:
    terms = ground_truth.get("required_terms", [])
    if not isinstance(terms, list):
        return []
    return [str(term).lower() for term in terms if str(term).strip()]


def _coverage_score(lower_solution: str, required_terms: list[str]) -> float:
    if not required_terms:
        return 0.0
    checked = required_terms[:48]
    hits = sum(1 for term in checked if term and term in lower_solution)
    return hits / max(1, len(checked))


def _keyword_fraction(lower_solution: str, keywords: list[str]) -> float:
    hits = sum(1 for keyword in keywords if keyword in lower_solution)
    return hits / max(1, len(keywords))


def _length_band_score(length: int, *, low: int, target_low: int, target_high: int, high: int) -> float:
    if length <= 0:
        return 0.0
    if target_low <= length <= target_high:
        return 1.0
    if low <= length < target_low:
        return (length - low) / max(1, target_low - low)
    if target_high < length <= high:
        return 1.0 - (length - target_high) / max(1, high - target_high)
    return 0.0


def _leakage_penalty(lower_solution: str) -> float:
    penalty = 0.0
    for term in FORBIDDEN_LEAKAGE_TERMS:
        if term in lower_solution:
            penalty += 0.25
    return min(0.75, penalty)


def _extract_questions(text: str) -> list[str]:
    parsed = _try_parse_json(text)
    questions: list[str] = []
    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
        for item in parsed["questions"]:
            if isinstance(item, dict):
                value = item.get("question")
            else:
                value = item
            if value:
                questions.append(str(value).strip())
    if not questions:
        questions = [q.strip() for q in re.split(r"[?\n]", text) if len(q.strip()) > 12]
    return questions


def _question_diversity_score(lower_solution: str) -> float:
    categories = [
        ["single-hop", "single hop"],
        ["multi-hop", "multi hop"],
        ["multi-session", "multi session", "cross-session"],
        ["temporal", "before", "after", "when"],
        ["preference", "prefer"],
        ["contradiction", "update", "changed"],
        ["summary", "abstraction", "overall"],
    ]
    hits = 0
    for aliases in categories:
        if any(alias in lower_solution for alias in aliases):
            hits += 1
    return hits / len(categories)


def _question_redundancy_penalty(questions: list[str]) -> float:
    if len(questions) <= 1:
        return 0.0
    normalized = [_normalize_question(question) for question in questions]
    unique = set(normalized)
    duplicate_ratio = 1.0 - (len(unique) / len(normalized))
    return min(0.4, duplicate_ratio)


def _normalize_question(question: str) -> str:
    question = question.lower()
    question = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", question)
    return re.sub(r"\s+", " ", question).strip()
