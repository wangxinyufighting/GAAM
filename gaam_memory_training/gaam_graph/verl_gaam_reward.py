"""
GAAM custom reward function for native VERL GRPO.

VERL's naive reward manager calls:
    compute_score(data_source, solution_str, ground_truth, extra_info=None)

The reward here is intentionally deterministic and multiprocessing-safe so it
can run inside native VERL/Ray workers.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


FORBIDDEN_LEAKAGE_TERMS = {
    "benchmark question",
    "target question",
    "gold answer",
    "oracle answer",
    "haystack_question_type",
}

MEMORY_FORBIDDEN_OUTPUT_KEYS = {
    "question",
    "target_question",
    "benchmark_question",
    "answer",
    "target_answer",
    "gold_answer",
    "oracle_answer",
    "haystack_question_type",
}

QUESTION_FORBIDDEN_OUTPUT_KEYS = {
    "target_question",
    "benchmark_question",
    "answer",
    "target_answer",
    "gold_answer",
    "oracle_answer",
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
    hard_failure = _hard_failure(solution_str, ground_truth, actor_role=str(actor_role or ""))
    if hard_failure:
        return {
            "score": 0.0,
            "data_source": data_source,
            "hard_failure": hard_failure,
        }

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

    heuristic_score = (
        parse_score
        + 0.30 * coverage
        + 0.15 * abstraction
        + 0.10 * compression
        - leakage_penalty
    )
    details = {
        "score": heuristic_score,
        "heuristic_score": heuristic_score,
        "parse_score": parse_score,
        "coverage_score": coverage,
        "abstraction_score": abstraction,
        "compression_score": compression,
        "leakage_penalty": leakage_penalty,
        "num_required_terms": len(required_terms),
    }
    judge = _maybe_llm_judge_memory(solution, ground_truth)
    if judge:
        details["llm_judge"] = judge
        if judge.get("status") == "succeeded":
            judge_score = _memory_judge_score(judge)
            details["llm_judge_score"] = judge_score
            details["score"] = 0.45 * heuristic_score + 0.55 * judge_score - leakage_penalty
        elif _judge_required():
            details["score"] = 0.0
    return details


def _score_question_agent(solution: str, ground_truth: dict[str, Any]) -> dict[str, Any]:
    text = solution.strip()
    lower = text.lower()
    required_terms = _required_terms(ground_truth)
    questions = _extract_questions(text)
    target_questions = _target_questions_per_case(ground_truth)

    count_score = _exact_count_score(len(questions), target_questions)
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

    heuristic_score = (
        0.25 * count_score
        + 0.25 * diversity_score
        + 0.25 * coverage
        + 0.15 * difficulty
        - redundancy_penalty
        - leakage_penalty
    )
    details = {
        "score": heuristic_score,
        "heuristic_score": heuristic_score,
        "question_count_score": count_score,
        "diversity_score": diversity_score,
        "coverage_score": coverage,
        "difficulty_score": difficulty,
        "redundancy_penalty": redundancy_penalty,
        "leakage_penalty": leakage_penalty,
        "num_questions": len(questions),
        "target_questions": target_questions,
        "num_required_terms": len(required_terms),
    }
    judge = _maybe_llm_judge_questions(solution, ground_truth, questions)
    if judge:
        details["llm_judge"] = judge
        if judge.get("status") == "succeeded":
            judge_score = _question_judge_score(judge)
            details["llm_judge_score"] = judge_score
            details["score"] = 0.40 * heuristic_score + 0.60 * judge_score - leakage_penalty
        elif _judge_required():
            details["score"] = 0.0
    return details


def _judge_enabled() -> bool:
    value = os.getenv("GAAM_REWARD_JUDGE_ENABLED", "0")
    return value in {"1", "true", "True", "yes", "YES"}


def _judge_required() -> bool:
    value = os.getenv("GAAM_REWARD_JUDGE_REQUIRED", "0")
    return value in {"1", "true", "True", "yes", "YES"}


def _judge_config() -> dict[str, Any]:
    timeout = 60.0
    try:
        timeout = float(os.getenv("GAAM_REWARD_JUDGE_TIMEOUT", "60"))
    except Exception:
        timeout = 60.0
    max_input_chars = 20000
    try:
        max_input_chars = max(1000, int(os.getenv("GAAM_REWARD_JUDGE_MAX_INPUT_CHARS", "20000")))
    except Exception:
        max_input_chars = 20000
    return {
        "base_url": os.getenv(
            "GAAM_REWARD_JUDGE_BASE_URL",
            os.getenv("DEEPSEEK_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")),
        ),
        "api_key": os.getenv(
            "GAAM_REWARD_JUDGE_API_KEY",
            os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        ),
        "model": os.getenv(
            "GAAM_REWARD_JUDGE_MODEL",
            os.getenv("DEEPSEEK_MODEL", os.getenv("LLM_MODEL", "deepseek-v4-flash")),
        ),
        "timeout": timeout,
        "max_input_chars": max_input_chars,
    }


def _redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_url": str(config.get("base_url", "")),
        "model": str(config.get("model", "")),
        "timeout": config.get("timeout"),
        "max_input_chars": config.get("max_input_chars"),
        "api_key_env": "GAAM_REWARD_JUDGE_API_KEY",
        "api_key_configured": bool(config.get("api_key")),
    }


def _maybe_llm_judge_memory(solution: str, ground_truth: dict[str, Any]) -> dict[str, Any] | None:
    if not _judge_enabled():
        return None
    config = _judge_config()
    config_error = _judge_config_error(config)
    if config_error:
        result = {
            "status": "skipped",
            "error": config_error,
            "config": _redacted_config(config),
        }
        return result if _judge_required() else result

    oracle_digest = _clip_for_judge(str(ground_truth.get("oracle_digest", "")), config, fraction=0.45)
    system = (
        "You are a strict reward judge for GAAM Memory Builder outputs. "
        "Return only JSON. Do not reveal or invent benchmark target questions or gold answers."
    )
    user = (
        "Judge whether Current Memory is useful for future answering, grounded in the oracle digest, "
        "not redundant, not over-compressed, and safely formatted.\n\n"
        "Return JSON with numeric scores in [0,1]: oracle_coverage, groundedness, "
        "non_redundancy, compression_balance, abstraction_quality, leakage_safety, overall. "
        "Also include short rationale.\n\n"
        f"Oracle digest:\n{oracle_digest}\n\n"
        f"Memory Builder output:\n{_clip_for_judge(solution, config)}"
    )
    return _call_judge_api(system=system, user=user, config=config)


def _maybe_llm_judge_questions(
    solution: str,
    ground_truth: dict[str, Any],
    questions: list[str],
) -> dict[str, Any] | None:
    if not _judge_enabled():
        return None
    config = _judge_config()
    config_error = _judge_config_error(config)
    if config_error:
        result = {
            "status": "skipped",
            "error": config_error,
            "config": _redacted_config(config),
        }
        return result if _judge_required() else result

    oracle_digest = _clip_for_judge(str(ground_truth.get("oracle_digest", "")), config, fraction=0.45)
    target_questions = _target_questions_per_case(ground_truth)
    system = (
        "You are a strict reward judge for GAAM Question Agent outputs. "
        "Return only JSON. Do not reveal or infer benchmark target questions."
    )
    user = (
        "Judge generated training questions against the oracle digest. The questions must be "
        "oracle-valid, answerable from the oracle graph, diverse, useful for exposing memory "
        "weaknesses, and free of leakage.\n\n"
        "Return JSON with numeric scores in [0,1]: oracle_validity, answerability, diversity, "
        "difficulty, weakness_targeting, non_redundancy, leakage_safety, overall. "
        "Also include short rationale.\n\n"
        f"Target number of questions: {target_questions}\n"
        f"Parsed question count: {len(questions)}\n\n"
        f"Oracle digest:\n{oracle_digest}\n\n"
        f"Question Agent output:\n{_clip_for_judge(solution, config)}"
    )
    return _call_judge_api(system=system, user=user, config=config)


def _judge_config_error(config: dict[str, Any]) -> str | None:
    if not str(config.get("api_key", "")).strip():
        return "GAAM_REWARD_JUDGE_API_KEY is not set"
    if not str(config.get("base_url", "")).strip():
        return "GAAM_REWARD_JUDGE_BASE_URL is not set"
    if not str(config.get("model", "")).strip():
        return "GAAM_REWARD_JUDGE_MODEL is not set"
    return None


def _clip_for_judge(text: str, config: dict[str, Any], *, fraction: float = 1.0) -> str:
    limit = int(float(config.get("max_input_chars", 20000)) * fraction)
    limit = max(1000, limit)
    if len(text) <= limit:
        return text
    return text[: limit - 40].rstrip() + "\n...[truncated for reward judge]"


def _call_judge_api(*, system: str, user: str, config: dict[str, Any]) -> dict[str, Any]:
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=config["api_key"],
            base_url=str(config["base_url"]).rstrip("/"),
            timeout=config["timeout"],
        )
        extra_body = None
        if "deepseek" in str(config["base_url"]).lower():
            extra_body = {"thinking": {"type": "disabled"}}
        request = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "stream": False,
            "extra_body": extra_body,
        }
        try:
            response = client.chat.completions.create(
                **request,
                response_format={"type": "json_object"},
            )
            response_format_used = True
        except Exception:
            response = client.chat.completions.create(**request)
            response_format_used = False
        parsed = _try_parse_json(response.choices[0].message.content or "")
        if not isinstance(parsed, dict):
            raise ValueError("Judge did not return a JSON object")
        parsed["status"] = "succeeded"
        parsed["model"] = config["model"]
        parsed["base_url"] = str(config["base_url"])
        parsed["config"] = _redacted_config(config)
        parsed["response_format_used"] = response_format_used
        return parsed
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "model": config.get("model"),
            "base_url": str(config.get("base_url", "")),
            "config": _redacted_config(config),
        }


def _memory_judge_score(judge: dict[str, Any]) -> float:
    keys = [
        "oracle_coverage",
        "groundedness",
        "non_redundancy",
        "compression_balance",
        "abstraction_quality",
        "leakage_safety",
        "overall",
    ]
    return _mean_score_fields(judge, keys)


def _question_judge_score(judge: dict[str, Any]) -> float:
    keys = [
        "oracle_validity",
        "answerability",
        "diversity",
        "difficulty",
        "weakness_targeting",
        "non_redundancy",
        "leakage_safety",
        "overall",
    ]
    return _mean_score_fields(judge, keys)


def _mean_score_fields(payload: dict[str, Any], keys: list[str]) -> float:
    values: list[float] = []
    for key in keys:
        try:
            values.append(float(payload[key]))
        except Exception:
            continue
    if not values:
        return 0.0
    return max(0.0, min(1.0, sum(values) / len(values)))


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


def _target_questions_per_case(ground_truth: dict[str, Any]) -> int:
    value = ground_truth.get("questions_per_case", 8)
    try:
        return max(1, int(value))
    except Exception:
        return 8


def _exact_count_score(num_questions: int, target_questions: int) -> float:
    if target_questions <= 0:
        return 0.0
    distance = abs(num_questions - target_questions)
    return max(0.0, 1.0 - (distance / target_questions))


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


def _hard_failure(solution: str, ground_truth: dict[str, Any], *, actor_role: str) -> dict[str, Any] | None:
    if not solution or not solution.strip():
        return {"reason": "empty_solution"}
    lower_solution = solution.lower()
    leaked_terms = [term for term in FORBIDDEN_LEAKAGE_TERMS if term in lower_solution]
    if leaked_terms:
        return {"reason": "forbidden_leakage_terms", "terms": sorted(leaked_terms)}

    parsed = _try_parse_json(solution)
    leaked_keys: list[str] = []
    if parsed is not None:
        forbidden_keys = (
            MEMORY_FORBIDDEN_OUTPUT_KEYS
            if actor_role == "memory_builder"
            else QUESTION_FORBIDDEN_OUTPUT_KEYS
        )
        leaked_keys = sorted(set(_find_forbidden_keys(parsed, forbidden_keys)))
    if leaked_keys:
        return {"reason": "forbidden_output_keys", "keys": leaked_keys}

    target_question = str(ground_truth.get("target_question", "") or ground_truth.get("benchmark_question", "")).strip()
    if len(target_question) >= 12 and target_question.lower() in lower_solution:
        return {"reason": "target_question_leakage"}
    return None


def _find_forbidden_keys(value: Any, forbidden_keys: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalize_key(str(key))
            if normalized in forbidden_keys:
                found.append(str(key))
            found.extend(_find_forbidden_keys(child, forbidden_keys))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_forbidden_keys(child, forbidden_keys))
    return found


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


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
