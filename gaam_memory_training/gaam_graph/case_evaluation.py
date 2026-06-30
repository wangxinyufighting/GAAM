"""Case-level GAAM evaluation.

Pipeline:
1. Read LongMemEval case sessions.
2. Build Current Memory without seeing the benchmark target question.
3. Answer the benchmark question from Current Memory.
4. Judge the answer against the benchmark answer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from gaam_graph.answer_agent import FrozenAnswerer
from gaam_graph.answer_schema import AnswerRequest
from gaam_graph.llm import LocalHFChatLLM, OpenAICompatibleLLM
from gaam_graph.lme_loader import LMERecord, LongMemEvalLoader
from gaam_graph.memory_builder import BaselineMemoryBuilder
from gaam_graph.raw_history import raw_history_from_lme_record
from gaam_graph.reward_manager import score_answer_correctness
from gaam_graph.stateful_incremental_memory import (
    StatefulIncrementalMemoryConfig,
    build_record_stateful_incremental_memory,
)
from gaam_graph.utils import stable_id, write_json


@dataclass(frozen=True)
class CaseEvaluationConfig:
    input_path: Path
    output_dir: Path
    split_manifest_path: Path | None = None
    evaluation_split: str | None = None
    record_id: str | None = None
    max_records: int | None = None
    memory_backend: str = "baseline"  # baseline, stateful_api, or stateful_local_hf
    answer_backend: str = "api"  # api, local_hf, or no_llm
    judge_backend: str = "api"  # api or heuristic
    session_chunk_size: int = 4
    max_chunk_chars: int = 12000
    max_previous_memory_chars: int = 12000
    memory_model: str | None = None
    memory_base_url: str | None = None
    memory_api_key: str | None = None
    memory_max_new_tokens: int | None = None
    memory_device_map: str | None = None
    memory_torch_dtype: str | None = None
    answer_model: str | None = None
    answer_base_url: str | None = None
    answer_api_key: str | None = None
    answer_max_new_tokens: int | None = None
    answer_device_map: str | None = None
    answer_torch_dtype: str | None = None
    judge_model: str | None = None
    judge_base_url: str | None = None
    judge_api_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "memory_model",
            self.memory_model
            or _env_first_nonempty(
                "GAAM_MEMORY_BUILDER_MODEL",
                "DEEPSEEK_MODEL",
                default="deepseek-v4-flash",
            ),
        )
        object.__setattr__(
            self,
            "memory_base_url",
            self.memory_base_url
            or _env_first_nonempty(
                "GAAM_MEMORY_BUILDER_BASE_URL",
                "DEEPSEEK_BASE_URL",
                "OPENAI_BASE_URL",
                default="https://api.deepseek.com",
            ),
        )
        object.__setattr__(
            self,
            "memory_api_key",
            self.memory_api_key
            or _env_first_nonempty(
                "GAAM_MEMORY_BUILDER_API_KEY",
                "DEEPSEEK_API_KEY",
                "OPENAI_API_KEY",
                default="",
            ),
        )
        object.__setattr__(
            self,
            "memory_max_new_tokens",
            self.memory_max_new_tokens
            or _env_int_first_nonempty("GAAM_MEMORY_BUILDER_MAX_NEW_TOKENS", default=2048),
        )
        object.__setattr__(
            self,
            "memory_device_map",
            self.memory_device_map
            or _env_first_nonempty("GAAM_MEMORY_BUILDER_DEVICE_MAP", default="auto"),
        )
        object.__setattr__(
            self,
            "memory_torch_dtype",
            self.memory_torch_dtype
            or _env_first_nonempty("GAAM_MEMORY_BUILDER_TORCH_DTYPE", default="auto"),
        )
        object.__setattr__(
            self,
            "answer_model",
            self.answer_model
            or _env_first_nonempty(
                "GAAM_ANSWERER_MODEL",
                "DEEPSEEK_MODEL",
                default="deepseek-v4-flash",
            ),
        )
        object.__setattr__(
            self,
            "answer_base_url",
            self.answer_base_url
            or _env_first_nonempty(
                "GAAM_ANSWERER_BASE_URL",
                "DEEPSEEK_BASE_URL",
                "OPENAI_BASE_URL",
                default="https://api.deepseek.com",
            ),
        )
        object.__setattr__(
            self,
            "answer_api_key",
            self.answer_api_key
            or _env_first_nonempty(
                "GAAM_ANSWERER_API_KEY",
                "DEEPSEEK_API_KEY",
                "OPENAI_API_KEY",
                default="",
            ),
        )
        object.__setattr__(
            self,
            "answer_max_new_tokens",
            self.answer_max_new_tokens
            or _env_int_first_nonempty("GAAM_ANSWERER_MAX_NEW_TOKENS", default=1024),
        )
        object.__setattr__(
            self,
            "answer_device_map",
            self.answer_device_map
            or _env_first_nonempty("GAAM_ANSWERER_DEVICE_MAP", default="auto"),
        )
        object.__setattr__(
            self,
            "answer_torch_dtype",
            self.answer_torch_dtype
            or _env_first_nonempty("GAAM_ANSWERER_TORCH_DTYPE", default="auto"),
        )
        object.__setattr__(
            self,
            "judge_model",
            self.judge_model
            or _env_first_nonempty(
                "GAAM_EVAL_JUDGE_MODEL",
                "DEEPSEEK_MODEL",
                default="deepseek-v4-flash",
            ),
        )
        object.__setattr__(
            self,
            "judge_base_url",
            self.judge_base_url
            or _env_first_nonempty(
                "GAAM_EVAL_JUDGE_BASE_URL",
                "DEEPSEEK_BASE_URL",
                "OPENAI_BASE_URL",
                default="https://api.deepseek.com",
            ),
        )
        object.__setattr__(
            self,
            "judge_api_key",
            self.judge_api_key
            or _env_first_nonempty(
                "GAAM_EVAL_JUDGE_API_KEY",
                "DEEPSEEK_API_KEY",
                "OPENAI_API_KEY",
                default="",
            ),
        )


def run_case_evaluation(config: CaseEvaluationConfig) -> dict[str, Any]:
    records = _select_records(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for record in records:
        reports.append(_evaluate_record(record, config))

    accuracy_values = [
        1.0 if report.get("is_correct") else 0.0
        for report in reports
        if report.get("judge_status") == "succeeded"
    ]
    summary = _build_evaluation_summary(
        config=config,
        reports=reports,
        accuracy_values=accuracy_values,
    )
    summary_path = config.output_dir / "evaluation_summary.json"
    write_json(summary_path, summary)
    manifest = {
        "manifest_version": "gaam_case_evaluation_v1",
        "status": "succeeded",
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "split_manifest_path": str(config.split_manifest_path) if config.split_manifest_path else None,
        "evaluation_split": config.evaluation_split,
        "selected_record_ids": [record.record_id for record in records],
        "record_id": config.record_id,
        "max_records": config.max_records,
        "memory_backend": config.memory_backend,
        "answer_backend": config.answer_backend,
        "judge_backend": config.judge_backend,
        "num_records": len(reports),
        "num_judged": len(accuracy_values),
        "accuracy": (sum(accuracy_values) / len(accuracy_values)) if accuracy_values else None,
        "summary_path": str(summary_path),
        "metrics_by_question_type": summary["metrics_by_question_type"],
        "reports": reports,
    }
    write_json(config.output_dir / "case_evaluation_manifest.json", manifest)
    return manifest


def _select_records(config: CaseEvaluationConfig) -> list[LMERecord]:
    records = LongMemEvalLoader(str(config.input_path)).load()
    if config.split_manifest_path:
        allowed_ids = _record_ids_from_split_manifest(
            config.split_manifest_path,
            config.evaluation_split,
        )
        records = [record for record in records if record.record_id in allowed_ids]
    if config.record_id:
        records = [record for record in records if record.record_id == config.record_id]
    if config.max_records is not None:
        records = records[: config.max_records]
    records = [record for record in records if record.question and record.answer]
    if not records:
        raise ValueError("No evaluable records selected. Records must contain question and answer fields.")
    return records


def _record_ids_from_split_manifest(
    split_manifest_path: Path,
    evaluation_split: str | None,
) -> set[str]:
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"split_manifest not found: {split_manifest_path}")
    split_data = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    records = split_data.get("records", []) if isinstance(split_data, dict) else []
    if not isinstance(records, list):
        raise ValueError(f"split_manifest records field is not a list: {split_manifest_path}")

    split = (evaluation_split or "test").strip().lower()
    if split not in {"train", "dev", "test", "all"}:
        raise ValueError(
            f"Unsupported evaluation_split: {evaluation_split}. "
            "Expected train, dev, test, or all."
        )

    selected: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            continue
        record_id = str(item.get("record_id", "")).strip()
        item_split = str(item.get("split", "")).strip().lower()
        if record_id and (split == "all" or item_split == split):
            selected.add(record_id)
    if not selected:
        raise ValueError(
            f"No records found for evaluation_split={split!r} in split manifest: "
            f"{split_manifest_path}"
        )
    return selected


def _evaluate_record(record: LMERecord, config: CaseEvaluationConfig) -> dict[str, Any]:
    record_dir = config.output_dir / record.record_id
    record_dir.mkdir(parents=True, exist_ok=True)

    current_memory = _build_memory(record, config, record_dir)
    memory_path = record_dir / "current_memory.json"
    write_json(memory_path, current_memory)

    answer_report = _answer_record_question(record, current_memory, config, memory_path)
    answer_path = record_dir / "answer_report.json"
    write_json(answer_path, answer_report.model_dump(mode="json"))

    judge_report = _judge_answer(record, answer_report.model_dump(mode="json"), config)
    judge_path = record_dir / "judge_report.json"
    write_json(judge_path, judge_report)

    return {
        "record_id": record.record_id,
        "question_id": stable_id("benchmark_question", record.record_id),
        "question_type": record.question_type,
        "memory_path": str(memory_path),
        "answer_report_path": str(answer_path),
        "judge_report_path": str(judge_path),
        "judge_status": judge_report.get("status"),
        "is_correct": judge_report.get("is_correct"),
        "score": judge_report.get("score"),
    }


def _build_evaluation_summary(
    *,
    config: CaseEvaluationConfig,
    reports: list[dict[str, Any]],
    accuracy_values: list[float],
) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    unjudged_record_ids: list[str] = []
    failed_record_ids: list[str] = []

    for report in reports:
        q_type = str(report.get("question_type") or "unknown")
        bucket = by_type.setdefault(
            q_type,
            {
                "num_records": 0,
                "num_judged": 0,
                "num_correct": 0,
                "accuracy": None,
                "scores": [],
            },
        )
        bucket["num_records"] += 1
        if report.get("judge_status") == "succeeded":
            bucket["num_judged"] += 1
            if report.get("is_correct"):
                bucket["num_correct"] += 1
            try:
                bucket["scores"].append(float(report.get("score")))
            except Exception:
                pass
        else:
            unjudged_record_ids.append(str(report.get("record_id", "")))
        if report.get("judge_status") == "succeeded" and not report.get("is_correct"):
            failed_record_ids.append(str(report.get("record_id", "")))

    for bucket in by_type.values():
        if bucket["num_judged"]:
            bucket["accuracy"] = bucket["num_correct"] / bucket["num_judged"]
            bucket["mean_score"] = (
                sum(bucket["scores"]) / len(bucket["scores"])
                if bucket["scores"]
                else None
            )
        else:
            bucket["mean_score"] = None
        del bucket["scores"]

    return {
        "manifest_version": "gaam_case_evaluation_summary_v1",
        "status": "succeeded",
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "split_manifest_path": str(config.split_manifest_path) if config.split_manifest_path else None,
        "evaluation_split": config.evaluation_split,
        "memory_backend": config.memory_backend,
        "answer_backend": config.answer_backend,
        "judge_backend": config.judge_backend,
        "num_records": len(reports),
        "num_judged": len(accuracy_values),
        "accuracy": (sum(accuracy_values) / len(accuracy_values)) if accuracy_values else None,
        "metrics_by_question_type": by_type,
        "unjudged_record_ids": [rid for rid in unjudged_record_ids if rid],
        "failed_record_ids": [rid for rid in failed_record_ids if rid],
        "artifact_paths": {
            "manifest": str(config.output_dir / "case_evaluation_manifest.json"),
            "summary": str(config.output_dir / "evaluation_summary.json"),
        },
    }


def _build_memory(record: LMERecord, config: CaseEvaluationConfig, record_dir: Path) -> dict[str, Any]:
    if config.memory_backend == "baseline":
        history = raw_history_from_lme_record(record)
        return BaselineMemoryBuilder().build_case_memory(history)

    if config.memory_backend in {"stateful_api", "stateful_local_hf"}:
        llm_backend = "api" if config.memory_backend == "stateful_api" else "local_hf"
        if llm_backend == "api":
            if not config.memory_api_key:
                raise ValueError("stateful_api memory backend requires GAAM_MEMORY_BUILDER_API_KEY")
            llm = OpenAICompatibleLLM(
                model=config.memory_model,
                api_key=config.memory_api_key,
                base_url=config.memory_base_url,
                temperature=0.0,
            )
        else:
            if not Path(config.memory_model).exists():
                raise FileNotFoundError(f"stateful_local_hf memory model path not found: {config.memory_model}")
            llm = LocalHFChatLLM(
                model_path=config.memory_model,
                max_new_tokens=config.memory_max_new_tokens,
                temperature=0.0,
                device_map=config.memory_device_map,
                torch_dtype=config.memory_torch_dtype,
            )
        mem_config = StatefulIncrementalMemoryConfig(
            input_path=config.input_path,
            output_dir=record_dir / "stateful_memory_build",
            record_id=record.record_id,
            session_chunk_size=config.session_chunk_size,
            max_chunk_chars=config.max_chunk_chars,
            max_previous_memory_chars=config.max_previous_memory_chars,
            llm_backend=llm_backend,
            model=config.memory_model,
            api_key=config.memory_api_key,
            base_url=config.memory_base_url,
            max_new_tokens=config.memory_max_new_tokens,
            device_map=config.memory_device_map,
            torch_dtype=config.memory_torch_dtype,
        )
        report = build_record_stateful_incremental_memory(record, mem_config, llm)
        return json.loads(Path(report["final_memory_path"]).read_text(encoding="utf-8"))

    raise ValueError(f"Unsupported memory_backend: {config.memory_backend}")


def _answer_record_question(
    record: LMERecord,
    current_memory: dict[str, Any],
    config: CaseEvaluationConfig,
    memory_path: Path,
):
    if config.answer_backend == "api":
        if not config.answer_api_key:
            raise ValueError("api answer backend requires GAAM_ANSWERER_API_KEY")
        llm = OpenAICompatibleLLM(
            model=config.answer_model,
            api_key=config.answer_api_key,
            base_url=config.answer_base_url,
            temperature=0.0,
        )
        answerer = FrozenAnswerer(llm=llm, no_llm=False, model_id=config.answer_model)
    elif config.answer_backend == "local_hf":
        if not Path(config.answer_model).exists():
            raise FileNotFoundError(f"local_hf answer model path not found: {config.answer_model}")
        llm = LocalHFChatLLM(
            model_path=config.answer_model,
            max_new_tokens=config.answer_max_new_tokens,
            temperature=0.0,
            device_map=config.answer_device_map,
            torch_dtype=config.answer_torch_dtype,
        )
        answerer = FrozenAnswerer(llm=llm, no_llm=False, model_id=config.answer_model)
    elif config.answer_backend == "no_llm":
        answerer = FrozenAnswerer(no_llm=True)
    else:
        raise ValueError(f"Unsupported answer_backend: {config.answer_backend}")

    request = AnswerRequest(
        record_id=record.record_id,
        question_id=stable_id("benchmark_question", record.record_id),
        question=str(record.question),
        current_memory_path=str(memory_path),
        metadata={},
    )
    return answerer.answer(current_memory=current_memory, request=request)


def _judge_answer(record: LMERecord, answer_report: dict[str, Any], config: CaseEvaluationConfig) -> dict[str, Any]:
    prediction = answer_report.get("prediction", "")
    expected_answer = str(record.answer or "")

    if config.judge_backend == "heuristic":
        score = score_answer_correctness(prediction, expected_answer)
        return {
            "status": "succeeded",
            "judge_backend": "heuristic",
            "score": score,
            "is_correct": score >= 0.7,
            "rationale": "Heuristic exact/containment/token-F1 answer check.",
        }

    if config.judge_backend != "api":
        raise ValueError(f"Unsupported judge_backend: {config.judge_backend}")
    if not config.judge_api_key:
        raise ValueError("api judge backend requires GAAM_EVAL_JUDGE_API_KEY")

    prompt = _answer_judge_prompt(
        question=str(record.question or ""),
        answer=expected_answer,
        prediction=prediction,
        question_type=str(record.question_type or ""),
    )
    try:
        llm = OpenAICompatibleLLM(
            model=config.judge_model,
            api_key=config.judge_api_key,
            base_url=config.judge_base_url,
            temperature=0.0,
        )
        parsed = llm.chat_json(
            system="You are a strict LongMemEval answer judge. Return only JSON.",
            user=prompt,
        )
        score = float(parsed.get("score", 1.0 if parsed.get("is_correct") else 0.0))
        return {
            "status": "succeeded",
            "judge_backend": "api",
            "judge_model": config.judge_model,
            "judge_base_url": config.judge_base_url,
            "score": max(0.0, min(1.0, score)),
            "is_correct": bool(parsed.get("is_correct", score >= 0.7)),
            "rationale": str(parsed.get("rationale", "")),
            "judge_client": "OpenAICompatibleLLM",
            "raw_judge": parsed,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "judge_backend": "api",
            "judge_model": config.judge_model,
            "judge_base_url": config.judge_base_url,
            "score": None,
            "is_correct": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _answer_judge_prompt(*, question: str, answer: str, prediction: str, question_type: str) -> str:
    return (
        "I will give you a LongMemEval question, the correct answer/rubric, and a model response. "
        "Judge whether the model response is correct. If the question is preference/personalized, "
        "the response is correct when it satisfies the rubric with correct personal information. "
        "If the question is temporal, do not over-penalize harmless off-by-one duration errors. "
        "Return JSON with keys is_correct (boolean), score (0 to 1), and rationale.\n\n"
        f"Question type: {question_type}\n\n"
        f"Question:\n{question}\n\n"
        f"Correct answer or rubric:\n{answer}\n\n"
        f"Model response:\n{prediction}"
    )


def _env_first_nonempty(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value)
    return default


def _env_int_first_nonempty(*names: str, default: int) -> int:
    value = _env_first_nonempty(*names, default=str(default))
    try:
        return int(value)
    except Exception:
        return default
