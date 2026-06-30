"""Production-path regression tests for GAAM training/evaluation contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gaam_graph.case_evaluation import CaseEvaluationConfig, run_case_evaluation
from gaam_graph.native_verl_grpo import NativeVerlExportConfig, export_native_verl_grpo_dataset
from gaam_graph.stateful_incremental_memory import StatefulIncrementalMemoryConfig
from gaam_graph.verl_gaam_reward import compute_score


def _minimal_case(record_id: str = "case_contract") -> dict:
    return {
        "id": record_id,
        "question": f"LEAK_TARGET_QUESTION_{record_id}: what should the model answer?",
        "answer": f"LEAK_GOLD_RUBRIC_{record_id}",
        "question_type": "personal_fact",
        "sessions": [
            {
                "session_id": "sess_001",
                "timestamp": "2024-01-01",
                "messages": [
                    {
                        "role": "user",
                        "content": "I prefer Python for data analysis and I use pandas often.",
                    },
                    {
                        "role": "assistant",
                        "content": "I will remember that preference.",
                    },
                ],
            }
        ],
    }


def _write_minimal_case(path: Path) -> None:
    data = [_minimal_case()]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _write_multi_case(path: Path, record_ids: list[str]) -> None:
    path.write_text(
        json.dumps([_minimal_case(record_id) for record_id in record_ids], ensure_ascii=False),
        encoding="utf-8",
    )


def _write_minimal_oracle_graph(graph_dir: Path, record_id: str = "case_contract") -> None:
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph = {
        "record_id": record_id,
        "nodes": [
            {
                "id": "oracle_fact_001",
                "type": "fact",
                "attrs": {
                    "text": "The user prefers Python for data analysis and often uses pandas.",
                    "topic": "programming preference",
                },
            },
            {
                "id": "oracle_event_001",
                "type": "event",
                "attrs": {
                    "text": "The user stated a Python and pandas preference.",
                    "event_category": "preference",
                },
            },
        ],
        "edges": [
            {"source": "oracle_event_001", "target": "oracle_fact_001", "type": "SUPPORTS"}
        ],
    }
    (graph_dir / f"{record_id}.graph.json").write_text(
        json.dumps(graph, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_split_manifest(
    path: Path,
    input_path: Path,
    graph_dir: Path,
    record_ids: list[str],
    *,
    split: str = "train",
) -> None:
    path.write_text(
        json.dumps(
            {
                "manifest_version": "phase4_dataset_split_v1",
                "dataset_name": "longmemeval",
                "source_records_path": str(input_path),
                "oracle_graph_dir": str(graph_dir),
                "seed": 0,
                "split_ratios": {"train": 1.0, "dev": 0.0, "test": 0.0},
                "created_at": "2026-01-01T00:00:00",
                "records": [
                    {
                        "record_id": record_id,
                        "split": split,
                        "source_index": index,
                        "raw_record_path": str(input_path),
                        "oracle_graph_path": str(graph_dir / f"{record_id}.graph.json"),
                        "has_benchmark_question": True,
                        "has_benchmark_answer": True,
                        "metadata": {"event_count": 1},
                    }
                    for index, record_id in enumerate(record_ids)
                ],
                "counts": {
                    "train": len(record_ids) if split == "train" else 0,
                    "dev": len(record_ids) if split == "dev" else 0,
                    "test": len(record_ids) if split == "test" else 0,
                },
                "record_id_hash": "test_hash",
                "issues": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _read_single_parquet_row(path: Path) -> dict:
    rows = pd.read_parquet(path).to_dict(orient="records")
    assert len(rows) == 1
    return _jsonable(rows[0])


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: _jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_jsonable(child) for child in value]
    return value


def test_production_readiness_warns_when_train_batch_exceeds_available_rows(tmp_path: Path):
    input_path = tmp_path / "records.json"
    graph_dir = tmp_path / "graphs"
    split_path = tmp_path / "split.json"
    model_dir = tmp_path / "model"
    code_a1_root = tmp_path / "Code-A1"
    model_dir.mkdir()
    (code_a1_root / "verl" / "verl").mkdir(parents=True)
    _write_minimal_case(input_path)
    _write_minimal_oracle_graph(graph_dir)
    _write_split_manifest(split_path, input_path, graph_dir, ["case_contract"])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_production_training_readiness.py",
            "--input",
            str(input_path),
            "--oracle_graph_dir",
            str(graph_dir),
            "--split_manifest",
            str(split_path),
            "--memory_model_path",
            str(model_dir),
            "--question_model_path",
            str(model_dir),
            "--code_a1_root",
            str(code_a1_root),
            "--train_batch_size",
            "8",
            "--questions_per_case",
            "3",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "succeeded"
    assert report["split_counts"]["train"] == 1
    assert report["train_batch_size"] == 8
    assert report["questions_per_case"] == 3
    assert any("train_batch_size=8 is larger" in warning for warning in report["warnings"])


def test_production_readiness_reward_judge_blank_key_falls_back_to_deepseek_key(
    tmp_path: Path,
    monkeypatch,
):
    input_path = tmp_path / "records.json"
    graph_dir = tmp_path / "graphs"
    split_path = tmp_path / "split.json"
    model_dir = tmp_path / "model"
    code_a1_root = tmp_path / "Code-A1"
    model_dir.mkdir()
    (code_a1_root / "verl" / "verl").mkdir(parents=True)
    _write_minimal_case(input_path)
    _write_minimal_oracle_graph(graph_dir)
    _write_split_manifest(split_path, input_path, graph_dir, ["case_contract"])
    monkeypatch.setenv("GAAM_REWARD_JUDGE_ENABLED", "1")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-fallback-key")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_production_training_readiness.py",
            "--input",
            str(input_path),
            "--oracle_graph_dir",
            str(graph_dir),
            "--split_manifest",
            str(split_path),
            "--memory_model_path",
            str(model_dir),
            "--question_model_path",
            str(model_dir),
            "--code_a1_root",
            str(code_a1_root),
            "--questions_per_case",
            "3",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "succeeded"
    assert not any("GAAM_REWARD_JUDGE_API_KEY is empty" in error for error in report["errors"])


def test_memory_builder_export_does_not_leak_benchmark_target(tmp_path: Path):
    input_path = tmp_path / "records.json"
    graph_dir = tmp_path / "graphs"
    output_dir = tmp_path / "memory_export"
    _write_minimal_case(input_path)
    _write_minimal_oracle_graph(graph_dir)

    manifest = export_native_verl_grpo_dataset(
        NativeVerlExportConfig(
            input_path=input_path,
            oracle_graph_dir=graph_dir,
            output_dir=output_dir,
            actor_role="memory_builder",
        )
    )

    row = _read_single_parquet_row(Path(manifest["train_path"]))
    serialized_prompt = json.dumps(row["prompt"], ensure_ascii=False)

    assert "LEAK_TARGET_QUESTION" not in serialized_prompt
    assert "LEAK_GOLD_RUBRIC" not in serialized_prompt
    assert "Python for data analysis" in serialized_prompt
    assert row["reward_model"]["ground_truth"]["actor_role"] == "memory_builder"


def test_question_agent_export_uses_configured_question_count_without_target_leakage(tmp_path: Path):
    input_path = tmp_path / "records.json"
    graph_dir = tmp_path / "graphs"
    output_dir = tmp_path / "question_export"
    _write_minimal_case(input_path)
    _write_minimal_oracle_graph(graph_dir)

    manifest = export_native_verl_grpo_dataset(
        NativeVerlExportConfig(
            input_path=input_path,
            oracle_graph_dir=graph_dir,
            output_dir=output_dir,
            actor_role="question_agent",
            questions_per_case=3,
        )
    )

    row = _read_single_parquet_row(Path(manifest["train_path"]))
    serialized_row = json.dumps(row, ensure_ascii=False)

    assert "Generate exactly 3 questions" in serialized_row
    assert row["reward_model"]["ground_truth"]["questions_per_case"] == 3
    assert "LEAK_TARGET_QUESTION" not in serialized_row
    assert "LEAK_GOLD_RUBRIC" not in serialized_row


def test_case_evaluation_builds_memory_before_using_target_question(tmp_path: Path):
    input_path = tmp_path / "records.json"
    output_dir = tmp_path / "case_eval"
    _write_minimal_case(input_path)

    manifest = run_case_evaluation(
        CaseEvaluationConfig(
            input_path=input_path,
            output_dir=output_dir,
            memory_backend="baseline",
            answer_backend="no_llm",
            judge_backend="heuristic",
        )
    )

    assert manifest["status"] == "succeeded"
    assert manifest["num_records"] == 1
    assert (output_dir / "evaluation_summary.json").exists()
    assert manifest["summary_path"] == str(output_dir / "evaluation_summary.json")
    assert manifest["metrics_by_question_type"]["personal_fact"]["num_records"] == 1

    memory_path = output_dir / "case_contract" / "current_memory.json"
    serialized_memory = memory_path.read_text(encoding="utf-8")
    assert "LEAK_TARGET_QUESTION" not in serialized_memory
    assert "LEAK_GOLD_RUBRIC" not in serialized_memory


def test_case_evaluation_can_restrict_to_test_split(tmp_path: Path):
    input_path = tmp_path / "records.json"
    output_dir = tmp_path / "case_eval_split"
    graph_dir = tmp_path / "graphs"
    split_path = tmp_path / "split.json"
    _write_multi_case(input_path, ["case_train", "case_test"])
    _write_minimal_oracle_graph(graph_dir, "case_train")
    _write_minimal_oracle_graph(graph_dir, "case_test")
    split_path.write_text(
        json.dumps(
            {
                "manifest_version": "phase4_dataset_split_v1",
                "dataset_name": "longmemeval",
                "source_records_path": str(input_path),
                "oracle_graph_dir": str(graph_dir),
                "seed": 0,
                "split_ratios": {"train": 0.5, "dev": 0.0, "test": 0.5},
                "created_at": "2026-01-01T00:00:00",
                "records": [
                    {
                        "record_id": "case_train",
                        "split": "train",
                        "source_index": 0,
                        "raw_record_path": str(input_path),
                        "oracle_graph_path": str(graph_dir / "case_train.graph.json"),
                        "has_benchmark_question": True,
                        "has_benchmark_answer": True,
                        "metadata": {"event_count": 1},
                    },
                    {
                        "record_id": "case_test",
                        "split": "test",
                        "source_index": 1,
                        "raw_record_path": str(input_path),
                        "oracle_graph_path": str(graph_dir / "case_test.graph.json"),
                        "has_benchmark_question": True,
                        "has_benchmark_answer": True,
                        "metadata": {"event_count": 1},
                    },
                ],
                "counts": {"train": 1, "dev": 0, "test": 1},
                "record_id_hash": "test_hash",
                "issues": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest = run_case_evaluation(
        CaseEvaluationConfig(
            input_path=input_path,
            output_dir=output_dir,
            split_manifest_path=split_path,
            evaluation_split="test",
            memory_backend="baseline",
            answer_backend="no_llm",
            judge_backend="heuristic",
        )
    )

    assert manifest["status"] == "succeeded"
    assert manifest["split_manifest_path"] == str(split_path)
    assert manifest["evaluation_split"] == "test"
    assert manifest["selected_record_ids"] == ["case_test"]
    assert manifest["num_records"] == 1
    assert (output_dir / "case_test" / "current_memory.json").exists()
    assert not (output_dir / "case_train" / "current_memory.json").exists()


def test_case_evaluation_blank_deepseek_env_falls_back_to_openai_env(tmp_path: Path, monkeypatch):
    """Evaluation config should not let blank DeepSeek values mask OpenAI-compatible fallbacks."""
    monkeypatch.setenv("GAAM_MEMORY_BUILDER_MODEL", "")
    monkeypatch.setenv("GAAM_MEMORY_BUILDER_BASE_URL", "")
    monkeypatch.setenv("GAAM_MEMORY_BUILDER_API_KEY", "")
    monkeypatch.setenv("GAAM_ANSWERER_MODEL", "")
    monkeypatch.setenv("GAAM_ANSWERER_BASE_URL", "")
    monkeypatch.setenv("GAAM_ANSWERER_API_KEY", "")
    monkeypatch.setenv("GAAM_EVAL_JUDGE_MODEL", "")
    monkeypatch.setenv("GAAM_EVAL_JUDGE_BASE_URL", "")
    monkeypatch.setenv("GAAM_EVAL_JUDGE_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_MODEL", "")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai-compatible.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")

    config = CaseEvaluationConfig(input_path=tmp_path / "records.json", output_dir=tmp_path / "eval")
    memory_config = StatefulIncrementalMemoryConfig(
        input_path=tmp_path / "records.json",
        output_dir=tmp_path / "memory",
    )

    assert config.memory_base_url == "https://openai-compatible.example/v1"
    assert config.memory_api_key == "fallback-key"
    assert config.answer_base_url == "https://openai-compatible.example/v1"
    assert config.answer_api_key == "fallback-key"
    assert config.judge_base_url == "https://openai-compatible.example/v1"
    assert config.judge_api_key == "fallback-key"
    assert memory_config.base_url == "https://openai-compatible.example/v1"
    assert memory_config.api_key == "fallback-key"


def test_case_evaluation_api_judge_uses_openai_compatible_llm_without_memory_leakage(
    tmp_path: Path,
    monkeypatch,
):
    input_path = tmp_path / "records.json"
    output_dir = tmp_path / "case_eval_api_judge"
    _write_minimal_case(input_path)
    calls = []

    class DummyJudgeLLM:
        def __init__(self, **kwargs):
            calls.append({"init": kwargs})

        def chat_json(self, system: str, user: str):
            calls.append({"system": system, "user": user})
            assert "LongMemEval answer judge" in system
            assert "LEAK_TARGET_QUESTION" in user
            assert "LEAK_GOLD_RUBRIC" in user
            return {
                "is_correct": True,
                "score": 1.0,
                "rationale": "The answer matches the rubric.",
            }

    import gaam_graph.case_evaluation as case_evaluation

    monkeypatch.setattr(case_evaluation, "OpenAICompatibleLLM", DummyJudgeLLM)

    manifest = run_case_evaluation(
        CaseEvaluationConfig(
            input_path=input_path,
            output_dir=output_dir,
            memory_backend="baseline",
            answer_backend="no_llm",
            judge_backend="api",
            judge_model="judge-model",
            judge_base_url="https://judge.example/v1",
            judge_api_key="judge-key",
        )
    )

    assert manifest["status"] == "succeeded"
    assert manifest["accuracy"] == 1.0
    assert calls[0]["init"]["model"] == "judge-model"
    assert calls[0]["init"]["base_url"] == "https://judge.example/v1"
    judge_report = json.loads((output_dir / "case_contract" / "judge_report.json").read_text(encoding="utf-8"))
    assert judge_report["judge_client"] == "OpenAICompatibleLLM"
    memory_text = (output_dir / "case_contract" / "current_memory.json").read_text(encoding="utf-8")
    assert "LEAK_TARGET_QUESTION" not in memory_text
    assert "LEAK_GOLD_RUBRIC" not in memory_text


def test_required_llm_judge_missing_key_forces_zero_reward(monkeypatch):
    monkeypatch.setenv("GAAM_REWARD_JUDGE_ENABLED", "1")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_REQUIRED", "1")
    monkeypatch.delenv("GAAM_REWARD_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    details = compute_score(
        "gaam_memory_builder",
        '{"memory_graph":{"nodes":[]},"memory_summaries":{"summary":"preference update"}}',
        {
            "actor_role": "memory_builder",
            "required_terms": ["preference"],
            "oracle_digest": "preference update",
        },
    )

    assert details["score"] == 0.0
    assert details["llm_judge"]["status"] == "skipped"


def test_reward_hard_fails_memory_output_with_target_question_key():
    details = compute_score(
        "gaam_memory_builder",
        json.dumps(
            {
                "memory_graph": {"nodes": []},
                "memory_summaries": {"summary": "safe"},
                "target_question": "LEAK_TARGET_QUESTION: what should the model answer?",
            }
        ),
        {"actor_role": "memory_builder", "required_terms": ["safe"]},
    )

    assert details["score"] == 0.0
    assert details["hard_failure"]["reason"] == "forbidden_output_keys"


def test_question_agent_reward_allows_question_field_but_blocks_gold_answer_key():
    allowed = compute_score(
        "gaam_question_agent",
        json.dumps(
            {
                "questions": [
                    {
                        "question": "What stable preference did the user express about Python?",
                        "type": "personal_fact",
                    }
                ]
            }
        ),
        {
            "actor_role": "question_agent",
            "required_terms": ["python"],
            "questions_per_case": 1,
        },
    )
    blocked = compute_score(
        "gaam_question_agent",
        json.dumps(
            {
                "questions": [
                    {
                        "question": "What stable preference did the user express about Python?",
                        "gold_answer": "Python for data analysis",
                    }
                ]
            }
        ),
        {
            "actor_role": "question_agent",
            "required_terms": ["python"],
            "questions_per_case": 1,
        },
    )

    assert "hard_failure" not in allowed
    assert blocked["score"] == 0.0
    assert blocked["hard_failure"]["reason"] == "forbidden_output_keys"


def test_reward_llm_judge_uses_openai_compatible_api(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            content = json.dumps(
                {
                    "oracle_coverage": 1.0,
                    "groundedness": 1.0,
                    "non_redundancy": 1.0,
                    "compression_balance": 1.0,
                    "abstraction_quality": 1.0,
                    "leakage_safety": 1.0,
                    "overall": 1.0,
                    "rationale": "Looks grounded.",
                }
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append({"client": kwargs})
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("GAAM_REWARD_JUDGE_ENABLED", "1")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_REQUIRED", "1")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_MODEL", "judge-model")

    details = compute_score(
        "gaam_memory_builder",
        json.dumps(
            {
                "memory_graph": {"nodes": [{"type": "fact", "content": "Python preference"}]},
                "memory_summaries": {"summary": "preference update evidence"},
            }
        ),
        {
            "actor_role": "memory_builder",
            "required_terms": ["python", "preference"],
            "oracle_digest": "The user prefers Python for data analysis.",
        },
    )

    assert details["llm_judge"]["status"] == "succeeded"
    assert details["llm_judge"]["base_url"] == "https://judge.example/v1"
    assert details["llm_judge"]["model"] == "judge-model"
    assert details["llm_judge"]["config"]["api_key_configured"] is True
    assert calls[0]["client"]["api_key"] == "test-key"
    assert calls[0]["client"]["base_url"] == "https://judge.example/v1"
    assert calls[1]["model"] == "judge-model"


def test_reward_llm_judge_blank_env_values_do_not_mask_fallbacks(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "oracle_validity": 1.0,
                                    "answerability": 1.0,
                                    "diversity": 1.0,
                                    "difficulty": 1.0,
                                    "weakness_targeting": 1.0,
                                    "non_redundancy": 1.0,
                                    "leakage_safety": 1.0,
                                    "overall": 1.0,
                                }
                            )
                        )
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append({"client": kwargs})
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("GAAM_REWARD_JUDGE_ENABLED", "1")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_REQUIRED", "1")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_BASE_URL", "")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_API_KEY", "")
    monkeypatch.setenv("GAAM_REWARD_JUDGE_MODEL", "")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fallback-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    details = compute_score(
        "gaam_question_agent",
        json.dumps(
            {
                "questions": [
                    {
                        "question": "What stable preference appears across sessions?",
                        "type": "multi_session",
                    }
                ]
            }
        ),
        {
            "actor_role": "question_agent",
            "required_terms": ["preference"],
            "oracle_digest": "The user repeatedly expressed a stable preference.",
            "questions_per_case": 1,
        },
    )

    assert details["llm_judge"]["status"] == "succeeded"
    assert details["llm_judge"]["base_url"] == "https://api.deepseek.com"
    assert details["llm_judge"]["model"] == "deepseek-v4-flash"
    assert calls[0]["client"]["api_key"] == "fallback-key"


def test_case_evaluation_can_use_local_hf_answer_backend_without_api_key(
    tmp_path: Path,
    monkeypatch,
):
    input_path = tmp_path / "records.json"
    output_dir = tmp_path / "case_eval_local_hf"
    model_dir = tmp_path / "answer_checkpoint"
    model_dir.mkdir()
    _write_minimal_case(input_path)

    class DummyLocalHF:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def chat_json(self, system: str, user: str):
            assert "Frozen Answerer" in system
            assert "LEAK_GOLD_RUBRIC" not in user
            return {
                "prediction": "The user prefers Python for data analysis.",
                "confidence": 0.9,
                "rationale": "Retrieved memory supports this answer.",
            }

    import gaam_graph.case_evaluation as case_evaluation

    monkeypatch.setattr(case_evaluation, "LocalHFChatLLM", DummyLocalHF)

    manifest = run_case_evaluation(
        CaseEvaluationConfig(
            input_path=input_path,
            output_dir=output_dir,
            memory_backend="baseline",
            answer_backend="local_hf",
            judge_backend="heuristic",
            answer_model=str(model_dir),
        )
    )

    assert manifest["status"] == "succeeded"
    answer_report = json.loads((output_dir / "case_contract" / "answer_report.json").read_text(encoding="utf-8"))
    assert answer_report["model_info"]["model"] == str(model_dir)


def test_case_evaluation_can_use_local_hf_stateful_memory_backend(
    tmp_path: Path,
    monkeypatch,
):
    input_path = tmp_path / "records.json"
    output_dir = tmp_path / "case_eval_local_hf_memory"
    model_dir = tmp_path / "memory_checkpoint"
    model_dir.mkdir()
    _write_minimal_case(input_path)

    class DummyLocalHF:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def chat_json(self, system: str, user: str):
            assert "GAAM Memory Builder" in system
            assert "LEAK_TARGET_QUESTION" not in user
            assert "LEAK_GOLD_RUBRIC" not in user
            return {
                "record_id": "case_contract",
                "memory_graph": {
                    "nodes": [
                        {
                            "id": "mem_fact_001",
                            "type": "fact",
                            "content": "The user prefers Python for data analysis.",
                            "status": "active",
                            "confidence": 0.9,
                            "source_session_ids": ["sess_001"],
                            "source_turn_ids": ["0"],
                            "source_event_ids": ["evt_001"],
                        }
                    ],
                    "edges": [],
                },
                "memory_summaries": {
                    "user_profile": "",
                    "stable_preferences": "The user prefers Python for data analysis.",
                    "active_plans": "",
                    "recent_changes": "",
                    "cross_session_abstractions": "",
                },
                "metadata": {},
            }

    import gaam_graph.case_evaluation as case_evaluation

    monkeypatch.setattr(case_evaluation, "LocalHFChatLLM", DummyLocalHF)

    manifest = run_case_evaluation(
        CaseEvaluationConfig(
            input_path=input_path,
            output_dir=output_dir,
            memory_backend="stateful_local_hf",
            answer_backend="no_llm",
            judge_backend="heuristic",
            memory_model=str(model_dir),
        )
    )

    assert manifest["status"] == "succeeded"
    memory_text = (output_dir / "case_contract" / "current_memory.json").read_text(encoding="utf-8")
    assert "Python for data analysis" in memory_text
    assert "LEAK_TARGET_QUESTION" not in memory_text
    assert "LEAK_GOLD_RUBRIC" not in memory_text
