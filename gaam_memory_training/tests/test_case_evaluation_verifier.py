"""Tests for case-evaluation output verification."""

from __future__ import annotations

import json
from pathlib import Path

from gaam_graph.case_evaluation_verifier import (
    CaseEvaluationVerificationConfig,
    verify_case_evaluation_output,
)


def _write_case_input(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "case_eval_verify",
                    "question": "LEAK_TARGET_QUESTION: what preference should be remembered?",
                    "answer": "LEAK_GOLD_RUBRIC",
                    "question_type": "personal_fact",
                    "sessions": [
                        {
                            "session_id": "sess_001",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "I prefer Python for data analysis.",
                                }
                            ],
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_eval_output(output_dir: Path, input_path: Path, *, memory: dict, accuracy: float = 1.0) -> None:
    case_dir = output_dir / "case_eval_verify"
    case_dir.mkdir(parents=True)
    memory_path = case_dir / "current_memory.json"
    answer_path = case_dir / "answer_report.json"
    judge_path = case_dir / "judge_report.json"
    memory_path.write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")
    answer_path.write_text(json.dumps({"prediction": "Python"}, ensure_ascii=False), encoding="utf-8")
    judge_path.write_text(
        json.dumps({"status": "succeeded", "is_correct": True, "score": 1.0}, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics_by_question_type = {
        "personal_fact": {
            "num_records": 1,
            "num_judged": 1,
            "num_correct": 1,
            "accuracy": 1.0,
            "mean_score": 1.0,
        }
    }
    summary_path = output_dir / "evaluation_summary.json"
    manifest = {
        "manifest_version": "gaam_case_evaluation_v1",
        "status": "succeeded",
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "num_records": 1,
        "num_judged": 1,
        "accuracy": accuracy,
        "summary_path": str(summary_path),
        "metrics_by_question_type": metrics_by_question_type,
        "reports": [
            {
                "record_id": "case_eval_verify",
                "question_type": "personal_fact",
                "memory_path": str(memory_path),
                "answer_report_path": str(answer_path),
                "judge_report_path": str(judge_path),
                "judge_status": "succeeded",
                "is_correct": True,
                "score": 1.0,
            }
        ],
    }
    summary = {
        "manifest_version": "gaam_case_evaluation_summary_v1",
        "status": "succeeded",
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "num_records": 1,
        "num_judged": 1,
        "accuracy": accuracy,
        "metrics_by_question_type": metrics_by_question_type,
        "unjudged_record_ids": [],
        "failed_record_ids": [],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    (output_dir / "case_evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )


def test_case_evaluation_verifier_passes_clean_output(tmp_path: Path):
    input_path = tmp_path / "records.json"
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    _write_case_input(input_path)
    _write_eval_output(
        output_dir,
        input_path,
        memory={
            "memory_graph": {"nodes": [{"type": "fact", "content": "The user prefers Python."}]},
            "memory_summaries": {"summary": "The user prefers Python."},
        },
    )

    report = verify_case_evaluation_output(
        CaseEvaluationVerificationConfig(output_dir=output_dir, input_path=input_path)
    )

    assert report["status"] == "passed"
    assert report["metrics"]["num_reports"] == 1
    assert report["metrics"]["num_judged"] == 1
    assert (output_dir / "case_evaluation_verification.json").exists()


def test_case_evaluation_verifier_fails_on_target_question_leakage(tmp_path: Path):
    input_path = tmp_path / "records.json"
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    _write_case_input(input_path)
    _write_eval_output(
        output_dir,
        input_path,
        memory={
            "memory_graph": {
                "nodes": [
                    {
                        "type": "fact",
                        "content": "LEAK_TARGET_QUESTION: what preference should be remembered?",
                    }
                ]
            },
            "memory_summaries": {"summary": "unsafe"},
        },
    )

    report = verify_case_evaluation_output(
        CaseEvaluationVerificationConfig(output_dir=output_dir, input_path=input_path)
    )

    assert report["status"] == "failed"
    assert any("leaks the exact benchmark target question" in error for error in report["errors"])


def test_case_evaluation_verifier_fails_on_accuracy_mismatch(tmp_path: Path):
    input_path = tmp_path / "records.json"
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    _write_case_input(input_path)
    _write_eval_output(
        output_dir,
        input_path,
        memory={
            "memory_graph": {"nodes": [{"type": "fact", "content": "The user prefers Python."}]},
            "memory_summaries": {"summary": "The user prefers Python."},
        },
        accuracy=0.0,
    )

    report = verify_case_evaluation_output(
        CaseEvaluationVerificationConfig(output_dir=output_dir, input_path=input_path)
    )

    assert report["status"] == "failed"
    assert any("accuracy mismatch" in error for error in report["errors"])


def test_case_evaluation_verifier_fails_on_summary_mismatch(tmp_path: Path):
    input_path = tmp_path / "records.json"
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    _write_case_input(input_path)
    _write_eval_output(
        output_dir,
        input_path,
        memory={
            "memory_graph": {"nodes": [{"type": "fact", "content": "The user prefers Python."}]},
            "memory_summaries": {"summary": "The user prefers Python."},
        },
    )
    summary_path = output_dir / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metrics_by_question_type"]["personal_fact"]["num_correct"] = 0
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")

    report = verify_case_evaluation_output(
        CaseEvaluationVerificationConfig(output_dir=output_dir, input_path=input_path)
    )

    assert report["status"] == "failed"
    assert any("summary metrics_by_question_type mismatch" in error for error in report["errors"])


def test_case_evaluation_verifier_fails_on_forbidden_memory_key(tmp_path: Path):
    input_path = tmp_path / "records.json"
    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    _write_case_input(input_path)
    _write_eval_output(
        output_dir,
        input_path,
        memory={
            "memory_graph": {"nodes": [{"type": "fact", "content": "The user prefers Python."}]},
            "memory_summaries": {"summary": "The user prefers Python."},
            "target_question": "unsafe schema field",
        },
    )

    report = verify_case_evaluation_output(
        CaseEvaluationVerificationConfig(output_dir=output_dir, input_path=input_path)
    )

    assert report["status"] == "failed"
    assert any("forbidden keys" in error for error in report["errors"])
