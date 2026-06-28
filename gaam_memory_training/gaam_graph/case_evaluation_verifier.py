"""Verification for GAAM case-evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from gaam_graph.lme_loader import LongMemEvalLoader
from gaam_graph.utils import write_json


FORBIDDEN_MEMORY_KEYS = {
    "question",
    "target_question",
    "benchmark_question",
    "answer",
    "target_answer",
    "gold_answer",
    "oracle_answer",
    "haystack_question_type",
}


@dataclass(frozen=True)
class CaseEvaluationVerificationConfig:
    output_dir: Path
    input_path: Path | None = None
    allow_unjudged: bool = False
    strict_answer_leakage: bool = False
    write_report: bool = True


def verify_case_evaluation_output(config: CaseEvaluationVerificationConfig) -> dict[str, Any]:
    output_dir = config.output_dir
    manifest_path = output_dir / "case_evaluation_manifest.json"
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        report = _report(
            status="failed",
            output_dir=output_dir,
            manifest_path=manifest_path,
            errors=[f"Missing case evaluation manifest: {manifest_path}"],
            warnings=warnings,
        )
        return _maybe_write(report, output_dir, config.write_report)

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report = _report(
            status="failed",
            output_dir=output_dir,
            manifest_path=manifest_path,
            errors=[f"Failed to parse case evaluation manifest: {type(exc).__name__}: {exc}"],
            warnings=warnings,
        )
        return _maybe_write(report, output_dir, config.write_report)

    if manifest.get("status") != "succeeded":
        errors.append(f"Manifest status is not succeeded: {manifest.get('status')}")

    input_path = config.input_path or _manifest_input_path(manifest)
    record_lookup = _load_record_lookup(input_path, errors, warnings)

    reports = manifest.get("reports", [])
    if not isinstance(reports, list):
        errors.append("Manifest reports field is not a list")
        reports = []

    if manifest.get("num_records") != len(reports):
        errors.append(
            f"num_records mismatch: manifest={manifest.get('num_records')} reports={len(reports)}"
        )

    judged_values: list[float] = []
    scanned_memory_files = 0
    for index, item in enumerate(reports):
        if not isinstance(item, dict):
            errors.append(f"Report #{index} is not an object")
            continue
        _verify_report_item(
            item=item,
            index=index,
            output_dir=output_dir,
            record_lookup=record_lookup,
            strict_answer_leakage=config.strict_answer_leakage,
            errors=errors,
            warnings=warnings,
        )
        if item.get("judge_status") == "succeeded":
            judged_values.append(1.0 if item.get("is_correct") else 0.0)
        elif not config.allow_unjudged:
            errors.append(
                f"{item.get('record_id', f'report #{index}')}: judge_status is not succeeded: "
                f"{item.get('judge_status')}"
            )
        memory_path = _resolve_path(item.get("memory_path"), output_dir)
        if memory_path and memory_path.exists():
            scanned_memory_files += 1

    if manifest.get("num_judged") != len(judged_values):
        errors.append(
            f"num_judged mismatch: manifest={manifest.get('num_judged')} judged_reports={len(judged_values)}"
        )

    expected_accuracy = (sum(judged_values) / len(judged_values)) if judged_values else None
    manifest_accuracy = manifest.get("accuracy")
    if expected_accuracy is None:
        if manifest_accuracy is not None:
            errors.append(f"accuracy mismatch: manifest={manifest_accuracy} expected=None")
    else:
        try:
            if abs(float(manifest_accuracy) - expected_accuracy) > 1e-9:
                errors.append(
                    f"accuracy mismatch: manifest={manifest_accuracy} expected={expected_accuracy}"
                )
        except Exception:
            errors.append(f"accuracy is not numeric: {manifest_accuracy}")

    _verify_summary(
        manifest=manifest,
        output_dir=output_dir,
        reports=reports,
        expected_accuracy=expected_accuracy,
        errors=errors,
        warnings=warnings,
    )

    report = _report(
        status="passed" if not errors else "failed",
        output_dir=output_dir,
        manifest_path=manifest_path,
        errors=errors,
        warnings=warnings,
        metrics={
            "num_reports": len(reports),
            "num_judged": len(judged_values),
            "scanned_memory_files": scanned_memory_files,
            "accuracy": expected_accuracy,
        },
    )
    return _maybe_write(report, output_dir, config.write_report)


def _verify_summary(
    *,
    manifest: dict[str, Any],
    output_dir: Path,
    reports: list[Any],
    expected_accuracy: float | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    summary_path = _resolve_path(manifest.get("summary_path"), output_dir)
    if not summary_path:
        warnings.append("Manifest does not contain summary_path; evaluation summary verification skipped.")
        return
    if not summary_path.exists():
        errors.append(f"evaluation summary does not exist: {summary_path}")
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Failed to parse evaluation summary: {type(exc).__name__}: {exc}")
        return

    if summary.get("manifest_version") != "gaam_case_evaluation_summary_v1":
        errors.append(f"Unexpected evaluation summary version: {summary.get('manifest_version')}")
    if summary.get("num_records") != manifest.get("num_records"):
        errors.append(
            f"summary num_records mismatch: summary={summary.get('num_records')} "
            f"manifest={manifest.get('num_records')}"
        )
    if summary.get("num_judged") != manifest.get("num_judged"):
        errors.append(
            f"summary num_judged mismatch: summary={summary.get('num_judged')} "
            f"manifest={manifest.get('num_judged')}"
        )
    _compare_optional_float(
        label="summary accuracy",
        actual=summary.get("accuracy"),
        expected=expected_accuracy,
        errors=errors,
    )

    expected_by_type = _expected_metrics_by_question_type(reports)
    summary_by_type = summary.get("metrics_by_question_type")
    manifest_by_type = manifest.get("metrics_by_question_type")
    if summary_by_type != expected_by_type:
        errors.append(
            "summary metrics_by_question_type mismatch: "
            f"summary={summary_by_type} expected={expected_by_type}"
        )
    if manifest_by_type != expected_by_type:
        errors.append(
            "manifest metrics_by_question_type mismatch: "
            f"manifest={manifest_by_type} expected={expected_by_type}"
        )


def _expected_metrics_by_question_type(reports: list[Any]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for item in reports:
        if not isinstance(item, dict):
            continue
        q_type = str(item.get("question_type") or "unknown")
        bucket = buckets.setdefault(
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
        if item.get("judge_status") == "succeeded":
            bucket["num_judged"] += 1
            if item.get("is_correct"):
                bucket["num_correct"] += 1
            try:
                bucket["scores"].append(float(item.get("score")))
            except Exception:
                pass
    for bucket in buckets.values():
        if bucket["num_judged"]:
            bucket["accuracy"] = bucket["num_correct"] / bucket["num_judged"]
            bucket["mean_score"] = sum(bucket["scores"]) / len(bucket["scores"]) if bucket["scores"] else None
        else:
            bucket["mean_score"] = None
        del bucket["scores"]
    return buckets


def _compare_optional_float(
    *,
    label: str,
    actual: Any,
    expected: float | None,
    errors: list[str],
) -> None:
    if expected is None:
        if actual is not None:
            errors.append(f"{label} mismatch: actual={actual} expected=None")
        return
    try:
        if abs(float(actual) - expected) > 1e-9:
            errors.append(f"{label} mismatch: actual={actual} expected={expected}")
    except Exception:
        errors.append(f"{label} is not numeric: {actual}")


def _verify_report_item(
    *,
    item: dict[str, Any],
    index: int,
    output_dir: Path,
    record_lookup: dict[str, dict[str, str]],
    strict_answer_leakage: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    record_id = str(item.get("record_id") or f"report #{index}")
    for field in ["memory_path", "answer_report_path", "judge_report_path"]:
        path = _resolve_path(item.get(field), output_dir)
        if not path:
            errors.append(f"{record_id}: missing {field}")
        elif not path.exists():
            errors.append(f"{record_id}: {field} does not exist: {path}")

    memory_path = _resolve_path(item.get("memory_path"), output_dir)
    if not memory_path or not memory_path.exists():
        return

    try:
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{record_id}: failed to parse current memory: {type(exc).__name__}: {exc}")
        return

    forbidden_keys = sorted(set(_find_forbidden_keys(memory)))
    if forbidden_keys:
        errors.append(f"{record_id}: current memory contains forbidden keys: {forbidden_keys}")

    serialized = json.dumps(memory, ensure_ascii=False)
    target = record_lookup.get(record_id, {})
    target_question = target.get("question", "").strip()
    if len(target_question) >= 12 and target_question in serialized:
        errors.append(f"{record_id}: current memory leaks the exact benchmark target question")

    target_answer = target.get("answer", "").strip()
    if len(target_answer) >= 12 and target_answer in serialized:
        message = f"{record_id}: current memory contains the exact benchmark answer/rubric"
        if strict_answer_leakage:
            errors.append(message)
        else:
            warnings.append(message)


def _manifest_input_path(manifest: dict[str, Any]) -> Path | None:
    value = manifest.get("input_path")
    return Path(value) if value else None


def _load_record_lookup(
    input_path: Path | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, dict[str, str]]:
    if input_path is None:
        warnings.append("No input path available; exact target question leakage scan is limited.")
        return {}
    if not input_path.exists():
        errors.append(f"Input path does not exist: {input_path}")
        return {}
    try:
        records = LongMemEvalLoader(str(input_path)).load()
    except Exception as exc:
        errors.append(f"Failed to load input records: {type(exc).__name__}: {exc}")
        return {}
    return {
        record.record_id: {
            "question": str(record.question or ""),
            "answer": str(record.answer or ""),
        }
        for record in records
    }


def _resolve_path(value: Any, output_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    output_relative = output_dir / path
    if output_relative.exists():
        return output_relative
    return path


def _find_forbidden_keys(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalize_key(str(key))
            if normalized in FORBIDDEN_MEMORY_KEYS:
                found.append(str(key))
            found.extend(_find_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_forbidden_keys(child))
    return found


def _normalize_key(key: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in key.lower()).strip("_")


def _report(
    *,
    status: str,
    output_dir: Path,
    manifest_path: Path,
    errors: list[str],
    warnings: list[str],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "manifest_version": "gaam_case_evaluation_verification_v1",
        "status": status,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "metrics": metrics or {},
        "errors": errors,
        "warnings": warnings,
    }


def _maybe_write(report: dict[str, Any], output_dir: Path, write_report: bool) -> dict[str, Any]:
    if write_report:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "case_evaluation_verification.json", report)
    return report
