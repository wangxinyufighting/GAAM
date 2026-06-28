"""Verification utilities for GAAM native Code-A1/VERL training outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


REQUIRED_HF_MARKERS = {"config.json"}
GRPO_LOG_MARKERS = (
    "verl.trainer.main_ppo",
    "algorithm.adv_estimator=grpo",
)


@dataclass
class NativeVerlTrainingVerificationReport:
    status: str
    output_dir: str
    manifest_path: str
    checks_passed: int = 0
    checks_failed: int = 0
    warnings: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    actor_step_reports: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "issues": self.issues,
            "metrics": self.metrics,
            "actor_step_reports": self.actor_step_reports,
        }


def verify_native_verl_training_output(
    output_dir: Path,
    *,
    require_hf_checkpoint: bool = True,
    require_grpo_log_markers: bool = True,
) -> NativeVerlTrainingVerificationReport:
    """Verify that a native dual VERL training run produced real train artifacts."""
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "dual_cotraining_manifest.json"
    report = NativeVerlTrainingVerificationReport(
        status="passed",
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
    )

    manifest = _load_manifest(manifest_path, report)
    if not manifest:
        return _finalize(report)

    _check(
        report,
        manifest.get("trainer") == "GAAMDualRayTrainer",
        "manifest trainer is GAAMDualRayTrainer",
        f"Unexpected trainer: {manifest.get('trainer')}",
    )
    _check(
        report,
        manifest.get("status") == "succeeded",
        "manifest status is succeeded",
        f"Manifest status is not succeeded: {manifest.get('status')}",
    )

    config = manifest.get("config", {})
    if isinstance(config, dict):
        _check(
            report,
            config.get("dry_run") is False,
            "run is not dry_run",
            "Run used dry_run=True; this is not real training.",
        )
        if require_hf_checkpoint:
            _check(
                report,
                config.get("save_hf_model") is True,
                "save_hf_model is enabled",
                "save_hf_model is not enabled; full HF checkpoint reuse is not guaranteed.",
            )
            _check(
                report,
                config.get("require_hf_checkpoint") is True,
                "require_hf_checkpoint is enabled",
                "require_hf_checkpoint is not enabled; actor steps could pass without updated weights.",
            )

    round_records = manifest.get("round_records", [])
    if not isinstance(round_records, list) or not round_records:
        report.issues.append("No round_records found in native dual training manifest.")
        report.checks_failed += 1
        return _finalize(report)

    final_model_paths = manifest.get("final_model_paths", {})
    actor_step_reports = []
    total_actor_steps = 0
    succeeded_actor_steps = 0
    checkpoint_actor_steps = 0

    for round_record in round_records:
        if not isinstance(round_record, dict):
            continue
        _check(
            report,
            round_record.get("status") == "succeeded",
            f"round {round_record.get('round_index')} succeeded",
            f"Round {round_record.get('round_index')} status is {round_record.get('status')}",
        )
        for actor_record in round_record.get("actor_records", []):
            if not isinstance(actor_record, dict):
                continue
            total_actor_steps += 1
            step_report = _verify_actor_step(
                actor_record,
                require_hf_checkpoint=require_hf_checkpoint,
                require_grpo_log_markers=require_grpo_log_markers,
            )
            actor_step_reports.append(step_report)
            if step_report["status"] == "passed":
                succeeded_actor_steps += 1
            if step_report.get("hf_checkpoint_valid"):
                checkpoint_actor_steps += 1
            for issue in step_report["issues"]:
                report.issues.append(
                    f"round={actor_record.get('round_index')} actor={actor_record.get('actor_role')}: {issue}"
                )
                report.checks_failed += 1
            for warning in step_report["warnings"]:
                report.warnings.append(
                    f"round={actor_record.get('round_index')} actor={actor_record.get('actor_role')}: {warning}"
                )
            if not step_report["issues"]:
                report.checks_passed += 1

    if isinstance(final_model_paths, dict):
        for actor_role, path_value in final_model_paths.items():
            path = Path(str(path_value))
            _check(
                report,
                path.exists(),
                f"final model path exists for {actor_role}",
                f"Final model path for {actor_role} does not exist: {path}",
            )
            if require_hf_checkpoint and path.exists():
                _check(
                    report,
                    _looks_like_hf_model_dir(path),
                    f"final model path is HF model for {actor_role}",
                    f"Final model path for {actor_role} is not a recognizable HF model dir: {path}",
                )

    report.actor_step_reports = actor_step_reports
    report.metrics = {
        "num_rounds": len(round_records),
        "num_actor_steps": total_actor_steps,
        "num_succeeded_actor_steps": succeeded_actor_steps,
        "num_actor_steps_with_hf_checkpoint": checkpoint_actor_steps,
    }
    return _finalize(report)


def write_native_verl_training_verification_report(
    report: NativeVerlTrainingVerificationReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_native_verl_checkpoint_paths(
    output_dir: Path,
    *,
    require_exists: bool = True,
) -> dict[str, Any]:
    """Resolve final actor checkpoint paths from a native dual training manifest."""
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "dual_cotraining_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing native dual training manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_model_paths = manifest.get("final_model_paths", {})
    if not isinstance(final_model_paths, dict):
        raise ValueError("dual_cotraining_manifest.json has no final_model_paths object")

    memory_builder = final_model_paths.get("memory_builder")
    question_agent = final_model_paths.get("question_agent")
    resolved = {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "status": manifest.get("status"),
        "memory_builder_checkpoint": str(memory_builder) if memory_builder else None,
        "question_agent_checkpoint": str(question_agent) if question_agent else None,
    }

    missing = []
    for key in ("memory_builder_checkpoint", "question_agent_checkpoint"):
        value = resolved.get(key)
        if not value:
            missing.append(key)
            continue
        if require_exists and not Path(value).exists():
            missing.append(f"{key} does not exist: {value}")
    if missing:
        raise FileNotFoundError("; ".join(missing))

    return resolved


def _load_manifest(
    manifest_path: Path,
    report: NativeVerlTrainingVerificationReport,
) -> dict[str, Any] | None:
    if not manifest_path.exists():
        report.issues.append(f"Missing native dual training manifest: {manifest_path}")
        report.checks_failed += 1
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.issues.append(f"Failed to parse native dual training manifest: {type(exc).__name__}: {exc}")
        report.checks_failed += 1
        return None


def _verify_actor_step(
    actor_record: dict[str, Any],
    *,
    require_hf_checkpoint: bool,
    require_grpo_log_markers: bool,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []

    if actor_record.get("status") != "succeeded":
        issues.append(f"Actor step status is not succeeded: {actor_record.get('status')}")
    if actor_record.get("dry_run"):
        issues.append("Actor step used dry_run=True.")
    if actor_record.get("returncode") not in (0, None):
        issues.append(f"Actor step returncode is non-zero: {actor_record.get('returncode')}")

    dataset_dir = Path(str(actor_record.get("dataset_dir", "")))
    checkpoint_dir = Path(str(actor_record.get("checkpoint_dir", "")))
    log_path = Path(str(actor_record.get("log_path", "")))
    latest_hf_model_path = actor_record.get("latest_hf_model_path")

    if not dataset_dir.exists():
        issues.append(f"Dataset directory not found: {dataset_dir}")
    if not checkpoint_dir.exists():
        issues.append(f"Checkpoint directory not found: {checkpoint_dir}")
    if not log_path.exists():
        issues.append(f"Actor log not found: {log_path}")

    hf_checkpoint_valid = False
    if latest_hf_model_path:
        hf_path = Path(str(latest_hf_model_path))
        if not hf_path.exists():
            issues.append(f"latest_hf_model_path does not exist: {hf_path}")
        elif not _looks_like_hf_model_dir(hf_path):
            issues.append(f"latest_hf_model_path is not a recognizable HF model dir: {hf_path}")
        else:
            hf_checkpoint_valid = True
    elif require_hf_checkpoint:
        issues.append("latest_hf_model_path is missing.")

    if require_grpo_log_markers and log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        for marker in GRPO_LOG_MARKERS:
            if marker not in log_text:
                issues.append(f"Actor log does not contain required GRPO marker: {marker}")
        if "lora_rank=0" not in log_text:
            warnings.append("Actor log does not show lora_rank=0; verify full-parameter config manually.")

    return {
        "status": "passed" if not issues else "failed",
        "round_index": actor_record.get("round_index"),
        "actor_role": actor_record.get("actor_role"),
        "dataset_dir": str(dataset_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "log_path": str(log_path),
        "latest_hf_model_path": latest_hf_model_path,
        "hf_checkpoint_valid": hf_checkpoint_valid,
        "issues": issues,
        "warnings": warnings,
    }


def _looks_like_hf_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return all((path / marker).exists() for marker in REQUIRED_HF_MARKERS)


def _check(
    report: NativeVerlTrainingVerificationReport,
    condition: bool,
    _passed_label: str,
    issue: str,
) -> None:
    if condition:
        report.checks_passed += 1
    else:
        report.checks_failed += 1
        report.issues.append(issue)


def _finalize(report: NativeVerlTrainingVerificationReport) -> NativeVerlTrainingVerificationReport:
    report.status = "failed" if report.issues else "passed"
    return report
