"""
Phase 4 Milestone 5: Leakage Audit

Scan production experiment artifacts for forbidden fields.

Key functions:
- run_phase4_leakage_audit: scan experiment directory
- scan_file_for_leakage: scan one file
- check_stage_leakage: check stage-specific rules

Design principle:
Fail closed when strict_no_leakage=True. Forbidden fields by stage.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gaam_graph.dataset_split_schema import DatasetSplitName


class LeakageIssue(BaseModel):
    """One leakage issue found during audit."""

    severity: str  # error, warning
    file_path: str
    stage: str  # memory_builder_input, question_agent_input, trainer_batch, answerer_input
    forbidden_field: str
    message: str


class Phase4LeakageAuditReport(BaseModel):
    """Leakage audit report."""

    status: str  # passed, failed
    scanned_files: int
    issues: list[LeakageIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# Stage-specific forbidden fields
FORBIDDEN_FIELDS = {
    "memory_builder_input": {
        "oracle_graph",
        "benchmark_target_question",
        "benchmark_question",
        "benchmark_answer",
        "gold_answer",
        "test_label",
        "generated_training_question",  # Memory Builder should not see questions
        "reward_report",  # No reward feedback during construction
    },
    "question_agent_training_input": {
        "benchmark_target_question",
        "benchmark_question",
        "benchmark_answer",
        "test_label",
    },
    "answerer_input": {
        "raw_history",  # Answerer should only see CurrentMemory
        "oracle_graph",
        "benchmark_answer",
        "gold_answer",
    },
    "trainer_batch": {
        "benchmark_target_question",
        "benchmark_question",
        "benchmark_answer",
        "gold_answer",
        "test_label",
    },
}


def run_phase4_leakage_audit(
    experiment_dir: Path,
    *,
    strict: bool = True,
) -> Phase4LeakageAuditReport:
    """
    Run leakage audit on production experiment artifacts.

    Args:
        experiment_dir: experiment output directory
        strict: if True, fail on any leakage issue

    Returns:
        Audit report
    """
    issues: list[LeakageIssue] = []
    warnings: list[str] = []
    scanned_files = 0

    # Scan cotraining artifacts
    cotraining_dir = experiment_dir / "cotraining"
    if cotraining_dir.exists():
        cotraining_issues, cotraining_count = scan_cotraining_artifacts(cotraining_dir)
        issues.extend(cotraining_issues)
        scanned_files += cotraining_count

    # Scan final evaluation artifacts
    final_eval_dir = experiment_dir / "final_evaluation"
    if final_eval_dir.exists():
        eval_issues, eval_count = scan_final_evaluation_artifacts(final_eval_dir)
        issues.extend(eval_issues)
        scanned_files += eval_count

    # Determine status
    has_errors = any(issue.severity == "error" for issue in issues)

    if has_errors and strict:
        status = "failed"
    elif has_errors:
        status = "failed"
        warnings.append("Leakage issues found but strict mode disabled")
    else:
        status = "passed"

    return Phase4LeakageAuditReport(
        status=status,
        scanned_files=scanned_files,
        issues=issues,
        warnings=warnings,
    )


def scan_cotraining_artifacts(cotraining_dir: Path) -> tuple[list[LeakageIssue], int]:
    """Scan cotraining artifacts for leakage."""
    issues: list[LeakageIssue] = []
    scanned_files = 0

    # Scan rounds
    rounds_dir = cotraining_dir / "rounds"
    if not rounds_dir.exists():
        return issues, scanned_files

    for round_dir in sorted(rounds_dir.iterdir()):
        if not round_dir.is_dir():
            continue

        # Scan train rollout
        train_rollout_dir = round_dir / "train_rollout"
        if train_rollout_dir.exists():
            round_issues, round_count = scan_rollout_artifacts(
                train_rollout_dir, split=DatasetSplitName.TRAIN
            )
            issues.extend(round_issues)
            scanned_files += round_count

        # Scan trainer step
        trainer_step_dir = round_dir / "trainer_step"
        if trainer_step_dir.exists():
            trainer_issues, trainer_count = scan_trainer_artifacts(trainer_step_dir)
            issues.extend(trainer_issues)
            scanned_files += trainer_count

        # Scan dev rollout
        dev_rollout_dir = round_dir / "dev_rollout"
        if dev_rollout_dir.exists():
            dev_issues, dev_count = scan_rollout_artifacts(
                dev_rollout_dir, split=DatasetSplitName.DEV
            )
            issues.extend(dev_issues)
            scanned_files += dev_count

    return issues, scanned_files


def scan_final_evaluation_artifacts(
    final_eval_dir: Path,
) -> tuple[list[LeakageIssue], int]:
    """Scan final evaluation artifacts for leakage."""
    issues: list[LeakageIssue] = []
    scanned_files = 0

    # Scan test rollout
    test_rollout_dir = final_eval_dir / "test_rollout"
    if test_rollout_dir.exists():
        test_issues, test_count = scan_rollout_artifacts(
            test_rollout_dir, split=DatasetSplitName.TEST
        )
        issues.extend(test_issues)
        scanned_files += test_count

    # Scan baselines
    baselines_dir = final_eval_dir / "baselines"
    if baselines_dir.exists():
        for baseline_dir in baselines_dir.iterdir():
            if not baseline_dir.is_dir():
                continue

            baseline_rollout = baseline_dir / "test_rollout"
            if baseline_rollout.exists():
                baseline_issues, baseline_count = scan_rollout_artifacts(
                    baseline_rollout, split=DatasetSplitName.TEST
                )
                issues.extend(baseline_issues)
                scanned_files += baseline_count

    return issues, scanned_files


def scan_rollout_artifacts(
    rollout_dir: Path, split: DatasetSplitName
) -> tuple[list[LeakageIssue], int]:
    """Scan rollout artifacts for leakage."""
    issues: list[LeakageIssue] = []
    scanned_files = 0

    # Scan per-record rollouts
    records_dir = rollout_dir / "records"
    if not records_dir.exists():
        return issues, scanned_files

    for record_dir in records_dir.iterdir():
        if not record_dir.is_dir():
            continue

        # Scan memory builder trace
        mb_trace = record_dir / "memory_builder_trace.json"
        if mb_trace.exists():
            mb_issues = scan_file_for_leakage(
                mb_trace, stage="memory_builder_input", split=split
            )
            issues.extend(mb_issues)
            scanned_files += 1

        # Scan question agent trace
        qa_trace = record_dir / "question_agent_trace.json"
        if qa_trace.exists():
            qa_issues = scan_file_for_leakage(
                qa_trace, stage="question_agent_training_input", split=split
            )
            issues.extend(qa_issues)
            scanned_files += 1

        # Scan answerer trace
        answerer_trace = record_dir / "answerer_trace.json"
        if answerer_trace.exists():
            answerer_issues = scan_file_for_leakage(
                answerer_trace, stage="answerer_input", split=split
            )
            issues.extend(answerer_issues)
            scanned_files += 1

    return issues, scanned_files


def scan_trainer_artifacts(trainer_dir: Path) -> tuple[list[LeakageIssue], int]:
    """Scan trainer artifacts for leakage."""
    issues: list[LeakageIssue] = []
    scanned_files = 0

    # Scan actor batches
    for actor_name in ["memory_builder", "question_agent"]:
        actor_dir = trainer_dir / actor_name
        if not actor_dir.exists():
            continue

        trainer_ready_batch = actor_dir / "trainer_ready_batch.json"
        if trainer_ready_batch.exists():
            batch_issues = scan_file_for_leakage(
                trainer_ready_batch,
                stage="trainer_batch",
                split=DatasetSplitName.TRAIN,
            )
            issues.extend(batch_issues)
            scanned_files += 1

    return issues, scanned_files


def scan_file_for_leakage(
    file_path: Path, stage: str, split: DatasetSplitName
) -> list[LeakageIssue]:
    """
    Scan one file for forbidden fields.

    Args:
        file_path: file to scan
        stage: stage name for field rules
        split: train/dev/test split

    Returns:
        List of leakage issues
    """
    issues: list[LeakageIssue] = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        forbidden = FORBIDDEN_FIELDS.get(stage, set())

        # Recursively check for forbidden fields
        found_fields = find_forbidden_fields(data, forbidden)

        for field_name in found_fields:
            issues.append(
                LeakageIssue(
                    severity="error",
                    file_path=str(file_path),
                    stage=stage,
                    forbidden_field=field_name,
                    message=f"Forbidden field '{field_name}' found in {stage} ({split.value} split)",
                )
            )

    except Exception as e:
        issues.append(
            LeakageIssue(
                severity="error",
                file_path=str(file_path),
                stage=stage,
                forbidden_field="__unreadable_json__",
                message=(
                    f"Could not read or parse audit file for {stage} "
                    f"({split.value} split): {type(e).__name__}: {e}"
                ),
            )
        )

    return issues


def find_forbidden_fields(data: Any, forbidden: set[str]) -> set[str]:
    """Recursively find forbidden field names in data."""
    found = set()

    if isinstance(data, dict):
        for key, value in data.items():
            if key in forbidden:
                found.add(key)
            found.update(find_forbidden_fields(value, forbidden))

    elif isinstance(data, list):
        for item in data:
            found.update(find_forbidden_fields(item, forbidden))

    return found


def write_leakage_audit_report(report: Phase4LeakageAuditReport, output_path: Path) -> None:
    """Write leakage audit report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)


def load_leakage_audit_report(report_path: Path) -> Phase4LeakageAuditReport:
    """Load leakage audit report from JSON."""
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Phase4LeakageAuditReport.model_validate(data)
