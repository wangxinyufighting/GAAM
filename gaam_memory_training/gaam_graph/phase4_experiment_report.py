"""
Phase 4 Milestone 5: Experiment Reporting

Export experiment metrics and reports.

Key functions:
- export_experiment_metrics: export metrics to CSV/JSONL
- write_paper_table: write paper-ready table
- collect_experiment_metrics: collect all metrics

Design principle:
Reports regenerate from existing experiment artifacts. Handle missing data gracefully.
"""

import csv
import json
from pathlib import Path
from typing import Any

from gaam_graph.phase4_cotraining_schema import Phase4CotrainingManifest
from gaam_graph.phase4_experiment_schema import (
    Phase4FinalEvaluationManifest,
    Phase4ProductionExperimentManifest,
)


def collect_experiment_metrics(
    experiment_dir: Path,
) -> dict[str, Any]:
    """
    Collect all metrics from production experiment.

    Args:
        experiment_dir: experiment output directory

    Returns:
        Dictionary with:
        - cotraining_metrics: per-round training metrics
        - final_evaluation_metrics: test metrics
        - baseline_metrics: baseline comparison metrics
    """
    metrics: dict[str, Any] = {
        "cotraining_metrics": [],
        "final_evaluation_metrics": None,
        "baseline_metrics": [],
    }

    # Load cotraining manifest
    cotraining_manifest_path = experiment_dir / "cotraining" / "cotraining_manifest.json"
    if cotraining_manifest_path.exists():
        with open(cotraining_manifest_path, "r", encoding="utf-8") as f:
            cotraining_data = json.load(f)

        cotraining_manifest = Phase4CotrainingManifest.model_validate(cotraining_data)

        # Extract per-round metrics
        for round_report in cotraining_manifest.round_reports:
            round_metrics = {
                "round_id": round_report.round_id,
                "status": round_report.status.value,
                "train_memory_reward_mean": round_report.train_memory_reward_mean,
                "train_question_reward_mean": round_report.train_question_reward_mean,
                "dev_memory_reward_mean": round_report.dev_memory_reward_mean,
                "dev_question_reward_mean": round_report.dev_question_reward_mean,
                "train_no_leakage_pass_rate": round_report.train_no_leakage_pass_rate,
                "dev_no_leakage_pass_rate": round_report.dev_no_leakage_pass_rate,
                "memory_builder_checkpoint_id": round_report.selected_memory_builder_checkpoint_id,
                "question_agent_checkpoint_id": round_report.selected_question_agent_checkpoint_id,
            }
            metrics["cotraining_metrics"].append(round_metrics)

    # Load final evaluation manifest
    final_eval_manifest_path = (
        experiment_dir / "final_evaluation" / "final_evaluation_manifest.json"
    )
    if final_eval_manifest_path.exists():
        with open(final_eval_manifest_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        final_eval = Phase4FinalEvaluationManifest.model_validate(eval_data)

        metrics["final_evaluation_metrics"] = {
            "status": final_eval.status,
            "num_test_records": final_eval.metrics.get("num_test_records"),
            "test_memory_reward_mean": final_eval.metrics.get("test_memory_reward_mean"),
            "test_question_reward_mean": final_eval.metrics.get("test_question_reward_mean"),
            "test_no_leakage_pass_rate": final_eval.metrics.get("test_no_leakage_pass_rate"),
            "num_succeeded": final_eval.metrics.get("num_succeeded"),
            "num_failed": final_eval.metrics.get("num_failed"),
        }

    # Load baseline evaluations
    baselines_dir = experiment_dir / "final_evaluation" / "baselines"
    if baselines_dir.exists():
        for baseline_dir in sorted(baselines_dir.iterdir()):
            if not baseline_dir.is_dir():
                continue

            baseline_manifest_path = baseline_dir / "final_evaluation_manifest.json"
            if not baseline_manifest_path.exists():
                continue

            with open(baseline_manifest_path, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)

            baseline_eval = Phase4FinalEvaluationManifest.model_validate(baseline_data)

            baseline_metrics = {
                "baseline_name": baseline_eval.baseline_name,
                "status": baseline_eval.status,
                "test_memory_reward_mean": baseline_eval.metrics.get("test_memory_reward_mean"),
                "test_question_reward_mean": baseline_eval.metrics.get(
                    "test_question_reward_mean"
                ),
                "test_no_leakage_pass_rate": baseline_eval.metrics.get(
                    "test_no_leakage_pass_rate"
                ),
            }
            metrics["baseline_metrics"].append(baseline_metrics)

    return metrics


def export_cotraining_metrics_csv(
    cotraining_metrics: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Export cotraining metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "round_id",
        "status",
        "train_memory_reward_mean",
        "train_question_reward_mean",
        "dev_memory_reward_mean",
        "dev_question_reward_mean",
        "train_no_leakage_pass_rate",
        "dev_no_leakage_pass_rate",
        "memory_builder_checkpoint_id",
        "question_agent_checkpoint_id",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cotraining_metrics)


def export_cotraining_metrics_jsonl(
    cotraining_metrics: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Export cotraining metrics to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for metric in cotraining_metrics:
            f.write(json.dumps(metric) + "\n")


def export_final_metrics_json(
    final_metrics: dict[str, Any] | None,
    baseline_metrics: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Export final and baseline metrics to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "final_evaluation": final_metrics,
        "baselines": baseline_metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_paper_table_markdown(
    final_metrics: dict[str, Any] | None,
    baseline_metrics: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Write paper-ready table in Markdown.

    Format:
    | Method | Test Memory Reward | Test Question Reward | No-Leakage | Notes |
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Test Results",
        "",
        "| Method | Test Memory Reward | Test Question Reward | No-Leakage | Notes |",
        "|--------|-------------------|---------------------|------------|-------|",
    ]

    # Add baselines
    for baseline in baseline_metrics:
        name = baseline.get("baseline_name", "unknown")
        memory_reward = baseline.get("test_memory_reward_mean")
        question_reward = baseline.get("test_question_reward_mean")
        no_leakage = baseline.get("test_no_leakage_pass_rate")

        memory_str = f"{memory_reward:.4f}" if memory_reward is not None else "N/A"
        question_str = f"{question_reward:.4f}" if question_reward is not None else "N/A"
        leakage_str = f"{no_leakage:.2%}" if no_leakage is not None else "N/A"

        if name == "no_llm_memory":
            notes = "baseline"
        elif name == "initial_checkpoint":
            notes = "before training"
        elif name == "final_checkpoint":
            notes = "trained"
        else:
            notes = ""

        lines.append(
            f"| {name} | {memory_str} | {question_str} | {leakage_str} | {notes} |"
        )

    # Add final evaluation (if not already in baselines)
    if final_metrics and not any(
        b.get("baseline_name") == "final_checkpoint" for b in baseline_metrics
    ):
        memory_reward = final_metrics.get("test_memory_reward_mean")
        question_reward = final_metrics.get("test_question_reward_mean")
        no_leakage = final_metrics.get("test_no_leakage_pass_rate")

        memory_str = f"{memory_reward:.4f}" if memory_reward is not None else "N/A"
        question_str = f"{question_reward:.4f}" if question_reward is not None else "N/A"
        leakage_str = f"{no_leakage:.2%}" if no_leakage is not None else "N/A"

        lines.append(
            f"| GAAM Final | {memory_str} | {question_str} | {leakage_str} | trained |"
        )

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_paper_table_json(
    final_metrics: dict[str, Any] | None,
    baseline_metrics: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write paper table data as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # Add baselines
    for baseline in baseline_metrics:
        rows.append(
            {
                "method": baseline.get("baseline_name", "unknown"),
                "test_memory_reward_mean": baseline.get("test_memory_reward_mean"),
                "test_question_reward_mean": baseline.get("test_question_reward_mean"),
                "test_no_leakage_pass_rate": baseline.get("test_no_leakage_pass_rate"),
            }
        )

    # Add final evaluation
    if final_metrics:
        rows.append(
            {
                "method": "final_checkpoint",
                "test_memory_reward_mean": final_metrics.get("test_memory_reward_mean"),
                "test_question_reward_mean": final_metrics.get("test_question_reward_mean"),
                "test_no_leakage_pass_rate": final_metrics.get("test_no_leakage_pass_rate"),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)


def write_learning_curves_json(
    cotraining_metrics: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write learning curves data as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rounds = []
    train_memory_rewards = []
    train_question_rewards = []
    dev_memory_rewards = []
    dev_question_rewards = []

    for metric in cotraining_metrics:
        rounds.append(metric["round_id"])
        train_memory_rewards.append(metric.get("train_memory_reward_mean"))
        train_question_rewards.append(metric.get("train_question_reward_mean"))
        dev_memory_rewards.append(metric.get("dev_memory_reward_mean"))
        dev_question_rewards.append(metric.get("dev_question_reward_mean"))

    data = {
        "rounds": rounds,
        "train_memory_reward": train_memory_rewards,
        "train_question_reward": train_question_rewards,
        "dev_memory_reward": dev_memory_rewards,
        "dev_question_reward": dev_question_rewards,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_experiment_reports(
    experiment_dir: Path,
    *,
    formats: list[str] = ["csv", "jsonl", "json", "markdown"],
) -> None:
    """
    Export all experiment reports.

    Args:
        experiment_dir: experiment output directory
        formats: report formats to generate
    """
    # Collect metrics
    metrics = collect_experiment_metrics(experiment_dir)

    reports_dir = experiment_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Export cotraining metrics
    if "csv" in formats:
        export_cotraining_metrics_csv(
            metrics["cotraining_metrics"], reports_dir / "metrics.csv"
        )

    if "jsonl" in formats:
        export_cotraining_metrics_jsonl(
            metrics["cotraining_metrics"], reports_dir / "metrics.jsonl"
        )

    # Export final metrics
    if "json" in formats:
        export_final_metrics_json(
            metrics["final_evaluation_metrics"],
            metrics["baseline_metrics"],
            reports_dir / "final_metrics.json",
        )

    # Export paper table
    if "markdown" in formats:
        write_paper_table_markdown(
            metrics["final_evaluation_metrics"],
            metrics["baseline_metrics"],
            reports_dir / "paper_table.md",
        )

    if "json" in formats:
        write_paper_table_json(
            metrics["final_evaluation_metrics"],
            metrics["baseline_metrics"],
            reports_dir / "paper_table.json",
        )

    # Export learning curves
    if "json" in formats:
        write_learning_curves_json(
            metrics["cotraining_metrics"], reports_dir / "learning_curves.json"
        )

    print(f"[Reports] Exported to {reports_dir}")
