"""
Phase 4 Milestone 4: Metrics Export

Export co-training metrics to CSV or JSONL for analysis and plotting.

Key functions:
- export_cotraining_metrics_csv: export to CSV format
- export_cotraining_metrics_jsonl: export to JSONL format
- collect_round_metrics: extract metrics from round reports

Design principle:
Handle missing metrics gracefully - use None or empty string for missing values.
"""

import csv
import json
from pathlib import Path
from typing import Any

from gaam_graph.phase4_cotraining_schema import (
    Phase4CotrainingManifest,
    Phase4RoundReport,
)


def collect_round_metrics(report: Phase4RoundReport) -> dict[str, Any]:
    """
    Collect metrics from one round report.

    Returns:
        Dictionary with all relevant metrics
    """
    return {
        "round_id": report.round_id,
        "status": report.status.value,
        "train_memory_reward_mean": report.train_memory_reward_mean,
        "train_question_reward_mean": report.train_question_reward_mean,
        "dev_memory_reward_mean": report.dev_memory_reward_mean,
        "dev_question_reward_mean": report.dev_question_reward_mean,
        "train_no_leakage_pass_rate": report.train_no_leakage_pass_rate,
        "dev_no_leakage_pass_rate": report.dev_no_leakage_pass_rate,
        "input_memory_builder_checkpoint_id": report.input_memory_builder_checkpoint_id,
        "input_question_agent_checkpoint_id": report.input_question_agent_checkpoint_id,
        "selected_memory_builder_checkpoint_id": report.selected_memory_builder_checkpoint_id,
        "selected_question_agent_checkpoint_id": report.selected_question_agent_checkpoint_id,
        "num_warnings": len(report.warnings),
        "num_errors": len(report.errors),
    }


def export_cotraining_metrics_csv(
    manifest: Phase4CotrainingManifest,
    output_path: Path,
) -> None:
    """
    Export co-training metrics to CSV.

    Columns:
    - round_id
    - status
    - train_memory_reward_mean
    - train_question_reward_mean
    - dev_memory_reward_mean
    - dev_question_reward_mean
    - train_no_leakage_pass_rate
    - dev_no_leakage_pass_rate
    - input_memory_builder_checkpoint_id
    - input_question_agent_checkpoint_id
    - selected_memory_builder_checkpoint_id
    - selected_question_agent_checkpoint_id
    - num_warnings
    - num_errors

    Args:
        manifest: co-training manifest
        output_path: output CSV file path
    """
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
        "input_memory_builder_checkpoint_id",
        "input_question_agent_checkpoint_id",
        "selected_memory_builder_checkpoint_id",
        "selected_question_agent_checkpoint_id",
        "num_warnings",
        "num_errors",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for report in manifest.round_reports:
            metrics = collect_round_metrics(report)
            # Replace None with empty string for CSV
            metrics = {k: (v if v is not None else "") for k, v in metrics.items()}
            writer.writerow(metrics)


def export_cotraining_metrics_jsonl(
    manifest: Phase4CotrainingManifest,
    output_path: Path,
) -> None:
    """
    Export co-training metrics to JSONL.

    Each line is a JSON object with all round metrics.

    Args:
        manifest: co-training manifest
        output_path: output JSONL file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for report in manifest.round_reports:
            metrics = collect_round_metrics(report)
            f.write(json.dumps(metrics) + "\n")


def export_cotraining_metrics(
    manifest: Phase4CotrainingManifest,
    output_path: Path,
    format: str = "csv",
) -> None:
    """
    Export co-training metrics in specified format.

    Args:
        manifest: co-training manifest
        output_path: output file path
        format: "csv" or "jsonl"
    """
    if format == "csv":
        export_cotraining_metrics_csv(manifest, output_path)
    elif format == "jsonl":
        export_cotraining_metrics_jsonl(manifest, output_path)
    else:
        raise ValueError(f"Unknown format: {format}. Supported: csv, jsonl")
