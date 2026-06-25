"""
Phase 4 Milestone 6: Suite Reporting

Generate paper-ready tables and figure data from aggregate metrics.
"""

import json
from pathlib import Path
from typing import Any


def format_metric_value(value: float | None, *, precision: int = 4) -> str:
    """Format metric value for display."""
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def format_metric_with_std(mean: float | None, std: float | None, *, precision: int = 4) -> str:
    """Format metric as mean ± std."""
    if mean is None:
        return "N/A"

    if std is None or std == 0.0:
        return format_metric_value(mean, precision=precision)

    return f"{format_metric_value(mean, precision=precision)} ± {format_metric_value(std, precision=precision)}"


def write_paper_table_markdown(
    aggregate_by_method: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    metrics: list[str] | None = None,
) -> None:
    """
    Write paper table in Markdown format.

    Args:
        aggregate_by_method: Aggregated metrics by method
        output_path: Output path for markdown file
        metrics: List of metrics to include (default: memory_reward, question_reward, no_leakage)
    """
    if metrics is None:
        metrics = [
            "test_memory_reward_mean",
            "test_question_reward_mean",
            "test_no_leakage_pass_rate",
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort methods: baselines first, then final
    method_order = ["no_llm_memory", "initial_checkpoint", "final_checkpoint"]
    methods = [m for m in method_order if m in aggregate_by_method]

    # Method display names
    method_names = {
        "no_llm_memory": "No-LLM Memory",
        "initial_checkpoint": "Initial Model",
        "final_checkpoint": "GAAM Final",
    }

    lines = []
    lines.append("# Phase 4 Benchmark Suite: Test Results")
    lines.append("")
    lines.append("| Method | Memory Reward | Question Reward | No-Leakage | Seeds | Notes |")
    lines.append("|--------|---------------|-----------------|------------|-------|-------|")

    for method in methods:
        method_data = aggregate_by_method[method]

        # Get metric statistics
        memory_reward_stats = method_data.get("test_memory_reward_mean", {})
        question_reward_stats = method_data.get("test_question_reward_mean", {})
        no_leakage_stats = method_data.get("test_no_leakage_pass_rate", {})

        # Format values
        memory_reward_str = format_metric_with_std(
            memory_reward_stats.get("mean"),
            memory_reward_stats.get("std"),
        )

        question_reward_str = format_metric_with_std(
            question_reward_stats.get("mean"),
            question_reward_stats.get("std"),
        )

        no_leakage_str = format_metric_value(no_leakage_stats.get("mean"))

        # Seed count
        seed_count = memory_reward_stats.get("count", 0)

        # Notes
        notes = ""
        if method == "no_llm_memory":
            notes = "baseline"
        elif method == "initial_checkpoint":
            notes = "before training"
        elif method == "final_checkpoint":
            notes = "trained"

        row = f"| {method_names.get(method, method)} | {memory_reward_str} | {question_reward_str} | {no_leakage_str} | {seed_count} | {notes} |"
        lines.append(row)

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_paper_table_latex(
    aggregate_by_method: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Write paper table in LaTeX format.

    Uses booktabs package for professional tables.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort methods
    method_order = ["no_llm_memory", "initial_checkpoint", "final_checkpoint"]
    methods = [m for m in method_order if m in aggregate_by_method]

    method_names = {
        "no_llm_memory": "No-LLM Memory",
        "initial_checkpoint": "Initial Model",
        "final_checkpoint": "GAAM Final",
    }

    lines = []
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Method & Memory Reward & Question Reward & No-Leakage & Seeds \\\\")
    lines.append("\\midrule")

    for method in methods:
        method_data = aggregate_by_method[method]

        memory_stats = method_data.get("test_memory_reward_mean", {})
        question_stats = method_data.get("test_question_reward_mean", {})
        no_leakage_stats = method_data.get("test_no_leakage_pass_rate", {})

        memory_str = format_metric_with_std(
            memory_stats.get("mean"),
            memory_stats.get("std"),
        )

        question_str = format_metric_with_std(
            question_stats.get("mean"),
            question_stats.get("std"),
        )

        no_leakage_str = format_metric_value(no_leakage_stats.get("mean"))
        seed_count = memory_stats.get("count", 0)

        row = f"{method_names.get(method, method)} & {memory_str} & {question_str} & {no_leakage_str} & {seed_count} \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_paper_table_json(
    aggregate_by_method: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Write paper table data as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    method_order = ["no_llm_memory", "initial_checkpoint", "final_checkpoint"]
    methods = [m for m in method_order if m in aggregate_by_method]

    rows = []
    for method in methods:
        method_data = aggregate_by_method[method]

        row = {
            "method": method,
            "test_memory_reward_mean": method_data.get("test_memory_reward_mean"),
            "test_question_reward_mean": method_data.get("test_question_reward_mean"),
            "test_no_leakage_pass_rate": method_data.get("test_no_leakage_pass_rate"),
        }

        rows.append(row)

    data = {"rows": rows}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_learning_curves_json(
    suite_dir: Path,
    output_path: Path,
) -> None:
    """
    Write learning curves JSON for plotting.

    Collects per-round metrics from all succeeded seeds.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load suite manifest
    manifest_path = suite_dir / "suite_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    run_records = manifest_data.get("run_records", [])

    # Collect learning curves per seed
    seed_curves = []

    for record in run_records:
        if record.get("status") != "succeeded":
            continue

        seed = record["seed"]
        output_dir = Path(record["output_dir"])

        # Load cotraining manifest
        cotraining_manifest_path = output_dir / "cotraining" / "cotraining_manifest.json"
        if not cotraining_manifest_path.exists():
            continue

        try:
            with open(cotraining_manifest_path, "r", encoding="utf-8") as f:
                cotraining_manifest = json.load(f)

            round_reports = cotraining_manifest.get("round_reports", [])

            # Extract per-round metrics
            rounds = []
            train_memory_reward = []
            train_question_reward = []
            dev_memory_reward = []
            dev_question_reward = []

            for report in round_reports:
                rounds.append(report.get("round_id"))
                train_memory_reward.append(report.get("train_memory_reward_mean"))
                train_question_reward.append(report.get("train_question_reward_mean"))
                dev_memory_reward.append(report.get("dev_memory_reward_mean"))
                dev_question_reward.append(report.get("dev_question_reward_mean"))

            seed_curves.append({
                "seed": seed,
                "rounds": rounds,
                "train_memory_reward": train_memory_reward,
                "train_question_reward": train_question_reward,
                "dev_memory_reward": dev_memory_reward,
                "dev_question_reward": dev_question_reward,
            })

        except Exception:
            continue

    data = {
        "seed_curves": seed_curves,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_figure_data_json(
    aggregate_by_method: dict[str, dict[str, Any]],
    suite_dir: Path,
    output_path: Path,
    *,
    reports_dir: Path | None = None,
) -> None:
    """
    Write figure data JSON for all visualizations.

    Includes:
    - Learning curves (per-round training progress)
    - Final bar chart (test results comparison)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load learning curves
    learning_curves_path = (reports_dir or suite_dir / "aggregate") / "learning_curves.json"
    learning_curves = {}
    if learning_curves_path.exists():
        with open(learning_curves_path, "r", encoding="utf-8") as f:
            learning_curves = json.load(f)

    # Build final bar chart data
    method_order = ["no_llm_memory", "initial_checkpoint", "final_checkpoint"]
    methods = [m for m in method_order if m in aggregate_by_method]

    final_bar_chart = []
    for method in methods:
        method_data = aggregate_by_method[method]

        memory_stats = method_data.get("test_memory_reward_mean", {})
        question_stats = method_data.get("test_question_reward_mean", {})

        final_bar_chart.append({
            "method": method,
            "test_memory_reward_mean": memory_stats.get("mean"),
            "test_memory_reward_stderr": memory_stats.get("stderr"),
            "test_question_reward_mean": question_stats.get("mean"),
            "test_question_reward_stderr": question_stats.get("stderr"),
        })

    data = {
        "learning_curves": learning_curves,
        "final_bar_chart": final_bar_chart,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def generate_suite_reports(
    suite_dir: Path,
    aggregate_by_method: dict[str, dict[str, Any]],
    *,
    formats: list[str] | None = None,
    output_dir: Path | None = None,
) -> None:
    """
    Generate all suite reports.

    Args:
        suite_dir: Suite directory
        aggregate_by_method: Aggregated metrics
        formats: List of formats to generate (default: all)
        output_dir: Optional directory for generated report files
    """
    if formats is None:
        formats = ["markdown", "latex", "json"]

    reports_dir = output_dir or suite_dir / "aggregate"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if "markdown" in formats:
        write_paper_table_markdown(
            aggregate_by_method,
            reports_dir / "paper_table.md",
        )

    if "latex" in formats:
        write_paper_table_latex(
            aggregate_by_method,
            reports_dir / "paper_table.tex",
        )

    if "json" in formats:
        write_paper_table_json(
            aggregate_by_method,
            reports_dir / "paper_table.json",
        )

    # Always write learning curves and figure data
    write_learning_curves_json(
        suite_dir,
        reports_dir / "learning_curves.json",
    )

    write_figure_data_json(
        aggregate_by_method,
        suite_dir,
        reports_dir / "figure_data.json",
        reports_dir=reports_dir,
    )
