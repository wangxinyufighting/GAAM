"""
Phase 4 Milestone 6: Suite Metrics Aggregation

Aggregate metrics across multiple seeds and baselines.
"""

import json
import math
import statistics
from pathlib import Path
from typing import Any

from gaam_graph.phase4_suite_schema import (
    Phase4AggregateMetrics,
    Phase4BenchmarkSuiteManifest,
    Phase4SuiteRunRecord,
)


def collect_seed_level_metrics(
    suite_dir: Path,
    run_records: list[Phase4SuiteRunRecord],
) -> list[dict[str, Any]]:
    """
    Collect metrics from each seed run.

    Returns list of seed-level metric dictionaries.
    """
    seed_metrics = []

    for record in run_records:
        seed_entry = {
            "seed": record.seed,
            "status": record.status,
        }

        # Add metrics from run record
        if record.metrics:
            seed_entry.update(record.metrics)

        # Try to load final evaluation manifest
        final_eval_path = Path(record.output_dir) / "final_evaluation" / "final_evaluation_manifest.json"
        if final_eval_path.exists():
            try:
                with open(final_eval_path, "r", encoding="utf-8") as f:
                    final_eval = json.load(f)

                if "metrics" in final_eval:
                    seed_entry.update(final_eval["metrics"])
            except Exception:
                pass

        # Try to load baseline metrics
        baselines_dir = Path(record.output_dir) / "final_evaluation" / "baselines"
        if baselines_dir.exists():
            for baseline_dir in baselines_dir.iterdir():
                if not baseline_dir.is_dir():
                    continue

                baseline_manifest_path = baseline_dir / "final_evaluation_manifest.json"
                if baseline_manifest_path.exists():
                    try:
                        with open(baseline_manifest_path, "r", encoding="utf-8") as f:
                            baseline_manifest = json.load(f)

                        baseline_name = baseline_manifest.get("baseline_name", baseline_dir.name)

                        if "metrics" in baseline_manifest:
                            for metric_key, metric_value in baseline_manifest["metrics"].items():
                                seed_entry[f"{baseline_name}_{metric_key}"] = metric_value
                    except Exception:
                        pass

        seed_metrics.append(seed_entry)

    return seed_metrics


def compute_aggregate_statistics(
    values: list[float],
) -> dict[str, float]:
    """
    Compute aggregate statistics for a list of values.

    Returns:
        Dictionary with count, mean, std, stderr, min, max, median
    """
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "stderr": None,
            "min": None,
            "max": None,
            "median": None,
        }

    sorted_values = sorted(values)
    count = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if count > 1 else 0.0

    return {
        "count": count,
        "mean": float(mean),
        "std": float(std),
        "stderr": float(std / math.sqrt(count)) if count > 1 else 0.0,
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "median": float(statistics.median(sorted_values)),
    }


def aggregate_metrics_by_method(
    seed_metrics: list[dict[str, Any]],
    *,
    include_failed: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Aggregate metrics by method (final_checkpoint, baselines).

    Returns:
        Dictionary mapping method name to aggregated statistics
    """
    # Filter to succeeded seeds unless include_failed is True
    if not include_failed:
        seed_metrics = [s for s in seed_metrics if s.get("status") == "succeeded"]

    if not seed_metrics:
        return {}

    # Identify all metric keys
    all_keys = set()
    for seed in seed_metrics:
        all_keys.update(seed.keys())

    # Remove non-metric keys
    exclude_keys = {"seed", "status"}
    metric_keys = all_keys - exclude_keys

    # Group metrics by method
    method_metrics = {}

    for key in metric_keys:
        # Determine method name
        if key.startswith("no_llm_memory_"):
            method = "no_llm_memory"
            metric_name = key[len("no_llm_memory_"):]
        elif key.startswith("initial_checkpoint_"):
            method = "initial_checkpoint"
            metric_name = key[len("initial_checkpoint_"):]
        elif key.startswith("final_checkpoint_"):
            method = "final_checkpoint"
            metric_name = key[len("final_checkpoint_"):]
        else:
            # Assume it's a final_checkpoint metric
            method = "final_checkpoint"
            metric_name = key

        # Collect values across seeds
        values = []
        for seed in seed_metrics:
            if key in seed and seed[key] is not None:
                try:
                    values.append(float(seed[key]))
                except (ValueError, TypeError):
                    pass

        if not values:
            continue

        # Compute statistics
        stats = compute_aggregate_statistics(values)

        # Store in method_metrics
        if method not in method_metrics:
            method_metrics[method] = {}

        method_metrics[method][metric_name] = stats

    return method_metrics


def write_aggregate_metrics_json(
    metrics: Phase4AggregateMetrics,
    output_path: Path,
) -> None:
    """Write aggregate metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics.model_dump(), f, indent=2)


def write_aggregate_metrics_csv(
    aggregate_by_method: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Write aggregate metrics to CSV file."""
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all metric names
    all_metrics = set()
    for method_data in aggregate_by_method.values():
        all_metrics.update(method_data.keys())

    all_metrics = sorted(all_metrics)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["method", "metric", "count", "mean", "std", "stderr", "min", "max", "median"]
        writer.writerow(header)

        # Rows
        for method in sorted(aggregate_by_method.keys()):
            method_data = aggregate_by_method[method]

            for metric_name in all_metrics:
                if metric_name not in method_data:
                    continue

                stats = method_data[metric_name]

                row = [
                    method,
                    metric_name,
                    stats.get("count", ""),
                    f"{stats.get('mean', ''):.4f}" if stats.get("mean") is not None else "",
                    f"{stats.get('std', ''):.4f}" if stats.get("std") is not None else "",
                    f"{stats.get('stderr', ''):.4f}" if stats.get("stderr") is not None else "",
                    f"{stats.get('min', ''):.4f}" if stats.get("min") is not None else "",
                    f"{stats.get('max', ''):.4f}" if stats.get("max") is not None else "",
                    f"{stats.get('median', ''):.4f}" if stats.get("median") is not None else "",
                ]

                writer.writerow(row)


def write_seed_level_metrics_jsonl(
    seed_metrics: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write seed-level metrics to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for seed_data in seed_metrics:
            f.write(json.dumps(seed_data) + "\n")


def aggregate_phase4_benchmark_suite(
    suite_dir: Path,
    *,
    include_failed: bool = False,
    output_dir: Path | None = None,
) -> Phase4AggregateMetrics:
    """
    Aggregate metrics from benchmark suite.

    Reads suite manifest and per-seed experiment manifests,
    computes aggregate statistics across seeds.

    Args:
        suite_dir: Root directory of benchmark suite
        include_failed: Include failed runs in aggregation
        output_dir: Optional directory for aggregate outputs

    Returns:
        Aggregated metrics
    """
    # Load suite manifest
    manifest_path = suite_dir / "suite_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest = Phase4BenchmarkSuiteManifest.model_validate(manifest_data)

    # Collect seed-level metrics
    seed_metrics = collect_seed_level_metrics(suite_dir, manifest.run_records)

    # Aggregate by method
    aggregate_by_method = aggregate_metrics_by_method(
        seed_metrics,
        include_failed=include_failed,
    )

    # Count statuses
    num_succeeded = sum(1 for r in manifest.run_records if r.status == "succeeded")
    num_partial = sum(1 for r in manifest.run_records if r.status == "partial")
    num_failed = sum(1 for r in manifest.run_records if r.status == "failed")

    # Build aggregate metrics
    metrics = Phase4AggregateMetrics(
        suite_id=manifest.suite_id,
        num_seeds_requested=len(manifest.run_records),
        num_seeds_succeeded=num_succeeded,
        num_seeds_partial=num_partial,
        num_seeds_failed=num_failed,
        final_metrics_by_method=[],
        aggregate_by_method=aggregate_by_method,
    )

    # Write outputs
    aggregate_dir = output_dir or suite_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    write_aggregate_metrics_json(metrics, aggregate_dir / "aggregate_metrics.json")
    write_aggregate_metrics_csv(aggregate_by_method, aggregate_dir / "aggregate_metrics.csv")
    write_seed_level_metrics_jsonl(seed_metrics, aggregate_dir / "seed_level_metrics.jsonl")

    return metrics
