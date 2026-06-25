"""
Phase 4 Milestone 6: Statistical Analysis

Compute statistical comparisons between methods.
"""

import random
import statistics
from typing import Any


def compute_paired_delta(
    primary_values: list[float],
    baseline_values: list[float],
) -> dict[str, Any]:
    """
    Compute paired delta statistics.

    Args:
        primary_values: Values for primary method (e.g., final_checkpoint)
        baseline_values: Values for baseline method

    Returns:
        Dictionary with delta statistics
    """
    if not primary_values or not baseline_values:
        return {
            "num_pairs": 0,
            "mean_delta": None,
            "note": "No paired values available",
        }

    if len(primary_values) != len(baseline_values):
        return {
            "num_pairs": 0,
            "mean_delta": None,
            "note": "Mismatched number of values",
        }

    deltas = [p - b for p, b in zip(primary_values, baseline_values)]
    mean_delta = float(statistics.fmean(deltas))

    result = {
        "num_pairs": len(deltas),
        "mean_delta": mean_delta,
    }

    # Add bootstrap CI if we have enough samples
    if len(deltas) >= 3:
        try:
            ci = bootstrap_confidence_interval(deltas, n_bootstrap=1000)
            result["confidence_interval_95"] = ci
        except Exception:
            result["confidence_interval_95"] = None

    # Add t-test if scipy is available
    try:
        import scipy.stats

        if len(deltas) >= 3:
            t_stat, p_value = scipy.stats.ttest_1samp(deltas, 0)
            result["t_statistic"] = float(t_stat)
            result["p_value"] = float(p_value)
    except ImportError:
        pass

    # Add warning for small sample size
    if len(deltas) < 3:
        result["note"] = "Too few seeds for reliable significance testing"

    return result


def bootstrap_confidence_interval(
    values: list[float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for mean.

    Args:
        values: List of values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_means = []

    rng = random.Random(42)  # Fixed seed for reproducibility

    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in values]
        bootstrap_means.append(statistics.fmean(sample))

    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    bootstrap_means.sort()
    lower_index = min(
        len(bootstrap_means) - 1,
        max(0, int(round((lower_percentile / 100) * (len(bootstrap_means) - 1)))),
    )
    upper_index = min(
        len(bootstrap_means) - 1,
        max(0, int(round((upper_percentile / 100) * (len(bootstrap_means) - 1)))),
    )

    lower = float(bootstrap_means[lower_index])
    upper = float(bootstrap_means[upper_index])

    return (lower, upper)


def extract_metric_values_by_seed(
    seed_metrics: list[dict[str, Any]],
    method_prefix: str,
    metric_name: str,
) -> list[float]:
    """
    Extract metric values for a specific method across seeds.

    Args:
        seed_metrics: List of seed-level metrics
        method_prefix: Method prefix (e.g., "final_checkpoint", "no_llm_memory")
        metric_name: Metric name (e.g., "test_memory_reward_mean")

    Returns:
        List of values (one per seed)
    """
    # Construct key
    if method_prefix == "final_checkpoint":
        key = metric_name
    else:
        key = f"{method_prefix}_{metric_name}"

    values = []
    for seed_data in seed_metrics:
        if seed_data.get("status") != "succeeded":
            continue

        if key in seed_data and seed_data[key] is not None:
            try:
                values.append(float(seed_data[key]))
            except (ValueError, TypeError):
                pass

    return values


def compute_phase4_statistical_tests(
    seed_metrics: list[dict[str, Any]],
    *,
    primary_method: str = "final_checkpoint",
    baseline_methods: list[str] | None = None,
    test_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute statistical tests comparing primary method to baselines.

    Args:
        seed_metrics: List of seed-level metrics
        primary_method: Primary method to compare (default: final_checkpoint)
        baseline_methods: List of baseline methods (default: no_llm_memory, initial_checkpoint)
        test_metrics: List of metrics to test (default: test_memory_reward_mean, test_question_reward_mean)

    Returns:
        Dictionary of statistical test results
    """
    if baseline_methods is None:
        baseline_methods = ["no_llm_memory", "initial_checkpoint"]

    if test_metrics is None:
        test_metrics = ["test_memory_reward_mean", "test_question_reward_mean"]

    results = {
        "primary_method": primary_method,
        "baseline_methods": baseline_methods,
        "test_metrics": test_metrics,
        "comparisons": [],
    }

    # Filter to succeeded seeds only
    succeeded_seeds = [s for s in seed_metrics if s.get("status") == "succeeded"]

    if not succeeded_seeds:
        results["note"] = "No succeeded seeds available for testing"
        return results

    # Run comparisons
    for baseline_method in baseline_methods:
        for metric_name in test_metrics:
            # Extract values
            primary_values = extract_metric_values_by_seed(
                succeeded_seeds,
                primary_method,
                metric_name,
            )

            baseline_values = extract_metric_values_by_seed(
                succeeded_seeds,
                baseline_method,
                metric_name,
            )

            # Compute comparison
            comparison = {
                "comparison": f"{primary_method}_vs_{baseline_method}",
                "metric": metric_name,
            }

            comparison.update(compute_paired_delta(primary_values, baseline_values))

            results["comparisons"].append(comparison)

    return results


def write_statistical_tests_json(
    tests: dict[str, Any],
    output_path,
) -> None:
    """Write statistical tests to JSON file."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2)
