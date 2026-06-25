"""
Phase 4 Milestone 5: Final Held-Out Evaluation

Run held-out test evaluation after training completes.

Key functions:
- run_phase4_final_evaluation: execute test rollout with selected checkpoints
- write_final_evaluation_manifest: write evaluation manifest

Design principle:
Test split is read-only. Never update models from test split.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from gaam_graph.dataset_split_schema import DatasetSplitName
from gaam_graph.dataset_splitter import load_dataset_split_manifest
from gaam_graph.phase4_experiment_schema import (
    Phase4FinalEvaluationConfig,
    Phase4FinalEvaluationManifest,
    generate_eval_id,
)
from gaam_graph.phase4_multi_case_rollout import run_multi_case_rollout


def run_phase4_final_evaluation(
    config: Phase4FinalEvaluationConfig,
) -> Phase4FinalEvaluationManifest:
    """
    Run held-out test evaluation with selected checkpoints.

    Rules:
    - Test split only
    - No model updates (trainer_mode=dry_run)
    - Selected checkpoints from training
    - Eval-only rollout

    Args:
        config: final evaluation configuration

    Returns:
        Final evaluation manifest
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split manifest
    split_manifest = load_dataset_split_manifest(Path(config.split_manifest_path))

    # Generate evaluation ID
    eval_id = generate_eval_id(
        baseline=config.baseline_name,
        timestamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )

    # Create manifest
    manifest = Phase4FinalEvaluationManifest(
        eval_id=eval_id,
        split=DatasetSplitName.TEST,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        selected_record_ids=[],
        test_rollout_dir="",
        memory_builder_checkpoint_id=config.memory_builder_checkpoint_id,
        question_agent_checkpoint_id=config.question_agent_checkpoint_id,
        checkpoint_registry_path=config.checkpoint_registry_path,
        baseline_name=config.baseline_name,
    )

    try:
        # Run test rollout
        test_rollout_dir = output_dir / "test_rollout"

        # Get test record IDs
        test_record_ids = [
            r.record_id for r in split_manifest.records if r.split == DatasetSplitName.TEST
        ]

        # Limit records if specified
        if config.test_records_limit is not None:
            test_record_ids = test_record_ids[: config.test_records_limit]

        manifest.selected_record_ids = test_record_ids

        print(f"\n[Final Evaluation] Running test rollout on {len(test_record_ids)} records")

        test_manifest = run_multi_case_rollout(
            split_manifest_path=Path(config.split_manifest_path),
            split=DatasetSplitName.TEST,
            output_dir=test_rollout_dir,
            backend="local_fallback",
            trainer_mode="dry_run",
            record_ids=test_record_ids,
            round_id=0,
            step_id=0,
            checkpoint_registry_path=config.checkpoint_registry_path,
            memory_builder_checkpoint_id=config.memory_builder_checkpoint_id,
            question_agent_checkpoint_id=config.question_agent_checkpoint_id,
            memory_builder_model_path=config.memory_builder_model_path,
            question_agent_model_path=config.question_agent_model_path,
            answerer_model_path=config.answerer_model_path,
            update_memory_builder=False,
            update_question_agent=False,
            continue_on_record_failure=True,
            overwrite=config.overwrite,
        )

        manifest.test_rollout_dir = str(test_rollout_dir)
        manifest.test_rollout_manifest_path = str(
            test_rollout_dir / "multi_case_rollout_manifest.json"
        )

        # Extract metrics
        manifest.metrics["test_memory_reward_mean"] = test_manifest.metrics.get(
            "memory_reward_mean"
        )
        manifest.metrics["test_question_reward_mean"] = test_manifest.metrics.get(
            "question_reward_mean"
        )
        manifest.metrics["test_no_leakage_pass_rate"] = test_manifest.metrics.get(
            "no_leakage_pass_rate"
        )
        manifest.metrics["num_test_records"] = len(test_record_ids)
        manifest.metrics["num_succeeded"] = test_manifest.metrics.get("num_succeeded", 0)
        manifest.metrics["num_failed"] = test_manifest.metrics.get("num_failed", 0)

        # Status
        if test_manifest.status == "succeeded":
            manifest.status = "succeeded"
        elif test_manifest.status == "partial":
            manifest.status = "partial"
        else:
            manifest.status = "failed"

    except Exception as e:
        manifest.status = "failed"
        manifest.errors.append(str(e))

    finally:
        manifest.finished_at = datetime.now(timezone.utc).isoformat()

        # Write manifest
        write_final_evaluation_manifest(manifest, output_dir)

    return manifest


def write_final_evaluation_manifest(
    manifest: Phase4FinalEvaluationManifest,
    output_dir: Path,
) -> None:
    """Write final evaluation manifest to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "final_evaluation_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)


def load_final_evaluation_manifest(manifest_path: Path) -> Phase4FinalEvaluationManifest:
    """Load final evaluation manifest from JSON."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Phase4FinalEvaluationManifest.model_validate(data)


def write_final_evaluation_summary(
    manifest: Phase4FinalEvaluationManifest,
    output_dir: Path,
) -> None:
    """Write human-readable summary markdown."""
    summary_path = output_dir / "summary.md"

    lines = [
        f"# Final Test Evaluation: {manifest.eval_id}",
        "",
        f"**Status:** {manifest.status}",
        f"**Baseline:** {manifest.baseline_name or 'final_checkpoint'}",
        f"**Started:** {manifest.started_at}",
        f"**Finished:** {manifest.finished_at or 'running'}",
        "",
        "## Configuration",
        "",
        f"- Memory Builder checkpoint: `{manifest.memory_builder_checkpoint_id or 'none'}`",
        f"- Question Agent checkpoint: `{manifest.question_agent_checkpoint_id or 'none'}`",
        f"- Test records: {len(manifest.selected_record_ids)}",
        "",
        "## Test Metrics",
        "",
    ]

    metrics = manifest.metrics
    if metrics.get("test_memory_reward_mean") is not None:
        lines.append(f"- Memory reward (mean): {metrics['test_memory_reward_mean']:.4f}")
    if metrics.get("test_question_reward_mean") is not None:
        lines.append(f"- Question reward (mean): {metrics['test_question_reward_mean']:.4f}")
    if metrics.get("test_no_leakage_pass_rate") is not None:
        lines.append(f"- No-leakage pass rate: {metrics['test_no_leakage_pass_rate']:.2%}")

    lines.extend([
        f"- Records succeeded: {metrics.get('num_succeeded', 0)}",
        f"- Records failed: {metrics.get('num_failed', 0)}",
        "",
    ])

    if manifest.warnings:
        lines.extend([
            "## Warnings",
            "",
        ])
        for w in manifest.warnings:
            lines.append(f"- {w}")
        lines.append("")

    if manifest.errors:
        lines.extend([
            "## Errors",
            "",
        ])
        for e in manifest.errors:
            lines.append(f"- {e}")
        lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
