"""
Phase 4 Milestone 5: Baseline Evaluation

Run baseline evaluations for comparison.

Baseline types:
- no_llm_memory: deterministic current memory baseline
- initial_checkpoint: model before co-training
- final_checkpoint: trained model (same as final evaluation)

Key functions:
- run_phase4_baseline_evaluations: run all baselines
- run_baseline_evaluation: run one baseline

Design principle:
Each baseline writes isolated output. Baselines are comparable.
"""

from pathlib import Path

from gaam_graph.phase4_experiment_schema import (
    Phase4FinalEvaluationConfig,
    Phase4FinalEvaluationManifest,
)
from gaam_graph.phase4_final_evaluation import run_phase4_final_evaluation


def run_phase4_baseline_evaluations(
    *,
    split_manifest_path: str,
    output_dir: Path,
    baselines: list[str],
    initial_memory_builder_checkpoint_id: str | None = None,
    initial_question_agent_checkpoint_id: str | None = None,
    final_memory_builder_checkpoint_id: str | None = None,
    final_question_agent_checkpoint_id: str | None = None,
    checkpoint_registry_path: str | None = None,
    memory_builder_model_path: str | None = None,
    question_agent_model_path: str | None = None,
    answerer_model_path: str | None = None,
    test_records_limit: int | None = None,
    seed: int = 0,
    overwrite: bool = False,
) -> list[Phase4FinalEvaluationManifest]:
    """
    Run multiple baseline evaluations.

    Args:
        split_manifest_path: path to split manifest
        output_dir: output directory for all baselines
        baselines: list of baseline names
        initial_memory_builder_checkpoint_id: pre-training checkpoint
        initial_question_agent_checkpoint_id: pre-training checkpoint
        final_memory_builder_checkpoint_id: post-training checkpoint
        final_question_agent_checkpoint_id: post-training checkpoint
        checkpoint_registry_path: checkpoint registry path
        memory_builder_model_path: model path
        question_agent_model_path: model path
        answerer_model_path: model path
        test_records_limit: max test records
        seed: random seed
        overwrite: overwrite existing results

    Returns:
        List of evaluation manifests
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for baseline_name in baselines:
        print(f"\n[Baseline] Running '{baseline_name}'")

        baseline_dir = output_dir / baseline_name

        try:
            manifest = run_baseline_evaluation(
                baseline_name=baseline_name,
                split_manifest_path=split_manifest_path,
                output_dir=baseline_dir,
                initial_memory_builder_checkpoint_id=initial_memory_builder_checkpoint_id,
                initial_question_agent_checkpoint_id=initial_question_agent_checkpoint_id,
                final_memory_builder_checkpoint_id=final_memory_builder_checkpoint_id,
                final_question_agent_checkpoint_id=final_question_agent_checkpoint_id,
                checkpoint_registry_path=checkpoint_registry_path,
                memory_builder_model_path=memory_builder_model_path,
                question_agent_model_path=question_agent_model_path,
                answerer_model_path=answerer_model_path,
                test_records_limit=test_records_limit,
                seed=seed,
                overwrite=overwrite,
            )

            results.append(manifest)

        except Exception as e:
            print(f"[Baseline] '{baseline_name}' failed: {e}")

    return results


def run_baseline_evaluation(
    *,
    baseline_name: str,
    split_manifest_path: str,
    output_dir: Path,
    initial_memory_builder_checkpoint_id: str | None = None,
    initial_question_agent_checkpoint_id: str | None = None,
    final_memory_builder_checkpoint_id: str | None = None,
    final_question_agent_checkpoint_id: str | None = None,
    checkpoint_registry_path: str | None = None,
    memory_builder_model_path: str | None = None,
    question_agent_model_path: str | None = None,
    answerer_model_path: str | None = None,
    test_records_limit: int | None = None,
    seed: int = 0,
    overwrite: bool = False,
) -> Phase4FinalEvaluationManifest:
    """
    Run one baseline evaluation.

    Args:
        baseline_name: baseline type
        split_manifest_path: path to split manifest
        output_dir: output directory for this baseline
        initial_memory_builder_checkpoint_id: pre-training checkpoint
        initial_question_agent_checkpoint_id: pre-training checkpoint
        final_memory_builder_checkpoint_id: post-training checkpoint
        final_question_agent_checkpoint_id: post-training checkpoint
        checkpoint_registry_path: checkpoint registry path
        memory_builder_model_path: model path
        question_agent_model_path: model path
        answerer_model_path: model path
        test_records_limit: max test records
        seed: random seed
        overwrite: overwrite existing results

    Returns:
        Evaluation manifest
    """
    # Select checkpoint IDs based on baseline type
    if baseline_name == "no_llm_memory":
        # No-LLM baseline: no checkpoint IDs (use deterministic memory builder)
        memory_checkpoint = None
        question_checkpoint = None

    elif baseline_name == "initial_checkpoint":
        # Pre-training baseline
        memory_checkpoint = initial_memory_builder_checkpoint_id
        question_checkpoint = initial_question_agent_checkpoint_id

    elif baseline_name == "final_checkpoint":
        # Post-training baseline (same as final evaluation)
        memory_checkpoint = final_memory_builder_checkpoint_id
        question_checkpoint = final_question_agent_checkpoint_id

    else:
        raise ValueError(f"Unknown baseline name: {baseline_name}")

    # Create evaluation config
    config = Phase4FinalEvaluationConfig(
        split_manifest_path=split_manifest_path,
        output_dir=str(output_dir),
        checkpoint_registry_path=checkpoint_registry_path,
        memory_builder_checkpoint_id=memory_checkpoint,
        question_agent_checkpoint_id=question_checkpoint,
        memory_builder_model_path=memory_builder_model_path,
        question_agent_model_path=question_agent_model_path,
        answerer_model_path=answerer_model_path,
        test_records_limit=test_records_limit,
        baseline_name=baseline_name,
        seed=seed,
        overwrite=overwrite,
    )

    # Run evaluation
    manifest = run_phase4_final_evaluation(config)

    return manifest
