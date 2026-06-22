"""
Phase 3 Milestone 6: End-to-End GRPO Orchestrator

Orchestrates the complete Phase 3 distributed training pipeline:
1. Preflight validation
2. Runtime plan generation
3. Rollout stage (Memory Builder + Question Agent + Answerer)
4. Reward stage (reward worker + weakness book updates)
5. Replay stage (replay buffer validation)
6. Trainer stage (trainer adapter)
7. Checkpoint registry update
8. Inspection (no-leakage validation)
9. Final report generation

Key principle: Thin orchestration layer that calls existing modules
and collects their reports. Does not duplicate rollout/reward/trainer logic.
"""

from __future__ import annotations

import json
import logging
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gaam_graph.code_a1_alignment import check_for_forbidden_fields
from gaam_graph.phase3_e2e_schema import (
    Phase3E2EBackend,
    Phase3E2ERunConfig,
    Phase3E2ERunReport,
    Phase3E2ERoundReport,
    Phase3E2EStageName,
    Phase3E2EStageReport,
    Phase3E2EStageStatus,
    create_stage_report,
    finalize_stage_report,
)

logger = logging.getLogger(__name__)

REAL_CASE_ID = "e47becba"


def _phase3_backend(config: Phase3E2ERunConfig):
    """Map E2E backend names onto the lower-level Phase 3 runtime backend."""
    from gaam_graph.distributed_runtime_schema import Phase3Backend

    if config.backend == Phase3E2EBackend.LOCAL_FALLBACK:
        return Phase3Backend.LOCAL_FAKE_DISTRIBUTED
    if config.backend == Phase3E2EBackend.RAY:
        return Phase3Backend.RAY
    if config.backend == Phase3E2EBackend.VENDORED_VERL:
        return Phase3Backend.VERL
    if config.backend == Phase3E2EBackend.CODE_A1_VERL:
        return Phase3Backend.CODE_A1_VERL
    raise ValueError(f"Unsupported backend: {config.backend}")


def _find_code_a1_root() -> Path | None:
    """Find the local Code-A1 root from common project working directories."""
    candidates = [
        Path("Code-A1/Code-A1"),
        Path("../Code-A1/Code-A1"),
        Path("gaam_memory_training/Code-A1/Code-A1"),
        Path(__file__).resolve().parents[1] / "Code-A1" / "Code-A1",
        Path(__file__).resolve().parents[2] / "Code-A1" / "Code-A1",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_record(config: Phase3E2ERunConfig):
    """Load the requested LongMemEval record."""
    from gaam_graph.lme_loader import LongMemEvalLoader

    loader = LongMemEvalLoader(config.input_records_path)
    records = loader.load()
    for record in records:
        if record.record_id == config.record_id:
            return record
    raise ValueError(f"Record {config.record_id} not found in {config.input_records_path}")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    """Copy an optional smoke artifact when it exists."""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def _materialize_current_memory(config: Phase3E2ERunConfig, job_dir: Path) -> Path:
    """Build or copy the CurrentMemory artifact consumed by reward and answer stages."""
    output_path = job_dir / "current_memory.json"
    smoke_path = Path("outputs/current_memory_test") / f"{config.record_id}.current_memory.json"
    if _copy_if_exists(smoke_path, output_path):
        return output_path

    from gaam_graph.memory_builder import BaselineMemoryBuilder, build_current_memory_from_record
    from gaam_graph.memory_schema import validate_current_memory

    record = _load_record(config)
    builder = BaselineMemoryBuilder()
    memory = build_current_memory_from_record(record, builder=builder)
    validate_current_memory(memory, strict=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)
    return output_path


def _materialize_question_set(config: Phase3E2ERunConfig, job_dir: Path) -> Path:
    """Build or copy oracle-valid generated questions for the rollout job."""
    output_path = job_dir / "questions.json"
    smoke_path = Path("outputs/question_sets_smoke") / f"{config.record_id}.accepted_questions.json"
    if _copy_if_exists(smoke_path, output_path):
        return output_path

    from gaam_graph.oracle_graph_loader import OracleGraphLoader
    from gaam_graph.oracle_validity_checker import OracleValidityChecker
    from gaam_graph.question_agent import QuestionAgent
    from gaam_graph.question_schema import OracleValidityVerdict, QuestionStatus

    graph_path = Path(config.oracle_graph_dir) / f"{config.record_id}.graph.json"
    loader = OracleGraphLoader(graph_path, strict=False, sanitize=True)
    loader.validate_no_leakage()
    loader.validate_integrity()

    # The QuestionAgent needs the graph dictionary for walking. The sanitized
    # loader keeps legacy benchmark metadata out of this teacher-side artifact.
    graph_dict = getattr(loader, "_graph_data")
    agent = QuestionAgent(llm=None, no_llm=True)
    trajectories = agent.sample_trajectories(
        graph=graph_dict,
        num_walks=8,
        walk_length=5,
        seed=config.seed,
        directed=False,
        require_multi_session=False,
        min_sessions=2,
        max_structural_ratio=0.35,
        max_consecutive_next=1,
    )
    candidates = agent.generate_candidates(
        record_id=config.record_id,
        graph_path=str(graph_path),
        trajectories=trajectories,
        num_questions=5,
        require_multi_session=False,
        min_sessions=2,
    )

    checker = OracleValidityChecker(
        oracle_loader=loader,
        allow_placeholder_questions=True,
    )
    reports = checker.validate_questions(
        questions=candidates,
        trajectories=trajectories,
        require_multi_session=False,
        min_sessions=2,
    )

    accepted = []
    for question, validity_report in zip(candidates, reports):
        if validity_report.verdict in {
            OracleValidityVerdict.ACCEPT,
            OracleValidityVerdict.REVISE,
        }:
            question.status = QuestionStatus.ACCEPTED
            accepted.append(question)
        else:
            question.status = QuestionStatus.REJECTED

    if not accepted:
        raise ValueError("Question Agent generated no oracle-valid questions")

    accepted_ids = {question.question_id for question in accepted}
    question_set = {
        "record_id": config.record_id,
        "graph_path": str(graph_path),
        "generation_config": {
            "num_walks": 8,
            "walk_length": 5,
            "num_questions": 5,
            "no_llm": True,
            "seed": config.seed,
        },
        "questions": [question.model_dump(mode="json") for question in accepted],
        "validity_reports": [
            report.model_dump(mode="json")
            for report, question in zip(reports, candidates)
            if question.question_id in accepted_ids
        ],
        "metadata": {"source": "phase3_e2e_orchestrator"},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(question_set, f, ensure_ascii=False, indent=2)
    return output_path


def _materialize_answers(config: Phase3E2ERunConfig, job_dir: Path) -> Path:
    """Build or copy Frozen Answerer reports for the rollout job."""
    output_path = job_dir / "answers.json"
    smoke_path = Path("outputs/answers_smoke") / f"{config.record_id}.answers.json"
    if _copy_if_exists(smoke_path, output_path):
        return output_path

    from gaam_graph.answer_agent import FrozenAnswerer
    from gaam_graph.answer_schema import answer_request_from_generated_question
    from gaam_graph.memory_retriever import CurrentMemoryRetriever

    current_memory_path = job_dir / "current_memory.json"
    question_set_path = job_dir / "questions.json"
    with open(current_memory_path, "r", encoding="utf-8") as f:
        current_memory = json.load(f)
    with open(question_set_path, "r", encoding="utf-8") as f:
        question_set = json.load(f)

    requests = []
    for question in question_set.get("questions", []):
        request = answer_request_from_generated_question(question)
        request.current_memory_path = str(current_memory_path)
        requests.append(request)

    if not requests:
        raise ValueError("No questions available for Frozen Answerer")

    answerer = FrozenAnswerer(
        llm=None,
        no_llm=True,
        retriever=CurrentMemoryRetriever(top_k=8, include_summaries=True),
        model_id="no_llm_baseline",
        max_answer_chars=500,
    )
    reports = answerer.batch_answer(current_memory=current_memory, requests=requests)

    answered_count = sum(1 for report in reports if report.answer_status == "answered")
    insufficient_count = sum(
        1 for report in reports if report.answer_status == "insufficient_evidence"
    )
    avg_confidence = (
        sum(report.confidence for report in reports) / len(reports)
        if reports
        else 0.0
    )

    answer_artifact = {
        "record_id": config.record_id,
        "current_memory_path": str(current_memory_path),
        "answerer_config": {
            "backend": "no_llm",
            "model": "no_llm_baseline",
            "top_k": 8,
            "include_summaries": True,
        },
        "answers": [report.model_dump(mode="json") for report in reports],
        "summary": {
            "num_questions": len(reports),
            "answered_count": answered_count,
            "insufficient_evidence_count": insufficient_count,
            "average_confidence": round(avg_confidence, 3),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answer_artifact, f, ensure_ascii=False, indent=2)
    return output_path


def _materialize_rollout_reward_artifacts(
    config: Phase3E2ERunConfig,
    job_dir: Path,
) -> dict[str, str]:
    """Ensure the rollout job has the artifacts required by the reward service."""
    current_memory_path = _materialize_current_memory(config, job_dir)
    question_set_path = _materialize_question_set(config, job_dir)
    answer_report_path = _materialize_answers(config, job_dir)
    return {
        "current_memory": str(current_memory_path),
        "questions": str(question_set_path),
        "answers": str(answer_report_path),
    }


def _assess_remote_gpu_readiness(
    *,
    config: Phase3E2ERunConfig,
    paths: Phase3E2EPaths,
    round_report: Phase3E2ERoundReport,
) -> tuple[bool, list[str]]:
    """Assess whether this E2E pipeline is ready to move to a remote GPU server."""
    gaps = []

    if round_report.status not in {
        Phase3E2EStageStatus.SUCCEEDED,
        Phase3E2EStageStatus.PARTIAL,
    }:
        gaps.append("phase3_e2e_run_not_successful")
    if not round_report.no_leakage_passed:
        gaps.append("no_leakage_not_passed")
    if not round_report.replay_buffer_path or not Path(round_report.replay_buffer_path).exists():
        gaps.append("replay_buffer_missing")
    if not paths.checkpoint_registry_path.exists():
        gaps.append("checkpoint_registry_missing")
    if config.backend == Phase3E2EBackend.RAY:
        try:
            import ray  # noqa: F401
        except ImportError:
            gaps.append("ray_not_installed_for_ray_backend")
    if config.trainer_mode in {"local_model", "vendored_verl"}:
        missing_model_paths = []
        if not config.memory_builder_model_path:
            missing_model_paths.append("memory_builder_model_path")
        if not config.question_agent_model_path:
            missing_model_paths.append("question_agent_model_path")
        if missing_model_paths:
            gaps.append("missing_model_paths:" + ",".join(missing_model_paths))

    return not gaps, gaps


def _assess_code_a1_verl_readiness(config: Phase3E2ERunConfig) -> tuple[bool, list[str]]:
    """Assess whether Code-A1/verl conversion is available in this environment."""
    gaps = []
    code_a1_path = _find_code_a1_root()
    if code_a1_path is None:
        gaps.append("code_a1_path_missing")
    try:
        from verl import protocol  # noqa: F401
    except Exception as exc:
        gaps.append(f"vendored_verl_not_importable:{type(exc).__name__}:{exc}")
    if not config.write_verl_dataproto:
        gaps.append("write_verl_dataproto_not_requested")
    return not gaps, gaps


# ============================================================================
# Path Planning
# ============================================================================


class Phase3E2EPaths(BaseModel):
    """Output path structure for E2E run."""

    model_config = {"arbitrary_types_allowed": True}

    output_dir: Path
    config_path: Path
    final_report_path: Path
    summary_path: Path
    stage_manifest_path: Path
    round_dir: Path
    rollout_dir: Path
    reward_dir: Path
    trainer_dir: Path
    inspection_dir: Path
    checkpoint_registry_path: Path
    checkpoints_dir: Path


def plan_output_paths(config: Phase3E2ERunConfig) -> Phase3E2EPaths:
    """Plan output directory structure."""
    output_dir = Path(config.output_dir)
    round_dir = output_dir / "rounds" / f"round_{config.round_id:03d}_step_{config.step_id:03d}"

    return Phase3E2EPaths(
        output_dir=output_dir,
        config_path=output_dir / "config.json",
        final_report_path=output_dir / "final_report.json",
        summary_path=output_dir / "summary.md",
        stage_manifest_path=output_dir / "stage_manifest.jsonl",
        round_dir=round_dir,
        rollout_dir=round_dir / "rollout",
        reward_dir=round_dir / "reward",
        trainer_dir=round_dir / "trainer",
        inspection_dir=round_dir / "inspections",
        checkpoint_registry_path=output_dir / "checkpoint_registry.json",
        checkpoints_dir=output_dir / "checkpoints",
    )


# ============================================================================
# Stage: Preflight
# ============================================================================


def run_preflight(config: Phase3E2ERunConfig) -> Phase3E2EStageReport:
    """
    Validate environment and required inputs before execution.

    Checks:
    - Input files exist
    - Oracle graph directory exists
    - Real case e47becba graph exists when required
    - Output directory policy (overwrite/resume)
    - Python imports for Phase 3 modules
    - Optional: Ray availability
    - Optional: vendored verl availability
    """
    report = create_stage_report(Phase3E2EStageName.PREFLIGHT, Phase3E2EStageStatus.RUNNING)

    metrics: dict[str, Any] = {}
    warnings: list[str] = []
    errors: list[str] = []

    try:
        # Check input records
        input_path = Path(config.input_records_path)
        if not input_path.exists():
            errors.append(f"Input records not found: {input_path}")
        else:
            metrics["input_records_exists"] = True

        # Check oracle graph directory
        oracle_graph_dir = Path(config.oracle_graph_dir)
        if not oracle_graph_dir.exists():
            errors.append(f"Oracle graph directory not found: {oracle_graph_dir}")
        else:
            metrics["oracle_graph_exists"] = True

            # Check real case
            real_case_graph = oracle_graph_dir / f"{REAL_CASE_ID}.graph.json"
            if config.require_real_case and config.record_id == REAL_CASE_ID:
                if not real_case_graph.exists():
                    errors.append(f"Real case {REAL_CASE_ID} graph not found: {real_case_graph}")
                else:
                    metrics["real_case_id"] = REAL_CASE_ID
                    metrics["real_case_graph_exists"] = True

        # Check output directory policy
        output_dir = Path(config.output_dir)
        if output_dir.exists():
            if not config.overwrite and not config.resume:
                errors.append(
                    f"Output directory exists: {output_dir}. Use --overwrite or --resume."
                )
            elif config.overwrite and config.resume:
                errors.append("Cannot use both --overwrite and --resume")
        else:
            metrics["output_dir_clean"] = True

        # Check Python imports
        try:
            from gaam_graph import distributed_reward_service  # noqa: F401
            from gaam_graph import ray_runtime  # noqa: F401
            from gaam_graph import replay_buffer  # noqa: F401
            from gaam_graph import verl_trainer_adapter  # noqa: F401

            metrics["phase3_imports_ok"] = True
        except ImportError as e:
            errors.append(f"Phase 3 import failed: {e}")

        # Check Ray availability
        try:
            import ray  # noqa: F401

            metrics["ray_available"] = True
        except ImportError:
            metrics["ray_available"] = False
            if config.backend == Phase3E2EBackend.RAY:
                errors.append("Ray backend requested but Ray is not available")

        # Check vendored verl availability
        try:
            from verl import protocol  # noqa: F401

            metrics["vendored_verl_available"] = True
        except Exception as e:
            metrics["vendored_verl_available"] = False
            metrics["vendored_verl_import_error"] = f"{type(e).__name__}: {e}"
            if config.backend in [Phase3E2EBackend.VENDORED_VERL, Phase3E2EBackend.CODE_A1_VERL]:
                if config.strict:
                    errors.append(
                        f"Vendored verl requested but not available: {type(e).__name__}: {e}"
                    )
                else:
                    warnings.append(
                        f"Vendored verl not available (will use fallback): {type(e).__name__}: {e}"
                    )

        # Check Code-A1 path for Code-A1/verl-oriented runs
        code_a1_path = _find_code_a1_root()
        metrics["code_a1_path_exists"] = code_a1_path is not None
        if code_a1_path is not None:
            metrics["code_a1_path"] = str(code_a1_path)
        if config.backend == Phase3E2EBackend.CODE_A1_VERL and code_a1_path is None:
            errors.append("Code-A1 path not found")

        # Check GPU availability
        try:
            import torch

            metrics["gpu_available"] = torch.cuda.is_available()
        except ImportError:
            metrics["gpu_available"] = False

        # Determine status
        status = (
            Phase3E2EStageStatus.FAILED
            if errors
            else Phase3E2EStageStatus.SUCCEEDED
        )

        report.status = status
        report.metrics = metrics
        report.warnings = warnings
        report.errors = errors

    except Exception as e:
        report.status = Phase3E2EStageStatus.FAILED
        report.errors.append(f"Preflight exception: {type(e).__name__}: {e}")
        logger.error(f"Preflight failed: {e}\n{traceback.format_exc()}")

    return finalize_stage_report(report)


# ============================================================================
# Stage: Runtime Plan
# ============================================================================


def build_runtime_plan(
    config: Phase3E2ERunConfig, paths: Phase3E2EPaths
) -> Phase3E2EStageReport:
    """
    Build runtime plan for rollout/reward workers.

    Reuses Phase 3 Milestone 1 runtime planning schemas.
    """
    report = create_stage_report(
        Phase3E2EStageName.RUNTIME_PLAN, Phase3E2EStageStatus.RUNNING
    )

    try:
        from gaam_graph.distributed_runtime import Phase3RuntimePlanner
        from gaam_graph.distributed_runtime_schema import (
            ActorRuntimeConfig,
            Phase3RuntimeConfig,
        )
        from gaam_graph.grpo_schema import ActorRole

        # Build runtime config
        runtime_config = Phase3RuntimeConfig(
            runtime_id=config.run_id,
            backend=_phase3_backend(config),
            input_records_path=Path(config.input_records_path),
            oracle_graph_dir=Path(config.oracle_graph_dir),
            output_dir=paths.rollout_dir,
            checkpoint_registry_path=Path(config.checkpoint_registry_path)
            if config.checkpoint_registry_path
            else None,
            record_id_filter=config.record_id,
            max_records=config.max_records,
            round_id=config.round_id,
            step_id=config.step_id,
            seed=config.seed,
            num_rollout_workers=config.num_rollout_workers,
            num_reward_workers=config.num_reward_workers,
            memory_builder=ActorRuntimeConfig(
                role=ActorRole.MEMORY_BUILDER,
                policy_id="memory_builder_policy",
                model_path=config.memory_builder_model_path or "stub",
                checkpoint_id=None,
            ),
            question_agent=ActorRuntimeConfig(
                role=ActorRole.QUESTION_AGENT,
                policy_id="question_agent_policy",
                model_path=config.question_agent_model_path or "stub",
                checkpoint_id=None,
            ),
            answerer_config={"model_path": config.answerer_model_path or "stub"},
            reward_config={"mode": config.reward_mode},
        )

        planner = Phase3RuntimePlanner(runtime_config)
        worker_plan = planner.build_worker_plan()

        # Write runtime plan
        plan_path = paths.round_dir / "runtime_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)

        plan_data = {
            "runtime_config": runtime_config.model_dump(mode="json"),
            "worker_plan": worker_plan.model_dump(mode="json"),
            "planner_issues": [issue.model_dump(mode="json") for issue in planner.issues],
        }

        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, indent=2)

        report.status = Phase3E2EStageStatus.SUCCEEDED
        report.primary_artifact_path = str(plan_path)
        report.metrics = {
            "num_jobs": len(worker_plan.jobs),
            "record_ids": [job.record_id for job in worker_plan.jobs],
            "num_workers": len(worker_plan.workers),
            "planner_issue_count": len(planner.issues),
        }
        if planner.issues:
            report.warnings.extend(issue.message for issue in planner.issues)

    except Exception as e:
        report.status = Phase3E2EStageStatus.FAILED
        report.errors.append(f"Runtime plan failed: {type(e).__name__}: {e}")
        logger.error(f"Runtime plan failed: {e}\n{traceback.format_exc()}")

    return finalize_stage_report(report)


# ============================================================================
# Stage: Rollout
# ============================================================================


def run_rollout_stage(
    config: Phase3E2ERunConfig, paths: Phase3E2EPaths
) -> Phase3E2EStageReport:
    """
    Run distributed rollout stage.

    Calls Phase 3 Milestone 3 Ray runtime or local fallback.
    Produces: current_memory, questions, answers, dataproto batches.
    """
    report = create_stage_report(Phase3E2EStageName.ROLLOUT, Phase3E2EStageStatus.RUNNING)

    try:
        from gaam_graph.distributed_runtime import Phase3RuntimePlanner
        from gaam_graph.distributed_runtime_schema import (
            ActorRuntimeConfig,
            Phase3JobStatus,
            Phase3RuntimeConfig,
        )
        from gaam_graph.grpo_schema import ActorRole
        from gaam_graph.ray_runtime import RayRuntimeConfig, RayRuntimeExecutor

        runtime_config = Phase3RuntimeConfig(
            runtime_id=config.run_id,
            backend=_phase3_backend(config),
            input_records_path=Path(config.input_records_path),
            oracle_graph_dir=Path(config.oracle_graph_dir),
            output_dir=paths.rollout_dir,
            checkpoint_registry_path=Path(config.checkpoint_registry_path)
            if config.checkpoint_registry_path
            else None,
            record_id_filter=config.record_id,
            max_records=config.max_records,
            round_id=config.round_id,
            step_id=config.step_id,
            seed=config.seed,
            num_rollout_workers=config.num_rollout_workers,
            num_reward_workers=config.num_reward_workers,
            memory_builder=ActorRuntimeConfig(
                role=ActorRole.MEMORY_BUILDER,
                policy_id="memory_builder_policy",
                model_path=config.memory_builder_model_path or "stub",
            ),
            question_agent=ActorRuntimeConfig(
                role=ActorRole.QUESTION_AGENT,
                policy_id="question_agent_policy",
                model_path=config.question_agent_model_path or "stub",
            ),
            answerer_config={"model_path": config.answerer_model_path or "stub"},
            reward_config={"mode": config.reward_mode},
        )
        worker_plan = Phase3RuntimePlanner(runtime_config).build_worker_plan()
        executor = RayRuntimeExecutor(
            config=runtime_config,
            worker_plan=worker_plan,
            ray_config=RayRuntimeConfig(
                require_ray=config.backend == Phase3E2EBackend.RAY,
                local_mode=config.backend != Phase3E2EBackend.RAY,
            ),
            write_torch_tensors=True,
        )
        rollout_results = executor.run()

        if not rollout_results:
            report.status = Phase3E2EStageStatus.FAILED
            report.errors.append("Rollout executor produced no job results")
            return finalize_stage_report(report)

        rollout_result = rollout_results[0]
        job_dir = Path(rollout_result.output_dir)
        reward_artifacts = _materialize_rollout_reward_artifacts(config, job_dir)

        # Check result
        if rollout_result.status == Phase3JobStatus.SUCCEEDED:
            report.status = Phase3E2EStageStatus.SUCCEEDED
        else:
            report.status = Phase3E2EStageStatus.FAILED

        report.output_dir = str(paths.rollout_dir)
        report.primary_artifact_path = str(paths.rollout_dir / "ray_runtime_summary.json")
        report.artifact_paths = {
            **reward_artifacts,
            "job_result": str(job_dir / "job_result.json"),
            "dataproto_manifest": rollout_result.dataproto_manifest_path or "",
        }
        report.metrics = {
            "num_jobs": len(rollout_results),
            "succeeded_jobs": sum(
                1 for result in rollout_results if result.status == Phase3JobStatus.SUCCEEDED
            ),
            "dataproto_exported": bool(rollout_result.dataproto_manifest_path),
            "materialized_reward_artifacts": sorted(reward_artifacts.keys()),
            **rollout_result.metrics,
        }
        if rollout_result.error:
            report.errors.append(rollout_result.error)

    except Exception as e:
        report.status = Phase3E2EStageStatus.FAILED
        report.errors.append(f"Rollout stage failed: {type(e).__name__}: {e}")
        logger.error(f"Rollout stage failed: {e}\n{traceback.format_exc()}")

    return finalize_stage_report(report)


# ============================================================================
# Stage: Reward
# ============================================================================


def run_reward_stage(
    config: Phase3E2ERunConfig, paths: Phase3E2EPaths, rollout_report: Phase3E2EStageReport
) -> Phase3E2EStageReport:
    """
    Run distributed reward stage.

    Calls Phase 3 Milestone 5 reward service.
    Produces: reward reports, weakness deltas, replay items.
    """
    report = create_stage_report(Phase3E2EStageName.REWARD, Phase3E2EStageStatus.RUNNING)

    try:
        from gaam_graph.distributed_reward_service import (
            build_reward_worker_input_from_job_dir,
            run_reward_worker_job_local,
        )
        from gaam_graph.replay_buffer import ReplayBufferWriter

        # Get rollout job directory
        rollout_job_dir = Path(paths.rollout_dir) / "jobs" / config.record_id

        if not rollout_job_dir.exists():
            report.status = Phase3E2EStageStatus.FAILED
            report.errors.append(f"Rollout job directory not found: {rollout_job_dir}")
            return finalize_stage_report(report)

        # Build reward worker input
        reward_output_dir = paths.reward_dir / "jobs" / config.record_id
        reward_input = build_reward_worker_input_from_job_dir(
            job_dir=rollout_job_dir,
            oracle_graph_dir=Path(config.oracle_graph_dir),
            output_dir=reward_output_dir,
            record_id=config.record_id,
            round_id=config.round_id,
            step_id=config.step_id,
            reward_mode=config.reward_mode,
            compute_memory_builder_reward=True,
            compute_question_agent_reward=True,
            update_weakness_book=True,
            write_replay_item=True,
        )

        # Run reward worker
        reward_result = run_reward_worker_job_local(reward_input)

        # Write replay buffer
        replay_buffer_path = paths.reward_dir / "replay_buffer.jsonl"
        if reward_result.replay_item_path:
            with open(reward_result.replay_item_path, "r", encoding="utf-8") as f:
                from gaam_graph.distributed_reward_schema import ReplayBufferItem

                item_data = json.load(f)
                item = ReplayBufferItem.model_validate(item_data)

            replay_writer = ReplayBufferWriter(replay_buffer_path)
            replay_writer.append(item)

        # Check result
        if reward_result.status.value == "succeeded":
            report.status = Phase3E2EStageStatus.SUCCEEDED
        elif reward_result.status.value == "partial":
            if config.require_memory_reward and reward_result.memory_update_reward is None:
                report.status = Phase3E2EStageStatus.FAILED
                report.errors.append("Memory reward is required but missing")
            elif config.require_question_reward and reward_result.question_agent_reward is None:
                report.status = Phase3E2EStageStatus.FAILED
                report.errors.append("Question reward is required but missing")
            else:
                report.status = Phase3E2EStageStatus.PARTIAL
        else:
            report.status = Phase3E2EStageStatus.FAILED

        report.output_dir = str(paths.reward_dir)
        report.primary_artifact_path = str(reward_output_dir / "reward_worker_report.json")
        report.artifact_paths = {
            "memory_reward": reward_result.memory_reward_path or "",
            "question_reward": reward_result.question_reward_path or "",
            "weakness_book": reward_result.weakness_book_path or "",
            "weakness_delta": reward_result.weakness_delta_path or "",
            "replay_item": reward_result.replay_item_path or "",
            "replay_buffer": str(replay_buffer_path),
        }
        report.metrics = {
            "memory_reward": reward_result.memory_update_reward,
            "question_reward": reward_result.question_agent_reward,
            "weakness_updates": reward_result.weakness_update_count,
        }
        report.warnings.extend(reward_result.warnings)
        report.errors.extend(reward_result.errors)

    except Exception as e:
        report.status = Phase3E2EStageStatus.FAILED
        report.errors.append(f"Reward stage failed: {type(e).__name__}: {e}")
        logger.error(f"Reward stage failed: {e}\n{traceback.format_exc()}")

    return finalize_stage_report(report)


# ============================================================================
# Stage: Replay
# ============================================================================


def run_replay_stage(
    config: Phase3E2ERunConfig, paths: Phase3E2EPaths, reward_report: Phase3E2EStageReport
) -> Phase3E2EStageReport:
    """
    Validate replay buffer before trainer execution.

    Checks:
    - Replay buffer exists
    - At least one item for record
    - Selected items have no_leakage_passed=True
    - Reward report paths exist
    - DataProto paths exist when needed
    """
    report = create_stage_report(Phase3E2EStageName.REPLAY, Phase3E2EStageStatus.RUNNING)

    try:
        from gaam_graph.replay_buffer import ReplayBufferReader

        replay_buffer_path = Path(reward_report.artifact_paths.get("replay_buffer", ""))

        if not replay_buffer_path.exists():
            report.status = Phase3E2EStageStatus.FAILED
            report.errors.append(f"Replay buffer not found: {replay_buffer_path}")
            return finalize_stage_report(report)

        # Read replay buffer
        reader = ReplayBufferReader(replay_buffer_path)
        all_items = list(reader.iter_items())

        # Filter for this record
        record_items = [item for item in all_items if item.record_id == config.record_id]

        if not record_items:
            report.status = Phase3E2EStageStatus.FAILED
            report.errors.append(f"No replay items found for record {config.record_id}")
            return finalize_stage_report(report)

        # Check selected items
        selected_items = [item for item in record_items if item.selected_for_training]

        if config.require_memory_reward and not selected_items:
            report.status = Phase3E2EStageStatus.FAILED
            report.errors.append("No replay items selected for training")
            return finalize_stage_report(report)

        # Check no-leakage
        no_leakage_failed = [
            item for item in selected_items if not item.no_leakage_passed
        ]
        if no_leakage_failed:
            report.status = Phase3E2EStageStatus.FAILED
            report.errors.append(
                f"{len(no_leakage_failed)} replay items failed no-leakage validation"
            )
            return finalize_stage_report(report)

        # Check reward paths exist
        for item in selected_items:
            if item.memory_reward_report_path:
                if not Path(item.memory_reward_report_path).exists():
                    report.warnings.append(
                        f"Memory reward report missing: {item.memory_reward_report_path}"
                    )

        report.status = Phase3E2EStageStatus.SUCCEEDED
        report.metrics = {
            "num_replay_items": len(record_items),
            "num_selected_for_training": len(selected_items),
            "num_low_memory_reward": len(
                [item for item in record_items if "low_memory_reward" in item.diagnostic_tags]
            ),
            "num_weakness_updated": len(
                [item for item in record_items if "weakness_updated" in item.diagnostic_tags]
            ),
        }
        report.artifact_paths = {"replay_buffer": str(replay_buffer_path)}

    except Exception as e:
        report.status = Phase3E2EStageStatus.FAILED
        report.errors.append(f"Replay stage failed: {type(e).__name__}: {e}")
        logger.error(f"Replay stage failed: {e}\n{traceback.format_exc()}")

    return finalize_stage_report(report)


# ============================================================================
# Stage: Trainer
# ============================================================================


def run_trainer_stage(
    config: Phase3E2ERunConfig, paths: Phase3E2EPaths, replay_report: Phase3E2EStageReport
) -> Phase3E2EStageReport:
    """
    Run trainer adapter stage.

    Calls Phase 3 Milestone 4 trainer adapter.
    Supports dry_run, local_model, vendored_verl modes.
    """
    report = create_stage_report(Phase3E2EStageName.TRAINER, Phase3E2EStageStatus.RUNNING)

    try:
        from gaam_graph.grpo_schema import ActorRole
        from gaam_graph.replay_buffer import ReplayBufferReader
        from gaam_graph.verl_trainer_adapter import (
            VerlTrainerAdapterMode,
            run_verl_trainer_adapter_step,
        )

        replay_buffer_path = Path(replay_report.artifact_paths.get("replay_buffer", ""))
        if not replay_buffer_path.exists():
            report.status = Phase3E2EStageStatus.FAILED
            report.errors.append("Replay buffer not found")
            return finalize_stage_report(report)

        # Get selected replay items
        reader = ReplayBufferReader(replay_buffer_path)
        selected_items = reader.sample(
            max_items=100,
            record_id=config.record_id,
            require_no_leakage=True,
            require_selected_for_training=True,
        )

        if not selected_items:
            report.status = Phase3E2EStageStatus.SKIPPED
            report.warnings.append("No selected replay items for training")
            return finalize_stage_report(report)

        # Process each actor
        actor_reports = {}
        actor_statuses = {}

        for actor_role in [ActorRole.MEMORY_BUILDER, ActorRole.QUESTION_AGENT]:
            if actor_role == ActorRole.QUESTION_AGENT and not config.update_question_agent:
                continue
            if actor_role == ActorRole.MEMORY_BUILDER and not config.update_memory_builder:
                continue

            actor_output_dir = paths.trainer_dir / actor_role.value
            actor_output_dir.mkdir(parents=True, exist_ok=True)

            # Find dataproto batch for this actor
            dataproto_batch_path = None
            for item in selected_items:
                batch_path = item.trainer_ready_batch_paths.get(actor_role.value)
                if batch_path and Path(batch_path).exists():
                    dataproto_batch_path = batch_path
                    break

            if not dataproto_batch_path:
                report.warnings.append(
                    f"No DataProto batch found for {actor_role.value}, skipping"
                )
                continue

            # Run trainer adapter
            try:
                adapter_result = run_verl_trainer_adapter_step(
                    dataproto_dir=Path(dataproto_batch_path).parent,
                    output_dir=actor_output_dir,
                    actor_role=actor_role,
                    mode=VerlTrainerAdapterMode(config.trainer_mode),
                    checkpoint_registry_path=paths.checkpoint_registry_path,
                    write_verl_dataproto=config.write_verl_dataproto,
                )

                report_path = actor_output_dir / "trainer_step_report.json"
                actor_reports[f"{actor_role.value}_report"] = str(report_path)
                if adapter_result.trainer_ready_batch_path:
                    actor_reports[f"{actor_role.value}_trainer_ready_batch"] = (
                        adapter_result.trainer_ready_batch_path
                    )
                actor_statuses[actor_role.value] = adapter_result.status

            except Exception as e:
                report.warnings.append(
                    f"Trainer adapter failed for {actor_role.value}: {type(e).__name__}: {e}"
                )

        if not actor_reports:
            report.status = Phase3E2EStageStatus.FAILED
            report.errors.append("No actors processed successfully")
        else:
            if any(status == "failed" for status in actor_statuses.values()):
                report.status = Phase3E2EStageStatus.PARTIAL
            else:
                report.status = Phase3E2EStageStatus.SUCCEEDED
            report.output_dir = str(paths.trainer_dir)
            report.primary_artifact_path = str(paths.trainer_dir / "adapter_report.json")
            report.metrics = {
                "actors_processed": list(actor_statuses.keys()),
                "actor_statuses": actor_statuses,
            }
            report.artifact_paths = actor_reports

            adapter_report_path = paths.trainer_dir / "adapter_report.json"
            with open(adapter_report_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "status": report.status.value,
                        "actor_statuses": actor_statuses,
                        "artifact_paths": actor_reports,
                        "trainer_mode": config.trainer_mode,
                    },
                    f,
                    indent=2,
                )

    except Exception as e:
        report.status = Phase3E2EStageStatus.FAILED
        report.errors.append(f"Trainer stage failed: {type(e).__name__}: {e}")
        logger.error(f"Trainer stage failed: {e}\n{traceback.format_exc()}")

    return finalize_stage_report(report)


# ============================================================================
# Stage: Checkpoint
# ============================================================================


def run_checkpoint_stage(
    config: Phase3E2ERunConfig, paths: Phase3E2EPaths, trainer_report: Phase3E2EStageReport
) -> Phase3E2EStageReport:
    """
    Update checkpoint registry after trainer execution.

    Uses Phase 2 checkpoint registry utilities.
    """
    report = create_stage_report(
        Phase3E2EStageName.CHECKPOINT, Phase3E2EStageStatus.RUNNING
    )

    try:
        from gaam_graph.checkpoint_registry import CheckpointRegistry, write_checkpoint_registry

        registry_path = write_checkpoint_registry(
            trainer_dir=paths.trainer_dir,
            output_path=paths.checkpoint_registry_path,
        )
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = CheckpointRegistry.model_validate(json.load(f))

        report.status = Phase3E2EStageStatus.SUCCEEDED
        report.primary_artifact_path = str(registry_path)
        report.metrics = {
            "num_checkpoints": len(registry.checkpoints),
            "latest_by_actor": registry.latest_by_actor,
            "registry_issue_count": len(registry.issues),
            "dry_run_registry": config.trainer_mode == "dry_run",
        }

    except Exception as e:
        report.status = Phase3E2EStageStatus.FAILED
        report.errors.append(f"Checkpoint stage failed: {type(e).__name__}: {e}")
        logger.error(f"Checkpoint stage failed: {e}\n{traceback.format_exc()}")

    return finalize_stage_report(report)


# ============================================================================
# Stage: Inspection
# ============================================================================


def run_inspection_stage(
    config: Phase3E2ERunConfig,
    paths: Phase3E2EPaths,
    rollout_report: Phase3E2EStageReport,
    reward_report: Phase3E2EStageReport,
    replay_report: Phase3E2EStageReport,
    trainer_report: Phase3E2EStageReport,
) -> Phase3E2EStageReport:
    """
    Run no-leakage inspection and artifact validation.

    Validates:
    - Actor-facing artifacts pass no-leakage
    - All JSON artifacts parse
    - Required artifacts exist
    """
    report = create_stage_report(
        Phase3E2EStageName.INSPECTION, Phase3E2EStageStatus.RUNNING
    )

    try:
        paths.inspection_dir.mkdir(parents=True, exist_ok=True)

        no_leakage_results = {}
        artifact_results = {}

        # Check rollout artifacts
        if rollout_report.artifact_paths.get("current_memory"):
            current_memory_path = Path(rollout_report.artifact_paths["current_memory"])
            if current_memory_path.exists():
                with open(current_memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                forbidden = check_for_forbidden_fields(data)
                no_leakage_results["current_memory"] = len(forbidden) == 0
                if forbidden:
                    report.errors.append(
                        f"Current memory contains forbidden fields: {forbidden}"
                    )

        # Check replay items
        if replay_report.artifact_paths.get("replay_buffer"):
            replay_buffer_path = Path(replay_report.artifact_paths["replay_buffer"])
            if replay_buffer_path.exists():
                from gaam_graph.replay_buffer import ReplayBufferReader

                reader = ReplayBufferReader(replay_buffer_path)
                items = list(reader.iter_items())

                all_pass = all(item.no_leakage_passed for item in items)
                no_leakage_results["replay_buffer"] = all_pass
                if not all_pass:
                    report.errors.append("Some replay items failed no-leakage validation")

        # Write inspection report
        inspection_report_path = paths.inspection_dir / "no_leakage_report.json"
        with open(inspection_report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "no_leakage_results": no_leakage_results,
                    "artifact_results": artifact_results,
                    "overall_passed": len(report.errors) == 0,
                },
                f,
                indent=2,
            )

        if report.errors:
            report.status = Phase3E2EStageStatus.FAILED
        else:
            report.status = Phase3E2EStageStatus.SUCCEEDED

        report.primary_artifact_path = str(inspection_report_path)
        report.metrics = {
            "no_leakage_checks": len(no_leakage_results),
            "no_leakage_passed": sum(1 for v in no_leakage_results.values() if v),
        }

    except Exception as e:
        report.status = Phase3E2EStageStatus.FAILED
        report.errors.append(f"Inspection stage failed: {type(e).__name__}: {e}")
        logger.error(f"Inspection stage failed: {e}\n{traceback.format_exc()}")

    return finalize_stage_report(report)


# ============================================================================
# Main Orchestrator
# ============================================================================


def run_phase3_end_to_end_grpo(config: Phase3E2ERunConfig) -> Phase3E2ERunReport:
    """
    Run complete Phase 3 E2E GRPO pipeline.

    Executes all stages in order:
    1. Preflight
    2. Runtime plan
    3. Rollout
    4. Reward
    5. Replay
    6. Trainer
    7. Checkpoint
    8. Inspection
    9. Finalize

    Returns final run report.
    """
    started_at = datetime.utcnow().isoformat()

    # Plan output paths
    paths = plan_output_paths(config)

    # Handle output directory
    if paths.output_dir.exists():
        if config.overwrite:
            logger.info(f"Overwriting output directory: {paths.output_dir}")
            shutil.rmtree(paths.output_dir)
        elif not config.resume:
            raise ValueError(
                f"Output directory exists: {paths.output_dir}. Use --overwrite or --resume."
            )

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    # Write config
    with open(paths.config_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2)

    # Stage manifest writer
    def write_stage_manifest(stage_report: Phase3E2EStageReport) -> None:
        with open(paths.stage_manifest_path, "a", encoding="utf-8") as f:
            manifest_entry = {
                "stage": stage_report.stage.value,
                "status": stage_report.status.value,
                "primary_artifact_path": stage_report.primary_artifact_path,
                "started_at": stage_report.started_at,
                "finished_at": stage_report.finished_at,
            }
            f.write(json.dumps(manifest_entry) + "\n")

    # Initialize round report
    round_report = Phase3E2ERoundReport(
        round_id=config.round_id,
        step_id=config.step_id,
        record_ids=[config.record_id],
        status=Phase3E2EStageStatus.PENDING,
        stage_reports=[],
    )

    warnings: list[str] = []
    errors: list[str] = []

    try:
        # Stage 1: Preflight
        logger.info("Running preflight...")
        preflight_report = run_preflight(config)
        write_stage_manifest(preflight_report)
        round_report.stage_reports.append(preflight_report)

        if preflight_report.status == Phase3E2EStageStatus.FAILED:
            round_report.status = Phase3E2EStageStatus.FAILED
            errors.extend(preflight_report.errors)
            raise RuntimeError("Preflight failed")

        warnings.extend(preflight_report.warnings)

        # Stage 2: Runtime plan
        logger.info("Building runtime plan...")
        plan_report = build_runtime_plan(config, paths)
        write_stage_manifest(plan_report)
        round_report.stage_reports.append(plan_report)

        if plan_report.status == Phase3E2EStageStatus.FAILED:
            round_report.status = Phase3E2EStageStatus.FAILED
            errors.extend(plan_report.errors)
            raise RuntimeError("Runtime plan failed")

        # Stage 3: Rollout
        logger.info("Running rollout stage...")
        rollout_report = run_rollout_stage(config, paths)
        write_stage_manifest(rollout_report)
        round_report.stage_reports.append(rollout_report)

        if rollout_report.status == Phase3E2EStageStatus.FAILED:
            round_report.status = Phase3E2EStageStatus.FAILED
            errors.extend(rollout_report.errors)
            raise RuntimeError("Rollout stage failed")

        round_report.rollout_report_path = rollout_report.primary_artifact_path
        warnings.extend(rollout_report.warnings)

        # Stage 4: Reward
        logger.info("Running reward stage...")
        reward_report = run_reward_stage(config, paths, rollout_report)
        write_stage_manifest(reward_report)
        round_report.stage_reports.append(reward_report)

        if reward_report.status == Phase3E2EStageStatus.FAILED:
            round_report.status = Phase3E2EStageStatus.FAILED
            errors.extend(reward_report.errors)
            raise RuntimeError("Reward stage failed")

        round_report.reward_summary_path = reward_report.primary_artifact_path
        round_report.memory_reward_mean = reward_report.metrics.get("memory_reward")
        round_report.question_reward_mean = reward_report.metrics.get("question_reward")
        round_report.num_weakness_updates = reward_report.metrics.get("weakness_updates", 0)
        warnings.extend(reward_report.warnings)

        # Stage 5: Replay
        logger.info("Running replay stage...")
        replay_report = run_replay_stage(config, paths, reward_report)
        write_stage_manifest(replay_report)
        round_report.stage_reports.append(replay_report)

        if replay_report.status == Phase3E2EStageStatus.FAILED:
            round_report.status = Phase3E2EStageStatus.FAILED
            errors.extend(replay_report.errors)
            raise RuntimeError("Replay stage failed")

        round_report.replay_buffer_path = replay_report.artifact_paths.get("replay_buffer")
        round_report.num_replay_items = replay_report.metrics.get("num_replay_items", 0)
        warnings.extend(replay_report.warnings)

        # Stage 6: Trainer
        logger.info("Running trainer stage...")
        trainer_report = run_trainer_stage(config, paths, replay_report)
        write_stage_manifest(trainer_report)
        round_report.stage_reports.append(trainer_report)

        if trainer_report.status == Phase3E2EStageStatus.FAILED:
            if config.strict:
                round_report.status = Phase3E2EStageStatus.FAILED
                errors.extend(trainer_report.errors)
                raise RuntimeError("Trainer stage failed")
            else:
                warnings.append("Trainer stage failed but continuing in non-strict mode")

        round_report.trainer_report_path = trainer_report.primary_artifact_path
        warnings.extend(trainer_report.warnings)

        # Stage 7: Checkpoint
        logger.info("Running checkpoint stage...")
        checkpoint_report = run_checkpoint_stage(config, paths, trainer_report)
        write_stage_manifest(checkpoint_report)
        round_report.stage_reports.append(checkpoint_report)

        round_report.checkpoint_registry_path = checkpoint_report.primary_artifact_path
        warnings.extend(checkpoint_report.warnings)

        # Stage 8: Inspection
        logger.info("Running inspection stage...")
        inspection_report = run_inspection_stage(
            config, paths, rollout_report, reward_report, replay_report, trainer_report
        )
        write_stage_manifest(inspection_report)
        round_report.stage_reports.append(inspection_report)

        if inspection_report.status == Phase3E2EStageStatus.FAILED:
            round_report.status = Phase3E2EStageStatus.FAILED
            round_report.no_leakage_passed = False
            errors.extend(inspection_report.errors)
        else:
            round_report.no_leakage_passed = True

        # Determine final round status
        if round_report.status != Phase3E2EStageStatus.FAILED:
            if any(r.status == Phase3E2EStageStatus.PARTIAL for r in round_report.stage_reports):
                round_report.status = Phase3E2EStageStatus.PARTIAL
            else:
                round_report.status = Phase3E2EStageStatus.SUCCEEDED

    except Exception as e:
        logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
        round_report.status = Phase3E2EStageStatus.FAILED
        errors.append(f"Pipeline exception: {type(e).__name__}: {e}")

    # Build final report
    finished_at = datetime.utcnow().isoformat()
    remote_gpu_ready, remote_gpu_gaps = _assess_remote_gpu_readiness(
        config=config,
        paths=paths,
        round_report=round_report,
    )
    code_a1_verl_ready, code_a1_verl_gaps = _assess_code_a1_verl_readiness(config)
    final_weakness_book_path = None
    for stage_report in round_report.stage_reports:
        if stage_report.stage == Phase3E2EStageName.REWARD:
            final_weakness_book_path = stage_report.artifact_paths.get("weakness_book") or None
            break

    final_report = Phase3E2ERunReport(
        run_id=config.run_id,
        status=round_report.status,
        backend=config.backend,
        output_dir=str(paths.output_dir),
        record_id=config.record_id,
        round_reports=[round_report],
        final_checkpoint_registry_path=str(paths.checkpoint_registry_path)
        if paths.checkpoint_registry_path.exists()
        else None,
        final_replay_buffer_path=round_report.replay_buffer_path,
        final_weakness_book_path=final_weakness_book_path,
        final_summary_path=str(paths.summary_path),
        no_leakage_passed=round_report.no_leakage_passed,
        remote_gpu_ready=remote_gpu_ready,
        code_a1_verl_ready=code_a1_verl_ready,
        remote_gpu_readiness_gaps=remote_gpu_gaps,
        code_a1_verl_readiness_gaps=code_a1_verl_gaps,
        warnings=warnings,
        errors=errors,
        started_at=started_at,
        finished_at=finished_at,
    )

    # Write final report
    with open(paths.final_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report.model_dump(mode="json"), f, indent=2)

    # Write summary markdown
    write_summary_markdown(final_report, paths)

    return final_report


def write_summary_markdown(report: Phase3E2ERunReport, paths: Phase3E2EPaths) -> None:
    """Write human-readable summary markdown."""
    lines = []
    lines.append("# Phase 3 E2E GRPO Smoke Run\n")
    lines.append(f"- **Run ID:** {report.run_id}")
    lines.append(f"- **Record ID:** {report.record_id}")
    lines.append(f"- **Status:** {report.status.value}")
    lines.append(f"- **Backend:** {report.backend.value}")
    lines.append(f"- **Remote GPU ready:** {'yes' if report.remote_gpu_ready else 'no'}")
    lines.append(f"- **Code-A1/verl ready:** {'yes' if report.code_a1_verl_ready else 'no'}")

    if report.round_reports:
        round_report = report.round_reports[0]
        if round_report.memory_reward_mean is not None:
            lines.append(f"- **Memory reward:** {round_report.memory_reward_mean:.3f}")
        if round_report.question_reward_mean is not None:
            lines.append(f"- **Question reward:** {round_report.question_reward_mean:.3f}")
        lines.append(f"- **Weakness updates:** {round_report.num_weakness_updates}")
        lines.append(f"- **Replay items:** {round_report.num_replay_items}")

    lines.append(f"- **No-leakage:** {'passed' if report.no_leakage_passed else 'failed'}")
    if report.remote_gpu_readiness_gaps:
        lines.append(
            f"- **Remote GPU readiness gaps:** {', '.join(report.remote_gpu_readiness_gaps)}"
        )
    if report.code_a1_verl_readiness_gaps:
        lines.append(
            f"- **Code-A1/verl readiness gaps:** {', '.join(report.code_a1_verl_readiness_gaps)}"
        )
    lines.append("")

    lines.append("## Stage Results\n")
    lines.append("| Stage | Status | Primary Artifact |")
    lines.append("| --- | --- | --- |")

    if report.round_reports:
        for stage_report in report.round_reports[0].stage_reports:
            artifact = stage_report.primary_artifact_path or "—"
            if artifact and len(artifact) > 60:
                artifact = "..." + artifact[-57:]
            lines.append(
                f"| {stage_report.stage.value} | {stage_report.status.value} | {artifact} |"
            )

    with open(paths.summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
