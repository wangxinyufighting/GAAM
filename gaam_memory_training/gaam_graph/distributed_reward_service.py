"""
Phase 3 Milestone 5: Distributed Reward Service

Service layer for distributed reward computation, weakness book updates,
and replay buffer item generation.

Key features:
- Consumes Phase 3 rollout job outputs
- Reuses existing RewardManager logic
- Updates Weakness Book with mergeable deltas
- Writes trainer-safe replay items
- Supports local fallback and optional Ray execution
- Validates no-leakage at reward boundary
"""

import json
import logging
import os
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gaam_graph.answer_schema import AnswerReport
from gaam_graph.code_a1_alignment import check_for_forbidden_fields
from gaam_graph.distributed_reward_schema import (
    ReplayBufferItem,
    RewardWorkerInput,
    RewardWorkerReport,
    RewardWorkerStatus,
    WeaknessBookDelta,
)
from gaam_graph.grpo_schema import ActorRole
from gaam_graph.memory_schema import CurrentMemory
from gaam_graph.memory_weakness_book import MemoryWeaknessBook, WeaknessEntry, WeaknessStatus
from gaam_graph.oracle_graph_loader import OracleGraphLoader
from gaam_graph.question_schema import QuestionSet
from gaam_graph.reward_manager import RewardManager
from gaam_graph.reward_schema import MemoryRewardReport, QuestionAgentRewardReport

logger = logging.getLogger(__name__)

REAL_CASE_ID = "e47becba"


def load_oracle_graph_for_reward(graph_path: str | Path) -> OracleGraphLoader:
    """
    Load an oracle graph for teacher-side reward computation.

    Older real artifacts may contain benchmark metadata in ``graph`` fields.
    Reward computation needs the oracle evidence graph, not benchmark targets,
    so this loader sanitizes forbidden metadata before scoring and then still
    validates the graph shape.
    """
    oracle_graph = OracleGraphLoader(graph_path, strict=False, sanitize=True)
    oracle_graph.validate_no_leakage()
    oracle_graph.validate_integrity()

    sanitized_keys = getattr(oracle_graph, "_sanitized_keys", [])
    if sanitized_keys:
        logger.warning(
            "Sanitized forbidden oracle graph fields before reward computation: %s",
            ", ".join(sanitized_keys),
        )

    return oracle_graph


# ============================================================================
# No-Leakage Validation
# ============================================================================


def validate_reward_worker_input_no_leakage(input_data: RewardWorkerInput) -> None:
    """Validate reward worker input contains no forbidden fields."""
    input_dict = input_data.model_dump(mode="json")
    forbidden_paths = check_for_forbidden_fields(input_dict)

    if forbidden_paths:
        unique_paths = ", ".join(sorted(set(forbidden_paths)))
        raise ValueError(
            f"Reward worker input contains forbidden oracle/benchmark fields: {unique_paths}"
        )


def validate_replay_item_no_leakage(item: ReplayBufferItem) -> None:
    """Validate replay item contains no forbidden fields."""
    item_dict = item.model_dump(mode="json")
    forbidden_paths = check_for_forbidden_fields(item_dict)

    if forbidden_paths:
        unique_paths = ", ".join(sorted(set(forbidden_paths)))
        raise ValueError(
            f"Replay item contains forbidden oracle/benchmark fields: {unique_paths}"
        )


def sanitize_weakness_delta_for_actor_context(delta: WeaknessBookDelta) -> WeaknessBookDelta:
    """
    Sanitize weakness delta to ensure it's safe for actor context.

    Weakness deltas may contain oracle node IDs and failure tags (teacher-side diagnostics),
    but must not contain gold answers or oracle answer text.
    """
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            sanitized = {}
            for key, value in obj.items():
                if check_for_forbidden_fields({key: value}):
                    continue
                sanitized[key] = _sanitize(value)
            return sanitized
        if isinstance(obj, list):
            return [_sanitize(item) for item in obj]
        return obj

    delta.weakness_updates = [_sanitize(update) for update in delta.weakness_updates]

    return delta


# ============================================================================
# Input Loading
# ============================================================================


def load_reward_worker_input(path: Path) -> RewardWorkerInput:
    """Load RewardWorkerInput from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RewardWorkerInput.model_validate(data)


def build_reward_worker_input_from_job_dir(
    *,
    job_dir: Path,
    oracle_graph_dir: Path,
    output_dir: Path,
    record_id: str,
    round_id: int = 0,
    step_id: int = 0,
    reward_mode: str = "heuristic",
    compute_memory_builder_reward: bool = True,
    compute_question_agent_reward: bool = True,
    update_weakness_book: bool = True,
    write_replay_item: bool = True,
) -> RewardWorkerInput:
    """
    Build RewardWorkerInput from Phase 3 job directory.

    Discovers artifacts from standard job layout:
      jobs/<record_id>/
        dataproto/
        rollout_result.json
        current_memory.json (optional)
        questions.json (optional)
        answers.json (optional)
    """
    job_id = f"{record_id}_round{round_id}_step{step_id}"

    # Oracle graph path (required)
    oracle_graph_path = oracle_graph_dir / f"{record_id}.graph.json"
    if not oracle_graph_path.exists():
        raise FileNotFoundError(f"Oracle graph not found: {oracle_graph_path}")

    # Optional artifacts from job directory
    current_memory_path = job_dir / "current_memory.json"
    question_set_path = job_dir / "questions.json"
    answer_report_path = job_dir / "answers.json"
    trainer_adapter_report_path = job_dir / "trainer_adapter_report.json"
    weakness_book_path = job_dir / "weakness_book.json"

    return RewardWorkerInput(
        job_id=job_id,
        record_id=record_id,
        round_id=round_id,
        step_id=step_id,
        rollout_job_dir=str(job_dir),
        oracle_graph_path=str(oracle_graph_path),
        current_memory_path=str(current_memory_path) if current_memory_path.exists() else None,
        question_set_path=str(question_set_path) if question_set_path.exists() else None,
        answer_report_path=str(answer_report_path) if answer_report_path.exists() else None,
        trainer_adapter_report_path=str(trainer_adapter_report_path) if trainer_adapter_report_path.exists() else None,
        weakness_book_path=str(weakness_book_path) if weakness_book_path.exists() else None,
        output_dir=str(output_dir),
        reward_mode=reward_mode,
        compute_memory_builder_reward=compute_memory_builder_reward,
        compute_question_agent_reward=compute_question_agent_reward,
        update_weakness_book=update_weakness_book,
        write_replay_item=write_replay_item,
    )


# ============================================================================
# Artifact Loading
# ============================================================================


class RewardArtifacts(BaseModel):
    """Loaded reward artifacts."""

    model_config = {"arbitrary_types_allowed": True}

    oracle_graph: Any
    current_memory: CurrentMemory | None = None
    question_set: QuestionSet | None = None
    answer_reports: list[AnswerReport] = Field(default_factory=list)
    weakness_book: MemoryWeaknessBook | None = None


def load_reward_artifacts(input_data: RewardWorkerInput) -> RewardArtifacts:
    """Load all reward artifacts from input paths."""
    # Load oracle graph (required)
    oracle_graph = load_oracle_graph_for_reward(Path(input_data.oracle_graph_path))

    # Load optional artifacts
    current_memory = None
    if input_data.current_memory_path:
        try:
            with open(input_data.current_memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            current_memory = CurrentMemory.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load current memory: {e}")

    question_set = None
    if input_data.question_set_path:
        try:
            with open(input_data.question_set_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            question_set = QuestionSet.model_validate(data)
        except Exception as e:
            logger.warning(f"Failed to load question set: {e}")

    answer_reports: list[AnswerReport] = []
    if input_data.answer_report_path:
        try:
            with open(input_data.answer_report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "answers" in data:
                answer_reports = [
                    AnswerReport.model_validate(answer)
                    for answer in data.get("answers", [])
                ]
            elif isinstance(data, list):
                answer_reports = [
                    AnswerReport.model_validate(answer)
                    for answer in data
                ]
            elif isinstance(data, dict) and "question_items" in data:
                # Backward-compatible debug path: existing reward reports can
                # supply answer-like question items, but production inputs
                # should come from Frozen Answerer outputs.
                reward_report = MemoryRewardReport.model_validate(data)
                answer_reports = [
                    AnswerReport(
                        record_id=item.record_id,
                        question_id=item.question_id,
                        question="",
                        prediction=item.prediction,
                        answer_status=item.answer_status,
                        confidence=item.correctness,
                        supporting_memory_ids=item.supporting_memory_ids,
                        supporting_evidence=[],
                    )
                    for item in reward_report.question_items
                ]
            else:
                raise ValueError("Unsupported answer report format")
        except Exception as e:
            logger.warning(f"Failed to load answer report: {e}")

    weakness_book = None
    if input_data.weakness_book_path:
        try:
            weakness_book = MemoryWeaknessBook.load(Path(input_data.weakness_book_path))
        except Exception as e:
            logger.warning(f"Failed to load weakness book: {e}")

    # If no existing weakness book, create empty one
    if weakness_book is None:
        weakness_book = MemoryWeaknessBook(
            record_id=input_data.record_id,
            weaknesses=[],
        )

    return RewardArtifacts(
        oracle_graph=oracle_graph,
        current_memory=current_memory,
        question_set=question_set,
        answer_reports=answer_reports,
        weakness_book=weakness_book,
    )


# ============================================================================
# Weakness Book Delta Building
# ============================================================================


def build_weakness_delta(
    *,
    old_book: MemoryWeaknessBook,
    new_book: MemoryWeaknessBook,
    record_id: str,
    source_report_id: str,
    round_id: int,
    step_id: int,
) -> WeaknessBookDelta:
    """Build mergeable delta from old and new weakness books."""
    old_ids = {w.weakness_id for w in old_book.weaknesses}
    new_ids = {w.weakness_id for w in new_book.weaknesses}

    added_weakness_ids = list(new_ids - old_ids)
    updated_weakness_ids = []
    mastered_weakness_ids = []

    # Find updated and mastered weaknesses
    for weakness in new_book.weaknesses:
        if weakness.weakness_id in old_ids:
            old_weakness = next(w for w in old_book.weaknesses if w.weakness_id == weakness.weakness_id)
            if weakness.status == WeaknessStatus.MASTERED and old_weakness.status != WeaknessStatus.MASTERED:
                mastered_weakness_ids.append(weakness.weakness_id)
            elif weakness.frequency != old_weakness.frequency or weakness.last_seen_round != old_weakness.last_seen_round:
                updated_weakness_ids.append(weakness.weakness_id)

    # Build weakness updates (only for added and updated)
    weakness_updates = []
    for weakness_id in added_weakness_ids + updated_weakness_ids:
        weakness = next(w for w in new_book.weaknesses if w.weakness_id == weakness_id)
        weakness_updates.append(weakness.model_dump(mode="json"))

    delta_id = f"{record_id}_round{round_id}_step{step_id}_delta"

    return WeaknessBookDelta(
        delta_id=delta_id,
        record_id=record_id,
        source_report_id=source_report_id,
        round_id=round_id,
        step_id=step_id,
        added_weakness_ids=added_weakness_ids,
        updated_weakness_ids=updated_weakness_ids,
        mastered_weakness_ids=mastered_weakness_ids,
        weakness_updates=weakness_updates,
        created_at=datetime.utcnow().isoformat(),
    )


# ============================================================================
# Replay Item Building
# ============================================================================


def build_replay_item(
    *,
    input_data: RewardWorkerInput,
    memory_reward_path: str | None,
    question_reward_path: str | None,
    weakness_delta_path: str | None,
    memory_update_reward: float | None,
    question_agent_reward: float | None,
    monitoring_score: float | None,
    trainer_ready_batch_paths: dict[str, str],
    diagnostic_tags: list[str],
    selected_for_training: bool,
) -> ReplayBufferItem:
    """Build trainer-safe replay buffer item."""
    replay_id = f"{input_data.record_id}_round{input_data.round_id}_step{input_data.step_id}_replay"

    return ReplayBufferItem(
        replay_id=replay_id,
        record_id=input_data.record_id,
        job_id=input_data.job_id,
        round_id=input_data.round_id,
        step_id=input_data.step_id,
        memory_builder_checkpoint_id=None,  # Would be set by real trainer
        question_agent_checkpoint_id=None,  # Would be set by real trainer
        memory_reward=memory_update_reward,
        question_reward=question_agent_reward,
        monitoring_score=monitoring_score,
        memory_reward_report_path=memory_reward_path,
        question_reward_report_path=question_reward_path,
        trainer_ready_batch_paths=trainer_ready_batch_paths,
        weakness_delta_path=weakness_delta_path,
        diagnostic_tags=diagnostic_tags,
        no_leakage_passed=True,
        selected_for_training=selected_for_training,
        metadata=input_data.metadata,
        created_at=datetime.utcnow().isoformat(),
    )


# ============================================================================
# Main Reward Worker
# ============================================================================


def run_reward_worker_job_local(input_data: RewardWorkerInput) -> RewardWorkerReport:
    """
    Run reward worker job in local fallback mode.

    This is the main implementation that does not require Ray.
    """
    started_at = datetime.utcnow().isoformat()
    report_id = f"{input_data.record_id}_round{input_data.round_id}_step{input_data.step_id}_reward"

    output_dir = Path(input_data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    errors: list[str] = []

    try:
        # Validate no leakage
        validate_reward_worker_input_no_leakage(input_data)

        # Load artifacts
        artifacts = load_reward_artifacts(input_data)

        # Check for missing critical artifacts
        if artifacts.current_memory is None:
            warnings.append("Current memory not available")

        if not artifacts.answer_reports:
            warnings.append("Answer report not available")

        # Initialize reward manager
        reward_manager = RewardManager(
            correctness_mode=input_data.reward_mode,
            weakness_book=artifacts.weakness_book,
        )

        # Compute Memory Builder reward
        memory_update_reward = None
        memory_monitoring_score = None
        memory_reward_path = None

        memory_reward_report: MemoryRewardReport | None = None
        if input_data.compute_memory_builder_reward and artifacts.current_memory and artifacts.question_set and artifacts.answer_reports:
            try:
                # Convert to dict format expected by RewardManager
                current_memory_dict = artifacts.current_memory.model_dump(mode="json")
                questions_list = [q.model_dump(mode="json") for q in artifacts.question_set.questions]
                answer_reports_list = [
                    report.model_dump(mode="json")
                    for report in artifacts.answer_reports
                ]

                memory_reward_report = reward_manager.score_memory_builder(
                    current_memory=current_memory_dict,
                    questions=questions_list,
                    answer_reports=answer_reports_list,
                    oracle_graph=artifacts.oracle_graph,
                )
                memory_update_reward = memory_reward_report.update_reward
                memory_monitoring_score = memory_reward_report.monitoring_score

                # Write memory reward report
                memory_reward_path = str(output_dir / f"{input_data.record_id}.reward_report.json")
                with open(memory_reward_path, "w", encoding="utf-8") as f:
                    json.dump(memory_reward_report.model_dump(mode="json"), f, indent=2)
            except Exception as e:
                warnings.append(f"Memory Builder reward failed: {type(e).__name__}: {e}")
        elif input_data.compute_memory_builder_reward:
            warnings.append("Memory Builder reward skipped: missing artifacts")

        # Compute Question Agent reward
        question_agent_reward = None
        question_reward_path = None

        if input_data.compute_question_agent_reward and artifacts.question_set:
            if not artifacts.question_set.validity_reports:
                warnings.append("Question Agent reward skipped: missing validity reports")
            else:
                try:
                    question_reward_report: QuestionAgentRewardReport = reward_manager.score_question_agent(
                        questions=[
                            question.model_dump(mode="json")
                            for question in artifacts.question_set.questions
                        ],
                        validity_reports=[
                            report.model_dump(mode="json")
                            for report in artifacts.question_set.validity_reports
                        ],
                        answer_reports=[
                            report.model_dump(mode="json")
                            for report in artifacts.answer_reports
                        ],
                        oracle_graph=artifacts.oracle_graph,
                    )
                    question_agent_reward = question_reward_report.total_reward
                    question_reward_path = str(output_dir / f"{input_data.record_id}.question_agent_reward_report.json")
                    with open(question_reward_path, "w", encoding="utf-8") as f:
                        json.dump(question_reward_report.model_dump(mode="json"), f, indent=2)
                except Exception as e:
                    warnings.append(f"Question Agent reward failed: {type(e).__name__}: {e}")
        elif input_data.compute_question_agent_reward:
            warnings.append("Question Agent reward skipped: missing question set")

        # Update Weakness Book
        weakness_delta_path = None
        weakness_book_path = None
        weakness_update_count = 0
        failure_type_counts: dict[str, int] = {}

        if input_data.update_weakness_book and memory_reward_report:
            try:
                old_book = deepcopy(artifacts.weakness_book)

                # Update weakness book from memory reward report
                artifacts.weakness_book.update_from_reward_report(
                    memory_reward_report,
                    round_id=input_data.round_id,
                )

                new_book = artifacts.weakness_book

                # Build delta
                delta = build_weakness_delta(
                    old_book=old_book,
                    new_book=new_book,
                    record_id=input_data.record_id,
                    source_report_id=report_id,
                    round_id=input_data.round_id,
                    step_id=input_data.step_id,
                )

                # Sanitize delta
                delta = sanitize_weakness_delta_for_actor_context(delta)

                # Write delta
                weakness_delta_path = str(output_dir / f"{input_data.record_id}.weakness_delta.json")
                with open(weakness_delta_path, "w", encoding="utf-8") as f:
                    json.dump(delta.model_dump(mode="json"), f, indent=2)

                # Write full weakness book
                weakness_book_path = str(output_dir / f"{input_data.record_id}.weakness_book.json")
                artifacts.weakness_book.save(Path(weakness_book_path))

                weakness_update_count = len(delta.added_weakness_ids) + len(delta.updated_weakness_ids)

                # Count failure types
                for weakness in artifacts.weakness_book.weaknesses:
                    failure_type = weakness.failure_type.value if hasattr(weakness.failure_type, "value") else str(weakness.failure_type)
                    failure_type_counts[failure_type] = failure_type_counts.get(failure_type, 0) + 1
            except Exception as e:
                warnings.append(f"Weakness Book update failed: {type(e).__name__}: {e}")
        elif input_data.update_weakness_book:
            warnings.append("Weakness Book update skipped: no memory reward report")

        # Build replay item
        replay_item_path = None

        if input_data.write_replay_item:
            try:
                # Find trainer ready batch paths
                trainer_ready_batch_paths = {}
                if input_data.rollout_job_dir:
                    dataproto_dir = Path(input_data.rollout_job_dir) / "dataproto"
                    if dataproto_dir.exists():
                        for actor_role in [ActorRole.MEMORY_BUILDER, ActorRole.QUESTION_AGENT]:
                            batch_files = list(dataproto_dir.glob(f"*.{actor_role.value}.local_dataproto.json"))
                            if batch_files:
                                trainer_ready_batch_paths[actor_role.value] = str(batch_files[0])

                # Diagnostic tags
                diagnostic_tags = []
                if memory_update_reward is not None and memory_update_reward < 0.5:
                    diagnostic_tags.append("low_memory_reward")
                if weakness_update_count > 0:
                    diagnostic_tags.append("weakness_updated")
                if memory_update_reward is None:
                    diagnostic_tags.append("missing_memory_reward")

                replay_item = build_replay_item(
                    input_data=input_data,
                    memory_reward_path=memory_reward_path,
                    question_reward_path=question_reward_path,
                    weakness_delta_path=weakness_delta_path,
                    memory_update_reward=memory_update_reward,
                    question_agent_reward=question_agent_reward,
                    monitoring_score=memory_monitoring_score,
                    trainer_ready_batch_paths=trainer_ready_batch_paths,
                    diagnostic_tags=diagnostic_tags,
                    selected_for_training=memory_update_reward is not None,
                )

                # Validate no leakage
                validate_replay_item_no_leakage(replay_item)

                # Write replay item
                replay_item_path = str(output_dir / f"{input_data.record_id}.replay_item.json")
                with open(replay_item_path, "w", encoding="utf-8") as f:
                    json.dump(replay_item.model_dump(mode="json"), f, indent=2)
            except Exception as e:
                warnings.append(f"Replay item creation failed: {type(e).__name__}: {e}")

        # Determine status
        status = RewardWorkerStatus.SUCCEEDED
        if memory_update_reward is None and input_data.compute_memory_builder_reward:
            status = RewardWorkerStatus.PARTIAL
        if question_agent_reward is None and input_data.compute_question_agent_reward:
            status = RewardWorkerStatus.PARTIAL

        # Build report
        finished_at = datetime.utcnow().isoformat()

        report = RewardWorkerReport(
            report_id=report_id,
            status=status,
            record_id=input_data.record_id,
            job_id=input_data.job_id,
            round_id=input_data.round_id,
            step_id=input_data.step_id,
            output_dir=str(output_dir),
            memory_reward_path=memory_reward_path,
            question_reward_path=question_reward_path,
            weakness_book_path=weakness_book_path,
            weakness_delta_path=weakness_delta_path,
            replay_item_path=replay_item_path,
            memory_update_reward=memory_update_reward,
            memory_monitoring_score=memory_monitoring_score,
            question_agent_reward=question_agent_reward,
            weakness_update_count=weakness_update_count,
            failure_type_counts=failure_type_counts,
            warnings=warnings,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

        # Write report
        report_path = output_dir / "reward_worker_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2)

        return report

    except Exception as e:
        finished_at = datetime.utcnow().isoformat()
        error_msg = f"{type(e).__name__}: {e}"
        error_trace = traceback.format_exc()

        errors.append(error_msg)

        report = RewardWorkerReport(
            report_id=report_id,
            status=RewardWorkerStatus.FAILED,
            record_id=input_data.record_id,
            job_id=input_data.job_id,
            round_id=input_data.round_id,
            step_id=input_data.step_id,
            output_dir=str(output_dir),
            warnings=warnings,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

        # Write report
        report_path = output_dir / "reward_worker_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2)

        logger.error(f"Reward worker failed: {error_msg}\n{error_trace}")

        return report


def run_reward_worker_job(input_data: RewardWorkerInput) -> RewardWorkerReport:
    """
    Run reward worker job (dispatches to local or Ray).

    For Milestone 5 MVP, always uses local fallback.
    """
    return run_reward_worker_job_local(input_data)


# Optional Ray wrapper (not required for MVP)
def run_reward_worker_job_ray(input_data: RewardWorkerInput) -> RewardWorkerReport:
    """
    Run reward worker job with Ray (optional).

    For Milestone 5 MVP, this is not implemented.
    Falls back to local execution.
    """
    try:
        import ray

        @ray.remote(num_cpus=1)
        def _run_reward_worker_remote(payload: dict) -> dict:
            input_obj = RewardWorkerInput.model_validate(payload)
            report = run_reward_worker_job_local(input_obj)
            return report.model_dump(mode="json")

        # Submit remote task
        payload = input_data.model_dump(mode="json")
        result_ref = _run_reward_worker_remote.remote(payload)
        result_dict = ray.get(result_ref)

        return RewardWorkerReport.model_validate(result_dict)

    except ImportError:
        # Ray not available, fall back to local
        return run_reward_worker_job_local(input_data)
