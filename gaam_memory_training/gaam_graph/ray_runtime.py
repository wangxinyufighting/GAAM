"""
Phase 3 Milestone 3: Ray Worker Prototype

This module provides Ray-compatible distributed execution for GAAM rollout jobs.
Supports two modes:
1. Local Ray execution (if Ray is available)
2. Local fallback execution (if Ray is unavailable)

The worker boundary is the key feature. Workers call existing local components
and write Phase 2-compatible artifacts.
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from gaam_graph.code_a1_alignment import check_for_forbidden_fields
from gaam_graph.distributed_runtime import _find_forbidden_runtime_keys
from gaam_graph.distributed_runtime_schema import (
    Phase3Backend,
    Phase3JobStatus,
    Phase3RolloutJob,
    Phase3RuntimeConfig,
    Phase3WorkerPlan,
)
from gaam_graph.grpo_schema import ActorRole, ActorUpdateBatch, ActorUpdateItem
from gaam_graph.lme_loader import LMEEvent, LMERecord, LongMemEvalLoader
from gaam_graph.local_dataproto_adapter import export_local_dataproto_batches
from gaam_graph.utils import normalize_text

# Real case ID for testing
REAL_CASE_ID = "e47becba"


def is_ray_available() -> bool:
    """Check if Ray is available."""
    try:
        import ray  # noqa: F401
        return True
    except ImportError:
        return False


def _validate_worker_payload_no_leakage(payload: "RayRolloutWorkerInput") -> None:
    """Validate actor-facing worker payload fields before any rollout artifacts are written."""
    memory_builder_payload = {
        "memory_builder": payload.config.memory_builder.model_dump(mode="json"),
    }
    forbidden_paths = []
    forbidden_paths.extend(check_for_forbidden_fields(memory_builder_payload))
    forbidden_paths.extend(
        _find_forbidden_runtime_keys(memory_builder_payload, "runtime_config")
    )

    if forbidden_paths:
        unique_paths = ", ".join(sorted(set(forbidden_paths)))
        raise ValueError(
            f"Worker payload contains forbidden Memory Builder teacher/oracle fields: {unique_paths}"
        )


def _load_worker_record(payload: "RayRolloutWorkerInput", record_id: str) -> LMERecord | None:
    """Load the LongMemEval record for a worker job without exposing target QA."""
    try:
        records = LongMemEvalLoader(str(payload.config.input_records_path)).load()
    except Exception as exc:
        logging.warning("Failed to load input records for %s: %s", record_id, exc)
        return None

    for record in records:
        if record.record_id == record_id:
            return record

    logging.warning("Record %s was not found in %s", record_id, payload.config.input_records_path)
    return None


def _event_line(event: LMEEvent) -> str:
    session = normalize_text(event.session_id)
    speaker = normalize_text(event.speaker)
    text = normalize_text(event.text)
    timestamp = f" time={normalize_text(event.timestamp)}" if event.timestamp else ""
    return f"[session={session} turn={event.turn_id}{timestamp}] {speaker}: {text}"


def _select_history_events(record: LMERecord | None, max_events: int = 24) -> list[LMEEvent]:
    """Select a compact, multi-session history window for actor-facing rollout samples."""
    if record is None or not record.events:
        return []

    by_session: dict[str, list[LMEEvent]] = {}
    for event in record.events:
        by_session.setdefault(event.session_id, []).append(event)

    selected: list[LMEEvent] = []
    sessions = list(by_session.values())

    # Keep early evidence from multiple sessions, then fill with late evidence.
    for session_events in sessions:
        selected.extend(session_events[:2])
        if len(selected) >= max_events:
            return selected[:max_events]

    for session_events in sessions:
        selected.extend(session_events[-2:])
        if len(selected) >= max_events:
            return selected[:max_events]

    return selected[:max_events]


def _history_excerpt(record: LMERecord | None, max_events: int = 24) -> str:
    events = _select_history_events(record, max_events=max_events)
    if not events:
        return "No conversation events were available for this record."
    return "\n".join(_event_line(event) for event in events)


def _memory_builder_response(record: LMERecord | None) -> str:
    """Create a safe Current Memory candidate from raw conversation events."""
    events = _select_history_events(record, max_events=18)
    if not events:
        return (
            "Summary:\nNo usable conversation history was available.\n\n"
            "Memory graph:\n- node: unknown_context | type: event | evidence: unavailable\n\n"
            "Abstractions:\n- Insufficient evidence to infer stable user facts or preferences."
        )

    session_ids = []
    for event in events:
        if event.session_id not in session_ids:
            session_ids.append(event.session_id)

    salient = events[:8]
    nodes = [
        f"- node: {event.event_id} | type: event | session: {event.session_id} | text: {normalize_text(event.text)}"
        for event in salient
    ]
    edges = []
    for prev, cur in zip(salient, salient[1:]):
        relation = "same_session_next" if prev.session_id == cur.session_id else "cross_session_context"
        edges.append(f"- edge: {prev.event_id} -> {cur.event_id} | relation: {relation}")

    summary_parts = [
        f"The record contains evidence from {len(session_ids)} session(s).",
        "The memory keeps concrete events first and leaves task-specific answers out.",
    ]

    return "\n".join(
        [
            "Summary:",
            " ".join(summary_parts),
            "",
            "Memory graph:",
            *nodes,
            "",
            "Relations:",
            *(edges or ["- no explicit relation inferred"]),
            "",
            "Abstractions:",
            "- Preserve stable user facts, preferences, updates, and cross-session links when evidence appears repeatedly.",
            "- Avoid adding conclusions that are not supported by the observed conversation events.",
        ]
    )


def _question_agent_response(record: LMERecord | None) -> str:
    """Create broad diagnostic questions without using the benchmark target question."""
    events = _select_history_events(record, max_events=12)
    sessions = []
    for event in events:
        if event.session_id not in sessions:
            sessions.append(event.session_id)

    if not events:
        return "\n".join(
            [
                "1. What stable facts can be recovered from the available conversation history?",
                "2. Which information should remain uncertain because the evidence is missing?",
            ]
        )

    first = normalize_text(events[0].text)
    last = normalize_text(events[-1].text)
    multi_session_hint = (
        "across multiple sessions" if len(sessions) > 1 else "within the available session"
    )
    return "\n".join(
        [
            f"1. What personal fact or preference is supported by the evidence {multi_session_hint}?",
            f"2. How did the user's context change between the earlier event '{first}' and the later event '{last}'?",
            "3. Which event should the memory retain because it may affect future personalization?",
        ]
    )


def _build_real_actor_batches(
    *,
    payload: "RayRolloutWorkerInput",
    record: LMERecord | None,
) -> tuple[ActorUpdateBatch, ActorUpdateBatch]:
    """Build non-empty actor update batches from safe rollout artifacts."""
    job = payload.job
    record_id = job.record_id
    excerpt = _history_excerpt(record)
    round_id = job.round_id
    step_id = job.step_id

    memory_prompt = "\n".join(
        [
            "Build a compact Current Memory from the conversation events below.",
            "Use graph-like nodes, a concise summary, and high-level abstractions.",
            "Use only the conversation events below.",
            "",
            "Conversation events:",
            excerpt,
        ]
    )
    question_prompt = "\n".join(
        [
            "Generate broad diagnostic questions that test whether a memory preserves useful information.",
            "Cover single-hop, multi-hop, temporal, preference, personal fact, update, and abstraction needs when evidence supports them.",
            "Use only the conversation evidence below.",
            "",
            "Conversation evidence:",
            excerpt,
        ]
    )

    memory_items = [
        ActorUpdateItem(
            sample_id=f"{record_id}.memory.compact",
            group_id=f"{record_id}.memory_group",
            record_id=record_id,
            role=ActorRole.MEMORY_BUILDER,
            prompt=memory_prompt,
            response=_memory_builder_response(record),
            reward=0.62,
            advantage=0.12,
            selected_for_update=True,
            metadata={
                "round_id": round_id,
                "step_id": step_id,
                "sample_kind": "history_grounded_current_memory",
            },
        ),
        ActorUpdateItem(
            sample_id=f"{record_id}.memory.minimal",
            group_id=f"{record_id}.memory_group",
            record_id=record_id,
            role=ActorRole.MEMORY_BUILDER,
            prompt=memory_prompt,
            response="Summary:\nOnly a minimal memory was built from the visible conversation events.\n\nAbstractions:\n- Important evidence may be under-covered.",
            reward=0.42,
            advantage=-0.12,
            selected_for_update=True,
            metadata={
                "round_id": round_id,
                "step_id": step_id,
                "sample_kind": "low_coverage_current_memory",
            },
        ),
    ]
    question_items = [
        ActorUpdateItem(
            sample_id=f"{record_id}.question.coverage",
            group_id=f"{record_id}.question_group",
            record_id=record_id,
            role=ActorRole.QUESTION_AGENT,
            prompt=question_prompt,
            response=_question_agent_response(record),
            reward=0.64,
            advantage=0.10,
            selected_for_update=True,
            metadata={
                "round_id": round_id,
                "step_id": step_id,
                "sample_kind": "coverage_oriented_questions",
            },
        ),
        ActorUpdateItem(
            sample_id=f"{record_id}.question.generic",
            group_id=f"{record_id}.question_group",
            record_id=record_id,
            role=ActorRole.QUESTION_AGENT,
            prompt=question_prompt,
            response="1. What should be remembered from this conversation?\n2. What changed over time?",
            reward=0.46,
            advantage=-0.10,
            selected_for_update=True,
            metadata={
                "round_id": round_id,
                "step_id": step_id,
                "sample_kind": "generic_questions",
            },
        ),
    ]

    memory_batch = ActorUpdateBatch(
        batch_id=f"{record_id}.memory_batch.r{round_id}.s{step_id}",
        role=ActorRole.MEMORY_BUILDER,
        round_id=round_id,
        step_id=step_id,
        items=memory_items,
        metadata={
            "worker_id": payload.worker_id,
            "dry_run": payload.dry_run,
            "no_llm": payload.no_llm,
            "source": "history_grounded_worker_rollout",
            "num_events_available": len(record.events) if record else 0,
        },
    )

    question_batch = ActorUpdateBatch(
        batch_id=f"{record_id}.question_batch.r{round_id}.s{step_id}",
        role=ActorRole.QUESTION_AGENT,
        round_id=round_id,
        step_id=step_id,
        items=question_items,
        metadata={
            "worker_id": payload.worker_id,
            "dry_run": payload.dry_run,
            "no_llm": payload.no_llm,
            "source": "history_grounded_worker_rollout",
            "num_events_available": len(record.events) if record else 0,
        },
    )

    return memory_batch, question_batch


# ============================================================================
# Worker Payload Schemas
# ============================================================================


class RayRolloutWorkerInput(BaseModel):
    """Input payload for a Ray rollout worker."""

    runtime_id: str
    worker_id: str
    job: Phase3RolloutJob
    config: Phase3RuntimeConfig
    output_dir: Path
    dry_run: bool = True
    no_llm: bool = True
    tokenizer_path: str | None = None
    max_prompt_length: int | None = None
    max_response_length: int | None = None
    truncation: str = "error"
    write_torch_tensors: bool = False


class RayRolloutWorkerResult(BaseModel):
    """Result from a Ray rollout worker."""

    runtime_id: str
    worker_id: str
    job_id: str
    record_id: str
    status: Phase3JobStatus
    output_dir: str
    rollout_trace_path: str | None = None
    memory_update_batch_path: str | None = None
    question_update_batch_path: str | None = None
    dataproto_manifest_path: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Ray Runtime Configuration
# ============================================================================


class RayRuntimeConfig(BaseModel):
    """Configuration for Ray runtime execution."""

    require_ray: bool = False
    ray_address: str | None = None
    local_mode: bool = False
    num_cpus: int | None = None
    num_gpus: float | None = None
    job_timeout_seconds: int | None = None


# ============================================================================
# Worker Implementation (Pure Local)
# ============================================================================


def run_rollout_worker_job(payload: RayRolloutWorkerInput) -> RayRolloutWorkerResult:
    """
    Pure local implementation of rollout worker.

    This function is called by both Ray workers and fallback executor.
    It produces Phase 2-compatible artifacts and exports local DataProto.
    """
    started_at = datetime.utcnow().isoformat()
    job = payload.job
    record_id = job.record_id

    # Create job output directory
    job_output_dir = Path(payload.output_dir) / "jobs" / record_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    # Create update_batches directory
    update_batches_dir = job_output_dir / "update_batches"
    update_batches_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Write job input
        job_input_path = job_output_dir / "job_input.json"
        with open(job_input_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "runtime_id": payload.runtime_id,
                    "worker_id": payload.worker_id,
                    "job": job.model_dump(mode="json"),
                    "dry_run": payload.dry_run,
                    "no_llm": payload.no_llm,
                },
                f,
                indent=2,
            )

        _validate_worker_payload_no_leakage(payload)

        record = _load_worker_record(payload, record_id)
        memory_batch, question_batch = _build_real_actor_batches(
            payload=payload,
            record=record,
        )

        memory_batch_path = update_batches_dir / f"{record_id}.memory_update_batch.json"
        with open(memory_batch_path, "w", encoding="utf-8") as f:
            json.dump(memory_batch.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        question_batch_path = update_batches_dir / f"{record_id}.question_update_batch.json"
        with open(question_batch_path, "w", encoding="utf-8") as f:
            json.dump(question_batch.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        # Export local DataProto
        dataproto_dir = job_output_dir / "dataproto"

        # Load tokenizer if specified
        tokenizer = None
        if payload.tokenizer_path:
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    payload.tokenizer_path,
                    trust_remote_code=False,
                )
            except ImportError:
                logging.warning("transformers not installed, using fallback tokenizer")
                tokenizer = None
            except Exception as e:
                logging.warning(f"Failed to load tokenizer: {e}, using fallback")
                tokenizer = None

        export_local_dataproto_batches(
            rollout_dir=job_output_dir,
            output_dir=dataproto_dir,
            record_id=record_id,
            tokenizer=tokenizer,
            max_prompt_length=payload.max_prompt_length,
            max_response_length=payload.max_response_length,
            truncation=payload.truncation,
            write_torch_tensors=payload.write_torch_tensors,
            include_unselected=False,
        )
        dataproto_manifest_path = str(dataproto_dir / "manifest.jsonl")

        # Write job result
        finished_at = datetime.utcnow().isoformat()
        result = RayRolloutWorkerResult(
            runtime_id=payload.runtime_id,
            worker_id=payload.worker_id,
            job_id=job.job_id,
            record_id=record_id,
            status=Phase3JobStatus.SUCCEEDED,
            output_dir=str(job_output_dir),
            rollout_trace_path=None,  # Would be written by real rollout
            memory_update_batch_path=str(memory_batch_path),
            question_update_batch_path=str(question_batch_path),
            dataproto_manifest_path=dataproto_manifest_path,
            started_at=started_at,
            finished_at=finished_at,
            error=None,
            metrics={
                "dry_run": payload.dry_run,
                "no_llm": payload.no_llm,
                "stub_batch": False,
                "memory_update_items": len(memory_batch.items),
                "question_update_items": len(question_batch.items),
                "record_loaded": record is not None,
            },
        )

        # Write job result
        job_result_path = job_output_dir / "job_result.json"
        with open(job_result_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2)

        return result

    except Exception as e:
        finished_at = datetime.utcnow().isoformat()
        error_msg = f"{type(e).__name__}: {e}"
        error_trace = traceback.format_exc()

        result = RayRolloutWorkerResult(
            runtime_id=payload.runtime_id,
            worker_id=payload.worker_id,
            job_id=job.job_id,
            record_id=record_id,
            status=Phase3JobStatus.FAILED,
            output_dir=str(job_output_dir),
            rollout_trace_path=None,
            memory_update_batch_path=None,
            question_update_batch_path=None,
            dataproto_manifest_path=None,
            started_at=started_at,
            finished_at=finished_at,
            error=error_msg,
            metrics={"error_trace": error_trace},
        )

        # Write job result even on failure
        job_result_path = job_output_dir / "job_result.json"
        with open(job_result_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2)

        return result


# ============================================================================
# Ray Worker Wrapper (Optional)
# ============================================================================

if is_ray_available():
    import ray

    @ray.remote
    class RayRolloutWorker:
        """Ray remote worker wrapper."""

        def run_job(self, payload_dict: dict[str, Any]) -> dict[str, Any]:
            """Run rollout job and return result."""
            payload = RayRolloutWorkerInput.model_validate(payload_dict)
            result = run_rollout_worker_job(payload)
            return result.model_dump(mode="json")


# ============================================================================
# Ray Runtime Executor
# ============================================================================


class RayRuntimeExecutor:
    """
    Executor for Ray-based or fallback distributed execution.

    Dispatches Phase3RolloutJobs to workers and collects results.
    """

    def __init__(
        self,
        config: Phase3RuntimeConfig,
        worker_plan: Phase3WorkerPlan,
        ray_config: RayRuntimeConfig,
        tokenizer_path: str | None = None,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
        truncation: str = "error",
        write_torch_tensors: bool = False,
    ):
        self.config = config
        self.worker_plan = worker_plan
        self.ray_config = ray_config
        self.tokenizer_path = tokenizer_path
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.truncation = truncation
        self.write_torch_tensors = write_torch_tensors

        self.ray_initialized = False
        self.use_ray: bool | None = None

    def _initialize_ray(self) -> bool:
        """Initialize Ray if available and requested."""
        # If require_ray is False and backend preference is local, skip Ray
        if not self.ray_config.require_ray and self.config.backend == Phase3Backend.LOCAL_FAKE_DISTRIBUTED:
            return False

        if not is_ray_available():
            if self.ray_config.require_ray:
                raise RuntimeError(
                    "Ray is required (--require_ray) but not available. "
                    "Install with: pip install ray>=2.9"
                )
            return False

        try:
            import ray

            # Initialize Ray
            init_kwargs = {
                "ignore_reinit_error": True,
            }

            if self.ray_config.ray_address:
                init_kwargs["address"] = self.ray_config.ray_address
            if self.ray_config.local_mode:
                init_kwargs["local_mode"] = True
            if self.ray_config.num_cpus is not None:
                init_kwargs["num_cpus"] = self.ray_config.num_cpus
            if self.ray_config.num_gpus is not None:
                init_kwargs["num_gpus"] = self.ray_config.num_gpus

            ray.init(**init_kwargs)
            self.ray_initialized = True
            return True

        except Exception as e:
            if self.ray_config.require_ray:
                raise RuntimeError(f"Failed to initialize Ray: {e}") from e
            logging.warning(f"Ray initialization failed: {e}")
            return False

    def run(self) -> list[RayRolloutWorkerResult]:
        """
        Execute all jobs in the worker plan.

        Returns list of results in the same order as worker_plan.jobs.
        """
        if self.use_ray is None:
            self.use_ray = self._initialize_ray()

        # Check if there are any jobs
        if not self.worker_plan.jobs:
            # No jobs to run - write empty manifest and summary
            self._write_manifest([])
            self._write_summary([])
            self._write_trace([])
            return []

        # Dispatch jobs
        if self.use_ray:
            results = self._run_with_ray()
        else:
            results = self._run_fallback()

        # Write manifest and summary
        self._write_manifest(results)
        self._write_summary(results)
        self._write_trace(results)

        return results

    def _run_with_ray(self) -> list[RayRolloutWorkerResult]:
        """Execute jobs with Ray remote workers."""
        import ray

        # Create worker pool (filter to rollout workers only)
        rollout_workers = [
            w for w in self.worker_plan.workers
            if w.role == "memory_builder" or w.role == "question_agent"
        ]
        num_workers = len(rollout_workers) if rollout_workers else len(self.worker_plan.workers)
        worker_pool = [RayRolloutWorker.remote() for _ in range(max(1, num_workers))]

        # Build worker payloads
        payloads = []
        for job in self.worker_plan.jobs:
            # Find assigned worker
            worker_id = job.assigned_rollout_worker
            worker_idx = next(
                (
                    i
                    for i, w in enumerate(self.worker_plan.workers)
                    if w.worker_id == worker_id
                ),
                0,
            )

            payload = RayRolloutWorkerInput(
                runtime_id=self.config.runtime_id,
                worker_id=worker_id,
                job=job,
                config=self.config,
                output_dir=self.config.output_dir,
                dry_run=True,  # Milestone 3 MVP: dry-run only
                no_llm=True,  # Milestone 3 MVP: no LLM calls
                tokenizer_path=self.tokenizer_path,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                truncation=self.truncation,
                write_torch_tensors=self.write_torch_tensors,
            )

            payloads.append((worker_idx % len(worker_pool), payload))

        # Dispatch jobs
        ray_refs = []
        for worker_idx, payload in payloads:
            worker = worker_pool[worker_idx]
            ref = worker.run_job.remote(payload.model_dump(mode="json"))
            ray_refs.append(ref)

        # Collect results
        result_dicts = ray.get(ray_refs, timeout=self.ray_config.job_timeout_seconds)
        results = [
            RayRolloutWorkerResult.model_validate(r) for r in result_dicts
        ]

        return results

    def _run_fallback(self) -> list[RayRolloutWorkerResult]:
        """Execute jobs with local fallback (no Ray)."""
        results = []

        for job in self.worker_plan.jobs:
            payload = RayRolloutWorkerInput(
                runtime_id=self.config.runtime_id,
                worker_id=job.assigned_rollout_worker,
                job=job,
                config=self.config,
                output_dir=self.config.output_dir,
                dry_run=True,
                no_llm=True,
                tokenizer_path=self.tokenizer_path,
                max_prompt_length=self.max_prompt_length,
                max_response_length=self.max_response_length,
                truncation=self.truncation,
                write_torch_tensors=self.write_torch_tensors,
            )

            result = run_rollout_worker_job(payload)
            results.append(result)

        return results

    def _write_manifest(self, results: list[RayRolloutWorkerResult]) -> None:
        """Write ray_runtime_manifest.jsonl."""
        manifest_path = Path(self.config.output_dir) / "ray_runtime_manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for result in results:
                entry = {
                    "runtime_id": result.runtime_id,
                    "job_id": result.job_id,
                    "record_id": result.record_id,
                    "worker_id": result.worker_id,
                    "status": result.status.value,
                    "job_result_path": f"jobs/{result.record_id}/job_result.json",
                    "dataproto_manifest_path": result.dataproto_manifest_path,
                }
                f.write(json.dumps(entry) + "\n")

    def _write_summary(self, results: list[RayRolloutWorkerResult]) -> None:
        """Write ray_runtime_summary.json."""
        succeeded = sum(1 for r in results if r.status == Phase3JobStatus.SUCCEEDED)
        failed = sum(1 for r in results if r.status == Phase3JobStatus.FAILED)

        summary = {
            "runtime_id": self.config.runtime_id,
            "backend": "ray" if self.use_ray else "local_fallback",
            "ray_available": is_ray_available(),
            "num_jobs": len(results),
            "succeeded_jobs": succeeded,
            "failed_jobs": failed,
            "records": [r.record_id for r in results],
            "output_dir": str(self.config.output_dir),
            "strict_no_leakage": True,
            "dataproto_exported": any(r.dataproto_manifest_path for r in results),
        }

        summary_path = Path(self.config.output_dir) / "ray_runtime_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _write_trace(self, results: list[RayRolloutWorkerResult]) -> None:
        """Write ray_runtime_trace.jsonl."""
        trace_path = Path(self.config.output_dir) / "ray_runtime_trace.jsonl"
        with open(trace_path, "w", encoding="utf-8") as f:
            for result in results:
                trace_entry = {
                    "runtime_id": result.runtime_id,
                    "job_id": result.job_id,
                    "record_id": result.record_id,
                    "worker_id": result.worker_id,
                    "status": result.status.value,
                    "started_at": result.started_at,
                    "finished_at": result.finished_at,
                    "error": result.error,
                    "metrics": result.metrics,
                }
                f.write(json.dumps(trace_entry) + "\n")
