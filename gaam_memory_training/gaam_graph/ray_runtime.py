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
from gaam_graph.grpo_schema import ActorRole, ActorUpdateBatch
from gaam_graph.local_dataproto_adapter import export_local_dataproto_batches

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

        # For Milestone 3 MVP: Create stub update batches
        # In a real implementation, this would:
        # 1. Load the LongMemEval record
        # 2. Load oracle graph for reward/question validation only
        # 3. Run policy rollout (dry-run or no-llm mode)
        # 4. Compute rewards
        # 5. Build ActorUpdateBatch artifacts

        # Create stub Memory Builder batch
        memory_batch = ActorUpdateBatch(
            batch_id=f"{record_id}.memory_batch",
            role=ActorRole.MEMORY_BUILDER,
            round_id=0,
            step_id=0,
            items=[],
            metadata={
                "worker_id": payload.worker_id,
                "dry_run": payload.dry_run,
                "no_llm": payload.no_llm,
                "stub_batch": True,
            },
        )

        memory_batch_path = update_batches_dir / f"{record_id}.memory_update_batch.json"
        with open(memory_batch_path, "w", encoding="utf-8") as f:
            json.dump(memory_batch.model_dump(mode="json"), f, indent=2)

        # Create stub Question Agent batch
        question_batch = ActorUpdateBatch(
            batch_id=f"{record_id}.question_batch",
            role=ActorRole.QUESTION_AGENT,
            round_id=0,
            step_id=0,
            items=[],
            metadata={
                "worker_id": payload.worker_id,
                "dry_run": payload.dry_run,
                "no_llm": payload.no_llm,
                "stub_batch": True,
            },
        )

        question_batch_path = update_batches_dir / f"{record_id}.question_update_batch.json"
        with open(question_batch_path, "w", encoding="utf-8") as f:
            json.dump(question_batch.model_dump(mode="json"), f, indent=2)

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
                "stub_batch": True,
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
