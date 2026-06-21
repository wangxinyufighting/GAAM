"""
Phase 3 Milestone 1: Distributed Runtime.

This module provides the local fake-distributed runtime that simulates
distributed execution without requiring Ray, verl, or GPUs.

Design principles:
1. Plan worker assignments and job distribution deterministically
2. Validate inputs and checkpoint resolution
3. Write Phase 2-compatible artifacts
4. Enforce no-leakage boundaries
5. Keep actual model execution optional (dry-run by default in Milestone 1)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gaam_graph.distributed_runtime_schema import (
    Phase3RuntimeConfig,
    Phase3WorkerPlan,
    Phase3WorkerSpec,
    Phase3RolloutJob,
    Phase3RuntimeTrace,
    Phase3ReadinessReport,
    Phase3WorkerRole,
    Phase3Backend,
    Phase3JobStatus,
)
from gaam_graph.code_a1_alignment import (
    AlignmentIssue,
    AlignmentIssueSeverity,
    check_for_forbidden_fields,
)
from gaam_graph.grpo_schema import ActorRole


RECORD_ID_KEYS = ("record_id", "question_id", "id", "case_id", "sample_id")
MEMORY_BUILDER_FORBIDDEN_RUNTIME_KEYS = {
    "oracle_graph_dir",
    "oracle_graph_path",
    "oracle_graph",
    "benchmark_question_path",
    "benchmark_question",
    "benchmark_target_question",
    "gold_answer",
    "expected_answer",
    "answer_reports",
    "reward_report",
    "raw_history",
}


def _find_forbidden_runtime_keys(data: Any, path: str = "root") -> list[str]:
    """Find runtime-only leakage keys in nested dict/list config payloads."""
    found = []
    if isinstance(data, dict):
        for key, value in data.items():
            child_path = f"{path}.{key}"
            if key in MEMORY_BUILDER_FORBIDDEN_RUNTIME_KEYS and value is not None:
                found.append(child_path)
            found.extend(_find_forbidden_runtime_keys(value, child_path))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            found.extend(_find_forbidden_runtime_keys(item, f"{path}[{idx}]"))
    elif isinstance(data, str):
        stripped = data.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, (dict, list)):
                found.extend(_find_forbidden_runtime_keys(parsed, path))
    return found


# ============================================================================
# Runtime Planner
# ============================================================================

class Phase3RuntimePlanner:
    """
    Plans worker assignments and job distribution for Phase 3 runtime.

    The planner:
    1. Loads input records
    2. Validates oracle graph availability
    3. Resolves actor checkpoints from registry
    4. Creates logical worker specs
    5. Assigns jobs round-robin to workers
    """

    def __init__(self, config: Phase3RuntimeConfig) -> None:
        self.config = config
        self.issues: list[AlignmentIssue] = []

    def build_worker_plan(self) -> Phase3WorkerPlan:
        """
        Build complete worker and job assignment plan.

        Returns:
            Worker plan with logical workers and job assignments
        """
        self.issues = []

        # Create worker specs
        workers = self._create_worker_specs()

        # Load and assign jobs
        jobs = self._create_and_assign_jobs(workers)

        return Phase3WorkerPlan(
            runtime_id=self.config.runtime_id,
            backend=self.config.backend,
            workers=workers,
            jobs=jobs,
        )

    def _create_worker_specs(self) -> list[Phase3WorkerSpec]:
        """Create logical worker specifications."""
        workers = []

        # Create rollout workers
        for i in range(self.config.num_rollout_workers):
            workers.append(Phase3WorkerSpec(
                worker_id=f"rollout_worker_{i}",
                role=Phase3WorkerRole.MEMORY_BUILDER,  # Rollout workers handle both actors
                backend=self.config.backend,
                resource_config=self.config.memory_builder.resource_config,
            ))

        # Create reward workers
        for i in range(self.config.num_reward_workers):
            workers.append(Phase3WorkerSpec(
                worker_id=f"reward_worker_{i}",
                role=Phase3WorkerRole.REWARD,
                backend=self.config.backend,
                resource_config=self.config.reward_config.get("resource_config", {}),
            ))

        # Create planner worker (self)
        workers.append(Phase3WorkerSpec(
            worker_id="planner",
            role=Phase3WorkerRole.PLANNER,
            backend=self.config.backend,
            resource_config={},
        ))

        return workers

    def _create_and_assign_jobs(self, workers: list[Phase3WorkerSpec]) -> list[Phase3RolloutJob]:
        """Load records and assign jobs to workers."""
        # Load record IDs
        record_ids = self._load_record_ids()

        # Get worker lists
        rollout_workers = [w for w in workers if w.role == Phase3WorkerRole.MEMORY_BUILDER]
        reward_workers = [w for w in workers if w.role == Phase3WorkerRole.REWARD]

        if not rollout_workers:
            self.issues.append(AlignmentIssue(
                severity=AlignmentIssueSeverity.ERROR,
                code="no_rollout_workers",
                message="No rollout workers available for job assignment",
            ))
            return []

        if not reward_workers:
            self.issues.append(AlignmentIssue(
                severity=AlignmentIssueSeverity.ERROR,
                code="no_reward_workers",
                message="No reward workers available for job assignment",
            ))
            return []

        # Resolve checkpoints
        memory_checkpoint_id = self._resolve_checkpoint(ActorRole.MEMORY_BUILDER)
        question_checkpoint_id = self._resolve_checkpoint(ActorRole.QUESTION_AGENT)

        # Create and assign jobs round-robin
        jobs = []
        for idx, record_id in enumerate(record_ids):
            # Validate oracle graph exists
            self._validate_oracle_graph(record_id)

            # Round-robin assignment
            rollout_worker = rollout_workers[idx % len(rollout_workers)]
            reward_worker = reward_workers[idx % len(reward_workers)]

            # Generate job ID
            job_id = self._generate_job_id(record_id, self.config.round_id, self.config.step_id)

            jobs.append(Phase3RolloutJob(
                job_id=job_id,
                record_id=record_id,
                round_id=self.config.round_id,
                step_id=self.config.step_id,
                seed=self.config.seed + idx,
                memory_checkpoint_id=memory_checkpoint_id,
                question_checkpoint_id=question_checkpoint_id,
                assigned_rollout_worker=rollout_worker.worker_id,
                assigned_reward_worker=reward_worker.worker_id,
                status=Phase3JobStatus.PENDING,
            ))

        return jobs

    def _load_record_ids(self) -> list[str]:
        """Load record IDs from input file."""
        input_path = Path(self.config.input_records_path)

        if not input_path.exists():
            self.issues.append(AlignmentIssue(
                severity=AlignmentIssueSeverity.ERROR,
                code="missing_input_records",
                message=f"Input records file not found: {input_path}",
                artifact_path=str(input_path),
            ))
            return []

        # Load JSON records
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both list of records and dict with 'records' key
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "records" in data:
                records = data["records"]
            else:
                records = []

            record_ids = []
            for i, record in enumerate(records):
                if not isinstance(record, dict):
                    record_ids.append(f"record_{i}")
                    continue

                record_id = None
                for key in RECORD_ID_KEYS:
                    value = record.get(key)
                    if value:
                        record_id = str(value)
                        break
                record_ids.append(record_id or f"record_{i}")

            if self.config.record_id_filter is not None:
                record_ids = [
                    record_id
                    for record_id in record_ids
                    if record_id == self.config.record_id_filter
                ]

            # Respect max_records
            if self.config.max_records is not None:
                record_ids = record_ids[:self.config.max_records]

            return record_ids

        except Exception as e:
            self.issues.append(AlignmentIssue(
                severity=AlignmentIssueSeverity.ERROR,
                code="failed_to_load_input_records",
                message=f"Failed to load input records: {e}",
                artifact_path=str(input_path),
            ))
            return []

    def _validate_oracle_graph(self, record_id: str) -> None:
        """Validate that oracle graph exists for a record."""
        graph_dir = Path(self.config.oracle_graph_dir)
        graph_path = graph_dir / f"{record_id}.graph.json"

        if not graph_path.exists():
            self.issues.append(AlignmentIssue(
                severity=AlignmentIssueSeverity.WARNING,
                code="missing_oracle_graph",
                message=f"Oracle graph not found for record: {record_id}",
                artifact_path=str(graph_path),
                record_id=record_id,
            ))

    def _resolve_checkpoint(self, actor_role: ActorRole) -> str | None:
        """Resolve checkpoint ID for an actor from config or registry."""
        # Get actor config
        if actor_role == ActorRole.MEMORY_BUILDER:
            actor_config = self.config.memory_builder
        else:
            actor_config = self.config.question_agent

        # Use explicitly provided checkpoint ID if present
        if actor_config.checkpoint_id:
            return actor_config.checkpoint_id

        # Try to load from checkpoint registry
        if self.config.checkpoint_registry_path:
            registry_path = Path(self.config.checkpoint_registry_path)
            if registry_path.exists():
                try:
                    with open(registry_path, "r", encoding="utf-8") as f:
                        registry = json.load(f)

                    latest_by_actor = registry.get("latest_by_actor", {})
                    checkpoint_id = latest_by_actor.get(actor_role.value)

                    if checkpoint_id:
                        actor_config.checkpoint_id = checkpoint_id
                        return checkpoint_id

                except Exception as e:
                    self.issues.append(AlignmentIssue(
                        severity=AlignmentIssueSeverity.WARNING,
                        code="failed_to_load_checkpoint_registry",
                        message=f"Failed to load checkpoint registry: {e}",
                        artifact_path=str(registry_path),
                        role=actor_role,
                    ))

        # No checkpoint resolved - cold start
        self.issues.append(AlignmentIssue(
            severity=AlignmentIssueSeverity.INFO,
            code="cold_start_actor",
            message=f"No checkpoint resolved for {actor_role.value}, will use model_path for cold start",
            role=actor_role,
        ))

        return None

    def _generate_job_id(self, record_id: str, round_id: int, step_id: int) -> str:
        """Generate stable job ID."""
        payload = f"{record_id}:round{round_id}:step{step_id}"
        job_hash = hashlib.sha256(payload.encode()).hexdigest()[:8]
        return f"job_{record_id}_{job_hash}"

    def write_worker_plan(self, plan: Phase3WorkerPlan | None = None) -> Path:
        """Write worker plan to output directory."""
        if plan is None:
            plan = self.build_worker_plan()

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plan_path = output_dir / "worker_plan.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        # Write case assignments manifest
        assignments_path = output_dir / "case_assignments.jsonl"
        with open(assignments_path, "w", encoding="utf-8") as f:
            for job in plan.jobs:
                f.write(json.dumps({
                    "job_id": job.job_id,
                    "record_id": job.record_id,
                    "round_id": job.round_id,
                    "step_id": job.step_id,
                    "rollout_worker": job.assigned_rollout_worker,
                    "reward_worker": job.assigned_reward_worker,
                    "memory_checkpoint_id": job.memory_checkpoint_id,
                    "question_checkpoint_id": job.question_checkpoint_id,
                }, ensure_ascii=False) + "\n")

        return plan_path


# ============================================================================
# Local Fake-Distributed Runtime
# ============================================================================

class LocalFakeDistributedRuntime:
    """
    Local fake-distributed runtime for Milestone 1.

    This runtime simulates distributed execution without Ray/verl:
    - Plans worker assignments
    - Validates inputs and checkpoints
    - Enforces no-leakage boundaries
    - Writes Phase 2-compatible output layout
    - Does NOT perform expensive model generation by default
    """

    def __init__(self, config: Phase3RuntimeConfig) -> None:
        self.config = config
        self.planner = Phase3RuntimePlanner(config)

    def run(self) -> Phase3ReadinessReport:
        """
        Run the local fake-distributed runtime.

        Returns:
            Readiness report with validation issues and metrics
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write runtime config
        config_path = output_dir / "runtime_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        # Build and write worker plan once so all artifacts share the same assignments.
        worker_plan = self.planner.build_worker_plan()
        self.planner.write_worker_plan(worker_plan)

        # Collect all issues from planner
        all_issues = self.planner.issues.copy()

        # Validate no-leakage if strict mode enabled
        if self.config.strict_no_leakage:
            leakage_issues = self._validate_no_leakage()
            all_issues.extend(leakage_issues)

        # Write runtime trace (stub in Milestone 1)
        self._write_runtime_trace_stub(worker_plan)

        # Build readiness report
        report = self._build_readiness_report(all_issues, worker_plan)

        # Write readiness report
        report_path = output_dir / "readiness_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        return report

    def _validate_no_leakage(self) -> list[AlignmentIssue]:
        """Validate no-leakage boundaries."""
        issues = []
        memory_builder_payload = {
            "memory_builder": self.config.memory_builder.model_dump(mode="json"),
        }

        forbidden_paths = []
        forbidden_paths.extend(check_for_forbidden_fields(memory_builder_payload))
        forbidden_paths.extend(_find_forbidden_runtime_keys(memory_builder_payload, "runtime_config"))

        for path in sorted(set(forbidden_paths)):
            issues.append(AlignmentIssue(
                severity=AlignmentIssueSeverity.ERROR,
                code="memory_builder_leakage",
                message=f"Memory Builder runtime config contains forbidden teacher field: {path}",
                role=ActorRole.MEMORY_BUILDER,
            ))

        return issues

    def _write_runtime_trace_stub(self, worker_plan: Phase3WorkerPlan) -> None:
        """Write stub runtime trace (no actual execution in Milestone 1)."""
        output_dir = Path(self.config.output_dir)
        trace_path = output_dir / "local_runtime_trace.jsonl"

        with open(trace_path, "w", encoding="utf-8") as f:
            # Write one stub trace per job
            for job in worker_plan.jobs:
                trace = Phase3RuntimeTrace(
                    trace_id=f"trace_{job.job_id}",
                    runtime_id=self.config.runtime_id,
                    job_id=job.job_id,
                    record_id=job.record_id,
                    status=Phase3JobStatus.SKIPPED,
                    issues=["Milestone 1: dry-run mode, no execution"],
                    metrics={"dry_run": True},
                )
                f.write(json.dumps(trace.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def _build_readiness_report(
        self,
        issues: list[AlignmentIssue],
        worker_plan: Phase3WorkerPlan,
    ) -> Phase3ReadinessReport:
        """Build Phase 3 readiness report."""
        # Count issues by severity
        error_count = sum(1 for issue in issues if issue.severity == AlignmentIssueSeverity.ERROR)
        warning_count = sum(1 for issue in issues if issue.severity == AlignmentIssueSeverity.WARNING)
        info_count = sum(1 for issue in issues if issue.severity == AlignmentIssueSeverity.INFO)

        # Determine status
        if error_count > 0:
            status = "failed"
        elif warning_count > 0:
            status = "warnings"
        else:
            status = "passed"

        # Check if jobs have resolved checkpoint IDs
        memory_checkpoint_resolved = False
        question_checkpoint_resolved = False

        if worker_plan.jobs:
            memory_checkpoint_resolved = worker_plan.jobs[0].memory_checkpoint_id is not None
            question_checkpoint_resolved = worker_plan.jobs[0].question_checkpoint_id is not None

        cold_start_count = sum(
            1 for issue in issues
            if issue.code == "cold_start_actor"
        )

        memory_reference_resolved = self.config.memory_builder.reference_checkpoint_id is not None
        question_reference_resolved = self.config.question_agent.reference_checkpoint_id is not None

        # Build report
        report_id = hashlib.sha256(self.config.runtime_id.encode()).hexdigest()[:8]

        return Phase3ReadinessReport(
            report_id=report_id,
            runtime_id=self.config.runtime_id,
            status=status,
            backend=self.config.backend,
            issues=issues,
            metrics={
                "total_issues": len(issues),
                "error_count": error_count,
                "warning_count": warning_count,
                "info_count": info_count,
                "memory_checkpoint_resolved": memory_checkpoint_resolved,
                "question_checkpoint_resolved": question_checkpoint_resolved,
                "memory_reference_resolved": memory_reference_resolved,
                "question_reference_resolved": question_reference_resolved,
                "cold_start_actor_count": cold_start_count,
                "num_jobs": len(worker_plan.jobs),
                "num_rollout_workers": self.config.num_rollout_workers,
                "num_reward_workers": self.config.num_reward_workers,
            },
        )
