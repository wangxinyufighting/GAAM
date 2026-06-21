"""
Local GRPO Trainer for Phase 2 Milestone 3.

This module orchestrates local trainer steps:
1. Load actor update batches from Milestone 2 rollout
2. Validate and select update items
3. Call Memory Builder policy update API
4. Call Question Agent policy update API
5. Write trainer step trace
6. Write local checkpoints / checkpoint metadata
7. Write training summary

This is still a local skeleton. It makes the training loop shape real without requiring
GPU training, Ray, verl, vLLM, LoRA, or distributed rollout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from gaam_graph.grpo_schema import (
    ActorRole,
    ActorUpdateBatch,
    GRPOTrainingTrace,
    GroupedRollout,
)
from gaam_graph.grpo_advantage import build_actor_update_batch
from gaam_graph.policy_clients import (
    BasePolicyClient,
    PolicyUpdateResult,
    StubMemoryBuilderPolicy,
    StubQuestionAgentPolicy,
)


# ============================================================================
# Configuration
# ============================================================================

class GRPOTrainerConfig(BaseModel):
    """Configuration for local GRPO trainer."""
    rollout_dir: Path
    output_dir: Path
    round_id: int = 0
    step_id: int = 0
    record_id: str | None = None
    max_records: int | None = None
    dry_run: bool = True
    fail_fast: bool = False
    require_both_actor_batches: bool = False
    reconstruct_missing_batches: bool = True
    selected_only: bool = True
    memory_policy_id: str = "stub_memory_builder"
    question_policy_id: str = "stub_question_agent"
    write_checkpoints: bool = True
    allow_teacher_fields_for_stub: bool = False


# ============================================================================
# Trainer Trace
# ============================================================================

class GRPOTrainerStepTrace(BaseModel):
    """Trace for one trainer step (one record)."""
    trace_id: str
    round_id: int
    step_id: int
    record_id: str
    status: str  # ok, skipped, failed, partial
    rollout_trace_path: str | None = None
    memory_update_batch_path: str | None = None
    question_update_batch_path: str | None = None
    memory_update_result: Dict[str, Any] | None = None
    question_update_result: Dict[str, Any] | None = None
    memory_checkpoint_path: str | None = None
    question_checkpoint_path: str | None = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# ============================================================================
# Trainer Summary
# ============================================================================

class GRPOTrainerSummary(BaseModel):
    """Summary of trainer run across multiple records."""
    round_id: int
    step_id: int
    rollout_dir: str
    output_dir: str
    total_records: int
    succeeded_records: int
    skipped_records: int
    failed_records: int
    partial_records: int
    average_memory_reward: float | None = None
    average_question_reward: float | None = None
    average_memory_advantage: float | None = None
    average_question_advantage: float | None = None
    manifest_path: str


# ============================================================================
# Helper Functions
# ============================================================================

def read_json(path: Path) -> Any:
    """Read JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """Write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_path_fragment(value: str) -> str:
    """Return a filesystem-safe path fragment while preserving readable IDs."""
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned.strip("._") or "unknown"


FORBIDDEN_KEYS = {
    "oracle_graph",
    "supporting_node_ids",
    "oracle_supporting_node_ids",
    "supporting_trajectory_ids",
    "gold_answer",
    "benchmark_answer",
    "benchmark_question",
    "raw_history",
    "answer_reports",
    "reward_report",
    "validity_reports",
    "expected_answer",
}


def check_for_teacher_fields(obj: Any, path: str = "root") -> List[str]:
    """
    Recursively check for teacher/oracle fields in an object.

    Returns:
        List of paths where forbidden keys were found
    """
    violations = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in FORBIDDEN_KEYS:
                violations.append(f"{path}.{key}")
            violations.extend(check_for_teacher_fields(value, f"{path}.{key}"))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            violations.extend(check_for_teacher_fields(item, f"{path}[{i}]"))
    elif isinstance(obj, str):
        # Try to parse JSON strings
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, (dict, list)):
                violations.extend(check_for_teacher_fields(parsed, path))
        except (json.JSONDecodeError, TypeError):
            pass

    return violations


def assert_actor_update_batch_safe(
    batch: ActorUpdateBatch,
    *,
    allow_teacher_fields_for_stub: bool = False,
    dry_run: bool = False,
) -> List[str]:
    """
    Check that actor update batch contains no teacher/oracle fields.

    Args:
        batch: Actor update batch to check
        allow_teacher_fields_for_stub: Allow teacher fields in dry-run mode
        dry_run: Whether this is a dry run

    Returns:
        List of violation paths (empty if safe)

    Raises:
        ValueError: If unsafe fields found and not allowed
    """
    violations = []

    batch_metadata_violations = check_for_teacher_fields(batch.metadata, "batch.metadata")
    violations.extend(batch_metadata_violations)

    for item in batch.items:
        # Check prompt
        prompt_violations = check_for_teacher_fields(item.prompt, f"item[{item.sample_id}].prompt")
        violations.extend(prompt_violations)

        # Check response
        response_violations = check_for_teacher_fields(item.response, f"item[{item.sample_id}].response")
        violations.extend(response_violations)

        # Check metadata
        metadata_violations = check_for_teacher_fields(item.metadata, f"item[{item.sample_id}].metadata")
        violations.extend(metadata_violations)

    if violations:
        if dry_run and allow_teacher_fields_for_stub:
            # Just warn
            return violations
        else:
            # Fail
            raise ValueError(
                f"Actor update batch contains teacher/oracle fields: {violations}\n"
                f"This is a leakage risk for real updates. "
                f"Use --allow_teacher_fields_for_stub only for dry-run diagnostics."
            )

    return []


# ============================================================================
# Local GRPO Trainer
# ============================================================================

class LocalGRPOTrainer:
    """
    Local GRPO trainer skeleton.

    Consumes Milestone 2 rollout artifacts and executes trainer-side control flow.
    """

    def __init__(
        self,
        config: GRPOTrainerConfig,
        *,
        memory_policy: BasePolicyClient | None = None,
        question_policy: BasePolicyClient | None = None,
    ) -> None:
        self.config = config
        self.memory_policy = memory_policy or StubMemoryBuilderPolicy(
            policy_id=config.memory_policy_id
        )
        self.question_policy = question_policy or StubQuestionAgentPolicy(
            policy_id=config.question_policy_id
        )

        # Validate config
        if not self.config.rollout_dir.exists():
            raise FileNotFoundError(f"Rollout dir does not exist: {self.config.rollout_dir}")

    def run(self) -> GRPOTrainerSummary:
        """
        Run trainer for all records.

        Returns:
            GRPOTrainerSummary with aggregate statistics
        """

        # Create output directories
        self._create_output_directories()

        # Discover records
        records = self._discover_records()

        # Filter records
        if self.config.record_id:
            records = [r for r in records if r == self.config.record_id]

        if self.config.max_records:
            records = records[:self.config.max_records]

        print(f"Processing {len(records)} records...")

        # Process each record
        traces = []
        succeeded = 0
        skipped = 0
        failed = 0
        partial = 0

        all_memory_rewards = []
        all_question_rewards = []
        all_memory_advantages = []
        all_question_advantages = []

        for i, record_id in enumerate(records):
            print(f"\n[{i+1}/{len(records)}] Processing {record_id}...")

            try:
                trace = self.run_record(record_id)
                traces.append(trace)

                if trace.status == "ok":
                    succeeded += 1
                elif trace.status == "skipped":
                    skipped += 1
                elif trace.status == "partial":
                    partial += 1
                else:
                    failed += 1

                # Collect metrics
                if trace.memory_update_result:
                    if trace.memory_update_result.get("mean_reward") is not None:
                        all_memory_rewards.append(trace.memory_update_result["mean_reward"])
                    if trace.memory_update_result.get("mean_advantage") is not None:
                        all_memory_advantages.append(trace.memory_update_result["mean_advantage"])

                if trace.question_update_result:
                    if trace.question_update_result.get("mean_reward") is not None:
                        all_question_rewards.append(trace.question_update_result["mean_reward"])
                    if trace.question_update_result.get("mean_advantage") is not None:
                        all_question_advantages.append(trace.question_update_result["mean_advantage"])

            except Exception as e:
                print(f"  ERROR: {e}")
                failed += 1

                if self.config.fail_fast:
                    raise

        # Write manifest
        manifest_path = self.config.output_dir / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for trace in traces:
                f.write(json.dumps(trace.model_dump(mode="json"), ensure_ascii=False) + "\n")

        # Compute averages
        avg_memory_reward = sum(all_memory_rewards) / len(all_memory_rewards) if all_memory_rewards else None
        avg_question_reward = sum(all_question_rewards) / len(all_question_rewards) if all_question_rewards else None
        avg_memory_advantage = sum(all_memory_advantages) / len(all_memory_advantages) if all_memory_advantages else None
        avg_question_advantage = sum(all_question_advantages) / len(all_question_advantages) if all_question_advantages else None

        # Create summary
        summary = GRPOTrainerSummary(
            round_id=self.config.round_id,
            step_id=self.config.step_id,
            rollout_dir=str(self.config.rollout_dir),
            output_dir=str(self.config.output_dir),
            total_records=len(records),
            succeeded_records=succeeded,
            skipped_records=skipped,
            failed_records=failed,
            partial_records=partial,
            average_memory_reward=avg_memory_reward,
            average_question_reward=avg_question_reward,
            average_memory_advantage=avg_memory_advantage,
            average_question_advantage=avg_question_advantage,
            manifest_path=str(manifest_path),
        )

        # Write summary
        summary_path = self.config.output_dir / "summary.json"
        write_json(summary_path, summary.model_dump(mode="json"))

        print(f"\n✅ Summary: {succeeded} succeeded, {partial} partial, {skipped} skipped, {failed} failed")
        print(f"📁 Output: {self.config.output_dir}")

        return summary

    def run_record(
        self,
        record_id: str,
        rollout_trace_path: Path | None = None,
    ) -> GRPOTrainerStepTrace:
        """
        Run trainer for one record.

        Args:
            record_id: Record identifier
            rollout_trace_path: Optional path to rollout trace

        Returns:
            GRPOTrainerStepTrace
        """

        trace_id = f"trainer_round{self.config.round_id}_step{self.config.step_id}_{record_id}"

        try:
            # Load rollout trace if available
            if rollout_trace_path is None:
                rollout_trace_path = self.config.rollout_dir / "traces" / f"{record_id}.grpo_trace.json"

            rollout_trace = None
            if rollout_trace_path.exists():
                rollout_trace_data = read_json(rollout_trace_path)
                rollout_trace = GRPOTrainingTrace.model_validate(rollout_trace_data)

            # Locate actor update batches
            memory_batch_path = self._locate_batch(record_id, ActorRole.MEMORY_BUILDER, rollout_trace)
            question_batch_path = self._locate_batch(record_id, ActorRole.QUESTION_AGENT, rollout_trace)

            # Check if we have at least one batch
            memory_batch = None
            question_batch = None

            if memory_batch_path and memory_batch_path.exists():
                memory_batch = ActorUpdateBatch.model_validate(read_json(memory_batch_path))
            elif self.config.reconstruct_missing_batches:
                memory_batch, memory_batch_path = self._reconstruct_batch(
                    record_id, ActorRole.MEMORY_BUILDER
                )

            if question_batch_path and question_batch_path.exists():
                question_batch = ActorUpdateBatch.model_validate(read_json(question_batch_path))
            elif self.config.reconstruct_missing_batches:
                question_batch, question_batch_path = self._reconstruct_batch(
                    record_id, ActorRole.QUESTION_AGENT
                )

            # Check if both required
            if self.config.require_both_actor_batches:
                if memory_batch is None or question_batch is None:
                    trace = GRPOTrainerStepTrace(
                        trace_id=trace_id,
                        round_id=self.config.round_id,
                        step_id=self.config.step_id,
                        record_id=record_id,
                        status="failed",
                        error="require_both_actor_batches=True but at least one batch missing",
                    )
                    self._write_trace(trace)
                    return trace

            # Check if both missing
            if memory_batch is None and question_batch is None:
                trace = GRPOTrainerStepTrace(
                    trace_id=trace_id,
                    round_id=self.config.round_id,
                    step_id=self.config.step_id,
                    record_id=record_id,
                    status="skipped",
                    error="no_actor_batches_available",
                )
                trace_path = self._write_trace(trace)
                print(f"  ⏭  skipped: {trace_path}")
                return trace

            # Update memory policy
            memory_result = None
            memory_checkpoint_path = None

            if memory_batch:
                memory_result = self._update_actor(memory_batch, self.memory_policy)

                if self.config.write_checkpoints:
                    memory_checkpoint_path = self._save_checkpoint(
                        self.memory_policy,
                        memory_result,
                    )

            # Update question policy
            question_result = None
            question_checkpoint_path = None

            if question_batch:
                question_result = self._update_actor(question_batch, self.question_policy)

                if self.config.write_checkpoints:
                    question_checkpoint_path = self._save_checkpoint(
                        self.question_policy,
                        question_result,
                    )

            # Determine status
            memory_ok = memory_result is not None and memory_result.error is None
            question_ok = question_result is not None and question_result.error is None

            if memory_ok and question_ok:
                status = "ok"
            elif memory_ok or question_ok:
                status = "partial"
            elif memory_batch is None and question_batch is None:
                status = "skipped"
            else:
                status = "failed"

            # Create trace
            trace = GRPOTrainerStepTrace(
                trace_id=trace_id,
                round_id=self.config.round_id,
                step_id=self.config.step_id,
                record_id=record_id,
                status=status,
                rollout_trace_path=str(rollout_trace_path) if rollout_trace_path and rollout_trace_path.exists() else None,
                memory_update_batch_path=str(memory_batch_path) if memory_batch_path else None,
                question_update_batch_path=str(question_batch_path) if question_batch_path else None,
                memory_update_result=memory_result.model_dump(mode="json") if memory_result else None,
                question_update_result=question_result.model_dump(mode="json") if question_result else None,
                memory_checkpoint_path=memory_checkpoint_path,
                question_checkpoint_path=question_checkpoint_path,
                metrics={
                    "memory_updated": memory_result.updated if memory_result else False,
                    "question_updated": question_result.updated if question_result else False,
                },
            )

            trace_path = self._write_trace(trace)

            print(f"  ✅ {status}: {trace_path}")

            return trace

        except Exception as e:
            print(f"  ❌ Failed: {e}")

            trace = GRPOTrainerStepTrace(
                trace_id=trace_id,
                round_id=self.config.round_id,
                step_id=self.config.step_id,
                record_id=record_id,
                status="failed",
                rollout_trace_path=str(rollout_trace_path) if rollout_trace_path and rollout_trace_path.exists() else None,
                error=str(e),
            )
            self._write_trace(trace)
            return trace

    def _create_output_directories(self) -> None:
        """Create output directory structure."""
        for name in ["traces", "update_batches", "checkpoints", "metrics"]:
            (self.config.output_dir / name).mkdir(parents=True, exist_ok=True)

        # Create actor checkpoint directories
        (self.config.output_dir / "checkpoints" / "memory_builder").mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "checkpoints" / "question_agent").mkdir(parents=True, exist_ok=True)

    def _discover_records(self) -> List[str]:
        """Discover record IDs from rollout directory."""
        records = set()

        # Check manifest
        manifest_path = self.config.rollout_dir / "manifest.jsonl"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    records.add(data["record_id"])

        # Check traces directory
        traces_dir = self.config.rollout_dir / "traces"
        if traces_dir.exists():
            for trace_file in traces_dir.glob("*.grpo_trace.json"):
                record_id = trace_file.stem.replace(".grpo_trace", "")
                records.add(record_id)

        # Check groups directory
        groups_dir = self.config.rollout_dir / "groups"
        if groups_dir.exists():
            for suffix in ("memory_group", "question_group"):
                for group_file in groups_dir.glob(f"*.{suffix}.json"):
                    record_id = group_file.name.removesuffix(f".{suffix}.json")
                    records.add(record_id)

        return sorted(records)

    def _trace_path(self, record_id: str) -> Path:
        """Return the trainer trace path for one record."""
        return self.config.output_dir / "traces" / f"{record_id}.trainer_trace.json"

    def _write_trace(self, trace: GRPOTrainerStepTrace) -> Path:
        """Write a trainer trace and return its path."""
        trace_path = self._trace_path(trace.record_id)
        write_json(trace_path, trace.model_dump(mode="json"))
        return trace_path

    def _locate_batch(
        self,
        record_id: str,
        role: ActorRole,
        rollout_trace: GRPOTrainingTrace | None,
    ) -> Path | None:
        """Locate actor update batch file."""

        # Try from rollout trace
        if rollout_trace:
            if role == ActorRole.MEMORY_BUILDER and rollout_trace.memory_update_batch_path:
                path = Path(rollout_trace.memory_update_batch_path)
                if path.exists():
                    return path
            elif role == ActorRole.QUESTION_AGENT and rollout_trace.question_update_batch_path:
                path = Path(rollout_trace.question_update_batch_path)
                if path.exists():
                    return path

        # Try conventional location
        role_suffix = "memory" if role == ActorRole.MEMORY_BUILDER else "question"
        conventional_path = self.config.rollout_dir / "update_batches" / f"{record_id}.{role_suffix}_update_batch.json"
        if conventional_path.exists():
            return conventional_path

        return None

    def _reconstruct_batch(
        self,
        record_id: str,
        role: ActorRole,
    ) -> tuple[ActorUpdateBatch | None, Path | None]:
        """Reconstruct actor update batch from group file."""

        role_suffix = "memory" if role == ActorRole.MEMORY_BUILDER else "question"
        group_path = self.config.rollout_dir / "groups" / f"{record_id}.{role_suffix}_group.json"

        if not group_path.exists():
            return None, None

        try:
            group_data = read_json(group_path)
            group = GroupedRollout.model_validate(group_data)

            if not group.advantages:
                print(f"  ⚠️  Cannot reconstruct {role.value} batch: no advantages in group")
                return None, None

            batch = build_actor_update_batch(
                group,
                batch_id=f"reconstructed_round{self.config.round_id}_step{self.config.step_id}_{record_id}_{role_suffix}",
                round_id=self.config.round_id,
                step_id=self.config.step_id,
                selected_only=self.config.selected_only,
            )

            # Write reconstructed batch
            batch_path = self.config.output_dir / "update_batches" / f"{record_id}.{role_suffix}_update_batch.json"
            write_json(batch_path, batch.model_dump(mode="json"))

            print(f"  ✅ Reconstructed {role.value} batch from group")

            return batch, batch_path

        except Exception as e:
            print(f"  ⚠️  Failed to reconstruct {role.value} batch: {e}")
            return None, None

    def _update_actor(
        self,
        batch: ActorUpdateBatch,
        policy: BasePolicyClient,
    ) -> PolicyUpdateResult:
        """Update actor policy."""

        # Validate batch safety
        violations = assert_actor_update_batch_safe(
            batch,
            allow_teacher_fields_for_stub=self.config.allow_teacher_fields_for_stub,
            dry_run=self.config.dry_run,
        )

        if violations and not self.config.dry_run:
            return PolicyUpdateResult(
                role=batch.role,
                policy_id=policy.policy_id,
                batch_id=batch.batch_id,
                round_id=batch.round_id,
                step_id=batch.step_id,
                updated=False,
                num_items=len(batch.items),
                num_selected_items=0,
                error=f"Unsafe batch: {violations}",
            )

        if violations:
            print(f"  ⚠️  Batch contains teacher fields (allowed in dry-run): {violations}")

        # Call policy update
        result = policy.update_grpo(batch, dry_run=self.config.dry_run)

        return result

    def _save_checkpoint(
        self,
        policy: BasePolicyClient,
        result: PolicyUpdateResult,
    ) -> str:
        """Save checkpoint metadata."""

        role_dir = "memory_builder" if policy.role == ActorRole.MEMORY_BUILDER else "question_agent"
        checkpoint_dir = (
            self.config.output_dir
            / "checkpoints"
            / role_dir
            / f"round_{self.config.round_id:03d}_step_{self.config.step_id:03d}"
            / safe_path_fragment(result.batch_id)
        )

        metadata = policy.save_checkpoint(
            checkpoint_dir,
            round_id=self.config.round_id,
            step_id=self.config.step_id,
            source_batch_id=result.batch_id,
            updated=result.updated,
            metrics=result.metrics,
        )

        return str(metadata.checkpoint_path)
