"""
Local Adversarial Rollout for Phase 2 Milestone 2.

This module orchestrates Memory Builder and Question Agent sampling,
evaluates them through the Frozen Answerer and Reward Manager,
and produces GRPO-ready rollout artifacts.

Key constraints:
- Memory Builder samples use raw history only (no questions, no oracle)
- Question Agent samples use oracle graph (no benchmark target questions)
- Frozen Answerer receives CurrentMemory + questions (no oracle, no raw history)
- Reward Manager may use oracle graph and CurrentMemory
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from gaam_graph.grpo_schema import (
    ActorRole,
    SampleStatus,
    GroupStatus,
    PolicySample,
    GRPORewardItem,
    GroupedRollout,
    GRPOTrainingTrace,
    GRPOTrainingSummary,
)
from gaam_graph.grpo_advantage import (
    attach_advantages_to_group,
    build_actor_update_batch,
)
from gaam_graph.lme_loader import LongMemEvalLoader, LMERecord
from gaam_graph.oracle_graph_loader import OracleGraphLoader
from gaam_graph.memory_builder import BaselineMemoryBuilder
from gaam_graph.raw_history import raw_history_from_lme_record, raw_history_to_prompt_payload
from gaam_graph.question_agent import QuestionAgent
from gaam_graph.oracle_validity_checker import OracleValidityChecker, OracleValidityVerdict
from gaam_graph.answer_agent import FrozenAnswerer
from gaam_graph.answer_schema import answer_request_from_generated_question
from gaam_graph.reward_manager import RewardManager
from gaam_graph.memory_weakness_book import MemoryWeaknessBook
from gaam_graph.question_schema import GeneratedQuestion


def _json_prompt(payload: Dict[str, Any]) -> str:
    """Serialize a policy prompt payload without ASCII-escaping user history."""
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _sanitize_generated_question_for_policy(question: GeneratedQuestion) -> Dict[str, Any]:
    """Return the Question Agent output fields that are safe for actor training."""
    return {
        "question_id": question.question_id,
        "record_id": question.record_id,
        "question": question.question,
        "answer": question.answer,
        "question_type": question.question_type.value,
        "status": question.status.value,
        "supporting_session_ids": list(question.supporting_session_ids),
        "reason": question.reason,
    }


# ============================================================================
# Configuration Models
# ============================================================================

class MemorySampleConfig(BaseModel):
    """Configuration for one Memory Builder sampling variant."""
    sample_index: int
    include_assistant_turns: bool = False
    min_user_text_chars: int = 20
    max_event_nodes: int | None = None
    temperature: float | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuestionSampleConfig(BaseModel):
    """Configuration for one Question Agent sampling variant."""
    sample_index: int
    num_walks: int = 8
    walk_length: int = 5
    num_questions: int = 5
    seed: int = 42
    directed: bool = False
    require_multi_session: bool = False
    min_sessions: int = 2
    max_structural_ratio: float = 0.35
    max_consecutive_next: int = 1
    question_types: List[str] | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AdversarialRolloutConfig(BaseModel):
    """Top-level config for local adversarial rollout."""
    input_path: Path
    graph_dir: Path
    output_dir: Path
    round_id: int = 0
    step_id: int = 0
    record_id: str | None = None
    max_records: int | None = None
    overwrite: bool = False
    fail_fast: bool = False

    num_memory_samples: int = 3
    num_question_samples: int = 3
    base_seed: int = 42

    no_llm_memory: bool = True
    no_llm_question: bool = True
    no_llm_answer: bool = True

    answer_top_k: int = 8
    answer_model: str | None = None
    answer_temperature: float = 0.0
    max_answer_chars: int = 500

    correctness_mode: str = "heuristic"
    split: str = "train"
    update_weakness_book: bool = True
    weakness_book_dir: Path | None = None

    normalize_advantages: bool = True
    skip_zero_variance: bool = True
    write_actor_update_batches: bool = True


class PairEvaluation(BaseModel):
    """Evaluation of one memory sample against one question-set sample."""
    record_id: str
    memory_sample_id: str
    question_sample_id: str
    accepted_question_ids: List[str] = Field(default_factory=list)
    answer_report_paths: List[str] = Field(default_factory=list)
    memory_reward_report_path: str | None = None
    question_reward_report_path: str | None = None
    weakness_book_path: str | None = None
    memory_reward: float | None = None
    question_reward: float | None = None
    status: str = "ok"
    error: str | None = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class LocalAdversarialRolloutResult(BaseModel):
    """Result for one record's adversarial rollout."""
    record_id: str
    trace_path: str
    memory_group_path: str
    question_group_path: str
    memory_update_batch_path: str | None = None
    question_update_batch_path: str | None = None
    reward_matrix_path: str | None = None
    status: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# ============================================================================
# Helper Functions
# ============================================================================

def write_json(path: Path, data: Any) -> None:
    """Write data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    """Read data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Main Rollout Runner
# ============================================================================

class LocalAdversarialRolloutRunner:
    """
    Local adversarial rollout orchestrator.

    For each record:
    1. Sample K memory candidates (baseline config variants)
    2. Sample M question-set candidates (seed/config variants)
    3. Run oracle-validity gate on question sets
    4. Evaluate all valid memory × question pairs
    5. Aggregate rewards
    6. Compute advantages
    7. Write GRPO trace and optional update batches
    """

    def __init__(
        self,
        config: AdversarialRolloutConfig,
        *,
        question_agent: QuestionAgent | None = None,
        answerer: FrozenAnswerer | None = None,
        reward_manager: RewardManager | None = None,
    ) -> None:
        self.config = config
        self.question_agent = question_agent
        self.answerer = answerer
        self.reward_manager = reward_manager
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.config.input_path}")
        if not self.config.graph_dir.exists():
            raise FileNotFoundError(f"Graph dir does not exist: {self.config.graph_dir}")

        # Check for LLM memory mode - not yet supported
        if not self.config.no_llm_memory:
            raise NotImplementedError(
                "LLM memory sampling is not yet implemented. "
                "Please use no_llm_memory=True for baseline memory builder."
            )

    def run(self) -> GRPOTrainingSummary:
        """
        Run local adversarial rollout for all records.

        Returns:
            GRPOTrainingSummary with aggregate statistics
        """
        # Create output directories
        self._create_output_directories()

        # Load records
        loader = LongMemEvalLoader(str(self.config.input_path))
        all_records = loader.load()

        # Filter records
        records = self._filter_records(all_records)

        print(f"Processing {len(records)} records...")

        # Process each record
        results = []
        succeeded = 0
        failed = 0
        skipped = 0

        for i, record in enumerate(records):
            print(f"\n[{i+1}/{len(records)}] Processing record {record.record_id}...")

            try:
                result = self.run_record(record, self.config.graph_dir / f"{record.record_id}.graph.json")
                results.append(result)

                if result.status == "ok":
                    succeeded += 1
                elif result.status == "skipped":
                    skipped += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"  ERROR: {e}")
                failed += 1
                if self.config.fail_fast:
                    raise

        # Write manifest
        manifest_path = self.config.output_dir / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")

        # Create summary
        summary = GRPOTrainingSummary(
            round_id=self.config.round_id,
            total_steps=len(records),
            succeeded_steps=succeeded,
            failed_steps=failed,
            skipped_steps=skipped,
            trace_dir=str(self.config.output_dir / "traces"),
        )

        # Write summary
        summary_path = self.config.output_dir / "summary.json"
        write_json(summary_path, summary.model_dump(mode="json"))

        print(f"\n✅ Summary: {succeeded} succeeded, {failed} failed, {skipped} skipped")
        print(f"📁 Output: {self.config.output_dir}")

        return summary

    def run_record(self, record: LMERecord, graph_path: Path) -> LocalAdversarialRolloutResult:
        """
        Run local adversarial rollout for one record.

        Args:
            record: LongMemEval record
            graph_path: Path to oracle graph JSON

        Returns:
            LocalAdversarialRolloutResult with paths and metrics
        """
        record_id = record.record_id

        try:
            # Load oracle graph
            oracle_loader = OracleGraphLoader(str(graph_path), strict=True)
            graph_dict = oracle_loader._graph_data

            # Create record-specific output directories
            record_memory_dir = self.config.output_dir / "memory_samples" / record_id
            record_question_dir = self.config.output_dir / "question_samples" / record_id
            record_answer_dir = self.config.output_dir / "answers" / record_id
            record_reward_dir = self.config.output_dir / "rewards" / record_id

            for d in [record_memory_dir, record_question_dir, record_answer_dir, record_reward_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # Step 1: Sample memory candidates
            print(f"  Sampling {self.config.num_memory_samples} memory candidates...")
            memory_configs = self._generate_memory_configs()
            memory_samples_with_artifacts = self.sample_memory_candidates(
                record, memory_configs, record_memory_dir
            )

            # Step 2: Sample question-set candidates
            print(f"  Sampling {self.config.num_question_samples} question-set candidates...")
            question_configs = self._generate_question_configs()
            question_samples_with_artifacts = self.sample_question_sets(
                record_id, graph_path, graph_dict, question_configs, record_question_dir
            )

            accepted_question_sample_count = sum(
                1
                for sample, questions in question_samples_with_artifacts
                if sample.status == SampleStatus.ACCEPTED and questions
            )

            # Step 3: Evaluate all memory × question pairs. Rejected question
            # sets still create rejected cells so skipped rollouts are auditable.
            print(
                f"  Evaluating {len(memory_samples_with_artifacts)} × "
                f"{len(question_samples_with_artifacts)} pairs..."
            )
            pair_evaluations = []

            for memory_sample, current_memory in memory_samples_with_artifacts:
                for question_sample, accepted_questions in question_samples_with_artifacts:
                    pair_eval = self.evaluate_pair(
                        record=record,
                        current_memory=current_memory,
                        memory_sample=memory_sample,
                        accepted_questions=accepted_questions,
                        question_sample=question_sample,
                        oracle_graph=oracle_loader,
                        answer_dir=record_answer_dir,
                        reward_dir=record_reward_dir,
                        weakness_book_dir=self.config.output_dir / "weakness_books",
                    )
                    pair_evaluations.append(pair_eval)

            # Step 4: Build reward matrix
            print("  Building reward matrix...")
            reward_matrix = self._build_reward_matrix(
                record_id, memory_samples_with_artifacts, question_samples_with_artifacts, pair_evaluations
            )
            reward_matrix_path = self.config.output_dir / "reward_matrices" / f"{record_id}.reward_matrix.json"
            write_json(reward_matrix_path, reward_matrix)

            # Step 5: Aggregate rewards
            print("  Aggregating rewards...")
            memory_rewards = self._aggregate_memory_rewards(
                [s for s, _ in memory_samples_with_artifacts], pair_evaluations
            )
            question_rewards = self._aggregate_question_rewards(
                [s for s, _ in question_samples_with_artifacts], pair_evaluations
            )

            # Step 6: Build grouped rollouts
            memory_group = self._build_memory_group(
                record_id, memory_samples_with_artifacts, memory_rewards
            )
            question_group = self._build_question_group(
                record_id, question_samples_with_artifacts, question_rewards
            )

            # Step 7: Attach advantages
            print("  Computing advantages...")
            memory_group = attach_advantages_to_group(
                memory_group,
                normalize_by_std=self.config.normalize_advantages,
                skip_zero_variance=self.config.skip_zero_variance,
            )
            question_group = attach_advantages_to_group(
                question_group,
                normalize_by_std=self.config.normalize_advantages,
                skip_zero_variance=self.config.skip_zero_variance,
            )

            # Write groups
            memory_group_path = self.config.output_dir / "groups" / f"{record_id}.memory_group.json"
            question_group_path = self.config.output_dir / "groups" / f"{record_id}.question_group.json"
            write_json(memory_group_path, memory_group.model_dump(mode="json"))
            write_json(question_group_path, question_group.model_dump(mode="json"))

            # Step 8: Optionally build actor update batches
            memory_update_batch_path = None
            question_update_batch_path = None

            if self.config.write_actor_update_batches:
                print("  Writing actor update batches...")

                if memory_group.advantages:
                    memory_batch = build_actor_update_batch(
                        memory_group,
                        batch_id=f"round{self.config.round_id}_step{self.config.step_id}_{record_id}_memory",
                        round_id=self.config.round_id,
                        step_id=self.config.step_id,
                        selected_only=True,
                    )
                    memory_update_batch_path = self.config.output_dir / "update_batches" / f"{record_id}.memory_update_batch.json"
                    write_json(memory_update_batch_path, memory_batch.model_dump(mode="json"))

                if question_group.advantages:
                    question_batch = build_actor_update_batch(
                        question_group,
                        batch_id=f"round{self.config.round_id}_step{self.config.step_id}_{record_id}_question",
                        round_id=self.config.round_id,
                        step_id=self.config.step_id,
                        selected_only=True,
                    )
                    question_update_batch_path = self.config.output_dir / "update_batches" / f"{record_id}.question_update_batch.json"
                    write_json(question_update_batch_path, question_batch.model_dump(mode="json"))

            # Step 9: Write GRPO trace
            trace = GRPOTrainingTrace(
                trace_id=f"round{self.config.round_id}_step{self.config.step_id}_{record_id}",
                round_id=self.config.round_id,
                step_id=self.config.step_id,
                status="ok" if accepted_question_sample_count else "skipped",
                config=self.config.model_dump(mode="json"),
                memory_groups=[memory_group],
                question_groups=[question_group],
                memory_update_batch_path=str(memory_update_batch_path) if memory_update_batch_path else None,
                question_update_batch_path=str(question_update_batch_path) if question_update_batch_path else None,
                metrics={
                    "num_memory_samples": len(memory_samples_with_artifacts),
                    "num_question_samples": len(question_samples_with_artifacts),
                    "num_accepted_question_samples": accepted_question_sample_count,
                    "num_pair_evaluations": len(pair_evaluations),
                    "memory_group_status": memory_group.status.value,
                    "question_group_status": question_group.status.value,
                },
            )

            trace_path = self.config.output_dir / "traces" / f"{record_id}.grpo_trace.json"
            write_json(trace_path, trace.model_dump(mode="json"))

            print(f"  ✅ Complete: {trace_path}")

            return LocalAdversarialRolloutResult(
                record_id=record_id,
                trace_path=str(trace_path),
                memory_group_path=str(memory_group_path),
                question_group_path=str(question_group_path),
                memory_update_batch_path=str(memory_update_batch_path) if memory_update_batch_path else None,
                question_update_batch_path=str(question_update_batch_path) if question_update_batch_path else None,
                reward_matrix_path=str(reward_matrix_path),
                status="ok" if accepted_question_sample_count else "skipped",
                error=None if accepted_question_sample_count else "no_accepted_question_sets",
                metrics=trace.metrics,
            )

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            return LocalAdversarialRolloutResult(
                record_id=record_id,
                trace_path="",
                memory_group_path="",
                question_group_path="",
                status="failed",
                error=str(e),
            )

    def _create_output_directories(self) -> None:
        """Create output directory structure."""
        for name in [
            "traces",
            "groups",
            "update_batches",
            "reward_matrices",
            "memory_samples",
            "question_samples",
            "answers",
            "rewards",
            "weakness_books",
        ]:
            (self.config.output_dir / name).mkdir(parents=True, exist_ok=True)

    def _filter_records(self, all_records: List[LMERecord]) -> List[LMERecord]:
        """Filter records by record_id and max_records."""
        records = all_records

        if self.config.record_id:
            records = [r for r in records if r.record_id == self.config.record_id]

        if self.config.max_records:
            records = records[:self.config.max_records]

        return records

    def _generate_memory_configs(self) -> List[MemorySampleConfig]:
        """Generate memory sample configurations."""
        configs = []

        # For MVP: deterministic config variants
        # Later milestones can replace with trainable policy sampling
        variants = [
            {"include_assistant_turns": False, "min_user_text_chars": 20, "max_event_nodes": None},
            {"include_assistant_turns": False, "min_user_text_chars": 40, "max_event_nodes": None},
            {"include_assistant_turns": True, "min_user_text_chars": 20, "max_event_nodes": None},
        ]

        for i in range(self.config.num_memory_samples):
            variant = variants[i % len(variants)]
            configs.append(MemorySampleConfig(sample_index=i, **variant))

        return configs

    def _generate_question_configs(self) -> List[QuestionSampleConfig]:
        """Generate question sample configurations."""
        configs = []

        # For MVP: deterministic seed/config variants
        for i in range(self.config.num_question_samples):
            configs.append(QuestionSampleConfig(
                sample_index=i,
                seed=self.config.base_seed + i,
                num_walks=8,
                walk_length=5,
                num_questions=5,
            ))

        return configs

    # ========================================================================
    # Memory Sampling
    # ========================================================================

    def sample_memory_candidates(
        self,
        record: LMERecord,
        configs: List[MemorySampleConfig],
        output_dir: Path,
    ) -> List[tuple[PolicySample, Dict[str, Any]]]:
        """
        Sample memory candidates using baseline memory builder.

        Args:
            record: LongMemEval record
            configs: List of memory sample configurations
            output_dir: Directory to write current memory files

        Returns:
            List of (PolicySample, current_memory_dict) tuples
        """
        results = []

        for config in configs:
            sample_id = f"{record.record_id}:memory:{config.sample_index}"
            group_id = f"round{self.config.round_id}_step{self.config.step_id}_{record.record_id}_memory"

            try:
                # Build raw history view
                raw_history = raw_history_from_lme_record(record)

                # Build CurrentMemory using baseline memory builder
                builder = BaselineMemoryBuilder(
                    include_assistant_turns=config.include_assistant_turns,
                    min_user_text_chars=config.min_user_text_chars,
                    max_event_nodes=config.max_event_nodes,
                )

                current_memory_dict = builder.build_case_memory(raw_history)

                # Write to file
                memory_path = output_dir / f"memory_s{config.sample_index}.current_memory.json"
                write_json(memory_path, current_memory_dict)

                memory_prompt = _json_prompt({
                    "task": "Build a compact CurrentMemory object from raw history only.",
                    "constraints": [
                        "Do not use benchmark questions or gold answers.",
                        "Preserve useful personal facts, preferences, events, summaries, and abstractions.",
                    ],
                    "history": raw_history_to_prompt_payload(raw_history),
                    "builder_config": config.model_dump(mode="json"),
                })

                # Create policy sample
                policy_sample = PolicySample(
                    sample_id=sample_id,
                    group_id=group_id,
                    record_id=record.record_id,
                    role=ActorRole.MEMORY_BUILDER,
                    status=SampleStatus.ACCEPTED,
                    prompt=memory_prompt,
                    response=json.dumps(current_memory_dict, ensure_ascii=False),
                    artifact={
                        "current_memory_path": str(memory_path),
                        "builder_config": config.model_dump(mode="json"),
                    },
                    model_id="baseline_memory_builder",
                    temperature=config.temperature,
                    metadata={"no_llm": True},
                )

                results.append((policy_sample, current_memory_dict))

            except Exception as e:
                print(f"    Memory sample {config.sample_index} failed: {e}")

                # Create failed policy sample
                policy_sample = PolicySample(
                    sample_id=sample_id,
                    group_id=group_id,
                    record_id=record.record_id,
                    role=ActorRole.MEMORY_BUILDER,
                    status=SampleStatus.FAILED,
                    metadata={"error": str(e)},
                )

                # Don't include failed samples in results for now
                # They won't be used for reward computation
                if self.config.fail_fast:
                    raise

        return results

    # ========================================================================
    # Question Sampling
    # ========================================================================

    def sample_question_sets(
        self,
        record_id: str,
        graph_path: Path,
        graph_dict: Dict[str, Any],
        configs: List[QuestionSampleConfig],
        output_dir: Path,
    ) -> List[tuple[PolicySample, List[GeneratedQuestion]]]:
        """
        Sample question-set candidates using Question Agent.

        Args:
            record_id: Record identifier
            graph_path: Path to oracle graph
            graph_dict: Oracle graph dictionary
            configs: List of question sample configurations
            output_dir: Directory to write question files

        Returns:
            List of (PolicySample, accepted_questions) tuples
        """
        results = []

        for config in configs:
            sample_id = f"{record_id}:question:{config.sample_index}"
            group_id = f"round{self.config.round_id}_step{self.config.step_id}_{record_id}_question"

            try:
                # Sample trajectories and generate questions
                question_agent = self.question_agent or QuestionAgent(
                    no_llm=self.config.no_llm_question,
                )

                # Sample trajectories
                trajectories = question_agent.sample_trajectories(
                    graph=graph_dict,
                    num_walks=config.num_walks,
                    walk_length=config.walk_length,
                    seed=config.seed,
                    directed=config.directed,
                    require_multi_session=config.require_multi_session,
                    min_sessions=config.min_sessions,
                    max_structural_ratio=config.max_structural_ratio,
                    max_consecutive_next=config.max_consecutive_next,
                )

                # Generate candidate questions
                candidate_questions = question_agent.generate_candidates(
                    record_id=record_id,
                    graph_path=str(graph_path),
                    trajectories=trajectories,
                    num_questions=config.num_questions,
                    require_multi_session=config.require_multi_session,
                    min_sessions=config.min_sessions,
                    desired_question_types=config.question_types,
                )

                # Write candidates
                candidates_path = output_dir / f"question_s{config.sample_index}.candidates.json"
                write_json(candidates_path, [q.model_dump(mode="json") for q in candidate_questions])

                # Run oracle validity check
                oracle_loader = OracleGraphLoader(str(graph_path), strict=True)
                validity_checker = OracleValidityChecker(oracle_loader=oracle_loader)
                validity_reports = validity_checker.validate_questions(
                    candidate_questions,
                    trajectories=trajectories,
                    require_multi_session=config.require_multi_session,
                    min_sessions=config.min_sessions,
                )

                # Filter accepted questions
                accepted_questions = [
                    q for q, report in zip(candidate_questions, validity_reports)
                    if report.verdict == OracleValidityVerdict.ACCEPT
                ]

                # Write validated questions
                validated_path = output_dir / f"question_s{config.sample_index}.validated.json"
                write_json(validated_path, {
                    "candidates": [q.model_dump(mode="json") for q in candidate_questions],
                    "validity_reports": [r.model_dump(mode="json") for r in validity_reports],
                    "accepted": [q.model_dump(mode="json") for q in accepted_questions],
                })

                # Determine status
                if accepted_questions:
                    status = SampleStatus.ACCEPTED
                elif candidate_questions:
                    status = SampleStatus.REJECTED
                else:
                    status = SampleStatus.FAILED

                question_prompt = _json_prompt({
                    "task": "Generate diverse training questions from oracle graph trajectories.",
                    "constraints": [
                        "Do not use benchmark target questions or gold answers.",
                        "Questions must be answerable from the provided trajectories.",
                        "Prefer broad coverage across single-hop, multi-hop, multi-session, temporal, preference, personal fact, contradiction/update, and summary types.",
                    ],
                    "record_id": record_id,
                    "trajectories": trajectories,
                    "generation_config": config.model_dump(mode="json"),
                })
                policy_response = json.dumps(
                    [_sanitize_generated_question_for_policy(q) for q in candidate_questions],
                    ensure_ascii=False,
                )

                # Create policy sample
                policy_sample = PolicySample(
                    sample_id=sample_id,
                    group_id=group_id,
                    record_id=record_id,
                    role=ActorRole.QUESTION_AGENT,
                    status=status,
                    prompt=question_prompt,
                    response=policy_response,
                    artifact={
                        "candidate_questions_path": str(candidates_path),
                        "validated_questions_path": str(validated_path),
                        "accepted_question_count": len(accepted_questions),
                        "question_config": config.model_dump(mode="json"),
                    },
                    model_id="question_agent_no_llm" if self.config.no_llm_question else "question_agent",
                    metadata={
                        "oracle_validity_gate": "enforced",
                        "num_candidates": len(candidate_questions),
                        "num_rejected": len(candidate_questions) - len(accepted_questions),
                    },
                )

                results.append((policy_sample, accepted_questions))

            except Exception as e:
                print(f"    Question sample {config.sample_index} failed: {e}")

                # Create failed policy sample
                policy_sample = PolicySample(
                    sample_id=sample_id,
                    group_id=group_id,
                    record_id=record_id,
                    role=ActorRole.QUESTION_AGENT,
                    status=SampleStatus.FAILED,
                    metadata={"error": str(e)},
                )

                results.append((policy_sample, []))

                if self.config.fail_fast:
                    raise

        return results

    # ========================================================================
    # Pair Evaluation
    # ========================================================================

    def evaluate_pair(
        self,
        *,
        record: LMERecord,
        current_memory: Dict[str, Any],
        memory_sample: PolicySample,
        accepted_questions: List[GeneratedQuestion],
        question_sample: PolicySample,
        oracle_graph: OracleGraphLoader,
        answer_dir: Path,
        reward_dir: Path,
        weakness_book_dir: Path | None = None,
    ) -> PairEvaluation:
        """
        Evaluate one memory sample against one question-set sample.

        Args:
            record: LongMemEval record
            current_memory: CurrentMemory dict
            memory_sample: Memory builder policy sample
            accepted_questions: Accepted generated questions
            question_sample: Question agent policy sample
            oracle_graph: Oracle graph loader
            answer_dir: Directory to write answer reports
            reward_dir: Directory to write reward reports

        Returns:
            PairEvaluation with rewards and paths
        """
        memory_sample_id = memory_sample.sample_id.split(":")[-1]  # Extract "memory:N" -> "N"
        question_sample_id = question_sample.sample_id.split(":")[-1]  # Extract "question:N" -> "N"
        pair_id = f"{memory_sample_id}__{question_sample_id}"

        # Handle empty question set
        if not accepted_questions:
            return PairEvaluation(
                record_id=record.record_id,
                memory_sample_id=memory_sample.sample_id,
                question_sample_id=question_sample.sample_id,
                status="rejected",
                error="no_accepted_questions",
            )

        try:
            # Answer questions using Frozen Answerer
            answerer = self.answerer or FrozenAnswerer(
                no_llm=self.config.no_llm_answer,
                model_id=self.config.answer_model or "",
                max_answer_chars=self.config.max_answer_chars,
            )

            # Convert questions to answer requests
            answer_requests = [
                answer_request_from_generated_question(q.model_dump(mode="json"))
                for q in accepted_questions
            ]

            # Batch answer
            answer_reports = answerer.batch_answer(
                current_memory=current_memory,
                requests=answer_requests,
            )

            # Write answer reports
            answers_path = answer_dir / f"{pair_id}.answers.json"
            write_json(answers_path, [r.model_dump(mode="json") for r in answer_reports])
            question_dicts = [q.model_dump(mode="json") for q in accepted_questions]
            answer_report_dicts = [r.model_dump(mode="json") for r in answer_reports]
            validity_report_dicts = [
                {
                    "question_id": q.question_id,
                    "record_id": q.record_id,
                    "verdict": OracleValidityVerdict.ACCEPT.value,
                }
                for q in accepted_questions
            ]

            # Score rewards
            reward_manager = self.reward_manager or RewardManager(
                correctness_mode=self.config.correctness_mode,
            )

            # Score Memory Builder
            memory_reward_report = reward_manager.score_memory_builder(
                current_memory=current_memory,
                questions=question_dicts,
                answer_reports=answer_report_dicts,
                oracle_graph=oracle_graph,
                split=self.config.split,
            )

            memory_reward_path = reward_dir / f"{pair_id}.memory_reward.json"
            write_json(memory_reward_path, memory_reward_report.model_dump(mode="json"))

            # Score Question Agent
            qa_reward_report = reward_manager.score_question_agent(
                questions=question_dicts,
                validity_reports=validity_report_dicts,
                answer_reports=answer_report_dicts,
                oracle_graph=oracle_graph,
            )

            qa_reward_path = reward_dir / f"{pair_id}.question_reward.json"
            write_json(qa_reward_path, qa_reward_report.model_dump(mode="json"))

            weakness_book_path = None
            if self.config.update_weakness_book and weakness_book_dir is not None:
                weakness_book_dir.mkdir(parents=True, exist_ok=True)
                weakness_book_path_obj = weakness_book_dir / f"{record.record_id}.weakness_book.json"
                if weakness_book_path_obj.exists():
                    weakness_book = MemoryWeaknessBook.load(
                        weakness_book_path_obj,
                        record_id=record.record_id,
                    )
                else:
                    weakness_book = MemoryWeaknessBook.empty(record.record_id)
                weakness_book.update_from_reward_report(
                    memory_reward_report,
                    round_id=self.config.round_id,
                )
                weakness_book.save(weakness_book_path_obj)
                weakness_book_path = str(weakness_book_path_obj)

            return PairEvaluation(
                record_id=record.record_id,
                memory_sample_id=memory_sample.sample_id,
                question_sample_id=question_sample.sample_id,
                accepted_question_ids=[q.question_id for q in accepted_questions],
                answer_report_paths=[str(answers_path)],
                memory_reward_report_path=str(memory_reward_path),
                question_reward_report_path=str(qa_reward_path),
                weakness_book_path=weakness_book_path,
                memory_reward=memory_reward_report.total_reward,
                question_reward=qa_reward_report.total_reward,
                status="ok",
                metrics={
                    "num_questions": len(accepted_questions),
                },
            )

        except Exception as e:
            print(f"    Pair evaluation failed: {e}")
            return PairEvaluation(
                record_id=record.record_id,
                memory_sample_id=memory_sample.sample_id,
                question_sample_id=question_sample.sample_id,
                status="failed",
                error=str(e),
            )

    # ========================================================================
    # Reward Aggregation
    # ========================================================================

    def _build_reward_matrix(
        self,
        record_id: str,
        memory_samples_with_artifacts: List[tuple[PolicySample, Dict[str, Any]]],
        question_samples_with_artifacts: List[tuple[PolicySample, List[GeneratedQuestion]]],
        pair_evaluations: List[PairEvaluation],
    ) -> Dict[str, Any]:
        """Build reward matrix for memory × question interactions."""
        memory_sample_ids = [s.sample_id for s, _ in memory_samples_with_artifacts]
        question_sample_ids = [s.sample_id for s, _ in question_samples_with_artifacts]

        # Build cells from pair evaluations
        cells = []
        for pair_eval in pair_evaluations:
            cells.append({
                "memory_sample_id": pair_eval.memory_sample_id,
                "question_sample_id": pair_eval.question_sample_id,
                "accepted_question_count": len(pair_eval.accepted_question_ids),
                "memory_reward": pair_eval.memory_reward,
                "question_reward": pair_eval.question_reward,
                "answer_report_paths": pair_eval.answer_report_paths,
                "memory_reward_report_path": pair_eval.memory_reward_report_path,
                "question_reward_report_path": pair_eval.question_reward_report_path,
                "status": pair_eval.status,
            })

        # Aggregate memory sample rewards (average across question sets)
        memory_sample_rewards = {}
        for memory_sample_id in memory_sample_ids:
            relevant_pairs = [
                pair for pair in pair_evaluations
                if pair.memory_sample_id == memory_sample_id and pair.status == "ok" and pair.memory_reward is not None
            ]
            if relevant_pairs:
                memory_sample_rewards[memory_sample_id] = sum(p.memory_reward for p in relevant_pairs) / len(relevant_pairs)
            else:
                memory_sample_rewards[memory_sample_id] = 0.0

        # Aggregate question sample rewards (average across memory samples)
        question_sample_rewards = {}
        for question_sample_id in question_sample_ids:
            relevant_pairs = [
                pair for pair in pair_evaluations
                if pair.question_sample_id == question_sample_id and pair.status == "ok" and pair.question_reward is not None
            ]
            if relevant_pairs:
                question_sample_rewards[question_sample_id] = sum(p.question_reward for p in relevant_pairs) / len(relevant_pairs)
            else:
                question_sample_rewards[question_sample_id] = 0.0

        return {
            "record_id": record_id,
            "round_id": self.config.round_id,
            "step_id": self.config.step_id,
            "memory_sample_ids": memory_sample_ids,
            "question_sample_ids": question_sample_ids,
            "cells": cells,
            "memory_sample_rewards": memory_sample_rewards,
            "question_sample_rewards": question_sample_rewards,
        }

    def _aggregate_memory_rewards(
        self,
        memory_samples: List[PolicySample],
        pair_evaluations: List[PairEvaluation],
    ) -> List[GRPORewardItem]:
        """
        Aggregate memory rewards across question sets.

        For each memory sample, average memory rewards across all valid question-set pairs.
        """
        rewards = []

        for memory_sample in memory_samples:
            # Find relevant pairs
            relevant_pairs = [
                pair for pair in pair_evaluations
                if pair.memory_sample_id == memory_sample.sample_id
                and pair.status == "ok"
                and pair.memory_reward is not None
            ]

            if relevant_pairs:
                avg_reward = sum(p.memory_reward for p in relevant_pairs) / len(relevant_pairs)
            else:
                avg_reward = 0.0

            reward_item = GRPORewardItem(
                sample_id=memory_sample.sample_id,
                group_id=memory_sample.group_id,
                record_id=memory_sample.record_id,
                role=ActorRole.MEMORY_BUILDER,
                reward=avg_reward,
                diagnostics={
                    "evaluated_question_sample_count": len(relevant_pairs),
                },
            )

            rewards.append(reward_item)

        return rewards

    def _aggregate_question_rewards(
        self,
        question_samples: List[PolicySample],
        pair_evaluations: List[PairEvaluation],
    ) -> List[GRPORewardItem]:
        """
        Aggregate question rewards across memory samples.

        For each question sample, average question rewards across all memory samples.
        If no accepted questions, reward is 0.0 (oracle-validity gate failure).
        """
        rewards = []

        for question_sample in question_samples:
            # Find relevant pairs
            relevant_pairs = [
                pair for pair in pair_evaluations
                if pair.question_sample_id == question_sample.sample_id
                and pair.status == "ok"
                and pair.question_reward is not None
            ]

            if relevant_pairs:
                avg_reward = sum(p.question_reward for p in relevant_pairs) / len(relevant_pairs)
            else:
                avg_reward = 0.0

            # Enforce oracle-validity gate penalty
            accepted_question_count = question_sample.artifact.get("accepted_question_count", 0)
            if accepted_question_count == 0:
                avg_reward = 0.0

            reward_item = GRPORewardItem(
                sample_id=question_sample.sample_id,
                group_id=question_sample.group_id,
                record_id=question_sample.record_id,
                role=ActorRole.QUESTION_AGENT,
                reward=avg_reward,
                diagnostics={
                    "evaluated_memory_sample_count": len(relevant_pairs),
                    "accepted_question_count": accepted_question_count,
                },
            )

            rewards.append(reward_item)

        return rewards

    # ========================================================================
    # Group Building
    # ========================================================================

    def _build_memory_group(
        self,
        record_id: str,
        memory_samples_with_artifacts: List[tuple[PolicySample, Dict[str, Any]]],
        memory_rewards: List[GRPORewardItem],
    ) -> GroupedRollout:
        """Build memory GroupedRollout."""
        group_id = f"round{self.config.round_id}_step{self.config.step_id}_{record_id}_memory"

        return GroupedRollout(
            group_id=group_id,
            record_id=record_id,
            role=ActorRole.MEMORY_BUILDER,
            status=GroupStatus.OK,
            samples=[s for s, _ in memory_samples_with_artifacts],
            rewards=memory_rewards,
            advantages=[],  # Will be populated by attach_advantages_to_group
            metadata={},
        )

    def _build_question_group(
        self,
        record_id: str,
        question_samples_with_artifacts: List[tuple[PolicySample, List[GeneratedQuestion]]],
        question_rewards: List[GRPORewardItem],
    ) -> GroupedRollout:
        """Build question GroupedRollout."""
        group_id = f"round{self.config.round_id}_step{self.config.step_id}_{record_id}_question"

        return GroupedRollout(
            group_id=group_id,
            record_id=record_id,
            role=ActorRole.QUESTION_AGENT,
            status=GroupStatus.OK,
            samples=[s for s, _ in question_samples_with_artifacts],
            rewards=question_rewards,
            advantages=[],  # Will be populated by attach_advantages_to_group
            metadata={},
        )
