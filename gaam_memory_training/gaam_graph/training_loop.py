"""
Training loop module.

Orchestrates one-round adversarial memory training.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from gaam_graph.answer_agent import FrozenAnswerer
from gaam_graph.answer_schema import answer_request_from_generated_question
from gaam_graph.llm import OpenAICompatibleLLM
from gaam_graph.oracle_graph_loader import OracleGraphLoader
from gaam_graph.lme_loader import LMERecord, LongMemEvalLoader
from gaam_graph.memory_builder import BaselineMemoryBuilder, build_current_memory_from_record
from gaam_graph.memory_retriever import CurrentMemoryRetriever
from gaam_graph.memory_schema import validate_current_memory
from gaam_graph.memory_weakness_book import MemoryWeaknessBook
from gaam_graph.oracle_validity_checker import OracleValidityChecker
from gaam_graph.question_agent import QuestionAgent
from gaam_graph.question_schema import OracleValidityVerdict, QuestionStatus
from gaam_graph.reward_manager import RewardManager
from gaam_graph.rollout_schema import (
    ArtifactPaths,
    OneRoundTrace,
    StageStatus,
    StageTrace,
    TrainingRoundSummary,
)
from gaam_graph.utils import write_json


class OneRoundTrainingConfig(BaseModel):
    """Configuration for one-round training."""
    input_path: Path
    graph_dir: Path
    output_dir: Path
    round_id: int = 0
    record_id: str | None = None
    max_records: int | None = None

    overwrite: bool = False
    fail_fast: bool = False

    include_assistant_turns: bool = False
    min_user_text_chars: int = 20
    max_event_nodes: int | None = None

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
    accept_revise: bool = False
    allow_placeholder_questions: bool = False
    allow_empty_question_set: bool = False

    no_llm_question: bool = False
    no_llm_answer: bool = False
    answer_top_k: int = 8
    answer_model: str | None = None
    answer_temperature: float = 0.0
    max_answer_chars: int = 500

    correctness_mode: str = "heuristic"
    split: str = "train"
    weakness_book_dir: Path | None = None
    write_question_agent_reward: bool = True


class OneRoundTrainingLoop:
    """One-round adversarial memory training loop."""

    def __init__(self, config: OneRoundTrainingConfig) -> None:
        """Initialize training loop."""
        self.config = config

    def run(self) -> TrainingRoundSummary:
        """Run one training round across all records."""
        # Validate inputs
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_path}")
        if not self.config.graph_dir.exists():
            raise FileNotFoundError(f"Graph directory not found: {self.config.graph_dir}")

        # Load records
        loader = LongMemEvalLoader(str(self.config.input_path))
        records = loader.load()

        # Filter records
        if self.config.record_id:
            records = [r for r in records if r.record_id == self.config.record_id]
            if not records:
                raise ValueError(f"Record {self.config.record_id} not found")

        if self.config.max_records:
            records = records[: self.config.max_records]

        if not records:
            raise ValueError("No records selected")

        # Create output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "current_memory").mkdir(exist_ok=True)
        (self.config.output_dir / "question_sets").mkdir(exist_ok=True)
        (self.config.output_dir / "answers").mkdir(exist_ok=True)
        (self.config.output_dir / "rewards").mkdir(exist_ok=True)
        (self.config.output_dir / "weakness_books").mkdir(exist_ok=True)
        (self.config.output_dir / "traces").mkdir(exist_ok=True)

        # Process records
        manifest_lines = []
        succeeded = 0
        failed = 0
        skipped = 0
        total_memory_reward = 0.0
        total_qa_reward = 0.0

        for record in records:
            # Locate graph
            graph_path = self.config.graph_dir / f"{record.record_id}.graph.json"

            try:
                trace = self.run_record(record, graph_path)
                manifest_lines.append(self._trace_to_manifest_line(trace))

                if trace.status == StageStatus.OK:
                    succeeded += 1
                    if trace.metrics.get("memory_update_reward") is not None:
                        total_memory_reward += trace.metrics["memory_update_reward"]
                    if trace.metrics.get("question_agent_total_reward") is not None:
                        total_qa_reward += trace.metrics["question_agent_total_reward"]
                elif trace.status == StageStatus.SKIPPED:
                    skipped += 1
                else:
                    failed += 1
                    if self.config.fail_fast:
                        break

            except Exception as e:
                failed += 1
                # Write minimal failed trace
                trace_path = self.config.output_dir / "traces" / f"{record.record_id}.rollout_trace.json"
                trace = OneRoundTrace(
                    round_id=self.config.round_id,
                    record_id=record.record_id,
                    status=StageStatus.FAILED,
                    config={},
                    artifacts=ArtifactPaths(
                        graph_path=str(graph_path),
                        rollout_trace_path=str(trace_path),
                    ),
                    stages=[],
                    error=str(e),
                )
                write_json(trace_path, trace.model_dump(mode="json"))
                manifest_lines.append(self._trace_to_manifest_line(trace))

                if self.config.fail_fast:
                    break

        # Write manifest
        manifest_path = self.config.output_dir / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for line in manifest_lines:
                f.write(json.dumps(line) + "\n")

        # Write summary
        summary = TrainingRoundSummary(
            round_id=self.config.round_id,
            input_path=str(self.config.input_path),
            graph_dir=str(self.config.graph_dir),
            output_dir=str(self.config.output_dir),
            attempted_records=len(records),
            succeeded_records=succeeded,
            failed_records=failed,
            skipped_records=skipped,
            average_memory_update_reward=total_memory_reward / succeeded if succeeded > 0 else None,
            average_question_agent_reward=total_qa_reward / succeeded if succeeded > 0 else None,
            manifest_path=str(manifest_path),
        )
        write_json(self.config.output_dir / "summary.json", summary.model_dump(mode="json"))

        return summary

    def run_record(self, record: LMERecord, graph_path: Path) -> OneRoundTrace:
        """Run one training round for a single record."""
        stages: List[StageTrace] = []
        artifacts = ArtifactPaths(graph_path=str(graph_path))
        config_snapshot = self.config.model_dump(mode="json")

        # Convert Path objects to strings
        for key, value in config_snapshot.items():
            if isinstance(value, Path):
                config_snapshot[key] = str(value)
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, Path):
                        value[k] = str(v)

        try:
            # Stage 1: Load oracle graph
            stage = self._start_stage("load_oracle_graph")
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph file not found: {graph_path}")

            oracle_loader = OracleGraphLoader(str(graph_path), strict=True)
            graph_dict = oracle_loader._graph_data

            if oracle_loader.record_id != record.record_id:
                raise ValueError(
                    f"Graph record_id {oracle_loader.record_id} does not match record {record.record_id}"
                )

            self._end_stage(stage, StageStatus.OK, metrics={"node_count": len(graph_dict.get("nodes", []))})
            stages.append(stage)

            # Stage 2: Build current memory
            stage = self._start_stage("build_current_memory")
            builder = BaselineMemoryBuilder(
                include_assistant_turns=self.config.include_assistant_turns,
                min_user_text_chars=self.config.min_user_text_chars,
                max_event_nodes=self.config.max_event_nodes,
            )
            current_memory = build_current_memory_from_record(record, builder=builder)
            validated_memory_model = validate_current_memory(current_memory, strict=True)
            validated_memory = validated_memory_model.model_dump(mode="json")

            memory_path = self.config.output_dir / "current_memory" / f"{record.record_id}.current_memory.json"
            write_json(memory_path, validated_memory)
            artifacts.current_memory_path = str(memory_path)

            session_ids = set()
            for node in validated_memory["memory_graph"]["nodes"]:
                session_ids.update(node.get("source_session_ids", []))

            self._end_stage(
                stage,
                StageStatus.OK,
                outputs={"memory_path": str(memory_path)},
                metrics={
                    "node_count": len(validated_memory["memory_graph"]["nodes"]),
                    "edge_count": len(validated_memory["memory_graph"]["edges"]),
                    "session_count": len(session_ids),
                },
            )
            stages.append(stage)

            # Stage 3: Generate candidate questions
            stage = self._start_stage("generate_questions")
            question_llm = None if self.config.no_llm_question else OpenAICompatibleLLM()
            question_agent = QuestionAgent(llm=question_llm, no_llm=self.config.no_llm_question)

            trajectories = question_agent.sample_trajectories(
                graph=graph_dict,
                num_walks=self.config.num_walks,
                walk_length=self.config.walk_length,
                seed=self.config.seed,
                directed=self.config.directed,
                require_multi_session=self.config.require_multi_session,
                min_sessions=self.config.min_sessions,
                max_structural_ratio=self.config.max_structural_ratio,
                max_consecutive_next=self.config.max_consecutive_next,
            )

            candidates = question_agent.generate_candidates(
                record_id=record.record_id,
                graph_path=str(graph_path),
                trajectories=trajectories,
                num_questions=self.config.num_questions,
                require_multi_session=self.config.require_multi_session,
                min_sessions=self.config.min_sessions,
                desired_question_types=self.config.question_types,
            )

            candidate_path = (
                self.config.output_dir / "question_sets" / f"{record.record_id}.candidate_questions.json"
            )
            write_json(
                candidate_path,
                {
                    "record_id": record.record_id,
                    "graph_path": str(graph_path),
                    "generation_config": {
                        "num_walks": self.config.num_walks,
                        "walk_length": self.config.walk_length,
                        "num_questions": self.config.num_questions,
                        "require_multi_session": self.config.require_multi_session,
                        "min_sessions": self.config.min_sessions,
                        "no_llm": self.config.no_llm_question,
                    },
                    "trajectories": trajectories,
                    "questions": [c.model_dump(mode="json") for c in candidates],
                },
            )
            artifacts.candidate_questions_path = str(candidate_path)

            self._end_stage(
                stage,
                StageStatus.OK,
                outputs={"candidate_path": str(candidate_path)},
                metrics={"candidate_count": len(candidates)},
            )
            stages.append(stage)

            # Stage 4: Oracle validity gate
            stage = self._start_stage("oracle_validity")
            checker = OracleValidityChecker(
                oracle_loader=oracle_loader,
                allow_placeholder_questions=self.config.allow_placeholder_questions,
            )

            reports = checker.validate_questions(
                questions=candidates,
                trajectories=trajectories,
                require_multi_session=self.config.require_multi_session,
                min_sessions=self.config.min_sessions,
            )

            validity_path = (
                self.config.output_dir / "question_sets" / f"{record.record_id}.validity_reports.json"
            )
            write_json(
                validity_path,
                {
                    "record_id": record.record_id,
                    "graph_path": str(graph_path),
                    "validity_config": {
                        "allow_placeholder_questions": self.config.allow_placeholder_questions,
                    },
                    "reports": [r.model_dump(mode="json") for r in reports],
                },
            )
            artifacts.validity_reports_path = str(validity_path)

            # Filter accepted questions
            candidate_by_id = {c.question_id: c for c in candidates}
            accepted_questions = []
            accepted_reports = []
            for report in reports:
                question = candidate_by_id.get(report.question_id)
                if question is None:
                    continue
                if report.verdict == OracleValidityVerdict.ACCEPT:
                    question.status = QuestionStatus.ACCEPTED
                    accepted_questions.append(question)
                    accepted_reports.append(report)
                elif report.verdict == OracleValidityVerdict.REVISE and self.config.accept_revise:
                    question.status = QuestionStatus.REVISED
                    accepted_questions.append(question)
                    accepted_reports.append(report)
                else:
                    question.status = QuestionStatus.REJECTED

            if not accepted_questions and not self.config.allow_empty_question_set:
                raise ValueError("No questions passed oracle validity")

            accepted_path = (
                self.config.output_dir / "question_sets" / f"{record.record_id}.accepted_questions.json"
            )
            question_type_counts: Dict[str, int] = {}
            for question in accepted_questions:
                question_type = question.question_type.value
                question_type_counts[question_type] = question_type_counts.get(question_type, 0) + 1

            write_json(
                accepted_path,
                {
                    "record_id": record.record_id,
                    "graph_path": str(graph_path),
                    "questions": [
                        question.model_dump(mode="json")
                        for question in accepted_questions
                    ],
                    "validity_reports": [
                        report.model_dump(mode="json")
                        for report in accepted_reports
                    ],
                    "summary": {
                        "candidate_count": len(candidates),
                        "accepted_count": len(accepted_questions),
                        "rejected_count": len(candidates) - len(accepted_questions),
                        "question_type_counts": question_type_counts,
                    },
                },
            )
            artifacts.accepted_questions_path = str(accepted_path)

            self._end_stage(
                stage,
                StageStatus.OK if accepted_questions else StageStatus.SKIPPED,
                outputs={"validity_path": str(validity_path), "accepted_path": str(accepted_path)},
                metrics={
                    "accepted_count": len(accepted_questions),
                    "rejected_count": len(candidates) - len(accepted_questions),
                },
            )
            stages.append(stage)

            if not accepted_questions:
                # Record skipped due to no accepted questions
                trace_path = self.config.output_dir / "traces" / f"{record.record_id}.rollout_trace.json"
                artifacts.rollout_trace_path = str(trace_path)
                trace = OneRoundTrace(
                    round_id=self.config.round_id,
                    record_id=record.record_id,
                    status=StageStatus.SKIPPED,
                    config=config_snapshot,
                    artifacts=artifacts,
                    stages=stages,
                )
                write_json(trace_path, trace.model_dump(mode="json"))
                return trace

            # Stage 5: Answer questions from current memory
            stage = self._start_stage("answer_questions")
            retriever = CurrentMemoryRetriever(top_k=self.config.answer_top_k, include_summaries=True)
            answer_llm = None
            if not self.config.no_llm_answer:
                answer_llm = OpenAICompatibleLLM(temperature=self.config.answer_temperature)
                if self.config.answer_model:
                    answer_llm.model = self.config.answer_model

            answerer = FrozenAnswerer(
                llm=answer_llm,
                no_llm=self.config.no_llm_answer,
                retriever=retriever,
                model_id=self.config.answer_model or ("no_llm_baseline" if self.config.no_llm_answer else answer_llm.model),
                max_answer_chars=self.config.max_answer_chars,
            )

            # Create sanitized requests
            requests = [
                answer_request_from_generated_question(question.model_dump(mode="json"))
                for question in accepted_questions
            ]

            answer_reports = answerer.batch_answer(current_memory=validated_memory, requests=requests)

            answers_path = self.config.output_dir / "answers" / f"{record.record_id}.answers.json"
            write_json(
                answers_path,
                {
                    "record_id": record.record_id,
                    "current_memory_path": str(memory_path),
                    "answerer_config": {
                        "backend": "no_llm" if self.config.no_llm_answer else "llm",
                        "model": self.config.answer_model or "no_llm_baseline",
                        "temperature": self.config.answer_temperature,
                        "top_k": self.config.answer_top_k,
                    },
                    "answers": [r.model_dump(mode="json") for r in answer_reports],
                    "summary": {
                        "num_questions": len(answer_reports),
                        "answered_count": sum(1 for r in answer_reports if r.answer_status.value == "answered"),
                        "insufficient_evidence_count": sum(
                            1 for r in answer_reports if r.answer_status.value == "insufficient_evidence"
                        ),
                    },
                },
            )
            artifacts.answers_path = str(answers_path)

            self._end_stage(
                stage,
                StageStatus.OK,
                outputs={"answers_path": str(answers_path)},
                metrics={
                    "answered_count": sum(1 for r in answer_reports if r.answer_status.value == "answered"),
                    "insufficient_evidence_count": sum(
                        1 for r in answer_reports if r.answer_status.value == "insufficient_evidence"
                    ),
                },
            )
            stages.append(stage)

            # Stage 6: Score rewards and update weakness book
            stage = self._start_stage("score_rewards")

            # Load or create weakness book
            weakness_book = None
            if self.config.weakness_book_dir:
                weakness_book_path = Path(self.config.weakness_book_dir) / f"{record.record_id}.weakness_book.json"
                if weakness_book_path.exists():
                    weakness_book = MemoryWeaknessBook.load(weakness_book_path, record_id=record.record_id)
                else:
                    weakness_book = MemoryWeaknessBook.empty(record.record_id)
            else:
                weakness_book = MemoryWeaknessBook.empty(record.record_id)

            # Score memory builder
            reward_manager = RewardManager(
                correctness_mode=self.config.correctness_mode,
                weakness_book=weakness_book,
            )

            accepted_question_dicts = [
                question.model_dump(mode="json")
                for question in accepted_questions
            ]

            memory_reward = reward_manager.score_memory_builder(
                current_memory=validated_memory,
                questions=accepted_question_dicts,
                answer_reports=[r.model_dump(mode="json") for r in answer_reports],
                oracle_graph=oracle_loader,
                split=self.config.split,
            )

            # Update weakness book
            weakness_updates = weakness_book.update_from_reward_report(
                memory_reward,
                round_id=self.config.round_id,
            )
            memory_reward.weakness_updates = [w.model_dump(mode="json") for w in weakness_updates]

            # Write reward report
            reward_path = self.config.output_dir / "rewards" / f"{record.record_id}.reward_report.json"
            write_json(reward_path, memory_reward.model_dump(mode="json"))
            artifacts.reward_report_path = str(reward_path)

            # Write weakness book
            weakness_book_path = (
                self.config.output_dir / "weakness_books" / f"{record.record_id}.weakness_book.json"
            )
            weakness_book.save(weakness_book_path)
            artifacts.weakness_book_path = str(weakness_book_path)

            # Optionally score question agent
            qa_reward = None
            if self.config.write_question_agent_reward:
                qa_reward = reward_manager.score_question_agent(
                    questions=accepted_question_dicts,
                    validity_reports=[r.model_dump(mode="json") for r in accepted_reports],
                    answer_reports=[r.model_dump(mode="json") for r in answer_reports],
                    oracle_graph=oracle_loader,
                )

                qa_reward_path = (
                    self.config.output_dir / "rewards" / f"{record.record_id}.question_agent_reward_report.json"
                )
                write_json(qa_reward_path, qa_reward.model_dump(mode="json"))
                artifacts.question_agent_reward_report_path = str(qa_reward_path)

            self._end_stage(
                stage,
                StageStatus.OK,
                outputs={
                    "reward_path": str(reward_path),
                    "weakness_book_path": str(weakness_book_path),
                },
                metrics={
                    "memory_update_reward": memory_reward.update_reward,
                    "memory_total_reward": memory_reward.total_reward,
                    "question_agent_reward": qa_reward.total_reward if qa_reward else None,
                    "active_weakness_count": len(weakness_book.get_active()),
                    "new_weakness_count": len(weakness_updates),
                },
            )
            stages.append(stage)

            # Stage 7: Write rollout trace
            stage = self._start_stage("write_trace")

            trace_path = self.config.output_dir / "traces" / f"{record.record_id}.rollout_trace.json"
            artifacts.rollout_trace_path = str(trace_path)
            self._end_stage(stage, StageStatus.OK, outputs={"trace_path": str(trace_path)})
            stages.append(stage)

            trace = OneRoundTrace(
                round_id=self.config.round_id,
                record_id=record.record_id,
                status=StageStatus.OK,
                config=config_snapshot,
                artifacts=artifacts,
                stages=stages,
                metrics={
                    "memory_total_reward": memory_reward.total_reward,
                    "memory_update_reward": memory_reward.update_reward,
                    "memory_monitoring_score": memory_reward.monitoring_score,
                    "question_agent_total_reward": qa_reward.total_reward if qa_reward else None,
                    "candidate_question_count": len(candidates),
                    "accepted_question_count": len(accepted_questions),
                    "answered_count": sum(1 for r in answer_reports if r.answer_status.value == "answered"),
                    "insufficient_evidence_count": sum(
                        1 for r in answer_reports if r.answer_status.value == "insufficient_evidence"
                    ),
                    "active_weakness_count": len(weakness_book.get_active()),
                    "new_weakness_count": len(weakness_updates),
                },
            )

            write_json(trace_path, trace.model_dump(mode="json"))

            return trace

        except Exception as e:
            # Mark last stage as failed
            if stages and stages[-1].status == StageStatus.RUNNING:
                self._end_stage(stages[-1], StageStatus.FAILED, error=str(e))
            elif "stage" in locals() and stage.status == StageStatus.RUNNING:
                self._end_stage(stage, StageStatus.FAILED, error=str(e))
                stages.append(stage)

            # Write failed trace
            trace_path = self.config.output_dir / "traces" / f"{record.record_id}.rollout_trace.json"
            artifacts.rollout_trace_path = str(trace_path)
            trace = OneRoundTrace(
                round_id=self.config.round_id,
                record_id=record.record_id,
                status=StageStatus.FAILED,
                config=config_snapshot,
                artifacts=artifacts,
                stages=stages,
                error=str(e),
            )

            write_json(trace_path, trace.model_dump(mode="json"))

            return trace

    def _start_stage(self, name: str) -> StageTrace:
        """Start a new stage."""
        return StageTrace(
            name=name,
            status=StageStatus.RUNNING,
            started_at=datetime.utcnow().isoformat(),
        )

    def _end_stage(
        self,
        stage: StageTrace,
        status: StageStatus,
        outputs: Dict[str, Any] | None = None,
        metrics: Dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """End a stage."""
        stage.status = status
        stage.ended_at = datetime.utcnow().isoformat()
        if stage.started_at:
            start = datetime.fromisoformat(stage.started_at)
            end = datetime.fromisoformat(stage.ended_at)
            stage.duration_seconds = (end - start).total_seconds()
        if outputs:
            stage.outputs = outputs
        if metrics:
            stage.metrics = metrics
        if error:
            stage.error = error

    def _trace_to_manifest_line(self, trace: OneRoundTrace) -> Dict[str, Any]:
        """Convert trace to manifest line."""
        return {
            "round_id": trace.round_id,
            "record_id": trace.record_id,
            "status": trace.status.value,
            "graph_path": trace.artifacts.graph_path,
            "current_memory_path": trace.artifacts.current_memory_path,
            "accepted_questions_path": trace.artifacts.accepted_questions_path,
            "answers_path": trace.artifacts.answers_path,
            "reward_report_path": trace.artifacts.reward_report_path,
            "question_agent_reward_report_path": trace.artifacts.question_agent_reward_report_path,
            "weakness_book_path": trace.artifacts.weakness_book_path,
            "rollout_trace_path": trace.artifacts.rollout_trace_path,
            "memory_update_reward": trace.metrics.get("memory_update_reward"),
            "question_agent_reward": trace.metrics.get("question_agent_total_reward"),
            "accepted_question_count": trace.metrics.get("accepted_question_count"),
        }
