"""
Answer agent module.

Implements the Frozen Answerer for answering questions from CurrentMemory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .answer_schema import (
    AnswerReport,
    AnswerRequest,
    AnswerStatus,
    RetrievedMemoryEvidence,
    assert_no_answer_input_leakage,
    validate_answer_request,
    validate_supporting_memory_ids,
)
from .memory_retriever import CurrentMemoryRetriever
from .memory_schema import validate_current_memory


class FrozenAnswerer:
    """
    Frozen Answerer for answering questions using CurrentMemory.

    The answerer is "frozen" - not trained by Memory Builder rewards.
    It acts as a stable probe for current memory quality.
    """

    def __init__(
        self,
        *,
        llm: Optional[Any] = None,
        no_llm: bool = False,
        prompt_path: Optional[str | Path] = None,
        retriever: Optional[CurrentMemoryRetriever] = None,
        model_id: str = "",
        max_answer_chars: int = 500,
    ) -> None:
        """
        Initialize Frozen Answerer.

        Args:
            llm: Optional LLM client for answer generation
            no_llm: If True, use deterministic no-LLM mode
            prompt_path: Optional path to answer generation prompt
            retriever: Optional CurrentMemoryRetriever instance
            model_id: Model identifier for tracking
            max_answer_chars: Max characters for no-LLM baseline predictions
        """
        self.llm = llm
        self.no_llm = no_llm
        self.prompt_path = prompt_path or (
            Path(__file__).parent / "prompts" / "answer_from_memory.txt"
        )
        self.retriever = retriever or CurrentMemoryRetriever()
        self.model_id = model_id or ("no_llm_baseline" if no_llm else "llm")
        self.max_answer_chars = max_answer_chars

    def answer(
        self,
        *,
        current_memory: dict,
        request: AnswerRequest,
    ) -> AnswerReport:
        """
        Answer a question using current memory.

        Args:
            current_memory: CurrentMemory dict
            request: AnswerRequest with question

        Returns:
            AnswerReport with prediction
        """
        # Validate inputs
        request = validate_answer_request(request.model_dump(), strict=True)
        validate_current_memory(current_memory, strict=True)

        # Retrieve evidence
        evidence = self.retriever.retrieve(
            current_memory=current_memory,
            question=request.question,
        )

        # Generate answer
        if self.no_llm:
            return self._answer_no_llm(request, evidence)
        else:
            return self._answer_with_llm(request, evidence, current_memory)

    def batch_answer(
        self,
        *,
        current_memory: dict,
        requests: List[AnswerRequest],
    ) -> List[AnswerReport]:
        """
        Answer multiple questions.

        Args:
            current_memory: CurrentMemory dict
            requests: List of AnswerRequest objects

        Returns:
            List of AnswerReport objects
        """
        reports = []
        for request in requests:
            report = self.answer(
                current_memory=current_memory,
                request=request,
            )
            reports.append(report)

        return reports

    def _answer_no_llm(
        self,
        request: AnswerRequest,
        evidence: List[RetrievedMemoryEvidence],
    ) -> AnswerReport:
        """
        Generate answer in no-LLM mode (deterministic baseline).

        Args:
            request: AnswerRequest
            evidence: Retrieved evidence

        Returns:
            AnswerReport
        """
        if not evidence:
            # No evidence available
            return AnswerReport(
                record_id=request.record_id,
                question_id=request.question_id,
                question=request.question,
                prediction="",
                answer_status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                supporting_memory_ids=[],
                supporting_evidence=[],
                rationale="No relevant memory evidence found",
                model_info={"model": "no_llm_baseline"},
            )

        # Use first evidence as answer
        top_evidence = evidence[0]
        prediction = top_evidence.content

        if self.max_answer_chars > 0 and len(prediction) > self.max_answer_chars:
            prediction = prediction[:self.max_answer_chars].rstrip() + "..."

        return AnswerReport(
            record_id=request.record_id,
            question_id=request.question_id,
            question=request.question,
            prediction=prediction,
            answer_status=AnswerStatus.ANSWERED,
            confidence=top_evidence.score,
            supporting_memory_ids=[ev.memory_id for ev in evidence[:3]],
            supporting_evidence=evidence[:3],
            rationale="Answer from top retrieved memory evidence",
            model_info={"model": "no_llm_baseline"},
        )

    def _answer_with_llm(
        self,
        request: AnswerRequest,
        evidence: List[RetrievedMemoryEvidence],
        current_memory: dict,
    ) -> AnswerReport:
        """
        Generate answer using LLM.

        Args:
            request: AnswerRequest
            evidence: Retrieved evidence
            current_memory: CurrentMemory dict

        Returns:
            AnswerReport
        """
        if not self.llm:
            raise ValueError("LLM client is required for LLM mode. Use no_llm=True or provide llm.")

        # Load prompt
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")

        system_prompt = self.prompt_path.read_text(encoding="utf-8")

        # Build payload
        payload = self._build_prompt_payload(request, evidence, current_memory)

        # Validate no forbidden keys
        assert_no_answer_input_leakage(payload)

        user_message = json.dumps(payload, ensure_ascii=False, indent=2)

        # Call LLM
        response = self.llm.chat_json(system=system_prompt, user=user_message)

        # Parse response
        report = self._parse_llm_response(request, response, evidence)

        return report

    def _build_prompt_payload(
        self,
        request: AnswerRequest,
        evidence: List[RetrievedMemoryEvidence],
        current_memory: dict,
    ) -> dict:
        """
        Build prompt payload for LLM.

        Args:
            request: AnswerRequest
            evidence: Retrieved evidence
            current_memory: CurrentMemory dict

        Returns:
            Prompt payload dict (validated for no leakage)
        """
        # Get memory summaries
        summaries = current_memory.get("memory_summaries", {})

        # Format retrieved memory
        retrieved_memory = []
        for ev in evidence:
            mem_dict = {
                "memory_id": ev.memory_id,
                "memory_type": ev.memory_type,
                "content": ev.content,
                "source_session_ids": ev.source_session_ids,
                "source_turn_ids": ev.source_turn_ids,
                "source_event_ids": ev.source_event_ids,
            }
            retrieved_memory.append(mem_dict)

        payload = {
            "record_id": request.record_id,
            "question_id": request.question_id,
            "question": request.question,
            "memory_summaries": {
                "user_profile": summaries.get("user_profile", ""),
                "stable_preferences": summaries.get("stable_preferences", ""),
                "active_plans": summaries.get("active_plans", ""),
                "recent_changes": summaries.get("recent_changes", ""),
                "cross_session_abstractions": summaries.get("cross_session_abstractions", ""),
            },
            "retrieved_memory": retrieved_memory,
        }

        return payload

    def _parse_llm_response(
        self,
        request: AnswerRequest,
        response: dict,
        evidence: List[RetrievedMemoryEvidence],
    ) -> AnswerReport:
        """
        Parse LLM response into AnswerReport.

        Args:
            request: AnswerRequest
            response: LLM response dict
            evidence: Retrieved evidence

        Returns:
            AnswerReport
        """
        try:
            assert_no_answer_input_leakage(response)
            prediction = response.get("prediction", "")
            answer_status = response.get("answer_status", AnswerStatus.ANSWERED)
            confidence = response.get("confidence", 0.5)
            supporting_memory_ids = response.get("supporting_memory_ids", [])
            rationale = response.get("rationale", "")

            # Validate answer_status
            if isinstance(answer_status, str):
                try:
                    answer_status = AnswerStatus(answer_status)
                except ValueError:
                    raise ValueError(f"Invalid answer_status: {answer_status}") from None

            if answer_status == AnswerStatus.ANSWERED and not prediction.strip():
                raise ValueError("Answered response must include a non-empty prediction")

            allowed_memory_ids = {ev.memory_id for ev in evidence}
            supporting_memory_ids = validate_supporting_memory_ids(
                supporting_memory_ids,
                allowed_memory_ids=allowed_memory_ids,
            )
            if answer_status == AnswerStatus.ANSWERED and not supporting_memory_ids:
                raise ValueError("Answered response must cite supporting_memory_ids")
            cited_evidence = [
                ev for ev in evidence
                if ev.memory_id in set(supporting_memory_ids)
            ]

            report = AnswerReport(
                record_id=request.record_id,
                question_id=request.question_id,
                question=request.question,
                prediction=prediction,
                answer_status=answer_status,
                confidence=confidence,
                supporting_memory_ids=supporting_memory_ids,
                supporting_evidence=cited_evidence,
                rationale=rationale,
                model_info={"model": self.model_id or "llm"},
            )

            return report

        except Exception as e:
            # Return error report
            return AnswerReport(
                record_id=request.record_id,
                question_id=request.question_id,
                question=request.question,
                prediction="",
                answer_status=AnswerStatus.INVALID_OUTPUT,
                confidence=0.0,
                supporting_memory_ids=[],
                supporting_evidence=[],
                rationale=f"Failed to parse LLM output: {e}",
                model_info={"model": self.model_id or "llm"},
            )
