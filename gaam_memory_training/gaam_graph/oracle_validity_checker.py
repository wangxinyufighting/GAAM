"""
Oracle validity checker module.

Validates that generated questions are grounded in oracle evidence.
"""

from __future__ import annotations

from typing import Dict, List

from .oracle_graph_loader import OracleGraphLoader
from .question_schema import (
    GeneratedQuestion,
    OracleValidityReport,
    OracleValidityVerdict,
    assert_no_question_artifact_leakage,
)
from .utils import normalize_text


class OracleValidityChecker:
    """
    Oracle validity checker for generated questions.

    This is a hard gate that validates questions can be answered from
    cited oracle evidence.
    """

    def __init__(
        self,
        oracle_loader: OracleGraphLoader,
        *,
        min_answer_token_overlap: float = 0.6,
        allow_placeholder_questions: bool = False,
    ) -> None:
        """
        Initialize oracle validity checker.

        Args:
            oracle_loader: OracleGraphLoader for accessing oracle graph
            min_answer_token_overlap: Minimum token overlap for answer grounding
            allow_placeholder_questions: If True, allow no-LLM placeholder questions
        """
        self.oracle_loader = oracle_loader
        self.min_answer_token_overlap = min_answer_token_overlap
        self.allow_placeholder_questions = allow_placeholder_questions

    def validate_question(
        self,
        question: GeneratedQuestion,
        *,
        trajectories: List[dict],
        require_multi_session: bool = False,
        min_sessions: int = 2,
    ) -> OracleValidityReport:
        """
        Validate a single question.

        Args:
            question: GeneratedQuestion to validate
            trajectories: List of trajectory dicts
            require_multi_session: If True, require multi-session evidence
            min_sessions: Minimum sessions for multi-session questions

        Returns:
            OracleValidityReport with verdict and scores
        """
        issues = []
        verdict = OracleValidityVerdict.ACCEPT
        leakage_risk = 0.0

        try:
            assert_no_question_artifact_leakage(question.model_dump())
        except ValueError as exc:
            issues.append(str(exc))
            verdict = OracleValidityVerdict.REJECT
            leakage_risk = 1.0

        # 1. Check required fields
        if not question.question:
            issues.append("Question text is empty")
            verdict = OracleValidityVerdict.REJECT

        if not question.answer:
            issues.append("Answer text is empty")
            verdict = OracleValidityVerdict.REJECT

        if not question.supporting_trajectory_ids:
            issues.append("No supporting trajectory IDs")
            verdict = OracleValidityVerdict.REJECT

        if not question.supporting_node_ids:
            issues.append("No supporting node IDs")
            verdict = OracleValidityVerdict.REJECT

        # 2. Check for placeholder questions
        if not self.allow_placeholder_questions:
            if self._is_placeholder_question(question):
                issues.append("Placeholder question not allowed in accepted output")
                verdict = OracleValidityVerdict.REJECT

        # 3. Validate cited trajectories exist
        trajectory_ids = {t.get("trajectory_id") for t in trajectories}
        for traj_id in question.supporting_trajectory_ids:
            if traj_id not in trajectory_ids:
                issues.append(f"Cited trajectory {traj_id} not found")
                verdict = OracleValidityVerdict.REJECT

        # 4. Validate cited nodes exist in trajectories
        trajectory_nodes = {}
        for traj in trajectories:
            traj_id = traj.get("trajectory_id")
            if traj_id in question.supporting_trajectory_ids:
                for node in traj.get("nodes", []):
                    trajectory_nodes[node.get("id")] = node

        for node_id in question.supporting_node_ids:
            if node_id not in trajectory_nodes:
                issues.append(f"Cited node {node_id} not found in trajectories")
                verdict = OracleValidityVerdict.REJECT

            # Also check node exists in oracle graph
            oracle_node = self.oracle_loader.get_node_or_none(node_id)
            if not oracle_node:
                issues.append(f"Cited node {node_id} not found in oracle graph")
                verdict = OracleValidityVerdict.REJECT

        # 5. Check evidence availability
        has_evidence = False
        for node_id in question.supporting_node_ids:
            node = trajectory_nodes.get(node_id)
            if node and node.get("type") in {"fact", "abstract_memory", "event"}:
                label = node.get("label", "")
                if label:
                    has_evidence = True
                    break

        if not has_evidence:
            issues.append("No evidence node with non-empty label found")
            verdict = OracleValidityVerdict.REJECT

        # 6. Check multi-session requirement
        if require_multi_session:
            session_ids = set()
            for node_id in question.supporting_node_ids:
                node_sessions = self.oracle_loader.get_session_ids_for_node(node_id)
                session_ids.update(node_sessions)

            if len(session_ids) < min_sessions:
                issues.append(
                    f"Multi-session required but only {len(session_ids)} sessions found (need {min_sessions})"
                )
                verdict = OracleValidityVerdict.REJECT

        # 7. Check answer grounding (heuristic)
        grounding_score = 0.0
        if verdict != OracleValidityVerdict.REJECT and question.answer:
            grounding_score = self._compute_answer_grounding(
                answer=question.answer,
                node_ids=question.supporting_node_ids,
                trajectory_nodes=trajectory_nodes,
            )

            if grounding_score < self.min_answer_token_overlap:
                issues.append(
                    f"Answer not sufficiently grounded in evidence (score={grounding_score:.2f}, need {self.min_answer_token_overlap})"
                )
                verdict = OracleValidityVerdict.REVISE

        # Compute scores
        answerability = 1.0 if not issues else 0.5
        grounding = grounding_score
        citation_validity = 1.0 if not any("not found" in issue for issue in issues) else 0.0
        # Build report
        report = OracleValidityReport(
            question_id=question.question_id,
            record_id=question.record_id,
            verdict=verdict,
            answerability=answerability,
            grounding=grounding,
            citation_validity=citation_validity,
            leakage_risk=leakage_risk,
            issues=issues,
            supporting_node_ids=list(question.supporting_node_ids),
            supporting_trajectory_ids=list(question.supporting_trajectory_ids),
            rationale="; ".join(issues) if issues else "Question is oracle-valid",
        )

        return report

    def validate_questions(
        self,
        questions: List[GeneratedQuestion],
        *,
        trajectories: List[dict],
        require_multi_session: bool = False,
        min_sessions: int = 2,
    ) -> List[OracleValidityReport]:
        """
        Validate multiple questions.

        Args:
            questions: List of GeneratedQuestion objects
            trajectories: List of trajectory dicts
            require_multi_session: If True, require multi-session evidence
            min_sessions: Minimum sessions for multi-session questions

        Returns:
            List of OracleValidityReport objects
        """
        reports = []

        for question in questions:
            report = self.validate_question(
                question=question,
                trajectories=trajectories,
                require_multi_session=require_multi_session,
                min_sessions=min_sessions,
            )
            reports.append(report)

        return reports

    def _is_placeholder_question(self, question: GeneratedQuestion) -> bool:
        """Check if question is a no-LLM placeholder."""
        # Check for placeholder text
        placeholder_markers = [
            "What user information is supported by the cited evidence",
            "What information is supported by this graph trajectory",
            "Generated in no-LLM mode",
        ]

        question_lower = question.question.lower()
        reason_lower = question.reason.lower()

        for marker in placeholder_markers:
            if marker.lower() in question_lower or marker.lower() in reason_lower:
                return True

        # Check metadata
        if question.metadata.get("mode") == "no_llm":
            return True

        return False

    def _compute_answer_grounding(
        self,
        answer: str,
        node_ids: List[str],
        trajectory_nodes: Dict[str, dict],
    ) -> float:
        """
        Compute answer grounding score using token overlap.

        Args:
            answer: Answer text
            node_ids: Supporting node IDs
            trajectory_nodes: Dict of node_id -> node dict

        Returns:
            Grounding score between 0 and 1
        """
        # Normalize answer
        answer_normalized = normalize_text(answer)
        answer_tokens = set(answer_normalized.split())

        if not answer_tokens:
            return 0.0

        # Collect evidence text from cited nodes
        evidence_text = []
        for node_id in node_ids:
            node = trajectory_nodes.get(node_id)
            if not node:
                # Try oracle loader
                oracle_node = self.oracle_loader.get_node_or_none(node_id)
                if oracle_node:
                    node = oracle_node

            if node:
                label = node.get("label", "")
                text = node.get("text", "")
                evidence_text.append(label or text)

        evidence_combined = " ".join(evidence_text)
        evidence_normalized = normalize_text(evidence_combined)

        # Check substring match first
        if answer_normalized in evidence_normalized:
            return 1.0

        # Compute token overlap
        evidence_tokens = set(evidence_normalized.split())
        overlap_tokens = answer_tokens & evidence_tokens
        overlap_ratio = len(overlap_tokens) / len(answer_tokens) if answer_tokens else 0.0

        return overlap_ratio
