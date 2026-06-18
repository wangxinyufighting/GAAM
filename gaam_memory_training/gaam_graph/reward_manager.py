"""
Reward Manager module.

Computes rewards for Memory Builder and Question Agent.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from gaam_graph.memory_schema import validate_current_memory
from gaam_graph.oracle_graph_loader import OracleGraphLoader
from gaam_graph.reward_schema import (
    FailureType,
    MemoryRewardReport,
    QuestionAgentRewardReport,
    QuestionRewardItem,
    RewardComponent,
    RewardTarget,
)


# Default component weights for Memory Builder
DEFAULT_MEMORY_BUILDER_WEIGHTS = {
    "answer_utility": 0.25,
    "oracle_coverage": 0.20,
    "groundedness": 0.15,
    "non_redundancy": 0.15,
    "compression_balance": 0.10,
    "abstraction_quality": 0.10,
    "format_safety": 0.05,
}

# Default component weights for Question Agent
DEFAULT_QUESTION_AGENT_WEIGHTS = {
    "oracle_validity": 0.35,
    "adversarial_success": 0.25,
    "coverage_gain": 0.15,
    "diagnostic_value": 0.15,
    "diversity": 0.10,
}


def normalize_for_reward(text: str) -> str:
    """
    Normalize text for reward scoring.

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation except spaces
    text = re.sub(r'[^\w\s]', ' ', text)

    # Collapse whitespace
    text = ' '.join(text.split())

    return text


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization for scoring.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens
    """
    normalized = normalize_for_reward(text)
    return normalized.split()


def token_f1(prediction: str, expected: str) -> float:
    """
    Compute token-level F1 score.

    Args:
        prediction: Predicted text
        expected: Expected text

    Returns:
        F1 score in [0, 1]
    """
    pred_tokens = set(tokenize(prediction))
    exp_tokens = set(tokenize(expected))

    if not exp_tokens:
        return 1.0 if not pred_tokens else 0.0

    if not pred_tokens:
        return 0.0

    intersection = pred_tokens & exp_tokens
    if not intersection:
        return 0.0

    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(exp_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def token_jaccard(a: str, b: str) -> float:
    """
    Compute token-level Jaccard similarity.

    Args:
        a: First text
        b: Second text

    Returns:
        Jaccard similarity in [0, 1]
    """
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))

    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union) if union else 0.0


def score_answer_correctness(prediction: str, expected_answer: str) -> float:
    """
    Score answer correctness using heuristics.

    Args:
        prediction: Predicted answer
        expected_answer: Expected answer

    Returns:
        Correctness score in [0, 1]
    """
    if not expected_answer:
        return 0.0

    if not prediction:
        return 0.0

    pred_norm = normalize_for_reward(prediction)
    exp_norm = normalize_for_reward(expected_answer)

    # Exact match
    if pred_norm == exp_norm:
        return 1.0

    # Containment (non-trivial)
    if len(exp_norm) >= 10:  # Non-trivial threshold
        if exp_norm in pred_norm or pred_norm in exp_norm:
            return 0.75

    # Token F1
    f1 = token_f1(prediction, expected_answer)
    return f1


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


class RewardManager:
    """
    Computes rewards for Memory Builder and Question Agent.

    Uses deterministic heuristic scoring by default.
    """

    def __init__(
        self,
        *,
        correctness_mode: str = "heuristic",
        weights: Dict[str, float] | None = None,
        weakness_book: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        """
        Initialize RewardManager.

        Args:
            correctness_mode: "heuristic" or "llm_judge"
            weights: Custom component weights
            weakness_book: MemoryWeaknessBook instance
            llm: LLM client for judge mode
        """
        self.correctness_mode = correctness_mode
        self.weights = weights or DEFAULT_MEMORY_BUILDER_WEIGHTS.copy()
        self.weakness_book = weakness_book
        self.llm = llm

    def score_memory_builder(
        self,
        *,
        current_memory: dict,
        questions: List[dict],
        answer_reports: List[dict],
        oracle_graph: OracleGraphLoader,
        split: str = "train",
    ) -> MemoryRewardReport:
        """
        Score Memory Builder output.

        Args:
            current_memory: CurrentMemory artifact
            questions: List of generated questions
            answer_reports: List of answer reports from Frozen Answerer
            oracle_graph: Oracle graph loader
            split: "train", "heldout", or "diagnostic"

        Returns:
            MemoryRewardReport with scores and diagnostics
        """
        record_id = current_memory.get("record_id", "unknown")

        # Build question->answer map
        answer_map = {a["question_id"]: a for a in answer_reports}

        # Score each question. Missing answers are explicit failures, not
        # silently dropped, because otherwise incomplete answer runs inflate
        # memory reward.
        question_items = []
        for q in questions:
            q_id = q["question_id"]
            answer = answer_map.get(q_id) or {
                "question_id": q_id,
                "prediction": "",
                "answer_status": "insufficient_evidence",
                "supporting_memory_ids": [],
                "supporting_evidence": [],
            }

            item = self._score_question_item(
                question=q,
                answer_report=answer,
                current_memory=current_memory,
                oracle_graph=oracle_graph,
            )
            question_items.append(item)

        # Compute component scores
        components = self._compute_memory_components(
            question_items=question_items,
            current_memory=current_memory,
            oracle_graph=oracle_graph,
        )

        # Aggregate scores
        total_reward = sum(c.weighted_score for c in components)
        total_reward = clamp(total_reward)

        # For MVP, monitoring score equals answer utility if only one split
        monitoring_score = total_reward

        # Failure summary
        failure_summary: Dict[str, int] = {}
        for item in question_items:
            for failure in item.failure_types:
                failure_key = failure.value if isinstance(failure, FailureType) else str(failure)
                failure_summary[failure_key] = failure_summary.get(failure_key, 0) + 1

        for failure in self._classify_global_memory_failures(components):
            failure_summary[failure.value] = failure_summary.get(failure.value, 0) + 1

        return MemoryRewardReport(
            record_id=record_id,
            target=RewardTarget.MEMORY_BUILDER,
            total_reward=total_reward,
            update_reward=total_reward,
            monitoring_score=monitoring_score,
            components=components,
            question_items=question_items,
            failure_summary=failure_summary,
            weakness_updates=[],
            config={
                "correctness_mode": self.correctness_mode,
                "weights": self.weights,
                "split": split,
            },
        )

    def _score_question_item(
        self,
        *,
        question: dict,
        answer_report: dict,
        current_memory: dict,
        oracle_graph: OracleGraphLoader,
    ) -> QuestionRewardItem:
        """Score a single question-answer pair."""
        q_id = question["question_id"]
        record_id = question["record_id"]
        q_type = question.get("question_type", "other")
        expected_answer = question.get("answer", "")
        prediction = answer_report.get("prediction", "")
        answer_status = answer_report.get("answer_status", "")

        # Correctness
        correctness = score_answer_correctness(prediction, expected_answer)

        # Evidence support
        evidence_support = self._score_evidence_support(
            expected_answer=expected_answer,
            supporting_evidence=answer_report.get("supporting_evidence", []),
        )

        # Citation quality
        citation_quality = self._score_citation_quality(
            answer_report=answer_report,
            current_memory=current_memory,
        )

        # Oracle coverage
        oracle_coverage = self._score_oracle_coverage(
            question=question,
            current_memory=current_memory,
            oracle_graph=oracle_graph,
        )

        # Classify failures
        failure_types = classify_question_failure(
            question=question,
            answer_report=answer_report,
            correctness=correctness,
            evidence_support=evidence_support,
            oracle_coverage=oracle_coverage,
            current_memory=current_memory,
            oracle_graph=oracle_graph,
        )

        return QuestionRewardItem(
            question_id=q_id,
            record_id=record_id,
            question_type=q_type,
            expected_answer=expected_answer,
            prediction=prediction,
            answer_status=answer_status,
            correctness=correctness,
            evidence_support=evidence_support,
            citation_quality=citation_quality,
            diagnostic_value=1.0 if failure_types else 0.0,
            failure_types=failure_types,
            supporting_oracle_node_ids=question.get("supporting_node_ids", []),
            supporting_memory_ids=answer_report.get("supporting_memory_ids", []),
            related_session_ids=question.get("supporting_session_ids", []),
        )

    def _score_evidence_support(
        self,
        *,
        expected_answer: str,
        supporting_evidence: List[dict],
    ) -> float:
        """Score whether evidence supports expected answer."""
        if not expected_answer:
            return 0.0

        if not supporting_evidence:
            return 0.0

        # Join all evidence content
        evidence_text = " ".join([ev.get("content", "") for ev in supporting_evidence])

        # Token F1 between evidence and expected answer
        return token_f1(evidence_text, expected_answer)

    def _score_citation_quality(
        self,
        *,
        answer_report: dict,
        current_memory: dict,
    ) -> float:
        """Score citation quality."""
        answer_status = answer_report.get("answer_status", "")
        cited_ids = answer_report.get("supporting_memory_ids", [])
        supporting_evidence = answer_report.get("supporting_evidence", [])

        # Get all valid memory IDs from current memory
        valid_ids = set()
        for node in current_memory.get("memory_graph", {}).get("nodes", []):
            valid_ids.add(node.get("id", ""))
        for key in current_memory.get("memory_summaries", {}):
            valid_ids.add(f"summary_{key}")

        # Check if answered
        if answer_status != "answered":
            # Insufficient evidence is OK if no relevant memory exists
            if answer_status == "insufficient_evidence" and not supporting_evidence:
                return 1.0
            return 0.5

        # Check all cited IDs are valid
        if cited_ids:
            invalid_count = sum(1 for cid in cited_ids if cid not in valid_ids)
            if invalid_count > 0:
                return 0.0

        # Check citations match evidence
        if cited_ids and supporting_evidence:
            evidence_ids = {ev.get("memory_id", "") for ev in supporting_evidence}
            if set(cited_ids) == evidence_ids:
                return 1.0
            return 0.7

        # Answered but missing citations while evidence exists
        if not cited_ids and supporting_evidence:
            return 0.5

        return 0.8

    def _score_oracle_coverage(
        self,
        *,
        question: dict,
        current_memory: dict,
        oracle_graph: OracleGraphLoader,
    ) -> float:
        """Score oracle coverage in current memory."""
        oracle_node_ids = question.get("supporting_node_ids", [])

        if not oracle_node_ids:
            return 1.0  # No oracle requirement

        # Get oracle node texts
        oracle_texts = []
        for node_id in oracle_node_ids:
            node = oracle_graph.get_node(node_id)
            if node:
                oracle_texts.append(node.get("text", ""))

        if not oracle_texts:
            return 0.0

        # Get all current memory content
        memory_texts = []
        for node in current_memory.get("memory_graph", {}).get("nodes", []):
            if node.get("status") == "active":
                memory_texts.append(node.get("content", ""))
        for summary_text in current_memory.get("memory_summaries", {}).values():
            if summary_text:
                memory_texts.append(summary_text)

        if not memory_texts:
            return 0.0

        # For each oracle node, find best match in memory
        scores = []
        for oracle_text in oracle_texts:
            best_score = max(
                token_f1(oracle_text, mem_text) for mem_text in memory_texts
            )
            scores.append(best_score)

        return sum(scores) / len(scores) if scores else 0.0

    def _compute_memory_components(
        self,
        *,
        question_items: List[QuestionRewardItem],
        current_memory: dict,
        oracle_graph: OracleGraphLoader,
    ) -> List[RewardComponent]:
        """Compute all memory builder components."""
        components = []

        # Answer utility
        if question_items:
            answer_scores = []
            for item in question_items:
                q_score = (
                    0.55 * item.correctness
                    + 0.25 * item.evidence_support
                    + 0.10 * item.citation_quality
                    + 0.10 * (1.0 if item.answer_status == "answered" else 0.0)
                )
                answer_scores.append(q_score)
            answer_utility = sum(answer_scores) / len(answer_scores)
        else:
            answer_utility = 0.0

        components.append(RewardComponent(
            name="answer_utility",
            score=answer_utility,
            weight=self.weights.get("answer_utility", 0.25),
            weighted_score=answer_utility * self.weights.get("answer_utility", 0.25),
            rationale=f"{len([i for i in question_items if i.correctness >= 0.7])}/{len(question_items)} questions answered correctly",
        ))

        # Oracle coverage
        if question_items:
            coverage_scores = [
                self._score_oracle_coverage(
                    question={"supporting_node_ids": item.supporting_oracle_node_ids},
                    current_memory=current_memory,
                    oracle_graph=oracle_graph,
                )
                for item in question_items
            ]
            oracle_coverage = sum(coverage_scores) / len(coverage_scores)
        else:
            oracle_coverage = 1.0

        components.append(RewardComponent(
            name="oracle_coverage",
            score=oracle_coverage,
            weight=self.weights.get("oracle_coverage", 0.20),
            weighted_score=oracle_coverage * self.weights.get("oracle_coverage", 0.20),
        ))

        # Groundedness
        groundedness = self._score_groundedness(current_memory)
        components.append(RewardComponent(
            name="groundedness",
            score=groundedness,
            weight=self.weights.get("groundedness", 0.15),
            weighted_score=groundedness * self.weights.get("groundedness", 0.15),
        ))

        # Non-redundancy
        non_redundancy = self._score_non_redundancy(current_memory)
        components.append(RewardComponent(
            name="non_redundancy",
            score=non_redundancy,
            weight=self.weights.get("non_redundancy", 0.15),
            weighted_score=non_redundancy * self.weights.get("non_redundancy", 0.15),
        ))

        # Compression balance
        compression_balance = self._score_compression_balance(
            current_memory=current_memory,
            oracle_coverage=oracle_coverage,
            non_redundancy=non_redundancy,
        )
        components.append(RewardComponent(
            name="compression_balance",
            score=compression_balance,
            weight=self.weights.get("compression_balance", 0.10),
            weighted_score=compression_balance * self.weights.get("compression_balance", 0.10),
        ))

        # Abstraction quality
        abstraction_quality = self._score_abstraction_quality(current_memory)
        components.append(RewardComponent(
            name="abstraction_quality",
            score=abstraction_quality,
            weight=self.weights.get("abstraction_quality", 0.10),
            weighted_score=abstraction_quality * self.weights.get("abstraction_quality", 0.10),
        ))

        # Format safety
        format_safety = self._score_format_safety(current_memory)
        components.append(RewardComponent(
            name="format_safety",
            score=format_safety,
            weight=self.weights.get("format_safety", 0.05),
            weighted_score=format_safety * self.weights.get("format_safety", 0.05),
        ))

        return components

    def _score_groundedness(self, current_memory: dict) -> float:
        """Score groundedness of memory nodes."""
        nodes = current_memory.get("memory_graph", {}).get("nodes", [])
        active_nodes = [n for n in nodes if n.get("status") == "active" and n.get("type") != "summary"]

        if not active_nodes:
            return 1.0

        scores = []
        for node in active_nodes:
            # Check provenance exists
            has_source_ids = bool(
                node.get("source_session_ids") or
                node.get("source_turn_ids") or
                node.get("source_event_ids")
            )

            if has_source_ids:
                scores.append(0.7)  # Has provenance
            else:
                scores.append(0.3)  # Missing provenance

        return sum(scores) / len(scores) if scores else 1.0

    def _score_non_redundancy(self, current_memory: dict) -> float:
        """Score non-redundancy (lack of duplicates)."""
        nodes = current_memory.get("memory_graph", {}).get("nodes", [])
        active_nodes = [n for n in nodes if n.get("status") == "active"]

        if len(active_nodes) < 2:
            return 1.0

        # Compare all pairs
        duplicate_count = 0
        total_pairs = 0

        for i, node_a in enumerate(active_nodes):
            for node_b in active_nodes[i+1:]:
                total_pairs += 1
                content_a = node_a.get("content", "")
                content_b = node_b.get("content", "")

                # Check similarity
                similarity = token_jaccard(content_a, content_b)
                if similarity >= 0.85:
                    duplicate_count += 1

        if total_pairs == 0:
            return 1.0

        redundancy_rate = duplicate_count / total_pairs
        return 1.0 - redundancy_rate

    def _score_compression_balance(
        self,
        *,
        current_memory: dict,
        oracle_coverage: float,
        non_redundancy: float,
    ) -> float:
        """Score compression balance (not too much, not too little)."""
        redundancy_rate = 1.0 - non_redundancy

        # Over-compression: low coverage and redundancy rate normal
        if oracle_coverage < 0.45:
            return 0.2

        # Excessive redundancy
        if redundancy_rate > 0.35:
            return 0.4

        # Balanced
        score = 0.5 + oracle_coverage - 0.5 * redundancy_rate
        return clamp(score)

    def _score_abstraction_quality(self, current_memory: dict) -> float:
        """Score abstraction quality."""
        nodes = current_memory.get("memory_graph", {}).get("nodes", [])

        # Count abstract nodes
        abstract_nodes = [
            n for n in nodes
            if n.get("type") in {"abstract", "abstract_memory", "summary"} and n.get("status") == "active"
        ]

        fact_nodes = [
            n for n in nodes
            if n.get("type") == "fact" and n.get("status") == "active"
        ]

        # Abstract node validity
        abstract_node_validity = 0.8 if abstract_nodes else 0.5

        # Summary coverage
        summaries = current_memory.get("memory_summaries", {})
        has_summaries = bool(summaries and any(v for v in summaries.values()))
        summary_coverage = 0.8 if has_summaries and fact_nodes else 0.6

        # Abstraction groundedness (check source IDs)
        if abstract_nodes:
            grounded_count = sum(
                1 for n in abstract_nodes
                if n.get("source_session_ids") or n.get("source_turn_ids") or n.get("source_event_ids")
            )
            abstraction_groundedness = grounded_count / len(abstract_nodes)
        else:
            abstraction_groundedness = 0.5

        # Weighted combination
        score = (
            0.40 * abstract_node_validity
            + 0.30 * summary_coverage
            + 0.30 * abstraction_groundedness
        )
        return score

    def _score_format_safety(self, current_memory: dict) -> float:
        """Score format safety."""
        try:
            validate_current_memory(current_memory, strict=True)
            return 1.0
        except Exception:
            return 0.0

    def _classify_global_memory_failures(
        self,
        components: List[RewardComponent],
    ) -> List[FailureType]:
        """Classify memory-level failures from component scores."""
        by_name = {component.name: component.score for component in components}
        failures = []

        if by_name.get("non_redundancy", 1.0) < 0.65:
            failures.append(FailureType.REDUNDANCY_NOISE)

        if (
            by_name.get("compression_balance", 1.0) <= 0.25
            and by_name.get("oracle_coverage", 1.0) < 0.45
        ):
            failures.append(FailureType.OVER_COMPRESSION)

        if by_name.get("abstraction_quality", 1.0) < 0.5:
            failures.append(FailureType.WRONG_ABSTRACTION)

        if by_name.get("format_safety", 1.0) < 1.0:
            failures.append(FailureType.FORMAT_SAFETY_ERROR)

        return failures

    def score_question_agent(
        self,
        *,
        questions: List[dict],
        validity_reports: List[dict],
        answer_reports: List[dict],
        oracle_graph: OracleGraphLoader,
        coverage_before: dict | None = None,
        coverage_after: dict | None = None,
    ) -> QuestionAgentRewardReport:
        """
        Score Question Agent output.

        Args:
            questions: Generated questions
            validity_reports: Oracle validity reports
            answer_reports: Answer reports from Frozen Answerer
            oracle_graph: Oracle graph loader
            coverage_before: Coverage before question generation
            coverage_after: Coverage after question generation

        Returns:
            QuestionAgentRewardReport
        """
        record_id = questions[0]["record_id"] if questions else "unknown"

        # Build maps
        validity_map = {r["question_id"]: r for r in validity_reports}
        answer_map = {a["question_id"]: a for a in answer_reports}

        # Per-question scores
        per_question_scores = []
        oracle_valid_questions = []

        for q in questions:
            q_id = q["question_id"]
            validity = validity_map.get(q_id, {})
            answer = answer_map.get(q_id)

            is_valid = validity.get("verdict") == "accept"
            if is_valid:
                oracle_valid_questions.append(q)

            per_question_scores.append({
                "question_id": q_id,
                "oracle_valid": is_valid,
                "correctness": score_answer_correctness(
                    answer.get("prediction", "") if answer else "",
                    q.get("answer", ""),
                ) if answer else 0.0,
            })

        # Oracle validity
        oracle_validity = sum(
            1 for s in per_question_scores if s["oracle_valid"]
        ) / len(per_question_scores) if per_question_scores else 0.0

        # Adversarial success
        adversarial_scores = [
            1.0 - s["correctness"]
            for s in per_question_scores
            if s["oracle_valid"]
        ]
        adversarial_success = sum(adversarial_scores) / len(adversarial_scores) if adversarial_scores else 0.0

        # Coverage gain
        if coverage_before and coverage_after:
            coverage_gain = 0.5  # Placeholder
        else:
            unique_node_ids = set()
            for q in oracle_valid_questions:
                unique_node_ids.update(q.get("supporting_node_ids", []))
            total_oracle_nodes = oracle_graph.node_count
            coverage_gain = len(unique_node_ids) / total_oracle_nodes if total_oracle_nodes else 0.0

        # Diagnostic value
        diagnostic_value = 0.5  # Placeholder for MVP

        # Diversity
        diversity = self._score_question_diversity(questions)

        # Components
        weights = DEFAULT_QUESTION_AGENT_WEIGHTS
        components = [
            RewardComponent(
                name="oracle_validity",
                score=oracle_validity,
                weight=weights["oracle_validity"],
                weighted_score=oracle_validity * weights["oracle_validity"],
            ),
            RewardComponent(
                name="adversarial_success",
                score=adversarial_success,
                weight=weights["adversarial_success"],
                weighted_score=adversarial_success * weights["adversarial_success"],
            ),
            RewardComponent(
                name="coverage_gain",
                score=coverage_gain,
                weight=weights["coverage_gain"],
                weighted_score=coverage_gain * weights["coverage_gain"],
            ),
            RewardComponent(
                name="diagnostic_value",
                score=diagnostic_value,
                weight=weights["diagnostic_value"],
                weighted_score=diagnostic_value * weights["diagnostic_value"],
            ),
            RewardComponent(
                name="diversity",
                score=diversity,
                weight=weights["diversity"],
                weighted_score=diversity * weights["diversity"],
            ),
        ]

        total_reward = sum(c.weighted_score for c in components)

        return QuestionAgentRewardReport(
            record_id=record_id,
            target=RewardTarget.QUESTION_AGENT,
            total_reward=clamp(total_reward),
            components=components,
            per_question_scores=per_question_scores,
            coverage_gain=coverage_gain,
            adversarial_success_rate=adversarial_success,
            diversity_score=diversity,
            config={"weights": weights},
        )

    def _score_question_diversity(self, questions: List[dict]) -> float:
        """Score question diversity."""
        if not questions:
            return 0.0

        # Question types
        q_types = set(q.get("question_type", "other") for q in questions)
        type_diversity = len(q_types) / 5.0  # Target ~5 types

        # Sessions
        all_sessions = set()
        for q in questions:
            all_sessions.update(q.get("supporting_session_ids", []))
        session_diversity = min(len(all_sessions) / 3.0, 1.0)  # Target ~3 sessions

        return (type_diversity + session_diversity) / 2.0


def classify_question_failure(
    *,
    question: dict,
    answer_report: dict,
    correctness: float,
    evidence_support: float,
    oracle_coverage: float,
    current_memory: dict,
    oracle_graph: OracleGraphLoader,
) -> List[FailureType]:
    """
    Classify question failure types.

    Args:
        question: Question dict
        answer_report: Answer report dict
        correctness: Correctness score
        evidence_support: Evidence support score
        oracle_coverage: Oracle coverage score
        current_memory: CurrentMemory dict
        oracle_graph: Oracle graph loader

    Returns:
        List of failure types
    """
    failures = []
    answer_status = answer_report.get("answer_status", "")
    q_type = question.get("question_type", "other")
    supporting_session_ids = question.get("supporting_session_ids", [])

    # Insufficient evidence
    if answer_status == "insufficient_evidence":
        failures.append(FailureType.ANSWERER_INSUFFICIENT_EVIDENCE)

    # Missing fact/abstraction (low correctness + low coverage)
    if correctness < 0.5 and oracle_coverage < 0.5:
        # Check oracle node types
        oracle_node_ids = question.get("supporting_node_ids", [])
        has_abstract = False
        for node_id in oracle_node_ids:
            node = oracle_graph.get_node(node_id)
            if node and node.get("type") in ["abstract_memory", "topic"]:
                has_abstract = True
                break

        if has_abstract:
            failures.append(FailureType.MISSING_ABSTRACTION)
        else:
            failures.append(FailureType.MISSING_FACT)

    # Wrong fact (high coverage but low correctness)
    if correctness < 0.5 and oracle_coverage >= 0.7:
        failures.append(FailureType.WRONG_FACT)

    # Missing multi-session link
    if len(set(supporting_session_ids)) > 1 and correctness < 0.7:
        failures.append(FailureType.MISSING_MULTI_SESSION_LINK)

    # Temporal error
    if q_type == "temporal" and correctness < 0.7:
        failures.append(FailureType.TEMPORAL_ERROR)

    # Contradiction/update error
    if q_type == "contradiction_update" and correctness < 0.7:
        failures.append(FailureType.CONTRADICTION_UPDATE_ERROR)

    # Unsupported answer
    if evidence_support < 0.3 and correctness >= 0.7:
        failures.append(FailureType.UNSUPPORTED_ANSWER)

    return failures
