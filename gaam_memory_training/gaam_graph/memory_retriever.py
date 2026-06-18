"""
Memory retriever module.

Retrieves relevant evidence from CurrentMemory for answering questions.
"""

from __future__ import annotations

from typing import List

from .answer_schema import RetrievedMemoryEvidence
from .utils import normalize_text


# Type weights for scoring
TYPE_WEIGHTS = {
    "fact": 1.15,
    "abstract": 1.10,
    "summary": 0.95,
    "preference": 1.20,
    "plan": 1.15,
    "event": 0.90,
    "entity": 0.75,
    "other": 0.70,
}


class CurrentMemoryRetriever:
    """
    Retriever for CurrentMemory evidence.

    Uses deterministic lexical scoring (no LLM).
    """

    def __init__(
        self,
        *,
        top_k: int = 8,
        include_summaries: bool = True,
        min_score: float = 0.0,
    ) -> None:
        """
        Initialize retriever.

        Args:
            top_k: Number of top memory nodes to retrieve
            include_summaries: If True, include global summaries
            min_score: Minimum score threshold
        """
        self.top_k = top_k
        self.include_summaries = include_summaries
        self.min_score = min_score

    def retrieve(
        self,
        *,
        current_memory: dict,
        question: str,
    ) -> List[RetrievedMemoryEvidence]:
        """
        Retrieve relevant memory evidence for a question.

        Args:
            current_memory: CurrentMemory dict
            question: Question text

        Returns:
            List of RetrievedMemoryEvidence, sorted by score (descending)
        """
        # Normalize question
        question_normalized = normalize_text(question)
        question_tokens = set(question_normalized.split())

        if not question_tokens:
            return []

        # Score memory nodes
        nodes = current_memory.get("memory_graph", {}).get("nodes", [])
        scored_nodes = []

        for node in nodes:
            # Skip archived nodes
            if node.get("status") == "archived":
                continue

            # Compute score
            score = self._score_node(node, question_tokens)

            if score >= self.min_score:
                scored_nodes.append((score, node))

        # Sort by score descending
        scored_nodes.sort(reverse=True, key=lambda x: x[0])

        # Take top k
        top_nodes = scored_nodes[:self.top_k]

        # Check for session diversity
        if len(top_nodes) > 0:
            top_nodes = self._ensure_session_diversity(top_nodes)

        # Convert to evidence objects
        evidence = []
        for score, node in top_nodes:
            ev = RetrievedMemoryEvidence(
                memory_id=node["id"],
                memory_type=node["type"],
                content=node.get("content", ""),
                score=score,
                source_session_ids=node.get("source_session_ids", []),
                source_turn_ids=node.get("source_turn_ids", []),
                source_event_ids=node.get("source_event_ids", []),
            )
            evidence.append(ev)

        # Add global summaries if enabled
        if self.include_summaries:
            summaries = current_memory.get("memory_summaries", {})
            for key, value in summaries.items():
                if value and value.strip():
                    ev = RetrievedMemoryEvidence(
                        memory_id=f"summary_{key}",
                        memory_type="summary",
                        content=f"{key}: {value}",
                        score=0.5,  # Fixed score for summaries
                        source_session_ids=[],
                        source_turn_ids=[],
                        source_event_ids=[],
                    )
                    evidence.append(ev)

        return evidence

    def _score_node(self, node: dict, question_tokens: set) -> float:
        """
        Score a memory node for relevance to question.

        Args:
            node: Memory node dict
            question_tokens: Set of normalized question tokens

        Returns:
            Relevance score
        """
        content = node.get("content", "")
        content_normalized = normalize_text(content)
        content_tokens = set(content_normalized.split())

        if not content_tokens:
            return 0.0

        # Token overlap
        overlap = question_tokens & content_tokens
        overlap_ratio = len(overlap) / len(question_tokens) if question_tokens else 0.0

        # Apply type weight
        node_type = node.get("type", "other")
        type_weight = TYPE_WEIGHTS.get(node_type, 1.0)

        # Check for preference/plan markers
        content_lower = content.lower()
        if node_type == "preference":
            if any(marker in content_lower for marker in ["prefer", "like", "favorite"]):
                type_weight *= 1.1

        if node_type == "plan":
            if any(marker in content_lower for marker in ["plan", "will", "going to"]):
                type_weight *= 1.1

        score = overlap_ratio * type_weight

        # Boost by confidence
        confidence = node.get("confidence", 0.8)
        score *= confidence

        return min(score, 1.0)

    def _ensure_session_diversity(self, scored_nodes: List[tuple]) -> List[tuple]:
        """
        Ensure session diversity in top results.

        If all top results come from one session but there are high-scoring
        nodes from other sessions, include one from another session.

        Args:
            scored_nodes: List of (score, node) tuples

        Returns:
            Adjusted list of (score, node) tuples
        """
        if len(scored_nodes) <= 1:
            return scored_nodes

        # Check sessions in top results
        top_sessions = set()
        for score, node in scored_nodes[:self.top_k]:
            sessions = node.get("source_session_ids", [])
            top_sessions.update(sessions)

        # If only one session, look for high-scoring nodes from other sessions
        if len(top_sessions) == 1:
            # Find best node from a different session
            for score, node in scored_nodes[self.top_k:]:
                node_sessions = set(node.get("source_session_ids", []))
                if node_sessions and not node_sessions.intersection(top_sessions):
                    # Add this node to results
                    scored_nodes = scored_nodes[:self.top_k] + [(score, node)]
                    break

        return scored_nodes
