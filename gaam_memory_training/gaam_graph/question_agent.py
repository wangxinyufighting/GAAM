"""
Question Agent module.

Generates questions from oracle graph evidence trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .llm import OpenAICompatibleLLM
from .question_schema import GeneratedQuestion, QuestionType
from .utils import stable_id


class QuestionAgent:
    """
    Question Agent for generating questions from oracle graph trajectories.

    Supports both no-LLM (deterministic placeholder) and LLM modes.
    """

    def __init__(
        self,
        *,
        llm: Optional[OpenAICompatibleLLM] = None,
        no_llm: bool = False,
        prompt_path: Optional[str | Path] = None,
    ) -> None:
        """
        Initialize Question Agent.

        Args:
            llm: Optional LLM client for question generation
            no_llm: If True, use deterministic no-LLM mode
            prompt_path: Optional path to question generation prompt
        """
        self.llm = llm
        self.no_llm = no_llm
        self.prompt_path = prompt_path or (
            Path(__file__).parent / "prompts" / "question_generation_from_walks.txt"
        )

    def sample_trajectories(
        self,
        graph: dict,
        *,
        num_walks: int,
        walk_length: int,
        seed: int,
        directed: bool = False,
        require_multi_session: bool = False,
        min_sessions: int = 2,
        max_structural_ratio: float = 0.35,
        max_consecutive_next: int = 1,
    ) -> List[dict]:
        """
        Sample evidence trajectories from graph.

        Args:
            graph: Graph dict with nodes and edges
            num_walks: Number of walks to sample
            walk_length: Length of each walk
            seed: Random seed
            directed: If True, respect edge directions
            require_multi_session: If True, filter to multi-session trajectories
            min_sessions: Minimum sessions for multi-session trajectories
            max_structural_ratio: Max ratio of structural nodes
            max_consecutive_next: Max consecutive NEXT edges

        Returns:
            List of trajectory dicts
        """
        from .graph_walker import GraphWalker

        # Create walker
        walker = GraphWalker(
            graph_json=graph,
            seed=seed,
            directed=directed,
        )

        # Sample walks
        trajectories = walker.sample_walks(
            num_walks=num_walks,
            walk_length=walk_length,
            min_nodes=2,
            max_structural_ratio=max_structural_ratio,
            require_evidence_node=True,
            max_consecutive_next=max_consecutive_next,
            require_multi_session=require_multi_session,
            min_sessions=min_sessions,
        )

        return trajectories

    def generate_candidates(
        self,
        *,
        record_id: str,
        graph_path: str,
        trajectories: List[dict],
        num_questions: int,
        require_multi_session: bool = False,
        min_sessions: int = 2,
        desired_question_types: Optional[List[str]] = None,
    ) -> List[GeneratedQuestion]:
        """
        Generate candidate questions from trajectories.

        Args:
            record_id: Record identifier
            graph_path: Path to source graph
            trajectories: List of trajectory dicts
            num_questions: Number of questions to generate
            require_multi_session: If True, questions should use multi-session evidence
            min_sessions: Minimum sessions for multi-session questions
            desired_question_types: Optional list of desired question types

        Returns:
            List of GeneratedQuestion objects
        """
        if self.no_llm:
            return self._generate_no_llm(
                record_id=record_id,
                graph_path=graph_path,
                trajectories=trajectories,
                num_questions=num_questions,
            )
        else:
            return self._generate_with_llm(
                record_id=record_id,
                graph_path=graph_path,
                trajectories=trajectories,
                num_questions=num_questions,
                require_multi_session=require_multi_session,
                min_sessions=min_sessions,
                desired_question_types=desired_question_types,
            )

    def _generate_no_llm(
        self,
        *,
        record_id: str,
        graph_path: str,
        trajectories: List[dict],
        num_questions: int,
    ) -> List[GeneratedQuestion]:
        """Generate deterministic placeholder questions (no-LLM mode)."""
        questions = []

        for i, trajectory in enumerate(trajectories[:num_questions]):
            # Choose best evidence node from trajectory
            evidence_node = self._select_best_evidence_node(trajectory)

            if not evidence_node:
                continue

            # Generate placeholder question
            question_id = stable_id("question", record_id, f"no_llm_{i}")

            question = GeneratedQuestion(
                question_id=question_id,
                record_id=record_id,
                question="What user information is supported by the cited evidence?",
                answer=evidence_node.get("label", "Evidence content"),
                question_type=QuestionType.OTHER,
                status="candidate",
                supporting_trajectory_ids=[trajectory.get("trajectory_id", f"traj_{i}")],
                supporting_node_ids=[evidence_node["id"]],
                supporting_session_ids=evidence_node.get("session_ids", []),
                reason="Generated in no-LLM mode for pipeline testing.",
                metadata={
                    "mode": "no_llm",
                    "trajectory_index": i,
                },
            )

            questions.append(question)

        return questions

    def _select_best_evidence_node(self, trajectory: dict) -> Optional[dict]:
        """Select best evidence node from trajectory."""
        nodes = trajectory.get("nodes", [])

        # Prefer in order: fact, abstract_memory, event
        type_priority = {"fact": 3, "abstract_memory": 2, "event": 1}

        best_node = None
        best_score = -1

        for node in nodes:
            node_type = node.get("type", "")
            score = type_priority.get(node_type, 0)

            if score > best_score:
                best_score = score
                best_node = node

        return best_node

    def _generate_with_llm(
        self,
        *,
        record_id: str,
        graph_path: str,
        trajectories: List[dict],
        num_questions: int,
        require_multi_session: bool,
        min_sessions: int,
        desired_question_types: Optional[List[str]],
    ) -> List[GeneratedQuestion]:
        """Generate questions using LLM."""
        if not self.llm:
            raise ValueError("LLM client is required for LLM mode. Use no_llm=True or provide llm.")

        # Load prompt
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")

        system_prompt = self.prompt_path.read_text(encoding="utf-8")

        # Build payload
        payload = {
            "record_id": record_id,
            "num_questions": num_questions,
            "require_multi_session": require_multi_session,
            "min_sessions": min_sessions,
            "desired_question_types": desired_question_types or [],
            "trajectories": self._format_trajectories(trajectories),
        }

        user_message = json.dumps(payload, ensure_ascii=False, indent=2)

        # Call LLM
        response = self.llm.chat_json(system=system_prompt, user=user_message)

        # Parse response
        questions = self._parse_llm_response(
            response=response,
            record_id=record_id,
            graph_path=graph_path,
        )

        return questions

    def _format_trajectories(self, trajectories: List[dict]) -> List[dict]:
        """Format trajectories for LLM prompt."""
        formatted = []

        for traj in trajectories:
            formatted_traj = {
                "trajectory_id": traj.get("trajectory_id", ""),
                "session_ids": traj.get("session_ids", []),
                "text": traj.get("text", ""),
                "nodes": [
                    {
                        "id": node.get("id", ""),
                        "type": node.get("type", ""),
                        "label": node.get("label", ""),
                    }
                    for node in traj.get("nodes", [])
                ],
            }
            formatted.append(formatted_traj)

        return formatted

    def _parse_llm_response(
        self,
        *,
        response: dict,
        record_id: str,
        graph_path: str,
    ) -> List[GeneratedQuestion]:
        """Parse LLM response into GeneratedQuestion objects."""
        questions = []

        for i, q_dict in enumerate(response.get("questions", [])):
            try:
                question_id = q_dict.get("question_id") or stable_id("question", record_id, f"llm_{i}")

                question = GeneratedQuestion(
                    question_id=question_id,
                    record_id=record_id,
                    question=q_dict.get("question", ""),
                    answer=q_dict.get("answer", ""),
                    question_type=q_dict.get("question_type", QuestionType.OTHER),
                    status="candidate",
                    supporting_trajectory_ids=q_dict.get("supporting_trajectory_ids", []),
                    supporting_node_ids=q_dict.get("supporting_node_ids", []),
                    supporting_session_ids=q_dict.get("supporting_session_ids", []),
                    reason=q_dict.get("reason", ""),
                    metadata=q_dict.get("metadata", {}),
                )

                questions.append(question)

            except Exception as e:
                # Skip malformed questions
                continue

        return questions
