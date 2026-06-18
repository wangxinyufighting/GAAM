"""
Memory Builder module.

Builds CurrentMemory artifacts from raw conversation history.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .memory_schema import (
    MemoryEdgeType,
    MemoryNodeType,
    MemoryStatus,
    empty_current_memory,
    validate_current_memory,
)
from .raw_history import RawHistoryView, RawSession, RawTurn, iter_history_slices
from .utils import normalize_text, stable_id


class BaselineMemoryBuilder:
    """
    Baseline deterministic Memory Builder (no-LLM MVP).

    Creates schema-valid CurrentMemory from raw history using conservative
    heuristics. Preserves user-derived memory-worthy turns, avoids storing
    target questions or answers.
    """

    def __init__(
        self,
        *,
        include_assistant_turns: bool = False,
        min_user_text_chars: int = 20,
        max_event_nodes: int | None = None,
    ) -> None:
        """
        Initialize baseline memory builder.

        Args:
            include_assistant_turns: If True, store assistant turns as event nodes
            min_user_text_chars: Minimum text length for user turns to be stored
            max_event_nodes: Optional cap on event nodes (for smoke tests)
        """
        self.include_assistant_turns = include_assistant_turns
        self.min_user_text_chars = min_user_text_chars
        self.max_event_nodes = max_event_nodes

    def build_case_memory(self, history: RawHistoryView) -> dict:
        """
        Build current memory from raw history.

        Args:
            history: RawHistoryView containing conversation history

        Returns:
            Valid CurrentMemory dict
        """
        memory = empty_current_memory(history.record_id)

        for session in iter_history_slices(history, slice_by="session"):
            memory = self.update_memory(memory, session)

        # Validate before returning
        validated = validate_current_memory(memory, strict=True)
        return validated.model_dump()

    def update_memory(
        self,
        previous_memory: dict,
        raw_slice: RawSession | RawTurn,
    ) -> dict:
        """
        Update memory with new raw history slice.

        Args:
            previous_memory: Previous CurrentMemory dict
            raw_slice: New RawSession or RawTurn to process

        Returns:
            Updated CurrentMemory dict
        """
        # Validate input
        validate_current_memory(previous_memory, strict=True)

        # Deep copy to avoid mutation
        memory = copy.deepcopy(previous_memory)
        memory["build_step"] += 1

        # Process based on slice type
        if isinstance(raw_slice, RawSession):
            self._process_session(memory, raw_slice)
        elif isinstance(raw_slice, RawTurn):
            self._process_turn(memory, raw_slice)
        else:
            raise TypeError(f"Unexpected slice type: {type(raw_slice)}")

        # Validate output
        validated = validate_current_memory(memory, strict=True)
        return validated.model_dump()

    def _process_session(self, memory: dict, session: RawSession) -> None:
        """Process a full session."""
        retained_turns = []

        for turn in session.turns:
            if self._should_retain_turn(turn, current_memory=memory):
                retained_turns.append(turn)
                self._add_event_node(memory, turn)
                self._add_fact_node(memory, turn)

        # If session has multiple retained turns, add session-level nodes
        if len(retained_turns) >= 2:
            self._add_session_summary(memory, session, retained_turns)
            self._add_session_abstract(memory, session, retained_turns)

        # Update global summaries
        self._update_global_summaries(memory)

    def _process_turn(self, memory: dict, turn: RawTurn) -> None:
        """Process a single turn."""
        if self._should_retain_turn(turn, current_memory=memory):
            self._add_event_node(memory, turn)
            self._add_fact_node(memory, turn)
            self._update_global_summaries(memory)

    def _should_retain_turn(self, turn: RawTurn, current_memory: Optional[dict] = None) -> bool:
        """Check if turn should be retained in memory."""
        # Check max_event_nodes cap
        if self.max_event_nodes is not None and current_memory is not None:
            # Count existing event nodes
            event_count = sum(
                1 for n in current_memory["memory_graph"]["nodes"]
                if n["type"] == MemoryNodeType.EVENT.value
            )
            if event_count >= self.max_event_nodes:
                return False

        # Filter by speaker
        speaker_lower = turn.speaker.lower()
        is_user = speaker_lower in {"user", "human"}
        is_assistant = speaker_lower in {"assistant", "ai", "system"}

        if not is_user and not (is_assistant and self.include_assistant_turns):
            return False

        # Filter by text length
        normalized_text = normalize_text(turn.text)
        if len(normalized_text) < self.min_user_text_chars:
            return False

        # Skip generic task instructions (simple heuristic)
        if self._is_generic_task(normalized_text):
            return False

        return True

    def _is_generic_task(self, text: str) -> bool:
        """Check if text looks like a generic task instruction."""
        text_lower = text.lower()

        # Skip very short generic prompts
        if len(text) < 50 and any(marker in text_lower for marker in [
            "solve this",
            "calculate",
            "translate",
            "write code",
            "debug this",
        ]):
            return True

        return False

    def _add_event_node(self, memory: dict, turn: RawTurn) -> None:
        """Add event memory node."""
        node_id = stable_id("mem_evt", memory["record_id"], turn.event_id)

        # Check if already exists
        if any(n["id"] == node_id for n in memory["memory_graph"]["nodes"]):
            return

        node = {
            "id": node_id,
            "type": MemoryNodeType.EVENT.value,
            "content": f"{turn.speaker} said: {turn.text}",
            "status": MemoryStatus.ACTIVE.value,
            "confidence": 0.6,
            "abstraction_level": 0,
            "source_session_ids": [turn.session_id],
            "source_turn_ids": [str(turn.turn_id)],
            "source_event_ids": [turn.event_id],
            "metadata": {
                "speaker": turn.speaker,
                "timestamp": turn.timestamp or "",
            },
        }

        memory["memory_graph"]["nodes"].append(node)

    def _add_fact_node(self, memory: dict, turn: RawTurn) -> None:
        """Add fact memory node derived from turn."""
        event_node_id = stable_id("mem_evt", memory["record_id"], turn.event_id)
        fact_node_id = stable_id("mem_fact", memory["record_id"], turn.event_id, turn.text[:128])

        # Check if already exists
        if any(n["id"] == fact_node_id for n in memory["memory_graph"]["nodes"]):
            return

        # Create conservative fact
        normalized_text = normalize_text(turn.text)
        content = f"The user mentioned: {normalized_text}"

        node = {
            "id": fact_node_id,
            "type": MemoryNodeType.FACT.value,
            "content": content,
            "status": MemoryStatus.ACTIVE.value,
            "confidence": 0.55,
            "abstraction_level": 1,
            "source_session_ids": [turn.session_id],
            "source_turn_ids": [str(turn.turn_id)],
            "source_event_ids": [turn.event_id],
            "metadata": {},
        }

        memory["memory_graph"]["nodes"].append(node)

        # Add DERIVED_FROM edge
        edge = {
            "source": fact_node_id,
            "target": event_node_id,
            "type": MemoryEdgeType.DERIVED_FROM.value,
            "confidence": 0.6,
            "metadata": {},
        }

        memory["memory_graph"]["edges"].append(edge)

    def _add_session_summary(
        self,
        memory: dict,
        session: RawSession,
        retained_turns: List[RawTurn]
    ) -> None:
        """Add session-level summary node."""
        summary_id = stable_id("mem_summary", memory["record_id"], session.session_id)

        # Check if already exists
        if any(n["id"] == summary_id for n in memory["memory_graph"]["nodes"]):
            return

        content = (
            f"Session {session.session_id} contains "
            f"{len(retained_turns)} user-provided pieces of information."
        )

        node = {
            "id": summary_id,
            "type": MemoryNodeType.SUMMARY.value,
            "content": content,
            "status": MemoryStatus.ACTIVE.value,
            "confidence": 0.5,
            "abstraction_level": 1,
            "source_session_ids": [session.session_id],
            "source_turn_ids": [],
            "source_event_ids": [],
            "metadata": {},
        }

        memory["memory_graph"]["nodes"].append(node)

    def _add_session_abstract(
        self,
        memory: dict,
        session: RawSession,
        retained_turns: List[RawTurn]
    ) -> None:
        """Add session-level abstract node."""
        abstract_id = stable_id("mem_abs", memory["record_id"], session.session_id, str(len(retained_turns)))

        # Check if already exists
        if any(n["id"] == abstract_id for n in memory["memory_graph"]["nodes"]):
            return

        content = "The user shared multiple pieces of information in this session."

        # Collect event IDs
        event_ids = [turn.event_id for turn in retained_turns]

        node = {
            "id": abstract_id,
            "type": MemoryNodeType.ABSTRACT.value,
            "content": content,
            "status": MemoryStatus.ACTIVE.value,
            "confidence": 0.45,
            "abstraction_level": 2,
            "source_session_ids": [session.session_id],
            "source_turn_ids": [],
            "source_event_ids": event_ids,
            "metadata": {},
        }

        memory["memory_graph"]["nodes"].append(node)

        # Add ABSTRACTS edges to facts
        fact_nodes = [
            n for n in memory["memory_graph"]["nodes"]
            if n["type"] == MemoryNodeType.FACT.value
            and session.session_id in n["source_session_ids"]
        ]

        for fact_node in fact_nodes[:5]:  # Limit to first 5 facts
            edge = {
                "source": abstract_id,
                "target": fact_node["id"],
                "type": MemoryEdgeType.ABSTRACTS.value,
                "confidence": 0.45,
                "metadata": {},
            }
            memory["memory_graph"]["edges"].append(edge)

    def _update_global_summaries(self, memory: dict) -> None:
        """Update global summary strings."""
        node_count = len(memory["memory_graph"]["nodes"])
        session_ids = set()

        for node in memory["memory_graph"]["nodes"]:
            session_ids.update(node.get("source_session_ids", []))

        retained_turn_count = sum(
            1 for n in memory["memory_graph"]["nodes"]
            if n["type"] == MemoryNodeType.EVENT.value
        )

        memory["memory_summaries"]["recent_changes"] = (
            f"The user provided information across {len(session_ids)} sessions "
            f"and {retained_turn_count} retained user turns."
        )

        # Simple heuristic for preferences
        preference_markers = ["like", "prefer", "favorite"]
        plan_markers = ["plan", "will", "going to", "need to"]

        has_preferences = any(
            any(marker in n["content"].lower() for marker in preference_markers)
            for n in memory["memory_graph"]["nodes"]
        )

        has_plans = any(
            any(marker in n["content"].lower() for marker in plan_markers)
            for n in memory["memory_graph"]["nodes"]
        )

        if has_preferences:
            memory["memory_summaries"]["stable_preferences"] = (
                "The user has expressed preferences in the conversation."
            )

        if has_plans:
            memory["memory_summaries"]["active_plans"] = (
                "The user has mentioned plans or intentions."
            )


def build_current_memory_from_record(
    record: Any,  # LMERecord
    *,
    builder: Optional[BaselineMemoryBuilder] = None,
) -> dict:
    """
    Build current memory from LMERecord.

    Args:
        record: LMERecord to process
        builder: Optional BaselineMemoryBuilder instance

    Returns:
        Valid CurrentMemory dict
    """
    from .raw_history import raw_history_from_lme_record

    if builder is None:
        builder = BaselineMemoryBuilder()

    # Convert to safe raw history (no question/answer)
    history = raw_history_from_lme_record(record)

    # Build memory
    return builder.build_case_memory(history)
