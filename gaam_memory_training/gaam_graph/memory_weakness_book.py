"""
Memory Weakness Book module.

Tracks recurring memory failures for curriculum and replay.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from gaam_graph.reward_schema import FailureType, MemoryRewardReport
from gaam_graph.utils import read_json, write_json


class WeaknessStatus(str, Enum):
    """Status of a weakness entry."""
    ACTIVE = "active"
    MASTERED = "mastered"
    STALE = "stale"


class WeaknessEntry(BaseModel):
    """Single weakness entry."""
    weakness_id: str
    record_id: str
    question_id: str
    question: str = ""
    question_type: str = "other"
    failure_type: FailureType
    severity: float = Field(ge=0.0, le=1.0)
    frequency: int = Field(default=1, ge=1)
    first_seen_round: int = 0
    last_seen_round: int = 0
    status: WeaknessStatus = WeaknessStatus.ACTIVE
    missing_oracle_node_ids: List[str] = Field(default_factory=list)
    related_session_ids: List[str] = Field(default_factory=list)
    related_memory_ids: List[str] = Field(default_factory=list)
    short_summary: str = ""
    recommended_focus: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


def stable_weakness_id(
    record_id: str,
    question_id: str,
    failure_type: str,
    oracle_node_ids: List[str],
) -> str:
    """
    Generate stable weakness ID.

    Args:
        record_id: Record ID
        question_id: Question ID
        failure_type: Failure type
        oracle_node_ids: Oracle node IDs

    Returns:
        Stable weakness ID
    """
    # Sort oracle node IDs for stability
    sorted_ids = sorted(oracle_node_ids)

    # Create hash input
    hash_input = f"{record_id}:{question_id}:{failure_type}:{'|'.join(sorted_ids)}"

    # Generate short hash
    hash_obj = hashlib.sha256(hash_input.encode())
    short_hash = hash_obj.hexdigest()[:8]

    return f"weak_{short_hash}"


class MemoryWeaknessBook:
    """
    Memory Weakness Book for tracking recurring failures.

    Inspired by Code-A1's Mistake Book pattern.
    """

    def __init__(
        self,
        *,
        record_id: str,
        weaknesses: List[WeaknessEntry] | None = None,
        version: int = 1,
        updated_round: int = 0,
    ) -> None:
        """
        Initialize MemoryWeaknessBook.

        Args:
            record_id: Record ID
            weaknesses: List of weakness entries
            version: Book version
            updated_round: Last updated round
        """
        self.record_id = record_id
        self.weaknesses = weaknesses or []
        self.version = version
        self.updated_round = updated_round

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        record_id: str | None = None,
    ) -> "MemoryWeaknessBook":
        """
        Load weakness book from file.

        Args:
            path: Path to weakness book JSON
            record_id: Expected record ID (optional)

        Returns:
            MemoryWeaknessBook instance
        """
        data = read_json(path)

        if record_id and data.get("record_id") != record_id:
            raise ValueError(
                f"Record ID mismatch: expected {record_id}, got {data.get('record_id')}"
            )

        weaknesses = [
            WeaknessEntry.model_validate(w) for w in data.get("weaknesses", [])
        ]

        return cls(
            record_id=data["record_id"],
            weaknesses=weaknesses,
            version=data.get("version", 1),
            updated_round=data.get("updated_round", 0),
        )

    @classmethod
    def empty(cls, record_id: str) -> "MemoryWeaknessBook":
        """
        Create empty weakness book.

        Args:
            record_id: Record ID

        Returns:
            Empty MemoryWeaknessBook
        """
        return cls(record_id=record_id)

    def update_from_reward_report(
        self,
        report: MemoryRewardReport,
        *,
        round_id: int,
    ) -> List[WeaknessEntry]:
        """
        Update weakness book from reward report.

        Args:
            report: Memory reward report
            round_id: Current round ID

        Returns:
            List of updated/new weaknesses
        """
        updated_weaknesses = []

        for item in report.question_items:
            if not item.failure_types:
                continue

            for failure_type in item.failure_types:
                # Generate weakness ID
                weakness_id = stable_weakness_id(
                    record_id=item.record_id,
                    question_id=item.question_id,
                    failure_type=failure_type.value,
                    oracle_node_ids=item.supporting_oracle_node_ids,
                )

                # Find existing weakness. Prefer exact ID, but also merge
                # equivalent active weaknesses with overlapping oracle nodes so
                # recurring failures across related questions become replayable.
                existing = self._find_weakness(weakness_id)
                if not existing:
                    existing = self._find_equivalent_weakness(
                        record_id=item.record_id,
                        question_id=item.question_id,
                        failure_type=failure_type,
                        oracle_node_ids=item.supporting_oracle_node_ids,
                    )

                if existing:
                    # Update existing
                    existing.frequency += 1
                    existing.last_seen_round = round_id

                    # Update severity (exponential moving average)
                    new_severity = 1.0 - item.correctness
                    existing.severity = max(
                        existing.severity * 0.8 + new_severity * 0.2,
                        new_severity,
                    )

                    # Merge related IDs
                    existing.related_session_ids = list(set(
                        existing.related_session_ids + item.related_session_ids
                    ))
                    existing.related_memory_ids = list(set(
                        existing.related_memory_ids + item.supporting_memory_ids
                    ))
                    existing.missing_oracle_node_ids = sorted(set(
                        existing.missing_oracle_node_ids + item.supporting_oracle_node_ids
                    ))

                    updated_weaknesses.append(existing)
                else:
                    # Create new weakness
                    new_weakness = WeaknessEntry(
                        weakness_id=weakness_id,
                        record_id=item.record_id,
                        question_id=item.question_id,
                        question_type=item.question_type,
                        failure_type=failure_type,
                        severity=1.0 - item.correctness,
                        frequency=1,
                        first_seen_round=round_id,
                        last_seen_round=round_id,
                        status=WeaknessStatus.ACTIVE,
                        missing_oracle_node_ids=item.supporting_oracle_node_ids,
                        related_session_ids=item.related_session_ids,
                        related_memory_ids=item.supporting_memory_ids,
                        short_summary=self._generate_summary(failure_type, item),
                        recommended_focus=self._generate_focus(failure_type),
                    )

                    self.weaknesses.append(new_weakness)
                    updated_weaknesses.append(new_weakness)

        self.updated_round = round_id
        return updated_weaknesses

    def _find_weakness(self, weakness_id: str) -> WeaknessEntry | None:
        """Find weakness by ID."""
        for w in self.weaknesses:
            if w.weakness_id == weakness_id:
                return w
        return None

    def _find_equivalent_weakness(
        self,
        *,
        record_id: str,
        question_id: str,
        failure_type: FailureType,
        oracle_node_ids: List[str],
    ) -> WeaknessEntry | None:
        """Find an active equivalent weakness for merge/update."""
        new_oracle_ids = set(oracle_node_ids)
        for weakness in self.weaknesses:
            if weakness.status != WeaknessStatus.ACTIVE:
                continue
            if weakness.record_id != record_id:
                continue
            if weakness.failure_type != failure_type:
                continue
            if weakness.question_id == question_id:
                return weakness
            if new_oracle_ids and new_oracle_ids.intersection(weakness.missing_oracle_node_ids):
                return weakness
        return None

    def _generate_summary(self, failure_type: FailureType, item: Any) -> str:
        """Generate short summary for weakness."""
        summaries = {
            FailureType.MISSING_FACT: "Memory missing required fact",
            FailureType.WRONG_FACT: "Memory contains wrong or outdated fact",
            FailureType.MISSING_ABSTRACTION: "Memory missing higher-level abstraction",
            FailureType.MISSING_MULTI_SESSION_LINK: "Memory failed to connect multi-session information",
            FailureType.TEMPORAL_ERROR: "Memory has temporal ordering error",
            FailureType.CONTRADICTION_UPDATE_ERROR: "Memory failed to handle contradictory update",
            FailureType.OVER_COMPRESSION: "Memory compressed away necessary details",
            FailureType.REDUNDANCY_NOISE: "Memory contains redundant information",
            FailureType.UNSUPPORTED_ANSWER: "Answerer guessed without memory support",
        }
        return summaries.get(failure_type, "Memory weakness detected")

    def _generate_focus(self, failure_type: FailureType) -> str:
        """Generate recommended focus."""
        focuses = {
            FailureType.MISSING_FACT: "Generate questions requiring this specific fact",
            FailureType.MISSING_ABSTRACTION: "Generate questions requiring abstraction",
            FailureType.MISSING_MULTI_SESSION_LINK: "Generate questions connecting multiple sessions",
            FailureType.TEMPORAL_ERROR: "Generate temporal reasoning questions",
            FailureType.CONTRADICTION_UPDATE_ERROR: "Generate update-sensitive questions",
            FailureType.OVER_COMPRESSION: "Avoid over-aggressive compression",
            FailureType.REDUNDANCY_NOISE: "Remove redundant memory nodes",
        }
        return focuses.get(failure_type, "Address memory weakness")

    def mark_mastered(
        self,
        *,
        question_id: str | None = None,
        weakness_id: str | None = None,
        round_id: int,
    ) -> None:
        """
        Mark weakness as mastered.

        Args:
            question_id: Question ID to mark mastered
            weakness_id: Weakness ID to mark mastered
            round_id: Current round ID
        """
        for weakness in self.weaknesses:
            if weakness.status != WeaknessStatus.ACTIVE:
                continue

            match = False
            if question_id and weakness.question_id == question_id:
                match = True
            if weakness_id and weakness.weakness_id == weakness_id:
                match = True

            if match:
                weakness.status = WeaknessStatus.MASTERED
                weakness.last_seen_round = round_id

    def get_active(
        self,
        *,
        max_count: int | None = None,
        failure_type: str | None = None,
    ) -> List[WeaknessEntry]:
        """
        Get active weaknesses.

        Args:
            max_count: Maximum number to return
            failure_type: Filter by failure type

        Returns:
            List of active weakness entries
        """
        active = [w for w in self.weaknesses if w.status == WeaknessStatus.ACTIVE]

        if failure_type:
            active = [w for w in active if w.failure_type.value == failure_type]

        # Sort by severity (descending) then frequency (descending)
        active.sort(key=lambda w: (-w.severity, -w.frequency))

        if max_count:
            active = active[:max_count]

        return active

    def to_question_agent_context(
        self,
        *,
        max_entries: int = 10,
        include_oracle_ids: bool = True,
    ) -> dict:
        """
        Export context for Question Agent.

        Args:
            max_entries: Max number of entries
            include_oracle_ids: Include oracle node IDs

        Returns:
            Context dict for Question Agent
        """
        active = self.get_active(max_count=max_entries)

        context_entries = []
        for weakness in active:
            entry = {
                "failure_type": weakness.failure_type.value,
                "question_type": weakness.question_type,
                "related_session_ids": weakness.related_session_ids,
                "recommended_focus": weakness.recommended_focus,
            }

            if include_oracle_ids:
                entry["missing_oracle_node_ids"] = weakness.missing_oracle_node_ids

            context_entries.append(entry)

        return {
            "active_weaknesses": context_entries,
        }

    def save(self, path: str | Path) -> None:
        """
        Save weakness book to file.

        Args:
            path: Output path
        """
        # Compute summary
        active_count = sum(1 for w in self.weaknesses if w.status == WeaknessStatus.ACTIVE)
        mastered_count = sum(1 for w in self.weaknesses if w.status == WeaknessStatus.MASTERED)

        top_failure_types: Dict[str, int] = {}
        for w in self.weaknesses:
            if w.status == WeaknessStatus.ACTIVE:
                ft = w.failure_type.value
                top_failure_types[ft] = top_failure_types.get(ft, 0) + 1

        data = {
            "record_id": self.record_id,
            "version": self.version,
            "updated_round": self.updated_round,
            "weaknesses": [w.model_dump(mode="json") for w in self.weaknesses],
            "summary": {
                "active_count": active_count,
                "mastered_count": mastered_count,
                "top_failure_types": top_failure_types,
            },
        }

        write_json(path, data)
