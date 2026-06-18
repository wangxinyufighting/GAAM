"""
Raw history view module.

Provides a safe, question-free view of LMERecord for Memory Builder input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .lme_loader import LMERecord


# Forbidden keys that must not appear in raw history
FORBIDDEN_RAW_HISTORY_KEYS = {
    "question",
    "answer",
    "gold_answer",
    "target",
    "query",
    "question_type",
    "haystack_question_type",
}


@dataclass(frozen=True)
class RawTurn:
    """A single turn in raw conversation history."""
    event_id: str
    session_id: str
    turn_id: int
    speaker: str
    text: str
    timestamp: Optional[str] = None


@dataclass(frozen=True)
class RawSession:
    """A session containing multiple turns."""
    session_id: str
    timestamp: Optional[str]
    turns: List[RawTurn]


@dataclass(frozen=True)
class RawHistoryView:
    """Safe raw-history-only view of an LMERecord."""
    record_id: str
    sessions: List[RawSession]


def raw_history_from_lme_record(record: LMERecord) -> RawHistoryView:
    """
    Create a safe raw history view from LMERecord.

    Only uses record.record_id and record.events.
    Never reads record.question, record.answer, or record.question_type.

    Args:
        record: LMERecord to convert

    Returns:
        RawHistoryView containing only conversation history
    """
    # Group events by session
    sessions_dict: Dict[str, List[RawTurn]] = {}
    session_order: List[str] = []  # Preserve first appearance order

    for event in record.events:
        turn = RawTurn(
            event_id=event.event_id,
            session_id=event.session_id,
            turn_id=event.turn_id,
            speaker=event.speaker,
            text=event.text,
            timestamp=event.timestamp,
        )

        if event.session_id not in sessions_dict:
            sessions_dict[event.session_id] = []
            session_order.append(event.session_id)

        sessions_dict[event.session_id].append(turn)

    # Build session objects, preserving order
    sessions = []
    for session_id in session_order:
        turns = sessions_dict[session_id]
        # Sort turns by turn_id within session
        turns_sorted = sorted(turns, key=lambda t: t.turn_id)

        # Get timestamp from first turn
        first_timestamp = turns_sorted[0].timestamp if turns_sorted else None

        session = RawSession(
            session_id=session_id,
            timestamp=first_timestamp,
            turns=turns_sorted,
        )
        sessions.append(session)

    return RawHistoryView(
        record_id=record.record_id,
        sessions=sessions,
    )


def iter_history_slices(
    history: RawHistoryView,
    *,
    slice_by: str = "session",
) -> Iterable[RawSession | RawTurn]:
    """
    Iterate over history slices.

    Args:
        history: RawHistoryView to slice
        slice_by: "session" or "turn"

    Yields:
        RawSession or RawTurn objects
    """
    if slice_by == "session":
        yield from history.sessions
    elif slice_by == "turn":
        for session in history.sessions:
            yield from session.turns
    else:
        raise ValueError(f"Invalid slice_by: {slice_by}")


def raw_history_to_prompt_payload(history: RawHistoryView) -> dict:
    """
    Convert raw history to prompt payload dict.

    Output contains no forbidden keys.

    Args:
        history: RawHistoryView to convert

    Returns:
        Dict suitable for LLM prompt input
    """
    payload = {
        "record_id": history.record_id,
        "sessions": [],
    }

    for session in history.sessions:
        session_dict = {
            "session_id": session.session_id,
            "timestamp": session.timestamp or "",
            "turns": [],
        }

        for turn in session.turns:
            turn_dict = {
                "event_id": turn.event_id,
                "turn_id": turn.turn_id,
                "speaker": turn.speaker,
                "text": turn.text,
            }
            if turn.timestamp:
                turn_dict["timestamp"] = turn.timestamp

            session_dict["turns"].append(turn_dict)

        payload["sessions"].append(session_dict)

    # Validate no forbidden keys
    assert_raw_history_no_leakage(payload)

    return payload


def assert_raw_history_no_leakage(payload: dict) -> None:
    """
    Assert that no forbidden keys exist in payload.

    Args:
        payload: Dict to check

    Raises:
        ValueError: If forbidden keys found
    """
    _recursive_check_leakage(payload, "", FORBIDDEN_RAW_HISTORY_KEYS)


def _recursive_check_leakage(
    obj: Any,
    path: str,
    forbidden_keys: set
) -> None:
    """Recursively check for forbidden keys."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            json_path = f"{path}.{key}" if path else key

            if key in forbidden_keys:
                raise ValueError(
                    f"Forbidden key '{key}' found at {json_path} in raw history payload"
                )

            _recursive_check_leakage(value, json_path, forbidden_keys)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            json_path = f"{path}[{i}]"
            _recursive_check_leakage(item, json_path, forbidden_keys)
