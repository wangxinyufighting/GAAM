"""
Tests for RawHistoryView.

Run with: pytest gaam_memory_training/tests/test_raw_history.py
"""

import pytest

from gaam_graph.lme_loader import LMEEvent, LMERecord
from gaam_graph.raw_history import (
    FORBIDDEN_RAW_HISTORY_KEYS,
    RawHistoryView,
    assert_raw_history_no_leakage,
    iter_history_slices,
    raw_history_from_lme_record,
    raw_history_to_prompt_payload,
)


@pytest.fixture
def sample_lme_record():
    """Create a sample LMERecord."""
    events = [
        LMEEvent(
            event_id="evt_001",
            session_id="sess_01",
            turn_id=0,
            speaker="user",
            text="I like Python programming.",
            timestamp="2024-01-01T10:00:00",
        ),
        LMEEvent(
            event_id="evt_002",
            session_id="sess_01",
            turn_id=1,
            speaker="assistant",
            text="That's great! Python is a versatile language.",
            timestamp="2024-01-01T10:01:00",
        ),
        LMEEvent(
            event_id="evt_003",
            session_id="sess_02",
            turn_id=0,
            speaker="user",
            text="I graduated with a degree in Computer Science.",
            timestamp="2024-01-02T10:00:00",
        ),
    ]

    return LMERecord(
        record_id="test_record",
        events=events,
        question="What does the user like?",  # Should be dropped
        answer="Python programming",  # Should be dropped
        question_type="preference",  # Should be dropped
        raw={},
    )


def test_raw_history_drops_leaked_fields(sample_lme_record):
    """Test that RawHistoryView drops question/answer/question_type."""
    history = raw_history_from_lme_record(sample_lme_record)

    # Should have record_id and sessions
    assert history.record_id == "test_record"
    assert len(history.sessions) == 2

    # Convert to payload and check no forbidden keys
    payload = raw_history_to_prompt_payload(history)

    assert "record_id" in payload
    assert "sessions" in payload

    # Check no forbidden keys exist anywhere
    for key in FORBIDDEN_RAW_HISTORY_KEYS:
        assert key not in str(payload)

    # Should not raise
    assert_raw_history_no_leakage(payload)


def test_raw_history_preserves_events(sample_lme_record):
    """Test that RawHistoryView preserves event data correctly."""
    history = raw_history_from_lme_record(sample_lme_record)

    # Check sessions grouped correctly
    assert len(history.sessions) == 2

    session_1 = history.sessions[0]
    assert session_1.session_id == "sess_01"
    assert len(session_1.turns) == 2

    session_2 = history.sessions[1]
    assert session_2.session_id == "sess_02"
    assert len(session_2.turns) == 1

    # Check turn order preserved
    turn_0 = session_1.turns[0]
    assert turn_0.turn_id == 0
    assert turn_0.speaker == "user"
    assert "Python" in turn_0.text

    turn_1 = session_1.turns[1]
    assert turn_1.turn_id == 1
    assert turn_1.speaker == "assistant"

    # Check session order follows first appearance
    assert history.sessions[0].session_id == "sess_01"
    assert history.sessions[1].session_id == "sess_02"


def test_iter_history_slices_by_session(sample_lme_record):
    """Test iteration by session."""
    history = raw_history_from_lme_record(sample_lme_record)

    sessions = list(iter_history_slices(history, slice_by="session"))
    assert len(sessions) == 2
    assert all(hasattr(s, "session_id") for s in sessions)


def test_iter_history_slices_by_turn(sample_lme_record):
    """Test iteration by turn."""
    history = raw_history_from_lme_record(sample_lme_record)

    turns = list(iter_history_slices(history, slice_by="turn"))
    assert len(turns) == 3
    assert all(hasattr(t, "event_id") for t in turns)


def test_raw_history_no_leakage_validation():
    """Test that leakage validation catches forbidden keys."""
    # Safe payload
    safe_payload = {
        "record_id": "test",
        "sessions": [
            {
                "session_id": "sess_01",
                "turns": [
                    {"event_id": "evt_001", "speaker": "user", "text": "Hello"}
                ]
            }
        ]
    }

    assert_raw_history_no_leakage(safe_payload)  # Should not raise

    # Leaky payload - top level
    leaky_payload_1 = {
        "record_id": "test",
        "question": "What did the user say?",  # Forbidden
        "sessions": []
    }

    with pytest.raises(ValueError) as exc_info:
        assert_raw_history_no_leakage(leaky_payload_1)
    assert "question" in str(exc_info.value)

    # Leaky payload - nested
    leaky_payload_2 = {
        "record_id": "test",
        "sessions": [
            {
                "session_id": "sess_01",
                "gold_answer": "leaked",  # Forbidden
                "turns": []
            }
        ]
    }

    with pytest.raises(ValueError) as exc_info:
        assert_raw_history_no_leakage(leaky_payload_2)
    assert "gold_answer" in str(exc_info.value)


def test_deterministic_output_with_different_questions():
    """
    Test that two LMERecords with identical events but different
    questions/answers produce identical RawHistoryView.
    """
    events = [
        LMEEvent(
            event_id="evt_001",
            session_id="sess_01",
            turn_id=0,
            speaker="user",
            text="I like cats.",
            timestamp="2024-01-01T10:00:00",
        ),
    ]

    record_a = LMERecord(
        record_id="test",
        events=events,
        question="What animal does the user like?",
        answer="Cats",
        question_type="preference",
        raw={},
    )

    record_b = LMERecord(
        record_id="test",
        events=events,
        question="Different question",
        answer="Different answer",
        question_type="different_type",
        raw={},
    )

    history_a = raw_history_from_lme_record(record_a)
    history_b = raw_history_from_lme_record(record_b)

    # Convert to dicts for comparison
    payload_a = raw_history_to_prompt_payload(history_a)
    payload_b = raw_history_to_prompt_payload(history_b)

    # Should be identical
    assert payload_a == payload_b


def test_empty_record():
    """Test handling of empty record."""
    record = LMERecord(
        record_id="empty",
        events=[],
        question=None,
        answer=None,
        question_type=None,
        raw={},
    )

    history = raw_history_from_lme_record(record)

    assert history.record_id == "empty"
    assert len(history.sessions) == 0

    payload = raw_history_to_prompt_payload(history)
    assert payload["record_id"] == "empty"
    assert payload["sessions"] == []


def test_session_timestamp_from_first_turn(sample_lme_record):
    """Test that session timestamp is taken from first turn."""
    history = raw_history_from_lme_record(sample_lme_record)

    session_1 = history.sessions[0]
    assert session_1.timestamp == "2024-01-01T10:00:00"

    session_2 = history.sessions[1]
    assert session_2.timestamp == "2024-01-02T10:00:00"
