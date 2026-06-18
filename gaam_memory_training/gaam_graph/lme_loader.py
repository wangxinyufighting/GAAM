from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .utils import normalize_text, read_json, stable_id


@dataclass
class LMEEvent:
    event_id: str
    session_id: str
    turn_id: int
    speaker: str
    text: str
    timestamp: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class LMERecord:
    record_id: str
    question: Optional[str]
    answer: Optional[str]
    question_type: Optional[str]
    events: List[LMEEvent]
    raw: Dict[str, Any]


class LongMemEvalLoader:
    """Defensive loader for LongMemEval-style JSON files.

    Official / cleaned variants have changed field names over time. This loader tries
    common shapes rather than assuming one exact schema.
    """

    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[LMERecord]:
        data = read_json(self.path)
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                items = data["data"]
            else:
                items = list(data.values())
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError(f"Unsupported JSON root type: {type(data)}")
        return [self._parse_record(item, idx) for idx, item in enumerate(items)]

    def _parse_record(self, item: Dict[str, Any], idx: int) -> LMERecord:
        rid = str(item.get("id") or item.get("question_id") or item.get("sample_id") or f"record_{idx:05d}")
        question = item.get("question") or item.get("query")
        answer = item.get("answer") or item.get("gold_answer") or item.get("target")
        qtype = item.get("question_type") or item.get("haystack_question_type") or item.get("type")
        sessions = self._extract_sessions(item)
        events: List[LMEEvent] = []
        for sidx, sess in enumerate(sessions):
            session_id = str(sess.get("session_id") or sess.get("id") or f"{rid}_sess_{sidx:04d}") if isinstance(sess, dict) else f"{rid}_sess_{sidx:04d}"
            timestamp = self._get_timestamp(sess)
            turns = self._extract_turns(sess)
            prev_speaker = None
            for tidx, turn in enumerate(turns):
                speaker, text = self._parse_turn(turn, prev_speaker=prev_speaker)
                prev_speaker = speaker
                text = normalize_text(text)
                if not text:
                    continue
                event_id = stable_id("evt", rid, session_id, tidx, speaker, text[:128])
                events.append(LMEEvent(
                    event_id=event_id,
                    session_id=session_id,
                    turn_id=tidx,
                    speaker=speaker,
                    text=text,
                    timestamp=self._get_timestamp(turn) or timestamp,
                    raw=turn if isinstance(turn, dict) else {"text": text},
                ))
        return LMERecord(rid, question, answer, qtype, events, item)

    def _extract_sessions(self, item: Dict[str, Any]) -> List[Any]:
        for key in ["haystack_sessions", "sessions", "history", "histories", "conversation_sessions", "conversations"]:
            if key in item and isinstance(item[key], list):
                return item[key]
        # Some variants store a single conversation/messages list directly.
        for key in ["messages", "conversation", "dialogue", "chat_history"]:
            if key in item and isinstance(item[key], list):
                return [{"session_id": "session_0000", "messages": item[key]}]
        return []

    def _extract_turns(self, sess: Any) -> List[Any]:
        if isinstance(sess, list):
            return sess
        if isinstance(sess, str):
            return [{"role": "unknown", "content": sess}]
        if not isinstance(sess, dict):
            return []
        for key in ["messages", "conversation", "dialogue", "turns", "chat", "history"]:
            if key in sess and isinstance(sess[key], list):
                return sess[key]
        # LongMemEval sometimes uses speaker-to-text fields in one session object.
        if "user" in sess or "assistant" in sess:
            turns = []
            if sess.get("user"):
                turns.append({"role": "user", "content": sess["user"]})
            if sess.get("assistant"):
                turns.append({"role": "assistant", "content": sess["assistant"]})
            return turns
        return []

    def _parse_turn(self, turn: Any, prev_speaker: Optional[str] = None) -> tuple[str, str]:
        if isinstance(turn, str):
            return "unknown", turn
        if not isinstance(turn, dict):
            return "unknown", str(turn)
        speaker = turn.get("role") or turn.get("speaker") or turn.get("from") or turn.get("author") or "unknown"
        text = turn.get("content") or turn.get("text") or turn.get("message") or turn.get("utterance") or ""
        if isinstance(text, list):
            text = "\n".join(str(x) for x in text)
        return str(speaker), str(text)

    def _get_timestamp(self, obj: Any) -> Optional[str]:
        if isinstance(obj, dict):
            for key in ["timestamp", "time", "date", "created_at", "datetime"]:
                if obj.get(key):
                    return str(obj[key])
        return None
