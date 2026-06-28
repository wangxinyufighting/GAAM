"""Tests for LLM client compatibility behavior."""

from __future__ import annotations

from openai import APIError

from gaam_graph.llm import LLMError, OpenAICompatibleLLM, parse_json_object


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


def test_parse_json_object_extracts_fenced_json():
    assert parse_json_object("```json\n{\"ok\": true}\n```") == {"ok": True}


def test_openai_compatible_llm_falls_back_without_response_format(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise APIError("unsupported response_format", request=None, body=None)
            return _Response('prefix {"ok": true, "mode": "fallback"} suffix')

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("gaam_graph.llm.OpenAI", FakeClient)

    llm = OpenAICompatibleLLM(
        model="fake-model",
        api_key="fake-key",
        base_url="https://api.example.test",
    )
    payload = llm.chat_json(system="system", user="user")

    assert payload == {"ok": True, "mode": "fallback"}
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_openai_compatible_llm_falls_back_from_generic_response_format_error(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise TypeError("response_format is not supported by this endpoint")
            return _Response('{"ok": true, "error_type": "generic"}')

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("gaam_graph.llm.OpenAI", FakeClient)

    llm = OpenAICompatibleLLM(
        model="fake-model",
        api_key="fake-key",
        base_url="https://api.example.test",
    )
    payload = llm.chat_json(system="system", user="user")

    assert payload == {"ok": True, "error_type": "generic"}
    assert len(calls) == 2


def test_openai_compatible_llm_wraps_fallback_failure(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("network unavailable")

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr("gaam_graph.llm.OpenAI", FakeClient)

    llm = OpenAICompatibleLLM(
        model="fake-model",
        api_key="fake-key",
        base_url="https://api.example.test",
    )

    try:
        llm.chat_json(system="system", user="user")
    except LLMError as exc:
        assert "network unavailable" in str(exc)
    else:
        raise AssertionError("Expected LLMError")


def test_openai_compatible_llm_raises_without_api_key():
    llm = OpenAICompatibleLLM(api_key=None)
    try:
        llm.chat_json(system="system", user="user")
    except LLMError as exc:
        assert "API_KEY" in str(exc)
    else:
        raise AssertionError("Expected LLMError")
