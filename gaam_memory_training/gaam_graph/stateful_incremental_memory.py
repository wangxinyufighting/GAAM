"""Stateful incremental Current Memory construction.

This module runs Memory Builder as a real sequential process: chunk ``k`` sees
the Current Memory generated after chunk ``k - 1`` plus the next raw-history
chunk. It is intended for production rollout/evaluation paths where independent
parquet rows are not sufficient to represent memory state.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from gaam_graph.llm import LocalHFChatLLM, OpenAICompatibleLLM, parse_json_object
from gaam_graph.lme_loader import LMERecord, LongMemEvalLoader
from gaam_graph.native_verl_grpo import (
    _chunk_sessions,
    _format_history_payload,
    _history_payload_from_sessions,
    _limit_text,
    _previous_current_memory_scaffold,
)
from gaam_graph.utils import write_json


@dataclass(frozen=True)
class StatefulIncrementalMemoryConfig:
    input_path: Path
    output_dir: Path
    record_id: str | None = None
    max_records: int | None = None
    session_chunk_size: int = 4
    max_chunk_chars: int = 12000
    max_previous_memory_chars: int = 12000
    llm_backend: str = "api"  # api or local_hf
    model: str = os.getenv("GAAM_MEMORY_BUILDER_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    api_key: str | None = os.getenv("GAAM_MEMORY_BUILDER_API_KEY", os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY")))
    base_url: str = os.getenv("GAAM_MEMORY_BUILDER_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")))
    temperature: float = 0.0
    max_new_tokens: int = 2048
    device_map: str = "auto"
    torch_dtype: str = "auto"


def build_stateful_incremental_memories(config: StatefulIncrementalMemoryConfig) -> dict[str, Any]:
    records = LongMemEvalLoader(str(config.input_path)).load()
    if config.record_id:
        records = [record for record in records if record.record_id == config.record_id]
    if config.max_records is not None:
        records = records[: config.max_records]
    if not records:
        raise ValueError("No records selected for stateful incremental memory building.")

    llm = create_stateful_memory_llm(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    record_reports = []
    for record in records:
        report = build_record_stateful_incremental_memory(record, config, llm)
        record_reports.append(report)

    manifest = {
        "manifest_version": "gaam_stateful_incremental_memory_v1",
        "status": "succeeded",
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "record_id": config.record_id,
        "max_records": config.max_records,
        "session_chunk_size": config.session_chunk_size,
        "max_chunk_chars": config.max_chunk_chars,
        "max_previous_memory_chars": config.max_previous_memory_chars,
        "llm_backend": config.llm_backend,
        "model": config.model,
        "base_url": config.base_url if config.llm_backend == "api" else None,
        "records": record_reports,
    }
    write_json(config.output_dir / "stateful_incremental_memory_manifest.json", manifest)
    return manifest


def build_record_stateful_incremental_memory(
    record: LMERecord,
    config: StatefulIncrementalMemoryConfig,
    llm: OpenAICompatibleLLM,
) -> dict[str, Any]:
    from gaam_graph.raw_history import raw_history_from_lme_record

    history = raw_history_from_lme_record(record)
    chunks = _chunk_sessions(history.sessions, config.session_chunk_size)
    previous_memory: dict[str, Any] = _previous_current_memory_scaffold(
        record_id=record.record_id,
        processed_session_ids=[],
        build_step=0,
    )
    processed_session_ids: list[str] = []
    trace = []

    for chunk_index, sessions in enumerate(chunks):
        session_ids = [str(session.session_id) for session in sessions]
        prompt = _stateful_incremental_prompt(
            record_id=record.record_id,
            previous_memory=previous_memory,
            sessions=sessions,
            processed_session_ids=processed_session_ids,
            chunk_index=chunk_index,
            num_chunks=len(chunks),
            max_chunk_chars=config.max_chunk_chars,
            max_previous_memory_chars=config.max_previous_memory_chars,
        )
        response = llm.chat_json(system=prompt[0]["content"], user=prompt[1]["content"])
        previous_memory = _normalize_memory_response(response, record.record_id)
        processed_session_ids.extend(session_ids)
        trace.append(
            {
                "chunk_index": chunk_index,
                "num_chunks": len(chunks),
                "session_ids": session_ids,
                "processed_session_ids": list(processed_session_ids),
                "memory_node_count": len(previous_memory.get("memory_graph", {}).get("nodes", [])),
                "memory_edge_count": len(previous_memory.get("memory_graph", {}).get("edges", [])),
            }
        )

    record_dir = config.output_dir / record.record_id
    record_dir.mkdir(parents=True, exist_ok=True)
    memory_path = record_dir / "current_memory.stateful_incremental.json"
    trace_path = record_dir / "stateful_incremental_trace.json"
    write_json(memory_path, previous_memory)
    write_json(trace_path, {"record_id": record.record_id, "trace": trace})

    return {
        "record_id": record.record_id,
        "status": "succeeded",
        "num_chunks": len(chunks),
        "final_memory_path": str(memory_path),
        "trace_path": str(trace_path),
    }


def create_stateful_memory_llm(config: StatefulIncrementalMemoryConfig) -> Any:
    if config.llm_backend == "api":
        return OpenAICompatibleLLM(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
        )
    if config.llm_backend == "local_hf":
        return LocalHFChatLLM(
            model_path=config.model,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            device_map=config.device_map,
            torch_dtype=config.torch_dtype,
        )
    raise ValueError(f"Unsupported stateful memory llm_backend: {config.llm_backend}")


def _stateful_incremental_prompt(
    *,
    record_id: str,
    previous_memory: dict[str, Any],
    sessions: list[Any],
    processed_session_ids: list[str],
    chunk_index: int,
    num_chunks: int,
    max_chunk_chars: int,
    max_previous_memory_chars: int,
) -> list[dict[str, str]]:
    payload = _history_payload_from_sessions(record_id, sessions)
    chunk_text = _limit_text(_format_history_payload(payload), max_chunk_chars)
    previous_memory_text = _limit_text(
        json.dumps(previous_memory, ensure_ascii=False, indent=2),
        max_previous_memory_chars,
    )
    system = (
        "You are the GAAM Memory Builder. You incrementally build Current Memory from raw "
        "conversation history only. Never use or mention benchmark target questions, gold answers, "
        "or evaluation metadata. Return only valid JSON."
    )
    user = (
        f"record_id: {record_id}\n"
        f"incremental_step: {chunk_index + 1}/{num_chunks}\n"
        f"already_processed_session_ids: {processed_session_ids}\n\n"
        "Previous Current Memory JSON:\n"
        f"{previous_memory_text}\n\n"
        "New raw-history chunk:\n"
        f"{chunk_text}\n\n"
        "Return the full updated Current Memory JSON with keys record_id, memory_graph, "
        "memory_summaries, and metadata. Merge useful new evidence, preserve provenance, remove "
        "redundancy, avoid over-compression, and keep stable abstractions."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _normalize_memory_response(response: dict[str, Any], record_id: str) -> dict[str, Any]:
    if not isinstance(response, dict):
        response = parse_json_object(str(response))
    response.setdefault("record_id", record_id)
    response.setdefault("memory_graph", {"nodes": [], "edges": []})
    response.setdefault("memory_summaries", {})
    response.setdefault("metadata", {})
    response["metadata"]["build_mode"] = "stateful_incremental"
    return response
