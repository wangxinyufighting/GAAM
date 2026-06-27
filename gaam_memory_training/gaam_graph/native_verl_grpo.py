"""
Native VERL GRPO dataset export utilities for GAAM.

This module builds VERL-compatible parquet rows from GAAM raw histories and
oracle graph artifacts. The exported rows are consumed by
``python -m verl.trainer.main_ppo`` with ``algorithm.adv_estimator=grpo``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gaam_graph.lme_loader import LMERecord, LongMemEvalLoader
from gaam_graph.raw_history import raw_history_from_lme_record, raw_history_to_prompt_payload
from gaam_graph.utils import read_json, stable_id, write_json


VALID_ACTOR_ROLES = {"memory_builder", "question_agent"}
TEXT_KEYS = {
    "text",
    "content",
    "summary",
    "description",
    "canonical_name",
    "surface",
    "subject",
    "predicate",
    "object",
    "topic",
    "event_category",
}


@dataclass(frozen=True)
class NativeVerlExportConfig:
    input_path: Path
    oracle_graph_dir: Path
    output_dir: Path
    actor_role: str
    split_manifest_path: Path | None = None
    max_history_chars: int = 16000
    max_oracle_chars: int = 12000
    max_records_per_split: int | None = None
    val_split_preference: tuple[str, ...] = ("dev", "test", "train")


def export_native_verl_grpo_dataset(config: NativeVerlExportConfig) -> dict[str, Any]:
    """Export GAAM actor prompts to VERL train/val parquet files."""
    if config.actor_role not in VALID_ACTOR_ROLES:
        raise ValueError(f"Unsupported actor_role: {config.actor_role}")

    records = LongMemEvalLoader(str(config.input_path)).load()
    records_by_id = {record.record_id: record for record in records}
    split_records = _resolve_split_records(config, records_by_id)

    train_ids = split_records.get("train", [])
    val_ids = _choose_val_ids(split_records, config.val_split_preference)
    if not train_ids:
        raise ValueError("No train records available for native VERL export.")
    if not val_ids:
        val_ids = train_ids[:1]

    train_rows = _build_rows(config, train_ids, records_by_id, split_name="train")
    val_rows = _build_rows(config, val_ids, records_by_id, split_name="val")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = config.output_dir / f"{config.actor_role}.train.parquet"
    val_path = config.output_dir / f"{config.actor_role}.val.parquet"

    _write_parquet(train_rows, train_path)
    _write_parquet(val_rows, val_path)

    manifest = {
        "manifest_version": "gaam_native_verl_grpo_dataset_v1",
        "actor_role": config.actor_role,
        "input_path": str(config.input_path),
        "oracle_graph_dir": str(config.oracle_graph_dir),
        "split_manifest_path": str(config.split_manifest_path) if config.split_manifest_path else None,
        "output_dir": str(config.output_dir),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "num_train_rows": len(train_rows),
        "num_val_rows": len(val_rows),
        "train_record_ids": train_ids,
        "val_record_ids": val_ids,
        "max_history_chars": config.max_history_chars,
        "max_oracle_chars": config.max_oracle_chars,
        "notes": [
            "Rows are compatible with verl.trainer.main_ppo.",
            "Memory Builder rows use raw history only and do not include benchmark target questions.",
            "Question Agent rows use sanitized oracle graph content and do not include benchmark target questions.",
        ],
    }
    manifest_path = config.output_dir / f"{config.actor_role}.dataset_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def _resolve_split_records(
    config: NativeVerlExportConfig,
    records_by_id: dict[str, LMERecord],
) -> dict[str, list[str]]:
    if config.split_manifest_path:
        split_manifest = read_json(config.split_manifest_path)
        split_records: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
        for item in split_manifest.get("records", []):
            record_id = str(item.get("record_id", ""))
            split = str(item.get("split", ""))
            if record_id in records_by_id and (config.oracle_graph_dir / f"{record_id}.graph.json").exists():
                split_records.setdefault(split, []).append(record_id)
        return _limit_split_records(split_records, config.max_records_per_split)

    graph_ids = {
        path.name.removesuffix(".graph.json")
        for path in config.oracle_graph_dir.glob("*.graph.json")
    }
    available_ids = [record_id for record_id in records_by_id if record_id in graph_ids]
    if not available_ids:
        raise ValueError(f"No records with oracle graphs found in {config.oracle_graph_dir}")

    if len(available_ids) == 1:
        split_records = {"train": available_ids, "dev": available_ids, "test": []}
    else:
        dev_count = max(1, round(len(available_ids) * 0.1))
        split_records = {
            "train": available_ids[:-dev_count],
            "dev": available_ids[-dev_count:],
            "test": [],
        }
        if not split_records["train"]:
            split_records["train"] = available_ids
    return _limit_split_records(split_records, config.max_records_per_split)


def _limit_split_records(
    split_records: dict[str, list[str]],
    max_records_per_split: int | None,
) -> dict[str, list[str]]:
    if max_records_per_split is None:
        return split_records
    return {
        split: record_ids[:max_records_per_split]
        for split, record_ids in split_records.items()
    }


def _choose_val_ids(split_records: dict[str, list[str]], preference: tuple[str, ...]) -> list[str]:
    for split in preference:
        ids = split_records.get(split, [])
        if ids:
            return ids
    return []


def _build_rows(
    config: NativeVerlExportConfig,
    record_ids: list[str],
    records_by_id: dict[str, LMERecord],
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, record_id in enumerate(record_ids):
        record = records_by_id[record_id]
        graph_path = config.oracle_graph_dir / f"{record_id}.graph.json"
        oracle_graph = read_json(graph_path)
        oracle_digest = _oracle_graph_digest(
            oracle_graph,
            max_chars=config.max_oracle_chars,
        )
        required_terms = _extract_required_terms(oracle_graph, limit=64)

        if config.actor_role == "memory_builder":
            prompt = _memory_builder_prompt(record, config.max_history_chars)
            ground_truth = {
                "actor_role": "memory_builder",
                "record_id": record_id,
                "required_terms": required_terms,
                "oracle_digest": oracle_digest,
            }
            ability = "memory_building"
            data_source = "gaam_memory_builder"
        else:
            prompt = _question_agent_prompt(oracle_digest)
            ground_truth = {
                "actor_role": "question_agent",
                "record_id": record_id,
                "required_terms": required_terms,
                "oracle_digest": oracle_digest,
            }
            ability = "question_generation"
            data_source = "gaam_question_agent"

        rows.append(
            {
                "data_source": data_source,
                "prompt": prompt,
                "ability": ability,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split_name,
                    "index": index,
                    "record_id": record_id,
                    "actor_role": config.actor_role,
                    "oracle_graph_path": str(graph_path),
                    "row_id": stable_id("verl_row", config.actor_role, split_name, record_id, index),
                },
            }
        )
    return rows


def _memory_builder_prompt(record: LMERecord, max_history_chars: int) -> list[dict[str, str]]:
    history = raw_history_from_lme_record(record)
    payload = raw_history_to_prompt_payload(history)
    history_text = _format_history_payload(payload)
    history_text = _limit_text(history_text, max_history_chars)
    system = (
        "You are the GAAM Memory Builder. Build a compact Current Memory from raw conversation "
        "history only. Do not mention benchmark questions or answers. Output JSON with keys "
        "record_id, memory_graph, memory_summaries, and metadata."
    )
    user = (
        "Raw history:\n"
        f"{history_text}\n\n"
        "Create a non-redundant Current Memory. Preserve concrete facts, updates, preferences, "
        "multi-session evidence, and useful abstractions. Include provenance fields when possible."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _question_agent_prompt(oracle_digest: str) -> list[dict[str, str]]:
    system = (
        "You are the GAAM Question Agent. Generate diverse training questions from an oracle "
        "memory graph digest. Do not copy or infer any benchmark target question. Output JSON "
        "with a questions array. Each item should include question, type, difficulty, and "
        "required_evidence."
    )
    user = (
        "Oracle graph digest:\n"
        f"{oracle_digest}\n\n"
        "Generate questions covering single-hop, multi-hop, multi-session, temporal, preference, "
        "personal fact, contradiction/update, and summary/abstraction cases when supported."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _format_history_payload(payload: dict[str, Any]) -> str:
    lines = [f"record_id: {payload.get('record_id', '')}"]
    for session in payload.get("sessions", []):
        session_id = session.get("session_id", "")
        timestamp = session.get("timestamp", "")
        lines.append(f"\n[session {session_id} timestamp={timestamp}]")
        for turn in session.get("turns", []):
            lines.append(
                f"- event_id={turn.get('event_id', '')} turn={turn.get('turn_id', '')} "
                f"speaker={turn.get('speaker', '')}: {turn.get('text', '')}"
            )
    return "\n".join(lines)


def _oracle_graph_digest(oracle_graph: dict[str, Any], *, max_chars: int) -> str:
    nodes_by_type: dict[str, list[str]] = {}
    for node in oracle_graph.get("nodes", []):
        node_type = str(node.get("type", "unknown"))
        text = _node_to_text(node)
        if text:
            nodes_by_type.setdefault(node_type, []).append(text)

    lines = []
    for node_type in sorted(nodes_by_type):
        lines.append(f"\n[{node_type}]")
        for text in nodes_by_type[node_type][:40]:
            lines.append(f"- {_limit_text(text, 500)}")

    edge_type_counts: dict[str, int] = {}
    for edge in oracle_graph.get("edges", []):
        edge_type = str(edge.get("type", "unknown"))
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
    if edge_type_counts:
        lines.append("\n[edge_type_counts]")
        for edge_type, count in sorted(edge_type_counts.items()):
            lines.append(f"- {edge_type}: {count}")

    return _limit_text("\n".join(lines).strip(), max_chars)


def _node_to_text(node: dict[str, Any]) -> str:
    attrs = node.get("attrs", {})
    chunks = [str(node.get("id", "")), str(node.get("type", ""))]
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            if key in TEXT_KEYS and value:
                chunks.append(f"{key}={value}")
    for key, value in node.items():
        if key in TEXT_KEYS and value:
            chunks.append(f"{key}={value}")
    return " | ".join(chunk for chunk in chunks if chunk)


def _extract_required_terms(oracle_graph: dict[str, Any], *, limit: int) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for node in oracle_graph.get("nodes", []):
        for text in _candidate_texts(node):
            for term in _split_terms(text):
                key = term.lower()
                if key not in seen:
                    terms.append(term)
                    seen.add(key)
                if len(terms) >= limit:
                    return terms
    return terms


def _candidate_texts(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in TEXT_KEYS and isinstance(child, str):
                texts.append(child)
            elif isinstance(child, (dict, list)):
                texts.extend(_candidate_texts(child))
    elif isinstance(value, list):
        for child in value:
            texts.extend(_candidate_texts(child))
    return texts


def _split_terms(text: str) -> list[str]:
    import re

    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{3,}|[\u4e00-\u9fff]{2,}", text)
    return [candidate[:80] for candidate in candidates]


def _limit_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 32].rstrip() + "\n...[truncated for VERL prompt budget]"


def _write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pandas is required to export VERL parquet files. Install requirements.txt "
            "or run: pip install pandas pyarrow"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)
