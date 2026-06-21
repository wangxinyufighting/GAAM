"""
Phase 3 Milestone 2: Local DataProto Adapter.

This module converts GAAM's Phase 2 actor update batches into tensor-ready
structures that mirror verl's DataProto contract, without requiring Ray, GPUs,
or the vendored verl runtime.

Design principles:
1. Tokenize prompts and responses deterministically
2. Create response-aligned reward/advantage tensors
3. Validate no teacher/oracle field leakage
4. Export JSON-safe batch representation
5. Optional PyTorch tensor export
6. Optional verl DataProto conversion (isolated boundary)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import json

from pydantic import BaseModel, Field

from gaam_graph.grpo_schema import ActorRole, ActorUpdateBatch, ActorUpdateItem
from gaam_graph.verl_batch_adapter import VerlLikeBatch
from gaam_graph.code_a1_alignment import check_for_forbidden_fields


# ============================================================================
# Simple Fallback Tokenizer
# ============================================================================

class SimpleFallbackTokenizer:
    """
    Deterministic whitespace/character tokenizer for CI testing.

    NOT production-ready. Use a real HuggingFace tokenizer when available.
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        self.backend = "simple_fallback"

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        """Tokenize text into deterministic integer IDs."""
        # Simple character-level tokenization
        char_ids = [ord(c) % 256 for c in text]

        # Add BOS token if requested
        if add_special_tokens:
            char_ids = [1] + char_ids  # BOS = 1

        return {
            "input_ids": char_ids,
            "attention_mask": [1] * len(char_ids),
        }

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text (best effort)."""
        # Skip special tokens
        filtered_ids = [tid for tid in token_ids if tid > 1]
        return "".join(chr(tid) if tid < 256 else "?" for tid in filtered_ids)


# ============================================================================
# Tensor Spec
# ============================================================================

class LocalTensorSpec(BaseModel):
    """Specification for a tensor field."""
    name: str
    dtype: str
    shape: list[int]
    required_for_verl: bool = True
    present: bool = True


# ============================================================================
# Local DataProto Sample
# ============================================================================

class LocalDataProtoSample(BaseModel):
    """A single training sample in local DataProto format."""
    sample_id: str
    group_id: str
    record_id: str
    actor_role: ActorRole
    selected_for_update: bool
    prompt_text: str
    response_text: str
    prompt_token_count: int
    response_token_count: int
    total_token_count: int
    reward: float
    advantage: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Local DataProto Batch
# ============================================================================

class LocalDataProtoBatch(BaseModel):
    """
    A batch of training samples in local DataProto format.

    This mirrors verl's DataProto structure but remains JSON-serializable.
    Actual tensors are stored separately in LocalDataProtoTensorBundle.
    """
    batch_id: str
    actor_role: ActorRole
    record_id: str | None = None
    round_id: int
    step_id: int
    tensorization_status: str
    samples: list[LocalDataProtoSample]
    tensor_specs: list[LocalTensorSpec]
    missing_fields_for_verl: list[str] = Field(default_factory=list)
    non_tensor_keys: list[str] = Field(default_factory=list)
    meta_info: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Tensor Bundle (Runtime-only)
# ============================================================================

@dataclass
class LocalDataProtoTensorBundle:
    """
    Runtime-only tensor bundle that mirrors verl DataProto structure.

    This uses PyTorch tensors if available, otherwise pure Python lists.
    """
    batch: dict[str, Any]  # Tensor fields (or lists for JSON)
    non_tensor_batch: dict[str, list[Any]]  # Object arrays
    meta_info: dict[str, Any]  # Runtime metadata


# ============================================================================
# Tokenization Helpers
# ============================================================================

def get_tokenizer(tokenizer: Any | None, pad_token_id: int = 0) -> Any:
    """
    Get tokenizer (HuggingFace or fallback).

    Args:
        tokenizer: Optional HuggingFace tokenizer
        pad_token_id: Padding token ID

    Returns:
        Tokenizer instance
    """
    if tokenizer is None:
        return SimpleFallbackTokenizer(pad_token_id=pad_token_id)
    return tokenizer


def tokenize_prompt_response(
    prompt: str,
    response: str,
    tokenizer: Any,
    max_prompt_length: int | None = None,
    max_response_length: int | None = None,
    truncation: Literal["error", "left", "right"] = "error",
    pad_token_id: int = 0,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Tokenize prompt and response separately.

    Args:
        prompt: Prompt text
        response: Response text
        tokenizer: Tokenizer instance
        max_prompt_length: Maximum prompt tokens
        max_response_length: Maximum response tokens
        truncation: Truncation strategy
        pad_token_id: Padding token ID

    Returns:
        (prompt_ids, response_ids, attention_mask, position_ids)
    """
    # Tokenize prompt with special tokens
    prompt_result = tokenizer(prompt, add_special_tokens=True)
    prompt_ids = prompt_result["input_ids"]

    # Tokenize response without special tokens
    response_result = tokenizer(response, add_special_tokens=False)
    response_ids = response_result["input_ids"]

    # Check truncation
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        if truncation == "error":
            raise ValueError(
                f"Prompt length {len(prompt_ids)} exceeds max_prompt_length {max_prompt_length}. "
                f"Set truncation='left' or 'right' to allow truncation."
            )
        elif truncation == "left":
            prompt_ids = prompt_ids[-max_prompt_length:]
        elif truncation == "right":
            prompt_ids = prompt_ids[:max_prompt_length]

    if max_response_length is not None and len(response_ids) > max_response_length:
        if truncation == "error":
            raise ValueError(
                f"Response length {len(response_ids)} exceeds max_response_length {max_response_length}. "
                f"Set truncation='left' or 'right' to allow truncation."
            )
        elif truncation == "left":
            response_ids = response_ids[-max_response_length:]
        elif truncation == "right":
            response_ids = response_ids[:max_response_length]

    # Concatenate prompt + response
    input_ids = prompt_ids + response_ids

    # Create attention mask and position IDs
    attention_mask = [1] * len(input_ids)
    position_ids = list(range(len(input_ids)))

    return input_ids, response_ids, attention_mask, position_ids


def create_response_aligned_tensors(
    total_length: int,
    response_start: int,
    response_length: int,
    reward: float,
    advantage: float,
) -> tuple[list[float], list[float], list[float]]:
    """
    Create response-aligned reward and advantage tensors.

    Places scalar reward/advantage values only on response token positions.

    Args:
        total_length: Total sequence length
        response_start: Response start position
        response_length: Response token count
        reward: Scalar reward
        advantage: Scalar advantage

    Returns:
        (token_level_scores, token_level_rewards, advantages)
    """
    # Initialize with zeros
    token_level_scores = [0.0] * total_length
    token_level_rewards = [0.0] * total_length
    advantages = [0.0] * total_length

    # Distribute reward/advantage across response tokens
    if response_length > 0:
        reward_per_token = reward / response_length
        advantage_per_token = advantage / response_length

        for i in range(response_start, response_start + response_length):
            if i < total_length:
                token_level_scores[i] = reward_per_token
                token_level_rewards[i] = reward_per_token
                advantages[i] = advantage_per_token

    return token_level_scores, token_level_rewards, advantages


# ============================================================================
# Batch Conversion Functions
# ============================================================================

def actor_update_batch_to_local_dataproto(
    batch: ActorUpdateBatch,
    *,
    tokenizer: Any | None = None,
    include_unselected: bool = False,
    max_prompt_length: int | None = None,
    max_response_length: int | None = None,
    truncation: Literal["error", "left", "right"] = "error",
    pad_token_id: int = 0,
) -> tuple[LocalDataProtoBatch, LocalDataProtoTensorBundle]:
    """
    Convert ActorUpdateBatch to local DataProto format.

    Args:
        batch: Actor update batch to convert
        tokenizer: Optional HuggingFace tokenizer (uses fallback if None)
        include_unselected: Whether to include unselected samples
        max_prompt_length: Maximum prompt tokens
        max_response_length: Maximum response tokens
        truncation: Truncation strategy ("error", "left", "right")
        pad_token_id: Padding token ID

    Returns:
        (LocalDataProtoBatch, LocalDataProtoTensorBundle)
    """
    # Validate batch safety
    from gaam_graph.grpo_trainer import assert_actor_update_batch_safe
    assert_actor_update_batch_safe(batch, allow_teacher_fields_for_stub=False, dry_run=False)

    # Get tokenizer
    tok = get_tokenizer(tokenizer, pad_token_id)
    tokenizer_backend = getattr(tok, "backend", "hf_tokenizer")

    # Filter items
    if include_unselected:
        items = batch.items
    else:
        items = [item for item in batch.items if item.selected_for_update]

    # Convert items to samples with tensors
    samples = []
    all_input_ids = []
    all_responses = []
    all_response_masks = []
    all_attention_masks = []
    all_position_ids = []
    all_token_level_scores = []
    all_token_level_rewards = []
    all_advantages = []

    sample_ids = []
    group_ids = []
    record_ids = []
    prompt_texts = []
    response_texts = []

    for item in items:
        # Check for empty response
        if not item.response.strip():
            if not include_unselected:
                # Skip empty responses for selected samples
                continue

        # Tokenize
        try:
            input_ids, response_ids, attention_mask, position_ids = tokenize_prompt_response(
                prompt=item.prompt,
                response=item.response,
                tokenizer=tok,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                truncation=truncation,
                pad_token_id=pad_token_id,
            )
        except ValueError as e:
            # Truncation error
            raise ValueError(f"Failed to tokenize sample {item.sample_id}: {e}") from e

        prompt_token_count = len(input_ids) - len(response_ids)
        response_token_count = len(response_ids)
        total_token_count = len(input_ids)

        # Create response mask
        response_mask = [0] * prompt_token_count + [1] * response_token_count

        # Create response-aligned tensors
        token_level_scores, token_level_rewards, advantages = create_response_aligned_tensors(
            total_length=total_token_count,
            response_start=prompt_token_count,
            response_length=response_token_count,
            reward=item.reward,
            advantage=item.advantage,
        )

        # Add sample
        samples.append(LocalDataProtoSample(
            sample_id=item.sample_id,
            group_id=item.group_id,
            record_id=item.record_id,
            actor_role=batch.role,
            selected_for_update=item.selected_for_update,
            prompt_text=item.prompt,
            response_text=item.response,
            prompt_token_count=prompt_token_count,
            response_token_count=response_token_count,
            total_token_count=total_token_count,
            reward=item.reward,
            advantage=item.advantage,
            metadata={},
        ))

        # Collect tensors
        all_input_ids.append(input_ids)
        all_responses.append(response_ids)
        all_response_masks.append(response_mask)
        all_attention_masks.append(attention_mask)
        all_position_ids.append(position_ids)
        all_token_level_scores.append(token_level_scores)
        all_token_level_rewards.append(token_level_rewards)
        all_advantages.append(advantages)

        # Collect non-tensor metadata
        sample_ids.append(item.sample_id)
        group_ids.append(item.group_id)
        record_ids.append(item.record_id)
        prompt_texts.append(item.prompt)
        response_texts.append(item.response)

    # Build tensor specs
    tensor_specs = [
        LocalTensorSpec(name="input_ids", dtype="int64", shape=[len(samples), -1], present=True),
        LocalTensorSpec(name="attention_mask", dtype="int64", shape=[len(samples), -1], present=True),
        LocalTensorSpec(name="position_ids", dtype="int64", shape=[len(samples), -1], present=True),
        LocalTensorSpec(name="responses", dtype="int64", shape=[len(samples), -1], present=True),
        LocalTensorSpec(name="response_mask", dtype="int64", shape=[len(samples), -1], present=True),
        LocalTensorSpec(name="token_level_scores", dtype="float32", shape=[len(samples), -1], present=True),
        LocalTensorSpec(name="token_level_rewards", dtype="float32", shape=[len(samples), -1], present=True),
        LocalTensorSpec(name="advantages", dtype="float32", shape=[len(samples), -1], present=True),
        LocalTensorSpec(name="old_log_probs", dtype="float32", shape=[len(samples), -1], required_for_verl=True, present=False),
        LocalTensorSpec(name="ref_log_probs", dtype="float32", shape=[len(samples), -1], required_for_verl=True, present=False),
        LocalTensorSpec(name="token_level_logprobs", dtype="float32", shape=[len(samples), -1], required_for_verl=True, present=False),
        LocalTensorSpec(name="returns", dtype="float32", shape=[len(samples), -1], required_for_verl=True, present=False),
    ]

    # Missing fields for full verl compatibility
    missing_fields = [
        "old_log_probs",
        "ref_log_probs",
        "token_level_logprobs",
        "returns",
    ]

    # Build tensor bundle
    tensor_bundle = LocalDataProtoTensorBundle(
        batch={
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "position_ids": all_position_ids,
            "responses": all_responses,
            "response_mask": all_response_masks,
            "token_level_scores": all_token_level_scores,
            "token_level_rewards": all_token_level_rewards,
            "advantages": all_advantages,
        },
        non_tensor_batch={
            "sample_id": sample_ids,
            "group_id": group_ids,
            "record_id": record_ids,
            "prompt_text": prompt_texts,
            "response_text": response_texts,
        },
        meta_info={
            "tokenizer_backend": tokenizer_backend,
            "batch_id": batch.batch_id,
            "actor_role": batch.role.value,
            "round_id": batch.round_id,
            "step_id": batch.step_id,
            "num_samples": len(samples),
            "include_unselected": include_unselected,
        },
    )

    # Build LocalDataProtoBatch
    local_batch = LocalDataProtoBatch(
        batch_id=batch.batch_id,
        actor_role=batch.role,
        record_id=samples[0].record_id if samples else None,
        round_id=batch.round_id,
        step_id=batch.step_id,
        tensorization_status="tensorized" if samples else "empty",
        samples=samples,
        tensor_specs=tensor_specs,
        missing_fields_for_verl=missing_fields,
        non_tensor_keys=list(tensor_bundle.non_tensor_batch.keys()),
        meta_info=tensor_bundle.meta_info,
    )

    return local_batch, tensor_bundle


def verl_like_batch_to_local_dataproto(
    batch: VerlLikeBatch,
    *,
    tokenizer: Any | None = None,
    include_unselected: bool = False,
    max_prompt_length: int | None = None,
    max_response_length: int | None = None,
    truncation: Literal["error", "left", "right"] = "error",
    pad_token_id: int = 0,
) -> tuple[LocalDataProtoBatch, LocalDataProtoTensorBundle]:
    """
    Convert VerlLikeBatch to local DataProto format.

    Args:
        batch: VerlLikeBatch to convert
        tokenizer: Optional HuggingFace tokenizer
        include_unselected: Whether to include unselected samples
        max_prompt_length: Maximum prompt tokens
        max_response_length: Maximum response tokens
        truncation: Truncation strategy
        pad_token_id: Padding token ID

    Returns:
        (LocalDataProtoBatch, LocalDataProtoTensorBundle)
    """
    _assert_no_forbidden_fields(
        batch.model_dump(mode="json"),
        f"VerlLikeBatch {batch.batch_id}",
    )

    # Convert VerlLikeBatch to ActorUpdateBatch format
    from gaam_graph.grpo_schema import ActorUpdateItem

    items = []
    for sample in batch.samples:
        if not include_unselected and not sample.selected_for_update:
            continue

        items.append(ActorUpdateItem(
            sample_id=sample.sample_id,
            group_id=sample.group_id,
            record_id=sample.record_id,
            role=sample.actor_role,
            prompt=sample.prompt,
            response=sample.response,
            reward=sample.reward,
            advantage=sample.advantage,
            selected_for_update=sample.selected_for_update,
            metadata=sample.metadata,
        ))

    temp_batch = ActorUpdateBatch(
        batch_id=batch.batch_id,
        role=batch.actor_role,
        round_id=batch.round_id,
        step_id=batch.step_id,
        items=items,
        metadata=batch.metadata,
    )

    # Use existing conversion
    return actor_update_batch_to_local_dataproto(
        batch=temp_batch,
        tokenizer=tokenizer,
        include_unselected=include_unselected,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        truncation=truncation,
        pad_token_id=pad_token_id,
    )


# ============================================================================
# Export Functions
# ============================================================================

class LocalDataProtoExportManifestEntry(BaseModel):
    """Manifest entry for exported local DataProto batch."""
    record_id: str
    role: ActorRole
    source_batch_path: str
    output_json_path: str
    output_tensor_path: str | None = None
    num_samples: int
    tensorization_status: str
    missing_fields_for_verl: list[str]


def _assert_no_forbidden_fields(data: dict[str, Any], source: str) -> None:
    """Fail closed before tensorization if any teacher/oracle field is present."""
    forbidden = check_for_forbidden_fields(data)
    if forbidden:
        raise ValueError(
            f"{source} contains forbidden teacher/oracle fields: {', '.join(forbidden)}"
        )


def _parse_batch_filename(batch_file: Path, source_type: str) -> tuple[str, ActorRole]:
    """Parse record id and actor role from a known Phase 2 batch filename."""
    name = batch_file.name
    suffixes = {
        "rollout": {
            ".memory_update_batch.json": ActorRole.MEMORY_BUILDER,
            ".question_update_batch.json": ActorRole.QUESTION_AGENT,
        },
        "verl_like": {
            ".memory.verl_like_batch.json": ActorRole.MEMORY_BUILDER,
            ".question.verl_like_batch.json": ActorRole.QUESTION_AGENT,
        },
    }

    for suffix, role in suffixes[source_type].items():
        if name.endswith(suffix):
            return name.removesuffix(suffix), role

    raise ValueError(f"Unrecognized {source_type} batch filename: {name}")


def _pad_numeric_rows(rows: list[list[Any]], *, pad_value: int | float, dtype: Any, torch: Any) -> Any:
    max_len = max((len(row) for row in rows), default=0)
    padded = [row + [pad_value] * (max_len - len(row)) for row in rows]
    return torch.tensor(padded, dtype=dtype)


def _tensor_bundle_to_torch_payload(bundle: LocalDataProtoTensorBundle, torch: Any) -> dict[str, Any]:
    """Convert ragged local lists to padded torch tensors for debug/loading exports."""
    int_fields = {"input_ids", "attention_mask", "position_ids", "responses", "response_mask"}
    float_fields = {"token_level_scores", "token_level_rewards", "advantages"}

    batch = {}
    sequence_lengths = {}
    for key, rows in bundle.batch.items():
        sequence_lengths[key] = [len(row) for row in rows]
        if key in int_fields:
            batch[key] = _pad_numeric_rows(rows, pad_value=0, dtype=torch.long, torch=torch)
        elif key in float_fields:
            batch[key] = _pad_numeric_rows(rows, pad_value=0.0, dtype=torch.float32, torch=torch)
        else:
            batch[key] = rows

    return {
        "batch": batch,
        "non_tensor_batch": bundle.non_tensor_batch,
        "meta_info": {
            **bundle.meta_info,
            "torch_export_padded": True,
            "sequence_lengths": sequence_lengths,
        },
    }


def export_local_dataproto_batches(
    *,
    rollout_dir: Path | None = None,
    verl_like_dir: Path | None = None,
    output_dir: Path,
    tokenizer: Any | None = None,
    record_id: str | None = None,
    max_records: int | None = None,
    include_unselected: bool = False,
    max_prompt_length: int | None = None,
    max_response_length: int | None = None,
    truncation: Literal["error", "left", "right"] = "error",
    write_torch_tensors: bool = False,
) -> list[Path]:
    """
    Export all actor update batches as local DataProto batches.

    Args:
        rollout_dir: Path to Phase 2 rollout directory
        verl_like_dir: Path to Phase 2 verl-like batches directory
        output_dir: Output directory
        tokenizer: Optional HuggingFace tokenizer
        record_id: Optional single record ID to export
        max_records: Maximum number of records to process
        include_unselected: Include unselected samples
        max_prompt_length: Maximum prompt tokens
        max_response_length: Maximum response tokens
        truncation: Truncation strategy
        write_torch_tensors: Write PyTorch tensor files (.pt)

    Returns:
        List of exported JSON file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths = []
    manifest_entries = []

    # Determine source directory
    if rollout_dir is not None:
        source_dir = Path(rollout_dir) / "update_batches"
        source_type = "rollout"
    elif verl_like_dir is not None:
        source_dir = Path(verl_like_dir)
        source_type = "verl_like"
    else:
        raise ValueError("Either rollout_dir or verl_like_dir must be provided")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Find batch files
    if source_type == "rollout":
        batch_files = list(source_dir.glob("*.memory_update_batch.json")) + \
                     list(source_dir.glob("*.question_update_batch.json"))
    else:
        batch_files = list(source_dir.glob("*.memory.verl_like_batch.json")) + \
                     list(source_dir.glob("*.question.verl_like_batch.json"))

    # Filter by record_id if specified
    if record_id is not None:
        batch_files = [
            f for f in batch_files
            if _parse_batch_filename(f, source_type)[0] == record_id
        ]

    # Respect max_records
    records_seen = set()
    filtered_files = []
    for f in batch_files:
        rec_id, _ = _parse_batch_filename(f, source_type)
        if rec_id not in records_seen:
            records_seen.add(rec_id)
            if max_records is not None and len(records_seen) > max_records:
                break
        filtered_files.append(f)

    batch_files = filtered_files

    # Convert each batch
    for batch_file in batch_files:
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)

        rec_id, role = _parse_batch_filename(batch_file, source_type)
        _assert_no_forbidden_fields(batch_data, str(batch_file))

        # Load batch
        if source_type == "rollout":
            batch = ActorUpdateBatch.model_validate(batch_data)
            local_batch, tensor_bundle = actor_update_batch_to_local_dataproto(
                batch=batch,
                tokenizer=tokenizer,
                include_unselected=include_unselected,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                truncation=truncation,
            )
        else:
            batch = VerlLikeBatch.model_validate(batch_data)
            local_batch, tensor_bundle = verl_like_batch_to_local_dataproto(
                batch=batch,
                tokenizer=tokenizer,
                include_unselected=include_unselected,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                truncation=truncation,
            )

        # Write JSON
        role_name = role.value
        output_json_path = output_dir / f"{rec_id}.{role_name}.local_dataproto.json"

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(local_batch.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        exported_paths.append(output_json_path)

        # Write tensors if requested
        output_tensor_path = None
        if write_torch_tensors:
            try:
                import torch
                output_tensor_path = output_dir / f"{rec_id}.{role_name}.local_dataproto.pt"
                torch.save(_tensor_bundle_to_torch_payload(tensor_bundle, torch), output_tensor_path)
            except ImportError:
                # PyTorch not available, skip tensor export
                pass

        # Add to manifest
        manifest_entries.append(LocalDataProtoExportManifestEntry(
            record_id=rec_id,
            role=role,
            source_batch_path=str(batch_file),
            output_json_path=str(output_json_path.relative_to(output_dir)),
            output_tensor_path=str(output_tensor_path.relative_to(output_dir)) if output_tensor_path else None,
            num_samples=len(local_batch.samples),
            tensorization_status=local_batch.tensorization_status,
            missing_fields_for_verl=local_batch.missing_fields_for_verl,
        ))

    # Write manifest
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n")

    # Write summary
    summary = {
        "source_type": source_type,
        "source_dir": str(source_dir),
        "num_batches": len(exported_paths),
        "include_unselected": include_unselected,
        "max_prompt_length": max_prompt_length,
        "max_response_length": max_response_length,
        "truncation": truncation,
        "write_torch_tensors": write_torch_tensors,
        "tokenizer_backend": "simple_fallback" if tokenizer is None else "hf_tokenizer",
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return exported_paths


# ============================================================================
# Optional verl DataProto Conversion
# ============================================================================

def local_bundle_to_verl_dataproto(bundle: LocalDataProtoTensorBundle) -> Any:
    """
    Convert LocalDataProtoTensorBundle to verl DataProto.

    This is an optional boundary for future verl integration.
    Requires verl to be installed.

    Args:
        bundle: Local tensor bundle

    Returns:
        verl DataProto instance

    Raises:
        RuntimeError: If verl is not installed
    """
    try:
        from verl import DataProto
    except ImportError as e:
        raise RuntimeError(
            "verl is not installed; use JSON/Torch local export instead. "
            "Install verl to enable DataProto conversion."
        ) from e

    # Convert to verl DataProto
    # This is a placeholder - actual conversion would depend on verl API
    raise NotImplementedError(
        "verl DataProto conversion will be implemented in Phase 3 Milestone 4 "
        "when verl trainer integration is added."
    )
