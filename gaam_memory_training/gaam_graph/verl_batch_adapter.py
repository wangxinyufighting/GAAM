"""
verl-like batch adapter for Phase 2 Milestone 5.

This module converts GAAM actor update batches into a verl-compatible JSON-safe
batch representation without introducing verl as a runtime dependency.

Design principles:
1. Export actor-safe batches in a structure compatible with future verl tensorization
2. Document missing fields needed for full PPO/GRPO (old_logprobs, ref_logprobs)
3. Provide JSONL manifest for exported batches
4. Enable Phase 3 distributed training handoff
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from pydantic import BaseModel, Field

from gaam_graph.grpo_schema import ActorRole, ActorUpdateBatch


# ============================================================================
# verl-like Data Models
# ============================================================================

class VerlLikeSample(BaseModel):
    """A single training sample in verl-compatible format."""
    sample_id: str
    group_id: str
    record_id: str
    actor_role: ActorRole
    prompt: str
    response: str
    reward: float
    advantage: float
    selected_for_update: bool
    prompt_token_count: int | None = None
    response_token_count: int | None = None
    old_logprob_sum: float | None = None
    ref_logprob_sum: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerlLikeBatch(BaseModel):
    """A batch of training samples in verl-compatible format."""
    batch_id: str
    actor_role: ActorRole
    round_id: int
    step_id: int
    samples: list[VerlLikeSample]
    tensorization_status: str = "not_tensorized"
    missing_fields_for_verl: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Conversion Functions
# ============================================================================

def actor_update_batch_to_verl_like_batch(
    batch: ActorUpdateBatch,
    *,
    tokenizer: Any | None = None,
    include_unselected: bool = False,
) -> VerlLikeBatch:
    """
    Convert ActorUpdateBatch to verl-like batch format.

    Args:
        batch: Actor update batch to convert
        tokenizer: Optional tokenizer for token counting
        include_unselected: Whether to include unselected items

    Returns:
        verl-like batch with JSON-safe format
    """
    # Validate batch safety
    from gaam_graph.grpo_trainer import assert_actor_update_batch_safe
    assert_actor_update_batch_safe(batch, allow_teacher_fields_for_stub=False, dry_run=False)

    # Filter items
    if include_unselected:
        items = batch.items
    else:
        items = [item for item in batch.items if item.selected_for_update]

    # Convert items to samples
    samples = []
    for item in items:
        # Count tokens if tokenizer provided
        prompt_token_count = None
        response_token_count = None

        if tokenizer is not None:
            try:
                prompt_tokens = tokenizer(item.prompt, add_special_tokens=True)
                prompt_token_count = len(prompt_tokens["input_ids"])

                response_tokens = tokenizer(item.response, add_special_tokens=False)
                response_token_count = len(response_tokens["input_ids"])
            except Exception:
                # Tokenization failed, leave counts as None
                pass

        sample = VerlLikeSample(
            sample_id=item.sample_id,
            group_id=item.group_id,
            record_id=item.record_id,
            actor_role=batch.role,
            prompt=item.prompt,
            response=item.response,
            reward=item.reward,
            advantage=item.advantage,
            selected_for_update=item.selected_for_update,
            prompt_token_count=prompt_token_count,
            response_token_count=response_token_count,
            old_logprob_sum=None,  # Not captured in Milestone 4
            ref_logprob_sum=None,  # Not captured in Milestone 4
            metadata={},
        )
        samples.append(sample)

    # Determine missing fields for full verl compatibility
    missing_fields = []

    # Check if any sample is missing token counts
    if any(s.prompt_token_count is None for s in samples):
        missing_fields.append("prompt_token_count")
    if any(s.response_token_count is None for s in samples):
        missing_fields.append("response_token_count")

    # Old and ref logprobs are always missing in Milestone 4/5
    missing_fields.extend([
        "old_logprob_sum",
        "ref_logprob_sum",
        "token_level_logprobs",
        "attention_mask",
        "position_ids",
    ])

    return VerlLikeBatch(
        batch_id=batch.batch_id,
        actor_role=batch.role,
        round_id=batch.round_id,
        step_id=batch.step_id,
        samples=samples,
        tensorization_status="not_tensorized",
        missing_fields_for_verl=missing_fields,
        metadata={
            "source_batch_id": batch.batch_id,
            "num_total_items": len(batch.items),
            "num_selected_items": sum(1 for item in batch.items if item.selected_for_update),
            "include_unselected": include_unselected,
        },
    )


# ============================================================================
# Export Functions
# ============================================================================

def export_verl_like_batches(
    *,
    rollout_dir: Path,
    output_dir: Path,
    include_unselected: bool = False,
    tokenizer: Any | None = None,
    max_records: int | None = None,
) -> list[Path]:
    """
    Export all actor update batches from rollout directory as verl-like batches.

    Args:
        rollout_dir: Path to rollout directory
        output_dir: Path to output directory
        include_unselected: Whether to include unselected items
        tokenizer: Optional tokenizer for token counting

    Returns:
        List of exported batch file paths
    """
    output_dir = Path(output_dir)
    batches_dir = output_dir / "verl_like_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    exported_paths = []
    manifest_lines = []

    # Read rollout manifest
    manifest_path = rollout_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Rollout manifest not found: {manifest_path}")

    records_seen = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_records is not None and records_seen >= max_records:
                break
            if not line.strip():
                continue
            records_seen += 1
            record = json.loads(line.strip())
            record_id = record["record_id"]

            # Export memory batch
            update_batches_dir = rollout_dir / "update_batches"
            memory_batch_path = update_batches_dir / f"{record_id}.memory_update_batch.json"

            if memory_batch_path.exists():
                with open(memory_batch_path, "r", encoding="utf-8") as bf:
                    batch_data = json.load(bf)
                    batch = ActorUpdateBatch.model_validate(batch_data)

                verl_batch = actor_update_batch_to_verl_like_batch(
                    batch,
                    tokenizer=tokenizer,
                    include_unselected=include_unselected,
                )

                # Write verl-like batch
                output_path = batches_dir / f"{record_id}.memory.verl_like_batch.json"
                with open(output_path, "w", encoding="utf-8") as outf:
                    json.dump(verl_batch.model_dump(mode="json"), outf, ensure_ascii=False, indent=2)

                exported_paths.append(output_path)

                # Add to manifest
                manifest_lines.append({
                    "record_id": record_id,
                    "role": "memory_builder",
                    "batch_path": str(output_path.relative_to(output_dir)),
                    "num_samples": len(verl_batch.samples),
                    "missing_fields_for_verl": verl_batch.missing_fields_for_verl,
                })

            # Export question batch
            question_batch_path = update_batches_dir / f"{record_id}.question_update_batch.json"

            if question_batch_path.exists():
                with open(question_batch_path, "r", encoding="utf-8") as bf:
                    batch_data = json.load(bf)
                    batch = ActorUpdateBatch.model_validate(batch_data)

                verl_batch = actor_update_batch_to_verl_like_batch(
                    batch,
                    tokenizer=tokenizer,
                    include_unselected=include_unselected,
                )

                # Write verl-like batch
                output_path = batches_dir / f"{record_id}.question.verl_like_batch.json"
                with open(output_path, "w", encoding="utf-8") as outf:
                    json.dump(verl_batch.model_dump(mode="json"), outf, ensure_ascii=False, indent=2)

                exported_paths.append(output_path)

                # Add to manifest
                manifest_lines.append({
                    "record_id": record_id,
                    "role": "question_agent",
                    "batch_path": str(output_path.relative_to(output_dir)),
                    "num_samples": len(verl_batch.samples),
                    "missing_fields_for_verl": verl_batch.missing_fields_for_verl,
                })

    # Write manifest
    manifest_output_path = output_dir / "verl_like_batches" / "manifest.jsonl"
    with open(manifest_output_path, "w", encoding="utf-8") as f:
        for line_data in manifest_lines:
            f.write(json.dumps(line_data, ensure_ascii=False) + "\n")

    return exported_paths
