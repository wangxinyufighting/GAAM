"""
Phase 3 Milestone 4: verl Trainer Adapter

This module bridges GAAM's local DataProto artifacts (from Milestone 2/3)
to trainer-ready batches with logprobs, returns, and optional verl conversion.

Key features:
- Enriches LocalDataProtoBatch with trainer-required fields
- Computes or stubs old_log_probs, ref_log_probs, token_level_logprobs, returns
- Explicit field provenance tracking (real/stub/missing/derived)
- Optional vendored verl DataProto conversion
- Dry-run and local-model modes
- No-leakage validation
"""

import json
import logging
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Keep subprocess CLI tests and lightweight help paths from crashing in
# constrained local environments where OpenMP cannot allocate shared memory.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import torch
from pydantic import BaseModel, Field

from gaam_graph.code_a1_alignment import check_for_forbidden_fields
from gaam_graph.grpo_schema import ActorRole
from gaam_graph.local_dataproto_adapter import (
    LocalDataProtoBatch,
    LocalDataProtoTensorBundle,
    LocalTensorSpec,
)

logger = logging.getLogger(__name__)

# Real case ID for testing
REAL_CASE_ID = "e47becba"


# ============================================================================
# Trainer Field Provenance
# ============================================================================


class TrainerFieldSource(str, Enum):
    """Provenance of trainer batch fields."""

    REAL = "real"  # Computed from real model inference
    STUB = "stub"  # Placeholder zeros/ones for testing
    MISSING = "missing"  # Field not present
    DERIVED = "derived"  # Computed from other fields


class VerlTrainerAdapterMode(str, Enum):
    """Trainer adapter execution modes."""

    DRY_RUN = "dry_run"  # Validate structure, stub logprobs
    LOCAL_MODEL = "local_model"  # Compute real logprobs with local models
    VENDORED_VERL = "vendored_verl"  # Convert to vendored verl DataProto


# ============================================================================
# Trainer Batch Schemas
# ============================================================================


class TrainerBatchMetadata(BaseModel):
    """Metadata for trainer-ready batch."""

    batch_id: str
    actor_role: ActorRole
    record_id: str | None = None
    round_id: int
    step_id: int
    tokenizer_backend: str
    source_dataproto_path: str | None = None
    sequence_count: int
    max_sequence_length: int
    field_sources: dict[str, TrainerFieldSource]
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    returns_strategy: str = "advantages_as_returns"


class TrainerReadyBatch(BaseModel):
    """Schema for trainer-ready batch (JSON-safe)."""

    batch_id: str
    actor_role: ActorRole
    metadata: TrainerBatchMetadata
    tensor_specs: list[LocalTensorSpec]
    non_tensor_keys: list[str]


@dataclass
class TrainerTensorBundle:
    """Runtime tensor payload for trainer (not serialized to JSON)."""

    batch: dict[str, Any]  # Tensor fields
    non_tensor_batch: dict[str, list[Any]]  # Non-tensor metadata
    meta_info: dict[str, Any]  # Batch metadata


class VerlTrainerAdapterReport(BaseModel):
    """Report from trainer adapter execution."""

    report_id: str
    status: str  # succeeded, failed, partial, skipped
    actor_role: ActorRole | None = None
    mode: VerlTrainerAdapterMode
    input_dataproto_dir: str
    output_dir: str
    trainer_ready_batch_path: str | None = None
    trainer_tensor_path: str | None = None
    verl_conversion_status: str = "not_requested"
    checkpoint_registry_path: str | None = None
    checkpoint_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None


# ============================================================================
# Loading Functions
# ============================================================================


def load_local_dataproto_batch(path: Path) -> LocalDataProtoBatch:
    """Load LocalDataProtoBatch from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return LocalDataProtoBatch.model_validate(data)


def load_torch_tensor_payload(path: Path) -> dict[str, Any]:
    """Load PyTorch tensor payload from .pt file."""
    if not path.exists():
        raise FileNotFoundError(f"Tensor payload not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


# ============================================================================
# No-Leakage Validation
# ============================================================================


def validate_trainer_input_no_leakage(batch: LocalDataProtoBatch) -> None:
    """Validate trainer input contains no forbidden oracle/benchmark fields."""
    batch_dict = batch.model_dump(mode="json")
    forbidden_paths = check_for_forbidden_fields(batch_dict)

    if forbidden_paths:
        unique_paths = ", ".join(sorted(set(forbidden_paths)))
        raise ValueError(
            f"Trainer input contains forbidden oracle/benchmark fields: {unique_paths}"
        )


# ============================================================================
# Logprob and Return Computation
# ============================================================================


def add_stub_logprob_fields(bundle: TrainerTensorBundle) -> TrainerTensorBundle:
    """Add stub logprob fields (zeros) for dry-run mode."""
    batch = bundle.batch

    # Get shape from input_ids
    if "input_ids" not in batch:
        raise ValueError("Cannot stub logprobs without input_ids")

    input_ids = batch["input_ids"]
    batch_size, seq_len = input_ids.shape

    # Add only missing stub logprobs. Existing real fields should not be
    # overwritten by a dry-run enrichment pass.
    bundle.meta_info["logprob_mode"] = "stub"
    bundle.meta_info["field_sources"] = bundle.meta_info.get("field_sources", {})
    for field in ("old_log_probs", "ref_log_probs", "token_level_logprobs"):
        if field not in batch:
            batch[field] = torch.zeros(batch_size, seq_len, dtype=torch.float32)
            bundle.meta_info["field_sources"][field] = TrainerFieldSource.STUB.value

    return bundle


def _mark_existing_field_sources(
    bundle: TrainerTensorBundle, required_fields: list[str]
) -> None:
    """Mark existing tensor fields that were not explicitly attributed."""
    field_sources = bundle.meta_info.setdefault("field_sources", {})
    for field in required_fields:
        if field not in bundle.batch or field in field_sources:
            continue
        if field in {"old_log_probs", "ref_log_probs", "token_level_logprobs"}:
            field_sources[field] = TrainerFieldSource.REAL.value
        else:
            field_sources[field] = TrainerFieldSource.DERIVED.value


def compute_returns(
    bundle: TrainerTensorBundle, strategy: str = "advantages_as_returns"
) -> TrainerTensorBundle:
    """Compute returns from advantages/rewards."""
    batch = bundle.batch

    if strategy == "advantages_as_returns":
        # Simple: returns = advantages
        if "advantages" in batch:
            batch["returns"] = batch["advantages"].clone()
            bundle.meta_info["returns_strategy"] = "advantages_as_returns"
            bundle.meta_info["field_sources"] = bundle.meta_info.get(
                "field_sources", {}
            )
            bundle.meta_info["field_sources"]["returns"] = TrainerFieldSource.DERIVED.value
        else:
            # No advantages - use zeros
            if "input_ids" in batch:
                batch_size, seq_len = batch["input_ids"].shape
                batch["returns"] = torch.zeros(batch_size, seq_len, dtype=torch.float32)
                bundle.meta_info["returns_strategy"] = "zero_returns"
                if "field_sources" not in bundle.meta_info:
                    bundle.meta_info["field_sources"] = {}
                bundle.meta_info["field_sources"]["returns"] = TrainerFieldSource.STUB.value

    elif strategy == "rewards_plus_advantages":
        # returns = rewards + advantages
        if "token_level_rewards" in batch and "advantages" in batch:
            batch["returns"] = batch["token_level_rewards"] + batch["advantages"]
            bundle.meta_info["returns_strategy"] = "rewards_plus_advantages"
            bundle.meta_info["field_sources"] = bundle.meta_info.get(
                "field_sources", {}
            )
            bundle.meta_info["field_sources"]["returns"] = TrainerFieldSource.DERIVED.value
        else:
            # Fallback to advantages_as_returns
            return compute_returns(bundle, strategy="advantages_as_returns")

    else:
        raise ValueError(f"Unknown returns strategy: {strategy}")

    return bundle


# ============================================================================
# Trainer Batch Building
# ============================================================================


def build_trainer_ready_batch(
    local_batch: LocalDataProtoBatch,
    tensor_payload: dict[str, Any] | None = None,
    *,
    mode: VerlTrainerAdapterMode = VerlTrainerAdapterMode.DRY_RUN,
    allow_stub_logprobs: bool = True,
    returns_strategy: str = "advantages_as_returns",
) -> tuple[TrainerReadyBatch, TrainerTensorBundle]:
    """
    Build trainer-ready batch from local DataProto batch.

    Args:
        local_batch: LocalDataProtoBatch from Milestone 2
        tensor_payload: Optional preloaded tensor dict
        mode: Adapter mode (dry_run/local_model/vendored_verl)
        allow_stub_logprobs: Allow stub logprobs in dry_run mode
        returns_strategy: How to compute returns

    Returns:
        (TrainerReadyBatch schema, TrainerTensorBundle runtime payload)
    """
    # Validate no leakage
    validate_trainer_input_no_leakage(local_batch)

    # Build tensor bundle
    if tensor_payload is None:
        # Load from local_batch tensor_specs
        if not local_batch.tensor_specs:
            raise ValueError("No tensor specs in local batch")

        # For MVP, create empty bundle if no real tensors
        batch_dict: dict[str, Any] = {}
        non_tensor_dict: dict[str, list[Any]] = {}
        meta_info: dict[str, Any] = {}
    else:
        # Use provided tensor payload
        batch_dict = tensor_payload.get("batch", {})
        non_tensor_dict = tensor_payload.get("non_tensor_batch", {})
        meta_info = tensor_payload.get("meta_info", {})

    bundle = TrainerTensorBundle(
        batch=batch_dict, non_tensor_batch=non_tensor_dict, meta_info=meta_info
    )

    # Initialize field sources
    field_sources: dict[str, TrainerFieldSource] = {}
    warnings: list[str] = []

    # Check required trainer fields
    required_fields = [
        "input_ids",
        "attention_mask",
        "responses",
        "response_mask",
        "token_level_rewards",
        "advantages",
        "old_log_probs",
        "ref_log_probs",
        "token_level_logprobs",
        "returns",
    ]

    logprob_fields = ["old_log_probs", "ref_log_probs", "token_level_logprobs"]

    # Handle logprobs based on mode
    if mode == VerlTrainerAdapterMode.DRY_RUN:
        if not allow_stub_logprobs:
            raise ValueError(
                "Dry-run mode requires stub logprobs but allow_stub_logprobs=False"
            )

        # Add stub logprobs
        if any(field not in bundle.batch for field in logprob_fields):
            bundle = add_stub_logprob_fields(bundle)
            warnings.append("Added stub logprobs for dry-run mode")

    elif mode == VerlTrainerAdapterMode.LOCAL_MODEL:
        # Real logprobs would be computed here
        # For Milestone 4 MVP, we still stub them
        if not allow_stub_logprobs:
            raise ValueError(
                "Local model mode not fully implemented, requires allow_stub_logprobs=True"
            )
        bundle = add_stub_logprob_fields(bundle)
        warnings.append("Local model mode not fully implemented, using stub logprobs")

    elif mode == VerlTrainerAdapterMode.VENDORED_VERL:
        if any(field not in bundle.batch for field in logprob_fields):
            if not allow_stub_logprobs:
                raise ValueError(
                    "vendored_verl mode requires logprobs but allow_stub_logprobs=False"
                )
            bundle = add_stub_logprob_fields(bundle)
            warnings.append("Added stub logprobs before vendored verl conversion")

    # Compute returns
    if "returns" not in bundle.batch:
        bundle = compute_returns(bundle, strategy=returns_strategy)

    # Build metadata
    _mark_existing_field_sources(bundle, required_fields)
    missing_fields = [field for field in required_fields if field not in bundle.batch]
    field_sources.update(bundle.meta_info.get("field_sources") or {})

    # Get tokenizer backend from meta_info or default
    tokenizer_backend = local_batch.meta_info.get("tokenizer_backend", "unknown")
    source_dataproto_path = local_batch.meta_info.get("source_rollout_dir", None)

    sequence_count = len(local_batch.samples)
    max_sequence_length = 0
    if "input_ids" in bundle.batch:
        input_ids = bundle.batch["input_ids"]
        sequence_count = input_ids.shape[0] if len(input_ids.shape) >= 2 else len(local_batch.samples)
        max_sequence_length = input_ids.shape[1] if len(input_ids.shape) >= 2 else 0

    metadata = TrainerBatchMetadata(
        batch_id=local_batch.batch_id,
        actor_role=local_batch.actor_role,
        record_id=local_batch.record_id,
        round_id=local_batch.round_id,
        step_id=local_batch.step_id,
        tokenizer_backend=tokenizer_backend,
        source_dataproto_path=str(source_dataproto_path) if source_dataproto_path else None,
        sequence_count=sequence_count,
        max_sequence_length=max_sequence_length,
        field_sources=field_sources,
        missing_fields=missing_fields,
        warnings=warnings,
        returns_strategy=returns_strategy,
    )

    # Build trainer ready batch schema
    trainer_batch = TrainerReadyBatch(
        batch_id=local_batch.batch_id,
        actor_role=local_batch.actor_role,
        metadata=metadata,
        tensor_specs=local_batch.tensor_specs,
        non_tensor_keys=local_batch.non_tensor_keys,
    )

    return trainer_batch, bundle


# ============================================================================
# vendored verl Conversion
# ============================================================================


def trainer_bundle_to_verl_dataproto(bundle: TrainerTensorBundle) -> Any:
    """
    Convert TrainerTensorBundle to vendored verl DataProto.

    This function attempts to import and use the vendored verl from Code-A1.
    Falls back with clear error if unavailable.
    """
    # Try to import vendored verl
    try:
        from verl import DataProto
    except ImportError:
        try:
            from verl.verl.protocol import DataProto
        except ImportError as e:
            raise RuntimeError(
                "vendored verl is not importable. "
                "Add Code-A1/Code-A1/verl to PYTHONPATH or skip verl conversion."
            ) from e

    # Convert to DataProto
    # The DataProto structure from verl typically has:
    # - batch: dict of torch tensors
    # - non_tensor_batch: dict of lists/arrays
    # - meta_info: dict of metadata

    dataproto = DataProto(
        batch=bundle.batch,
        non_tensor_batch=bundle.non_tensor_batch,
        meta_info=bundle.meta_info,
    )

    return dataproto


def _write_report(report: VerlTrainerAdapterReport, output_dir: Path) -> None:
    """Write a trainer step report for every terminal status."""
    report_path = output_dir / "trainer_step_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2)


# ============================================================================
# Main Adapter Function
# ============================================================================


def run_verl_trainer_adapter_step(
    *,
    dataproto_dir: Path,
    output_dir: Path,
    actor_role: ActorRole | str,
    mode: VerlTrainerAdapterMode = VerlTrainerAdapterMode.DRY_RUN,
    checkpoint_registry_path: Path | None = None,
    allow_stub_logprobs: bool = True,
    returns_strategy: str = "advantages_as_returns",
    write_verl_dataproto: bool = False,
) -> VerlTrainerAdapterReport:
    """
    Run trainer adapter step for one actor role.

    Args:
        dataproto_dir: Input directory with LocalDataProto artifacts
        output_dir: Output directory for trainer artifacts
        actor_role: Actor role to process
        mode: Adapter mode
        checkpoint_registry_path: Optional checkpoint registry path
        allow_stub_logprobs: Allow stub logprobs
        returns_strategy: Returns computation strategy
        write_verl_dataproto: Write vendored verl DataProto

    Returns:
        VerlTrainerAdapterReport
    """
    started_at = datetime.utcnow().isoformat()
    report_id = f"trainer_adapter_{actor_role}_{started_at}"

    if isinstance(actor_role, str):
        actor_role = ActorRole(actor_role)

    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    errors: list[str] = []
    metrics: dict[str, Any] = {}

    try:
        # Find local dataproto batch for this actor
        batch_files = list(dataproto_dir.glob(f"*.{actor_role.value}.local_dataproto.json"))

        if not batch_files:
            # No batch found - return skipped
            finished_at = datetime.utcnow().isoformat()
            report = VerlTrainerAdapterReport(
                report_id=report_id,
                status="skipped",
                actor_role=actor_role,
                mode=mode,
                input_dataproto_dir=str(dataproto_dir),
                output_dir=str(output_dir),
                warnings=[f"No {actor_role.value} batch found in {dataproto_dir}"],
                started_at=started_at,
                finished_at=finished_at,
            )
            _write_report(report, output_dir)
            return report

        # Load first batch
        batch_file = batch_files[0]
        local_batch = load_local_dataproto_batch(batch_file)

        # Check for empty batch
        if len(local_batch.samples) == 0:
            warnings.append(f"Batch has zero samples: {batch_file.name}")
            finished_at = datetime.utcnow().isoformat()
            report = VerlTrainerAdapterReport(
                report_id=report_id,
                status="partial",
                actor_role=actor_role,
                mode=mode,
                input_dataproto_dir=str(dataproto_dir),
                output_dir=str(output_dir),
                warnings=warnings,
                metrics={"num_sequences": 0},
                started_at=started_at,
                finished_at=finished_at,
            )
            _write_report(report, output_dir)
            return report

        # Try to load tensor payload if available
        tensor_file = batch_file.with_suffix(".pt")
        tensor_payload = None
        if tensor_file.exists():
            try:
                tensor_payload = load_torch_tensor_payload(tensor_file)
            except Exception as e:
                warnings.append(f"Failed to load tensor payload: {e}")

        # Build trainer ready batch
        trainer_batch, trainer_bundle = build_trainer_ready_batch(
            local_batch=local_batch,
            tensor_payload=tensor_payload,
            mode=mode,
            allow_stub_logprobs=allow_stub_logprobs,
            returns_strategy=returns_strategy,
        )

        # Write trainer ready batch (JSON)
        trainer_batch_path = output_dir / "trainer_ready_batch.json"
        with open(trainer_batch_path, "w", encoding="utf-8") as f:
            json.dump(trainer_batch.model_dump(mode="json"), f, indent=2)

        # Write trainer tensor payload (.pt)
        trainer_tensor_path = output_dir / "trainer_tensor_payload.pt"
        torch.save(
            {
                "batch": trainer_bundle.batch,
                "non_tensor_batch": trainer_bundle.non_tensor_batch,
                "meta_info": trainer_bundle.meta_info,
            },
            trainer_tensor_path,
        )

        # Optional: convert to verl DataProto
        verl_conversion_status = "not_requested"
        if write_verl_dataproto:
            try:
                verl_dataproto = trainer_bundle_to_verl_dataproto(trainer_bundle)
                verl_path = output_dir / "verl_dataproto.pt"
                torch.save(verl_dataproto, verl_path)
                verl_conversion_status = "succeeded"
            except Exception as e:
                verl_conversion_status = "failed"
                warnings.append(f"verl conversion failed: {type(e).__name__}: {e}")

        # Collect metrics
        metrics = {
            "num_sequences": trainer_batch.metadata.sequence_count,
            "max_sequence_length": trainer_batch.metadata.max_sequence_length,
            "returns_strategy": returns_strategy,
            "stub_logprobs": any(
                v == TrainerFieldSource.STUB
                for v in trainer_batch.metadata.field_sources.values()
            ),
        }

        # Write trainer step report
        finished_at = datetime.utcnow().isoformat()
        report = VerlTrainerAdapterReport(
            report_id=report_id,
            status="succeeded",
            actor_role=actor_role,
            mode=mode,
            input_dataproto_dir=str(dataproto_dir),
            output_dir=str(output_dir),
            trainer_ready_batch_path=str(trainer_batch_path),
            trainer_tensor_path=str(trainer_tensor_path),
            verl_conversion_status=verl_conversion_status,
            checkpoint_registry_path=str(checkpoint_registry_path)
            if checkpoint_registry_path
            else None,
            checkpoint_id=None,  # Would be set by real trainer
            metrics=metrics,
            warnings=warnings + trainer_batch.metadata.warnings,
            errors=errors,
            started_at=started_at,
            finished_at=finished_at,
        )

        # Write report
        _write_report(report, output_dir)

        return report

    except Exception as e:
        finished_at = datetime.utcnow().isoformat()
        error_msg = f"{type(e).__name__}: {e}"
        error_trace = traceback.format_exc()

        report = VerlTrainerAdapterReport(
            report_id=report_id,
            status="failed",
            actor_role=actor_role,
            mode=mode,
            input_dataproto_dir=str(dataproto_dir),
            output_dir=str(output_dir),
            warnings=warnings,
            errors=[error_msg],
            metrics={"error_trace": error_trace},
            started_at=started_at,
            finished_at=finished_at,
        )
        _write_report(report, output_dir)
        return report
