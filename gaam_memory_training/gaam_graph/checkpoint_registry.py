"""
Checkpoint registry for Phase 2 Milestone 5.

This module makes checkpoint state explicit and resume-friendly for distributed training.

Design principles:
1. Discover and catalog all checkpoints from trainer directory
2. Track which actor, round, step, and source batch produced each checkpoint
3. Support both stub and local_hf backend checkpoints
4. Validate checkpoint integrity (model/tokenizer/optimizer paths)
5. Enable Phase 3 distributed checkpoint orchestration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import hashlib

from pydantic import BaseModel, Field

from gaam_graph.grpo_schema import ActorRole
from gaam_graph.code_a1_alignment import AlignmentIssue, AlignmentIssueSeverity


# ============================================================================
# Checkpoint Registry Models
# ============================================================================

class RegisteredCheckpoint(BaseModel):
    """A registered checkpoint with full metadata."""
    checkpoint_id: str
    actor_role: ActorRole
    policy_id: str
    backend: str
    round_id: int
    step_id: int
    source_batch_id: str | None = None
    checkpoint_path: str
    model_path: str | None = None
    tokenizer_path: str | None = None
    optimizer_path: str | None = None
    train_lora: bool = False
    updated: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)


class CheckpointRegistry(BaseModel):
    """Registry of all checkpoints from a trainer directory."""
    registry_id: str
    trainer_dir: str
    checkpoints: list[RegisteredCheckpoint]
    latest_by_actor: dict[str, str] = Field(default_factory=dict)
    issues: list[AlignmentIssue] = Field(default_factory=list)


# ============================================================================
# Checkpoint ID Generation
# ============================================================================

def generate_checkpoint_id(
    actor_role: ActorRole,
    policy_id: str,
    round_id: int,
    step_id: int,
    source_batch_id: str | None,
) -> str:
    """
    Generate stable checkpoint ID.

    Format: {actor_role}:{policy_id}:round{round_id}:step{step_id}:{batch_hash}

    Args:
        actor_role: Actor role
        policy_id: Policy ID
        round_id: Round ID
        step_id: Step ID
        source_batch_id: Source batch ID (optional)

    Returns:
        Stable checkpoint ID
    """
    # Hash source batch ID if provided
    if source_batch_id:
        batch_hash = hashlib.sha256(source_batch_id.encode()).hexdigest()[:8]
    else:
        batch_hash = "none"

    return f"{actor_role.value}:{policy_id}:round{round_id:03d}:step{step_id:03d}:{batch_hash}"


# ============================================================================
# Checkpoint Discovery
# ============================================================================

def discover_checkpoints(trainer_dir: Path) -> list[RegisteredCheckpoint]:
    """
    Discover all checkpoints in trainer directory.

    Args:
        trainer_dir: Path to trainer directory

    Returns:
        List of registered checkpoints
    """
    trainer_dir = Path(trainer_dir)
    checkpoints_dir = trainer_dir / "checkpoints"

    if not checkpoints_dir.exists():
        return []

    discovered = []

    # Scan actor directories
    for actor_dir in checkpoints_dir.iterdir():
        if not actor_dir.is_dir():
            continue

        actor_name = actor_dir.name

        # Map actor name to role
        if actor_name == "memory_builder":
            actor_role = ActorRole.MEMORY_BUILDER
        elif actor_name == "question_agent":
            actor_role = ActorRole.QUESTION_AGENT
        else:
            continue

        # Scan round directories
        for round_dir in actor_dir.iterdir():
            if not round_dir.is_dir():
                continue

            # Parse round_id and step_id from directory name
            # Format: round_XXX_step_YYY
            try:
                parts = round_dir.name.split("_")
                if len(parts) != 4:
                    continue
                round_id = int(parts[1])
                step_id = int(parts[3])
            except (ValueError, IndexError):
                continue

            # Scan batch directories
            for batch_dir in round_dir.iterdir():
                if not batch_dir.is_dir():
                    continue

                source_batch_id_from_dir = batch_dir.name

                # Check for checkpoint metadata
                metadata_path = batch_dir / "checkpoint_metadata.json"
                if not metadata_path.exists():
                    continue

                # Load metadata
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # Extract fields
                policy_id = metadata.get("policy_id", "unknown")
                backend = metadata.get("backend", "unknown")
                updated = metadata.get("updated", False)
                metrics = metadata.get("metrics", {})
                train_lora = metrics.get("train_lora", False)
                source_batch_id = metadata.get("source_batch_id") or source_batch_id_from_dir

                # Determine model/tokenizer/optimizer paths
                model_path = None
                tokenizer_path = None
                optimizer_path = None

                if backend == "local_hf":
                    model_subdir = metrics.get("model_subdir", "adapter" if train_lora else "model")
                    tokenizer_subdir = metrics.get("tokenizer_subdir", "tokenizer")
                    optimizer_file = metrics.get("optimizer_path", "optimizer.pt")

                    model_candidate = batch_dir / model_subdir
                    if model_candidate.exists():
                        model_path = str(model_candidate)

                    tokenizer_candidate = batch_dir / tokenizer_subdir
                    if tokenizer_candidate.exists():
                        tokenizer_path = str(tokenizer_candidate)

                    optimizer_candidate = batch_dir / optimizer_file
                    if optimizer_candidate.exists():
                        optimizer_path = str(optimizer_candidate)

                # Generate checkpoint ID
                checkpoint_id = generate_checkpoint_id(
                    actor_role=actor_role,
                    policy_id=policy_id,
                    round_id=round_id,
                    step_id=step_id,
                    source_batch_id=source_batch_id,
                )

                checkpoint = RegisteredCheckpoint(
                    checkpoint_id=checkpoint_id,
                    actor_role=actor_role,
                    policy_id=policy_id,
                    backend=backend,
                    round_id=round_id,
                    step_id=step_id,
                    source_batch_id=source_batch_id,
                    checkpoint_path=str(batch_dir),
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    optimizer_path=optimizer_path,
                    train_lora=train_lora,
                    updated=updated,
                    metrics=metrics,
                )

                discovered.append(checkpoint)

    return discovered


# ============================================================================
# Checkpoint Validation
# ============================================================================

def validate_checkpoints(checkpoints: list[RegisteredCheckpoint]) -> list[AlignmentIssue]:
    """
    Validate checkpoint integrity and consistency.

    Args:
        checkpoints: List of checkpoints to validate

    Returns:
        List of validation issues
    """
    issues = []

    # Check for duplicate checkpoint IDs
    seen_ids = set()
    for checkpoint in checkpoints:
        if checkpoint.checkpoint_id in seen_ids:
            issues.append(AlignmentIssue(
                severity=AlignmentIssueSeverity.ERROR,
                code="duplicate_checkpoint_id",
                message=f"Duplicate checkpoint ID: {checkpoint.checkpoint_id}",
                artifact_path=checkpoint.checkpoint_path,
                role=checkpoint.actor_role,
            ))
        seen_ids.add(checkpoint.checkpoint_id)

    # Validate individual checkpoints
    for checkpoint in checkpoints:
        # Check checkpoint path exists
        if not Path(checkpoint.checkpoint_path).exists():
            issues.append(AlignmentIssue(
                severity=AlignmentIssueSeverity.ERROR,
                code="checkpoint_path_not_found",
                message=f"Checkpoint path does not exist: {checkpoint.checkpoint_path}",
                artifact_path=checkpoint.checkpoint_path,
                role=checkpoint.actor_role,
            ))

        # Check local_hf backend requirements
        if checkpoint.backend == "local_hf":
            # Check required artifacts for updated checkpoints
            if checkpoint.updated and not checkpoint.model_path:
                issues.append(AlignmentIssue(
                    severity=AlignmentIssueSeverity.ERROR,
                    code="missing_model_for_updated_checkpoint",
                    message=f"Updated local_hf checkpoint missing model directory",
                    artifact_path=checkpoint.checkpoint_path,
                    role=checkpoint.actor_role,
                ))

            if checkpoint.updated and not checkpoint.tokenizer_path:
                issues.append(AlignmentIssue(
                    severity=AlignmentIssueSeverity.ERROR,
                    code="missing_tokenizer_for_updated_checkpoint",
                    message=f"Updated local_hf checkpoint missing tokenizer directory",
                    artifact_path=checkpoint.checkpoint_path,
                    role=checkpoint.actor_role,
                ))

            if checkpoint.updated and not checkpoint.optimizer_path:
                issues.append(AlignmentIssue(
                    severity=AlignmentIssueSeverity.ERROR,
                    code="missing_optimizer_for_updated_checkpoint",
                    message=f"Updated local_hf checkpoint missing optimizer state",
                    artifact_path=checkpoint.checkpoint_path,
                    role=checkpoint.actor_role,
                ))

            # Check model directory exists
            if checkpoint.model_path and not Path(checkpoint.model_path).exists():
                issues.append(AlignmentIssue(
                    severity=AlignmentIssueSeverity.ERROR,
                    code="model_path_not_found",
                    message=f"Model path does not exist: {checkpoint.model_path}",
                    artifact_path=checkpoint.model_path,
                    role=checkpoint.actor_role,
                ))

            # Check tokenizer directory exists
            if checkpoint.tokenizer_path and not Path(checkpoint.tokenizer_path).exists():
                issues.append(AlignmentIssue(
                    severity=AlignmentIssueSeverity.WARNING,
                    code="tokenizer_path_not_found",
                    message=f"Tokenizer path does not exist: {checkpoint.tokenizer_path}",
                    artifact_path=checkpoint.tokenizer_path,
                    role=checkpoint.actor_role,
                ))

            # Check optimizer file exists
            if checkpoint.optimizer_path and not Path(checkpoint.optimizer_path).exists():
                issues.append(AlignmentIssue(
                    severity=AlignmentIssueSeverity.WARNING,
                    code="optimizer_path_not_found",
                    message=f"Optimizer path does not exist: {checkpoint.optimizer_path}",
                    artifact_path=checkpoint.optimizer_path,
                    role=checkpoint.actor_role,
                ))

    return issues


# ============================================================================
# Registry Builder
# ============================================================================

def build_checkpoint_registry(trainer_dir: Path) -> CheckpointRegistry:
    """
    Build checkpoint registry from trainer directory.

    Args:
        trainer_dir: Path to trainer directory

    Returns:
        Checkpoint registry with validation issues
    """
    trainer_dir = Path(trainer_dir)

    # Discover checkpoints
    checkpoints = discover_checkpoints(trainer_dir)

    # Validate checkpoints
    issues = validate_checkpoints(checkpoints)

    # Find latest checkpoint for each actor
    latest_by_actor = {}

    for actor_role in [ActorRole.MEMORY_BUILDER, ActorRole.QUESTION_AGENT]:
        actor_checkpoints = [
            cp for cp in checkpoints
            if cp.actor_role == actor_role
        ]

        if actor_checkpoints:
            # Use a deterministic tie-breaker for multiple checkpoints in the same step.
            actor_checkpoints.sort(
                key=lambda cp: (
                    cp.round_id,
                    cp.step_id,
                    cp.updated,
                    cp.source_batch_id or "",
                    cp.checkpoint_path,
                    cp.checkpoint_id,
                ),
                reverse=True,
            )
            latest = actor_checkpoints[0]
            latest_by_actor[actor_role.value] = latest.checkpoint_id

    # Generate registry ID from trainer dir name
    registry_id = hashlib.sha256(str(trainer_dir).encode()).hexdigest()[:8]

    return CheckpointRegistry(
        registry_id=registry_id,
        trainer_dir=str(trainer_dir),
        checkpoints=checkpoints,
        latest_by_actor=latest_by_actor,
        issues=issues,
    )


# ============================================================================
# Registry Writer
# ============================================================================

def write_checkpoint_registry(
    trainer_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Write checkpoint registry to JSON file.

    Args:
        trainer_dir: Path to trainer directory
        output_path: Optional output path (defaults to trainer_dir/checkpoint_registry.json)

    Returns:
        Path to written registry file
    """
    trainer_dir = Path(trainer_dir)

    # Build registry
    registry = build_checkpoint_registry(trainer_dir)

    # Determine output path
    if output_path is None:
        output_path = trainer_dir / "checkpoint_registry.json"
    else:
        output_path = Path(output_path)

    # Write registry
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    return output_path
