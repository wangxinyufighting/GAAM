"""
Policy client interfaces and stub implementations for Phase 2 Milestone 3.

This module defines local policy-client abstractions and deterministic stub implementations.
Milestone 3 needs policy clients because the trainer must not know whether the actor is:
- a deterministic baseline
- a local model
- a future LoRA adapter
- a future Code-A1/verl actor worker

Design principles:
1. Policy clients are actor-specific
2. Update APIs consume sanitized ActorUpdateBatch, not full rollout trace
3. Stub clients are deterministic and JSON-safe
4. No torch/transformers dependency in Milestone 3
5. Checkpoint writes are metadata-only in this skeleton
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

from gaam_graph.grpo_schema import ActorRole, ActorUpdateBatch


# ============================================================================
# Policy Update Result
# ============================================================================

class PolicyUpdateResult(BaseModel):
    """Result of a policy update call."""
    role: ActorRole
    policy_id: str
    batch_id: str
    round_id: int
    step_id: int
    updated: bool
    num_items: int
    num_selected_items: int
    mean_reward: float | None = None
    mean_advantage: float | None = None
    loss: float | None = None
    grad_norm: float | None = None
    checkpoint_path: str | None = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# ============================================================================
# Policy Checkpoint Metadata
# ============================================================================

class PolicyCheckpointMetadata(BaseModel):
    """Checkpoint metadata for Milestone 3 (no model weights)."""
    role: ActorRole
    policy_id: str
    round_id: int
    step_id: int
    checkpoint_path: str
    source_batch_id: str | None = None
    updated: bool = False
    backend: str = "stub"
    metrics: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Base Policy Client
# ============================================================================

class BasePolicyClient:
    """Abstract base for policy clients."""

    def __init__(self, *, role: ActorRole, policy_id: str | None = None) -> None:
        self.role = role
        self.policy_id = policy_id or f"policy_{role.value}"

    def update_grpo(
        self,
        batch: ActorUpdateBatch,
        *,
        dry_run: bool = False,
    ) -> PolicyUpdateResult:
        """
        Update policy using GRPO batch.

        Args:
            batch: Sanitized actor update batch
            dry_run: If True, validate but don't update weights

        Returns:
            PolicyUpdateResult with metrics
        """
        raise NotImplementedError

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        round_id: int,
        step_id: int,
        source_batch_id: str | None = None,
        updated: bool = False,
        metrics: Dict[str, Any] | None = None,
    ) -> PolicyCheckpointMetadata:
        """
        Save checkpoint metadata.

        Args:
            checkpoint_dir: Directory to write checkpoint
            round_id: Training round ID
            step_id: Training step ID
            source_batch_id: Source batch ID if available
            updated: Whether this checkpoint reflects a real update
            metrics: Optional metrics to save

        Returns:
            PolicyCheckpointMetadata
        """
        raise NotImplementedError


# ============================================================================
# Stub Policy Client
# ============================================================================

class StubPolicyClient(BasePolicyClient):
    """
    Deterministic stub policy client for Milestone 3.

    Behavior:
    - Validates batch.role == self.role
    - Filters selected items
    - Computes mean reward and mean advantage
    - Returns updated=not dry_run and selected_count > 0
    - Never mutates the batch
    - Writes checkpoint metadata JSON when asked
    """

    def update_grpo(
        self,
        batch: ActorUpdateBatch,
        *,
        dry_run: bool = False,
    ) -> PolicyUpdateResult:
        """Update policy using GRPO batch (stub implementation)."""

        # Validate role
        if batch.role != self.role:
            raise ValueError(
                f"Batch role mismatch: expected {self.role.value}, got {batch.role.value}"
            )

        # Filter selected items
        selected_items = [item for item in batch.items if item.selected_for_update]

        # Compute mean reward and advantage
        mean_reward = None
        mean_advantage = None

        if batch.items:
            rewards = [item.reward for item in batch.items]
            mean_reward = sum(rewards) / len(rewards)

            advantages = [item.advantage for item in batch.items]
            mean_advantage = sum(advantages) / len(advantages)

        # Determine if updated
        updated = not dry_run and len(selected_items) > 0

        return PolicyUpdateResult(
            role=self.role,
            policy_id=self.policy_id,
            batch_id=batch.batch_id,
            round_id=batch.round_id,
            step_id=batch.step_id,
            updated=updated,
            num_items=len(batch.items),
            num_selected_items=len(selected_items),
            mean_reward=mean_reward,
            mean_advantage=mean_advantage,
            loss=None,  # Stub has no loss
            grad_norm=None,  # Stub has no gradients
            checkpoint_path=None,
            metrics={
                "backend": "stub",
                "dry_run": dry_run,
            },
        )

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        round_id: int,
        step_id: int,
        source_batch_id: str | None = None,
        updated: bool = False,
        metrics: Dict[str, Any] | None = None,
    ) -> PolicyCheckpointMetadata:
        """Save checkpoint metadata (stub implementation)."""

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / "checkpoint_metadata.json"

        metadata = PolicyCheckpointMetadata(
            role=self.role,
            policy_id=self.policy_id,
            round_id=round_id,
            step_id=step_id,
            checkpoint_path=str(checkpoint_path),
            source_batch_id=source_batch_id,
            updated=updated,
            backend="stub",
            metrics=metrics or {},
        )

        # Write metadata JSON
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        return metadata


# ============================================================================
# Actor-Specific Aliases
# ============================================================================

class StubMemoryBuilderPolicy(StubPolicyClient):
    """Stub policy for Memory Builder."""

    def __init__(self, *, policy_id: str | None = None) -> None:
        super().__init__(
            role=ActorRole.MEMORY_BUILDER,
            policy_id=policy_id or "stub_memory_builder",
        )


class StubQuestionAgentPolicy(StubPolicyClient):
    """Stub policy for Question Agent."""

    def __init__(self, *, policy_id: str | None = None) -> None:
        super().__init__(
            role=ActorRole.QUESTION_AGENT,
            policy_id=policy_id or "stub_question_agent",
        )
