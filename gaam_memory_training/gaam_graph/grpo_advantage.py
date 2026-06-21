"""
GRPO Advantage Computation Utilities.

Pure Python functions for group-relative advantage computation.
No model loading, no Ray, no verl, no torch dependencies.
"""

from __future__ import annotations

import math

from gaam_graph.grpo_schema import (
    GRPOAdvantageItem,
    GRPORewardItem,
    GroupedRollout,
    GroupStatus,
    ActorUpdateBatch,
    ActorUpdateItem,
)


def _validate_finite_rewards(rewards: list[float]) -> None:
    for reward in rewards:
        if not math.isfinite(reward):
            raise ValueError(f"All rewards must be finite, got {reward}")


def _reward_stats(rewards: list[float]) -> tuple[float, float]:
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    return mean, math.sqrt(variance)


def compute_grpo_advantages(
    rewards: list[float],
    *,
    normalize_by_std: bool = True,
    eps: float = 1e-6,
) -> list[float]:
    """
    Compute group-relative advantages from rewards.

    Args:
        rewards: List of reward values
        normalize_by_std: If True, divide by population std
        eps: Small constant to avoid division by zero

    Returns:
        List of advantages (same length as rewards)

    Rules:
    - Empty input returns []
    - One reward returns [0.0]
    - Mean is arithmetic mean
    - Std is population std (not sample std)
    - If normalize_by_std=True and std <= eps, return all zeros
    - If normalize_by_std=False, return reward minus mean
    - All inputs must be finite numbers
    """
    if not rewards:
        return []

    _validate_finite_rewards(rewards)

    if len(rewards) == 1:
        return [0.0]

    mean, std = _reward_stats(rewards)
    if normalize_by_std:
        if std <= eps:
            return [0.0] * len(rewards)
        return [(reward - mean) / std for reward in rewards]
    return [reward - mean for reward in rewards]


def summarize_reward_group(
    rewards: list[float],
    *,
    eps: float = 1e-6,
) -> dict:
    """
    Summarize statistics for a reward group.

    Args:
        rewards: List of reward values
        eps: Small constant for zero-variance detection

    Returns:
        Dictionary with count, mean, std, min, max, is_zero_variance
    """
    if not rewards:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "is_zero_variance": None,
        }

    _validate_finite_rewards(rewards)
    mean, std = _reward_stats(rewards)

    return {
        "count": len(rewards),
        "mean": mean,
        "std": std,
        "min": min(rewards),
        "max": max(rewards),
        "is_zero_variance": std <= eps,
    }


def compute_advantage_items(
    rewards: list[GRPORewardItem],
    *,
    normalize_by_std: bool = True,
    skip_zero_variance: bool = True,
    eps: float = 1e-6,
) -> list[GRPOAdvantageItem]:
    """
    Compute advantage items from reward items.

    Args:
        rewards: List of GRPORewardItem
        normalize_by_std: If True, divide by population std
        skip_zero_variance: If True, set selected_for_update=False for zero-variance groups
        eps: Small constant for zero-variance detection

    Returns:
        List of GRPOAdvantageItem in same order as input rewards

    Rules:
    - All rewards must share the same group_id, record_id, and role
    - Compute advantages from reward values
    - If skip_zero_variance=True, set selected_for_update=False for zero-variance groups
    - Rank items by reward descending; highest reward gets rank_in_group=1
    - Preserve input order in returned list
    """
    if not rewards:
        return []

    # Validate consistency
    first = rewards[0]
    for r in rewards:
        if r.group_id != first.group_id:
            raise ValueError(
                f"All rewards must have same group_id, got {r.group_id} and {first.group_id}"
            )
        if r.record_id != first.record_id:
            raise ValueError(
                f"All rewards must have same record_id, got {r.record_id} and {first.record_id}"
            )
        if r.role != first.role:
            raise ValueError(
                f"All rewards must have same role, got {r.role} and {first.role}"
            )

    # Extract reward values
    reward_values = [r.reward for r in rewards]

    # Compute advantages
    advantages = compute_grpo_advantages(
        reward_values,
        normalize_by_std=normalize_by_std,
        eps=eps,
    )

    # Compute group statistics
    mean, std = _reward_stats(reward_values)

    # Determine if zero variance
    is_zero_variance = std <= eps

    # Compute ranks (highest reward = rank 1)
    # Create list of (index, reward_value) and sort by reward descending
    indexed_rewards = [(i, r.reward) for i, r in enumerate(rewards)]
    sorted_by_reward = sorted(indexed_rewards, key=lambda x: x[1], reverse=True)

    # Assign ranks
    rank_map = {}
    for rank, (idx, _) in enumerate(sorted_by_reward, start=1):
        rank_map[idx] = rank

    # Build advantage items
    advantage_items = []
    for i, reward in enumerate(rewards):
        selected = True
        if skip_zero_variance and is_zero_variance:
            selected = False

        advantage_items.append(
            GRPOAdvantageItem(
                sample_id=reward.sample_id,
                group_id=reward.group_id,
                record_id=reward.record_id,
                role=reward.role,
                reward=reward.reward,
                group_mean=mean,
                group_std=std,
                advantage=advantages[i],
                normalized=normalize_by_std,
                selected_for_update=selected,
                rank_in_group=rank_map[i],
            )
        )

    return advantage_items


def attach_advantages_to_group(
    group: GroupedRollout,
    *,
    normalize_by_std: bool = True,
    skip_zero_variance: bool = True,
    eps: float = 1e-6,
) -> GroupedRollout:
    """
    Attach advantages to a grouped rollout.

    Args:
        group: GroupedRollout with rewards populated
        normalize_by_std: If True, divide by population std
        skip_zero_variance: If True, set selected_for_update=False for zero-variance groups
        eps: Small constant for zero-variance detection

    Returns:
        New GroupedRollout with advantages populated and status updated

    Rules:
    - Do not mutate the input group
    - Return a copy with advantages populated
    - Set group status:
        - TOO_FEW_SAMPLES if fewer than 2 rewards
        - ZERO_VARIANCE if reward std <= eps
        - OK otherwise
    - If no rewards, status should be ALL_FAILED
    """
    # Handle no rewards case
    if not group.rewards:
        return GroupedRollout(
            group_id=group.group_id,
            record_id=group.record_id,
            role=group.role,
            status=GroupStatus.ALL_FAILED,
            samples=group.samples,
            rewards=group.rewards,
            advantages=[],
            metadata=group.metadata,
        )

    # Handle too few samples
    if len(group.rewards) < 2:
        return GroupedRollout(
            group_id=group.group_id,
            record_id=group.record_id,
            role=group.role,
            status=GroupStatus.TOO_FEW_SAMPLES,
            samples=group.samples,
            rewards=group.rewards,
            advantages=[],
            metadata=group.metadata,
        )

    # Compute advantages
    advantages = compute_advantage_items(
        group.rewards,
        normalize_by_std=normalize_by_std,
        skip_zero_variance=skip_zero_variance,
        eps=eps,
    )

    # Determine status
    reward_values = [r.reward for r in group.rewards]
    _, std = _reward_stats(reward_values)

    if std <= eps:
        status = GroupStatus.ZERO_VARIANCE
    else:
        status = GroupStatus.OK

    # Return new group with advantages
    return GroupedRollout(
        group_id=group.group_id,
        record_id=group.record_id,
        role=group.role,
        status=status,
        samples=group.samples,
        rewards=group.rewards,
        advantages=advantages,
        metadata=group.metadata,
    )


def build_actor_update_batch(
    group: GroupedRollout,
    *,
    batch_id: str,
    round_id: int,
    step_id: int,
    selected_only: bool = True,
) -> ActorUpdateBatch:
    """
    Build an actor update batch from a grouped rollout.

    Args:
        group: GroupedRollout with advantages populated
        batch_id: Unique batch identifier
        round_id: Training round ID
        step_id: Training step ID
        selected_only: If True, include only samples with selected_for_update=True

    Returns:
        ActorUpdateBatch with sanitized update items

    Rules:
    - Requires group.advantages
    - Match each advantage to its PolicySample
    - Include only selected items if selected_only=True
    - Use sample prompt and response
    - Do not include full reward report or oracle artifacts
    """
    if not group.advantages:
        raise ValueError("Group must have advantages populated")

    # Build sample lookup
    sample_map = {s.sample_id: s for s in group.samples}

    # Build update items
    items = []
    for adv in group.advantages:
        # Skip if not selected and selected_only=True
        if selected_only and not adv.selected_for_update:
            continue

        # Find corresponding sample
        sample = sample_map.get(adv.sample_id)
        if sample is None:
            raise ValueError(
                f"Advantage sample_id {adv.sample_id} not found in samples"
            )

        # Create sanitized update item
        items.append(
            ActorUpdateItem(
                sample_id=adv.sample_id,
                group_id=adv.group_id,
                record_id=adv.record_id,
                role=adv.role,
                prompt=sample.prompt,
                response=sample.response,
                reward=adv.reward,
                advantage=adv.advantage,
                selected_for_update=adv.selected_for_update,
                metadata={},  # Explicitly empty to avoid leaking oracle data
            )
        )

    return ActorUpdateBatch(
        role=group.role,
        batch_id=batch_id,
        round_id=round_id,
        step_id=step_id,
        items=items,
        metadata={},
    )
