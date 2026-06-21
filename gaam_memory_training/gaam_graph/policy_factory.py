"""
Policy factory for building policy clients.

This module provides a factory function for constructing policy clients
based on backend type (stub or local_hf).
"""

from __future__ import annotations

from gaam_graph.grpo_schema import ActorRole
from gaam_graph.policy_clients import (
    BasePolicyClient,
    StubMemoryBuilderPolicy,
    StubQuestionAgentPolicy,
)


def build_policy_client(
    *,
    role: ActorRole,
    backend: str,
    policy_id: str | None = None,
    local_hf_config: dict | None = None,
) -> BasePolicyClient:
    """
    Build a policy client based on backend type.

    Args:
        role: Actor role (MEMORY_BUILDER or QUESTION_AGENT)
        backend: Backend type ('stub' or 'local_hf')
        policy_id: Optional policy ID
        local_hf_config: Optional LocalHFPolicyConfig dict for local_hf backend

    Returns:
        BasePolicyClient instance

    Raises:
        ValueError: If backend is unknown or config is invalid
    """

    if backend == "stub":
        if role == ActorRole.MEMORY_BUILDER:
            return StubMemoryBuilderPolicy(policy_id=policy_id)
        elif role == ActorRole.QUESTION_AGENT:
            return StubQuestionAgentPolicy(policy_id=policy_id)
        else:
            raise ValueError(f"Unknown role: {role}")

    elif backend == "local_hf":
        # Import local HF module (may fail if dependencies missing)
        try:
            from gaam_graph.local_policy_clients import (
                LocalHFPolicyClient,
                LocalHFPolicyConfig,
            )
        except ImportError as e:
            raise ImportError(
                f"Cannot use local_hf backend: {e}\n"
                "Install required dependencies: pip install torch transformers"
            )

        if local_hf_config is None:
            raise ValueError("local_hf backend requires local_hf_config")

        # Build config
        config_dict = {**local_hf_config, "role": role}
        if policy_id is not None:
            config_dict["policy_id"] = policy_id

        config = LocalHFPolicyConfig(**config_dict)

        return LocalHFPolicyClient(config)

    else:
        raise ValueError(f"Unknown backend: {backend}")
