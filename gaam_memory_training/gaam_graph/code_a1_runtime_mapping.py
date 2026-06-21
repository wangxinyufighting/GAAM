"""
Phase 3 Milestone 1: Code-A1 Runtime Mapping.

This module maps GAAM runtime concepts to Code-A1/verl concepts for future
distributed training integration.

Design principles:
1. Map GAAM actors to Code-A1 actor and reference roles
2. Define worker group and resource pool names
3. Provide draft OmegaConf-style config generation (not executable in Milestone 1)
4. Document the mapping without requiring Code-A1 runtime
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from gaam_graph.grpo_schema import ActorRole
from gaam_graph.distributed_runtime_schema import (
    Phase3RuntimeConfig,
    ActorRuntimeConfig,
)


# ============================================================================
# Code-A1 Role Constants
# ============================================================================

# GAAM → Code-A1 actor role mapping
GAAM_TO_CODE_A1_ROLE = {
    ActorRole.MEMORY_BUILDER: "Role.ActorRollout",
    ActorRole.QUESTION_AGENT: "Role.ActorRollout_B",
}

# GAAM → Code-A1 reference policy role mapping
GAAM_TO_CODE_A1_REF_ROLE = {
    ActorRole.MEMORY_BUILDER: "Role.RefPolicy",
    ActorRole.QUESTION_AGENT: "Role.RefPolicy_B",
}


# ============================================================================
# Runtime Mapping Model
# ============================================================================

class CodeA1RuntimeMapping(BaseModel):
    """
    Mapping from GAAM actor to Code-A1 runtime concepts.

    This mapping defines how GAAM's Memory Builder and Question Agent
    correspond to Code-A1's dual-actor training setup.
    """
    gaam_actor_role: ActorRole
    code_a1_actor_role: str
    code_a1_ref_role: str | None = None
    worker_group_name: str
    resource_pool_name: str
    rollout_data_dir_name: str
    checkpoint_subdir_name: str


# ============================================================================
# Mapping Builder
# ============================================================================

def build_code_a1_runtime_mappings() -> list[CodeA1RuntimeMapping]:
    """
    Build default GAAM → Code-A1 runtime mappings.

    Returns:
        List of mappings for Memory Builder and Question Agent
    """
    mappings = []

    # Memory Builder → Code-A1 Code LLM
    mappings.append(CodeA1RuntimeMapping(
        gaam_actor_role=ActorRole.MEMORY_BUILDER,
        code_a1_actor_role=GAAM_TO_CODE_A1_ROLE[ActorRole.MEMORY_BUILDER],
        code_a1_ref_role=GAAM_TO_CODE_A1_REF_ROLE[ActorRole.MEMORY_BUILDER],
        worker_group_name="memory_builder_worker_group",
        resource_pool_name="memory_builder_pool",
        rollout_data_dir_name="memory_builder_rollout",
        checkpoint_subdir_name="memory_builder",
    ))

    # Question Agent → Code-A1 Test LLM
    mappings.append(CodeA1RuntimeMapping(
        gaam_actor_role=ActorRole.QUESTION_AGENT,
        code_a1_actor_role=GAAM_TO_CODE_A1_ROLE[ActorRole.QUESTION_AGENT],
        code_a1_ref_role=GAAM_TO_CODE_A1_REF_ROLE[ActorRole.QUESTION_AGENT],
        worker_group_name="question_agent_worker_group",
        resource_pool_name="question_agent_pool",
        rollout_data_dir_name="question_agent_rollout",
        checkpoint_subdir_name="question_agent",
    ))

    return mappings


# ============================================================================
# Draft Config Generator
# ============================================================================

def _resolve_checkpoint_id(runtime_config: Phase3RuntimeConfig, actor_config: ActorRuntimeConfig) -> str | None:
    """Resolve explicit or registry-backed checkpoint id for draft config metadata."""
    if actor_config.checkpoint_id:
        return actor_config.checkpoint_id

    if not runtime_config.checkpoint_registry_path:
        return None

    registry_path = runtime_config.checkpoint_registry_path
    if not registry_path.exists():
        return None

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except Exception:
        return None

    latest_by_actor = registry.get("latest_by_actor", {})
    return latest_by_actor.get(actor_config.role.value)


def draft_code_a1_omegaconf(runtime_config: Phase3RuntimeConfig) -> dict[str, Any]:
    """
    Generate a draft Code-A1 OmegaConf-style configuration.

    This is NOT executable in Milestone 1. It documents the config shape
    needed for future Code-A1/verl integration.

    Args:
        runtime_config: Phase 3 runtime configuration

    Returns:
        JSON-safe dictionary shaped like Code-A1 OmegaConf config
    """
    # Get mappings
    mappings = build_code_a1_runtime_mappings()
    memory_mapping = [m for m in mappings if m.gaam_actor_role == ActorRole.MEMORY_BUILDER][0]
    question_mapping = [m for m in mappings if m.gaam_actor_role == ActorRole.QUESTION_AGENT][0]
    memory_checkpoint_id = _resolve_checkpoint_id(runtime_config, runtime_config.memory_builder)
    question_checkpoint_id = _resolve_checkpoint_id(runtime_config, runtime_config.question_agent)

    # Build draft config
    draft_config = {
        # Primary actor (Memory Builder)
        "actor_rollout_ref": {
            "model": {
                "path": runtime_config.memory_builder.model_path or "unknown",
                "tokenizer_path": runtime_config.memory_builder.tokenizer_path,
            },
            "rollout": {
                "name": memory_mapping.worker_group_name,
                "n": runtime_config.num_rollout_workers,
                "log_prob": True,
                "temperature": runtime_config.memory_builder.generation_config.get("temperature", 1.0),
                "top_p": runtime_config.memory_builder.generation_config.get("top_p", 1.0),
            },
            "actor": {
                "optim": runtime_config.memory_builder.update_config.get("optimizer", "AdamW"),
                "lr": runtime_config.memory_builder.update_config.get("learning_rate", 1e-5),
                "ppo_mini_batch_size": runtime_config.memory_builder.update_config.get("batch_size", 8),
            },
            "ref": {
                "log_prob": True,
                "enable": runtime_config.memory_builder.reference_checkpoint_path is not None,
            },
        },
        # Secondary actor (Question Agent)
        "other": {
            "actor_rollout_ref": {
                "model": {
                    "path": runtime_config.question_agent.model_path or "unknown",
                    "tokenizer_path": runtime_config.question_agent.tokenizer_path,
                },
                "rollout": {
                    "name": question_mapping.worker_group_name,
                    "n": runtime_config.num_rollout_workers,
                    "log_prob": True,
                    "temperature": runtime_config.question_agent.generation_config.get("temperature", 1.0),
                    "top_p": runtime_config.question_agent.generation_config.get("top_p", 1.0),
                },
                "actor": {
                    "optim": runtime_config.question_agent.update_config.get("optimizer", "AdamW"),
                    "lr": runtime_config.question_agent.update_config.get("learning_rate", 1e-5),
                    "ppo_mini_batch_size": runtime_config.question_agent.update_config.get("batch_size", 8),
                },
                "ref": {
                    "log_prob": True,
                    "enable": runtime_config.question_agent.reference_checkpoint_path is not None,
                },
            },
            "trainer": {
                "default_local_dir": str(runtime_config.output_dir),
                "rollout_data_dir": question_mapping.rollout_data_dir_name,
            },
            "update_n": runtime_config.question_agent.update_config.get("num_epochs", 1),
        },
        # Trainer config
        "trainer": {
            "n_gpus_per_node": runtime_config.memory_builder.resource_config.get("gpus_per_node", 1),
            "nnodes": runtime_config.memory_builder.resource_config.get("num_nodes", 1),
            "default_local_dir": str(runtime_config.output_dir),
            "rollout_data_dir": memory_mapping.rollout_data_dir_name,
            "project_name": "gaam_phase3",
            "experiment_name": f"round{runtime_config.round_id}_step{runtime_config.step_id}",
            "total_epochs": 1,
            "save_freq": -1,
        },
        # Data config
        "data": {
            "max_prompt_length": runtime_config.memory_builder.generation_config.get("max_prompt_length", 2048),
            "max_response_length": runtime_config.memory_builder.generation_config.get("max_response_length", 1024),
        },
        # Algorithm config
        "algorithm": {
            "adv_estimator": "grpo",  # GAAM uses GRPO
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": 0.0,  # GRPO doesn't use KL divergence
            },
        },
        # GAAM-specific metadata
        "gaam_metadata": {
            "runtime_id": runtime_config.runtime_id,
            "backend": runtime_config.backend.value,
            "round_id": runtime_config.round_id,
            "step_id": runtime_config.step_id,
            "checkpoint_registry_path": str(runtime_config.checkpoint_registry_path) if runtime_config.checkpoint_registry_path else None,
            "memory_checkpoint_id": memory_checkpoint_id,
            "question_checkpoint_id": question_checkpoint_id,
        },
    }

    return draft_config
