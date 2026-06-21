"""
Local trainable policy clients for Phase 2 Milestone 4.

This module provides real local trainable model backends using HuggingFace transformers.
It replaces Milestone 3 stub policy clients with actual advantage-weighted causal language
modeling updates.

Design principles:
1. Use existing BasePolicyClient interface from Milestone 3
2. Support LoRA and full-model training
3. Advantage-weighted response NLL as training objective
4. Optional stabilizers: advantage clipping, gradient clipping
5. Real checkpoints with model/tokenizer/optimizer state
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator

from gaam_graph.grpo_schema import ActorRole, ActorUpdateBatch
from gaam_graph.policy_clients import (
    BasePolicyClient,
    PolicyUpdateResult,
    PolicyCheckpointMetadata,
)


QWEN_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ============================================================================
# Dependency Guards
# ============================================================================

def require_local_training_dependencies():
    """
    Check and import local training dependencies.

    Returns:
        Tuple of (torch, transformers, peft_available)

    Raises:
        ImportError: If required dependencies are missing
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "Local HF backend requires torch. "
            "Install with: pip install torch transformers"
        )

    try:
        import transformers
    except ImportError:
        raise ImportError(
            "Local HF backend requires transformers. "
            "Install with: pip install torch transformers"
        )

    # Check for optional PEFT
    try:
        import peft
        peft_available = True
    except ImportError:
        peft_available = False

    return torch, transformers, peft_available


# ============================================================================
# Configuration
# ============================================================================

class LocalHFPolicyConfig(BaseModel):
    """Configuration for local HuggingFace policy client."""
    role: ActorRole
    model_path: str
    policy_id: str | None = None
    tokenizer_path: str | None = None
    output_dtype: str = "float32"
    device: str = "auto"
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    max_seq_length: int = 2048
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    advantage_clip: float | None = 5.0
    loss_clip: float | None = None
    normalize_advantages_in_batch: bool = False
    train_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    trust_remote_code: bool = False

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        if not v:
            raise ValueError("model_path must be non-empty")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        return v

    @field_validator("max_seq_length")
    @classmethod
    def validate_max_seq_length(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_seq_length must be positive")
        return v

    @field_validator("micro_batch_size")
    @classmethod
    def validate_micro_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("micro_batch_size must be positive")
        return v

    @field_validator("gradient_accumulation_steps")
    @classmethod
    def validate_gradient_accumulation_steps(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        return v


# ============================================================================
# Tokenization Helper
# ============================================================================

def build_prompt_response_labels(
    tokenizer,
    prompt: str,
    response: str,
    max_seq_length: int,
):
    """
    Build input_ids, attention_mask, and labels for advantage-weighted training.

    Rules:
    - Tokenize prompt and response separately
    - Labels are -100 for prompt positions
    - Labels are token IDs for response positions
    - Truncate from left if sequence exceeds max_seq_length
    - Preserve as much response as possible

    Args:
        tokenizer: HuggingFace tokenizer
        prompt: Prompt text
        response: Response text
        max_seq_length: Maximum sequence length

    Returns:
        Dict with input_ids, attention_mask, labels (all torch tensors)
        None if response becomes empty after truncation
    """
    torch, _, _ = require_local_training_dependencies()

    # Tokenize prompt and full sequence
    prompt_tokens = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    full_tokens = tokenizer(prompt + response, add_special_tokens=True, return_tensors="pt")

    prompt_length = prompt_tokens["input_ids"].shape[1]
    full_length = full_tokens["input_ids"].shape[1]
    response_length = full_length - prompt_length

    # If too long, truncate from left while preserving response
    if full_length > max_seq_length:
        # Keep as much response as possible
        if response_length > max_seq_length:
            # Response alone is too long, truncate it
            start_idx = full_length - max_seq_length
            input_ids = full_tokens["input_ids"][:, start_idx:]
            attention_mask = full_tokens["attention_mask"][:, start_idx:]
            # All positions are response now
            labels = input_ids.clone()
        else:
            # Truncate prompt, keep full response
            tokens_to_keep = max_seq_length
            start_idx = full_length - tokens_to_keep
            input_ids = full_tokens["input_ids"][:, start_idx:]
            attention_mask = full_tokens["attention_mask"][:, start_idx:]

            # Recompute prompt length after truncation
            new_prompt_length = tokens_to_keep - response_length
            labels = input_ids.clone()
            labels[:, :new_prompt_length] = -100
    else:
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        labels = input_ids.clone()
        labels[:, :prompt_length] = -100

    # Check if response has any non-masked positions
    response_mask = labels != -100
    if not response_mask.any():
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def infer_lora_target_modules(model) -> list[str] | None:
    """Infer safe LoRA target modules for common Qwen/LLaMA-style architectures."""
    module_leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    qwen_targets = [name for name in QWEN_LORA_TARGET_MODULES if name in module_leaf_names]
    if {"q_proj", "v_proj"}.issubset(set(qwen_targets)):
        return qwen_targets
    return None


def load_torch_state(torch, path: Path, *, map_location: str):
    """Load a local torch state dict without triggering unsafe default warnings."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ============================================================================
# Local HF Policy Client
# ============================================================================

class LocalHFPolicyClient(BasePolicyClient):
    """
    Local HuggingFace trainable policy client.

    Uses advantage-weighted causal language modeling as training objective.
    Supports LoRA and full-model training.
    """

    def __init__(self, config: LocalHFPolicyConfig) -> None:
        super().__init__(role=config.role, policy_id=config.policy_id)

        self.config = config

        # Import dependencies
        torch, transformers, peft_available = require_local_training_dependencies()
        self.torch = torch
        self.transformers = transformers

        # Check PEFT availability
        if config.train_lora and not peft_available:
            raise ImportError(
                "train_lora=True but peft is not installed. "
                "Install with: pip install peft"
            )

        if config.train_lora:
            from peft import LoraConfig, get_peft_model
            self.peft = __import__("peft")

        # Resolve device
        self.device = self._resolve_device(config.device)

        # Load tokenizer
        tokenizer_path = config.tokenizer_path or config.model_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=config.trust_remote_code,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
        )

        # Move to device
        self.model = self.model.to(self.device)

        # Attach LoRA if requested
        if config.train_lora:
            target_modules = config.lora_target_modules or infer_lora_target_modules(self.model)
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        # Count parameters
        self.num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.num_total_params = sum(p.numel() for p in self.model.parameters())

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Put model in train mode
        self.model.train()

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        torch, _, _ = require_local_training_dependencies()

        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return device

    def update_grpo(
        self,
        batch: ActorUpdateBatch,
        *,
        dry_run: bool = False,
    ) -> PolicyUpdateResult:
        """Update policy using advantage-weighted GRPO batch."""

        # Validate role
        if batch.role != self.role:
            raise ValueError(
                f"Batch role mismatch: expected {self.role.value}, got {batch.role.value}"
            )

        # Real model backends must enforce no-leakage safety even when called directly.
        from gaam_graph.grpo_trainer import assert_actor_update_batch_safe
        assert_actor_update_batch_safe(batch, allow_teacher_fields_for_stub=False, dry_run=False)

        # Filter selected items
        selected_items = [item for item in batch.items if item.selected_for_update]

        # Compute mean reward and advantage from all items
        mean_reward = None
        mean_advantage = None

        if batch.items:
            rewards = [item.reward for item in batch.items]
            mean_reward = sum(rewards) / len(rewards)

            advantages = [item.advantage for item in batch.items]
            mean_advantage = sum(advantages) / len(advantages)

        # If no selected items, return early
        if not selected_items:
            return PolicyUpdateResult(
                role=self.role,
                policy_id=self.policy_id,
                batch_id=batch.batch_id,
                round_id=batch.round_id,
                step_id=batch.step_id,
                updated=False,
                num_items=len(batch.items),
                num_selected_items=0,
                mean_reward=mean_reward,
                mean_advantage=mean_advantage,
                metrics={
                    "backend": "local_hf",
                    "dry_run": dry_run,
                    "device": str(self.device),
                    "num_trainable_parameters": self.num_trainable_params,
                    "num_total_parameters": self.num_total_params,
                    "skipped_reason": "no_selected_items",
                },
            )

        # Check for all-zero advantages
        selected_advantages = [item.advantage for item in selected_items]
        if all(abs(adv) < 1e-9 for adv in selected_advantages):
            return PolicyUpdateResult(
                role=self.role,
                policy_id=self.policy_id,
                batch_id=batch.batch_id,
                round_id=batch.round_id,
                step_id=batch.step_id,
                updated=False,
                num_items=len(batch.items),
                num_selected_items=len(selected_items),
                mean_reward=mean_reward,
                mean_advantage=mean_advantage,
                metrics={
                    "backend": "local_hf",
                    "dry_run": dry_run,
                    "device": str(self.device),
                    "num_trainable_parameters": self.num_trainable_params,
                    "num_total_parameters": self.num_total_params,
                    "skipped_reason": "zero_advantage",
                },
            )

        # Dry run: tokenize one small batch then return
        if dry_run:
            # Try tokenizing first item
            first_item = selected_items[0]
            tokens = build_prompt_response_labels(
                self.tokenizer,
                first_item.prompt,
                first_item.response,
                self.config.max_seq_length,
            )

            return PolicyUpdateResult(
                role=self.role,
                policy_id=self.policy_id,
                batch_id=batch.batch_id,
                round_id=batch.round_id,
                step_id=batch.step_id,
                updated=False,
                num_items=len(batch.items),
                num_selected_items=len(selected_items),
                mean_reward=mean_reward,
                mean_advantage=mean_advantage,
                metrics={
                    "backend": "local_hf",
                    "dry_run": True,
                    "device": str(self.device),
                    "num_trainable_parameters": self.num_trainable_params,
                    "num_total_parameters": self.num_total_params,
                    "tokenized_sample": tokens is not None,
                },
            )

        # Real update
        return self._real_update(batch, selected_items, mean_reward, mean_advantage)

    def _real_update(
        self,
        batch: ActorUpdateBatch,
        selected_items: list,
        mean_reward: float | None,
        mean_advantage: float | None,
    ) -> PolicyUpdateResult:
        """Perform real model update."""

        torch = self.torch

        # Prepare micro-batches
        total_loss = 0.0
        total_grad_norm = 0.0
        num_backward_steps = 0
        num_optimizer_steps = 0
        num_skipped_items = 0

        response_token_counts = []
        abs_advantages = []

        # Process items in micro-batches
        micro_batches = []
        for i in range(0, len(selected_items), self.config.micro_batch_size):
            micro_batches.append(selected_items[i:i + self.config.micro_batch_size])

        self.optimizer.zero_grad()

        for batch_idx, micro_batch in enumerate(micro_batches):
            micro_batch_data = []
            micro_batch_advantages = []

            # Tokenize micro-batch
            for item in micro_batch:
                tokens = build_prompt_response_labels(
                    self.tokenizer,
                    item.prompt,
                    item.response,
                    self.config.max_seq_length,
                )

                if tokens is None:
                    num_skipped_items += 1
                    continue

                # Clip advantage
                advantage = item.advantage
                if self.config.advantage_clip is not None:
                    advantage = max(-self.config.advantage_clip, min(self.config.advantage_clip, advantage))

                micro_batch_data.append(tokens)
                micro_batch_advantages.append(advantage)
                abs_advantages.append(abs(advantage))

                # Count response tokens
                response_mask = tokens["labels"] != -100
                response_token_counts.append(response_mask.sum().item())

            if not micro_batch_data:
                continue

            # Stack tensors
            input_ids = torch.cat([d["input_ids"] for d in micro_batch_data], dim=0).to(self.device)
            attention_mask = torch.cat([d["attention_mask"] for d in micro_batch_data], dim=0).to(self.device)
            labels = torch.cat([d["labels"] for d in micro_batch_data], dim=0).to(self.device)
            advantages_tensor = torch.tensor(micro_batch_advantages, dtype=torch.float32, device=self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Compute per-token loss
            logits = outputs.logits[:, :-1, :]  # Shift by 1
            labels_shifted = labels[:, 1:]  # Shift by 1

            # Compute cross-entropy per token
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_loss = loss_fct(
                logits.reshape(-1, logits.size(-1)),
                labels_shifted.reshape(-1),
            ).reshape(labels_shifted.shape)

            # Mask to response tokens only
            response_mask = labels_shifted != -100

            # Compute mean NLL per item
            item_nll = (token_loss * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)

            # Advantage-weighted loss
            weighted_loss = (advantages_tensor * item_nll).mean()

            # Apply loss clipping if configured
            if self.config.loss_clip is not None:
                weighted_loss = torch.clamp(weighted_loss, -self.config.loss_clip, self.config.loss_clip)

            # Check for non-finite loss
            if not torch.isfinite(weighted_loss):
                return PolicyUpdateResult(
                    role=self.role,
                    policy_id=self.policy_id,
                    batch_id=batch.batch_id,
                    round_id=batch.round_id,
                    step_id=batch.step_id,
                    updated=False,
                    num_items=len(batch.items),
                    num_selected_items=len(selected_items),
                    mean_reward=mean_reward,
                    mean_advantage=mean_advantage,
                    error="non_finite_loss",
                )

            # Backward
            (weighted_loss / self.config.gradient_accumulation_steps).backward()
            total_loss += weighted_loss.item()
            num_backward_steps += 1

            # Optimizer step after accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(micro_batches):
                # Gradient clipping
                if self.config.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    total_grad_norm += grad_norm.item()

                self.optimizer.step()
                self.optimizer.zero_grad()
                num_optimizer_steps += 1

        # Compute metrics
        avg_loss = total_loss / num_backward_steps if num_backward_steps > 0 else 0.0
        avg_grad_norm = total_grad_norm / num_optimizer_steps if num_optimizer_steps > 0 else 0.0
        mean_response_tokens = sum(response_token_counts) / len(response_token_counts) if response_token_counts else 0.0
        mean_abs_advantage = sum(abs_advantages) / len(abs_advantages) if abs_advantages else 0.0
        max_abs_advantage = max(abs_advantages) if abs_advantages else 0.0

        return PolicyUpdateResult(
            role=self.role,
            policy_id=self.policy_id,
            batch_id=batch.batch_id,
            round_id=batch.round_id,
            step_id=batch.step_id,
            updated=True,
            num_items=len(batch.items),
            num_selected_items=len(selected_items),
            mean_reward=mean_reward,
            mean_advantage=mean_advantage,
            loss=avg_loss,
            grad_norm=avg_grad_norm,
            metrics={
                "backend": "local_hf",
                "dry_run": False,
                "device": str(self.device),
                "num_trainable_parameters": self.num_trainable_params,
                "num_total_parameters": self.num_total_params,
                "num_optimizer_steps": num_optimizer_steps,
                "num_backward_steps": num_backward_steps,
                "num_skipped_items": num_skipped_items,
                "mean_response_tokens": mean_response_tokens,
                "mean_abs_advantage": mean_abs_advantage,
                "max_abs_advantage": max_abs_advantage,
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
        metrics: dict[str, Any] | None = None,
    ) -> PolicyCheckpointMetadata:
        """Save checkpoint with model, tokenizer, and optimizer state."""

        torch = self.torch
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model or adapter
        if self.config.train_lora:
            # Save LoRA adapter
            adapter_dir = checkpoint_dir / "adapter"
            self.model.save_pretrained(adapter_dir)
            model_subdir = "adapter"
        else:
            # Save full model
            model_dir = checkpoint_dir / "model"
            self.model.save_pretrained(model_dir)
            model_subdir = "model"

        # Save tokenizer
        tokenizer_dir = checkpoint_dir / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_dir)

        # Save optimizer state
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # Create metadata
        checkpoint_metadata = PolicyCheckpointMetadata(
            role=self.role,
            policy_id=self.policy_id,
            round_id=round_id,
            step_id=step_id,
            checkpoint_path=str(checkpoint_dir / "checkpoint_metadata.json"),
            source_batch_id=source_batch_id,
            updated=updated,
            backend="local_hf",
            metrics={
                **(metrics or {}),
                "model_path": self.config.model_path,
                "tokenizer_path": self.config.tokenizer_path or self.config.model_path,
                "train_lora": self.config.train_lora,
                "model_subdir": model_subdir,
                "tokenizer_subdir": "tokenizer",
                "optimizer_path": "optimizer.pt",
                "num_trainable_parameters": self.num_trainable_params,
                "num_total_parameters": self.num_total_params,
            },
        )

        # Write metadata
        metadata_path = checkpoint_dir / "checkpoint_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_metadata.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        return checkpoint_metadata

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        config_overrides: dict[str, Any] | None = None,
    ) -> "LocalHFPolicyClient":
        """
        Load policy client from checkpoint.

        Args:
            checkpoint_dir: Path to checkpoint directory
            config_overrides: Optional config overrides

        Returns:
            LocalHFPolicyClient restored from checkpoint
        """
        torch, _, _ = require_local_training_dependencies()

        checkpoint_dir = Path(checkpoint_dir)

        # Load metadata
        metadata_path = checkpoint_dir / "checkpoint_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Extract config from metadata
        metrics = metadata["metrics"]
        train_lora = metrics["train_lora"]
        model_subdir = metrics.get("model_subdir", "adapter" if train_lora else "model")
        tokenizer_subdir = metrics.get("tokenizer_subdir", "tokenizer")
        tokenizer_path = str(checkpoint_dir / tokenizer_subdir)

        # Build config
        if train_lora:
            model_path = metrics["model_path"]
            init_train_lora = False
        else:
            model_path = str(checkpoint_dir / model_subdir)
            init_train_lora = False

        config_dict = {
            "role": ActorRole(metadata["role"]),
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "policy_id": metadata["policy_id"],
            "train_lora": init_train_lora,
        }

        if config_overrides:
            config_dict.update(config_overrides)

        config = LocalHFPolicyConfig(**config_dict)

        # Create client
        client = cls(config)

        # Load model weights
        if train_lora:
            adapter_dir = checkpoint_dir / "adapter"
            if adapter_dir.exists():
                from peft import PeftModel
                client.model = PeftModel.from_pretrained(
                    client.model,
                    str(adapter_dir),
                    is_trainable=True,
                )
                client.config.train_lora = True
                client.model = client.model.to(client.device)
                client.num_trainable_params = sum(
                    p.numel() for p in client.model.parameters() if p.requires_grad
                )
                client.num_total_params = sum(p.numel() for p in client.model.parameters())
                client.optimizer = torch.optim.AdamW(
                    client.model.parameters(),
                    lr=client.config.learning_rate,
                    weight_decay=client.config.weight_decay,
                )
                client.model.train()

        # Load optimizer state
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            client.optimizer.load_state_dict(
                load_torch_state(torch, optimizer_path, map_location=client.device)
            )

        return client
