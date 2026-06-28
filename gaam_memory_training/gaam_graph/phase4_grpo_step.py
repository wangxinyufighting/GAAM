"""
Phase 4 Milestone 3: GRPO Trainer Step Execution

Runs GRPO trainer step over aggregated multi-case actor batches.

Key functions:
- run_phase4_grpo_trainer_step: main orchestration
- run_actor_grpo_update: per-actor update execution
- write_trainer_step_summary_markdown: human-readable summary

Design principle:
Narrow trainer boundary: safe aggregated batches in → validated actor-specific
model updates → auditable checkpoints out.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from gaam_graph.phase4_trainer_input import (
    load_actor_batch,
    load_and_validate_actor_inputs,
    normalize_actor_batch_samples,
)
from gaam_graph.phase4_trainer_schema import (
    Phase4ActorTrainerReport,
    Phase4TrainerBackend,
    Phase4TrainerStepConfig,
    Phase4TrainerStepManifest,
    aggregate_actor_trainer_status,
    compute_trainer_step_metrics,
    generate_trainer_step_run_id,
    sanitize_metrics,
)
from gaam_graph.grpo_schema import ActorRole, ActorUpdateBatch
from gaam_graph.local_dataproto_adapter import (
    _tensor_bundle_to_torch_payload,
    actor_update_batch_to_local_dataproto,
)


def _numeric_values(values: list[Any]) -> list[float]:
    """Return finite numeric values from a heterogeneous list."""
    import math

    numeric: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            value_float = float(value)
            if math.isfinite(value_float):
                numeric.append(value_float)
    return numeric


def _write_checkpoint_registry(
    manifest: Phase4TrainerStepManifest,
    output_dir: Path,
    registry_path: Path | None,
) -> Path:
    """Write a lightweight Phase 4 checkpoint registry for downstream runs."""
    if registry_path is None:
        registry_path = output_dir / "checkpoint_registry.json"

    registry_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoints = []
    latest_by_actor: dict[str, str] = {}
    for report in manifest.actor_reports:
        if not report.written_checkpoint_id:
            continue
        checkpoint = {
            "checkpoint_id": report.written_checkpoint_id,
            "actor_role": report.actor_role.value,
            "backend": report.backend.value,
            "round_id": manifest.round_id,
            "step_id": manifest.step_id,
            "checkpoint_path": report.written_checkpoint_path,
            "source_batch_path": report.input_batch_path,
            "updated": report.weights_written,
            "handoff_only": report.handoff_only,
            "metrics": {
                "reward_mean": report.reward_mean,
                "loss": report.loss,
                "approx_kl": report.approx_kl,
                "num_selected_samples": report.num_selected_samples,
            },
        }
        checkpoints.append(checkpoint)
        if report.status == "succeeded" and (
            report.weights_written or report.backend == Phase4TrainerBackend.DRY_RUN
        ):
            latest_by_actor[report.actor_role.value] = report.written_checkpoint_id

    registry = {
        "registry_id": f"{manifest.run_id}_checkpoint_registry",
        "trainer_dir": str(output_dir),
        "manifest_version": "phase4_checkpoint_registry_v1",
        "checkpoints": checkpoints,
        "latest_by_actor": latest_by_actor,
    }

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    return registry_path


def _load_actor_tokenizer_for_handoff(
    actor_role: ActorRole,
    config: Phase4TrainerStepConfig,
    report: Phase4ActorTrainerReport,
) -> Any | None:
    """Load the actor tokenizer for VERL handoff, falling back to local deterministic tokenization."""
    model_path = (
        config.memory_builder_model_path
        if actor_role == ActorRole.MEMORY_BUILDER
        else config.question_agent_model_path
    )
    if not model_path:
        report.warnings.append(
            f"No model path configured for {actor_role.value}; using fallback tokenizer for handoff."
        )
        return None

    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        report.warnings.append(
            f"Failed to load tokenizer from {model_path}; using fallback tokenizer: {type(exc).__name__}: {exc}"
        )
        return None


def _write_phase4_verl_handoff(
    *,
    actor_role: ActorRole,
    config: Phase4TrainerStepConfig,
    raw_batch: dict[str, Any],
    source_batch_path: Path,
    output_dir: Path,
    report: Phase4ActorTrainerReport,
) -> None:
    """
    Convert an aggregated ActorUpdateBatch into trainer-ready VERL artifacts.

    This is a real handoff boundary: it validates the sanitized actor batch,
    tensorizes it, enriches trainer fields, and optionally materializes a vendored
    VERL DataProto. It does not perform a distributed weight update inside this
    process.
    """
    import torch
    from gaam_graph.verl_trainer_adapter import (
        VerlTrainerAdapterMode,
        build_trainer_ready_batch,
        trainer_bundle_to_verl_dataproto,
    )

    actor_batch = ActorUpdateBatch.model_validate(raw_batch)
    if actor_batch.role != actor_role:
        raise ValueError(
            f"Batch role mismatch: expected {actor_role.value}, got {actor_batch.role.value}"
        )

    tokenizer = _load_actor_tokenizer_for_handoff(actor_role, config, report)
    local_batch, local_tensor_bundle = actor_update_batch_to_local_dataproto(
        actor_batch,
        tokenizer=tokenizer,
        include_unselected=False,
        max_prompt_length=None,
        max_response_length=None,
        truncation="error",
    )
    if not local_batch.samples:
        raise ValueError(f"VERL handoff received zero selected samples for {actor_role.value}")

    handoff_dir = output_dir / "verl_handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    local_dataproto_path = handoff_dir / f"{actor_role.value}.local_dataproto.json"
    with open(local_dataproto_path, "w", encoding="utf-8") as f:
        json.dump(local_batch.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    tensor_payload = _tensor_bundle_to_torch_payload(local_tensor_bundle, torch)
    local_tensor_path = handoff_dir / f"{actor_role.value}.local_dataproto.pt"
    torch.save(tensor_payload, local_tensor_path)

    trainer_ready_batch, trainer_bundle = build_trainer_ready_batch(
        local_batch=local_batch,
        tensor_payload=tensor_payload,
        mode=VerlTrainerAdapterMode.VENDORED_VERL,
        allow_stub_logprobs=True,
        returns_strategy="advantages_as_returns",
    )

    trainer_ready_batch_path = output_dir / "trainer_ready_batch.json"
    with open(trainer_ready_batch_path, "w", encoding="utf-8") as f:
        json.dump(trainer_ready_batch.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    trainer_tensor_path = output_dir / "trainer_tensor_payload.pt"
    torch.save(
        {
            "batch": trainer_bundle.batch,
            "non_tensor_batch": trainer_bundle.non_tensor_batch,
            "meta_info": trainer_bundle.meta_info,
        },
        trainer_tensor_path,
    )

    verl_dataproto_path: Path | None = None
    verl_conversion_status = "not_requested"
    try:
        verl_dataproto = trainer_bundle_to_verl_dataproto(trainer_bundle)
        verl_dataproto_path = output_dir / "verl_dataproto.pt"
        torch.save(verl_dataproto, verl_dataproto_path)
        verl_conversion_status = "succeeded"
    except Exception as exc:
        verl_conversion_status = "failed"
        report.warnings.append(
            f"Vendored VERL DataProto conversion failed after tensor handoff: {type(exc).__name__}: {exc}"
        )

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = {
        "checkpoint_id": f"{actor_role.value}_round{config.round_id:03d}_step{config.step_id:03d}",
        "actor_role": actor_role.value,
        "round_id": config.round_id,
        "step_id": config.step_id,
        "backend": config.backend.value,
        "dry_run": False,
        "handoff_only": True,
        "weights_written": False,
        "source_batch_path": str(source_batch_path),
        "local_dataproto_path": str(local_dataproto_path),
        "local_tensor_path": str(local_tensor_path),
        "trainer_ready_batch_path": str(trainer_ready_batch_path),
        "trainer_tensor_path": str(trainer_tensor_path),
        "verl_dataproto_path": str(verl_dataproto_path) if verl_dataproto_path else None,
        "verl_conversion_status": verl_conversion_status,
        "created_at": datetime.utcnow().isoformat(),
    }
    checkpoint_metadata_path = checkpoint_dir / "checkpoint_metadata.json"
    with open(checkpoint_metadata_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_metadata, f, indent=2)

    report.status = "succeeded"
    report.handoff_only = True
    report.weights_written = False
    report.written_checkpoint_id = checkpoint_metadata["checkpoint_id"]
    report.written_checkpoint_path = str(checkpoint_dir)
    report.trainer_ready_batch_path = str(trainer_ready_batch_path)
    report.tensor_payload_path = str(trainer_tensor_path)
    report.warnings.extend(trainer_ready_batch.metadata.warnings)
    report.warnings.append(
        "VERL handoff completed; this step wrote trainer-ready artifacts but did not update model weights in-process."
    )


def _actor_model_path(actor_role: ActorRole, config: Phase4TrainerStepConfig) -> str | None:
    return (
        config.memory_builder_model_path
        if actor_role == ActorRole.MEMORY_BUILDER
        else config.question_agent_model_path
    )


def _actor_checkpoint_path(actor_role: ActorRole, config: Phase4TrainerStepConfig) -> str | None:
    return (
        config.memory_builder_checkpoint_path
        if actor_role == ActorRole.MEMORY_BUILDER
        else config.question_agent_checkpoint_path
    )


def _actor_checkpoint_id(actor_role: ActorRole, config: Phase4TrainerStepConfig) -> str | None:
    return (
        config.memory_builder_checkpoint_id
        if actor_role == ActorRole.MEMORY_BUILDER
        else config.question_agent_checkpoint_id
    )


def _write_full_checkpoint_metadata(
    *,
    checkpoint_dir: Path,
    actor_role: ActorRole,
    config: Phase4TrainerStepConfig,
    source_batch_path: Path,
    actor_batch: ActorUpdateBatch,
    update_result: Any,
    policy_metadata: Any,
    loaded_checkpoint_path: str | None,
    loaded_model_path: str | None,
    handoff_dir: Path,
) -> dict[str, Any]:
    """Augment LocalHF checkpoint metadata with Phase 4/Code-A1 details."""
    metadata_path = checkpoint_dir / "checkpoint_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint_id = f"{actor_role.value}_round{config.round_id:03d}_step{config.step_id:03d}"
    metadata.update(
        {
            "checkpoint_id": checkpoint_id,
            "actor_role": actor_role.value,
            "role": actor_role.value,
            "round_id": config.round_id,
            "step_id": config.step_id,
            "phase4_backend": config.backend.value,
            "backend": "local_hf_full_checkpoint",
            "code_a1_compatible": True,
            "verl_compatible": True,
            "handoff_only": False,
            "weights_written": bool(update_result.updated),
            "source_batch_path": str(source_batch_path),
            "source_batch_id": actor_batch.batch_id,
            "loaded_checkpoint_id": _actor_checkpoint_id(actor_role, config),
            "loaded_checkpoint_path": loaded_checkpoint_path,
            "loaded_model_path": loaded_model_path,
            "trainer_ready_batch_path": str(handoff_dir.parent / "trainer_ready_batch.json"),
            "trainer_tensor_path": str(handoff_dir.parent / "trainer_tensor_payload.pt"),
            "created_at": datetime.utcnow().isoformat(),
            "update_result": update_result.model_dump(mode="json"),
            "policy_checkpoint_metadata": policy_metadata.model_dump(mode="json"),
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _run_phase4_full_checkpoint_update(
    *,
    actor_role: ActorRole,
    config: Phase4TrainerStepConfig,
    raw_batch: dict[str, Any],
    source_batch_path: Path,
    output_dir: Path,
    report: Phase4ActorTrainerReport,
) -> None:
    """Run a real full-model HF update and save a full checkpoint."""
    from gaam_graph.local_policy_clients import LocalHFPolicyClient, LocalHFPolicyConfig

    actor_batch = ActorUpdateBatch.model_validate(raw_batch)
    if actor_batch.role != actor_role:
        raise ValueError(
            f"Batch role mismatch: expected {actor_role.value}, got {actor_batch.role.value}"
        )
    if not actor_batch.items:
        raise ValueError(f"Full checkpoint update received zero items for {actor_role.value}")

    selected_count = sum(1 for item in actor_batch.items if item.selected_for_update)
    print(
        f"[GRPO update] {actor_role.value}: loading policy for "
        f"{len(actor_batch.items)} items ({selected_count} selected)",
        flush=True,
    )

    checkpoint_path = _actor_checkpoint_path(actor_role, config)
    base_model_path = _actor_model_path(actor_role, config)
    if checkpoint_path and Path(checkpoint_path).exists():
        report.loaded_checkpoint_path = checkpoint_path
        report.loaded_checkpoint_id = _actor_checkpoint_id(actor_role, config)
        client = LocalHFPolicyClient.from_checkpoint(
            checkpoint_path,
            config_overrides={
                "learning_rate": config.learning_rate,
                "max_grad_norm": config.max_grad_norm,
                "micro_batch_size": max(1, config.mini_batch_size),
                "gradient_accumulation_steps": max(1, config.gradient_accumulation_steps),
                "train_lora": False,
                "trust_remote_code": True,
            },
        )
        loaded_model_path = None
    else:
        if checkpoint_path:
            report.warnings.append(
                f"Configured checkpoint path does not exist for {actor_role.value}: {checkpoint_path}; loading base model."
            )
        if not base_model_path:
            raise ValueError(f"No model path configured for {actor_role.value}")
        loaded_model_path = base_model_path
        client = LocalHFPolicyClient(
            LocalHFPolicyConfig(
                role=actor_role,
                policy_id=f"{actor_role.value}_phase4_policy",
                model_path=base_model_path,
                tokenizer_path=base_model_path,
                learning_rate=config.learning_rate,
                max_grad_norm=config.max_grad_norm,
                micro_batch_size=max(1, config.mini_batch_size),
                gradient_accumulation_steps=max(1, config.gradient_accumulation_steps),
                train_lora=False,
                trust_remote_code=True,
            )
        )

    print(
        f"[GRPO update] {actor_role.value}: running advantage-weighted update "
        f"lr={config.learning_rate} mini_batch_size={config.mini_batch_size} "
        f"grad_accum={config.gradient_accumulation_steps}",
        flush=True,
    )
    update_result = client.update_grpo(actor_batch, dry_run=False)
    print(
        f"[GRPO update] {actor_role.value}: updated={update_result.updated} "
        f"loss={update_result.loss} grad_norm={update_result.grad_norm} "
        f"optimizer_steps={update_result.metrics.get('num_optimizer_steps')}",
        flush=True,
    )

    checkpoint_dir = output_dir / "checkpoint"
    policy_metadata = client.save_checkpoint(
        checkpoint_dir,
        round_id=config.round_id,
        step_id=config.step_id,
        source_batch_id=actor_batch.batch_id,
        updated=update_result.updated,
        metrics={
            **update_result.metrics,
            "reward_mean": update_result.mean_reward,
            "advantage_mean": update_result.mean_advantage,
            "loss": update_result.loss,
            "grad_norm": update_result.grad_norm,
            "phase4_backend": config.backend.value,
            "code_a1_compatible": True,
            "verl_compatible": True,
        },
    )

    handoff_dir = output_dir / "verl_handoff"
    metadata = _write_full_checkpoint_metadata(
        checkpoint_dir=checkpoint_dir,
        actor_role=actor_role,
        config=config,
        source_batch_path=source_batch_path,
        actor_batch=actor_batch,
        update_result=update_result,
        policy_metadata=policy_metadata,
        loaded_checkpoint_path=checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else None,
        loaded_model_path=loaded_model_path,
        handoff_dir=handoff_dir,
    )

    report.status = "succeeded" if update_result.updated else "partial"
    report.handoff_only = False
    report.weights_written = bool(update_result.updated)
    report.written_checkpoint_id = metadata["checkpoint_id"]
    report.written_checkpoint_path = str(checkpoint_dir)
    report.loss = update_result.loss
    report.grad_norm = update_result.grad_norm
    report.learning_rate = config.learning_rate
    report.reward_mean = update_result.mean_reward
    report.advantage_mean = update_result.mean_advantage
    if not update_result.updated:
        report.warnings.append(
            f"{actor_role.value} full checkpoint was saved, but update_result.updated=False"
        )
    report.warnings.append(
        "Full-model checkpoint written with model/, tokenizer/, and optimizer.pt."
    )
    print(
        f"[GRPO update] {actor_role.value}: checkpoint written to {checkpoint_dir} "
        f"weights_written={report.weights_written}",
        flush=True,
    )


def run_actor_grpo_update(
    actor_role: ActorRole,
    config: Phase4TrainerStepConfig,
    actor_input_summary: dict[str, Any],
    output_dir: Path,
) -> Phase4ActorTrainerReport:
    """
    Run GRPO update for one actor.

    Args:
        actor_role: Actor to update
        config: Trainer step configuration
        actor_input_summary: Actor input metadata (from phase4_trainer_input)
        output_dir: Actor-specific output directory

    Returns:
        Phase4ActorTrainerReport with update results
    """
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize report
    report = Phase4ActorTrainerReport(
        actor_role=actor_role,
        status="pending",
        backend=config.backend,
        input_batch_path=actor_input_summary["batch_path"],
        input_batch_manifest_path=actor_input_summary.get("batch_manifest_path"),
        output_dir=str(output_dir),
        num_input_samples=actor_input_summary.get("num_samples", 0),
        no_leakage_passed=actor_input_summary.get("no_leakage_passed", False),
    )
    if actor_role == ActorRole.MEMORY_BUILDER:
        report.loaded_checkpoint_id = config.memory_builder_checkpoint_id
        report.loaded_checkpoint_path = config.memory_builder_checkpoint_path
    elif actor_role == ActorRole.QUESTION_AGENT:
        report.loaded_checkpoint_id = config.question_agent_checkpoint_id
        report.loaded_checkpoint_path = config.question_agent_checkpoint_path

    try:
        # Load batch
        batch_path = Path(actor_input_summary["batch_path"])
        batch = load_actor_batch(batch_path)

        samples = normalize_actor_batch_samples(batch)
        num_selected = sum(
            1 for sample in samples if sample.get("selected_for_update", True)
        )

        report.num_selected_samples = num_selected

        # Compute reward stats if available
        rewards = _numeric_values(
            [
                sample.get("reward", sample.get("reward_score", sample.get("score")))
                for sample in samples
            ]
        )

        if rewards:
            import statistics

            report.reward_mean = statistics.mean(rewards)

        # Backend-specific execution
        if config.backend == Phase4TrainerBackend.DRY_RUN:
            # Dry run: validate only, no weight updates
            report.status = "succeeded"
            report.warnings.append("Dry-run mode: no model weights updated")

            # Write dry-run checkpoint metadata
            checkpoint_dir = output_dir / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_metadata = {
                "checkpoint_id": f"{actor_role.value}_round{config.round_id:03d}_step{config.step_id:03d}",
                "actor_role": actor_role.value,
                "round_id": config.round_id,
                "step_id": config.step_id,
                "backend": config.backend.value,
                "dry_run": True,
                "weights_written": False,
                "source_batch_path": str(batch_path),
                "created_at": datetime.utcnow().isoformat(),
            }

            checkpoint_metadata_path = checkpoint_dir / "checkpoint_metadata.json"
            with open(checkpoint_metadata_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_metadata, f, indent=2)

            report.written_checkpoint_id = checkpoint_metadata["checkpoint_id"]
            report.written_checkpoint_path = str(checkpoint_dir)
            report.weights_written = False
            report.handoff_only = False

        elif config.backend == Phase4TrainerBackend.LOCAL_GRPO:
            # Local GRPO: minimal local implementation
            report.status = "partial"
            report.warnings.append(
                "LOCAL_GRPO backend did not update model weights in this implementation"
            )

            # Write checkpoint metadata
            checkpoint_dir = output_dir / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_metadata = {
                "checkpoint_id": f"{actor_role.value}_round{config.round_id:03d}_step{config.step_id:03d}",
                "actor_role": actor_role.value,
                "round_id": config.round_id,
                "step_id": config.step_id,
                "backend": config.backend.value,
                "dry_run": False,
                "weights_written": False,  # Placeholder until real local GRPO implemented
                "source_batch_path": str(batch_path),
                "created_at": datetime.utcnow().isoformat(),
                "metrics": {
                    "reward_mean": report.reward_mean,
                    "num_selected_samples": num_selected,
                },
            }

            checkpoint_metadata_path = checkpoint_dir / "checkpoint_metadata.json"
            with open(checkpoint_metadata_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_metadata, f, indent=2)

            report.written_checkpoint_id = checkpoint_metadata["checkpoint_id"]
            report.written_checkpoint_path = str(checkpoint_dir)
            report.weights_written = False
            report.handoff_only = False

        elif config.backend == Phase4TrainerBackend.VERL:
            _write_phase4_verl_handoff(
                actor_role=actor_role,
                config=config,
                raw_batch=batch,
                source_batch_path=batch_path,
                output_dir=output_dir,
                report=report,
            )
            _run_phase4_full_checkpoint_update(
                actor_role=actor_role,
                config=config,
                raw_batch=batch,
                source_batch_path=batch_path,
                output_dir=output_dir,
                report=report,
            )

        else:
            report.errors.append(f"Unknown backend: {config.backend}")
            report.status = "failed"

        # Write trainer-ready batch (normalized format)
        if not report.trainer_ready_batch_path:
            trainer_ready_batch_path = output_dir / "trainer_ready_batch.json"
            with open(trainer_ready_batch_path, "w", encoding="utf-8") as f:
                json.dump(batch, f, indent=2)
            report.trainer_ready_batch_path = str(trainer_ready_batch_path)

    except Exception as e:
        report.status = "failed"
        report.errors.append(f"Actor update failed: {e}")

    # Record update time
    report.update_seconds = time.time() - start_time

    # Write actor report
    report_path = output_dir / "trainer_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    return report


def run_phase4_grpo_trainer_step(
    config: Phase4TrainerStepConfig,
) -> Phase4TrainerStepManifest:
    """
    Run Phase 4 GRPO trainer step over aggregated multi-case batches.

    Pipeline:
    1. Load multi-case rollout manifest
    2. Locate and load actor batches
    3. Validate no-leakage and split policy
    4. Run per-actor GRPO updates
    5. Write checkpoints and reports
    6. Update checkpoint registry (if provided)
    7. Write trainer step manifest

    Args:
        config: Trainer step configuration

    Returns:
        Phase4TrainerStepManifest with results
    """
    if config.backend in {Phase4TrainerBackend.LOCAL_GRPO, Phase4TrainerBackend.VERL} and not config.allow_mvp_backends:
        raise ValueError(
            f"Phase4TrainerBackend.{config.backend.value} is a legacy MVP trainer path. "
            "LOCAL_GRPO does not update weights, and the old VERL backend is only a handoff/local-HF shim. "
            "Use scripts/run_native_verl_dual_cotraining.sh for real Code-A1/VERL GRPO training, "
            "or set allow_mvp_backends=True only for smoke/debug runs."
        )

    run_dir = Path(config.source_multi_case_run_dir)
    output_dir = Path(config.output_dir)

    # Check output directory
    if output_dir.exists() and not config.overwrite and not config.resume:
        raise FileExistsError(
            f"Output directory exists: {output_dir}. Use --overwrite or --resume."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate run ID
    multi_case_manifest, actor_inputs, validation_errors = load_and_validate_actor_inputs(
        run_dir,
        config.actors,
        require_non_empty=config.require_non_empty_batch
        and not config.allow_empty_batch,
        strict_no_leakage=config.strict_no_leakage,
    )

    if not multi_case_manifest:
        raise ValueError(f"Failed to load multi-case manifest: {validation_errors}")

    run_id = generate_trainer_step_run_id(
        multi_case_manifest.split, config.round_id, config.step_id
    )

    # Initialize manifest
    manifest = Phase4TrainerStepManifest(
        run_id=run_id,
        source_multi_case_run_dir=config.source_multi_case_run_dir,
        source_multi_case_manifest_path=str(
            run_dir / "multi_case_rollout_manifest.json"
        ),
        output_dir=str(output_dir),
        split=multi_case_manifest.split,
        round_id=config.round_id,
        step_id=config.step_id,
        backend=config.backend,
        status="running",
        started_at=datetime.utcnow().isoformat(),
    )

    # Fail early if validation errors
    if validation_errors:
        manifest.status = "failed"
        manifest.errors = validation_errors
        manifest.finished_at = datetime.utcnow().isoformat()
        write_phase4_trainer_step_manifest(manifest, output_dir)
        return manifest

    # Write input summaries
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    for actor_role, actor_input in actor_inputs.items():
        input_summary_path = inputs_dir / f"{actor_role.value}.input_summary.json"
        with open(input_summary_path, "w", encoding="utf-8") as f:
            json.dump(actor_input.model_dump(), f, indent=2)

    # Run per-actor updates
    actor_reports: list[Phase4ActorTrainerReport] = []

    for actor_role in config.actors:
        if actor_role not in actor_inputs:
            # Actor input loading failed
            report = Phase4ActorTrainerReport(
                actor_role=actor_role,
                status="failed",
                backend=config.backend,
                input_batch_path="",
                output_dir="",
                errors=[f"Failed to load {actor_role.value} input"],
            )
            actor_reports.append(report)
            continue

        actor_input = actor_inputs[actor_role]
        actor_output_dir = output_dir / actor_role.value

        try:
            report = run_actor_grpo_update(
                actor_role,
                config,
                actor_input.model_dump(),
                actor_output_dir,
            )
            actor_reports.append(report)

        except Exception as e:
            report = Phase4ActorTrainerReport(
                actor_role=actor_role,
                status="failed",
                backend=config.backend,
                input_batch_path=actor_input.batch_path,
                output_dir=str(actor_output_dir),
                errors=[f"Update failed: {e}"],
            )
            actor_reports.append(report)

    # Aggregate status
    manifest.actor_reports = actor_reports
    manifest.status = aggregate_actor_trainer_status(actor_reports)
    manifest.finished_at = datetime.utcnow().isoformat()

    # Compute metrics
    manifest.metrics = sanitize_metrics(compute_trainer_step_metrics(actor_reports))

    # Write checkpoint registry
    registry_path = _write_checkpoint_registry(
        manifest,
        output_dir,
        Path(config.checkpoint_registry_path)
        if config.checkpoint_registry_path
        else None,
    )
    manifest.checkpoint_registry_path = str(registry_path)

    # Write manifest
    write_phase4_trainer_step_manifest(manifest, output_dir)

    # Write summary markdown
    write_trainer_step_summary_markdown(manifest, output_dir)

    return manifest


def write_phase4_trainer_step_manifest(
    manifest: Phase4TrainerStepManifest,
    output_dir: Path,
) -> None:
    """Write trainer step manifest to JSON."""
    manifest_path = output_dir / "trainer_step_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)


def write_trainer_step_summary_markdown(
    manifest: Phase4TrainerStepManifest,
    output_dir: Path,
) -> None:
    """Write human-readable summary in markdown format."""
    summary_path = output_dir / "summary.md"

    lines = [
        f"# Phase 4 Trainer Step: {manifest.run_id}",
        "",
        f"**Status:** {manifest.status}",
        f"**Split:** {manifest.split.value}",
        f"**Round:** {manifest.round_id}, **Step:** {manifest.step_id}",
        f"**Backend:** {manifest.backend.value}",
        f"**Started:** {manifest.started_at}",
        f"**Finished:** {manifest.finished_at or 'N/A'}",
        "",
        "## Source",
        "",
        f"- Multi-case run: `{manifest.source_multi_case_run_dir}`",
        "",
        "## Actor Updates",
        "",
    ]

    for report in manifest.actor_reports:
        lines.append(f"### {report.actor_role.value}")
        lines.append("")
        lines.append(f"- **Status:** {report.status}")
        lines.append(f"- **Input samples:** {report.num_input_samples}")
        lines.append(f"- **Selected samples:** {report.num_selected_samples}")

        if report.reward_mean is not None:
            lines.append(f"- **Reward mean:** {report.reward_mean:.4f}")

        if report.loss is not None:
            lines.append(f"- **Loss:** {report.loss:.4f}")

        if report.approx_kl is not None:
            lines.append(f"- **Approx KL:** {report.approx_kl:.4f}")

        if report.written_checkpoint_id:
            lines.append(f"- **Checkpoint:** {report.written_checkpoint_id}")

        if report.no_leakage_passed:
            lines.append("- **No-leakage:** ✅ passed")
        else:
            lines.append("- **No-leakage:** ❌ failed")

        if report.warnings:
            lines.append("- **Warnings:**")
            for warning in report.warnings:
                lines.append(f"  - {warning}")

        if report.errors:
            lines.append("- **Errors:**")
            for error in report.errors:
                lines.append(f"  - {error}")

        lines.append("")

    lines.append("## Metrics")
    lines.append("")

    for key, value in manifest.metrics.items():
        if isinstance(value, float):
            lines.append(f"- **{key}:** {value:.4f}")
        else:
            lines.append(f"- **{key}:** {value}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
