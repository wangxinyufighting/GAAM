__version__ = "0.1.0"

# Phase 2 GRPO exports
from gaam_graph.grpo_schema import (
    ActorRole,
    SampleStatus,
    GroupStatus,
    PolicySample,
    GRPORewardItem,
    GRPOAdvantageItem,
    GroupedRollout,
    ActorUpdateItem,
    ActorUpdateBatch,
    GRPOTrainingTrace,
    GRPOTrainingSummary,
)

from gaam_graph.grpo_advantage import (
    compute_grpo_advantages,
    summarize_reward_group,
    compute_advantage_items,
    attach_advantages_to_group,
    build_actor_update_batch,
)

from gaam_graph.policy_clients import (
    BasePolicyClient,
    StubPolicyClient,
    StubMemoryBuilderPolicy,
    StubQuestionAgentPolicy,
    PolicyUpdateResult,
    PolicyCheckpointMetadata,
)

from gaam_graph.grpo_trainer import (
    GRPOTrainerConfig,
    GRPOTrainerStepTrace,
    GRPOTrainerSummary,
    LocalGRPOTrainer,
)

from gaam_graph.policy_factory import build_policy_client

# Phase 2 Milestone 5: Code-A1/verl alignment
from gaam_graph.code_a1_alignment import (
    CodeA1Role,
    GAAMActorMapping,
    AlignmentIssueSeverity,
    AlignmentIssue,
    CodeA1AlignmentReport,
    build_default_actor_mappings,
    check_for_forbidden_fields,
    inspect_rollout_alignment,
    inspect_trainer_alignment,
    build_alignment_report,
)

from gaam_graph.verl_batch_adapter import (
    VerlLikeSample,
    VerlLikeBatch,
    actor_update_batch_to_verl_like_batch,
    export_verl_like_batches,
)

from gaam_graph.checkpoint_registry import (
    RegisteredCheckpoint,
    CheckpointRegistry,
    generate_checkpoint_id,
    discover_checkpoints,
    validate_checkpoints,
    build_checkpoint_registry,
    write_checkpoint_registry,
)

# Phase 3 Milestone 1: Distributed runtime
from gaam_graph.distributed_runtime_schema import (
    Phase3WorkerRole,
    Phase3Backend,
    Phase3JobStatus,
    ActorRuntimeConfig,
    Phase3RuntimeConfig,
    Phase3RolloutJob,
    Phase3WorkerSpec,
    Phase3WorkerPlan,
    Phase3RuntimeTrace,
    Phase3ReadinessReport,
)

from gaam_graph.code_a1_runtime_mapping import (
    GAAM_TO_CODE_A1_ROLE,
    GAAM_TO_CODE_A1_REF_ROLE,
    CodeA1RuntimeMapping,
    build_code_a1_runtime_mappings,
    draft_code_a1_omegaconf,
)

from gaam_graph.distributed_runtime import (
    Phase3RuntimePlanner,
    LocalFakeDistributedRuntime,
)

# Phase 3 Milestone 2: Local DataProto adapter
from gaam_graph.local_dataproto_adapter import (
    SimpleFallbackTokenizer,
    LocalTensorSpec,
    LocalDataProtoSample,
    LocalDataProtoBatch,
    LocalDataProtoTensorBundle,
    LocalDataProtoExportManifestEntry,
    actor_update_batch_to_local_dataproto,
    verl_like_batch_to_local_dataproto,
    export_local_dataproto_batches,
)

# Phase 3 Milestone 3: Ray Worker Prototype
from gaam_graph.ray_runtime import (
    RayRolloutWorkerInput,
    RayRolloutWorkerResult,
    RayRuntimeConfig,
    RayRuntimeExecutor,
    is_ray_available,
    run_rollout_worker_job,
)

# Phase 3 Milestone 4: verl Trainer Adapter
from gaam_graph.verl_trainer_adapter import (
    TrainerFieldSource,
    VerlTrainerAdapterMode,
    TrainerBatchMetadata,
    TrainerReadyBatch,
    TrainerTensorBundle,
    VerlTrainerAdapterReport,
    load_local_dataproto_batch,
    load_torch_tensor_payload,
    validate_trainer_input_no_leakage,
    add_stub_logprob_fields,
    compute_returns,
    build_trainer_ready_batch,
    trainer_bundle_to_verl_dataproto,
    run_verl_trainer_adapter_step,
)

# Phase 3 Milestone 5: Distributed Reward Service
from gaam_graph.distributed_reward_schema import (
    RewardWorkerStatus,
    RewardWorkerInput,
    RewardWorkerReport,
    WeaknessBookDelta,
    ReplayBufferItem,
    ReplayBufferManifestEntry,
    DistributedRewardSummary,
)

from gaam_graph.distributed_reward_service import (
    validate_reward_worker_input_no_leakage,
    validate_replay_item_no_leakage,
    sanitize_weakness_delta_for_actor_context,
    load_reward_worker_input,
    build_reward_worker_input_from_job_dir,
    load_reward_artifacts,
    build_weakness_delta,
    build_replay_item,
    run_reward_worker_job_local,
    run_reward_worker_job,
    run_reward_worker_job_ray,
)

from gaam_graph.replay_buffer import (
    ReplayBufferWriter,
    ReplayBufferReader,
)

# Phase 3 Milestone 6: End-to-End GRPO Orchestration
from gaam_graph.phase3_e2e_schema import (
    Phase3E2EBackend,
    Phase3E2EStageName,
    Phase3E2EStageStatus,
    Phase3E2ERunConfig,
    Phase3E2EStageReport,
    Phase3E2ERoundReport,
    Phase3E2ERunReport,
    create_stage_report,
    finalize_stage_report,
)

from gaam_graph.phase3_e2e_orchestrator import (
    Phase3E2EPaths,
    plan_output_paths,
    run_preflight,
    build_runtime_plan,
    run_rollout_stage,
    run_reward_stage,
    run_replay_stage,
    run_trainer_stage,
    run_checkpoint_stage,
    run_inspection_stage,
    run_phase3_end_to_end_grpo,
    write_summary_markdown,
)

__all__ = [
    # Version
    "__version__",
    # GRPO Schema
    "ActorRole",
    "SampleStatus",
    "GroupStatus",
    "PolicySample",
    "GRPORewardItem",
    "GRPOAdvantageItem",
    "GroupedRollout",
    "ActorUpdateItem",
    "ActorUpdateBatch",
    "GRPOTrainingTrace",
    "GRPOTrainingSummary",
    # GRPO Advantage
    "compute_grpo_advantages",
    "summarize_reward_group",
    "compute_advantage_items",
    "attach_advantages_to_group",
    "build_actor_update_batch",
    # Policy Clients
    "BasePolicyClient",
    "StubPolicyClient",
    "StubMemoryBuilderPolicy",
    "StubQuestionAgentPolicy",
    "PolicyUpdateResult",
    "PolicyCheckpointMetadata",
    # GRPO Trainer
    "GRPOTrainerConfig",
    "GRPOTrainerStepTrace",
    "GRPOTrainerSummary",
    "LocalGRPOTrainer",
    # Policy Factory
    "build_policy_client",
    # Code-A1 Alignment
    "CodeA1Role",
    "GAAMActorMapping",
    "AlignmentIssueSeverity",
    "AlignmentIssue",
    "CodeA1AlignmentReport",
    "build_default_actor_mappings",
    "check_for_forbidden_fields",
    "inspect_rollout_alignment",
    "inspect_trainer_alignment",
    "build_alignment_report",
    # verl Batch Adapter
    "VerlLikeSample",
    "VerlLikeBatch",
    "actor_update_batch_to_verl_like_batch",
    "export_verl_like_batches",
    # Checkpoint Registry
    "RegisteredCheckpoint",
    "CheckpointRegistry",
    "generate_checkpoint_id",
    "discover_checkpoints",
    "validate_checkpoints",
    "build_checkpoint_registry",
    "write_checkpoint_registry",
    # Phase 3 Runtime Schema
    "Phase3WorkerRole",
    "Phase3Backend",
    "Phase3JobStatus",
    "ActorRuntimeConfig",
    "Phase3RuntimeConfig",
    "Phase3RolloutJob",
    "Phase3WorkerSpec",
    "Phase3WorkerPlan",
    "Phase3RuntimeTrace",
    "Phase3ReadinessReport",
    # Code-A1 Runtime Mapping
    "GAAM_TO_CODE_A1_ROLE",
    "GAAM_TO_CODE_A1_REF_ROLE",
    "CodeA1RuntimeMapping",
    "build_code_a1_runtime_mappings",
    "draft_code_a1_omegaconf",
    # Distributed Runtime
    "Phase3RuntimePlanner",
    "LocalFakeDistributedRuntime",
    # Local DataProto Adapter
    "SimpleFallbackTokenizer",
    "LocalTensorSpec",
    "LocalDataProtoSample",
    "LocalDataProtoBatch",
    "LocalDataProtoTensorBundle",
    "LocalDataProtoExportManifestEntry",
    "actor_update_batch_to_local_dataproto",
    "verl_like_batch_to_local_dataproto",
    "export_local_dataproto_batches",
    # Ray Runtime
    "RayRolloutWorkerInput",
    "RayRolloutWorkerResult",
    "RayRuntimeConfig",
    "RayRuntimeExecutor",
    "is_ray_available",
    "run_rollout_worker_job",
    # verl Trainer Adapter
    "TrainerFieldSource",
    "VerlTrainerAdapterMode",
    "TrainerBatchMetadata",
    "TrainerReadyBatch",
    "TrainerTensorBundle",
    "VerlTrainerAdapterReport",
    "load_local_dataproto_batch",
    "load_torch_tensor_payload",
    "validate_trainer_input_no_leakage",
    "add_stub_logprob_fields",
    "compute_returns",
    "build_trainer_ready_batch",
    "trainer_bundle_to_verl_dataproto",
    "run_verl_trainer_adapter_step",
    # Distributed Reward Service
    "RewardWorkerStatus",
    "RewardWorkerInput",
    "RewardWorkerReport",
    "WeaknessBookDelta",
    "ReplayBufferItem",
    "ReplayBufferManifestEntry",
    "DistributedRewardSummary",
    "validate_reward_worker_input_no_leakage",
    "validate_replay_item_no_leakage",
    "sanitize_weakness_delta_for_actor_context",
    "load_reward_worker_input",
    "build_reward_worker_input_from_job_dir",
    "load_reward_artifacts",
    "build_weakness_delta",
    "build_replay_item",
    "run_reward_worker_job_local",
    "run_reward_worker_job",
    "run_reward_worker_job_ray",
    "ReplayBufferWriter",
    "ReplayBufferReader",
    # Phase 3 E2E Schema
    "Phase3E2EBackend",
    "Phase3E2EStageName",
    "Phase3E2EStageStatus",
    "Phase3E2ERunConfig",
    "Phase3E2EStageReport",
    "Phase3E2ERoundReport",
    "Phase3E2ERunReport",
    "create_stage_report",
    "finalize_stage_report",
    # Phase 3 E2E Orchestrator
    "Phase3E2EPaths",
    "plan_output_paths",
    "run_preflight",
    "build_runtime_plan",
    "run_rollout_stage",
    "run_reward_stage",
    "run_replay_stage",
    "run_trainer_stage",
    "run_checkpoint_stage",
    "run_inspection_stage",
    "run_phase3_end_to_end_grpo",
    "write_summary_markdown",
]
