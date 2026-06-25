__version__ = "0.1.0"


def _missing_optional_dependency(symbol_name: str, dependency_name: str, exc: Exception):
    """Build a placeholder for optional exports when dependencies are absent."""

    def _raise_missing_optional_dependency(*args, **kwargs):
        raise ImportError(
            f"{symbol_name} requires optional dependency '{dependency_name}'. "
            "Install the local training dependencies before using this export."
        ) from exc

    _raise_missing_optional_dependency.__name__ = symbol_name
    return _raise_missing_optional_dependency

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
try:
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
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    TrainerFieldSource = _missing_optional_dependency("TrainerFieldSource", "torch", exc)
    VerlTrainerAdapterMode = _missing_optional_dependency("VerlTrainerAdapterMode", "torch", exc)
    TrainerBatchMetadata = _missing_optional_dependency("TrainerBatchMetadata", "torch", exc)
    TrainerReadyBatch = _missing_optional_dependency("TrainerReadyBatch", "torch", exc)
    TrainerTensorBundle = _missing_optional_dependency("TrainerTensorBundle", "torch", exc)
    VerlTrainerAdapterReport = _missing_optional_dependency("VerlTrainerAdapterReport", "torch", exc)
    load_local_dataproto_batch = _missing_optional_dependency("load_local_dataproto_batch", "torch", exc)
    load_torch_tensor_payload = _missing_optional_dependency("load_torch_tensor_payload", "torch", exc)
    validate_trainer_input_no_leakage = _missing_optional_dependency("validate_trainer_input_no_leakage", "torch", exc)
    add_stub_logprob_fields = _missing_optional_dependency("add_stub_logprob_fields", "torch", exc)
    compute_returns = _missing_optional_dependency("compute_returns", "torch", exc)
    build_trainer_ready_batch = _missing_optional_dependency("build_trainer_ready_batch", "torch", exc)
    trainer_bundle_to_verl_dataproto = _missing_optional_dependency("trainer_bundle_to_verl_dataproto", "torch", exc)
    run_verl_trainer_adapter_step = _missing_optional_dependency("run_verl_trainer_adapter_step", "torch", exc)

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

# Phase 4 Milestone 1: Dataset Split and Training Protocol
from gaam_graph.dataset_split_schema import (
    DatasetSplitName,
    DatasetSplitIssueSeverity,
    DatasetSplitIssue,
    DatasetSplitRecord,
    DatasetSplitManifest,
    find_leakage_sensitive_metadata_paths,
    split_counts,
    compute_split_assignment_hash,
)

from gaam_graph.dataset_splitter import (
    load_lme_records_for_split,
    read_record_id_file,
    validate_oracle_graph_record_id,
    create_random_split,
    create_explicit_split,
    validate_split_manifest,
    write_split_manifest,
    load_dataset_split_manifest,
)

from gaam_graph.training_protocol_schema import (
    TrainingProtocolPolicy,
    TrainingProtocolManifest,
    SplitRunRecordReport,
    SplitRunReport,
    get_default_policy_for_split,
)

from gaam_graph.training_protocol import (
    select_split_records,
    build_training_protocol_manifest,
    run_split_smoke,
    aggregate_split_reports,
    write_split_summary_markdown,
)

# Phase 4 Milestone 2: Multi-Case GRPO Rollout and Replay Aggregation
from gaam_graph.phase4_rollout_schema import (
    MultiCaseRolloutStatus,
    MultiCaseRecordRun,
    MultiCaseRolloutManifest,
    AggregatedReplayManifest,
    ActorBatchManifest,
    aggregate_record_status,
    split_counts as multi_case_split_counts,
    compute_aggregate_metrics,
    generate_run_id,
)

from gaam_graph.phase4_replay_aggregation import (
    build_replay_item_key,
    collect_record_replay_items,
    filter_replay_items_for_split,
    detect_duplicate_replay_keys,
    aggregate_replay_buffer,
    read_replay_buffer,
)

from gaam_graph.phase4_batch_export import (
    collect_actor_update_batches,
    merge_actor_update_batches,
    compute_batch_reward_stats,
    check_actor_batch_no_leakage,
    write_actor_batch_manifest,
    export_phase4_grpo_batches,
)

from gaam_graph.phase4_multi_case_rollout import (
    select_records_for_multi_case_rollout,
    build_phase3_config_for_record,
    run_record_rollout,
    run_multi_case_rollout,
    write_multi_case_summary_markdown,
)

# Phase 4 Milestone 3: True GRPO Trainer Step
from gaam_graph.phase4_trainer_schema import (
    Phase4TrainerBackend,
    Phase4TrainerStepConfig,
    Phase4ActorTrainerInput,
    Phase4ActorTrainerReport,
    Phase4TrainerStepManifest,
    CheckpointSelectionReport,
    aggregate_actor_trainer_status,
    compute_trainer_step_metrics,
    generate_trainer_step_run_id,
    sanitize_metric_value,
    sanitize_metrics,
)

from gaam_graph.phase4_trainer_input import (
    load_multi_case_rollout_manifest,
    locate_actor_batch_paths,
    load_actor_batch,
    normalize_actor_batch_samples,
    validate_actor_batch_for_update,
    summarize_actor_trainer_input,
    load_and_validate_actor_inputs,
)

from gaam_graph.phase4_grpo_step import (
    run_actor_grpo_update,
    run_phase4_grpo_trainer_step,
    write_phase4_trainer_step_manifest,
    write_trainer_step_summary_markdown,
)

from gaam_graph.phase4_checkpoint_selection import (
    select_latest_successful_checkpoints,
    select_checkpoints_with_optional_dev_metrics,
    write_checkpoint_selection_report,
    load_checkpoint_selection_report,
)

from gaam_graph.phase4_trainer_inspection import (
    load_trainer_step_manifest,
    check_trainer_step_artifacts,
    check_trainer_step_no_leakage,
    check_trainer_step_metrics,
    check_trainer_step_split_policy,
    inspect_phase4_trainer_step,
    write_trainer_step_inspection_report,
)

# Phase 4 Milestone 4: Multi-Round Adversarial Co-Training
from gaam_graph.phase4_cotraining_schema import (
    Phase4RoundStatus,
    Phase4CotrainingConfig,
    Phase4RoundPlan,
    Phase4RoundReport,
    Phase4CotrainingManifest,
    Phase4EarlyStopReport,
    aggregate_round_status,
    compute_experiment_metrics,
    generate_cotraining_run_id,
)

from gaam_graph.phase4_round_scheduler import (
    select_round_records,
    build_round_plan,
    get_previous_round_report,
)

from gaam_graph.phase4_cotraining_loop import (
    run_phase4_cotraining_loop,
    run_phase4_cotraining_round,
    evaluate_early_stopping,
    write_cotraining_manifest,
    load_cotraining_manifest,
    write_round_summary,
    write_cotraining_summary,
)

from gaam_graph.phase4_cotraining_inspection import (
    load_cotraining_manifest_for_inspection,
    check_cotraining_artifacts,
    check_cotraining_split_policy,
    check_cotraining_no_leakage,
    check_cotraining_checkpoint_consistency,
    check_cotraining_metrics,
    inspect_phase4_cotraining,
    write_cotraining_inspection_report,
)

from gaam_graph.phase4_metrics_export import (
    collect_round_metrics,
    export_cotraining_metrics_csv,
    export_cotraining_metrics_jsonl,
    export_cotraining_metrics,
)

# Phase 4 Milestone 5: Production Training and Evaluation Hardening
from gaam_graph.phase4_experiment_schema import (
    Phase4ProductionExperimentConfig,
    Phase4ProductionExperimentManifest,
    Phase4FinalEvaluationConfig,
    Phase4FinalEvaluationManifest,
    generate_experiment_id,
    generate_eval_id,
)

from gaam_graph.phase4_backend_readiness import (
    Phase4BackendReadinessReport,
    check_python_version,
    check_torch_available,
    check_cuda_available,
    check_transformers_available,
    check_verl_available,
    check_model_path,
    check_output_dir_writable,
    check_phase4_backend_readiness,
)

from gaam_graph.phase4_final_evaluation import (
    run_phase4_final_evaluation,
    write_final_evaluation_manifest,
    load_final_evaluation_manifest,
    write_final_evaluation_summary,
)

from gaam_graph.phase4_baseline_eval import (
    run_phase4_baseline_evaluations,
    run_baseline_evaluation,
)

from gaam_graph.phase4_leakage_audit import (
    LeakageIssue,
    Phase4LeakageAuditReport,
    find_forbidden_fields,
    scan_file_for_leakage,
    run_phase4_leakage_audit,
    write_leakage_audit_report,
    load_leakage_audit_report,
)

from gaam_graph.phase4_experiment_report import (
    collect_experiment_metrics,
    export_cotraining_metrics_csv as export_experiment_cotraining_csv,
    export_cotraining_metrics_jsonl as export_experiment_cotraining_jsonl,
    export_final_metrics_json,
    write_paper_table_markdown,
    write_paper_table_json,
    write_learning_curves_json,
    export_experiment_reports,
)

# Phase 4 Milestone 6: Benchmark Reproducibility and Release
from gaam_graph.phase4_suite_schema import (
    Phase4BenchmarkSuiteConfig,
    Phase4SuiteRunRecord,
    Phase4BenchmarkSuiteManifest,
    Phase4AggregateMetrics,
    aggregate_suite_status,
    generate_suite_id,
)

from gaam_graph.phase4_suite_config import (
    load_json_config,
    load_yaml_config,
    load_suite_config,
    apply_cli_overrides,
    write_resolved_config,
    load_and_resolve_suite_config,
)

from gaam_graph.phase4_suite_runner import (
    build_experiment_config_for_seed,
    should_run_seed,
    run_seed_experiment,
    write_suite_manifest,
    load_suite_manifest,
    run_phase4_benchmark_suite,
)

from gaam_graph.phase4_suite_aggregation import (
    collect_seed_level_metrics,
    compute_aggregate_statistics,
    aggregate_metrics_by_method,
    write_aggregate_metrics_json,
    write_aggregate_metrics_csv,
    write_seed_level_metrics_jsonl,
    aggregate_phase4_benchmark_suite,
)

from gaam_graph.phase4_statistical_analysis import (
    compute_paired_delta,
    bootstrap_confidence_interval,
    extract_metric_values_by_seed,
    compute_phase4_statistical_tests,
    write_statistical_tests_json,
)

from gaam_graph.phase4_suite_reporting import (
    format_metric_value,
    format_metric_with_std,
    write_paper_table_markdown as write_suite_paper_table_markdown,
    write_paper_table_latex,
    write_paper_table_json as write_suite_paper_table_json,
    write_learning_curves_json as write_suite_learning_curves_json,
    write_figure_data_json,
    generate_suite_reports,
)

from gaam_graph.phase4_release_package import (
    compute_file_sha256,
    write_checksums,
    write_environment_report,
    write_artifact_manifest,
    write_readme,
    write_reproduce_script,
    build_phase4_release_package,
)

from gaam_graph.phase4_artifact_verification import (
    Phase4ArtifactVerificationReport,
    check_required_files_exist,
    check_json_files_parse,
    check_checksums_match,
    check_suite_manifest_consistency,
    check_no_test_split_violation,
    check_no_hidden_failed_runs,
    verify_phase4_release_artifacts,
    write_verification_report,
)

from gaam_graph.phase4_remote_scripts import (
    generate_remote_run_script,
    generate_slurm_script,
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
    # Phase 4 Dataset Split Schema
    "DatasetSplitName",
    "DatasetSplitIssueSeverity",
    "DatasetSplitIssue",
    "DatasetSplitRecord",
    "DatasetSplitManifest",
    "find_leakage_sensitive_metadata_paths",
    "split_counts",
    "compute_split_assignment_hash",
    # Dataset Splitter
    "load_lme_records_for_split",
    "read_record_id_file",
    "validate_oracle_graph_record_id",
    "create_random_split",
    "create_explicit_split",
    "validate_split_manifest",
    "write_split_manifest",
    "load_dataset_split_manifest",
    # Training Protocol Schema
    "TrainingProtocolPolicy",
    "TrainingProtocolManifest",
    "SplitRunRecordReport",
    "SplitRunReport",
    "get_default_policy_for_split",
    # Training Protocol
    "select_split_records",
    "build_training_protocol_manifest",
    "run_split_smoke",
    "aggregate_split_reports",
    "write_split_summary_markdown",
    # Phase 4 Rollout Schema
    "MultiCaseRolloutStatus",
    "MultiCaseRecordRun",
    "MultiCaseRolloutManifest",
    "AggregatedReplayManifest",
    "ActorBatchManifest",
    "aggregate_record_status",
    "multi_case_split_counts",
    "compute_aggregate_metrics",
    "generate_run_id",
    # Phase 4 Replay Aggregation
    "build_replay_item_key",
    "collect_record_replay_items",
    "filter_replay_items_for_split",
    "detect_duplicate_replay_keys",
    "aggregate_replay_buffer",
    "read_replay_buffer",
    # Phase 4 Batch Export
    "collect_actor_update_batches",
    "merge_actor_update_batches",
    "compute_batch_reward_stats",
    "check_actor_batch_no_leakage",
    "write_actor_batch_manifest",
    "export_phase4_grpo_batches",
    # Phase 4 Multi-Case Rollout
    "select_records_for_multi_case_rollout",
    "build_phase3_config_for_record",
    "run_record_rollout",
    "run_multi_case_rollout",
    "write_multi_case_summary_markdown",
    # Phase 4 Trainer Schema
    "Phase4TrainerBackend",
    "Phase4TrainerStepConfig",
    "Phase4ActorTrainerInput",
    "Phase4ActorTrainerReport",
    "Phase4TrainerStepManifest",
    "CheckpointSelectionReport",
    "aggregate_actor_trainer_status",
    "compute_trainer_step_metrics",
    "generate_trainer_step_run_id",
    "sanitize_metric_value",
    "sanitize_metrics",
    # Phase 4 Trainer Input
    "load_multi_case_rollout_manifest",
    "locate_actor_batch_paths",
    "load_actor_batch",
    "normalize_actor_batch_samples",
    "validate_actor_batch_for_update",
    "summarize_actor_trainer_input",
    "load_and_validate_actor_inputs",
    # Phase 4 GRPO Step
    "run_actor_grpo_update",
    "run_phase4_grpo_trainer_step",
    "write_phase4_trainer_step_manifest",
    "write_trainer_step_summary_markdown",
    # Phase 4 Checkpoint Selection
    "select_latest_successful_checkpoints",
    "select_checkpoints_with_optional_dev_metrics",
    "write_checkpoint_selection_report",
    "load_checkpoint_selection_report",
    # Phase 4 Trainer Inspection
    "load_trainer_step_manifest",
    "check_trainer_step_artifacts",
    "check_trainer_step_no_leakage",
    "check_trainer_step_metrics",
    "check_trainer_step_split_policy",
    "inspect_phase4_trainer_step",
    "write_trainer_step_inspection_report",
    # Phase 4 Co-Training Schema
    "Phase4RoundStatus",
    "Phase4CotrainingConfig",
    "Phase4RoundPlan",
    "Phase4RoundReport",
    "Phase4CotrainingManifest",
    "Phase4EarlyStopReport",
    "aggregate_round_status",
    "compute_experiment_metrics",
    "generate_cotraining_run_id",
    # Phase 4 Round Scheduler
    "select_round_records",
    "build_round_plan",
    "get_previous_round_report",
    # Phase 4 Co-Training Loop
    "run_phase4_cotraining_loop",
    "run_phase4_cotraining_round",
    "evaluate_early_stopping",
    "write_cotraining_manifest",
    "load_cotraining_manifest",
    "write_round_summary",
    "write_cotraining_summary",
    # Phase 4 Co-Training Inspection
    "load_cotraining_manifest_for_inspection",
    "check_cotraining_artifacts",
    "check_cotraining_split_policy",
    "check_cotraining_no_leakage",
    "check_cotraining_checkpoint_consistency",
    "check_cotraining_metrics",
    "inspect_phase4_cotraining",
    "write_cotraining_inspection_report",
    # Phase 4 Metrics Export
    "collect_round_metrics",
    "export_cotraining_metrics_csv",
    "export_cotraining_metrics_jsonl",
    "export_cotraining_metrics",
    # Phase 4 Experiment Schema
    "Phase4ProductionExperimentConfig",
    "Phase4ProductionExperimentManifest",
    "Phase4FinalEvaluationConfig",
    "Phase4FinalEvaluationManifest",
    "generate_experiment_id",
    "generate_eval_id",
    # Phase 4 Backend Readiness
    "Phase4BackendReadinessReport",
    "check_python_version",
    "check_torch_available",
    "check_cuda_available",
    "check_transformers_available",
    "check_verl_available",
    "check_model_path",
    "check_output_dir_writable",
    "check_phase4_backend_readiness",
    # Phase 4 Final Evaluation
    "run_phase4_final_evaluation",
    "write_final_evaluation_manifest",
    "load_final_evaluation_manifest",
    "write_final_evaluation_summary",
    # Phase 4 Baseline Evaluation
    "run_phase4_baseline_evaluations",
    "run_baseline_evaluation",
    # Phase 4 Leakage Audit
    "LeakageIssue",
    "Phase4LeakageAuditReport",
    "find_forbidden_fields",
    "scan_file_for_leakage",
    "run_phase4_leakage_audit",
    "write_leakage_audit_report",
    "load_leakage_audit_report",
    # Phase 4 Experiment Report
    "collect_experiment_metrics",
    "export_experiment_cotraining_csv",
    "export_experiment_cotraining_jsonl",
    "export_final_metrics_json",
    "write_paper_table_markdown",
    "write_paper_table_json",
    "write_learning_curves_json",
    "export_experiment_reports",
    # Phase 4 Suite Schema
    "Phase4BenchmarkSuiteConfig",
    "Phase4SuiteRunRecord",
    "Phase4BenchmarkSuiteManifest",
    "Phase4AggregateMetrics",
    "aggregate_suite_status",
    "generate_suite_id",
    # Phase 4 Suite Config
    "load_json_config",
    "load_yaml_config",
    "load_suite_config",
    "apply_cli_overrides",
    "write_resolved_config",
    "load_and_resolve_suite_config",
    # Phase 4 Suite Runner
    "build_experiment_config_for_seed",
    "should_run_seed",
    "run_seed_experiment",
    "write_suite_manifest",
    "load_suite_manifest",
    "run_phase4_benchmark_suite",
    # Phase 4 Suite Aggregation
    "collect_seed_level_metrics",
    "compute_aggregate_statistics",
    "aggregate_metrics_by_method",
    "write_aggregate_metrics_json",
    "write_aggregate_metrics_csv",
    "write_seed_level_metrics_jsonl",
    "aggregate_phase4_benchmark_suite",
    # Phase 4 Statistical Analysis
    "compute_paired_delta",
    "bootstrap_confidence_interval",
    "extract_metric_values_by_seed",
    "compute_phase4_statistical_tests",
    "write_statistical_tests_json",
    # Phase 4 Suite Reporting
    "format_metric_value",
    "format_metric_with_std",
    "write_suite_paper_table_markdown",
    "write_paper_table_latex",
    "write_suite_paper_table_json",
    "write_suite_learning_curves_json",
    "write_figure_data_json",
    "generate_suite_reports",
    # Phase 4 Release Package
    "compute_file_sha256",
    "write_checksums",
    "write_environment_report",
    "write_artifact_manifest",
    "write_readme",
    "write_reproduce_script",
    "build_phase4_release_package",
    # Phase 4 Artifact Verification
    "Phase4ArtifactVerificationReport",
    "check_required_files_exist",
    "check_json_files_parse",
    "check_checksums_match",
    "check_suite_manifest_consistency",
    "check_no_test_split_violation",
    "check_no_hidden_failed_runs",
    "verify_phase4_release_artifacts",
    "write_verification_report",
    # Phase 4 Remote Scripts
    "generate_remote_run_script",
    "generate_slurm_script",
]
