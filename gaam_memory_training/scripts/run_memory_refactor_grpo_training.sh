#!/usr/bin/env bash
set -euo pipefail

# Production GRPO startup script for the GAAM Memory Refactoring Policy.
#
# Required:
#   INPUT_PATH=/path/to/memory_refactor_samples.jsonl
#   MODEL_PATH=/path/to/hf_model
#   CODE_A1_ROOT=/path/to/Code-A1/Code-A1
#
# Example:
#   INPUT_PATH=data/memory_refactor/train_samples.jsonl \
#   MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   CODE_A1_ROOT=/workspace/Code-A1/Code-A1 \
#   NUM_GPUS=2 \
#   bash scripts/run_memory_refactor_grpo_training.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_PATH="${INPUT_PATH:-data/memory_refactor/train_samples.jsonl}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/memory_refactor_grpo}"
DATASET_DIR="${DATASET_DIR:-${OUTPUT_ROOT}/dataset}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_ROOT}/checkpoints}"

NUM_GPUS="${NUM_GPUS:-1}"
NNODES="${NNODES:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:-1}"
LR="${LR:-1e-6}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
LOGGER="${LOGGER:-console}"
PROJECT_NAME="${PROJECT_NAME:-gaam_memory_refactor_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-memory_refactor_policy}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
MAX_RECORDS_PER_SPLIT="${MAX_RECORDS_PER_SPLIT:-}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 1
fi

if [[ -z "${CODE_A1_ROOT:-}" ]]; then
  CODE_A1_ROOT="${REPO_DIR}/Code-A1/Code-A1"
fi
VERL_ROOT="${VERL_ROOT:-${CODE_A1_ROOT}/verl}"

export PYTHONPATH="${ROOT_DIR}:${CODE_A1_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"

mkdir -p "${DATASET_DIR}" "${CHECKPOINT_DIR}"

EXPORT_ARGS=(
  scripts/export_memory_refactor_grpo_dataset.py
  --input "${INPUT_PATH}"
  --output_dir "${DATASET_DIR}"
)

if [[ -n "${MAX_RECORDS_PER_SPLIT}" ]]; then
  EXPORT_ARGS+=(--max_records_per_split "${MAX_RECORDS_PER_SPLIT}")
fi

echo "== Exporting Memory Refactor GRPO dataset =="
"${PYTHON_BIN}" "${EXPORT_ARGS[@]}"

TRAIN_FILE="${DATASET_DIR}/memory_refactor.train.parquet"
VAL_FILE="${DATASET_DIR}/memory_refactor.val.parquet"
MANIFEST="${DATASET_DIR}/memory_refactor.dataset_manifest.json"
REWARD_FILE="${ROOT_DIR}/gaam_graph/verl_gaam_reward.py"

NUM_TRAIN_ROWS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_train_rows"])' "${MANIFEST}")"
if [[ "${TRAIN_BATCH_SIZE}" -gt "${NUM_TRAIN_ROWS}" ]]; then
  TRAIN_BATCH_SIZE="${NUM_TRAIN_ROWS}"
fi
if [[ "${PPO_MINI_BATCH_SIZE}" -gt "${TRAIN_BATCH_SIZE}" ]]; then
  PPO_MINI_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
fi

VERL_ARGS=(
  -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=True
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=False
  data.truncation=left
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr="${LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.clip_ratio=0.2
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.temperature=0.8
  actor_rollout_ref.rollout.top_p=0.95
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  critic.model.path="${MODEL_PATH}"
  critic.optim.lr=1e-5
  critic.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  critic.model.fsdp_config.param_offload=True
  critic.model.fsdp_config.optimizer_offload=True
  algorithm.use_kl_in_reward=False
  trainer.critic_warmup=0
  trainer.logger="['${LOGGER}']"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.nnodes="${NNODES}"
  trainer.n_gpus_per_node="${NUM_GPUS}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.val_before_train="${VAL_BEFORE_TRAIN}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.device=cuda
  custom_reward_function.path="${REWARD_FILE}"
  custom_reward_function.name=compute_score
)

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  VERL_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

echo "== Launching Memory Refactor GRPO training =="
echo "dataset_manifest=${MANIFEST}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
"${PYTHON_BIN}" "${VERL_ARGS[@]}"
