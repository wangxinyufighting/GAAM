#!/usr/bin/env bash
set -euo pipefail

# Native GAAM dual-agent co-training with Code-A1 vendored VERL.
#
# This script launches GAAMDualRayTrainer. Each actor step uses the real native
# VERL GRPO kernel through scripts/run_native_verl_grpo_training.sh.
#
# Example:
#   MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   SPLIT_MANIFEST=outputs/splits/longmemeval_s.phase4_split.seed0.json \
#   MEMORY_NUM_GPUS=2 \
#   QUESTION_NUM_GPUS=2 \
#   ROUNDS=1 \
#   bash scripts/run_native_verl_dual_cotraining.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_PATH="${INPUT_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"
ORACLE_GRAPH_DIR="${ORACLE_GRAPH_DIR:-outputs/longmemeval_s_graph}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/native_verl_dual_cotraining}"

MEMORY_MODEL_PATH="${MEMORY_MODEL_PATH:-}"
QUESTION_MODEL_PATH="${QUESTION_MODEL_PATH:-}"

ROUNDS="${ROUNDS:-1}"
ORDER="${ORDER:-question_agent,memory_builder}"
MEMORY_NUM_GPUS="${MEMORY_NUM_GPUS:-${NUM_GPUS:-1}}"
QUESTION_NUM_GPUS="${QUESTION_NUM_GPUS:-${NUM_GPUS:-1}}"
NNODES="${NNODES:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
MEMORY_ROLLOUT_N="${MEMORY_ROLLOUT_N:-}"
QUESTION_ROLLOUT_N="${QUESTION_ROLLOUT_N:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MEMORY_TRAIN_BATCH_SIZE="${MEMORY_TRAIN_BATCH_SIZE:-}"
QUESTION_TRAIN_BATCH_SIZE="${QUESTION_TRAIN_BATCH_SIZE:-}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MEMORY_MAX_PROMPT_LENGTH="${MEMORY_MAX_PROMPT_LENGTH:-}"
QUESTION_MAX_PROMPT_LENGTH="${QUESTION_MAX_PROMPT_LENGTH:-}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MEMORY_MAX_RESPONSE_LENGTH="${MEMORY_MAX_RESPONSE_LENGTH:-}"
QUESTION_MAX_RESPONSE_LENGTH="${QUESTION_MAX_RESPONSE_LENGTH:-}"
MAX_HISTORY_CHARS="${MAX_HISTORY_CHARS:-16000}"
MAX_ORACLE_CHARS="${MAX_ORACLE_CHARS:-12000}"
MAX_RECORDS_PER_SPLIT="${MAX_RECORDS_PER_SPLIT:-}"
TOTAL_EPOCHS_PER_ACTOR_STEP="${TOTAL_EPOCHS_PER_ACTOR_STEP:-1}"
TOTAL_TRAINING_STEPS_PER_ACTOR_STEP="${TOTAL_TRAINING_STEPS_PER_ACTOR_STEP:-}"
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:-1}"
LR="${LR:-1e-6}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
LOGGER="${LOGGER:-console}"
SAVE_HF_MODEL="${SAVE_HF_MODEL:-1}"
DUAL_DRY_RUN="${DUAL_DRY_RUN:-0}"

if [[ -z "${CODE_A1_ROOT:-}" ]]; then
  CODE_A1_ROOT="${REPO_DIR}/Code-A1/Code-A1"
fi

if [[ -z "${MEMORY_MODEL_PATH}" ]]; then
  echo "MEMORY_MODEL_PATH is required." >&2
  exit 1
fi

if [[ -z "${QUESTION_MODEL_PATH}" ]]; then
  echo "QUESTION_MODEL_PATH is required." >&2
  exit 1
fi

if [[ ! -d "${CODE_A1_ROOT}/verl/verl" ]]; then
  echo "Code-A1 vendored VERL package not found at ${CODE_A1_ROOT}/verl/verl" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${CODE_A1_ROOT}:${CODE_A1_ROOT}/verl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"

ARGS=(
  scripts/run_native_verl_dual_cotraining.py
  --root_dir "${ROOT_DIR}"
  --code_a1_root "${CODE_A1_ROOT}"
  --python_bin "${PYTHON_BIN}"
  --input "${INPUT_PATH}"
  --oracle_graph_dir "${ORACLE_GRAPH_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --rounds "${ROUNDS}"
  --order "${ORDER}"
  --memory_model_path "${MEMORY_MODEL_PATH}"
  --question_model_path "${QUESTION_MODEL_PATH}"
  --memory_num_gpus "${MEMORY_NUM_GPUS}"
  --question_num_gpus "${QUESTION_NUM_GPUS}"
  --nnodes "${NNODES}"
  --rollout_n "${ROLLOUT_N}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --ppo_mini_batch_size "${PPO_MINI_BATCH_SIZE}"
  --ppo_micro_batch_size_per_gpu "${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_response_length "${MAX_RESPONSE_LENGTH}"
  --max_history_chars "${MAX_HISTORY_CHARS}"
  --max_oracle_chars "${MAX_ORACLE_CHARS}"
  --total_epochs_per_actor_step "${TOTAL_EPOCHS_PER_ACTOR_STEP}"
  --save_freq "${SAVE_FREQ}"
  --test_freq "${TEST_FREQ}"
  --lr "${LR}"
  --rollout_tp_size "${ROLLOUT_TP_SIZE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --logger "${LOGGER}"
)

if [[ -n "${SPLIT_MANIFEST}" ]]; then
  ARGS+=(--split_manifest "${SPLIT_MANIFEST}")
fi

if [[ -n "${MEMORY_ROLLOUT_N}" ]]; then
  ARGS+=(--memory_rollout_n "${MEMORY_ROLLOUT_N}")
fi

if [[ -n "${QUESTION_ROLLOUT_N}" ]]; then
  ARGS+=(--question_rollout_n "${QUESTION_ROLLOUT_N}")
fi

if [[ -n "${MEMORY_TRAIN_BATCH_SIZE}" ]]; then
  ARGS+=(--memory_train_batch_size "${MEMORY_TRAIN_BATCH_SIZE}")
fi

if [[ -n "${QUESTION_TRAIN_BATCH_SIZE}" ]]; then
  ARGS+=(--question_train_batch_size "${QUESTION_TRAIN_BATCH_SIZE}")
fi

if [[ -n "${MEMORY_MAX_PROMPT_LENGTH}" ]]; then
  ARGS+=(--memory_max_prompt_length "${MEMORY_MAX_PROMPT_LENGTH}")
fi

if [[ -n "${QUESTION_MAX_PROMPT_LENGTH}" ]]; then
  ARGS+=(--question_max_prompt_length "${QUESTION_MAX_PROMPT_LENGTH}")
fi

if [[ -n "${MEMORY_MAX_RESPONSE_LENGTH}" ]]; then
  ARGS+=(--memory_max_response_length "${MEMORY_MAX_RESPONSE_LENGTH}")
fi

if [[ -n "${QUESTION_MAX_RESPONSE_LENGTH}" ]]; then
  ARGS+=(--question_max_response_length "${QUESTION_MAX_RESPONSE_LENGTH}")
fi

if [[ -n "${MAX_RECORDS_PER_SPLIT}" ]]; then
  ARGS+=(--max_records_per_split "${MAX_RECORDS_PER_SPLIT}")
fi

if [[ -n "${TOTAL_TRAINING_STEPS_PER_ACTOR_STEP}" ]]; then
  ARGS+=(--total_training_steps_per_actor_step "${TOTAL_TRAINING_STEPS_PER_ACTOR_STEP}")
fi

if [[ "${SAVE_HF_MODEL}" == "0" || "${SAVE_HF_MODEL}" == "False" || "${SAVE_HF_MODEL}" == "false" ]]; then
  ARGS+=(--no_save_hf_model)
fi

if [[ "${DUAL_DRY_RUN}" == "1" || "${DUAL_DRY_RUN}" == "True" || "${DUAL_DRY_RUN}" == "true" ]]; then
  ARGS+=(--dry_run)
fi

echo "== GAAM native dual VERL co-training =="
echo "ROOT_DIR=${ROOT_DIR}"
echo "CODE_A1_ROOT=${CODE_A1_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ROUNDS=${ROUNDS}"
echo "ORDER=${ORDER}"
echo "MEMORY_NUM_GPUS=${MEMORY_NUM_GPUS}"
echo "QUESTION_NUM_GPUS=${QUESTION_NUM_GPUS}"
echo "DUAL_DRY_RUN=${DUAL_DRY_RUN}"

"${PYTHON_BIN}" "${ARGS[@]}"
