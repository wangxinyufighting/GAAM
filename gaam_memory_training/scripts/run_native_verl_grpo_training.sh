#!/usr/bin/env bash
set -euo pipefail

# Native Code-A1/VERL GRPO entrypoint for one GAAM actor.
#
# This script is intentionally separate from the Phase 4 benchmark-suite wrapper:
# it runs the real vendored VERL trainer:
#   python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...
#
# Example:
#   ACTOR_ROLE=memory_builder \
#   MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   NUM_GPUS=2 \
#   SPLIT_MANIFEST=outputs/splits/longmemeval_s.phase4_split.seed0.json \
#   bash scripts/run_native_verl_grpo_training.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_PATH="${INPUT_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"
ORACLE_GRAPH_DIR="${ORACLE_GRAPH_DIR:-outputs/longmemeval_s_graph}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"
ACTOR_ROLE="${ACTOR_ROLE:-memory_builder}"
MODEL_PATH="${MODEL_PATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/native_verl_grpo}"
DATASET_DIR="${DATASET_DIR:-${OUTPUT_ROOT}/datasets/${ACTOR_ROLE}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_ROOT}/checkpoints/${ACTOR_ROLE}}"

NUM_GPUS="${NUM_GPUS:-1}"
NNODES="${NNODES:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_HISTORY_CHARS="${MAX_HISTORY_CHARS:-16000}"
MAX_ORACLE_CHARS="${MAX_ORACLE_CHARS:-12000}"
QUESTIONS_PER_CASE="${QUESTIONS_PER_CASE:-8}"
MEMORY_INPUT_MODE="${MEMORY_INPUT_MODE:-full}"
MEMORY_SESSION_CHUNK_SIZE="${MEMORY_SESSION_CHUNK_SIZE:-4}"
MAX_MEMORY_CHUNK_CHARS="${MAX_MEMORY_CHUNK_CHARS:-}"
MAX_PREVIOUS_MEMORY_CHARS="${MAX_PREVIOUS_MEMORY_CHARS:-6000}"
ALLOW_STATIC_INCREMENTAL_SCAFFOLD="${ALLOW_STATIC_INCREMENTAL_SCAFFOLD:-0}"
MAX_RECORDS_PER_SPLIT="${MAX_RECORDS_PER_SPLIT:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:-1}"
LR="${LR:-1e-6}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
LOGGER="${LOGGER:-console}"
PROJECT_NAME="${PROJECT_NAME:-gaam_native_verl_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${ACTOR_ROLE}_qwen3_0_6b}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
SAVE_HF_MODEL="${SAVE_HF_MODEL:-True}"
REQUIRE_HF_CHECKPOINT="${REQUIRE_HF_CHECKPOINT:-1}"

if [[ "${ACTOR_ROLE}" != "memory_builder" && "${ACTOR_ROLE}" != "question_agent" ]]; then
  echo "ACTOR_ROLE must be memory_builder or question_agent, got: ${ACTOR_ROLE}" >&2
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  if [[ "${ACTOR_ROLE}" == "memory_builder" ]]; then
    MODEL_PATH="${MEMORY_MODEL_PATH:-}"
  else
    MODEL_PATH="${QUESTION_MODEL_PATH:-}"
  fi
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required. Set MODEL_PATH, MEMORY_MODEL_PATH, or QUESTION_MODEL_PATH." >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist or is not a directory: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ -z "${CODE_A1_ROOT:-}" ]]; then
  CODE_A1_ROOT="${REPO_DIR}/Code-A1/Code-A1"
fi
VERL_ROOT="${VERL_ROOT:-${CODE_A1_ROOT}/verl}"

if [[ ! -d "${VERL_ROOT}/verl" ]]; then
  echo "Vendored VERL package not found at ${VERL_ROOT}/verl" >&2
  echo "Set CODE_A1_ROOT or VERL_ROOT to the Code-A1 checkout." >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}:${CODE_A1_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"

mkdir -p "${DATASET_DIR}" "${CHECKPOINT_DIR}"

echo "== GAAM native Code-A1/VERL GRPO training =="
echo "ROOT_DIR=${ROOT_DIR}"
echo "CODE_A1_ROOT=${CODE_A1_ROOT}"
echo "VERL_ROOT=${VERL_ROOT}"
echo "ACTOR_ROLE=${ACTOR_ROLE}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "INPUT_PATH=${INPUT_PATH}"
echo "ORACLE_GRAPH_DIR=${ORACLE_GRAPH_DIR}"
echo "SPLIT_MANIFEST=${SPLIT_MANIFEST:-<auto-from-oracle-graphs>}"
echo "DATASET_DIR=${DATASET_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "ROLLOUT_N=${ROLLOUT_N}"
echo "MEMORY_INPUT_MODE=${MEMORY_INPUT_MODE}"
echo "SAVE_HF_MODEL=${SAVE_HF_MODEL}"
echo "REQUIRE_HF_CHECKPOINT=${REQUIRE_HF_CHECKPOINT}"

EXPORT_ARGS=(
  scripts/export_native_verl_grpo_dataset.py
  --input "${INPUT_PATH}"
  --oracle_graph_dir "${ORACLE_GRAPH_DIR}"
  --output_dir "${DATASET_DIR}"
  --actor_role "${ACTOR_ROLE}"
  --max_history_chars "${MAX_HISTORY_CHARS}"
  --max_oracle_chars "${MAX_ORACLE_CHARS}"
  --questions_per_case "${QUESTIONS_PER_CASE}"
  --memory_input_mode "${MEMORY_INPUT_MODE}"
  --memory_session_chunk_size "${MEMORY_SESSION_CHUNK_SIZE}"
  --max_previous_memory_chars "${MAX_PREVIOUS_MEMORY_CHARS}"
)

if [[ -n "${MAX_MEMORY_CHUNK_CHARS}" ]]; then
  EXPORT_ARGS+=(--max_memory_chunk_chars "${MAX_MEMORY_CHUNK_CHARS}")
fi

if [[ "${ALLOW_STATIC_INCREMENTAL_SCAFFOLD}" == "1" || "${ALLOW_STATIC_INCREMENTAL_SCAFFOLD}" == "True" || "${ALLOW_STATIC_INCREMENTAL_SCAFFOLD}" == "true" ]]; then
  EXPORT_ARGS+=(--allow_static_incremental_scaffold)
fi

if [[ -n "${SPLIT_MANIFEST}" ]]; then
  EXPORT_ARGS+=(--split_manifest "${SPLIT_MANIFEST}")
fi

if [[ -n "${MAX_RECORDS_PER_SPLIT}" ]]; then
  EXPORT_ARGS+=(--max_records_per_split "${MAX_RECORDS_PER_SPLIT}")
fi

"${PYTHON_BIN}" "${EXPORT_ARGS[@]}"

TRAIN_FILE="${DATASET_DIR}/${ACTOR_ROLE}.train.parquet"
VAL_FILE="${DATASET_DIR}/${ACTOR_ROLE}.val.parquet"
DATASET_MANIFEST="${DATASET_DIR}/${ACTOR_ROLE}.dataset_manifest.json"
REWARD_FILE="${ROOT_DIR}/gaam_graph/verl_gaam_reward.py"

NUM_TRAIN_ROWS="$("${PYTHON_BIN}" - "${DATASET_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(manifest.get("num_train_rows", 0)))
PY
)"

if [[ "${NUM_TRAIN_ROWS}" -lt 1 ]]; then
  echo "Native VERL dataset has zero train rows: ${DATASET_MANIFEST}" >&2
  exit 1
fi

if [[ "${TRAIN_BATCH_SIZE}" -gt "${NUM_TRAIN_ROWS}" ]]; then
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} is larger than num_train_rows=${NUM_TRAIN_ROWS}; capping to ${NUM_TRAIN_ROWS} to avoid an empty VERL dataloader."
  TRAIN_BATCH_SIZE="${NUM_TRAIN_ROWS}"
fi

if [[ "${PPO_MINI_BATCH_SIZE}" -gt "${TRAIN_BATCH_SIZE}" ]]; then
  echo "PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} is larger than TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}; capping to ${TRAIN_BATCH_SIZE}."
  PPO_MINI_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
fi

if [[ "${SAVE_HF_MODEL}" == "1" || "${SAVE_HF_MODEL}" == "True" || "${SAVE_HF_MODEL}" == "true" ]]; then
  CHECKPOINT_CONTENTS="['model','hf_model','optimizer','extra']"
else
  CHECKPOINT_CONTENTS="['model','optimizer','extra']"
fi

VERL_ARGS=(
  -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=False
  data.truncation=left
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.lora_rank=0
  actor_rollout_ref.model.lora_alpha=0
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr="${LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_CONTENTS}"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
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

echo "== Launching native VERL GRPO =="
echo "${PYTHON_BIN} ${VERL_ARGS[*]}"
"${PYTHON_BIN}" "${VERL_ARGS[@]}"

if [[ "${REQUIRE_HF_CHECKPOINT}" == "1" || "${REQUIRE_HF_CHECKPOINT}" == "True" || "${REQUIRE_HF_CHECKPOINT}" == "true" ]]; then
  if [[ "${SAVE_HF_MODEL}" == "0" || "${SAVE_HF_MODEL}" == "False" || "${SAVE_HF_MODEL}" == "false" ]]; then
    echo "REQUIRE_HF_CHECKPOINT=1 conflicts with SAVE_HF_MODEL=${SAVE_HF_MODEL}." >&2
    echo "Enable SAVE_HF_MODEL or set REQUIRE_HF_CHECKPOINT=0 only for smoke tests." >&2
    exit 1
  fi

  CHECKPOINT_FOUND="$("${PYTHON_BIN}" - "${CHECKPOINT_DIR}" <<'PY'
from pathlib import Path
import sys

checkpoint_dir = Path(sys.argv[1])
candidates = []
for name in ("huggingface", "hf_model"):
    candidates.extend(path for path in checkpoint_dir.rglob(name) if path.is_dir())
print("1" if candidates else "0")
PY
)"
  if [[ "${CHECKPOINT_FOUND}" != "1" ]]; then
    echo "Native VERL finished without writing a Hugging Face checkpoint under: ${CHECKPOINT_DIR}" >&2
    echo "This is treated as a failed training step because downstream rounds need updated full-model weights." >&2
    exit 1
  fi
fi

echo "== Native VERL GRPO finished =="
echo "Checkpoints: ${CHECKPOINT_DIR}"
