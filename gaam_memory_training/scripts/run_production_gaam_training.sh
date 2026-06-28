#!/usr/bin/env bash
set -euo pipefail

# GAAM production training entrypoint.
#
# This script intentionally uses the native Code-A1/VERL dual-agent GRPO path:
#   scripts/run_native_verl_dual_cotraining.sh
#
# It does NOT use the old Phase4 benchmark-suite MVP trainer path.
#
# Typical usage on a GPU server:
#
#   MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   SPLIT_MANIFEST=outputs/splits/longmemeval_s.phase4_split.seed0.json \
#   GAAM_REWARD_JUDGE_ENABLED=1 \
#   GAAM_REWARD_JUDGE_BASE_URL=https://api.deepseek.com \
#   GAAM_REWARD_JUDGE_API_KEY="${DEEPSEEK_API_KEY}" \
#   bash scripts/run_production_gaam_training.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Python executable inside the training environment.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Path to the raw LongMemEval-style records.
INPUT_PATH="${INPUT_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"

# Directory containing prebuilt oracle graph files: <record_id>.graph.json.
ORACLE_GRAPH_DIR="${ORACLE_GRAPH_DIR:-outputs/longmemeval_s_graph}"

# Dataset split manifest. It should contain train/dev/test record IDs and must not
# contain benchmark question or answer text.
SPLIT_MANIFEST="${SPLIT_MANIFEST:-outputs/splits/longmemeval_s.phase4_split.seed0.json}"

# Output directory for native dual-agent training artifacts.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/production_native_verl_dual_cotraining}"

# Initial trainable Memory Builder model path. Required.
MEMORY_MODEL_PATH="${MEMORY_MODEL_PATH:-}"

# Initial trainable Question Agent model path. Required.
QUESTION_MODEL_PATH="${QUESTION_MODEL_PATH:-}"

# Code-A1 checkout root. The script expects ${CODE_A1_ROOT}/verl/verl to exist.
CODE_A1_ROOT="${CODE_A1_ROOT:-$(cd "${ROOT_DIR}/.." && pwd)/Code-A1/Code-A1}"

# Number of adversarial co-training rounds. One round trains the actors in ORDER.
ROUNDS="${ROUNDS:-1}"

# Actor update order per round. Use question_agent first to refresh adversarial
# questions before Memory Builder learns from them.
ORDER="${ORDER:-question_agent,memory_builder}"

# Number of GPUs assigned to Memory Builder's native VERL process.
MEMORY_NUM_GPUS="${MEMORY_NUM_GPUS:-${NUM_GPUS:-1}}"

# Number of GPUs assigned to Question Agent's native VERL process.
QUESTION_NUM_GPUS="${QUESTION_NUM_GPUS:-${NUM_GPUS:-1}}"

# Number of rollouts sampled per prompt by VERL. GRPO needs multiple samples per
# group; 4 is a small smoke value, 8+ is more meaningful.
ROLLOUT_N="${ROLLOUT_N:-4}"

# Global train batch size consumed by VERL for each actor step.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"

# PPO/GRPO mini-batch size. Must be <= TRAIN_BATCH_SIZE.
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"

# Micro-batch size per GPU. Lower this if CUDA memory is tight.
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"

# Max tokenized prompt length for VERL. Increase if raw histories/oracle digests
# are truncated too aggressively; lower if memory is insufficient.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"

# Max generated response length. Memory JSON may need more than question JSON.
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"

# Max raw-history characters used in full Memory Builder prompts.
MAX_HISTORY_CHARS="${MAX_HISTORY_CHARS:-16000}"

# Max oracle digest characters used in Question Agent prompts.
MAX_ORACLE_CHARS="${MAX_ORACLE_CHARS:-12000}"

# Exact target number of questions generated per case by Question Agent.
QUESTIONS_PER_CASE="${QUESTIONS_PER_CASE:-8}"

# Memory input mode for native parquet training. For production native VERL, keep
# this as full. The static incremental parquet mode is blocked by default because
# real incremental memory requires stateful rollout.
MEMORY_INPUT_MODE="${MEMORY_INPUT_MODE:-full}"

# Limit records per split for debugging. Leave empty for full split.
MAX_RECORDS_PER_SPLIT="${MAX_RECORDS_PER_SPLIT:-}"

# Number of native VERL training epochs per actor step.
TOTAL_EPOCHS_PER_ACTOR_STEP="${TOTAL_EPOCHS_PER_ACTOR_STEP:-1}"

# Optional hard cap for native VERL training steps per actor step. Useful for
# smoke tests; leave empty for epoch-driven training.
TOTAL_TRAINING_STEPS_PER_ACTOR_STEP="${TOTAL_TRAINING_STEPS_PER_ACTOR_STEP:-}"

# Native VERL checkpoint save frequency.
SAVE_FREQ="${SAVE_FREQ:-1}"

# Native VERL validation/test frequency.
TEST_FREQ="${TEST_FREQ:-1}"

# Actor learning rate.
LR="${LR:-1e-6}"

# Tensor parallel size for vLLM rollout workers.
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"

# Fraction of GPU memory vLLM may use.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"

# Logger backend for VERL. Use console for simple server runs.
LOGGER="${LOGGER:-console}"

# Save Hugging Face model directories in checkpoints. Keep enabled for practical
# checkpoint reuse.
SAVE_HF_MODEL="${SAVE_HF_MODEL:-1}"

# Require each native VERL actor step to write a real full-model Hugging Face
# checkpoint. Keep this enabled for production; disable only for smoke tests.
REQUIRE_HF_CHECKPOINT="${REQUIRE_HF_CHECKPOINT:-1}"

# LLM-as-judge reward switch. When enabled, gaam_graph/verl_gaam_reward.py calls
# an OpenAI-compatible API in addition to deterministic reward checks.
export GAAM_REWARD_JUDGE_ENABLED="${GAAM_REWARD_JUDGE_ENABLED:-0}"

# OpenAI-compatible judge endpoint, for example DeepSeek or a local vLLM server.
export GAAM_REWARD_JUDGE_BASE_URL="${GAAM_REWARD_JUDGE_BASE_URL:-https://api.deepseek.com}"

# Judge API key. Required when GAAM_REWARD_JUDGE_ENABLED=1.
export GAAM_REWARD_JUDGE_API_KEY="${GAAM_REWARD_JUDGE_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}"

# Judge model name.
export GAAM_REWARD_JUDGE_MODEL="${GAAM_REWARD_JUDGE_MODEL:-deepseek-v4-flash}"

# If 1, missing/failed judge calls force reward to 0. If 0, heuristic reward is
# still used and judge failure is recorded in reward details.
export GAAM_REWARD_JUDGE_REQUIRED="${GAAM_REWARD_JUDGE_REQUIRED:-0}"

# Verify native dual training artifacts after the training script exits.
# Keep enabled for production; set to 0 only if you want to inspect partial
# artifacts after a failed low-level VERL run.
VERIFY_AFTER_TRAINING="${VERIFY_AFTER_TRAINING:-1}"

if [[ -z "${MEMORY_MODEL_PATH}" ]]; then
  echo "MEMORY_MODEL_PATH is required." >&2
  exit 1
fi

if [[ -z "${QUESTION_MODEL_PATH}" ]]; then
  echo "QUESTION_MODEL_PATH is required." >&2
  exit 1
fi

if [[ "${MEMORY_INPUT_MODE}" == "incremental" && "${ALLOW_STATIC_INCREMENTAL_SCAFFOLD:-0}" != "1" ]]; then
  echo "MEMORY_INPUT_MODE=incremental is not a production native parquet mode." >&2
  echo "Use scripts/run_stateful_incremental_memory_builder.py for real stateful incremental rollout," >&2
  echo "or set ALLOW_STATIC_INCREMENTAL_SCAFFOLD=1 only for smoke tests." >&2
  exit 1
fi

echo "== GAAM production native VERL dual training =="
echo "INPUT_PATH=${INPUT_PATH}"
echo "ORACLE_GRAPH_DIR=${ORACLE_GRAPH_DIR}"
echo "SPLIT_MANIFEST=${SPLIT_MANIFEST}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ROUNDS=${ROUNDS}"
echo "ORDER=${ORDER}"
echo "MEMORY_INPUT_MODE=${MEMORY_INPUT_MODE}"
echo "QUESTIONS_PER_CASE=${QUESTIONS_PER_CASE}"
echo "SAVE_HF_MODEL=${SAVE_HF_MODEL}"
echo "REQUIRE_HF_CHECKPOINT=${REQUIRE_HF_CHECKPOINT}"
echo "GAAM_REWARD_JUDGE_ENABLED=${GAAM_REWARD_JUDGE_ENABLED}"
echo "GAAM_REWARD_JUDGE_BASE_URL=${GAAM_REWARD_JUDGE_BASE_URL}"
echo "GAAM_REWARD_JUDGE_MODEL=${GAAM_REWARD_JUDGE_MODEL}"
echo "VERIFY_AFTER_TRAINING=${VERIFY_AFTER_TRAINING}"

READINESS_ARGS=(
  scripts/check_production_training_readiness.py
  --input "${INPUT_PATH}"
  --oracle_graph_dir "${ORACLE_GRAPH_DIR}"
  --split_manifest "${SPLIT_MANIFEST}"
  --memory_model_path "${MEMORY_MODEL_PATH}"
  --question_model_path "${QUESTION_MODEL_PATH}"
  --code_a1_root "${CODE_A1_ROOT}"
  --memory_input_mode "${MEMORY_INPUT_MODE}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --questions_per_case "${QUESTIONS_PER_CASE}"
  --reward_judge_enabled "${GAAM_REWARD_JUDGE_ENABLED}"
  --reward_judge_api_key "${GAAM_REWARD_JUDGE_API_KEY}"
)

if [[ "${ALLOW_STATIC_INCREMENTAL_SCAFFOLD:-0}" == "1" || "${ALLOW_STATIC_INCREMENTAL_SCAFFOLD:-0}" == "True" || "${ALLOW_STATIC_INCREMENTAL_SCAFFOLD:-0}" == "true" ]]; then
  READINESS_ARGS+=(--allow_static_incremental_scaffold)
fi

echo "== Checking production training readiness =="
"${PYTHON_BIN}" "${READINESS_ARGS[@]}"

TRAINING_STATUS=0
PYTHON_BIN="${PYTHON_BIN}" \
INPUT_PATH="${INPUT_PATH}" \
ORACLE_GRAPH_DIR="${ORACLE_GRAPH_DIR}" \
SPLIT_MANIFEST="${SPLIT_MANIFEST}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MEMORY_MODEL_PATH="${MEMORY_MODEL_PATH}" \
QUESTION_MODEL_PATH="${QUESTION_MODEL_PATH}" \
CODE_A1_ROOT="${CODE_A1_ROOT}" \
ROUNDS="${ROUNDS}" \
ORDER="${ORDER}" \
MEMORY_NUM_GPUS="${MEMORY_NUM_GPUS}" \
QUESTION_NUM_GPUS="${QUESTION_NUM_GPUS}" \
ROLLOUT_N="${ROLLOUT_N}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
MAX_HISTORY_CHARS="${MAX_HISTORY_CHARS}" \
MAX_ORACLE_CHARS="${MAX_ORACLE_CHARS}" \
QUESTIONS_PER_CASE="${QUESTIONS_PER_CASE}" \
MEMORY_INPUT_MODE="${MEMORY_INPUT_MODE}" \
MAX_RECORDS_PER_SPLIT="${MAX_RECORDS_PER_SPLIT}" \
TOTAL_EPOCHS_PER_ACTOR_STEP="${TOTAL_EPOCHS_PER_ACTOR_STEP}" \
TOTAL_TRAINING_STEPS_PER_ACTOR_STEP="${TOTAL_TRAINING_STEPS_PER_ACTOR_STEP}" \
SAVE_FREQ="${SAVE_FREQ}" \
TEST_FREQ="${TEST_FREQ}" \
LR="${LR}" \
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
LOGGER="${LOGGER}" \
SAVE_HF_MODEL="${SAVE_HF_MODEL}" \
REQUIRE_HF_CHECKPOINT="${REQUIRE_HF_CHECKPOINT}" \
bash scripts/run_native_verl_dual_cotraining.sh || TRAINING_STATUS=$?

VERIFY_STATUS=0
if [[ "${VERIFY_AFTER_TRAINING}" == "1" || "${VERIFY_AFTER_TRAINING}" == "True" || "${VERIFY_AFTER_TRAINING}" == "true" ]]; then
  echo "== Verifying native VERL training outputs =="
  "${PYTHON_BIN}" scripts/verify_native_verl_training_output.py \
    --output_dir "${OUTPUT_DIR}" || VERIFY_STATUS=$?
fi

if [[ "${TRAINING_STATUS}" -ne 0 ]]; then
  echo "== Production training failed with status ${TRAINING_STATUS} ==" >&2
fi

if [[ "${VERIFY_STATUS}" -ne 0 ]]; then
  echo "== Production training verification failed with status ${VERIFY_STATUS} ==" >&2
fi

if [[ "${TRAINING_STATUS}" -ne 0 ]]; then
  exit "${TRAINING_STATUS}"
fi

exit "${VERIFY_STATUS}"
