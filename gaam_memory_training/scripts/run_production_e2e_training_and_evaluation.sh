#!/usr/bin/env bash
set -euo pipefail

# GAAM production end-to-end entrypoint.
#
# Pipeline:
#   existing oracle graphs
#     -> graph-backed train/dev/test split
#     -> native Code-A1/VERL dual-agent GRPO training
#     -> checkpoint verification
#     -> post-training case evaluation
#     -> evaluation artifact verification
#
# Typical remote GPU usage:
#
#   MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   DEEPSEEK_API_KEY="..." \
#   GAAM_REWARD_JUDGE_ENABLED=1 \
#   ANSWER_BACKEND=api \
#   JUDGE_BACKEND=api \
#   bash scripts/run_production_e2e_training_and_evaluation.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Python executable inside the training/evaluation environment.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Path to raw LongMemEval-style records. The Memory Builder sees only sessions
# during memory construction; benchmark question/answer are used only in final
# evaluation.
INPUT_PATH="${INPUT_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"

# Directory containing prebuilt oracle graph files: <record_id>.graph.json.
ORACLE_GRAPH_DIR="${ORACLE_GRAPH_DIR:-outputs/longmemeval_s_graph}"

# If 1, create SPLIT_MANIFEST from oracle graphs that already exist and are also
# present in INPUT_PATH. This avoids missing-graph noise when only a subset of
# cases has been built.
AUTO_CREATE_SPLIT="${AUTO_CREATE_SPLIT:-1}"

# Random seed used when AUTO_CREATE_SPLIT=1.
SPLIT_SEED="${SPLIT_SEED:-0}"

# Output split manifest. If unset, a deterministic graph-backed split name is
# used under outputs/splits.
SPLIT_MANIFEST="${SPLIT_MANIFEST:-outputs/splits/longmemeval_s.existing_graphs.seed${SPLIT_SEED}.json}"

# Split ratios for AUTO_CREATE_SPLIT=1. For real experiments, build enough
# oracle graphs to keep train/dev/test non-empty.
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
DEV_RATIO="${DEV_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"

# Allow empty dev/test split when graph-backed case count is too small. Useful
# for smoke tests, not recommended for final experiments.
ALLOW_EMPTY_SPLIT="${ALLOW_EMPTY_SPLIT:-0}"

# Optional cap on graph-backed cases after sorting. Leave empty for all existing
# oracle graphs.
SPLIT_MAX_CASES="${SPLIT_MAX_CASES:-}"

# If 1, replace an existing split manifest when AUTO_CREATE_SPLIT=1.
OVERWRITE_SPLIT="${OVERWRITE_SPLIT:-1}"

# Output directory for native dual-agent GRPO training artifacts.
TRAINING_OUTPUT_DIR="${TRAINING_OUTPUT_DIR:-outputs/production_native_verl_dual_cotraining}"

# Output directory for post-training evaluation reports.
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/production_post_training_case_evaluation}"

# If 1, run post-training evaluation after training and checkpoint verification.
RUN_EVALUATION="${RUN_EVALUATION:-1}"

# If 1, run a full preflight check after split creation and before launching
# expensive native VERL training. Keep enabled for remote GPU runs.
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

# If 1, preflight requires CUDA and enough visible GPUs for the configured
# Memory Builder / Question Agent actor steps.
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

# If 1, preflight checks Python package imports such as torch, transformers,
# pyarrow, ray, vLLM, OpenAI, and Code-A1/VERL.
CHECK_IMPORTS="${CHECK_IMPORTS:-1}"

# If 1, preflight imports the vendored Code-A1/VERL package. Set 0 only for
# shell-level smoke tests where the filesystem scaffold is intentionally fake.
REQUIRE_VERL_IMPORT="${REQUIRE_VERL_IMPORT:-1}"

# If 1, write a final production_run_audit.json summarizing training
# verification, checkpoint availability, evaluation split, accuracy, and
# evaluation leakage verification.
RUN_FINAL_AUDIT="${RUN_FINAL_AUDIT:-1}"

# If 1, export a compact checkpoint-free evidence bundle after final audit. The
# bundle contains manifests, verification reports, evaluation summary, split
# manifest, audit report, checksums, and a redacted environment snapshot.
EXPORT_ARTIFACT_BUNDLE="${EXPORT_ARTIFACT_BUNDLE:-1}"

# Output directory for the compact production evidence bundle.
ARTIFACT_BUNDLE_DIR="${ARTIFACT_BUNDLE_DIR:-outputs/production_artifact_bundle}"

# Optional run id stored in the bundle manifest. Useful for remote server runs.
RUN_ID="${RUN_ID:-}"

# Split to evaluate after training. Defaults to test for final held-out
# evaluation. Use dev for model selection/debugging, train for debugging, or all
# to evaluate every record in the split manifest.
EVALUATION_SPLIT="${EVALUATION_SPLIT:-test}"

# Optional evaluation case filter. Leave empty to evaluate all selected records.
RECORD_ID="${RECORD_ID:-}"

# Optional maximum number of evaluation records.
MAX_RECORDS="${MAX_RECORDS:-}"

# Initial trainable Memory Builder model path. Required by training.
MEMORY_MODEL_PATH="${MEMORY_MODEL_PATH:-}"

# Initial trainable Question Agent model path. Required by training.
QUESTION_MODEL_PATH="${QUESTION_MODEL_PATH:-}"

# Code-A1 checkout root. The child scripts expect ${CODE_A1_ROOT}/verl/verl.
CODE_A1_ROOT="${CODE_A1_ROOT:-$(cd "${ROOT_DIR}/.." && pwd)/Code-A1/Code-A1}"

# Native GRPO co-training rounds.
ROUNDS="${ROUNDS:-1}"

# Actor update order per round.
ORDER="${ORDER:-question_agent,memory_builder}"

# GPU counts. You can also set MEMORY_NUM_GPUS and QUESTION_NUM_GPUS separately.
NUM_GPUS="${NUM_GPUS:-1}"
MEMORY_NUM_GPUS="${MEMORY_NUM_GPUS:-${NUM_GPUS}}"
QUESTION_NUM_GPUS="${QUESTION_NUM_GPUS:-${NUM_GPUS}}"

# Number of sampled responses per prompt in VERL rollout. GRPO benefits from
# multiple samples per group.
ROLLOUT_N="${ROLLOUT_N:-4}"

# VERL train batch size. Child scripts cap it to available train rows when
# necessary to avoid an empty dataloader.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"

# PPO/GRPO mini-batch size.
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"

# PPO/GRPO micro-batch size per GPU.
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"

# Max tokenized prompt length for native VERL.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"

# Max generated response length for native VERL.
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"

# Max raw-history characters in Memory Builder full prompts.
MAX_HISTORY_CHARS="${MAX_HISTORY_CHARS:-16000}"

# Max oracle digest characters in Question Agent prompts.
MAX_ORACLE_CHARS="${MAX_ORACLE_CHARS:-12000}"

# Exact target number of questions generated per case by Question Agent.
QUESTIONS_PER_CASE="${QUESTIONS_PER_CASE:-8}"

# Native parquet Memory Builder input mode. Keep "full" for production native
# VERL; real stateful incremental memory construction is used in evaluation.
MEMORY_INPUT_MODE="${MEMORY_INPUT_MODE:-full}"

# Optional record cap per split for debugging.
MAX_RECORDS_PER_SPLIT="${MAX_RECORDS_PER_SPLIT:-}"

# Native VERL training epochs per actor step.
TOTAL_EPOCHS_PER_ACTOR_STEP="${TOTAL_EPOCHS_PER_ACTOR_STEP:-1}"

# Optional hard cap on native VERL training steps per actor step.
TOTAL_TRAINING_STEPS_PER_ACTOR_STEP="${TOTAL_TRAINING_STEPS_PER_ACTOR_STEP:-}"

# Native VERL checkpoint save/test frequencies.
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:-1}"

# Actor learning rate.
LR="${LR:-1e-6}"

# vLLM tensor parallel size and GPU memory utilization.
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"

# VERL logger backend.
LOGGER="${LOGGER:-console}"

# Full-model checkpoint settings. Keep both enabled for production.
SAVE_HF_MODEL="${SAVE_HF_MODEL:-1}"
REQUIRE_HF_CHECKPOINT="${REQUIRE_HF_CHECKPOINT:-1}"

# LLM-as-judge reward config. API key can also come from DEEPSEEK_API_KEY or
# OPENAI_API_KEY in child scripts.
export GAAM_REWARD_JUDGE_ENABLED="${GAAM_REWARD_JUDGE_ENABLED:-0}"
export GAAM_REWARD_JUDGE_BASE_URL="${GAAM_REWARD_JUDGE_BASE_URL:-https://api.deepseek.com}"
export GAAM_REWARD_JUDGE_API_KEY="${GAAM_REWARD_JUDGE_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}"
export GAAM_REWARD_JUDGE_MODEL="${GAAM_REWARD_JUDGE_MODEL:-deepseek-v4-flash}"
export GAAM_REWARD_JUDGE_REQUIRED="${GAAM_REWARD_JUDGE_REQUIRED:-0}"

# Post-training evaluation backends and generation settings.
MEMORY_BACKEND="${MEMORY_BACKEND:-stateful_local_hf}"
ANSWER_BACKEND="${ANSWER_BACKEND:-api}"
JUDGE_BACKEND="${JUDGE_BACKEND:-api}"
SESSION_CHUNK_SIZE="${SESSION_CHUNK_SIZE:-4}"
MAX_CHUNK_CHARS="${MAX_CHUNK_CHARS:-12000}"
MAX_PREVIOUS_MEMORY_CHARS="${MAX_PREVIOUS_MEMORY_CHARS:-12000}"
MEMORY_MAX_NEW_TOKENS="${MEMORY_MAX_NEW_TOKENS:-2048}"
MEMORY_DEVICE_MAP="${MEMORY_DEVICE_MAP:-auto}"
MEMORY_TORCH_DTYPE="${MEMORY_TORCH_DTYPE:-auto}"
ANSWER_MODEL="${ANSWER_MODEL:-${GAAM_ANSWERER_MODEL:-deepseek-v4-flash}}"
ANSWER_BASE_URL="${ANSWER_BASE_URL:-${GAAM_ANSWERER_BASE_URL:-https://api.deepseek.com}}"
ANSWER_API_KEY="${ANSWER_API_KEY:-${GAAM_ANSWERER_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-1024}"
ANSWER_DEVICE_MAP="${ANSWER_DEVICE_MAP:-auto}"
ANSWER_TORCH_DTYPE="${ANSWER_TORCH_DTYPE:-auto}"
JUDGE_MODEL="${JUDGE_MODEL:-${GAAM_EVAL_JUDGE_MODEL:-deepseek-v4-flash}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${GAAM_EVAL_JUDGE_BASE_URL:-https://api.deepseek.com}}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${GAAM_EVAL_JUDGE_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}}"
VERIFY_TRAINING_OUTPUT="${VERIFY_TRAINING_OUTPUT:-1}"
VERIFY_EVALUATION_OUTPUT="${VERIFY_EVALUATION_OUTPUT:-1}"

if [[ -z "${MEMORY_MODEL_PATH}" ]]; then
  echo "MEMORY_MODEL_PATH is required." >&2
  exit 1
fi

if [[ -z "${QUESTION_MODEL_PATH}" ]]; then
  echo "QUESTION_MODEL_PATH is required." >&2
  exit 1
fi

echo "== GAAM production E2E training + evaluation =="
echo "INPUT_PATH=${INPUT_PATH}"
echo "ORACLE_GRAPH_DIR=${ORACLE_GRAPH_DIR}"
echo "AUTO_CREATE_SPLIT=${AUTO_CREATE_SPLIT}"
echo "SPLIT_MANIFEST=${SPLIT_MANIFEST}"
echo "TRAINING_OUTPUT_DIR=${TRAINING_OUTPUT_DIR}"
echo "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
echo "RUN_EVALUATION=${RUN_EVALUATION}"
echo "RUN_PREFLIGHT=${RUN_PREFLIGHT}"
echo "REQUIRE_CUDA=${REQUIRE_CUDA}"
echo "CHECK_IMPORTS=${CHECK_IMPORTS}"
echo "RUN_FINAL_AUDIT=${RUN_FINAL_AUDIT}"
echo "EXPORT_ARTIFACT_BUNDLE=${EXPORT_ARTIFACT_BUNDLE}"
echo "ARTIFACT_BUNDLE_DIR=${ARTIFACT_BUNDLE_DIR}"
echo "EVALUATION_SPLIT=${EVALUATION_SPLIT}"
echo "ROUNDS=${ROUNDS}"
echo "ORDER=${ORDER}"
echo "MEMORY_NUM_GPUS=${MEMORY_NUM_GPUS}"
echo "QUESTION_NUM_GPUS=${QUESTION_NUM_GPUS}"
echo "GAAM_REWARD_JUDGE_ENABLED=${GAAM_REWARD_JUDGE_ENABLED}"

if [[ "${AUTO_CREATE_SPLIT}" == "1" || "${AUTO_CREATE_SPLIT}" == "True" || "${AUTO_CREATE_SPLIT}" == "true" ]]; then
  SPLIT_ARGS=(
    scripts/create_split_from_existing_graphs.py
    --input "${INPUT_PATH}"
    --oracle_graph_dir "${ORACLE_GRAPH_DIR}"
    --output "${SPLIT_MANIFEST}"
    --train_ratio "${TRAIN_RATIO}"
    --dev_ratio "${DEV_RATIO}"
    --test_ratio "${TEST_RATIO}"
    --seed "${SPLIT_SEED}"
  )
  if [[ "${ALLOW_EMPTY_SPLIT}" == "1" || "${ALLOW_EMPTY_SPLIT}" == "True" || "${ALLOW_EMPTY_SPLIT}" == "true" ]]; then
    SPLIT_ARGS+=(--allow_empty_split)
  fi
  if [[ -n "${SPLIT_MAX_CASES}" ]]; then
    SPLIT_ARGS+=(--max_cases "${SPLIT_MAX_CASES}")
  fi
  if [[ "${OVERWRITE_SPLIT}" == "1" || "${OVERWRITE_SPLIT}" == "True" || "${OVERWRITE_SPLIT}" == "true" ]]; then
    SPLIT_ARGS+=(--overwrite)
  fi
  echo "== Creating graph-backed split =="
  "${PYTHON_BIN}" "${SPLIT_ARGS[@]}"
fi

if [[ "${RUN_PREFLIGHT}" == "1" || "${RUN_PREFLIGHT}" == "True" || "${RUN_PREFLIGHT}" == "true" ]]; then
  PREFLIGHT_ARGS=(
    scripts/check_production_e2e_preflight.py
    --input "${INPUT_PATH}"
    --oracle_graph_dir "${ORACLE_GRAPH_DIR}"
    --split_manifest "${SPLIT_MANIFEST}"
    --memory_model_path "${MEMORY_MODEL_PATH}"
    --question_model_path "${QUESTION_MODEL_PATH}"
    --code_a1_root "${CODE_A1_ROOT}"
    --training_output_dir "${TRAINING_OUTPUT_DIR}"
    --eval_output_dir "${EVAL_OUTPUT_DIR}"
    --run_evaluation "${RUN_EVALUATION}"
    --evaluation_split "${EVALUATION_SPLIT}"
    --memory_backend "${MEMORY_BACKEND}"
    --answer_backend "${ANSWER_BACKEND}"
    --judge_backend "${JUDGE_BACKEND}"
    --memory_api_key "${GAAM_MEMORY_BUILDER_API_KEY:-}"
    --answer_api_key "${ANSWER_API_KEY}"
    --judge_api_key "${JUDGE_API_KEY}"
    --reward_judge_enabled "${GAAM_REWARD_JUDGE_ENABLED}"
    --reward_judge_api_key "${GAAM_REWARD_JUDGE_API_KEY}"
    --require_cuda "${REQUIRE_CUDA}"
    --check_imports "${CHECK_IMPORTS}"
    --require_verl_import "${REQUIRE_VERL_IMPORT}"
    --memory_num_gpus "${MEMORY_NUM_GPUS}"
    --question_num_gpus "${QUESTION_NUM_GPUS}"
    --train_batch_size "${TRAIN_BATCH_SIZE}"
    --questions_per_case "${QUESTIONS_PER_CASE}"
  )
  echo "== Running production E2E preflight =="
  "${PYTHON_BIN}" "${PREFLIGHT_ARGS[@]}"
fi

echo "== Running production training =="
PYTHON_BIN="${PYTHON_BIN}" \
INPUT_PATH="${INPUT_PATH}" \
ORACLE_GRAPH_DIR="${ORACLE_GRAPH_DIR}" \
SPLIT_MANIFEST="${SPLIT_MANIFEST}" \
OUTPUT_DIR="${TRAINING_OUTPUT_DIR}" \
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
VERIFY_AFTER_TRAINING="${VERIFY_TRAINING_OUTPUT}" \
bash scripts/run_production_gaam_training.sh

if [[ "${RUN_EVALUATION}" == "1" || "${RUN_EVALUATION}" == "True" || "${RUN_EVALUATION}" == "true" ]]; then
  echo "== Running production post-training evaluation =="
  PYTHON_BIN="${PYTHON_BIN}" \
  TRAINING_OUTPUT_DIR="${TRAINING_OUTPUT_DIR}" \
  VERIFY_TRAINING_OUTPUT="${VERIFY_TRAINING_OUTPUT}" \
  VERIFY_EVALUATION_OUTPUT="${VERIFY_EVALUATION_OUTPUT}" \
  INPUT_PATH="${INPUT_PATH}" \
  SPLIT_MANIFEST="${SPLIT_MANIFEST}" \
  EVALUATION_SPLIT="${EVALUATION_SPLIT}" \
  EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
  RECORD_ID="${RECORD_ID}" \
  MAX_RECORDS="${MAX_RECORDS}" \
  MEMORY_BACKEND="${MEMORY_BACKEND}" \
  ANSWER_BACKEND="${ANSWER_BACKEND}" \
  JUDGE_BACKEND="${JUDGE_BACKEND}" \
  SESSION_CHUNK_SIZE="${SESSION_CHUNK_SIZE}" \
  MAX_CHUNK_CHARS="${MAX_CHUNK_CHARS}" \
  MAX_PREVIOUS_MEMORY_CHARS="${MAX_PREVIOUS_MEMORY_CHARS}" \
  MEMORY_MAX_NEW_TOKENS="${MEMORY_MAX_NEW_TOKENS}" \
  MEMORY_DEVICE_MAP="${MEMORY_DEVICE_MAP}" \
  MEMORY_TORCH_DTYPE="${MEMORY_TORCH_DTYPE}" \
  ANSWER_MODEL="${ANSWER_MODEL}" \
  ANSWER_BASE_URL="${ANSWER_BASE_URL}" \
  ANSWER_API_KEY="${ANSWER_API_KEY}" \
  ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS}" \
  ANSWER_DEVICE_MAP="${ANSWER_DEVICE_MAP}" \
  ANSWER_TORCH_DTYPE="${ANSWER_TORCH_DTYPE}" \
  JUDGE_MODEL="${JUDGE_MODEL}" \
  JUDGE_BASE_URL="${JUDGE_BASE_URL}" \
  JUDGE_API_KEY="${JUDGE_API_KEY}" \
  bash scripts/run_production_post_training_evaluation.sh
fi

if [[ "${RUN_FINAL_AUDIT}" == "1" || "${RUN_FINAL_AUDIT}" == "True" || "${RUN_FINAL_AUDIT}" == "true" ]]; then
  AUDIT_ARGS=(
    scripts/audit_production_run.py
    --training_output_dir "${TRAINING_OUTPUT_DIR}"
    --split_manifest "${SPLIT_MANIFEST}"
    --expected_evaluation_split "${EVALUATION_SPLIT}"
  )
  if [[ "${RUN_EVALUATION}" == "1" || "${RUN_EVALUATION}" == "True" || "${RUN_EVALUATION}" == "true" ]]; then
    AUDIT_ARGS+=(--evaluation_output_dir "${EVAL_OUTPUT_DIR}")
  fi
  echo "== Auditing production E2E artifacts =="
  "${PYTHON_BIN}" "${AUDIT_ARGS[@]}"
fi

if [[ "${EXPORT_ARTIFACT_BUNDLE}" == "1" || "${EXPORT_ARTIFACT_BUNDLE}" == "True" || "${EXPORT_ARTIFACT_BUNDLE}" == "true" ]]; then
  BUNDLE_ARGS=(
    scripts/export_production_artifact_bundle.py
    --training_output_dir "${TRAINING_OUTPUT_DIR}"
    --split_manifest "${SPLIT_MANIFEST}"
    --bundle_output_dir "${ARTIFACT_BUNDLE_DIR}"
  )
  if [[ "${RUN_EVALUATION}" == "1" || "${RUN_EVALUATION}" == "True" || "${RUN_EVALUATION}" == "true" ]]; then
    BUNDLE_ARGS+=(--evaluation_output_dir "${EVAL_OUTPUT_DIR}")
  fi
  if [[ -n "${RUN_ID}" ]]; then
    BUNDLE_ARGS+=(--run_id "${RUN_ID}")
  fi
  echo "== Exporting compact production artifact bundle =="
  "${PYTHON_BIN}" "${BUNDLE_ARGS[@]}"
fi

echo "== GAAM production E2E complete =="
echo "Training output: ${TRAINING_OUTPUT_DIR}"
if [[ "${RUN_EVALUATION}" == "1" || "${RUN_EVALUATION}" == "True" || "${RUN_EVALUATION}" == "true" ]]; then
  echo "Evaluation output: ${EVAL_OUTPUT_DIR}"
fi
if [[ "${EXPORT_ARTIFACT_BUNDLE}" == "1" || "${EXPORT_ARTIFACT_BUNDLE}" == "True" || "${EXPORT_ARTIFACT_BUNDLE}" == "true" ]]; then
  echo "Artifact bundle: ${ARTIFACT_BUNDLE_DIR}"
fi
