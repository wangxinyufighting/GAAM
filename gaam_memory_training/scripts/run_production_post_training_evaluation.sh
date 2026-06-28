#!/usr/bin/env bash
set -euo pipefail

# GAAM post-training evaluation entrypoint.
#
# Pipeline:
#   native dual VERL output directory
#     -> verify real GRPO training artifacts and HF checkpoints
#     -> resolve final Memory Builder / Question Agent checkpoint paths
#     -> evaluate cases with the trained Memory Builder checkpoint
#
# Typical usage:
#
#   TRAINING_OUTPUT_DIR=outputs/production_native_verl_dual_cotraining \
#   ANSWER_BACKEND=api \
#   JUDGE_BACKEND=api \
#   ANSWER_API_KEY="${DEEPSEEK_API_KEY}" \
#   JUDGE_API_KEY="${DEEPSEEK_API_KEY}" \
#   RECORD_ID=e47becba \
#   bash scripts/run_production_post_training_evaluation.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Python executable inside the evaluation environment.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Output directory produced by scripts/run_production_gaam_training.sh.
TRAINING_OUTPUT_DIR="${TRAINING_OUTPUT_DIR:-outputs/production_native_verl_dual_cotraining}"

# If 1, require the training output verifier to pass before evaluation.
VERIFY_TRAINING_OUTPUT="${VERIFY_TRAINING_OUTPUT:-1}"

# If 1, verify evaluation artifacts and leakage safety after case evaluation.
VERIFY_EVALUATION_OUTPUT="${VERIFY_EVALUATION_OUTPUT:-1}"

# If 1, fail when resolved checkpoint paths do not exist.
REQUIRE_RESOLVED_CHECKPOINTS="${REQUIRE_RESOLVED_CHECKPOINTS:-1}"

# LongMemEval-style input file containing sessions plus benchmark question/answer.
INPUT_PATH="${INPUT_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"

# Optional split manifest from training. If provided, final evaluation defaults
# to EVALUATION_SPLIT=test so held-out test records are used instead of the
# entire raw input file.
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"

# Split to evaluate when SPLIT_MANIFEST is set.
EVALUATION_SPLIT="${EVALUATION_SPLIT:-test}"

# Output directory for post-training evaluation reports.
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/production_post_training_case_evaluation}"

# Optional single case ID. Leave empty to evaluate all selected records.
RECORD_ID="${RECORD_ID:-}"

# Optional maximum number of records. Useful for smoke tests.
MAX_RECORDS="${MAX_RECORDS:-}"

# Memory backend defaults to the trained local HF Memory Builder checkpoint.
MEMORY_BACKEND="${MEMORY_BACKEND:-stateful_local_hf}"

# Frozen Answerer backend. Use api for a stable external/frozen Answerer, or
# local_hf if you have a local frozen Answerer checkpoint.
ANSWER_BACKEND="${ANSWER_BACKEND:-api}"

# Correctness judge backend: api or heuristic.
JUDGE_BACKEND="${JUDGE_BACKEND:-api}"

# Number of sessions per stateful Memory Builder chunk.
SESSION_CHUNK_SIZE="${SESSION_CHUNK_SIZE:-4}"

# Character budget for each new raw-history chunk.
MAX_CHUNK_CHARS="${MAX_CHUNK_CHARS:-12000}"

# Character budget for Previous Current Memory in each stateful update.
MAX_PREVIOUS_MEMORY_CHARS="${MAX_PREVIOUS_MEMORY_CHARS:-12000}"

# Local HF generation settings for trained Memory Builder evaluation.
MEMORY_MAX_NEW_TOKENS="${MEMORY_MAX_NEW_TOKENS:-2048}"
MEMORY_DEVICE_MAP="${MEMORY_DEVICE_MAP:-auto}"
MEMORY_TORCH_DTYPE="${MEMORY_TORCH_DTYPE:-auto}"

# Answerer model/base URL/key. If ANSWER_BACKEND=local_hf, ANSWER_MODEL must be a
# local Hugging Face checkpoint directory.
ANSWER_MODEL="${ANSWER_MODEL:-${GAAM_ANSWERER_MODEL:-deepseek-v4-flash}}"
ANSWER_BASE_URL="${ANSWER_BASE_URL:-${GAAM_ANSWERER_BASE_URL:-https://api.deepseek.com}}"
ANSWER_API_KEY="${ANSWER_API_KEY:-${GAAM_ANSWERER_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-1024}"
ANSWER_DEVICE_MAP="${ANSWER_DEVICE_MAP:-auto}"
ANSWER_TORCH_DTYPE="${ANSWER_TORCH_DTYPE:-auto}"

# Evaluation judge model/base URL/key.
JUDGE_MODEL="${JUDGE_MODEL:-${GAAM_EVAL_JUDGE_MODEL:-deepseek-v4-flash}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${GAAM_EVAL_JUDGE_BASE_URL:-https://api.deepseek.com}}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${GAAM_EVAL_JUDGE_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}}"

if [[ "${VERIFY_TRAINING_OUTPUT}" == "1" || "${VERIFY_TRAINING_OUTPUT}" == "True" || "${VERIFY_TRAINING_OUTPUT}" == "true" ]]; then
  echo "== Verifying native VERL training output before evaluation =="
  "${PYTHON_BIN}" scripts/verify_native_verl_training_output.py \
    --output_dir "${TRAINING_OUTPUT_DIR}"
fi

RESOLVE_ARGS=(
  scripts/resolve_native_verl_checkpoints.py
  --output_dir "${TRAINING_OUTPUT_DIR}"
  --format shell
)

if [[ "${REQUIRE_RESOLVED_CHECKPOINTS}" == "0" || "${REQUIRE_RESOLVED_CHECKPOINTS}" == "False" || "${REQUIRE_RESOLVED_CHECKPOINTS}" == "false" ]]; then
  RESOLVE_ARGS+=(--allow_missing)
fi

eval "$("${PYTHON_BIN}" "${RESOLVE_ARGS[@]}")"

if [[ -z "${MEMORY_MODEL:-}" ]]; then
  echo "Resolved MEMORY_MODEL is empty. Check ${TRAINING_OUTPUT_DIR}/dual_cotraining_manifest.json." >&2
  exit 1
fi

echo "== GAAM production post-training evaluation =="
echo "TRAINING_OUTPUT_DIR=${TRAINING_OUTPUT_DIR}"
echo "INPUT_PATH=${INPUT_PATH}"
echo "SPLIT_MANIFEST=${SPLIT_MANIFEST:-<none>}"
echo "EVALUATION_SPLIT=${EVALUATION_SPLIT}"
echo "EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
echo "RECORD_ID=${RECORD_ID:-<all>}"
echo "MAX_RECORDS=${MAX_RECORDS:-<none>}"
echo "MEMORY_BACKEND=${MEMORY_BACKEND}"
echo "MEMORY_MODEL=${MEMORY_MODEL}"
echo "ANSWER_BACKEND=${ANSWER_BACKEND}"
echo "JUDGE_BACKEND=${JUDGE_BACKEND}"

EVAL_STATUS=0
INPUT_PATH="${INPUT_PATH}" \
OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
SPLIT_MANIFEST="${SPLIT_MANIFEST}" \
EVALUATION_SPLIT="${EVALUATION_SPLIT}" \
RECORD_ID="${RECORD_ID}" \
MAX_RECORDS="${MAX_RECORDS}" \
MEMORY_BACKEND="${MEMORY_BACKEND}" \
MEMORY_MODEL="${MEMORY_MODEL}" \
MEMORY_MAX_NEW_TOKENS="${MEMORY_MAX_NEW_TOKENS}" \
MEMORY_DEVICE_MAP="${MEMORY_DEVICE_MAP}" \
MEMORY_TORCH_DTYPE="${MEMORY_TORCH_DTYPE}" \
ANSWER_BACKEND="${ANSWER_BACKEND}" \
ANSWER_MODEL="${ANSWER_MODEL}" \
ANSWER_BASE_URL="${ANSWER_BASE_URL}" \
ANSWER_API_KEY="${ANSWER_API_KEY}" \
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS}" \
ANSWER_DEVICE_MAP="${ANSWER_DEVICE_MAP}" \
ANSWER_TORCH_DTYPE="${ANSWER_TORCH_DTYPE}" \
JUDGE_BACKEND="${JUDGE_BACKEND}" \
JUDGE_MODEL="${JUDGE_MODEL}" \
JUDGE_BASE_URL="${JUDGE_BASE_URL}" \
JUDGE_API_KEY="${JUDGE_API_KEY}" \
SESSION_CHUNK_SIZE="${SESSION_CHUNK_SIZE}" \
MAX_CHUNK_CHARS="${MAX_CHUNK_CHARS}" \
MAX_PREVIOUS_MEMORY_CHARS="${MAX_PREVIOUS_MEMORY_CHARS}" \
bash scripts/run_production_case_evaluation.sh || EVAL_STATUS=$?

VERIFY_STATUS=0
if [[ "${VERIFY_EVALUATION_OUTPUT}" == "1" || "${VERIFY_EVALUATION_OUTPUT}" == "True" || "${VERIFY_EVALUATION_OUTPUT}" == "true" ]]; then
  echo "== Verifying case evaluation output =="
  "${PYTHON_BIN}" scripts/verify_case_evaluation_output.py \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --input "${INPUT_PATH}" || VERIFY_STATUS=$?
fi

if [[ "${EVAL_STATUS}" != "0" ]]; then
  exit "${EVAL_STATUS}"
fi

if [[ "${VERIFY_STATUS}" != "0" ]]; then
  exit "${VERIFY_STATUS}"
fi
