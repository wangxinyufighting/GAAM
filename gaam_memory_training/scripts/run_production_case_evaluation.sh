#!/usr/bin/env bash
set -euo pipefail

# GAAM production case-level evaluation.
#
# Pipeline:
#   LongMemEval case sessions
#     -> build Current Memory without seeing benchmark question/answer
#     -> Frozen Answerer answers the benchmark question from Current Memory
#     -> API or heuristic judge checks correctness against benchmark answer/rubric
#
# Example:
#
#   GAAM_MEMORY_BUILDER_API_KEY="${DEEPSEEK_API_KEY}" \
#   GAAM_ANSWERER_API_KEY="${DEEPSEEK_API_KEY}" \
#   GAAM_EVAL_JUDGE_API_KEY="${DEEPSEEK_API_KEY}" \
#   MEMORY_BACKEND=stateful_api \
#   ANSWER_BACKEND=api \
#   JUDGE_BACKEND=api \
#   RECORD_ID=e47becba \
#   bash scripts/run_production_case_evaluation.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Python executable inside the evaluation environment.
PYTHON_BIN="${PYTHON_BIN:-python}"

# LongMemEval-style input file containing sessions plus benchmark question/answer.
INPUT_PATH="${INPUT_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"

# Output directory for evaluation manifest and per-case reports.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/production_case_evaluation}"

# Optional train/dev/test split manifest. When set, evaluation is restricted to
# EVALUATION_SPLIT instead of all records in INPUT_PATH.
SPLIT_MANIFEST="${SPLIT_MANIFEST:-}"

# Split to evaluate when SPLIT_MANIFEST is set. Use test for final evaluation,
# dev for checkpoint/model selection, train for debugging, or all for every
# split record in the manifest.
EVALUATION_SPLIT="${EVALUATION_SPLIT:-test}"

# Optional single case ID. Leave empty to evaluate all selected records.
RECORD_ID="${RECORD_ID:-}"

# Optional maximum number of records. Useful for smoke tests.
MAX_RECORDS="${MAX_RECORDS:-}"

# Memory construction backend:
#   baseline          deterministic non-LLM memory builder
#   stateful_api      real sequential API Memory Builder over session chunks
#   stateful_local_hf real sequential local-HF Memory Builder over session chunks
MEMORY_BACKEND="${MEMORY_BACKEND:-stateful_api}"

# Answerer backend:
#   api      Frozen Answerer uses OpenAI-compatible API
#   local_hf Frozen Answerer loads a local Hugging Face checkpoint
#   no_llm   deterministic retrieval baseline
ANSWER_BACKEND="${ANSWER_BACKEND:-api}"

# Correctness judge backend:
#   api       LLM-as-judge, LongMemEval-style yes/no scoring
#   heuristic token/exact/containment heuristic
JUDGE_BACKEND="${JUDGE_BACKEND:-api}"

# Number of sessions per stateful Memory Builder chunk.
SESSION_CHUNK_SIZE="${SESSION_CHUNK_SIZE:-4}"

# Character budget for each new raw-history chunk.
MAX_CHUNK_CHARS="${MAX_CHUNK_CHARS:-12000}"

# Character budget for Previous Current Memory in each stateful update.
MAX_PREVIOUS_MEMORY_CHARS="${MAX_PREVIOUS_MEMORY_CHARS:-12000}"

# Memory Builder API model/base URL/key.
MEMORY_MODEL="${MEMORY_MODEL:-${GAAM_MEMORY_BUILDER_MODEL:-deepseek-v4-flash}}"
MEMORY_BASE_URL="${MEMORY_BASE_URL:-${GAAM_MEMORY_BUILDER_BASE_URL:-https://api.deepseek.com}}"
MEMORY_API_KEY="${MEMORY_API_KEY:-${GAAM_MEMORY_BUILDER_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}}"
MEMORY_MAX_NEW_TOKENS="${MEMORY_MAX_NEW_TOKENS:-${GAAM_MEMORY_BUILDER_MAX_NEW_TOKENS:-2048}}"
MEMORY_DEVICE_MAP="${MEMORY_DEVICE_MAP:-${GAAM_MEMORY_BUILDER_DEVICE_MAP:-auto}}"
MEMORY_TORCH_DTYPE="${MEMORY_TORCH_DTYPE:-${GAAM_MEMORY_BUILDER_TORCH_DTYPE:-auto}}"

# Frozen Answerer API model/base URL/key.
ANSWER_MODEL="${ANSWER_MODEL:-${GAAM_ANSWERER_MODEL:-deepseek-v4-flash}}"
ANSWER_BASE_URL="${ANSWER_BASE_URL:-${GAAM_ANSWERER_BASE_URL:-https://api.deepseek.com}}"
ANSWER_API_KEY="${ANSWER_API_KEY:-${GAAM_ANSWERER_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-${GAAM_ANSWERER_MAX_NEW_TOKENS:-1024}}"
ANSWER_DEVICE_MAP="${ANSWER_DEVICE_MAP:-${GAAM_ANSWERER_DEVICE_MAP:-auto}}"
ANSWER_TORCH_DTYPE="${ANSWER_TORCH_DTYPE:-${GAAM_ANSWERER_TORCH_DTYPE:-auto}}"

# Evaluation judge API model/base URL/key.
JUDGE_MODEL="${JUDGE_MODEL:-${GAAM_EVAL_JUDGE_MODEL:-deepseek-v4-flash}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${GAAM_EVAL_JUDGE_BASE_URL:-https://api.deepseek.com}}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${GAAM_EVAL_JUDGE_API_KEY:-${DEEPSEEK_API_KEY:-${OPENAI_API_KEY:-}}}}"

ARGS=(
  scripts/run_case_evaluation.py
  --input "${INPUT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --memory_backend "${MEMORY_BACKEND}"
  --answer_backend "${ANSWER_BACKEND}"
  --judge_backend "${JUDGE_BACKEND}"
  --session_chunk_size "${SESSION_CHUNK_SIZE}"
  --max_chunk_chars "${MAX_CHUNK_CHARS}"
  --max_previous_memory_chars "${MAX_PREVIOUS_MEMORY_CHARS}"
  --memory_model "${MEMORY_MODEL}"
  --memory_base_url "${MEMORY_BASE_URL}"
  --memory_max_new_tokens "${MEMORY_MAX_NEW_TOKENS}"
  --memory_device_map "${MEMORY_DEVICE_MAP}"
  --memory_torch_dtype "${MEMORY_TORCH_DTYPE}"
  --answer_model "${ANSWER_MODEL}"
  --answer_base_url "${ANSWER_BASE_URL}"
  --answer_max_new_tokens "${ANSWER_MAX_NEW_TOKENS}"
  --answer_device_map "${ANSWER_DEVICE_MAP}"
  --answer_torch_dtype "${ANSWER_TORCH_DTYPE}"
  --judge_model "${JUDGE_MODEL}"
  --judge_base_url "${JUDGE_BASE_URL}"
)

if [[ -n "${SPLIT_MANIFEST}" ]]; then
  ARGS+=(--split_manifest "${SPLIT_MANIFEST}")
  ARGS+=(--evaluation_split "${EVALUATION_SPLIT}")
fi

if [[ -n "${RECORD_ID}" ]]; then
  ARGS+=(--record_id "${RECORD_ID}")
fi

if [[ -n "${MAX_RECORDS}" ]]; then
  ARGS+=(--max_records "${MAX_RECORDS}")
fi

if [[ -n "${MEMORY_API_KEY}" ]]; then
  ARGS+=(--memory_api_key "${MEMORY_API_KEY}")
fi

if [[ -n "${ANSWER_API_KEY}" ]]; then
  ARGS+=(--answer_api_key "${ANSWER_API_KEY}")
fi

if [[ -n "${JUDGE_API_KEY}" ]]; then
  ARGS+=(--judge_api_key "${JUDGE_API_KEY}")
fi

echo "== GAAM production case evaluation =="
echo "INPUT_PATH=${INPUT_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SPLIT_MANIFEST=${SPLIT_MANIFEST:-<none>}"
echo "EVALUATION_SPLIT=${EVALUATION_SPLIT}"
echo "RECORD_ID=${RECORD_ID:-<all>}"
echo "MAX_RECORDS=${MAX_RECORDS:-<none>}"
echo "MEMORY_BACKEND=${MEMORY_BACKEND}"
echo "ANSWER_BACKEND=${ANSWER_BACKEND}"
echo "JUDGE_BACKEND=${JUDGE_BACKEND}"
echo "MEMORY_BASE_URL=${MEMORY_BASE_URL}"
echo "MEMORY_DEVICE_MAP=${MEMORY_DEVICE_MAP}"
echo "MEMORY_TORCH_DTYPE=${MEMORY_TORCH_DTYPE}"
echo "ANSWER_BASE_URL=${ANSWER_BASE_URL}"
echo "ANSWER_DEVICE_MAP=${ANSWER_DEVICE_MAP}"
echo "ANSWER_TORCH_DTYPE=${ANSWER_TORCH_DTYPE}"
echo "JUDGE_BASE_URL=${JUDGE_BASE_URL}"

"${PYTHON_BIN}" "${ARGS[@]}"
