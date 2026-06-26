#!/usr/bin/env bash
set -euo pipefail

# Remote Phase 4 benchmark-suite entrypoint.
#
# Default mode:
#   Runs the real e47becba debug smoke suite with dry-run training.
#
# GPU/verl mode:
#   Set TRAINER_BACKEND=verl and provide model paths. For a real benchmark,
#   also provide SUITE_CONFIG pointing to a multi-case train/dev/test suite.
#
# Example:
#   MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   ANSWERER_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   TRAINER_BACKEND=verl \
#   SUITE_CONFIG=configs/phase4/my_full_suite.json \
#   OUTPUT_DIR=outputs/phase4_benchmark_suites/full_verl \
#   SEEDS="0 1 2" \
#   bash scripts/run_remote_phase4_benchmark_suite.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Neither python nor python3 was found on PATH. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python >= 3.10 is required, got {sys.version.split()[0]}. "
        "Activate the correct conda environment or set PYTHON_BIN explicitly."
    )
PY

INPUT_PATH="${INPUT_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"
ORACLE_GRAPH_DIR="${ORACLE_GRAPH_DIR:-outputs/longmemeval_s_graph}"
RECORD_ID="${RECORD_ID:-e47becba}"
DEBUG_SPLIT_PATH="${DEBUG_SPLIT_PATH:-outputs/splits/${RECORD_ID}.debug.json}"

SUITE_CONFIG="${SUITE_CONFIG:-configs/phase4/debug_smoke_suite.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase4_benchmark_suites/debug_smoke}"
TRAINER_BACKEND="${TRAINER_BACKEND:-dry_run}"
SEEDS="${SEEDS:-0}"

MEMORY_MODEL_PATH="${MEMORY_MODEL_PATH:-}"
QUESTION_MODEL_PATH="${QUESTION_MODEL_PATH:-}"
ANSWERER_MODEL_PATH="${ANSWERER_MODEL_PATH:-}"

# Set INSTALL_DEPS=1 on a fresh server. Default is off so the script does not
# mutate a managed environment unexpectedly.
INSTALL_DEPS="${INSTALL_DEPS:-0}"

# Debug split preparation is useful for the default e47becba smoke suite.
PREPARE_DEBUG_SPLIT="${PREPARE_DEBUG_SPLIT:-1}"

# Use OVERWRITE=1 for a fresh run, or OVERWRITE=0 to resume.
OVERWRITE="${OVERWRITE:-1}"

# These post-processing steps are cheap and validate that the run is publishable.
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
RUN_RELEASE_PACKAGE="${RUN_RELEASE_PACKAGE:-1}"
RUN_ARTIFACT_VERIFY="${RUN_ARTIFACT_VERIFY:-1}"

if [[ -z "${CODE_A1_ROOT:-}" ]]; then
  if [[ -d "${REPO_DIR}/Code-A1/Code-A1" ]]; then
    CODE_A1_ROOT="${REPO_DIR}/Code-A1/Code-A1"
  elif [[ -d "${ROOT_DIR}/Code-A1/Code-A1" ]]; then
    CODE_A1_ROOT="${ROOT_DIR}/Code-A1/Code-A1"
  else
    CODE_A1_ROOT="${REPO_DIR}/Code-A1/Code-A1"
  fi
fi
VERL_ROOT="${VERL_ROOT:-${CODE_A1_ROOT}/verl}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"

if [[ "${TRAINER_BACKEND}" == "verl" ]]; then
  export PYTHONPATH="${ROOT_DIR}:${CODE_A1_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
fi

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
}

require_model_path_for_verl() {
  local name="$1"
  local path="$2"
  if [[ -z "${path}" ]]; then
    echo "TRAINER_BACKEND=verl requires ${name} to be set." >&2
    exit 1
  fi
  require_dir "${path}"
}

install_if_present() {
  local requirements_file="$1"
  if [[ -f "${requirements_file}" ]]; then
    "${PYTHON_BIN}" -m pip install -r "${requirements_file}"
  fi
}

echo "== GAAM Phase 4 benchmark-suite script =="
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "SUITE_CONFIG=${SUITE_CONFIG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRAINER_BACKEND=${TRAINER_BACKEND}"
echo "SEEDS=${SEEDS}"

require_file "${INPUT_PATH}"
require_dir "${ORACLE_GRAPH_DIR}"
require_file "${ORACLE_GRAPH_DIR}/${RECORD_ID}.graph.json"
require_file "${SUITE_CONFIG}"

if [[ "${TRAINER_BACKEND}" == "verl" ]]; then
  require_dir "${CODE_A1_ROOT}"
  require_dir "${VERL_ROOT}"
  require_model_path_for_verl "MEMORY_MODEL_PATH" "${MEMORY_MODEL_PATH}"
  require_model_path_for_verl "QUESTION_MODEL_PATH" "${QUESTION_MODEL_PATH}"
  require_model_path_for_verl "ANSWERER_MODEL_PATH" "${ANSWERER_MODEL_PATH}"
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "== Installing GAAM dependencies =="
  install_if_present "requirements.txt"
  install_if_present "requirements-local-training.txt"

  if [[ "${TRAINER_BACKEND}" == "verl" && -f "${VERL_ROOT}/requirements.txt" ]]; then
    echo "== Installing vendored verl dependencies =="
    "${PYTHON_BIN}" -m pip install -r "${VERL_ROOT}/requirements.txt"
  fi
fi

echo "== Python / CUDA check =="
"${PYTHON_BIN}" - <<'PY'
import os
import sys

print("python:", sys.version.replace("\n", " "))
print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda_device_0:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_check_failed:", repr(exc))

try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as exc:
    print("transformers_check_failed:", repr(exc))

try:
    import verl
    print("verl:", getattr(verl, "__file__", "<unknown>"))
except Exception as exc:
    print("verl_check:", repr(exc))
PY

if [[ "${PREPARE_DEBUG_SPLIT}" == "1" ]]; then
  echo "== Preparing debug split for ${RECORD_ID} if needed =="
  "${PYTHON_BIN}" scripts/create_dataset_split.py \
    --input "${INPUT_PATH}" \
    --oracle_graph_dir "${ORACLE_GRAPH_DIR}" \
    --output "${DEBUG_SPLIT_PATH}" \
    --record_id "${RECORD_ID}" \
    --debug_single_case \
    --allow_empty_split \
    --overwrite
fi

read -r -a SEED_ARGS <<< "${SEEDS}"

READINESS_ARGS=(
  --trainer_backend "${TRAINER_BACKEND}"
  --output_dir "${OUTPUT_DIR}"
)

RUN_ARGS=(
  --suite_config "${SUITE_CONFIG}"
  --output_dir "${OUTPUT_DIR}"
  --seeds "${SEED_ARGS[@]}"
  --trainer_backend "${TRAINER_BACKEND}"
)

if [[ "${TRAINER_BACKEND}" == "verl" ]]; then
  READINESS_ARGS+=(--require_cuda)
  READINESS_ARGS+=(--memory_builder_model_path "${MEMORY_MODEL_PATH}")
  READINESS_ARGS+=(--question_agent_model_path "${QUESTION_MODEL_PATH}")
  READINESS_ARGS+=(--answerer_model_path "${ANSWERER_MODEL_PATH}")

  RUN_ARGS+=(--memory_builder_model_path "${MEMORY_MODEL_PATH}")
  RUN_ARGS+=(--question_agent_model_path "${QUESTION_MODEL_PATH}")
  RUN_ARGS+=(--answerer_model_path "${ANSWERER_MODEL_PATH}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  RUN_ARGS+=(--overwrite)
else
  RUN_ARGS+=(--resume)
fi

echo "== Checking Phase 4 backend readiness =="
"${PYTHON_BIN}" scripts/check_phase4_backend_readiness.py "${READINESS_ARGS[@]}"

echo "== Running Phase 4 benchmark suite =="
"${PYTHON_BIN}" scripts/run_phase4_benchmark_suite.py "${RUN_ARGS[@]}"

if [[ "${RUN_AGGREGATE}" == "1" ]]; then
  echo "== Aggregating Phase 4 benchmark suite =="
  "${PYTHON_BIN}" scripts/aggregate_phase4_benchmark_suite.py \
    --suite_dir "${OUTPUT_DIR}"
fi

if [[ "${RUN_RELEASE_PACKAGE}" == "1" ]]; then
  echo "== Building Phase 4 release package =="
  "${PYTHON_BIN}" scripts/build_phase4_release_package.py \
    --suite_dir "${OUTPUT_DIR}"
fi

if [[ "${RUN_ARTIFACT_VERIFY}" == "1" ]]; then
  echo "== Verifying Phase 4 release artifacts =="
  "${PYTHON_BIN}" scripts/verify_phase4_artifacts.py \
    --release_dir "${OUTPUT_DIR}/release" \
    --strict
fi

echo "== Done =="
echo "Suite output: ${OUTPUT_DIR}"
echo "Suite manifest: ${OUTPUT_DIR}/suite_manifest.json"
echo "Aggregate metrics: ${OUTPUT_DIR}/aggregate/aggregate_metrics.json"
echo "Release package: ${OUTPUT_DIR}/release"
