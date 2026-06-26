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
# Examples:
#   # 1) Real e47becba smoke test, no custom suite config required:
#   bash scripts/run_remote_phase4_benchmark_suite.sh
#
#   # 2) Real full benchmark after you have created a real split manifest and
#   # copied configs/phase4/full_verl_suite.template.json to your own config:
#   MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   ANSWERER_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
#   TRAINER_BACKEND=verl \
#   SUITE_CONFIG=configs/phase4/full_verl_suite.json \
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

DEFAULT_SUITE_CONFIG="configs/phase4/debug_smoke_suite.json"
SUITE_CONFIG="${SUITE_CONFIG:-${DEFAULT_SUITE_CONFIG}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase4_benchmark_suites/debug_smoke}"
TRAINER_BACKEND="${TRAINER_BACKEND:-dry_run}"
SEEDS="${SEEDS:-0}"

MEMORY_MODEL_PATH="${MEMORY_MODEL_PATH:-}"
QUESTION_MODEL_PATH="${QUESTION_MODEL_PATH:-}"
ANSWERER_MODEL_PATH="${ANSWERER_MODEL_PATH:-}"

# Set INSTALL_DEPS=1 on a fresh server. Default is off so the script does not
# mutate a managed environment unexpectedly.
INSTALL_DEPS="${INSTALL_DEPS:-0}"

# Debug split preparation is useful only for the default e47becba smoke suite.
if [[ -z "${PREPARE_DEBUG_SPLIT:-}" ]]; then
  if [[ "${SUITE_CONFIG}" == "${DEFAULT_SUITE_CONFIG}" ]]; then
    PREPARE_DEBUG_SPLIT="1"
  else
    PREPARE_DEBUG_SPLIT="0"
  fi
fi

# Full split preparation is useful when ORACLE_GRAPH_DIR contains a subset of
# prebuilt oracle graphs and you want the suite to use only those record IDs.
PREPARE_FULL_SPLIT="${PREPARE_FULL_SPLIT:-0}"
FULL_SPLIT_PATH="${FULL_SPLIT_PATH:-outputs/splits/longmemeval_s.phase4_split.seed0.json}"
SPLIT_SEED="${SPLIT_SEED:-0}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
DEV_RATIO="${DEV_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"

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
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

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

install_vendored_verl_requirements() {
  local requirements_file="$1"
  local filtered_requirements
  local flash_attn_requirements

  filtered_requirements="$(mktemp)"
  flash_attn_requirements="$(mktemp)"
  trap 'rm -f "${filtered_requirements}" "${flash_attn_requirements}"' RETURN

  "${PYTHON_BIN}" - "${requirements_file}" "${filtered_requirements}" "${flash_attn_requirements}" <<'PY'
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
flash_dst = pathlib.Path(sys.argv[3])

kept = []
flash_attn_specs = []
for raw_line in src.read_text(encoding="utf-8").splitlines():
    stripped = raw_line.strip()
    normalized = stripped.split("#", 1)[0].strip().lower().replace("_", "-")
    if (
        normalized == "flash-attn"
        or normalized.startswith("flash-attn==")
        or normalized.startswith("flash-attn>")
        or normalized.startswith("flash-attn<")
    ):
        flash_attn_specs.append(stripped)
    else:
        kept.append(raw_line)

dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
flash_dst.write_text(
    "\n".join(flash_attn_specs) + ("\n" if flash_attn_specs else ""),
    encoding="utf-8",
)
PY

  echo "== Installing vendored verl dependencies except flash-attn =="
  "${PYTHON_BIN}" -m pip install -r "${filtered_requirements}"

  if [[ -s "${flash_attn_requirements}" ]]; then
    echo "== Installing flash-attn with --no-build-isolation =="
    "${PYTHON_BIN}" - <<'PY'
import torch

print("torch_available_for_flash_attn_build:", torch.__version__, "cuda", torch.version.cuda)
PY
    "${PYTHON_BIN}" -m pip install -r "${flash_attn_requirements}" --no-build-isolation
  fi
}

echo "== GAAM Phase 4 benchmark-suite script =="
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "SUITE_CONFIG=${SUITE_CONFIG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRAINER_BACKEND=${TRAINER_BACKEND}"
echo "SEEDS=${SEEDS}"
echo "PREPARE_DEBUG_SPLIT=${PREPARE_DEBUG_SPLIT}"
echo "PREPARE_FULL_SPLIT=${PREPARE_FULL_SPLIT}"

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
    install_vendored_verl_requirements "${VERL_ROOT}/requirements.txt"
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

if [[ "${PREPARE_FULL_SPLIT}" == "1" ]]; then
  echo "== Preparing full split from oracle graphs =="
  mapfile -t GRAPH_RECORD_IDS < <(
    find "${ORACLE_GRAPH_DIR}" -maxdepth 1 -type f -name '*.graph.json' \
      -exec basename {} .graph.json \; | sort
  )
  if [[ "${#GRAPH_RECORD_IDS[@]}" -lt 3 ]]; then
    echo "Need at least 3 oracle graphs for train/dev/test split, found ${#GRAPH_RECORD_IDS[@]}." >&2
    exit 1
  fi

  SPLIT_ARGS=(
    scripts/create_dataset_split.py
    --input "${INPUT_PATH}"
    --oracle_graph_dir "${ORACLE_GRAPH_DIR}"
    --output "${FULL_SPLIT_PATH}"
    --train_ratio "${TRAIN_RATIO}"
    --dev_ratio "${DEV_RATIO}"
    --test_ratio "${TEST_RATIO}"
    --seed "${SPLIT_SEED}"
    --overwrite
  )
  for rid in "${GRAPH_RECORD_IDS[@]}"; do
    SPLIT_ARGS+=(--record_id "${rid}")
  done
  "${PYTHON_BIN}" "${SPLIT_ARGS[@]}"
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
set +e
"${PYTHON_BIN}" scripts/run_phase4_benchmark_suite.py "${RUN_ARGS[@]}"
SUITE_EXIT_CODE=$?
set -e

if [[ "${SUITE_EXIT_CODE}" -ne 0 ]]; then
  echo "== Phase 4 benchmark suite failed =="
  if [[ -f "${OUTPUT_DIR}/suite_manifest.json" ]]; then
    "${PYTHON_BIN}" - "${OUTPUT_DIR}/suite_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
print(f"Suite status: {manifest.get('status')}")
print(f"Manifest: {manifest_path}")
print("")
for run in manifest.get("run_records", []):
    print(f"seed={run.get('seed')} status={run.get('status')}")
    errors = run.get("errors") or []
    warnings = run.get("warnings") or []
    experiment_manifest_path = run.get("experiment_manifest_path")
    if errors:
        print("  errors:")
        for error in errors:
            print(f"    - {error}")
    if warnings:
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    if experiment_manifest_path:
        print(f"  experiment_manifest: {experiment_manifest_path}")
    print(f"  output_dir: {run.get('output_dir')}")
    print("")
PY
  else
    echo "Suite manifest was not written: ${OUTPUT_DIR}/suite_manifest.json" >&2
  fi
  exit "${SUITE_EXIT_CODE}"
fi

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
