#!/usr/bin/env bash
set -euo pipefail

# Remote Phase 3 training/conversion entrypoint with Code-A1 vendored verl.
#
# This script runs:
#   1. environment checks;
#   2. optional dependency installation;
#   3. local fallback dry-run smoke test;
#   4. Code-A1/verl-oriented Phase 3 E2E run;
#   5. strict inspection of both runs.
#
# Usage:
#   bash scripts/run_remote_phase3_verl_training.sh
#
# Common overrides:
#   MEMORY_MODEL_PATH=/models/Qwen3-0.6B \
#   QUESTION_MODEL_PATH=/models/Qwen3-0.6B \
#   ANSWERER_MODEL_PATH=/models/Qwen3-0.6B \
#   bash scripts/run_remote_phase3_verl_training.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_PATH="${INPUT_PATH:-data/longmemeval/longmemeval_s_cleaned.json}"
ORACLE_GRAPH_DIR="${ORACLE_GRAPH_DIR:-outputs/longmemeval_s_graph}"
RECORD_ID="${RECORD_ID:-e47becba}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/remote_phase3_verl}"
DRY_RUN_OUTPUT_DIR="${DRY_RUN_OUTPUT_DIR:-${OUTPUT_ROOT}/dry_run_smoke}"
VERL_OUTPUT_DIR="${VERL_OUTPUT_DIR:-${OUTPUT_ROOT}/code_a1_verl_run}"

MEMORY_MODEL_PATH="${MEMORY_MODEL_PATH:-/path/to/Qwen3-0.6B}"
QUESTION_MODEL_PATH="${QUESTION_MODEL_PATH:-/path/to/Qwen3-0.6B}"
ANSWERER_MODEL_PATH="${ANSWERER_MODEL_PATH:-/path/to/Qwen3-0.6B}"

NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS:-4}"
NUM_REWARD_WORKERS="${NUM_REWARD_WORKERS:-2}"
ROUND_ID="${ROUND_ID:-0}"
STEP_ID="${STEP_ID:-0}"
SEED="${SEED:-0}"

# Set INSTALL_DEPS=1 on a fresh remote server. Keep default off so the script
# does not mutate a managed environment unexpectedly.
INSTALL_DEPS="${INSTALL_DEPS:-0}"

# Set RUN_DRY_SMOKE=0 if you already verified the server and want to run only
# the Code-A1/verl path.
RUN_DRY_SMOKE="${RUN_DRY_SMOKE:-1}"

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

# OpenMP guards for CI, containers, and restricted /dev/shm environments.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"

export PYTHONPATH="${ROOT_DIR}:${CODE_A1_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"

echo "== GAAM Phase 3 Code-A1/verl training script =="
echo "ROOT_DIR=${ROOT_DIR}"
echo "CODE_A1_ROOT=${CODE_A1_ROOT}"
echo "VERL_ROOT=${VERL_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "RECORD_ID=${RECORD_ID}"

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
    if normalized == "flash-attn" or normalized.startswith("flash-attn==") or normalized.startswith("flash-attn>") or normalized.startswith("flash-attn<"):
        flash_attn_specs.append(stripped)
    else:
        kept.append(raw_line)

dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
flash_dst.write_text("\n".join(flash_attn_specs) + ("\n" if flash_attn_specs else ""), encoding="utf-8")
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

require_file "${INPUT_PATH}"
require_dir "${ORACLE_GRAPH_DIR}"
require_file "${ORACLE_GRAPH_DIR}/${RECORD_ID}.graph.json"
require_dir "${CODE_A1_ROOT}"
require_dir "${VERL_ROOT}"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  echo "== Installing GAAM dependencies =="
  "${PYTHON_BIN}" -m pip install -r requirements.txt
  "${PYTHON_BIN}" -m pip install -r requirements-local-training.txt

  echo "== Installing optional Ray dependency =="
  "${PYTHON_BIN}" -m pip install "ray>=2.9"

  if [[ -f "${VERL_ROOT}/requirements.txt" ]]; then
    echo "== Installing vendored verl dependencies =="
    install_vendored_verl_requirements "${VERL_ROOT}/requirements.txt"
  fi
fi

echo "== Python / CUDA / verl check =="
"${PYTHON_BIN}" - <<'PY'
import os
import sys
import traceback

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
    import verl
    print("verl_package:", getattr(verl, "__file__", "<unknown>"))
    from verl import protocol
    print("verl_protocol:", getattr(protocol, "__file__", "<unknown>"))
except Exception:
    print("verl_protocol_import_failed:")
    traceback.print_exc()
    raise SystemExit(
        "verl.protocol is not importable. Install vendored verl dependencies "
        "or set CODE_A1_ROOT/VERL_ROOT/PYTHONPATH correctly."
    )
PY

if [[ "${RUN_DRY_SMOKE}" == "1" ]]; then
  echo "== Running local fallback dry-run smoke =="
  "${PYTHON_BIN}" scripts/run_phase3_end_to_end_grpo.py \
    --input "${INPUT_PATH}" \
    --oracle_graph_dir "${ORACLE_GRAPH_DIR}" \
    --record_id "${RECORD_ID}" \
    --output_dir "${DRY_RUN_OUTPUT_DIR}" \
    --backend local_fallback \
    --trainer_mode dry_run \
    --round_id "${ROUND_ID}" \
    --step_id "${STEP_ID}" \
    --seed "${SEED}" \
    --overwrite

  echo "== Inspecting dry-run smoke =="
  "${PYTHON_BIN}" scripts/inspect_phase3_e2e_run.py \
    --run_dir "${DRY_RUN_OUTPUT_DIR}" \
    --strict
fi

echo "== Running Code-A1/verl-oriented Phase 3 run =="
"${PYTHON_BIN}" scripts/run_phase3_end_to_end_grpo.py \
  --input "${INPUT_PATH}" \
  --oracle_graph_dir "${ORACLE_GRAPH_DIR}" \
  --record_id "${RECORD_ID}" \
  --output_dir "${VERL_OUTPUT_DIR}" \
  --backend code_a1_verl \
  --trainer_mode vendored_verl \
  --memory_builder_model_path "${MEMORY_MODEL_PATH}" \
  --question_agent_model_path "${QUESTION_MODEL_PATH}" \
  --answerer_model_path "${ANSWERER_MODEL_PATH}" \
  --num_rollout_workers "${NUM_ROLLOUT_WORKERS}" \
  --num_reward_workers "${NUM_REWARD_WORKERS}" \
  --round_id "${ROUND_ID}" \
  --step_id "${STEP_ID}" \
  --seed "${SEED}" \
  --write_verl_dataproto \
  --use_vendored_verl_if_available \
  --overwrite

echo "== Inspecting Code-A1/verl run =="
"${PYTHON_BIN}" scripts/inspect_phase3_e2e_run.py \
  --run_dir "${VERL_OUTPUT_DIR}" \
  --strict

echo "== Done =="
echo "Dry-run output: ${DRY_RUN_OUTPUT_DIR}"
echo "Code-A1/verl output: ${VERL_OUTPUT_DIR}"
