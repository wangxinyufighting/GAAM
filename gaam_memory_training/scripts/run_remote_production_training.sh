#!/usr/bin/env bash
set -euo pipefail

# GAAM remote production launcher.
#
# This is the highest-level command intended for a remote GPU server. It:
#   1. loads a sourceable env file such as production.env;
#   2. optionally writes a remote diagnostics report without training;
#   3. launches the production E2E training/evaluation/audit/bundle pipeline.
#
# Typical usage:
#
#   cp production.env.example production.env
#   # edit production.env with model paths, API keys, GPU counts, split settings
#   bash scripts/run_remote_production_training.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Env file to load before running diagnostics/training. Keep real API keys in
# this untracked file or in the server environment.
ENV_FILE="${ENV_FILE:-production.env}"

# Python executable inside the remote training environment.
PYTHON_BIN="${PYTHON_BIN:-python}"

# If 1, source ENV_FILE before running diagnostics/training.
LOAD_ENV_FILE="${LOAD_ENV_FILE:-1}"

# If 1, run scripts/install_remote_production_deps.sh before diagnostics. Keep
# this off by default so a managed remote environment is not mutated unless you
# explicitly request installation.
INSTALL_DEPS_BEFORE_TRAINING="${INSTALL_DEPS_BEFORE_TRAINING:-0}"

# If 1, collect a remote diagnostics report before launching training. This
# records package versions, CUDA status, nvidia-smi output, redacted env values,
# and production preflight results.
RUN_DIAGNOSTICS_BEFORE_TRAINING="${RUN_DIAGNOSTICS_BEFORE_TRAINING:-1}"

# Where the diagnostics JSON report is written.
DIAGNOSTICS_OUTPUT="${DIAGNOSTICS_OUTPUT:-outputs/remote_training_diagnostics.json}"

# If 1, continue to training even if diagnostics fail. Default 0 prevents
# expensive training from starting when environment/data/config are not ready.
ALLOW_FAILED_DIAGNOSTICS="${ALLOW_FAILED_DIAGNOSTICS:-0}"

# If 1, launch the production E2E pipeline after diagnostics.
RUN_TRAINING="${RUN_TRAINING:-1}"

if [[ "${LOAD_ENV_FILE}" == "1" || "${LOAD_ENV_FILE}" == "True" || "${LOAD_ENV_FILE}" == "true" ]]; then
  if [[ -f "${ENV_FILE}" ]]; then
    echo "== Loading GAAM production env: ${ENV_FILE} =="
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
  else
    echo "ENV_FILE was requested but not found: ${ENV_FILE}" >&2
    echo "Create it from production.env.example, or set LOAD_ENV_FILE=0." >&2
    exit 1
  fi
fi

echo "== GAAM remote production launcher =="
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "ENV_FILE=${ENV_FILE}"
echo "INSTALL_DEPS_BEFORE_TRAINING=${INSTALL_DEPS_BEFORE_TRAINING}"
echo "RUN_DIAGNOSTICS_BEFORE_TRAINING=${RUN_DIAGNOSTICS_BEFORE_TRAINING}"
echo "DIAGNOSTICS_OUTPUT=${DIAGNOSTICS_OUTPUT}"
echo "ALLOW_FAILED_DIAGNOSTICS=${ALLOW_FAILED_DIAGNOSTICS}"
echo "RUN_TRAINING=${RUN_TRAINING}"

if [[ "${INSTALL_DEPS_BEFORE_TRAINING}" == "1" || "${INSTALL_DEPS_BEFORE_TRAINING}" == "True" || "${INSTALL_DEPS_BEFORE_TRAINING}" == "true" ]]; then
  echo "== Installing remote production dependencies =="
  bash scripts/install_remote_production_deps.sh
fi

DIAGNOSTICS_STATUS=0
if [[ "${RUN_DIAGNOSTICS_BEFORE_TRAINING}" == "1" || "${RUN_DIAGNOSTICS_BEFORE_TRAINING}" == "True" || "${RUN_DIAGNOSTICS_BEFORE_TRAINING}" == "true" ]]; then
  echo "== Collecting remote training diagnostics =="
  "${PYTHON_BIN}" scripts/collect_remote_training_diagnostics.py \
    --output "${DIAGNOSTICS_OUTPUT}" || DIAGNOSTICS_STATUS=$?
  if [[ "${DIAGNOSTICS_STATUS}" != "0" && ! ( "${ALLOW_FAILED_DIAGNOSTICS}" == "1" || "${ALLOW_FAILED_DIAGNOSTICS}" == "True" || "${ALLOW_FAILED_DIAGNOSTICS}" == "true" ) ]]; then
    echo "Diagnostics failed. Fix ${DIAGNOSTICS_OUTPUT}, or set ALLOW_FAILED_DIAGNOSTICS=1 to continue anyway." >&2
    exit "${DIAGNOSTICS_STATUS}"
  fi
fi

if [[ "${RUN_TRAINING}" == "1" || "${RUN_TRAINING}" == "True" || "${RUN_TRAINING}" == "true" ]]; then
  echo "== Launching production E2E training/evaluation =="
  "${PYTHON_BIN}" - <<'PY'
import sys
print("python:", sys.version)
print("executable:", sys.executable)
PY
  bash scripts/run_production_e2e_training_and_evaluation.sh
fi

echo "== GAAM remote production launcher complete =="
if [[ "${RUN_DIAGNOSTICS_BEFORE_TRAINING}" == "1" || "${RUN_DIAGNOSTICS_BEFORE_TRAINING}" == "True" || "${RUN_DIAGNOSTICS_BEFORE_TRAINING}" == "true" ]]; then
  echo "Diagnostics: ${DIAGNOSTICS_OUTPUT}"
fi
