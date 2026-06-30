#!/usr/bin/env bash
set -euo pipefail

# GAAM remote production dependency installer.
#
# This script is intentionally optional. It mutates the active Python
# environment, so run it only inside a dedicated conda/venv on the remote GPU
# server. It installs GAAM runtime dependencies, optional local/HF training
# dependencies, optional vLLM, and Code-A1 vendored VERL dependencies with
# flash-attn handled separately.
#
# Typical usage:
#
#   conda activate gaam-verl
#   bash scripts/install_remote_production_deps.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Python executable in the target environment.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Code-A1 checkout root. Its vendored VERL requirements are installed when
# INSTALL_VERL_DEPS=1.
CODE_A1_ROOT="${CODE_A1_ROOT:-${REPO_DIR}/Code-A1/Code-A1}"
VERL_ROOT="${VERL_ROOT:-${CODE_A1_ROOT}/verl}"

# Install GAAM core runtime dependencies from requirements.txt.
INSTALL_GAAM_DEPS="${INSTALL_GAAM_DEPS:-1}"

# Install local/HF training dependencies from requirements-local-training.txt.
# This file includes torch/transformers/ray. Install the correct CUDA PyTorch
# build before this script when your server needs a specific wheel index.
INSTALL_LOCAL_TRAINING_DEPS="${INSTALL_LOCAL_TRAINING_DEPS:-1}"

# Install Code-A1 vendored VERL requirements except flash-attn first.
INSTALL_VERL_DEPS="${INSTALL_VERL_DEPS:-1}"

# Install vLLM explicitly. Code-A1 currently expects vLLM 0.8.4 for native VERL
# rollout.
INSTALL_VLLM="${INSTALL_VLLM:-1}"
VLLM_VERSION="${VLLM_VERSION:-0.8.4}"

# Install flash-attn after PyTorch is installed/importable.
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn}"

# If 1, use --no-build-isolation when installing flash-attn.
FLASH_ATTN_NO_BUILD_ISOLATION="${FLASH_ATTN_NO_BUILD_ISOLATION:-1}"

# If 1, run pip check at the end.
RUN_PIP_CHECK="${RUN_PIP_CHECK:-1}"

echo "== GAAM remote production dependency installer =="
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CODE_A1_ROOT=${CODE_A1_ROOT}"
echo "VERL_ROOT=${VERL_ROOT}"
echo "INSTALL_GAAM_DEPS=${INSTALL_GAAM_DEPS}"
echo "INSTALL_LOCAL_TRAINING_DEPS=${INSTALL_LOCAL_TRAINING_DEPS}"
echo "INSTALL_VERL_DEPS=${INSTALL_VERL_DEPS}"
echo "INSTALL_VLLM=${INSTALL_VLLM}"
echo "VLLM_VERSION=${VLLM_VERSION}"
echo "INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN}"

if [[ "${INSTALL_GAAM_DEPS}" == "1" || "${INSTALL_GAAM_DEPS}" == "True" || "${INSTALL_GAAM_DEPS}" == "true" ]]; then
  echo "== Installing GAAM runtime dependencies =="
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

if [[ "${INSTALL_LOCAL_TRAINING_DEPS}" == "1" || "${INSTALL_LOCAL_TRAINING_DEPS}" == "True" || "${INSTALL_LOCAL_TRAINING_DEPS}" == "true" ]]; then
  echo "== Installing local/HF training dependencies =="
  "${PYTHON_BIN}" -m pip install -r requirements-local-training.txt
fi

install_vendored_verl_requirements() {
  local requirements_file="$1"
  local filtered_requirements
  local flash_attn_requirements

  filtered_requirements="$(mktemp)"
  flash_attn_requirements="$(mktemp)"
  trap 'rm -f "${filtered_requirements}" "${flash_attn_requirements}"' RETURN

  "${PYTHON_BIN}" - "${requirements_file}" "${filtered_requirements}" "${flash_attn_requirements}" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
flash_dst = Path(sys.argv[3])

kept = []
flash_lines = []
for raw_line in src.read_text(encoding="utf-8").splitlines():
    stripped = raw_line.strip()
    normalized = stripped.split("#", 1)[0].strip().lower().replace("_", "-")
    if normalized == "flash-attn" or normalized.startswith("flash-attn==") or normalized.startswith("flash-attn>") or normalized.startswith("flash-attn<"):
        flash_lines.append(stripped)
    else:
        kept.append(raw_line)

dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
flash_dst.write_text("\n".join(flash_lines) + ("\n" if flash_lines else ""), encoding="utf-8")
print(dst)
if flash_lines:
    print(flash_dst)
PY

  echo "== Installing Code-A1/VERL dependencies except flash-attn =="
  "${PYTHON_BIN}" -m pip install -r "${filtered_requirements}"
}

if [[ "${INSTALL_VERL_DEPS}" == "1" || "${INSTALL_VERL_DEPS}" == "True" || "${INSTALL_VERL_DEPS}" == "true" ]]; then
  if [[ ! -f "${VERL_ROOT}/requirements.txt" ]]; then
    echo "Missing VERL requirements file: ${VERL_ROOT}/requirements.txt" >&2
    exit 1
  fi
  install_vendored_verl_requirements "${VERL_ROOT}/requirements.txt"
fi

if [[ "${INSTALL_VLLM}" == "1" || "${INSTALL_VLLM}" == "True" || "${INSTALL_VLLM}" == "true" ]]; then
  echo "== Installing vLLM =="
  "${PYTHON_BIN}" -m pip install "vllm==${VLLM_VERSION}"
fi

if [[ "${INSTALL_FLASH_ATTN}" == "1" || "${INSTALL_FLASH_ATTN}" == "True" || "${INSTALL_FLASH_ATTN}" == "true" ]]; then
  echo "== Checking PyTorch before flash-attn =="
  "${PYTHON_BIN}" - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cxx11_abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
PY
  echo "== Installing flash-attn =="
  if [[ "${FLASH_ATTN_NO_BUILD_ISOLATION}" == "1" || "${FLASH_ATTN_NO_BUILD_ISOLATION}" == "True" || "${FLASH_ATTN_NO_BUILD_ISOLATION}" == "true" ]]; then
    "${PYTHON_BIN}" -m pip install "${FLASH_ATTN_SPEC}" --no-build-isolation
  else
    "${PYTHON_BIN}" -m pip install "${FLASH_ATTN_SPEC}"
  fi
fi

if [[ "${RUN_PIP_CHECK}" == "1" || "${RUN_PIP_CHECK}" == "True" || "${RUN_PIP_CHECK}" == "true" ]]; then
  echo "== Running pip check =="
  "${PYTHON_BIN}" -m pip check
fi

echo "== Dependency installation complete =="
