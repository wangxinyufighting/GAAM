#!/usr/bin/env bash
set -euo pipefail

# Bulk oracle graph construction for LongMemEval.
#
# Token-saving defaults:
# - batch event analysis: one LLM call per EVENT_BATCH_SIZE events
# - heuristic fact relation detection: no relation LLM calls
# - heuristic abstract memories: no abstraction LLM calls
# - skip existing graphs: safe resume
#
# Examples:
#   # Build first 20 records with LLM event extraction:
#   MAX_RECORDS=20 bash scripts/run_build_oracle_graphs_bulk.sh
#
#   # Build all missing records:
#   MAX_RECORDS= bash scripts/run_build_oracle_graphs_bulk.sh
#
#   # Dry-run without API cost:
#   USE_LLM=0 MAX_RECORDS=5 bash scripts/run_build_oracle_graphs_bulk.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/longmemeval_s_graph}"
MAX_RECORDS="${MAX_RECORDS:-20}"
USE_LLM="${USE_LLM:-1}"
EVENT_BATCH_SIZE="${EVENT_BATCH_SIZE:-20}"
RELATION_MODE="${RELATION_MODE:-heuristic}"
ABSTRACT_MODE="${ABSTRACT_MODE:-heuristic}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ "${OVERWRITE}" == "1" ]]; then
  SKIP_EXISTING="0"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

args=(
  scripts/build_lme_graph.py
  --input "${INPUT_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --event_batch_size "${EVENT_BATCH_SIZE}"
  --relation_mode "${RELATION_MODE}"
  --abstract_mode "${ABSTRACT_MODE}"
)

if [[ -n "${MAX_RECORDS}" ]]; then
  args+=(--max_records "${MAX_RECORDS}")
fi

if [[ "${USE_LLM}" == "1" ]]; then
  args+=(--llm)
else
  args+=(--no-llm)
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  args+=(--skip_existing)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

echo "== GAAM bulk oracle graph builder =="
echo "INPUT_PATH=${INPUT_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_RECORDS=${MAX_RECORDS:-all}"
echo "USE_LLM=${USE_LLM}"
echo "EVENT_BATCH_SIZE=${EVENT_BATCH_SIZE}"
echo "RELATION_MODE=${RELATION_MODE}"
echo "ABSTRACT_MODE=${ABSTRACT_MODE}"
echo "OVERWRITE=${OVERWRITE}"
echo "SKIP_EXISTING=${SKIP_EXISTING}"
echo "PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" "${args[@]}"

echo "== Done =="
echo "Graphs: ${OUTPUT_DIR}"
echo "Manifest: ${OUTPUT_DIR}/manifest.jsonl"
