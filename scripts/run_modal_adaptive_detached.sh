#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-smoke}"
MODEL_NAME_OR_PATH="${2:-mistralai/Mistral-7B-Instruct-v0.2}"
GROUP_SIZE="${3:-64}"
RESIDUAL_LENGTH="${4:-128}"
SEGMENT_LENGTHS="${5:-2048,4096,8192}"
K_BITS_LIST="${6:-8,4,2}"
V_BITS_LIST="${7:-8,4,2}"
RUN_LONG_E="${8:-0}"
BATCH_SIZE="${9:-96}"
PROMPT_LENGTH="${10:-160}"
OUTPUT_LENGTH="${11:-338}"
NUM_REPEATS="${12:-3}"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/modal_${ACTION}_${STAMP}.log"

if [[ "${ACTION}" == "smoke" ]]; then
  CMD=(
    modal run --detach scripts/modal_adaptive_run.py::smoke_test
  )
elif [[ "${ACTION}" == "repro" ]]; then
  CMD=(
    modal run --detach scripts/modal_adaptive_run.py::run_adaptive_repro
    --model-name-or-path "${MODEL_NAME_OR_PATH}"
    --group-size "${GROUP_SIZE}"
    --residual-length "${RESIDUAL_LENGTH}"
    --segment-lengths "${SEGMENT_LENGTHS}"
    --k-bits-list "${K_BITS_LIST}"
    --v-bits-list "${V_BITS_LIST}"
  )

  if [[ "${RUN_LONG_E}" == "1" ]]; then
    CMD+=(--run-long-e)
  fi
elif [[ "${ACTION}" == "benchmark" ]]; then
  CMD=(
    modal run --detach scripts/modal_adaptive_run.py::run_adaptive_benchmark
    --model-name-or-path "${MODEL_NAME_OR_PATH}"
    --group-size "${GROUP_SIZE}"
    --residual-length "${RESIDUAL_LENGTH}"
    --segment-lengths "${SEGMENT_LENGTHS}"
    --k-bits-list "${K_BITS_LIST}"
    --v-bits-list "${V_BITS_LIST}"
    --batch-size "${BATCH_SIZE}"
    --prompt-length "${PROMPT_LENGTH}"
    --output-length "${OUTPUT_LENGTH}"
    --num-repeats "${NUM_REPEATS}"
  )
elif [[ "${ACTION}" == "longbench" ]]; then
  CMD=(
    modal run --detach scripts/modal_adaptive_run.py::run_adaptive_longbench
    --model-name-or-path "${MODEL_NAME_OR_PATH}"
    --group-size "${GROUP_SIZE}"
    --residual-length "${RESIDUAL_LENGTH}"
    --segment-lengths "${SEGMENT_LENGTHS}"
    --k-bits-list "${K_BITS_LIST}"
    --v-bits-list "${V_BITS_LIST}"
  )
else
  echo "Unsupported action: ${ACTION}" >&2
  exit 1
fi

echo "Launching detached Modal job..."
echo "Log: ${LOG_PATH}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
