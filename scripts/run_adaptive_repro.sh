#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 9 ]; then
  echo "Usage: $0 GPU_ID MODEL_NAME_OR_PATH GROUP_SIZE RESIDUAL_LENGTH SEGMENT_LENGTHS K_BITS_LIST V_BITS_LIST RUN_LONG_E CACHE_DIR"
  echo "Example:"
  echo "  $0 0 meta-llama/Llama-3.1-8B-Instruct 64 128 2048,4096,8192 8,4,2 8,4,2 0 ./cached_models"
  exit 1
fi

GPU_ID="$1"
MODEL_NAME_OR_PATH="$2"
GROUP_SIZE="$3"
RESIDUAL_LENGTH="$4"
SEGMENT_LENGTHS="$5"
K_BITS_LIST="$6"
V_BITS_LIST="$7"
RUN_LONG_E="$8"
CACHE_DIR="$9"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p outputs pred pred_e

MODEL_NAME="$(basename "${MODEL_NAME_OR_PATH}")"
MAX_LENGTH="$(python - <<PY
import json
from utils.adaptive_kv import resolve_model_max_length
model_name_or_path = "${MODEL_NAME_OR_PATH}"
with open("config/model2maxlen.json", "r") as f:
    _, max_length = resolve_model_max_length(model_name_or_path, json.load(f))
    print(max_length)
PY
)"

EXP_TAG="$(python - <<PY
from types import SimpleNamespace
from utils.adaptive_kv import adaptive_experiment_tag
args = SimpleNamespace(
    adaptive_kv=True,
    adaptive_policy="static_distance",
    adaptive_segment_lengths="${SEGMENT_LENGTHS}",
    adaptive_k_bits="${K_BITS_LIST}",
    adaptive_v_bits="${V_BITS_LIST}",
    group_size=int("${GROUP_SIZE}"),
    residual_length=int("${RESIDUAL_LENGTH}"),
    k_bits=2,
    v_bits=2,
)
print(adaptive_experiment_tag(args))
PY
)"

PRED_DIR="pred"
E_FLAG=""
if [ "${RUN_LONG_E}" = "1" ]; then
  PRED_DIR="pred_e"
  E_FLAG="--e"
fi

MODEL_TAG="${MODEL_NAME}_${MAX_LENGTH}_${EXP_TAG}"

echo "[1/3] Adaptive memory/speed benchmark"
python scripts/run_adaptive_mem_speed.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --cache_dir "${CACHE_DIR}" \
  --batch_size 96 \
  --prompt_length 160 \
  --output_length 338 \
  --num_repeats 3 \
  --group_size "${GROUP_SIZE}" \
  --residual_length "${RESIDUAL_LENGTH}" \
  --use_flash_attention_2 false \
  --adaptive_kv \
  --adaptive_policy static_distance \
  --adaptive_segment_lengths "${SEGMENT_LENGTHS}" \
  --adaptive_k_bits "${K_BITS_LIST}" \
  --adaptive_v_bits "${V_BITS_LIST}" \
  --out_json "outputs/adaptive_mem_speed.json"

echo "[2/3] Adaptive LongBench prediction"
python pred_long_bench_adaptive.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --cache_dir "${CACHE_DIR}" \
  --group_size "${GROUP_SIZE}" \
  --residual_length "${RESIDUAL_LENGTH}" \
  --use_flash_attention_2 false \
  --adaptive_kv \
  --adaptive_policy static_distance \
  --adaptive_segment_lengths "${SEGMENT_LENGTHS}" \
  --adaptive_k_bits "${K_BITS_LIST}" \
  --adaptive_v_bits "${V_BITS_LIST}" \
  ${E_FLAG}

echo "[3/3] LongBench evaluation"
python eval_long_bench.py --model "${MODEL_TAG}" ${E_FLAG}

RESULT_PATH="${PRED_DIR}/${MODEL_TAG}/result.json"
if [ -f "${RESULT_PATH}" ]; then
  cp "${RESULT_PATH}" outputs/adaptive_longbench_result.json
fi

echo "Adaptive run complete"
echo "  Memory/speed: outputs/adaptive_mem_speed.json"
echo "  Metrics:      outputs/adaptive_longbench_result.json"
