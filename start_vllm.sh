#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/public/qwen3.5/Qwen3.5-27B-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B-FP8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-200000}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_BIN="${VLLM_BIN:-vllm}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"

# GPU 수 자동 감지
if [ -z "${TP_SIZE:-}" ]; then
  TP_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
  [ "${TP_SIZE}" -lt 1 ] && TP_SIZE=1
  echo "[start_vllm] GPU 자동 감지: ${TP_SIZE}개"
fi

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "Model config not found: ${MODEL_DIR}/config.json" >&2
  exit 1
fi

ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${MODEL_NAME}"
  --api-key "${API_KEY}"
  --tensor-parallel-size "${TP_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --reasoning-parser "${REASONING_PARSER}"
  --enforce-eager
)

if [ "${ENABLE_AUTO_TOOL_CHOICE}" != "0" ]; then
  ARGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi

exec "${VLLM_BIN}" serve "${MODEL_DIR}" "${ARGS[@]}"
