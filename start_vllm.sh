#!/usr/bin/env bash
set -euo pipefail

# WSL2에서 io_uring 문제로 Uvicorn accept이 안 되는 현상 방지
export UVICORN_NO_IOURING=1
export UV_USE_IO_URING=0

MODEL_DIR="${MODEL_DIR:-/data/public/qwen3.5/Qwen3.5-27B-FP8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8015}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B-FP8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-200000}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_BIN="${VLLM_BIN:-vllm}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "Model config not found: ${MODEL_DIR}/config.json" >&2
  exit 1
fi

ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${MODEL_NAME}"
  --api-key "${API_KEY}"
  --tensor-parallel-size 1
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --reasoning-parser "${REASONING_PARSER}"
  --enforce-eager
)

if [ "${ENABLE_AUTO_TOOL_CHOICE}" != "0" ]; then
  ARGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi

exec "${VLLM_BIN}" serve "${MODEL_DIR}" "${ARGS[@]}"
