#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 MODE MODEL [vLLM arguments...]" >&2
    echo "MODE: stock|shadow|active-validation|controller|sjf256|sjf512" >&2
}

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

readonly MODE="$1"
readonly MODEL="$2"
shift 2
readonly ROOT="${VIDUR_MEW1_ROOT:-/home/shaz/vidur}"
readonly SOURCE="${VIDUR_REAL_SOURCE_ROOT:-$ROOT/source/vidur-classical-search}"
readonly ENV_DIR="${VIDUR_REAL_ENV_DIR:-$ROOT/envs/vllm-0.13.0-cu129}"
readonly CUDA_HOME="${VIDUR_REAL_CUDA_HOME:-/usr/local/cuda-13.0}"
readonly PHYSICAL_GPU="${VIDUR_PHYSICAL_GPU:-2}"
readonly PORT="${VIDUR_VLLM_PORT:-8000}"

if [[ "$(hostname -s)" != "mew1" || "$(id -un)" != "shaz" ]]; then
    echo "this launcher is restricted to shaz@mew1" >&2
    exit 2
fi
if [[ "$PHYSICAL_GPU" != "2" && "$PHYSICAL_GPU" != "3" ]]; then
    echo "VIDUR_PHYSICAL_GPU must be 2 or 3" >&2
    exit 2
fi
case "$MODE" in
    stock|shadow|active-validation|controller|sjf256|sjf512) ;;
    *) usage; exit 2 ;;
esac

active_processes="$(nvidia-smi -i "$PHYSICAL_GPU" --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_processes" ]]; then
    echo "GPU $PHYSICAL_GPU is in use; refusing to launch" >&2
    exit 2
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:/home/shaz/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SOURCE"
export HF_HOME="${VIDUR_REAL_HF_HOME:-$ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$ROOT/cache/torch"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export VIDUR_PACKAGE_ROOT="$SOURCE/vidur_vllm_real_testing"
export VIDUR_MODEL_TOKENIZER="$MODEL"
export VIDUR_VLLM_CANONICAL_TRACE="${VIDUR_VLLM_CANONICAL_TRACE:-$SOURCE/vidur_vllm_real_testing/traces/splitwise_conv_20s_english_canonical.csv}"
export VIDUR_VLLM_SCHEDULER_LOG="${VIDUR_VLLM_SCHEDULER_LOG:-$ROOT/runs/scheduler-${MODE}-gpu${PHYSICAL_GPU}.jsonl}"

mkdir -p "$ROOT/runs"
exec bash "$SOURCE/vidur_vllm_real_testing/docker/production_entrypoint.sh" \
    "serve-$MODE" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --scheduling-policy fcfs \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    "$@"
