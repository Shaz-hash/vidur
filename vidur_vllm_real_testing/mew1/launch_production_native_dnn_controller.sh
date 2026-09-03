#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MODEL [vLLM arguments...]" >&2
    exit 2
fi
readonly MODEL="$1"
shift
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
if [[ -n "$(nvidia-smi -i "$PHYSICAL_GPU" --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; then
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
export VIDUR_VLLM_SCHEDULER_MODE=controller
export VIDUR_VLLM_CANONICAL_TRACE="${VIDUR_VLLM_CANONICAL_TRACE:-$SOURCE/vidur_vllm_real_testing/traces/splitwise_conv_20s_english_canonical.csv}"
export VIDUR_VLLM_SCHEDULER_LOG="${VIDUR_VLLM_SCHEDULER_LOG:-$ROOT/runs/scheduler-controller-native-dnn-gpu${PHYSICAL_GPU}.jsonl}"

export VIDUR_VLLM_GV3_PLANNER="vidur_vllm_real_testing.gv3_adapter:GV3PersistentAdapter"
export VIDUR_VLLM_GV3_STATE_PLANNER="${VIDUR_VLLM_GV3_STATE_PLANNER:-vidur_vllm_real_testing.native_dnn_mcts_state_planner:ProductionNativeDNNMCTSPlanner}"
export VIDUR_VLLM_GV3_MODEL_BUNDLE="${VIDUR_VLLM_GV3_MODEL_BUNDLE:-$ROOT/artifacts/promoted_models/xl3_controller_v197_adv_v212}"
export VIDUR_VLLM_GV3_NATIVE_CFG="${VIDUR_VLLM_GV3_NATIVE_CFG:-$SOURCE/vidur_vllm_real_testing/artifacts/native_mcts_cfg.json}"
export VIDUR_VLLM_GV3_MCTS_ITERATIONS="${VIDUR_VLLM_GV3_MCTS_ITERATIONS:-2000}"
export VIDUR_VLLM_GV3_DISCOUNT_FACTOR="${VIDUR_VLLM_GV3_DISCOUNT_FACTOR:-0.98}"
export VIDUR_VLLM_GV3_PUCT_C="${VIDUR_VLLM_GV3_PUCT_C:-0.5}"
export VIDUR_VLLM_GV3_UCT_C="${VIDUR_VLLM_GV3_UCT_C:-1.0}"
export VIDUR_VLLM_GV3_NATIVE_SEARCH_MODE="${VIDUR_VLLM_GV3_NATIVE_SEARCH_MODE:-full_tree_rollout}"
export VIDUR_VLLM_GV3_ROLLOUT_COUNT="${VIDUR_VLLM_GV3_ROLLOUT_COUNT:-1}"
export VIDUR_VLLM_GV3_ROLLOUT_HORIZON_S="${VIDUR_VLLM_GV3_ROLLOUT_HORIZON_S:-3.0}"
export VIDUR_VLLM_GV3_ROLLOUT_THREADS="${VIDUR_VLLM_GV3_ROLLOUT_THREADS:-1}"
export VIDUR_VLLM_GV3_ROLLOUT_POLICY_THREADS="${VIDUR_VLLM_GV3_ROLLOUT_POLICY_THREADS:-1}"
export VIDUR_VLLM_GV3_NATIVE_THREADS="${VIDUR_VLLM_GV3_NATIVE_THREADS:-1}"

mkdir -p "$ROOT/runs"
python -m vidur_vllm_real_testing.container_smoke \
    --package-root "$VIDUR_PACKAGE_ROOT" \
    --model-tokenizer "$MODEL" \
    --report "$ROOT/runs/controller-tokenizer-gate.json" \
    --require-vllm

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --scheduler-cls vidur_vllm_real_testing.gv3_persistent_scheduler.GV3PersistentScheduler \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --scheduling-policy fcfs \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    "$@"
