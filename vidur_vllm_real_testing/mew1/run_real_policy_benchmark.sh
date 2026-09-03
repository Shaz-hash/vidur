#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 controller|sjf256|sjf512 OUTPUT_DIR" >&2
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

readonly MODE="$1"
readonly OUTPUT_DIR="$2"
case "$MODE" in
    controller|sjf256|sjf512) ;;
    *) usage; exit 2 ;;
esac

readonly ROOT="${VIDUR_MEW1_ROOT:-/home/shaz/vidur}"
readonly SOURCE="${VIDUR_REAL_SOURCE_ROOT:-$ROOT/source/vidur-classical-search}"
readonly ENV_DIR="${VIDUR_REAL_ENV_DIR:-$ROOT/envs/vllm-0.13.0-cu129}"
readonly CUDA_HOME="${VIDUR_REAL_CUDA_HOME:-/usr/local/cuda-13.0}"
readonly PHYSICAL_GPU="${VIDUR_PHYSICAL_GPU:-2}"
readonly MODEL="${VIDUR_REAL_MODEL:-$ROOT/models/Meta-Llama-3-8B}"
readonly TRACE="${VIDUR_REAL_TRACE:-$SOURCE/vidur_vllm_real_testing/traces/splitwise_conv_20s_english_mew1_a100_canonical.csv}"
readonly BUNDLE="${VIDUR_REAL_MODEL_BUNDLE:-$ROOT/artifacts/promoted_models/mew1_controller_v107_adv_v110}"
readonly NATIVE_CFG="${VIDUR_REAL_NATIVE_CFG:-$BUNDLE/native_mcts_cfg.json}"
readonly PORT="${VIDUR_VLLM_PORT:-8000}"
readonly SCHEDULER_LOG="$OUTPUT_DIR/scheduler.jsonl"
readonly SERVER_LOG="$OUTPUT_DIR/server.log"
readonly GPU_TIMING_LOG="$OUTPUT_DIR/gpu_forward_timing.jsonl"
readonly PERSISTENT_SJF="${VIDUR_VLLM_PERSISTENT_SJF:-0}"

if [[ "$(hostname -s)" != "mew1" || "$(id -un)" != "shaz" ]]; then
    echo "this benchmark is restricted to shaz@mew1" >&2
    exit 2
fi
if [[ "$PHYSICAL_GPU" != "2" && "$PHYSICAL_GPU" != "3" ]]; then
    echo "VIDUR_PHYSICAL_GPU must be 2 or 3" >&2
    exit 2
fi
if [[ "$PERSISTENT_SJF" != "0" && "$PERSISTENT_SJF" != "1" ]]; then
    echo "VIDUR_VLLM_PERSISTENT_SJF must be 0 or 1" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "output directory is not empty: $OUTPUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"
: > "$SCHEDULER_LOG"
: > "$GPU_TIMING_LOG"

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$ENV_DIR/bin:/home/shaz/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SOURCE"
export HF_HOME="${VIDUR_REAL_HF_HOME:-$ROOT/cache/huggingface}"
export VIDUR_PHYSICAL_GPU="$PHYSICAL_GPU"
export VIDUR_VLLM_PORT="$PORT"
export VIDUR_VLLM_CANONICAL_TRACE="$TRACE"
export VIDUR_VLLM_SCHEDULER_LOG="$SCHEDULER_LOG"
export VIDUR_VLLM_WARMUP_REQUEST_PREFIX="cmpl-vidur-warmup-"
export VIDUR_VLLM_GV3_IMPLICIT_PREFILL_OUTPUT_TOKENS=1
export VIDUR_VLLM_GV3_BATCH_DURATION_SOURCE=gpu_forward
export VIDUR_VLLM_GPU_TIMING_LOG="$GPU_TIMING_LOG"
export VIDUR_VLLM_GPU_TIMING_SCOPE="${VIDUR_VLLM_GPU_TIMING_SCOPE:-full_model_forward}"
export VIDUR_SMOKE_REPORT="$OUTPUT_DIR/tokenizer_compatibility.json"

server_pid=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM -- "-$server_pid" 2>/dev/null || kill -TERM "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

python -m vidur_vllm_real_testing.patch_vllm_scheduler install

common_args=(
    --no-async-scheduling
    --attention-backend "${VIDUR_VLLM_ATTENTION_BACKEND:-FLASHINFER}"
    --worker-cls vidur_vllm_real_testing.vllm_gpu_timing_worker.TimedGPUWorker
    --enforce-eager
    --max-model-len 8192
    --max-num-batched-tokens "${VIDUR_VLLM_MAX_NUM_BATCHED_TOKENS:-4608}"
    --max-num-seqs "${VIDUR_VLLM_MAX_NUM_SEQS:-512}"
    --block-size 16
    --dtype float16
    --gpu-memory-utilization "${VIDUR_VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
    --no-enable-log-requests
)

if [[ "$MODE" == "controller" || ( "$PERSISTENT_SJF" == "1" && ( "$MODE" == "sjf256" || "$MODE" == "sjf512" ) ) ]]; then
    export VIDUR_VLLM_GV3_MODEL_BUNDLE="$BUNDLE"
    export VIDUR_VLLM_GV3_NATIVE_CFG="$NATIVE_CFG"
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
    if [[ "$MODE" == "sjf256" ]]; then
        export VIDUR_VLLM_GV3_STATE_PLANNER="vidur_vllm_real_testing.baseline_state_planners:sjf256"
    elif [[ "$MODE" == "sjf512" ]]; then
        export VIDUR_VLLM_GV3_STATE_PLANNER="vidur_vllm_real_testing.baseline_state_planners:sjf512"
    fi
    server_command=(
        bash
        "$SOURCE/vidur_vllm_real_testing/mew1/launch_production_native_dnn_controller.sh"
        "$MODEL"
        "${common_args[@]}"
    )
else
    server_command=(
        "$SOURCE/vidur_vllm_real_testing/mew1/launch_server.sh"
        "$MODE"
        "$MODEL"
        "${common_args[@]}"
    )
fi

setsid "${server_command[@]}" > "$SERVER_LOG" 2>&1 &
server_pid=$!
echo "$server_pid" > "$OUTPUT_DIR/server.pid"

ready=0
for _ in $(seq 1 600); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "vLLM exited during startup; see $SERVER_LOG" >&2
        exit 1
    fi
    if curl --silent --fail "http://127.0.0.1:$PORT/health" >/dev/null; then
        ready=1
        break
    fi
    sleep 1
done
if [[ "$ready" != 1 ]]; then
    echo "vLLM did not become healthy within 600 seconds" >&2
    exit 1
fi
python -m vidur_vllm_real_testing.mew1.vllm_warmup \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --output "$OUTPUT_DIR/warmup.json"
: > "$SCHEDULER_LOG"
: > "$GPU_TIMING_LOG"


python -m vidur_vllm_real_testing.real_trace_benchmark \
    --trace "$TRACE" \
    --scheduler-log "$SCHEDULER_LOG" \
    --output-dir "$OUTPUT_DIR" \
    --policy "$MODE" \
    --model "$MODEL" \
    --base-url "http://127.0.0.1:$PORT" \
    --timeout-s 7200
