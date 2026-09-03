#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${VIDUR_MEW1_ROOT:-/home/shaz/vidur}"
readonly SOURCE="$ROOT/source/vidur-classical-search"
readonly ENV_DIR="$ROOT/envs/vidur-profiler-cu121"
readonly PHYSICAL_GPU="${VIDUR_PHYSICAL_GPU:-2}"
readonly PROFILE_TAG="${VIDUR_PROFILE_TAG:-mew1_a100_80gb_pcie_sarathi017_cu121_tp1_8k}"
readonly RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
readonly OUTPUT_ROOT="${VIDUR_PROFILE_OUTPUT:-$ROOT/profiling/$PROFILE_TAG/$RUN_STAMP}"
readonly LOCK_DIR="$ROOT/locks/vidur-profile-gpu${PHYSICAL_GPU}.lock"

if [[ "$(hostname -s)" != "mew1" || "$(id -un)" != "shaz" ]]; then
    echo "this profiler is restricted to shaz@mew1" >&2
    exit 2
fi
if [[ "$PHYSICAL_GPU" != "2" && "$PHYSICAL_GPU" != "3" ]]; then
    echo "only physical GPUs 2 and 3 are authorized" >&2
    exit 2
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "missing isolated profiler environment: $ENV_DIR" >&2
    exit 2
fi

gpu_pids() {
    nvidia-smi -i "$PHYSICAL_GPU" \
        --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | sed '/^[[:space:]]*$/d'
}

foreign_gpu_pids() {
    local pid owner
    while read -r pid; do
        [[ -n "$pid" ]] || continue
        owner="$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)"
        if [[ -n "$owner" && "$owner" != "shaz" ]]; then
            printf '%s\n' "$pid"
        fi
    done < <(gpu_pids)
}

if [[ -n "$(gpu_pids)" ]]; then
    echo "GPU $PHYSICAL_GPU is in use; refusing to start" >&2
    nvidia-smi -i "$PHYSICAL_GPU" \
        --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
    exit 3
fi

mkdir -p "$ROOT/locks" "$OUTPUT_ROOT" "$ROOT/tmp"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "profiling lock already exists: $LOCK_DIR" >&2
    exit 3
fi

cleanup() {
    local status=$?
    rmdir "$LOCK_DIR" 2>/dev/null || true
    if [[ $status -eq 0 ]]; then
        touch "$OUTPUT_ROOT/COMPLETE"
    else
        printf '%s\n' "$status" > "$OUTPUT_ROOT/FAILED_EXIT_CODE"
    fi
}
trap cleanup EXIT

{
    echo "profile_tag=$PROFILE_TAG"
    echo "started_utc=$RUN_STAMP"
    echo "hostname=$(hostname -f)"
    echo "user=$(id -un)"
    echo "physical_gpu=$PHYSICAL_GPU"
    echo "vidur_commit=$(git -C "$SOURCE" rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "sarathi_commit=$(git -C "$ROOT/source/sarathi-serve-vidur" rev-parse HEAD)"
    "$ENV_DIR/bin/python" -c 'import torch, sarathi; print(f"torch={torch.__version__}\ntorch_cuda={torch.version.cuda}\nsarathi={sarathi.__version__}")'
    nvidia-smi -i "$PHYSICAL_GPU" \
        --query-gpu=name,uuid,memory.total,power.limit,clocks.max.sm,compute_cap,driver_version \
        --format=csv,noheader
    echo "mlp_max_tokens=8192"
    echo "attention_max_seq_len=8192"
    echo "attention_max_batch_size=256"
    echo "attention_max_chunk_size=4096"
    echo "tensor_parallel_size=1"
} > "$OUTPUT_ROOT/metadata.txt"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export CUDA_HOME=/usr/local/cuda-12.1
export PYTHONPATH="$SOURCE"
export RAY_DEDUP_LOGS=0
export TMPDIR="$ROOT/tmp"

run_guarded() {
    local label="$1"
    shift
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting $label"
    "$@" &
    local child=$!
    while kill -0 "$child" 2>/dev/null; do
        local foreign
        foreign="$(foreign_gpu_pids)"
        if [[ -n "$foreign" ]]; then
            echo "foreign GPU process appeared on GPU $PHYSICAL_GPU: $foreign" >&2
            echo "terminating only profiler driver PID $child" >&2
            kill -TERM "$child" 2>/dev/null || true
            wait "$child" || true
            return 90
        fi
        sleep 5
    done
    wait "$child"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] completed $label"
}

cd "$SOURCE"
run_guarded mlp \
    "$ENV_DIR/bin/python" vidur/profiling/mlp/main.py \
    --models meta-llama/Meta-Llama-3-8B \
    --num_gpus 1 \
    --num_tensor_parallel_workers 1 \
    --max_tokens 8192 \
    --profile_method cuda_event \
    --output_dir "$OUTPUT_ROOT"

run_guarded attention \
    "$ENV_DIR/bin/python" vidur/profiling/attention/main.py \
    --models meta-llama/Meta-Llama-3-8B \
    --num_gpus 1 \
    --num_tensor_parallel_workers 1 \
    --max_seq_len 8192 \
    --min_batch_size 1 \
    --max_batch_size 256 \
    --max_chunk_size 4096 \
    --attention_backend flashinfer \
    --output_dir "$OUTPUT_ROOT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] profile complete: $OUTPUT_ROOT"
