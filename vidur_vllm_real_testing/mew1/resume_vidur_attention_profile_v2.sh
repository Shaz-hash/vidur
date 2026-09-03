#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${VIDUR_MEW1_ROOT:-/home/shaz/vidur}"
readonly SOURCE="$ROOT/source/vidur-classical-search"
readonly ENV_DIR="$ROOT/envs/vidur-profiler-cu121"
readonly PHYSICAL_GPU="${VIDUR_PHYSICAL_GPU:-2}"
readonly OUTPUT_ROOT="${VIDUR_PROFILE_OUTPUT:?set VIDUR_PROFILE_OUTPUT to the existing profile directory}"
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
if ! find "$OUTPUT_ROOT/mlp" -path '*/meta-llama/Meta-Llama-3-8B/mlp.csv' \
    -type f -size +0c -print -quit 2>/dev/null | grep -q .; then
    echo "completed MLP CSV not found under $OUTPUT_ROOT" >&2
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
        touch "$OUTPUT_ROOT/ATTENTION_COMPLETE"
    else
        printf '%s\n' "$status" > "$OUTPUT_ROOT/ATTENTION_FAILED_EXIT_CODE"
    fi
}
trap cleanup EXIT

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export CUDA_HOME=/usr/local/cuda-12.1
export PYTHONPATH="$SOURCE"
export RAY_DEDUP_LOGS=0
export TMPDIR="$ROOT/tmp"

cd "$SOURCE"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] resuming attention profile"
"$ENV_DIR/bin/python" vidur/profiling/attention/main.py \
    --models meta-llama/Meta-Llama-3-8B \
    --num_gpus 1 \
    --num_tensor_parallel_workers 1 \
    --max_seq_len 8192 \
    --min_batch_size 1 \
    --max_batch_size 256 \
    --max_chunk_size 4096 \
    --attention_backend FLASHINFER \
    --output_dir "$OUTPUT_ROOT" &
readonly CHILD_PID=$!

while kill -0 "$CHILD_PID" 2>/dev/null; do
    foreign="$(foreign_gpu_pids)"
    if [[ -n "$foreign" ]]; then
        echo "foreign GPU process appeared on GPU $PHYSICAL_GPU: $foreign" >&2
        echo "terminating only profiler driver PID $CHILD_PID" >&2
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" || true
        exit 90
    fi
    sleep 5
done

wait "$CHILD_PID"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] attention profile complete: $OUTPUT_ROOT"
