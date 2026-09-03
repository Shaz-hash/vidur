#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-mew1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO/.venv/bin/python"
REMOTE_ROOT="/home/shaz/vidur_vllm_profile_accuracy/runs/full"
PROFILE_DIR="$REPO/data/profiling/compute/a100_mew1_gpu2_vllm026_flashinfer0614/meta-llama/Meta-Llama-3-8B"
OUTPUT_DIR="$REPO/simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING"

mkdir -p "$PROFILE_DIR" "$OUTPUT_DIR"
rsync -az "$HOST:$REMOTE_ROOT/raw_profiles/mlp.csv" "$PROFILE_DIR/mlp.csv"
rsync -az "$HOST:$REMOTE_ROOT/raw_profiles/attention.csv" "$PROFILE_DIR/attention.csv"
rsync -az "$HOST:$REMOTE_ROOT/vllm_actual_batches.csv" "$OUTPUT_DIR/vllm_actual_batches.csv"
rsync -az "$HOST:$REMOTE_ROOT/vllm_raw_cuda_events.jsonl" "$OUTPUT_DIR/vllm_raw_cuda_events.jsonl"
rsync -az "$HOST:$REMOTE_ROOT/manifest.json" "$OUTPUT_DIR/remote_manifest.json"
rsync -az "$HOST:$REMOTE_ROOT/config.json" "$OUTPUT_DIR/config.json"
rsync -az "$HOST:$REMOTE_ROOT/sha256sums.txt" "$OUTPUT_DIR/remote_sha256sums.txt"

cd "$REPO"
$PYTHON -m vidur_vllm_real_testing.profiling_accuracy.predict_vidur_batches \
    --cache-mode use_cache \
    --output "$OUTPUT_DIR/vidur_predictions.csv"
$PYTHON -m vidur_vllm_real_testing.profiling_accuracy.predict_vidur_batches \
    --cache-mode require_cache \
    --output "$OUTPUT_DIR/vidur_predictions_require_cache.csv"
cmp "$OUTPUT_DIR/vidur_predictions.csv" "$OUTPUT_DIR/vidur_predictions_require_cache.csv"
$PYTHON -m vidur_vllm_real_testing.profiling_accuracy.compare_results \
    --actual "$OUTPUT_DIR/vllm_actual_batches.csv" \
    --vidur "$OUTPUT_DIR/vidur_predictions.csv" \
    --output "$OUTPUT_DIR/comparison.csv" \
    --summary "$OUTPUT_DIR/summary.json"

sha256sum \
    "$PROFILE_DIR/mlp.csv" \
    "$PROFILE_DIR/attention.csv" \
    "$OUTPUT_DIR/vllm_actual_batches.csv" \
    "$OUTPUT_DIR/vidur_predictions.csv" \
    "$OUTPUT_DIR/comparison.csv" \
    > "$OUTPUT_DIR/local_sha256sums.txt"
echo "Local raw comparison completed in $OUTPUT_DIR"
