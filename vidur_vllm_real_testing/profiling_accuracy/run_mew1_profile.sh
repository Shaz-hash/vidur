#!/usr/bin/env bash
set -euo pipefail

ROOT="${VIDUR_PROFILE_REMOTE_ROOT:-/home/shaz/vidur_vllm_profile_accuracy}"
REPO="$ROOT/source/vidur-classical-search"
PYTHON="$ROOT/.venv/bin/python"
RUN_DIR="$ROOT/runs/full"
PROFILE_DIR="$RUN_DIR/raw_profiles"
cd "$REPO"
GPU="$($PYTHON -m vidur_vllm_real_testing.profiling_accuracy.select_gpu)"
CUDA_HOME="$ROOT/.venv/lib/python3.10/site-packages/nvidia/cu13"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="$ROOT/hf_cache"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$PROFILE_DIR" "$RUN_DIR"

echo "Using physical GPU $GPU; no other GPU will be touched."
$PYTHON -m vidur_vllm_real_testing.profiling_accuracy.config \
    --output "$RUN_DIR/config.json"
$PYTHON -m vidur_vllm_real_testing.profiling_accuracy.write_manifest \
    --output "$RUN_DIR/manifest.json"
$PYTHON -m vidur_vllm_real_testing.profiling_accuracy.profile_compute_ops \
    --output "$PROFILE_DIR/mlp.csv"
$PYTHON -m vidur_vllm_real_testing.profiling_accuracy.profile_flashinfer_attention \
    --output "$PROFILE_DIR/attention.csv"
$PYTHON -m vidur_vllm_real_testing.profiling_accuracy.profile_vllm_batches \
    --output "$RUN_DIR/vllm_actual_batches.csv" \
    --raw-log "$RUN_DIR/vllm_raw_cuda_events.jsonl"

sha256sum \
    "$PROFILE_DIR/mlp.csv" \
    "$PROFILE_DIR/attention.csv" \
    "$RUN_DIR/vllm_actual_batches.csv" \
    > "$RUN_DIR/sha256sums.txt"
echo "Remote profiling completed in $RUN_DIR"
