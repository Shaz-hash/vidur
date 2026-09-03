#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke-bundled}"
if [[ $# -gt 0 ]]; then
    shift
fi

root="${VIDUR_PACKAGE_ROOT:-/opt/vidur/vidur_vllm_real_testing}"
bundled_tokenizer="$root/tokenizer/llama3_8b"
model_tokenizer="${VIDUR_MODEL_TOKENIZER:-}"
pinned_revision="${VIDUR_PINNED_TOKENIZER_REVISION:-}"
model_revision="${VIDUR_MODEL_TOKENIZER_REVISION:-}"
report="${VIDUR_SMOKE_REPORT:-/results/tokenizer_compatibility.json}"

smoke() {
    local tokenizer="$1"
    local require_vllm="$2"
    local revision="$3"
    local args=(
        -m vidur_vllm_real_testing.container_smoke
        --package-root "$root"
        --model-tokenizer "$tokenizer"
        --report "$report"
    )
    if [[ -n "$revision" ]]; then
        args+=(--revision "$revision")
    fi
    if [[ "$require_vllm" == "1" ]]; then
        args+=(--require-vllm)
    fi
    python3 "${args[@]}"
}

case "$mode" in
    smoke-bundled)
        smoke "$bundled_tokenizer" 1 "$pinned_revision"
        ;;
    smoke-model)
        if [[ -z "$model_tokenizer" ]]; then
            echo "VIDUR_MODEL_TOKENIZER must name a mounted model directory or Hub repository" >&2
            exit 2
        fi
        smoke "$model_tokenizer" 1 "$model_revision"
        ;;
    serve|serve-stock|serve-shadow|serve-active-validation|serve-controller|serve-sjf256|serve-sjf512)
        if [[ -z "$model_tokenizer" ]]; then
            echo "VIDUR_MODEL_TOKENIZER must be set before serving" >&2
            exit 2
        fi
        case "$mode" in
            serve-stock) export VIDUR_VLLM_SCHEDULER_MODE="stock" ;;
            serve-shadow) export VIDUR_VLLM_SCHEDULER_MODE="shadow" ;;
            serve-active-validation) export VIDUR_VLLM_SCHEDULER_MODE="active-validation" ;;
            serve-controller) export VIDUR_VLLM_SCHEDULER_MODE="controller" ;;
            serve-sjf256) export VIDUR_VLLM_SCHEDULER_MODE="sjf-256" ;;
            serve-sjf512) export VIDUR_VLLM_SCHEDULER_MODE="sjf-512" ;;
        esac
        smoke "$model_tokenizer" 1 "$model_revision"
        server_args=(
            --model "$model_tokenizer"
            --tokenizer "$model_tokenizer"
            --scheduler-cls vidur_vllm_real_testing.vllm_scheduler.GV3Scheduler
        )
        if [[ -n "$model_revision" ]]; then
            server_args+=(--tokenizer-revision "$model_revision")
        fi
        exec python3 -m vllm.entrypoints.openai.api_server "${server_args[@]}" "$@"
        ;;
    shell)
        exec bash "$@"
        ;;
    *)
        echo "unknown mode: $mode" >&2
        exit 2
        ;;
esac
