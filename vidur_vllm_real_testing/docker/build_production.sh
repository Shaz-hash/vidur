#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
context_dir="$(cd -- "$script_dir/.." && pwd)"
image_tag="${VIDUR_VLLM_IMAGE_TAG:-vidur-vllm-real-testing:v0.13.0-scheduler}"
platform="${VIDUR_VLLM_PLATFORM:-linux/amd64}"

docker build \
    --platform "$platform" \
    --file "$script_dir/production.Dockerfile" \
    --tag "$image_tag" \
    "$context_dir"

printf 'built %s for %s\n' "$image_tag" "$platform"
