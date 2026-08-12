#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="${SCRIPT_DIR}/build_context"
ARTIFACT_DIR="${SCRIPT_DIR}/artifacts"
IMAGE_NAME="${AGZ_IMAGE_NAME:-vidur-agz-exp3-worker}"
IMAGE_TAG="${AGZ_IMAGE_TAG:-rollout-modes-3892281}"

if [[ ! -f "${CONTEXT_DIR}/build-metadata.env" ]]; then
    "${SCRIPT_DIR}/prepare_build_context.sh"
fi
if ! docker buildx version >/dev/null 2>&1; then
    echo "docker buildx is required for the ARM64 export" >&2
    exit 2
fi

set -a
source "${CONTEXT_DIR}/build-metadata.env"
set +a

mkdir -p "${ARTIFACT_DIR}"
OUTPUT="${ARTIFACT_DIR}/${IMAGE_NAME}-${IMAGE_TAG}-linux-arm64.tar"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

docker buildx build \
    --platform linux/arm64 \
    --build-arg "VIDUR_GIT_SHA=${VIDUR_GIT_SHA}" \
    --build-arg "BUILD_DATE=${BUILD_DATE}" \
    --build-arg "NATIVE_BUILD_JOBS=${AGZ_NATIVE_BUILD_JOBS:-8}" \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --output "type=docker,dest=${OUTPUT}" \
    "${CONTEXT_DIR}"

(
    cd "${ARTIFACT_DIR}"
    sha256sum "$(basename "${OUTPUT}")" > "$(basename "${OUTPUT}").sha256"
)
ls -lh "${OUTPUT}" "${OUTPUT}.sha256"
