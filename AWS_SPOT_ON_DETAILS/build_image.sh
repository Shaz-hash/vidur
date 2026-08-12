#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="${SCRIPT_DIR}/build_context"
IMAGE_NAME="${AGZ_IMAGE_NAME:-vidur-agz-exp3-worker}"
IMAGE_TAG="${AGZ_IMAGE_TAG:-rollout-modes-3892281}"
case "$(uname -m)" in
    x86_64) HOST_PLATFORM="linux/amd64" ;;
    aarch64|arm64) HOST_PLATFORM="linux/arm64" ;;
    *)
        echo "unsupported Docker host architecture: $(uname -m)" >&2
        exit 2
        ;;
esac
PLATFORM="${AGZ_IMAGE_PLATFORM:-${HOST_PLATFORM}}"

if [[ ! -f "${CONTEXT_DIR}/build-metadata.env" ]]; then
    "${SCRIPT_DIR}/prepare_build_context.sh"
fi

set -a
source "${CONTEXT_DIR}/build-metadata.env"
set +a

BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

if docker buildx version >/dev/null 2>&1; then
    docker buildx build \
        --platform "${PLATFORM}" \
        --build-arg "VIDUR_GIT_SHA=${VIDUR_GIT_SHA}" \
        --build-arg "BUILD_DATE=${BUILD_DATE}" \
        --build-arg "NATIVE_BUILD_JOBS=${AGZ_NATIVE_BUILD_JOBS:-8}" \
        --tag "${FULL_IMAGE}" \
        --load \
        "${CONTEXT_DIR}"
else
    if [[ "${PLATFORM}" != "${HOST_PLATFORM}" ]]; then
        echo "docker buildx is required to build ${PLATFORM} on ${HOST_PLATFORM}" >&2
        exit 2
    fi
    docker build \
        --build-arg "VIDUR_GIT_SHA=${VIDUR_GIT_SHA}" \
        --build-arg "BUILD_DATE=${BUILD_DATE}" \
        --build-arg "NATIVE_BUILD_JOBS=${AGZ_NATIVE_BUILD_JOBS:-8}" \
        --tag "${FULL_IMAGE}" \
        "${CONTEXT_DIR}"
fi

docker image inspect "${FULL_IMAGE}" > "${SCRIPT_DIR}/image_metadata.local.json"
echo "loaded_image=${FULL_IMAGE}"
docker image inspect --format 'id={{.Id}} arch={{.Architecture}} size={{.Size}}' "${FULL_IMAGE}"
