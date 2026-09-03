#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-mew1}"
REMOTE_ROOT="/home/shaz/vidur_vllm_profile_accuracy"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_REPO="$REMOTE_ROOT/source/vidur-classical-search"
readarray -t VERSION_PINS < <(
    cd "$LOCAL_REPO"
    "$LOCAL_REPO/.venv/bin/python" - <<'PY'
from vidur_vllm_real_testing.profiling_accuracy.config import load_config
config = load_config()
print(config.vllm_version)
print(config.flashinfer_version)
print(config.cuda_compiler_version)
PY
)
VLLM_VERSION="${VERSION_PINS[0]}"
FLASHINFER_VERSION="${VERSION_PINS[1]}"
CUDA_COMPILER_VERSION="${VERSION_PINS[2]}"


ssh "$HOST" "mkdir -p '$REMOTE_REPO/vidur_vllm_real_testing' '$REMOTE_REPO/simulator_output' '$REMOTE_ROOT/runs'"
rsync -az \
    "$LOCAL_REPO/vidur_vllm_real_testing/profiling_accuracy/" \
    "$HOST:$REMOTE_REPO/vidur_vllm_real_testing/profiling_accuracy/"
rsync -az \
    "$LOCAL_REPO/vidur_vllm_real_testing/__init__.py" \
    "$LOCAL_REPO/vidur_vllm_real_testing/canonicalization.py" \
    "$LOCAL_REPO/vidur_vllm_real_testing/trace_contract.py" \
    "$HOST:$REMOTE_REPO/vidur_vllm_real_testing/"
rsync -az \
    "$LOCAL_REPO/simulator_output/prefill_profile.csv" \
    "$HOST:$REMOTE_REPO/simulator_output/prefill_profile.csv"

ssh "$HOST" "
set -euo pipefail
if [[ ! -x '$REMOTE_ROOT/.venv/bin/python' ]]; then
    /home/shaz/.local/bin/uv venv --python 3.10 '$REMOTE_ROOT/.venv'
fi
/home/shaz/.local/bin/uv pip install --python '$REMOTE_ROOT/.venv/bin/python' \
    'vllm==$VLLM_VERSION' 'flashinfer-python==$FLASHINFER_VERSION'
# FlashInfer JIT must use compiler components matching Torch's CUDA 13.0 runtime.
/home/shaz/.local/bin/uv pip install --python '$REMOTE_ROOT/.venv/bin/python' \
    'nvidia-cuda-nvcc==$CUDA_COMPILER_VERSION' 'nvidia-cuda-crt==$CUDA_COMPILER_VERSION' 'nvidia-nvvm==$CUDA_COMPILER_VERSION'
"

echo "mew1 profiling environment and source are synchronized at $REMOTE_ROOT"
