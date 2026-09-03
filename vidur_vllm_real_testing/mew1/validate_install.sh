#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${VIDUR_MEW1_ROOT:-/home/shaz/vidur}"
readonly SOURCE="$ROOT/source/vidur-classical-search"
readonly ENV_DIR="$ROOT/envs/vllm-0.13.0-cu129"
readonly PHYSICAL_GPU="${VIDUR_PHYSICAL_GPU:-2}"

if [[ "$(hostname -s)" != "mew1" || "$(id -un)" != "shaz" ]]; then
    echo "this validation is restricted to shaz@mew1" >&2
    exit 2
fi
if [[ "$PHYSICAL_GPU" != "2" && "$PHYSICAL_GPU" != "3" ]]; then
    echo "VIDUR_PHYSICAL_GPU must be 2 or 3" >&2
    exit 2
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "bootstrap has not created $ENV_DIR" >&2
    exit 2
fi

active_processes="$(nvidia-smi -i "$PHYSICAL_GPU" --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_processes" ]]; then
    echo "GPU $PHYSICAL_GPU is in use; refusing to run validation" >&2
    exit 2
fi

export PATH="$ENV_DIR/bin:/home/shaz/.local/bin:$PATH"
export PYTHONPATH="$SOURCE"
export HF_HOME="$ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$ROOT/cache/huggingface/hub"
export TORCH_HOME="$ROOT/cache/torch"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"

python - <<'PY'
import json
import torch

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
device = torch.device("cuda:0")
x = torch.randn((1024, 1024), device=device, dtype=torch.float16)
y = x @ x
torch.cuda.synchronize(device)
props = torch.cuda.get_device_properties(device)
print(json.dumps({
    "logical_device": 0,
    "name": props.name,
    "total_memory_bytes": props.total_memory,
    "matmul_finite": bool(torch.isfinite(y).all().item()),
}, sort_keys=True))
PY

python -m vidur_vllm_real_testing.patch_vllm_scheduler verify
python -m vidur_vllm_real_testing.vllm_scheduler_smoke \
    --trace "$SOURCE/vidur_vllm_real_testing/traces/splitwise_conv_20s_english_canonical.csv"
python -c 'from vidur.Game_Version3_Cpp import mcts_native_gv2; print(mcts_native_gv2.__file__)'

mkdir -p "$ROOT/manifests"
nvidia-smi -i "$PHYSICAL_GPU" \
    --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv,noheader > "$ROOT/manifests/gpu${PHYSICAL_GPU}.csv"
touch "$ROOT/manifests/validation.gpu${PHYSICAL_GPU}.complete"
echo "GPU $PHYSICAL_GPU and the installed scheduler passed validation"
