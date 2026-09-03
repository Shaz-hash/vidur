#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_HOST="${VIDUR_MEW1_HOST:-mew1}"
readonly EXPECTED_USER="${VIDUR_MEW1_USER:-shaz}"
readonly ROOT="${VIDUR_MEW1_ROOT:-/home/shaz/vidur}"
readonly SOURCE="$ROOT/source/vidur-classical-search"
readonly ENV_DIR="$ROOT/envs/vllm-0.13.0-cu129"
readonly BUILD_DIR="$ROOT/build/gv3-native"
readonly UV_VERSION="${VIDUR_UV_VERSION:-0.12.5}"
readonly PYTHON_VERSION="${VIDUR_PYTHON_VERSION:-3.12}"
readonly BUILD_JOBS="${VIDUR_NATIVE_BUILD_JOBS:-8}"
readonly MIN_FREE_GIB="${VIDUR_MIN_FREE_GIB:-60}"

if [[ "$(hostname -s)" != "$EXPECTED_HOST" ]]; then
    echo "refusing to install on host $(hostname -s); expected $EXPECTED_HOST" >&2
    exit 2
fi
if [[ "$(id -un)" != "$EXPECTED_USER" ]]; then
    echo "refusing to install as $(id -un); expected $EXPECTED_USER" >&2
    exit 2
fi
case "$ROOT" in
    /home/shaz/*) ;;
    *) echo "VIDUR_MEW1_ROOT must remain under /home/shaz" >&2; exit 2 ;;
esac
if [[ ! -d "$SOURCE/vidur_vllm_real_testing" ]]; then
    echo "deploy source to $SOURCE before bootstrapping" >&2
    exit 2
fi

free_kib="$(df -Pk /home/shaz | awk 'NR == 2 {print $4}')"
min_kib="$((MIN_FREE_GIB * 1024 * 1024))"
if (( free_kib < min_kib )); then
    echo "insufficient free disk: need at least ${MIN_FREE_GIB} GiB" >&2
    exit 2
fi

mkdir -p \
    "$ROOT/artifacts" \
    "$ROOT/build" \
    "$ROOT/cache/huggingface" \
    "$ROOT/cache/torch" \
    "$ROOT/cache/uv" \
    "$ROOT/envs" \
    "$ROOT/logs" \
    "$ROOT/manifests" \
    "$ROOT/models" \
    "$ROOT/runs"

python3 -m pip install --user --disable-pip-version-check --no-warn-script-location \
    "uv==$UV_VERSION"

readonly UV="/home/shaz/.local/bin/uv"
export UV_CACHE_DIR="$ROOT/cache/uv"
"$UV" python install "$PYTHON_VERSION"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$UV" venv --python "$PYTHON_VERSION" --seed "$ENV_DIR"
fi

"$UV" pip install \
    --python "$ENV_DIR/bin/python" \
    --torch-backend=cu129 \
    "vllm==0.13.0" \
    "numpy==1.26.4" \
    "scikit-learn==1.5.0" \
    "joblib==1.5.0" \
    "pybind11==3.0.2" \
    "pandas==2.2.3" \
    "pyyaml==6.0.1"

export PATH="$ENV_DIR/bin:/home/shaz/.local/bin:$PATH"
export PYTHONPATH="$SOURCE"
export HF_HOME="$ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$ROOT/cache/huggingface/hub"
export TORCH_HOME="$ROOT/cache/torch"

python -m vidur_vllm_real_testing.patch_vllm_scheduler install

pybind11_dir="$(python -m pybind11 --cmakedir)"
torch_prefix="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake \
    -S "$SOURCE/vidur/Game_Version3_Cpp" \
    -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$ENV_DIR/bin/python" \
    -DPython_ROOT_DIR="$ENV_DIR" \
    -Dpybind11_DIR="$pybind11_dir" \
    -DCMAKE_PREFIX_PATH="$torch_prefix"
cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"

python -m unittest discover -s "$SOURCE/vidur_vllm_real_testing/tests" -v
python -m vidur_vllm_real_testing.container_smoke \
    --package-root "$SOURCE/vidur_vllm_real_testing" \
    --model-tokenizer "$SOURCE/vidur_vllm_real_testing/tokenizer/llama3_8b" \
    --require-vllm \
    --local-files-only \
    --report "$ROOT/manifests/bundled_scheduler_smoke.json"
python -m vidur_vllm_real_testing.vllm_scheduler_smoke \
    --trace "$SOURCE/vidur_vllm_real_testing/traces/splitwise_conv_20s_english_canonical.csv"
python -c 'from vidur.Game_Version3_Cpp import mcts_native_gv2; print(mcts_native_gv2.__file__)'

python -m pip freeze | LC_ALL=C sort > "$ROOT/manifests/environment.freeze.txt"
{
    printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "$(hostname -s)"
    printf 'user=%s\n' "$(id -un)"
    printf 'uv=%s\n' "$("$UV" --version)"
    printf 'python=%s\n' "$(python --version 2>&1)"
    printf 'vllm=%s\n' "$(python -c 'import vllm; print(vllm.__version__)')"
    printf 'torch=%s\n' "$(python -c 'import torch; print(torch.__version__)')"
    printf 'torch_cuda=%s\n' "$(python -c 'import torch; print(torch.version.cuda)')"
    printf 'source_manifest=%s\n' "$ROOT/manifests/source.sha256"
} > "$ROOT/manifests/environment.txt"

touch "$ROOT/manifests/bootstrap.complete"
echo "mew1 user-space environment is ready at $ENV_DIR"
