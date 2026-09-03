#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="${VIDUR_MEW1_ROOT:-/home/shaz/vidur}"
readonly SOURCE="$ROOT/source/vidur-classical-search"
readonly ENV_DIR="$ROOT/envs/vllm-0.13.0-cu129"
readonly PHYSICAL_GPU="${VIDUR_PHYSICAL_GPU:-2}"
readonly MODEL="${VIDUR_SMOKE_MODEL:-facebook/opt-125m}"
readonly PORT="${VIDUR_SMOKE_PORT:-18080}"
readonly LOG="$ROOT/logs/vllm-gpu${PHYSICAL_GPU}-smoke.log"

if [[ "$(hostname -s)" != "mew1" || "$(id -un)" != "shaz" ]]; then
    echo "this smoke test is restricted to shaz@mew1" >&2
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
    echo "GPU $PHYSICAL_GPU is in use; refusing to run smoke test" >&2
    exit 2
fi

export PATH="$ENV_DIR/bin:/home/shaz/.local/bin:$PATH"
export PYTHONPATH="$SOURCE"
export HF_HOME="$ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$ROOT/cache/huggingface/hub"
export TORCH_HOME="$ROOT/cache/torch"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export VIDUR_VLLM_SCHEDULER_MODE=stock

mkdir -p "$ROOT/logs" "$ROOT/manifests"
server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --scheduler-cls vidur_vllm_real_testing.vllm_scheduler.GV3Scheduler \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.15 \
    --enforce-eager \
    --disable-log-requests \
    >"$LOG" 2>&1 &
server_pid="$!"

ready=0
for _ in $(seq 1 180); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "vLLM exited before becoming ready; see $LOG" >&2
        tail -80 "$LOG" >&2
        exit 1
    fi
    if python - "$PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=1) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
    then
        ready=1
        break
    fi
    sleep 2
done
if [[ "$ready" != "1" ]]; then
    echo "vLLM did not become ready; see $LOG" >&2
    tail -80 "$LOG" >&2
    exit 1
fi

python - "$PORT" "$MODEL" "$ROOT/manifests/vllm_gpu_smoke.json" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

port, model, output = sys.argv[1:]
payload = json.dumps({
    "model": model,
    "prompt": "A deterministic scheduler smoke test says",
    "max_tokens": 8,
    "temperature": 0,
}).encode("utf-8")
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)
completion = result["choices"][0]["text"]
if not isinstance(completion, str):
    raise RuntimeError(f"unexpected completion: {result}")
report = {
    "status": "passed",
    "model": model,
    "scheduler": "vidur_vllm_real_testing.vllm_scheduler.GV3Scheduler",
    "mode": "stock",
    "completion": completion,
}
Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, sort_keys=True))
PY

touch "$ROOT/manifests/vllm_gpu_smoke.complete"
echo "real vLLM request passed through GV3Scheduler on physical GPU $PHYSICAL_GPU"
