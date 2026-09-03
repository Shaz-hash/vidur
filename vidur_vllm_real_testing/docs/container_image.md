# Tokenizer-Gated vLLM Image

## Frozen Base

The production image uses the official multi-architecture vLLM base:

```text
vllm/vllm-openai:v0.13.0
sha256:d623253f2ba246378421c9642e20885e65257f38418ff26d48c81aea1702521b
```

The base manifest resolves to ARM64 and AMD64 images. ARM64 is useful for the
tokenizer/container smoke test on XL4. AMD64 is the deployment architecture for
the A100 host.

## Build

From the repository root:

```bash
bash vidur_vllm_real_testing/docker/build_production.sh
```

Override the platform or image tag when required:

```bash
VIDUR_VLLM_PLATFORM=linux/arm64 \
VIDUR_VLLM_IMAGE_TAG=vidur-vllm-real-testing:arm64 \
bash vidur_vllm_real_testing/docker/build_production.sh
```

## Acceptance Gates

Validate the tokenizer and prepared artifacts bundled in the image:

```bash
mkdir -p /tmp/vidur-vllm-results
docker run --rm \
  -v /tmp/vidur-vllm-results:/results \
  vidur-vllm-real-testing:v0.13.0-token-gate \
  smoke-bundled
```

Validate the tokenizer from the actual mounted model directory:

```bash
docker run --rm \
  -v /data/models/Meta-Llama-3-8B:/models/llama3:ro \
  -v /data/results:/results \
  -e VIDUR_MODEL_TOKENIZER=/models/llama3 \
  vidur-vllm-real-testing:v0.13.0-token-gate \
  smoke-model
```

For a Hub model, set `VIDUR_MODEL_TOKENIZER` to the repository and optionally
set `VIDUR_MODEL_TOKENIZER_REVISION`. Do not reuse the pinned mirror revision
for a different repository.

The gate fails before serving when any of the following changes:

- frozen prefill or decode profile checksum;
- raw trace, canonical trace, or prompt catalog checksum;
- bundled tokenizer checksum;
- model `tokenizer.json` checksum or vocabulary size;
- token IDs produced by the model tokenizer or vLLM for any of the 110 prompts.
- vLLM is not exactly `0.13.0` or the guarded scheduler cap hook is absent.

The JSON report is written to `/results/tokenizer_compatibility.json`.

## Serving Boundary

The `serve` mode runs `smoke-model` before starting the vLLM OpenAI server:

```bash
docker run --rm --gpus all --ipc=host \
  -v /data/models/Meta-Llama-3-8B:/models/llama3:ro \
  -v /data/results:/results \
  -e VIDUR_MODEL_TOKENIZER=/models/llama3 \
  -p 8000:8000 \
  vidur-vllm-real-testing:v0.13.0-scheduler \
  serve-stock --tensor-parallel-size 1
```

The image contains the pinned custom scheduler class and keeps the tokenizer
gate mandatory. Available serving modes are `serve-stock`, `serve-shadow`,
`serve-active-validation`, `serve-controller`, `serve-sjf256`, and
`serve-sjf512`. See `scheduler_integration.md` for required trace/planner
environment variables and initial vLLM flags.

Run the CPU-only vLLM scheduler integration gate inside the image:

```bash
python3 -m vidur_vllm_real_testing.vllm_scheduler_smoke \
  --trace /opt/vidur/vidur_vllm_real_testing/traces/splitwise_conv_20s_english_canonical.csv
```

This verifies stock equivalence, mixed running-decode/waiting-prefill batches,
SJF-256, SJF-512, controller application, shadow non-interference, and stale
active-validation fallback through vLLM's actual KV cache manager.
