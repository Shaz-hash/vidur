# Vidur versus current vLLM GPU profiling

This directory implements `../task.md` as an isolated experiment. It does not
modify Game Version 3, AlphaGo Zero, the existing A100 profile, or any prior
predictor cache.

## Fixed configuration

- vLLM: `0.26.0`
- FlashInfer Python API: `0.6.14`
- Torch: `2.11.0`, CUDA runtime `13.0`
- FlashInfer JIT compiler components: CUDA `13.0.88`
- GPU: physical `mew1` GPU 2 or 3, selected only when idle
- Tensor/pipeline parallelism: TP1/PP1
- Model geometry: `meta-llama/Meta-Llama-3-8B`
- Actual vLLM weight source: `NousResearch/Meta-Llama-3-8B`, an ungated mirror
  with the same Llama-3-8B tensor geometry. Weight values do not change kernel
  shapes or execution-time modeling.
- Attention backend: `FLASHINFER`
- Predictor: Vidur random forest, one training thread
- Limits: 8192 tokens/request, batch size 256, prefill chunk 4096
- Test batches: one and two equal-size prefill requests for every token count in
  `simulator_output/prefill_profile.csv`

The Sarathi `vidur` branch is from 2024 and uses an obsolete FlashInfer
`begin_forward` API. It cannot be imported into current vLLM/FlashInfer without
changing the kernel stack being tested. The operation profiler therefore emits
Vidur-compatible CSVs using current vLLM primitives and the current FlashInfer
paged-KV API directly.

vLLM V1's offline scheduler can serialize two waiting prefills even when the
token and sequence budgets fit both. The actual-batch harness therefore invokes
vLLM's native model-runner `_dummy_run` through a worker RPC and forces the exact
equal-size physical batch shape. This still executes the loaded vLLM model,
FlashInfer attention, KV-cache preparation, and transformer kernels; only the
online scheduling decision is bypassed. The scheduler remains configured with a
512-sequence cap. Raw records retain every synthetic request and the aggregator
rejects missing, split, or extra shapes.

## Measurement boundary

The actual vLLM timing uses CUDA events from entry into the first Llama decoder
layer through exit from the final decoder layer. This includes the QKV/MLP
GEMMs, normalization, RoPE, KV-cache write, and FlashInfer attention for all 32
layers. It excludes scheduler/CPU overhead, embedding, token sampling, and the
final vocabulary projection.

The comparison uses `ExecutionTime.model_time` on the Vidur side. That property
also excludes CPU overhead, embedding, and final softmax. Full vLLM model
forward time is retained in the output only as a diagnostic.

No measured or predicted value is multiplied by a calibration factor. Every
manifest and output row records `calibration_applied=false`.

## Files

- `config.py`: single source of truth for versions, model, limits, cases, paths,
  and repetitions.
- `profile_compute_ops.py`: creates Vidur-compatible `mlp.csv` using current
  vLLM CUDA primitives.
- `profile_flashinfer_attention.py`: creates `attention.csv` using current
  FlashInfer paged prefill/decode kernels.
- `vllm_block_timing_worker.py`: vLLM Worker subclass that places CUDA events at
  the exact decoder-stack boundary and exposes exact-shape model-runner RPCs.
- `profile_vllm_batches.py`: runs configurable equal-size, multiple-request
  batches through that RPC and aggregates warmup-free CUDA timings.
- `predict_vidur_batches.py`: builds the raw Vidur RF cache and queries the same
  batch shapes. A second `require_cache` run verifies the cache is complete.
- `compare_results.py`: writes per-shape signed and absolute errors plus summary
  statistics.
- `select_gpu.py`: fails closed unless GPU 2 or 3 is idle; it never kills a
  process.
- `write_manifest.py`: records software, GPU, boundary, configuration, and the
  explicit absence of calibration.
- `setup_mew1.sh`, `run_mew1_profile.sh`, `sync_and_compare.sh`: reproducible
  setup, remote measurement, and local fitting/comparison entry points.

## Runbook

From the repository root on the local machine:

```bash
vidur_vllm_real_testing/profiling_accuracy/setup_mew1.sh mew1
ssh mew1 '/home/shaz/vidur_vllm_profile_accuracy/source/vidur-classical-search/vidur_vllm_real_testing/profiling_accuracy/run_mew1_profile.sh'
vidur_vllm_real_testing/profiling_accuracy/sync_and_compare.sh mew1
```

Future equal-size request counts can be selected without editing code. Increase
the vLLM token budget whenever the largest requested physical batch needs it:

```bash
VIDUR_PROFILE_REQUEST_COUNTS=1,2,4 \
VIDUR_PROFILE_MAX_BATCHED_TOKENS=16384 \
vidur_vllm_real_testing/profiling_accuracy/run_mew1_profile.sh
```

The configuration validator rejects a request-count/token-size cross product
that cannot fit in the configured token budget.

The GPU selector exits without launching if both permitted GPUs are in use.
The remote run is stored under
`/home/shaz/vidur_vllm_profile_accuracy/runs/full`. The final local artifacts
are written under:

```text
simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING/
  comparison.csv
  summary.json
  vllm_actual_batches.csv
  vllm_raw_cuda_events.jsonl
  vidur_predictions.csv
  remote_manifest.json
```

The uncalibrated profile consumed by Vidur is separate from every previous A100
profile:

```text
data/profiling/compute/a100_mew1_gpu2_vllm026_flashinfer0614/
  meta-llama/Meta-Llama-3-8B/mlp.csv
  meta-llama/Meta-Llama-3-8B/attention.csv
```

## Multi-request interpretation

Vidur's current prefill model represents a multi-request prefill with aggregate
KV-cache and an L2 aggregate chunk size, rather than request-count as an
independent feature. Its configured prefill lookup is capped at 4096 chunk
tokens. A batch of two 4096-token requests therefore exposes this limitation:
the real vLLM batch has 8192 tokens, Vidur's rounded L2 aggregate is 5824, and
Vidur queries the 4096-token attention boundary.
`vidur_attention_lookup_clamped` marks every affected row so unclamped and
clamped errors can be interpreted separately rather than hidden.
