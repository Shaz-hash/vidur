# Scheduler And A100 Handoff

The tokenizer-gated image and pinned scheduler boundary are complete. See
`container_image.md` for the build evidence. The remaining phase installs the
full promoted native planner/model artifacts and runs the A100 gates without
mutating the existing CPU AlphaGoZero worker image.

## Remaining Image Plan

1. Install the repository at the current classical-branch commit.
2. Compile the GV3 native module inside the pinned vLLM/CUDA/PyTorch image.
3. Mount the promoted controller planner and model bundles.
4. Mount model artifacts and benchmark outputs at runtime rather than baking
   large mutable weights/results into the image.

Suggested runtime mounts:

```text
/data/huggingface -> /root/.cache/huggingface
/data/vllm        -> /root/.cache/vllm
/data/models      -> /models
/data/results     -> /results
/data/traces      -> /traces
```

Suggested image modes:

```text
smoke       verify GPU, model/profile hashes, native import, and schemas
calibrate   run batch-shape mismatch measurements only
baseline    run stock vLLM on a prepared trace
shadow      compute GV3 plans without applying them
controller  apply fingerprint-validated GV3 plans
sjf256      apply the fixed SJF 256-token baseline
sjf512      apply the fixed SJF 512-token baseline
```

## Container Acceptance Gates

Before active scheduling:

- NVIDIA driver/runtime sees the selected A100.
- Prefill and decode profile hashes match this directory's documented values.
- Controller/adversary artifacts report expected model and feature schemas.
- Python/native DNN inference differs by at most `1e-4` for single and batched
  inputs.
- Noise-disabled Python/native MCTS agrees at 1, 10, and 2,000 simulations.
- Every prompt tokenizes to its declared actual prefill length.
- Every prepared trace and manifest checksum validates.
- Shadow mode emits only legal actions and does not alter vLLM output.
- Initial active runs have no invalid plan, stale-plan application, KV OOM, or
  output-token mismatch.

## Calibration

Calibration compares the frozen Vidur prediction to the real A100 result for:

- prefills from 128 through 4096;
- decode batch sizes from 1 through 512;
- mixed decode plus prefill 128, 256, 512, and 1024.

After warmup and repeated measurements, write:

```text
batch_shape,vidur_predicted_s,real_p50_s,real_p95_s,relative_error
```

Do not feed these measurements back into MCTS for the first experiment. Their
purpose is to quantify simulator-to-hardware mismatch.

## Server Prerequisites

The host only needs the NVIDIA driver, Docker Engine, NVIDIA Container Toolkit,
and sufficient model/result disk. vLLM, CUDA user-space libraries, PyTorch, and
the custom scheduling code live in the pinned image.
