# vLLM Scheduler Integration

## Boundary

The policy chooses request IDs and token counts. vLLM 0.13.0 remains the sole
owner of KV blocks, preemption, connectors, model execution, and output tokens.
The image applies a guarded per-request cap at the two points where upstream
vLLM computes running/waiting token counts. If either source location changes,
the image build fails.

## Required Server Shape

Use the initial correctness configuration:

```text
--scheduler-cls vidur_vllm_real_testing.vllm_scheduler.GV3Scheduler
--scheduling-policy fcfs
--enable-chunked-prefill
--no-enable-prefix-caching
--max-num-batched-tokens 4608
--max-num-seqs 512
--block-size 16
```

The package entrypoint adds `--scheduler-cls`; pass the remaining vLLM flags to
the selected `serve-*` mode.

## Mode Commands

With `/models`, `/traces`, and `/results` mounted:

```bash
docker run --gpus device=0 --rm \
  -e VIDUR_MODEL_TOKENIZER=/models/Meta-Llama-3-8B \
  -e VIDUR_VLLM_CANONICAL_TRACE=/traces/workload_canonical.csv \
  -e VIDUR_VLLM_SCHEDULER_LOG=/results/scheduler.jsonl \
  IMAGE serve-sjf256 --enable-chunked-prefill --no-enable-prefix-caching

docker run --gpus device=0 --rm \
  -e VIDUR_MODEL_TOKENIZER=/models/Meta-Llama-3-8B \
  -e VIDUR_VLLM_CANONICAL_TRACE=/traces/workload_canonical.csv \
  -e VIDUR_VLLM_GV3_PLANNER=deployment.gv3_planner:PromotedPlanner \
  -e VIDUR_VLLM_SCHEDULER_LOG=/results/scheduler.jsonl \
  IMAGE serve-shadow --enable-chunked-prefill --no-enable-prefix-caching
```

Replace `serve-sjf256` with `serve-sjf512` for the second baseline. Promotion
from `serve-shadow` to `serve-active-validation`, then `serve-controller`, is a
deployment decision and does not require rebuilding the image.

## Safety Sequence

1. Convert each live request through the checked canonical trace metadata.
2. Fingerprint all live request phases, tails, SLOs, queue locations, and the
   vLLM token limit.
3. Invoke the policy synchronously and measure its blocking time.
4. Re-snapshot and reject a stale fingerprint.
5. Reject missing IDs, duplicate allocations, phase mismatches, multi-token
   decodes, over-budget plans, and empty work for a non-empty state.
6. Project canonical chunks to real tails with `min(canonical, actual_tail)`.
7. Let stock vLLM allocate KV and construct `SchedulerOutput`.
8. Compare vLLM's emitted token map to the validated plan exactly; mismatch is
   fatal in active modes.

`active-validation` falls back only for errors before vLLM state mutation.
`controller`, SJF-256, and SJF-512 fail closed.
