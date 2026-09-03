# Task 1.1.2 Implementation

## Purpose

Task 1.1.2 runs exact-token synthetic request traces through real vLLM execution on
`shaz@mew1`, restricted to physical GPU 2 or 3. A persistent GV3 state mirrors the
real queue, and the promoted Markov-v2 DNN/native-MCTS controller selects each real
batch. This path applies no timing calibration.

## Configuration

The single source of truth is:

`vidur_vllm_real_testing/test_traces_on_GPU_with_AlphaGOZERO_models_config.py`

It records and validates:

- Served model: `meta-llama/Meta-Llama-3-8B` geometry using the
  `NousResearch/Meta-Llama-3-8B` weight mirror.
- Task 1.1.1 runtime: vLLM `0.26.0`, FlashInfer `0.6.14`, PyTorch `2.11.0`,
  CUDA compiler `13.0.88`, `float16`, and `FLASHINFER`.
- Hardware: one A100, TP=1, PP=1, and physical mew1 GPU 2 by default. GPU 3 is
  accepted explicitly. A launcher refuses to start when the selected GPU has a
  compute process.
- Promoted DNN bundle paths: controller value, controller policy, adversary policy,
  and the frozen native MCTS configuration. HGB artifacts are rejected by the
  existing native planner validation.
- Search: 2,000 MCTS iterations, `full_tree_rollout`, one 3-second leaf-relative
  rollout, discount `0.98`, PUCT `0.5`, and one native/rollout thread.
- Task 1.1.1 self-play provenance: Dirichlet epsilon `0.25`, concentration `12/N`,
  and sampling for the first 15 controller actions. Real-GPU evaluation does not
  inject root noise.
- Static trace duration, trace seed, trace type, GPU number, and output paths.

Environment overrides use the `VIDUR_TASK112_` prefix. The important deployment
ones are:

```bash
VIDUR_TASK112_GPU=2
VIDUR_TASK112_TRACE_TYPE=in_distribution
VIDUR_TASK112_TRACE_LENGTH_S=20
VIDUR_TASK112_MODEL_BUNDLE=/home/shaz/.../promoted_bundle
VIDUR_TASK112_NATIVE_MCTS_CONFIG=/home/shaz/.../native_mcts_cfg.json
VIDUR_TASK112_PREFILL_PROFILE=/home/shaz/.../flash-infer_prefill_profile.csv
VIDUR_TASK112_REMOTE_HF_HOME=/home/shaz/vidur_vllm_profile_accuracy/hf_cache
VIDUR_TASK112_CUDA_HOME=/usr/local/cuda-13.0
```

Changing `VIDUR_TASK112_MODEL_BUNDLE` automatically derives the controller value,
controller policy, adversary policy, and native-config paths unless their specific
overrides are also supplied.

## Trace Contract

`vidur_vllm_real_testing/task_1_1_2_trace.py` creates actual English prompts whose
BOS-inclusive Llama-3 tokenization exactly equals each physical prefill length.
Every prompt and token-ID artifact is checksummed and verified before execution.

`in_distribution` uses only GV3's trained request support:

```text
128, 256, 512, 1024, 1536, 2048, 3072, 4096
```

Physical and canonical prefill lengths are identical.

`out_distribution` generates physical lengths in `[128, 4096]`. The real length is
sent to vLLM, while GV3/DNN/MCTS receives a ceiling-to-128 representation. For
example, 157 physical tokens become 256 canonical tokens. This is an opt-in
canonicalization mode; the legacy nearest-128 behavior remains the default for all
existing traces.

Arrival groups are 1.2 seconds apart, so no two groups overlap in GV3's inclusive
one-second launch window. Each group obeys both caps: at most seven requests and at
most 7,168 canonical prefill tokens. Groups intentionally contain simultaneous
arrivals so the scheduler's atomic arrival barrier is exercised.

Prepared output contains:

```text
template_raw_trace.csv
raw_trace.csv
canonical_trace.csv
trace_manifest.json
experiment_config.json
prompts/text/*
prompts/token_ids/*
prompts/prompt_catalog.json
```

The manifest explicitly records `calibration: null`.

## Persistent State And Time

The real scheduler uses `GV3PersistentAdapter`; a fresh state is not reconstructed at
each decision.

1. The live vLLM running/waiting queues are converted to an immutable snapshot.
2. Equal-time trace requests are held until the complete arrival group is visible.
3. The snapshot reconciles against the persistent request ledger. Deadlines,
   canonical/physical progress, decode credits, adversary ticks, lateness, completed
   requests, and stopped/dropped requests remain from the prior state.
4. Native MCTS receives the persistent GV3 payload and the admitted live snapshot.
5. Planning wall time is logged but never added to `sim_time`.
6. vLLM executes the selected physical token allocation.
7. CUDA events measure from entry to the first transformer layer through exit from
   the final transformer layer, matching Task 1.1.1's profiling boundary.
8. Only that completed transformer-block duration advances canonical time.
9. Physical progress is reconciled back into the persistent ledger before the next
   decision.

When no prefill remains but decode work exists, the adapter returns decode-only
batches repeatedly. Each real GPU batch advances canonical time, so this continues
until a new trace arrival/adversary tick makes prefill work visible. If no work is
active, the ledger advances to the earliest visible arrival; no fake GPU duration is
invented.

## Runtime

The runner is:

`vidur_vllm_real_testing/task_1_1_2_runner.py`

Prepare and validate locally:

```bash
VIDUR_TASK112_TRACE_TYPE=in_distribution \
python -m vidur_vllm_real_testing.task_1_1_2_runner prepare \
  --output-root /tmp/task112-in

VIDUR_TASK112_TRACE_TYPE=in_distribution \
python -m vidur_vllm_real_testing.task_1_1_2_runner validate \
  --prepared-root /tmp/task112-in
```

Repeat with `VIDUR_TASK112_TRACE_TYPE=out_distribution` for the ceiling-mapped test.

On mew1, after transferring the source, prepared trace, promoted bundle, frozen
native config, and uncalibrated Task 1.1.1 profile:

```bash
VIDUR_TASK112_GPU=2 \
VIDUR_TASK112_TRACE_TYPE=in_distribution \
python -m vidur_vllm_real_testing.task_1_1_2_runner validate \
  --prepared-root /home/shaz/.../prepared \
  --runtime --require-free-gpu

VIDUR_TASK112_GPU=2 \
VIDUR_TASK112_TRACE_TYPE=in_distribution \
python -m vidur_vllm_real_testing.task_1_1_2_runner run \
  --prepared-root /home/shaz/.../prepared \
  --output-dir /home/shaz/.../runs/controller-in \
  --mode controller
```

The runtime preflight verifies vLLM/FlashInfer/PyTorch versions, the selected GPU,
Markov-v2 native configuration, and absence of calibration. It also instantiates the
CUDA compiler `13.0.88`, then instantiates the native planner and loads the promoted
controller value, controller policy, and
adversary policy DNNs before any server is started.

`VIDUR_TASK112_DECODE_TOKENS` may reduce output length for an isolated integration
smoke test. Production comparisons leave it at the validated default of 864.

The guarded per-request token-cap patch now recognizes tested vLLM versions 0.13.0
and 0.26.0, but still verifies the exact scheduler source needles and exactly two
patch markers before scheduling.

SJF baselines can use `--mode sjf256` or `--mode sjf512` with the same physical trace.

## Tests

`vidur_vllm_real_testing/tests/test_task_1_1_2.py` covers:

- Pinned Task 1.1.1 runtime and MCTS values.
- Opt-in 157-to-256 ceiling mapping without changing legacy defaults.
- In-distribution support and out-distribution non-grid prompts.
- Exact prompt/token parity and canonical trace validation.
- Simultaneous-arrival groups and GV3 launch-window legality.
- Canonical time advancing only by completed GPU duration, independent of snapshot
  monotonic time or controller wall time.

Existing scheduler tests additionally cover persistent progress, prefill tails,
decode credit accounting, repeated decode fast-forward, SLO timing, and rejection of
queue-state drift.
