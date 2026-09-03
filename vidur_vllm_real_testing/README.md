# Vidur GV3 AlphaGoZero on Real vLLM

This directory is the integration boundary between three systems:

1. A real vLLM server executing Llama-3-8B requests on an A100 GPU.
2. The persistent Game Version 3 (GV3) scheduling state and its SLO semantics.
3. A promoted AlphaGoZero controller whose native MCTS uses DNN policy and value models plus a frozen Vidur execution-time predictor.

The pipeline generates exact-token request traces, starts vLLM with a custom scheduler, converts each live vLLM scheduling state into a persistent GV3 state, runs the native DNN MCTS controller, applies the selected root action through vLLM's stock scheduler, measures the completed GPU batch, and feeds that duration and progress back into the persistent GV3 ledger.

This README is the operational and architectural reference for that complete path. The task-specific implementation notes in [`../vidur/AlphaGoZero/new_vidur_cache_experiment/task 1.1.2_implementation.md`](../vidur/AlphaGoZero/new_vidur_cache_experiment/task%201.1.2_implementation.md) remain useful historical context, but this file describes the current code.

## 1. Purpose

The main question answered by this pipeline is:

> How does a controller trained against a Vidur digital twin behave when its selected batches are executed by real vLLM on a real GPU?

The same prepared trace can run under three policies:

| Mode | Decision source | Purpose |
| --- | --- | --- |
| `controller` | Promoted native DNN MCTS controller | Evaluate the learned policy on real hardware. |
| `sjf256` | Deterministic SJF with a 256-token prefill budget | First real-hardware baseline. |
| `sjf512` | Deterministic SJF with a 512-token prefill budget | Second real-hardware baseline. |

This is an evaluation pipeline, not an online-training pipeline. It loads frozen promoted models, evaluates without root Dirichlet noise, and records enough state to audit every decision and physical batch.

## 2. Non-negotiable invariants

The code fails closed when one of these invariants is violated.

### 2.1 Actual and canonical values are separate contracts

Every request keeps two representations:

```text
actual fields
    Real prompt length, physical vLLM progress, GPU execution, and raw
    real-request SLO observations.

canonical fields
    GV3 state, DNN features, MCTS actions, canonical deadlines, and
    authoritative GV3 accounting.
```

Canonicalization never overwrites actual data. An out-of-distribution request with 157 physical prefill tokens remains a 157-token vLLM request while the controller sees a canonical 256-token request. At its final tail, a canonical allocation is projected down to the actual tokens remaining; spare canonical capacity is not reassigned.

### 2.2 GV3 state is persistent

The scheduler does not reconstruct an independent game state from the vLLM queues at every decision. [`gv3_live_adapter.py`](gv3_live_adapter.py) owns a persistent ledger of simulator time, deadlines, lateness, decode credits, adversary ticks, launch history, progress, and terminal state. Current vLLM requests reconcile and validate that ledger.

### 2.3 vLLM still owns execution mechanics

The integration selects requests and token counts. It does not reimplement vLLM's KV-cache allocation, block management, preemption, request lifecycle, or model execution. It constrains eligible work, calls stock `Scheduler.schedule()`, and verifies that vLLM emitted exactly the requested physical batch.

### 2.4 Controller wall time is not game time

Native MCTS planning can take substantial wall time. Planning delay is excluded from both logical trace arrivals and the persistent GV3 clock. The GV3 clock advances only by completed measured GPU batches or an explicit legal idle jump.

### 2.5 No calibration is applied

The code may compare real GPU duration with Vidur prediction, but it never rescales either result. Native MCTS uses its frozen uncalibrated Vidur predictor for hypothetical tree transitions.

### 2.6 Evaluation has no root noise

Dirichlet fields are retained as self-play provenance. `root_noise_enabled_for_gpu_evaluation=True` is rejected.

## 3. End-to-end architecture

```text
AlphaGoZeroGPUTraceConfig
        |
        +--> deterministic request groups
        |       +--> exact English text
        |       +--> exact Llama-3 token IDs
        |       +--> raw and canonical CSVs
        |       `--> immutable manifests
        |
        +--> runtime preflight
        |       +--> host/user/GPU checks
        |       +--> pinned runtime checks
        |       +--> promoted DNN bundle checks
        |       `--> native planner construction
        |
        `--> run_real_policy_benchmark.sh
                +--> guarded vLLM scheduler patch
                +--> vLLM API server
                +--> GV3PersistentScheduler
                |       +--> live vLLM snapshot
                |       +--> GV3PersistentAdapter
                |       |       +--> persistent GV3 ledger
                |       |       `--> ProductionNativeDNNMCTSPlanner
                |       |               +--> native GV3 transitions
                |       |               +--> controller policy DNN
                |       |               +--> controller value DNN
                |       |               +--> adversary policy DNN
                |       |               `--> frozen Vidur predictor
                |       +--> plan validation and physical projection
                |       `--> stock vLLM Scheduler.schedule()
                +--> TimedGPUWorker executes the physical batch
                `--> Scheduler.update_from_output()
                        +--> verify exact progress and timing
                        +--> advance persistent GV3 state
                        `--> write decision and batch audits
```

## 4. Main entrypoints

| File | Responsibility |
| --- | --- |
| [`test_traces_on_GPU_with_AlphaGOZERO_models_config.py`](test_traces_on_GPU_with_AlphaGOZERO_models_config.py) | Single experiment configuration and environment overrides. |
| [`task_1_1_2_runner.py`](task_1_1_2_runner.py) | Normal `prepare`, `validate`, and `run` CLI. |
| [`task_1_1_2_trace.py`](task_1_1_2_trace.py) | Deterministic trace generation and artifact assembly. |
| [`prompt_materialization.py`](prompt_materialization.py) | Exact-token English prompt generation and verification. |
| [`mew1/run_real_policy_benchmark.sh`](mew1/run_real_policy_benchmark.sh) | vLLM lifecycle, warmup, benchmark, and cleanup. |
| [`gv3_persistent_scheduler.py`](gv3_persistent_scheduler.py) | Scheduling-cycle and completed-batch integration. |
| [`gv3_adapter.py`](gv3_adapter.py) | Production GV3 transition, drop, credit, and cost semantics. |
| [`native_dnn_mcts_state_planner.py`](native_dnn_mcts_state_planner.py) | Frozen predictor/config loading and state-planner entrypoint. |
| [`native_dnn_mcts_planner.py`](native_dnn_mcts_planner.py) | DNN loading, native search, winner selection, and action translation. |
| [`vllm_gpu_timing_worker.py`](vllm_gpu_timing_worker.py) | CUDA-event measurement of real GPU execution. |
| [`real_trace_benchmark.py`](real_trace_benchmark.py) | Async HTTP replay and real-request metrics. |

## 5. Configuration

`AlphaGoZeroGPUTraceConfig` in [`test_traces_on_GPU_with_AlphaGOZERO_models_config.py`](test_traces_on_GPU_with_AlphaGOZERO_models_config.py) is the source of truth for a run started through `task_1_1_2_runner`.

Print the effective config before preparing or running:

```bash
cd /home/shazer/Desktop/Research/Vidur/vidur-classical-search
python -m vidur_vllm_real_testing.test_traces_on_GPU_with_AlphaGOZERO_models_config
```

Write it to JSON:

```bash
python -m vidur_vllm_real_testing.test_traces_on_GPU_with_AlphaGOZERO_models_config \
  --output /tmp/task_1_1_2_config.json
```

### 5.1 Model and runtime

| Parameter | Default | Meaning and impact |
| --- | --- | --- |
| `model_type` | `meta-llama/Meta-Llama-3-8B` | Model geometry and profile provenance. |
| `vllm_model_name` | `NousResearch/Meta-Llama-3-8B` | Compatible model/tokenizer source. |
| `controller_model_family` | `dnn` | Only native DNN model bundles are accepted. HGB artifacts fail validation. |
| `vllm_version` | `0.26.0` | Exact scheduler/runtime contract. |
| `flashinfer_version` | `0.6.14` | Exact attention package contract. |
| `torch_version` | `2.11.0` | Required Torch version prefix. |
| `cuda_compiler_version` | `13.0.88` | Required `nvcc` version. |
| `cuda_home` | `/usr/local/cuda-13.0` | CUDA toolkit path. |
| `attention_backend` | `FLASHINFER` | Must match the execution/profile contract. |
| `dtype` | `float16` | Real vLLM model dtype. |

Changing model geometry, backend, dtype, TP/PP, or predictor profiles invalidates direct timing comparability with training.

### 5.2 GPU and vLLM capacity

| Parameter | Default | Meaning and impact |
| --- | --- | --- |
| `gpu_index` | `2` | Physical mew1 GPU; only 2 or 3 is accepted. |
| `tensor_parallel_size` | `1` | Required single-GPU tensor parallelism. |
| `pipeline_parallel_size` | `1` | Required single-stage pipeline parallelism. |
| `gpu_memory_utilization` | `0.50` | vLLM GPU memory reservation fraction. |
| `max_num_batched_tokens` | `8192` | Physical batch token budget. |
| `max_num_seqs` | `512` | Maximum sequences in a vLLM batch. |
| `block_size` | `16` | vLLM KV-cache block size. |
| `async_scheduling` | `False` | Required so one batch completes before the next persistent decision. |
| `frontend_multiprocessing` | `False` | Required by this deterministic integration. |

The launcher also uses eager execution, chunked prefill, no prefix caching, and FCFS internally. FCFS does not choose the AlphaGoZero action. The custom scheduler first restricts eligible requests and caps their tokens, then stock vLLM performs resource allocation.

### 5.3 Trace generation

| Parameter | Default | Meaning and impact |
| --- | --- | --- |
| `static_trace_length_s` | `20.0` | Logical arrival window, not rollout depth. |
| `trace_type` | `in_distribution` | Exact training support or non-grid physical requests rounded upward canonically. |
| `trace_seed` | `2026` | Reproducible layout and prompts. |
| `arrival_group_interval_s` | `1.2` | Interval between simultaneous-arrival groups. |
| `decode_tokens_per_request` | `864` | Exact logical decode length. |
| `prefill_grid_tokens` | `128` | Required canonical grid. |
| `min_prefill_tokens` | `128` | Minimum canonical prefill length. |
| `max_prefill_tokens` | `4096` | Maximum canonical prefill length. |
| `adversary_tick_s` | `0.2` | GV3 adversary and fast-forward interval. |
| `launch_window_s` | `1.0` | Recent adversary launch window. |
| `launch_window_request_cap` | `7` | Maximum requests in that window. |
| `launch_window_prefill_cap` | `7168` | Maximum canonical prefill tokens in that window. |

The in-distribution support is exactly:

```text
128, 256, 512, 1024, 1536, 2048, 3072, 4096
```

### 5.4 MCTS and promoted models

| Parameter | Default | Meaning and impact |
| --- | --- | --- |
| `mcts_iterations` | `2000` | Tree simulations per real controller decision. |
| `search_mode` | `full_tree_rollout` | Required native search implementation. |
| `rollout_count` | `1` | Policy-guided rollouts per newly expanded child. |
| `rollout_horizon_s` | `3.0` | Leaf-relative simulated-time horizon. |
| `rollout_threads` | `1` | Rollout worker count. |
| `rollout_policy_threads` | `1` | Policy inference threads in rollouts. |
| `native_threads` | `1` | General native search threads. |
| `discount_factor` | `0.98` | Time-varying search discount. |
| `puct_c` | `0.5` | Weight of the policy-prior exploration term. |
| `uct_c` | `1.0` | Additional native UCT-related constant. |
| `rollout_policy_temperature` | `1.0` | Policy rollout sampling temperature. |
| `policy_prior_temperature` | `1.0` | Search-prior temperature. |
| `game_horizon_s` | `5.0` | Training provenance; the real scheduler is continuing. |

The root action is selected deterministically from search statistics. Evaluation uses no root noise.

### 5.5 Self-play provenance

These fields describe the matching training experiment but do not alter real evaluation:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `dirichlet_epsilon` | `0.25` | Self-play root-noise mixture. |
| `dirichlet_total_concentration` | `12.0` | Self-play alpha is `12 / N` for `N` canonical legal actions. |
| `sampled_controller_actions` | `15` | Initial self-play controller actions sampled during training. |
| `root_noise_enabled_for_gpu_evaluation` | `False` | Required real-evaluation setting. |

### 5.6 Time and completion

| Parameter | Default | Meaning and impact |
| --- | --- | --- |
| `timing_scope` | `transformer_blocks` | CUDA interval from entry to first transformer block through exit from last block. |
| `batch_duration_source` | `gpu_forward` | Persistent clock uses measured GPU duration. |
| `implicit_prefill_output_tokens` | `1` | Removes vLLM's prefill-completion sampled token from logical decode progress. |
| `scheduler_mode` | `controller` | Config provenance; runner `--mode` selects the actual policy. |
| `output_root` | `/home/shaz/vidur_vllm_task_1_1_2/runs` | Default remote results root. |

The Python runner exports `timing_scope=transformer_blocks`. Calling [`mew1/run_real_policy_benchmark.sh`](mew1/run_real_policy_benchmark.sh) directly without that variable currently defaults to `full_model_forward`. Those scopes are not directly comparable.

### 5.7 Paths and promoted bundle

| Parameter | Purpose |
| --- | --- |
| `remote_root` | User-space deployment root on mew1. |
| `remote_environment` | Pinned Python/vLLM environment. |
| `remote_hf_home` | Hugging Face cache. |
| `remote_source_root` | Deployed repository import root. |
| `vllm_model_path` | Real model weights/tokenizer. |
| `model_bundle_path` | Promoted AlphaGoZero bundle. |
| `controller_value_model_path` | Native controller value DNN. |
| `controller_policy_model_path` | Native controller policy DNN. |
| `adversary_policy_model_path` | Native adversary policy DNN. |
| `native_mcts_config_path` | Frozen simulator/search/predictor config. |
| `prefill_profile_path` | Profile used to assign canonical request SLOs during preparation. |

A valid bundle contains:

```text
MODEL_BUNDLE/
|-- current_model.json
|-- native_mcts_cfg.json
|-- controller_value/
|   `-- native_model.tsv
|-- controller_prior/
|   `-- native_model.tsv
`-- adversary_prior/
    `-- native_model.tsv
```

Historical native binding names may contain `hgb`, but the production loader validates DNN export metadata and rejects HGB models.

### 5.8 Environment overrides

`load_config()` supports these `VIDUR_TASK112_*` suffixes:

```text
GPU
CUDA_HOME
TRACE_LENGTH_S
TRACE_TYPE
TRACE_SEED
VLLM_MODEL
MODEL_BUNDLE
CONTROLLER_VALUE_MODEL
CONTROLLER_POLICY_MODEL
ADVERSARY_POLICY_MODEL
NATIVE_MCTS_CONFIG
PREFILL_PROFILE
REMOTE_ROOT
REMOTE_ENVIRONMENT
REMOTE_HF_HOME
REMOTE_SOURCE_ROOT
OUTPUT_ROOT
DECODE_TOKENS
MCTS_ITERATIONS
ROLLOUT_COUNT
ROLLOUT_HORIZON_S
ROLLOUT_THREADS
ROLLOUT_POLICY_THREADS
NATIVE_THREADS
DISCOUNT_FACTOR
PUCT_C
UCT_C
ROOT_NOISE
```

Example:

```bash
export VIDUR_TASK112_GPU=3
export VIDUR_TASK112_TRACE_TYPE=in_distribution
export VIDUR_TASK112_MODEL_BUNDLE=/home/shaz/vidur_vllm_task_1_1_2/artifacts/promoted_models/controller_v125
export VIDUR_TASK112_MCTS_ITERATIONS=2000
export VIDUR_TASK112_PUCT_C=0.5
```

Changing `MODEL_BUNDLE` derives the three DNN paths and native config from the new bundle unless an individual path is also overridden.

Not every dataclass field has a `VIDUR_TASK112_*` override. Capacity fields such as `max_num_batched_tokens`, `max_num_seqs`, and `gpu_memory_utilization` currently require changing/constructing the config. The runner converts those fields to lower-level `VIDUR_VLLM_*` variables after loading it.

## 6. Normal workflow

Run from the deployed repository with its environment active.

### 6.1 Prepare

```bash
python -m vidur_vllm_real_testing.task_1_1_2_runner prepare \
  --output-root /home/shaz/vidur_vllm_task_1_1_2/prepared/in_distribution_20s \
  --tokenizer-dir /path/to/pinned/llama3/tokenizer
```

Preparation is deterministic for the same effective config, tokenizer, profile, and seed.

### 6.2 Validate artifacts

```bash
python -m vidur_vllm_real_testing.task_1_1_2_runner validate \
  --prepared-root /home/shaz/vidur_vllm_task_1_1_2/prepared/in_distribution_20s \
  --tokenizer-dir /path/to/pinned/llama3/tokenizer
```

This validates schemas, counts, prompt hashes and IDs, rounding, simultaneous arrivals, clock metadata, and explicit no-calibration metadata.

### 6.3 Validate runtime and GPU

```bash
python -m vidur_vllm_real_testing.task_1_1_2_runner validate \
  --prepared-root /home/shaz/vidur_vllm_task_1_1_2/prepared/in_distribution_20s \
  --tokenizer-dir /path/to/pinned/llama3/tokenizer \
  --runtime \
  --require-free-gpu
```

Runtime preflight verifies:

- The process is `shaz@mew1`.
- The selected GPU is 2 or 3 and is free when requested.
- vLLM, FlashInfer, Torch, CUDA, and backend versions match.
- Bundle and frozen config files exist.
- Value and policy feature schemas are `markov_v2`.
- Controller value, controller policy, and adversary policy DNNs load.
- No calibration is configured.

### 6.4 Run controller

The output directory must not contain prior files.

```bash
python -m vidur_vllm_real_testing.task_1_1_2_runner run \
  --prepared-root /home/shaz/vidur_vllm_task_1_1_2/prepared/in_distribution_20s \
  --output-dir /home/shaz/vidur_vllm_task_1_1_2/runs/controller_v125_trace01 \
  --mode controller
```

### 6.5 Run baselines

```bash
python -m vidur_vllm_real_testing.task_1_1_2_runner run \
  --prepared-root /home/shaz/vidur_vllm_task_1_1_2/prepared/in_distribution_20s \
  --output-dir /home/shaz/vidur_vllm_task_1_1_2/runs/sjf256_trace01 \
  --mode sjf256

python -m vidur_vllm_real_testing.task_1_1_2_runner run \
  --prepared-root /home/shaz/vidur_vllm_task_1_1_2/prepared/in_distribution_20s \
  --output-dir /home/shaz/vidur_vllm_task_1_1_2/runs/sjf512_trace01 \
  --mode sjf512
```

Set `VIDUR_VLLM_PERSISTENT_SJF=1` when SJF must use the same persistent GV3 accounting/audit path as the controller. The simple non-persistent SJF launcher is useful for throughput checks but does not provide the equivalent GV3 batch ledger.

## 7. Trace and prompt preparation

### 7.1 Arrival groups

[`task_1_1_2_trace.py`](task_1_1_2_trace.py) creates groups at:

```text
arrival_time = group_index * arrival_group_interval_s
```

Groups continue while arrival time is below `static_trace_length_s`. Every request in one group shares the timestamp, intentionally testing atomic simultaneous admission.

Each group obeys both caps:

```text
request_count <= 7
request_count * canonical_prefill_tokens <= 7168
```

### 7.2 In-distribution mode

Physical and canonical prefill lengths are identical and belong to the exact support:

```text
128, 256, 512, 1024, 1536, 2048, 3072, 4096
```

The generated group uses:

```text
support[(group_index + 3) % len(support)]
```

Generated requests in one arrival group are homogeneous.

### 7.3 Out-of-distribution mode

Physical prefill length can be any integer in `[128, 4096]`. Controller length is ceiling-rounded:

```text
canonical_prefill = ceil(actual_prefill / 128) * 128
```

| Actual | Canonical |
| ---: | ---: |
| 128 | 128 |
| 129 | 256 |
| 157 | 256 |
| 255 | 256 |
| 257 | 384 |

The physical prompt remains its actual size.

### 7.4 SLO construction

The configured prefill profile must contain all 128-token points from 128 through 4096. The canonical SLO contract is:

```text
canonical prefill SLO = profiled prefill execution time * 3.0
canonical decode SLO  = 0.05 seconds
```

Preparation stores explicit actual/canonical SLO fields. Decode length remains exact and is not bucketed.

Later runtime code consumes the stored trace contract rather than silently recomputing deadlines from another profile.

### 7.5 Exact English prompts

[`prompt_materialization.py`](prompt_materialization.py) creates deterministic readable English with exact Llama-3 token counts:

1. Reserve one BOS token.
2. Build deterministic headers and paragraph text.
3. Fill the body with known one-token words until it has exactly `target - 1` tokens.
4. Add special tokens and verify exactly `target` IDs.
5. Save text and token IDs.
6. Re-tokenize saved text and verify IDs and hashes.

The benchmark sends saved token IDs, not newly tokenized text. Text exists for human inspection. Token JSON is the physical payload contract.

The tokenizer contract records repository, revision, tokenizer JSON SHA, BOS ID, vocabulary size, and special-token behavior. Any mismatch stops execution before the GPU run.

### 7.6 Prepared artifacts

```text
PREPARED_ROOT/
|-- template_raw_trace.csv
|-- raw_trace.csv
|-- canonical_trace.csv
|-- trace_manifest.json
|-- experiment_config.json
`-- prompts/
    |-- prompt_catalog.json
    |-- text/
    |   |-- req-000000.txt
    |   `-- ...
    `-- token_ids/
        |-- req-000000.json
        `-- ...
```

| File | Meaning |
| --- | --- |
| `template_raw_trace.csv` | Request layout before prompt references are installed. |
| `raw_trace.csv` | Physical request contract and prompt references. |
| `canonical_trace.csv` | Actual and canonical fields used by the registry. |
| `trace_manifest.json` | Hashes, rounding, arrival groups, clocks, and `calibration: null`. |
| `experiment_config.json` | Effective preparation config. |
| `prompts/text/*.txt` | Readable exact-token prompts. |
| `prompts/token_ids/*.json` | One immutable token payload per request. |
| `prompts/prompt_catalog.json` | Request/artifact index and tokenizer manifest. |

One JSON file per request is intentional. It makes each physical payload independently verifiable.

Each token-ID JSON records:

```text
schema version
request ID
text-file reference and SHA256
tokenizer repository, revision, and tokenizer.json SHA256
add_special_tokens setting
BOS token ID
exact token count
ordered token ID array
```

## 8. Request schemas and registry

### 8.1 Raw fields

```text
request_id
arrived_at_s
num_prefill_tokens
num_decode_tokens
prefill_slo_s
decode_slo_s
prompt_mode
prompt_ref
seed
ignore_eos
```

`prompt_mode=token_ids_file` directs the benchmark to the exact JSON payload.

### 8.2 Canonical registry

[`scheduler_contract.py`](scheduler_contract.py) loads `canonical_trace.csv` into `TraceMetadataRegistry`. It preserves physical and canonical totals/SLOs and maps vLLM completion IDs such as `cmpl-<trace-id>-<prompt-index>` back to stable trace identities.

### 8.3 Live snapshot

[`vllm_live_state.py`](vllm_live_state.py) turns vLLM waiting/running objects into immutable snapshots containing:

- Stable request identity and queue.
- Prefill or decode phase.
- Arrival time.
- Actual/canonical totals and remaining work.
- vLLM computed/output-token progress.
- Actual/canonical SLO values.

Duplicate queue membership, prompt-length mismatch, impossible progress, and missing registry metadata are fatal.

### 8.4 Implicit prefill output token

vLLM samples one physical output token in the model step that completes prefill. GV3 advances decode only for explicit decode allocations. The client therefore asks for:

```text
physical max_tokens = logical decode tokens + 1
```

`implicit_prefill_output_tokens=1` removes the first physical output from logical decode progress. It consumes no GV3 decode credit.

## 9. Server process lifecycle

The runner converts config fields into `VIDUR_*` variables and invokes:

```text
mew1/run_real_policy_benchmark.sh MODE OUTPUT_DIR
```

The launcher:

1. Restricts execution to `shaz@mew1` and GPU 2 or 3.
2. Rejects a non-empty output directory.
3. Initializes scheduler and GPU timing logs.
4. installs/verifies the guarded vLLM token-cap patch.
5. Builds a synchronous FlashInfer vLLM command with `TimedGPUWorker`.
6. Chooses controller or SJF planner.
7. Starts vLLM in a new process group with `setsid`.
8. Writes `server.pid` and redirects output to `server.log`.
9. Polls `/health` for up to 600 seconds.
10. Runs a warmup request.
11. Truncates warmup scheduling/timing records.
12. Runs the trace benchmark with a 7200-second timeout.
13. On every exit path, terminates the whole vLLM process group, waits, then uses `KILL` only if needed.

This process-group trap prevents orphaned vLLM workers after a supported run.

## 10. Scheduler construction

vLLM starts with:

```text
--scheduler-cls vidur_vllm_real_testing.gv3_persistent_scheduler.GV3PersistentScheduler
```

The inheritance chain is:

```text
vLLM Scheduler
    ^
GV3Scheduler
    ^
GV3PersistentScheduler
```

[`vllm_scheduler.py`](vllm_scheduler.py) defines the generic snapshot, plan, validation, and stock-vLLM application boundary. [`gv3_persistent_scheduler.py`](gv3_persistent_scheduler.py) adds simultaneous-arrival barriers, exact post-batch progress, persistent updates, timing consumption, and batch audits.

Production binds two plug-ins:

```text
VIDUR_VLLM_GV3_PLANNER=
    vidur_vllm_real_testing.gv3_adapter:GV3PersistentAdapter

VIDUR_VLLM_GV3_STATE_PLANNER=
    vidur_vllm_real_testing.native_dnn_mcts_state_planner:
    ProductionNativeDNNMCTSPlanner
```

The first is the persistent state adapter. The second is the decision algorithm called by that adapter.

## 11. Adapter versus planner

### Adapter

The adapter answers:

> What is the correct GV3 state now, and how does a selected canonical action map to current physical requests?

It owns simulator time, request identities/progress, deadlines, lateness, decode credits, launch history, adversary ticks, terminal state, fast-forward, vLLM reconciliation, and post-batch updates.

### Planner

The planner answers:

> Given the complete GV3 state, which controller action should execute next?

It owns promoted DNN loading, native MCTS, PUCT, rollouts, controller value bootstrap, adversary-policy decisions, root-child ranking, and canonical action generation.

The adapter exposes `plan()` because it is the scheduler plug-in, but learned selection is delegated to the separate state planner after the persistent state is valid.

## 12. One complete scheduling cycle

### 12.1 Submit requests

[`real_trace_benchmark.py`](real_trace_benchmark.py) creates one async HTTP task per request. Each task waits for its logical arrival then posts exact token IDs to `/v1/completions`.

Arrival waits subtract cumulative controller planning. Long MCTS calls therefore do not shift later logical arrivals.

### 12.2 vLLM calls `schedule()`

The engine invokes `GV3PersistentScheduler.schedule()`. A second batch cannot be scheduled while completion feedback for the prior batch is pending. The supported runtime is synchronous with one in-flight batch.

### 12.3 Snapshot vLLM

`GV3Scheduler._snapshot()` reads waiting/running queues and constructs immutable records. Its fingerprint covers ordered request state and the current token budget but excludes capture wall time. A returned plan with a different fingerprint is stale.

### 12.4 Enforce atomic arrivals

Before planning, the persistent scheduler waits until all trace requests sharing the next canonical timestamp are visible. HTTP admission order cannot turn one simultaneous group into several game states.

`VIDUR_VLLM_ARRIVAL_GROUP_TIMEOUT_S` defaults to 10 seconds. Missing peers fail closed.

### 12.5 Reconcile persistent state

The adapter:

1. Registers newly visible requests from canonical metadata.
2. Rejects progress regression and unexplained disappearance.
3. Keeps requests hidden until persistent `sim_time` reaches arrival.
4. Consumes reached 0.2-second adversary ticks.
5. Applies pending terminal/eviction work before another learned action.

Midstream initialization is rejected because missing history would corrupt deadlines, credits, and lateness.

### 12.6 Fast-forward or MCTS

If no prefill is pending but decodes are active, GV3 schedules a real one-token decode batch for every eligible active decode request. That batch executes and advances time; the process repeats until an arrival/adversary boundary creates a meaningful choice.

If nothing is active, state can jump legally to the next event without invented GPU time.

Otherwise the adapter serializes the full GV3 payload and calls `ProductionNativeDNNMCTSPlanner`.

### 12.7 Native search

The planner runs the configured 2000 simulations. Hypothetical tree actions use native GV3 transitions and frozen Vidur execution predictions; they do not use the physical GPU.

Controller nodes use controller policy priors. Adversary nodes and policy continuations use the promoted adversary policy. Leaf/final rollout states use the controller-perspective value DNN. There is no adversary value model in this controller decision path.

Each rollout begins after expanding/selecting a child, follows policy with temperature 1 for the leaf-relative 3-second simulated horizon, bootstraps from controller value, and backs up immediate rewards, discounted continuation rewards, and discounted bootstrap.

### 12.8 Choose root child

Root children are ordered by:

1. Visit count.
2. Q on a visit tie.
3. Stable action index on a remaining tie.

The winner is translated into canonical per-request allocations and optional evictions.

### 12.9 Validate and project

[`scheduler_contract.py`](scheduler_contract.py) checks:

- Current fingerprint.
- Known, unique request IDs.
- Correct prefill/decode phase.
- Exactly one logical token per selected decode.
- No canonical prefill over-allocation.
- Projected physical total within vLLM budget.
- No empty action for a non-empty schedulable state.

A canonical 128-token tail with 37 physical tokens left becomes a 37-token physical allocation.

### 12.10 Build stock vLLM batch

`GV3Scheduler._apply_validated()` temporarily hides unselected requests, installs per-request token caps, and sets the global token budget. It calls `super().schedule()`, restores original structures, and asserts that vLLM's exact token map equals the projected plan.

[`patch_vllm_scheduler.py`](patch_vllm_scheduler.py) installs only the guarded per-request cap hook at pinned source sites. It does not replace vLLM's resource scheduler.

### 12.11 Execute GPU batch

vLLM executes `SchedulerOutput` using [`TimedGPUWorker`](vllm_gpu_timing_worker.py). Only the winning root action reaches the real GPU. Other MCTS branches are simulator-only counterfactuals.

### 12.12 Complete and update

In the pinned synchronous engine:

```text
scheduler.schedule()
model_executor.execute_model(...)
future.result()
scheduler.update_from_output(...)
```

`GV3PersistentScheduler.update_from_output()`:

1. Verifies a pending batch.
2. Verifies exact completed token map.
3. Preserves pre-update request references.
4. Calls stock `super().update_from_output()`.
5. Computes exact before/after progress.
6. Reads the exact matching GPU timing record.
7. Calls `adapter.on_batch_completed()`.

The adapter then advances:

```text
sim_time_after = sim_time_before + measured_gpu_batch_duration
```

It applies progress, deadlines, credits, lateness, drop/stop logic, reached ticks, launch-window pruning, and writes `state_after`. Only then can the next scheduling decision occur.

## 13. Native DNN usage

The production controller loads:

| Artifact | Purpose |
| --- | --- |
| Controller value DNN | Controller-perspective value at leaf/final rollout states. |
| Controller policy DNN | Controller priors and rollout choices. |
| Adversary policy DNN | Adversary choices in search and rollouts. |

Runtime preflight requires both value and policy feature schemas to be `markov_v2`.

Controller value remains controller-perspective at adversary turns. Search handles player/turn semantics rather than switching to an adversary value model.

The adapter passes both persistent payload and live snapshot:

```text
persistent payload
    Simulator time, deadlines, credits, launches, canonical progress,
    terminal state, and GV3 statistics needed by native search.

live snapshot
    Current vLLM identity, phase, actual progress, and physical-tail state
    needed to validate/map the winning canonical action.
```

Snapshot alone loses game history. Payload alone loses the current physical execution boundary.

## 14. SJF semantics

Persistent SJF planners in [`baseline_state_planners.py`](baseline_state_planners.py):

1. Include one decode token for each available decode request.
2. Sort pending prefills by shortest canonical remaining work.
3. Spend up to 256 or 512 canonical prefill tokens.
4. Project rounded tails to physical remaining work.

With persistent SJF enabled, controller and SJF share prompt payload, snapshot, physical projection, vLLM execution, GPU timing, and GV3 completion accounting. Only action selection differs.

## 15. GPU batch timing

### 15.1 CUDA hooks

[`vllm_gpu_timing_worker.py`](vllm_gpu_timing_worker.py) finds the transformer layer list and registers:

- `forward_pre_hook` on the first block: record CUDA start event.
- `forward_hook` on the final block: record CUDA end event.

Events enter the same GPU stream as model kernels. After base execution, the end event is synchronized and `start.elapsed_time(end)` gives elapsed GPU milliseconds.

`transformer_blocks` includes kernels from entry into the first transformer block through completion of the last, including block attention, MLP, and internal normalization/dataflow. It excludes scheduler planning, HTTP, queueing, pre-block CPU work, input embedding before the boundary, output projection after the final block, and sampling.

The worker can also record full-model-forward diagnostic timing, but normal configured GV3 time is transformer-block timing.

### 15.2 File timing channel

Arbitrary worker attributes do not reliably survive vLLM output serialization. [`vllm_gpu_timing_channel.py`](vllm_gpu_timing_channel.py) therefore uses:

```text
TimedGPUWorker -> gpu_forward_timing.jsonl -> GV3PersistentScheduler
```

Each timing row carries the exact token map. The scheduler consumes each record once by byte offset, skips explicit warmups, and rejects missing, non-positive, or mismatched timing.

### 15.3 Wall time is audit-only

Engine wall time can include dispatch, CPU work, synchronization, and serialization. It is diagnostic only when `batch_duration_source=gpu_forward`.

### 15.4 Vidur comparison

[`batch_execution_comparison.py`](batch_execution_comparison.py) reconstructs the exact batch shape and reports:

```text
real transformer-block duration
uncalibrated Vidur prediction
signed/absolute error
within-10-percent flag
```

It asserts persistent clock advance equals the raw measured duration. `calibration_applied` is always `False`.

## 16. Time and cost semantics

### 16.1 Four clocks

| Clock | Definition | Use |
| --- | --- | --- |
| Wall clock | Planning, waiting, and execution | Operational throughput. |
| Logical trace clock | Static arrivals with planning delay removed | Fair request release. |
| Persistent GV3 clock | Completed GPU durations plus legal idle jumps | State, ticks, deadlines, and GV3 cost. |
| MCTS hypothetical time | Native transitions using frozen Vidur prediction | Counterfactual action comparison. |

The root batch advances persistent time using real GPU measurement. Tree branches use Vidur. Planning advances neither game clock.

### 16.2 Credits and fast-forward

Canonical prefill completion mints 216 decode credits when decode remains. Each explicit decode consumes one credit. The transport-only prefill output consumes none. The mint is configurable through `VIDUR_VLLM_GV3_DECODE_CREDIT_MINT`, but changing it changes GV3 semantics and must match the native game configuration.

When only decodes remain, fast-forward performs multiple real decode batches as needed. One decode batch never represents an entire interval.

### 16.3 Lateness and drops

At each completed batch, production applies GV3 deadline and terminal rules. The production defaults are a 2.0-second automatic-drop lateness threshold and fixed drop cost 3.0. A drop truncates work, updates credits, and terminates the vLLM request. It does not apply an unrelated fixed cost 16. The corresponding environment variables are `VIDUR_VLLM_GV3_AUTO_DROP_LATENESS_S` and `VIDUR_VLLM_GV3_DROP_COST`; changing either must remain synchronized with native GV3 semantics.

The final persistent `state_after` is authoritative for violations, lateness, drops, and total GV3 cost.

### 16.4 Two result families

| Result | Source | Meaning |
| --- | --- | --- |
| Real-request metrics | `request_results.csv`, `summary.json` | Streamed TTFT/inter-token observations, raw and planning-corrected. |
| Authoritative GV3 metrics | `scheduler.jsonl.batches.jsonl` final state | Canonical game cost comparable with simulator results. |

Do not substitute HTTP summary cost for GV3 game cost.

## 17. Run artifacts

```text
RUN_DIR/
|-- server.pid
|-- server.log
|-- tokenizer_compatibility.json
|-- warmup.json
|-- scheduler.jsonl
|-- scheduler.jsonl.batches.jsonl
|-- gpu_forward_timing.jsonl
|-- request_results.csv
`-- summary.json
```

| File | Purpose |
| --- | --- |
| `server.pid` | Process-group leader while alive. |
| `server.log` | vLLM startup/runtime output. |
| `tokenizer_compatibility.json` | Runtime tokenizer gate. |
| `warmup.json` | Warmup result; warmup timing logs are removed. |
| `scheduler.jsonl` | Decisions, actions, state, and search diagnostics. |
| `scheduler.jsonl.batches.jsonl` | Completed batches with before/after state, progress, shape, and timing. |
| `gpu_forward_timing.jsonl` | Raw CUDA timing channel. |
| `request_results.csv` | Per-real-request streamed metrics. |
| `summary.json` | Aggregate real-request metrics. |

A transient planning sidecar tracks controller blocking. It is not a durable result replacement.

## 18. Post-processing

### Decision CSV

```bash
python -m vidur_vllm_real_testing.scheduler_log_to_csv \
  RUN_DIR/scheduler.jsonl \
  RUN_DIR/scheduler_actions.csv
```

This includes action labels, actual/canonical allocations, simulator time, priors, Q, visits, and model versions when present.

### Decision plus SLO CSV

```bash
python -m vidur_vllm_real_testing.scheduler_logs_with_slo_to_csv \
  RUN_DIR/scheduler.jsonl \
  RUN_DIR/scheduler.jsonl.batches.jsonl \
  RUN_DIR/steps_with_slo.csv
```

This joins decisions to completed batches by exact token allocation and adds incremental/cumulative GV3 cost.

### Authoritative GV3 summary

```bash
python -m vidur_vllm_real_testing.persistent_run_summary \
  RUN_DIR/scheduler.jsonl.batches.jsonl \
  RUN_DIR/gv3_summary.json \
  --policy controller
```

It reports batch count, final simulator time, generated/completed/active/stopped/dropped counts, violations, lateness, total cost, and timing source.

## 19. Tests

Run the full local suite:

```bash
cd /home/shazer/Desktop/Research/Vidur/vidur-classical-search
python -m unittest discover -s vidur_vllm_real_testing/tests -v
```

### Trace and canonicalization tests

They cover strict schemas, duplicate IDs, in/out-distribution conversion, actual/canonical preservation, 128-token boundaries, tail projection, decode identity, profile provenance, window caps, deterministic sorting, and manifests.

### Prompt and tokenizer tests

They cover all supported exact prompt sizes, re-tokenization, BOS behavior, tokenizer hashes, artifact corruption, and runtime tokenizer compatibility.

### Scheduler contract tests

They cover controller/SJF plans, stale fingerprints, duplicate/unknown IDs, phase validation, one-token decode, empty action, budgets, registry mapping, deterministic fingerprints, atomic arrivals, and timeout.

### Persistent-state tests

They cover future arrivals, repeated decode fast-forward, prefill transport-token correction, lateness boundaries, credits, evictions, automatic drops, drift, midstream rejection, payload fields, and invalid durations.

### Native DNN tests

They cover action translation, over-allocation, root ordering by visits/Q/index, HGB rejection, DNN metadata, Markov schemas, and bundle/config loading.

### GPU timing and comparison tests

They cover exact timing-map matching, warmup skipping, missing/mismatched records, exact multi-request shapes, predictor requests, clock equality, and no-calibration output.

### Real benchmark tests

They cover planning-frozen arrivals, request-scoped overlap, corrected SLOs, and final persistent summary accounting.

## 20. Smoke and hardware acceptance

### Scheduler smoke

```bash
python -m vidur_vllm_real_testing.vllm_scheduler_smoke \
  --trace /path/to/prepared/canonical_trace.csv
```

This tests snapshot, planner contract, validation, and projection without a real vLLM server.

### Native planner smoke

```bash
python -m vidur_vllm_real_testing.native_dnn_planner_smoke \
  --model-bundle /path/to/promoted/bundle \
  --iterations 16 \
  --rollout-count 1 \
  --rollout-horizon-s 0.2 \
  --puct-c 0.5
```

Reduced search values test loading/plumbing only, not policy quality.

### Real GPU acceptance

Before a long run:

1. Run runtime validation with `--require-free-gpu`.
2. Run a short prepared trace.
3. Confirm health and warmup.
4. Confirm decision and completed-batch logs.
5. Confirm one usable timing per batch.
6. Confirm each clock delta equals authoritative duration.
7. Compare first action with local native search from the identical initial state.
8. Confirm no calibration field/scale.

## 21. Deployment on mew1

The deployment is user-space under `/home/shaz`; Docker and system changes are not required.

[`mew1/deploy_from_local.sh`](mew1/deploy_from_local.sh) copies source, profiles, promoted artifacts, and source hashes. Its historical default root/model source may differ from Task 1.1.2 defaults. Set `VIDUR_MEW1_ROOT` and verify its model source before a new deployment.

[`mew1/bootstrap_user_env.sh`](mew1/bootstrap_user_env.sh) builds/validates the user environment after source deployment. [`mew1/validate_install.sh`](mew1/validate_install.sh) verifies it.

Supported runtime:

```text
host: mew1
user: shaz
GPU: 2 or 3
TP: 1
PP: 1
```

Never terminate another user's process. Preflight and launchers refuse an occupied selected GPU.

## 22. Failure diagnosis

### GPU occupied

Use `nvidia-smi -i 2` or `nvidia-smi -i 3`, choose a free allowed GPU, or wait.

### Non-empty output

Use a new immutable result directory. Do not mix two runs.

### Prompt/tokenizer mismatch

Regenerate the prepared trace with the intended pinned tokenizer. Never bypass the hash/token gate.

### Arrival-group timeout

A same-time peer did not reach vLLM. Inspect HTTP/client and server admission failures. Planning from a partial group is invalid.

### Stale fingerprint

The queue changed while planning or a plan was reused. Recompute from the new snapshot.

### Scheduled-token mismatch

Inspect the pinned scheduler patch, vLLM version, tail projection, token budget, and phase. Never advance GV3 from a batch different from the requested one.

### Timing missing/mismatched

Confirm `TimedGPUWorker`, writable timing path, warmup truncation, and exact token maps. No wall-time/predictor fallback is silently substituted.

### DNN bundle rejected

Inspect manifest, `native_model.tsv` headers, roles, versions, and Markov schemas. Production requires controller value, controller policy, and adversary policy DNNs compatible with the frozen config.

### First action differs from simulator

At the initial decision no real batch has completed. Compare registry IDs, simultaneous arrivals, actual/canonical conversion, model/config paths, deterministic seed, and frozen predictor bundle.

### Cost looks impossible

Identify whether it came from HTTP metrics or persistent GV3 accounting. Regenerate `steps_with_slo.csv` and `gv3_summary.json` from the batch audit for game cost.

### Timing differs across runs

Check `VIDUR_VLLM_GPU_TIMING_SCOPE`. `transformer_blocks` and `full_model_forward` measure different boundaries. Also compare model, dtype, backend/version, eager mode, TP/PP, batch shape, and GPU.

## 23. File map

### Trace and prompt

| File | Role |
| --- | --- |
| [`canonicalization.py`](canonicalization.py) | Schemas, conversion, profile SLOs, manifests. |
| [`task_1_1_2_trace.py`](task_1_1_2_trace.py) | Deterministic task trace and artifacts. |
| [`prompt_materialization.py`](prompt_materialization.py) | Exact English prompts and verification. |
| [`prepare_trace.py`](prepare_trace.py) | General canonicalization CLI. |
| [`generate_english_trace.py`](generate_english_trace.py) | General prompt CLI. |
| [`verify_ready_trace.py`](verify_ready_trace.py) | Ready-trace integrity gate. |
| [`feature_audit.py`](feature_audit.py) | Static Markov feature audit. |

### Scheduler

| File | Role |
| --- | --- |
| [`scheduler_contract.py`](scheduler_contract.py) | Registry, snapshots, plans, validation, projection. |
| [`vllm_live_state.py`](vllm_live_state.py) | vLLM objects to live snapshots. |
| [`vllm_scheduler.py`](vllm_scheduler.py) | Generic custom/stock scheduler boundary. |
| [`gv3_persistent_scheduler.py`](gv3_persistent_scheduler.py) | Persistent completed-batch lifecycle. |
| [`patch_vllm_scheduler.py`](patch_vllm_scheduler.py) | Guarded per-request token caps. |
| [`baseline_state_planners.py`](baseline_state_planners.py) | SJF-256/SJF-512 planners. |

### GV3 and MCTS

| File | Role |
| --- | --- |
| [`gv3_live_adapter.py`](gv3_live_adapter.py) | Ledger, reconciliation, time, fast-forward, mapping. |
| [`gv3_adapter.py`](gv3_adapter.py) | Production terminal/drop/credit semantics. |
| [`scheduler_planner.py`](scheduler_planner.py) | Plug-in loader and callable boundary. |
| [`native_dnn_mcts_state_planner.py`](native_dnn_mcts_state_planner.py) | Frozen config/predictor construction. |
| [`native_dnn_mcts_planner.py`](native_dnn_mcts_planner.py) | DNN validation and native MCTS. |
| [`export_native_mcts_cfg.py`](export_native_mcts_cfg.py) | Lean frozen deployment config. |

### Real execution and results

| File | Role |
| --- | --- |
| [`real_trace_benchmark.py`](real_trace_benchmark.py) | HTTP replay and real-request metrics. |
| [`vllm_gpu_timing_worker.py`](vllm_gpu_timing_worker.py) | CUDA instrumentation. |
| [`vllm_gpu_timing_channel.py`](vllm_gpu_timing_channel.py) | Batch-keyed timing transport. |
| [`batch_execution_comparison.py`](batch_execution_comparison.py) | Real versus uncalibrated Vidur. |
| [`scheduler_log_to_csv.py`](scheduler_log_to_csv.py) | Decision CSV. |
| [`scheduler_logs_with_slo_to_csv.py`](scheduler_logs_with_slo_to_csv.py) | Decision plus GV3 SLO CSV. |
| [`persistent_run_summary.py`](persistent_run_summary.py) | Final authoritative GV3 summary. |
| [`mew1/run_real_policy_benchmark.sh`](mew1/run_real_policy_benchmark.sh) | Complete run lifecycle. |
| [`mew1/launch_production_native_dnn_controller.sh`](mew1/launch_production_native_dnn_controller.sh) | Production vLLM command and bindings. |

[`profiling_accuracy/`](profiling_accuracy/) is a separate supporting subsystem. It measures vLLM transformer-block batches and compares them with uncalibrated Vidur predictions. It does not change the controller or predictor automatically.

## 24. Safe modification checklist

### New promoted model

1. Deploy a complete immutable bundle.
2. Set `VIDUR_TASK112_MODEL_BUNDLE`.
3. Verify derived/explicit component paths.
4. Print effective config.
5. Run runtime preflight.
6. Run native smoke.
7. Run first-action parity before a long benchmark.

### New profile or predictor

1. Use a new versioned path.
2. Update trace profile and frozen native config consistently.
3. Regenerate prepared traces because SLOs/hashes are immutable.
4. Verify no calibration.
5. Run profile, batch-shape, and clock tests.
6. Never overwrite a historical training profile.

### New canonicalization

1. Update trace and model-support contracts together.
2. Preserve actual fields.
3. Add physical-tail boundary tests.
4. Regenerate prompts/manifests.
5. Re-run tokenizer and first-action parity.

### New vLLM version

1. Treat it as an integration change.
2. Revalidate guarded patch source markers.
3. Verify exact `schedule -> execute_model -> update_from_output` order.
4. Re-test one-in-flight-batch behavior.
5. Re-test exact token/timing matching.
6. Re-run real GPU smoke.

### New timing scope

1. Give it a distinct name.
2. Record it in results.
3. Do not compare it directly with transformer-block data.
4. Document included boundaries.
5. Keep raw CUDA measurements unscaled.

## 25. Current limitations

- Runtime is restricted to `shaz@mew1`, GPU 2 or 3.
- Only TP=1 and PP=1 are supported.
- Async scheduling and frontend multiprocessing are disabled.
- vLLM, FlashInfer, Torch, and CUDA versions are pinned.
- The source token-cap hook is guarded but version-sensitive.
- This directory evaluates promoted models; it does not train them.
- Counterfactual MCTS uses frozen Vidur; only the selected root batch uses real GPU time.
- Timing mismatch is reported, never calibrated away.
- HTTP SLO metrics and canonical GV3 cost answer different questions.

Within these limits, the pipeline gives a reproducible path from exact prompt generation through real GPU execution and auditable persistent GV3 state updates.
