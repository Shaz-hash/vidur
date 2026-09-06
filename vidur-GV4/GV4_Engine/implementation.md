GV4_Engine/
├── config.py               # topology, KV, PP and schema configuration
├── state.py                # compact request/replica/in-flight state
├── kv_ledger.py            # finite block accounting
├── pipeline_calendar.py    # TP/PP stage timing
├── action_resolver.py      # validation and canonicalization
├── transition_engine.py    # boundary advancement and completion commits
├── fast_forward.py         # deterministic decode-only progression
├── vidur_timing_provider.py # Vidur cache construction and timing adaptation
├── logger/
│   ├── __init__.py         # public logger exports
│   └── node_csv_logger.py  # joined MCTS/request/pipeline/KV node snapshots
└── virtual_environment.py  # thin MCTS-facing orchestration



The existing GV3 implementation remains available as the known-correct baseline. A GV4VirtualVidurMCTSEnvironment will expose the same methods used by MCTS:
initial_state()
sample_controller_actions()
sample_adversary_actions()
apply_controller_action_only()
apply_adversary_action_only()
evaluate_objective()
describe_state()


ideally with the right params GV4 should be able to replicate the environment of the GV3


# Implemented Foundation

The first GV4 engine layer now consists of two deliberately separate files:

- config.py defines the immutable experiment and engine contract.
- state.py defines the mutable runtime data that must be copied for an MCTS branch.

Keeping these responsibilities separate is important for both correctness and speed. A
configuration is created and validated once, then shared by every branch. Only the
compact runtime state is cloned while MCTS explores alternative actions.


## config.py: Immutable Engine Contract

### Purpose

config.py describes the exact system that GV4 is simulating. It fixes the model,
hardware topology, Vidur cache settings, KV-cache geometry, scheduling limits, timing,
adversary workload budgets, SLOs, costs, action schemas, routing rules, and native
array bounds.

The configuration classes use frozen, slotted dataclasses so that:

- an experiment cannot silently change its semantics after startup;
- configuration records do not carry per-instance dictionaries;
- one configuration can be safely shared by all MCTS branches;
- the same resolved values can be serialized for Python/native parity checks.

Invalid or unsupported combinations fail during construction through GV4ConfigError.
GV4 therefore fails closed instead of silently approximating a topology or feature
that schema v1 does not support.


### Configuration Records

| Record | Responsibility |
| --- | --- |
| LayerRange | Represents the half-open transformer-layer interval [begin, end) assigned to one PP stage. |
| ReplicaPlacement | Stores the ordered TP rank IDs belonging to every PP stage of one replica. |
| TopologyConfig | Defines the number of replicas, TP width, PP width, stage layer ranges, and physical rank placement. |
| ModelConfig | Stores model dimensions and dtypes needed to validate tensor partitioning and derive KV bytes per token per rank. |
| VidurPredictorConfig | Defines the device, network, cache mode/directory, and lookup-table bounds needed to construct Vidur's predictor once at startup. Model, TP, PP, and KV block size are derived from existing records rather than repeated. |
| KVCacheConfig | Defines per-rank KV byte budgets, one hard safety margin, token block size, and allocation granularity. |
| SchedulerConfig | Defines legal batch size, sequence count, prefill chunk size, in-flight depth, PP queue capacity, and stage-ordering rules. |
| TimingConfig | Defines adversary ticks, launch windows, floating-point tolerance, and zero-time transition guards. |
| DecodeCreditConfig | Defines the pooled decode-credit mint per completed prefill and nonnegative overspend protection. Prefill has no credit configuration. |
| RequestConfig | Defines legal prefill/decode lengths and the adversary target-average constraints. |
| SLOConfig | Defines the prefill slowdown target and per-token decode deadline interval. |
| CostConfig | Defines violation, lateness, terminal-drop, and automatic-drop costs. |
| RewardConfig | Defines elapsed-time discounting and the minimize-cost value convention used by MCTS and replay targets. |
| ControllerActionConfig | Defines and indexes the fixed raw controller action product: resumable preemption rule, terminal eviction rule, shared prefill/recompute budget, and ordering heuristic. |
| AdversaryActionConfig | Defines and indexes launch count, prefill-size template, and decode stop rule. |
| RoutingConfig | Defines sequential assignment to replicas and forbids combinatorial multi-request routing actions. |
| NativeLayoutConfig | Versions the manifest/state/action/feature/native schemas and fixes maximum array dimensions for a later native layout. |
| GV4EngineConfig | Composes every record, checks cross-record consistency, derives capacities and action dimensions, and produces the canonical manifest hash. |


### Topology and Placement

TopologyConfig models homogeneous replicas with configurable TP and PP dimensions. For
each replica, ReplicaPlacement.stage_rank_ids[k] lists the TP ranks that execute PP
stage k. Rank IDs must be unique and contiguous because the intended native state uses
direct array indexing rather than maps.

TopologyConfig.contiguous() is a convenience constructor for the common equal-stage
case. It divides model layers evenly across PP stages and assigns contiguous rank IDs.
An explicitly constructed TopologyConfig can represent a profiled uneven layer
partition when require_equal_stage_layers is false.

For multiple replicas, routing must be enabled. A single replica may use implicit
assignment and therefore avoid a router decision entirely.


### Vidur Predictor Startup and Accounting

`GV4VirtualVidurMCTSEnvironment.from_config(config)` is the complete startup entry
point. It derives Vidur's model name from `ModelConfig`, TP and PP from
`TopologyConfig`, and block size from `KVCacheConfig`. `VidurPredictorConfig` supplies
only values that are not already represented: device, network device, cache location,
cache mode, predictor bounds, and lookup granularities.

Run from the repository root and import `GV4_Engine` through the top-level package
bridge. Do not prepend the complete `vidur-GV4/` copy to `PYTHONPATH`: that legacy
tree contains a package named `types`, which can shadow Python's standard-library
module. The bridge exposes only `vidur-GV4/GV4_Engine` and leaves the standard library
and the canonical `vidur` package unambiguous.

The startup path constructs Vidur's random-forest predictor directly. It does not run
`vidur.main` or a workload simulation. Cache behavior follows Vidur's native modes:

- `use_cache` loads existing tables and creates missing models/tables from the raw
  profiling CSVs;
- `require_cache` loads only complete existing caches and fails when one is missing;
- `ignore_cache` neither reads nor writes predictor caches.

AlphaGoZero workers should use `require_cache` after one controlled `use_cache` build.
Cache generation is startup work and is never performed inside an MCTS transition.

A ready TP2/PP2 environment is created from one manifest as follows:

```python
from GV4_Engine.config import (
    GV4EngineConfig,
    KVCacheConfig,
    ModelConfig,
    TopologyConfig,
    VidurPredictorConfig,
)
from GV4_Engine.virtual_environment import GV4VirtualVidurMCTSEnvironment

model = ModelConfig(
    model_id="meta-llama/Meta-Llama-3-8B",
    model_revision="<immutable-model-revision>",
)
topology = TopologyConfig.contiguous(
    num_replicas=1,
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    num_layers=model.num_layers,
)
kv_bytes_per_rank = 8 * 1024**3  # Replace with the measured post-reservation budget.
config = GV4EngineConfig(
    model=model,
    topology=topology,
    vidur_predictor=VidurPredictorConfig(
        device="h100",
        network_device="h100_dgx",
        cache_dir="/absolute/path/to/gv4_h100_tp2_pp2_cache",
        cache_mode="use_cache",  # Use require_cache after this one-time build.
    ),
    kv_cache=KVCacheConfig(
        kv_budget_bytes_per_rank=(kv_bytes_per_rank,) * topology.total_ranks,
    ),
)
environment = GV4VirtualVidurMCTSEnvironment.from_config(config)
```

Constructing `GV4EngineConfig` itself is intentionally side-effect free. Predictor
training or cache loading occurs only when
`config.create_vidur_timing_provider()` is called. The virtual-environment
`from_config()` factory delegates to that method. This keeps manifest validation,
serialization, and MCTS state operations from accidentally triggering expensive
cache generation.

The startup sequence is:

1. Build Vidur's random-forest execution predictor from the resolved model,
   TP/PP, device, network, block size, and prediction bounds.
2. Let Vidur apply `cache_mode`. With `use_cache`, existing models/tables are
   loaded and missing artifacts are trained from the repository profiling CSVs.
   With `require_cache`, any missing Vidur artifact is an error.
3. Load or derive the single-request prefill SLO profile from that predictor.
4. Keep the predictor, the SLO profile, and the bounded runtime batch LRU in one
   `VidurTimingProvider` shared by the environment and tests.

There is deliberately no separate cache-generator module and no
`python -m vidur.main` subprocess. Predictor construction itself is Vidur's
supported cache bootstrap operation.

#### Derived Prefill SLO Profile

The provider stores the derived file inside `VidurPredictorConfig.cache_dir`:

    prefill_profile_TP{TP}_PP{PP}_{device}_{network_device}.csv

For TP2/PP2 on H100 DGX this is:

    prefill_profile_TP2_PP2_h100_h100_dgx.csv

`prefill_profile_step_tokens` defaults to 128. The file contains one fresh,
single-request prefill prediction for 128, 256, ..., through
`RequestConfig.max_prefill_tokens_per_request` (4096 in the test manifest).
Its columns are interleaved in pipeline order:

    prefill_request_size_tokens
    stage_0_computation_time_sec
    stage_0_1_pp_communication_time_sec
    stage_1_computation_time_sec
    ...
    end_to_end_prefill_time_sec

Each stage computation value includes that stage's GPU kernels and TP collectives
but excludes PP transfer. Each boundary column contains only PP send/receive time.
The final column is the sum of all stage and boundary values and is the base
prefill duration used by the SLO deadline calculation.

At startup the provider validates the exact header, complete token grid, finite
durations, and every row's component sum. A missing, partial, stale-shape, or
malformed CSV is regenerated from the already loaded predictor, even in
`require_cache` mode. Generation uses an inter-process lock and atomic rename,
so concurrent workers cannot observe a partial file. The validated totals are
then loaded into a tuple; request-deadline lookups perform no disk I/O and no
Vidur call. A non-grid residual prefill amount uses the next 128-token profile
point conservatively.

Because the requested filename identifies hardware topology but not model name,
each model/profile combination must use its own dedicated `cache_dir`.

For every resolved batch, `VidurTimingProvider` builds Vidur predictor requests and
returns:

    stage service time = stage GPU compute + TP communication
    PP boundary time   = send/receive communication to the next stage

Vidur's `ExecutionTime.model_time` already includes PP communication, so the provider
subtracts that component before returning stage service and emits it separately for
the pipeline calendar. It never uses `ExecutionTime.total_time`, which would add CPU
overheads. Equivalent batch shapes are retained in a bounded in-memory LRU cache for
fast repeated MCTS queries.


### KV-Capacity Derivation

For each PP stage, GV4EngineConfig.kv_bytes_per_token_by_stage() derives:

    KV bytes/token/rank =
        2 * KV element bytes * layers in stage
          * head dimension * KV heads on TP rank

The factor 2 accounts for both key and value. Grouped-query KV heads are either
partitioned or replicated across TP ranks according to the model and TP dimensions.

KVCacheConfig.kv_budget_bytes_per_rank is the memory remaining after model weights,
runtime buffers, and other non-KV reservations. GV4 applies the configured
memory_safety_margin_fraction once:

    hard KV bytes/rank =
        floor(KV budget/rank * (1 - safety margin))

    KV blocks/rank =
        hard KV bytes/rank // bytes per KV block on that stage

The logical capacity of a replica is the minimum capacity across its ranks because a
request logical KV block must be representable on every participating rank. GV4 has
no second allocator watermark, prefix caching, or CPU/disk KV offload. Resumable KV
preemption is implemented in both Python and native GV4 using the same authoritative
request and per-rank ledgers; no second allocator or KV-manager state is introduced.


### Scheduler and Action Bounds

SchedulerConfig.max_batch_tokens bounds the total ordinary prefill, reconstruction,
and decode tokens admitted in one microbatch. max_sequences independently bounds the
number of distinct requests in that microbatch. max_prefill_chunk_tokens bounds one
prefill-class contribution, and one batch may reserve at most one decode token per
request. request_preemption_enabled is a real default-on feature gate. Disabling it
keeps the fixed raw policy width but removes all preemption effects through normal
masking/canonicalization.

max_inflight_microbatches limits all admitted but not yet pipeline-complete batches for
one replica. inter_stage_queue_capacity limits work waiting between PP stages. Until
explicit PP backpressure is implemented, the configuration requires queue capacity to
be at least the in-flight limit.

The controller and adversary expose stable raw action indices. Their encode/decode
methods make the mapping deterministic for training data, Python execution, and the
future native implementation. Legality and canonicalization of a raw action remain the
responsibility of action_resolver.py; config.py only defines the raw schema and bounds.

The eviction names in ControllerActionConfig describe terminal removal policies. They
must not be interpreted as resumable preemption: eviction permanently drops a request,
whereas preemption preserves logical progress and discards only physical KV. Waiting
preemption releases memory immediately. In-flight preemption is drain-before-release;
in-flight terminal eviction remains deferred through DROP_PENDING.


### Timing, Launch Windows, Decode Credits, SLOs, Costs, and Rewards

TimingConfig preserves the GV3-style alternating boundary model, including the
adversary tick interval, launch window, floating-point epsilon, time rounding, and a
guard against an infinite chain of zero-time decisions.

There is no prefill-credit ledger. Adversary prefill pressure is bounded directly by
the sliding launch window: both request count and aggregate launched prefill tokens
must remain within their configured caps. Controller prefill admission is therefore
limited only by the selected action budget, remaining request work, scheduler limits,
KV capacity, and pipeline capacity.

DecodeCreditConfig controls one pooled, non-expiring adversary workload budget. A
request mints 216 credits exactly once when its final prefill token commits at the
last PP stage. One credit is charged for each decode token that actually commits. An
issued in-flight token is reserved first and charged when it drains, preventing
multiple in-flight batches from spending the same credit.

Decode credit is not a controller strategy or a controller raw-action dimension. The
controller cannot decide to mint, discard, or transfer it. Automatic decode inclusion
uses only adversary-funded decode work; the controller then decides the physical batch
subject to token, sequence, KV, and pipeline limits.

Reserving the final available credit is a system-wide terminal boundary for the decode
requests that already exist. Every WAITING_DECODE request becomes STOPPED immediately
and releases its KV. Every INFLIGHT_DECODE request becomes STOP_PENDING; its
already-reserved token is its final token, drains through the pipeline, is charged, and
then releases KV. A later prefill completion may mint new credits, but it cannot revive
those stopped requests.

For an initial balance of zero, the runtime invariant is:

    available_decode_credit
    + reserved_decode_credit
    + total_committed_decode_tokens
    = decode_credit_mint * requests_that_reached_decode

The balance is pooled across requests. Stopping a request early charges no future
tokens and leaves unused pooled credit available; any token already in flight still
commits and is charged. This permits an individual request to exceed 216 tokens, up to
the 864-token request cap, when earlier short requests banked enough credit. It also
bounds the cumulative average at 216 tokens per request that reached decode. The
average is near 216 only when the adversary spends most of its available budget. If
the available pool reaches exactly zero, all then-active decodes terminalize as
described above; credits minted later fund new decode entrants rather than resuming
them.

RequestConfig defines legal request lengths and workload-average constraints.
SLOConfig defines prefill and decode deadlines. CostConfig defines the capped violation
and terminal costs. RewardConfig converts elapsed simulator time into the discount:

    discount(elapsed) =
        discount_factor ** (elapsed / discount_reference_step_sec)

This keeps reward backup based on elapsed simulated time rather than assuming every
transition has equal duration.


### Cross-Configuration Validation and Manifest

GV4EngineConfig.validate() checks relationships that no individual record can check
alone, including:

- model layers, profile layer ranges, and PP dimensions agree;
- hidden size, attention heads, and KV heads are compatible with TP;
- one KV budget exists for every rank and every rank can hold at least one block;
- scheduler limits do not exceed the measured execution-profile range;
- request sizes and action templates are aligned to the profile lookup interval;
- multiple replicas have an explicit routing policy;
- all configured dimensions fit the fixed native-layout bounds;
- launch-history and zero-time guards are large enough for their configured windows;
- the decode-credit mint equals the configured target decode average and does not
  exceed the per-request decode cap.

to_manifest_json() serializes both configured values and derived dimensions in a
deterministic order. manifest_sha256() hashes the canonical representation. Every
runtime state stores this hash, so restoring or evaluating a state with a different
configuration fails instead of producing mixed semantics.


## state.py: Compact Branchable Runtime State

### Purpose and Boundary

state.py contains only authoritative data that can differ between MCTS branches. It
does not choose actions, allocate KV blocks, predict service times, advance the
pipeline, commit completions, or perform disk snapshot I/O. Those operations belong in
action_resolver.py, kv_ledger.py, pipeline_calendar.py, transition_engine.py, and
virtual_simulator.py.

This boundary prevents duplicated derived state from drifting out of sync and keeps the
hot clone() path straightforward enough to translate to fixed native arrays.


### IDs, Turns, and Lifecycle

Player identifies whose decision is next: ADVERSARY, CONTROLLER, or ROUTER. NO_ID,
UNASSIGNED_REPLICA, and UNSET_TIME use -1 sentinels so IDs and optional times fit
compact scalar fields.

RequestLifecycle describes the one authoritative phase of each request:

| Lifecycle | Meaning |
| --- | --- |
| WAITING_PREFILL | The request is assigned but still has prefill work not in flight. |
| INFLIGHT_PREFILL | Prefill work and required resources are reserved in one admitted microbatch. |
| WAITING_DECODE | Prefill is complete and decode work remains. |
| INFLIGHT_DECODE | One decode step is reserved in an admitted microbatch. |
| INFLIGHT_RECOMPUTE | A preempted request is rebuilding part of its missing physical KV context. Its committed logical progress does not change. |
| PREEMPT_PENDING | The controller selected an in-flight request for preemption. Its already-admitted work must finish before all updated physical KV is released. |
| STOP_PENDING | An adversary stop was requested while in-flight work prevents immediate removal. |
| DROP_PENDING | A terminal drop was requested while in-flight work prevents immediate removal. |
| COMPLETED | All requested work completed naturally. |
| STOPPED | The adversary terminated the request. |
| DROPPED | The controller or automatic SLO rule terminated the request. |

TerminalReason records why a terminal or pending-terminal transition occurred.
Preemption is not terminal and therefore does not assign a terminal reason. It does
not erase deadlines, accumulated lateness, completed tokens, or decode-credit history.
For PREEMPT_PENDING, the admitted allocation can still update these fields normally
before its physical KV is released.


### Runtime Records

| Record | Authoritative data retained |
| --- | --- |
| LaunchRecord | One adversary launch used by the sliding launch-window constraint. |
| BatchAllocation | Exactly one kind of request work: prefill, decode, or KV recomputation, plus newly reserved logical KV blocks. |
| InflightMicrobatchState | Batch/action IDs, request allocations, and precomputed ready/start/finish time for every PP stage. |
| RequestState | Request identity, owner, lifecycle, logical token progress, physical KV progress, KV ownership, SLO data, in-flight link, one-time decode-credit mint flag, lateness, and terminal fields. |
| ObjectiveState | Cumulative completion, stop, drop, violation, lateness, terminal-cost, and total-cost counters. |
| ReplicaState | Rank placement/capacity, committed and reserved KV ledgers, compact PP stage tails, and the ordered in-flight batch ring. |
| GV4State | Global time/turn, deterministic ID and RNG counters, launch history, pooled available/reserved decode credit, cumulative minted/committed decode-token counters, all requests, all replicas, and objective totals. |


### Committed, Reserved, and Remaining Work

GV4 separates logical request progress from physical KV state:

- committed prefill/decode fields mean useful request work passed the final PP stage;
- reserved prefill/decode fields mean useful work is admitted but has not passed the final PP stage;
- kv_computed_tokens is the logical context currently represented by physical GPU KV;
- reserved_recompute_tokens is missing context being rebuilt by an in-flight batch;
- remaining work is derived as original minus committed minus reserved and is not stored separately.

For example, while one decode token is in flight, it is reserved but not committed. On
final-stage completion, transition_engine.py will move that token and its new KV blocks
from reserved to committed. If admission is rejected, no reservation is made.

For every nonterminal request:

    logical_context_tokens = committed_prefill_tokens + committed_decode_tokens
    remaining_recompute_tokens =
        logical_context_tokens - kv_computed_tokens - reserved_recompute_tokens
    resident_tokens =
        kv_computed_tokens
        + reserved_recompute_tokens
        + reserved_prefill_tokens
        + reserved_decode_tokens

Before preemption, kv_computed_tokens normally equals logical_context_tokens. A
preemption releases all blocks and resets kv_computed_tokens to zero while preserving
logical_context_tokens. RequestState.resident_tokens therefore means physical KV
occupancy, not historical or logical progress. The minimum block count is:

    minimum blocks =
        (resident_tokens + block_size_tokens - 1) // block_size_tokens

Terminal requests retain historical token counters for reporting but must have zero
physical KV tokens and zero committed/reserved KV blocks.

Recomputation and ordinary prefill remain distinct accounting kinds, but they form one
controller scheduling class. They use the same queue ordering, prefill action budget,
batch token/sequence limits, pipeline stages, and Vidur prefill timing. One allocation
contains exactly one of prefill, decode, or recompute work. If reconstruction remains,
new prefill and decode work for that request are illegal.


### Compact Pipeline Calendar

Every in-flight microbatch stores immutable tuples of per-stage ready, start, and
finish times. For each stage the state requires:

    ready <= start < finish

The next stage cannot become ready before the prior stage finishes. Batches are kept in
ascending microbatch-ID order, and FIFO work on the same stage cannot overlap.

ReplicaState.stage_tail_finish_times[k] is the right edge of all work currently
scheduled on PP stage k: the latest finish time known for that stage. Therefore each
in-flight batch finish on stage k must be less than or equal to that tail. The tail is
not necessarily the finish time of the batch currently executing at state.now; it can
be a future finish time for queued pipeline work.

ReplicaState.stage_last_microbatch_ids retains the ID associated with the latest
scheduled work on each stage. pipeline_calendar.py will update this alongside the stage
tail whenever a new microbatch is admitted.


### KV Ledgers and Cross-Links

Each request owns logical committed and reserved KV block counts. The replica mirrors
their aggregate on every participating rank using rank_kv_committed_blocks and
rank_kv_reserved_blocks. GV4State.assert_valid() recomputes the aggregate from requests
and requires the rank ledgers to match exactly.

An in-flight request points to exactly one microbatch. That microbatch must point back
to the same request through a BatchAllocation, use the same replica, and contain the
same prefill, decode, recompute, and KV reservations. Reserved decode credit must equal
the sum of in-flight decode allocations. Prefill and recompute allocations have no
credit reservation. These two-way checks prevent a branch from gaining decode work
without funding or any physical work without KV capacity.

The global counters additionally enforce the conservation equation:

    decode_credits_available
      + decode_credits_reserved
      + decode_tokens_committed_total
      == decode_credits_minted_total

Each request can set decode_credit_minted only once, after its final prefill token
commits. This makes duplicate minting and double spending fail during state
validation. Zero available credit cannot coexist with WAITING_DECODE or
INFLIGHT_DECODE. A decode-phase INFLIGHT_RECOMPUTE request is also treated as active
decode work. Only a STOP_PENDING request may retain its already-reserved final decode
token.


### Initial State, Lookup, and MCTS Cloning

GV4State.initial(config) creates an empty topology-shaped state. It copies derived
per-rank capacities into each replica, initializes every PP tail at now, stores the
config manifest hash, and initializes deterministic ID/RNG counters.

Requests and replicas are append-only arrays whose IDs equal their positions. This
allows request(id) and replica(id) to be O(1) and provides a direct path to native array
indexing.

GV4State.clone() copies all mutable branch data:

- request, replica, and objective records are independently copied;
- mutable KV arrays, stage-tail arrays, and in-flight lists are independently copied;
- frozen BatchAllocation records and immutable timing tuples are safely shared.

The clone method intentionally does not run the full validator. Debug/test boundaries
call assert_valid() explicitly, avoiding a full cross-state scan on every MCTS branch
copy.


### Full-State Validation

GV4State.assert_valid(config) is the expensive debug and parity checker. It verifies:

- state schema and configuration manifest identity;
- legal time, player, ID, decode-credit, tie-break, and RNG counters;
- ordered launch history and its sliding-window bounds;
- configured replica placement, PP width, rank capacities, and in-flight limits;
- globally unique batch IDs and ordered FIFO pipeline calendars;
- request lifecycle, progress, deadline, terminal, and KV invariants;
- logical-context, physical-KV, and partial-recomputation invariants;
- request-to-batch and batch-to-request links;
- request KV ownership against every rank ledger;
- in-flight decode reservations against the pooled decode-credit ledger;
- objective counters against terminal and violation state of requests.

This validator is intended for tests, debug transitions, snapshot restore, and
Python/native parity. It is not intended to run inside every optimized native MCTS
selection step.



## kv_ledger.py: Atomic Logical-Block Accounting

### Purpose and Boundary

kv_ledger.py is the only module that changes request KV ownership and the mirrored
per-rank counters in ReplicaState. It has no block objects, free lists, dictionaries,
event records, hidden allocator state, prefix cache, or offload. Resumable preemption
uses this existing ledger rather than introducing a second KV manager.

Every request owns a logical number of blocks. Reserving one logical block increments
the reserved counter on every rank of that request's replica. Because rank capacities
can differ across PP stages, admission succeeds only when every rank can accept the
same logical delta.


### Public Operations

| Operation | Responsibility |
| --- | --- |
| blocks_for_tokens() | Computes exact block demand with integer ceiling division. |
| additional_blocks_for_work() | Computes incremental blocks for exactly one of prefill, decode, or recomputation after reusing existing partial blocks. |
| free_logical_blocks() | Returns the least free-block count across all ranks. |
| can_reserve_blocks() | Checks every rank without mutation and can include blocks from validated terminal evictions. |
| reserve_batch_blocks() | Verifies exact block deltas, checks all ranks, and reserves the entire batch atomically. |
| commit_batch_blocks() | Moves a completed batch from reserved to committed without changing total KV usage. |
| release_request_blocks() | Releases all physical KV for non-in-flight terminal cleanup and resets physical computed-token state. |
| preempt_request_blocks() | Validates a live, non-in-flight resident request, releases its blocks on every rank, and resets only its physical KV progress. |


### Incremental Demand

For one pre-admission request allocation:

    resident_after =
        resident_tokens
        + admitted_prefill
        + admitted_decode
        + admitted_recompute
    blocks_after = ceil(resident_after / block_size_tokens)
    blocks_owned = committed_kv_blocks + reserved_kv_blocks
    new_blocks = max(0, blocks_after - blocks_owned)

This provides the required full-cache behavior. A decode inside an existing partial
block has new_blocks equal to zero and remains feasible when global free capacity is
zero. A decode at an exact block boundary requires one new block on every rank. A
prefill or recomputation may require multiple blocks.

The operation accepts exactly one positive work kind. If
remaining_recompute_tokens is nonzero, ordinary prefill and decode are illegal: the
request must rebuild its missing physical context first. Recomputation itself may be
split across multiple batches and cannot exceed remaining_recompute_tokens.

Already owned excess blocks are reused. The ledger itself does not decide when to add
speculative preallocation; it only accounts for blocks already represented in the
authoritative request state.


### Preemption and Recomputation

Preemption deliberately preserves the request rather than converting it into a new
request. For an eligible waiting request, preempt_request_blocks() performs:

    released_blocks = committed_kv_blocks
    committed_kv_blocks = 0
    kv_computed_tokens = 0

The same released-block delta is subtracted from every TP/PP rank ledger in the
request's replica. The request ID, lifecycle phase, committed prefill/decode counts,
decode-credit mint state, deadlines, accumulated lateness, and violation state are
unchanged. Consequently:

    remaining_recompute_tokens == logical_context_tokens

Recovery batches reserve recompute tokens and blocks just like other physical work.
At final-stage completion they increase kv_computed_tokens and release the recompute
reservation, but do not increase committed prefill/decode progress, consume decode
credit, mint credit, or reset an SLO deadline. Partial recovery returns the request to
its original waiting phase; normal work becomes legal only after the missing context
reaches zero.

The controller may preempt one eligible request whether or not KV is currently full.
For a waiting request, the action calls preempt_request_blocks() immediately. For an
in-flight request, no scheduled GPU work is cancelled and no reserved memory is made
available early. The request becomes PREEMPT_PENDING and follows this final-stage
order:

1. Move the batch's block reservation from reserved to committed.
2. Commit its prefill, decode, or recompute allocation normally.
3. Record prefill/decode lateness and update the next decode deadline normally.
4. Mint credit for final prefill or consume already-reserved decode credit normally.
5. Release every physical block now owned by the request.
6. Set kv_computed_tokens to zero and return to WAITING_PREFILL or WAITING_DECODE.

The missing recovery after step 6 is the complete updated logical context. Thus an
in-flight decode preemption includes the just-completed decode token, and an in-flight
recompute preemption discards the partial reconstruction that just drained. This
drain-before-release rule prevents impossible cancellation, early block reuse, lost
logical progress, and decode-credit refunds.


### Atomicity and Transition Ordering

reserve_batch_blocks() follows validate-then-apply ordering:

1. Validate the nonempty, sorted, unique request allocations.
2. Resolve requests by O(1) array index and verify replica ownership.
3. Reject terminal, unassigned, or already in-flight requests.
4. Reject normal work until any missing KV context has been recomputed.
5. Recalculate and verify every allocation's exact new_kv_blocks value.
6. Sum the batch delta and check it independently against every rank.
7. Only after every check passes, update request and rank reserved counters.

Therefore a failed capacity or consistency check leaves every counter unchanged.
The transition engine subsequently sets token reservations and in-flight links as one
completed engine transition.

When a microbatch exits the final PP stage, commit_batch_blocks() moves the same delta
from reserved to committed on requests and all ranks. Free capacity does not change
during that move. Terminal release is legal only after a request has no in-flight
batch, and release is checked on every rank before mutation.


### Complexity

The hot operations are linear only in ranks plus selected requests:

    block calculation: O(1)
    capacity check: O(number of ranks in one replica)
    reserve/commit: O(number of allocations + number of ranks)
    release/preempt: O(number of ranks)

No operation scans unrelated requests or replicas. On the local Python environment,
the measured best-of-five microbenchmarks were approximately:

    incremental block calculation: 0.531 us
    capacity check across 8 ranks: 0.814 us
    reserve + commit across 8 ranks: 4.604 us

These measurements are development checks rather than performance guarantees. The
same array-oriented operations can be translated directly into native loops later.


## pipeline_calendar.py: Compact FIFO PP Timing

### Purpose and Predictor Boundary

pipeline_calendar.py converts one already-resolved microbatch and its predicted
durations into a deterministic PP timetable. It does not choose requests, reserve KV,
change decode funding, load profile artifacts, advance global time, or commit
completed work.

The caller resolves the configured predictor IDs and supplies two immutable tuples:

    stage_service_times[k] =
        GPU compute + TP collectives inside PP stage k

    pp_communication_times[k] =
        transfer from PP stage k to PP stage k + 1

The calendar never scales stage service by TP or PP. In particular, PP communication
must be excluded from stage_service_times because it is added separately at each PP
boundary.


### FIFO Calculation

For a batch admitted at time t, the calendar performs one loop over the PP stages:

    ready[0] = t
    start[k] = max(ready[k], stage_tail_finish[k])
    finish[k] = round(start[k] + stage_service_times[k])
    ready[k + 1] = round(finish[k] + pp_communication_times[k])

The maximum against the stage tail creates downstream stalls naturally. An interval
where a stage has no ready batch remains a pipeline bubble. A later small batch keeps
its own service duration even when it waits behind a larger batch.

The resulting ready, start, and finish tuples are stored in an
InflightMicrobatchState. The final stage finish is the batch completion time used by
transition_engine.py later.


### Public Operations and Atomicity

| Operation | Responsibility |
| --- | --- |
| can_admit_microbatch() | Checks stage-0 availability and the bounded in-flight ring. |
| next_pipeline_admission_time() | Returns the earliest time allowed by stage 0 and in-flight occupancy alone. |
| build_microbatch_calendar() | Purely calculates and validates a batch timetable without mutation. |
| admit_microbatch() | Builds the timetable, then atomically updates stage tails, last-batch IDs, and the in-flight ring. |

Every rejecting check occurs before admit_microbatch() changes the replica. Invalid
durations, PP widths, action IDs, allocation ordering, stale microbatch IDs, busy stage
0, or a full in-flight ring therefore leave the calendar unchanged.

The next pipeline admission time is only a lower bound for the controller. KV, funded
decode availability, request dependencies, and action legality may move the actual
controller decision later. Final-stage completions remain internal transition-engine
events.

GV4 v1 requires:

    inter_stage_queue_capacity >= max_inflight_microbatches

This guarantees enough waiting capacity for every admitted batch and permits the
compact tail-time calculation. Smaller queues require explicit backpressure and are
rejected until that separate algorithm is implemented and parity-tested.


### Complexity

Calendar construction is O(PP) in time and stores three PP-width timing tuples per
in-flight batch. Admission appends one bounded ring record and updates two PP-width
replica arrays. It does not scan requests, ranks, or other replicas.

On the local Python environment, a PP4 calendar build with validation and object
construction measured approximately 5.61 us per operation, excluding execution-time
prediction. This is a development measurement rather than a performance guarantee;
the same fixed arrays and single stage loop map directly to native code.


## action_resolver.py: Pure Action Resolution and Canonicalization

### Responsibility Boundary

action_resolver.py is the mutation-free bridge between a policy output index and a
physical GV4 transition. It reads GV4State and GV4EngineConfig, but it never changes
requests, decode funding, KV counters, pipeline tails, clocks, or objective values. This is
important for MCTS: hundreds of actions can be expanded from one parent without
cloning and repairing the state for every rejected action.

The module exposes these compact immutable records:

| Record | Meaning |
| --- | --- |
| ResolvedControllerAction | Exact preemption/eviction IDs, deferred in-flight victims, per-request work, KV effects, and transition kind for one legal raw index. |
| CanonicalControllerAction | One MCTS edge, its smallest representative raw index, and every equivalent raw alias. |
| ResolvedAdversaryAction | Exact launch count/template and deterministic decode-stop target IDs. |
| CanonicalAdversaryAction | One adversary MCTS edge, its smallest representative raw index, and every equivalent raw alias. |
| ControllerTransitionKind | WAIT, EVICT_ONLY, BATCH, PREEMPT_ONLY, or EVICT_AND_PREEMPT. BATCH takes precedence when the same action also admits work. |

LST and the two timing-aware preemption policies receive a PrefillTimeEstimator
callback. The resolver does not load or guess a profile itself. The callback takes a
request plus a prefill-class token count and returns predicted service time. Invalid
or missing timing fails closed.


### Controller Resolution

resolve_controller_action() resolves one raw index. resolve_controller_actions()
reuses one cached context to resolve the entire dynamic raw action space and returns
both the fixed raw-index array and the canonical edge tuple.

The fixed raw product is:

    preemption rule x eviction rule x prefill budget x ordering heuristic
      = 5 x 9 x 9 x 4 = 1620 raw actions

Preemption is the outermost index dimension and `preempt_none` is first, so the former
0..323 controller indices retain their exact meanings. For each raw action the resolver
performs this deterministic sequence:

1. Decode preemption, terminal eviction, shared prefill-class budget, and ordering.
2. Require a controller turn. Memory-only choices remain legal while stage 0 is busy;
   only batch allocation requires a free admission slot.
3. Resolve waiting terminal-eviction targets, then resolve at most one distinct
   preemption target. Preemption is legal even when free KV exists.
4. Split that victim into immediate or PREEMPT_PENDING according to whether it has an
   in-flight allocation. Only immediate releases increase scratch free KV.
5. Exclude eviction/preemption victims and jointly order ordinary waiting prefills and
   missing-context reconstructions by SJF, EDF, LST, or LJF.
6. Apply strict masks for ineffective memory rules, zero-budget aliases, unavailable
   prefill-class work, invalid over-budget choices, and closed-pipeline batch fields.
7. Allocate reconstruction or ordinary prefill from the same selected budget, max
   batch tokens, max sequences, and exact block capacity. Reconstruction always wins
   within one request; new work cannot bypass missing context.
8. Add funded one-token decodes that need zero blocks first, then boundary-crossing
   decodes in request-ID order. Decode credit remains feasibility accounting, not a
   controller action dimension.
9. Sort allocations by request ID. Any admitted work produces BATCH; otherwise classify
   the concrete memory effects or use the unique WAIT fallback.

Exact new_kv_blocks is stored in every BatchAllocation. The net KV effect is also
stored per physical rank for the canonical key. Prefill allocation can consume unused
space in an existing final block and can be shortened to the maximum exact amount
that fits; it does not round request work up to an artificial token allocation.

### Single-Victim Preemption Policies

Every non-`none` rule selects one resumable resident request, including an in-flight
request whose admitted work has not completed. The candidate's recompute size and
released blocks are evaluated at the time preemption will actually take effect. For an
in-flight candidate that is its final PP completion; for a waiting candidate it is
`state.now`. A final in-flight decode token is excluded because it naturally completes
the request and leaves nothing to resume.

| Rule | Deterministic choice |
| --- | --- |
| `preempt_min_recompute` | Smallest updated logical context, then more released blocks, then lowest request ID. This minimizes recovery work. |
| `preempt_largest_kv` | Most released blocks, then fewer recovery tokens, then lowest request ID. This maximizes immediate/eventual memory relief. |
| `preempt_max_recovery_slack` | Largest `recovery_deadline - release_time - estimated_recompute_time`, then more blocks, then lowest ID. This chooses the request safest to delay. |
| `preempt_best_relief_cost` | Largest released-blocks/recovery-cost ratio. Recovery cost combines predicted recompute time, any newly introduced violation base cost, and capped predicted lateness; ties prefer more blocks then lower ID. |

For prefill-phase requests the recovery deadline is the original prefill deadline. For
decode-phase requests it is the next-token deadline. If an in-flight allocation finishes
prefill or decode and creates the next decode deadline, the candidate uses that updated
deadline. Policy selection is cached while all raw actions for one state are expanded.


### Canonical Edges

The key is formed only from physical effects:

    (
        transition_kind,
        sorted_evicted_request_ids,
        sorted_preempted_request_ids,
        sorted_pending_preemption_request_ids,
        sorted(request_id, work_kind, prefill_tokens, decode_tokens, recompute_tokens),
        per_rank_net_kv_delta,
    )

Actions with the same key share one CanonicalControllerAction. Aliases remain listed
so the policy layer can sum their priors. The representative is always the smallest
raw index, and canonical indices follow first raw-index appearance. Python iteration
therefore has no hash-order dependency and can be matched directly in native code.


### Adversary Resolution

resolve_adversary_action() and resolve_adversary_actions() preserve the configured
fixed policy dimension while strictly masking illegal choices. They check:

- the current player and adversary tick;
- the open one-second launch window request count and aggregate prefill cap;
- append-only native request/unassigned-request bounds;
- stop-rule eligibility for waiting or in-flight decode requests;
- deterministic longest/shortest/threshold target selection and request-ID ties.

The resolved launch record carries the configured prefill template. As in GV3, the
transition engine gives every adversarially generated request the configured maximum
decode length. Before a scheduled adversary tick, only raw index zero is exposed as a
forced no-op; it does not alter launch history, decode funding, or the adversary
clock.

Legal adversary actions are canonicalized by physical effect:

    (launch_count, prefill_tokens, sorted_stop_request_ids)

The raw index and symbolic stop rule are intentionally excluded. For example, when
only one decode exists, stop_longest_decode and stop_shortest_decode select the same
request and therefore share one CanonicalAdversaryAction. The raw policy dimension is
preserved for masking and prior lookup; equivalent raw priors must be summed before
MCTS expands the canonical edge. Canonical indices follow first raw-index appearance,
and the smallest raw index is the deterministic representative.


### Complexity

One controller context scans the target replica's requests once. Eviction targets and
prefill orderings are cached across the raw Cartesian product. Each physical plan then
uses bounded linear loops over selected prefills, eligible decodes, and replica ranks.
No scratch GV4State clone is created. Canonical grouping uses immutable tuple keys.

The prior 324-action performance measurement no longer applies because the raw product
is now 1620. Python still scans requests once per controller context and caches eviction
targets, preemption candidates/timing estimates, and prefill-class orderings across that
product. Native parity preserves deterministic resolution order; performance should be
remeasured before declaring a new benchmark.


## transition_engine.py: Atomic Time, Work, and Cost Transitions

### Public Operations

| Operation | Responsibility |
| --- | --- |
| apply_adversary_action() | Apply decode stops, create launch-window-bounded requests, and move to the controller turn. |
| apply_controller_action() | Revalidate a canonical plan, apply terminal evictions and immediate/deferred preemptions, reserve work/KV and funded decode tokens, and admit an optional batch. |
| advance_to() | Process every due final-stage completion, prune launch history, update time, and apply automatic drops. |
| next_internal_completion_time() | Return the earliest final-stage completion across replicas. |
| next_wait_boundary_time() | Return the earliest future adversary, stage-0, or completion boundary that can change legality. |

Every operation returns TransitionOutcome. It contains the resulting state, elapsed
simulator time, objective before/after, edge_reward = before - after, and the exact
time-dependent discount from RewardConfig. Zero-time BATCH and memory-only transitions
consequently receive discount exponent zero; a WAIT edge receives the elapsed boundary
time.

Controller application uses one fixed order: terminal evictions, preemption marking or
release, batch KV/decode-credit reservation, and pipeline admission. Immediate
preemption blocks may fund another request in the same action. PREEMPT_PENDING blocks
may not, because they remain occupied until the victim's final-stage completion. The
selected victim is excluded from the action's new batch in both cases.


### Adversary-Tick Replay Across In-Flight Work

Adversary ticks are hard external decision boundaries. They cannot be skipped or
coalesced merely because a controller-admitted microbatch finishes after one or more
ticks. A controller BATCH transition is zero-time admission at `t_admit`: it reserves
request work, KV, and decode funding and installs the complete immutable PP calendar.
It does not jump `state.now` directly to the batch's final completion time.

At every crossed adversary tick, the adversary receives the controller's post-admission
state advanced to that exact tick. This is not a rollback to the state before the
controller action. The exposed state includes:

- the admitted microbatch and all stage ready/start/finish times;
- request token and KV reservations made by that microbatch;
- its current in-flight request lifecycles and pending terminal markers;
- all launches and stops chosen at earlier adversary ticks;
- every internal final-stage completion whose completion time is at or before this
  tick.

Work that has not crossed the final PP stage remains reserved rather than committed.
The calendar itself does not need per-stage event mutation as time passes; `state.now`
plus its precomputed times identify where the work lies.

Advancement toward a later controller-admission time must repeatedly stop at each
earlier adversary tick. At a timestamp shared by several effects, GV4 uses this order:

1. Commit due final-stage completions in
   `(final_completion_time, microbatch_id)` order.
2. Refresh lateness and apply automatic drops at that timestamp.
3. If it is an adversary tick, expose the resulting snapshot and apply exactly one
   adversary action.
4. Expose a controller decision only when its next admission boundary is legal;
   otherwise the controller takes the canonical WAIT edge to the next enabling
   boundary.

Therefore, an API call or environment loop that wants to reach time `T` must not call
a bulk fast-forward past `next_adversary_tick`. It must yield every adversary
decision in order, update `next_adversary_tick`, and only then continue toward
`T`.

#### Stop and Launch Effects During the Replay

An adversary stop never cancels GPU work already admitted by the controller:

- a WAITING_DECODE request becomes STOPPED immediately and releases its KV;
- an INFLIGHT_DECODE request becomes STOP_PENDING, keeps the existing batch
  reservation, and treats that admitted token as its final token;
- a STOP_PENDING request becomes STOPPED only when that microbatch reaches its final
  PP stage, at which point its KV is released.

Consequently, saying that an in-flight request is "stopped" at an intermediate tick
means stop has been requested. Its concrete lifecycle is STOP_PENDING until final
completion. If completion occurs before or exactly at a later adversary tick, the
completion is committed first and that later adversary sees STOPPED.

A request launched at a tick is appended with that exact arrival time and a
profile-derived prefill deadline. It is visible at every subsequent tick and controller
decision, but it cannot be inserted retroactively into a microbatch admitted before its
arrival.

#### Concrete 0.9 to 1.3 Example

Assume the controller cannot admit another batch until 1.3:

| Time | Required exposed state and action |
| --- | --- |
| 0.9 | Controller admits batch B. B, its allocations, reservations, and complete PP calendar become part of the state immediately. |
| 1.0 | Advance only to 1.0. Commit any work due by 1.0, then expose B as in flight to the adversary. Launches are appended at 1.0. A stop targeting a request in B changes it to STOP_PENDING. |
| 1.2 | Expose a state containing the same still-in-flight calendar for B plus all effects from 1.0 and any completions due by 1.2. If B still finishes at 1.3, its stopped request remains STOP_PENDING. Apply the 1.2 adversary action. |
| 1.3 | Commit B first. STOP_PENDING requests become STOPPED and release KV; ordinary requests commit their work. The next legal controller decision sees the launches and stop effects from both 1.0 and 1.2. |

For PP greater than one, stage 0 may become free before the batch reaches its final
stage. In that case the next controller admission may occur before 1.3 if in-flight,
queue, KV, and turn constraints permit. The table intentionally assumes that 1.3 is
the next legal admission boundary.

#### Current Support and Remaining Integration

The Python transition primitives already preserve the main safety properties:

- `next_wait_boundary_time()` includes the next adversary tick, stage-0 release,
  and internal final completion;
- `_advance_to_inplace()` rejects any target beyond an unprocessed adversary tick;
- controller admission stores the microbatch calendar and request reservations before
  time advances;
- adversary launches and STOP_PENDING mutations persist in the same state.

The future GV4 environment and native loop must preserve these primitives while
replaying an arbitrary number of crossed ticks. They must never replace the sequence
of adversary decisions with one action at the final batch-completion time. Required
parity tests must cover zero, one, and multiple crossed ticks, stops of in-flight
requests, launches at each tick, completion exactly on a tick, and snapshot/restore
from every exposed boundary.


### Atomic Controller Admission

The transition engine re-resolves the representative raw action against the current
state and rejects stale plans. For BATCH it then builds the complete PP calendar
without mutation. Only after action and timing validation succeed does it:

1. Terminally drop resolved non-in-flight eviction targets and release their KV.
2. Release waiting preemption victims or mark in-flight victims PREEMPT_PENDING.
3. Reserve exact KV blocks on every rank through kv_ledger.py.
4. Move only the automatically included decode-token funding from available to the
   global reserved count. Prefill reserves no credit.
5. Copy token reservations and the microbatch backlink into each request.
6. Admit the batch through pipeline_calendar.py.
7. If the reservation exhausts available decode credit, stop every waiting decode
   across all replicas and mark every in-flight decode STOP_PENDING.
8. Increment the monotonic microbatch ID.

All capacity and timing failures occur before the first mutation. Memory-only actions
perform only their resolved release/marking at the current time. WAIT invokes
next_wait_boundary_time(), advances to that boundary, and cannot silently spin at the
same timestamp.


### Completion Commit

advance_to() orders due work by (final_completion_time, microbatch_id), even across
replicas. A batch finishing stage 0 is still in flight; request progress changes only
after its final PP stage finishes. At final completion the engine:

1. Moves reserved KV to committed KV without changing total occupied capacity.
2. Charges each completed decode token by removing its reserved decode credit.
   Prefill completion has no credit to consume.
3. Moves reserved request work to committed prefill/decode progress or restored
   physical recompute progress, then clears the batch backlink.
4. Transitions unfinished prefill back to WAITING_PREFILL.
5. On prefill completion, records final prefill lateness, enters WAITING_DECODE, sets
   the first decode deadline, and mints the configured decode credits exactly once.
6. On ordinary decode completion, charges that token against its prior deadline and
   sets the next token deadline from the actual completion time.
7. Naturally completes a finished decode request and releases all committed KV.
8. Physically finalizes STOP_PENDING or DROP_PENDING after issued work drains,
   without returning it to WAITING_DECODE or assigning another deadline.
9. For PREEMPT_PENDING, performs the normal effects above, then releases all updated
   KV and returns the request to its logical waiting phase with full reconstruction due.
10. Removes the batch from the bounded in-flight ring and reconciles objective totals.

An adversary stop charges no additional decode credit. A non-in-flight request stops
immediately and cannot consume another token. An in-flight request becomes
STOP_PENDING; its already-issued token drains, consumes its reservation, and is
charged before terminal release. Unused positive pooled decode credit does not expire.

Exact credit exhaustion uses the same physical drain rule but records
DECODE_CREDIT_EXHAUSTED. Waiting decodes stop immediately; in-flight decodes consume
their final reservation and then stop. Subsequent credit minting never changes a
STOPPED request back to WAITING_DECODE.


### Adversary, Terminal, and SLO Semantics

At a real adversary tick, apply_adversary_action() prunes the launch history and
strictly enforces both the request-count cap and aggregate prefill-token cap. It does
not mint prefill credit. Prefill deadlines are calculated from the injected profile
estimate multiplied by SLOConfig.prefill_slowdown_factor. Single-replica launches are
assigned directly; multi-replica launches remain unassigned for the future router.

A waiting decode stop immediately releases KV and becomes STOPPED. An in-flight stop
becomes STOP_PENDING, consumes its already issued work, and releases KV only at final
completion. Controller eviction and automatic SLO drop use the terminal drop path:
they replace earlier violation/lateness cost with terminal_drop_cost. An in-flight
automatic drop becomes DROP_PENDING, records terminal cost immediately, and delays
physical release.

At each reached external boundary, unfinished prefill lateness is refreshed and
requests at or above automatic_drop_lateness_sec are dropped. Objective rebuilding is
linear in the bounded request array and derives counts from authoritative lifecycles,
which prevents incremental counters from drifting between MCTS branches.

advance_to() rejects backward time and any target beyond an unprocessed adversary
tick before mutating state. This preserves the required completion/drop/adversary/
controller ordering instead of accidentally skipping a player decision.


### Snapshot and Native Suitability

The transition engine accepts inplace=False by default and modifies an isolated state
clone. inplace=True is available for a caller that already owns a branch; all
externally rejectable action, timing, and skipped-boundary checks still happen before
mutation. The implementation uses stable integer IDs, sorted tuples, bounded arrays,
and simple loops only. No closures, profile objects, Python simulator instances, or
generic event heap are stored in GV4State, so the same operations can be translated to
fixed native structs later.


## fast_forward.py: Deterministic Internal Progression

The fast-forward module does not own a simulator and does not implement another
scheduler. It reuses raw controller action 0, which means no preemption, no eviction,
zero prefill-class budget, and automatic inclusion of every feasible funded decode.
The normal resolver therefore remains the only source of batching, sequence, KV, and
credit legality.

`fast_forward_decode_only_to_next_tick()` clones once unless the caller already owns
the branch. It then repeats this bounded sequence:

1. Auto-apply the unique pre-tick adversary no-op when strict alternation requires it.
2. Resolve raw controller action 0 independently for replicas in stable ID order.
3. Ask the injected batch-timing provider for per-stage service and PP-boundary times.
4. Atomically admit one legal decode-only batch through `apply_controller_action()`.
5. When admission is temporarily blocked by stage or in-flight occupancy, advance to
   the earliest stage-0, final-completion, or adversary boundary.
6. Stop exactly at the next adversary tick and expose `Player.ADVERSARY`.

A microbatch whose final PP completion is after the tick remains in the state with
reserved tokens, KV, and credit. Completion exactly on the tick is committed first by
`advance_to()`, so the adversary sees the committed result. An empty terminal state
jumps directly to the tick without invoking a predictor.

Fast-forward deliberately yields without advancing when ordinary prefill or KV
reconstruction is waiting/in flight, including reconstruction for a decode-phase
request. It also yields a controller state when stage 0 is free but a waiting decode
cannot obtain its next KV block. Those cases may require strategic eviction or
preemption, so they are not safe to hide from MCTS. The zero-time transition guard
prevents an incorrect callback or turn cycle from spinning forever.

An explicit controller memory action is also an observable boundary. After any
concrete eviction or resumable preemption, the virtual environment applies the KV,
request, and optional batch effects but does not enter automatic decode fast-forward.
This remains true when the same action automatically admits decode work. The resulting
state is returned at the action timestamp so the adversary and then the controller can
observe the new memory layout. The sole exception is a genuinely idle result with no
nonterminal request and no in-flight batch; that state still jumps directly to the next
adversary tick. Ordinary decode-only actions with no eviction/preemption retain the
normal fast-forward behavior.

This gameplay change advances the engine manifest contract to
`gv4_engine_manifest_v7`. State, action, feature, and native-layout schema versions do
not change because no serialized field or tensor shape changed.

The timing callback contract is intentionally small:

    timing_provider(state, resolved_controller_action)
        -> (stage_service_times, pp_communication_times)

Stage service times must already include TP-local compute and collectives. PP transfer
times remain separate, exactly as required by `pipeline_calendar.py`.


## virtual_environment.py: Thin MCTS Facade

`GV4VirtualVidurMCTSEnvironment` stores only three shared objects: the immutable
configuration, the batch-timing callback, and the prefill-time callback. It has no
request registry, event heap, hidden clock, or mutable simulator copy. `GV4State` is
the complete snapshot, and `GV4State.clone()` is the only branch snapshot operation.
Consequently, a separate `virtual_simulator.py` is unnecessary for GV4.

The facade exposes the intended MCTS operations:

- `initial_state()` constructs the compact configured snapshot.
- `sample_controller_actions()` and `sample_adversary_actions()` return canonical
  action objects indexed by the fixed raw policy dimension plus a Boolean mask.
  Equivalent raw indices reference the same canonical edge.
- `apply_controller_action_only()` obtains timing only for a real batch, delegates the
  transition, and optionally invokes decode-only fast-forward. Concrete eviction or
  preemption suppresses that fast-forward unless the resulting system is fully idle.
- `apply_adversary_action_only()` delegates launches and stops with the injected
  prefill estimate used to construct deadlines.
- `evaluate_objective()` returns `(SLO violations, total cost)`.
- `describe_state()` returns compact lifecycle, credit, replica, calendar, KV, and
  objective diagnostics without mutating state.

When stage 0 is occupied, raw controller action 0 now canonicalizes to `WAIT`. This
keeps the fixed action space nonempty after a forced adversary no-op and advances only
to the earliest deterministic boundary. No stage timing callback is used for WAIT.


## Python MCTS Connection

The top-level `mcts_value_prior.py` is now the shared GV4 Python tree implementation.
It no longer imports GV3's MCTS, environment, state, action types, virtual simulator,
or clock. Its engine boundary is `GV4VirtualVidurMCTSEnvironment` plus `GV4State`:

- tree snapshots are independent `GV4State.clone()` results;
- root and child turns come from `state.next_player`, including non-alternating
  controller states created by pipeline or KV blocking;
- raw policy indices are grouped using GV4's canonical action wrappers;
- tree actions call only GV4 adversary/controller transition methods;
- node time is `state.now`, objective cost is `state.objective.total_cost`, and edge
  discount uses `config.reward.discount_for_elapsed()`;
- the caller's root state is never mutated by search;
- multi-replica search fails explicitly until router actions are connected.

`mcts_value_prior_rollout.py` subclasses that GV4-native base and contains only the
fixed-horizon rollout behavior. Rollout branches use the same clone, action,
transition, turn, cost, and timing contracts as normal tree expansion.

Two compatibility seams are intentionally deferred, as agreed for this phase:

- value bootstrap still imports the existing GV3 DNN input builder;
- the MCTS call site still invokes the existing GV3 logger until it is explicitly
  switched to the new four-file GV4 logger described below.

Those are the only remaining `Game_Version3` imports in the two MCTS modules. They
must be migrated or copied before deleting the corresponding DNN and logger
directories. They do not control GV4 state progression or scheduling semantics.


## GV4 DNN Feature Contract

### Status and Scope

This section defines the Python GV4 feature schema implemented by
`dnn_inference/dnn_features.py` and the model boundary implemented by
`dnn_inference/inference.py`. It builds on useful GV3 features while reading
game-dependent scales from `GV4EngineConfig` instead of scattering constants
through training and inference code. The AlphaGoZero models, replay layout, MCTS
wiring, and native implementation must consume this contract; they are not changed
by these two files.

The initial scheduler model is not a load balancer and does not receive router state.
It models a continuing, time-homogeneous system rather than an episode:

- no episode number, horizon progress, or time-until-episode-end is an input;
- no router turn, unassigned-request queue, or routing action is an input;
- no `layout.max_requests`, remaining request-ID capacity, or equivalent artificial
  simulator limit is an input;
- no absolute request ID, microbatch ID, raw action index, canonical index, MCTS
  depth, visit count, Q-value, replay target, RNG counter, or model version is an
  input;
- no cumulative completed/stopped/dropped counters or already-paid objective cost is
  an input to a model trained on future discounted return.

The simulator has deterministic precomputed stage calendars, but the policy must not
receive future stage ready, start, finish, or final batch-completion timestamps. A
real scheduler observes current occupancy and submitted batch composition but does
not know exact future completion times. The models must learn timing consequences
from observed state, batch shape, and training targets. Vidur is therefore not called
once per legal action merely to construct DNN features.

### Shared Normalization Scales

The feature builder computes normalization scales once from the immutable config. The
only intentional schema constant is `WORKLOAD_WINDOW_MULTIPLIER = 20`; it supplies
a stable active-workload scale without exposing `layout.max_requests`.

```text
window_request_cap = timing.max_requests_per_launch_window
window_prefill_cap = (
    request.target_prefill_tokens_per_request_window_average
    * window_request_cap
)

active_request_scale = WORKLOAD_WINDOW_MULTIPLIER * window_request_cap
system_prefill_scale = WORKLOAD_WINDOW_MULTIPLIER * window_prefill_cap
system_decode_scale = (
    active_request_scale
    * request.target_decode_tokens_per_request_average
)

request_prefill_scale = request.max_prefill_tokens_per_request
request_decode_scale = request.max_decode_tokens_per_request
decode_credit_scale = request.target_decode_tokens_per_request_average
adversary_time_scale = timing.adversary_tick_sec
launch_age_scale = timing.launch_window_sec
lateness_scale = cost.lateness_cap_sec
block_token_scale = kv_cache.block_size_tokens
controller_prefill_action_scale = max(controller_actions.prefill_budget_options)
```

KV capacity is logical replicated capacity, not the sum of mirrored physical-rank
capacities:

```text
replica_logical_blocks = min(rank capacity for each rank in the replica)
system_logical_blocks = sum(replica_logical_blocks for every replica)
system_logical_tokens = system_logical_blocks * kv_cache.block_size_tokens
```

Ordinary normalized counts and token totals use division without clipping. Values
above one retain overload information. Time ages and signed deltas use
`asinh(raw / scale)`, which is approximately linear near zero and logarithmic for
large magnitudes. Every emitted scalar must be finite. A fraction whose natural
denominator is zero is defined as zero.

### Structured State Layout

The schema is structured rather than a single fixed 226D vector:

```text
GV4StateFeatures
  global_features        [fixed]
  request_rows           [number of live requests, fixed request width]
  launch_rows            [number of retained launch records, fixed launch width]
  replica_rows           [configured replicas, fixed replica width]
  microbatch_rows        [number of in-flight microbatches, fixed batch width]
```

Padding and masks may be introduced only while batching examples for a DNN. Padding
is not semantic state. Live request and launch rows must not be truncated, sampled,
or selected as a top-k subset. Request order is structural only; numeric request IDs
are not model features. In a future multi-replica scheduler, request ownership must
be represented by grouping rows under a replica or by an explicit structural link,
not by treating a replica ID as a continuous scalar. Router-only and unassigned
requests remain outside this scheduler schema.

### Global State Features

The global vector begins with retained GV3 workload features translated to GV4
committed/reserved semantics, followed by new GV4 aggregates.

| Feature | Definition and normalization |
| --- | --- |
| `active_request_count` | Nonterminal requests divided by `active_request_scale`. This is also the single `live_request_count`; it is not duplicated under two names. |
| `active_prefill_count` | Live requests with remaining or reserved prefill work divided by `active_request_scale`. |
| `active_decode_count` | Live requests with completed prefill and remaining or reserved decode work divided by `active_request_scale`. |
| `remaining_prefill_tokens` | Sum of `remaining_prefill_tokens` divided by `system_prefill_scale`. Reserved work is excluded because it is already attached to an in-flight batch. |
| `remaining_decode_tokens` | Sum of `remaining_decode_tokens` divided by `system_decode_scale`. |
| `committed_prefill_tokens` | Fully completed prefill tokens divided by `system_prefill_scale`. |
| `committed_decode_tokens` | Fully completed decode tokens divided by `system_decode_scale`. |
| `committed_context_tokens` | Committed prefill plus committed decode of live requests divided by `system_logical_tokens`. |
| `decode_credit_balance` | `asinh(decode_credits_available / decode_credit_scale)`. The v1 ledger keeps it nonnegative, while the signed transform permits a future signed ledger. |
| `decode_credits_reserved` | `asinh(decode_credits_reserved / decode_credit_scale)`. |
| `next_adversary_tick_delta` | `asinh((next_adversary_tick - now) / adversary_time_scale)`. Absolute simulator time is excluded. |
| `logical_tokens_free_fraction` | Free logical KV tokens across replicas divided by `system_logical_tokens`. Each replica contributes its bottleneck free-block count, not a sum over mirrored ranks. |

The eight nonterminal lifecycle counts are separate features, each divided by
`active_request_scale`:

| Lifecycle feature | Requests counted |
| --- | --- |
| `waiting_prefill_count` | `WAITING_PREFILL` |
| `inflight_prefill_count` | `INFLIGHT_PREFILL` |
| `waiting_decode_count` | `WAITING_DECODE` |
| `inflight_decode_count` | `INFLIGHT_DECODE` |
| `stop_pending_count` | `STOP_PENDING` |
| `drop_pending_count` | `DROP_PENDING` |
| `inflight_recompute_count` | `INFLIGHT_RECOMPUTE` |
| `preempt_pending_count` | `PREEMPT_PENDING` |

Violation aggregates use only currently live requests. Historical terminal violations
have already contributed transition reward and must not be counted again:

| Feature | Definition and normalization |
| --- | --- |
| `active_violation_fraction` | Violated live requests divided by live request count. |
| `active_prefill_violation_fraction` | Violated active-prefill requests divided by live request count. |
| `active_decode_violation_fraction` | Violated active-decode requests divided by live request count. |
| `waiting_prefill_violation_fraction` | Violated `WAITING_PREFILL` requests divided by live request count. |
| `inflight_prefill_violation_fraction` | Violated `INFLIGHT_PREFILL` requests divided by live request count. |
| `waiting_decode_violation_fraction` | Violated `WAITING_DECODE` requests divided by live request count. |
| `inflight_decode_violation_fraction` | Violated `INFLIGHT_DECODE` requests divided by live request count. |
| `stop_pending_violation_fraction` | Violated `STOP_PENDING` requests divided by live request count. |
| `drop_pending_violation_fraction` | Violated `DROP_PENDING` requests divided by live request count. |
| `inflight_recompute_violation_fraction` | Violated `INFLIGHT_RECOMPUTE` requests divided by live request count. |
| `preempt_pending_violation_fraction` | Violated `PREEMPT_PENDING` requests divided by live request count. |

### Launch-History Rows

Every `LaunchRecord` still inside the rolling launch window gets one row. Exact ages
are retained because equal aggregate usage can have different expiration times and
therefore different future legal adversary actions.

| Feature | Definition and normalization |
| --- | --- |
| `launch_age` | `asinh((now - launch_time) / launch_age_scale)`. |
| `request_count` | Launched request count divided by `window_request_cap`. |
| `prefill_tokens` | Aggregate launched prefill tokens divided by `window_prefill_cap`. |

The builder rejects future-dated records and ignores only records the transition
contract considers expired. It does not replace exact rows with a previous-second
bucket.

### Per-Request Rows

One row is emitted for every live request. `processed` means committed work;
reserved work is encoded separately and is not treated as completed.

| Feature | Definition and normalization |
| --- | --- |
| `decode_phase` | One when prefill is complete and the request is in decode phase; zero for prefill. Pending states retain the phase implied by reserved or remaining work. |
| `prefill_total` | `original_prefill_tokens / request_prefill_scale`. |
| `prefill_committed` | `committed_prefill_tokens / request_prefill_scale`. |
| `prefill_remaining` | `remaining_prefill_tokens / request_prefill_scale`. |
| `decode_total` | `original_decode_tokens / request_decode_scale`. |
| `decode_committed` | `committed_decode_tokens / request_decode_scale`. |
| `decode_remaining` | `remaining_decode_tokens / request_decode_scale`. |
| `committed_context` | Committed prefill plus decode divided by `request_prefill_scale + request_decode_scale`. |
| `kv_computed_context` | Physical context currently represented by KV, divided by `request_prefill_scale + request_decode_scale`. |
| `recompute_remaining` | Missing physical context, excluding reconstruction already in flight, divided by the combined request scale. |
| `arrival_age` | `asinh((now - arrival_time) / launch_age_scale)`. |
| `current_lateness` | For prefill, maximum of recorded lateness and current deadline overrun. For decode, accumulated prefill plus decode lateness. It uses `asinh(lateness / launch_age_scale)`; current signed deadline deltas remain separate. |
| `prefill_deadline_delta` | `asinh((prefill_deadline - now) / launch_age_scale)`. Negative means overdue. |
| `decode_deadline_present` | One when `next_decode_deadline` is set. |
| `decode_deadline_delta` | `asinh((next_decode_deadline - now) / decode_token_slo_sec)` when present, otherwise zero. This avoids a hardcoded 0.05 seconds. |
| `violated` | `violation_recorded` as zero or one. |
| `lifecycle_one_hot` | Eight positions for the four normal waiting/in-flight phases, `STOP_PENDING`, `DROP_PENDING`, `INFLIGHT_RECOMPUTE`, and `PREEMPT_PENDING`. |
| `reserved_tokens` | Reserved prefill, decode, and recompute work divided by `request_prefill_scale + request_decode_scale`. |
| `partial_block_used_fraction` | Tokens used in the final owned block divided by `block_token_scale`; zero when there are no resident tokens. |
| `tokens_until_next_block_fraction` | Tokens available in the partial block before another block is needed, divided by `block_token_scale`. |
| `has_inflight_work` | One when `inflight_microbatch_id != NO_ID`. The numeric ID is excluded. |
| `active_pipeline_stage_one_hot` | One position per PP stage. A position is one only while the request's batch is executing on that stage at `now`. Exact timestamps are hidden. |
| `pipeline_wait_or_transfer` | One when the request is in flight but no PP stage is executing it, such as before stage 0 or during an inter-stage wait/transfer. |
| `inflight_prefill_tokens` | `reserved_prefill_tokens / request_prefill_scale`. |
| `inflight_decode_token` | `reserved_decode_tokens`, which must be zero or one. |
| `inflight_recompute_tokens` | Reserved reconstruction divided by `request_prefill_scale + request_decode_scale`. |
| `pending_stop` | One for `STOP_PENDING`. |
| `pending_drop` | One for `DROP_PENDING`. |
| `pending_termination_age` | `asinh((now - terminal_requested_at) / lateness_scale)` for pending states, otherwise zero. |

The proposed `age_since_prefill_completion` feature cannot yet be emitted correctly:
`RequestState` does not store a prefill-completion timestamp. It must either be
omitted from schema v1 or added as authoritative state with snapshot, logger, test,
and native-layout coverage. It must not be reconstructed from the decode deadline,
which changes after every decode token.

The phrase "pipeline FIFO slot" must not be used as a synonym for PP stage. FIFO
position and active stage are different. This schema exposes current stage occupancy
and the wait/transfer flag while hiding FIFO IDs and future completion times.

### Per-Replica Scheduler Rows

One row is emitted per configured replica. These are scheduler and resource features,
not router features.

| Feature | Definition and normalization |
| --- | --- |
| `committed_logical_kv_blocks` | Replica committed blocks divided by `system_logical_blocks`. |
| `reserved_logical_kv_blocks` | Replica reserved blocks divided by `system_logical_blocks`. |
| `free_logical_kv_blocks` | Replica bottleneck free blocks divided by `system_logical_blocks`. |
| `min_rank_free_fraction` | Minimum free fraction among the replica's physical ranks. |
| `mean_rank_free_fraction` | Mean free fraction among the replica's physical ranks. |
| `max_rank_free_fraction` | Maximum free fraction among the replica's physical ranks. |
| `inflight_microbatch_count` | `replica.inflight_count / scheduler.max_inflight_microbatches`. |
| `stage_free_one_hot` | One Boolean per PP stage. A stage is free when no microbatch is executing there at `now`. No tail-finish timestamp or time-to-free value is exposed. |

Current KV operations mirror each logical block across every rank in a replica. The
minimum free-rank count is therefore the admission bottleneck. Physical rank IDs are
not inputs; minimum, mean, and maximum fractions retain imbalance information.

### Per-Microbatch Rows

One row is emitted for every admitted microbatch still in a replica's in-flight ring.
The model sees current placement and composition, not the future calendar.

| Feature | Definition and normalization |
| --- | --- |
| `active_stage_one_hot` | One position per PP stage. Exactly one is active while a stage executes the batch; all are zero while queued or between stages. |
| `waiting_or_transfer` | One when the batch is in flight but currently executing on no stage. |
| `prefill_request_count` | Prefill allocations divided by `active_request_scale`. |
| `decode_request_count` | Decode allocations divided by `active_request_scale`. |
| `recompute_request_count` | Reconstruction allocations divided by `active_request_scale`. |
| `prefill_tokens` | Total batch prefill tokens divided by `system_prefill_scale`. |
| `decode_tokens` | Total batch decode tokens divided by `system_decode_scale`. |
| `recompute_tokens` | Total reconstruction divided by `system_prefill_scale`. |
| `prefill_reserved_kv_blocks` | New blocks for prefill allocations divided by `system_logical_blocks`. |
| `decode_reserved_kv_blocks` | New blocks for decode allocations divided by `system_logical_blocks`. |
| `recompute_reserved_kv_blocks` | New blocks for reconstruction divided by `system_logical_blocks`. |
| `violated_prefill_request_count` | Violated prefill allocations divided by `active_request_scale`. |
| `violated_decode_request_count` | Violated decode allocations divided by `active_request_scale`. |
| `violated_recompute_request_count` | Violated reconstruction allocations divided by `active_request_scale`. |

The row excludes stage ready/start/finish arrays, final completion time, action
indices, and microbatch ID. Hidden calendar times may only be used to derive current
Boolean occupancy.

### Canonical Controller Action Features

Policy rows are built only for legal canonical controller edges. State features are
encoded once per node. Each action has a fixed header and a variable set of affected-request rows.
Raw aliases must map to the same feature object.

| Header feature | Definition and normalization |
| --- | --- |
| `transition_kind_one_hot` | Five positions for `WAIT`, `EVICT_ONLY`, `BATCH`, `PREEMPT_ONLY`, and `EVICT_AND_PREEMPT`. |
| `evicted_prefill_count` | Evicted prefills divided by `active_request_scale`. |
| `evicted_decode_count` | Evicted decodes divided by `active_request_scale`. |
| `preempted_prefill_count` | Concrete prefill-phase victims divided by `active_request_scale`. |
| `preempted_decode_count` | Concrete decode-phase victims divided by `active_request_scale`. |
| `preempted_inflight_count` | PREEMPT_PENDING victims divided by `active_request_scale`. |
| `decode_request_count` | Decode allocations divided by `active_request_scale`. |
| `total_allocated_prefill_tokens` | Total prefill allocation divided by `controller_prefill_action_scale`. |
| `total_allocated_recompute_tokens` | Total reconstruction allocation divided by `controller_prefill_action_scale`. |

One affected-request row is emitted for every prefill/recompute allocation and every
evicted or preempted request:

| Affected-request feature | Definition and normalization |
| --- | --- |
| `allocated_prefill` | One when this request receives prefill work. |
| `allocated_recompute` | One when this request rebuilds previously committed context. |
| `evicted` | One when it is evicted. It cannot also be allocated by the same canonical edge. |
| `preempted` | One when it is selected for resumable KV release. It cannot also be allocated by the same edge. |
| `allocated_prefill_tokens` | Allocation divided by `controller_prefill_action_scale`; zero for eviction-only rows. |
| `allocated_recompute_tokens` | Reconstruction divided by `controller_prefill_action_scale`; zero for other rows. |
| `remaining_prefill_tokens` | Parent remaining prefill divided by `request_prefill_scale`. |
| `remaining_recompute_tokens` | Parent missing physical context divided by `request_prefill_scale + request_decode_scale`. |
| `total_prefill_tokens` | Original prefill divided by `request_prefill_scale`. |
| `arrival_age` | Same transformation as the state request row. |
| `current_lateness` | Same transformation as the state request row. |
| `violated` | Parent request violation flag. |
| `new_reserved_kv_blocks` | Required blocks divided by `ceil(controller_prefill_action_scale / block_token_scale)`; zero for eviction rows. |
| `already_inflight` | Parent in-flight flag. It is zero for allocations and terminal evictions, but may be one for a deferred preemption victim. |

Features describe physical allocations, evictions, and preemptions. They do not encode
the symbolic victim-rule name, budget label, ordering heuristic, raw index, canonical
index, or alias count. Different raw labels producing the same
`ResolvedControllerAction.canonical_key` are one policy edge.

Vidur service times, PP communication times, predicted stage-free times, and batch
completion times are absent. They are computed only when an edge is applied.

### Canonical Adversary Action Features

Policy rows are built only for legal canonical adversary edges and describe physical
launch and stop effects.

| Feature | Definition and normalization |
| --- | --- |
| `launched_request_count` | `launch_count / window_request_cap`. |
| `prefill_tokens_per_launched_request` | Requested prefill size divided by `request_prefill_scale`; zero for no launch. |
| `total_launched_prefill_tokens` | Total launch prefill divided by `window_prefill_cap`. |
| `stopped_decode_count` | Concrete decode stop targets divided by `window_request_cap`. |
| `stopped_inflight_decode_count` | Stop targets that become `STOP_PENDING` divided by `window_request_cap`. |

Launch prefill size is mandatory. Launch count alone aliases, for example, one
128-token request with one 4096-token request even though they create different
states and SLOs.

The current GV4 adversary cannot stop prefill-phase requests. It can select ordinary
waiting/in-flight decodes and decode-phase reconstruction/preempt-pending requests.
Every in-flight target becomes STOP_PENDING, which supersedes deferred preemption and
still lets admitted work drain. A `stopped_prefill_count` would therefore be
permanently zero and is omitted unless game semantics change first.

Stop count alone can alias actions that stop different decode requests. The
implemented adversary representation therefore includes one affected-request row
per concrete `stop_request_id` with these ordered fields:

| Affected-request feature | Definition and normalization |
| --- | --- |
| `decode_total` | Original decode cap divided by `request_decode_scale`. |
| `decode_committed` | Completed decode tokens divided by `request_decode_scale`. |
| `decode_remaining` | Unreserved remaining decode tokens divided by `request_decode_scale`. |
| `current_lateness` | Same accumulated lateness transform as the state request row. |
| `decode_deadline_present` | One when the next-token deadline exists. |
| `decode_deadline_delta` | Same signed deadline transform as the state request row. |
| `violated` | Parent request violation flag. |
| `reserved_decode_token` | Reserved decode count, currently zero or one. |
| `inflight` | One when stopping produces `STOP_PENDING`; zero for immediate stop. |

This is required disambiguation, not an optional enhancement. The stop-rule label
is not encoded because multiple rules can resolve to the same physical target set.

### Implemented Python Boundary

`GV4FeatureBuilder` is created once per immutable `GV4EngineConfig`. Construction
precomputes the feature names, dimensions, config-derived scales, logical KV
capacity, and manifest hash. Its three hot-path methods are:

```python
state_features = builder.build_state(state)
controller_features = builder.build_controller_action(state, canonical_edge)
adversary_features = builder.build_adversary_action(state, canonical_edge)
```

`GV4StateFeatures` contains unpadded `float32` matrices. It also contains
`request_replica_offsets` and `microbatch_replica_offsets`. For replica `r`, rows in
`offsets[r]:offsets[r + 1]` belong to that replica. These integer offsets are
structural metadata, not continuous neural inputs. They preserve ownership for the
future multi-replica model without encoding numeric replica IDs. All returned arrays
are read-only so an MCTS caller can safely reuse one state serialization for value
and policy inference.

The dimensions are topology-dependent only through the PP stage count `P`:

| Matrix/vector | Width |
| --- | ---: |
| Global state | 31 |
| Request row | `35 + P` |
| Launch row | 3 |
| Replica row | `7 + P` |
| Microbatch row | `13 + P` |
| Controller action header | 13 |
| Controller affected-request row | 14 |
| Adversary action header | 5 |
| Adversary affected-request row | 9 |

Thus TP2/PP2 uses widths 31, 37, 3, 9, and 15 for the five state components.
TP affects KV capacities and normalized state values through the resolved config;
it does not add one-hot rank IDs.

Rows use deterministic structural order: launch records remain chronological,
replicas remain in configured order, and request/microbatch rows are grouped by
replica while retaining their stable engine order. Action affected-request rows are
sorted by request ID, but request IDs themselves are not emitted. Terminal requests
and router-only unassigned requests are excluded. No live scheduler row is
truncated, sampled, or selected by top-k.

`GV4FeatureBuilder.schema_metadata()` returns the schema version, config manifest,
PP width, every ordered feature name, and every normalization scale. Model artifacts
and replay shards must persist this metadata. A model trained for another manifest
or feature schema must fail before prediction rather than silently accepting a
same-shaped tensor.

`GV4DNNInference` is the PyTorch-independent runtime facade. It accepts separate
controller/adversary value and policy models. Value states are evaluated in one
batch through:

```python
value_model.predict_structured(sequence_of_state_features)
```

All canonical actions at one root are scored in one call through:

```python
policy_model.predict_root_structured(state_features, sequence_of_action_features)
```

The state is serialized once and can be supplied back to both calls. The policy
result is one raw logit per canonical edge in the same order as the input sequence.
Softmax, temperature, legal masks, alias expansion, PUCT, and root Dirichlet noise
remain MCTS responsibilities. Tensor conversion, device placement, row
padding/masking, value-target normalization, and neural architecture remain
`AlphaGoZero/dnn_models.py` responsibilities.

The adapter checks model `role` and requires matching
`feature_schema_version` and `config_manifest_sha256` metadata. It also checks
every output length and value for finiteness.
It has no dependency on Torch, MCTS nodes, replay code, or Vidur timing predictors.
Consequently, the same feature objects can be used by Python training, CPU/GPU
inference, replay serialization, and Python/native parity fixtures.

### Schema Freeze and Parity Requirements

The Python/native reference uses `layout.feature_schema_version` (currently
`gv4_markov_v5`) and `WORKLOAD_WINDOW_MULTIPLIER = 20`. Version 5 adds
PREEMPT_PENDING state plus preemption/recompute action and microbatch features while
retaining deterministic row ordering. The
unavailable `age_since_prefill_completion` feature is omitted rather than reconstructed
from a changing decode deadline. Adding that feature later requires authoritative
runtime state plus a schema-version bump.

Native extraction is accepted only after the same samples produce identical row
counts, replica offsets, ordering, categorical bits, and float values within a
declared tolerance. Gates must also cover no truncation, canonical-alias identity,
finite values, pipeline status without future-timing leakage, batched output order,
model metadata rejection, and Python/native model-output parity. The focused Python
coverage is in `tests/GV4_tests/dnn_inference_test.py`.


## logger/: Four Joined Node Snapshots

### Purpose and Boundary

`GV4NodeCSVLogger` records the complete diagnostic view of every materialized MCTS
node without changing the node, resolving another action, advancing time, or invoking
Vidur. It reads the authoritative `GV4State` stored in `Node.cached_sim_snapshot` and
the search-only statistics stored on `Node`.

The logger writes four files:

| File | Responsibility |
| --- | --- |
| `mcts_nodes.csv` | Tree identity, incoming action, time, objective, reward/discount, action mask, and search statistics. |
| `requests.csv` | Complete per-request lifecycle, progress, KV, SLO, lateness, and terminal records grouped by outcome. |
| `pipeline_nodes.csv` | One JSON calendar column per replica and PP stage, including active and queued in-flight batches. |
| `kv_cache_nodes.csv` | Logical replica-level KV totals and one physical accounting JSON column per configured rank. |

Every file has exactly one row per logged node and begins with the same join key:

    (run_id, game_id, root_id, node_id)

`run_id` is supplied by the caller. It identifies the MCTS search/iteration snapshot
being recorded; it is not inferred from node visits. `game_id` identifies the game,
`root_id` identifies the external decision root, and `node_id` identifies one node in
that tree. The same logger may record multiple runs, but it rejects a duplicate
four-part key within one output set.

### How to Read One Joined Row

One joined row contains four different kinds of information. They must not be mixed:

| Perspective | Meaning |
| --- | --- |
| Current node state | `state_time`, objective counters, request records, pipeline calendars, KV state, IDs, credits, and `next_player` describe the state **at this node after the incoming action**. |
| Incoming edge | `parent_node_id`, `player_acted`, `incoming_action_json`, the three incoming action-index fields, `parent_time`, `reward`, and `discount` describe the transition **from the parent into this node**. |
| Outgoing action space | `canonical_action_indices_json` and `valid_mask_json` describe actions that may be taken **from this node**, not the action that created it. |
| Search result | `visits`, `value_sum`, and `mean_value` are the node statistics after backpropagation for the logged MCTS iteration. |

This distinction is especially important for a newly created leaf. MCTS creates the
leaf by applying one parent action, performs rollout/backpropagation, and may log the
leaf before calling `ensure_expanded()` on it. Such a row has a populated
`incoming_action_json` but has `canonical_action_indices_json=[]` and
`valid_mask_json=[]`. The empty lists mean **the current leaf's outgoing action space
has not yet been materialized**. They do not mean that no action created the node,
that the state has no legal action, or that a logged decode batch was automatic.

For example, a list such as `[0,4,8,12,16,180,184,188,192,196]` in
`canonical_action_indices_json` is the set of representative MCTS edge indices that
can be selected from that row's state. It is not the history of actions used to
reach the row. To find the controller batch that created a row, inspect
`incoming_action_json.transition_kind` and `incoming_action_json.allocations`.

### Columns Shared by All Four Files

Every file begins with these columns. For a valid joined row, their values must agree
across all four files.

| Column | Exact meaning |
| --- | --- |
| `logger_schema_version` | Version of the four-CSV layout and JSON payload format. It is independent of the simulator state schema. |
| `run_id` | Caller-supplied logging run identifier. In `GV4_MCTS_Test`, this is the MCTS iteration whose selected path was captured. |
| `game_id` | Caller-supplied identifier for the game/search episode. |
| `root_id` | Caller-supplied identifier for the external decision root. It can differ from the MCTS node's internal ID. |
| `node_id` | Internal MCTS node ID. Together with the prior three IDs, it forms the join key. |
| `state_schema_version` | Version of the serialized `GV4State` contract represented by the row. |
| `config_manifest_sha256` | Hash of the fully resolved engine configuration. It prevents rows from different configurations being compared accidentally. |
| `state_hash` | SHA-256 of the complete deterministic `GV4State` snapshot. Equal state hashes mean equal authoritative simulator state. Search-only values such as visits are not part of this hash. |
| `state_time` | Virtual simulator time of the current node after the incoming action and any macro fast-forward performed by that action. |


### Construction and Tree Logging

The logger is created once for one resolved engine configuration:

```python
from GV4_Engine.logger import GV4NodeCSVLogger

with GV4NodeCSVLogger(output_dir, config) as logger:
    logged_count = logger.log_tree(
        run_id=mcts_iteration,
        game_id=game_id,
        root_id=root_id,
        root=root_node,
    )
```

`log_tree()` performs a deterministic depth-first traversal ordered by canonical edge
index. `log_node()` is also public when a caller wants to record selected nodes or one
tree snapshot per search iteration. By default the logger validates every state using
`GV4State.assert_valid(config)` before writing. Missing snapshots, mismatched manifest
hashes, duplicate node IDs/keys, broken parent links, rank-ledger disagreement, and
overlapping active work on one FIFO stage fail closed.

Existing output files are rejected unless `overwrite=True`. JSON uses sorted keys,
compact separators, finite numbers only, and no truncation or Python `repr` strings.
Consequently, future trace tests can parse actions and states without importing the
class that originally produced them.


### mcts_nodes.csv

The columns below are in addition to the shared columns.

#### Tree, Turn, and Incoming Edge

| Column | Exact meaning |
| --- | --- |
| `root_node_id` | Internal `Node.node_id` of this search tree's root. Unlike `root_id`, this value comes from the MCTS tree itself. |
| `parent_node_id` | Internal node ID of the parent. It is blank for the root. |
| `depth` | Number of MCTS edges from the root to this node. |
| `root_player` | Player that acts at the search root, taken from `root.player`. It does not mean the player that created every row. |
| `player_acted` | Player at the parent node, therefore the player whose incoming action created this node. It is blank for the root. |
| `next_player` | Authoritative `GV4State.next_player` at this node. This player acts from the current state. |
| `incoming_action_json` | Fully resolved action applied to the parent state to create this node. It is JSON `null` at the root. Its nested schema is defined below. |
| `raw_action_index` | Representative raw policy-space index embedded in the resolved incoming action. Blank at the root. |
| `canonical_action_index` | MCTS edge key in `parent.children`, which is the representative raw policy index used by the Python tree. This is the index selected at the parent. |
| `resolver_canonical_action_index` | Compact zero-based ordinal assigned by the GV4 action resolver after equivalent physical transitions are grouped. It is not a `Node.children` key. |
| `alias_indices_json` | All raw policy-space indices that resolve to the same physical incoming transition. The representative is included. |

There are deliberately two canonical index namespaces. Use
`canonical_action_index` when traversing the MCTS CSV/tree, and use
`resolver_canonical_action_index` when checking Python/native resolver parity. The
batch calendar stores the resolver ordinal, so use `raw_action_index` when joining an
incoming controller action to its pipeline batch without ambiguity.

#### Time, Objective, and Edge Value

| Column | Exact meaning |
| --- | --- |
| `parent_time` | Parent state's virtual time before the incoming action. Blank at the root. |
| `child_time` | Current state's virtual time after the incoming action and any fast-forward. It must equal `state_time`. |
| `next_adversary_tick` | Next adversary grid time not yet processed in the current state. |
| `requests_generated` | Total number of request records created so far, including requests that are now terminal. |
| `requests_completed` | Number currently in natural `COMPLETED` lifecycle. |
| `requests_stopped` | Number currently in terminal `STOPPED` lifecycle. `STOP_PENDING` is not included. |
| `requests_dropped` | Number currently in terminal `DROPPED` lifecycle. `DROP_PENDING` is not included. This includes controller eviction and automatic SLO drop. |
| `slo_violations` | Number of non-dropped requests whose `violation_recorded` flag is set. A dropped request is charged terminal drop cost instead. |
| `prefill_lateness_sec` | Sum of recorded prefill lateness over non-dropped requests. Prefill lateness is finalized when prefill completes. |
| `decode_lateness_sec` | Sum of accumulated late decode-token time over non-dropped requests. |
| `total_lateness_sec` | `prefill_lateness_sec + decode_lateness_sec`; logged for convenience. |
| `terminal_cost` | Sum of configured terminal-drop cost for requests in `DROP_PENDING` or `DROPPED`. |
| `objective_total_cost` | Authoritative objective at this state: terminal-drop costs plus configured violation base/capped-lateness costs for non-dropped violating requests. |
| `reward` | Immediate incoming-edge reward, `parent objective cost - child objective cost`. A newly incurred cost therefore gives a negative reward. Root reward is `0`. |
| `discount` | Incoming-edge time discount computed from `child_time - parent_time`; it includes elapsed time hidden inside fast-forward. Root discount is `1`. |

#### Current-State Counters

| Column | Exact meaning |
| --- | --- |
| `next_request_id` | ID that will be assigned to the next adversary-launched request. It is not the number of active requests. |
| `next_microbatch_id` | ID that will be assigned to the next admitted microbatch. Automatic decode fast-forward may increase it several times within one MCTS edge. |
| `adversary_launch_history_json` | Launch records still inside the configured sliding window. Each object has `launch_time`, `request_count`, and aggregate `prefill_tokens`. |
| `total_completed_requests` | Number of request objects in the `COMPLETED` group in `requests.csv`. |
| `total_evicted_requests` | Number with terminal reason `CONTROLLER_EVICTION` and lifecycle `DROP_PENDING` or `DROPPED`. Automatic drops are excluded. |
| `total_requests_in_system` | Number of nonterminal requests, including waiting, in-flight, `STOP_PENDING`, and `DROP_PENDING` requests. |
| `total_stopped_requests` | Number in terminal `STOPPED` lifecycle. |
| `total_stop_pending_requests` | Number marked to stop after its already admitted final microbatch drains. |
| `total_drop_pending_requests` | Number marked to drop after its already admitted final microbatch drains. |
| `decode_credits_available` | Pooled adversary-funded decode tokens that have not yet been admitted into a batch. |
| `decode_credits_reserved` | Decode tokens already placed in in-flight batches but not yet committed at final-stage completion. |

#### Search and Outgoing Actions

| Column | Exact meaning |
| --- | --- |
| `visits` | Number of MCTS visits to this node after backpropagation of the logged iteration. |
| `value_sum` | Sum of controller-valued rollout/bootstrap returns accumulated at this node. The immediate reward from its parent edge is stored separately in `reward`. |
| `mean_value` | `value_sum / visits`, or `0` when `visits == 0`. |
| `canonical_action_indices_json` | Sorted representative raw indices for canonical actions available **from this node** after it has been expanded. These are keys for future `Node.children`. It is `[]` for an unexpanded leaf. |
| `valid_mask_json` | Boolean mask over the complete raw policy action space **from this node**. Aliases may have multiple `true` entries for one physical transition. It is `[]` until this node is expanded. |

The logger deliberately excludes policy priors and a separate final MCTS value. The
retained visits, value sum, and mean value are the authoritative search statistics at
the moment this path was logged.

#### `incoming_action_json` for a Controller Edge

| JSON member | Exact meaning |
| --- | --- |
| `actor` | Always `"controller"`. |
| `raw_action_index` | Representative raw controller-policy index that was resolved and applied. |
| `canonical_action_index` | Resolver canonical ordinal retained for backward readability; prefer the explicitly named member below. |
| `mcts_canonical_action_index` | Parent MCTS edge key, equal to the row's scalar `canonical_action_index`. |
| `resolver_canonical_action_index` | Resolver's compact canonical ordinal, equal to the row's scalar field of the same name. |
| `alias_indices` | Raw indices with identical physical effects. |
| `replica_id` | Replica on which memory effects and admission were resolved. |
| `preemption_rule` | Raw resumable-preemption policy before its single concrete victim was resolved. |
| `eviction_rule` | Raw controller eviction rule selected before concrete targets were resolved. |
| `prefill_budget` | Maximum ordinary prefill plus recompute tokens requested by this action. Decode tokens are scheduled independently. |
| `ordering_heuristic` | Shared prefill/recompute ordering rule, such as SJF, EDF, LJF, or LST. |
| `transition_kind` | `WAIT`, `EVICT_ONLY`, `BATCH`, `PREEMPT_ONLY`, or `EVICT_AND_PREEMPT`. A BATCH may also carry memory effects. |
| `evicted_request_ids` | Concrete request IDs selected by the eviction rule. |
| `preempted_request_ids` | Zero or one concrete request selected for resumable preemption. |
| `pending_preemption_request_ids` | The selected victim when it was in flight and must drain before releasing KV; otherwise empty. |
| `allocations` | Ordered list of concrete per-request work included in the admitted batch. It is empty for memory-only and WAIT transitions. |
| `released_kv_blocks` | Logical blocks released immediately by terminal evictions plus waiting preemptions. Deferred victim blocks are excluded. |
| `preempted_kv_blocks` | Immediate-release portion attributable only to waiting preemption. |
| `reserved_kv_blocks` | New logical blocks reserved for the action's batch allocations. Existing partially occupied blocks do not appear as new blocks. |
| `rank_kv_delta` | List of `[rank_id, net_block_delta]`, where the net delta is newly reserved blocks minus blocks released on that mirrored rank. |

Each object inside `allocations` has:

| Allocation member | Exact meaning |
| --- | --- |
| `request_id` | Request receiving work in this batch. |
| `prefill_tokens` | Prefill tokens reserved for the request; zero for decode work. |
| `decode_tokens` | Decode tokens reserved for the request; zero for prefill work and at most one in GV4 v1. |
| `recompute_tokens` | Previously committed context rebuilt by this allocation; zero for new prefill/decode work. |
| `new_kv_blocks` | Additional logical KV blocks needed for this allocation after reusing blocks already owned by the request. |

#### `incoming_action_json` for an Adversary Edge

| JSON member | Exact meaning |
| --- | --- |
| `actor` | Always `"adversary"`. |
| `raw_action_index` | Representative raw adversary-policy index that was applied. |
| `canonical_action_index` | Resolver canonical ordinal retained for backward readability. |
| `mcts_canonical_action_index` | Parent MCTS edge key. |
| `resolver_canonical_action_index` | Resolver's compact canonical ordinal. |
| `alias_indices` | Raw adversary indices with identical launches and concrete stops. |
| `launch_count` | Number of new requests launched at this adversary tick. |
| `prefill_tokens` | Prefill size assigned to each launched request, or JSON `null` when no request is launched. |
| `stop_rule` | Requested decode-stop policy before concrete target resolution. |
| `stop_request_ids` | Concrete decode request IDs stopped or marked `STOP_PENDING`. |

Root rows have `incoming_action_json=null`, blank scalar incoming-action fields, and
`alias_indices_json=[]` because no parent edge created the root.


### requests.csv

Every request appears in exactly one JSON list. The scalar counts are the lengths of
those lists.

| Column | Exact meaning |
| --- | --- |
| `completed_count` | Number of request objects in `completed_requests_json`. |
| `in_progress_count` | Number in `in_progress_requests_json`. This includes waiting, in-flight, `PREEMPT_PENDING`, `STOP_PENDING`, and `DROP_PENDING`. |
| `evicted_count` | Number in `evicted_requests_json`. Despite the historical column name, this includes every terminal `DROPPED` request; inspect `terminal_reason` to distinguish controller eviction from automatic SLO drop. |
| `stopped_count` | Number in `stopped_requests_json`. `STOP_PENDING` remains in progress until its final batch drains. |
| `completed_requests_json` | JSON list whose members have lifecycle `COMPLETED`. |
| `in_progress_requests_json` | JSON list of every nonterminal request. Pending terminal requests remain here because they can still own in-flight/KV resources. |
| `evicted_requests_json` | JSON list whose members have lifecycle `DROPPED`. |
| `stopped_requests_json` | JSON list whose members have lifecycle `STOPPED`. |

All four JSON columns use the same request-object schema:

| Request member | Exact meaning |
| --- | --- |
| `request_id` | Stable request ID assigned at launch. |
| `owner_replica_id` | Replica that owns the request and its logical KV. A negative value means not assigned. |
| `lifecycle` | Exact `RequestLifecycle` name: waiting, in-flight, pending terminal, or terminal. |
| `current_phase` | Derived phase. It is `prefill` while any prefill work remains or is reserved; otherwise it is `decode`. This is not a replacement for `lifecycle`. |
| `arrival_time` | Adversary tick at which the request was created. |
| `completion_time` | `terminal_time` for `COMPLETED`, `STOPPED`, or `DROPPED`; JSON `null` while nonterminal. |
| `terminal_requested_at` | Time a stop/drop was requested, including a request that must drain in-flight work; JSON `null` when no terminal request exists. |
| `terminal_reason` | Exact reason enum, such as natural completion, adversary stop, controller eviction, automatic SLO drop, or decode-credit exhaustion. |
| `original_prefill_tokens` | Total prompt tokens assigned when the request was launched. |
| `committed_prefill_tokens` | Prefill tokens whose microbatch has completed the final PP stage. |
| `reserved_prefill_tokens` | Prefill tokens in the request's currently in-flight microbatch. |
| `remaining_prefill_tokens` | `original_prefill_tokens - committed_prefill_tokens - reserved_prefill_tokens`. |
| `original_decode_tokens` | Per-request maximum decode work assigned at launch, capped by request configuration. |
| `committed_decode_tokens` | Decode tokens whose microbatch has completed the final PP stage. |
| `reserved_decode_tokens` | Decode tokens already in flight. GV4 v1 permits at most one per request. |
| `remaining_decode_tokens` | `original_decode_tokens - committed_decode_tokens - reserved_decode_tokens`. |
| `total_committed_tokens` | Committed prefill plus committed decode tokens. |
| `logical_context_tokens` | Committed prefill plus committed decode tokens. This progress survives preemption. |
| `kv_computed_tokens` | Logical context tokens whose KV is currently computed and resident on the replica. Preemption resets this to zero. |
| `reserved_recompute_tokens` | Missing context currently being rebuilt by an in-flight recovery batch. |
| `remaining_recompute_tokens` | Logical context still absent from physical KV after accounting for in-flight recovery. |
| `resident_tokens` | Physical KV tokens: computed KV plus reserved recompute, prefill, and decode tokens. It becomes zero immediately after preemption. |
| `committed_kv_blocks` | Logical KV blocks permanently owned by currently committed request context. Zero after terminal release. |
| `reserved_kv_blocks` | Additional logical blocks reserved for in-flight work but not yet committed. |
| `inflight_microbatch_id` | Microbatch currently carrying this request, or JSON `null`. |
| `prefill_deadline` | Absolute virtual time by which final prefill completion should occur. |
| `decode_token_slo_sec` | Allowed interval between successive committed decode tokens for this request. |
| `next_decode_deadline` | Absolute deadline for the currently awaited/in-flight decode token, or JSON `null` before decode starts/after terminal completion. |
| `prefill_lateness_sec` | Recorded final-prefill lateness for this request. Partial prefill chunks do not finalize it. |
| `decode_lateness_sec` | Sum of this request's per-token decode lateness. |
| `violation_recorded` | Whether the request has incurred a counted SLO violation. |
| `decode_credit_minted` | One-time guard showing that final prefill completion already minted this request's decode-credit grant. |

Terminal records retain their historical token/deadline/lateness fields so objective
reconstruction remains possible, but their committed and reserved KV ownership is
zero. A pending stop/drop has no `completion_time` yet because its already admitted
microbatch still has to reach the final PP stage.


### pipeline_nodes.csv

The fixed pipeline columns are:

| Column | Exact meaning |
| --- | --- |
| `replica_count` | Number of replicas represented in this state/configuration. |
| `pipeline_parallel_size` | Number of PP stages per replica. Stage IDs are zero based. |
| `total_inflight_microbatches` | Sum of `ReplicaState.inflight_count` over all replicas. A microbatch remains in flight until final-stage completion is committed. |
| `inflight_microbatch_ids_json` | JSON object from string replica ID to FIFO list of microbatch IDs currently in that replica's in-flight ring. |

The file then creates one dynamic column for every configured stage:

    replica_<replica_id>_stage_<zero_based_stage_id>_json

Each dynamic cell is one complete stage object:

| Stage member | Exact meaning |
| --- | --- |
| `stage_index` | Zero-based PP stage represented by this object. |
| `tail_finish_time` | Finish time of the latest batch scheduled on this stage. A new batch cannot start on this stage before this tail. This value can refer to work that has already completed by `state_time`. |
| `last_microbatch_id` | ID of the most recently scheduled microbatch on the stage, or JSON `null` if none has ever been scheduled. It is historical and need not still be in flight. |
| `active_batch` | Full calendar object for the one batch executing on this stage at `state_time`, or JSON `null`. FIFO validation permits at most one. |
| `inflight_batch_calendars` | Calendar objects for every replica microbatch that has not yet committed final-stage completion, including batches queued here or already finished on this early stage. |

Every object in `active_batch`/`inflight_batch_calendars` has:

| Batch member | Exact meaning |
| --- | --- |
| `microbatch_id` | Stable pipeline microbatch ID. The same ID appears in every stage calendar for that batch. |
| `raw_action_index` | Representative raw controller action that admitted the batch. This is the safest action-to-pipeline join key. |
| `canonical_action_index` | Resolver canonical ordinal stored at admission. It is not necessarily the MCTS representative index in `mcts_nodes.csv.canonical_action_index`. |
| `request_tokens` | JSON object keyed by request ID containing this batch's exact allocation for each request. |
| `ready_time` | Earliest time the batch's input is available to this stage. For stage 0 this is admission time; for later stages it includes prior-stage completion and PP communication. |
| `start_time` | Actual FIFO start, `max(ready_time, prior stage-tail value)` at admission. |
| `stage_completion_time` | `start_time + Vidur stage service time` after configured rounding. |
| `batch_completion_time` | Finish time on the final PP stage. It is identical in every stage's copy of this batch calendar. |
| `status` | Stage-local status derived from `state_time`: `queued` before start, `active` from start until finish, or `stage_complete` after this stage finishes while the batch remains in flight elsewhere. |

Each value inside `request_tokens` has:

| Token member | Exact meaning |
| --- | --- |
| `prefill_tokens` | Prefill tokens for this request in the microbatch. |
| `decode_tokens` | Decode tokens for this request in the microbatch. |
| `recompute_tokens` | Previously committed context tokens whose evicted KV is being rebuilt. |
| `total_tokens` | Sum of the three values; exactly one work kind is positive per request allocation. |

All PP stages carry the same microbatch allocations, but each stage has its own ready,
start, finish, and status. Keeping all in-flight calendars is necessary even though
FIFO guarantees at most one active batch per stage. A batch may be `stage_complete`
on stage 0, active downstream, while a newer batch is active on stage 0. This is how
the CSV exposes pipeline overlap, bubbles, and downstream stalls without an event
heap.


### kv_cache_nodes.csv

Aggregate block fields use **logical blocks**, not the sum of mirrored physical rank
counters. For one replica:

    logical capacity  = minimum rank capacity
    logical available = minimum(rank capacity - committed - reserved)

The multi-replica fields sum these per-replica logical values. Therefore a TP2/PP2
request block is counted once in aggregate columns rather than four times.

| Column | Exact meaning |
| --- | --- |
| `block_size_tokens` | Number of request tokens represented by one logical KV block. |
| `total_capacity_blocks` | Sum over replicas of the minimum per-rank block capacity. This is logical cluster capacity. |
| `total_available_blocks` | Sum over replicas of the minimum fully free block count after committed and reserved ownership. |
| `total_consumed_blocks` | Sum of logical committed block ownership over replicas. `consumed` and `committed` mean the same thing in this CSV. |
| `total_reserved_blocks` | Sum of logical blocks reserved for in-flight work but not yet committed. |
| `total_occupied_blocks` | `total_consumed_blocks + total_reserved_blocks`. |
| `total_tokens_in_memory` | Sum of physical `resident_tokens` for nonterminal assigned requests, counted once per request rather than once per rank. It excludes preempted logical context until recomputation is reserved/completed. |
| `total_available_tokens` | `total_capacity_blocks * block_size_tokens - total_tokens_in_memory`. This includes unused slots inside request-owned partial blocks, so it is a geometry diagnostic rather than a promise that all slots can be reassigned to arbitrary requests. |
| `total_free_block_token_slots` | `total_available_blocks * block_size_tokens`. This counts only wholly free logical blocks and excludes slack inside occupied blocks. |

The distinction between the last two columns is intentional. If a request owns a
16-token block containing 13 resident tokens, `total_available_tokens` includes its
three request-local unused slots, while `total_free_block_token_slots` does not count
that block at all.

The file creates `rank_<rank_id>_json` for every configured GPU rank. Each object has:

| Rank member | Exact meaning |
| --- | --- |
| `rank_id` | Global configured GPU rank ID. |
| `replica_id` | Replica containing this rank. |
| `pipeline_stage` | Zero-based PP stage to which the rank belongs. Multiple TP ranks can report the same stage. |
| `capacity_blocks` | Physical block capacity configured for this rank after safety margin. |
| `available_blocks` | `capacity_blocks - consumed_blocks - reserved_blocks`. |
| `consumed_blocks` | Committed block ownership mirrored onto this rank. |
| `reserved_blocks` | In-flight block reservations mirrored onto this rank. |
| `occupied_blocks` | `consumed_blocks + reserved_blocks`. |
| `resident_tokens` | Physical resident-token total of this rank's replica. It is intentionally repeated on every rank because every PP/TP rank must hold its portion of the same resident contexts. |
| `capacity_token_slots` | `capacity_blocks * block_size_tokens`. |
| `available_token_slots` | `capacity_token_slots - resident_tokens`, including slack in partial blocks. |
| `free_block_token_slots` | `available_blocks * block_size_tokens`, counting only fully free blocks. |

Committed and reserved block counts are required to agree across ranks of one replica;
the logger rejects a snapshot if those mirrors disagree. Capacities may differ, which
is why the logical aggregate uses the bottleneck rank rather than the first or the
sum of ranks.


### MCTS Integration Boundary

The new logger is intentionally usable before replacing the existing MCTS logger. It
accepts the current GV4 `Node` contract through `cached_sim_snapshot`, so tests can
call it immediately after a uniform-prior search. The automatic call at the end of
`mcts_value_prior.py` still targets the legacy two-file logger when `log_flag=True`.
The four-file test logger is connected independently through the optional
`MCTSConfig.iteration_observer`. This leaves production logging unchanged while tests
capture the exact selected path immediately after each simulation's backpropagation.


## Current Verification

The standard-library suites are located at:

- vidur-GV4/tests/GV4_tests/state_test.py
- vidur-GV4/tests/GV4_tests/kv_ledger_test.py
- vidur-GV4/tests/GV4_tests/pipeline_calendar_test.py
- vidur-GV4/tests/GV4_tests/action_transition_test.py
- vidur-GV4/tests/GV4_tests/environment_test.py
- vidur-GV4/tests/GV4_tests/mcts_integration_test.py

The state suite covers:

- TP2/PP2 multi-replica initial layout and capacity construction;
- derived request token/KV accounting;
- a valid in-flight PP prefill with batch links and rank ledgers and no prefill-credit state;
- decode admission on a KV-block boundary;
- complete clone isolation while immutable tuples/allocations are shared;
- terminal completion with KV release and objective reconciliation;
- rejection of zero-token launches and missing decode deadlines;
- rejection of lifecycle/progress, request/batch, rank-ledger, and batch-order errors;
- fail-closed request and replica ID lookup.

The KV-ledger suite covers:

- exact ceiling division and partial-block reuse;
- boundary-crossing decode and multi-block prefill demand;
- reuse of already preallocated request blocks;
- bottleneck capacity across heterogeneous rank limits;
- atomic TP2/PP2 batch reservation and atomic capacity failure;
- rejection of incorrect allocation block deltas;
- zero-increment decode while every rank is full;
- reserved-to-committed transfer without changing free capacity;
- terminal release on every rank and rejection of in-flight release;
- parent/child MCTS branch isolation;
- a full reserve and commit sequence accepted by GV4State.assert_valid().

The pipeline-calendar suite covers:

- pure PP1 timetable construction and atomic calendar admission;
- PP stalls, bubbles, FIFO overlap, and separate boundary communication;
- stage-0 and in-flight-capacity admission clocks;
- rejection without mutation when stage 0 is busy or the ring is full;
- malformed predictor outputs, PP widths, IDs, and allocation order;
- TP-inclusive stage durations without scaling or double counting;
- deterministic time rounding and parent/child calendar isolation;
- reconciliation with request, KV, decode-budget, and full-state invariants.

The action/transition suite covers:

- deterministic 512-token SJF allocation across multiple requests;
- canonical merging of physically equivalent heuristic aliases;
- zero-block decode at full KV capacity and WAIT at the next block boundary;
- adversary launch creation, profile-derived deadlines, and launch-window enforcement;
- atomic prefill admission with no prefill-credit dependency;
- one-time decode minting at final-stage prefill completion;
- decode reservation/commit accounting and system-wide last-credit exhaustion;
- early adversary stop preserving unused pooled credit while charging issued work;
- PP2 progress remaining uncommitted until the final stage;
- exact per-token decode lateness, one violation, and objective cost;
- decode-only admission and completion from pooled adversary-funded credit;
- zero-time controller eviction and terminal-cost replacement;
- in-flight adversary stop, drain, and delayed KV release;
- WAIT boundary advancement and automatic terminal drop behavior;
- strict one-second launch-window masking;
- in-place invalid timing and skipped-tick rejection without state mutation.

The environment/fast-forward suite covers:

- direct idle advancement without a predictor call;
- repeated decode-only batches through exact pooled-credit exhaustion;
- a batch crossing the adversary tick remaining reserved and in flight;
- active prefill disabling decode fast-forward;
- KV-blocked decode yielding a controller decision instead of hiding eviction;
- PP stage-0 release admitting a second batch while earlier work is downstream;
- preemption with no decode retaining the exact controller-action timestamp;
- eviction plus an admitted decode batch remaining visible at that timestamp;
- fully idle eviction still jumping to the next adversary tick;
- ordinary decode-only actions retaining automatic fast-forward;
- raw/canonical action sampling, parent/child isolation, and occupied-stage WAIT.

The MCTS integration suite covers:

- adversary and controller search over compact GV4 state;
- canonical raw-index aliases and fixed policy dimensions;
- clone-only snapshots and root-state immutability;
- child turn selection from authoritative GV4 state;
- fixed-horizon policy rollouts through real GV4 transitions;
- root-player validation and the explicit pre-router one-replica guard.

Run all current GV4 engine tests from the repository root with:

    python3 -m unittest discover -s vidur-GV4/tests/GV4_tests -p '*_test.py' -v

Current verification status after the decode-only credit migration:

    75 tests discovered
    75 tests passed

The new logger has additionally passed two isolated construction checks:

- a TP2/PP2 in-flight prefill produced all four files, one joined row, an active
  stage-0 batch, a queued stage-1 calendar, eight logical reserved blocks, and eight
  mirrored reserved blocks on each of four physical ranks;
- a six-iteration uniform-prior GV4 MCTS search logged its complete seven-node tree,
  including root null-edge handling, structured child actions, and both MCTS and
  resolver canonical indices.

The dedicated four-file trace suite below now supersedes these early smoke checks.

The suite contains no prefill-credit fixtures. It covers one-time minting, pooled
unused credit, system-wide exhaustion, immediate waiting-request stop, final-token
STOP_PENDING drain, the 7-request/7168-token launch window, and the 864-token
per-request decode cap.

There is not yet a dedicated config_test.py. The state tests construct real
GV4EngineConfig objects and therefore exercise successful cross-configuration setup,
but invalid configuration combinations and artifact checksum behavior still need a
focused configuration test suite before the configuration layer is called complete.

## GV4 MCTS Four-File Trace Test Harness

### Purpose

The package GV4_Engine/GV4_MCTS_Test validates complete MCTS-selected paths
without a DNN. It runs PUCT with uniform priors, records the exact root-to-leaf
path selected by every simulation, splits those observations into one directory
per MCTS iteration, reconstructs engine state from the four CSV rows, and runs
independent state, request, KV, and pipeline checks.

This differs from logging the materialized tree after search. A final tree cannot
tell which path simulation 17 selected or what a reused node's visit/value totals
were immediately after simulation 17. MCTSConfig.iteration_observer is called
after each backpropagation and receives that exact path. The logger uses the
iteration number as run_id; therefore the same node_id may correctly appear
under several run IDs with progressively updated search statistics.

The observer is diagnostic only. It does not select actions, restore snapshots,
advance time, query Vidur, or mutate MCTS nodes.

### Test Configuration

GV4_MCTS_Test/config.py creates one immutable engine manifest shared by MCTS,
the logger, reconstruction, and every validator.

| Setting | Test value | Reason |
| --- | --- | --- |
| Model | meta-llama/Meta-Llama-3-8B | Matches the selected Vidur H100 profiles. |
| Replicas | 1 | Router/load-balancer MCTS is not connected yet. |
| TP | 2 | Exercises per-stage TP-inclusive execution predictions and mirrored KV ledgers. |
| PP | 2 | Exercises stage overlap, PP transfer, stalls, bubbles, and final-stage commits. |
| Device/network | h100 / h100_dgx | Selects compute, all-reduce, and send/receive profiles. |
| Scheduler | 4608 tokens, 256 sequences | Supports one 4096-token prefill plus decode work. |
| In-flight limit | 2 | Allows stage 0 to admit work while an older batch is downstream. |
| Inter-stage capacity | 2 | Avoids unsupported explicit backpressure while matching the in-flight bound. |
| KV block | 16 tokens | Matches Vidur/vLLM block accounting. |
| KV budget | 512 MiB per rank before 10% safety margin | Keeps the integration run finite enough to exercise meaningful occupancy. |
| MCTS | 100 iterations, seed 7 | Reproducible uniform-prior baseline. |
| DNN/bootstrap/noise | Disabled | Tests engine semantics without model-dependent behavior. |

timing_mode=vidur is the real integration mode. It constructs one
VidurTimingProvider, loads or builds its predictor cache once, and reuses it for
MCTS and independent timing checks. timing_mode=deterministic is only a fast
harness smoke mode; it proves capture, splitting, reconstruction, and assertions
without claiming Vidur timing coverage.

### Output Layout

For an output root <run> the raw logger writes:

    <run>/raw/mcts_nodes.csv
    <run>/raw/requests.csv
    <run>/raw/pipeline_nodes.csv
    <run>/raw/kv_cache_nodes.csv

split_iteration_logs() verifies that all four raw files have exactly the same
(run_id, game_id, root_id, node_id) key set. It rejects duplicate keys, missing
rows, unexpected run IDs, stale iteration directories, and unrelated files.
For 100 simulations it then writes exactly:

    <run>/mcts_iterations/mcts_iter_000001/<four CSV files>
    ...
    <run>/mcts_iterations/mcts_iter_000100/<four CSV files>

Rows inside each mini-file are ordered by MCTS depth. No manifest or fifth file is
placed in an iteration directory. The global result is written separately to
<run>/validation_summary.json.

### Trace Context and State Reconstruction

trace_context.py performs a strict one-to-one join before semantic tests run.
Common schema version, configuration hash, state hash, and state time must agree
across all four rows.

For each node it reconstructs:

- every RequestState, including progress, reservations, SLOs, terminal state,
  KV ownership, in-flight link, and decode-credit mint flag;
- every ReplicaState, including physical rank placement, capacity, committed
  and reserved mirrors, stage tails, last batch IDs, and the in-flight ring;
- every InflightMicrobatchState, with request allocations and ready/start/finish
  arrays rebuilt from all PP-stage columns;
- launch history, next IDs, next player/tick, pooled decode-credit totals, and the
  complete ObjectiveState.

The reconstructed object is passed to GV4State.assert_valid(config) at every
logged node. This catches cross-file inconsistencies that isolated CSV checks
would miss, including request/batch backlink errors, credit non-conservation,
rank-ledger disagreement, invalid lifecycle reservations, and malformed PP arrays.

### State and Tree Tests

state_tests.py runs the following checks on every iteration path:

1. The folder run_id, CSV run_id, and root identity agree.
2. Every path starts at depth zero with no parent and JSON null incoming action.
3. Root visits equal the completed MCTS iteration number because logging occurs
   immediately after that iteration's backup.
4. Every child names the preceding path node as parent and increments depth once.
5. parent_time equals the parent snapshot, and child time never moves backward.
6. player_acted equals the parent state's turn; action actor and acting player
   agree; the resulting next player follows strict adversary/controller alternation.
7. A request terminal in a parent can never become active in a child.
8. Edge reward equals parent_cost - child_cost.
9. Edge discount equals RewardConfig.discount_for_elapsed(child_time-parent_time).
10. Representative raw action, MCTS edge index, and alias list agree; the MCTS edge
    is the minimum raw alias used by the current Python tree.
11. Expanded nodes have the configured fixed raw-mask width. A just-created leaf
    may have an empty mask only when it also has no canonical action list.
12. Canonical action indices are sorted, unique, in range, and marked valid.
13. Reconstructed request count equals append-only next_request_id.
14. Logged in-system count equals all nonterminal request lifecycles.
15. Objective generated/completed/stopped/dropped/violation counters are recomputed
    directly from requests.
16. Prefill lateness, decode lateness, terminal replacement cost, and total objective
    are independently recomputed with the configured lateness cap.
17. next_adversary_tick never trails state time.
18. A pre-tick adversary edge must be raw action zero, launch nothing, stop nothing,
    and preserve the pending tick.
19. A real adversary decision occurs exactly at the exposed 0.2-second grid point
    and advances the tick by exactly one interval. Missed ticks therefore appear as
    separate ordered adversary turns rather than one hidden multi-tick jump.
20. The retained one-second launch history never exceeds seven requests or the
    7168-token aggregate prefill cap.

### Request and Action Tests

request_tests.py checks:

1. Completed, in-progress, dropped/evicted, and stopped JSON lists form a disjoint
   partition and match lifecycle membership.
2. Request IDs remain contiguous and append-only.
3. Committed plus reserved prefill/decode never exceeds original work, and physical
   computed plus reserved reconstruction never exceeds logical context.
4. New prefill/decode never begins while reconstruction remains.
5. At most one decode token is reserved per request per batch, and exactly one work
   kind is reserved for each in-flight request.
6. Logged logical prefill/decode phase remains stable while KV is reconstructed.
7. Every request uses the configured maximum 864-token decode bound.
8. Available plus reserved plus committed decode tokens equals 216 times the number
   of requests that minted credit.
9. Zero available credit leaves no active decode, including decode-phase reconstruction
   or preempt-pending work; the final issued token may remain STOP_PENDING while it drains.
10. Every adversary raw index decodes to the logged launch count, template, and stop
    rule.
11. Longest, shortest, 216-threshold, and 512-threshold stop targets are recomputed
    from the parent snapshot with deterministic tie breaking.
12. New IDs are exactly the sequential range beginning at parent next_request_id.
13. New requests use an allowed template, arrive at the action time, start in
    WAITING_PREFILL, and receive the profile-derived prefill deadline.
14. Waiting stop targets become STOPPED; in-flight decode targets become
    STOP_PENDING.
15. Every controller raw index decodes to the logged preemption rule, eviction rule,
    shared prefill-class budget, and ordering heuristic.
16. Every eviction rule independently recomputes its target from parent waiting
    requests, deadlines, progress, and lateness.
17. A preemption selects at most one resident request, never overlaps an eviction,
    and identifies an in-flight target exactly as pending.
18. Batch request IDs are sorted/unique and obey token and sequence limits.
19. Ordinary prefill plus recomputation is no larger than the selected shared budget.
20. A zero prefill budget forbids both prefill-class kinds but may still schedule
    decode. The budget is a cap because remaining work or KV can be lower.
21. Every allocation targets an existing waiting request in the matching logical and
    reconstruction phase.
22. New KV blocks equal the request's exact post-allocation ceiling demand.
23. All five transition kinds agree with concrete batch/eviction/preemption effects.
24. Reserved/released block totals include immediate preemption but exclude deferred
    in-flight release.
25. Evicted requests become terminal DROPPED records; immediate preemption preserves
    logical progress with zero physical KV, while visible in-flight victims become
    PREEMPT_PENDING.
26. If the new batch remains visible, child prefill/decode/recompute reservations and
    the in-flight ID exactly match the controller allocation.

### KV-Cache Tests

kv_cache_tests.py independently recomputes:

1. Per-replica committed and reserved ownership from nonterminal requests.
2. Every physical rank's configured capacity and mirrored committed/reserved values.
3. Per-rank free blocks, occupied blocks, resident tokens, available token slots,
   and free-block token slots.
4. Logical replica capacity as the minimum rank capacity, avoiding TP/PP
   multiplication of one logical block.
5. Multi-replica logical capacity/available/committed/reserved/occupied totals.
6. Total resident tokens and both definitions of remaining token capacity.
7. Exact per-request block ownership as ceil(resident_tokens/block_size).
8. Controller rank_kv_delta as the same net logical delta on every replica rank.
9. Immediate parent/child occupancy delta when the action snapshot remains at the
   same time or the newly admitted batch remains visible.
10. WAIT or fast-forward edges are not falsely treated as action-only deltas; their
    node-level ledgers are still checked completely after all intervening commits.

The focused `kv_ledger_test.py` and `action_transition_test.py` coverage additionally
verifies the complete preemption contract:

1. Preemption releases the same logical block count on every rank while preserving
   request progress, deadlines, lateness, violation state, and decode-credit state.
2. A preempted request reports zero physical resident tokens and the full logical
   context as missing recomputation.
3. Ordinary prefill/decode allocation is rejected while context is missing.
4. Recomputation can complete across multiple pipeline batches, restores physical KV,
   and never changes useful token progress or consumes/mints decode credit.
5. All four deterministic victim rules choose the expected single request.
6. Recomputation and ordinary prefill share one SJF queue and one action budget while
   retaining distinct allocation fields.
7. In-flight prefill and decode work commits lateness/credit effects before release.
8. In-flight reconstruction commits physically, is then discarded, and restarts the
   full updated logical-context recovery.
9. Disabling request_preemption_enabled masks preemption effects.
10. Python and native engines produce identical state/action results for primitive,
    immediate, deferred, and mixed recovery cases.

### Pipeline and Vidur Timing Tests

pipeline_tests.py checks:

1. Every PP stage lists the same in-flight microbatch IDs.
2. Replica in-flight count respects the configured bound.
3. At most one microbatch is active on each FIFO stage.
4. Every calendar satisfies ready <= start < finish.
5. Consecutive batches never overlap on one stage.
6. Logged queued/active/stage-complete status matches node time.
7. Stage tails never precede scheduled work.
8. A downstream stage's ready time never precedes upstream completion.
9. Request/token composition is identical across all stages.
10. Batch completion equals the final PP stage's finish.
11. Pipeline aggregate in-flight count equals stage calendars.
12. For every new controller batch, the parent state and resolved action are passed
    directly to the configured timing provider.
13. The provider must return one TP-inclusive service duration per PP stage and one
    PP communication duration per boundary.
14. Expected ready/start/finish arrays are recomputed independently with parent
    stage tails and configured rounding.
15. If the batch remains in flight, every logged timestamp and first-stage request
    composition must equal this independent calendar.
16. Stage-0 tail gives the next stage-0 admission time after a visible new batch.
17. If decode fast-forward completes the batch before the child snapshot, child time
    must reach final completion and allocated work must be committed.
18. Any pre-existing batch disappearing between adjacent path nodes must have reached
    its recorded final-stage completion.

The focused `vidur_timing_provider_test.py` suite additionally checks the startup
artifact contract without training real random forests:

1. A TP2/PP2 profile has exactly 32 rows for the 128-to-4096 token grid.
2. Its header contains two stage-service columns, one PP-boundary column, and the
   derived end-to-end column in pipeline order.
3. Every end-to-end value equals the TP-inclusive stage services plus PP transfer.
4. A second startup reuses the valid CSV without another predictor query.
5. A malformed CSV is detected and regenerated completely.
6. A non-grid residual chunk rounds to the next profile point without a live
   predictor query.
7. Runtime batch prediction and CSV generation share the same PP subtraction
   function, preventing the two timing paths from drifting.

The real Vidur provider's stage service already includes kernels and TP collectives.
The separately returned boundary duration contains PP send/receive only. These tests
therefore also guard against dividing whole-model time by PP or counting PP transfer
twice.

### Coverage Behavior

Checks for an event run whenever that event appears. The summary records coverage
counts for launches, stops, controller actions, evictions, timed batches, visible
batches, fast-forward completions, pipeline completions, real ticks, and forced
pre-tick no-ops. The 100-iteration baseline additionally requires at least one
adversary action, launch, controller action, controller batch, and timing query.
Stop and eviction counts are reported but are not mandatory for a random uniform
sample; a zero count means a targeted scenario must be added before claiming coverage
of that policy.

### Running the Harness

Fast deterministic smoke test:

    python3 -m GV4_Engine.GV4_MCTS_Test.runner \
      --iterations 100 \
      --timing-mode deterministic \
      --output-dir /tmp/gv4_mcts_trace_smoke_100 \
      --overwrite

First real Vidur run, allowing predictor cache construction:

    python3 -m GV4_Engine.GV4_MCTS_Test.runner \
      --iterations 100 \
      --timing-mode vidur \
      --cache-mode use_cache \
      --output-dir simulator_output/GV4_MCTS_Test/h100_tp2_pp2_uniform

Reproducible rerun after the cache exists:

    python3 -m GV4_Engine.GV4_MCTS_Test.runner \
      --iterations 100 \
      --timing-mode vidur \
      --cache-mode require_cache \
      --output-dir simulator_output/GV4_MCTS_Test/h100_tp2_pp2_uniform_rerun

Revalidate existing split logs without rerunning MCTS:

    python3 -m GV4_Engine.GV4_MCTS_Test.runner \
      --iterations 100 \
      --timing-mode vidur \
      --cache-mode require_cache \
      --output-dir <existing-output-root> \
      --validate-only

With the Cache dir as dir :

    cd /home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur-GV4

    /home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
    -m GV4_Engine.GV4_MCTS_Test.runner \
    --iterations 100 \
    --timing-mode vidur \
    --cache-mode require_cache \
    --output-dir /home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/GV4_MCTS_Test/h100_tp2_pp2_uniform_rerun \
    --overwrite

The verified deterministic baseline produced 100 iteration directories and passed
37,467 assertions over 388 joined node observations. It exercised 196 launches,
112 controller edges, 55 timed batches, 16 pipeline completions, three adversary
stops, 106 processed ticks, and 70 forced pre-tick no-ops. This validates the harness
and engine semantics; it is not a substitute for the real Vidur-profile timing run.

## Native GV4 Engine and MCTS

`GV4_Cpp/` is the native, single-replica implementation of the Python GV4
contract. The current topology may use TP and PP, including the tested TP2/PP2
configuration, but routing between replicas is deliberately deferred. Native
state and transition code does not import or call the old GV3 game engine.

The source is split by responsibility:

- `include/gv4/config.hpp` and `src/config.cpp` hold the validated native subset
  of the immutable GV4 configuration.
- `include/gv4/state.hpp` and `src/state.cpp` hold requests, replicas,
  microbatches, launch history, decode credit, objective state, and validation.
- `src/action_resolver.cpp` reproduces GV4 raw-action expansion,
  canonicalization, aliasing, shared prefill/recompute scheduling, KV feasibility,
  and stop, eviction, or four-policy preemption target selection.
- `src/engine.cpp` owns controller/adversary transitions, immediate/deferred
  preemption, completion commits, reconstruction, decode-credit exhaustion,
  adversary tick replay, fast-forward, KV updates, and compact PP advancement.
- `src/features.cpp` builds the same structured state and action tensors as the
  Python GV4 feature builder. The permanent parity test compares every float32
  byte and every ragged offset.
- `src/inference.cpp` validates schema/manifest identity and presents one native
  value/policy interface. At this phase its callbacks may still invoke the
  Python model object; replacing those callbacks with the eventual native model
  runtime does not change MCTS or game semantics.
- `src/mcts.cpp` implements canonical-action PUCT selection, expansion,
  Bellman backup, root action statistics, and the unchanged zero-bootstrap
  uniform path.
- `src/rollout.cpp` implements fixed-simulated-time policy rollouts.
- `src/mcts_rollout_bindings.cpp` exposes rollout search without changing the
  existing `run_uniform_mcts` API.

### Native Rollout Contract

`run_policy_rollout_mcts` evaluates every newly expanded tree leaf with
`rollout_count` independent trajectories. Its deadline is:

    expansion_parent_time + rollout_horizon_sec

The parent time is captured before applying the selected tree edge. Therefore
sibling actions receive the same absolute horizon even when one selected action
advances simulated time farther than another. If selection reaches a tree node
with no action to expand, the deadline is that leaf's current time and it is
bootstrapped in place.

Each trajectory follows these steps:

1. Copy the leaf `State`; never mutate a tree snapshot.
2. Resolve fresh legal canonical actions for `state.next_player` through the
   normal native action resolver.
3. Use uniform priors when `use_policy_prior=false`. Otherwise obtain logits
   from `InferenceRuntime`, apply temperature softmax and the configured minimum
   prior, and use the same priors for tree selection and rollout sampling.
4. Quantize sampling probabilities by `rollout_probability_quantum`, normalize,
   and sample a canonical representative. The sampler reproduces CPython's
   seeded MT19937 `random.Random.random()` behavior, including Python integer
   seeding.
5. Apply the action through the normal native environment. Record
   `parent_cost-child_cost` and the standard elapsed-time discount.
6. Continue until state time reaches or passes the fixed deadline, no legal action
   remains, or `rollout_max_actions` is reached. Because preemption is legal without
   memory pressure, the cap is a valid truncation boundary rather than an exception.
7. Bootstrap the reached state, including an action-capped state. Use zero by default;
   with `use_model_bootstrap=true`, query the value model and reject non-finite or
   positive controller-valued output.
8. Compose the trajectory backward as
   `reward + discount * continuation`, then average all trajectory returns and
   back up that mean through the MCTS path.

Trajectory seed `j` for native leaf evaluation `i` is:

    rollout_seed + i * 1_000_003 + j * 9_176   (modulo 2^64)

The result includes `used_rollout`, `used_bootstrap`, and `rollout_stats`.
Statistics cover leaf evaluations, trajectory/action counts, terminal exits,
bootstrap calls, start/final/deadline ranges, remaining horizon, and the first
sampled action-history hash. These fields make Python/native rollout drift
observable without logging every internal trajectory.

A zero `rollout_count`, or a zero horizon, disables rollout and preserves the
previous zero-bootstrap search exactly. `run_uniform_mcts` remains unchanged.
The rollout binding defaults to 10 trajectories, a 0.4-second horizon,
temperature 1.0, probability quantum `1e-6`, and a 4096-action safety limit.

### Native Verification

`GV4_Cpp/tests/test_engine_parity.py` compares scripted Python/native action
spaces and transitions. `GV4_Cpp/tests/test_mcts_feature_parity.py` checks:

1. Structured state/action features match Python bit for bit.
2. Uniform root visits match Python exactly at 100 and 1000 iterations.
3. Structured value and policy callbacks enforce schema and manifest identity.
4. `rollout_count=0` produces exactly the original uniform root action values,
   visits, and best action.
5. A seeded Python and C++ rollout search produces the same rollout statistics,
   first action-history hash, root visits, value sums, and best action.
6. Policy-guided rollouts can use both controller/adversary policy models and
   final-state value bootstrap.
7. Resolved preemption fields, immediate release, PREEMPT_PENDING drain, and mixed
   reconstruction/prefill batches match Python exactly.

The fixed seeded parity case produced four trajectories, eight internal actions,
and first-history hash `84696351` identically in Python and C++. Repeated native
stress runs at 100 and 1000 MCTS iterations were deterministic. The 1000-iteration
run evaluated 2000 trajectories and 40,716 rollout actions with identical root
statistics on repetition.

Build from the classical-search repository root with:

    cmake -S vidur-GV4/GV4_Cpp -B vidur-GV4/GV4_Cpp/build \
      -DPython_EXECUTABLE=/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3
    cmake --build vidur-GV4/GV4_Cpp/build -j 8

### Completion Boundary

For the current one-replica GV4 phase, the native game core is complete:
configuration, state, canonical actions, KV accounting, TP/PP timing calendar,
transitions, fast-forward, features, inference interface, PUCT, rollout,
bootstrap, logging adapter, and parity tests are present. The remaining work is
not another game-engine component. It is the AlphaGo Zero integration layer:
root payload construction, arena/self-play selection, worker invocation,
checkpoint/model loading, replay generation, and eventual replacement of the
Python timing/model callbacks with production native runtimes.

## Alpha Go Zero

The first GV4 AlphaGoZero layer is implemented in `AlphaGoZeroGV4/`. Its main
design rule is that AlphaGoZero orchestrates the game but does not implement a
second copy of GV4 semantics. Topology, request rules, action canonicalization,
KV accounting, pipeline timing, adversary ticks, objective updates, reward
discounting, and feature construction continue to come from `GV4_Engine/` or
its parity-tested `GV4_Cpp/` equivalent.

The initial scope is one replica per game. TP and PP remain configurable inside
that replica, so this layer works with the validated TP2/PP2 engine. Multiple
independent games and workers will be added above this boundary rather than by
putting routing or multiprocessing into the game runner.

### Implemented Files

- `GV4_Engine/history_root.py` creates a reproducible random legal state before
  a played cycle begins.
- `AlphaGoZeroGV4/engine_runtime.py` is the only Python-versus-native dispatch
  point. It converts both backends into the same small set of immutable result,
  metadata, and state-summary records.
- `AlphaGoZeroGV4/runner.py` owns one game cycle: history generation, MCTS at
  branching states, visit-based action selection, forced advancement, horizon
  handling, and replay finalization.
- `AlphaGoZeroGV4/replay_runtime.py` records meaningful decisions and computes
  exact discounted training targets.
- `AlphaGoZeroGV4/dnn_models.py` defines the four structured GV4 value/policy
  networks, their trainers, and their checkpoint/native-export formats.
- `AlphaGoZeroGV4/model_bundle.py` validates and atomically publishes one
  immutable set of all four model artifacts.
- `AlphaGoZeroGV4/bootstrap_untrained_dnn_v100.py` creates the reproducible
  neutral version-100 bundle used to start self-play before trained models
  exist.
- `AlphaGoZeroGV4/__init__.py` defines the package without import-time work.

`AlphaGoZeroGV4` does not import `Game_Version3`, its feature builders, its
arena runner, or its replay format. The existing GV3 AlphaGoZero directory has
not been modified or used as a fallback.

### Backend-Neutral Runtime

`create_engine_runtime()` constructs either `PythonEngineRuntime` or
`NativeEngineRuntime`. After construction, the runner uses only the
`EngineRuntime` contract:

    initial_state()
    clone_state(state)
    state_time(state)
    objective_cost(state)
    player_to_move(state)
    metadata
    summarize_state(state)
    canonical_actions(state)
    apply_action(state, action)
    search(state, context)
    bootstrap_value(state)
    close()

The runtime returns immutable backend-neutral records rather than exposing
Python dataclasses or pybind objects to replay and runner code:

- `CanonicalActionRef` carries player, canonical index, representative raw
  index, all equivalent raw aliases, and one private backend payload.
- `AppliedEdge` carries the resulting state, acting/next player, start/end
  times, objective before/after, controller-valued reward, elapsed-time
  discount, and transition kind.
- `SearchResult` carries root identity, player, best action, root value, one
  `RootActionStats` per legal canonical action, the raw valid mask, structured
  state features, and rollout/bootstrap diagnostics.
- `StateFeatureSnapshot` and `ActionFeatureSnapshot` convert NumPy and native
  feature matrices to the same finite, serializable representation.
- `EngineMetadata` freezes the backend, engine/config hash, all semantic schema
  versions, and time epsilon used by one runtime.
- `StateSummary` recursively copies request, launch-window, KV-rank, PP-calendar,
  decode-credit, and objective data into immutable backend-neutral records.
  Loggers therefore never inspect mutable Python state or pybind internals.

The Python adapter creates `GV4VirtualVidurMCTSEnvironment`, uses
`vidur-GV4/mcts_value_prior.py` or its rollout subclass, and uses the existing
GV4 feature builder. The native adapter creates the C++ environment through
`GV4_Cpp/runtime.py`, calls `run_policy_rollout_mcts`, and uses the native
feature builder. Missing unvisited native root actions are restored as
zero-visit rows so replay has the complete canonical legal action set.

Both adapters calculate the public Bellman edge consistently:

    reward = objective_before - objective_after
    discount = config.reward.discount_for_elapsed(finished_at - started_at)

These calculations are adapter normalization, not new game policy. Times,
objectives, and resulting states come from the selected engine. Non-finite
values, backward time, cross-backend actions, wrong-player actions, schema
mismatches, and invalid discounts fail immediately.

The engine configuration remains the semantic entry point. If a timing
provider is not explicitly injected for a test, the runtime obtains the Vidur
provider from `GV4EngineConfig`. Therefore the runner does not know cache paths,
TP/PP stage predictors, KV limits, or action-space dimensions.

Model objects are dependency-injected into the runtime. The CLI supports both
the uniform/no-model integration path and a validated model bundle through
`--model-bundle`. Policy priors, internal leaf bootstrap, and final-state model
bootstrap remain separate explicit switches. Native root Dirichlet noise is
still rejected until that option is exposed by the native binding rather than
being silently ignored.

### Random History Root

`HistoryRootGenerator.generate()` accepts a runtime, hop count, seed, optional
initial state, and optional absolute history horizon. For this GV4 contract:

    one history hop = one canonical action applied to the game

A forced turn with one canonical action still counts as one hop. Internal
completion, pipeline, and time advancement performed while applying that
action does not create additional hops. This matches the requested action-based
history definition and avoids depending on how much internal simulator work an
action triggers.

The generator:

1. Clones the supplied state, so generating a history never mutates a reusable
   initial state.
2. Resolves canonical actions through the selected backend.
3. Sorts by canonical and representative identity before sampling, making the
   seed independent of container iteration order.
4. Samples uniformly over canonical actions rather than raw aliases.
5. Uses a local `random.Random(seed)` and never global RNG state or Python's
   randomized `hash()`.
6. Applies exactly one selected action and records its action identity, aliases,
   player, time interval, reward, discount, and transition kind.
7. Stops after the requested hops, at the optional horizon, or if no legal
   action exists.

The returned `HistoryRoot` contains the root state plus the complete audit path,
requested/achieved hops, seed, next player, final simulation time, and stop
reason. Python and native runtimes therefore receive the same history algorithm
rather than maintaining separate random-root implementations.

### One Game Cycle

`run_game_cycle()` starts its time horizon at the generated history root and
then repeatedly examines the number of legal canonical actions:

- Zero actions ends the cycle with `no_legal_actions`.
- One action is forced and is applied directly without spending MCTS work or
  creating a policy target.
- Two or more actions form a meaningful decision and invoke MCTS.

At a meaningful decision, the runner creates a `SearchContext`, performs one
root search, selects a canonical action from root visits, and applies that exact
action through the runtime. Each root receives a disjoint node-ID range of
`mcts_iterations + 2`, so independently built trees can later share one logger
without node-ID collisions.

For self-play temperature greater than zero, selection samples with weight
`visits ** (1 / temperature)`. If all visits are zero, it falls back to a
uniform canonical choice. Temperature zero is deterministic: it first chooses
the largest visit count, then the value preferred by the acting player, then
the smallest representative raw index. No raw alias gets a second chance to be
sampled.

After a searched action, the runner immediately applies every following
single-action turn until another branch, the horizon, or the action safety cap.
Those forced actions remain visible to an optional observer and in replay's
`forced_chain`, but they do not become separate policy examples. This is
important for forced adversary no-ops, missed-tick replay, and decode-only fast
forward.

The observer receives every actually applied action as a `PlayedStep`, including
immutable state summaries immediately before and after the edge. Searched
events also include their `SearchResult`; forced events deliberately do not
invent an MCTS result. This is the integration hook for arena/debug loggers, so
logging logic remains outside the game loop and outside engine state classes.

`GameCycleResult` reports history provenance, engine metadata, role model
versions, backend, end reason, searched and forced action counts, final
time/objective, an immutable final-state summary, explicit bootstrap choice,
and replay paths. `max_actions` is a safety bound against zero-time loops rather
than a normal game-ending rule.

The local command-line entry point accepts an engine-config factory, backend,
game/cycle identity, seed, history depth, horizon, MCTS settings, rollout
settings, output directory, and overwrite policy. A config factory is written
as `module:function` and must return one validated `GV4EngineConfig`. The CLI
writes `game_result.json` atomically and refuses to overwrite any completed
result unless `--overwrite` is explicit.

### Replay Runtime

`GV4ReplayRecorder` stores only meaningful searched decisions. A decision owns
its selected edge and the subsequent forced one-action chain. The chain is
collapsed mathematically but retained for auditing. For edges `(r_i, d_i)`:

    composed_reward = r0 + d0*r1 + d0*d1*r2 + ...
    composed_discount = d0*d1*d2*...

After the cycle, targets are backed up from the explicitly named bootstrap:

    target[i] = reward[i] + discount[i] * target[i + 1]

Supported bootstrap labels are:

- `terminal_zero`: the backend exposed no further action and the value is zero.
- `neutral_zero`: a configured nonterminal zero bootstrap, recorded explicitly
  rather than inferred from model version.
- `model`: a value predicted for the final state by the injected runtime.

The explicit `use_model_bootstrap` argument in Python MCTS now controls
bootstrap directly. Consequently a deliberately requested version-0 model is
valid; model version is artifact metadata and is no longer an implicit hard
off-switch.

One cycle publishes three files:

- `replay_states.jsonl` has one row per meaningful root. It stores game/cycle
  identity, decision/root-node IDs, player, time/objective edge, composed
  reward/discount, target and root values, selected canonical identity, raw
  valid mask, structured state features, search diagnostics, forced-chain
  audit, and bootstrap metadata.
- `replay_actions.jsonl` has one row per canonical legal action at each stored
  root. It stores action identity/aliases, visits, normalized visit probability,
  value sum/mean, optional prior, selected flag, and structured action features.
- `replay_manifest.json` is the completion marker. It records complete engine
  metadata, replay/feature schemas, engine config hash, all model versions,
  game seed, history-hop count, row counts, SHA-256 hashes of both JSONL files,
  bootstrap, end reason, and final time.

The recorder validates contiguous decision IDs, selected-action membership,
schema/config identity, unique canonical representatives, nonnegative visits,
finite values, player/time/objective continuity through forced chains, reward
agreement with objective deltas, and valid discounts. It buffers one game,
computes all targets, writes temporary files with `fsync`, atomically renames
the two JSONL files, and publishes the manifest last. A downstream worker must
treat only a directory with a valid complete manifest as ingestible replay.

### GV4 DNN Models

`AlphaGoZeroGV4/dnn_models.py` is the trainable implementation behind the
feature and inference contracts above. It follows the useful parts of the GV3
Markov design while binding every artifact to GV4 rather than reusing the GV3
226-element vector. Four independent artifacts are expected:

- controller value model;
- adversary value model;
- controller policy model;
- adversary policy model.

The two player roles do not share weights. They observe the same state schema,
but the controller and adversary policies have different action headers and
different affected-request row widths. `GV4ModelSpec.from_config()` records the
feature schema, full engine-manifest SHA-256, PP stage count, replica count, and
every state/action dimension. Model loading fails if the requested role,
schema, manifest, architecture, or model type differs. A model trained under
one topology, action space, KV setup, timing contract, or cost contract is
therefore not silently reused under another.

GV4 AlphaGoZero v1 deliberately accepts exactly one replica. Requests,
microbatches, and the replica row are still represented as variable sets, but a
future multi-replica policy needs an explicit ownership-aware hierarchy before
pooling. Rejecting multiple replicas is safer than losing request-to-replica
relationships in one global pool.

#### State Encoder

The state encoder consumes the existing structured feature object directly:

1. The fixed global vector is projected once.
2. Every request row is encoded independently by the same small MLP.
3. Launch-history, replica, and in-flight-microbatch rows each use their own
   independent row MLP.
4. Each variable set is reduced with both sum pooling and max pooling.
5. The global embedding and eight pooled vectors are fused into one 192-wide
   state embedding and passed through two bottleneck residual blocks.

Sum pooling preserves aggregate load, while max pooling preserves the most
urgent or extreme member. Since a row MLP is shared within a set and pooling is
commutative, changing request or microbatch storage order cannot change the
prediction. Empty sets receive a masked zero maximum. Minibatches are padded
only to the largest row count in that minibatch; masks ensure padding never
contributes, and no live row is truncated.

The model accepts all three representations used by GV4 without rebuilding
features: NumPy feature dataclasses from the Python engine, tuple/list payloads
read from replay JSON, and native pybind `FeatureMatrix` objects. Feature
meaning and normalization remain solely in `GV4FeatureBuilder`; the DNN module
does not inspect requests, clocks, actions, or simulator internals.

#### Value Model and Huber Loss

`GV4ValueDeepSet` maps the shared state embedding through a small scalar head.
All MCTS values use the controller-valued convention:

    edge_reward = objective_before - objective_after

Because objective cost is nondecreasing, valid replay returns are non-positive.
The final transform is therefore:

    predicted_value = value_scale * -softplus(raw_output)

Unlike GV3's fixed `tanh` interval, this does not impose an incorrect hard
lower bound on a GV4 trajectory. The default scale is 25 and controls numerical
conditioning, not game semantics. The output-head initialization starts close
to the explicit neutral-zero bootstrap without making trained inference equal
to a hard-coded bootstrap.

`fit_value_dnn()` divides replay targets by `value_scale` and minimizes PyTorch
Huber loss. Huber behaves quadratically for small errors and linearly for large
errors, so an unusually expensive trajectory does not dominate a minibatch as
strongly as it would under pure MSE. Training rejects positive or non-finite
controller-valued targets, clips the gradient norm to 1, and uses AdamW. The
GV3-compatible name `fit_markov_value_dnn` is an alias to the same structured
GV4 trainer.

#### Policy Model and Visit Targets

`GV4PolicyDeepSet` encodes a root state once, then reuses that 192-wide embedding
for every canonical legal action at the root. An action encoder projects its
fixed header and DeepSets-pools its variable affected-request rows. The state
and action embeddings are fused to produce one unrestricted logit. Softmax,
legal masks, temperature, and root Dirichlet noise remain MCTS responsibilities
and are not duplicated in the model.

`fit_policy_dnn()` receives one state per root, a contiguous action list, root
offset pairs, and MCTS visit probabilities. It normalizes targets independently
inside each root and minimizes soft cross-entropy against the root logits. A
zero-mass root falls back to a uniform target, matching the safe GV3 behavior.
The implementation batches roots, repeats only their already-computed state
embeddings, clips gradients to 1, and uses AdamW. `fit_markov_policy_dnn` is the
equivalent GV3-compatible entry-point name.

#### Incremental Training and Artifacts

Both trainers accept an optional previous checkpoint. Warm start restores the
model and AdamW optimizer state, then applies the newly requested learning rate
and weight decay. It refuses to warm-start from a different role or engine
manifest. Training metadata records row/root counts, epochs, loss, whether the
run was warm-started, and the policy's encode-once property.

`save_dnn_model()` writes a data-only Torch checkpoint atomically with `fsync`
and rename. It stores version tags, role, complete model specification, tensor
state dictionary, optimizer state, and training metadata; it does not pickle a
live Python model object. `load_dnn_model()` uses Torch's restricted
`weights_only` loader and reconstructs only a known GV4 architecture.
`write_artifact_metadata()` emits a readable JSON summary.

`export_dnn_to_native()` emits versioned metadata followed by named float32
tensors. This freezes the interchange boundary for the later standalone C++
model loader. The current compiled C++ inference runtime can already pass its
native structured features into these Python models through pybind callbacks;
the standalone C++ tensor parser and forward kernels remain a later phase.

### Model Bundle and Version 100

`model_bundle.py` is the only loader/publisher for a playable model generation.
A bundle is all-or-nothing and contains exactly these four named artifacts:

- `controller_value`;
- `adversary_value`;
- `controller_policy`;
- `adversary_policy`.

Each artifact directory contains `model.pt`, readable `metadata.json`, and
`native_model.tsv`. The bundle manifest stores the role, model kind, model
version, and SHA-256 of every file. Loading verifies every hash before exposing
any model, reconstructs only known GV4 architectures, and checks the complete
engine-manifest hash plus feature schema. Value and policy versions must agree
within each player role. A partially copied, corrupted, stale-topology, or
wrong-schema bundle therefore fails closed.

Publishing writes a fresh temporary directory, flushes each artifact and its
manifest, and renames the complete directory into its immutable versioned
location. `current_model.json` is a small atomic pointer to an already-published
manifest; workers resolve and verify that pointer rather than observing a model
directory while it is being written. Existing version directories are never
silently overwritten.

`bootstrap_untrained_dnn_v100.py` builds the initial four-model generation with
deterministic seeds and fresh AdamW states. Hidden layers retain their normal
random initialization, but policy output weights/biases are zero, producing
uniform softmax priors. Value output weights are zero with a large negative
pre-softplus bias, producing a stable nonpositive value very close to zero.
Thus version 100 exercises real model loading and inference while preserving
the intended neutral starting policy/value. It publishes
`models/Model_Version100/model_bundle.json` and, by default,
`models/current_model.json`.

The same loaded bundle constructs either `GV4DNNInference` for Python or the
pybind `InferenceRuntime` for native MCTS. The runner puts the acting role's
model version in each `SearchContext`, all four artifact versions in replay,
and both role versions in its game result. Model version is provenance only;
the explicit policy/bootstrap flags still decide whether predictions are used.

### Verification and Current Boundary

`tests/GV4_tests/agz_runtime_test.py` checks:

1. The same history seed repeats the same six canonical actions and times,
   exactly one action counts as one hop, and the source state remains unchanged.
2. A searched edge followed by a forced edge composes reward and discount
   correctly, backs up the exact target, normalizes visit probabilities, and
   atomically publishes a complete replay manifest.
3. Python and native TP2/PP2 runtimes play the same seeded uniform cycle and
   agree on history, end reason, searched/forced counts, final time, and final
   objective. Both backends also write nonempty structured replay, exercising
   conversion from both NumPy and native feature matrices.
4. Python and native initial states produce identical `StateSummary` values and
   matching semantic metadata apart from the backend name.
5. Observer events are contiguous and expose consistent immutable pre/post
   state summaries for both searched and forced actions.

`tests/GV4_tests/dnn_models_test.py` adds eight model checks: non-positive value
outputs, request-order invariance, inference-facade integration, action-order
equivariance, native-shaped matrix ingestion, the explicit one-replica guard,
Huber training plus checkpoint identity checks, and controller/adversary policy
training dimensions. A direct compiled-runtime smoke test also passes one
native state and all 39 canonical adversary actions through the new Python
value and policy callbacks.

`tests/GV4_tests/model_bundle_test.py` checks immutable four-artifact bootstrap
publication, pointer loading, neutral version-100 predictions, config mismatch
rejection, checksum corruption rejection, model-version propagation into game
and replay output, final-state model bootstrap, and use of the same bundle by
native policy/value callbacks.

The complete GV4 Python test directory now passes 106 tests. All new modules
pass Python bytecode compilation. PyTorch
prints a CUDA-driver compatibility warning on the current host while initializing
autograd, but the requested test training device is CPU and every test completes.

### Training and Evaluation Layer

The local training/evaluation pipeline is implemented in
`AlphaGoZeroGV4/training_and_evaluation/`:

    replay_dataset.py
    indexed_replay.py
    trainer.py
    arena.py
    baselines.py
    sjf_runner.py
    promotion.py
    agz_train_eval_promote.py
    evaluation_pipeline_logger.py

These modules consume only the stable replay, model-bundle, runner, and
`EngineRuntime` contracts. They do not import request-transition logic, decode
credit rules, KV accounting, pipeline scheduling, or Python/native state
classes. A future GV4 semantic change should therefore affect this layer only
when it deliberately changes the config hash, feature schema, replay schema, or
public runtime records.

#### Replay Dataset Validation

`replay_dataset.py` is the strict decoding boundary between immutable self-play
files and model training. It discovers only files named
`replay_manifest.json`; an incomplete directory without the final manifest is
ignored. Before exposing a sample, it checks:

1. The manifest status and replay schema.
2. The complete GV4 config hash and feature schema.
3. Every engine metadata schema recorded by the worker.
4. Presence and SHA-256 of both JSONL files.
5. Declared state/action row counts.
6. Game, cycle, decision, root-node, and player identity.
7. Exact fixed-vector widths, variable-row widths, per-replica offsets, and one
   replica feature row per configured replica.
8. Unique canonical actions and raw aliases containing their representative.
9. Nonnegative visits/priors, finite values, exactly one selected action, and a
   visit distribution summing to one.
10. Finite, non-positive controller-valued training targets.

One `ReplayRootSample` retains one structured state, its target value, and all
canonical action rows. `materialize_training_data()` flattens action rows only
at the final trainer boundary and records contiguous offsets for each root.
No request or action is truncated, padded into a global fixed maximum, or
silently dropped. `load_roots_at()` supports grouped byte-range reads so an
index can materialize multiple roots while opening each touched state/action
file once.

#### Indexed Replay

`indexed_replay.py` makes large replay histories randomly accessible without
loading all historical JSON into RAM. Its `addresses.npy` is a compact NumPy
structured array containing, per root:

- replay partition number;
- state byte offset and length;
- contiguous action byte offset, length, and action count;
- player role, game ID, decision index, and root-node ID.

The address array is memory-mapped on open. Sampling first filters addresses by
controller/adversary role and then uses seeded NumPy sampling without
replacement. Requested roots are grouped by partition and read in byte order,
but returned in the original random sample order.

Index construction streams state rows and contiguous action groups together.
It requires decisions to be contiguous within a replay partition, checks all
declared row totals, and passes every indexed root through the strict dataset
parser in bounded chunks. The index metadata records the config/schema,
validated manifest records, source paths, file sizes and modification times,
and manifest SHA-256 values. A changed source or config makes the index stale
and causes a rebuild. Construction is protected by an inter-process file lock;
the address array and metadata are atomically replaced, with metadata published
last.

#### Four-Model Trainer

`trainer.py` samples controller and adversary roots independently, converts each
role's roots to the existing structured DNN inputs, and trains exactly four
models:

- controller value with normalized Huber loss;
- controller policy with per-root visit-distribution cross entropy;
- adversary value with normalized Huber loss;
- adversary policy with per-root visit-distribution cross entropy.

Each artifact warm-starts from the matching incumbent checkpoint. The existing
model trainer restores AdamW state, validates role/config/schema identity, clips
gradients, and records losses and row counts. The coordinator cannot pair a
controller value model with an adversary checkpoint or publish only part of a
candidate. `train_candidate()` publishes one immutable four-model candidate
bundle only after all four fits succeed. Bundle metadata records the incumbent
manifest, replay index, available/sample root counts, action counts, and all
training metrics.

`TrainerConfig` bounds roots per role and owns epochs, value/policy batch sizes,
learning rate, weight decay, Huber delta, value scale, Torch thread count,
device, and seed. The same selected roots train a role's value and policy heads,
which keeps both targets aligned and avoids decoding replay twice.

#### Paired Arena

`arena.py` evaluates a candidate with paired seeds. Every pair runs three games:

1. incumbent controller versus incumbent adversary;
2. candidate controller versus incumbent adversary;
3. incumbent controller versus candidate adversary.

The history hop count, game seed, horizon, MCTS settings, and timing-provider
factory are identical inside a pair. This isolates one candidate role at a time
instead of changing both players and making attribution ambiguous. The mixed
inference facade selects both value and policy models from the same role bundle
and supports either the Python or native runtime.

All objective values retain the controller-cost convention. A controller
improvement is:

    incumbent_cost - candidate_controller_cost

An adversary improvement is:

    candidate_adversary_cost - incumbent_cost

Positive is therefore better for the role being evaluated in both cases.
`RoleArenaStats` reports wins, ties, losses, score rate with a half point for a
tie, and mean signed improvement. Evaluation uses zero action temperature and
model bootstrap; no root exploration noise is introduced by this layer.

#### SJF256 Baseline

`baselines.py` defines SJF256 as a semantic controller action:

    eviction = evict_none
    prefill budget = 256
    ordering = SJF

The selector computes the corresponding raw action index from the active
`ControllerActionConfig`, then searches every legal canonical action's alias
set. It does not assume that the representative raw index remains SJF256 after
canonicalization. If no 256-token prefill is feasible, it selects the matching
zero-prefill-budget SJF action. Under GV4 semantics that fallback still admits
mandatory decode work, so it is not equivalent to disabling decode service.

`sjf_runner.py` runs direct SJF controller decisions while leaving adversary
turns on the same model-backed MCTS used by normal evaluation. It compares that
cycle with a fully model-controlled cycle under the same adversary model,
history seed, search settings, timing provider, and horizon. It reports model
wins/ties/SJF wins, score rate, signed cost improvement, action counts, and how
often SJF needed the zero-budget fallback.

#### Independent Role Promotion

`promotion.py` applies independent thresholds to controller and adversary arena
statistics. `PromotionPolicy` requires a minimum paired-game count, role score
rate, and mean improvement. Four outcomes are possible:

- neither role passes: retain the incumbent bundle;
- both roles pass: point directly at the candidate bundle;
- only controller passes: candidate controller pair plus incumbent adversary
  pair;
- only adversary passes: incumbent controller pair plus candidate adversary
  pair.

A one-role outcome creates a new immutable composite bundle. Value and policy
versions always move together within a role, all artifact hashes/config schemas
are revalidated, and the worker-visible `current_model.json` pointer is replaced
only after the complete bundle exists. A per-candidate promotion JSON records
the decision, source hashes, promoted role versions, and final manifest.

#### Thin Cycle Orchestrator

`agz_train_eval_promote.py` deliberately contains no training loss, arena game
logic, SJF action rule, or promotion formula. One call to
`run_train_eval_promote()` performs these steps:

1. Open or rebuild the replay index.
2. Resolve and verify the current incumbent bundle.
3. choose the next unused candidate version, unless explicitly supplied;
4. train and atomically publish all four candidate models;
5. run the paired role-isolation arena;
6. make and publish the independent role promotion decision;
7. optionally compare the promoted result with SJF256;
8. atomically write one cycle summary JSON.

Its dataclass config receives paths and the dedicated trainer, arena, promotion,
and optional SJF configs. A small CLI accepts an engine-config factory and the
main local-cycle controls. The forthcoming shared top-level config can construct
this dataclass without changing orchestration code.

#### Evaluation Logging

`evaluation_pipeline_logger.py` receives `PlayedStep`, `GameCycleResult`, and
immutable `StateSummary` records only. It creates:

- `arena_games/game_<id>_<cycle>.csv`, one row per applied action plus an
  `arena_end` row;
- `arena_results.csv`, one final row per arena or SJF game;
- `sjf_results.csv`, one paired model-versus-SJF comparison per row;
- `training_cycles.csv`, `arena_summary.csv`, `promotion_results.csv`, and
  `sjf_summary.csv` for cycle-level records.

Per-action rows retain the useful GV3 columns for player/turn, action identity,
times, objective, MCTS visits/Q/prior summaries, and top-five actions. They add
GV4 request terminal sets, separate prefill/decode lateness, available/reserved/
minted decode credit, committed decode tokens, per-rank KV capacity/committed/
reserved blocks, in-flight microbatch IDs, and the next adversary tick. Fields
that are not present in the backend-neutral MCTS result remain explicitly blank
rather than being reconstructed from engine internals. CSV append is protected
by thread and process locks; headers are written once.

#### Training Pipeline Verification

`tests/GV4_tests/training_pipeline_test.py` performs six integration checks on
real GV4 replay:

1. The disk index finds both roles, samples deterministically without
   replacement, and reconstructs normalized structured roots.
2. All four candidate models warm-start and publish with one matched version.
3. Paired arena and SJF runs produce the expected five game rows and per-game
   backend-neutral CSV traces.
4. SJF256 is recovered through canonical alias membership and decodes to the
   exact no-eviction/256/SJF tuple.
5. Controller-only promotion creates a composite whose controller value/policy
   are version 101 while both adversary artifacts remain version 100; loading
   through the current pointer revalidates that composition.
6. The thin orchestrator completes index, train, arena, non-promotion, logging,
   and atomic summary publication in one local cycle.

Before the distributed layer was added, the full `tests/GV4_tests` discovery
run passed 106 tests. The sections below complete the shared top-level
configuration, deployment, transfer, coordinator, worker, and distributed
evaluation lifecycle. Standalone native DNN execution remains later work;
current native search uses parity-tested pybind callbacks into the same Python
models.

### Shared Experiment Configuration

The shared orchestration manifest is implemented in
`AlphaGoZeroGV4/config.py`. It is intentionally separate from
`GV4_Engine/config.py`:

- `GV4EngineConfig` remains the only source of game semantics, including TP/PP,
  KV limits, request bounds, action spaces, SLOs, objective costs, reward
  discounting, predictor configuration, and feature/native schema versions.
- `ExperimentConfig` owns only AlphaGoZero orchestration: self-play search,
  worker process limits, replay retention, training, arena evaluation,
  promotion, optional SJF evaluation, distributed evaluation, and deployment
  process settings.
- A semantic change therefore does not require edits to worker, coordinator,
  transfer, deployment, trainer, or arena logic. It changes the engine factory
  output and its manifest hash; feature or replay changes additionally require
  their explicit schema-version bumps.

The experiment records are:

| Record | Responsibility |
| --- | --- |
| `SelfPlayConfig` | Backend, complete MCTS/rollout settings, base seed, random-history hop range, horizon, action safety limit, played-action temperature, bootstrap mode, and model device. |
| `WorkerConfig` | Parallel game count, threads per child, disjoint game-ID stride, shard thresholds, ready-shard backpressure, poll/retry intervals, upload switch, and optional completed-game retention. |
| `CoordinatorConfig` | Bounded replay size, new-root training trigger, minimum replay per role, ingestion work per pass, poll interval, training switch, and promotion broadcast switch. |
| `TrainerConfig` | Existing bounded per-role sample and optimizer configuration. |
| `ArenaConfig` | Existing paired role-isolation evaluation configuration. |
| `PromotionPolicy` | Existing independent controller/adversary promotion thresholds. |
| `SJFBenchmarkConfig` | Optional post-promotion model-versus-SJF256 evaluation. |
| `DistributedEvaluationConfig` | Distributed-arena switch, pairs per chunk, concurrency per host, and remote-output retention. |
| `DeploymentConfig` | Native build parallelism and explicit environment variables shared by launched processes. |

`ExperimentConfig.engine_config_factory` has the form `module:function`. The
function must take no arguments and return one validated `GV4EngineConfig`.
The experiment file stores both this import path and the exact engine-manifest
SHA-256. On every load, the factory is called again and the resulting hash is
compared with the stored hash. Editing a factory underneath an existing
experiment therefore fails before self-play or training rather than silently
mixing incompatible replay.

`write_experiment_config()` publishes JSON with temporary-file, `fsync`, and
atomic-rename semantics. `load_experiment_config()` reconstructs every nested
dataclass and validates the engine identity. The manifest is portable because
machine-specific repository, experiment, SSH, and Python paths are not stored
in it.

`ExperimentPaths` gives every host the same logical layout below its own
experiment root:

    experiment_config.json
    cluster.json
    deployment_manifest.json
    models/current_model.json
    coordinator/incoming/
    coordinator/incoming_uploading/
    coordinator/global_replay/partitions/
    coordinator/acks/
    coordinator/training/
    workers/<worker_id>/game_runs/
    workers/<worker_id>/completed_games/
    workers/<worker_id>/replay/active/
    workers/<worker_id>/replay/ready/

The TP2/PP2 integration config now also exposes
`GV4_Engine.GV4_MCTS_Test.config:build_default_engine_config` as a
zero-argument factory. It is useful for deterministic integration manifests;
production experiments should point at their own immutable zero-argument
factory.

### Cluster Inventory and Process Boundary

`distributed_operations/cluster.py` defines the infrastructure manifest.
`HostSpec` records a path-safe host ID, SSH address or `local`, integral
ordinal, role, absolute repository root, absolute experiment root, and that
host's Python executable. `ClusterSpec` requires exactly one coordinator, at
least one worker, unique host IDs, and unique positive worker ordinals.
Evaluation hosts are optional; when absent, workers also form the arena pool.

The module contains the deliberately small local/SSH primitives used everywhere
else:

- safely quoted argv and environment execution;
- non-destructive rsync/copy in both directions;
- source-tree synchronization with generated-directory exclusions;
- atomic cluster-manifest publication;
- guarded detached launch with one PID file and one append-only log;
- PID-scoped status and TERM shutdown.

Source synchronization never uses `--delete`. It omits generated directories
such as `.git`, virtual environments, caches, simulator output, bytecode, and
native build directories. Host-local predictor caches and experiment data must
be provisioned independently and are never deleted by deployment.

There is one non-obvious Python startup rule. The repository contains a
top-level `vidur-GV4/types/` package, which can shadow Python's standard
library `types` if `vidur-GV4` is placed in `PYTHONPATH` before interpreter
startup. `AlphaGoZeroGV4/process_entrypoint.py` avoids that collision. It is
invoked by absolute path, lets Python finish startup, then inserts the
repository paths and runs the requested module. Worker games, coordinator
training children, distributed-arena chunks, deployment validation, and
long-running daemons all use this entrypoint. `PYTHONPATH` contains only the
repository root during startup.

### Durable Replay Transfer

`distributed_operations/durable_transfer.py` defines an explicit replay
lifecycle:

    mutable active games
      -> checksum-complete ready shard
      -> coordinator/incoming_uploading
      -> coordinator/incoming
      -> global_replay/partitions
      -> durable acknowledgement
      -> worker deletion

`freeze_replay_shard()` first validates every contained replay partition,
renames the mutable active directory out of service, writes one
`shard_manifest.json`, writes `SHA256SUMS`, validates the result, and finally
renames it into `ready/<shard_id>`. If construction fails, the active
directory is restored. The manifest binds worker/shard identity, engine and
feature schemas, game IDs, role root counts, model lineages, byte counts, and
SHA-256 values.

`publish_replay_shard()` is idempotent. A local coordinator receives an
independently validated copy; a remote coordinator receives rsync data in
`incoming_uploading`, verifies checksums, and atomically renames it into
`incoming`. Reusing the same worker/shard identity with another digest is an
error.

The coordinator writes an acknowledgement only after the shard has reached
the immutable global replay partition. A worker compares the acknowledged
digest before deleting its ready copy. A lost response therefore causes a safe
retry, not replay loss or duplicate training data.

The same module distributes models. Each already-validated bundle is copied to
a content-addressed directory containing the bundle manifest hash. Every
checkpoint, metadata file, native export, and bundle manifest is SHA-256
checked on the destination before `current_model.json` is atomically replaced.
Workers can finish games against their launch-time bundle while the next
generation is staged.

### Worker Daemon

`distributed_operations/worker_daemon.py` supervises independent game
processes. One daemon:

1. Loads restart-safe counters and recovers unfinished child records.
2. Resolves `current_model.json` and freezes its complete four-model identity
   before each game launch.
3. Allocates a disjoint game ID as
   `worker_ordinal * game_id_stride + local_sequence`.
4. Derives the game seed and random history-hop count deterministically.
5. Translates every self-play and MCTS field into one explicit
   `AlphaGoZeroGV4.runner` command.
6. Runs each game in an isolated process with bounded thread environment.
7. Accepts output only after game identity, config hash, model lineage, replay
   schema, row counts, and replay checksums validate.
8. Moves valid games into the active replay area and invalid games into
   `failed_games` with a readable reason.
9. Optionally copies complete raw game directories into `completed_games`
   without leaving them in the recovery queue.
10. Freezes active data when any game, root, or byte threshold is reached.
11. Uploads ready shards, polls acknowledgements at a separate interval, and
    removes only acknowledged local data.
12. Stops launching games when the configured ready-shard bound is reached.

Persistent sequence counters are saved before a child starts, so a crash may
leave a harmless ID gap but cannot reuse a game or shard identity. Rollout seed
uses the configured rollout offset plus the unique game seed. A model pointer
change affects only newly launched games.

### XL Coordinator

`distributed_operations/xl_coordinator.py` owns global replay and exactly one
training child. Its state file records accepted shard identities and digests,
accept order, roots received since the previous cycle, completed cycle count,
active child metadata, and the last error.

Each pass:

1. Reconciles a completed or failed training child.
2. Validates and ingests a bounded number of incoming shards.
3. Acknowledges duplicates with the same digest and rejects identity collisions.
4. Prunes the oldest replay partitions only when no training process is using
   them and only while preserving configured controller/adversary minima.
5. Writes a visible training-gate record.
6. Starts at most one train/evaluate/promote child when new-root and per-role
   minima are satisfied and a current model exists.

The child calls the existing thin `run_train_eval_promote()`; the coordinator
does not implement losses, arena scoring, promotion, or game transitions.
Completed cycles subtract only the roots visible at launch, so replay arriving
during training remains eligible for the next cycle. Successful promotion
broadcasts the atomically published current bundle to workers. Failed cycles
retain replay and expose their error rather than advancing counters.

### Distributed Arena

`distributed_operations/distributed_eval.py` preserves the exact paired arena
contract while distributing work. It partitions `ArenaConfig.games` into
contiguous, non-overlapping pair ranges and assigns chunks to arena hosts in
stable round-robin order. A per-host semaphore bounds concurrent chunks.

Before a chunk starts, the orchestrator copies the experiment manifest and
content-addressed incumbent/candidate bundles. Each chunk runs all three
required scenarios for every assigned pair:

    incumbent controller vs incumbent adversary
    candidate controller vs incumbent adversary
    incumbent controller vs candidate adversary

Its local seed base is shifted by the global pair start, then local pair IDs
are rewritten to global IDs. Collection rejects missing ranges, overlapping
ranges, duplicate scenarios, mixed model versions, or different seeds inside
a pair. Controller and adversary aggregate statistics are recomputed from the
merged raw games using the configured tie tolerance; chunk summaries are never
averaged.

`run_train_eval_promote()` now accepts one optional `ArenaEvaluator`
callback. Its default is the original local arena. When
`DistributedEvaluationConfig.enabled` is true, the coordinator injects the
distributed evaluator. Trainer, promotion, SJF, and logger code are unchanged.
SJF remains a coordinator-local optional benchmark; only the paired promotion
arena is distributed.

### Deployment Orchestrator

`distributed_operations/deploy.py` provides composable operations and one
`all` path:

- `sync-source` merges source onto each distinct machine while preserving
  caches and experiment outputs.
- `stage` writes a deployment record and copies experiment/cluster manifests;
  it can also stage the initial current model.
- `build-native` runs CMake only on hosts whose configured self-play, arena,
  or SJF role needs the native backend.
- `validate` starts a short subprocess on every host and checks the engine
  factory/hash, four-model bundle, feature schema, imports, and native module
  when required.
- `start` repeats preflight, then starts one coordinator and all workers with
  guarded PID files.
- `status` reports only this experiment's PID files.
- `stop` sends TERM only to those recorded worker/coordinator PIDs.
- `all` performs source sync, deployment record, manifest/model staging,
  selective native build, preflight, and daemon launch in that order.

The deployment record stores UTC time, experiment and cluster hashes, engine
manifest hash through the experiment, source Git revision/dirty state, and
initial model identity. It is copied to every host for later audit.

Because of the startup-path rule above, invoke deployment through the safe
entrypoint from the classical-search repository root:

    /home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
      vidur-GV4/AlphaGoZeroGV4/process_entrypoint.py \
      AlphaGoZeroGV4.distributed_operations.deploy \
      --experiment-config /path/to/experiment_config.json \
      --cluster /path/to/cluster.json \
      all \
      --source-root /home/shazer/Desktop/Research/Vidur/vidur-classical-search \
      --current-model /path/to/models/current_model.json

The same prefix supports `status`, `validate`, and `stop`; those commands
require only the two manifest arguments and their command name. SSH
authentication, remote Python environments, and any engine predictor caches
referenced by the immutable engine factory must already exist.

### Distributed Operations Verification

`tests/GV4_tests/distributed_operations_test.py` adds seven focused checks:

1. Experiment and cluster manifests round-trip and an engine-factory hash drift
   is rejected.
2. Worker command construction preserves model identity and every relevant
   search switch, including rollout seed.
3. Non-destructive source synchronization copies code but excludes generated
   cache data.
4. A real four-model bundle is staged through atomic pointers on two local host
   roots, then both hosts pass subprocess preflight through the safe entrypoint.
5. A real GV4 replay is frozen, checksum-validated, idempotently published,
   accepted once, acknowledged, retired by the worker, and reconstructed by a
   restarted coordinator without duplication.
6. Modifying a frozen replay payload is rejected by transport checksums.
7. Distributed chunk planning is stable and merged arena results require exact
   pair/scenario/seed coverage before role statistics are recomputed.

After this layer was added, the complete `tests/GV4_tests` discovery run
passes 113 tests. The distributed tests execute local host specifications, real
subprocess preflight, real replay files, and real model artifacts. They do not
claim to test site-specific SSH authentication, network interruption, remote
filesystem permissions, scheduler integration, or predictor-cache
provisioning; one deployment smoke run on the target machines remains required
before a production experiment.

The remaining planned AlphaGoZero work is the requested logger revision.
Standalone C++ tensor execution also remains separate: current native search
continues to use the parity-tested pybind model callbacks.
