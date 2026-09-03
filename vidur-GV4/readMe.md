# GV4: Multi-Replica AlphaGoZero Scheduling

## Status

`vidur-GV4` is an isolated copy of the current `vidur` implementation for
designing the next game version. The existing GV3 and real-vLLM pipelines are
preserved as the starting baseline. This document is the normative design
contract for GV4. The implementation must not silently diverge from the state,
action, timing, memory, reward, snapshot, and validation rules defined here.

## Goal

GV4 will learn end-to-end scheduling policies for an inference deployment with:

- Multiple data-parallel replicas.
- Tensor and/or pipeline parallelism inside each replica.
- Finite, block-allocated KV caches.
- Request admission, batching, stalling, and terminal eviction decisions.
- A global request router/load balancer above the replica schedulers.
- Exact deterministic Vidur transitions using a compact TP/PP stage calendar
  suitable for high-throughput AlphaGoZero self-play and MCTS.

The objective is a scalable policy that minimizes the configured system cost,
including request SLO violations, while respecting execution and memory
constraints.

## System Hierarchy

```text
Request arrivals / adversary
            |
            v
Global load balancer
  - route request to a replica
  - optionally defer an unassigned request
            |
            v
+-------------------+  ...  +-------------------+
| Replica scheduler |       | Replica scheduler |
| local request set |       | local request set |
| local KV cache    |       | local KV cache    |
| TP x PP workers   |       | TP x PP workers   |
+-------------------+       +-------------------+
```

Data parallelism creates replicas. TP and PP are model-parallel dimensions
inside one replica. A request is initially routed to one replica, whose
scheduler controls batching and the request's KV-cache lifetime.

## Controller Decisions

### Global Routing

The load balancer handles one unassigned request at a time and chooses a
replica or a legal hold action. Simultaneous arrivals are assigned through a
sequence of zero-time decisions rather than one combinatorial joint action.

The routing policy observes the pending requests and summaries of every
replica, including queued work, deadline pressure, KV availability, in-flight
pipeline work, and predicted service cost.

### Replica Scheduling

When a replica can accept another batch, its scheduler constructs a legal
batch from local requests. The final action design may be sequential, for
example adding request-token allocations followed by a commit action, so that
arbitrary request subsets do not become one enormous flat action space.

### KV Feasibility And Terminal Eviction

Before a batch executes, the simulator computes its exact incremental KV-block
requirement. If immediately available blocks are insufficient, the controller
may select terminal eviction targets whose released blocks make the batch
feasible, or select another legal scheduling decision.

A controller eviction preserves the current GV3 semantics: the target leaves
the active system permanently, is marked dropped, releases its KV blocks, and
contributes terminal cost 3. It cannot later resume.

Cache fullness alone does not make every decode illegal. A decode that fits in
an already allocated block can still run, while one crossing a block boundary
may require more memory. Finishing requests can also release cache capacity.

### Pipeline Execution

Committed batches enter a compact TP/PP stage calendar. GV4 tracks stage
availability, in-flight microbatches, inter-stage dependencies, communication,
and KV reservations without allocating a general-purpose event object for
every stage transition. The calendar must be mathematically equivalent to the
corresponding FIFO event trace. A replica decision admits work at stage 0; it
does not treat a multi-stage pipeline as one indivisible time jump.

## Markov State

The full state will include at least:

- Current simulated time, adversary state, and pending global arrivals.
- Static replica topology and execution-profile identity.
- Every active request's replica ownership, progress, deadline/SLO state, and
  terminal lifecycle state.
- Per-request KV allocation and partial-block occupancy.
- Per-replica free, allocated, reserved, and pending-release KV blocks.
- Replica waiting/running sets and scheduler limits.
- PP stage queues, active stages, and in-flight batches.

The Python and native implementations must encode the same state and produce
the same legal actions, transitions, rewards, and discounting.

## Learning Structure

The intended training progression is:

1. Train and validate a single-replica scheduler with finite KV capacity.
2. Add TP/PP-aware execution and pipeline state.
3. Share or condition the replica policy across equivalent replicas.
4. Train the global load balancer using the learned replica scheduler and
   value estimates.
5. Alternate or jointly fine-tune routing and replica scheduling with a shared
   system-level objective.

The load balancer may use the local value change caused by assigning a request
to each replica as an input or prior. Freezing local schedulers is useful for
initial router training, but final optimization must account for the workload
distribution induced by the router.

## Core Correctness Invariants

- Allocated and reserved blocks never exceed physical logical-block capacity.
- A block is not simultaneously free and referenced by an active request.
- A request has one owning replica unless migration is explicitly modeled.
- In-flight requests cannot be selected by a controller eviction rule.
- Batch execution, completion, terminal eviction, and waiting accrue the
  configured time-based rewards and discounts exactly once.
- Zero-time subdecisions do not introduce artificial discounting.
- Waiting cannot create a permanent deadlock when a legal progress action is
  available.
- Python, native, and real-vLLM state transitions remain trace-compatible.

## Validation Direction

GV4 will require:

- Unit tests for block allocation, release, reservation, and terminal eviction.
- Python/native parity tests for complete event traces.
- TP1/PP1, TP2/PP1, TP1/PP2, and TP2/PP2 timing and state tests.
- Forced low-memory scenarios covering terminal eviction and stalling.
- Small exhaustive-search environments for measuring policy regret against a
  true optimum.
- Comparisons against standard routing, batching, and eviction heuristics.
- Shadow tests against the corresponding real-vLLM configuration.

# Implementation Design for the GV4 Structure

This section is normative. Terms such as **must**, **shall**, and **cannot**
define implementation requirements. A behavior marked **deferred** is outside
the first implementation and must not be approximated silently.

## 1. Scope and Terminology

- A **replica** is one data-parallel copy of the model.
- `TP` is the tensor-parallel degree inside each pipeline stage.
- `PP` is the number of pipeline stages inside a replica.
- A **rank** is one GPU process. A homogeneous replica therefore owns
  `TP * PP` ranks.
- A **controller batch** or **microbatch** is one scheduler action admitted at
  stage 0. It may contain prefill allocations, one decode token for eligible
  decode requests, or both.
- A **decision boundary** is a time at which the adversary or controller can
  choose an action.
- An **internal transition** is a deterministic pipeline completion, KV
  release, decode-budget update, or lifecycle update that occurs while advancing
  between decision boundaries.
- An **eviction rule** selects active, non-in-flight requests for a terminal
  drop. Each target leaves the system permanently, releases its KV, and incurs
  the configured terminal drop cost.

The first milestone is one replica with finite KV and configurable TP/PP. The
multi-replica router is layered on only after the local scheduler is correct.

## 2. Initially Supported Execution Model

The compact calendar described below is exact under these initial constraints:

- Replicas are homogeneous within an experiment.
- `TP >= 1`, `PP >= 1`, and `TP * PP` equals the ranks assigned to a
  replica.
- Model layers are assigned deterministically to PP stages. The initial
  implementation requires equal layer counts unless the execution profile
  explicitly contains per-stage layer ranges.
- Every stage executes one microbatch at a time and downstream service is FIFO.
- Service time is deterministic for a fixed stage, batch shape, model, GPU SKU,
  TP degree, PP degree, kernel set, and profile version.
- A request may belong to at most one in-flight microbatch.
- In-flight requests cannot be reordered, migrated, or terminally evicted.
- The inter-stage queue capacity is at least the configured maximum number of
  in-flight microbatches. This prevents downstream queue backpressure in the
  first implementation while retaining an explicit capacity check.
- KV allocation is finite and block based on every rank.
- CPU/disk KV offload is disabled.
- Prefix sharing and prefix caching are disabled.
- The production model family is DNN. HGB artifacts are legacy-only and must
  not be loaded by a GV4 experiment.

Smaller inter-stage queues, heterogeneous replicas, live request migration,
prefix sharing, KV offload, and non-FIFO stage scheduling are later extensions.
Each requires a new state-schema and transition-contract version.

## 3. Configuration Contract

One resolved, immutable experiment manifest must contain all topology, profile,
memory, game, feature, search, model, and training values. Python, native,
self-play, evaluation, synthetic testing, and real-vLLM adapters must consume
the same manifest hash.

### 3.1 Topology and profile identity

The manifest must define:

- `num_replicas`.
- `tensor_parallel_size`.
- `pipeline_parallel_size`.
- Ordered GPU rank membership for every replica and PP stage.
- Model identifier and exact model revision.
- Number of layers, hidden dimension, attention head dimension, KV-head count,
  vocabulary size, weight dtype, and KV dtype.
- GPU SKU, interconnect topology, attention backend, MLP backend, and collective
  implementation.
- Execution-profile path, profile checksum, and predictor-cache checksum.
- Per-stage layer range and per-stage execution/communication predictor.
- Scheduler batch-token limit, sequence limit, and maximum in-flight
  microbatches.
- Inter-stage queue capacity.

Changing any item invalidates serialized states, replay, native caches, and
model compatibility unless an explicit migration exists.

### 3.2 Derived dimensions

The implementation derives rather than hardcodes:

```text
ranks_per_replica = TP * PP
total_ranks = num_replicas * TP * PP
layers_on_stage[k] = stage_layer_end[k] - stage_layer_begin[k]
controller_raw_actions =
    len(eviction_rules)
    * len(prefill_budgets)
    * len(ordering_heuristics)
```

The action-space specification, feature dimensions, policy action-feature
width, and native array limits are serialized with every model and replay
shard. Startup fails on a dimension or checksum mismatch.

## 4. Authoritative GV4 State

The complete simulator snapshot, not the DNN tensor, is the source of truth.
It contains the following state.

### 4.1 Global state

- Current simulator time.
- Player whose turn is next.
- Next adversary tick.
- Sliding one-second launch history and expiration times.
- Pooled, non-expiring available and reserved decode credits.
- Launch-window request-count and aggregate-prefill usage.
- Current objective/cost bookkeeping.
- Global request ID allocator and deterministic tie-break counter.
- Reproducible RNG state when stochastic behavior is enabled.
- Configuration and state-schema versions.

### 4.2 Per-request state

- Request ID and stable ordering key.
- Owning replica or `UNASSIGNED`.
- Lifecycle state:
  `WAITING_PREFILL`, `INFLIGHT_PREFILL`, `WAITING_DECODE`,
  `INFLIGHT_DECODE`, `STOP_PENDING`, `DROP_PENDING`, `COMPLETED`,
  `STOPPED`, or `DROPPED`.
- Arrival time, prefill deadline, decode deadline/SLO, age, slack, lateness, and
  already-recorded violation state.
- Original prefill/decode tokens.
- Committed, reserved, in-flight, and remaining token counts.
- Per-stage and per-rank KV block ownership.
- Used tokens in the final partial KV block and tokens until the next block
  boundary.
- In-flight microbatch ID and predicted stage start/finish times.
- Terminal stop/drop reason and whether physical KV release is pending.
- Pending stop/drop reason and timestamp.

Request progress is committed only when its microbatch exits the final PP
stage. Admission reserves work and KV; an admitted decode also reserves one
already-funded decode credit. Prefill has no credit reservation. Admission does
not pretend the token has completed.

### 4.3 Per-replica state

- Static TP/PP topology and profile identity.
- Waiting prefill/decode request sets.
- Number and token count of reserved and in-flight requests.
- Maximum in-flight capacity and current occupancy.
- Earliest next controller admission time.
- Per-stage availability/tail time and active microbatch.
- Per-stage FIFO/in-flight ordering.
- Per-stage/per-rank KV counters.
- Partial-block waste and number of decodes at a block boundary.

### 4.4 Per-stage and in-flight records

Each stage stores `tail_finish_time`, active microbatch ID, queue occupancy,
and profile identity. Each in-flight microbatch stores:

- Microbatch ID and controller action/canonical action IDs.
- Sorted request IDs.
- Prefill and decode token allocation per request.
- KV reservations and issued decode-credit reservations; prefill has no credit.
- Stage-ready, stage-start, and stage-finish times.
- Final completion time.
- Whether final completion effects have already been applied.

These fields are sufficient to reconstruct every future transition without
consulting hidden history.

## 5. Game-Engine Semantics

### 5.1 Two external decision clocks

GV4 exposes only two kinds of external decision boundary:

```text
next_time = min(
    next_adversary_tick,
    next_controller_admission_time,
)
```

`next_controller_admission_time` is not merely the time at which stage 0's
current kernel ends. It is the earliest time at which:

- Stage 0 can accept work.
- An in-flight slot is available.
- The admission queue has capacity.
- Internal completions needed to free a required resource have occurred.
- At least one controller transition can differ from a forced no-op.

Final-stage batch completions are internal transitions. They do not create an
extra player turn by themselves. They may, however, determine
`next_controller_admission_time` because they release in-flight capacity, KV,
or request dependencies.

### 5.2 Exact advancement to a boundary

Before either player acts at target time `t`, the engine must:

1. Find every in-flight completion with `completion_time <= t + eps`.
2. Process completions in ascending time and then microbatch-ID order.
3. Set the internal clock to each completion time while applying its effects.
4. Commit completed prefill/decode progress exactly once.
5. Apply prefill-to-decode transitions and mint decode credits.
6. Complete requests and release all of their KV when appropriate.
7. Resolve `STOP_PENDING` and `DROP_PENDING` requests whose issued work has
   drained.
8. Prune launch-window entries through time `t`.
9. Set `now = t`.
10. Apply automatic SLO drops due at `t`.

This ordering prevents a request completing exactly at a tick from being
incorrectly stopped, dropped, or charged after its completion.

### 5.3 Strict GV3 player alternation

The logical order remains:

```text
adversary -> controller -> adversary -> controller -> ...
```

Alternation does not require both players to have a meaningful action at every
boundary:

- If the next boundary is a controller admission before the next adversary
  tick, the adversary turn is a forced no-op and the controller acts.
- If the next boundary is an adversary tick while stage 0 cannot admit work,
  the adversary acts and the following controller turn is a forced no-op.
- If both clocks are equal, internal completions and automatic drops run first,
  then the adversary acts, then the controller sees the updated request set and
  acts if admission is legal.

Forced no-ops are auto-resolved. They do not invoke a model, consume an MCTS
branch, enter replay, or apply a nonzero discount. The state still records the
logical player transition so Python and native traces agree.

### 5.4 Controller decision

At a controller turn:

1. If stage 0 or admission capacity is unavailable, resolve a forced no-op.
2. Reconcile all completed, stopped, dropped, and newly decode-ready requests.
3. Generate raw actions from the configured dynamic action specification.
4. Resolve each action's terminal eviction targets.
5. Reject targets that are in-flight, inactive, or own no releasable blocks.
6. On a scratch state, terminally drop the targets and release their KV.
7. Build the candidate prefill/decode allocation.
8. Calculate incremental KV blocks on every stage and rank.
9. Validate funded decode availability, token limits, sequence limits,
   in-flight capacity, and inter-stage capacity. Prefill has no credit check.
10. Reserve KV and any automatically included decode funding atomically.
11. Admit at most one microbatch to stage 0.

No partial mutation is allowed when an action fails validation.

Controller transitions are classified as:

- `BATCH`: terminal eviction may occur and a nonempty microbatch is admitted.
- `EVICT_ONLY`: at least one valid target is terminally dropped, but no batch is
  admitted. Simulated time does not advance.
- `WAIT`: no eviction target and no batch. Time advances to the next boundary that can
  change legality, usually an adversary tick or an enabling completion.

Repeated zero-time `EVICT_ONLY` transitions must be finite: every such
transition must remove at least one active request and release its resident KV.
An identical state/action loop is an invariant failure.

### 5.5 Adversary decision

At each adversary tick, the adversary may launch requests and stop eligible
decode requests subject to strict masks, launch-window request-count and
aggregate-prefill caps, request bounds, and decode stop eligibility. New
requests become visible to the controller at the same timestamp after the
adversary action.

If the adversary stops an in-flight request, the request becomes
`STOP_PENDING`. Already-issued GPU work drains and is counted as consumed
compute. Its KV is not released until the final in-flight stage completes.
Stopping a non-in-flight request releases its KV immediately and transitions it
to `STOPPED`.

An automatic drop follows the same drain rule using `DROP_PENDING`. The drop
cost is recorded once at the logical drop time, while physical KV release may
occur later.

## 6. Compact Pipeline-Parallel Calendar

GV4 does not require a heap of generic stage events inside every MCTS state.
For FIFO stages with bounded in-flight work, an array of stage tail times and a
ring of in-flight records is sufficient.

For microbatch `b`, admitted at time `t`, and stage `k`:

```text
ready[b, 0]  = t
start[b, k]  = max(ready[b, k], stage_tail_finish[k])
service[b,k] = VidurStageTime(stage=k, batch_shape=b)
finish[b, k] = start[b, k] + service[b, k]
ready[b,k+1] = finish[b, k] + pp_communication_time[b, k]
stage_tail_finish[k] = finish[b, k]
completion[b] = finish[b, PP - 1]
```

`VidurStageTime` must use that stage's layer range, TP collectives, kernel
configuration, KV/context shape, and batch composition. It cannot reuse a
whole-replica latency and divide it by `PP` unless the profile explicitly
proves that model.

### 6.1 Stall and bubble interpretation

- If `ready[b,k] < stage_tail_finish[k]`, the microbatch waits. This is a
  downstream dependency stall.
- While a stage has no ready microbatch, it is idle. That idle interval is a
  pipeline bubble.
- A small 128-token microbatch queued behind a 4096-token microbatch does not
  acquire the 4096-token service time. It keeps its own service time but starts
  later.
- Pipeline latency is the final-stage completion time minus admission time.
- Pipeline throughput is governed by stage availability and overlap, not by
  treating the maximum stage service as a single global batch jump.

### 6.2 Controller re-admission

Stage 0 can usually accept the next microbatch at `finish[b,0]`, even though
`b` remains in later stages. The actual controller admission time is:

```text
next_controller_admission_time =
    earliest time satisfying(
        stage_0_is_free,
        inflight_count < max_inflight_microbatches,
        admission_queue_has_capacity,
        some legal controller transition is enabled,
    )
```

This value may be later than `finish[b,0]` when the in-flight limit is full or
when a required final completion has not released memory.

### 6.3 PP example

Suppose `PP=2`, time is 10.0 s, stage 0 is free, and stage 1 is busy until
10.20 s. A new 128-token batch needs 0.02 s per stage:

```text
stage 0: start=10.00, finish=10.02
stage 1: ready=10.02, start=max(10.02, 10.20)=10.20, finish=10.22
```

The controller may admit another batch at 10.02 if the in-flight and memory
constraints allow it. The first batch commits request progress at 10.22, not
10.02. Stage 1's 0.18 s wait is a stall for this batch; stage 0 being free after
10.02 is not evidence that the whole batch completed.

### 6.4 Inter-stage capacity

For the initial exact calendar:

```text
inter_stage_queue_capacity >= max_inflight_microbatches
```

Startup rejects a smaller value. Supporting a smaller queue requires explicit
backpressure: a stage cannot release its output until the downstream queue has
a slot, and that release time changes the stage tail. That extension must be
tested against a reference event simulator before being enabled.

## 7. Finite KV-Cache Model

### 7.1 Capacity per rank

For one rank on PP stage `k`:

```text
kv_bytes_per_token_rank =
    2
    * kv_element_bytes
    * layers_on_stage[k]
    * head_dim
    * kv_heads_on_tp_rank

capacity_blocks_rank =
    floor(
        usable_kv_bytes_rank
        / (kv_bytes_per_token_rank * block_size_tokens)
    )
```

The factor 2 represents key and value. `kv_heads_on_tp_rank` reflects TP
sharding. Usable bytes are GPU memory remaining after weights, runtime
workspace, graph buffers, communication buffers, and the configured safety
margin.

A logical request block must be realizable on every rank that stores its KV.
The replica's logical capacity is therefore constrained by the least-capable
required rank, not by summing memory across ranks.

### 7.2 Allocator accounting

Every stage/rank tracks mutually exclusive block classes:

- `free`.
- `allocated_live`.
- `reserved_inflight`.
- `pending_release_inflight_stop_or_drop`.

For every rank:

```text
free
+ allocated_live
+ reserved_inflight
+ pending_release_inflight_stop_or_drop
= physical_capacity
```

### 7.3 Incremental block requirement

The engine computes the block delta per selected request, stage, and rank.
A decode fitting in the request's existing partial block needs zero new blocks.
A decode crossing the next block boundary needs one additional logical block
on every required stage/rank. A prefill allocation may require multiple blocks.

An action is memory-feasible only if, atomically:

```text
free_blocks
+ blocks_released_by_selected_evictions
>= incremental_blocks_required
```

The comparison is checked independently on every required rank.

### 7.4 Terminal controller eviction

Only active, non-in-flight, resident requests can be selected. Eviction:

1. Removes the request from the active scheduler state.
2. Marks it completed and dropped at the current simulator time.
3. Releases all KV blocks owned by the request.
4. Cleans up its deadline and future decode eligibility. Already committed
   decode spend and already minted pooled credit are not retroactively changed.
5. Removes any previously accrued violation/lateness contribution for that
   request.
6. Adds the configured terminal drop cost, which is 3 by default.

Controller eviction consumes zero simulated GPU time. The request cannot
return to a waiting state or resume later.

### 7.5 Full-cache behavior

- Full cache does not disable all decode.
- A decode with zero incremental block demand remains legal.
- A boundary-crossing decode is masked unless terminal eviction or a completion
  makes a block available.
- Prefill is masked unless the action's selected eviction targets make all
  required block reservations feasible.
- If no batch is feasible and no eviction target is selected, the only
  canonical action is `WAIT`.
- If eviction targets are valid but no batch is selected, the transition is
  `EVICT_ONLY` and time remains unchanged.

## 8. Controller Action Space

The raw controller action is the Cartesian product:

```text
terminal eviction rule
x prefill token budget
x prefill ordering heuristic
```

The baseline has 9 rules, 9 budgets, and 4 heuristics, producing 324 raw
indices. No implementation may hardcode 324. The index formula is:

```text
raw_index =
    (eviction_rule_index * num_prefill_budgets + budget_index)
    * num_ordering_heuristics
    + heuristic_index
```

### 8.1 Baseline terminal eviction rules

| Rule | Deterministic target set |
|---|---|
| `evict_none` | No targets. |
| `evict_largest_prefill` | One waiting resident prefill with largest remaining prefill; smallest request ID breaks ties. |
| `evict_earliest_prefill_deadline` | One waiting resident prefill with earliest prefill deadline; smallest request ID breaks ties. |
| `evict_prefill_missed_deadline` | All eligible resident prefills with positive current prefill lateness. |
| `evict_prefill_lateness_over_0p5` | All eligible resident prefills with prefill lateness greater than 0.5 s. |
| `evict_longest_decode` | One waiting resident decode with greatest processed decode length; smallest request ID breaks ties. |
| `evict_decode_lateness_over_0p5` | All eligible resident decodes with total lateness greater than 0.5 s. |
| `evict_prefill_highest_lateness` | One eligible prefill with maximum positive prefill lateness. |
| `evict_decode_highest_lateness` | One eligible decode with maximum positive total lateness. |

In-flight, already pending-stop/drop, completed, stopped, dropped, unassigned,
and zero-block requests are excluded before applying a rule.

### 8.2 Baseline prefill budgets

```text
0, 128, 256, 512, 1024, 1536, 2048, 3072, 4096
```

The budget is the total prefill allocation across requests, not a per-request
budget. A positive budget may finish a final tail smaller than 128 tokens.
Allocation cannot exceed remaining logical work, batch-token limits, or KV
capacity. There is no prefill-credit check.

### 8.3 Baseline ordering heuristics

| Heuristic | Ordering key |
|---|---|
| `SJF` | Ascending remaining prefill tokens, then request ID. |
| `EDF` | Ascending absolute prefill deadline, then request ID. |
| `LST` | Ascending `remaining_slo - predicted_remaining_prefill_time`, then request ID. |
| `LJF` | Descending remaining prefill tokens, then request ID. |

### 8.4 Decode inclusion

GV3-style decode inclusion remains automatic after prefill allocation:

- At most one decode token is allocated per eligible decode request.
- Decode funding is an adversary workload constraint, not a raw controller
  action component. Automatic inclusion cannot issue more tokens than the
  pooled available balance.
- Batch sequence and batch token limits still apply.
- Requests already in-flight are excluded.
- Decodes requiring no new block are considered first.
- Remaining free blocks may then admit boundary-crossing decodes in stable
  request-ID order.
- KV is reserved before the microbatch enters stage 0.

### 8.5 Resolution order

For each raw action, resolution is deterministic:

1. Resolve and sort valid terminal eviction targets.
2. Apply the terminal drops to a scratch allocator.
3. Build prefill allocation using the budget and heuristic.
4. Add zero-increment decodes.
5. Reserve prefill blocks.
6. Add block-boundary decodes while capacity remains.
7. Validate all non-memory constraints.
8. Produce a canonical transition descriptor.

### 8.6 Canonicalization

Different raw actions often produce the same physical transition. The
canonical key is:

```text
(
    transition_kind,
    sorted_evicted_request_ids,
    sorted(request_id, phase, prefill_tokens, decode_tokens),
    sorted_per_rank_kv_delta,
)
```

Equivalent raw actions are one MCTS edge. Their policy prior probabilities are
summed, and the smallest raw index is retained as the representative for logs
and replay. Canonicalization must happen identically in Python and native code.

### 8.7 Controller examples

**Normal mixed batch:** With pending prefills of 300 and 700 tokens, budget 512
and SJF allocate 300 to the first request and 212 to the second, subject to
scheduler and KV limits. Eligible decodes are then added.

**Full cache with partial decode blocks:** If no free block remains, a decode
whose current block has space may run. A decode at a block boundary and all
prefill allocations are masked unless terminal evictions or an internal
completion free the required blocks.

**Evict only:** If a selected rule terminally drops requests but budget 0 and
no decode is eligible, the action is `EVICT_ONLY`. The next controller
decision may use the newly free capacity at the same simulator time after the
forced adversary no-op.

## 9. Adversary Action Space

The inherited adversary action chooses:

- Launch count from 0 through 7.
- For a positive launch count, one prefill-size template from
  `128, 256, 512, 1024, 1536, 2048, 3072, 4096`.
- One stop rule:
  `stop_none`, `stop_longest_decode`, `stop_shortest_decode`,
  `stop_all_decodes_over_512`, or `stop_all_decodes_over_216`.

Launch count 0 has only the five stop choices. The baseline flattened size is:

```text
5 + (7 * 8 * 5) = 285
```

This dimension is also derived dynamically. Strict masking rejects actions
that violate the launch-window request-count or aggregate-prefill cap, native
request capacity, or stop eligibility. There is no adversary prefill-credit
lot. The default launch window is one second with at most seven
requests. Each request has at most 4096 prefill and 864 decode tokens.

## 10. Request, Credit, and SLO Rules

### 10.1 Request bounds

- Allowed adversary prefill templates:
  `128, 256, 512, 1024, 1536, 2048, 3072, 4096`.
- Maximum prefill tokens per request: 4096.
- Decode tokens per request: 1 through 864.
- Target long-run average decode size: 216.
- Monitoring target average prefill size in the one-second window: 1024.

### 10.2 Adversary workload budgets

**Prefill:** GV4 has no prefill credits. The adversary launch history directly
enforces both limits over the sliding window:

```text
window_request_count <= max_requests_per_launch_window
window_prefill_tokens <=
    max_requests_per_launch_window
    * target_prefill_tokens_per_request_window_average
```

With the baseline values, this is at most seven requests and 7168 aggregate
prefill tokens in one second. These limits constrain adversary launches only;
they never reduce the controller's prefill action budget.

**Decode:** Completing prefill at the final PP stage mints 216 pooled,
non-expiring decode credits exactly once for that request. Every decode token
actually issued reserves one credit, and final-stage completion consumes that
reservation. Decode funding is automatic adversary-side workload eligibility,
not a controller action choice.

When admission reserves the final available credit, every existing
WAITING_DECODE request stops immediately and releases KV. Existing
INFLIGHT_DECODE requests become STOP_PENDING; their reserved token is their
last token and must drain before KV release. Credits minted by later prefill
completions cannot revive these stopped requests.

```text
available_decode_credit
+ reserved_decode_credit
+ total_committed_decode_tokens
= 216 * requests_that_reached_decode
```

The available and reserved balances cannot become negative. A request can
consume at most 864 decode tokens. Because the balance is global, short requests
bank unused credits and a later request may exceed 216 tokens when the bank can
fund it. Therefore individual requests and short samples can diverge greatly,
while cumulative decode length is bounded by 216 per request that reached
decode. It approaches 216 only when the adversary spends most of the budget.
If the available pool reaches zero, the active decode generation terminates;
later mints fund new decode entrants rather than resuming stopped requests.

Stopping a non-in-flight decode charges nothing at the stop and prevents all
future token spending. Stopping an in-flight decode creates `STOP_PENDING`;
already-issued work drains and consumes its reserved credit before the request
becomes `STOPPED`. Credit-exhaustion stops follow the same drain rule and use
`DECODE_CREDIT_EXHAUSTED` as their terminal reason. No already-committed token
is refunded.

### 10.3 SLO and drop cost

The inherited per-request cost contract is:

```text
0                         if all relevant deadlines are met
1 + min(lateness, 2.0)   for a violation
3                         for a terminal automatic drop
```

Default automatic-drop lateness is 2.0 s. A request's violation or drop cost is
recorded exactly once. A controller eviction uses this same terminal drop path
and therefore charges cost 3 for every evicted request.

## 11. Reward, Discounting, Horizon, and Bootstrap

The controller minimizes cost. Values therefore use the existing nonpositive
reward/value convention:

```text
edge_reward = objective_before - objective_after
Q(s,a) = edge_reward + gamma ** (elapsed_time / discount_step_sec) * V(s')
```

- The objective is evaluated before and after the complete transition,
  including every internal completion and drop crossed by the edge.
- Internal events are not independently charged again.
- Zero-time transitions use discount exponent zero.
- Time is rounded only through the configured deterministic time utility.
- The default numerical epsilon is `1e-9` and time is rounded to 10 decimal
  digits.
- GV4 is a continuing game. A self-play horizon is an external data-generation
  boundary, not a natural terminal state.
- At a truncated horizon, the target is the complete sequence of discounted
  simulator rewards plus the configured final value bootstrap.
- Only states inside the configured replay sampling window are emitted.

Inherited fallback search values are `gamma=0.995` and
`discount_step_sec=0.015725797204323228`. An experiment manifest may override
them, but self-play, native search, training-target construction, and evaluation
must use the same resolved values.

## 12. Markov-Sufficient DNN Representation

The DNN representation is derived from the complete snapshot. It must not
become the source of truth or discard variables that alter legal actions,
transition times, rewards, or future arrivals.

### 12.1 Global feature group

- Player-to-act and whether that turn is forced.
- Time until next adversary tick.
- Launch-window remaining count and prefill budget.
- Pooled available and reserved decode credits, plus cumulative mint/spend
  information needed to verify the budget invariant.
- Current objective and counts near each SLO/drop boundary.
- Number of replicas and topology/profile embeddings.
- Number and age of unassigned requests when routing is enabled.

Absolute simulator time may be omitted from the DNN only if every dependency is
encoded as a relative duration. It remains exact in the snapshot.

### 12.2 Request feature rows

Every active request row includes:

- Presence mask and stable tie-order feature.
- Lifecycle and owner replica.
- Original, committed, reserved, in-flight, and remaining tokens.
- Prefill/decode phase indicators.
- Age, relative deadlines, slack, lateness, and violation state.
- Resident KV blocks by stage, partial-block fill, and distance to a block
  boundary.
- In-flight/pending-stop/pending-drop indicators.
- Remaining stage latency and final completion latency if in-flight.
- Terminal stop/drop reason and pending physical-release state.

Completed, stopped, and dropped requests are removed once they have no active
KV, in-flight, decode-budget, or cost effect. Historical completion counters are not
features unless they affect a live rule.

### 12.3 Replica feature rows

Every replica row includes:

- TP/PP topology and profile identity.
- Waiting prefill/decode counts and token totals.
- Earliest deadline/slack summaries.
- In-flight count and admission capacity.
- Time to next controller admission.
- Free, allocated, reserved, and pending-release KV blocks.
- Partial-block waste and boundary-crossing decode count.
- Predicted service summaries for candidate canonical batch shapes.

### 12.4 Stage feature rows

Every PP-stage row includes:

- Replica and stage identity.
- Busy flag and active microbatch type.
- Time until stage tail is free.
- Queue occupancy and capacity.
- Active batch prefill/decode shape.
- Time until next final completion for dependent work.
- Free/reserved KV summaries for that stage's ranks.

### 12.5 Variable-size encoding

GV4 uses a hierarchical masked set encoder:

1. Encode request rows and aggregate with masked sum and max.
2. Encode stage rows within each replica.
3. Fuse request, stage, KV, and replica summaries into a replica embedding.
4. Aggregate replica embeddings for the system value.
5. Reuse the state embedding while scoring dynamic action-feature rows.

If a fixed row cap is required for batching, overflow must be represented by
explicit count, token, deadline, lateness, KV, and in-flight aggregates.
Silently truncating requests or stages is forbidden.

### 12.6 Feature transforms

- Bounded quantities use `x / scale`.
- Nonnegative heavy-tailed quantities use `asinh(x / scale)`.
- Signed heavy-tailed quantities use signed `asinh(x / scale)`.
- Booleans and masks remain explicit.
- Bucketization is not used unless the bucket schema is versioned and parity
  tested.
- Python and native use the same operation order, scales, dtype, and clipping.

The existing `legacy_226` schema is not Markov-sufficient for finite KV and
PP. GV4 uses `gv4_markov_v3`; loading a legacy feature model must fail
closed.

## 13. DNN and Policy Baseline

The current DNN models are the implementation starting point, not a reason to
retain the old feature schema:

- Markov value encoder: global width 64, request width 64, launch width 32,
  masked sum/max aggregation, 192-wide fused trunk, two 64-wide bottleneck
  residual blocks, and a 32-wide value head.
- Markov policy encoder: global width 32, request width 32, launch width 16,
  192-wide state embedding, 64-wide action embedding, 128-wide fused trunk,
  one 32-wide bottleneck residual block, and a 64-wide policy head.
- Value output range: `[-50, 0]` through the inherited tanh transform.

GV4 adds replica and stage encoders and a dynamic action-feature contract.
Controller and adversary retain separate policy models. Value-model ownership
must be explicit in the experiment manifest and identical in Python/native
search.

## 14. Snapshot, Restore, and Native Layout

MCTS requires cheap exact branching. The initial native state uses contiguous,
bounded arrays and ring buffers rather than object graphs:

```text
GV4State {
    config_and_schema_version
    now, player, next_adversary_tick
    launch_history, decode_credit_ledger
    requests[max_requests]
    replicas[max_replicas]
    stages[max_replicas][max_pp]
    kv[max_replicas][max_pp][max_tp]
    inflight[max_replicas][max_inflight]
    objective_and_cost_ledger
    deterministic_counters_and_rng
}
```

The first correct implementation may copy this fixed-size state per branch.
After parity is established, apply/undo may replace copying:

- Every mutation writes an undo record before changing state.
- Undo restores arrays, counters, ledgers, tail times, and RNG exactly.
- A snapshot hash before apply must equal the hash after undo.
- No pointer in a snapshot may refer to mutable state outside the snapshot.

The compact calendar advances in `O(PP)` per admitted microbatch plus the
bounded number of completions crossed. It does not scan all historical events.

## 15. Global Load Balancer Design

The router must not use a combinatorial assignment such as choosing all
replica/request pairings at once. It handles one unassigned request per
zero-time subdecision:

```text
route(request_i) -> replica_0 | ... | replica_(R-1) | HOLD
```

Thus 40 queued requests and 5 replicas do not create `40 choose 5` actions.
They create at most six actions for the current request, followed by the next
request. The state includes all remaining unassigned requests so the value
function can reason about future assignments.

`HOLD` is legal only when a deterministic future boundary can improve
feasibility or when every replica assignment is masked. Zero-time routing must
strictly reduce the unassigned work for that routing cycle; otherwise time
advances.

Initial router training freezes a validated shared replica scheduler. Final
fine-tuning may alternate router and scheduler updates under the same
system-level cost.

## 16. Rules and Invariants

Every transition must assert the following in debug/parity builds:

- Each request has at most one owning replica.
- Each request belongs to at most one in-flight microbatch.
- In-flight requests cannot be terminally evicted or admitted again.
- Committed plus reserved plus remaining accounting matches original work for
  every active request.
- No token is committed before final PP-stage completion.
- No final completion is applied twice.
- Per-rank KV ownership classes sum to physical capacity.
- No block is both free and referenced.
- Every logical block reserved by a request exists on all required ranks.
- Reserved KV and issued decode credits cannot be double spent; prefill has no
  credit reservation.
- Stage tail times are nondecreasing.
- A stage never runs two microbatches simultaneously.
- FIFO stage ordering is preserved.
- In-flight count never exceeds capacity.
- Inter-stage occupancy never exceeds capacity.
- Player alternation is preserved after auto-resolving forced turns.
- A same-time transition either changes state monotonically or is rejected.
- Cost and reward telescope exactly across split versus combined time advances.
- Python and native canonical actions, next times, rewards, and resulting
  snapshot hashes match.

## 17. Hyperparameter Baseline

These are inherited defaults from the copied GV3/AlphaGoZero code. They are
starting values, not hidden constants. The resolved manifest is authoritative.

### 17.1 System and execution defaults

| Parameter | Baseline |
|---|---:|
| Model | `meta-llama/Meta-Llama-3-8B` |
| Device / network | `a100` / `a100_dgx` |
| Replicas / TP / PP | 1 / 1 / 1 |
| Scheduler batch-token cap | 512 |
| Execution predictor max batch size | 256 |
| Maximum prefill chunk | 4096 |
| Profile lookup interval | 128 tokens |
| Prefill slowdown bridge | 3.0 |
| Decode SLO bridge | 50 ms |
| Predictor threads | 1 |
| KV block size | 16 tokens |
| KV memory safety margin | 0.10 |
| Allocator watermark | 0.01 |
| Block preallocation granularity | 64 |
| Prefix caching / disk offload | disabled / disabled |

GV4 must add explicit weight dtype, KV dtype, usable bytes per rank,
`max_inflight_microbatches`, and `inter_stage_queue_capacity`.

### 17.2 Game defaults

| Parameter | Baseline |
|---|---:|
| Adversary tick | 0.2 s |
| Launch window | 1.0 s |
| Max requests per window | 7 |
| Max prefill / decode per request | 4096 / 864 |
| Min decode per request | 1 |
| Prefill launch-window token cap | 7168 per 1.0 s window |
| Decode credits per prefill completion | 216, pooled and non-expiring |
| Violation base / lateness cap / drop cost | 1 / 2.0 s / 3 |
| Automatic drop lateness | 2.0 s |
| Numeric epsilon / round digits | `1e-9` / 10 |
| Global seed | 6 |
| Deterministic Torch kernels | false |

### 17.3 Search defaults

| Parameter | Baseline |
|---|---:|
| Native search mode | `full_tree` |
| Self-play / evaluation simulations | 1000 / 1000 |
| Self-play / evaluation PUCT constant | 2.5 / 1.0 |
| Policy-prior temperature | 1.0 |
| Root Dirichlet alpha | 0.05 |
| Root Dirichlet total concentration | 0.0 |
| Root Dirichlet epsilon | 0.25 |
| Initial sampled moves | 20 |
| MCTS action temperature | 1.0 |
| Discount factor | 0.995 |
| Discount reference step | 0.015725797204323228 s |
| Rollouts per leaf | 10 |
| Rollout horizon | 0.4 s |
| Rollout policy temperature | 1.0 |
| Rollout threads | 1 |
| Probability quantum | `1e-6` |
| Maximum rollout actions | 4096 |

The lower-level copied GV3 search config still contains fallback Dirichlet
`alpha=0.6`, `epsilon=0.30`, `pb_c_base=1500`, and
`pb_c_init=1.25`. GV4 must resolve values once in the experiment manifest;
lower-level fallbacks cannot override AlphaGoZero settings.

### 17.4 DNN and training defaults

| Parameter | Baseline |
|---|---:|
| DNN epochs per update | 5 |
| Value batch size | 4096 |
| Policy root batch size | 256 |
| Initial / update learning rate | `3e-4` / `1e-4` |
| Weight decay | `1e-4` |
| Gradient clip norm | 5.0 |
| Torch threads per model | 24 |
| New-state training trigger | 600,000 |
| Min controller / adversary states for evaluation | 250,000 / 25,000 |
| Controller / adversary policy sample cap | 250,000 / 250,000 |
| Maximum adversary value states | 300,000 |
| Policy root oversample factor | 1.25 |
| Total / controller / adversary replay cap | 35M / 20M / 15M |
| Evaluation / benchmark games | 100 / 50 |
| Promotion win-rate threshold | 0.55 |
| Maximum promotions | 50 |
| Policy cache / metrics workers | 60 / 80 |
| Replay index / extraction workers | 60 / 16 |

### 17.5 Inherited feature scales

Existing scales may initialize GV4 where the meaning is unchanged:

- Prefill total/remaining: 4096.
- Decode total/remaining/processed: 864.
- Age: 5 s.
- Lateness/slack: 2 s.
- Prefill/decode SLO: 2 s / 0.2 s.
- Objective and total lateness: 50.
- System load: 120 requests.
- Recent launch count/prefill: 7 / 7168.
- Decode credit: 21,600.
- Near-drop bands: 0.5 s and 1.5 s.
- Launch EWMA alpha/window: 0.37 / 1 s.

New KV, replica, stage, in-flight, and routing scales must be explicit and
serialized. They cannot be inferred differently in Python and native code.

## 18. Determinism and Logging

Every controller/adversary trace row must include:

- Schema/config/profile/model checksums.
- Simulator time, player, and boundary reason.
- Raw and canonical action IDs.
- Resolved terminal eviction targets and request-token allocations.
- Per-rank KV before, reserved/released delta, and after.
- Stage start/finish calendar for an admitted microbatch.
- Internal completions crossed before the decision.
- Objective before/after, immediate reward, elapsed time, and discount.
- Legal-action count, priors, visits, Q, normalized Q, exploration, and PUCT
  when MCTS logging is enabled.
- Snapshot hash before and after the transition.

Stable request IDs, sorted iteration, fixed tie-breaks, common probability
quantization, and explicit RNG state are mandatory for Python/native parity.

## 19. Validation and Acceptance Tests

### 19.1 Configuration tests

- Reject invalid rank counts, layer partitions, and profile/topology mismatch.
- Reject action/model/schema dimension mismatch.
- Reject profile checksum mismatch.
- Reject inter-stage capacity below max in-flight for the initial calendar.
- Verify every derived action index round-trips to its three components.

### 19.2 KV tests

- Allocate, reserve, commit, release, and terminally evict at exact block boundaries.
- Decode inside a partial block while global free blocks are zero.
- Mask a boundary-crossing decode at zero free blocks.
- Evict one and multiple targets and verify terminal cost and immediate KV release.
- Stop/drop an in-flight request and delay physical release until completion.
- Verify every rank's accounting invariant after every operation.

### 19.3 PP timing tests

- TP1/PP1 parity with the existing indivisible GV3 execution path.
- TP1/PP2, TP2/PP1, and TP2/PP2 stage timing.
- 128-token batch queued behind a 4096-token batch.
- Pipeline fill, steady overlap, drain, bubble, and downstream stall traces.
- Final completion exactly equal to an adversary tick.
- Stage 0 becoming free while the in-flight cap remains full.
- Compact calendar versus a reference event simulator on randomized legal
  traces, with identical completion order and final state.

### 19.4 Turn and reward tests

- Controller admission before adversary tick with forced adversary no-op.
- Adversary tick before admission with forced controller no-op.
- Equal clocks with completion, drop, adversary, then controller ordering.
- `BATCH`, `EVICT_ONLY`, and `WAIT` time semantics.
- No same-time infinite loop.
- Split transition rewards equal combined transition reward after discounting.
- Truncated-horizon targets contain all intervening rewards plus one bootstrap.

### 19.5 Action and model tests

- Exhaustively compare Python/native legal raw and canonical actions.
- Verify alias priors sum and visits map back to the canonical edge.
- Add a rule/budget/heuristic in a test config and prove all dimensions update.
- Verify GV4 feature tensors match between Python and native CSV exports.
- Permute request/replica rows and verify permutation-invariant value output.
- Reject legacy/HGB artifacts and wrong feature schemas.

### 19.6 Snapshot tests

- Snapshot/restore at every request lifecycle state.
- Snapshot while multiple microbatches occupy different PP stages.
- Apply/undo restores byte-identical state and snapshot hash.
- Branches from the same parent cannot mutate one another.

### 19.7 Policy-quality tests

- Tiny exhaustive environments with a known optimal policy.
- Baselines: SJF 128/256/512, EDF, LST, no eviction, and largest-request
  terminal eviction.
- Stress traces at 0%, 50%, 90%, and 100% KV occupancy.
- Generalization across legal TP/PP and memory configurations represented in
  training.
- Shadow parity against real vLLM for batch shape, completion
  ordering, KV occupancy, and measured stage/batch timing.

GV4 is not ready for AlphaGoZero training until configuration, KV, PP calendar,
turn ordering, snapshot, and Python/native parity suites pass.

## 20. Implementation Sequence

1. Freeze this contract and assign `gv4_state_v3`, `gv4_actions_v1`,
   `gv4_transition_v3`, and `gv4_native_layout_v3` version IDs.
2. Add resolved topology/profile/KV configuration and validation.
3. Implement finite-KV request lifecycle for TP1/PP1.
4. Add dynamic action specification and canonicalization.
5. Implement the compact PP stage calendar and internal completion drain.
6. Add complete snapshot/restore and Python/native parity.
7. Implement `gv4_markov_v3` features and DNN export.
8. Run exhaustive and baseline tests for one replica.
9. Add sequential global routing and shared replica encoding.
10. Validate against real vLLM before large-scale self-play.

## 21. Explicitly Deferred Features

The following are not silently approximated in `gv4_transition_v3`:

- Prefix caching or shared KV blocks.
- CPU/disk KV swap or offload.
- Request migration between replicas.
- Resuming or reconstructing a request after a terminal controller eviction.
- Non-FIFO PP stage scheduling.
- Inter-stage queue backpressure below the max-in-flight capacity.
- Heterogeneous model/GPU replicas in one routing game.
- Disaggregated prefill/decode serving.

Adding any of these requires updating this document, the state/action/transition
versions, parity fixtures, and replay/model compatibility checks first.
