# GV3 Distributed Network Configuration

This note describes how the current GV3 distributed Python experiment distributes
self-play sample generation across machines and worker processes.

The concrete run discussed here is:

```text
output_name = Game_Version3_Fresh8_Hops200
total_roots = 350000
sample_cycles_per_generation = 8
history_hops_min = 0
history_hops_max = 200
history_hop_interval_width = 5
worker_processes = 10
max_concurrent_workers = 67
max_workers_per_interval = 14
selfplay_dynamic_chunk_roots = 64
worker_cpu_fraction = 0.70
worker_model_device = cpu
local_training_device = cuda
environment_lang = python
```

## Coordinator Role

The local machine is the coordinator and trainer. For each generation it:

1. Copies the current checkpoint into `selfplay_weights_gen_XXXXXX.pt`.
2. Splits the generation into collection cycles.
3. Builds one top-level task per remote machine for each cycle.
4. Sends each task to a remote machine with `task.json` and `weights.pt`.
5. Starts the remote client with SSH.
6. Rsyncs remote results back into `network/received`.
7. Mirrors received replay shards into `mcts_dnn_dataset`.
8. Trains the value model locally on GPU after all cycles finish.

Relevant files:

```text
Network/server/orchestrator.py
Network/client/run_task.py
Network/common/executor.py
Network/common/types.py
Network/server/training.py
```

## Cycle-Level Root Split

The current run asks for:

```text
350000 roots / generation
8 cycles / generation
```

So each collection cycle generates:

```text
350000 / 8 = 43750 roots / cycle
```

There are 8 selected machines, so each cycle gives each machine about:

```text
43750 / 8 = 5468 or 5469 roots / machine / cycle
```

The task ids include generation, cycle, machine, and machine index. Example:

```text
gv3_fresh_8machines_hops200_v2_gen_000030_c007_aws-gv3-machine3_002
```

This means:

```text
generation = 30
cycle = 7   # zero-indexed, so cycle 8/8
machine = aws-gv3-machine3
machine index = 2
```

## Machine-Level Hop Split

The global history hop range is:

```text
0..200
```

The coordinator partitions this inclusive range across the 8 machines. With the
current setup, each machine gets a broad slice like:

```text
machine 0: 0-25
machine 1: 26-50
machine 2: 51-75
machine 3: 76-100
machine 4: 101-125
machine 5: 126-150
machine 6: 151-175
machine 7: 176-200
```

So a remote machine is not sampling all depths `0..200` in a task. It receives
only its assigned slice for that cycle.

This split is done in:

```text
Network/server/orchestrator.py
```

by `_partition_inclusive_range(...)`.

## Remote Task Contents

Each remote machine receives a `task.json` plus `weights.pt`.

The task tells the machine:

```text
session_id
task_id
generation
cycle_index
model_version
num_roots
start_root_id
history_hops_min
history_hops_max
history_hop_interval_width
worker_processes
max_concurrent_workers
max_workers_per_interval
selfplay_dynamic_chunk_roots
out_dir_train
out_dir_eval
logs_dir
weights_path
environment_lang
model_device
```

The remote client runs:

```bash
python -m vidur.mcts.Game_Versions.Game_Version3.Network.client.run_task --task task.json
```

## Worker Process Split On One Remote Machine

Inside a remote machine, "workers" means local Python worker processes, not more
machines.

The remote executor first converts the machine's broad hop slice into 5-hop
intervals using:

```text
history_hop_interval_width = 5
```

Example for a machine assigned `26..50`:

```text
26-30
31-35
36-40
41-45
46-50
```

That is 5 interval states.

The config says:

```text
worker_processes = 10
```

but this is capped by the number of intervals inside the machine's assigned hop
slice. Therefore, for most machines, the effective interval count is 5, not 10.

The first machine gets `0..25`, which is 26 hop values, so it has 6 intervals:

```text
0-4
5-9
10-14
15-19
20-24
25-25
```

This logic is in:

```text
Network/common/executor.py
```

through `_resolve_worker_processes(...)` and `_worker_hop_ranges(...)`.

## Roots Per Interval

The remote task has about `5469` roots.

For a 5-interval machine:

```text
5469 / 5 ~= 1093 roots / interval
```

For the first 6-interval machine:

```text
5469 / 6 ~= 911 roots / interval
```

The dynamic scheduler tracks each interval's target roots and launches smaller
chunks until each interval reaches its target unique frontier roots.

## Dynamic Chunk Scheduling

The scheduler does not launch one long process per interval. Instead it launches
shorter worker chunks.

Current chunk size:

```text
selfplay_dynamic_chunk_roots = 64
```

So one spawned process may receive:

```text
num_roots = 64
history_hops_min = 76
history_hops_max = 80
```

That means:

```text
Generate up to 64 frontier roots whose random history depth is sampled from 76..80.
```

The scheduler keeps launching chunks until the interval reaches its target
number of unique roots.

## Concurrency Limits

The configured limit is:

```text
max_concurrent_workers = 67
```

But the effective cap is also constrained by:

```text
num_intervals * max_workers_per_interval
```

With:

```text
max_workers_per_interval = 7
```

most machines have:

```text
5 intervals * 7 = 35 active worker processes max
```

The first machine has:

```text
6 intervals * 7 = 42 active worker processes max
```

So `67` is an upper bound, but the current interval structure normally limits
the process count to about 35 or 42 per remote machine.

The scheduler also checks RSS memory pressure:

```text
selfplay_launch_rss_limit_gb = 120
```

If active workers exceed the memory threshold, new launches pause until memory
falls.

## How One Worker Chunk Generates Roots

Each worker process runs:

```text
DNN/selfPlay.py
  SelfPlayRunner.run_n_roots(...)
```

The worker receives a small hop interval and a target number of roots. With the
current dynamic scheduler, a typical chunk is:

```text
num_roots = 128
history_hops_min = 76
history_hops_max = 80
```

This means the worker should emit up to 128 frontier root samples whose history
depth is inside `76..80`.

### Base State

For the current network path, no custom `initial_state` is passed into
`run_n_roots(...)`. Therefore history generation starts from:

```text
VirtualVidurMCTSEnvironment.initial_state()
```

That restores the environment's base simulator snapshot. In this experiment,
that base snapshot is the empty-system state before any adversary requests have
been injected.

So all generated frontier roots, across all hop intervals, are descendants of
the same empty initial state. The hop interval only controls how many
nontrivial history decisions away from that initial state a frontier root should
be.

### Anchor Node

The history generator does not restart from the empty state for every emitted
root. It first builds an anchor node at the minimum hop for the interval.

For an interval like:

```text
10..15
```

the generator calls:

```text
HistoryRootGenerator._roll_to_target_hops(target_hops=10)
```

This produces:

```text
empty initial state
  -> random nontrivial action
  -> random nontrivial action
  -> ...
  -> anchor at history_hops = 10
```

Forced single-child steps are applied along the way, but they do not increment
`history_hops`. A hop is counted only when the current state has multiple valid
actions and the generator chooses one of them.

After reaching the target hop count, the generator advances through any forced
single-child chain so the anchor is positioned at a legal branching decision
state when possible.

### Frontier Nodes

A frontier node is a candidate root sample. It stores:

```text
state snapshot
stats snapshot
player to act
tree depth
history_hops
history trace metadata
```

When the worker emits a frontier node, that frontier state becomes one training
or eval root sample. The model target is then computed at that frontier by
`mctsDNN.py`.

### Expanding From The Anchor

After creating the anchor, the generator prepares a shuffled list of untried
actions from that anchor.

An anchor or frontier node is expandable only if:

```text
node.history_hops < history_hops_max
len(valid_actions) > 1
```

The generator then uses a randomized depth-first expansion:

```text
anchor
  -> child from one shuffled action
      -> grandchild from one shuffled action
  -> child from another shuffled action
  -> ...
```

Each selected child action increments `history_hops` by 1, because expansion is
only prepared for nodes with multiple valid actions. After applying the child
action, the generator again advances through forced single-child steps to reach
the next branching frontier.

This means a 128-root chunk usually shares some common prefix from the empty
state to an anchor, then emits many unique roots by expanding different branches
below that anchor. If the active branch path is exhausted, the generator creates
another anchor and continues.

### Uniqueness Criterion

Each candidate frontier node is checked against a history signature before it
is emitted.

The signature is computed from:

```text
player_to_act
history_hops
sim_time rounded to 9 digits
pending_adv_tick
decode_credit_balance
active_request_ids
completed_request_ids
```

If the signature has already been seen inside that interval's shared set, the
candidate is skipped and the generator continues expanding. If the signature is
new, the frontier node is emitted as a root sample.

The shared uniqueness set is interval-scoped on a remote machine. Multiple
worker chunks launched for the same hop interval share this set, so they avoid
counting the same frontier twice.

### Bellman Target At The Frontier

After a frontier node is emitted, `SelfPlayRunner.run_n_roots(...)` computes the
GV3 Bellman target for that root:

```text
controller root:
  evaluate valid controller actions
  select max Q, because Q is controller-valued negative cost

adversary root:
  evaluate adversary action
  evaluate controller response after that adversary action
  controller selects max Q response
  adversary selects min Q over adversary actions
```

The root sample stores the frontier features, the selected value target, player
type, root metadata, and history signature. It is then written into either the
train or eval replay shard.

Relevant files:

```text
DNN/history_root.py
DNN/selfPlay.py
mctsDNN.py
multiProcessUtils.py
```

## Frontier Root Uniqueness

The current implementation tries to generate unique frontier roots inside each
remote machine task.

The main uniqueness mechanism is based on history signatures. For each interval
on a remote machine, the dynamic scheduler creates a shared dictionary:

```text
shared_seen_proxy = manager.dict()
```

All worker chunks launched for that interval share this dictionary. When a worker
generates a random history frontier, the history generator checks whether that
frontier signature has already been emitted. If it has been seen, it tries to
generate a different frontier state.

The worker returns emitted signatures in:

```text
run_stats["history_signatures"]
```

The interval scheduler uses those signatures to update:

```text
completed_unique_roots
completed_history_signatures
```

This means the scheduler counts unique generated roots, not just requested roots.
If a chunk produces no new unique signatures, the interval gets a
`zero_progress_completions` increment. If this happens repeatedly and there are
no active workers left for that interval, the interval can be marked exhausted.

Relevant code:

```text
multiProcessUtils.py
  _run_selfplay_cycle(...)
  _make_selfplay_chunk_task(...)

DNN/selfPlay.py
  _iter_prepared_history_root_batches(...)
  run_n_roots(...)
```

The uniqueness scope is important:

```text
Strongest uniqueness scope: one interval on one remote machine
Weaker scope: different intervals on the same machine
Weakest scope: different machines
```

So uniqueness is not currently enforced by one global cluster-wide set. That is
acceptable for the current setup because machines receive disjoint hop slices:

```text
machine 0: 0-25
machine 1: 26-50
...
machine 7: 176-200
```

This makes cross-machine duplicates unlikely, though not impossible. Different
hop depths can theoretically converge to equivalent simulator states.

The practical signal to monitor is:

```text
unique_roots / roots_generated
```

Recent runs show this is very high, for example:

```text
unique_roots = 43747 / 43750
```

That indicates the current duplicate pressure is low.

Increasing:

```text
max_workers_per_interval: 7 -> 14
```

will increase concurrent workers racing inside the same interval. The shared
seen-signature mechanism should handle this, but duplicate pressure may rise
slightly. If `unique_roots / roots_generated` starts dropping, the first knobs to
adjust are:

```text
reduce max_workers_per_interval
reduce selfplay_dynamic_chunk_roots from 64 to 32
add a machine-wide shared seen set across intervals
add a coordinator-level global seen set across machines
```

## Train/Eval Split

Each root sample is independently assigned to train or eval based on:

```text
eval_split_ratio
```

In the current observed runs this gives roughly 90% train and 10% eval.

The resulting remote directory has:

```text
network/results/<session>/<task_id>/train/proc_.../replay_000000.pt
network/results/<session>/<task_id>/eval/proc_.../replay_000000.pt
```

After rsync back to the local machine, the result is stored under:

```text
network/received/<session>/<task_id>/
```

Then it is mirrored into the canonical dataset:

```text
mcts_dnn_dataset/gen_XXXXXX/train/proc_net_.../replay_000000.pt
mcts_dnn_dataset/gen_XXXXXX/eval/proc_net_.../replay_000000.pt
```

## Training Phase

After all machines finish all 8 cycles, the local machine trains on GPU:

```text
local_training_device = cuda
train_batch_size = 256
train_target_epochs = 20
```

The local trainer uses replay shards from:

```text
mcts_dnn_dataset/gen_XXXXXX/train
```

and evaluates using:

```text
mcts_dnn_dataset/gen_XXXXXX/eval
```

Metrics are written to:

```text
mcts_dnn_logs/eval_metrics.csv
```

Checkpoints are written to:

```text
mcts_dnn_checkpoints/
```

## Practical Consequence Of Current Settings

The current settings emphasize breadth across history depth:

```text
0..200 history hops
5-hop intervals
8 machines
dynamic 128-root chunks
```

This gives good coverage across many frontier depths while keeping multiple
processes active per depth interval.

However, because the server first splits the global hop range across machines,
each machine only sees a slice of the global range. The 5-hop interval scheduler
then operates inside that slice.

So the actual hierarchy is:

```text
generation
  -> 8 collection cycles
    -> 8 machine tasks per cycle
      -> about 5 or 6 hop intervals per machine
        -> many 128-root worker chunks per interval
          -> individual frontier root samples
```
