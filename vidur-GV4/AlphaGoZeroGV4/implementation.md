# GV4 AlphaGoZero Implementation Plan

## 1. Objective

AlphaGoZeroGV4 owns the orchestration that is specific to the GV4 game. It must
run the same single-replica game through Python or native MCTS, generate replay,
train four role models, evaluate candidates, and distribute promoted bundles to
many self-play workers.

The first production scope is intentionally narrow:

- One replica per game.
- TP and PP are supported inside that replica, initially TP=2 and PP=2.
- A worker may run many independent game processes.
- Many workers may run against one XL coordinator.
- Native MCTS is the production path; Python MCTS is the correctness reference.
- DNN value and policy models are the initial model family.
- Ordinary leaf-bootstrap MCTS comes before optional policy rollouts.

Single replica does not mean single process. Every independent game has one
replica, while a worker can run many games and the cluster can contain many
workers.

GV3 must remain runnable during migration. GV4 must never silently fall back to
the GV3 engine, GV3 features, GV3 arena runner, or GV3 model artifacts.

## 2. Minimal architecture

    AlphaGoZeroGV4/deploy.py
      -> starts one XL coordinator and one daemon on every self-play worker

    AlphaGoZeroGV4/worker_daemon.py
      -> snapshots the promoted model manifest
      -> launches N isolated gv4_arena_runner.py subprocesses
      -> merges completed replay into durable shards
      -> uploads immutable shards and waits for acknowledgement

    AlphaGoZeroGV4/gv4_arena_runner.py
      -> creates the validated GV4 environment and timing provider
      -> creates a seeded random-history root
      -> advances forced single-action states
      -> calls Python or native MCTS only at a real branching decision
      -> applies the chosen action and records replay plus diagnostics

    AlphaGoZeroGV4/replay_runtime.py
      -> records meaningful decisions for one game
      -> composes rewards and discounts through hidden forced transitions
      -> emits state/value and canonical policy records

    AlphaGoZero/durable_transfer.py
      -> checksums, publishes, transfers, acknowledges, and retires shards

    AlphaGoZeroGV4/xl_coordinator.py
      -> validates and ingests GV4 shards
      -> owns bounded replay and the training gate
      -> launches at most one trainer/evaluator/promoter cycle

    AlphaGoZeroGV4/agz_train_eval_promote.py
      -> samples GV4 replay
      -> trains and exports four GV4 artifacts
      -> evaluates the two roles independently
      -> publishes and distributes the promoted bundle

The four artifacts remain:

| Artifact | Input | Output |
| --- | --- | --- |
| Controller value | GV4 state features | Controller-perspective cost-to-go |
| Adversary value | GV4 state features | Controller-perspective cost-to-go |
| Controller policy | State plus canonical controller-action features | One logit per canonical action |
| Adversary policy | State plus canonical adversary-action features | One logit per canonical action |

Both value models predict the same controller-perspective target. MCTS owns the
player-dependent minimization/maximization convention.

## 3. Files that are intentionally unnecessary

### Do not create AlphaGoZeroGV4/gv4_backends.py

That file would duplicate interfaces that already exist:

- Python environment: GV4_Engine/virtual_environment.py.
- Python MCTS: vidur-GV4/mcts_value_prior.py.
- Native config/environment adapter: GV4_Cpp/runtime.py.
- Native search: the GV4_Cpp pybind module.

Keep two short private search functions in gv4_arena_runner.py, one Python and
one native. Make both return the same small arena result: selected canonical
action, root visits, root value, and diagnostics. Extract a backend module only
if this dispatch later becomes substantial or a third backend is added.

GV4_Cpp/runner.py is not the production arena. It is a uniform-MCTS parity and
trace-validation program. The production arena should reuse GV4_Cpp/runtime.py
and call the pybind search API directly.

### Do not create GV4_Engine/decision_boundary.py

Decision-boundary normalization is an arena/replay presentation rule, not a new
transition rule. Keep one tested helper such as
_advance_to_branching_state() in gv4_arena_runner.py.

The helper repeatedly applies the only canonical legal action until:

- More than one canonical action is legal, so MCTS is required.
- The game reaches its horizon.
- The state becomes terminal.
- A nonterminal state has no legal action, which is an invariant failure.
- The zero-time transition safety limit is reached, which is also a failure.

Count canonical actions, not raw aliases. Several raw indices that resolve to
one effect still represent one forced choice.

Forced actions are omitted from the main meaningful-decision rows but not from
the mathematics. For edge pairs (reward, discount):

    composed_reward = r0 + d0*r1 + d0*d1*r2 + ...
    composed_discount = d0*d1*d2*...

Store every hidden action and timestamp in forced_chain_json. This preserves
exact value targets and makes missed adversary ticks, decode fast-forward, and
automatic completions auditable.

Run this helper before the first MCTS call, after every random-history choice,
and after each played action. MCTS may still traverse forced nodes internally;
the rule only controls played-game rows and replay targets.

## 4. File migration map

### Create or adapt

| GV4 destination | GV3 source | Purpose |
| --- | --- | --- |
| AlphaGoZeroGV4/__init__.py | New | Define the package with no import-time work. |
| AlphaGoZeroGV4/config.py | AlphaGoZero/config.py | Own high-level self-play, MCTS, replay, training, evaluation, process, and deployment settings. |
| AlphaGoZeroGV4/gv4_arena_runner.py | New | Replace cpp_selfplay_runner.py and the GV3 arena bridge with one GV4 game loop. |
| GV4_Engine/history_root.py | Game_Version3/DNN/history_root.py algorithm only | Create reproducible random legal GV4 roots without knowing about models or workers. |
| AlphaGoZeroGV4/replay_runtime.py | AlphaGoZero/replay_runtime.py | Replace GV3 rows with GV4 structured state/action replay and composed edges. |
| AlphaGoZeroGV4/dnn_models.py | AlphaGoZero/dnn_models.py | Define, train, checkpoint, and export the four GV4 DNN artifacts. |
| AlphaGoZeroGV4/indexed_replay.py | AlphaGoZero/indexed_replay.py | Index and sample variable-size GV4 state/action examples with strict schema checks. |
| AlphaGoZeroGV4/worker_daemon.py | AlphaGoZero/worker_daemon.py | Supervise many isolated game processes and durable worker shards. |
| AlphaGoZeroGV4/xl_coordinator.py | AlphaGoZero/xl_coordinator.py | Ingest GV4 replay, bound it, gate training, and own trainer state. |
| AlphaGoZeroGV4/agz_train_eval_promote.py | AlphaGoZero/agz_train_eval_promote.py | Train four DNNs, evaluate roles, promote, and publish bundles. |
| AlphaGoZeroGV4/bootstrap_untrained_dnn_v100.py | AlphaGoZero/bootstrap_untrained_dnn_v100.py | Create a neutral schema-valid native-ready initial bundle. |
| AlphaGoZeroGV4/distributed_eval.py | AlphaGoZero/distributed_eval.py | Later distribute paired GV4 evaluations over workers. |
| AlphaGoZeroGV4/deploy.py | AlphaGoZero/deploy.py | Later build/sync GV4 and launch GV4 daemons. |

### Reuse without copying initially

| Existing file | Reason |
| --- | --- |
| AlphaGoZero/durable_transfer.py | Checksums, atomic rename, acknowledgement, and safe deletion are game-independent and already tested. |
| AlphaGoZero/cluster.py | Host definitions and SSH/rsync helpers are infrastructure, not GV3 semantics. |

This avoids two copies of reliability code. If GV3 is retired, move these files
once into a shared infrastructure package rather than maintaining divergent
copies.

### Defer or omit

| GV3 file or branch | Decision |
| --- | --- |
| cpp_selfplay_runner.py | Do not copy; gv4_arena_runner.py is both local CLI and worker entry point. |
| markov_value_features.py | Do not copy; GV4_Engine/dnn_inference/dnn_features.py is the feature source of truth. |
| replay_target_smoke.py | Do not copy; replay comes directly from the game recorder. |
| adaptive_rollout.py | Defer until ordinary leaf-bootstrap MCTS is correct end to end. |
| spot_work_protocol.py and all spot worker/eval files | Defer until fixed workers are stable. |
| bootstrap_markov_v100.py | Omit from the initial DNN-only path. |
| distill_dnn_v100.py | Optional experiment, not core pipeline code. |
| HGB and GV3 compatibility branches in the trainer | Remove from the GV4 copy instead of retaining dead paths. |
| prebuild_replay_caches.py | Optional after indexed replay is proven. |
| GV3 analysis/tests | Do not bulk-copy; write focused GV4 contract tests. |

## 5. Detailed ownership by file

### AlphaGoZeroGV4/config.py

This is the high-level experiment gate, not a duplicate of
GV4_Engine/config.py.

It owns:

- Output roots, worker identity, process count, shard thresholds, disk limits,
  retry behavior, and resource limits.
- Self-play/evaluation MCTS iterations, PUCT, noise, temperature, horizon, and
  random-history distribution.
- Replay capacities, sample sizes, training gates, optimizer settings,
  promotion thresholds, and evaluation counts.
- Python/native backend selection for validation.
- Paths to the GV4 engine manifest, Vidur timing profile/cache, current model
  manifest, and output directories.
- State, action, feature, root-payload, replay, and model-export schema IDs.

It must not redefine topology, KV, scheduler, SLO, action, reward, request, or
timing semantics. Build or load one GV4EngineConfig, validate it, and record its
manifest hash.

Fail before launching a game if:

- num_replicas is not one in the initial implementation.
- Model feature dimensions differ from the engine feature contract.
- A model has the wrong role, schema, config hash, or checksum.
- Native self-play lacks any of the four native-ready artifacts.
- The Vidur profile does not match model, TP, PP, device, or network.
- Self-play points to mutable candidate files instead of promoted artifacts.

Use a separate root such as simulator_output/GV4_Agent/AlphaGoZero and a
GV4-specific environment prefix so GV3 shell variables cannot silently alter a
GV4 run.

### GV4_Engine/history_root.py

This is a small engine-level helper because Python tests, native parity,
self-play, and paired evaluation all need identical random roots.

Inputs:

- Validated environment/config.
- Initial state and player.
- Stable random seed.
- Requested range of meaningful history decisions.
- A forced-advance callback supplied by the arena.

Outputs:

- Root state and next player.
- Canonical history choices and aliases.
- Seed, requested/achieved depth, and final simulation time.
- Early terminal/horizon status or retry reason.

Choose uniformly over canonical actions, never raw aliases. Count only
branching choices toward history depth. Derive seeds from stable values such as
global seed, worker ID, and game ID; never use Python's randomized hash().

For strict backend parity, generate a versioned root-state payload in Python and
put Python/native payload conversion in GV4_Cpp/runtime.py. The payload must
contain time, player/tick, ID counters, launch history, decode credits,
objective, requests, rank KV ledgers, stage tails, and in-flight microbatches,
plus the payload schema and engine hash. Do not use pickle.

history_root.py should accept the arena's forced-advance callback. It must not
import AlphaGoZeroGV4 back from the engine package, which would create a circular
dependency.

### AlphaGoZeroGV4/gv4_arena_runner.py

This is the central integration point, but it remains orchestration rather than
a second simulator.

One game performs:

1. Parse one immutable game specification.
2. Validate engine, timing, features, replay, and model manifests.
3. Snapshot the four artifact paths and hashes for the entire game.
4. Construct Python environment and, if selected, native environment/inference.
5. Build or load the seeded history root.
6. Advance forced actions to a branch, terminal state, or horizon.
7. Call MCTS only when at least two canonical actions exist.
8. Select from root visits using self-play or deterministic evaluation rules.
9. Apply the selected canonical action through the selected backend.
10. Advance forced actions and compose reward/discount.
11. Record the root, visits, selected action, result, forced chain, and replay
    edge.
12. Repeat and atomically finalize the game outputs.

The arena must not canonicalize actions itself, compute KV/pipeline/SLO logic,
recalculate reward or discount, know DNN layers, mutate model files, or import a
GV3 fallback.

Use canonical visits as policy targets. Raw representative/alias indices remain
only for exact application and diagnostics. Evaluation disables noise and
sampling and selects the most-visited root action. All self-play randomness is
derived from the game seed.

The primary arena CSV has one row per meaningful MCTS decision. It should carry
game/step IDs, root payload hash, player/time, canonical actions, visits,
selected action, composed reward/discount, objective before/after, model
versions, history provenance, and forced_chain_json.

Detailed request, pipeline, and KV snapshots can use the existing GV4 logger in
debug mode. Keep them separate from replay and disable full MCTS-node logging in
production by default.

### AlphaGoZeroGV4/replay_runtime.py

Reuse the one-game trajectory and backward-target approach, not the GV3 schema.

Each decision retains:

- GV4 structured state features.
- Root player and simulation time.
- Canonical action features and identities.
- Root visits, valid canonical count, and selected action.
- Composed reward and discount to the next meaningful decision.
- Terminal/horizon status and explicit bootstrap kind/value.
- Engine, timing, feature, action, replay, and model schema metadata.
- Controller/adversary model versions used for the game.

Do not silently apply zero bootstrap at a nonterminal horizon. A configured
neutral-zero bootstrap is allowed, but bootstrap_kind and bootstrap_value must
make it explicit. Backward targets use stored composed discounts, not a fixed
number of arena rows.

Only states with at least two canonical actions receive policy targets. Write
temporary outputs and rename them only after a complete successful game.

### AlphaGoZeroGV4/dnn_models.py

Reuse checkpoint, optimizer, loss, and export organization while replacing all
GV3 dimensions and parsing.

GV4_Engine/dnn_inference/dnn_features.py remains the only feature builder.
dnn_models.py consumes those structured state/action tensors and masks; it must
not reconstruct request, pipeline, or KV features itself.

Required model contracts:

- Variable request counts use the documented mask/pooling path.
- Variable action counts produce one logit per canonical action.
- Controller/adversary policies consume their respective action feature type.
- Checkpoints record role, version, architecture, dimensions, schemas, config
  hash, training parent, and optimizer state.
- Native exports record checksums and reject incompatible loads before search.
- Python/native values and logits meet a declared numeric tolerance.

The arena constructs an inference runtime from a manifest and should not import
model classes or know layer sizes.

### AlphaGoZeroGV4/indexed_replay.py

Retain partition indexes, bounded sampling, and useful multiprocessing. Replace
GV3 Markov/226D parsing with GV4 structured state and canonical action parsing.

Every index entry identifies partition, row offset, role, model lineage,
feature/action/replay schemas, and engine config hash. Reject incompatible
partitions rather than padding unrelated schemas. State/value sampling and
policy sampling stay separate because one state can have many action rows.

### AlphaGoZeroGV4/worker_daemon.py

Keep process-per-game isolation. Do not share native or Torch runtime state
through fork.

The daemon:

1. Polls and validates models/current_model.json.
2. Allocates globally unique game IDs from worker ID plus a persistent counter.
3. Copies the immutable model manifest into each game directory.
4. Launches python -m vidur.AlphaGoZeroGV4.gv4_arena_runner.
5. Enforces configured parallel game count and per-process thread limits.
6. Accepts only games whose result/replay manifests validate.
7. Merges accepted replay into worker-local active shards under a lock.
8. Atomically finalizes immutable ready shards at row/byte thresholds.
9. Uploads in the background with the durable-transfer protocol.
10. Deletes local data only after a matching XL acknowledgement.

Running games keep their launch bundle; only newly launched games see a
promotion. Apply disk/upload backpressure before starting games when ready or
uploading data exceeds configured limits.

### AlphaGoZeroGV4/xl_coordinator.py

Keep durable ingestion and single-trainer ownership, but validate GV4 metadata.

A shard is accepted only when checksums are complete, its worker/shard/hash is
not already accepted, all schemas match, engine/timing hashes are allowed, and
controller/adversary lineage is present.

After acceptance, index replay by training role, update bounded FIFO
partitions, write the acknowledgement atomically, and update the training gate.
Never ingest incoming_uploading and never start a second trainer while a
train/evaluate/promote cycle is active.

### AlphaGoZeroGV4/agz_train_eval_promote.py

Start from a reduced DNN-only copy. Remove GV3 HGB, Markov, 226D, old arena, and
compatibility branches instead of carrying them forward.

One cycle:

1. Snapshots eligible replay and the promoted bundle.
2. Samples role-specific value and policy examples.
3. Trains controller value, adversary value, controller policy, and adversary
   policy from the configured optimization parents.
4. Exports and checksums all native artifacts.
5. Runs Python/native inference parity.
6. Evaluates candidate adversary against promoted controller.
7. Evaluates candidate controller against promoted adversary.
8. Promotes each role only if its own rule passes.
9. Creates one atomic four-role current_model.json, even if one role changed.
10. Broadcasts immutable files first and the manifest last.

Self-play lineage and optimization lineage stay separate. A rejected candidate
may remain an optimizer parent when explicitly configured, but workers only use
promoted artifacts.

Paired candidate/baseline games must start from the same root payload and
history seed.

### AlphaGoZeroGV4/distributed_eval.py

Add this only after local candidate evaluation passes. Stage exact immutable
artifacts, allocate disjoint paired game IDs, launch gv4_arena_runner.py in
evaluation mode, verify complete coverage, and merge by role. Do not invoke the
GV3 arena module.

### AlphaGoZeroGV4/deploy.py

Adapt deployment last. Build GV4_Cpp, run import/schema smoke tests, sync source
plus one immutable experiment manifest, then launch:

    python -m vidur.AlphaGoZeroGV4.xl_coordinator
    python -m vidur.AlphaGoZeroGV4.worker_daemon

Log source commit/dirty state, engine hash, timing-profile hash, model-manifest
hash, host map, and effective settings before launch.

## 6. Multi-process and multi-worker durability

Use the proven directory state machine:

    worker/game_runs/game_<id>/       isolated subprocess output
    worker/replay/active/             mutable worker-local merge target
    worker/replay/ready/<shard>/      immutable finalized shard
    xl/incoming_uploading/<shard>/    incomplete transfer, never ingest
    xl/incoming/<shard>/              atomically published transfer
    xl/global_replay/<role>/<part>/   accepted bounded replay
    xl/acks/<worker>/<shard>.json     acceptance acknowledgement

Concurrency invariants:

- A game writes only to its own directory.
- Only the worker daemon merges into active replay.
- Active-to-ready publication is an atomic same-filesystem rename.
- Upload publishes incoming only after checksum verification.
- XL ingestion is idempotent by worker ID, shard ID, and hash.
- Only XL writes global replay and current_model.json.
- Only one trainer/evaluator/promoter owns candidate state.
- Referenced artifact files are published before their manifest.

Changing parallel process or worker counts may change throughput, but never game
semantics, seeds, replay schemas, or model lineage.

## 7. Required manifests

### Experiment manifest

Record engine/config hash, all schema IDs, model/TP/PP/device/network identity,
Vidur profile/cache hash, high-level MCTS/training settings, source commit, and
dirty-tree status.

### Model bundle manifest

For each role record role, version, immutable path, SHA256, native-ready flag,
feature/action schemas and dimensions, engine compatibility hash, training
parent, promoted parent, and training data range.

### Replay shard manifest

Record worker/shard IDs, game range, row/byte counts, checksums, schemas,
experiment hashes, controller/adversary versions, and completion marker.

Validate metadata at game finalization, worker merge, XL ingestion, training
load, model load, evaluation, and promotion.

## 8. Implementation order and review gates

### Phase 0: freeze interfaces

1. Freeze engine and schema IDs.
2. Freeze Python/native root-state payload roundtrip.
3. Freeze canonical action identity/aliases.
4. Freeze structured feature dimensions and native export format.
5. Confirm TP=2/PP=2 Python/native transition parity.

Exit: a Python payload round-trips through native without losing any
authoritative field.

### Phase 1: one local uniform game

1. Add __init__.py, config.py, GV4_Engine/history_root.py, and
   gv4_arena_runner.py.
2. Run one Python uniform game.
3. Run identical root/history through native uniform MCTS.
4. Confirm every primary row has at least two canonical actions.
5. Validate forced-chain time, reward, discount, pipeline, KV, and ticks.

Exit: noise-disabled Python/native action and resulting-state parity.

### Phase 2: DNN and replay

1. Adapt dnn_models.py and neutral bootstrap.
2. Connect four models to Python and native inference.
3. Adapt replay_runtime.py.
4. Verify feature, value, logit, and MCTS root parity.
5. Independently recompute replay targets from logged composed edges.

Exit: one native game emits replay loadable by GV4 training.

### Phase 3: many local processes

1. Adapt worker_daemon.py with upload disabled.
2. Run several isolated game processes.
3. Test game failure, daemon restart, partial active data, and ready recovery.
4. Verify unique IDs and immutable per-game model hashes.

Exit: no interleaved rows, duplicate IDs, leaked processes, or premature
deletion.

### Phase 4: one-worker durable transfer

1. Adapt XL ingestion/acknowledgement.
2. Reuse durable_transfer.py.
3. Test interruption, duplicate upload, bad checksum, and restart around ack.

Exit: accepted shards appear once and workers delete only after acknowledgement.

### Phase 5: training and local evaluation

1. Adapt indexed_replay.py.
2. Adapt the DNN-only trainer.
3. Train/export all four artifacts from a known replay sample.
4. Evaluate locally through gv4_arena_runner.py.
5. Test independent role promotion and atomic bundles.

Exit: promoted bundles load natively; rejected models never reach self-play.

### Phase 6: many workers and distributed evaluation

1. Connect several fixed workers to one XL coordinator.
2. Adapt distributed_eval.py with paired roots.
3. Adapt deploy.py and exercise stop/restart/redeploy.
4. Add disk, upload, CPU, and trainer backpressure.

Exit: scaling workers changes throughput only.

### Phase 7: optional features

Add rollouts, spot workers, prebuilt replay caches, distillation, and additional
benchmarks one at a time after the fixed-worker non-rollout path is stable.

## 9. Required tests

### Engine and decision boundary

- Root payload roundtrip preserves every state field.
- Python/native canonical actions and aliases match.
- One canonical action advances without MCTS.
- Two or more canonical actions invoke MCTS exactly once.
- Missed adversary ticks replay in order.
- Decode fast-forward stops at the proper decision/tick boundary.
- Composed reward/discount equals explicit edge-by-edge calculation.
- Zero-time loops respect the safety limit.
- Every primary arena row is a meaningful branch.

### History roots

- The same seed gives the same canonical history and payload.
- Raw aliases do not bias random choices.
- History depth counts branches rather than forced transitions.
- Python/native replay of history reaches equivalent state.
- Early terminal/horizon roots retry according to config.

### Features, models, and search

- Python/native state and action feature parity.
- Masks/padding do not alter real entries.
- Python/native values and logits agree for all roles.
- Noise-disabled root visits and selected actions agree.
- Wrong dimensions, schemas, roles, or hashes fail before search.

### Replay

- Canonical visit targets normalize correctly.
- Aliases do not produce duplicate policy targets.
- Backward targets use composed edge discounts.
- Nonterminal bootstrap kind/value is explicit.
- A trajectory never mixes model versions.
- Readers reject mixed schemas/config hashes.

### Worker and transfer

- Parallel subprocesses use unique IDs and isolated directories.
- Failed games do not enter replay.
- Shard finalization is atomic.
- Interrupted/duplicate upload is safe.
- Bad checksums receive no ack.
- Cleanup happens only after matching ack.
- Backpressure pauses launches without killing active games.

### Coordinator, training, and promotion

- Duplicate shards ingest once.
- Replay bounds prune only eligible oldest partitions.
- Training gate counts accepted compatible rows.
- At most one trainer owns a cycle.
- Four artifacts export with correct role metadata.
- Controller/adversary promotion is independent.
- Candidate evaluation uses paired roots.
- Manifest-last publication hides partial bundles.
- Running games retain old bundle; new games adopt promotion.

## 10. Recommended review boundaries

Do not copy the entire GV3 directory in one change. Adapt one phase at a time and
remove unsupported GV3 branches immediately.

First review:

    AlphaGoZeroGV4/__init__.py
    AlphaGoZeroGV4/config.py
    AlphaGoZeroGV4/gv4_arena_runner.py
    GV4_Engine/history_root.py
    focused decision/history tests

Second review:

    AlphaGoZeroGV4/dnn_models.py
    AlphaGoZeroGV4/replay_runtime.py
    AlphaGoZeroGV4/bootstrap_untrained_dnn_v100.py
    feature/inference/MCTS/replay parity tests

Worker, transfer, coordinator, trainer, distributed evaluation, and deployment
follow in that order. This prevents cluster orchestration from hiding an
engine, feature, or replay error.

## 11. Definition of done

The initial pipeline is complete when:

- GV4 games never import or invoke GV3 engine, arena, features, or model loader.
- Python/native GV4 agree on payloads, features, legal canonical actions,
  transitions, rewards, discounts, inference, and deterministic MCTS.
- Main arena/replay rows contain only branching choices while hidden forced
  transitions remain exactly reconstructable.
- Many processes on many workers safely produce durable compatible replay.
- XL can ingest bounded replay, train/export four artifacts, evaluate roles,
  promote atomically, and broadcast a complete bundle.
- Running games retain their model snapshot and new games use the promotion.
- Restarting any worker, transfer, coordinator, trainer, or deploy process
  cannot duplicate replay, expose partial models, or delete unacknowledged data.





-------------


**Important Runtime Detail**
Commands such as `python -m vidur.AlphaGoZero.worker_daemon` currently load the top-level [`vidur/`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur), not [`vidur-GV4/`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur-GV4). The two copies are separate and some files differ. Editing only `vidur-GV4/` will not change the current `python -m vidur...` runtime unless deployment explicitly remaps it.

**GV3 AGZ Runtime Path**
1. [`AlphaGoZero/config.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/config.py) defines replay limits, model family, training thresholds, MCTS iterations, PUCT constants, root noise, evaluation counts, and SJF settings.

2. [`AlphaGoZero/cluster.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/cluster.py:17) defines the XL coordinator and worker machines.

3. [`AlphaGoZero/deploy.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/deploy.py:195) is the operator entry point. It synchronizes the repository, builds `mcts_native_gv2`, starts [`xl_coordinator.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/xl_coordinator.py), and starts one [`worker_daemon.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/worker_daemon.py:616) per worker.

4. The worker reads `models/current_model.json` before each game through [`_current_model_paths()`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/worker_daemon.py:189). This supplies four promoted artifacts: controller value/prior and adversary value/prior.

5. The worker selects a parent-state ID, creates `runs/game_<id>/`, and launches [`arena_mcts_value_runnerCPP.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/bellman_v4_adv/arena_mcts_value_runnerCPP.py:615) through [`_build_game_command()`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/worker_daemon.py:271). It passes `--only-model-ctrl-cycle`, so self-play is model adversary versus model controller. The trivial-controller cycle is skipped.

6. The arena wrapper calls the GV3 [`Model_Tester/runner.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/mcts/Game_Versions/Game_Version3/Model_Tester/runner.py:1042). This Python code owns the real game trajectory: it builds the simulator/environment, exposes turns, applies selected actions, updates objective accounting, and writes arena logs.

7. For every state with multiple legal actions, the wrapper serializes the Python state and calls the `mcts_native_gv2` pybind module. Its build sources are listed in [`CMakeLists.txt`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/Game_Version3_Cpp/CMakeLists.txt:46). Native MCTS uses `gv2_mcts_value_prior.cpp`, `gv2_mcts_dnn.cpp`, `gv2_virtual_environment.cpp`, native feature builders, and native inference.

8. Native MCTS returns the selected action and root statistics. The Python arena applies that selected action to the authoritative Python GV3 state. Therefore Python runs the actual game, while C++ explores hypothetical tree transitions.

9. [`replay_runtime.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/replay_runtime.py:158) records the complete trajectory, performs backward discounted target calculation, and retains states with `canonical_action_count > 2`.

10. [`worker_daemon.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/worker_daemon.py:384) merges game replay into `active/`. [`durable_transfer.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/durable_transfer.py:217) freezes immutable shards with manifests and checksums and transfers them to XL.

11. [`xl_coordinator.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/xl_coordinator.py:1291) ingests shards, partitions replay by player role, enforces FIFO capacity, and launches [`agz_train_eval_promote.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/agz_train_eval_promote.py) once the training gate is reached.

12. The trainer samples replay, trains four models using [`dnn_models.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/dnn_models.py), exports native artifacts, evaluates both roles independently, promotes successful roles, writes `current_model.json`, and broadcasts it to workers.

`cpp_selfplay_runner.py` is only a local smoke-test wrapper in the current implementation. Production workers directly invoke `arena_mcts_value_runnerCPP.py`. Likewise, Python `mcts_value_prior.py` is a reference implementation; production search uses C++.

**Logging Files**
- Worker control logs: `worker_daemon.log`, `worker_state.json`, `replay_buffer.csv`, `model communication.csv`, and `last_game_error.log`.
- Temporary per-game files: `launch_command.json`, `game_process.log`, `arena_results.csv`, `arena_games/game_<id>_model_adv_depth1_vs_model_ctrl_depth1.csv`, `replay_target_runtime.csv`, and `replay_policy_rows.csv`.
- Arena action CSVs contain time, acting/next player, action, objective values, active/completed requests, decode state, valid/canonical action counts, chosen MCTS values, and top-five action diagnostics. They are produced by [`ArenaGameCycleFileLogger`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/mcts/Game_Versions/Game_Version3/logger/evaluation_pipeline_logger.py:46).
- Worker replay shards contain `replay_target_runtime.csv`, `replay_policy_rows.csv`, `game_outputs/*.json`, `shard_manifest.json`, and `SHA256SUMS`.
- XL logs include `xl_coordinator.log`, `xl_state.json`, `training_gate_status.json`, `replay_distribution.csv`, `train_model.csv`, replay partitions, candidate manifests, and model artifacts.
- Evaluation output includes four role-comparison directories, each with `arena_results.csv` and `arena_games/*.csv`, plus `eval_game_details.csv` and `eval_summary.json`.
- `mcts_visit_logs/*_root.csv`, `*_children.csv`, and `*_model_action_details.csv` are optional debug logs and are disabled during normal self-play.
- Worker per-game directories are deleted after replay merging unless `--keep-game-runs` is enabled. Evaluation and SJF arena logs are retained.

**How SJF-256 Works**
[`_run_sjf_benchmark()`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/agz_train_eval_promote.py:2973) runs only after at least one role is promoted. It launches two matched blocks:

- `cycle1_trivial`: promoted adversary versus fixed SJF-256 controller.
- `cycle2_model`: the same promoted adversary versus promoted model controller.

Both blocks use identical game IDs, history-hop selections, seeds, initial model bundle, and no root noise. The fixed controller chooses an already-legal action with `heuristic="SJF"`, `evict_none`, and prefill target `min(256, total remaining prefill)` through [`trivial_controller.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/mcts/Game_Versions/Game_Version3/Model_Tester/trivial_controller.py:43).

[`merge_split_sjf_cycles()`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/distributed_eval.py:1017) matches both runs by game ID/history hops, combines their costs, copies their per-game logs, and writes `SJF_256_Game/arena_results.csv`. Lower controller cost wins. SJF does not influence promotion; it is a post-promotion benchmark.

**Cached History Roots**
Your recollection is correct about the intended large self-play design. [`deploy.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/deploy.py:132) assigns pre-generated ModelSearchBed parent datasets to workers, and [`worker_daemon.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/worker_daemon.py:235) randomly selects parent IDs without reuse until the available set is exhausted.

However, the current checkout has a wiring defect: the arena wrapper tries to patch `_make_base_snapshot` at [`arena_mcts_value_runnerCPP.py:1600`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/bellman_v4_adv/arena_mcts_value_runnerCPP.py:1600), but the current tester defines and calls `_prepare_base_state_for_game()` at [`runner.py:514`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/mcts/Game_Versions/Game_Version3/Model_Tester/runner.py:514). Consequently, the cached-parent arguments are currently passed but not injected. With the worker default `history_hops=0`, current games fall back to the initial state.

Evaluation and SJF intentionally do not use those cached datasets. They create roots live using [`history_root.py`](/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/mcts/Game_Versions/Game_Version3/DNN/history_root.py:417): forced one-action transitions are skipped, while random branching decisions count as history hops.