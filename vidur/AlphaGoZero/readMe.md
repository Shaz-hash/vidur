# AlphaGoZero GV3 System

This directory contains the AlphaGoZero-style training loop for GV3. It uses
the existing C++ game engine and native MCTS with either classical HGB or
incrementally trained DNN bundles. EXP2 uses DNN value/policy models,
`markov_v2` value features, and native `full_tree` MCTS. Every current bundle
has four artifacts:

- `controller_value`: controller-valued state value used when the controller model is being tested/promoted.
- `adversary_value`: controller-valued state value used when the adversary model is being tested/promoted.
- `controller_prior`: scores controller canonical actions from 226D state features plus 43D controller action features.
- `adversary_prior`: scores adversary canonical actions from 226D state features plus 7D adversary action features.

Both value models train on the same sampled states and controller-valued
targets. Separate artifacts allow controller and adversary promotion to proceed
independently.

## Current Architecture

The selected XL host is coordinator, replay owner, trainer, evaluator, and
promoter. Its selected eight workers generate self-play replay.

Workers:

- Keep a local shard of parent controller/adversary states.
- Launch C++ self-play games in parallel, normally `60` processes per worker.
- Start each game from a random parent state, without reusing the same parent state inside one local replay-buffer cycle.
- Use the latest promoted role bundle available on the worker: controller value/prior and adversary value/prior can come from different model versions.
- Store replay states where `canonical_action_count > 2`.
- Flush a local replay shard after about `10,000` replay states.
- Upload immutable replay shards to XL with checksums and wait for ack before deleting local ready shards.
- Delete per-game run directories after merge unless debug retention is explicitly enabled.

Classical XL:

- Receives worker shards under `incoming/`.
- Verifies manifests and SHA256 checksums.
- Ingests rows into replay partitions under `global_replay/partitions/<worker>/<shard>/`.
- Keeps replay bounded with `XL_MAX_REPLAY_STATES = 15,000,000`, split into role budgets.
- Starts training once `TRAIN_TRIGGER_NEW_STATES = 600,000` new replay states arrive after the last completed train/eval/benchmark cycle.
- Trains one candidate bundle: controller value, adversary value, controller prior, adversary prior.
- Evaluates candidate controller and candidate adversary independently against the current promoted role bundle.
- Promotes controller artifacts if the candidate controller passes its 100-game role threshold.
- Promotes adversary artifacts if the candidate adversary passes its 100-game role threshold.
- Broadcasts the resulting mixed current role bundle to workers.
- Runs SJF-256 benchmark games for promoted versions and keeps arena CSV logs.

## Important Defaults

These live in `config.py` and training/eval launch code.

```text
MAX_PROMOTIONS                 = 50
PROMOTION_WIN_RATE_THRESHOLD   = 0.55
ROLE_PROMOTION_WIN_THRESHOLD   = 55 wins out of 100
TRAIN_TRIGGER_NEW_STATES       = 600,000
XL_MAX_REPLAY_STATES           = 15,000,000
XL_CONTROLLER_MAX_REPLAY       = 10,000,000
XL_ADVERSARY_MAX_REPLAY        = 5,000,000
VALUE_SAMPLE_CAP               = 500,000 by default, often 700,000 by launch env
CONTROLLER_POLICY_ROOT_CAP     = 200,000 by default, often 250,000 by launch env
ADVERSARY_POLICY_ROOT_CAP      = 50,000 by default, often 150,000 by launch env
MIN_ADVERSARY_STATES_FOR_EVAL  = 25,000

self-play mcts_iterations      = 2,000
self-play puct_c               = 2.5
self-play root noise           = enabled
self-play Dirichlet alpha      = 0.05
self-play Dirichlet epsilon    = 0.25
self-play sample initial moves = enabled for moves 1-20
self-play after move 20        = deterministic most-visited action

eval mcts_iterations           = 1,000
eval puct_c                    = 1.0
eval root noise                = disabled
eval prior temperature         = 1.0
eval sample initial moves      = disabled
arena time limit               = 5 seconds
```

The value target is controller-valued. Good controller outcomes are less negative / higher. The adversary value model also predicts controller-valued returns; the adversary still selects actions from the adversary objective by inverting normalized exploitation during tree selection.

## End-to-End Loop

1. Workers generate replay using the current promoted role bundle.
2. Each worker accumulates local active replay until the shard threshold is reached.
3. Worker finalizes the shard with a manifest and SHA256 checksums.
4. Worker uploads through the crash-safe `incoming_uploading/` handoff, verifies remotely, and atomically publishes the shard for XL.
5. XL eagerly drains every published shard into replay and writes an ack. There is no model-version admission queue or per-loop shard throttle.
6. XL updates replay counters and training gate status.
7. If no train/eval cycle is running and enough new states arrived, XL starts `agz_train_eval_promote.py`.
8. Trainer streams replay partitions, samples bounded training sets, continues all four DNNs from the latest published candidate checkpoint, and writes metrics/artifacts.
9. XL evaluates candidate controller and candidate adversary separately against the current promoted role bundle.
10. If either role reaches its promotion threshold, XL updates only that role in `current_model.json` and broadcasts the mixed role bundle to workers.
11. XL runs the SJF-256 benchmark when at least one role is promoted.
12. Only after train + eval + optional benchmark complete, XL advances the coordinator-owned lifetime-ingestion watermark to the count captured at launch. States admitted after launch remain eligible for the next cycle.


Training and self-play use deliberately separate lineages. Workers always use
the current promoted controller/adversary bundle. A rejected candidate is never
broadcast for self-play, but its DNN and AdamW state remain the parent of the
next candidate. Therefore optimization continues through failed arena
checkpoints while promotion still compares against the current best bundle.
This means if another 600k states arrive while training/eval/benchmark is still running, a second trainer is not launched concurrently. The next trainer becomes eligible only after the current full cycle is finalized. Training and evaluation never rewrite replay counters; this avoids losing ingestion progress to concurrent `xl_state.json` writes.

## Replay Storage and Disk Boundaries

The replay is a plain role-separated FIFO sliding window. Model versions are
recorded for diagnostics but never control admission, sampling, or eviction.

Current intended behavior:

- Controller replay retains the newest 20M controller states.
- Adversary replay retains the newest 15M adversary states.
- Total configured replay capacity is therefore 35M feature-complete states.
- Every valid row is admitted regardless of which model version produced it.
- When a role exceeds its capacity, XL deletes that role's oldest replay partitions until it is back within capacity.
- No current-model quota, previous-five history, minimum historical quota, partial version admission, or obsolete-version rejection is applied.
- `replay_distribution.csv` still logs the exact retained state count for the current promoted model and every other model version, independently for controller and adversary.
- Worker keeps only `active/` plus unacked `ready/` shards.
- Worker uploads finalized shards in FIFO order. It does not reorder or drop them based on model freshness.
- Workers read `models/current_model.json` before each game launch, so games launched after a promotion use the newest promoted controller/adversary bundle.
- Worker deletes each per-game output directory after `_merge_game_into_active()` unless `--keep-game-runs` is set.
- XL stores accepted replay as prunable partitions, not as one append-only monolithic CSV.
- XL accepted shard directories keep only small manifests/checksums by default.
- `XL_KEEP_ACCEPTED_SHARDS = False`.
- `XL_WRITE_AUDIT_REPLAY = False`.
- XL drains all available published shards on every coordinator pass, including while training and evaluation run. `incoming/` is only a short-lived durable transfer handoff, not a training gate or replay-policy queue.
- If pruning is unsafe while a trainer is reading partitions, ingestion continues and pruning catches up immediately after that cycle completes.

### Simple FIFO Ingestion (2026-07-12)

Each mixed worker shard is split into controller and adversary partitions. The
role split exists only so each role can enforce its own capacity without deleting
the other role's rows. Shard model-version fields remain metadata used by
`replay_distribution.csv` and ingestion auditing.

The resulting behavior is the standard AlphaGo Zero sliding replay window:
new samples enter immediately, the newest 20M controller and 15M adversary
states remain available for training, and old samples age out solely by arrival
order. Promotions affect future worker games but do not rewrite, reserve, or
filter the replay already retained.

Replay files:

```text
replay_target_runtime.csv
  One row per retained replay state.
  Contains target value, 226D state feature JSON, root metadata, MCTS top5 summaries.

replay_policy_rows.csv
  One row per canonical action for retained replay states.
  Contains action features, MCTS visit probability target, model prior, Q value, immediate reward, discount, bootstrap.
```

The trainer can read legacy monolithic replay files, but new ingestion should write partitioned replay only.

## Training

Training is XL-only and happens in `agz_train_eval_promote.py`.

Sampling:

- Both value models sample the same value-state set and train against the same controller-valued targets.
- Value sampling defaults to `500,000` feature-complete replay states and can be raised with `AGZ_MAX_VALUE_STATES`.
- Controller policy samples up to `AGZ_CONTROLLER_POLICY_SAMPLE_CAP` controller roots.
- Adversary policy samples up to `AGZ_ADVERSARY_POLICY_SAMPLE_CAP` adversary roots.
- Policy caps are maxima, not additional training minima. When fewer actionable
  roots are available, policy training uses all available roots after the
  corresponding controller/adversary value-state minimum has been satisfied.
- The trainer uses streaming/reservoir sampling over replay partitions so it does not load the full replay buffer into memory.
- Training is blocked if sampled adversary rows are below `MIN_ADVERSARY_STATES_FOR_EVAL`.

Value model:

```text
HistGradientBoostingRegressor(
  loss="squared_error",
  max_leaf_nodes=63,
  max_iter=1050,
  learning_rate=0.05,
  l2_regularization=1.0,
  early_stopping=False,
)
sample_weight = 1 + 2 * abs(target_value)
```

Policy models:

- Same HGB regressor family.
- One regressor predicts action logits/scores.
- Targets are centered per root from `log(visit_count + POLICY_ALPHA)`.
- `POLICY_ALPHA = 1.0`.
- At inference, valid canonical action scores are softmaxed with temperature `1.0`.

Training outputs:

```text
simulator_output/GV3_Agent/AlphaGoZero/models/Model_Version<N>/
  candidate_manifest.json
  controller_value/hgb_sq_63leaf_1050iter_a2/model.joblib
  controller_value/hgb_sq_63leaf_1050iter_a2/native_model.tsv
  adversary_value/hgb_sq_63leaf_1050iter_a2/model.joblib
  adversary_value/hgb_sq_63leaf_1050iter_a2/native_model.tsv
  controller_prior/hgb_policy_63leaf_1050iter/model.joblib
  controller_prior/hgb_policy_63leaf_1050iter/native_model.tsv
  adversary_prior/hgb_policy_63leaf_1050iter/model.joblib
  adversary_prior/hgb_policy_63leaf_1050iter/native_model.tsv
```

Global training metrics are appended to:

```text
simulator_output/GV3_Agent/AlphaGoZero/train_model.csv
```

Important metrics:

- `controller_value_rmse` / `adversary_value_rmse` and corresponding p95/max absolute errors.
- `controller_policy_cross_entropy`, `controller_policy_top1`, `controller_policy_top3`.
- `adversary_policy_cross_entropy`, `adversary_policy_top1`, `adversary_policy_top3`.

Policy cross entropy is computed after softmax over each root's canonical actions. It measures how well the model distribution matches the MCTS visit distribution, not whether the final arena policy is strong by itself.

## DNN Architecture

The incremental DNN family keeps the existing GV3 feature, game, and search
contracts. It adds no scheduling rules, future-cost auxiliary head, or MCTS
logic. The four artifacts remain `controller_value`, `adversary_value`,
`controller_prior`, and `adversary_prior`.

### Markov Value-State Contract

The selectable replacement for `legacy_226` is `markov_v2`. The legacy vector
keeps only selected request slots and aggregates; it can therefore alias states
whose hidden requests or launch-event expiry times produce different futures.
`markov_v2` instead serializes all live transition state as variable-length sets:

```text
MarkovValueStateV2
  global_features       [19]
  request_tokens        [number of active requests, 24]
  launch_history_tokens [number of retained one-second-window events, 3]
```

Padding masks exist only inside a training batch. They are not semantic input
features. No top-k selection, truncation, sampling, or numeric request ID is
allowed. Missing or duplicate active request records are hard errors. Request
and launch tokens are consumed by permutation-invariant encoders.

Both role-specific value artifacts predict discounted cost-to-go from the
controller's perspective. The player-to-act selects the controller or adversary
value artifact; it is routing metadata, not a numeric feature. Absolute episode
time, turn number, and evaluation/self-play horizon are also excluded because
GV3 is a continuing, time-homogeneous game.

#### Global Features (19D)

All divisions below are ordinary division with no clipping. `asinh` always
means `asinh(raw_value / scale)`, never a bucket index.

| Index | Feature |
| ---: | --- |
| 0 | active request count / `120` |
| 1 | active prefill count / `20` |
| 2 | active decode count / `100` |
| 3 | total remaining prefill tokens / `(20 * 4096)` |
| 4 | total remaining decode tokens / `(100 * 864)` |
| 5 | total processed prefill tokens / `(20 * 4096)` |
| 6 | total processed decode tokens / `(100 * 864)` |
| 7 | total processed context tokens / `(120 * (4096 + 864))` |
| 8 | `asinh(raw signed decode-credit balance / 216)` |
| 9 | non-negative usable decode credit / `216` |
| 10 | `asinh((next_adversary_tick - simulator_time) / 0.2)` |
| 11 | pending-adversary-tick boolean |
| 12-14 | one-hot missed-tick source: none, controller-cross, fast-forward/jump-cross |
| 15 | requests launched in the retained one-second window / `7` |
| 16 | prefill tokens launched in that window / `(7 * 1024)` |
| 17 | active violated-request count / `100` |
| 18 | active finalized-prefill-lateness count / `20` |

`next_adversary_tick` is required. It is not replaced by a generic clock-phase
feature. The missed-tick source remains because native search uses it to decide
whether a crossed tick produces a forced adversary continuation. Historical
generated/completed/dropped/stopped counts and cumulative objective cost are
not inputs: terminal requests no longer affect the current transition, and the
training target already represents future cost-to-go.

#### Active Request Tokens (24D Each)

There is exactly one token for every ID in `active_request_ids`. Terminal and
feature-only records are rejected from the active set.

| Index | Feature |
| ---: | --- |
| 0 | decode-phase boolean (`0` means prefill phase) |
| 1 | total prefill tokens / `4096` |
| 2 | processed prefill tokens / `4096` |
| 3 | remaining prefill tokens / `4096` |
| 4 | total decode tokens / `864` |
| 5 | processed decode tokens / `864` |
| 6 | remaining decode tokens / `864` |
| 7 | total processed context tokens / `(4096 + 864)` |
| 8 | `asinh((simulator_time - arrived_at) / 1 second)` |
| 9 | `asinh((simulator_time - queued_at) / 1 second)` |
| 10 | `asinh(prefill_slo / 1 second)` |
| 11 | `asinh((prefill_deadline - simulator_time) / 1 second)` |
| 12 | `asinh(decode_slo / 0.05 second)` |
| 13 | decode-deadline presence bit |
| 14 | `asinh((decode_deadline - simulator_time) / 0.05 second)`, or zero if absent |
| 15 | prefill-completion-time presence bit |
| 16 | `asinh((simulator_time - prefill_completed_at) / 1 second)`, or zero if absent |
| 17 | `asinh(prefill_lateness / 1 second)` |
| 18 | `asinh(cumulative decode lateness / 1 second)` |
| 19 | decode tokens already counted by the credit ledger / `864` |
| 20 | credit-ledger-entry presence bit |
| 21 | violated boolean |
| 22 | prefill-lateness-finalized boolean |
| 23 | `is_prefill_complete` boolean, consistency-checked against remaining prefill |

The serializer checks non-negative token counts, processed <= total, remaining
= total - processed, active/terminal consistency, deadline presence, and finite
values. Arrival and queue ages are retained because action generation and
deadline fallback use them. A mask is not listed: an active token is present by
definition, and padded rows are ignored by the batch mask.

#### Launch-History Tokens (3D Each)

`recent_launches` is the complete rolling one-second launch window, not a fixed
previous-second bucket. Each retained event contains:

| Index | Feature |
| ---: | --- |
| 0 | `asinh((simulator_time - launch_timestamp) / 1 second)` |
| 1 | request count / `7` |
| 2 | prefill-token count / `(7 * 1024)` |

The encoder ignores simulator-retained records older than the configured
one-second window; future-dated events are validation errors. Every event still
inside the window is encoded. Exact ages are necessary: equal aggregate usage with
different expiration times permits different future adversary launches.

#### Deliberately Excluded Inputs

The value model does not receive player/turn bits, absolute simulator time,
tick-grid phase, turn/depth/root IDs, model or worker IDs, historical terminal
request counters, cumulative objective cost, MCTS priors/Q/visits, chosen
actions, replay targets, future rewards, or horizon progress. Edge outputs such
as `transition_discount_time` and `transition_final_time` are produced after an
action and are not root-state inputs.

#### Sufficiency and Parity Gates

`markov_v2` is accepted only when Python/native CSV parity, no-alias mutation
tests, request/launch permutation invariance, prefill-versus-decode transition
separation, finite-value validation, and model-inference parity all pass. Every
model artifact records the schema name, dimensions, normalization scales, and
game limits; loaders reject mismatches instead of silently falling back.

### Value Models

`legacy_226` remains available and keeps its approximately 101k-parameter flat
residual MLP. `markov_v2` uses a residual DeepSets network:

```text
global[19] -> Linear(19,64) -> LayerNorm -> SiLU

each request[24]
  -> Linear(24,64) -> SiLU -> Linear(64,64) -> LayerNorm -> SiLU
  -> masked sum pool || masked max pool                         [128]

each launch[3]
  -> Linear(3,32) -> SiLU -> Linear(32,32) -> LayerNorm -> SiLU
  -> masked sum pool || masked max pool                          [64]

concatenate global/request/launch                              [256]
  -> Linear(256,192) -> SiLU
  -> two LayerNorm(192) / Linear(192,64) / SiLU /
     Linear(64,192) residual blocks
  -> LayerNorm(192) -> Linear(192,32) -> SiLU
  -> Linear(32,1) -> tanh -> denormalize to [-50,0]
```

Sum pooling preserves backlog magnitude; max pooling preserves the most urgent
or late request. Together they let the network learn that many individually
safe prefills can still form a dangerous backlog while remaining invariant to
request ordering.

The controller-perspective target is normalized as:

```text
normalized_value = value / 25 + 1
value             = 25 * (normalized_value - 1)
```

Thus `-50 -> -1`, `-25 -> 0`, and `0 -> 1`. Python and native inference
denormalize before returning values to MCTS, so reward, Q, and Bellman units do
not change. Training rejects targets outside `[-50,0]`; it never silently clips.
`markov_v2` uses normalized Huber loss with delta `0.1`, AdamW, shuffled
mini-batches, and gradient-norm clipping at `1.0`. It has one scalar output and
no handcrafted future-cost or action-specific auxiliary head. Each role has a
separate artifact, but both train on the same sampled states and the same
controller-perspective targets.
### Policy Models

The controller uses 43D action features and the adversary uses 7D action
features. Both use the same ranking architecture:

```text
state tower:  226 -> Linear(226, 192) -> LayerNorm -> SiLU
action tower: D   -> Linear(D, 64)     -> LayerNorm -> SiLU
concatenate: 256
  -> Linear(256, 128) -> LayerNorm -> SiLU
  -> LayerNorm(128) -> Linear(128, 32) -> SiLU
       -> Linear(32, 128) -> residual add
  -> LayerNorm(128) -> Linear(128, 64) -> SiLU
  -> Linear(64, 1) -> unbounded action logit
```

This is approximately 97k parameters for controller policy and 95k for
adversary policy. The state tower is evaluated once per root; action and fusion
work is batched across valid canonical actions. Individual logits remain
unbounded. Softmax over valid canonical actions produces the bounded policy.

Each replay root is one training example:

```text
target_policy = visit_counts / sum(visit_counts)
policy_loss   = -sum(target_policy * log_softmax(action_logits))
```

Rows from a root stay grouped and are not optimized as independent regression
examples.

### Incremental DNN Training

- AdamW uses `3e-4` for initialization, `1e-4` for incremental updates,
  weight decay `1e-4`, and global gradient clipping at `1.0`.
- Each candidate warm-starts all four models from the latest fully published,
  native-parity-validated candidate checkpoint and preserves optimizer state.
- If no newer candidate exists, the corresponding promoted role is the safe
  fallback. Failed evaluations do not roll training weights back, but workers
  continue self-play exclusively from promoted role models.
- Training uses 3-5 replay passes, value batches of 4096 states, and policy
  batches of 128-512 complete roots.
- Replay scanning/cache construction remains multiprocess; tensor optimization
  is batched.
- Dropout and BatchNorm are not used. LayerNorm supports deterministic
  single-state and variable-action inference.
- HGB remains available as the default/fallback family. DNN runs are explicitly
  selected with `AGZ_MODEL_FAMILY=dnn`.

### DNN Runtime Contract

Artifacts contain model kind, role, architecture version, dimensions, value
bounds, FP32 parameters, optimizer checkpoint, and feature-schema metadata.
Python and native load the same parameters; native self-play never calls Python.

Before deployment, a DNN bundle must pass:

- exact state/action feature and canonical-order parity;
- single and batched value/logit Python-native maximum absolute error `<=1e-4`;
- matching policy top-1/top-3 outside declared near-ties;
- noise-disabled MCTS agreement at 1, 10, and 1,000 simulations without
  modifying GV3 game or MCTS logic; and
- native 1,000-simulation throughput no more than 5% slower than the HGB
  baseline on identical hardware and roots.

Only bundles passing export, inference parity, MCTS parity, and throughput gates
may be marked `native_ready` and broadcast.

## Candidate Evaluation and Promotion

Candidate eval is role independent. Every candidate version runs two 100-game comparisons:

```text
Testing Adversary:
  promoted adversary vs promoted controller
  candidate adversary vs promoted controller

Testing Controller:
  promoted adversary vs promoted controller
  promoted adversary vs candidate controller
```

The candidate adversary is promoted when it produces higher total SLO cost than the promoted adversary in more than 55 of its 100 comparison games. The candidate controller is promoted when it produces lower total SLO cost than the promoted controller in more than 55 of its 100 comparison games.

Controller and adversary versions can diverge. For example, the current bundle can contain controller v105 with adversary v125. `current_model.json` stores both role versions and all four model paths.

After a role promotion:

- `xl_state.json` is updated for the promoted role only.
- `models/current_model.json` is written with the mixed current role bundle.
- Model artifacts are rsynced to each worker.
- XL runs 50 SJF-256 benchmark games when at least one role promoted.
- Arena CSV logs are kept for all candidate evals and SJF-256 benchmark games.

Eval output:

```text
simulator_output/GV3_Agent/AlphaGoZero/eval_of_models/eval_000<N>/
  promoted_adversary_vs_promoted_controller_for_adversary/arena_results.csv
  candidate_adversary_vs_promoted_controller/arena_results.csv
  promoted_adversary_vs_promoted_controller_for_controller/arena_results.csv
  promoted_adversary_vs_candidate_controller/arena_results.csv
  promotion_eval_summary.csv
  eval_game_details.csv
  SJF_256_Game/arena_results.csv
  SJF_256_Game/jobs/*/arena_games/*.csv
```

The SJF-256 benchmark is not part of the promotion decision. It is an external sanity check against a fixed trivial policy.

## MCTS With Value and Priors

Python reference implementation: `vidur/Game_Version3/mcts_value_prior.py`

C++ production implementation: `vidur/Game_Version3_Cpp/src/gv2_mcts_value_prior.cpp`

The Python file is the clearest reference for the algorithm. The C++ file mirrors it for arena/self-play speed.

### Expansion

At a node:

1. Sample all valid actions for the player.
2. Canonicalize actions.
3. Build policy feature rows for canonical actions only.
4. Predict prior logits with the correct player policy model.
5. Softmax logits over canonical actions.
6. Optionally mix root Dirichlet noise for self-play only.

Controller canonicalization key:

```text
(
  token_alloc,
  prefill_alloc,
  decode_alloc,
  evicted_ids,
)
```

This intentionally ignores old heuristic labels. If two controller action aliases produce the same concrete allocation and eviction set, they are the same canonical action.

Adversary actions currently canonicalize by their valid action index.

### Selection

When policy priors are enabled, selection uses PUCT:

```text
score(s, a) = exploit(s, a) + puct_c * prior(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

Where:

- `prior(s,a)` is the model policy prior for the canonical action.
- `N(s)` is parent visits.
- `N(s,a)` is child visits.
- `exploit` is normalized to `[0, 1]` from the parent node's observed child value range.
- Controller uses normalized value directly.
- Adversary uses `1 - normalized_value` because the value is controller-valued.
- Unvisited actions use neutral exploitation `0.5`.

Self-play uses `puct_c = 2.5` to generate broader samples. Arena/eval uses `puct_c = 1.0`.

### Leaf Value, Policy Rollout, and Backpropagation

Standard `full_tree` search has no rollout: the leaf value comes directly from
the value model when `model_version > 0` and `use_model_bootstrap=True`;
otherwise bootstrap is `0.0`.

EXP3 uses `full_tree_rollout`. When PUCT reaches an unexpanded leaf, the leaf is
expanded and one child action is selected. Ten independent continuations then
sample both player policy networks with temperature `1.0`. All sibling child
actions are compared over the same simulated-time interval:

```text
deadline          = expanded_leaf.sim_time + rollout_horizon_sec
remaining_rollout = deadline - selected_child.sim_time
```

The selected child action therefore consumes part of the one-second horizon. A
short decode action receives a longer continuation than a long prefill action,
but both are evaluated at the same absolute deadline relative to their common
expanded parent. Normal game-engine transitions are retained during the
continuation, including fast-forward and adversary turns. A final action may
cross the deadline; the value model bootstraps that resulting state. This is
expansion-leaf-relative, not search-root-relative.

Backpropagation uses Bellman-style controller-valued returns:

```text
value(parent -> child) = immediate_reward + discount * value(child)
immediate_reward       = parent_cost - child_cost
```

Because costs are bad, values are normally negative or zero. A child that reduces future cost is less negative / better for the controller.

For rollout search, the discounted rewards from every continuation transition
plus its final value bootstrap are backed up through the selected tree edge and
then through the explicit tree using the same time-varying discount rule.

### Final Action Selection

Arena action selection uses most visits:

```text
best_action = argmax_a N(root, a)
```

Ties use root-player objective over mean value, then lower action index. This was changed away from raw Q-max because Q-max made arena behavior too sensitive to small value noise.

Arena CSV rows include top action visits and priors so root policy behavior can be inspected.

## Arena Games

All official arena/self-play/eval games use the C++ runner launched from Python:

```text
vidur.bellman_v4_adv.arena_mcts_value_runnerCPP
```

Important flags used by AlphaGoZero:

```text
--value-model-path
--controller-prior-model-path
--adversary-prior-model-path
--model-version
--mcts-iterations
--puct-c
--policy-prior-temperature
--prior-min-prob
--root-dirichlet-alpha
--root-dirichlet-epsilon
--agz-replay-target-csv
--agz-sample-initial-moves
--agz-mcts-action-temperature
```

Self-play writes replay targets. Eval/benchmark writes arena results and, when enabled, arena game CSV logs. We disabled unnecessary historical/trivial debug output in production paths to avoid disk bloat.

## Synthetic Trace Testing

The next evaluation phase is fixed-trace testing: run the promoted AlphaGoZero controller on request traces from `data/processed_traces/`, especially:

```text
data/processed_traces/splitwise_conv.csv
data/processed_traces/splitwise_code.csv
```

Those CSVs provide externally observed arrivals:

```text
arrived_at,num_prefill_tokens,num_decode_tokens
```

The goal is not to retrain MCTS to know the future trace. In real deployment the model will not know future arrivals. Therefore the model should keep using the existing AlphaGoZero value+prior MCTS internally and continue planning against its learned adversary. The fixed trace only replaces the real outer adversary that injects requests into the environment.

### Test Modes

There are two useful trace-testing modes.

`gv3-compatible-random-history`

This is the first validation mode. It converts trace rows into normal GV3 adversary-style request launches, then lets a random controller choose valid actions. It emits a standard `mcts_iter`-style CSV and runs the existing game-engine invariants from `vidur/tests/game_engine_tests.py`.

This mode intentionally preserves GV3 timing/window assumptions while allowing
natural trace token counts by default:

- arrivals are aligned to GV3 adversary ticks where needed
- `token_policy=clip` preserves raw prefill/decode lengths except for hard caps
- `token_policy=bucket_gv3` is available only when explicitly requested
- prefill SLO is derived from the nearest prefill profile entry with the usual GV3 slowdown
- decode SLO remains the GV3 default 0.05 seconds per token
- launch-window constraints are respected
- generated rows remain compatible with existing trace validators

The purpose is to prove that trace-shaped request streams do not break the GV3 game engine or controller action semantics.

`fixed-trace-controller-eval`

This is the later evaluation mode. It preserves trace arrival times as the real request source, injects due requests into the live environment, and compares:

- AlphaGoZero controller using existing C++ value+prior MCTS
- SJF-256 controller
- SJF-512 controller
- optionally EDF/LST baselines

The MCTS implementation is not changed. The only difference from arena games is the real outer arrival source.

### Planned Directory

Synthetic trace code should live under:

```text
vidur/Game_Version3/Model_Tester/Synthetic_Trace_Tester/
```

This keeps trace evaluation separate from the existing model-vs-model and model-vs-trivial arena runner.

### New Files

`trace_types.py`

Defines small dataclasses used across the trace runner:

```text
TraceRequest
TraceWindow
TraceRunSummary
```

`TraceRequest` should preserve both original and effective token counts, because real traces may contain requests outside the GV3 training range.

`trace_loader.py`

Loads trace CSVs, validates required columns, sorts by `arrived_at`, and can produce deterministic windows from a longer trace.

Important responsibilities:

- parse `arrived_at,num_prefill_tokens,num_decode_tokens`
- support `--window-sec`
- support `--num-games`
- support seeded random window selection
- optionally rebase each sampled window to start at time `0`

`trace_token_policy.py`

Centralizes token handling for out-of-range trace rows.

Initial policies:

```text
clip
  prefill = min(max(prefill, 1), 4096)
  decode  = min(max(decode, 1), 864)

bucket_gv3
  prefill is mapped to the nearest allowed GV3 prefill bucket
  decode is clipped to the GV3 decode cap
```

Later policies can include request splitting, but the first implementation should keep clipping explicit and logged.

`trace_request_injector.py`

Injects trace requests into a `VirtualVidurMCTSEnvironment` state.

This should create real `Request(...)` objects directly and add them to the replica scheduler. It should not use `AdversaryAction`, because `AdversaryAction` applies synthetic adversary tick/window behavior.

For the `gv3-compatible-random-history` mode, this file can also expose a helper that converts trace rows into GV3-compatible `AdversaryAction` objects when we specifically want validator-compatible history traces.

`random_trace_history_runner.py`

Builds a random-action game history from CSV arrivals.

This is the first implementation target. It should:

- load a trace window
- create a GV3 environment
- generate trace-shaped adversary launches
- choose random valid controller actions
- write a normal `mcts_iter`-style trace CSV
- run or support running `vidur/tests/run_game_engine_trace_tests.py`

The output is a test artifact, not a model benchmark.

`fixed_trace_runner.py`

Runs the actual fixed-trace model evaluation.

This should:

- initialize a live GV3 state
- inject all trace rows with `arrived_at <= simulator_time`
- if the system is idle, fast-forward to the next trace arrival
- if requests are active, select a controller action
- apply the controller action with trace-safe fast-forward behavior
- inject any trace rows that arrived during the controller batch
- log per-step and per-request results

The model action selector should reuse the existing C++ value+prior arena selector from `vidur/bellman_v4_adv/arena_mcts_value_runnerCPP.py`.

`trace_logger.py`

Writes trace-evaluation CSVs:

```text
trace_results.csv
trace_steps/game_<id>_<policy>.csv
per_request_results/game_<id>_<policy>.csv
```

The logs should include:

- trace name and window id
- controller policy
- original/effective token counts
- injected request count
- active/completed/dropped request ids
- action repr
- total cost, SLO violations, total lateness
- model MCTS diagnostics when the policy is AlphaGoZero

`run_random_trace_history.py`

CLI entrypoint for generating validator-compatible random history traces.

Example:

```bash
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur-classical-search \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
  -m vidur.Game_Version3.Model_Tester.Synthetic_Trace_Tester.run_random_trace_history \
  --trace-csv data/processed_traces/splitwise_conv.csv \
  --output-csv simulator_output/GV3_Agent/Synthetic_Trace_Tester/splitwise_conv_random_history.csv \
  --num-requests 500 \
  --seed 2026 \
  --mode gv3-compatible
```

`run_fixed_trace_eval.py`

CLI entrypoint for model and baseline fixed-trace evaluation.

Example:

```bash
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur-classical-search \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
  -m vidur.Game_Version3.Model_Tester.Synthetic_Trace_Tester.run_fixed_trace_eval \
  --trace-csv data/processed_traces/splitwise_conv.csv \
  --output-dir simulator_output/GV3_Agent/Synthetic_Trace_Tester/eval_000139_splitwise_conv \
  --num-games 50 \
  --window-sec 5 \
  --num-parallel-games 20 \
  --policies model_mcts_cpp,SJF256,SJF512 \
  --shared-root-mcts-iterations 1000 \
  --puct-c 1.0 \
  --token-policy clip
```

`analysis.py`

Builds per-window comparison CSVs and plots:

- model vs SJF-256
- model vs SJF-512
- model vs per-window best trivial baseline
- game-only cost plots without unrelated history cost

### Existing Files Touched

`vidur/Game_Version3/virtual_environment.py`

Add a narrow trace-injection API:

```text
inject_trace_requests(...)
prepare_trace_controller_turn(...)
```

`inject_trace_requests(...)` should be used by the fixed-trace runner to add external requests directly.

`prepare_trace_controller_turn(...)` should prevent the real outer trace runner from being blocked by stale GV3 adversary-tick metadata before applying a real controller action. This must not change MCTS internals.

`vidur/bellman_v4_adv/arena_mcts_value_runnerCPP.py`

Expose/refactor the C++ value+prior action selector so `fixed_trace_runner.py` can call the same model selection logic used by arena games.

The MCTS algorithm itself should not change.

`vidur/tests/game_engine_tests.py`

No immediate changes for `gv3-compatible-random-history`. The generated CSV should be compatible with existing tests.

Later, true fixed-trace tests may need a trace-specific invariant mode that skips synthetic-adversary checks like adversary tick progression, adversary action-index matching, and sliding-window launch constraints.

### Validation Plan

1. Generate a small random history trace from the first few hundred `splitwise_conv.csv` rows.
2. Run `vidur/tests/run_game_engine_trace_tests.py` on the generated CSV.
3. Confirm request ids are sequential, controller allocations are valid, objective logs match reconstructed objective, and finalized ids do not reappear.
4. Run fixed-trace eval on a small 5-second window with SJF-256 and SJF-512 only.
5. Add AlphaGoZero model policy using the existing C++ selector.
6. Scale to 50 windows and produce model-vs-best-trivial summaries.

## EXP2 Implementation and Native Map

This local tree is the maintained superset of the code deployed on the
`bellman-classical-exp2-*` cluster. It also contains EXP3 rollout support, so
the cluster and search mode must always be explicit. EXP2 uses ordinary
leaf-bootstrap MCTS, not rollout MCTS:

```bash
AGZ_CLUSTER=exp2
AGZ_MODEL_FAMILY=dnn
AGZ_VALUE_FEATURE_SCHEMA=markov_v2
AGZ_NATIVE_SEARCH_MODE=full_tree
AGZ_MCTS_ITERATIONS=1000
AGZ_EVAL_MCTS_ITERATIONS=1000
AGZ_DISCOUNT_FACTOR=0.995
AGZ_XL_CONTROLLER_MAX_REPLAY_STATES=20000000
AGZ_XL_ADVERSARY_MAX_REPLAY_STATES=15000000
```

Launch scripts must additionally set the intended output root, training gate,
sample caps, horizon, and parallelism. Do not infer a historical EXP2 setting
from a current module default. `full_tree_rollout` and `AGZ_ROLLOUT_*` belong
to EXP3 and are inactive when `full_tree` is selected.

### End-to-End Ownership

```text
deploy.py / cluster.py
  -> select EXP2, sync/build code, and start XL plus eight workers
worker_daemon.py / cpp_selfplay_runner.py
  -> load the promoted role bundle, run native games, and finalize replay shards
durable_transfer.py
  -> checksum, publish, acknowledge, and retire shards atomically
xl_coordinator.py
  -> ingest role FIFO partitions, prune at 20M/15M, gate and launch training
agz_train_eval_promote.py
  -> sample, incrementally train four DNNs, export, evaluate, and promote roles
```

Workers generate replay only with the currently promoted controller and
adversary. Candidate training may continue from the latest candidate checkpoint,
but failed candidates are not broadcast to self-play. Both role-specific value
models predict controller-perspective cost-to-go.

### AlphaGoZero Python Files

| File | EXP2 responsibility |
| --- | --- |
| `config.py` | Environment-backed replay, training, model, MCTS, discount, noise, and resource defaults. |
| `cluster.py` | Maps `AGZ_CLUSTER=exp2` to EXP2 XL and its eight workers; contains SSH/rsync helpers. |
| `deploy.py` | Syncs source, builds the native extension, launches daemons, and forwards experiment environment variables. |
| `worker_daemon.py` | Supervises process-isolated self-play, switches models between games, builds shards, and uploads them. |
| `cpp_selfplay_runner.py` | Converts worker/config arguments to the native arena command and logs the effective setup. |
| `durable_transfer.py` | Implements manifests, SHA256 validation, atomic publication, acknowledgement, and safe deletion. |
| `replay_runtime.py` | Records full trajectories and writes backward discounted value targets and canonical policy rows. |
| `xl_coordinator.py` | Drains incoming shards, role-splits FIFO replay, evicts oldest partitions, writes distribution/gate status, and owns trainer state. |
| `prebuild_replay_caches.py` | Multiprocess construction of reusable state and policy training caches. |
| `agz_train_eval_promote.py` | Samples replay, trains/exports four artifacts, runs evaluation, promotes roles independently, and publishes bundles. |
| `dnn_models.py` | Value/policy architectures, losses, optimizer warm starts, checkpoints, and deterministic native export. |
| `markov_value_features.py` | Python `markov_v2` serialization, scaling, invariance contract, and validation. |
| `distributed_eval.py` | Partitions role/SJF games across hosts, stages exact artifacts, validates coverage, and merges results. |
| `bootstrap_untrained_dnn_v100.py` | Creates the neutral native-ready v100 DNN bundle. |
| `bootstrap_markov_v100.py` | Builds or validates a v100 bundle using the Markov value schema. |
| `distill_dnn_v100.py` | Optional classical-to-DNN v100 distillation; not used by the neutral-v100 run. |
| `test_experiment_configuration.py` | Guards cluster, discount, MCTS, and replay-target argument propagation. |
| `test_distributed_eval.py` | Guards game partitioning, complete coverage, staging, and merged outputs. |
| `test_and_analysis/test_dnn_native_parity.py` | Python/native single and batched value/logit parity. |
| `test_and_analysis/test_dnn_mcts_parity.py` | Deterministic noise-disabled Python/native MCTS parity. |
| `test_and_analysis/test_markov_value_features.py` | Markov dimensions, invariance, validation, and Python/native feature parity. |
| `test_and_analysis/smoke_resource_fixes.py` | FIFO ingestion, bounded replay, streaming reads, policy alignment, and cleanup smoke tests. |

### Native Call Path

```text
arena_mcts_value_runnerCPP.py
  -> mcts_native_gv2 (pybind)
  -> pybind_module.cpp parses simulator/search/model payloads
  -> gv2_mcts_value_prior.cpp dispatches value+prior search
  -> gv2_mcts_dnn.cpp runs full-tree DNN PUCT
  -> gv2_virtual_environment.cpp applies exact GV3 transitions and rewards
  -> replay_runtime.py receives trajectory and root diagnostics
```

The Python runner is orchestration/logging glue. EXP2 native search does not
call Python or Torch for model inference.

### Native Files

| File | Responsibility |
| --- | --- |
| `vidur/Game_Version3_Cpp/CMakeLists.txt` | Builds `mcts_native_gv2` from simulator, feature, DNN, MCTS, self-play, and pybind sources. |
| `include/gv2_mcts_dnn.hpp` | Declares DNN MCTS options/results and the full-tree interface. |
| `src/gv2_mcts_value_prior.cpp` | Public value/prior wrapper and model-family dispatch used by arena integration. |
| `src/gv2_mcts_dnn.cpp` | Canonical expansion, batched policy logits, PUCT/root noise, leaf `V(s)`, time-discounted backup, Q/visits, and most-visited selection. EXP2 uses `full_tree`. |
| `agz_dense_dnn.hpp/.cpp` | Loads `dnn_models.py` exports and performs FP32 residual value/policy inference without Python. |
| `markov_value_features.hpp/.cpp` | Native `markov_v2` extraction, normalization, validation, and variable-length token construction. |
| `new_features_226_inference.hpp/.cpp` | Legacy 226D state plus controller/adversary action features used by policy and compatibility paths. |
| `gv2_player_sample_actions.hpp/.cpp` | Generates legal actions and canonical identities before policy scoring. |
| `gv2_virtual_environment.hpp/.cpp` | Search-state copies, controller/adversary transitions, batch timing, ticks, fast-forward, reward/cost, and duration. |
| `gv2_credits.hpp/.cpp` | Decode-credit ledger included in transition state. |
| `gv2_infer_runtime.hpp/.cpp` | Retained non-DNN native inference runtime. |
| `gv3_native_selfplay.hpp/.cpp` | Native self-play loop and trajectory/root packaging. |
| `src/pybind_module.cpp` | Python/C++ ABI, payload conversion, feature/inference parity helpers, and result diagnostics. |

`vidur/Game_Version3/mcts_value_prior.py` is the deterministic non-rollout
Python MCTS reference. `vidur/Game_Version3/virtual_environment.py` is the
Python transition reference. Any feature, action, reward, discount, model
export, or backup change requires matching Python/native tests.

### Consistency Map

| Change | Files that must remain consistent |
| --- | --- |
| Replay admission/capacity | `config.py`, `xl_coordinator.py`, resource smoke tests. |
| Training gates/sample sizes | `config.py`, `xl_coordinator.py`, `agz_train_eval_promote.py`. |
| DNN/loss/export | `dnn_models.py`, `agz_dense_dnn.*`, DNN parity tests. |
| Markov value state | `markov_value_features.py`, `markov_value_features.*`, pybind helper, feature tests. |
| Policy/action features | Python feature builder, `new_features_226_inference.*`, action parity tests. |
| MCTS semantics | Python reference, `gv2_mcts_value_prior.cpp`, `gv2_mcts_dnn.cpp`, MCTS parity tests. |
| Transition/reward timing | Python/native virtual environments and game-engine trace tests. |
| Distributed evaluation | `distributed_eval.py`, `agz_train_eval_promote.py`, `worker_daemon.py`, eval tests. |

## File Map

### `config.py`

Central defaults: output roots, initial model paths, promotion threshold, replay caps, sample caps, self-play/eval MCTS defaults.

### `cluster.py`

Defines XL and worker hostnames and basic SSH/rsync helpers.

### `deploy.py`

Deployment helper for syncing code and launching/stopping AlphaGoZero daemons on the cluster.

### `durable_transfer.py`

File-transfer and integrity primitives:

- SHA256 generation/verification.
- Atomic JSON writes.
- CSV append helpers.
- Shard manifest creation.
- Finalize active worker replay into ready shards.
- Rsync helpers.
- Remote verify/publish.
- Ack polling.

This is the source of truth for safe replay movement between machines.

### `worker_daemon.py`

Worker-side self-play daemon. It launches C++ game processes, merges finished game replay into local `active/`, finalizes ready shards, uploads them to XL, waits for acks, and deletes local completed shards and game run dirs.

Key behavior:

- Process-per-game isolates native memory.
- Default parallelism is high, normally 60 games per worker.
- Existing games keep the model they launched with.
- New games use the newest worker `models/current_model.json`.

### `xl_coordinator.py`

XL-side coordinator daemon. It ingests replay shards, updates replay counters, prunes replay partitions, writes training-gate status, launches one trainer at a time, and reconciles promotion state.

Key behavior:

- Does not ingest from `incoming_uploading/`.
- Does not write new monolithic replay CSVs.
- Does not launch a new trainer while a current train/eval/benchmark cycle is active.

### `agz_train_eval_promote.py`

XL-side candidate lifecycle:

1. Stream sample replay.
2. Train value model.
3. Train controller prior.
4. Train adversary prior.
5. Export native TSVs.
6. Append `train_model.csv`.
7. Run candidate-vs-promoted eval.
8. Promote if threshold is met.
9. Broadcast promoted artifacts.
10. Run SJF-256 benchmark.

The trainer explicitly deletes large arrays and model objects and calls `gc.collect()` before eval to avoid the prior OOM/memory creep issue.

### `replay_runtime.py`

Runtime replay recorder used by C++ arena integration. It stores full trajectories in memory for one game, computes discounted targets backward at cycle end, filters states with insufficient canonical actions, then appends:

- state/value rows to `replay_target_runtime.csv`
- per-canonical-action policy rows to `replay_policy_rows.csv`

### `cpp_selfplay_runner.py`

Small local smoke-test wrapper for launching the C++ arena runner with AlphaGoZero value/prior arguments. Useful for validating one or a few games outside the full daemon.

### `replay_target_smoke.py`

Older log-based replay target smoke tool. It builds replay rows from existing arena CSV logs. Production replay should come from `replay_runtime.py`.

### `test_and_analysis/smoke_resource_fixes.py`

Smoke test for the resource fixes:

- XL ingest writes prunable partitions.
- Accepted shard directories keep manifests only.
- Trainer can stream sample from partitions.
- Worker deletes per-game run dirs after merge.

### `test_and_analysis/calculate_discounted_reward.py`

Debug utility for manually recomputing discounted immediate-cost returns from arena CSV logs.

### `test_and_analysis/candidate_prefill_q_trend.py`

Reusable controller-candidate trend report across every available evaluation
version. It only includes controller decisions where
`prefill_remaining_by_id` contains a positive remaining-token count; canonical
action count is not used as a pending-prefill proxy. For each candidate it
compares the candidate-controller block with that evaluation's promoted
controller baseline and reports chosen-action, MCTS-visit, backed-up-Q, and raw
policy-prior preference for prefill actions. It also records promotion result,
training parent, value error, category margins, and residual decode choices.

Run it from the repository root whenever a historical or current trend is
needed:

```bash
python3 -m vidur.AlphaGoZero.test_and_analysis.candidate_prefill_q_trend \
  /path/to/experiment \
  --start-version 100 \
  --csv /path/to/experiment/candidate_prefill_q_trend.csv \
  --json /path/to/experiment/candidate_prefill_q_trend.json
```

An experiment normally has no `eval_000100` directory because v100 is the
initial baseline. The first row is therefore v101, whose baseline columns
measure promoted v100. Incomplete evaluations remain visible with an
`in_progress` status and should not be interpreted as failed candidates.

## Related Non-AlphaGoZero Files

### `vidur/Game_Version3/mcts_value_prior.py`

Python reference MCTS with HGB value and policy prior. Use this to understand algorithm semantics.

### `vidur/Game_Version3_Cpp/src/gv2_mcts_value_prior.cpp`

C++ production MCTS with value and priors. This is what arena/self-play uses for speed.

### `vidur/Game_Version3_Cpp/src/gv2_mcts_dnn.cpp`

Native value-only MCTS implementation. It was also changed to choose the root action by most visits and to log top action visit diagnostics where applicable.

### `vidur/bellman_v4_adv/arena_mcts_value_runnerCPP.py`

Python launcher around the native C++ arena. AlphaGoZero calls this module for games, eval, and benchmark.

## Results and Milestones

The important milestone is that promoted AlphaGoZero HGB models started beating strong SJF baselines, not just earlier promoted versions.

### Version 102

- Candidate beat previous promoted version by `62/100`.
- Promoted.

### Version 105

- Candidate beat previous promoted version by `66/100`.
- Promoted.
- SJF-256 arena logs were initially missing due to a logging flag; this was fixed and logs are now retained under `eval_<version>/SJF_256_Game/`.

### Version 109

Promotion:

- Candidate beat previous promoted version by `60/100`.
- Promoted.

SJF-256 benchmark:

- Model won `45/50`.
- SJF-256 trivial won `5/50`.

Local SJF-512 benchmark:

- Model won `30/50`.
- SJF-512 trivial won `11/50`.
- Ties: `9/50`.

Best-trivial comparison per game, where best trivial is `min(SJF-256 cost, SJF-512 cost)`:

- Model won `26/50`.
- Best trivial won `15/50`.
- Ties: `9/50`.

Relevant local files:

```text
simulator_output/GV3_Agent/AlphaGoZero/eval_of_models/eval_000109/
  sjf256_vs_sjf512_summary.csv
  model_vs_best_trivial_per_game.csv
  model_vs_best_trivial_summary.csv
  SJF_512_Game/arena_results.csv
```

### Version 124

Promotion:

- Candidate beat previous promoted version by `61/100`.
- Promoted.

SJF-256 benchmark:

- Model won `46/50`.
- SJF-256 trivial won `4/50`.

Local SJF-512 benchmark:

- Model won `33/50`.
- SJF-512 trivial won `12/50`.
- Ties: `5/50`.

Best-trivial comparison per game, where best trivial is `min(SJF-256 cost, SJF-512 cost)`:

- Model won `29/50`.
- Best trivial won `16/50`.
- Ties: `5/50`.

Relevant local files:

```text
simulator_output/GV3_Agent/AlphaGoZero/eval_of_models/eval_000124/
  sjf256_vs_sjf512_summary.csv
  model_vs_best_trivial_per_game.csv
  model_vs_best_trivial_summary.csv
  SJF_512_Game/arena_results.csv
```

Interpretation:

- SJF-512 is harder than SJF-256, but version 124 still wins a clear majority against SJF-512 alone.
- Against the per-game best of SJF-256 and SJF-512, version 124 still wins more games than the baseline.
- The trend from v109 to v124 improved both SJF-512 wins and best-trivial wins.

## Operational Notes

Memory:

- Worker game memory is reclaimed by process exit.
- Trainer memory is bounded by streaming replay sampling and explicit release before eval.
- If XL memory climbs during eval, check for orphaned trainer or arena processes first.

Disk:

- Worker disk should not grow beyond active replay plus unacked ready shards unless `--keep-game-runs` is set.
- XL disk should grow with replay partitions up to the replay cap plus model/eval artifacts.
- If disk grows unexpectedly, check for old `runs/`, `incoming_uploading/`, old accepted full shard copies, and monolithic replay CSVs.

Versioning:

- Candidate versions are computed from existing `Model_Version*` directories and the promoted version.
- An earlier bug skipped some version numbers, but it did not affect model semantics. The code now uses the next actual version safely.

Arena logs:

- Promotion eval logs candidate-vs-promoted summaries.
- SJF-256 benchmark logs are now retained for promoted versions.
- Root prior/visit behavior should be inspected in per-game arena CSV rows when debugging policy behavior.

## Known Constraints

- Prior inference makes MCTS slower than value-only MCTS. The policy model must score all canonical root actions at each newly expanded node.
- C++ inference is required for scale; Python MCTS is a reference implementation, not the production path.
- Current adversary policy sample cap is much smaller than controller cap because adversary roots are less frequent and more expensive to balance.
- `AGZ_ADVERSARY_POLICY_ROOT_SAMPLE_TARGET` can independently oversample adversary roots before the feature-complete policy cap is applied. A value of `400000` with `AGZ_ADVERSARY_POLICY_SAMPLE_CAP=350000` samples 400k replay roots and trains on at most 350k roots with canonical-action rows, without changing controller sampling.
- The promotion threshold measures head-to-head improvement against the current model, while SJF benchmark measures external strength.
- Ties in baseline comparison are counted explicitly and should not be silently assigned to either side.
