# AlphaGoZero GV3 HGB System

This directory contains the AlphaGoZero-style training loop for GV3. The system uses the existing C++ game engine and C++ MCTS, with three HGB models per promoted version:

- `value`: predicts controller-valued state value from the 226D state feature vector.
- `controller_prior`: scores controller canonical actions from 226D state features plus 43D controller action features.
- `adversary_prior`: scores adversary canonical actions from 226D state features plus 7D adversary action features.

The current production model family is intentionally the single classical 200k-capacity HGB family because the earlier 200k and 400k HGB families behaved similarly in arena tests. The current config is:

```text
value:             hgb_sq_63leaf_1050iter_a2
controller_prior:  hgb_policy_63leaf_1050iter
adversary_prior:   hgb_policy_63leaf_1050iter
```

## Current Architecture

Classical XL is the coordinator, replay owner, trainer, evaluator, and promoter. The eight Bellman workers are self-play replay generators.

Workers:

- Keep a local shard of parent controller/adversary states.
- Launch C++ self-play games in parallel, normally `60` processes per worker.
- Start each game from a random parent state, without reusing the same parent state inside one local replay-buffer cycle.
- Use the latest promoted value/controller-prior/adversary-prior model bundle available on the worker.
- Store replay states where `canonical_action_count > 2`.
- Flush a local replay shard after about `10,000` replay states.
- Upload immutable replay shards to XL with checksums and wait for ack before deleting local ready shards.
- Delete per-game run directories after merge unless debug retention is explicitly enabled.

Classical XL:

- Receives worker shards under `incoming/`.
- Verifies manifests and SHA256 checksums.
- Ingests rows into replay partitions under `global_replay/partitions/<worker>/<shard>/`.
- Keeps replay bounded with `XL_MAX_REPLAY_STATES = 5,000,000`.
- Starts training once `TRAIN_TRIGGER_NEW_STATES = 200,000` new replay states arrive after the last completed train/eval/benchmark cycle.
- Trains one candidate model bundle.
- Evaluates candidate vs current promoted model.
- Promotes the candidate if it wins enough games.
- Broadcasts promoted model artifacts to workers.
- Runs SJF-256 benchmark games for promoted versions and keeps arena CSV logs.

## Important Defaults

These live in `config.py` and training/eval launch code.

```text
MAX_PROMOTIONS                 = 50
PROMOTION_WIN_RATE_THRESHOLD   = 0.57
TRAIN_TRIGGER_NEW_STATES       = 200,000
XL_MAX_REPLAY_STATES           = 5,000,000
VALUE_SAMPLE_CAP               = 500,000
CONTROLLER_POLICY_ROOT_CAP     = 200,000
ADVERSARY_POLICY_ROOT_CAP      = 50,000
MIN_ADVERSARY_STATES_FOR_EVAL  = 25,000

self-play mcts_iterations      = 1,000
self-play puct_c               = 1.5
self-play root noise           = enabled
self-play sample initial moves = enabled

eval mcts_iterations           = 1,000
eval puct_c                    = 1.0
eval root noise                = disabled
eval prior temperature         = 1.0
eval sample initial moves      = disabled
arena time limit               = 5 seconds
```

The value target is controller-valued. Good controller outcomes are less negative / higher. The adversary uses the same value model but selects actions from the adversary objective by inverting normalized exploitation during tree selection.

## End-to-End Loop

1. Workers generate replay using the current promoted model.
2. Each worker accumulates local active replay until the shard threshold is reached.
3. Worker finalizes the shard with a manifest and SHA256 checksums.
4. Worker uploads to XL `incoming_uploading/`, verifies remotely, publishes to `incoming/`, and waits for ack.
5. XL ingests the shard into a replay partition and writes an ack.
6. XL updates replay counters and training gate status.
7. If no train/eval cycle is running and enough new states arrived, XL starts `agz_train_eval_promote.py`.
8. Trainer streams replay partitions, samples bounded training sets, trains a candidate model bundle, writes metrics and artifacts.
9. XL evaluates candidate vs the current promoted model in both model-role assignments.
10. If the candidate reaches the promotion threshold, XL marks it promoted and broadcasts it to workers.
11. XL runs the SJF-256 benchmark for the promoted model.
12. Only after train + eval + optional benchmark complete, XL resets `new_states_since_last_training`.

This means if another 200k states arrive while training/eval/benchmark is still running, a second trainer is not launched concurrently. The next trainer becomes eligible only after the current full cycle is finalized.

## Replay Storage and Disk Boundaries

The replay design was changed after the first large run caused disk bloat.

Current intended behavior:

- Worker keeps only `active/` plus unacked `ready/` shards.
- Worker deletes each per-game output directory after `_merge_game_into_active()` unless `--keep-game-runs` is set.
- XL stores accepted replay as prunable partitions, not as one append-only monolithic CSV.
- XL accepted shard directories keep only small manifests/checksums by default.
- `XL_KEEP_ACCEPTED_SHARDS = False`.
- `XL_WRITE_AUDIT_REPLAY = False`.
- XL prunes oldest replay partitions to enforce `XL_MAX_REPLAY_STATES`.

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

- Value model samples up to `500,000` feature-complete replay states.
- Controller policy samples up to `200,000` controller roots.
- Adversary policy samples up to `50,000` adversary roots.
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
  value/hgb_sq_63leaf_1050iter_a2/model.joblib
  value/hgb_sq_63leaf_1050iter_a2/native_model.tsv
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

- `value_rmse`, `value_p95_abs_error`, `value_max_abs_error`.
- `controller_policy_cross_entropy`, `controller_policy_top1`, `controller_policy_top3`.
- `adversary_policy_cross_entropy`, `adversary_policy_top1`, `adversary_policy_top3`.

Policy cross entropy is computed after softmax over each root's canonical actions. It measures how well the model distribution matches the MCTS visit distribution, not whether the final arena policy is strong by itself.

## Candidate Evaluation and Promotion

Candidate eval compares the candidate bundle against the current promoted bundle using 100 games total across two role assignments:

```text
candidate controller vs promoted adversary
promoted controller vs candidate adversary
```

Each paired game uses the same start-state/hop plan. The candidate wins a pair when the candidate-controller leg has lower total SLO cost than the candidate-adversary leg. The model is promoted when:

```text
candidate_win_rate >= 0.57
```

After promotion:

- `state.json` is updated with the new promoted version.
- `current_model.json` is written for workers.
- Model artifacts are rsynced to each worker.
- XL runs 50 SJF-256 benchmark games under the promoted eval directory.
- Arena CSV logs are kept for SJF-256 benchmark games.

Eval output:

```text
simulator_output/GV3_Agent/AlphaGoZero/eval_of_models/eval_000<N>/
  candidate_controller_vs_promoted_adversary/arena_results.csv
  promoted_controller_vs_candidate_adversary/arena_results.csv
  promotion_eval_summary.csv
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

Self-play uses `puct_c = 1.5` to generate broader samples. Arena/eval uses `puct_c = 1.0`.

### Leaf Value and Backpropagation

There is no random rollout. The leaf value comes from the value model when `model_version > 0` and `use_model_bootstrap=True`; otherwise bootstrap is `0.0`.

Backpropagation uses Bellman-style controller-valued returns:

```text
value(parent -> child) = immediate_reward + discount * value(child)
immediate_reward       = parent_cost - child_cost
```

Because costs are bad, values are normally negative or zero. A child that reduces future cost is less negative / better for the controller.

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
- The promotion threshold measures head-to-head improvement against the current model, while SJF benchmark measures external strength.
- Ties in baseline comparison are counted explicitly and should not be silently assigned to either side.
