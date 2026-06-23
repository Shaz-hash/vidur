# Bellman V4 Adv 2.25M Parent Root Generation

## Purpose

Generate a larger fixed parent/root-state dataset for the GV3 Bellman V4 adversary-aware classical experiments.

This dataset is intended to replace the smaller repaired parent stores with a broader state distribution:

- More history depth coverage.
- More high-immediate-cost controller states.
- More diverse controller decision frontiers for Bellman iteration and feature/model training.

The dataset should still follow the current repaired parent-generation semantics:

- Store controller-to-act roots only.
- Refresh objective stats before scoring/storing roots.
- Compute first-layer no-bootstrap target with `mctsDNN.search_dnn(..., model_version=0, use_model_bootstrap=False)`.
- Store simulator snapshots, stats, frontier snapshots, history metadata, target pieces, and manifest shards through `ModelSearchBed/root_storage.py`.

## Target Dataset

Total accepted roots:

```text
2,250,000 controller roots
```

History hop range:

```text
0 to 750 nontrivial history hops
```

Target/high-signal filter:

```text
At least 40% selected/high-signal roots.
```

Selected/high-signal means:

```text
abs(target_value) >= 1.0
```

Because targets are controller-perspective and normally non-positive, this is usually equivalent to:

```text
target_value <= -1.0
```

In the current root-generation CLI this maps to:

```text
--target-abs-threshold 1.0
--min-large-abs-target-ratio 0.40
```

## Root Type

Only controller roots should be stored:

```text
root_player_filter = "controller"
```

Adversary roots may be traversed during history generation, but they are not accepted into this parent dataset.

## Deduplication

Use history signatures as before:

```text
deduplicate_history_signatures = True
allow_duplicate_history_fallback = False
history_signature_cache_size = 10000
```

Within each server/task, this prevents repeatedly storing the same history frontier signature.

Important limitation:

```text
If servers generate independently, cross-server duplicate signatures are not prevented unless we add a shared global signature exchange/merge step.
```

For this experiment the hop intervals are non-overlapping, which should substantially reduce duplicates across servers, but it is not a formal global uniqueness guarantee.

## Five-Server Split

Total roots are split evenly across five servers:

```text
2,250,000 / 5 = 450,000 roots per server
```

History hop range is split into five non-overlapping intervals over `[0, 750]`.

Current inclusive intervals used by the coordinator:

```text
server_0: hops    0-150
server_1: hops  151-300
server_2: hops  301-450
server_3: hops  451-600
server_4: hops  601-750
```

The first interval has 151 hop values because the global range is inclusive and contains 751 total hop values.

Each server should run local multiprocessing using its available cores. The root-generation process should use short-lived worker tasks as implemented in `root_storage.py`:

```text
num_processes > 1
worker_roots_per_task bounded
maxtasksperchild = 1
```

This is important because simulator/MCTS allocations can otherwise accumulate in long-lived processes.

## Responsible Files

Primary wrapper to launch root generation across servers:

```text
vidur/bellman_v4_adv_2000k_multiprocess/multi_server_root_generation.py
```

Per-server root generation entry point:

```text
vidur/Game_Version3/ModelSearchBed/analysis_testing/rootGeneration.py
```

Main generation/storage implementation:

```text
vidur/Game_Version3/ModelSearchBed/root_storage.py
```

History/frontier state generation:

```text
vidur/Game_Version3/DNN/history_root.py
```

No-bootstrap first-layer target computation:

```text
vidur/Game_Version3/mctsDNN.py
```

Next-phase child cache coordinator:

```text
vidur/bellman_v4_adv_2000k_multiprocess/multi_server_child_cache_generation.py
```

Next-phase parent feature coordinator:

```text
vidur/bellman_v4_adv_2000k_multiprocess/multi_server_parent_feature_building.py
```

Next-phase child feature coordinator:

```text
vidur/bellman_v4_adv_2000k_multiprocess/multi_server_child_feature_building.py
```

Underlying feature builders:

```text
vidur/bellman_v4_adv/build_state_local_features_adv.py
vidur/bellman_v4_adv/build_child_features_v4Adv.py
```

## Required Correctness Conditions

Before launching large generation, confirm the current code has:

- `refresh_root_objective_stats(env, state)` called before target scoring.
- Correct history decision-time logging.
- Correct controller action canonicalization if `mctsDNN.py` is used for target scoring.
- Current GV3 engine fast-forward behavior for pending adversary ticks.
- Existing game-engine trace tests passing on generated smoke traces.

## Expected Server Command Shape

Each server should run a command equivalent to:

```bash
python -m vidur.Game_Version3.ModelSearchBed.analysis_testing.rootGeneration \
  --output-dir <server_specific_output_dir> \
  --num-roots 450000 \
  --max-candidate-roots 22000000 \
  --num-processes <server_core_count> \
  --candidate-batch-size 256 \
  --generation-batch-size 8 \
  --shard-size 128 \
  --worker-roots-per-task 64 \
  --max-processes-per-interval 4 \
  --history-signature-cache-size 10000 \
  --min-large-abs-target-ratio 0.40 \
  --target-abs-threshold 1.0 \
  --history-hops-min <server_hop_min> \
  --history-hops-max <server_hop_max> \
  --history-max-total-steps 20000 \
  --seed <server_unique_seed>
```

`max_candidate_roots` should be chosen high enough to satisfy the 60% selected-target quota at each hop interval. Deeper intervals may require more candidates.

## Expected Outputs Per Server

Each server output directory should contain:

```text
root_generation_config.json
manifest.jsonl
summary.json
summary.csv
roots_000000.pt
roots_000001.pt
...
```

Each shard stores root records containing:

```text
simulator_snapshot
stats
frontier_simulator_snapshot
frontier_stats
root_player
root_depth
history_hops
history_signature
target_value
best_action_index
best_action_repr
best_reward
best_discount
best_bootstrap
best_child_cost
best_child_time
```

## Merge Plan

After all five servers finish:

1. Sync all server output directories back to the main machine.
2. Validate each `manifest.jsonl` and shard count.
3. Confirm each server has `450000` accepted roots.
4. Confirm selected/high-signal ratio is at least `0.60`.
5. Optionally run a cross-server signature duplicate audit.
6. Merge shards into one canonical parent dataset directory with rewritten manifest entries.

## Next Phase: Child Cache And Features

Once the root dataset is complete, each server should build the next artifacts locally for its own partition first. This avoids moving large simulator snapshots between machines before deriving child transitions/features.

Child transition cache:

```bash
python vidur/bellman_v4_adv_2000k_multiprocess/multi_server_child_cache_generation.py launch \
  --num-processes 64 \
  --parents-per-task 1000 \
  --transition-shard-size 4096
```

This calls:

```bash
python -m vidur.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv
```

Parent features:

```bash
python vidur/bellman_v4_adv_2000k_multiprocess/multi_server_parent_feature_building.py launch \
  --num-processes 64
```

Child features:

```bash
python vidur/bellman_v4_adv_2000k_multiprocess/multi_server_child_feature_building.py launch \
  --num-processes 48 \
  --flush-every-shards 256
```

The child feature builder writes fixed `.npy` outputs incrementally using bounded in-flight shards, so it should not retain the full child-state workload in process memory.

Small local smoke validation already checks:

```text
2 parent roots -> 24 child transitions
parent feature shape -> (2, 226)
child feature shape -> (24, 226)
```


## Multiprocess Pipeline File Map

This directory contains coordinators and local runners for the full Bellman V4 Adv data/model pipeline. The scripts should be kept role-specific so future runs do not accidentally mix controller roots, adversary roots, value targets, and policy-prior targets.

### Dataset Generation

```text
multi_server_root_generation.py
```

Generates the main controller parent/root-state dataset across the bellman workers. It launches `rootGeneration.py` remotely, uses non-overlapping history-hop intervals, stores only `root_player="controller"` roots by default, and enforces the high-signal target ratio through `min_large_abs_target_ratio` and `target_abs_threshold`.

```text
multi_server_adv_root_generation.py
```

Generates adversary parent/root states for adversary policy-head work. It uses the same lower-level `rootGeneration.py`/`root_storage.py` path, but passes `--root-player-filter adversary` and `--min-canonical-actions 2`, so every accepted adversary state has more than one canonical action available.

```text
count_remote_roots.py
```

Lightweight status/audit helper for parent root datasets on remote machines. Use it to count completed roots per worker without manually SSHing into every output directory.

```text
count_adv_remote_roots.py
```

Lightweight status/audit helper for the adversary root dataset on workers 5-8. It points at `bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon` by default and reports completed root shard counts per worker.

### Current Adversary Root Generation Run

Purpose:

```text
Generate 300,000 adversary-to-act root states for adversary policy-head training.
```

Acceptance criteria:

```text
root_player_filter = "adversary"
min_canonical_actions = 2
```

This means each accepted adversary root must have more than one canonical action available.

Workers and hop partitions:

```text
bellman-classical-worker-5: hops   0-100, 75,000 roots
bellman-classical-worker-6: hops 100-200, 75,000 roots
bellman-classical-worker-7: hops 200-300, 75,000 roots
bellman-classical-worker-8: hops 300-400, 75,000 roots
```

Large-run generation settings:

```text
num_processes = 64
worker_roots_per_task = 64
candidate_batch_size = 256
generation_batch_size = 8
shard_size = 128
max_candidate_multiplier = 100
include_history_trace_logs = false
```

Launch command:

```bash
python vidur/bellman_v4_adv_2000k_multiprocess/multi_server_adv_root_generation.py launch \
  --hosts bellman-classical-worker-5,bellman-classical-worker-6,bellman-classical-worker-7,bellman-classical-worker-8 \
  --hop-intervals 0-100,100-200,200-300,300-400 \
  --roots-per-server 75000 \
  --max-candidates-per-server 0 \
  --max-candidate-multiplier 100 \
  --num-processes 64 \
  --worker-roots-per-task 64 \
  --candidate-batch-size 256 \
  --generation-batch-size 8 \
  --shard-size 128 \
  --progress-every 1000 \
  --force
```

Progress command:

```bash
python vidur/bellman_v4_adv_2000k_multiprocess/count_adv_remote_roots.py
```

Remote output base:

```text
/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/ModelSearchBed/bellman_v4_adv_300k_adversary_roots_hops0_750_min2canon
```

Smoke validation before the large run:

```text
40 adversary history trace CSVs were generated on workers 5-8 and pulled locally.
Game engine validation passed: files=40, traces=40, adv_actions_checked=1551.
Local trace directory:
/home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/GV3_Agent/history_adv_traces
```

### Child Cache And 226D Feature Building

```text
multi_server_child_cache_generation.py
```

Builds child transition caches for each worker's parent-root shard. It expands each stored parent root into valid child transitions and writes transition shards locally on the same machine to avoid moving large simulator snapshots before derivation.

```text
multi_server_parent_feature_building.py
```

Builds 226D value-function state features for parent roots. These are the value-model inputs for the current state.

```text
multi_server_child_feature_building.py
```

Builds 226D value-function state features for cached child transitions. These are needed for Bellman target updates because each parent action evaluates `reward + discount * V(child)`.

```text
assemble_parent_child_dataset.py
```

Assembles/synchronizes parent features, child features, and target files so each training machine can see the complete dataset. Use this before model training if the training job needs all shards, not just its local worker partition.

### Value Model Training

```text
multi_server_HGB_train_bellman_model.py
```

Coordinates multi-server HGB value-model training. Each server is assigned a set of HGB capacities and alpha settings. It writes per-iteration model directories and train/eval CSVs, then these can be pulled back into `simulator_output/GV3_Agent/bellman_multiserver_HGB`.

```text
multi_server_hgb_feature_building_training.py
```

Legacy/combined feature-building and training helper. Prefer the separated parent-feature, child-feature, assembly, and HGB training scripts above for the current pipeline unless this file is intentionally being revived.

### MCTS Visit Targets For Policy Heads

```text
multi_server_hgb_Mcts_value_function_targets.py
```

Generates MCTS visit targets from stored roots using the native C++ MCTS and an HGB value-model bootstrap. For controller policy targets, run it with the controller root experiment and `--root-player-filter controller`. For adversary policy targets, reuse the same file with the adversary root experiment and `--root-player-filter adversary`. It writes one row per canonical action with `visit_count`, `visit_prob`, action index, root metadata, and timing/root-value summaries.

```text
count_mcts_targets_remote.py
```

Counts remote MCTS target progress by reading target output directories and partial files. Use this while target generation is running to check how many roots/actions have finished per worker.

### Controller Policy-Prior Features And Training

```text
features_value_and_prior.md
```

Feature specification document. It records the 226D value-state features and the action-feature design for policy/prior training, including slot-aligned prefill action features and canonical-action-compatible eviction summary features.

```text
multi_server_hgb_controller_prior_feature_bulding.py
```

Builds action features for each canonical controller action in each controller root. It should produce one action-feature row per MCTS target row and sanity-check that `feature_action_count == canonical_action_count`.

```text
multi_server_hgb_training_controller_prior.py
```

Trains the controller HGB policy/prior head once action features and MCTS visit targets are available. The expected target transforms are `visit_prob = visits / sum(visits)` and centered logits `log(visits + alpha) - mean(log(visits + alpha))`, with `alpha=1` unless explicitly changed.


### Controller Policy-Prior Training Mechanism

The controller policy/prior head is trained as a state-action scorer, not as a fixed action-id classifier. This is necessary because each root has a variable number of canonical actions.

Inputs used during assembly:

```text
226D state features
```

These are already produced by the parent feature pipeline and stored as NumPy arrays such as `parent_features.npy` with accompanying metadata. Each row corresponds to one controller root state.

```text
43D action features
```

These are produced by `multi_server_hgb_controller_prior_feature_bulding.py`. The full output is currently written as `target_smoke_feature.csv` for historical naming reasons. Each row corresponds to one `(state_id, canon_action_index)` and contains `action_feature_repr`, a JSON representation of the 43D action vector.

```text
MCTS visit targets
```

These are produced by `multi_server_hgb_Mcts_value_function_targets.py`. For each `(state_id, canon_action_index)`, the raw visit count is converted into:

```text
visit_prob = visits / sum(visits over canonical actions for this root)
centered_logit = log(visits + alpha) - mean(log(visits + alpha))
```

Use `alpha=1` by default.

Assembly logic:

```text
For each canonical action row:
    state_vec  = parent_features[state_id]          # 226D
    action_vec = parse(action_feature_repr)         # 43D
    X_row      = concat(state_vec, action_vec)      # 269D
    y_row      = centered_logit target from MCTS
```

The assembled training matrix is therefore row-level over canonical actions, with shape approximately:

```text
num_canonical_action_rows x 269
```

Training model:

```text
HistGradientBoostingRegressor(X_269D -> centered_logit)
```

Current planned configs are approximately 200k-parameter HGB regressors:

```text
hgb_policy_47leaf_1410iter
hgb_policy_63leaf_1050iter
hgb_policy_95leaf_0700iter
```

Evaluation metrics:

```text
row-level MSE/RMSE/MAE against centered_logit
top1_match: argmax predicted score == argmax MCTS visits
top3_contains_best
cross_entropy / KL after softmaxing predicted scores within each root
```

Inference usage:

```text
For a new controller state:
    build the 226D state vector once
    enumerate canonical controller actions
    build the 43D action vector for each canonical action
    score each concat(state, action) row with the HGB policy model
    rank or softmax scores within the root to select/prioritize actions
```

### Arena And Evaluation Runners

```text
runner.py
```

Python Bellman one-step arena runner. It selects actions by evaluating immediate reward plus discounted model bootstrap over valid children, without doing full MCTS iterations.

```text
runnerCPP.py
```

C++/native equivalent path for the one-step Bellman runner. It exists so the reward-plus-bootstrap action-selection arena can run faster while staying aligned with Python.

```text
launch_latest_hgb_sjf256_arena.py
```

Launch helper for arena runs against SJF-256 using the latest pulled HGB value models. This is for non-depth1 or older arena invocation style.

```text
launch_latest_hgb_sjf256_depth1cpp_arena.py
```

Launch helper for depth-1/native C++ arena runs against SJF-256 using the latest pulled HGB value models. This is the preferred helper for the current native C++ arena experiments when comparing many latest model versions.

## Open Questions Before Launch

- Exact five server hostnames.
- Exact core count to use per server.
- Candidate budget per interval.
- Whether cross-server global signature deduplication is required, or non-overlapping hop intervals are sufficient.
- Whether to update `mctsDNN.py` controller canonicalization to match the newer effect-only `mcts.py` key before generation.
