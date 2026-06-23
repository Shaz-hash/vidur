# V1 Classical-Model Pipeline (`experiments/new_features_v1/`)

This directory holds every script used on **`bellman-classical`** (3.218.145.231,
`/home/ubuntu/vidur-classical-search`) to train and evaluate the GV3 controller
value function under the 200k effective-scalar-param budget defined in
`TASK.md`. The local copy here is a mirror of the remote, plus a few inspect/plot
one-offs that never ran on remote.

The artifacts those scripts produced live under
`simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v{1..6}/` —
also synced from remote. Every file in this folder is paired with a directory
in that tree (mapping is in [§ 6](#6-output-tree-on-remote)).

---

## 1. Top-level pipeline

```
                        manifest of stored root shards
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       │                             │                             │
       ▼ (parent rows)               ▼ (child rows / per action)   ▼ (logs)
  build_state_local_features.py  build_child_features_v4.py     n/a
  build_state_local_features_v5  (parent's children for         (results.csv,
  build_state_local_features_v6   bellman bootstrap)             metadata.json)
       │                             │
       ▼                             ▼
  features_d{1,2}.npy           v4_child_d{1,2}/child_features.npy
  targets_d{1,2}.npy            v4_child_d{1,2}/child_meta.npy
  meta_d{1,2}.json              v4_child_d{1,2}/parent_index.npz
  split.json
       │
       ├─── train_v1_supervised*.py ─────── HGB regressors / hierarchical / ensemble / bucket
       │                                      │
       │                                      ▼
       │                              cached_state_local_v4/<run>/<config>/model.joblib
       │
       └─── bellman_iterate_v4.py ────────── V_n iteration loop using V_{n-1} on children
                                              │
                                              ▼
                                cached_state_local_v4/V_hier_iter/Model_Version{n}/{model.joblib,
                                  train_results.csv, eval_results.csv,
                                  *_same_version.csv, metadata.json}
                                              │
                                              ▼
                                run_arena_v_n.sh / run_arena_v1_hier.sh
                                  (wraps model in V4HGBWrapper / V4HierWrapper,
                                   runs Model_Tester arena vs SJF-128/256/512)
```

The Bellman loop is the closing of the cycle: V_n is trained on
target_n(parent) = max_a [r_a + γ_a · V_{n-1}(child_a)], using child features
already built once by `build_child_features_v4.py`.

---

## 2. Feature builders

All builders write `(features.npy, targets.npy, meta.json)` per dataset
("d1" = full data, "d2" = held-out eval). They are deterministic and parallel
over root shards. `feature_dim` grew across versions but the parent schema is
stable inside each version (24 globals at the front, then per-prefill and
per-decode slot blocks).

| File | Schema | feature_dim | Out dir |
|---|---|---|---|
| `build_state_local_features.py` | v4 base — 21 globals + 7×25 prefill + 7×4 decode | **224** | `cached_state_local_v4/` |
| `build_state_local_features_v5.py` | v4 + 4×5 analytical lookahead per token-budget B∈{128,256,512,1024} | **244** | `cached_state_local_v5/` |
| `build_state_local_features_v6.py` | v5 + 4 rescue scalars (smallest B that zeros violations) | **248** | `cached_state_local_v6/` |
| `build_child_features_v4.py` | v4 schema for **every child** of every parent | 224 | `<run>/v4_child_d{1,2}/` |

Earlier checkpoints (v1, v2, v3) live in their own dirs and were used while
iterating on the global-block design (slack buckets, EDF-margin). The current
champion features (used in every recent training and arena run) are **v4**.

### Run on remote

```bash
ssh bellman-classical
cd /home/ubuntu/vidur-classical-search && export PYTHONPATH=$PWD

# Parent features (D1, D2 split)
python -m experiments.new_features_v1.build_state_local_features \
  --manifest simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40/manifest.jsonl \
  --shard-dir simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40 \
  --root-player-filter controller \
  --num-processes 48 \
  --out-dir simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4

# Child features (used by bellman_iterate to bootstrap V on every child)
python -m experiments.new_features_v1.build_child_features_v4 \
  --child-cache-dir simulator_output/GV3_Agent/<child_transitions_dir> \
  --out-dir simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4/v4_child_d1 \
  --num-processes 48
```

Logs from these runs are in `cached_state_local_v4/build_d{1,2}.log`.

---

## 3. Trainers

All trainers expect the v4 (or v5/v6) feature dump and `split.json` produced by
the builder. Each writes one or more configs into a sibling directory under
`cached_state_local_v4/<run_name>/<config>/`. Param budgets are tracked via
`_hgb_n_params(model) = 3 × Σ leaves`.

| File | Architecture | Param budget | Out subdir |
|---|---|---|---|
| `train_v1_supervised.py` | Initial single-HGB sweep (sq + abs loss + small ExtraTrees) | < 120k | `V1_models/` |
| `train_v1_supervised_xl.py` | Three near-120k HGB shapes: 31×1280, 47×850, 63×630 | ~120k | `V1_models_xl/` |
| `train_v1_supervised_xl_alpha.py` | XL shapes at a single fixed `tail_alpha` (script arg) | ~120k | `V1_models_xl_alpha{1,2}/` |
| `train_v1_supervised_200k.py` | Capacity sweep at 200k: leaves × iters × tail_alpha × loss | ~200k | `V1_models_200k_*/` |
| `train_v1_supervised_200k_targeted.py` | 200k with α=0 / β-boost / abs-loss variants for cliff handling | ~200k | `V1_models_200k_targeted{,2}/` |
| `train_v1_bucket_classifier_200k.py` | Multiclass HGBClassifier over y∈{−7..0} buckets, then bucket-mean (and optional residual reg) | ~200k | `V1_bucket_200k{,_v2,_soft}/` |
| `train_v1_hierarchical_200k.py` | HGBClassifier `g(x)=P(y≥−0.05)` + HGBRegressor `r(x)` on `y<−0.05`; `y_hat = 0 if g>τ else min(r,0)`. Sweeps τ ∈ {0.50, 0.70, 0.85, 0.95} | ~50k cls + ~150k reg | `V1_hier_200k{,_v2}/` |
| `train_v1_ensemble_200k.py` | 0.5·(HGB_A + HGB_B): A weighted by tail magnitude, B weighted by cliff-escapee mass | 2 × ~100k | `V1_ensemble_200k/` (if it exists) |

Each config dir contains:

```
model.joblib                 — sklearn estimator (or hier dict/ensemble dict)
train_results.csv            — per-row predictions on D1
eval_results.csv             — per-row predictions on D2
metadata.json                — n_params, MSE/RMSE/MAE/p95_abs/max_abs (train+eval), config
```

### Pass / gate criteria (TASK.md)

Per-iteration model passes if **on both train and eval**:
- MSE < 0.05, RMSE < 0.05, MAE < 0.05, p95_abs < 0.05, max_abs < 0.5

Champion so far is `V1_hier_200k/.../τ=0.95` (cls+reg) — passes on most metrics
but max_abs sits ~1.7–1.9 at the Bellman fixed point, not under 0.5.

### Run on remote

```bash
ssh bellman-classical
cd /home/ubuntu/vidur-classical-search && export PYTHONPATH=$PWD

# 200k capacity sweep on v4 features
python -m experiments.new_features_v1.train_v1_supervised_200k \
  --features-d1 simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4/features_d1.npy \
  --targets-d1  simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4/targets_d1.npy \
  --features-d2 simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4/features_d2.npy \
  --targets-d2  simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4/targets_d2.npy \
  --split-json  simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4/split.json \
  --output-dir  simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4/V1_models_200k

# Hierarchical 200k
python -m experiments.new_features_v1.train_v1_hierarchical_200k \
  --features-d1 ... --targets-d1 ... \
  --features-d2 ... --targets-d2 ... \
  --split-json ... \
  --output-dir simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4/V1_hier_200k
```

Each `train_*` run on remote dumped its stdout to `<out_dir>.log` (e.g.
`V1_models_200k.log`).

---

## 4. Bellman iteration

`bellman_iterate_v4.py` does the value-iteration loop in N×D batches.

For each `n` in `[start..end]`:

1. Load V_{n-1} from `Model_Version{n-1}/model.joblib` (or compute the
   non-bootstrap target on iter 1: `target_1 = act_max_reward`).
2. Predict V_{n-1} on every child in `v4_child_d{1,2}/child_features.npy`.
3. Reduce to per-parent `target_n(p) = max_a [reward_a + γ_a · clip_max(V_{n-1}(child_a), 0)]`
   over the `is_valid==1` children indexed by `parent_index.npz`.
4. Train HGB with `sample_weight = 1 + tail_alpha · |target|`.
5. Save `Model_Version{n}/{model.joblib, train_results.csv, eval_results.csv, metadata.json}`.
6. **Same-version trick**: at the start of iter n+1, after computing
   `target_{n+1}` (which uses V_n on the child side, i.e. is the same-version
   target for V_n), write
   `Model_Version{n}/{train,eval}_results_same_version.csv` with
   `y_true = target_{n+1}` and `y_pred = V_n(parent)` from the cached forward
   predictions. Saves a second 30M-row pass per iter.

V1 bootstrap also supports the **hierarchical V1** wrapper as the iter-1
seed (`_hier_predict_block` + `_is_hier` detect it; classifier+regressor
inference matches `V4HierWrapper`).

### Run on remote

```bash
ssh bellman-classical
cd /home/ubuntu/vidur-classical-search && export PYTHONPATH=$PWD

# Continue iter 2..50 from a hierarchical V_1
python -m experiments.new_features_v1.bellman_iterate_v4 \
  --features-d1 .../cached_state_local_v4/features_d1.npy \
  --features-d2 .../cached_state_local_v4/features_d2.npy \
  --child-d1    .../cached_state_local_v4/v4_child_d1 \
  --child-d2    .../cached_state_local_v4/v4_child_d2 \
  --split-json  .../cached_state_local_v4/split.json \
  --start-iter  2 --end-iter 50 \
  --bootstrap-from .../cached_state_local_v4/V1_hier_200k/v4_hier_wrapper.joblib \
  --tail-alpha 2.0 \
  --max-leaves 47 --max-iter 850 \
  --out-dir    .../cached_state_local_v4/V_hier_iter
```

The 47-leaf × 850-iter shape (~120k params) was the champion arena/Bellman
trade-off; the run-name suffix `47x850_a2` traces back to that.

---

## 5. Arena drivers

The arena harness is `vidur.mcts.Game_Versions.Game_Version3.Model_Tester.run`
(under the main project tree). Two thin shell wrappers in this folder pin the
"`v4_*_wrapper.joblib`" path and a hard-coded threading config:

| Script | Wraps | Notes |
|---|---|---|
| `run_arena_v_n.sh` | `Model_Version{N}/model.joblib` → `V4HGBWrapper` (auto-built if missing) | Generic per-iteration driver. Args: `ITER_N SJF_BUDGET NUM_GAMES ARENA_PROCESSES RUN_DIR_NAME` |
| `run_arena_v1_hier.sh` | `V1_hier_200k/v4_hier_wrapper.joblib` (`V4HierWrapper`) | The first-iter hierarchical model only |

Both use `--arena-time-limit-sec 5.0`, `--arena-num-processes 25`,
`--history-hops-min 0 --history-hops-max 100 --history-seed 2026`. They write
into `simulator_output/GV3_Agent/Model_Tester_Results/<NAME_SUFFIX>/`:

```
arena_results.csv                                  — one row per game
arena_games/game_<id>_*_trivial_ctrl.csv           — cycle1 (model_adv vs SJF)
arena_games/game_<id>_*_model_ctrl_depth1.csv      — cycle2 (model_adv vs model_ctrl)
```

### Run on remote

```bash
ssh bellman-classical
# V_5 arena vs SJF-128, 50 games on 25 procs, 5 sec/decision
bash experiments/new_features_v1/run_arena_v_n.sh 5 128 50 25 V_hier_iter

# V_1 hierarchical baseline arena vs SJF-512
bash experiments/new_features_v1/run_arena_v1_hier.sh 512 50 25
```

`parse_arena_results.py` summarises the resulting `arena_results.csv` (cycle-1
and cycle-2 cost means / wins). For per-game **discounted-return** comparison,
see `vidur/mcts/Game_Versions/Game_Version3/analysis/compare_discounted.py`
(local-only).

---

## 6. Output tree on remote

```
simulator_output/GV3_Agent/BellmanConvergence/
├── cached_state_local_v1/  ← initial 21-global schema, slack 6-bucket
├── cached_state_local_v2/  ← +17-bucket slack
├── cached_state_local_v3/  ← +decode_profile-driven decode_time_at_max
├── cached_state_local_v4/  ← +EDF-margin global; **active champion**
│   ├── features_d{1,2}.npy, targets_d{1,2}.npy, meta_d{1,2}.json
│   ├── split.json, build_d{1,2}.log
│   ├── v4_child_d{1,2}/
│   │   ├── child_features.npy, child_meta.npy, parent_index.npz
│   │   └── child_features.summary.json
│   ├── V1_models/, V1_models_xl/, V1_models_xl_alpha{1,2}/
│   ├── V1_models_200k{,_targeted,_targeted2}/, V1_bucket_200k{,_v2,_soft}/
│   ├── V1_hier_200k{,_v2}/  ← classifier+regressor + v4_hier_wrapper.joblib
│   └── V_hier_iter/
│       ├── Model_Version{1..50}/
│       │   ├── model.joblib
│       │   ├── train_results.csv, eval_results.csv
│       │   ├── train_results_same_version.csv, eval_results_same_version.csv
│       │   ├── metadata.json
│       │   └── v4_hgb_wrapper.joblib   (built lazily by run_arena_v_n.sh)
│       └── *.log
├── cached_state_local_v5/  ← v4 + 20-d analytical lookahead block
└── cached_state_local_v6/  ← v5 + 4 rescue scalars
```

Each `<run>.log` next to a run-dir is the captured stdout of the corresponding
training/Bellman invocation.

---

## 7. Wrappers (`vidur/.../ModelSearchBed/`)

The arena harness loads wrappers via `joblib.load`; they expose a stable
`infer_from_inputs(inputs)` interface and read the v4 feature vector from
`inputs.extras` (populated by
`vidur/mcts/Game_Versions/Game_Version3/DNN/infer.build_model_inputs` after
`enable_inputs_extras()` is called).

| Wrapper | Wraps | Built by |
|---|---|---|
| `v4_hgb_wrapper.py` (`V4HGBWrapper`) | A bare HGBRegressor | `pack_v4_hgb(raw_path, out_path, model_tag)` — auto-invoked from `run_arena_v_n.sh` |
| `v4_hier_wrapper.py` (`V4HierWrapper`) | `(cls, reg, tau)` triple | Written by `train_v1_hierarchical_200k.py` (look for `v4_hier_wrapper.joblib` next to the cls/reg pair) |
| `v15_wrapper.py` | Older neural V15; kept for reference | n/a |

---

## 8. Inspection / one-off scripts

These never ran on remote (or only locally) and are not part of the training
loop, but are useful when chasing a specific failure mode:

| File | Use |
|---|---|
| `inspect_outliers.py` | Dumps top-N eval-set outliers' raw + normalised features per row + per request slot |
| `parse_arena_results.py` | Cycle1/cycle2 summary from `arena_results.csv` |
| `plot_v100_vs_best_trivial.py` | Bar chart of model cost vs best-trivial cost across games (V100 era) |
| `plot_large_error_violins.py` | Per-feature violin plots on large-error rows |
| `repeat_controller_action0_until_cost.py` | Manual single-policy debug runner |
| `run_mcts_for_outlier_psid.py`, `run_mcts_for_large_error_psids.py` | Replays the MCTS root for a given parent state id to print the action ranking the model produced |

---

## 9. End-to-end sequence (the recipe used to get V_1..V_50)

```bash
ssh bellman-classical
cd /home/ubuntu/vidur-classical-search && export PYTHONPATH=$PWD

# 1) Build v4 parent features (D1, D2)
python -m experiments.new_features_v1.build_state_local_features \
  --manifest .../manifest.jsonl --shard-dir .../shards \
  --root-player-filter controller --num-processes 48 \
  --out-dir simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4

# 2) Build v4 child features for both datasets
python -m experiments.new_features_v1.build_child_features_v4 \
  --child-cache-dir .../child_transitions_d1 \
  --out-dir         .../cached_state_local_v4/v4_child_d1 --num-processes 48
# (repeat for d2)

# 3) Train V_1 hierarchical (cliff-classifier + cliff-regressor)
python -m experiments.new_features_v1.train_v1_hierarchical_200k \
  --features-d1 .../features_d1.npy --targets-d1 .../targets_d1.npy \
  --features-d2 .../features_d2.npy --targets-d2 .../targets_d2.npy \
  --split-json  .../split.json --output-dir .../V1_hier_200k

# 4) Bellman iterate V_2..V_50 starting from the hierarchical V_1
python -m experiments.new_features_v1.bellman_iterate_v4 \
  --features-d1 ... --features-d2 ... \
  --child-d1 .../v4_child_d1 --child-d2 .../v4_child_d2 \
  --split-json .../split.json \
  --start-iter 2 --end-iter 50 \
  --bootstrap-from .../V1_hier_200k/v4_hier_wrapper.joblib \
  --tail-alpha 2.0 --max-leaves 47 --max-iter 850 \
  --out-dir .../V_hier_iter

# 5) Arena V_5 vs SJF-128 (50 games, 25 procs)
bash experiments/new_features_v1/run_arena_v_n.sh 5 128 50 25 V_hier_iter
```

After step 5 you can pull `simulator_output/GV3_Agent/Model_Tester_Results/`
back to local and run
`PYTHONPATH=$PWD python -m vidur.mcts.Game_Versions.Game_Version3.analysis.compare_discounted`
to score the discounted-return win/loss table.
