# Task: GV3 Controller Value Function, Classical and NUERAL Model Search

## Worktree Contract

Familarise/Refresh yourself with how the GV3 game runs : by reading this : /home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/mcts/Game_Versions/Game_Version3/readMe.md

You are assigned to this worktree only:

```text
/home/shazer/Desktop/Research/Vidur/vidur-classical-search
```

Expected branch:

```text
model-search-classical
```

Before editing files, run:

```bash
pwd
git branch --show-current
git status --short
```

Stop immediately if the directory or branch is not the expected one. Do not edit files in `/home/shazer/Desktop/Research/Vidur/vidur` or in any other worktree.

## Objective

Build the best either classical or Nueral Network model now you can for the GV3 controller value function on the fixed ModelSearchBed root-state dataset.

This is controller-perspective value learning only. The game is zero-sum/minimax, so we are not trying to learn a separate adversary value function here. Focus on `root_player == "controller"` records and predict `target_value`.

Allowed approaches include (everything is allowed including DNN side ):

```text
decision trees, random forests, extra trees, gradient boosted trees, linear models, generalized additive models, nearest-neighbor style methods, rule-based models, symbolic/hand-engineered formulas, feature crosses, calibration layers that are not neural networks.
```

OR/AND

```text
neural networks, MLPs, DNNs, attention, transformers, torch trainable neural modules, learned embeddings, differentiable NN-style representation learning.
```

Hard parameter/model-size budget IMPORTANT !!:

```text
effective trainable/scalar parameters <= 120,000
```

For tree models, count approximately by number of learned split/value scalars. Keep the final model compact and explain the model-size estimate.

## FEATURES :
You now have a lot of the data , around 650+k samples. And you have already generated children of these states. Ideally your features should be designed from the states of this dataset without creating features from the children. You may also use the prefill_profile.csv to create features that gives you a better idea about what batches and their consequences on the current states are if they are applied. You may use vidur's batch execution table that the GV3 in the cache dir to get batch execution times more accurately for prefill + decoode mixed workloads.


## Optimisations :
Now that you have the dataset , and their cached children formed, you may create features etc of both these states and their children at once for faster iterations etc. Iteration time shouldnt be long if you create the features once for both the dataset and their children. 

## Fixed Dataset


Dataset 1 — original 292k:
  - Path: simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40/
  - 292,713 controller records
  - Manifest: manifest.jsonl with 4,288 shards

  Dataset 2 — new 400k:
  - Path: simulator_output/GV3_Agent/model_search_roots_controller_extra_400k_abs1_ratio40_unique_from_292k_hops0_320/
  - 399,488 controller records
  - Manifest: manifest.jsonl with 15,160 shards (smaller shards: 128 records each vs 64 records each)

For the eval dataset, use the same 50k samples created from the earlier Dataset 1 — original 292k, basically the eval ratio was 0.2 , eval split seed : 12345


## Target Goals & Metrics

For all of the goals below, use the remote machine we have provided you, not this local machine. Once your goals are done, bring all the csvs in the simulator output of this classical branch from the remote machine ! For training the model, evaluation etc use multiple processes like we currently did in our previous classical approach

- The goal is extremely low error on the supervised learning like the current classical model is able to achieve for example both train and eval splits for every iteration 1 to n iterations, n could be 100 or 200 etc for example

```text
MSE < 0.05
RMSE < 0.05
MAE < 0.05
p95_abs_error < 0.05
max asbolute error < 0.5 on both training and eval
```

if at any iteration your model fails to get the error to meet these constraints then stop the experiment and revisit your model and features and loss and other hyper params.

- Secondly, your model should be designed such that max_abs_error clearly converges i.e. bellman convergence i.e, unlike the current classical model which is failing to bring the max absolute error to very small value such as around 0.5 or less . we need bellman convergence from your model as we go along the iterations ! and there is clear strong pattern of convergence. We need that your bellman convergence happens and becomes stabilised as you train the model with more iterations within the bounds above (i.e. < 0.5). If this does not happen, revisit your model and featues and hyper params 

- Create the csvs for each of the iteration like we currently do in the classical so that we can better understand that whether you were able to reach convergence or not. Save the model versions along each iteration if you were able to reach convergence.

- Once your model has reached convergence , use that model version in the Model Tester to run arena games where each game is of 5s length and run 50 games. Ideally your model should be able to have lower Cost in the end than best trivial policy that we currently have i.e. SJF 512. Right now we run games sequentially, it would be better if you change model tester code so that each game runs in parrallel, will give you more speed to get results quicker!

## Terminologies : 

- Iteration n-1 to n and its csvs are essentially using the previous model n-1 to provide us with the bootstrap value applied on the child state (supervised learning) and we train the model version n on it (except for the first iteraiton where the boostrap value is 0, hence controller value function learns the immediate max value action!). 
- Iteration n to n and its csvs are essentially the convergence (bellman convergence) happening, where the boostrap value comes from the model version n and we see how close is the prediction converging. Use this on the eval set to see if our max absolute error is within constraints or not  

## Clarifications & Plan (Q&A — reference, do not deviate)

These are the locked-in answers from the planning conversation. Re-read this section any time before changing approach.

### Feature rules

- FORBIDDEN as inputs to V: parent-side features that aggregate the parent's children
  (e.g. `la_max_qfloor = max_a [r + gamma * child_doomed_floor]` from prior v9/v10/v14 work).
- ALLOWED at iteration time: querying V at a child using the child's own self-features
  (the 498-d cliff+action+forecast schema). That is how the Bellman bootstrap target is
  computed: target_n(parent) = max_a [r + gamma * V_{n-1}(child_a)].
- ALLOWED parent-only inputs:
  - The 498-d cliff + action + forecast features extracted from the parent state.
  - On-the-fly transforms of those parent features (chunk-aware, realizability sigmoid,
    eta_to_tick ratio, etc.) computed from columns that exist in BOTH parent and child
    feature rows so train and inference stay consistent.
  - Analytical lookahead features built from `prefill_profile.csv`: e.g., for each candidate
    token_budget B in {128, 256, 512, 1024}, simulate one tick of greedy chunked prefill
    using the profile's batch-time table on the parent's pending prefills; report counts
    of prefills that miss SLO. Parent-only — NOT children.

### Target (label) per iteration

- Iteration 1 (V_0 = 0): target_1(parent) = max_a [reward_a + gamma_a * 0] = act_max_reward(parent).
  The dataset's stored `target_value` and `act_max_reward` are equal at iteration 1 by definition. Hence your model should be able to learn the max reward in V_0 accurately !! Your features and model design should support that ! Dont have it directly as a feature !
- Iteration n >= 2: target_n(parent) = max_a [reward_a + gamma_a * V_{n-1}(child_a)] using the
  cached children for both datasets.

### Eval split (locked, do not regenerate)

- Use the same 58,543 PSIDs from Dataset 1 split with `eval_ratio=0.2, split_seed=12345`
  (file: `simulator_output/GV3_Agent/BellmanConvergence/cached_v4_actfc_25v/split_indices.json`).
- Train = the other 234,170 PSIDs from Dataset 1 + all 399,488 PSIDs from Dataset 2 = 633,658.
- Total combined parents indexed = 692,201 (Dataset 1 first, Dataset 2 appended).

### Param budget

- Hard cap: <= 120,000 effective trainable scalar params.
- Counted as approx `3 * (sum of leaves across all trees)` for HGB-style ensembles.
- One model per iteration; V_n supersedes V_{n-1}. Budget applies to a single iteration's model.
- Initial target shape: HGB max_iter ~ 1500, max_leaf_nodes ~ 31 -> ~ 93k params.

### Iteration loop

- Save every iteration's V_n joblib for now (auditability).
- Forward CSVs (train_results.csv, eval_results.csv) emitted every iteration.
- Same-version CSVs (train_results_same_version.csv, eval_results_same_version.csv) emitted
  every k = 5 iterations as the convergence diagnostic.
- Once a k is found where same-version max < 0.5 on eval AND the prior ks already show a
  decreasing pattern in same-version max_abs_error (i.e. the maxes were trending downward
  toward < 0.5), switch to running same-version every iteration to confirm >= 5 consecutive
  iterations where the max stays < 0.5 (oscillation allowed if it stabilises within bounds).

### Pass / fail per iteration

Every iteration must satisfy on BOTH train and eval, on BOTH forward and same-version CSVs:

```
MSE < 0.05
RMSE < 0.05
MAE < 0.05
p95_abs_error < 0.05
max_abs_error < 0.5
```

If ANY iteration breaches max_abs_error < 0.5 (or any other constraint):

1. STOP the run.
2. Fix one of:
   (a) HGB hyperparams (max_leaf_nodes, max_iter, learning_rate, sample weighting, loss).
   (b) Add / refine features (e.g., new prefill_profile-derived features).
3. Restart from V_0 (rebuild from scratch).

### "Stabilised convergence" definition

- Search for the first k where same-version max_abs_error < 0.5 on eval, AND the prior ks
  show the pattern that same-version max_abs_error is essentially decreasing
  (monotone-ish downward trend toward < 0.5).
- From that k, run consecutive iterations with same-version every iteration to verify
  >= 5 consecutive iterations within the bounds (oscillation OK if it stays under 0.5).
- After convergence is detected, KEEP iterating to N = 100 or 200 to build long-term
  confidence in stability.

### Arena (after convergence)

- Use a converged V_n joblib in the Model Tester.
- Run 50 games x 5 sec each, vs SJF-512 trivial (the strongest baseline).
- Modify Model Tester to run games in parallel (currently sequential) to speed up.
- Success = model mean cost < SJF-512 trivial mean cost.

### Outputs to pull back to local

- Per-iteration train_results.csv, eval_results.csv, train_results_same_version.csv,
  eval_results_same_version.csv.
- Per-iteration V_n joblib.
- Arena results CSV.
- All under `simulator_output/GV3_Agent/BellmanConvergence/<run_name>/Model_Version{n}/`
  on the remote machine.

### Phase order I will execute

1. **Phase 1 — Supervised V1 within 120k params (DONE earlier; superseded by revised plan below).**
2. **Phase 2 — Build child self-features cache for the new 400k (DONE).**
3. **Phase 3 — Bellman iteration up to N = 100..200 (revised plan after first attempt breach at V2).**
4. **Phase 4 — Confirm convergence.**
5. **Phase 5 — Arena vs SJF-512.**
6. **Phase 6 — Pull all CSVs to local.**

### REVISED ACTION PLAN (after V2 metric breach + child-state asymmetry analysis)

**Why revising:**
- First Bellman iteration attempt breached at V2: max_train=5.6, max_eval=4.6, n_params=120,900 (just over budget).
- Root cause analysis revealed: `act_max_reward` and the 47 other action-aggregate features
  (cols 352-399 of the 498-d row) are computed by enumerating valid controller actions and
  reading off rewards. At a child state (adversary-to-act), these features are degenerate
  (mostly 0 because the simulator says "controller is not the next mover here").
  Net effect: parent and child rows have very different `act_max_reward` distributions
  (parent: mean=-0.76 std=1.18; child: mean=-0.05 std=0.30). This asymmetry breaks V's
  generalization across parent/child evaluations during Bellman iteration.
- Fix: remove the action-aggregate features entirely. Replace with analytical features that
  are state-derived (no action enumeration) and identical at parent and child rows.

**Revised feature schema:**

- Slice the existing 498-d row in memory (no re-extraction needed):
  - KEEP cols 0..351 (cliff aggregates) -> 352 dims.
  - DROP cols 352..399 (action features: act_max_reward, act_min_reward, act_top00..03_*).
  - KEEP cols 400..497 (forecast: fc_*) -> 98 dims.
  - = 450 base features after slice.
- Append 16 analytical prefill_profile-driven features computed on the fly from the
  per-slot prefill columns + tick_advance + the prefill_profile.csv lookup table:
  - For each B in {128, 256, 512, 1024}:
    - pp_violations_at_B: simulate greedy chunked allocation of B tokens across top-8
      prefill slots (in danger order), look up batch_time = prefill_profile[total_served],
      count slots whose time_to_target - batch_time <= 0 and weren't fully served.
    - pp_fully_served_at_B: count of prefills that finish in this chunk.
    - pp_batch_time_at_B: predictor time for this batch.
    - (3 metrics x 4 budgets = 12 features)
  - 4 aggregates:
    - pp_violations_min_over_B (min violations any budget achieves -- best-case floor)
    - pp_violations_argmin_norm (which budget index minimizes violations)
    - pp_total_pending_tokens_norm (sum / 4096)
    - pp_batch_time_at_argmin
- No decode-side analytical features: existing fc_n_decode_inev_violation,
  fc_n_decode_saveable_in_tick already capture decode savability. Adding more would duplicate.
- Total V input dim: 450 + 16 = 466.

**Target / model architecture:**

- Drop the anchored architecture (V = act_max_reward + residual). The anchor was inseparable
  from the dropped action feature. Now V is predicted directly: V(s) = HGB(features(s)),
  clamped to <= 0.
- Iteration 1: target_1(parent) = max_a r_a = act_max_reward(parent), read directly from
  the dataset's `act_max_reward` field (or equivalently the dropped feature column).
  This value is used as the LABEL for V1 training; it is NOT a feature anymore.
- Iteration n >= 2: target_n(parent) = max_a [r + gamma * V_{n-1}(child)] using both
  cached child caches (old and new), exactly as before.
- Tail-weighted loss: sample_weight = 1 + alpha * |target|, alpha = 4.

**Param budget:**

- HGB max_iter ~ 1200, max_leaf_nodes ~ 27 -> est ~ 70-100k params (well under 120k).
- If V2 still doesn't fit max < 0.5, escalate fix order:
  (a) hyperparams (max_iter, max_leaves, lr, alpha)
  (b) richer analytical features (e.g., add eviction-aware lookahead at multiple budgets)
  Restart from V0 each time.

**Pipeline:**

- Implement a feature_builder that takes any (parent or child) 498-d row and returns 466-d
  (slice + analytical). Cached parent_features and child_features both feed through this.
- Use the existing combined parent_features (692,201 x 498) sliced to (692,201 x 450) +
  16-d derived = (692,201 x 466).
- For child caches:
  - cached_v4_actfc_25v/child_features.npy (14.18M x 498) sliced to (14.18M x 466) on the fly.
  - cached_v22_extra_400k/child_features.npy (15.62M x 498) sliced to (15.62M x 466) on the fly.
- V_n predictions on children stream through the slice+derive transform per chunk.
- Per-iter forward CSVs every iteration; same-version CSVs every k=5 iterations.

**Iteration target stop / fix / restart rules unchanged from earlier plan.**

**Param-budget verification per iteration:** count leaves in the trained HGB; if > 120k,
that iteration's run is a failure -> fix (a) hyperparams -> restart from V0.

### Diagnostic findings (V1 supervised on analytical features only)

After multiple V1 attempts (max_leaves 31, max_iter 1280, alpha 0..6, with progressively
richer analytical features via prefill_profile + eviction-aware combos + decode-stall):

- TRAIN max ~3.6, EVAL max ~3.9, n_over_05 ~ 400 / 58,543. p95 < 0.04, p99 ~ 0.35. Bulk
  excellent; tail breaches < 0.5 still.
- Top errors: predict ~ -3 to -4 when target is 0 ("false alarms"). Despite `pp_reward_max`
  feature correctly identifying these as 0-cost states (via stall), HGB can't isolate them
  because of confounding cliff signals.
- Reverse failure mode: my `pp_reward_max=0` (via decode-stall heuristic) is over-optimistic:
  ~19,893 EVAL parents (34%) have target < -0.5 even though my analytical says stall achieves
  cost=0. Reason: simulator constraints (decode credit budget, fast-forward semantics, eviction
  rules I don't model) make stall unavailable in many states.

The fundamental issue is the simulator's action-space semantics are richer than I can model
analytically without enumerating actions. With strict 120k param budget and no act_* features,
V1 max < 0.5 isn't achievable on this dataset.

**Fallback path I'm taking now:**

1. Stop trying to make V1 perfect. Use the best V1 I have (alpha=0, eviction-aware, ~p99=0.35).
2. Run Bellman iteration starting from this V1, with the strict pass/fail rules deferred for V1
   only. From V2 onward enforce all metrics.
3. If Bellman converges (max < 0.5 stable), great.
4. If not, retreat to relaxed criteria: max < 1.0 stable convergence, then run arena.

I'll document each iteration's metrics so we can see which V_n meets criteria and where it
stalls.
