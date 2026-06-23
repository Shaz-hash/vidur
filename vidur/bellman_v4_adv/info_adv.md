# Bellman V4 Adv Experiment Bundle

This directory is the local bundle for the current GV3 Bellman experiment where a controller action is followed by explicit adversary responses before Bellman bootstrapping.

Path:

```text
vidur/bellman_v4_adv/
```

The goal is to keep the experiment-specific code in one manageable place while still using the canonical GV3/MCTS implementation from `vidur.mcts.Game_Versions.Game_Version3`. This directory should be treated as the working source for the Adv experiment. The older files under `experiments/new_features_v1/` are kept for compatibility/reference until we fully migrate launch scripts.

## Core Idea

The previous child cache represented only:

```text
parent controller state --controller action--> child controller/adversary boundary state
```

The Adv cache represents:

```text
parent controller state
  --controller action, fast_forward=False-->
intermediate adversary state
  --adversary action-->
final child controller state used for bootstrap
```

The controller edge owns the immediate `reward` and `discount`. The adversary edge changes the final child state but does not add controller reward or advance time. Bellman backup is therefore minimax:

```text
V_n(parent) = max_controller_action min_adversary_action [reward + discount * V_{n-1}(final_child)]
```

For iteration 1, `V_0 = 0`, so the adversary fanout does not affect the target except through the shared controller reward/discount.

## Files

| File | Purpose |
|---|---|
| `rootChildGenerationAdv.py` | Builds raw Adv transition shards from repaired parent/root datasets. Applies controller action with `fast_forward=False`, then enumerates valid adversary actions from the intermediate state. |
| `build_state_local_features_adv.py` | Builds parent/root feature matrices and immediate targets from stored controller states. |
| `build_child_features_v4Adv.py` | Builds feature matrices for final Adv child states and writes `parent_index.npz` containing controller and adversary action indices for minimax Bellman backup. |
| `bellman_iterate_v4Adv.py` | Runs Bellman value iteration using `max_controller min_adversary` reduction over the Adv child cache. |
| `rootChildSelectedActionTraceTesting.py` | Generates selected parent/controller/adversary trace CSVs and validates them with GV3 history-node tests. |
| `info_adv.md` | This experiment note. |

## Import Layout

Run from the repo root:

```bash
cd /home/shazer/Desktop/Research/Vidur/vidur-classical-search
export PYTHONPATH=$PWD
```

Then use module commands such as:

```bash
python -m vidur.bellman_v4_adv.rootChildGenerationAdv --help
python -m vidur.bellman_v4_adv.build_state_local_features_adv --help
python -m vidur.bellman_v4_adv.build_child_features_v4Adv --help
python -m vidur.bellman_v4_adv.bellman_iterate_v4Adv --help
python -m vidur.bellman_v4_adv.rootChildSelectedActionTraceTesting --help
```

The bundle imports GV3/MCTS internals from canonical paths like:

```python
vidur.mcts.Game_Versions.Game_Version3...
```

It should not depend on `experiments.new_features_v1` for its copied Adv modules.

## Current Remote Context

Primary Adv/cache machine:

```text
bellman-classical-xl
/home/ubuntu/vidur-classical-search
```

The required Vidur execution-time cache and simulator profiles must stay consistent across local/remote machines:

```text
cache/
simulator_output/prefill_profile.csv
simulator_output/decode_profile.csv
```

Current repaired parent datasets on `bellman-classical-xl`:

```text
simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40_repaired_stats_v1_mp20
simulator_output/GV3_Agent/model_search_roots_controller_extra_400k_abs1_ratio40_unique_from_292k_hops0_320_repaired_stats_v1_mp20
```

Current raw Adv child caches on `bellman-classical-xl`:

```text
simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40_repaired_stats_v1_mp20_child_transitions_adv
simulator_output/GV3_Agent/model_search_roots_controller_extra_400k_abs1_ratio40_unique_from_292k_hops0_320_repaired_stats_v1_mp20_child_transitions_adv
```

D1 raw Adv cache summary:

```text
parents: 292,713
transitions: 55,587,695
shards: 13,717
tasks: 293
```

D2 raw Adv cache summary:

```text
parents: 399,488
transitions: 41,053,802
shards: 10,225
tasks: 400
```

## Current Feature Contract From TASK.md

## FEATURES :

Use **only** the feature schema below for now. This feature contract supersedes the older
498-d / 466-d feature-plan notes later in this task. The main lesson from the successful
`experiments/cliff_aware_bellman/report.md` run is that the value model should read
state-local GV3 request features in a fixed layout, with the same extraction logic used
for parent states and child states during Bellman bootstrap. Our hypothesis is that using these features , we will be able to converge more closely to the bellman with lower max absolute error.


```text
target_n(parent) = max_a [reward_a + gamma_a * V_{n-1}(child_a)]
```

but the model input for both parent and child must be built from that state's own local
features only.

### A. Global State Features

Start from GV3 `global_features` as built by:

```text
vidur/mcts/Game_Versions/Game_Version3/DNN/infer.py:build_model_inputs
```

Do **not** include:

- player one-hot bits: `global_features[0]`, `global_features[1]`
- `global_002 objective_cost`
- `global_003 slo_violations`
- `global_004 slo_lateness_sum`
- `global_022 has_prefill`
- `global_023 has_decode`

Include the following global features, with names and meanings matching the style used
in `experiments/cliff_aware_bellman/report.md`:

| Column name | Source field in `infer.py` | Definition |
|---|---|---|
| `global_005_num_prefill` | `num_prefill` | active prefill request count, `_norm01` by `F.active_prefill_count_den` |
| `global_006_num_decode` | `num_decode` | active decode request count, `_norm01` by `F.active_decode_count_den` |
| `global_007_num_active` | `num_active` | total active request count, `_norm01` by `F.active_total_count_den` |
| `global_008_total_remaining_prefill` | `total_remaining_prefill` | sum of remaining prefill tokens across active prefill reqs, `_norm01` by `F.total_remaining_prefill_den` |
| `global_009_total_remaining_decode` | `total_remaining_decode` | sum of remaining decode tokens across active decode reqs, `_norm01` by `F.total_remaining_decode_den` |
| `global_010_total_decode_generated_active` | `total_decode_generated_active` | total decode tokens already produced for active decode reqs, `_norm01` by `F.total_decode_generated_active_den` |
| `global_011_num_violated_active` | `num_violated_active` | total active requests whose SLO is already violated, `_norm01` by `F.violated_count_den` |
| `global_012_p_late_05_15` | `p_late_05_15` | # active prefill reqs with `near_drop_low <= lateness < near_drop_high`, `_norm01` by `F.prefill_near_drop_den` |
| `global_013_p_late_15` | `p_late_15` | # active prefill reqs with `lateness >= near_drop_high`, `_norm01` by `F.prefill_near_drop_den` |
| `global_014_d_late_05_15` | `d_late_05_15` | # active decode reqs with `near_drop_low <= lateness < near_drop_high`, `_norm01` by `F.decode_near_drop_den` |
| `global_015_d_late_15` | `d_late_15` | # active decode reqs with `lateness >= near_drop_high`, `_norm01` by `F.decode_near_drop_den` |
| `global_016_launch_count` | `launch_count` | # requests launched in the recent launch window, `_norm01` by `F.recent_launch_count_den` |
| `global_017_launch_prefill` | `launch_prefill` | total prefill tokens launched in the recent launch window, `_norm01` by `F.recent_launch_prefill_den` |
| `global_018_remaining_launch_request_headroom` | `remaining_launch_request_headroom` | `MAX_REQUESTS_PER_LAUNCH_WINDOW - launch_count`, clipped at `0`, `_norm01` by `MAX_REQUESTS_PER_LAUNCH_WINDOW` |
| `global_019_remaining_launch_prefill_headroom` | `remaining_launch_prefill_headroom` | `MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW - launch_prefill`, clipped at `0`, `_norm01` by `MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW` |
| `global_020_ewma` | `ewma` | EWMA of recent launch counts, `_norm01` by `F.recent_launch_count_den` |
| `global_021_decode_credit` | `decode_credit` | available decode credit, `_norm01` by `F.decode_credit_den` |
| `g_delta_next_adv_tick_norm` | compute from `decode_next_deadline_by_id[-9100001]` | `max(0, next_adv_tick - sim_time)`, normalized by `adversary_tick_sec = 0.2` |
| `g_delta_since_last_adv_launch_norm` | compute from `recent_arrivals` | `sim_time - latest_recent_adv_launch_time`, normalized by `launch_window_sec = 1.0`; set to 1.0 if there is no recent launch |
| `global_extra_prefill_violated_active` | compute from active prefill req ids and violated req ids | # active prefill requests whose SLO is already violated, `_norm01` by `F.active_prefill_count_den` |
| `global_extra_decode_violated_active` | compute from active decode req ids and violated req ids | # active decode requests whose SLO is already violated, `_norm01` by `F.active_decode_count_den` |

The two `global_extra_*` columns are not currently separate columns in GV3
`global_features`; compute them directly from the same active request sets used by
`build_model_inputs`. Keep them state-local and compute them identically for parent and
child states.

#### A.1 EDF / decode-batch-time scalar globals

**Motivation.** The 17-bucket prefill-slack scheme cut V1 max_abs error from 6.05 → 2.93,
but ~39 eval samples remain with `abs_err > 0.5`. Outlier inspection
(`request_edf_analysis.csv` for v2) shows the model still cannot cleanly express
"min EDF slack > decode batch time → cost = 0", because:

1. Slack is encoded only as per-slot one-hots; min-over-slots requires AND-ing many
   one-hots inside a single tree path (expensive within a 31-leaf budget).
2. There is no continuous scalar for the most predictive number — the *minimum* slack
   across all active requests vs the time of one decode batch.
3. The decode batch time depends on the largest-context request currently active,
   which is *also* not exposed.

**Decode batch time lookup.** Use `simulator_output/decode_profile.csv` (columns
`decode tokens, decode_time_seconds`). For a state, define the **largest active
context length** as only for decode requests !:

```text
context_tokens(req) = num_prefill_tokens # done prefill
                      + max(0, num_processed_tokens - num_prefill_tokens)  # done decode
max_context_tokens(state) = max_{r in active_requests} context_tokens(r)
```

If there are no active decode requests, `max_context_tokens = 0`.

Look up `decode_time_at_max = decode_profile[bucket(max_context_tokens)]`, where
`bucket(x)` picks the row whose `decode tokens` value is **closest** to `x` (ties
break to the smaller bucket). Treat the table as the canonical batch-execution
time for the next decode tick at this state.

**Min prefill slack.** Across active **prefill** requests only:

```text
prefill_slack(r) = (arrived_at(r) + prefill_slo_time(r)) - sim_time
min_prefill_slack(state) = min_{r in active prefill} prefill_slack(r)
```

If there are no active prefill requests, set `min_prefill_slack = +inf` so the
EDF margin below clips to 1 (no prefill urgency).

**The new global features.** Append the following columns to section A:

| Column name | Definition |
|---|---|
| `global_edf_minus_batch_norm` | `clip( (min_prefill_slack - decode_time_at_max) , 0, 1 )`. Zero means "the most-urgent active prefill is at or below the next decode batch's execution time" (cliff regime). One means "≥ 1 second of headroom" (clearly safe). HGB can express the cliff with a single threshold on this scalar. |
| `global_n_active_with_edf_margin_gt_batch_norm` | Count of active requests (prefill + decode) whose own per-request deadline-vs-batch margin is strictly positive, divided by `F.active_total_count_den` (120). For each active request, define `request_deadline = arrived_at + prefill_slo_time` if the request is in the prefill phase (`is_prefill_complete == False`), otherwise `request_deadline = decode_next_deadline_by_id[rid]` (fall back to `arrived_at + decode_slo_time` if missing). The per-request margin is `(request_deadline - sim_time) - decode_time_at_max`. Count requests with margin > 0 and normalise by `F.active_total_count_den`. **Why:** v3 outlier inspection shows several cases where `min_prefill_slack` is just barely above batch time but 5–6 active requests are crowded near deadline. The model needs a single-threshold scalar that says "all (or most) active requests are clearly safe" so HGB can isolate "many crowded but each above batch time → cost ≈ 0" without ANDing across per-slot one-hots. |

`D_GLOBAL` is now 23: 19 base globals, 2 EDF/decode-batch globals, and 2 adversary-timing globals. Sections B and C unchanged. New
`D_TOTAL = 23 + 7*25 + 7*4 = 226`.

#### A.2 Adversary timing scalar globals

These two scalar globals expose timing that is otherwise only implicit in GV3 metadata and recent launch history:

| Column name | Definition |
|---|---|
| `g_delta_next_adv_tick_norm` | `clip((next_adv_tick - sim_time) / 0.2, 0, 1)`, where `next_adv_tick` is stored under metadata key `-9100001` in `decode_next_deadline_by_id`. |
| `g_delta_since_last_adv_launch_norm` | `clip((sim_time - latest_recent_adv_launch_time) / 1.0, 0, 1)`, using the newest launch timestamp inside `recent_arrivals`; if no recent launch exists, use `1.0`. |

**Determinism.** All inputs (snapshot + stats + `decode_profile.csv`) are state-local
and identical at parent and child rows, satisfying the iteration-time symmetry rule
in the Q&A section below.

### B. Prefill Request Slots

Use at most:

```text
MAX_REQUESTS_PER_LAUNCH_WINDOW = 7
```

prefill request slots.

Selection rule:

1. Consider currently active prefill requests only.
2. Compute remaining prefill slack time:

```text
prefill_slack_sec = prefill_slo_deadline - current_sim_time
prefill_slack_sec_clamped = max(0.0, prefill_slack_sec)
```

3. Sort requests by `prefill_slack_sec_clamped` ascending. The most urgent request
   goes first. Break ties by request id ascending.
4. If all active prefill requests have already violated their SLO, ignore slack ordering
   and select the first 7 by request id ascending.
5. If fewer than 7 requests exist, pad the remaining slots with zeros. Each slot should
   include a `present` bit so padded zeros are distinguishable from real zero-valued
   features.

For each selected prefill slot, emit only:

- `prefill_slot_present`: `1` for a real slot, `0` for padding.
- `prefill_remaining_tokens`: remaining prefill tokens for that request.
- `prefill_total_tokens`: total prefill tokens for that request.
- `prefill_violated_bit`: `1` if the request has violated prefill SLO, else `0`.
- `prefill_lateness_bucket`: one-hot bucket over request lateness:
  - `[0.0, 0.5)`
  - `[0.5, 1.0)`
  - `[1.0, 1.5)`
  - `[1.5, +inf)`
- `prefill_slack_bucket`: one-hot bucket over `prefill_slack_sec_clamped`.

Use the following prefill slack bucket boundaries. The first sub-bucket range
(`0.0 < slack <= 0.015725797204323228`, the 128-prefill-token profile time) is split
into 6 finer sub-buckets to give the model granular resolution around the
~13.29 ms decode-batch execution time. Rationale: V1 outlier analysis
(`request_edf_analysis.csv`) showed all 40 worst eval predictions have minimum
slack in `[0.0133, 0.0157]`. As long as the earliest deadline is slightly above
the decode-batch time (~0.01329 s), a near-zero immediate cost is reachable, but
HGB cannot separate "EDF slightly above batch time → 0 cost" from "EDF below
batch time → cliff" with the original single coarse bucket.

| Bucket | Slack range |
|---:|---|
| 0 | `slack == 0.0` |
| 1 | `0.0 < slack <= 0.0133` |
| 2 | `0.0133 < slack <= 0.0135` |
| 3 | `0.0135 < slack <= 0.0137` |
| 4 | `0.0137 < slack <= 0.0140` |
| 5 | `0.0140 < slack <= 0.0145` |
| 6 | `0.0145 < slack <= 0.015725797204323228` |
| 7 | `0.015725797204323228 < slack <= 0.023274675327417962` |
| 8 | `0.023274675327417962 < slack <= 0.031963770276289896` |
| 9 | `0.031963770276289896 < slack <= 0.03888623299112536` |
| 10 | `0.03888623299112536 < slack <= 0.06091750997076902` |
| 11 | `0.06091750997076902 < slack <= 0.07016849423102292` |
| 12 | `0.07016849423102292 < slack <= 0.08612442901238251` |
| 13 | `0.08612442901238251 < slack <= 0.09850190759874299` |
| 14 | `0.09850190759874299 < slack <= 0.19613847773632437` |
| 15 | `0.19613847773632437 < slack <= 0.28408680179190937` |
| 16 | `slack > 0.28408680179190937` |

Buckets 7..15 still correspond to the prefill_profile token times:

```text
128, 256, 384, 512, 640, 768, 896, 1024, 2048, 3072 prefill-token profile times
```

with an explicit zero-slack bucket (0), 6 fine sub-buckets within the
128-token-time band (1..6), and a final `>3072-token-time` bucket (16).
Total: 17 prefill slack buckets.

### C. Decode Request Slots

Use at most:

```text
MAX_REQUESTS_PER_LAUNCH_WINDOW = 7
```

decode request slots.

Selection rule:

1. Consider only active decode requests that have **not** yet experienced SLO violation.
2. If there are more than 7 non-violated decode requests, choose 7 randomly.
3. The random choice must be deterministic for a given state. Use a stable seed derived
   from the root/sample id when available; otherwise use a deterministic hash of the
   state signature and active request ids. Do not use nondeterministic process-global
   randomness, because train-time and bootstrap-time feature extraction must agree.
4. If fewer than 7 non-violated decode requests exist, pad the remaining slots with zeros.
   Each slot should include a `present` bit so padding is explicit.

For each selected decode slot, emit only:

- `decode_slot_present`: `1` for a real slot, `0` for padding.
- `decode_remaining_norm`: remaining decode tokens normalized to `[0, 1]`.
- `decode_done_gt_216_bit`: `1` if decoded tokens processed so far is greater than `216`,
  else `0`.
- `decode_done_gt_512_bit`: `1` if decoded tokens processed so far is greater than `512`,
  else `0`.

Do not include already-violated decode requests in the decode slots.

## Raw Child Generation

D1 example:

```bash
ssh bellman-classical-xl
cd /home/ubuntu/vidur-classical-search
export PYTHONPATH=$PWD

python -m vidur.bellman_v4_adv.rootChildGenerationAdv \
  --dataset-dir simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40_repaired_stats_v1_mp20 \
  --output-dir simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40_repaired_stats_v1_mp20_child_transitions_adv \
  --root-player-filter controller \
  --transition-shard-size 4096 \
  --num-processes 64 \
  --parents-per-task 1000
```

D2 is the same pattern with the extra-400k repaired dataset and `_child_transitions_adv` output dir.

Important invariant: parent to intermediate must not fast-forward. The code path is:

```python
mcts._env.apply_controller_action_only(..., fast_forward=False)
```

## Feature Building

Parent features:

```bash
python -m vidur.bellman_v4_adv.build_state_local_features_adv \
  --manifest <parent_dataset>/manifest.jsonl \
  --shard-dir <parent_dataset> \
  --root-player-filter controller \
  --num-processes 64 \
  --out-dir <bellman_run_dir>
```

Adv child features:

```bash
python -m vidur.bellman_v4_adv.build_child_features_v4Adv \
  --child-cache-dir <raw_adv_child_cache_dir> \
  --out-dir <bellman_run_dir>/v4_child_d1 \
  --num-processes 64
```

Repeat for D2 using `<bellman_run_dir>/v4_child_d2`.

`build_child_features_v4Adv.py` writes:

```text
child_features.npy
child_meta.npy
child_features.names.json
child_features.summary.json
parent_index.npz
```

The Adv `parent_index.npz` must include controller/adversary grouping fields used by `bellman_iterate_v4Adv.py`, especially:

```text
controller_action_indices
adversary_action_indices
rewards
discounts
is_valid
parent_ids
offsets
order
```


## Training Scripts

The XL and 200k classical training scripts are copied into this bundle. They do not build features themselves; they consume the feature/target arrays produced by `build_state_local_features_adv.py` and the split JSON for the current Bellman run. Exploratory targeted/hierarchical/bucket/ensemble scripts are intentionally not part of this cleaned Adv bundle.

| File | Purpose |
|---|---|
| `train_v1_supervised_xl.py` | Trains the three ~120k HGB XL shapes: 31x1280, 47x850, 63x630. |
| `train_v1_supervised_xl_alpha.py` | Same XL shape set, but with `--tail-alpha` supplied explicitly for alpha sweeps. |
| `train_v1_supervised_200k.py` | 200k HGB capacity sweep over larger leaf/iteration shapes, alpha values, and losses. |

Run from repo root:

```bash
cd /home/ubuntu/vidur-classical-search
export PYTHONPATH=$PWD

python -m vidur.bellman_v4_adv.train_v1_supervised_xl   --features-d1 <bellman_run_dir>/features_d1.npy   --targets-d1 <bellman_run_dir>/targets_d1.npy   --features-d2 <bellman_run_dir>/features_d2.npy   --targets-d2 <bellman_run_dir>/targets_d2.npy   --split-json <bellman_run_dir>/split_indices_combined.json   --output-dir <bellman_run_dir>/V1_models_xl

python -m vidur.bellman_v4_adv.train_v1_supervised_200k   --features-d1 <bellman_run_dir>/features_d1.npy   --targets-d1 <bellman_run_dir>/targets_d1.npy   --features-d2 <bellman_run_dir>/features_d2.npy   --targets-d2 <bellman_run_dir>/targets_d2.npy   --split-json <bellman_run_dir>/split_indices_combined.json   --output-dir <bellman_run_dir>/V1_models_200k
```

These scripts should be preferred over the old `experiments.new_features_v1` training modules for new Adv runs, so the experiment remains self-contained under `vidur.bellman_v4_adv`.

## Bellman Iteration

Run after parent features and Adv child features exist:

```bash
python -m vidur.bellman_v4_adv.bellman_iterate_v4Adv \
  --features-d1 <bellman_run_dir>/features_d1.npy \
  --features-d2 <bellman_run_dir>/features_d2.npy \
  --child-d1 <bellman_run_dir>/v4_child_d1 \
  --child-d2 <bellman_run_dir>/v4_child_d2 \
  --split-json <bellman_run_dir>/split_indices_combined.json \
  --start-iter 1 \
  --end-iter 100 \
  --tail-alpha 1.0 \
  --max-leaves <leaves> \
  --max-iter <iters> \
  --out-dir <bellman_run_dir>/V_iter_adv
```

Core reduction in `bellman_iterate_v4Adv.py`:

```text
candidate = reward + discount * V_prev(final_child)
controller_value = min(candidate over adversary actions for that controller action)
parent_target = max(controller_value over controller actions)
```

The candidate is clipped with `min(candidate, 0)` to preserve the value convention that costs are non-positive.

## Trace Testing

Use this before trusting a new child-generation change:

```bash
python -m vidur.bellman_v4_adv.rootChildSelectedActionTraceTesting \
  --dataset-dir <small_or_full_parent_dataset> \
  --output-dir simulator_output/GV3_Agent/rootChildAdvTraceSmoke \
  --max-traces 5 \
  --overwrite
```

The trace tester writes CSVs showing:

```text
existing parent history rows
selected parent state row
controller action row / intermediate state
adversary action row / final child state
```

Then it validates the CSVs through the GV3 history-node tests. This catches state/list consistency issues such as request IDs disappearing from active/waiting sets.

## Known Sanity Checks

Repaired classical raw child cache vs XL Adv cache:

- The classical repaired raw child state matches the Adv `intermediate_simulator_snapshot`.
- Sampled rewards/discounts matched exactly for `(parent_state_id, controller_action)`.
- Adv final child state is expected to differ because adversary action is applied after the intermediate state.

Observed sampled adversary fanout rates from raw Adv caches:

```text
Sample A, 1000 parents:
  parents with at least one controller action having >1 valid adversary action: 315 / 1000 = 31.5%
  parents where every valid controller action has >1 valid adversary action: 28 / 1000 = 2.8%

Sample B, 1000 parents:
  parents with at least one controller action having >1 valid adversary action: 289 / 1000 = 28.9%
  parents where every valid controller action has >1 valid adversary action: 22 / 1000 = 2.2%
```

Estimated full-dataset count for the stricter “every valid controller action has >1 valid adversary action” case is roughly `15k-19k` parent states out of `692,201`.

## Migration Notes

This bundle is currently a copy/cleanup of the Adv-specific files. Do not delete the old `experiments/new_features_v1` files yet. Safe migration path:

1. Keep running old experiment commands if a remote job depends on them.
2. Prefer new commands under `python -m vidur.bellman_v4_adv...` for new Adv runs.
3. After the bundle is validated remotely, turn old Adv files into thin wrappers or leave them as archival copies.
4. Training scripts in this bundle are limited to XL, XL-alpha, and 200k HGB model families. Keep older exploratory files under `experiments/new_features_v1` only for compatibility/reference.

## Stats Bug And Repaired Parent Datasets

The repaired parent datasets are required because the original root-storage path had stale objective stats at some decision frontiers.

Bug location:

```text
vidur/mcts/Game_Versions/Game_Version3/ModelSearchBed/root_storage.py
```

Problem:

```text
Root generation could stop at a controller decision frontier after simulator time had advanced past request deadlines, but the state.stats SLO/lateness counters were not always refreshed before scoring/storing the root.
```

Consequence:

```text
The first-layer target and immediate reward could be computed from stale stats.
Some parent records therefore had incorrect target_value / best_reward / stored stats.
```

Fix in `root_storage.py`:

```python
def refresh_root_objective_stats(env, state):
    ...
```

This is called inside `evaluate_first_layer_target(...)` before `search_dnn(...)` and before the model-state stats are cloned for storage.

Repair/migration script:

```text
vidur/mcts/Game_Versions/Game_Version3/ModelSearchBed/repair_dataset.py
```

The repair script reloads each stored root, calls `refresh_root_objective_stats(...)`, recomputes the first-layer target, and writes a repaired dataset with updated fields such as:

```text
target_value
best_action_index
best_action_repr
best_reward
best_discount
best_bootstrap
best_child_cost
best_child_time
stats
frontier_stats
repair_metadata
```

Current repaired parent datasets:

```text
model_search_roots_controller_350k_abs1_ratio40_repaired_stats_v1_mp20
model_search_roots_controller_extra_400k_abs1_ratio40_unique_from_292k_hops0_320_repaired_stats_v1_mp20
```

Important ordering:

```text
1. Fix/repair parent datasets.
2. Generate raw child/Adv-child transitions from repaired parents.
3. Build parent and child features from those repaired artifacts.
4. Train V1 / run Bellman iterations.
```

Do not mix old raw child caches with repaired parent features. If parent stats/targets are repaired, child transitions must be regenerated from those repaired parent records.
