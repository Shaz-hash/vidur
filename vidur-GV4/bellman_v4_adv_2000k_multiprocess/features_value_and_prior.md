# Value And Prior Feature Contract

This file records the feature contracts for the current GV3 Bellman / HGB work.

There are two separate model-input use cases:

1. `V(s)` value model: uses only state-local features from the current simulator state.
2. `Q(s, a)` or prior/action model: uses `[state_features(s), action_features(a | s)]`.

For `Q(s, a)` and action-prior work, action features must be built from the parent
state and the controller action only. They must not use reward, child cost, discount,
bootstrap value, or any post-action state statistic.

## Value Function State Features

This is the current 226D state-local schema from `vidur/bellman_v4_adv/info_adv.md`.
The same extractor must be used for parent states and child states during Bellman
bootstrap.

```text
target_n(parent) = max_a [reward_a + gamma_a * V_{n-1}(child_a)]
```

The model input for both parent and child must be built from that state's own local
features only.

### A. Global State Features

Start from GV3 `global_features` as built by:

```text
vidur/mcts/Game_Versions/Game_Version3/DNN/infer.py:build_model_inputs
```

Do not include:

- player one-hot bits: `global_features[0]`, `global_features[1]`
- `global_002 objective_cost`
- `global_003 slo_violations`
- `global_004 slo_lateness_sum`
- `global_022 has_prefill`
- `global_023 has_decode`

Include the following global features:

| Column name | Source field | Definition |
|---|---|---|
| `global_005_num_prefill` | `num_prefill` | active prefill request count, normalized by `F.active_prefill_count_den` |
| `global_006_num_decode` | `num_decode` | active decode request count, normalized by `F.active_decode_count_den` |
| `global_007_num_active` | `num_active` | total active request count, normalized by `F.active_total_count_den` |
| `global_008_total_remaining_prefill` | `total_remaining_prefill` | sum of remaining prefill tokens across active prefill requests, normalized by `F.total_remaining_prefill_den` |
| `global_009_total_remaining_decode` | `total_remaining_decode` | sum of remaining decode tokens across active decode requests, normalized by `F.total_remaining_decode_den` |
| `global_010_total_decode_generated_active` | `total_decode_generated_active` | total decode tokens already produced for active decode requests, normalized by `F.total_decode_generated_active_den` |
| `global_011_num_violated_active` | `num_violated_active` | total active requests whose SLO is already violated, normalized by `F.violated_count_den` |
| `global_012_p_late_05_15` | `p_late_05_15` | active prefill requests with `near_drop_low <= lateness < near_drop_high`, normalized by `F.prefill_near_drop_den` |
| `global_013_p_late_15` | `p_late_15` | active prefill requests with `lateness >= near_drop_high`, normalized by `F.prefill_near_drop_den` |
| `global_014_d_late_05_15` | `d_late_05_15` | active decode requests with `near_drop_low <= lateness < near_drop_high`, normalized by `F.decode_near_drop_den` |
| `global_015_d_late_15` | `d_late_15` | active decode requests with `lateness >= near_drop_high`, normalized by `F.decode_near_drop_den` |
| `global_016_launch_count` | `launch_count` | requests launched in the recent launch window, normalized by `F.recent_launch_count_den` |
| `global_017_launch_prefill` | `launch_prefill` | total prefill tokens launched in the recent launch window, normalized by `F.recent_launch_prefill_den` |
| `global_018_remaining_launch_request_headroom` | computed | `MAX_REQUESTS_PER_LAUNCH_WINDOW - launch_count`, clipped at `0`, normalized by `MAX_REQUESTS_PER_LAUNCH_WINDOW` |
| `global_019_remaining_launch_prefill_headroom` | computed | `MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW - launch_prefill`, clipped at `0`, normalized by `MAX_PREFILL_TOKENS_PER_LAUNCH_WINDOW` |
| `global_020_ewma` | `ewma` | EWMA of recent launch counts, normalized by `F.recent_launch_count_den` |
| `global_021_decode_credit` | `decode_credit` | available decode credit, normalized by `F.decode_credit_den` |
| `global_extra_prefill_violated_active` | computed | active prefill requests whose SLO is already violated, normalized by `F.active_prefill_count_den` |
| `global_extra_decode_violated_active` | computed | active decode requests whose SLO is already violated, normalized by `F.active_decode_count_den` |
| `global_edf_minus_batch_norm` | computed | `clip(min_prefill_slack - decode_time_at_max, 0, 1)` |
| `global_n_active_with_edf_margin_gt_batch_norm` | computed | count of active requests whose per-request deadline-vs-batch margin is positive, normalized by `F.active_total_count_den` |
| `g_delta_next_adv_tick_norm` | computed from `decode_next_deadline_by_id[-9100001]` | `clip((next_adv_tick - sim_time) / 0.2, 0, 1)` |
| `g_delta_since_last_adv_launch_norm` | computed from `recent_arrivals` | `clip((sim_time - latest_recent_adv_launch_time) / 1.0, 0, 1)`, or `1.0` if no recent launch exists |

`D_GLOBAL = 23`.

### A.1 EDF / Decode-Batch Globals

For a state, define largest active decode context length:

```text
context_tokens(req) = num_prefill_tokens
                      + max(0, num_processed_tokens - num_prefill_tokens)
max_context_tokens(state) = max over active decode requests of context_tokens(req)
```

If there are no active decode requests, `max_context_tokens = 0`.

Look up:

```text
decode_time_at_max = decode_profile[nearest_bucket(max_context_tokens)]
```

where `simulator_output/decode_profile.csv` is canonical and ties choose the smaller
bucket.

Across active prefill requests:

```text
prefill_slack(r) = (arrived_at(r) + prefill_slo_time(r)) - sim_time
min_prefill_slack(state) = min over active prefill requests of prefill_slack(r)
```

If there are no active prefill requests, set `min_prefill_slack = +inf`.

Then:

```text
global_edf_minus_batch_norm = clip(min_prefill_slack - decode_time_at_max, 0, 1)
```

For `global_n_active_with_edf_margin_gt_batch_norm`, each active request gets:

```text
request_deadline = arrived_at + prefill_slo_time       if request is prefill phase
request_deadline = decode_next_deadline_by_id[request] if request is decode phase
margin = (request_deadline - sim_time) - decode_time_at_max
```

Count requests with `margin > 0`, divide by `F.active_total_count_den`.

### B. Prefill Request Slots

Use at most:

```text
MAX_REQUESTS_PER_LAUNCH_WINDOW = 7
```

prefill slots.

Selection rule:

1. Consider currently active prefill requests only.
2. Compute `prefill_slack_sec = prefill_slo_deadline - current_sim_time`.
3. Clamp slack with `prefill_slack_sec_clamped = max(0.0, prefill_slack_sec)`.
4. Sort by `prefill_slack_sec_clamped` ascending, tie-break by request id ascending.
5. If all active prefill requests have already violated SLO, ignore slack ordering and select first 7 by request id ascending.
6. If fewer than 7 requests exist, pad remaining slots with zeros.

Each selected prefill slot emits 25 dimensions:

```text
prefill_slot_present
prefill_remaining_tokens
prefill_total_tokens
prefill_violated_bit
prefill_lateness_bucket_0..3
prefill_slack_bucket_0..16
```

The slack buckets are:

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

Total prefill slot features:

```text
7 slots * 25 dims = 175 dims
```

### C. Decode Request Slots

Use at most:

```text
MAX_REQUESTS_PER_LAUNCH_WINDOW = 7
```

decode slots.

Selection rule:

1. Consider only active decode requests that have not yet experienced SLO violation.
2. If there are more than 7 non-violated decode requests, choose 7 deterministically.
3. The deterministic choice should use the root/sample id when available, otherwise a stable hash of state signature and active request ids.
4. If fewer than 7 non-violated decode requests exist, pad remaining slots with zeros.

Each selected decode slot emits 4 dimensions:

```text
decode_slot_present
decode_remaining_norm
decode_done_gt_216_bit
decode_done_gt_512_bit
```

Total decode slot features:

```text
7 slots * 4 dims = 28 dims
```

Total value-state feature dimension:

```text
D_TOTAL = 23 + 7*25 + 7*4 = 226
```

## Action Features For Q(s, a) / Prior Models

Action features are built from:

```text
parent simulator state
parent stats
controller action
```

They must not use:

```text
reward
discount
child state
child cost
bootstrap value
post-action stats
post-action time
```

This makes the feature contract safe for both training and arena inference, where we
only have the parent state and valid controller actions before choosing an action.

### Constants

Use these normalizers:

```text
MAX_PREFILL_ACTION_ALLOC = max(controller_action.prefill_budget_options) = 4096
MAX_DECODE_REQUESTS = F_ACTIVE_DECODE_COUNT_DEN = 100
MAX_REQUESTS_PER_LAUNCH_WINDOW = 7
```

`MAX_PREFILL_ACTION_ALLOC` is the maximum prefill budget option in the GV3 controller
action config. If the config changes, this denominator must be read from the config.

### A. Global Action Summary Features

| Feature name | Definition |
|---|---|
| `a_total_prefill_alloc_norm` | `sum(action.prefill_allocations.values()) / MAX_PREFILL_ACTION_ALLOC` |
| `a_total_decode_alloc_norm` | `sum(action.decode_allocations.values()) / MAX_DECODE_REQUESTS` |
| `a_n_prefill_alloc_reqs_norm` | `len(action.prefill_allocations) / MAX_REQUESTS_PER_LAUNCH_WINDOW` |
| `a_n_decode_alloc_reqs_norm` | `len(action.decode_allocations) / MAX_DECODE_REQUESTS` |
| `a_n_evicted_prefill_norm` | `num_evicted_prefill / max(1, num_active_prefill)` |
| `a_n_evicted_decode_norm` | `num_evicted_decode / MAX_DECODE_REQUESTS` |
| `a_has_prefill_alloc` | `1` if action allocates any prefill token, else `0` |
| `a_has_decode_alloc` | `1` if action allocates any decode token, else `0` |
| `a_has_eviction` | `1` if the canon action evicts any request, else `0` |
| `a_strict_noop` | `1` if no prefill allocation, no decode allocation, and no eviction, else `0` |

`num_evicted_prefill` and `num_evicted_decode` must be computed from the canon
action's `evicted_request_ids` and the parent state's request phases. Do not infer
eviction from the child state. If an uncannonicalized action alias is being processed,
first resolve its evicted request ids using the same GV3 eviction semantics as the
environment, then build features from those ids.

### B. Canon Eviction Effect Features

Do not include one-hot features for the prefill heuristic or eviction rule. The native
MCTS/action canon now merges action aliases by actual action effect:

```text
token_allocations + prefill_allocations + decode_allocations + evicted_request_ids
```

Therefore, the action feature extractor must not depend on the aliasing heuristic that
created the action. Instead, derive generalized eviction features from:

```text
parent simulator state
canon action evicted_request_ids
```

Emit these features:

| Feature name | Definition |
|---|---|
| `a_n_evicted_decode_late_over_0p5_norm` | number of evicted decode requests with lateness `> 0.5`, divided by `MAX_DECODE_REQUESTS` |
| `a_n_evicted_prefill_late_over_0p5_norm` | number of evicted prefill requests with lateness `> 0.5`, divided by `MAX_REQUESTS_PER_LAUNCH_WINDOW` |
| `a_n_evicted_prefill_missed_deadline_norm` | number of evicted prefill requests whose prefill deadline is already missed, divided by `MAX_REQUESTS_PER_LAUNCH_WINDOW` |
| `a_evict_includes_highest_lateness_prefill` | `1` if the evicted prefill set includes the active prefill request with highest lateness, else `0` |
| `a_evict_includes_highest_lateness_decode` | `1` if the evicted decode set includes the active decode request with highest lateness, else `0` |

Definitions:

```text
prefill deadline missed:
  parent_time > request.arrived_at + prefill_slo_time(request)

prefill lateness:
  same lateness definition used by the 226D state feature extractor for active
  prefill requests

decode lateness:
  same lateness definition used by the 226D state feature extractor for active
  decode requests

highest lateness prefill:
  active prefill request with maximum prefill lateness; tie-break by request id

highest lateness decode:
  active decode request with maximum decode lateness; tie-break by request id
```

If there is no active request of that type, or if no request of that type is evicted,
the corresponding highest-lateness inclusion bit is `0`.

### C. Slot-Aligned Prefill Action Features

The action feature extractor must rebuild the same prefill slot map used by the 226D
state feature extractor:

```text
prefill_req_id -> prefill_state_slot in [0..6]
```

This lets the model learn interactions such as:

```text
state says prefill_slot_0 is urgent
AND action allocates to prefill_slot_0
```

For each prefill state slot `i = 0..6`, emit:

| Feature name | Definition |
|---|---|
| `a_prefill_slot_i_selected` | `1` if the action allocates prefill tokens to the request currently represented by prefill slot `i`, else `0` |
| `a_prefill_slot_i_alloc_norm` | allocated prefill tokens for prefill slot `i` divided by `MAX_PREFILL_ACTION_ALLOC` |
| `a_prefill_slot_i_alloc_frac_of_remaining` | allocated prefill tokens for prefill slot `i` divided by that slot request's remaining prefill tokens, clipped to `[0, 1]`; use `0` for padded/missing slots |
| `a_prefill_slot_i_evicted` | `1` if the request currently represented by prefill slot `i` is present in the canon action's `evicted_request_ids`, else `0` |

Expanded names:

```text
a_prefill_slot_0_selected
a_prefill_slot_0_alloc_norm
a_prefill_slot_0_alloc_frac_of_remaining
a_prefill_slot_0_evicted
...
a_prefill_slot_6_selected
a_prefill_slot_6_alloc_norm
a_prefill_slot_6_alloc_frac_of_remaining
a_prefill_slot_6_evicted
```

If an action allocates prefill tokens to a request that is not visible in the 7 selected
prefill state slots, the per-slot features remain zero for that request. Such cases
should be tracked separately in a future `a_prefill_alloc_outside_visible_slots_norm`
feature if they become common.

### D. Current Action Feature Dimension

With the starter feature set requested here:

```text
global action summary: 10
canon eviction effect features: 5
prefill slot-aligned features: 7 * 4 = 28
```

Total action feature dimension:

```text
D_ACTION_STARTER = 10 + 5 + 28 = 43
```

If concatenated with the current state features:

```text
D_STATE_ACTION_STARTER = 226 + 43 = 269
```

### E. Intended Targets

For controller-action value learning:

```text
Q_target(s, a_controller)
  = min_adversary [reward(s, a_controller)
                   + discount(s, a_controller)
                   * V_prev(final_child_after_adversary)]
```

The model input is:

```text
[state_features_226(parent_state), action_features_43(parent_state, controller_action)]
```

At inference time:

1. Build `state_features_226(s)` once.
2. Enumerate valid canonical controller actions.
3. Build action features for each valid action.
4. Predict `Q(s, a)` for each action.
5. Choose the valid action with maximum predicted Q.
