# Linear LP Submodule

This package assembles and solves LPs from linear sampler merged tables with selectable objective modes.

## Formulation

- Value model: `V_w(s) = phi(s)^T w`
- For each transition sample `i`:
  - `a_i = phi(s_i) - gamma * phi(s'_i)`
  - `b_i = delta_cost_i = cost_next_i - cost_s_i`

### Mode 1: `minimax_directional_slack_eliminated` (default)

Directional-equivalent inequalities (eliminated-slack form):

- Controller sample:
  - `a_i^T w <= b_i`
  - `-a_i^T w - t <= -b_i`
- Adversary sample:
  - `-a_i^T w <= -b_i`
  - `a_i^T w - t <= b_i`

Objective:

- `min t`

Decision vector:

- `[w, t]` where `w` are policy weights and `t` is worst-case directional error.

Stability:

- Box bounds on policy weights: `-Wmax <= w_i <= Wmax`
- Additional structural preference constraints enforce small positive value gaps on
  synthetic monotone comparisons such as:
  - fresh burst vs empty
  - more decode load vs less

### Mode 2: `avg_anchor_value_gap`

Uses classic one-sided Bellman constraints:

- Controller sample: `a_i^T w <= b_i`
- Adversary sample: `-a_i^T w <= -b_i`

Objective:

- `min( mean_{s in S_adv} phi(s)^T w - mean_{s in S_ctrl} phi(s)^T w )`
- `S_adv` and `S_ctrl` are unique anchor states by player.

Decision vector:

- `[w]` only (no `t` variable).

## `baseline_v1` Feature Set (7 dims)

The LP feature vector is current-burst/decode oriented and is used for both
anchor and successor states in constraints.

Global features:

1. `bias`
2. `current_burst_size_norm` = request count in the current prefill burst divided by `6`
3. `current_burst_remaining_chunks_norm` = `current_burst_prefill_remaining / (512 * 36)`
4. `current_burst_deadline_urgency_norm` = `1 / max(current_prefill_deadline_distance, 1e-3)` when a burst is active, else `0`
5. `current_burst_lateness_sec_norm` = `max(0, -current_prefill_deadline_distance)` when a burst is active, else `0`
6. `decode_active_norm` = `active_decodes / 200`
7. `decode_violated_count_norm` = `active_decode_violated / 200`

The "current burst" is selected as the live prefill burst with the smallest
deadline distance. Requests that already completed prefill but share that burst
deadline are still counted in `current_burst_size_norm`, which lets the value
function separate burst size from the remaining active prefill work.

`baseline_v1` also hard-codes sign constraints directly into the LP bounds:

- `current_burst_size_norm >= 0`
- `current_burst_remaining_chunks_norm >= 0`
- `current_burst_deadline_urgency_norm >= 0`
- `current_burst_lateness_sec_norm >= 0`
- `decode_active_norm >= 0`
- `decode_violated_count_norm >= 0`

The LP also appends structural margin constraints of the form:

- `V(worse_state) >= V(better_state) + eps`

with `eps` controlled by `--structural-margin-eps` (default `1e-3`).

## CLI

```bash
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
-m vidur.mcts.linear.lp.run \
  --sampler-round-dir simulator_output/linear_sampler_fast_LP/round_000/merged \
  --out-dir simulator_output/linear_lp_solution \
  --discount-factor 0.98 \
  --weight-bound-abs 100.0 \
  --structural-margin-eps 1e-3 \
  --feature-set baseline_v1 \
  --objective-mode minimax_directional_slack_eliminated \
  --skip-default-evaluator \
  --seed 12345
```

Average-anchor objective mode:

```bash
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
-m vidur.mcts.linear.lp.run \
  --sampler-round-dir simulator_output/linear_sampler_fast_LP/round_000/merged \
  --out-dir simulator_output/linear_lp_solution_avg \
  --discount-factor 0.98 \
  --weight-bound-abs 100.0 \
  --structural-margin-eps 1e-3 \
  --feature-set baseline_v1 \
  --objective-mode avg_anchor_value_gap \
  --skip-default-evaluator \
  --seed 12345
```

## Outputs

- `lp_dataset_summary.json`
- `lp_problem_stats.json`
- `lp_solution.npz`
- `lp_feature_names.json`
- `lp_weights_by_feature.json`
- `lp_constraint_violations.csv`
- `mcts_iter_random_state.csv`
- `normalised_features_random_state.csv`
- `linear_eval_game_h0.csv` when the default evaluator is not skipped

`lp_solution.npz` includes:

- `weights` (policy weights only, feature dimension; backward-compatible consumer key)
- `policy_weights` (same as `weights`)
- `sample_errors` (one value per transition sample, derived post-solve from directional residuals)
- `max_error` (`t`)
- `full_solution` (concatenated `[w, t]` in minimax mode; `[w]` in avg mode)
