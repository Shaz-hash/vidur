# Game Version 3: Bellman Value Learning for Vidur Scheduling

This document describes the current Game Version 3 implementation in the Python GV3 pipeline. The main implementation lives under `vidur/mcts/Game_Versions/Game_Version3/`.

The system is a two-player sequential scheduling game over the Vidur simulator. The controller tries to keep request service within SLOs. The adversary creates or stops workload to expose schedules that cause SLO cost. The ML model is trained as a value model that estimates the controller-perspective value of a game state, and the search code uses this value estimate inside a Bellman-style one-step backup.

## Core Files

The main code paths are:

| Area | File |
| --- | --- |
| Main GV3 config | `config.py` |
| Virtual game environment | `virtual_environment.py` |
| Action generation and action masks | `player_sample_actions.py` |
| Bellman/depth-1 search | `mctsDNN.py` |
| State feature construction | `DNN/infer.py` |
| Model architecture | `DNN/value_models.py` and `DNN/dnn_spec.py` |
| History/frontier root generation | `DNN/history_root.py` |
| Self-play sample writer flow | `DNN/selfPlay.py` |
| Replay sample format | `DNN/replay_write.py` |
| Replay loading | `DNN/replay_buffer.py` |
| Training loop | `DNN/trainer.py` and `Network/server/training.py` |
| Distributed experiment launcher | `Network/server/run_experiment.py` |
| Distributed orchestration | `Network/server/orchestrator.py` |
| Worker task execution | `multiProcessUtils.py` and `Network/client/run_task.py` |
| Test documentation | `tests/README.md` |

## Game Setting

GV3 models scheduling as an alternating game between two players:

| Player | Goal | Turn effect |
| --- | --- | --- |
| `adversary` | Make the controller incur high SLO cost | Launches new requests and may stop active decode requests |
| `controller` | Minimize SLO cost | Chooses prefill allocation, decode allocation, and eviction behavior |

The environment state is a `VidurMCTSState` wrapping a `VirtualSimulator`, plus game stats tracking SLO violations, lateness, credits, adversary ticks, and transition timing. The main state transition code is in `virtual_environment.py`.

The objective cost used by the value backup is implemented in `mctsDNN.py` through `_state_cost`, and the underlying violation/lateness accounting is exposed through `VirtualVidurMCTSEnvironment.evaluate_objective()`.

Current cost configuration in `config.py`:

| Config | Value | Meaning |
| --- | ---: | --- |
| `CostConfig.violation_base_cost` | `1.0` | Base penalty after a request violates SLO |
| `CostConfig.drop_cost` | `3.0` | Cost for dropped/evicted request path |
| `CostConfig.lateness_cap_sec` | `2.0` | Cap for lateness contribution |
| `CostConfig.auto_drop_lateness_sec` | `2.0` | Lateness threshold for automatic drop behavior |

The scalar cost currently used by search is:

```text
state_cost = number_of_violated_requests + total_lateness_sum
```

The immediate reward used by the Bellman backup is the negative increase in that cost:

```text
reward = -(state_cost_after - state_cost_before)
```

The reward is then shaped by `reward_knee` and `reward_max_penalty` in `MCTSSearchConfig` before it is used in Q-value computation.

## Config 3 Settings

The authoritative config is `config.py`. The top-level object is named `DEFAULT_GAME_V2_CONFIG` for historical compatibility, but it is the GV3 game config used by this directory.

### Timing

From `TimingConfig`:

| Setting | Value | Meaning |
| --- | ---: | --- |
| `adversary_tick_sec` | `0.2` | Adversary decision grid |
| `launch_window_sec` | `1.0` | Sliding request-launch window |
| `max_requests_per_launch_window` | `7` | Maximum requests adversary can launch per 1 second window |
| `controller_noop_prefill_only_jump_to_next_adv_tick` | `True` | If controller strict no-ops while only prefill exists, jump to next adversary tick |

The environment has an important timing distinction:

| Time | Meaning |
| --- | --- |
| `transition_discount_time` | Time immediately after the selected action finishes |
| `transition_final_time` | Time after environment-internal fast-forward, if decode-only progress occurs before the next adversary tick |

`mctsDNN.py` uses `transition_discount_time` for Bellman discounting. This avoids over-penalizing actions when time advances only because the simulator fast-forwards decode-only internal work after the action.

### Credits

From `CreditConfig`:

| Setting | Value | Meaning |
| --- | ---: | --- |
| `prefill_credit_mint_per_adv_tick` | `1024` | Prefill credit created per adversary tick |
| `prefill_credit_expiry_sec` | `1.0` | Prefill credit lifetime |
| `decode_credit_mint_per_prefill_complete` | `216` | Decode credit created when a request finishes prefill |
| `enforce_nonnegative_decode_credits` | `True` | Decode credits cannot go below zero |

The controller sampler uses these credits when generating valid controller actions. See `player_sample_actions.py`.

### Request Space

From `RequestConfig`:

| Setting | Value |
| --- | ---: |
| `max_prefill_tokens_per_request` | `4096` |
| `max_decode_tokens_per_request` | `864` |
| `min_decode_tokens_per_request` | `1` |
| `allowed_prefill_tokens` | `128, 256, 512, 1024, 1536, 2048, 3072, 4096` |
| `target_decode_tokens_per_request_avg` | `216` |
| `target_prefill_tokens_per_request_avg_window` | `1024` |

### Bellman Search

From `MCTSSearchConfig`:

| Setting | Value | Meaning |
| --- | ---: | --- |
| `discount_factor` | `0.98` | Base time discount factor |
| `discount_time_denominator_sec` | `0.015725797204323228` | Time denominator for exponent |
| `reward_knee` | `25.0` | Reward shaping knee |
| `reward_max_penalty` | `40.0` | Reward shaping cap |

The time discount is computed as:

```text
discount = discount_factor ** ((child_discount_time - parent_time) / discount_time_denominator_sec)
```

### Training and Data

From `MultipleProcessTrainingConfig` and `TrainerHyperParams`:

| Setting | Current value |
| --- | ---: |
| `environment_lang` | `python` by default, can be `native` |
| `num_processes` | `60` |
| `max_concurrent_selfplay_workers` | `60` |
| `roots_per_generation` | `40000` |
| `sample_cycles_per_generation` | `8` |
| `train_batch_size` | `256` |
| `train_target_epochs_per_generation` | `20.0` |
| `eval_split_ratio` | `0.1` |
| `replay_capacity_samples` | `400000` |
| `replay_max_cached_shards` | `5000` |
| `trainer.lr` | `1e-4` |
| `trainer.weight_decay` | `1e-4` |
| `trainer.value_only` | `True` |
| `trainer.policy_weight` | `0.0` |
| `trainer.value_weight` | `1.0` |

The fields `adv_iterations_per_root` and `cont_iterations_per_root` are still present because the native pipeline has an iteration budget. The current Python GV3 self-play path uses depth-one value backup and does not consume those fields for Python search.

## Bellman Hypothesis

The central hypothesis is that if the model learns the controller-perspective value function accurately, then greedy Bellman selection should produce a strong scheduling policy.

The value is always controller-perspective:

```text
higher value = better for controller
lower value = better for adversary
```

For a controller state:

```text
V(s, controller) = max_a [ r(s, a) + discount(s, a) * V(s_after_a, adversary) ]
```

For an adversary state, the adversary chooses the action that minimizes the controller value. In the current Python search code, adversary root actions are evaluated with an immediate best controller response:

```text
V(s, adversary) = min_a_adv [
    r_adv + discount_adv * max_a_ctrl [
        r_ctrl + discount_ctrl * V(s_after_adv_ctrl, adversary)
    ]
]
```

This is implemented in `mctsDNN.py`:

| Function | Role |
| --- | --- |
| `_compose_q_from_state` | Builds `Q = reward + discount * bootstrap` |
| `_evaluate_depth1_action_q` | Evaluates one controller action |
| `_evaluate_adversary_action_q_two_step` | Evaluates one adversary action followed by best controller response |
| `_select_depth1_best_action_index` | Selects `max Q` for controller and `min Q` for adversary |
| `search_dnn` | Produces the root target value and selected action |

The Bellman target is not an external supervised label. It is generated by this search procedure, then written into replay as `mcts_value_controller`.

## Methodology

Each generation follows this flow:

1. Build frontier states from the initial simulator state using `HistoryRootGenerator` in `DNN/history_root.py`.
2. At each frontier root, run Python depth-one Bellman search through `SelfPlayRunner.run_single_root()` in `DNN/selfPlay.py`.
3. Build model features from that root state through `DNN/infer.py`.
4. Store the action mask, selected-action one-hot target, value target, and metadata through `DNN/replay_write.py`.
5. Write replay shards under `simulator_output/Game_Version3/mcts_dnn_dataset`.
6. Train `AlphaZeroModel` locally from replay shards through `Network/server/training.py` and `DNN/trainer.py`.
7. Save checkpoints under `simulator_output/Game_Version3/mcts_dnn_checkpoints`.
8. Append train/eval losses to `simulator_output/Game_Version3/mcts_dnn_logs/eval_metrics.csv`.

The training objective is value-only. Older replay structures may still carry compatibility fields, but the active GV3 objective is only the scalar value target.

## What the Adversary Can Do

The adversary action space is generated in `player_sample_actions.py` by `GV2PlayerActionSampler.sample_adversary_actions()`.

The flattened adversary action space has:

```text
num_stop_rules + max_launch_count * num_prefill_templates * num_stop_rules
= 5 + 7 * 8 * 5
= 285 actions
```

The adversary can:

| Capability | Details |
| --- | --- |
| Launch zero requests | One branch per stop rule |
| Launch 1 to 7 requests | Request count is bounded by `max_launch_count_per_tick` |
| Select a prefill template | One of `128, 256, 512, 1024, 1536, 2048, 3072, 4096` prefill tokens |
| Stop decode requests | Based on stop rules |
| Obey launch-window limits | Request count and prefill-token window caps are enforced by mask |

Stop rules are configured in `AdversaryActionConfig`:

| Stop rule | Meaning |
| --- | --- |
| `stop_none` | Stop no decode request |
| `stop_longest_decode` | Stop active decode with largest processed decode |
| `stop_shortest_decode` | Stop active decode with smallest processed decode |
| `stop_all_decodes_over_512` | Stop all active decodes with more than 512 processed decode tokens |
| `stop_all_decodes_over_216` | Stop all active decodes with more than 216 processed decode tokens |

The adversary mask is strict. Invalid launches or stops are masked out instead of being silently clamped.

## What the Controller Can Do

The controller action space is generated in `player_sample_actions.py` by `GV2PlayerActionSampler.sample_controller_actions()`.

The flattened controller action space has:

```text
num_eviction_rules * num_prefill_budget_options * num_ordering_heuristics
= 9 * 9 * 4
= 324 actions
```

The controller action contains:

| Component | Meaning |
| --- | --- |
| Eviction rule | Which request, if any, to evict or drop |
| Prefill budget | How many prefill tokens to allocate this turn |
| Ordering heuristic | How to order eligible prefill requests for allocation |
| Decode allocation | Decode credit allocation, generally 1 token per eligible decode while credit allows |

Eviction rules:

| Rule |
| --- |
| `evict_none` |
| `evict_largest_prefill` |
| `evict_earliest_prefill_deadline` |
| `evict_prefill_missed_deadline` |
| `evict_prefill_lateness_over_0p5` |
| `evict_longest_decode` |
| `evict_decode_lateness_over_0p5` |
| `evict_prefill_highest_lateness` |
| `evict_decode_highest_lateness` |

Prefill budgets:

```text
0, 128, 256, 512, 1024, 1536, 2048, 3072, 4096
```

Ordering heuristics:

| Heuristic | Meaning |
| --- | --- |
| `SJF` | Shortest job first |
| `EDF` | Earliest deadline first |
| `LST` | Least slack time first |
| `LJF` | Longest job first |

If there are no requests, only the no-op controller action is valid. If the budget is zero, duplicate zero-budget heuristic branches are suppressed so only the canonical zero-budget action remains valid.

## Environment Transitions

The environment applies adversary and controller actions in `virtual_environment.py`.

Adversary transition:

| Step | Implementation |
| --- | --- |
| Drain simulator arrivals | `_drain_arrivals()` |
| Apply launches/stops | `apply_adversary_action_only()` |
| Update stats and timing | `VidurGameStats` fields |

Controller transition:

| Step | Implementation |
| --- | --- |
| Drain arrivals and check pending adversary tick | `apply_controller_action_only()` |
| Apply evictions | Controller action handling |
| Run selected batch through predictor | Virtual simulator execution path |
| Update SLO stats and credit state | `GV2RuntimeOps` helpers |
| Record action-end timing | `transition_discount_time` |
| Optionally fast-forward decode-only work | `_maybe_fast_forward_decode_only_to_next_adv_second()` |
| Record final transition time | `transition_final_time` |

The distinction between action-end time and final fast-forward time matters for Bellman discounting. The search uses action-end time for discounting but the simulator state is still advanced correctly to the next decision point when decode-only internal events must happen.

## ML Model Inputs

Feature construction is implemented in `DNN/infer.py`.

The model receives split tensors:

| Tensor | Shape |
| --- | --- |
| `prefill_req_features` | `[1, 10, 10]` |
| `decode_req_features` | `[1, 50, 13]` |
| `global_features` | `[1, 24]` |
| `prefill_req_mask` | `[1, 10]` |
| `decode_req_mask` | `[1, 50]` |
| `action_mask` | `[1, num_actions_for_player]`, used by search to know which actions are valid |

Prefill request rows are sorted by deadline pressure and truncated/padded to 10 rows. Decode request rows are sorted by violation/lateness/processed-token priority and truncated/padded to 50 rows.

The feature conversion tests under `tests/feature_conversion_tests.py` verify that the tensors built for frontier states match the simulator state and history traces.

## ML Model Architecture

The model is `AlphaZeroModel` in `DNN/value_models.py`. The spec is defined in `DNN/dnn_spec.py`.

Default architecture:

| Component | Value |
| --- | ---: |
| Request embedding width | `64` |
| Transformer trunk width | `128` feed-forward |
| Attention heads | `4` |
| Attention layers | `2` |
| Dropout | `0.10` |
| Prefill rows | `10` |
| Decode rows | `50` |
| Global features | `24` |

The architecture is:

1. Encode prefill request rows with a small MLP.
2. Encode decode request rows with a small MLP.
3. Encode global features with a small MLP.
4. Add type embeddings for CLS, prefill, and decode tokens.
5. Add player embedding to the CLS/global token.
6. Run a Transformer encoder over `[CLS, prefill tokens, decode tokens]`.
7. Pool prefill and decode token outputs with masked mean.
8. Concatenate CLS output, prefill pool, decode pool, and global token.
9. Feed the summary through separate controller/adversary value heads.

The GV3 pipeline uses the model as a value predictor. Search passes a state through the model to get a scalar value estimate for that state, then Bellman backup chooses actions from simulated Q-values. The model is not used to infer a policy in the current GV3 pipeline.

## Value Scaling and Loss

The model predicts a normalized value in `[-1, 0]`, then denormalizes it back to real controller-perspective value.

Value range settings from `DNN/dnn_spec.py`:

| Setting | Value |
| --- | ---: |
| `v_max` | `0.0` |
| `v_min` | `-50.0` |
| `v_linear_min` | `-48.0` |
| `v_norm_min` | `-1.0` |
| `v_norm_max` | `0.0` |
| `v_linear_norm_min` | `-0.98` |
| `v_tail_compress_power` | `2.0` |

Training loss in `DNN/trainer.py`:

```text
value_loss = SmoothL1(normalized_prediction, normalized_target)
```

Real-scale MSE and MAE are also logged for both players. Because `value_only=True` and `policy_weight=0.0`, training optimizes only the value prediction.

## Sample Generation

Sample generation is driven by `SelfPlayRunner.run_n_roots()` in `DNN/selfPlay.py`.

The process is:

1. The worker receives a root quota and a history-hop range.
2. `HistoryRootGenerator` builds frontier roots by walking from the initial state.
3. Forced single-action chains are applied but do not count as history hops.
4. Branching nodes with multiple valid actions count as nontrivial history hops.
5. A random valid action is selected during history generation.
6. Each emitted frontier is advanced through any forced chain until it reaches a branchable root.
7. The root is searched with `search_dnn(..., one_step_value_mode=True)`.
8. The root sample is written to replay.

History-root settings in `config.py`:

| Setting | Current value |
| --- | ---: |
| `history_hops_min` | `0` |
| `history_hops_max` | `199` |
| `history_hop_interval_width` | `5` |
| `history_root_batch_size` | `64` |
| `history_max_total_steps` | `20000` |
| `max_forced_hops_per_root` | `1024` |
| `history_allow_duplicate_root_fallback` | `False` |
| `log_history_rows` | `True` |

If explicit `history_hops_per_worker` is empty, `selfplay_hop_ranges_for_generation()` assigns process `wid` the interval:

```text
lo = history_hops_min + wid * history_hop_interval_width
hi = lo + history_hop_interval_width - 1
```

With the current defaults, process 0 gets hops `[0, 4]`, process 1 gets `[5, 9]`, and so on.

## Replay Dataset

Replay samples are written by `DNN/replay_write.py`.

Each root sample contains:

| Field | Meaning |
| --- | --- |
| `feature_version` | Feature schema version |
| `game_id`, `root_id`, `root_node_id`, `root_depth` | Root identity |
| `player` | Root player |
| `model_inputs` | Prefill, decode, global, and mask tensors |
| `action_mask` | Valid action mask for the root player |
| `mcts_policy` | One-hot selected action retained in replay for old data structure compatibility; not used as the active GV3 learning objective |
| `mcts_value_controller` | Bellman target value from search |
| `meta.best_action_index` | Selected action index |
| `meta.search_mode` | Currently `depth1_value_backup` |
| `meta.used_bootstrap` | Whether model bootstrap was used |
| `meta.history_hops` | Number of history branching hops |

Shards are written under:

```text
simulator_output/Game_Version3/mcts_dnn_dataset
```

The network server validates and repairs generation datasets from received worker results before training. See `Network/server/dataset_integrity.py` and the callsite in `Network/server/orchestrator.py`.

## Distributed Sampling Over Machines

The distributed experiment is launched by `Network/server/run_experiment.py`, which starts one orchestrator process per generation.

The orchestrator in `Network/server/orchestrator.py` does the following:

1. Resolves the output namespace, dataset directories, checkpoint directories, and logs.
2. Splits `roots_per_generation` into `sample_cycles_per_generation` collection cycles.
3. Splits each cycle across the configured remote machines.
4. Writes a `task.json` for each machine.
5. Sends task JSON and model weights through SSH/rsync.
6. Runs `Network/client/run_task.py` remotely.
7. Receives replay shards and metadata back from each machine.
8. Mirrors results into the canonical generation dataset layout.
9. Validates or repairs replay shards.
10. Trains locally on the configured training device.
11. Writes checkpoints and eval metrics.
12. Cleans up consumed remote/local task artifacts after successful training.

A typical run command shape is:

```bash
cd /home/shazer/Desktop/Research/Vidur/vidur
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -u -m \
vidur.mcts.Game_Versions.Game_Version3.Network.server.run_experiment \
  --start-generation 0 \
  --num-generations 100 \
  --initial-model-version 1 \
  --session-prefix gv3_experiment \
  --machine aws-native-1-gv3 \
  --machine aws-native-2-gv3 \
  --weights-path simulator_output/Game_Version3/mcts_dnn_checkpoints/best.pt \
  --local-training-device cuda
```

Additional flags after `--` are passed through to the orchestrator. Examples include:

```text
--output-name Game_Version3_Native
--environment-lang native
--worker-model-device cpu
--local-training-device cuda
```

The current Python environment language is controlled by `MultipleProcessTrainingConfig.environment_lang` or by the orchestrator CLI. Native mode has separate implementation details and still uses native iteration-budget fields. This README is primarily describing the Python GV3 logic because that is where the current Bellman-selection fixes and tests were reviewed.

## Evaluation

Evaluation settings live in `EvaluationGroup` inside `config.py`.

Important fields:

| Setting | Current value |
| --- | ---: |
| `num_games` | `70` |
| `max_history_hops` | `100` |
| `arena_time_limit_sec` | `5.0` |
| `arena_iters_adversary` | `2000` |
| `arena_iters_controller` | `2000` |
| `arena_win_threshold` | `0.70` |

The model tester code lives under `Model_Tester/`. Arena outputs are written under:

```text
simulator_output/Game_Version3/Model_Tester_Results
```

The arena compares controller/adversary pairings and trivial baselines, depending on `Model_Tester/config.py`.

## Tests and Verification

The GV3 tests live under `tests/` and are documented in `tests/README.md`.

Current test coverage includes:

| Test file | What it checks |
| --- | --- |
| `history_node_tests.py` | Frontier/history traces are replayable and consistent with GV2-style simulator invariants |
| `test_checking_depth1.py` | Depth-one Bellman selection chooses max Q for controller and min Q for adversary |
| `feature_conversion_tests.py` | Model features match simulator/frontier state |
| `consolidated_frontier_tests.py` | Generates frontier roots once, then runs trace, Bellman selection, and feature conversion checks together |

These tests are the main guard against hidden bugs in:

| Risk | Test coverage |
| --- | --- |
| History trace does not reconstruct the frontier state | `history_node_tests.py` |
| Duplicate frontier nodes | `history_node_tests.py` uniqueness checks |
| Controller selects worst action by mistake | `test_checking_depth1.py` |
| Adversary selects controller-friendly action by mistake | `test_checking_depth1.py` |
| Feature tensors do not match active simulator requests | `feature_conversion_tests.py` |
| Combined pipeline drift | `consolidated_frontier_tests.py` |

Before starting an expensive GPU experiment, run the consolidated test from the repository root:

```bash
cd /home/shazer/Desktop/Research/Vidur/vidur
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m \
vidur.mcts.Game_Versions.Game_Version3.tests.consolidated_frontier_tests \
  --num-roots 32
```

## Known Implementation Notes

The Python GV3 path currently uses depth-one value backup. The names `adv_iterations_per_root` and `cont_iterations_per_root` remain in config because the native path still has iteration budgets and older interfaces still pass these fields.

The model is value-only in training. The current GV3 pipeline should be evaluated through value error, Bellman action selection, arena outcomes, and Bellman convergence diagnostics.

The controller must select max Q because Q is controller-perspective. The adversary must select min Q. This is implemented explicitly in `mctsDNN.py`.

The adversary root backup includes a controller response in Python. This is intentional for the current approximation because an adversary launch is only meaningful after considering the controller's immediate best response.

Discounting uses the action-end transition time, not the final post-fast-forward simulator time. This is necessary for states where a controller action leaves no active prefill, causing decode-only internal progress before the next adversary tick.

The native implementation should be reviewed separately before assuming full parity with the Python Bellman search behavior.

## Practical Experiment Checklist

Before launching a long run:

1. Run `tests.consolidated_frontier_tests` with at least 32 roots.
2. Check that `test_checking_depth1_details.csv` shows controller roots selecting the highest valid Q and adversary roots selecting the lowest valid Q.
3. Check that feature conversion summaries have zero or near-zero diffs for generated frontier states.
4. Confirm `config.py` has the intended `environment_lang`, `roots_per_generation`, `sample_cycles_per_generation`, `train_target_epochs_per_generation`, and trainer LR.
5. Confirm `best.pt` exists in the intended output namespace.
6. Confirm remote machines have the same branch/code if running distributed.
7. Start with a small smoke generation if the code or config changed.
