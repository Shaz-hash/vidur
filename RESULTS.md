# Classical Model Search Results

## Best Candidate

`bellman_shaped_classical_full`

This is a learned, non-neural controller value surrogate. It is not the
symbolic Bellman recomputation path. The model predicts `V_controller(s)` from
state-derived features only, so it can be used later as the bootstrap term in:

```text
reward(s, a) + discount * V_controller(s')
```

The model shape is:

```text
ExtraTreesRegressor + HistGradientBoostingRegressor base blend
HistGradientBoostingClassifier zero-value gate
HistGradientBoostingRegressor residual tail corrector
```

The design is "Bellman-shaped" because it encodes two properties of the GV3
target without recomputing labels:

```text
controller values are non-positive penalties
many states have exactly/near-zero immediate penalty
rare tail states need a separate residual correction
```

## Model Size

Estimated learned scalar budget:

```text
144,852 tree/node scalars
```

This stays below the requested `<= 150,000` budget.

## Feature Design

Inputs are state-derived only:

```text
simulator_snapshot
stats
root_depth
history_hops
active request state
recent arrivals
deadline/lateness/slack summaries
fixed slots for the 12 most urgent active requests
```

The model does not use:

```text
target_value
best_reward
best_child_cost
best_child_time
best_action_index
best_action_repr
history_signature
```

## Full Dataset Metrics

Dataset:

```text
records: 292,713 controller roots
train/eval split: deterministic 80/20
train samples: 234,170
eval samples: 58,543
```

Train:

```text
MSE:  0.00048147584311664104
RMSE: 0.021942557807070738
MAE:  0.007562050595879555
p95_abs_error: 0.032879289239645004
max_abs_error: 0.8960984349250793
abs_error > 1.0: 0
```

Eval:

```text
MSE:  0.0017447256250306964
RMSE: 0.041769912916245065
MAE:  0.009136867709457874
p95_abs_error: 0.03467610850930214
max_abs_error: 2.597357749938965
abs_error > 1.0: 17
```

This meets the MSE/RMSE/MAE/p95 targets on eval, but it does not meet the
strict `max_abs_error < 0.1` target. The remaining failures are rare
discontinuous states where the true first-layer penalty is near zero but the
tree model predicts about `-1` to `-2.6`, or vice versa.

## Exact Command

```bash
cd /home/shazer/Desktop/Research/Vidur/vidur-classical-search

PYTHONPATH=$PWD \
/usr/bin/time -f 'elapsed=%E maxrss_kb=%M' \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
  -m vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.self_model_test \
  --dataset-dir /home/shazer/Desktop/Research/Vidur/vidur/simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40 \
  --output-dir $PWD/simulator_output/GV3_Agent/model_search_results/classical_agent_full_bellman_shaped \
  --num-roots 292713 \
  --max-candidate-roots 292713 \
  --root-player-filter controller \
  --eval-ratio 0.2 \
  --batch-size 4096 \
  --model-name bellman_shaped_classical_full \
  --extra-config-json '{"classical_backend":"bellman_shaped","n_jobs":-1}'
```

Runtime:

```text
elapsed: 15:40.65
maxrss_kb: 11318432
```

## Files Changed

```text
vidur/mcts/Game_Versions/Game_Version3/ModelSearchBed/classical_value_model.py
vidur/mcts/Game_Versions/Game_Version3/DNN/trainer.py
vidur/mcts/Game_Versions/Game_Version3/DNN/infer.py
vidur/mcts/Game_Versions/Game_Version3/ModelSearchBed/self_model_test.py
RESULTS.md
```

## Known Limitation

The current classical model is a strong average-error bootstrap candidate, but
not a safe exact value oracle. If max error must be below `0.1`, the next step
should be either:

```text
a learned neural model with better representation of request interactions
a classical action-conditional model that predicts immediate reward for each controller action
a small exact/probe feature set that computes selected action-transition risk without full symbolic Bellman recomputation
```
