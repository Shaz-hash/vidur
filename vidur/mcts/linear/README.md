# Linear Bellman Pipeline

This directory implements a standalone **value-only** self-improvement loop for Vidur MCTS environments.

It is intentionally separate from:
- `vidur/mcts/alphaZeroParrallel.py`
- `vidur/mcts/mctsDNN.py`
- native C++ MCTS/inference paths

No policy heads are trained here. No arena matchup logic is used.

## Goal

Learn a linear controller value function `V(s)` from Bellman-style targets over branching controller states.

For each sampled branching state `s`:
- enumerate valid controller actions `a`
- compute one-step targets
- set:
  - `Q(s,a) = r(s,a) + gamma_eff * V_hat(s')`
  - `y(s) = max_a Q(s,a)`

Train a linear model to regress `V(s) -> y(s)` with MSE.

## Bellman Definitions

Controller perspective only.

- `cost(s) = slo_violations(s) + lateness_sum(s)`
- `r(s,a) = cost(s) - cost(s')`
- `gamma_eff = discount_factor ** (delta_time / base_step_time)`
- `Q(s,a) = r(s,a) + gamma_eff * V_hat(s')`
- `target y(s) = max_a Q(s,a)`

`base_step_time` is calibrated the same way as current MCTS-DNN style:
- lookup prefill time for step tokens (usually 512)
- divide by prefill slowdown

## Feature Vector (30 dims)

Implemented in `features.py` as `extract_features(...)`.

### A) Queue/load
1. `n_waiting_total`
2. `n_prefill_active`
3. `n_decode_active`
4. `sum_prefill_tokens_remaining`
5. `sum_decode_tokens_remaining`
6. `max_prefill_tokens_remaining`
7. `max_decode_tokens_remaining`
8. `mean_prefill_tokens_remaining`
9. `mean_decode_tokens_remaining`
10. `sum_total_tokens_remaining`

### B) Deadline/lateness pressure
11. `min_prefill_slack_sec`
12. `mean_prefill_slack_sec`
13. `prefill_overdue_count`
14. `prefill_overdue_lateness_sum_sec`
15. `min_decode_slack_sec`
16. `mean_decode_slack_sec`
17. `decode_overdue_count`
18. `decode_overdue_lateness_sum_sec`
19. `cum_slo_violations`
20. `cum_lateness_sum_sec`

### C) Time-phase
21. `sim_time_sec`
22. `phase_in_1s_cycle` (`sim_time % 1.0`)
23. `time_to_next_adversary_inject_sec` (`1.0 - phase`, clipped)

### D) Piecewise buckets
24. `decode_bucket_0`
25. `decode_bucket_1_2`
26. `decode_bucket_3_4`
27. `decode_bucket_5_8`
28. `decode_bucket_9_plus`
29. `prefill_bucket_0`
30. `prefill_bucket_1_plus`

## Model

`model.py`:
- `LinearValueModel(num_features=30)`
- single linear head: `nn.Linear(30, 1, bias=True)`
- stores feature standardization stats as buffers:
  - `feat_mean`
  - `feat_std`

Training standardizes input as:
- `z = (x - mean) / std`
- `V_hat = Wz + b`

## Data Collection Strategy

Implemented in:
- `collector_worker.py`
- `collector_parallel.py`

Per worker:
1. Build isolated env/simulator clone.
2. Create root state:
   - optionally replay `--history-csv`
   - apply history hops via `HistoryRootGenerator`
3. Advance to controller branching states.
4. BFS over branching controller states.
5. For each branching state:
   - enumerate all valid controller actions
   - step to `s'` (controller + deterministic adversary progression to next controller branching point)
   - compute `Q(s,a)` with model bootstrap
   - store one sample with `target = max_a Q(s,a)`
6. Fill eval set first, then train set.
7. If queue drains, reseed from previously seen branching snapshots.

### Deterministic adversary behavior

When adversary turn is needed during progression:
- choose max valid action index (`max(valid)`), corresponding to max request action in current indexing.

## Parallelism

Configured by `--workers`.

Normal path (`workers > 1`):
- multiprocessing spawn workers
- one worker result shard each

Single-worker fallback (`workers == 1`):
- runs in-process (no multiprocessing queue/semaphore)
- useful in restricted environments

## Training

Implemented in `trainer.py`.

- loss: MSE only
- optimizer: Adam (default) or SGD
- random shuffled minibatches
- metrics per epoch:
  - train/eval MSE
  - train/eval MAE
  - train/eval R²

## Greedy Rollout Evaluation

Implemented in `rollout_eval.py`.

After each round:
- run `--num-traces` traces from seeded roots
- at each controller step choose greedy action by max one-step `Q(s,a)`
- log per-step:
  - state hash
  - chosen action
  - all Q values
  - `V_hat(s)`
  - cost/reward/time discount details

## Orchestration

`pipeline.py` (`run_self_improvement`):
1. Save current model snapshot for collection.
2. Collect train/eval samples in parallel.
3. Train for `epochs`.
4. Evaluate and save metrics.
5. Run greedy rollouts and save traces.
6. Save updated model checkpoint.
7. Append one row to `summary.csv`.

## CLI

Entrypoint:

```bash
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m vidur.mcts.linear.run --help
```

Main options:
- `--rounds`
- `--workers`
- `--train-samples-per-worker`
- `--eval-samples-per-worker`
- `--history-hops`
- `--history-csv`
- `--root-player`
- `--discount-factor`
- `--epochs`
- `--batch-size`
- `--optimizer`
- `--device`
- `--use-virtual-env` / `--use-real-env`
- `--out-dir`

### Example (full shape)

```bash
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m vidur.mcts.linear.run \
  --rounds 2 \
  --workers 8 \
  --train-samples-per-worker 10000 \
  --eval-samples-per-worker 1000 \
  --history-hops 0,5,10,15,20,25,30,35 \
  --history-csv simulator_output/mcts_dnn_logs/sample_history.csv \
  --discount-factor 0.98 \
  --epochs 8 \
  --batch-size 32 \
  --optimizer adam \
  --device cpu \
  --use-virtual-env \
  --out-dir simulator_output/linear_value
```

### Example (smoke)

```bash
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m vidur.mcts.linear.run \
  --rounds 1 \
  --workers 1 \
  --train-samples-per-worker 32 \
  --eval-samples-per-worker 8 \
  --epochs 1 \
  --batch-size 8 \
  --num-traces 1 \
  --max-rollout-steps 8 \
  --history-hops 0 \
  --device cpu \
  --out-dir simulator_output/linear_value_smoke
```

## Output Layout

Under `--out-dir`, per round:

- `round_XXX/model_for_collection_round_XXX.pt`
- `round_XXX/worker_YY_train.npz`
- `round_XXX/worker_YY_eval.npz`
- `round_XXX/worker_YY_train_meta.csv`
- `round_XXX/worker_YY_eval_meta.csv`
- `round_XXX/train_samples_round_XXX.npz`
- `round_XXX/eval_samples_round_XXX.npz`
- `round_XXX/train_samples_round_XXX_meta.csv`
- `round_XXX/eval_samples_round_XXX_meta.csv`
- `round_XXX/metrics_round_XXX.csv`
- `round_XXX/greedy_trace_worker_YY.csv`
- `round_XXX/rollout_summary_round_XXX.csv`
- `round_XXX/model_round_XXX.pt`

Top-level:
- `config.json`
- `summary.csv`

## Tests

Tests live in `vidur/mcts/tests/linear/`:
- `test_bellman.py`
- `test_features.py`
- `test_trainer_smoke.py`

If `pytest` is available:

```bash
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m pytest -q vidur/vidur/mcts/tests/linear
```

## Current Limitations / Notes

- `--history-hop-mode` currently exists but is functionally an alias (`fixed` behavior).
- Multi-worker path relies on multiprocessing spawn; ensure host allows process semaphores.
- In some environments, simulator startup may trigger expensive predictor warm-up.
- This pipeline is value-only by design; no policy/adversary model training is done here.

## File Map

- `config.py` - dataclasses for run/config surface
- `bellman.py` - cost/reward/discount/Q helpers
- `features.py` - 30-feature extraction + state hashing + stats helpers
- `model.py` - linear value model
- `collector_worker.py` - per-worker BFS collector
- `collector_parallel.py` - worker orchestration + shard merge
- `trainer.py` - value-only training loop
- `rollout_eval.py` - greedy post-train trace evaluation
- `pipeline.py` - end-to-end round orchestration
- `run.py` - CLI entrypoint
