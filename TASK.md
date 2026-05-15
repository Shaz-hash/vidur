# Task: GV3 Controller Value Function, Classical Model Search

## Worktree Contract

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

Build the best non-neural model you can for the GV3 controller value function on the fixed ModelSearchBed root-state dataset.

This is controller-perspective value learning only. The game is zero-sum/minimax, so we are not trying to learn a separate adversary value function here. Focus on `root_player == "controller"` records and predict `target_value`.

Allowed approaches include:

```text
decision trees, random forests, extra trees, gradient boosted trees, linear models, generalized additive models, nearest-neighbor style methods, rule-based models, symbolic/hand-engineered formulas, feature crosses, calibration layers that are not neural networks.
```

Not allowed:

```text
neural networks, MLPs, DNNs, attention, transformers, torch trainable neural modules, learned embeddings, differentiable NN-style representation learning.
```

Hard parameter/model-size budget:

```text
effective trainable/scalar parameters <= 150,000
```

For tree models, count approximately by number of learned split/value scalars. Keep the final model compact and explain the model-size estimate.

## Fixed Dataset

Use this dataset as read-only input:

```text
/home/shazer/Desktop/Research/Vidur/vidur/simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40
```

Current known size:

```text
292,713 controller root records
~40.5% have abs(target_value) >= 1
```

Do not generate a new root dataset.

Do not pass:

```text
--allow-dataset-generation
--overwrite
```

Use deterministic train/eval split with:

```text
--eval-ratio 0.2
```

## Target Metrics

The goal is extremely low error on both train and eval splits:

```text
MSE < 0.1
RMSE < 0.05
MAE < 0.05
p95_abs_error < 0.05
max_abs_error < 0.1
```

If those are not achievable, provide the best model you can find and clearly report the closest metrics. Do not hide failure cases. Include what you tried and why the final model is the best candidate.

## Allowed Code Changes

You may modify files inside this worktree only.

Primary files you may modify:

```text
vidur/mcts/Game_Versions/Game_Version3/DNN/trainer.py
vidur/mcts/Game_Versions/Game_Version3/DNN/infer.py
vidur/mcts/Game_Versions/Game_Version3/ModelSearchBed/self_model_test.py
```

You may add classical-model helper files under:

```text
vidur/mcts/Game_Versions/Game_Version3/ModelSearchBed/
```

Avoid modifying neural-model files unless it is only to bypass/disable unused neural fallback paths:

```text
vidur/mcts/Game_Versions/Game_Version3/DNN/models.py
vidur/mcts/Game_Versions/Game_Version3/DNN/value_models.py
vidur/mcts/Game_Versions/Game_Version3/DNN/dnn_spec.py
```

Do not modify root generation/storage code unless absolutely necessary:

```text
vidur/mcts/Game_Versions/Game_Version3/ModelSearchBed/root_storage.py
```

If you believe `root_storage.py` must change, document the reason first. The stored dataset is already produced and should remain read-only.

## Harness Contract

`self_model_test.py` is the experiment harness. It should:

1. Load the fixed dataset.
2. Split train/eval with `eval_ratio=0.2`.
3. Call trainer code to fit the candidate classical model.
4. Call infer code to produce one scalar prediction per record.
5. Report train/eval metrics.

Feature extraction, model choice, objective, fitting procedure, and inference representation are your responsibility. You can implement custom hooks:

```python
# in DNN/trainer.py
def train_model_search(train_records, eval_records, cfg, state_loader, output_dir):
    ...

# in DNN/infer.py
def predict_model_search_values(model, records, cfg, state_loader, split_name):
    ...
```

The harness passes raw stored root records. Each record contains simulator snapshot, stats, metadata, and `target_value`. Use `state_loader(record)` only if you need to reconstruct the GV3 state; avoid doing that in hot loops unless needed.

## Feature Guidance

You are free to build any non-neural feature representation from:

```text
simulator_snapshot
stats
request metadata
active/prefill/decode request state
timing/cost/SLO fields
history depth
best_action metadata, if using it does not leak target directly
```

Do not leak `target_value` into features. Do not use `best_reward`, `best_child_cost`, or equivalent target-construction fields as input features if the deployed inference path would not know them. Those fields are labels/diagnostics, not state features.

## Run Commands

Use the original repo venv but this worktree as `PYTHONPATH`:

```bash
cd /home/shazer/Desktop/Research/Vidur/vidur-classical-search

PYTHONPATH=$PWD \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
  -m vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.self_model_test \
  --dataset-dir /home/shazer/Desktop/Research/Vidur/vidur/simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40 \
  --output-dir $PWD/simulator_output/GV3_Agent/model_search_results/classical_agent \
  --num-roots 292713 \
  --max-candidate-roots 292713 \
  --root-player-filter controller \
  --eval-ratio 0.2 \
  --batch-size 256 \
  --model-name classical_agent_candidate
```

For quick smoke tests, use a smaller `--num-roots`, for example `2048`, but final reported metrics must use the largest practical dataset size.

## Output Requirements

Write outputs only under:

```text
$PWD/simulator_output/GV3_Agent/model_search_results/classical_agent/
```

Do not commit generated checkpoints, CSV outputs, or large artifacts unless explicitly requested.

At the end, provide:

```text
best model description
model-size / parameter estimate
feature design
training/fitting setup
train metrics
eval metrics
exact command used
files changed
known failure modes or limitations
```
