# Promoted Native DNN MCTS Planner

The production state-planner entry point is:

```bash
VIDUR_VLLM_GV3_STATE_PLANNER=vidur_vllm_real_testing.native_dnn_mcts_planner:PromotedNativeDNNMCTSPlanner
```

It is nested under the persistent real-execution adapter:

```bash
VIDUR_VLLM_GV3_PLANNER=vidur_vllm_real_testing.gv3_adapter:GV3PersistentAdapter
```

Despite the historical pybind function name
`search_mcts_hgb226_value_prior_hgb`, this integration accepts **only DNNs**.
Startup fails unless `current_model.json` declares `model_family=dnn` and
`native_ready=true`, and unless all native exports use `agz_dnn_v2` with the
Markov-v2 architectures:

- controller value: `agz_markov_value_deepset_v2`
- controller policy: `agz_markov_policy_deepset_v3`
- adversary policy: `agz_markov_policy_deepset_v3`

The controller value remains controller-perspective. The adversary policy is
used only when native search or a policy rollout reaches an adversary turn.

## Search Contract

For each nontrivial controller state, the planner:

1. Receives the persistent adapter's complete native-compatible state.
2. Runs native GV3 MCTS with the promoted controller value, promoted
   controller policy, and promoted adversary policy DNNs.
3. Disables root Dirichlet noise for deterministic real scheduling.
4. Selects the root child by visits, then controller Q, then action index.
5. Parses the native action JSON and validates token budget, active request
   IDs, prefill remaining tokens, decode phase, and evictions.
6. Returns canonical allocations to the persistent adapter, which maps GV3
   integer IDs back to real vLLM request IDs.

No MCTS or GV3 game-engine semantics are reimplemented in the adapter.

## Production Defaults

```bash
VIDUR_VLLM_GV3_MODEL_BUNDLE=/home/shaz/vidur/artifacts/promoted_models/xl3_controller_v197_adv_v212
VIDUR_VLLM_GV3_MCTS_ITERATIONS=2000
VIDUR_VLLM_GV3_DISCOUNT_FACTOR=0.98
VIDUR_VLLM_GV3_PUCT_C=0.5
VIDUR_VLLM_GV3_NATIVE_SEARCH_MODE=full_tree_rollout
VIDUR_VLLM_GV3_ROLLOUT_COUNT=1
VIDUR_VLLM_GV3_ROLLOUT_HORIZON_S=3.0
VIDUR_VLLM_GV3_ROLLOUT_THREADS=1
VIDUR_VLLM_GV3_NATIVE_THREADS=1
```

All values can be overridden before launch. Root noise is always disabled by
the real scheduler planner.

## Mew1 Launch

Physical GPU 2 only:

```bash
VIDUR_PHYSICAL_GPU=2 \
bash /home/shaz/vidur/source/vidur-classical-search/vidur_vllm_real_testing/mew1/launch_native_dnn_controller.sh \
  /home/shaz/vidur/models/Meta-Llama-3-8B
```

The launcher uses `GV3PersistentScheduler`, not the stateless scheduler hook.
The exact Llama weights remain an external artifact and are not included in
the repository.

## Verification

CPU/native integration smoke:

```bash
python -m vidur_vllm_real_testing.native_dnn_planner_smoke \
  --model-bundle "$VIDUR_VLLM_GV3_MODEL_BUNDLE" \
  --iterations 2000 \
  --rollout-count 1 \
  --rollout-horizon-s 3.0
```

Regression tests:

```bash
python -m unittest \
  vidur_vllm_real_testing.tests.test_native_dnn_mcts_planner \
  vidur_vllm_real_testing.tests.test_gv3_adapter \
  vidur_vllm_real_testing.tests.test_gv3_live_adapter
```

