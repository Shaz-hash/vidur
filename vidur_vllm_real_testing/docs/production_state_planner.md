# Production GV3 DNN State Planner

The tested production state-planner entry point is:

```bash
VIDUR_VLLM_GV3_STATE_PLANNER=vidur_vllm_real_testing.native_dnn_mcts_state_planner:ProductionNativeDNNMCTSPlanner
VIDUR_VLLM_GV3_PLANNER=vidur_vllm_real_testing.gv3_adapter:GV3PersistentAdapter
```

Launch it on physical GPU 2 of `mew1` with:

```bash
VIDUR_PHYSICAL_GPU=2 \
bash /home/shaz/vidur/source/vidur-classical-search/vidur_vllm_real_testing/mew1/launch_production_native_dnn_controller.sh \
  /home/shaz/vidur/models/Meta-Llama-3-8B
```

The production class uses the checksummed frozen native MCTS configuration in
`vidur_vllm_real_testing/artifacts/native_mcts_cfg.json`. This avoids importing
the training and analysis dependency stack in the vLLM process.

Only promoted DNN exports are accepted. Startup rejects a bundle unless its
manifest declares `model_family=dnn` and `native_ready=true`, its native files
use `agz_dnn_v2`, and its value and policy metadata use the expected Markov-v2
DNN architectures. The historical native pybind search function still contains
`hgb` in its name for API compatibility; it does not make this an HGB planner.

