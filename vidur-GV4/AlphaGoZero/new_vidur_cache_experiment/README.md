# New Vidur FlashInfer AlphaGoZero experiment

This experiment uses the uncalibrated Vidur random-forest execution-time model
fitted from the vLLM 0.26.0 / FlashInfer 0.6.14 operation profiles measured on
`mew1`. Measured vLLM batch times are comparison data only and are never used
as simulator targets or calibration factors.

The runtime source of truth is:

```text
settingUpServers/exp3_new_vidur_flashinfer_env.sh
```

The deployed artifacts are isolated under the experiment root:

```text
AlphaGOZERO_new_vidur_flash_infer_models/
  flash-infer_prefill_profile.csv
  vidur_execution_profile/
    raw/meta-llama/Meta-Llama-3-8B/{mlp.csv,attention.csv}
    predictor_cache/
```

Python constructs the predictor in `require_cache` mode. The distributed arena
runner serializes the same predictor component tables into native MCTS, while
the prefill CSV supplies GV3 SLO/profile lookups. The validator checks the
32-point grid, exact agreement with the one-request Vidur predictions, explicit
absence of calibration, cache completeness, and native profile lookup parity.

