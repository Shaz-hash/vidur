# A100 TP1 mew1 GPU 2 Simulator Profile

This package builds the Vidur execution-time predictor used by GV3 from measurements taken on physical GPU 2 of `mew1`.

## Fixed Configuration

- Hardware: NVIDIA A100 80GB PCIe, physical GPU 2 on `mew1`.
- Model geometry: `meta-llama/Meta-Llama-3-8B`.
- Precision: FP16.
- Tensor parallelism: TP1.
- Pipeline parallelism: PP1.
- Predictor: Vidur random forest with the repository's default estimator grid.
- Maximum tokens per request: 8192.
- Maximum prediction batch size: 256.
- Maximum prefill chunk: 4096.
- KV-cache prediction granularity: 64 tokens.
- Prefill prediction granularity: 32 tokens.
- Exported GV3 prefill table step: 128 tokens.

The raw MLP sweep has 326 rows. The raw attention sweep has 14,734 rows covering prefill and decode configurations through an 8192-token context.

## Layout

```text
A100_TP1_MEW1_GPU2/
  raw/compute/                 immutable copies of GPU profiling CSVs
  raw/real_vllm/               measured vLLM prefill reference data
  artifacts/raw_cache/         uncalibrated sklearn models and prediction tables
  artifacts/cache/             vLLM-calibrated models and prediction tables
  artifacts/raw_prefill_profile.csv
  artifacts/raw_decode_profile.csv
  artifacts/prefill_profile.csv
  artifacts/decode_profile.csv
  artifacts/calibration.json   auditable backend calibration and evidence
  artifacts/validation/        comparison against real vLLM
  build_simulator_model.sh      reproducible end-to-end build
```

The uncalibrated predictor consumes stable copies under:

```text
data/profiling/compute/a100_mew1_gpu2/meta-llama/Meta-Llama-3-8B/
```

The deployable predictor consumes calibrated copies under:

```text
data/profiling/compute/a100_mew1_gpu2_vllm013_calibrated/meta-llama/Meta-Llama-3-8B/
```

Paths below `data/profiling` keep Vidur cache hashes stable when the repository is copied to another machine.

## Build

From the repository root:

```bash
VIDUR_PROFILE_TRAIN_JOBS=32 \
  vidur/Profiler/A100_TP1_MEW1_GPU2/build_simulator_model.sh
```

The script does not replace the repository-wide `simulator_output/prefill_profile.csv`. It builds and validates isolated artifacts first.

## Backend Calibration

The real-vLLM comparison uses median end-to-end batch duration from five runs per prefill size. The raw simulator predictor uses isolated Sarathi/FlashInfer kernel timing, while the real run used vLLM 0.13 with FlashAttention and eager execution.

The build records both models and applies one global backend factor to every profiled GPU-operation timing. The factor is the median `real_vllm / raw_simulator` ratio across the measured prefill anchors. It is not a per-token fit. Its derivation and all input points are preserved in `artifacts/calibration.json`.

The current factor is `0.883200462385`. The calibrated seven-point validation has 3.02% mean absolute error, 6.86% p95 error, and 7.68% maximum error. All seven measured points are within 10% of real vLLM.

## Deployment Contract

Future GV3 experiments must use all three components together:

1. The calibrated profile paths under `data/profiling/compute/a100_mew1_gpu2_vllm013_calibrated`.
2. The matching files under `artifacts/cache`.
3. The generated `artifacts/prefill_profile.csv` and `artifacts/decode_profile.csv` for Python/native feature and SLO consistency.

Runtime predictor limits must remain 8192 tokens, batch size 256, and maximum prefill chunk 4096. A changed limit changes predictor cache identity and requires regeneration.

Use predictor cache mode `require_cache` after deployment. A strict cache-only load was verified locally, and Python/native prefill lookup matched exactly over 39 profile and boundary queries.
