# Build Results

The A100 TP1 simulator model was built from the `mew1` physical GPU 2 profile and validated against the real vLLM 0.13 FP16 measurements.

| Prefill tokens | Real vLLM | Calibrated simulator | Absolute error |
|---:|---:|---:|---:|
| 128 | 16.673 ms | 17.953 ms | 7.68% |
| 256 | 23.106 ms | 23.970 ms | 3.74% |
| 512 | 39.465 ms | 38.000 ms | 3.71% |
| 1024 | 77.015 ms | 73.190 ms | 4.97% |
| 2048 | 143.128 ms | 143.083 ms | 0.03% |
| 3072 | 208.928 ms | 210.783 ms | 0.89% |
| 4096 | 284.457 ms | 284.137 ms | 0.11% |

Summary: 3.02% mean absolute error, 6.86% p95 absolute error, and 7.68% maximum absolute error. Six of seven anchors are within 5%; all seven are within 10%.

The raw Sarathi/FlashInfer predictor had 13.80% mean absolute error against vLLM/FlashAttention. The deployable model applies one global, documented backend factor of `0.883200462385`; the raw predictor and raw exports remain available for audit.

Validation also confirmed:

- Strictly positive and increasing prefill times at all 32 points from 128 to 4096 tokens.
- Strict `require_cache` loading without model retraining.
- Exact Python/native prefill lookup parity across 39 anchor and boundary queries.
