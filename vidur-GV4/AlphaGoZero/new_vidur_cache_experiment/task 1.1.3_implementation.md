# Task 1.1.3 Implementation And Result

## Scope

Task 1.1.3 validates the uncalibrated Vidur execution-time predictor against exact
batches selected by the promoted AlphaGoZero controller and executed by real vLLM
on `shaz@mew1`. The test uses the Task 1.1.2 persistent GV3 scheduler, DNN/native
MCTS, and transformer-block CUDA timing boundary.

The frozen promoted bundle used for this run is controller `v118` and adversary
`v117`. The in-distribution trace has a two-second arrival window, 11 requests,
and exact canonical prefill sizes. The search configuration is 2,000 MCTS
iterations, one 3-second leaf-relative rollout, discount `0.98`, PUCT `0.5`, one
thread, and no evaluation root noise.

## Batch Audit

Before each real batch executes, the scheduler records the exact live state and
scheduled token map. After execution it records the reconciled state and measured
transformer-block CUDA duration. Each audit row includes:

- Total scheduled prefill tokens and prefill/decode request counts.
- Per-request physical prefill total, completed prefill, remaining prefill,
  context, and scheduled prefill tokens.
- Per-request decode context, completed/remaining decode tokens, and scheduled
  decode tokens.
- Simulator state before and after execution.
- Authoritative real-GPU duration and its timing source.

The comparison reconstructs the same `ExecutionTimePredictorRequest` objects and
queries only:

`simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING/vidur_predictor_cache/`

No calibration or fitted correction is applied. The comparison fails closed if the
logical-time increment differs from the measured real-GPU duration.

Warm-up scheduler and GPU timing logs are both cleared at the warm-up boundary.
The timing reader also skips only vLLM's explicitly named `_warmup_*` records and
still rejects every other batch-identity mismatch.

## Real Run

The successful run used physical mew1 GPU 3 after an uncontended GPU preflight.
It produced 242 one-to-one batch audit and GPU timing records. The final GV3 state
had no active requests; all 11 request IDs were completed/stopped. Because the
trace defines arrivals over two seconds, draining those requests advanced logical
simulator time to `4.2559232113` seconds. Every logical-time increment exactly
matched its measured batch duration.

Artifacts are stored at:

```text
simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING/task_1_1_3_2s_run/
simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING/comparison_test.csv
```

The vLLM engine became idle after GV3 stopped/completed all requests, but the HTTP
benchmark client did not receive terminal stream events for GV3-stopped requests.
The idle isolated server was therefore shut down after the complete batch audit was
persisted. This stream-finalization issue does not alter the recorded batch shapes,
CUDA durations, or logical-time accounting.

## Accuracy Result

The requested 10% criterion did not pass overall:

| Batch category | Batches | Mean real time | Mean Vidur time | Mean absolute error | Within 10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 242 | 17.586 ms | 20.330 ms | 17.581% | 6 |
| Prefill only | 1 | 210.188 ms | 196.065 ms | 6.719% | 1 |
| Mixed prefill/decode | 23 | 39.089 ms | 43.590 ms | 11.593% | 5 |
| Decode only | 218 | 14.434 ms | 17.069 ms | 18.263% | 0 |

Vidur overpredicted 241 of 242 batches. Median absolute error was `18.924%`, p95
was `19.628%`, and maximum error was `20.244%`. The maximum occurred for an
11-request decode-only batch: real `14.286 ms`, Vidur `17.178 ms`.

The new prefill predictor is within the requested margin for the initial
three-request `3 x 1024` prefill batch (`6.719%` error). The overall failure is
primarily the decode-only mismatch, not that initial prefill batch.

## Reproduction

The required test and comparison entry point is:

`vidur_vllm_real_testing/tests/test_batch_execution_error.py`

```bash
PYTHONPATH=. .venv/bin/python \
  -m vidur_vllm_real_testing.tests.test_batch_execution_error \
  --batch-audit simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING/task_1_1_3_2s_run/scheduler.jsonl.batches.jsonl \
  --output simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING/comparison_test.csv
```

The local vLLM integration suite passes 89 tests.
