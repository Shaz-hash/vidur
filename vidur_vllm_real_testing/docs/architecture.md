# Runtime Architecture

## Data Flow

```text
fixed request trace
  -> real arrival dispatcher
  -> vLLM live requests (actual lengths and SLOs)
  -> canonical state adapter
  -> frozen Vidur GV3 digital twin
  -> native DNN + MCTS
  -> canonical ControllerAction
  -> projection to actual remaining tokens
  -> stock vLLM KV validation and batch execution
```

The trace is visible to the load generator and evaluator, not to MCTS. MCTS may
use the trained adversary for future workload, exactly as in self-play, but it
must not inspect future trace rows.

## Ownership Boundaries

The real vLLM engine owns request admission, KV allocation, preemption safety,
GPU execution, generated token correctness, and actual timestamps. The GV3
controller chooses a desired batch only. The integration must pass that plan
through vLLM's normal safety checks rather than replacing them.

The Vidur shadow owns canonical request state, game credits, deadlines, learned
adversary continuations, model inference, and MCTS. Its prefill/decode profiles
remain frozen. Calibration is observational and never modifies these profiles.

## Scheduler Integration

The target implementation is a narrow scheduling-plan hook in a pinned vLLM
version. Replacing the entire scheduler is avoided because vLLM's scheduler
contains critical KV and preemption logic.

Implemented bring-up modes:

1. `stock`: stock vLLM FCFS reference with no controller.
2. `shadow`: GV3 computes decisions but vLLM ignores them; logs prove state and
   action validity without affecting service.
3. `active-validation`: valid GV3 plans drive execution; invalid/stale plans
   fall back to stock and are logged.
4. `controller`: valid GV3 plans drive execution and any failure is fatal.
5. `sjf-256` and `sjf-512`: exact fixed-policy comparison baselines.

The pinned vLLM source receives two guarded token-cap reads, one in its running
loop and one in its waiting loop. The patch installer rejects every vLLM version
other than `0.13.0` and rejects source whose expected lines changed. The custom
scheduler temporarily exposes only selected requests and their validated token
caps for one stock scheduling call. KV allocation, block ownership, preemption,
connector metadata, and output construction remain in upstream vLLM.

## Decision-Time Accounting

Report two clocks:

```text
raw_wall_clock = observed real system time
adjusted_clock = raw_wall_clock - scheduler_blocking_time
```

Only time that actually blocks GPU dispatch is subtracted. Background planning
that overlaps GPU work is not subtracted. Both raw and adjusted SLO costs must
be retained; adjusted numbers cannot replace the real result.

## Initial vLLM Shape

The first image/server configuration should pin:

```text
model: Meta-Llama-3-8B
dtype: bfloat16
tensor parallel: 1
pipeline parallel: 1
max model length: 8192
max sequences: 512
max batched tokens: 4608
chunked prefill: enabled
prefix caching: disabled
KV block size: 16
swap: disabled
```

`4608` permits the largest GV3 prefill allocation (`4096`) plus one decode token
for up to 512 active decode requests. The AlphaGoZero experiment assumed
infinite KV. Initial real tests therefore use traces that do not preempt or OOM;
finite-KV behavior requires later features and retraining, not a hidden runtime
workaround.

## Logging Contract

Each scheduling decision logs the state fingerprint, canonical/projected plan,
planning/blocking time, fallback reason, and actual scheduled token map. Real
GPU calibration extends those rows with:

```text
decision_id, live_state_fingerprint, plan_state_fingerprint,
actual_time_before_s, actual_time_after_s, controller_blocking_s,
canonical_action, projected_actual_action, fallback_reason,
actual_batch_shape, vidur_predicted_batch_time_s, real_batch_time_s,
raw_cost_delta, adjusted_cost_delta
```

Per-request output retains actual/canonical lengths and SLOs from the prepared
trace. This is necessary to distinguish model-domain mismatch from real serving
performance.
