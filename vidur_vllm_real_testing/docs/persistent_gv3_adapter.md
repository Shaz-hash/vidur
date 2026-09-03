# Persistent GV3 Adapter

## Production entrypoints

Use these classes with vLLM 0.13:

```bash
export VIDUR_VLLM_SCHEDULER_MODE=controller
export VIDUR_VLLM_GV3_PLANNER=vidur_vllm_real_testing.gv3_adapter:GV3PersistentAdapter
export VIDUR_VLLM_GV3_STATE_PLANNER=your_module:your_native_mcts_planner
export VIDUR_VLLM_CANONICAL_TRACE=/absolute/path/to/canonical_trace.csv
```

Pass the scheduler class as:

```text
vidur_vllm_real_testing.gv3_persistent_scheduler.GV3PersistentScheduler
```

The adapter fails closed when request progress regresses, a request disappears
without a completed batch, an arrival is submitted ahead of canonical time, a
decode exceeds the credit balance, or initialization begins in the middle of a
request.

## State ownership

`GV3PersistentAdapter` owns one state ledger for the complete benchmark. A vLLM
queue snapshot reconciles and validates that ledger; it never replaces it.

The ledger retains:

- Stable integer GV3 request IDs for real string request IDs.
- Canonical prefill and decode progress.
- Prefill completion times and fixed prefill deadlines.
- Per-token decode deadlines and accumulated lateness.
- Decode-credit mint, spend, drop reclaim, and credit-overflow stop state.
- Completed, dropped, stopped, and violated request sets.
- Recent one-second launch history used to derive adversary budget usage.
- Last and next 0.2-second adversary ticks.

After a real batch completes:

```text
canonical_time = previous_canonical_time + measured_real_batch_duration
```

The scheduler reports progress only after vLLM has processed the GPU output.
Controller planning time is therefore not charged as simulated service time.
Native MCTS receives a copy of the resulting state and uses the frozen Vidur
profile only for hypothetical successor states.

## Decode fast-forward

When no prefill is pending but decode requests remain, the adapter schedules one
decode token for every eligible active decode request. vLLM executes that real
batch, its observed duration advances canonical time, and the next scheduler
call repeats the operation if the next adversary tick has not been reached.
This matches the GV3 loop: it is not a direct clock jump while real decode work
exists.

When no request is active, canonical time advances to the next submitted trace
arrival. The trace driver must submit arrivals on adjusted experiment time.

## GV3 terminal behavior

The production module mirrors native `DecodeCreditLedger` behavior:

- Prefill completion mints 216 decode credits once.
- Every generated output token consumes one credit.
- Automatic drop at two seconds of request lateness subtracts that request's
  unspent minted credit, removes live lateness and violation accounting, and
  adds the fixed terminal drop cost of 16.
- Credit overflow stops the most decoded requests first and erases their credit
  entries without changing the remaining global credit balance.

The trained GV3 contract requires a maximum of 864 decode tokens per request.
The production adapter rejects any other configured cap rather than silently
creating a state outside the training distribution.

## Verification

Focused and package tests:

```bash
python -m unittest discover -s vidur_vllm_real_testing/tests -p 'test_*.py'
```

The current remote installation passes 47 tests. The production payload also
passes both Python and native Markov-v2 feature construction with zero maximum
absolute error for the checked state.

The real-vLLM smoke uses physical GPU 2 only, an exact Llama-3-8B architecture,
vLLM dummy weights, and the bundled Llama-3 tokenizer. Dummy values do not alter
the model's kernel shapes, but the test is not a learned-weight correctness test.

## Prefill profile measurement

Run on `mew1` with:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python vidur_vllm_real_testing/mew1/run_prefill_profile_v2.py \
  --warmups 3 --repetitions 5
```

The duration starts after scheduling/controller work and ends when the model
runner output reaches the scheduler. The warmed median comparison was:

| Prefill tokens | Real vLLM median (s) | Frozen profile (s) | Real/profile |
|---:|---:|---:|---:|
| 128 | 0.016673 | 0.015726 | 1.060 |
| 256 | 0.023106 | 0.023275 | 0.993 |
| 512 | 0.039465 | 0.038886 | 1.015 |
| 1024 | 0.077015 | 0.098502 | 0.782 |
| 2048 | 0.143128 | 0.196138 | 0.730 |
| 3072 | 0.208928 | 0.284087 | 0.735 |
| 4096 | 0.284457 | 0.386693 | 0.736 |

The profile is close for 128-512 tokens. On this A100/vLLM configuration,
1024-4096-token prefills are approximately 22-27 percent faster than the frozen
profile, so the large-prefill profile does not match real execution closely.

Remote artifacts are written below:

```text
/home/shaz/vidur/runs/prefill_profile_gpu2/
```
