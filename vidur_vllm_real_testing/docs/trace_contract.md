# Trace and Canonicalization Contract

## Raw CSV

The strict raw schema is `schemas/raw_trace_row.schema.json`.

| Column | Meaning |
| --- | --- |
| `request_id` | Stable unique identifier |
| `arrived_at_s` | Real arrival time relative to benchmark start |
| `num_prefill_tokens` | Exact intended token count passed to vLLM |
| `num_decode_tokens` | Maximum generated tokens for the request |
| `prefill_slo_s` | Real time-to-first-token SLO used for raw scoring |
| `decode_slo_s` | Real per-token decode SLO used for raw scoring |
| `prompt_mode` | `synthetic_token_ids`, `text_file`, or `token_ids_file` |
| `prompt_ref` | File reference for non-synthetic prompt modes |
| `seed` | Deterministic generation seed |
| `ignore_eos` | Whether generation must continue to the requested output length |

Real traces must provide both SLOs. Synthetic imports may explicitly derive
them from the frozen Vidur contract.

## Canonical CSV

`prepare_trace.py` emits `schemas/canonical_trace_row.schema.json`. Every size
and SLO exists as an `actual_*` and a `canonical_*` value.

### Prefill Tokens

Input must be in `[1, 4096]`. It is rounded to the nearest 128-token profile
entry using half-up ties:

```text
canonical = 128 * floor((actual + 64) / 128)
canonical = clamp(canonical, 128, 4096)
```

This is deliberately not Python `round`, which uses ties-to-even. Examples:

| Actual | Canonical |
| ---: | ---: |
| 64 | 128 |
| 191 | 128 |
| 192 | 256 |
| 764 | 768 |
| 4096 | 4096 |

Values above 4096 are rejected instead of silently clipped.

### Decode Tokens

Decode totals are not bucketed. Values in `[1, 864]` are preserved because the
Markov-v2 feature is exact `decode_tokens / 864`. Values outside that trained
range are rejected.

### SLOs

Actual positive SLOs are preserved for real scoring. Model-facing SLOs are:

```text
canonical_prefill_slo_s = prefill_profile[canonical_prefill_tokens] * 3.0
canonical_decode_slo_s  = 0.05
```

Thus an actual 764-token prompt is represented to the digital twin as 768
tokens and uses `3 * prefill_profile[768]`. The real request remains 764 tokens
and retains its supplied SLO.

### Arrival and Window Validation

Arrival times are preserved; they are not rounded to the 0.2-second adversary
grid. The live system can receive requests at arbitrary times. Initial traces
must nevertheless remain inside the trained workload envelope: in every
inclusive one-second window there may be at most seven arrivals and at most
7168 canonical prefill tokens. Validation can be disabled only for an explicit
out-of-distribution experiment.

## Prompt Materialization

The ready English traces store both readable `.txt` files and complete Llama-3
token-ID arrays. Counts include BOS (`128000`) and use `add_special_tokens=true`.
The pinned tokenizer revision and every text/token checksum are recorded in the
prompt catalog and trace manifest. Before execution, the GPU image must
retokenize each text and verify exact array equality:

```text
retokenized_ids == stored_token_ids
len(stored_token_ids) == actual_prefill_tokens
```

A mismatch is a hard error before the benchmark starts. vLLM should receive the
stored token IDs directly so its request length cannot drift because of an
implicit tokenizer or BOS setting.

## Action Projection

MCTS returns canonical token allocations. The integration projects each
allocation onto real remaining work:

```text
actual_allocation = min(canonical_allocation, actual_remaining_tokens)
```

For example, the last canonical 128-token chunk of a 764-token prompt executes
the actual 124-token tail. The digital twin still consumes its canonical
128-token tail. Both transitions are logged.

## Reproducibility Manifest

Every prepared trace has a JSON manifest with source/canonical SHA256 hashes,
the frozen prefill profile hash, row count, bounds, SLO rules, launch-window
policy, and feature schema names. The future container must reject an input
whose current checksum differs from its manifest.
