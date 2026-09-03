# English Prompt Artifacts

The processed Splitwise files contain arrival times and token counts only. They
do not contain user prompt text or token IDs. This directory therefore creates
deterministic English prompts that exactly reproduce each declared prefill
length with the tokenizer used by `Meta-Llama-3-8B`.

## Tokenizer Contract

```text
repository: NousResearch/Meta-Llama-3-8B
revision: 315b20096dc791d381d514deb5f8bd9c8d6d3061
tokenizer.json SHA256: e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7
add_special_tokens: true
BOS token ID: 128000
```

The declared prefill length includes BOS. A 374-token request therefore stores
one BOS token plus 373 tokens obtained from its English text. The token-ID JSON
contains the complete array sent to vLLM, while a sidecar `.txt` contains the
human-readable source.

Sending token IDs directly avoids a hidden tokenizer-version or BOS mismatch.
Before execution, the Docker image must still retokenize every `.txt` file and
assert exact equality with the stored array.

## Ready Traces

### Splitwise Conversation, First 20 Seconds

This is the primary natural-length trace. It preserves the first 31 rows from
`data/processed_traces/splitwise_conv.csv` through 19.945197 seconds.

```text
traces/splitwise_conv_20s_english_raw.csv
traces/splitwise_conv_20s_english_canonical.csv
traces/splitwise_conv_20s_english_manifest.json
prompts/splitwise_conv_20s/prompt_catalog.json
audits/splitwise_conv_20s_static_markov_features.csv
```

Actual prompt lengths range from 91 to 4085 and decode limits from 12 to 194.
The source CSV provides neither prompt content nor SLOs. These prompts are generated
locally, prefill SLOs use the frozen canonical profile times three, and decode SLOs
are 0.05 seconds; the manifest records this experimental contract.
Canonical prefill sizes are rounded half-up to the 128-token Vidur profile grid.
Actual sizes drive vLLM; canonical sizes drive the digital twin and model.

### Arena-Style Stress Trace, 20 Seconds

This retains the 79-request GV3 stress workload used by prior synthetic tests:

```text
traces/gv3_legal_20s_english_raw.csv
traces/gv3_legal_20s_english_canonical.csv
traces/gv3_legal_20s_english_manifest.json
prompts/gv3_legal_20s/prompt_catalog.json
audits/gv3_legal_20s_static_markov_features.csv
```

### Exact-Grid Mew1 A100 Debug Workload

This workload reproduces the legal 20-second synthetic trace with real English
prompts whose tokenized prefill lengths already lie on the controller's
128-token grid. No request is rounded during canonicalization, so differences
between real-vLLM and simulator results cannot be attributed to prefill-length
rounding.

```text
traces/gv3_legal_20s_exact_grid_mew1_a100_template_raw.csv
traces/gv3_legal_20s_exact_grid_mew1_a100_raw.csv
traces/gv3_legal_20s_exact_grid_mew1_a100_canonical.csv
traces/gv3_legal_20s_exact_grid_mew1_a100_manifest.json
prompts/gv3_legal_20s_exact_grid_mew1_a100/prompt_catalog.json
```

The 79 requests use prefill sizes `128`, `256`, `512`, `1024`, `1536`,
`2048`, `3072`, and `4096`; every decode length is `864`. Actual and canonical
prefill sizes are identical for every row. Prefill SLOs are three times the
calibrated Mew1 A100 TP1 profile and the manifest pins that profile by hash.

On `mew1`, select this workload for the next real-GPU benchmark with:

```bash
export VIDUR_REAL_TRACE=/home/shaz/vidur/source/vidur-classical-search/vidur_vllm_real_testing/traces/gv3_legal_20s_exact_grid_mew1_a100_canonical.csv
```

Restart the benchmark process after setting the variable; an already-running
server retains its previous trace and imported adapter code.

## Feature Boundary

The canonical trace fully determines static request inputs: total prefill and
decode lengths, request SLOs, arrival time, and prompt references. The feature
audit passes these values through the real `markov_v2` builder.

The trace cannot determine live features in advance. Processed tokens, queue
age, active backlog, lateness, credits, deadlines after decode progress, recent
launch state, and the set of live requests depend on actual vLLM execution.
The scheduler adapter must build those fields from the live engine state at
every decision and then invoke the same Python/native feature parity checks.

Policy inputs also depend on the live state and current canonical action set;
they cannot be safely precomputed from arrivals alone.
