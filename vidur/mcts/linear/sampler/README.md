# Linear Sampler (Module 1)

This package implements a depth-aware, deduplicated state sampler for LP-oriented linear experiments.

## What it writes

Per round, worker shards are written under:

- `round_XXX/shards/worker_YY/states.parquet`
- `round_XXX/shards/worker_YY/requests.parquet`
- `round_XXX/shards/worker_YY/actions.parquet`
- `round_XXX/shards/worker_YY/transitions.parquet`
- `round_XXX/shards/worker_YY/nodes.parquet`
- `round_XXX/shards/worker_YY/transition_request_deltas.parquet`
- `round_XXX/shards/worker_YY/anchors.parquet`

Merged, deduplicated tables are written under:

- `round_XXX/merged/*.parquet`
- `round_XXX/merged/manifest.json`

Compatibility CSV exports (optional) are written under:

- `round_XXX/compat/mcts_root_compat.csv`
- `round_XXX/compat/mcts_iter_compat.csv`

## CLI

Run:

```bash
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
-m vidur.mcts.linear.sampler.run \
  --target-unique-states 400000 \
  --workers 8 \
  --max-trace-length 10 \
  --max-branching 10 \
  --max-forced-hops 20000 \
  --history-hops 0,10,20,30,40,50,60,70 \
  --out-dir simulator_output/linear_sampler \
  --export-compat-root \
  --export-compat-iter
```

## Notes

- Controller canonicalization follows token-allocation signature parity with current MCTS logic.
- Sampling draws `L ~ Uniform[1, max_trace_length]` per attempt and traverses at most `L` branching decisions ahead of each history root.
- Unique target counting is based on unique accepted anchor `state_id`s (branching anchors only).
- `manifest.json` now records LP sample counts split by transition actor:
  - `controller_lp_samples`
  - `adversary_lp_samples`
- Snapshot restore uses an in-process LRU cache.
- Compat CSV export is disabled by default and enabled only with explicit flags.
