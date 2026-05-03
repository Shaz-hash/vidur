# Game Version 3 Tests

This directory contains Python-side tests for GV3 MCTS/DNN correctness. These tests are designed to validate the pipeline before running expensive self-play/training experiments.

All commands below assume:

```bash
cd /home/shazer/Desktop/Research/Vidur/vidur
export PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur
```

Outputs are written to:

```text
simulator_output/Game_Version3/tests/
```

## 1. `history_node_tests.py`

Validates that history/frontier nodes are generated correctly from the initial state.

What it checks:

- Generates frontier roots through the optimized GV3 history-root generator.
- Logs the shared action trace from the initial state to frontier nodes.
- Reuses GV2-style semantic trace checks from `vidur/mcts/tests/DNN_TESTS/Game_Version2/mcts_tests.py`.
- Verifies that applying each logged action keeps simulator/request state consistent.
- Verifies frontier uniqueness using history signatures and stricter state signatures.
- Rejects forced single-action frontier roots.

Main outputs:

```text
history_node_tests_mcts_iter.csv
history_node_tests_frontiers.csv
history_node_tests_failed_trace.csv   # only if a trace check fails
```

Command:

```bash
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m \
vidur.mcts.Game_Versions.Game_Version3.tests.history_node_tests \
  --num-roots 32 \
  --history-hops-min 0 \
  --history-hops-max 32 \
  --batch-size 16
```

Useful CSVs:

- `history_node_tests_mcts_iter.csv`: shared trace rows. This may have one `root_id` because shared prefixes are logged once.
- `history_node_tests_frontiers.csv`: the actual frontier roots. `history_log_node_id` maps each frontier root to the final node in the trace CSV.

## 2. `test_checking_depth1.py`

Validates the depth-1 Bellman action selection used by `mctsDNN.py`.

What it checks:

- Generates frontier roots through the same self-play history-root path.
- Runs the actual `SelfPlayRunner.run_single_root(...)` / `VidurMCTS.search_dnn(...)` path.
- Logs all possible actions from each root.
- Recomputes expected best action independently from logged Q values.
- Checks that controller roots select max Q.
- Checks that adversary roots select min Q.

Main outputs:

```text
test_checking_depth1_search.csv
test_checking_depth1_details.csv
```

Command:

```bash
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m \
vidur.mcts.Game_Versions.Game_Version3.tests.test_checking_depth1 \
  --num-roots 32 \
  --history-hops-min 0 \
  --history-hops-max 32
```

Useful CSVs:

- `test_checking_depth1_search.csv`: one row per root, selected action, expected action, pass/fail.
- `test_checking_depth1_details.csv`: one row per possible action. This is the best file for manually inspecting Q, reward, discount, bootstrap, validity, and action representation.

Important detail:

- For adversary roots, the logged action is the outer adversary action. Its Q may include an inner controller best response because GV3 adversary Q is evaluated as adversary action followed by controller response.

## 3. `feature_conversion_tests.py`

Validates that simulator/frontier states are converted into DNN model tensors correctly.

What it checks:

- Generates frontier roots through the same history-root path.
- Calls production `build_model_inputs(...)`.
- Independently recomputes expected prefill, decode, global, legacy feature tensors from simulator/request state.
- Verifies tensor values with max absolute tolerance.
- Verifies prefill/decode masks.
- Verifies action masks.
- Verifies legacy `req_features` and `req_mask` compatibility tensors.

Main outputs:

```text
feature_conversion_summary.csv
feature_conversion_prefill.csv
feature_conversion_decode.csv
feature_conversion_global.csv
```

Command:

```bash
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m \
vidur.mcts.Game_Versions.Game_Version3.tests.feature_conversion_tests \
  --num-roots 32 \
  --history-hops-min 0 \
  --history-hops-max 32 \
  --batch-size 16
```

Useful CSVs:

- `feature_conversion_summary.csv`: one row per root with max feature diffs and mask pass/fail.
- `feature_conversion_prefill.csv`: per-prefill-request feature vectors.
- `feature_conversion_decode.csv`: per-decode-request feature vectors.
- `feature_conversion_global.csv`: per-global-feature scalar values.

Expected pass condition:

```text
prefill_max_abs_diff = 0
decode_max_abs_diff = 0
global_max_abs_diff = 0
legacy_max_abs_diff = 0
passed = true
```

## 4. `consolidated_frontier_tests.py`

Runs the main correctness tests together on the same generated frontier roots.

What it does:

1. Generates `x` frontier nodes once.
2. Validates history/environment traces through `mcts_tests.py`.
3. Checks frontier uniqueness and rejects forced frontiers.
4. Runs depth-1 Bellman selection checks on those same roots.
5. Runs feature conversion checks on those same roots.

Main outputs:

```text
consolidated_history_mcts_iter.csv
consolidated_frontiers.csv
consolidated_depth1_search.csv
consolidated_depth1_details.csv
consolidated_feature_summary.csv
consolidated_feature_prefill.csv
consolidated_feature_decode.csv
consolidated_feature_global.csv
```

Command:

```bash
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m \
vidur.mcts.Game_Versions.Game_Version3.tests.consolidated_frontier_tests \
  --num-roots 32 \
  --history-hops-min 0 \
  --history-hops-max 32 \
  --batch-size 16
```

Use this before launching a new experiment. It is the most useful end-to-end Python-side sanity check.

Expected output:

```text
consolidated GV3 tests passed: frontiers=32, trace_chains=..., depth1_roots=32, feature_roots=32
```

## Direct CSV Trace Validation

If you manually edit a history trace CSV and want to validate that exact file without regenerating it, use this command:

```bash
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 - <<'PY'
from pathlib import Path

from vidur.mcts.Game_Versions.Game_Version3.tests.history_node_tests import _patch_gv2_validator_for_gv3
from vidur.mcts.tests.DNN_TESTS.Game_Version2 import mcts_tests as tests

csv_path = Path("simulator_output/Game_Version3/tests/history_node_tests_mcts_iter.csv")

_patch_gv2_validator_for_gv3()
rows, fieldnames, raw_by_rownum = tests.load_rows(str(csv_path))
traces = tests.build_leaf_traces(rows)

for trace in traces:
    tests.run_trace(
        trace,
        prefill_profile={},
        assume_extracted_trace=False,
        unsupported_hits=set(),
    )

print(f"PASSED: file={csv_path}, rows={len(rows)}, traces={len(traces)}")
PY
```

Do not run `history_node_tests.py` if your goal is to test a manually edited CSV, because that test regenerates and overwrites the CSV first.

## Recommended Pre-Experiment Checklist

Run these in order:

```bash
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m py_compile \
  vidur/mcts/environment.py \
  vidur/mcts/Game_Versions/Game_Version3/virtual_environment.py \
  vidur/mcts/Game_Versions/Game_Version3/mctsDNN.py \
  vidur/mcts/Game_Versions/Game_Version3/tests/history_node_tests.py \
  vidur/mcts/Game_Versions/Game_Version3/tests/test_checking_depth1.py \
  vidur/mcts/Game_Versions/Game_Version3/tests/feature_conversion_tests.py \
  vidur/mcts/Game_Versions/Game_Version3/tests/consolidated_frontier_tests.py
```

Then:

```bash
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m \
vidur.mcts.Game_Versions.Game_Version3.tests.consolidated_frontier_tests \
  --num-roots 32 \
  --history-hops-min 0 \
  --history-hops-max 32 \
  --batch-size 16
```
