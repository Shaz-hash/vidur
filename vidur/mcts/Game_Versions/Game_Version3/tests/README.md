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

## 3. `frontier_feature_trace_tests.py`

Validates that frontier states are converted into DNN model tensors correctly, using CSV trace/frontier rows as the source of truth.

What it checks:

- Generates frontier roots through the same history-root path.
- Writes an expanded per-root `final_root_iter.csv`: each root keeps its history trace and appends an `internal:frontier_state` row.
- Writes `frontiers.csv`: one serialized frontier state per root, including active, completed, dropped, stopped, lateness, deadlines, decode counts, and request snapshots.
- Writes `frontier_feature.csv`: the production features produced by `build_model_inputs(...)`.
- Rebuilds expected prefill, decode, global, and legacy feature tensors only from `frontiers.csv`.
- Verifies tensor values with max absolute tolerance.
- Verifies prefill/decode masks.
- Verifies legacy `req_features` and `req_mask` compatibility tensors.
- Verifies feature request ids are a subset of active request ids.
- Verifies feature request ids are disjoint from completed, dropped, and stopped request ids.

Main outputs:

```text
shared_history_iter.csv
final_root_iter.csv
frontiers.csv
frontier_feature.csv
frontier_feature_compare.csv
```

Command:

```bash
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m \
vidur.mcts.Game_Versions.Game_Version3.tests.frontier_feature_trace_tests \
  --num-roots 128 \
  --history-hops-min 0 \
  --history-hops-max 8 \
  --history-seed 2027 \
  --batch-size 16
```

Useful CSVs:

- `final_root_iter.csv`: easy-to-read trace for each emitted root, with the selected frontier state appended.
- `frontiers.csv`: serialized state used to rebuild expected features independently from production code.
- `frontier_feature.csv`: production feature tensors logged at the same roots.
- `frontier_feature_compare.csv`: one row per root with max diffs, mask checks, and active/terminal invariants.

Expected pass condition:

```text
passed = true
max_feature_diff <= tolerance
feature_ids_subset_active = true
feature_ids_disjoint_terminal = true
prefill_mask_match = true
decode_mask_match = true
req_mask_match = true
```

Why this replaced the older feature test:

```text
feature_conversion_tests.py recomputes expected features directly from live Python objects.
That is useful as a low-level unit check, but it can miss stale-state bugs when the live object lookup itself is wrong.
frontier_feature_trace_tests.py rebuilds expected features from serialized trace/frontier rows, so it catches ghost requests and terminal requests leaking into features.
```

## 4. `feature_conversion_tests.py`

Legacy low-level feature conversion test.

What it checks:

- Generates frontier roots.
- Calls production `build_model_inputs(...)`.
- Recomputes expected features directly from live simulator/request objects.
- Compares prefill, decode, global, and legacy tensors.

Use this only when debugging a narrow feature formula mismatch. For pre-experiment validation, prefer `frontier_feature_trace_tests.py` or `consolidated_frontier_tests.py`.

## 5. `consolidated_frontier_tests.py`

Runs the main correctness tests together on the same generated frontier roots.

What it does:

1. Generates `x` frontier nodes once.
2. Validates history/environment traces through `mcts_tests.py`.
3. Checks frontier uniqueness and rejects forced frontiers.
4. Runs depth-1 Bellman selection checks on those same roots.
5. Runs the trace-derived frontier feature invariant on those same roots.

Main outputs:

```text
consolidated_history_mcts_iter.csv
consolidated_frontiers.csv
consolidated_depth1_search.csv
consolidated_depth1_details.csv
consolidated_feature_final_root_iter.csv
consolidated_feature_frontiers.csv
consolidated_frontier_feature.csv
consolidated_frontier_feature_compare.csv
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
consolidated GV3 tests passed: frontiers=32, trace_chains=..., depth1_roots=32, feature_roots=32, feature_max_diff=...
```

The consolidated feature phase uses the same logic as `frontier_feature_trace_tests.py`: production features are logged separately, expected features are rebuilt from serialized frontier rows, and active/terminal request invariants are enforced.

## 6. `native_logger_tests.py`

Runs the native sample-generation correctness gate.

What it checks:

- Runs native root generation and writes native `history_mcts_iter.csv`, `frontiers.csv`, `depth1_search.csv`, and `depth1_details.csv`.
- Validates the native history trace with the same GV3/GV2 trace validator.
- Verifies native depth-1 Bellman selected actions match the expected best action from native Q rows.
- Writes `native_frontier_feature.csv` from the native sample tensors that would be sent to training.
- Rebuilds expected features from the enriched native `frontiers.csv` log and compares them in `native_frontier_feature_compare.csv`.
- Runs bootstrap Bellman parity against Python on the same Python roots. Best action must match; Q values are allowed small TorchScript/Python numeric drift.

Main outputs:

```text
history_mcts_iter.csv
frontiers.csv
depth1_search.csv
depth1_details.csv
native_frontier_feature.csv
native_frontier_feature_compare.csv
native_bellman_bootstrap_parity.csv
```

Command:

```bash
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 -m \
vidur.mcts.Game_Versions.Game_Version3.tests.native_logger_tests \
  --num-roots 128 \
  --history-hops-min 0 \
  --history-hops-max 8 \
  --history-seed 2027 \
  --checkpoint-path simulator_output/Game_Version3_Fresh8_Hops200/mcts_dnn_checkpoints/best.pt
```

Expected output:

```text
native GV3 tests passed: native_roots=128, trace_chains=..., feature_roots=128, max_feature_diff=..., bellman_roots=128, max_controller_q_diff=...
```

Important detail:

- The default bootstrap Q tolerance is `1e-2`. The native and Python best actions are still required to match, but exact Q equality is too strict because the comparison crosses Python eager model execution and TorchScript/native execution.

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
  vidur/mcts/Game_Versions/Game_Version3/tests/frontier_feature_trace_logger.py \
  vidur/mcts/Game_Versions/Game_Version3/tests/frontier_feature_trace_tests.py \
  vidur/mcts/Game_Versions/Game_Version3/tests/consolidated_frontier_tests.py \
  vidur/mcts/Game_Versions/Game_Version3/tests/native_logger_tests.py
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
