#!/usr/bin/env bash
# Run arena: V_1 hierarchical (cls+reg, 200k) vs SJF-128/256/512.
set -euo pipefail
cd /home/ubuntu/vidur-classical-search
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

SJF_BUDGET="${1:-512}"
NUM_GAMES="${2:-50}"
ARENA_PROCESSES="${3:-25}"
NAME_SUFFIX="V1_hier_200k_vs_sjf${SJF_BUDGET}_${NUM_GAMES}games"
MODEL_PATH=$PWD/simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4/V1_hier_200k/v4_hier_wrapper.joblib
OUT=$PWD/simulator_output/GV3_Agent/Model_Tester_Results/${NAME_SUFFIX}
LOG=$OUT.log

mkdir -p "$OUT"
[ -f "$MODEL_PATH" ] || { echo "model not found: $MODEL_PATH" >&2; exit 1; }
echo "starting V1_hier vs sjf$SJF_BUDGET $(date)" > "$LOG"

(
  /usr/bin/time -f 'elapsed=%E maxrss_kb=%M' \
  /home/ubuntu/vidur-classical-search/.venv/bin/python -u -m \
    vidur.mcts.Game_Versions.Game_Version3.Model_Tester.run \
    --model-kind classical_joblib \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUT" \
    --num-games "$NUM_GAMES" \
    --environment-lang python \
    --history-hops-min 0 \
    --history-hops-max 100 \
    --history-seed 2026 \
    --arena-time-limit-sec 5.0 \
    --arena-num-processes "$ARENA_PROCESSES" \
    --arena-worker-threads 1 \
    --arena-mp-start-method spawn \
    --trivial-budget-tokens "$SJF_BUDGET" \
    --bootstrap-model-version 1
) >> "$LOG" 2>&1
echo "completed $(date)" >> "$LOG"
