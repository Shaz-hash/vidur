#!/usr/bin/env bash
# Run a 50-game parallel arena tournament against trivial SJF-128 controller,
# using the converged cliff-aware model version as the model under test.
set -euo pipefail
cd /home/ubuntu/vidur-classical-search
export PYTHONPATH=$PWD

MODEL_VERSION="${1:-50}"
NUM_GAMES="${2:-50}"
ARENA_PROCESSES="${3:-25}"
NAME_SUFFIX="${4:-cliff_aware_v${MODEL_VERSION}_${NUM_GAMES}games_5sec}"
RUN_DIR="${5:-cliff_aware_full_50v}"

MODEL_PATH=$PWD/simulator_output/GV3_Agent/BellmanConvergence/${RUN_DIR}/Model_Version${MODEL_VERSION}/cliff_aware_controller_value.joblib
OUT=$PWD/simulator_output/GV3_Agent/Model_Tester_Results/${NAME_SUFFIX}
LOG=$OUT.log

mkdir -p "$OUT"
[ -f "$MODEL_PATH" ] || { echo "model not found: $MODEL_PATH" >&2; exit 1; }
echo "starting arena $(date)" > "$LOG"

# Each worker reuses its own simulator, so worker_threads=1 keeps fits/sims
# from oversubscribing the box.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

(
  /usr/bin/time -f 'elapsed=%E maxrss_kb=%M' \
  /home/ubuntu/vidur/.venv/bin/python3 -u -m \
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
    --bootstrap-model-version 1
) >> "$LOG" 2>&1
echo "completed $(date)" >> "$LOG"
