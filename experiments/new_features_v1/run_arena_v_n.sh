#!/usr/bin/env bash
# Run arena for an arbitrary V_n joblib (HGBRegressor) wrapped as V4HGBWrapper.
# Args:
#   $1 — iter index N (e.g. 5, 10, 15)
#   $2 — SJF budget tokens (e.g. 128, 256, 512)
#   $3 — num games (default 50)
#   $4 — arena processes (default 25)
#   $5 — run dir name (default V_hier_iter)
set -euo pipefail
cd /home/ubuntu/vidur-classical-search
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

ITER_N="${1:?iter required}"
SJF_BUDGET="${2:-512}"
NUM_GAMES="${3:-50}"
ARENA_PROCESSES="${4:-25}"
RUN_DIR_NAME="${5:-V_hier_iter}"

NAME_SUFFIX="V${ITER_N}_${RUN_DIR_NAME}_vs_sjf${SJF_BUDGET}_${NUM_GAMES}games"
RAW_MODEL=$PWD/simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4/${RUN_DIR_NAME}/Model_Version${ITER_N}/model.joblib
WRAPPED_MODEL=$PWD/simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4/${RUN_DIR_NAME}/Model_Version${ITER_N}/v4_hgb_wrapper.joblib
OUT=$PWD/simulator_output/GV3_Agent/Model_Tester_Results/${NAME_SUFFIX}
LOG=$OUT.log

mkdir -p "$OUT"
[ -f "$RAW_MODEL" ] || { echo "model not found: $RAW_MODEL" >&2; exit 1; }

# Build wrapper if missing.
if [ ! -f "$WRAPPED_MODEL" ]; then
  echo "wrapping $RAW_MODEL -> $WRAPPED_MODEL"
  /home/ubuntu/vidur-classical-search/.venv/bin/python -c "
import sys
sys.path.insert(0, '/home/ubuntu/vidur-classical-search')
from vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.v4_hgb_wrapper import pack_v4_hgb
pack_v4_hgb('$RAW_MODEL', '$WRAPPED_MODEL', model_tag='V${ITER_N}_${RUN_DIR_NAME}')
print('wrote', '$WRAPPED_MODEL')
"
fi

echo "starting V${ITER_N} vs sjf$SJF_BUDGET $(date)" > "$LOG"
(
  /usr/bin/time -f 'elapsed=%E maxrss_kb=%M' \
  /home/ubuntu/vidur-classical-search/.venv/bin/python -u -m \
    vidur.mcts.Game_Versions.Game_Version3.Model_Tester.run \
    --model-kind classical_joblib \
    --model-path "$WRAPPED_MODEL" \
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
    --bootstrap-model-version "$ITER_N"
) >> "$LOG" 2>&1
echo "completed $(date)" >> "$LOG"
