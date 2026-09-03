#!/usr/bin/env bash
# Run v4-Adv arena games sequentially for the latest completed XL checkpoints.
set -euo pipefail

cd /home/ubuntu/vidur-classical-search
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SJF_BUDGET="${1:-512}"
NUM_GAMES="${2:-50}"
ARENA_PROCESSES="${3:-25}"
TIME_LIMIT_SEC="${4:-5.0}"
RUN_SUFFIX="${5:-}"

BASE="$PWD/simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v4_adv_226/V_iter_xl_50"
OUT_BASE="$PWD/simulator_output/GV3_Agent/Model_Tester_Results/adv226_xl_sjf${SJF_BUDGET}_${NUM_GAMES}games_${TIME_LIMIT_SEC}s${RUN_SUFFIX}"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"

run_one() {
  local model_name="$1"
  local version="$2"
  local model_dir="$BASE/$model_name/Model_Version$version"
  local wrapped_model="$model_dir/v4_adv_hgb_wrapper.joblib"
  local out_dir="$OUT_BASE/${model_name}_v${version}"
  local log_path="$LOG_DIR/${model_name}_v${version}.log"

  if [[ ! -f "$wrapped_model" ]]; then
    echo "missing wrapped model: $wrapped_model" >&2
    return 1
  fi

  mkdir -p "$out_dir"
  echo "[arena] start ${model_name} V${version} $(date)" | tee "$log_path"
  /usr/bin/time -f 'elapsed=%E maxrss_kb=%M' \
    /home/ubuntu/vidur-classical-search/.venv/bin/python -u -m \
      vidur.Game_Version3.Model_Tester.run \
      --model-kind classical_joblib \
      --model-path "$wrapped_model" \
      --output-dir "$out_dir" \
      --num-games "$NUM_GAMES" \
      --environment-lang python \
      --history-hops-min 0 \
      --history-hops-max 100 \
      --history-seed 2026 \
      --arena-time-limit-sec "$TIME_LIMIT_SEC" \
      --arena-num-processes "$ARENA_PROCESSES" \
      --arena-worker-threads 1 \
      --arena-mp-start-method spawn \
      --trivial-budget-tokens "$SJF_BUDGET" \
      --bootstrap-model-version "$version" \
      --write-model-action-detail-logs \
      >> "$log_path" 2>&1
  echo "[arena] done ${model_name} V${version} $(date)" | tee -a "$log_path"
}

echo "[arena] output base: $OUT_BASE"
run_one "hgb_sq_31leaf_1280iter" "40"
run_one "hgb_sq_47leaf_850iter" "47"
run_one "hgb_sq_63leaf_630iter" "50"
echo "[arena] all done $(date)"
