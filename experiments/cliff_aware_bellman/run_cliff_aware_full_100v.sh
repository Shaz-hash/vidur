#!/usr/bin/env bash
# Run cliff-aware Bellman convergence for 100 versions on the full dataset.
# Writes per-version Model_Version{N}/ subdirs with metrics and per-sample CSVs,
# plus version_(N-1)_to_N.csv (supervised fit) and version_N_to_N.csv (same-
# model Bellman residual) under the output dir.
set -euo pipefail
cd /home/ubuntu/vidur-classical-search
export PYTHONPATH=$PWD
# Allow the parent to use a few cores so HistGradientBoosting fits don't run
# fully serial; workers will throttle themselves to 1 thread via env in
# initializer.
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

BASE=$PWD/simulator_output/GV3_Agent/BellmanConvergence/cliff_aware_full_100v
LOG=$BASE.log
MON=$BASE.monitor.log
CACHE=$PWD/simulator_output/GV3_Agent/BellmanConvergence/_cliff_feature_cache_full

EXTRA=$(printf '{"classical_backend":"cliff_aware","cliff_feature_num_processes":64,"cliff_feature_chunk_records":2048,"cliff_feature_cache_dir":"%s","cliff_worker_threads":1,"bootstrap_batch_roots":64,"bootstrap_child_batch_size":4096,"target_worker_threads":1,"n_jobs":24,"max_trainable_params":150000}' "$CACHE")

mkdir -p "$BASE"
echo "started $(date)" > "$MON"
echo "extra=$EXTRA" >> "$MON"

(
  /usr/bin/time -f 'elapsed=%E maxrss_kb=%M' \
  /home/ubuntu/vidur/.venv/bin/python3 -u -m \
    vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.bellman_convergence \
    --dataset-dir /home/ubuntu/vidur/simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40 \
    --output-dir "$BASE" \
    --num-versions 100 \
    --num-roots 292713 \
    --eval-ratio 0.2 \
    --split-seed 12345 \
    --seed 12345 \
    --batch-size 256 \
    --root-player-filter controller \
    --model-name-prefix cliff_aware_full \
    --same-model-analysis-start-version 999 \
    --target-num-processes 64 \
    --extra-config-json "$EXTRA"
) > "$LOG" 2>&1 &
PID=$!
echo "runner_pid=$PID" >> "$MON"

while kill -0 "$PID" 2>/dev/null; do
  rss_kb=$(ps --no-headers -o rss= -p "$PID" 2>/dev/null | awk '{print $1+0}')
  rss_gib=$(awk -v kb="$rss_kb" 'BEGIN {printf "%.2f", kb/1024/1024}')
  workers=$(ps -eo cmd | grep -E 'multiprocessing.spawn|multiprocessing\.resource' | grep -v grep | wc -l)
  load=$(awk '{print $1","$2","$3}' /proc/loadavg)
  versions_done=$(ls -d "$BASE"/Model_Version* 2>/dev/null | wc -l)
  echo "ts=$(date +%F_%T) rss_gib=$rss_gib workers=$workers load=$load versions_done=$versions_done" >> "$MON"
  sleep 60
done
wait "$PID"
RC=$?
echo "completed rc=$RC ts=$(date +%F_%T)" >> "$MON"
exit "$RC"
