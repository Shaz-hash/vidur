#!/usr/bin/env bash

# Fresh EXP3 run from the untrained v100 bundle. Reuse the validated adaptive
# p95 DNN/Markov/replay/training contract and override only search exploration
# and one-thread scheduling settings.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp3_98_500_adaptive_p95_rollout_env.sh"

export AGZ_REMOTE_OUTPUT_ROOT=/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_EXP3_UNTRAINED_.98_2k_MCTS_3s_p95_DEPENDENCE_LEAF_RELATIVE_1ROLLOUT_DNN_MARKOV_1_THREAD_PER_GAME_PUCT_1p25_DIRICHLET12_OVER_N_SAMPLE15

export AGZ_MCTS_ITERATIONS=2000
export AGZ_EVAL_MCTS_ITERATIONS=2000
export AGZ_SELFPLAY_PUCT_C=1.25
export AGZ_EVAL_PUCT_C=1.25
export AGZ_SJF_PUCT_C=1.25

# Setting total concentration makes per-action alpha = 12 / canonical actions.
# The fixed alpha remains a compatibility fallback and is not used when N > 0.
export AGZ_ROOT_DIRICHLET_ALPHA=0.05
export AGZ_ROOT_DIRICHLET_TOTAL_CONCENTRATION=12
export AGZ_ROOT_DIRICHLET_EPSILON=0.25
export AGZ_INITIAL_SAMPLE_MOVE_COUNT=15

export AGZ_ROLLOUT_COUNT=1
export AGZ_EVAL_ROLLOUT_COUNT=1
export AGZ_SJF_ROLLOUT_COUNT=1
export AGZ_ROLLOUT_PARALLEL_THREADS=1
export AGZ_EVAL_ROLLOUT_PARALLEL_THREADS=1
export AGZ_SJF_ROLLOUT_PARALLEL_THREADS=1

# One core per game: 90 usable cores on each c8g.24xlarge worker.
export AGZ_WORKER_PARALLEL_GAMES=90
export AGZ_DISTRIBUTED_EVAL_ROLE_THREADS_PER_GAME=1
export AGZ_DISTRIBUTED_EVAL_SJF_THREADS_PER_GAME=1
export AGZ_DISTRIBUTED_EVAL_XL_PARALLEL=90
export AGZ_DISTRIBUTED_EVAL_WORKER_PARALLEL=90
export AGZ_DISTRIBUTED_EVAL_SJF_HOST_PARALLEL=90
