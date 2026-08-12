#!/usr/bin/env bash

# Reuse the validated adaptive-p95 EXP3 contract and change only the fresh
# experiment output root and self-play exploration constant.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp3_98_500_adaptive_p95_rollout_env.sh"

export AGZ_REMOTE_OUTPUT_ROOT=/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_EXP3_UNTRAINED_.98_1k_MCTS_3s_p95_DEPENDENCE_LEAF_RELATIVE_4ROLLOUT_DNN_MARKOV_2_THREADS_PER_GAME_PUCT_0p05
export AGZ_SELFPLAY_PUCT_C=0.05
