# EXP3: 2K MCTS / One Rollout / Dynamic Dirichlet Concentration 12

Output root:

`/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_EXP3_UNTRAINED_.98_2k_MCTS_3s_p95_DEPENDENCE_LEAF_RELATIVE_1ROLLOUT_DNN_MARKOV_1_THREAD_PER_GAME_PUCT_1p25_DIRICHLET12_OVER_N_SAMPLE15`

This is a fresh run from the untrained DNN v100 bundle. It retains the previous
adaptive-p95 Markov-v2 DNN training, FIFO replay, sampling, arena, promotion,
and discount-0.98 contract, with these search changes:

- Self-play, role evaluation, and SJF use 2,000 MCTS simulations.
- Every expanded leaf receives one policy-guided rollout on one thread.
- Rollout horizon starts at 3 seconds and follows the existing adaptive-p95
  calculation after candidate training.
- PUCT constant is 1.25 in self-play, role evaluation, and SJF.
- Self-play root Dirichlet epsilon is 0.25 and per-action alpha is
  `12 / canonical_action_count`. Evaluation root noise remains disabled.
- Visit-temperature action sampling is enabled for the first 15 game actions;
  later actions select the most-visited root child deterministically.
- Each worker runs 90 concurrent self-play games. Distributed evaluation also
  uses one thread per game and up to 90 slots per c8g.24xlarge host.

The executable environment contract is
`settingUpServers/exp3_98_2k_rollout1_dynamic_alpha12_puct125_env.sh`.
