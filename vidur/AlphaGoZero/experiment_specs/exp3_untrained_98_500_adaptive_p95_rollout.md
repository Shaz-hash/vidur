# EXP3 Adaptive p95 Rollout Experiment

## Output

`/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_EXP3_UNTRAINED_.98_1k_MCTS_3s_p95_DEPENDENCE_LEAF_RELATIVE_4ROLLOUT_DNN_MARKOV_2_THREADS_PER_GAME`

The directory label contains `1k_MCTS`, but this run intentionally uses **500 MCTS
iterations**, as specified for the experiment.

## Search and training settings

- Fresh neutral/untrained Markov-v2 DNN bundle starts at v100.
- Discount factor: `0.98`.
- MCTS simulations per non-trivial state: `500`.
- Self-play PUCT constant: `1.20`.
- Evaluation and SJF PUCT constant: `1.0`.
- Leaf-relative policy rollouts: `4`.
- Parallel rollout threads per game: `2`.
- Initial and maximum rollout horizon: `3.0s`.
- Self-play and replay sample window: `5s`.
- Worker shard threshold: `4,000` states.
- New-version gate: `100,000` newly admitted states.
- First training requires at least `100,000` controller and `50,000`
  adversary states.
- Controller/adversary policy sample caps remain `500,000/300,000`.
- Arena evaluation uses four role blocks of `140` games; promotion remains
  strict wins `>81`. SJF uses `50` paired games.

## Adaptive horizon

Candidate v101 uses the initial `3.0s` horizon. After candidate `k` is
trained, its controller `p95_abs_error` determines the horizon for candidate
`k+1`:

```text
raw_horizon =
  prefill_128_time * ln(v_e / controller_p95_abs_error) / ln(discount)

calculated_horizon = clamp(raw_horizon, 0.0, 3.0)
rounded_horizon = nearest(calculated_horizon, 0.2s)
```

Constants:

- `v_e = 0.09`
- `discount = 0.98`
- `prefill_128_time = 0.015725797204323228s`
- adversary tick = `0.2s`
- rounding is nearest-tick, half-up: `2.46 -> 2.4`, `2.78 -> 2.8`

The horizon is not changed while a candidate is training or being evaluated.
After that candidate's evaluation completes, the coordinator atomically writes
`runtime_search_config.json` and broadcasts it to every worker. Workers read
that file before launching each new self-play game. The controller error
controls both controller and adversary rollout horizons.

`train_model.csv` records the horizon used by the current candidate, source
controller p95 error, raw horizon, capped/calculated horizon, rounded horizon,
all constants, and the residual discounted error.

## Parallelism

Each c8g.24xlarge contributes 90 selected vCPUs. At two rollout threads per
game, capacity is `floor(90 / 2) = 45` simultaneous games per machine.

- Self-play: 8 worker machines x 45 = **360** concurrent games.
- Evaluation: self-play pauses and XL joins, giving 9 x 45 = **405** slots.
- Four role blocks contain 560 game jobs, so they run in two bounded waves.
  Paired baseline/candidate games remain on the same host with identical game
  and history-hop offsets.
- The two SJF cycles contain 100 jobs and fit in one wave.

The implementation never oversubscribes the configured CPU slots; live CPU and
RAM checks may lower concurrency further.
