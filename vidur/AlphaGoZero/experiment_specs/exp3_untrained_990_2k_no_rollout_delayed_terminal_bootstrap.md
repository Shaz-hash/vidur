# EXP3: 0.990, 2K MCTS, No Rollout, Delayed Terminal Bootstrap

Experiment output directory:

```text
/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_EXP3_UNTRAINED_.990_2k_MCTS_SIMULTATIONS_NO_ROLLOUT_BUT_TERMINAL_VALUE_AFTER_5_SECs
```

## Purpose

Measure whether moving the trajectory-target bootstrap farther into the future
reduces value-target bias without using policy rollouts inside MCTS.

## Search Contract

- Model family: untrained-at-v100 DNN pipeline used by EXP3.
- MCTS simulations per decision: `2000`.
- MCTS leaf policy rollouts: disabled (`rollout_count = 0`).
- A newly expanded MCTS leaf is evaluated directly by its value model.
- Self-play root exploration remains independently configurable.
- Discount factor: `0.99`, consistently in Python MCTS, native MCTS,
  self-play trajectory backup, and evaluation.

## Trajectory And Replay Contract

All durations below are simulated time relative to the history-provided root.

- Replay sample window: first `5.0` seconds.
- Full self-play trajectory horizon: `12.0` seconds.
- Continue choosing actions with the same `2000`-simulation MCTS from 5 seconds
  through 12 seconds.
- Do not create replay samples for states after the 5-second sample window.
- Keep rewards and transition times from the complete 12-second trajectory.
- At the final 12-second state, evaluate the appropriate controller-perspective
  value model once.
- For every retained state, compute its value target from all subsequent
  simulator rewards through 12 seconds plus the time-discounted final value
  bootstrap.
- Retain the MCTS visit distribution generated at each sampled state.

For a retained state at time `t_i <= 5.0`, the target is:

```text
Z_i = r_i + d_i r_(i+1) + ... + D(i,T) V(s_T)
```

where each discount uses the existing time-varying game discount convention,
`T` is the state at or just beyond 12 simulated seconds, and `gamma = 0.99`.

States from 5 to 12 seconds affect the retained targets through their real
rewards and elapsed times, but they are not written as value or policy training
samples.

## Isolation Requirements

- This behavior must be opt-in.
- Existing 5-second direct-bootstrap self-play must remain unchanged.
- Existing fixed-horizon rollout MCTS must remain unchanged.
- Evaluation horizon and replay sample window must remain separate settings.
- Native and Python paths must expose equivalent settings and target semantics.

## Required Validation

1. Legacy modes produce unchanged outputs when delayed-terminal mode is off.
2. Only states in the first 5 simulated seconds are emitted to replay.
3. Rewards from the complete 12-second trajectory contribute to those targets.
4. The final value is evaluated at the 12-second state, not the 5-second state.
5. Time-varying discounted targets match an independently calculated reference.
6. Python and native MCTS agree at 2,000 simulations with rollouts disabled.
7. Native game traces pass Game Version 3 semantic and reward checks.
8. One-core native wall time is consistent with:

```text
number of MCTS searches x average one-core 2K no-rollout search time
```
