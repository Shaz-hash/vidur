# Delayed-Terminal 0.990 / 2K No-Rollout Validation

## Configuration

- Full self-play horizon: 12 simulated seconds.
- Replay sample window: first 5 simulated seconds from the history root.
- Search: native full-tree MCTS, 2,000 simulations per non-trivial decision.
- Leaf evaluation: value-network bootstrap only; no policy rollout.
- Terminal evaluation: one controller-perspective value prediction at the final state.
- Discount factor: 0.99 with the existing time-varying discount denominator.
- Target backup: all 12-second rewards and discounts are backed up from the terminal value before replay rows are filtered to the first 5 seconds.

## Search Parity

Python and native results matched exactly at 2,000 simulations for both players.

| Player | Best canonical action | Visit L1 | Root value error | Native time | Python time |
|---|---:|---:|---:|---:|---:|
| Controller | 16 | 0.0 | 0.0 | 0.1095 s | 6.0006 s |
| Adversary | 250 | 0.0 | 0.0 | 0.1662 s | 8.5371 s |

Both native searches reported 2,000 value-inference calls and zero rollout actions.

## Native Game Validation

Both games were pinned to one CPU core with BLAS/OpenMP thread counts set to one.

| Game | Wall time | Full transitions | 2K searches | Replay rows | Terminal simulated time |
|---|---:|---:|---:|---:|---:|
| 1 | 48.50 s | 198 | 159 | 55 | 12.008797 s |
| 2 | 41.87 s | 172 | 140 | 42 | 12.006860 s |

The effective full-game cost per 2K search, including game transitions and amortized
startup, was 0.3050 seconds in game 1 and 0.2991 seconds in game 2.

The semantic audit checked:

- Consecutive turns/depths and alternating actors.
- Continuous, non-decreasing simulated time.
- Instantaneous adversary actions.
- Monotonic total objective cost and `cost = violations + lateness`.
- Request lifecycle, active/completed disjointness, and contiguous request IDs.
- Controller token budget and prefill/decode allocation conservation.
- Non-negative decode credits.
- Adversary prefill sizes against the authoritative GV3 configuration.

All 370 committed transitions passed.

## Target and Discount Validation

Targets were independently recomputed from the complete arena transition logs.

| Game | Max discount error | Max target error | Post-window transitions | Effect on first target |
|---|---:|---:|---:|---:|
| 1 | 0.0 | 1.78e-15 | 126 | 0.654787 |
| 2 | 0.0 | 3.55e-15 | 106 | 0.398958 |

The non-zero target effects prove that transitions after 5 seconds are used in the
targets retained from the first 5 seconds. They are not emitted as replay rows.

## Regression Coverage

- Focused replay target/window tests pass.
- Existing experiment configuration tests pass.
- Native and Python reverse target backup match.
- Legacy zero-window behavior still emits the full trajectory.
- The prior rollout mode remains operational: a 100-iteration, 10-rollout,
  one-second-horizon native regression produced exactly 1,000 rollout trajectories
  and passed rollout-deadline assertions.
