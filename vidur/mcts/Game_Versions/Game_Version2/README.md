# Game Version 2: Adversary 0.2s Tick Scheduling

This module defines the **Version 2** gameplay semantics for the virtual MCTS environment.

## Goal
Increase adversary branching by allowing decisions at mini-intervals inside each 1-second window, while preserving the rule that adversary can send at most once per second-window.

## Time Grid
- Mini tick size: `0.2s`
- Decision grid: `..., 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, ...`
- A state tracks `next_adv_tick` (the next adversary decision epoch).

## Adversary Action Space (7 options)
At each adversary decision epoch, action indices are:
- `0`: no-op (send 0 requests)
- `1`: send 1 request
- `2`: send 2 requests
- `3`: send 3 requests
- `4`: send 4 requests
- `5`: send 5 requests
- `6`: send 6 requests

All send actions use the same request shape as v1 (max prefill tokens, fixed decode tokens, SLO assignment logic).

## One-Send-Per-Window Rule
- A window is `[k, k+1)` for integer `k`.
- If adversary has already sent in current window, only no-op remains valid until the next window starts.
- If adversary sends at e.g. `0.8`, it cannot send again at `1.0`? Actually `1.0` is the **next window** and becomes legal again.

## Tick Progression
Given current adversary decision tick `t`:
- If adversary **sends**, next adversary decision tick becomes the next second boundary:
  - `next_adv_tick = floor(t) + 1.0`
- If adversary chooses **no-op**, next adversary decision tick becomes:
  - `next_adv_tick = t + 0.2`

## Controller Behavior Between Adversary Ticks
Controller can execute batches normally, even if a batch end time overshoots one or more adversary ticks.

If overshoot happens, adversary ticks are replayed as a backlog:
- While `next_adv_tick <= sim_time`, controller is forced to no-op.
- Adversary is asked to decide at each pending tick in order (`1.0`, then `1.2`, etc.).
- If adversary sends at a pending tick, injected requests use that pending tick as `arrived_at`.
- If adversary no-ops at a pending tick, the next pending tick is evaluated immediately.
- Controller is released again only once `next_adv_tick > sim_time`.

This preserves missed branching points without rewinding simulator time.

## Arrival/Apply Semantics
- Adversary actions are applied at the current decision tick.
- If simulator time is below the tick, simulator time is advanced to the tick first.
- Requests are injected at that tick.

## Internal Metadata (stored in stats map)
Version 2 keeps scheduling metadata in reserved keys inside `state.stats.decode_next_deadline_by_id`:
- next adversary tick
- last adversary decision tick
- window in which a send was consumed
- controller idle-until tick

This keeps metadata clone-safe through existing `VidurGameStats.clone()` without changing global state schemas.

## Current Scope
- Implemented for `Game_Version2/virtual_environment.py`.
- Python controller sampling path only.
- Native/full-native parity for v2 is not implemented yet.
