**Vidur MCTS Environment: API and Internals**

- Purpose
  - Provides MCTS-compatible transitions over a Vidur simulator: sampling actions, applying adversary/controller moves, advancing the event loop, and scoring states.

- Key Data Classes
  - `VidurMCTSState`: `{ simulator: Simulator, stats: VidurGameStats }` with `fork()` (deep snapshot via `Simulator.fork()` and cloned stats).
  - `VidurGameStats`: requests generated/completed, SLO counters, recent arrivals (for QPS), and `completed_request_ids`.
  - `AdversaryAction`/`AdversaryRequestSpec`: requests to inject (prefill/decode sizes and SLOs).
  - `ControllerAction`: token budget + per-request allocations split into `prefill_allocations` and `decode_allocations`.

- Construction
  - Seeds RNG from `base_simulator._config.seed`.
  - Loads `PrefillProfile` from CSV or generates; applies `prefill_slowdown` to times.

- State Helpers
  - `initial_state()` returns a forked simulator snapshot with clean stats.

- Action Sampling
  - `sample_adversary_actions(state, max_samples)`
    - Respects `maximum_qps` window; for each sample:
      - Chooses prefill/decode sizes within `[min_request_tokens, max_request_tokens]` in `interval_request_size` steps.
      - `prefill_slo = lookup(prefill_tokens) * random_choice(prefill_slos)`.
      - `decode_slo = random_choice(decode_slos) / 1000.0`.
  - `sample_controller_actions(state, max_samples)`
    - Drains arrivals, enumerates feasible token budgets (`_enumerate_token_budgets`), builds request lookup.
    - Picks a selection set (by default all waiting ids when `len(waiting) <= max_branching`), then calls `_generate_allocation_variants` to produce allocations.
    - Falls back to a minimal action when nothing else is possible.

- Applying Actions
  - `apply_adversary_action_only(state, action)`
    - For each request spec: constructs `Request`, sets SLOs via `SLOManager`, then overrides the prefill/decode SLOs with the sampled values. Enqueues a `RequestArrivalEvent` at current sim time.
    - Stats: updates `requests_generated`, trims `recent_arrivals` to 1s window.

  - `apply_controller_action_only(state, action)`
    - Drains arrivals.
    - Configures controller move via `_configure_controller_action`:
      - Validates allocations; clamps to remaining work.
      - Builds a `_ControllerBudgetTracker` that tracks per-request token budgets against progress baselines.
      - Restricts each replica scheduler to the targeted requests only (both waiting and running sets are temporarily masked); prioritizes targeted ids in the waiting queue.
      - Pushes per-request overrides and caps per-replica `chunk_size` to the total override sum.
    - Enqueues a `GlobalScheduleEvent` and a `ReplicaScheduleEvent` for each activated replica at the same sim time.
    - Advances the event queue (`_advance_simulation`) until the tracker budgets are satisfied, then prunes same-timestamp replica schedules.
    - Restores schedulers’ overrides and `chunk_size`, and restores hidden waiting/running entries.
    - Updates stats (records completions and SLO contributions for completed requests only).

- Simulation Advance
  - `_advance_simulation(state, tracker)`
    - Pops events (excluding future arrivals), feeds them to schedulers, enqueues follow-up events.
    - After each event, rebuilds a request lookup and checks `tracker.is_satisfied`; when true, prunes any same-time `REPLICA_SCHEDULE` events and returns.

- Budget Tracking
  - `_ControllerBudgetTracker`
    - Holds `{ rid: (prefill_budget, decode_budget) }` and baselines for processed prefill/decode tokens.
    - `is_satisfied(lookup)`: returns true once each targeted request has advanced by at least its requested amounts.

- Lateness and Objective
  - `_compute_lateness(request, sim_time)` returns the sum of:
    - Prefill lateness: `max(0, actual_prefill_completed_at − (arrived_at + prefill_slo_time))`.
    - Decode lateness: worst of
      - lag relative to the deadline for the last completed decode token, and
      - lag relative to the next decode token deadline (so waiting contributes),
      where deadlines are computed from `prefill_completed_at + k * decode_slo_time` (with one implicit token granted at prefill end accounted via a baseline).
  - `evaluate_objective(state)`
    - Starts from accumulated stats for completed requests.
    - Scans live requests and adds any positive lateness; increments violation count per late request; computes `avg_lateness` as `sum_lateness / violations`.
  - Cost used by MCTS: `violations + avg_lateness`.

- Utilities
  - `_build_request_lookup(simulator)` collects all live requests from each replica scheduler in waiting, running, and in-replica maps.
  - `_restrict_waiting_queue` masks non-targeted waiters; `_restrict_running_set` masks non-targeted running requests; `_restore_hidden_requests` restores them after the step.
  - `_snapshot_scheduler_budget_state`/`_restore_scheduler_budget_state` capture and restore per-replica overrides and `chunk_size`.
  - `_drain_arrivals` drains `REQUEST_ARRIVAL`/`GLOBAL_SCHEDULE` events at the controller step boundary and reapplies adversary SLO overrides attached to request objects.

- Randomness and Seeds
  - Environment RNG is seeded from `SimulationConfig.seed` (default 42). This affects adversary action sampling and controller allocation variants.

- Invariants and Guards
  - Decode tokens are only allocated to requests whose prefill has completed.
  - Controller per-request decode allocations are capped at one token per step; prefill allocations are multiples of `interval_request_size` up to remaining prefill.
  - Per-step simulator advance stops as soon as the requested budgets are actually reflected in request progress (enforced by the tracker).

- Where to Look in Code
  - Public API: `apply_adversary_action_only`, `apply_controller_action_only`, `sample_adversary_actions`, `sample_controller_actions`, `evaluate_objective`, `describe_state`.
  - Core internals: `_configure_controller_action`, `_advance_simulation`, `_compute_lateness`, masking/restoration helpers.

