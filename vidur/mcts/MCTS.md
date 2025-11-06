**Vidur MCTS: Structure and Flow**

- Purpose
  - Coordinates two players — adversary (generates requests) and controller (allocates tokens) — over a Vidur simulator snapshot to explore action sequences and evaluate cost.

- Key Types
  - `MCTSNode` (vidur/vidur/mcts/mcts.py): wraps a `VidurMCTSState`, the `player` to act next, tree bookkeeping (`visits`, `cumulative_cost`, `children`, `untried_actions`).
  - `MCTSChildEdge`: pairs an action with a child node for selection/expansion.

- Entry Point
  - `search(iterations: int) -> ControllerAction`
    - Creates root node with `player='adversary'`.
    - Iterates: select → expand → simulate → backpropagate.
    - Logs each tree/rollout step to CSV via `_MCTSLogger`.

- Core Phases
  - `_create_root()`
    - Initializes `VidurMCTSState` from `VidurMCTSEnvironment.initial_state()` and sets `player='adversary'`.
    - Precomputes `untried_actions` for the root by calling `_enumerate_actions`.

  - `_select(root)`
    - Walks down using `_best_child` (UCB1) while node is fully expanded.

  - `_expand(node)`
    - Pops one action from `node.untried_actions` and applies it using the environment:
      - If node.player == 'adversary' → `env.apply_adversary_action_only`.
      - Else (controller) → `env.apply_controller_action_only`.
    - Creates a child with `player` flipped (`controller` after adversary; `adversary` after controller).
    - Prepares child’s `untried_actions` with `_enumerate_actions`.
    - Logs a “tree” row capturing state snapshot and action.

  - `_simulate(node)`
    - Performs `simulation_random_tries` rollouts from `node.state` (forked).
    - For each depth step (0..`simulation_depth`-1): alternates current player; samples a candidate action list from the environment and either applies a random action or skips if none.
    - Each applied action is logged as a “rollout” row.
    - At the end of a rollout, evaluates the state objective via `env.evaluate_objective` and accumulates cost.

  - `_backpropagate(node, cost)`
    - Adds the rollout’s scalar cost to every ancestor and increments visits.

- Action Enumeration
  - `_enumerate_actions(node)`
    - If `node.player == 'adversary'` → `env.sample_adversary_actions(state, max_branching)`.
    - Else → `env.sample_controller_actions(...)`.
    - Shuffles the list using the MCTS RNG (seeded with 0 by default).

- Cost
  - `_compute_objective_cost(violations, avg_lateness)` in mcts.py: `cost = violations + avg_lateness`.
  - `env.evaluate_objective` supplies the pair by scanning current requests.

- Logger Output (CSV)
  - Columns include: `phase` (root/tree/rollout), `player_to_act`, `next_player`, `sim_time`, objective metrics, and serialized actions.
  - Rows do not strictly alternate A→C→A; tree expansion may expand multiple adversary nodes consecutively, and rollout logging skips turns with no action.

- Determinism and Variation
  - Environment RNG is seeded by `SimulationConfig.seed` (default 42). MCTS RNG is seeded with 0.
  - To induce variation: change seeds or set `--mcts_max_branching` below number of waiters so sampling chooses subsets.

**File Map**
- Coordinator: `vidur/vidur/mcts/mcts.py`
- Environment: `vidur/vidur/mcts/environment.py`
- Config and SLO options: `vidur/vidur/mcts/config.py`
- Runner CLI: `vidur/vidur/mcts/run_mcts.py`

