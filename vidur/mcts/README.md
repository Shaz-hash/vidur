# Vidur MCTS Playground

This package contains an experimental two–player Monte‑Carlo Tree Search loop
for Vidur.  It treats request generation as an adversary and the global
controller/scheduler as the protagonist.  The goal is to explore “what‑if”
scenarios by cloning the simulator via the snapshot support that was added for
MCTS work.

## Concepts

| Component                | Description                                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------- |
| `AdversaryAction`       | Requests to inject in the next turn.  Each request comes with (prefill, decode) size and SLO choices. |
| `ControllerAction`      | Controller hints: the chunk/token budget to use and an optional subset of requests to prioritise.     |
| `VidurMCTSEnvironment`  | Wraps a `Simulator`, applies actions, advances the run for a few events and evaluates SLO violations. |
| `VidurMCTS`             | UCB1 tree search alternating between adversary and controller moves.                                  |

The environment keeps the adversary bounded by:

* maximum QPS within a one‑second window,
* request token size / chunk granularity,
* a finite catalogue of SLO options for each stage.

The controller is allowed to:

* tweak the VLLM v1 scheduler token budget (chunk size) in multiples of the interval size,
* reorder the waiting queue to promote a subset of requests.

After each joint move the simulator advances for a limited number of events
(`simulation_depth`).  During evaluation the environment counts SLO violations
and the average lateness per violating request.  Those become the scalar cost
minimised by the controller and maximised by the adversary.

## Files

```
vidur/vidur/mcts/
├── __init__.py                 # convenience exports
├── config.py                   # constraint + search configuration objects
├── environment.py              # game state wrapper around the simulator
├── mcts.py                     # UCB1 search implementation
└── README.md                   # this document
```

There is no direct CLI yet.  A typical integration looks like:

```python
from vidur.simulator import Simulator
from vidur.config import SimulationConfig
from vidur.mcts import (
    MCTSConstraintConfig,
    MCTSExploreConfig,
    VidurMCTSEnvironment,
    VidurMCTS,
)

cfg = SimulationConfig.create_from_cli_args()
sim = Simulator(cfg, register_atexit=False)

env = VidurMCTSEnvironment(
    base_simulator=sim,
    constraints=MCTSConstraintConfig(maximum_qps=4),
    explore_cfg=MCTSExploreConfig(simulation_depth=3),
)
mcts = VidurMCTS(env, MCTSExploreConfig())

best_action = mcts.search(iterations=200)
print(best_action)
```

You can plug this into higher‑level optimisation loops by repeatedly applying
`env.apply_actions(...)` on snapshots produced during the tree search.

## Notes and limitations

* The implementation uses VLLM‑v1 controllers underneath; it does **not**
  modify core scheduling logic in Vidur.
* The action spaces are sampled – only a handful of controller/adversary
  candidates are considered per node to keep the branching factor manageable.
* Rollouts use random actions bounded by the constraints.  You can experiment
  with smarter simulation policies by swapping in your own heuristics.
* This is currently a research harness, not a production‑ready planner.
*** End Patch
