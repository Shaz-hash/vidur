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
├── prefill_calibrator.py       # generate/load baseline prefill timing tables
└── README.md                   # this document
```

A typical programmatic integration looks like:

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

### Prefill timing profiles

`prefill_calibrator.py` can be run directly to create the baseline prefill table
used for SLO assignment:

```
python -m vidur.mcts.prefill_calibrator \
  --step 64 \
  --max_tokens 2048 \
  --output data/prefill_profile.csv \
  --time_limit 10800 ... (other SimulationConfig flags)
```

Passing `prefill_profile_path` in `MCTSConstraintConfig` will reuse a saved CSV;
otherwise the environment generates the table on the fly.

## CLI runner & CSV trace

A small wrapper makes it easy to launch the MCTS loop from the command line and
record every explored transition:

```
python -m vidur.mcts.run_mcts \
  --mcts_iterations 50 \
  --mcts_log_csv simulator_output/mcts_trace.csv \
  --mcts_interval_request_size 1024 \
  --mcts_max_request_tokens 10240 \
  --mcts_prefill_profile simulator_output/prefill_profile.csv \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_device h100 \
  --replica_config_network_device h100_dgx \
  --cluster_config_num_replicas 1 \
  --replica_config_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --global_scheduler_config_type round_robin \
  --replica_scheduler_config_type vllm_v1
```

The CSV emitted by the runner/logging hook contains one row per joint action
(`phase="tree"` for nodes that became part of the search tree,
`phase="rollout"` for stochastic simulations, and `phase="root"` for the
initial snapshot).  Columns include:

| Column                          | Meaning                                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------------------- |
| `iteration`                     | MCTS iteration when the transition was generated (0 is the root snapshot).               |
| `phase`                         | `root`, `tree`, or `rollout`.                                                            |
| `depth`                         | Depth in the alternating game (root = 0).                                                |
| `parent_node_id` / `node_id`    | Identifiers of the predecessor and the resulting state.                                  |
| `player_to_act` / `next_player` | Player that acted to create the edge, and the player who will act next.                  |
| `sim_time`                      | Simulator time after applying the joint action.                                          |
| `requests_*` / `slo_*`          | Aggregated stats tracked by the environment (generated/completed/violations/lateness).   |
| `objective_cost`                | Scalar cost used for back-propagation (`violations + avg_lateness`).                     |
| `state_waiting_ids`             | Request IDs currently waiting in the replica queues.                                     |
| `state_completed_request_ids`   | Requests marked completed so far.                                                        |
| `adversary_requests`            | JSON payload describing each injected request (token sizes + SLOs).                      |
| `adversary_prefill_slos`        | Prefill SLOs chosen for the injected requests (JSON array).                               |
| `adversary_decode_slos`         | Decode SLOs chosen for the injected requests (JSON array).                                |
| `controller_token_budget`       | Chunk/token budget chosen by the controller for this step.                               |
| `controller_selected_ids`       | Request IDs prioritised by the controller (JSON array).                                  |
| `controller_allocations`        | Per-request token allocations passed to the VLLM v1 scheduler (JSON object).             |

## Notes and limitations

* The implementation uses VLLM‑v1 controllers underneath; it does **not**
  modify core scheduling logic in Vidur.
* The action spaces are sampled – only a handful of controller/adversary
  candidates are considered per node to keep the branching factor manageable.
* Rollouts use random actions bounded by the constraints.  You can experiment
  with smarter simulation policies by swapping in your own heuristics.
* This is currently a research harness, not a production‑ready planner.
*** End Patch
