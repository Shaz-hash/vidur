# DNN-MCTS Terminology & Logging Notes

This note documents how `mctsDNN.py` defines **trivial/forced nodes** and how to interpret the iteration-level CSV (`mcts_iter.csv`) fields and phases.

---

## 1) What is a “trivial / forced node” in `mctsDNN._expand_node`?

### Action-space at a node
At every node we call the environment to get:

- `actions_by_index`: fixed-size list (action-space) where each index corresponds to a deterministic action
- `mask`: fixed-size list/bool tensor where `True` means “this action index is valid in this state”

We then compute:

- `valid = [i for i, ok in enumerate(mask) if ok and actions_by_index[i] is not None]`
- `num_valid_actions = len(valid)`

### Terminal node
- **Terminal** means: `num_valid_actions == 0`
- There are no children to expand.
- In logs: `phase = "expand_terminal"`

### Forced node (previously called “trivial”)
- **Forced** means: `num_valid_actions == 1`
- There is only **one** valid action index, so this node has **exactly one child**.
- In logs: `phase = "expand_forced"`

#### Important refinement: when do we call the NN at a forced node?
A forced node can still carry important value information.

**New rule** (the one we want):
- If the forced node’s **parent had branching** (i.e., `len(parent.children) > 1`), then this forced node has **siblings**, meaning it lies on a meaningful branch selected from a larger choice set.
  - In this case: we still call the NN to evaluate the node’s **value** (controller perspective) so it can be backed up correctly to the root.
  - Priors are irrelevant here because there is only one outgoing action (PUCT exploration doesn’t matter).
- If the forced node’s parent also had no choice (parent had 1 child), this forced node is truly trivial.
  - In this case: skip NN (value can be treated as 0.0 for backup) because there is no meaningful decision boundary upstream.

**Why this matters**
Example: adversary is forced to no-op because the 1-second arrival window hasn’t elapsed yet, but we reached that adversary node due to a controller decision among many budgets/heuristics. Even though adversary has only one action, the value of that forced state should still be evaluated to correctly score the controller’s earlier branching choice.

---

## 2) Iteration-level logging: what is a “simulation iteration”?

- `sim_iteration` is one rollout / simulation inside one `search_dnn` call.
- Each `search_dnn` runs `iterations` rollouts on the same root.
- Each rollout:
  1) traverses down the tree using `select_child` until it reaches an unexpanded leaf
  2) reconstructs the leaf state (via replay/apply action)
  3) expands the leaf node (creates children)
  4) (maybe) calls NN at that node (depending on forced/branching logic)
  5) backpropagates the NN value + rewards up the path

---

## 3) Meaning of `phase` in `mcts_iter.csv`

### `phase = "expand"`
- The leaf node had **multiple valid actions** (`num_valid_actions > 1`)
- We called the NN at this node to get:
  - `nn_value_controller`
  - `nn_priors` (used to set `child.prior`)

### `phase = "expand_forced"`
- The leaf node had exactly **one valid action** (`num_valid_actions == 1`)
- We still create the single child.
- NN call behavior depends on the “forced node rule”:
  - If parent had branching → NN **can** be called (for value only).
  - If parent also forced → NN not called.
- The CSV explicitly shows this using:
  - `nn_called` = True/False

### `phase = "expand_terminal"`
- The leaf node has **no valid actions** (`num_valid_actions == 0`)
- No children are created; NN not called.

---

## 4) Column definitions in `mcts_iter.csv` (core fields)

### Root identity fields
- `game_id`: self-play episode id (one “game” trajectory).
- `root_id`: index of the root within the game. Root changes after you pick best action and advance state.
- `sim_iteration`: rollout index within this root’s MCTS search.
- `root_depth`: depth of the root within the game trajectory.
- `root_node_id`: node_id of the root node in the search tree for that root.
- `root_player`: player to act at the root (`"adversary"` or `"controller"`).

### Node identity fields
- `node_depth`: depth of this expanded node in the search tree (relative to root).
- `parent_node_id`: parent node id (blank if root).
- `node_id`: node id of the expanded node.

### Player fields
We store two different “player” meanings:

- `player_acted_to_create_this_node`:
  - The player who took the action from the parent to reach this node.
  - For root: `"root_no_parent"`.

- `player_to_act_in_this_node`:
  - The player who is supposed to act at THIS node (the node’s `node.player`).

These are not the same, and mixing them makes logs confusing.

### Action fields (incoming action from parent → this node)
- `action_index`: deterministic index in fixed action space (None/blank at root).
- `action_repr`: string repr of the action object.
- `prior`: prior probability assigned to the incoming edge (from parent’s NN priors, or 1.0 if forced).
- `reward`: immediate reward for taking parent→child transition:
  - computed from simulator cost delta, typically `reward = parent_cost - child_cost`
  - (controller perspective: higher is better)

### NN / expansion fields
- `nn_called`: whether NN inference was called at this node.
- `num_valid_actions`: number of valid actions at this node (computed from env mask + action list).
- `nn_value_controller`:
  - if `nn_called=True`: the NN value prediction at this node (controller perspective)
  - if `nn_called=False`: blank/0 depending on logging code (conceptually: no NN value was used)

### Cost/state snapshot fields (from simulator at this node)
- `objective_cost`: cost value of the simulator state at this node (as defined by your cost function).
- `sim_time`, `requests_in_system`, `requests_generated`, `requests_completed`, `slo_violations`, `avg_lateness`
- `state_waiting_ids`, `state_completed_request_ids`

### Action-detail JSON fields (optional debugging)
If the incoming action is adversary/controller, these store the action payload in JSON-ish strings:
- adversary: `adversary_requests`, `adversary_prefill_slos`, `adversary_decode_slos`
- controller: `controller_token_budget`, `controller_selected_ids`, `controller_allocations`,
  `controller_prefill_allocations`, `controller_decode_allocations`,
  `controller_prefill_total`, `controller_decode_total`, `controller_heuristic`, `controller_strategy`

---

## 5) Quick example interpretation (forced expansion case)

If you see:

- `phase = expand_forced`
- `num_valid_actions = 1`
- `nn_called = False`

This means:
- environment mask allowed exactly one action (forced)
- NN was skipped (either truly trivial forced, or you intentionally skipped evaluation)

If instead you see:

- `phase = expand_forced`
- `num_valid_actions = 1`
- `nn_called = True`

This means:
- forced node but parent had branching
- NN was still used to evaluate the node value for correct backup

---
