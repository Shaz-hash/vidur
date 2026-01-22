# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import time 

# Misc Imports for debugging the tree and logging
from .DNN.misc import dump_tree_snapshot_csv


# Imports for excuting DNN model during MCTS
import torch 
from .DNN import infer as dnn_infer

# Logger Imports
from .logger.mctsDNN_logger import DNNMCTSIterationLogger, DNNMCTSRootSummaryLogger

from .launch_mcts_job import MCTSExploreConfig
from .environment import (
    AdversaryAction,
    ControllerAction,
    VidurMCTSEnvironment,
    VidurMCTSState,
)

class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
 

@dataclass
class MCTSNode:
    player: str  # "adversary" or "controller"
    node_id: int = 0
    depth: int = 0
    parent: Optional["MCTSNode"] = None
    
    # Action that led from parent -> this node
    parent_action: Optional[Union[AdversaryAction, ControllerAction]] = None
    parent_action_index: Optional[int] = None

    # MuZero stats
    prior: float = 0.0          # π(a|s_parent) for this node's incoming edge
    reward: float = 0.0         # immediate controller-reward from parent->this node, essentially incoming edge/action cost from parent state to this state
    visits: int = 0
    value_sum: float = 0.0 
    state_cost: float = 0.0  # absolute cost at this node's state (used for effficient cost calculation during the 1 simulator step)
    
    # Expanded children: index -> child node (this property is for Tree structure & navigation) 
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    
    # To help logging and debugging and training IA
    sim_time: float = 0.0   # NEW: simulator time at this node after performing action
    nn_value_controller: float | None = None        # value from controller perspective
    nn_priors: list[float] | None = None            # length = full action space size
    nn_valid_mask: list[bool] | None = None         # length = full action space size
    num_valid_actions: int = 0  # last computed valid-actions count at this node

    # Controller action deduplication (only meaningful when player == "controller")
    action_alias_to_canonical: Dict[int, int] = field(default_factory=dict)      # alias_idx -> canonical_idx
    canonical_to_action_aliases: Dict[int, List[int]] = field(default_factory=dict)  # canonical_idx -> [idxs...]


    def expanded(self) -> bool:
        return len(self.children) > 0

    def mean_value(self) -> float:
        return (self.value_sum / self.visits) if self.visits > 0 else 0.0

    # TODO: Remove this function if not needed
    # def is_fully_expanded(self) -> bool:
    #     return len(self.untried_actions) == 0


# Essentially an action from a given node leading to its child
@dataclass
class MCTSChildEdge:
    action: Union[AdversaryAction, ControllerAction]
    child: "MCTSNode"
    prior: float = 0.0 # probability from policy network for this action
    action_index: Optional[int] = None  # index of this action in the action space
    action_cost : float = 0.0 # True Cost incurred by simulator when applying this action (we wont be using this at the moment, this has been shifted as reward in MCTSNode)
    # node: MCTSNode


class VidurMCTS:
    """UCB1-based Monte Carlo Tree Search coordinating adversary and controller."""

    def __init__(
        self,
        env: VidurMCTSEnvironment,
        explore_cfg: MCTSExploreConfig,
        rng: Optional[random.Random] = None,
        log_path: Optional[Union[str, Path]] = None,
        tree_log_path: Optional[Union[str, Path]] = None,   # NEW
        *,
        logger_flush_every: int = 1,
        verbose: bool = False,
        ## A flag to enable logs for normalised version of the inputs given to the model at root inference
        _did_root_infer_debug = False
        # history_depth: int = 0,          # NEW
    ) -> None:
        self._env = env
        self._cfg = explore_cfg
        self._rng = rng or random.Random(0)
        self._verbose = verbose

        self._node_counter = 0
        self._root: Optional[MCTSNode] = None
        self._history_root_state: Optional[VidurMCTSState] = None
        self._history_root_node: Optional[MCTSNode] = None

        self._iter_logger = DNNMCTSIterationLogger(log_path, flush_every=logger_flush_every)
        self._root_logger = DNNMCTSRootSummaryLogger(tree_log_path, flush_every=logger_flush_every)

        # --- Time-based discount calibration ---
        step_tokens = int(getattr(getattr(self._env, "_constraints", None), "interval_request_size", 512) or 512)
        try:
            slowdown = float(getattr(self._env._constraints, "prefill_slowdown", 1.0) or 1.0)
            if slowdown <= 0:
                slowdown = 1.0

            scaled = float(self._env._prefill_profile.lookup(step_tokens) or 1e-9)
            self._prefill_step_time = scaled / slowdown  # undo slowdown for discount calibration

        except Exception:
            self._prefill_step_tokens = step_tokens
            self._prefill_step_time = 0.0388862329  # safe fallback



    # TODO: call this function in either SelfPlay or AlphaZero.py class
    def close(self) -> None:
        if getattr(self, "_iter_logger", None) is not None:
            self._iter_logger.close()
            self._iter_logger = None
        if getattr(self, "_root_logger", None) is not None:
            self._root_logger.close()
            self._root_logger = None

    # ------------------------------------------------------------------ #
    # Core phases
    # ------------------------------------------------------------------ #


    def _state_cost(self, state: VidurMCTSState) -> float:
        violations, avg_lateness = self._env.evaluate_objective(state)
        return float(violations) + float(avg_lateness)

    def _transition_reward(self, parent_cost: float, child_cost: float) -> float:
        # controller-reward (higher better): reward = -(child_cost - parent_cost)
        # Essentially the cost will always increase therefore, reward will be negative 
        return parent_cost - child_cost

    def _time_discount(self, t_child_time: float, t_parent_branch: float) -> float:
        """
        discount = gamma ^ ((t_child_time - t_parent_branch) / prefill_time(step_tokens))
        """
        gamma = float(getattr(self._cfg, "discount_factor", 0.98))
        denom = max(float(getattr(self, "_prefill_step_time", 0.0388862329)), 1e-9)
        dt = max(0.0, float(t_child_time) - float(t_parent_branch))
        return gamma ** (dt / denom)


    """
        This function is there to prevent going over the same actions for a given state multiple times during expansion.
    """
    def _controller_action_key(self, action: ControllerAction) -> tuple:
        """
        Two controller actions are 'equivalent' if they allocate the same tokens
        to the same request ids (token_allocations dict is identical).
        """
        alloc = action.token_allocations or {}
        # stable, hashable
        return tuple(sorted((int(rid), int(tok)) for rid, tok in alloc.items()))



    def _replay_to_node(self, node: MCTSNode) -> VidurMCTSState:
        """Reconstruct a fresh state at `node` by replaying actions.

        If a random history prefix was generated, we treat the history root
        state as the baseline and only replay actions *below* that node.
        Otherwise, we start from a fresh initial state and replay from the
        true root.
        """
        # Decide the baseline state and the cut node in the ancestry.
        if self._history_root_node is not None and self._history_root_state is not None:
            # baseline_state = self._history_root_state
            # cut_node = self._history_root_node

            # Clone from the frozen history-root snapshot, with stats cloned
            baseline_state = self._env.clone_history_root_state(
                self._history_root_state.stats
            )
            cut_node = self._history_root_node
        else:
            baseline_state = self._env.initial_state()
            cut_node = None

        # Collect path from `cut_node` (exclusive) down to `node`.
        path: List[MCTSNode] = []
        cur = node
        while cur is not None and cur is not cut_node:
            path.append(cur)
            cur = cur.parent
        path.reverse()

        # Work on a fork so we never mutate the stored baseline state.
        # state = baseline_state.fork()
        state = baseline_state
        # Replay actions inplace along the path segment.
        for n in path:
            parent = n.parent
            action = n.parent_action
            if parent is None or action is None:
                continue
            if parent.player == "adversary":
                state = self._env.apply_adversary_action_only(state, action, inplace=True)
            else:
                state = self._env.apply_controller_action_only(state, action, inplace=True)

        return state


    def _nn_value_and_priors(
        self,
        dnn_model: Any,
        state: VidurMCTSState,
        player: str,
        action_mask: Optional[torch.Tensor],
    ) -> Tuple[float, List[float]]:
        device = next(dnn_model.parameters()).device if hasattr(dnn_model, "parameters") else torch.device("cpu")
        do_debug = (getattr(state, "simulator", None) is not None) and (player in ("adversary","controller"))
        do_debug = do_debug and (self._root is not None) and (self._root.player == player) and (self._did_root_infer_debug is False)


        inputs = dnn_infer.build_model_inputs(
            state,
            player,
            device,
            max_prefill_tokens=int(getattr(self._env._constraints, "max_request_tokens", 3072) or 3072),
            prefill_slowdown=float(getattr(self._env._constraints, "prefill_slowdown", 3.0) or 3.0),
            debug=do_debug,
            debug_out_path="simulator_output/mcts_dnn_logs/infer_root_debug.txt" if do_debug else None,
        )

        if do_debug:
            self._did_root_infer_debug = True

        # Build base inputs from state
        # inputs = dnn_infer.build_model_inputs(
        #             state,
        #             player,
        #             device,
        #             max_prefill_tokens=int(getattr(self._env._constraints, "max_request_tokens", 3072) or 3072),
        #             prefill_slowdown=float(getattr(self._env._constraints, "prefill_slowdown", 3.0) or 3.0),
        #         )


        # Override action mask from env (convert list -> tensor [1, A])
        if action_mask is not None:
            
            inputs = type(inputs)(
                req_features=inputs.req_features,
                global_features=inputs.global_features,
                req_mask=inputs.req_mask,
                action_mask=action_mask,
            )

        value, priors = dnn_model.infer_from_inputs(inputs, player, device=device)
        return float(value), list(priors)

    def _actions_and_mask(self, state: VidurMCTSState, player: str) -> Tuple[List[Optional[object]], torch.Tensor]:
        if player == "controller":
            actions_by_index, mask = self._env.sample_controller_actions(state, self._cfg.max_branching)
        else:
            actions_by_index, mask = self._env.sample_adversary_actions(state, self._cfg.max_branching)

        # mask must be 1D bool, length == len(actions_by_index)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        else:
            mask = mask.to(dtype=torch.bool)

        return actions_by_index, mask

    # TODO: Let the environment produce the action space with fixed indexing & MORE IMPORTANTLY, we need to have cost and violations here added for each child node created!
    def _expand_node(self, node: MCTSNode, state: VidurMCTSState, dnn_model: Any) -> Tuple[float, bool, int]:
        """
        Expand `node` by creating children nodes for all valid indexed actions, setting their priors.
        Returns the NN value at this node (controller perspective) for backup.
        """
        actions_by_index, mask = self._actions_and_mask(state, node.player)

        # collect valid indices
        valid = [i for i, ok in enumerate(mask) if ok and actions_by_index[i] is not None]
        num_valid_actions = len(valid)
        node.num_valid_actions = int(num_valid_actions)


        # Forced / terminal cases: skip NN entirely
        if len(valid) == 0:
            # terminal: no children, leaf value estimate = 0
            return 0.0, False , 0

        # parent_had_multiple_children = (node.parent is None) or (len(node.parent.children) > 1)
        # should_call_nn = bool(parent_has_multiple_children)



        # if len(valid) == 1:
        #     i = valid[0]
        #     next_player = "controller" if node.player == "adversary" else "adversary"

        #     if i not in node.children:
        #         node.children[i] = MCTSNode(
        #             player=next_player,
        #             node_id=self._next_node_id(),
        #             depth=node.depth + 1,
        #             parent=node,
        #             parent_action=actions_by_index[i],
        #             parent_action_index=i,
        #             prior=1.0,      # forced move
        #             reward=0.0,     # will be filled when traversed in run_one_simulation
        #         )
    

        #     # NN only if parent's branching (for VALUE evaluation)
        #     if parent_had_multiple_children:
        #         value, priors = self._nn_value_and_priors(
        #             dnn_model, state, node.player, action_mask=mask.unsqueeze(0)
        #         )
        #         node.nn_value_controller = float(value)
        #         node.nn_priors = list(priors)
        #         node.nn_valid_mask = [bool(x) for x in mask.tolist()]
        #         return float(value), True, 1

        #     return 0.0, False, 1

        if num_valid_actions == 1:
            idx = valid[0]
            next_player = "controller" if node.player == "adversary" else "adversary"

            if idx not in node.children:
                node.children[idx] = MCTSNode(
                    player=next_player,
                    node_id=self._next_node_id(),
                    depth=node.depth + 1,
                    parent=node,
                    parent_action=actions_by_index[idx],
                    parent_action_index=idx,
                    prior=1.0,
                    reward=0.0,
                    visits=0,
                    value_sum=0.0,
                    sim_time=state.simulator._time,
                    state_cost=0.0,
                    num_valid_actions=0,
                    branch_anchor_time=node.branch_anchor_time,
                )

            # NEW: never call NN on single-child states
            return 0.0, False, 1



        # ------------------------
        # MULTIPLE-CHILD
        # ------------------------
        # Always call NN to get PRIORS for multi-child nodes
        model_value, priors = self._nn_value_and_priors(
            dnn_model, state, node.player, action_mask=mask.unsqueeze(0)
        )
        node.nn_value_controller = float(model_value)
        node.nn_priors = list(priors)
        node.nn_valid_mask = [bool(x) for x in mask.tolist()]

        # But only USE the value for backup if parent had multiple children.
        # If parent had a single child -> "trivial-multiple-child": value passed upward is 0.0.
        # value_used_for_backup = float(model_value) if parent_had_multiple_children else 0.0
        value_used_for_backup = float(model_value) 
        next_player = "controller" if node.player == "adversary" else "adversary"

        "Creating only child nodes with unique valid actions"
        # -----------------------------
        # Controller action dedup (multi-child only)
        # -----------------------------
        canonical_indices: List[int] = []
        canonical_prior: Dict[int, float] = {}

        # Default: no dedup for adversary nodes
        node.action_alias_to_canonical = {}
        node.canonical_to_action_aliases = {}

        if node.player == "controller":
            sig_to_canon: Dict[tuple, int] = {}
            canon_to_aliases: Dict[int, List[int]] = {}
            alias_to_canon: Dict[int, int] = {}

            # Group by identical token_allocations
            for idx in valid:
                act = actions_by_index[idx]
                if act is None:
                    continue
                if not isinstance(act, ControllerAction):
                    continue
                sig = self._controller_action_key(act)
                canon = sig_to_canon.get(sig)
                if canon is None:
                    canon = idx
                    sig_to_canon[sig] = canon
                    canon_to_aliases[canon] = [idx]
                else:
                    canon_to_aliases[canon].append(idx)
                alias_to_canon[idx] = canon

            node.action_alias_to_canonical = alias_to_canon
            node.canonical_to_action_aliases = canon_to_aliases

            canonical_indices = sorted(canon_to_aliases.keys())

            # Prior mass for canonical child = sum of priors for its aliases
            for canon, aliases in canon_to_aliases.items():
                psum = 0.0
                for aidx in aliases:
                    if 0 <= aidx < len(priors):
                        psum += float(priors[aidx])
                canonical_prior[canon] = psum

        else:
            # No dedup: all valid indices are canonical
            canonical_indices = valid
            for idx in canonical_indices:
                canonical_prior[idx] = float(priors[idx]) if 0 <= idx < len(priors) else 0.0



        # Create children ONLY for canonical indices
        for idx in canonical_indices:
            action = actions_by_index[idx]
            if action is None:
                continue
            if idx in node.children:
                continue

            node.children[idx] = MCTSNode(
                player=next_player,
                node_id=self._next_node_id(),
                depth=node.depth + 1,
                parent=node,
                parent_action=action,
                parent_action_index=idx,
                prior=float(canonical_prior.get(idx, 0.0)),
                reward=0.0,
                visits=0,
                value_sum=0.0,
                sim_time=state.simulator._time,
            )

        return value_used_for_backup, True, num_valid_actions



    def ucb_score(self, parent: MCTSNode, child: MCTSNode, min_max_stats: MinMaxStats) -> float:
        pb_c_base = getattr(self._cfg, "pb_c_base", 19652)
        pb_c_init = getattr(self._cfg, "pb_c_init", 1.25)

        parent_is_branching = (getattr(parent, "num_valid_actions", 0) > 1) or (len(parent.children) > 1)

        pb_c = math.log((parent.visits + pb_c_base + 1.0) / pb_c_base) + pb_c_init
        pb_c *= math.sqrt(parent.visits + 1.0) / (child.visits + 1.0)

        prior_score = pb_c * float(child.prior)

        if child.visits > 0:
            # Time-based discount ONLY when parent is branching
            disc = self._time_discount(float(child.sim_time), float(parent.sim_time)) if parent_is_branching else 1.0

            q_controller = float(child.reward) + disc * float(child.mean_value())
            q_norm = min_max_stats.normalize(q_controller)

            # controller maximizes controller-Q; adversary minimizes it
            value_score = q_norm if parent.player == "controller" else -q_norm
        else:
            value_score = 0.0

        return float(prior_score) + float(value_score)


    def select_child(self, node: MCTSNode, min_max_stats: MinMaxStats) -> Tuple[int, MCTSNode]:
        assert node.children, "select_child called on unexpanded node"

        if len(node.children) == 1:
            idx, only_child = next(iter(node.children.items()))
            return int(idx), only_child

        max_ucb = max(self.ucb_score(node, child, min_max_stats) for child in node.children.values())
        best = [idx for idx, child in node.children.items() if self.ucb_score(node, child, min_max_stats) == max_ucb]
        action_index = self._rng.choice(best)
        return int(action_index), node.children[action_index]

    
    def _advance_through_single_child_chain(
        self,
        node: MCTSNode,
        state: VidurMCTSState,
        search_path: List[MCTSNode],
        *,
        forced_step_logs: Optional[List[Tuple[MCTSNode, Dict[str, Any], int, int]]] = None,
        max_hops: int = 2000,
    ) -> Tuple[MCTSNode, VidurMCTSState]:
        """
        Keep advancing while the current node has exactly 1 valid action.
        Stop at:
        - 0 valid actions (terminal), or
        - >1 valid actions (branching)
        ** FOR LOGGING PURPOSES **
        If forced_step_logs is provided, we record a log payload for each *skipped* node
        (the single-child nodes we are bypassing).
        """
        hops = 0
        while True:
            if hops >= max_hops:
                return node, state

            actions_by_index, mask = self._actions_and_mask(state, node.player)
            valid = [i for i, ok in enumerate(mask) if ok and actions_by_index[i] is not None]
            n_valid = len(valid)
            node.num_valid_actions = int(n_valid)

            if n_valid != 1:
                return node, state
            
            # DEBUG: controller should not be "forced" if any request still has prefill remaining
            # TODO : SHIFT THIS TO checks.py file later in function
            if n_valid == 1 and node.player == "controller":
                try:
                    req_lookup_fn = getattr(self._env, "_build_request_lookup", None)
                    if callable(req_lookup_fn):
                        reqs = req_lookup_fn(state.simulator).values()
                    else:
                        reqs = []
                        print("[WARN] no _build_request_lookup function in environment for prefill debug check")

                    pending = []
                    total_remaining = 0
                    for req in reqs:
                        
                        req_status = getattr(req, "is_prefill_complete", None)
                        if None == req_status:
                            print("ERROR IN STATUS")
                        elif req_status is False:
                            print("PREFILL REQUEST NOT COMPLETE!!")

                        cached = int(getattr(req, "num_prefill_tokens_cached", 0) or 0)
                        remaining = int(req.num_prefill_tokens) - int(req.num_processed_prefill_tokens) - cached
                        if remaining > 0 and not bool(getattr(req, "is_prefill_complete", False)):
                            pending.append((int(req.id), int(remaining)))
                            total_remaining += int(remaining)

                    if pending:
                        print(
                            "[ERROR] controller single-child while prefill pending: "
                            f"node_id={node.node_id} depth={node.depth} sim_time={state.simulator._time:.6f} "
                            f"pending_count={len(pending)} total_remaining={total_remaining} "
                            f"pending_sample={pending[:10]}"
                        )
                except Exception as e:
                    print(f"[WARN] failed prefill-pending debug check: {e}")


            # record the CURRENT node (the one we're about to skip through)
            if forced_step_logs is not None:
                snap = self._env.describe_state(state)
                forced_step_logs.append((node, snap, int(n_valid), int(len(node.children))))

            idx = valid[0]
            next_player = "controller" if node.player == "adversary" else "adversary"

            if idx not in node.children:
                node.children[idx] = MCTSNode(
                    player=next_player,
                    node_id=self._next_node_id(),
                    depth=node.depth + 1,
                    parent=node,
                    parent_action=actions_by_index[idx],
                    parent_action_index=idx,
                    prior=1.0,
                    reward=0.0,
                    visits=0,
                    value_sum=0.0,
                    sim_time=state.simulator._time,
                    state_cost=0.0,
                    num_valid_actions=0,
                    # branch_anchor_time=node.branch_anchor_time,
                )

            child = node.children[idx]
            action = child.parent_action
            assert action is not None

            parent_cost = float(node.state_cost)

            # Apply forced action
            if node.player == "adversary":
                state = self._env.apply_adversary_action_only(state, action, inplace=True)
            else:
                state = self._env.apply_controller_action_only(state, action, inplace=True)

            child_cost = self._state_cost(state)
            step_reward = self._transition_reward(parent_cost, child_cost)

            child.reward = float(step_reward)
            child.state_cost = float(child_cost)
            child.sim_time = state.simulator._time
            # branching anchor doesn't change inside a forced chain
            # child.branch_anchor_time = node.branch_anchor_time

            search_path.append(child)
            node = child
            hops += 1



    # def backpropagate(self, search_path: List[MCTSNode], value: float, min_max_stats: MinMaxStats) -> None:
    #     """
    #     New backup rule:
    #     - If parent has 1 child => propagate value upward with NO discount and reward = 0
    #     - If parent has >1 children => apply time-based discount from its child time to parent's sim_time:
    #         value = reward + disc * value
    #     and reset "last branching time" to the parent branching time.
    #     """
    #     value = float(value)
    #     # last_branch_time = float(search_path[-1].sim_time)  # time_child_branching (starts at leaf)

    #     for node in reversed(search_path):
    #         node.value_sum += value
    #         node.visits += 1

    #         parent = node.parent
    #         if parent is None:
    #             break

    #         parent_is_branching = bool(getattr(parent, "num_valid_actions", 0) > 1) or (len(parent.children) > 1)

    #         if parent_is_branching:
    #             disc = self._time_discount(node.sim_time, parent.sim_time)

    #             # print(
    #             #     f"[DISCDBG] parent={parent.node_id} node={node.node_id} "
    #             #     f"parent_time={parent.sim_time:.12f} node_time={node.sim_time:.12f} "
    #             #     f"dt={(node.sim_time-parent.sim_time):.12f} denom={self._prefill_step_time:.12f} "
    #             #     f"gamma={getattr(self._cfg,'discount_factor',0.98)} disc={disc:.12f}"
    #             # )


    #             if (disc == 1 and parent.player != "adversary"):
    #                 print("Warning: Discount factor is 1.0 at adversary branching node during backpropagation. Check time settings.")
                    

    #             if(disc > 1 or disc <= 0) :
    #                 print("Abnormal discount factor detected during backpropagation:", disc)
    #             # update minmax using the same Q-shape you're backing up at branching boundaries
    #             min_max_stats.update(float(node.reward) + disc * float(node.mean_value()))

    #             value = float(node.reward) + disc * value
    #             # last_branch_time = float(parent.sim_time)
    #         else:
    #             # parent has 1 child => forced edge contributes 0 reward and no discount
    #             # value unchanged, last_branch_time unchanged
    #             pass

    def backpropagate(self, search_path: List[MCTSNode], value: float, min_max_stats: MinMaxStats) -> None:
        """
        Backup rule (per-edge discount accumulation):
        - Always apply time discount based on (node.sim_time - parent.sim_time).
        - Use reward ONLY when parent is branching; otherwise reward=0 (your desired semantics).
        """
        value = float(value)

        for node in reversed(search_path):
            node.value_sum += value
            node.visits += 1

            parent = node.parent
            if parent is None:
                break

            parent_is_branching = bool(getattr(parent, "num_valid_actions", 0) > 1) or (len(parent.children) > 1)

            disc = self._time_discount(float(node.sim_time), float(parent.sim_time))

            reward_used = float(node.reward) if parent_is_branching else 0.0

            if parent_is_branching:
                min_max_stats.update(reward_used + disc * float(node.mean_value()))

            value = reward_used + disc * value


    def run_one_simulation(self, root: MCTSNode, root_state: VidurMCTSState, dnn_model: Any, min_max_stats: MinMaxStats , game_id: Optional[str] = None, root_id: Optional[str] = None, sim_iteration: Optional[int] = None) -> None:
        node = root
        search_path = [node]

        # Selection: descend until we hit an unexpanded node
        while node.expanded():
            _, node = self.select_child(node, min_max_stats)
            search_path.append(node)

        # Compute state at this leaf node AND its reward (from parent transition)
        if node.parent is None:
            # leaf_state = root_state
            # fresh clone so MCTS never mutates the caller’s rootState
            leaf_state = self._env.clone_history_root_state(root_state.stats)
            node.reward = 0.0
            node.state_cost = root.state_cost
        else:
            # Reconstruct parent state once, apply this node's incoming action
            parent = node.parent
            parent_state = self._replay_to_node(parent)
            parent_cost = parent.state_cost

            # Apply incoming action to parent_state to reach leaf_state
            action = node.parent_action
            assert action is not None
            if parent.player == "adversary":
                leaf_state = self._env.apply_adversary_action_only(parent_state, action, inplace=True)
            else:
                leaf_state = self._env.apply_controller_action_only(parent_state, action, inplace=True)

            child_cost = self._state_cost(leaf_state)
            node.reward = self._transition_reward(parent_cost, child_cost)
            node.sim_time = leaf_state.simulator._time
            node.state_cost = child_cost

        # Expansion + NN evaluation at leaf
        # only skip trivial forced chains
        forced_logs: List[Tuple[MCTSNode, Dict[str, Any], int, int]] = []   


        node, leaf_state = self._advance_through_single_child_chain(
            node, leaf_state, search_path, forced_step_logs=forced_logs
        )

        ## LOGGGING FOR FORCED STEPS ##
        for forced_node, forced_snap, forced_n_valid, forced_unique in forced_logs:
            parent_multi = (forced_node.parent is None) or (len(forced_node.parent.children) > 1)

            if forced_n_valid == 0:
                forced_phase = "terminal"
            elif forced_n_valid == 1:
                forced_phase = "single-child" if parent_multi else "trivial-single-child"
            else:
                forced_phase = "multiple-child" if parent_multi else "trivial-multiple-child"

            self._iter_logger.log_expand(
                game_id=game_id,
                root_id=root_id,
                sim_iteration=sim_iteration,
                root_depth=root.depth,
                root_node_id=root.node_id,
                root_player=root.player,

                node_depth=forced_node.depth,
                parent_node_id=(forced_node.parent.node_id if forced_node.parent else None),
                node_id=forced_node.node_id,
                player_acted_to_create_this_node=(forced_node.parent.player if forced_node.parent else "root_no_parent"),
                player_to_act=forced_node.player,

                action_index=forced_node.parent_action_index,
                action_repr=(repr(forced_node.parent_action) if forced_node.parent_action else ""),
                prior=float(getattr(forced_node, "prior", 0.0)),
                reward=float(getattr(forced_node, "reward", 0.0)),

                nn_called=False,
                num_valid_actions=int(forced_n_valid),
                unique_actions=int(forced_unique),
                nn_value_controller=None,

                objective_cost=float(getattr(forced_node, "state_cost", 0.0)),
                state_snapshot=forced_snap,
                phase=f"forced_step:{forced_phase}",
            )

        # now expand/eval at the first “meaningful” node
        nn_value, nn_called, num_valid = self._expand_node(node, leaf_state, dnn_model)


        parent_multi = (node.parent is None) or (len(node.parent.children) > 1)

        if num_valid == 0:
            phase = "terminal"
        elif num_valid == 1:
            phase = "single-child" if parent_multi else "trivial-single-child"
        else:
            phase = "multiple-child" if parent_multi else "trivial-multiple-child"

        # Logging for this iteration
        snap = self._env.describe_state(leaf_state)
        unique_actions = len(node.children)  # after expansion, children dict keys are the unique/canonical actions

        adv_deadlines = "{}"
        if node.parent_action is not None and isinstance(node.parent_action, AdversaryAction):
            adv_deadlines = self._adversary_prefill_deadlines_by_id_json(leaf_state, node.parent_action)

        self._iter_logger.log_expand(
                game_id=game_id,
                root_id=root_id,
                sim_iteration=sim_iteration,
                root_depth=root.depth,
                root_node_id=root.node_id,
                root_player=root.player,

                node_depth=node.depth,
                parent_node_id=(node.parent.node_id if node.parent else None),
                node_id=node.node_id,
                player_acted_to_create_this_node=(node.parent.player if node.parent else "root_no_parent"),
                player_to_act=node.player,
                

                action_index=node.parent_action_index,
                action_repr=(repr(node.parent_action) if node.parent_action else ""),
                prior=float(node.prior),
                reward=float(node.reward),

                nn_called=nn_called,
                num_valid_actions=num_valid,
                unique_actions=unique_actions,
                nn_value_controller=nn_value,   # None for forced/terminal

                objective_cost=float(node.state_cost),
                adversary_prefill_deadlines_by_id_json=adv_deadlines,
                state_snapshot=snap,
                phase=phase,
            )

        # Backup
        self.backpropagate(search_path, nn_value, min_max_stats)

        # LOGGING MCTS:
        # DUMP_EVERY = 1  # or 50/100 to reduce overhead
        # if sim_iteration is not None and (sim_iteration % DUMP_EVERY == 0):
        #     dump_tree_snapshot_csv(
        #         out_path=f"simulator_output/mcts_dnn_logs/tree_snap_iter_{sim_iteration}.csv",
        #         game_id=int(game_id),
        #         root_id=int(root_id),
        #         sim_iteration=int(sim_iteration),
        #         root_node=root,
        #         minmax_min=min_max_stats.minimum,
        #         minmax_max=min_max_stats.maximum,
        #         max_nodes=None,   # or set a cap while debugging
        #     )
 


    def search_dnn(self, dnn_model: Any, rootState: VidurMCTSState, root_player: str, iterations: int , * , game_id: int , root_id: int , root_depth: int) -> Tuple[str, List[Union[AdversaryAction, ControllerAction]]]:
        self._history_root_state = rootState
        # TODO: Code review the state cost logic for root and child from the Codex here
        self._root = MCTSNode(player=root_player, node_id=self._next_node_id(), depth=0, sim_time=rootState.simulator._time, state_cost = self._state_cost(rootState))
        self._history_root_node = self._root
        self._env.snapshot_history_root(self._history_root_state)
        self._did_root_infer_debug = False

        min_max_stats = MinMaxStats()

        # --- Root evaluation (NOT counted as a simulation) ---
        # This seeds priors / nn cache so PUCT is defined, but we do NOT let it bias MCTS value.
        _ignored_value, _ignored_called, _ignored_num_valid = self._expand_node(self._root, rootState, dnn_model)
        # IMPORTANT: no self.backpropagate(...) here


        for sim_iteration in range(int(iterations)):
            self.run_one_simulation(
                self._root,
                rootState,
                dnn_model,
                min_max_stats,
                game_id=game_id,
                root_id=root_id,
                sim_iteration=sim_iteration,
            )

        next_player = "controller" if self._root.player == "adversary" else "adversary"
        action_space = [child.parent_action for child in self._root.children.values()]

        if (
            self._root.nn_value_controller is None
            or self._root.nn_priors is None
            or self._root.nn_valid_mask is None
        ):
            return next_player, action_space

        ## Filling Logging details for root node after MCTS Search is done :

        model_v, model_prior, mask, mcts_prior, best_idx = self._compute_root_log_payload_from_cache(
            root=self._root,
        )
       
        ### DEBUGGING LOG ###
        best_action = None
        if best_idx is not None and self._root is not None:
            best_child = self._root.children.get(best_idx)
            if best_child is not None:
                best_action = best_child.parent_action

        ## CREATING LOGS FOR THE ROOT :

        best_action_repr = repr(best_action) if best_action is not None else ""
        best_action_json = ""

        try:
            if isinstance(best_action, ControllerAction):
                best_action_json = json.dumps(
                    {
                        "type": "controller",
                        "token_budget": int(best_action.token_budget),
                        "selected_request_ids": [int(x) for x in (best_action.selected_request_ids or [])],
                        "token_allocations": {str(int(k)): int(v) for k, v in (best_action.token_allocations or {}).items()},
                        "prefill_allocations": {str(int(k)): int(v) for k, v in (best_action.prefill_allocations or {}).items()},
                        "decode_allocations": {str(int(k)): int(v) for k, v in (best_action.decode_allocations or {}).items()},
                        "heuristic": best_action.heuristic,
                        "strategy": best_action.strategy,
                    },
                    ensure_ascii=False,
                )

            elif isinstance(best_action, AdversaryAction):
                best_action_json = json.dumps(
                    {
                        "type": "adversary",
                        "requests": [
                            {
                                "prefill_tokens": int(r.prefill_tokens),
                                "decode_tokens": int(r.decode_tokens),
                                "prefill_slo": float(r.prefill_slo),
                                "decode_slo": float(r.decode_slo),
                            }
                            for r in (best_action.requests or [])
                        ],
                        "stop_decode_ids": [int(x) for x in (best_action.stop_decode_ids or [])],
                    },
                    ensure_ascii=False,
                )
        except Exception:
            best_action_json = ""


        # print(
        #     "Model value at root:", model_v,
        #     "best_idx:", best_idx,
        #     "best_action:", repr(best_action),
        #     "best_action_mcts_prob:", mcts_prior[best_idx] if best_idx < len(mcts_prior) else None,
        #     "best_action_model_prob:", model_prior[best_idx] if best_idx < len(model_prior) else None,
        # )


        # Only log roots where we actually queried the NN (i.e., real branching).
        # Forced-move roots (<=1 valid action) intentionally skip NN + skip root logging.
        

        self._root_logger.log_root(
            game_id=game_id,
            root_id=root_id,
            root_depth=root_depth, # TODO: remove this feild here from the logger, not needed
            root_node_id=self._root.node_id,
            root_player=self._root.player,
            num_simulations=iterations,
            model_root_value_controller=model_v,
            model_root_prior=model_prior,
            valid_action_mask=mask,
            mcts_root_value_controller= self._root.mean_value(),
            mcts_root_prior=mcts_prior,
            best_action_index=best_idx,
            best_action_repr=best_action_repr,
            best_action_json=best_action_json,
        )


        return next_player, action_space

    # CORE PHASE ENDS HERE #

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #
    
    def _compute_root_log_payload_from_cache(
        self,
        root: MCTSNode,
    ) -> tuple[float, list[float], list[bool], list[float], int]:
        if root.nn_value_controller is None or root.nn_priors is None or root.nn_valid_mask is None:
            raise RuntimeError("Root NN eval missing; evaluate+cache root before logging")

        model_v = root.nn_value_controller
        model_prior = root.nn_priors
        mask = root.nn_valid_mask
        num_actions = len(mask)

        # MCTS prior from visits (expanded back to full action space)
        visit_mass = [0.0] * num_actions

        if root.canonical_to_action_aliases:
            for canon, aliases in root.canonical_to_action_aliases.items():
                child = root.children.get(canon)
                v = float(child.visits) if child is not None else 0.0
                share = v / float(len(aliases)) if aliases else 0.0
                for aidx in aliases:
                    if 0 <= aidx < num_actions:
                        visit_mass[aidx] = share
        else:
            for action_index, child in root.children.items():
                if 0 <= action_index < num_actions:
                    visit_mass[action_index] = float(child.visits)

        total = sum(visit_mass[i] for i in range(num_actions) if mask[i])
        if total > 0:
            mcts_prior = [(visit_mass[i] / total) if mask[i] else 0.0 for i in range(num_actions)]
        else:
            valid_idx = [i for i, ok in enumerate(mask) if ok]
            mcts_prior = [0.0] * num_actions
            if valid_idx:
                p = 1.0 / len(valid_idx)
                for i in valid_idx:
                    mcts_prior[i] = p

        best_idx = max((i for i, ok in enumerate(mask) if ok), key=lambda i: visit_mass[i], default=0)

        
        return model_v, model_prior, mask, mcts_prior, best_idx

    def _adversary_prefill_deadlines_by_id_json(
        self, state: VidurMCTSState, action: AdversaryAction
    ) -> str:
        if action is None or not getattr(action, "requests", None):
            return "{}"

        # After apply_adversary_action_only, arrivals are drained, so requests exist in scheduler.
        lookup = self._env._build_request_lookup(state.simulator)
        if not lookup:
            return "{}"

        k = len(action.requests)
        ids = sorted(lookup.keys())[-k:]  # assumes newly created requests have largest ids

        out: Dict[int, float] = {}
        for rid in ids:
            req = lookup.get(rid)
            if req is None:
                continue
            queued_at = float(getattr(req, "queued_at", getattr(req, "arrived_at", 0.0)) or 0.0)
            slo = float(getattr(req, "prefill_slo_time", 0.0) or 0.0)
            out[int(rid)] = queued_at + slo  # absolute deadline time
        return json.dumps(out)


    def _next_node_id(self) -> int:
        node_id = self._node_counter
        self._node_counter += 1
        return node_id
