# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict

import time 

# Misc Imports for debugging the tree and logging
from .DNN.misc import dump_tree_snapshot_csv


# Imports for excuting DNN model during MCTS
import torch 
from .DNN import infer as dnn_infer
try:
    from . import mcts_native as _mcts_native
except Exception:  # pragma: no cover - optional native runtime
    _mcts_native = None

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
    nn_priors_after_threshold: list[float] | None = None  # length = full action space size, nornalized after min_prior enforcement
    nn_valid_mask: list[bool] | None = None         # length = full action space size
    num_valid_actions: int = 0  # last computed valid-actions count at this node

    # Controller action deduplication (only meaningful when player == "controller")
    action_alias_to_canonical: Dict[int, int] = field(default_factory=dict)      # alias_idx -> canonical_idx
    canonical_to_action_aliases: Dict[int, List[int]] = field(default_factory=dict)  # canonical_idx -> [idxs...]

    # Optional cached state to accelerate _replay_to_node()
    cached_sim_snapshot: Any | None = None
    cached_stats: Any | None = None

    # Debug fields for PUCT trace logging
    last_expand_children_created: List[list] = field(default_factory=list)
    last_expand_dedup: List[list] = field(default_factory=list)


    def expanded(self) -> bool:
        return len(self.children) > 0

    def mean_value(self) -> float:
        return (self.value_sum / self.visits) if self.visits > 0 else 0.0

    # TODO: Remove this function if not needed
    # def is_fully_expanded(self) -> bool:
    #     return len(self.untried_actions) == 0



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
        complete_log: bool = False,
        ## A flag to enable logs for normalised version of the inputs given to the model at root inference
        _did_root_infer_debug = False
        # history_depth: int = 0,          # NEW
    ) -> None:
        self._env = env
        self._cfg = explore_cfg
        self._rng = rng or random.Random(0)
        self._verbose = verbose
        self._scratch_state: Optional[VidurMCTSState] = None
        self._node_counter = 0
        self._root: Optional[MCTSNode] = None
        self._history_root_state: Optional[VidurMCTSState] = None
        self._history_root_node: Optional[MCTSNode] = None


        self._use_fast_sim_snapshot = True

        # Iteration-level logging is intentionally opt-in via existing `verbose` flag.
        # This keeps native/Python search hot paths unaffected unless explicitly enabled.
        self._iter_log_enabled = bool(verbose)
        self._iter_complete_log = bool(complete_log)
        self._iter_logger = (
            DNNMCTSIterationLogger(log_path, flush_every=logger_flush_every)
            if (self._iter_log_enabled and log_path is not None)
            else None
        )

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
        self.clear_search_state(drop_scratch=True)

    # ------------------------------------------------------------------ #
    # Core phases
    # ------------------------------------------------------------------ #


    def _state_cost(self, state: VidurMCTSState) -> float:
        violations, avg_lateness = self._env.evaluate_objective(state)
        return float(violations) + float(avg_lateness)

    def _seed_node_runtime_from_state(self, node: MCTSNode, state: VidurMCTSState) -> None:
        node.state_cost = float(self._state_cost(state))
        node.sim_time = float(getattr(state.simulator, "_time", 0.0))

    def _prior_value_mode(self) -> str:
        """
        Mode switch for expansion priors/value:
        - "model": normal NN inference
        - "uniform": uniform priors over valid actions, bootstrap value = 0
        """
        mode = str(
            getattr(
                self._cfg,
                "prior_value_mode",
                getattr(self._cfg, "prior_mode", "model"),
            )
            or "model"
        ).strip().lower()
        if mode not in ("model", "uniform"):
            raise ValueError(f"Unsupported prior/value mode: {mode!r}")
        return mode

    def _get_prior_value_mode(self) -> str:
        return self._prior_value_mode()

    def _uniform_value_and_priors(
        self,
        actions_by_index: List[Optional[object]],
        mask: torch.Tensor,
    ) -> Tuple[float, List[float]]:
        """
        Uniform policy on valid actions, value=0 (controller perspective).
        """
        mask_list = (
            mask.to(dtype=torch.bool).tolist()
            if isinstance(mask, torch.Tensor)
            else [bool(x) for x in mask]
        )
        priors = [0.0] * len(actions_by_index)
        valid = [
            i
            for i, ok in enumerate(mask_list)
            if ok and i < len(actions_by_index) and actions_by_index[i] is not None
        ]
        if valid:
            p = 1.0 / float(len(valid))
            for i in valid:
                priors[i] = p
        return 0.0, priors

    # TODO: pass this via the config and experiment with different reward shaping functions
    def _transition_reward(self, parent_cost: float, child_cost: float) -> float:
        delta = max(0.0, float(child_cost) - float(parent_cost))

        knee = float(getattr(self._cfg, "reward_knee", 25.0))
        max_penalty = float(getattr(self._cfg, "reward_max_penalty", 40.0))

        if max_penalty <= knee:
            # fallback: pure linear
            return -min(delta, max_penalty)

        headroom = max_penalty - knee

        # alpha controls how fast we saturate after the knee.
        # alpha = 1/headroom makes slope continuous at the knee (linear slope=1).
        alpha = float(getattr(self._cfg, "reward_tail_alpha", 1.0 / headroom))

        if delta <= knee:
            penalty = delta
        else:
            penalty = knee + headroom * math.tanh(alpha * (delta - knee))

        return -penalty



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



    def _scratch_restore(self, snapshot: Any, stats_template: Any) -> VidurMCTSState:
        """
        Restore `snapshot` into a single reusable scratch VidurMCTSState.

        - Creates ONE Simulator the first time (via env.initial_state()).
        - Subsequent calls reuse the same Simulator object and just restore_state(...).
        - IMPORTANT: the returned state is a scratch buffer and will be overwritten
        by the next _scratch_restore call. Don’t keep multiple restored states alive.
        """
        if snapshot is None:
            raise ValueError("_scratch_restore: snapshot is None")

        # Create the scratch simulator exactly once (per search_dnn, you should reset self._scratch_state=None).
        if self._scratch_state is None:
            self._scratch_state = self._env.initial_state()  # creates Simulator(...) once

        # Reuse the same simulator instance; restore in-place.
        sim = self._scratch_state.simulator
        if (
            isinstance(snapshot, dict)
            and snapshot.get("__mode__") == "mcts_fast"
            and hasattr(sim, "restore_state_fast")
        ):
            sim.restore_state_fast(snapshot)
        else:
            sim.restore_state(snapshot)

        # Stats must correspond to that snapshot prefix.
        clone_fn = getattr(stats_template, "clone", None)
        if not callable(clone_fn):
            raise TypeError("_scratch_restore: stats_template must have a .clone() method")
        self._scratch_state.stats = clone_fn()

        return self._scratch_state


    def _store_node_snapshot(self, node: MCTSNode, state: VidurMCTSState) -> None:
        # node.cached_sim_snapshot = state.simulator.snapshot_state()
        # node.cached_stats = state.stats.clone()
        sim = state.simulator
        if hasattr(sim, "snapshot_state_fast"):
            node.cached_sim_snapshot = sim.snapshot_state_fast()
        else:
            node.cached_sim_snapshot = sim.snapshot_state()
        node.cached_stats = state.stats.clone()


    def _restore_state_for_node(self, node: MCTSNode) -> VidurMCTSState:
        """
        Restores scratch simulator to `node` state.
        If `node` has no snapshot yet, it materializes it by restoring parent and applying the parent_action once.
        """
        # Fast path: node already has snapshot
        # t = time.perf_counter()
        if node.cached_sim_snapshot is not None and node.cached_stats is not None:
            return self._scratch_restore(node.cached_sim_snapshot, node.cached_stats)

        parent = node.parent
        if parent is None:
            raise RuntimeError("Root node has no cached snapshot/stats")

        # Restore parent first (recursive; typically 1 hop in practice)
        state = self._restore_state_for_node(parent)

        action = node.parent_action
        if action is None:
            raise RuntimeError(f"Missing parent_action for node_id={node.node_id}")

        parent_cost = float(parent.state_cost)
        # self._perf["recursive_restore"] += time.perf_counter() - t

        # Apply parent->child transition on scratch state
        if parent.player == "adversary":
            # t = time.perf_counter()
            state = self._env.apply_adversary_action_only(state, action, inplace=True)
            # self._perf["adv_apply_actions"] += time.perf_counter() - t
        else:
            # t = time.perf_counter()
            state = self._env.apply_controller_action_only(state, action, inplace=True)
            # self._perf["ctrl_apply_actions"] += time.perf_counter() - t

        # Fill edge reward/cost/time now that we have the true child state
        # t = time.perf_counter()
        child_cost = float(self._state_cost(state))
        node.reward = float(self._transition_reward(parent_cost, child_cost))
        self._seed_node_runtime_from_state(node, state)

        # Snapshot this node so future rollouts can jump here
        self._store_node_snapshot(node, state)
        # self._perf["cost_&_node_snap"] += time.perf_counter() - t
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

        # t_nn = time.perf_counter()
        inputs = dnn_infer.build_model_inputs(
            state,
            player,
            device,
            max_prefill_tokens=int(getattr(self._env._constraints, "max_request_tokens", 3072) or 3072),
            prefill_slowdown=float(getattr(self._env._constraints, "prefill_slowdown", 3.0) or 3.0),
            debug=do_debug,
            debug_out_path="simulator_output/mcts_dnn_logs/infer_root_debug.txt" if do_debug else None,
        )
        # self._perf["nn_build_inputs"] += time.perf_counter() - t_nn

        if do_debug:
            self._did_root_infer_debug = True

        # Override action mask from env (convert list -> tensor [1, A])
        if action_mask is not None:
            
            inputs = type(inputs)(
                req_features=inputs.req_features,
                global_features=inputs.global_features,
                req_mask=inputs.req_mask,
                action_mask=action_mask,
            )

        # t_nn = time.perf_counter()
        value, priors = dnn_model.infer_from_inputs(inputs, player, device=device)
        # self._perf["nn_infer"] += time.perf_counter() - t_nn
        return float(value), list(priors)


    def _maybe_add_root_dirichlet_noise(
        self,
        root: MCTSNode,
        *,
        nn_called: bool,
        num_valid_actions: int,
    ) -> None:
        """
        Apply AlphaZero-style root Dirichlet noise once:
        p' = (1 - eps) * p + eps * Dir(alpha)

        Applied only when enabled in config and only for true branching roots.
        """
        if not bool(getattr(self._cfg, "root_dirichlet_noise_enabled", False)):
            return
        if not bool(nn_called):
            return
        if int(num_valid_actions) <= 1:
            return
        if not root.children:
            return

        alpha = float(getattr(self._cfg, "root_dirichlet_alpha", 0.3) or 0.0)
        eps = float(getattr(self._cfg, "root_dirichlet_epsilon", 0.25) or 0.0)

        if alpha <= 0.0 or eps <= 0.0:
            return
        eps = max(0.0, min(1.0, eps))

        child_indices = sorted(int(i) for i in root.children.keys())
        n = len(child_indices)
        if n <= 1:
            return

        # Sample Dir(alpha) via Gamma(alpha, 1) normalization, using MCTS RNG.
        noise_raw = [self._rng.gammavariate(alpha, 1.0) for _ in range(n)]
        s = float(sum(noise_raw))
        if s <= 1e-12:
            noise = [1.0 / float(n)] * n
        else:
            inv_s = 1.0 / s
            noise = [x * inv_s for x in noise_raw]

        mixed = {}
        for j, idx in enumerate(child_indices):
            p = max(0.0, float(root.children[idx].prior))
            mixed[idx] = (1.0 - eps) * p + eps * float(noise[j])

        z = float(sum(mixed.values()))
        if z <= 1e-12:
            u = 1.0 / float(n)
            for idx in child_indices:
                root.children[idx].prior = u
        else:
            inv_z = 1.0 / z
            for idx in child_indices:
                root.children[idx].prior = float(mixed[idx]) * inv_z

        # Optional: keep normalized prior log payload aligned with actual noisy root priors for debug consistency.
        if root.nn_priors_after_threshold is not None:
            a = len(root.nn_priors_after_threshold)
            noisy_full = [0.0] * a

            if root.canonical_to_action_aliases:
                for canon, aliases in root.canonical_to_action_aliases.items():
                    child = root.children.get(int(canon))
                    if child is None:
                        continue
                    alias_ids = [int(x) for x in aliases if 0 <= int(x) < a]
                    if not alias_ids:
                        continue
                    share = float(child.prior) / float(len(alias_ids))
                    for ai in alias_ids:
                        noisy_full[ai] = share
            else:
                for idx, child in root.children.items():
                    ii = int(idx)
                    if 0 <= ii < a:
                        noisy_full[ii] = float(child.prior)

            root.nn_priors_after_threshold = noisy_full





    def _apply_min_prior_threshold_dict(
        self,
        prior_by_idx: Dict[int, float],
        *,
        min_prior: float,
        eps: float = 1e-12,
        max_iters: int = 64,
    ) -> Dict[int, float]:
        """
        Enforce p_i >= min_prior for all keys in prior_by_idx, while keeping sum(p)=1.
        If min_prior is infeasible (min_prior * N >= 1), fall back to uniform over keys.

        Important: expects prior_by_idx to represent ONLY valid actions you want MCTS to consider
        (e.g., canonical controller children, or adversary valid indices).
        """
        if not prior_by_idx:
            return {}

        keys = list(prior_by_idx.keys())
        n = len(keys)

        mp = float(min_prior or 0.0)
        if mp <= 0.0 or n <= 1:
            # just normalize and return
            s = float(sum(max(0.0, float(prior_by_idx[k])) for k in keys))
            if s <= eps:
                u = 1.0 / float(n)
                return {k: u for k in keys}
            return {k: max(0.0, float(prior_by_idx[k])) / s for k in keys}

        if mp * float(n) >= 1.0 - eps:
            u = 1.0 / float(n)
            return {k: u for k in keys}

        # normalize original (p0) -> used as proportional weights when subtracting mass
        p0 = {k: max(0.0, float(prior_by_idx[k])) for k in keys}
        s0 = float(sum(p0.values()))
        if s0 <= eps:
            u = 1.0 / float(n)
            p0 = {k: u for k in keys}
        else:
            inv = 1.0 / s0
            p0 = {k: v * inv for k, v in p0.items()}

        # clamp low probs to min_prior
        q = {k: (p0[k] if p0[k] >= mp else mp) for k in keys}

        # if we increased some entries, we must remove the excess from entries above mp
        for _ in range(max_iters):
            total = float(sum(q.values()))
            over = total - 1.0
            if abs(over) <= 1e-10:
                break

            if over > 0.0:
                adjustable = [k for k in keys if q[k] > mp + eps]
                if not adjustable:
                    u = 1.0 / float(n)
                    return {k: u for k in keys}

                wsum = float(sum(p0[k] for k in adjustable))
                if wsum <= eps:
                    u = 1.0 / float(n)
                    return {k: u for k in keys}

                # subtract overage proportional to original p0 mass (NN prior)
                for k in adjustable:
                    q[k] -= over * (p0[k] / wsum)

                # re-clamp any that dropped below mp, then loop again if needed
                for k in keys:
                    if q[k] < mp:
                        q[k] = mp

            else:
                # (rare) total < 1 due to numerical issues: add missing mass proportional to p0
                under = -over
                wsum = float(sum(p0.values()))
                for k in keys:
                    q[k] += under * (p0[k] / wsum)

        # final tiny correction (keep sum=1 without breaking the floor)
        total = float(sum(q.values()))
        if abs(total - 1.0) > 1e-8:
            # adjust the largest entry (must exist and should be >= mp)
            kmax = max(keys, key=lambda k: q[k])
            q[kmax] = max(mp, q[kmax] + (1.0 - total))

        return q



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
        # t = time.perf_counter()
        valid = [i for i, ok in enumerate(mask) if ok and actions_by_index[i] is not None] 
        num_valid_actions = len(valid)
        node.num_valid_actions = int(num_valid_actions)
        # self._perf["expand_valid_scan"] += time.perf_counter() - t

        # Forced / terminal cases: skip NN entirely
        if len(valid) == 0:
            # terminal: no children, leaf value estimate = 0
            # self._perf["expand_terminal_count"] += 1
            return 0.0, False , 0

        # parent_had_multiple_children = (node.parent is None) or (len(node.parent.children) > 1)
        # should_call_nn = bool(parent_has_multiple_children)

        if num_valid_actions == 1:
            # self._perf["expand_single_count"] += 1
            # t_single = time.perf_counter()
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
                )

            # self._perf["expand_single_path"] += time.perf_counter() - t_single
            # NEW: never call NN on single-child states
            return 0.0, False, 1

        # self._perf["expand_multi_count"] += 1

        # ------------------------
        # MULTIPLE-CHILD
        # ------------------------
        # Always call NN to get PRIORS for multi-child nodes
        # t = time.perf_counter()
        mode = self._prior_value_mode()
        if mode == "uniform":
            model_value, priors = self._uniform_value_and_priors(actions_by_index, mask)
            used_model = False
        else:
            model_value, priors = self._nn_value_and_priors(
                dnn_model, state, node.player, action_mask=mask.unsqueeze(0)
            )
            used_model = True
        # self._perf["expand_nn_total"] += time.perf_counter() - t
        node.nn_value_controller = float(model_value)
        node.nn_priors = list(priors)
        node.nn_valid_mask = [bool(x) for x in mask.tolist()]


        # --- NEW: apply min-prior threshold on ALL valid indices (not canonical) ---
        # t = time.perf_counter()
        if node.player == "controller":
            min_p = float(getattr(self._cfg, "controller_min_prior_threshold", 0.0) or 0.0)
        else:
            min_p = float(getattr(self._cfg, "adversary_min_prior_threshold", 0.0) or 0.0)

        valid_prior_by_idx = {
            int(i): (float(priors[i]) if 0 <= i < len(priors) else 0.0)
            for i in valid
        }

        # This normalizes across valid indices and applies the floor.
        # (If min_p==0 it still normalizes and cleans negatives.)
        norm_prior_by_idx = self._apply_min_prior_threshold_dict(
            valid_prior_by_idx,
            min_prior=float(min_p),
        )

        # store per-index thresholded prior vector for debugging/logging
        thr_vec = [0.0] * len(priors)
        for i, p in norm_prior_by_idx.items():
            if 0 <= int(i) < len(thr_vec):
                thr_vec[int(i)] = float(p)
        node.nn_priors_after_threshold = thr_vec
        # self._perf["expand_threshold"] += time.perf_counter() - t

        # But only USE the value for backup if parent had multiple children.
        # If parent had a single child -> "trivial-multiple-child": value passed upward is 0.0.
        # value_used_for_backup = float(model_value) if parent_had_multiple_children else 0.0
        value_used_for_backup = float(model_value) 
        next_player = "controller" if node.player == "adversary" else "adversary"

        "Creating only child nodes with unique valid actions"
        # -----------------------------
        # Controller action dedup (multi-child only)
        # -----------------------------
        # t = time.perf_counter() 
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
                        # psum += float(priors[aidx])
                        psum += float(norm_prior_by_idx.get(int(aidx), 0.0))
                canonical_prior[canon] = psum

        else:
            # No dedup: all valid indices are canonical
            canonical_indices = valid
            for idx in canonical_indices:
                # canonical_prior[idx] = float(priors[idx]) if 0 <= idx < len(priors) else 0.0
                canonical_prior[int(idx)] = float(norm_prior_by_idx.get(int(idx), 0.0))

        # self._perf["expand_dedup"] += time.perf_counter() - t

        # Create children ONLY for canonical indices
        # t = time.perf_counter()
        record_puct = getattr(self._iter_logger, "_path", None) is not None
        if record_puct:
            node.last_expand_children_created = []
            node.last_expand_dedup = []
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
            if record_puct:
                node.last_expand_children_created.append(
                    [int(node.children[idx].node_id), float(node.children[idx].prior), int(idx)]
                )

        if record_puct and node.player == "controller":
            for canon_idx, aliases in node.canonical_to_action_aliases.items():
                child = node.children.get(canon_idx)
                if child is None:
                    continue
                node.last_expand_dedup.append(
                    [
                        int(child.node_id),
                        float(canonical_prior.get(canon_idx, 0.0)),
                        [int(a) for a in aliases],
                    ]
                )
        # self._perf["expand_child_create"] += time.perf_counter() - t

        return value_used_for_backup, bool(used_model), num_valid_actions



    def ucb_score(self, parent: MCTSNode, child: MCTSNode, min_max_stats: MinMaxStats) -> float:
        pb_c_base = getattr(self._cfg, "pb_c_base", 5000)
        pb_c_init = getattr(self._cfg, "pb_c_init", 0.75)

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

    def _ucb_components(self, parent: MCTSNode, child: MCTSNode, min_max_stats: MinMaxStats) -> Dict[str, float]:
        pb_c_base = getattr(self._cfg, "pb_c_base", 5000)
        pb_c_init = getattr(self._cfg, "pb_c_init", 0.75)
        parent_is_branching = (getattr(parent, "num_valid_actions", 0) > 1) or (len(parent.children) > 1)

        pb_c = math.log((parent.visits + pb_c_base + 1.0) / pb_c_base) + pb_c_init
        pb_c *= math.sqrt(parent.visits + 1.0) / (child.visits + 1.0)

        prior_score = pb_c * float(child.prior)
        if child.visits > 0:
            disc = self._time_discount(float(child.sim_time), float(parent.sim_time)) if parent_is_branching else 1.0
            q_controller = float(child.reward) + disc * float(child.mean_value())
            q_norm = min_max_stats.normalize(q_controller)
            value_score = q_norm if parent.player == "controller" else -q_norm
        else:
            q_controller = 0.0
            q_norm = 0.0
            value_score = 0.0

        ucb = float(prior_score) + float(value_score)
        return {
            "pb_c": float(pb_c),
            "prior_score": float(prior_score),
            "value_score": float(value_score),
            "q_controller": float(q_controller),
            "q_norm": float(q_norm),
            "ucb": float(ucb),
        }

    def _build_selection_candidates(
        self,
        node: MCTSNode,
        min_max_stats: MinMaxStats,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for idx, child in node.children.items():
            comp = self._ucb_components(node, child, min_max_stats)
            candidates.append(
                {
                    "action_index": int(idx),
                    "child_node_id": int(child.node_id),
                    **comp,
                }
            )
        candidates.sort(key=lambda x: x["action_index"])
        return candidates

    def _select_child_with_trace(
        self,
        node: MCTSNode,
        min_max_stats: MinMaxStats,
        *,
        enable_trace: bool,
    ) -> Tuple[int, MCTSNode, Optional[Dict[str, Any]]]:
        trace = None
        candidates = self._build_selection_candidates(node, min_max_stats) if enable_trace else None
        action_index, child = self.select_child(node, min_max_stats)
        if enable_trace:
            trace = {
                "parent_node_id": int(node.node_id),
                "parent_player": str(node.player),
                "parent_visits": int(node.visits),
                "candidates": candidates,
                "chosen_action_index": int(action_index),
                "chosen_child_node_id": int(child.node_id),
            }
        return int(action_index), child, trace

    
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

            # inside _advance_through_single_child_chain
            # if n_valid != 1:  # branching/terminal checkpoint only
            #     self._cache_node_state(node, state)


            if n_valid != 1:
                if node.cached_sim_snapshot is None or node.cached_stats is None:
                    self._store_node_snapshot(node, state)
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
            # time0 = time.time()
            # Apply forced action
            if node.player == "adversary":
                state = self._env.apply_adversary_action_only(state, action, inplace=True)
            else:
                state = self._env.apply_controller_action_only(state, action, inplace=True)
            # time1 = time.time()
            # print(f"  [MCTS] apply_action took {time1 - time0:.6f} sec for hop {hops} player={node.player}")
            

            child_cost = self._state_cost(state)
            step_reward = self._transition_reward(parent_cost, child_cost)

            child.reward = float(step_reward)
            self._seed_node_runtime_from_state(child, state)
            # branching anchor doesn't change inside a forced chain
            # child.branch_anchor_time = node.branch_anchor_time

            search_path.append(child)
            node = child
            hops += 1

    def _ancestor_chain(self, node: MCTSNode) -> List[Dict[str, Any]]:
        out = []
        cur = node
        while cur is not None:
            out.append(
                {
                    "node_id": int(cur.node_id),
                    "parent_node_id": None if cur.parent is None else int(cur.parent.node_id),
                    "value_sum": float(cur.value_sum),
                    "visits": int(cur.visits),
                    "mean_value": float(cur.mean_value()),
                }
            )
            cur = cur.parent
        return out



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



    def run_one_simulation(
        self,
        root: MCTSNode,
        dnn_model: Any,
        min_max_stats: MinMaxStats,
        game_id: Optional[str] = None, root_id: Optional[str] = None, sim_iteration: Optional[int] = None
    ) -> None:
        # 1) Selection (TREE ONLY)
        node = root
        search_path: List[MCTSNode] = [node]
        selection_trace: List[Dict[str, Any]] = []
        iter_logging = getattr(self._iter_logger, "_path", None) is not None
        # t_phase = time.perf_counter()
        while node.expanded():
            if iter_logging:
                _, child, hop = self._select_child_with_trace(
                    node,
                    min_max_stats,
                    enable_trace=True,
                )
                if hop is not None:
                    selection_trace.append(hop)
                node = child
            else:
                _, node = self.select_child(node, min_max_stats)
            search_path.append(node)
        # self._perf["selection"] += time.perf_counter() - t_phase
        # 2) Restore scratch sim straight to the selected node state (or materialize it once)
     
        # t_phase = time.perf_counter()
        state = self._restore_state_for_node(node)
        # self._perf["restore"] += time.perf_counter() - t_phase
       

        forced_logs = [] if iter_logging else None

        # 3) Skip forced single-child chains so we end on branching/terminal
        # t_phase = time.perf_counter()
        leaf_node, leaf_state = self._advance_through_single_child_chain(node, state, search_path, forced_step_logs=forced_logs)
        # self._perf["forced_chain"] += time.perf_counter() - t_phase

        # (Safety) If we bailed out due to max_hops etc, don’t silently proceed
        if getattr(leaf_node, "num_valid_actions", 0) == 1:
            raise RuntimeError("advance_through_single_child_chain ended on single-child node; check max_hops/no-progress")

        # t0 = time.time()
        # 4) Expand + NN evaluation at final node
        # t_phase = time.perf_counter()
        leaf_value, _nn_called, _num_valid = self._expand_node(leaf_node, leaf_state, dnn_model)
        # self._perf["expand"] += time.perf_counter() - t_phase

        if iter_logging:
            parent_multi = (leaf_node.parent is None) or (len(leaf_node.parent.children) > 1)

            if _num_valid == 0:
                phase = "terminal"
            elif _num_valid == 1:
                phase = "single-child" if parent_multi else "trivial-single-child"
            else:
                phase = "multiple-child" if parent_multi else "trivial-multiple-child"

            # Logging for this iteration

            # 1. Forced Steps ##
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
                    model_prior_json="[]",
                    normalized_prior_json="[]",
                    reward=float(getattr(forced_node, "reward", 0.0)),
                    # action_cost_softcap=float(
                    #     max(0.0, min(1.0, -float(getattr(forced_node, "reward", 0.0))))
                    # ),


                    nn_called=False,
                    num_valid_actions=int(forced_n_valid),
                    unique_actions=int(forced_unique),
                    nn_value_controller=None,

                    objective_cost=float(getattr(forced_node, "state_cost", 0.0)),
                    state_snapshot=forced_snap,
                    phase=f"forced_step:{forced_phase}",
                )

            snap = self._env.describe_state(leaf_state)
            unique_actions = len(leaf_node.children)

            adv_deadlines = "{}"
            if isinstance(leaf_node.parent_action, AdversaryAction):
                adv_deadlines = self._adversary_prefill_deadlines_by_id_json(leaf_state, leaf_node.parent_action)

            nn_value_controller = float(leaf_node.nn_value_controller) if _nn_called else None

            model_prior_json = "[]"
            normalized_prior_json = "[]"
            if _nn_called and leaf_node.nn_priors is not None:
                model_prior_json = json.dumps(list(leaf_node.nn_priors))
            if _nn_called and leaf_node.nn_priors_after_threshold is not None:
                normalized_prior_json = json.dumps(list(leaf_node.nn_priors_after_threshold))


            self._iter_logger.log_expand(
                game_id=int(game_id),
                root_id=int(root_id),
                sim_iteration=int(sim_iteration),
                root_depth=root.depth,
                root_node_id=root.node_id,
                root_player=root.player,

                node_depth=leaf_node.depth,
                parent_node_id=(leaf_node.parent.node_id if leaf_node.parent else None),
                node_id=leaf_node.node_id,
                player_acted_to_create_this_node=(leaf_node.parent.player if leaf_node.parent else "root_no_parent"),
                player_to_act=leaf_node.player,

                action_index=leaf_node.parent_action_index,
                action_repr=(repr(leaf_node.parent_action) if leaf_node.parent_action else ""),
                prior=float(getattr(leaf_node, "prior", 0.0)),
                model_prior_json=model_prior_json,
                normalized_prior_json=normalized_prior_json,    

                reward=float(getattr(leaf_node, "reward", 0.0)),

                nn_called=bool(_nn_called),
                num_valid_actions=int(_num_valid),
                unique_actions=int(unique_actions),
                nn_value_controller=nn_value_controller,

                objective_cost=float(getattr(leaf_node, "state_cost", 0.0)),
                adversary_prefill_deadlines_by_id_json=adv_deadlines,
                state_snapshot=snap,
                phase=phase,
            )


        # 5) Backprop
        # t_phase = time.perf_counter()
        self.backpropagate(search_path, float(leaf_value), min_max_stats)
        # self._perf["backprop"] += time.perf_counter() - t_phase
        # self._perf["sim_count"] += 1

        if self._iter_logger is not None:
            self._iter_logger.puct_log_expand(
                game_id=int(game_id),
                root_id=int(root_id),
                sim_iteration=int(sim_iteration),
                root_node_id=int(root.node_id),
                parent_node_id=(leaf_node.parent.node_id if leaf_node.parent else None),
                node_id=int(leaf_node.node_id),
                player_acted_to_create_this_node=(leaf_node.parent.player if leaf_node.parent else "root_no_parent"),
                reward=float(getattr(leaf_node, "reward", 0.0)),
                node_dnn_value=(
                    None if leaf_node.nn_value_controller is None else float(leaf_node.nn_value_controller)
                ),
                children_created=list(getattr(leaf_node, "last_expand_children_created", [])),
                dedup_children=list(getattr(leaf_node, "last_expand_dedup", [])),
                ancestor_chain=self._ancestor_chain(leaf_node),
                selection_trace=selection_trace,
                minmax_min=float(min_max_stats.minimum),
                minmax_max=float(min_max_stats.maximum),
            )

        # # LOGGING MCTS:
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



    def _native_mcts_enabled(self) -> bool:
        return bool(getattr(self._cfg, "native_mcts_enabled", False)) and _mcts_native is not None


    def _native_infer_callback(self, dnn_model: Any):
        def _cb(state: VidurMCTSState, player: str, mask_list: List[bool]):
            if self._prior_value_mode() == "uniform":
                actions_by_index, mask = self._actions_and_mask(state, str(player))
                value, priors = self._uniform_value_and_priors(actions_by_index, mask)
                return float(value), [float(x) for x in priors]
            action_mask = torch.tensor(
                [bool(x) for x in mask_list],
                dtype=torch.bool,
            ).unsqueeze(0)
            value, priors = self._nn_value_and_priors(
                dnn_model,
                state,
                str(player),
                action_mask=action_mask,
            )
            return float(value), [float(x) for x in priors]

        return _cb


    def _apply_native_search_result(
        self,
        *,
        native_result: dict,
        root_node_id: int,
        root_depth: int,
        root_player: str,
    ) -> None:
        root = self._root
        if root is None:
            raise RuntimeError("Root must be initialized before applying native result")

        root.player = str(root_player)
        root.node_id = int(root_node_id)
        root.depth = int(root_depth)
        root.parent = None
        root.parent_action = None
        root.parent_action_index = None

        root.visits = int(native_result.get("root_visits", 0))
        root.value_sum = float(native_result.get("root_value_sum", 0.0))
        root.state_cost = float(native_result.get("root_state_cost", 0.0))
        root.sim_time = float(native_result.get("root_sim_time", 0.0))
        root.num_valid_actions = int(native_result.get("root_num_valid_actions", 0))

        nn_v = native_result.get("root_nn_value_controller", None)
        root.nn_value_controller = (None if nn_v is None else float(nn_v))
        nn_priors = native_result.get("root_nn_priors", None)
        root.nn_priors = (None if nn_priors is None else [float(x) for x in nn_priors])
        nn_priors_thr = native_result.get("root_nn_priors_after_threshold", None)
        root.nn_priors_after_threshold = (
            None if nn_priors_thr is None else [float(x) for x in nn_priors_thr]
        )
        nn_mask = native_result.get("root_nn_valid_mask", None)
        root.nn_valid_mask = (None if nn_mask is None else [bool(x) for x in nn_mask])

        root.action_alias_to_canonical = {
            int(k): int(v)
            for k, v in dict(native_result.get("action_alias_to_canonical", {})).items()
        }
        root.canonical_to_action_aliases = {
            int(k): [int(x) for x in list(v)]
            for k, v in dict(native_result.get("canonical_to_action_aliases", {})).items()
        }

        root.children.clear()
        max_seen_node_id = int(root.node_id)
        for row in list(native_result.get("children", [])):
            idx = int(row.get("index"))
            child_node_id = int(row.get("node_id", idx))
            max_seen_node_id = max(max_seen_node_id, child_node_id)
            child = MCTSNode(
                player=str(row.get("player", "controller" if root.player == "adversary" else "adversary")),
                node_id=child_node_id,
                depth=int(row.get("depth", root.depth + 1)),
                parent=root,
                parent_action=row.get("parent_action", None),
                parent_action_index=idx,
                prior=float(row.get("prior", 0.0)),
                reward=float(row.get("reward", 0.0)),
                visits=int(row.get("visits", 0)),
                value_sum=float(row.get("value_sum", 0.0)),
                sim_time=float(row.get("sim_time", 0.0)),
                state_cost=float(row.get("state_cost", 0.0)),
                num_valid_actions=int(row.get("num_valid_actions", 0)),
            )
            root.children[idx] = child

        self._node_counter = max(self._node_counter, max_seen_node_id + 1)


    def _search_dnn_native(
        self,
        *,
        dnn_model: Any,
        rootState: VidurMCTSState,
        root_player: str,
        iterations: int,
        root_node_id: int,
        root_depth: int,
        game_id: int = 0,
        root_id: int = 0,
    ) -> None:
        if _mcts_native is None:
            raise RuntimeError("native_mcts_enabled=True but vidur.mcts.mcts_native is unavailable")
        if not hasattr(_mcts_native, "search_mcts_dnn"):
            raise RuntimeError("mcts_native.search_mcts_dnn is missing; rebuild native module")

        reward_knee = float(getattr(self._cfg, "reward_knee", 25.0))
        reward_max_penalty = float(getattr(self._cfg, "reward_max_penalty", 40.0))
        headroom = max(reward_max_penalty - reward_knee, 1e-9)
        reward_tail_alpha = float(getattr(self._cfg, "reward_tail_alpha", 1.0 / headroom))

        iter_log_path = ""
        if bool(getattr(self, "_iter_log_enabled", False)):
            p = getattr(getattr(self, "_iter_logger", None), "_path", None)
            if p is not None:
                iter_log_path = str(p)

        common_kwargs = dict(
            env=self._env,
            root_state=rootState,
            root_player=str(root_player),
            iterations=int(iterations),
            uniform_prior_value=bool(self._prior_value_mode() == "uniform"),
            max_branching=int(getattr(self._cfg, "max_branching", 10)),
            controller_min_prior_threshold=float(
                getattr(self._cfg, "controller_min_prior_threshold", 0.0) or 0.0
            ),
            adversary_min_prior_threshold=float(
                getattr(self._cfg, "adversary_min_prior_threshold", 0.0) or 0.0
            ),
            root_dirichlet_noise_enabled=bool(
                getattr(self._cfg, "root_dirichlet_noise_enabled", False)
            ),
            root_dirichlet_alpha=float(getattr(self._cfg, "root_dirichlet_alpha", 0.3) or 0.3),
            root_dirichlet_epsilon=float(
                getattr(self._cfg, "root_dirichlet_epsilon", 0.25) or 0.25
            ),
            pb_c_base=float(getattr(self._cfg, "pb_c_base", 5000)),
            pb_c_init=float(getattr(self._cfg, "pb_c_init", 0.75)),
            discount_factor=float(getattr(self._cfg, "discount_factor", 0.98)),
            prefill_step_time=float(getattr(self, "_prefill_step_time", 0.0388862329)),
            reward_knee=reward_knee,
            reward_max_penalty=reward_max_penalty,
            reward_tail_alpha=reward_tail_alpha,
            seed=int(self._rng.randint(0, 2**31 - 1)),
            root_node_id=int(root_node_id),
            root_depth=int(root_depth),
            game_id=int(game_id),
            root_id=int(root_id),
            iter_log_path=str(iter_log_path),
            iter_complete_log=bool(getattr(self, "_iter_complete_log", True)),
        )

        native_ts_runtime = getattr(dnn_model, "_native_ts_runtime", None)
        native_ts_model_version = getattr(dnn_model, "_native_ts_model_version", None)
        # If a native TorchScript runtime is present, prefer full-native search.
        # This avoids the mixed callback bridge (C++ -> Python infer callback),
        # which has been unstable under multiprocess load.
        use_full_native_ts = bool(getattr(self._cfg, "torchscript_full_native_search", False))
        if (
            not use_full_native_ts
            and native_ts_runtime is not None
            and native_ts_model_version is not None
            and hasattr(_mcts_native, "search_mcts_dnn_torchscript")
        ):
            use_full_native_ts = True
            if not bool(getattr(self, "_auto_native_ts_full_search_logged", False)):
                print(
                    "[VidurMCTS] auto-enabled full native torchscript search "
                    "(avoids unstable callback bridge path)",
                    flush=True,
                )
                self._auto_native_ts_full_search_logged = True
        if (
            use_full_native_ts
            and
            native_ts_runtime is not None
            and native_ts_model_version is not None
            and hasattr(_mcts_native, "search_mcts_dnn_torchscript")
        ):
            native_search_fn = _mcts_native.search_mcts_dnn_torchscript
            try:
                if (
                    hasattr(_mcts_native, "NativeInferServiceRuntime")
                    and hasattr(_mcts_native, "search_mcts_dnn_torchscript_service")
                    and isinstance(native_ts_runtime, _mcts_native.NativeInferServiceRuntime)
                ):
                    native_search_fn = _mcts_native.search_mcts_dnn_torchscript_service
            except Exception:
                pass

            native_result = native_search_fn(
                infer_runtime=native_ts_runtime,
                model_version=int(native_ts_model_version),
                **common_kwargs,
            )
        else:
            native_result = _mcts_native.search_mcts_dnn(
                infer_cb=self._native_infer_callback(dnn_model),
                **common_kwargs,
            )
        if not isinstance(native_result, dict):
            raise RuntimeError("native search returned invalid payload")

        perf = native_result.get("perf")
        if isinstance(perf, dict) and bool(getattr(self._cfg, "native_profile", True)):
            try:
                print(
                    "[NATIVE_SEARCH_PERF] "
                    f"root_node_id={int(root_node_id)} player={str(root_player)} iters={int(iterations)} "
                    f"total={float(perf.get('total_sec', 0.0)):.3f}s "
                    f"state={float(perf.get('state_build_sec', 0.0)):.3f}s "
                    f"predictor={float(perf.get('predictor_load_sec', 0.0)):.3f}s "
                    f"root_expand={float(perf.get('root_expand_sec', 0.0)):.3f}s "
                    f"selection={float(perf.get('selection_sec', 0.0)):.3f}s "
                    f"restore={float(perf.get('restore_sec', 0.0)):.3f}s "
                    f"forced={float(perf.get('forced_chain_sec', 0.0)):.3f}s "
                    f"forced_mask={float(perf.get('forced_actions_mask_sec', 0.0)):.3f}s "
                    f"forced_apply={float(perf.get('forced_apply_sec', 0.0)):.3f}s "
                    f"leaf_expand={float(perf.get('leaf_expand_sec', 0.0)):.3f}s "
                    f"backprop={float(perf.get('backprop_sec', 0.0)):.3f}s "
                    f"actions_mask_total={float(perf.get('actions_mask_total_sec', 0.0)):.3f}s "
                    f"infer_total={float(perf.get('infer_total_sec', 0.0)):.3f}s "
                    f"infer_build={float(perf.get('infer_build_sec', 0.0)):.3f}s "
                    f"infer_forward={float(perf.get('infer_forward_sec', 0.0)):.3f}s "
                    f"expand_threshold={float(perf.get('expand_threshold_sec', 0.0)):.3f}s "
                    f"expand_dedup={float(perf.get('expand_dedup_sec', 0.0)):.3f}s "
                    f"expand_child_create={float(perf.get('expand_child_create_sec', 0.0)):.3f}s "
                    f"infer_calls={int(perf.get('infer_calls', 0))} "
                    f"expand_calls={int(perf.get('expand_calls', 0))} "
                    f"selection_steps={int(perf.get('selection_steps', 0))} "
                    f"forced_steps={int(perf.get('forced_steps', 0))} "
                    f"restore_missing={int(perf.get('restore_missing_nodes_total', 0))} "
                    f"nodes_capacity_grows={int(perf.get('nodes_capacity_grows', 0))} "
                    f"expand_ctrl_actions={int(perf.get('expand_controller_actions_total', 0))} "
                    f"expand_ctrl_alloc_pairs={int(perf.get('expand_controller_alloc_pairs_total', 0))}",
                    flush=True,
                )
            except Exception:
                pass

        self._apply_native_search_result(
            native_result=native_result,
            root_node_id=int(root_node_id),
            root_depth=int(root_depth),
            root_player=str(root_player),
        )



    def search_dnn(self, dnn_model: Any, rootState: VidurMCTSState, root_player: str, iterations: int , * , game_id: int , root_id: int, root_node_id_override: int | None , root_depth: int , root_phase: str = "train_root", cycle_label: str = "",) -> Tuple[str, List[Union[AdversaryAction, ControllerAction]]]:
        
        self.clear_search_state(drop_scratch=True)
        self._history_root_state = rootState
        if root_node_id_override is None:
            root_node_id = self._next_node_id()
        else:
            root_node_id = int(root_node_id_override)
            self._node_counter = max(self._node_counter, root_node_id + 1)  # so children get fresh ids

        self._root = MCTSNode(player=root_player, node_id=root_node_id, depth=int(root_depth), parent=None)
        
        # Store the exact root state on the root node
        # self._root.cached_sim_snapshot = rootState.simulator.snapshot_state()
        # self._root.cached_stats = rootState.stats.clone()

        sim = rootState.simulator
        if hasattr(sim, "snapshot_state_fast"):
            self._root.cached_sim_snapshot = sim.snapshot_state_fast()
        else:
            self._root.cached_sim_snapshot = sim.snapshot_state()
        self._root.cached_stats = rootState.stats.clone()
        # IMPORTANT: root reward/discount baseline must come from actual root state
        self._seed_node_runtime_from_state(self._root, rootState)

        # One scratch simulator per search_dnn
        self._scratch_state = None  # forces _scratch_restore to create it once
        self._did_root_infer_debug = False

        min_max_stats = MinMaxStats()

        # PROFILING :
        self._perf = {
            "restore": 0.0,
            "selection": 0.0,
            "forced_chain": 0.0,
            "expand": 0.0,
            "backprop": 0.0,
            "nn_build_inputs": 0.0,
            "nn_infer": 0.0,
            "sim_count": 0,

            "recursive_restore": 0.0,
            "adv_apply_actions" : 0.0,
            "ctrl_apply_actions" : 0.0,
            "cost_&_node_snap" : 0.0,

            "expand_actions_mask": 0.0,
            "expand_valid_scan": 0.0,
            "expand_single_path": 0.0,
            "expand_nn_total": 0.0,
            "expand_threshold": 0.0,
            "expand_dedup": 0.0,
            "expand_child_create": 0.0,

            # NEW: branch counters
            "expand_terminal_count": 0,
            "expand_single_count": 0,
            "expand_multi_count": 0,
        }
        # if hasattr(self._env, "_perf"):
        #     for k in self._env._perf:
        #         self._env._perf[k] = 0.0

        if self._native_mcts_enabled():
            self._search_dnn_native(
                dnn_model=dnn_model,
                rootState=rootState,
                root_player=root_player,
                iterations=int(iterations),
                root_node_id=int(root_node_id),
                root_depth=int(root_depth),
                game_id=int(game_id),
                root_id=int(root_id),
            )
        else:
            # --- Root evaluation (NOT counted as a simulation) ---
            # This seeds priors / nn cache so PUCT is defined, but we do NOT let it bias MCTS value.
            root_work = self._scratch_restore(self._root.cached_sim_snapshot, self._root.cached_stats)
            _ignored_value, _ignored_called, _ignored_num_valid = self._expand_node(self._root, root_work, dnn_model)

            self._maybe_add_root_dirichlet_noise(
                self._root,
                nn_called=bool(_ignored_called),
                num_valid_actions=int(_ignored_num_valid),
            )

            for sim_iteration in range(int(iterations)):
                self.run_one_simulation(
                    self._root,
                    dnn_model,
                    min_max_stats,
                    game_id=game_id,
                    root_id=root_id,
                    sim_iteration=sim_iteration,
                )

        next_player = "controller" if self._root.player == "adversary" else "adversary"
        action_space = [child.parent_action for child in self._root.children.values()]

        ## Filling Logging details for root node after MCTS Search is done :
        # Always log a root row. For forced/terminal roots (no NN call), use a
        # safe fallback payload so native/Python paths still emit CSV output.
        if (
            self._root.nn_value_controller is None
            or self._root.nn_priors is None
            or self._root.nn_valid_mask is None
        ):
            model_v, model_prior, normalized_prior, mask, mcts_prior, best_idx = (
                self._compute_root_log_payload_without_nn(
                    root=self._root,
                    root_state=rootState,
                )
            )
        else:
            model_v, model_prior, normalized_prior, mask, mcts_prior, best_idx = (
                self._compute_root_log_payload_from_cache(root=self._root)
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


        # Only log roots where we actually queried the NN (i.e., real branching).
        # Forced-move roots (<=1 valid action) intentionally skip NN + skip root logging.
        viol, lateness = self._env.evaluate_objective(rootState)
        total_cost = float(viol) + float(lateness)

        self._root_logger.log_root(
            game_id=game_id,
            root_id=root_id,
            root_depth=root_depth, # TODO: remove this feild here from the logger, not needed
            root_node_id=self._root.node_id,
            root_player=self._root.player,
            num_simulations=iterations,
            model_root_value_controller=model_v,
            model_root_prior=model_prior,
            normalized_root_prior=normalized_prior,
            valid_action_mask=mask,
            mcts_root_value_controller= self._root.mean_value(),
            mcts_root_prior=mcts_prior,
            best_action_index=best_idx,
            best_action_repr=best_action_repr,
            best_action_json=best_action_json,
            phase=str(root_phase),
            cycle_label=str(cycle_label),
            sim_time=float(rootState.simulator._time),
            slo_violations=int(viol),
            total_lateness=float(lateness),
            total_cost=float(total_cost),
        )


        # p = getattr(self, "_perf", None)
        # if p is not None:
        #     total_core = p["restore"] + p["forced_chain"] + p["expand"] + p["backprop"]
        #     print(
        #         "[MCTS_PERF] "
        #         f"root_id={root_id} iters={iterations} \n"
        #         f"total_core={total_core:.3f}s \n"
        #         f"restore={p['restore']:.3f}s \n selection={p['selection']:.3f}s \n forced_chain={p['forced_chain']:.3f}s \n "
        #         f"expand={p['expand']:.3f}s \n backprop={p['backprop']:.3f}s \n"
        #         f"sim_count={p.get('sim_count', 0)}\n"
        #     )

        #PRINTING PROFILING :
        # p = getattr(self, "_perf", None)
        # if p is not None:
        #     n = max(1, int(p.get("sim_count", 0)))
        #     total_core = p["restore"] + p["forced_chain"] + p["expand"] + p["backprop"]
        #     print(
        #         "[MCTS_PERF] "
        #         f"root_id={root_id} iters={iterations} "
        #         f"restore={p['restore']:.3f}s forced_chain={p['forced_chain']:.3f}s "
        #         f"expand={p['expand']:.3f}s backprop={p['backprop']:.3f}s "
        #         f"nn_build={p['nn_build_inputs']:.3f}s nn_infer={p['nn_infer']:.3f}s \n"
        #         f"core_total={total_core:.3f}s core_per_sim={total_core/n:.6f}s \n"
        #         f"adv_apply_actions={p['adv_apply_actions']:.3f}s ctrl_apply_actions={p['ctrl_apply_actions']:.3f}s\n"
        #         f"cost_&_node_snap={p['cost_&_node_snap']:.3f}s recursive_restore={p['recursive_restore']:.3f}s \n"
        #     )

        #     ex_total = max(float(p.get("expand", 0.0)), 1e-12)

        #     def pct(v: float, tot: float) -> float:
        #         return (100.0 * v / tot) if tot > 1e-12 else 0.0

        #     print(
        #         "[EXPAND_PERF]\n"
        #         f"  root_id={root_id} iters={n}\n"
        #         f"  expand_total={ex_total:.3f}s  expand_per_sim={ex_total/n:.6f}s\n"
        #         f"  actions_mask={p['expand_actions_mask']:.3f}s ({pct(p['expand_actions_mask'], ex_total):.1f}%)\n"
        #         f"  valid_scan={p['expand_valid_scan']:.3f}s ({pct(p['expand_valid_scan'], ex_total):.1f}%)\n"
        #         f"  nn_total={p['expand_nn_total']:.3f}s ({pct(p['expand_nn_total'], ex_total):.1f}%)\n"
        #         f"  threshold={p['expand_threshold']:.3f}s ({pct(p['expand_threshold'], ex_total):.1f}%)\n"
        #         f"  dedup={p['expand_dedup']:.3f}s ({pct(p['expand_dedup'], ex_total):.1f}%)\n"
        #         f"  child_create={p['expand_child_create']:.3f}s ({pct(p['expand_child_create'], ex_total):.1f}%)\n"
        #         f"  single_path={p['expand_single_path']:.3f}s ({pct(p['expand_single_path'], ex_total):.1f}%)\n"
        #         f"  counts(term/single/multi)="
        #         f"({int(p['expand_terminal_count'])}/{int(p['expand_single_count'])}/{int(p['expand_multi_count'])})"
        #     )

        #     ep = getattr(self._env, "_perf", None)
        #     if ep is not None:
        #         print(
        #             "[ENV_PERF] "
        #             f"ctrl_calls={int(ep.get('apply_ctrl_calls', 0))} "
        #             f"ctrl_total={ep.get('apply_ctrl_total', 0.0):.3f}s "
        #             f"lookup={ep.get('lookup', 0.0):.3f}s "
        #             f"alloc_norm={ep.get('alloc_norm', 0.0):.3f}s "
        #             f"predictor={ep.get('predictor', 0.0):.3f}s "
        #             f"stats={ep.get('stats', 0.0):.3f}s "
        #             f"rebuild={ep.get('rebuild', 0.0):.3f}s "
        #             f"ff_decode={ep.get('ff_decode', 0.0):.3f}s"
        #         )

        #     stats_total = ep.get("stats_total_internal", 0.0)
        #     stats_calls = max(1, int(ep.get("stats_calls", 0)))
        #     if stats_total > 0.0:
        #         def _pct(x: float) -> float:
        #             return (100.0 * x / stats_total) if stats_total > 1e-12 else 0.0

        #         print(
        #             "[ENV_STATS_PERF] "
        #             f"calls={stats_calls} total={stats_total:.3f}s per_call={stats_total/stats_calls:.6f}s "
        #             f"batch_unpack={ep.get('stats_batch_unpack',0.0):.3f}s({_pct(ep.get('stats_batch_unpack',0.0)):.1f}%) "
        #             f"ids_union={ep.get('stats_ids_union',0.0):.3f}s({_pct(ep.get('stats_ids_union',0.0)):.1f}%) "
        #             f"req_get={ep.get('stats_req_get',0.0):.3f}s({_pct(ep.get('stats_req_get',0.0)):.1f}%) "
        #             f"hooks={ep.get('stats_hooks',0.0):.3f}s({_pct(ep.get('stats_hooks',0.0)):.1f}%) "
        #             f"prefill={ep.get('stats_prefill',0.0):.3f}s({_pct(ep.get('stats_prefill',0.0)):.1f}%) "
        #             f"decode={ep.get('stats_decode',0.0):.3f}s({_pct(ep.get('stats_decode',0.0)):.1f}%) "
        #             f"violation={ep.get('stats_violation',0.0):.3f}s({_pct(ep.get('stats_violation',0.0)):.1f}%) "
        #             f"complete={ep.get('stats_complete',0.0):.3f}s({_pct(ep.get('stats_complete',0.0)):.1f}%)"
        #         )


        return next_player, action_space

    # CORE PHASE ENDS HERE #


    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #
    
    def _compute_root_log_payload_from_cache(
        self,
        root: MCTSNode,
    ) -> tuple[float, list[float], list[float], list[bool], list[float], Optional[int]]:
        if root.nn_value_controller is None or root.nn_priors is None or root.nn_valid_mask is None:
            raise RuntimeError("Root NN eval missing; evaluate+cache root before logging")

        model_v = root.nn_value_controller
        model_prior = root.nn_priors
        normalized_prior = root.nn_priors_after_threshold
        if normalized_prior is None:
            # fallback (shouldn't happen for multi-child roots)
            normalized_prior = model_prior

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

        # best_idx = max((i for i, ok in enumerate(mask) if ok), key=lambda i: visit_mass[i], default=0)
        # Pick best action by CANONICAL visits (children keys), not alias-split visit_mass
        valid_canons = [int(i) for i in root.children.keys() if 0 <= int(i) < num_actions and mask[int(i)]]
        if valid_canons:
            best_idx = max(valid_canons, key=lambda i: int(root.children[i].visits))
        else:
            best_idx = next((i for i, ok in enumerate(mask) if ok), 0)

        
        return model_v, model_prior, normalized_prior, mask, mcts_prior, best_idx

    def _compute_root_log_payload_without_nn(
        self,
        *,
        root: MCTSNode,
        root_state: VidurMCTSState,
    ) -> tuple[float, list[float], list[float], list[bool], list[float], Optional[int]]:
        if root.player == "controller":
            _, mask_raw = self._env.sample_controller_actions(root_state, self._cfg.max_branching)
        else:
            _, mask_raw = self._env.sample_adversary_actions(root_state, self._cfg.max_branching)

        if isinstance(mask_raw, torch.Tensor):
            mask = [bool(x) for x in mask_raw.to(dtype=torch.bool).tolist()]
        else:
            mask = [bool(x) for x in mask_raw]

        num_actions = len(mask)
        model_v = 0.0
        model_prior = [0.0] * num_actions
        normalized_prior = [0.0] * num_actions

        visit_mass = [0.0] * num_actions
        if root.canonical_to_action_aliases:
            for canon, aliases in root.canonical_to_action_aliases.items():
                child = root.children.get(int(canon))
                visits = float(child.visits) if child is not None else 0.0
                alias_ids = [int(aidx) for aidx in aliases if 0 <= int(aidx) < num_actions]
                if not alias_ids:
                    continue
                share = visits / float(len(alias_ids))
                for aidx in alias_ids:
                    visit_mass[aidx] = share
        else:
            for action_index, child in root.children.items():
                idx = int(action_index)
                if 0 <= idx < num_actions:
                    visit_mass[idx] = float(child.visits)

        total = sum(visit_mass[i] for i in range(num_actions) if mask[i])
        if total > 0:
            mcts_prior = [(visit_mass[i] / total) if mask[i] else 0.0 for i in range(num_actions)]
        else:
            valid_idx = [i for i, ok in enumerate(mask) if ok]
            mcts_prior = [0.0] * num_actions
            if valid_idx:
                p = 1.0 / float(len(valid_idx))
                for i in valid_idx:
                    mcts_prior[i] = p

        valid_canons = [
            int(i)
            for i in root.children.keys()
            if 0 <= int(i) < num_actions and mask[int(i)]
        ]
        if valid_canons:
            best_idx: Optional[int] = max(valid_canons, key=lambda i: int(root.children[i].visits))
        else:
            best_idx = next((i for i, ok in enumerate(mask) if ok), None)

        return model_v, model_prior, normalized_prior, mask, mcts_prior, best_idx

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


    # add near other methods in VidurMCTS
    def clear_search_state(self, *, drop_scratch: bool = True) -> None:
        root = self._root
        self._root = None
        self._history_root_state = None
        self._history_root_node = None
        self._did_root_infer_debug = False
        if drop_scratch:
            self._scratch_state = None

        if root is None:
            return

        stack = [root]
        seen = set()
        while stack:
            n = stack.pop()
            oid = id(n)
            if oid in seen:
                continue
            seen.add(oid)

            if n.children:
                stack.extend(n.children.values())
                n.children.clear()

            n.parent = None
            n.parent_action = None
            n.cached_sim_snapshot = None
            n.cached_stats = None
            n.action_alias_to_canonical.clear()
            n.canonical_to_action_aliases.clear()
