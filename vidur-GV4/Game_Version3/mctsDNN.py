# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

import torch

from .launch_mcts_job import MCTSExploreConfig
from .game_types import AdversaryAction, ControllerAction
from .environment import VidurMCTSState
from .virtual_environment import VirtualVidurMCTSEnvironment
from .DNN import infer as dnn_infer
 

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
    nn_priors_after_threshold: list[float] | None = None  # length = full action space size, normalized priors actually used by search
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


@dataclass
class DepthOneSearchResult:
    root_node_id: int
    root_player: str
    next_player: str
    best_action_index: Optional[int]
    best_action: Optional[Union[AdversaryAction, ControllerAction]]
    best_action_value: float
    action_values: List[float]   # full action space, invalid = -inf
    valid_mask: List[bool]
    used_bootstrap: bool

    ## For only logging purpose for now 
    best_reward: float = 0.0
    best_discount: float = 1.0
    best_bootstrap: float = 0.0
    best_child_cost: float = 0.0
    best_child_time: float = 0.0



class VidurMCTS:
    """Depth-1 value search used by the current GV3 pipeline."""

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
        del log_path, tree_log_path, logger_flush_every, complete_log
        self._env = env
        self._cfg = explore_cfg
        self._rng = rng or random.Random(0)
        self._verbose = bool(verbose)
        self._scratch_state: Optional[VidurMCTSState] = None
        self._node_counter = 0
        self._root: Optional[MCTSNode] = None

        step_tokens = int(getattr(getattr(self._env, "_constraints", None), "interval_request_size", 128) or 128)
        try:
            slowdown = float(getattr(self._env._constraints, "prefill_slowdown", 1.0) or 1.0)
            if slowdown <= 0:
                slowdown = 1.0

            scaled = float(self._env._prefill_profile.lookup(step_tokens) or 1e-9)
            self._prefill_step_time = scaled / slowdown  # undo slowdown for discount calibration

        except Exception:
            self._prefill_step_tokens = step_tokens
            self._prefill_step_time = 0.015725797204323228  # safe fallback


        # Discount denominator: config override or profile-derived default
        denom_override = getattr(self._cfg, "discount_time_denominator_sec", None)
        if denom_override is None:
            self._discount_time_denom = max(float(getattr(self, "_prefill_step_time", 0.015725797204323228)), 1e-9)
        else:
            self._discount_time_denom = max(float(denom_override), 1e-9)




    # TODO: call this function in either SelfPlay or AlphaZero.py class
    def close(self) -> None:
        self.clear_search_state(drop_scratch=True)

    # ------------------------------------------------------------------ #
    # Core phases
    # ------------------------------------------------------------------ #


    def _state_cost(self, state: VidurMCTSState) -> float:
        violations, total_lateness = self._env.evaluate_objective(state)
        return float(violations) + float(total_lateness)

    # TODO : Use this function in the search_dnn_depth1 & compose_q_from_state
    def _seed_node_runtime_from_state(self, node: MCTSNode, state: VidurMCTSState) -> None:
        node.state_cost = float(self._state_cost(state))
        node.sim_time = float(getattr(state.simulator, "_time", 0.0))

    # TODO: pass this via the config and experiment with different reward shaping functions & Change the value reward knee and reward max penalty from config to bigger values
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
        # alpha = float(getattr(self._cfg, "reward_tail_alpha", 1.0 / headroom))
        alpha_cfg = getattr(self._cfg, "reward_tail_alpha", None)
        alpha = float(alpha_cfg) if alpha_cfg is not None else (1.0 / headroom)


        if delta <= knee:
            penalty = delta
        else:
            penalty = knee + headroom * math.tanh(alpha * (delta - knee))

        return -penalty



    def _time_discount(self, t_child_time: float, t_parent_branch: float) -> float:
        """
        discount = gamma ^ ((t_child_time - t_parent_branch) / prefill_time(step_tokens))
        """
        gamma = float(getattr(self._cfg, "discount_factor", 0.995))
        denom = max(float(getattr(self, "_discount_time_denom", getattr(self, "_prefill_step_time", 0.015725797204323228))), 1e-9)
        dt = max(0.0, float(t_child_time) - float(t_parent_branch))
        return gamma ** (dt / denom)


    """
        This function is there to prevent going over the same actions for a given state multiple times during expansion.
    """
    def _controller_action_key(self, action: ControllerAction) -> tuple:
        alloc = tuple(sorted((int(rid), int(tok)) for rid, tok in (action.token_allocations or {}).items()))

        # Prefer explicit eviction ids if sampler/env attaches them later.
        evicted_ids = tuple(sorted(int(x) for x in (getattr(action, "_evicted_request_ids", ()) or ())))

        # Fallback eviction identity from strategy/mapping.
        strategy = str(action.strategy or "")
        evict_rule = strategy.split("|", 1)[1] if strategy.startswith("GV2|") else strategy
        mapping = tuple(int(x) for x in (action.mapping or ()))

        if evicted_ids:
            return (alloc, evicted_ids)
        return (alloc, evict_rule, mapping)


    def _adversary_action_key(self, action: AdversaryAction) -> tuple:
        request_specs = tuple(
            sorted(
                (
                    int(req.prefill_tokens),
                    int(req.decode_tokens),
                    float(req.prefill_slo),
                    float(req.decode_slo),
                )
                for req in (action.requests or [])
            )
        )
        stop_decode_ids = tuple(sorted(set(int(x) for x in (action.stop_decode_ids or []))))
        return (request_specs, stop_decode_ids)


    """
        These functions are used for the case when Adversary missed its turn due to the controller's action which progressed the time :
    """

    def _is_missed_adv_tick(self, state: VidurMCTSState) -> Tuple[bool, Optional[float]]:
        # GV2/3-specific hook (safe no-op for other envs)
        next_tick_fn = getattr(self._env, "_v2_next_adv_tick", None)
        if not callable(next_tick_fn):
            return False, None
        try:
            next_tick = float(next_tick_fn(state))
        except Exception:
            return False, None
        sim_time = float(state.simulator._time)
        return sim_time > (next_tick + 1e-9), next_tick



    def _decision_state(self, node: MCTSNode, state: VidurMCTSState, player: str) -> Tuple[VidurMCTSState, Optional[float]]:

        # Root: use current state
        if node.parent is None:
            return state, None

        # Only adversary decisions need missed-tick correction
        if player != "adversary":
            return state, None

        # Only when coming from controller step
        if node.parent.player != "controller":
            return state, None
        # If no missed adversary tick, use current state
        flag, next_tick = self._is_missed_adv_tick(state)
        if not flag:
            return state, None

        # Finding the true source for the Adv missed action i.e. either thru controller's meaningful action or just doing a FF decode .
        get_src = getattr(self._env, "_v2_missed_adv_source", None)
        miss_src = int(get_src(state)) if callable(get_src) else 1

        # 2 => missed due FF/decode/no-request progression: use post-trivial (current) state
        if miss_src == 2:
            return state, next_tick

        # 1 (or fallback) => missed due controller batch: use parent snapshot

        # Build inference state from parent snapshot (pre-controller-action context)
        clone_fn = getattr(self._env, "clone_state_from_snapshot", None)
        if not callable(clone_fn):
            return state, None
        if node.parent.cached_sim_snapshot is None or node.parent.cached_stats is None:
            return state, None

        # try:
        #     return clone_fn(node.parent.cached_sim_snapshot, node.parent.cached_stats), next_tick
        # except Exception:
        #     return state, None

        ## TODO: this is causing sim time to become tick , so model is also interpretted as that. We may want to consider this change later
        try:
            decision_state = clone_fn(node.parent.cached_sim_snapshot, node.parent.cached_stats)

            # Missed-tick adversary decision must be sampled AT the missed tick boundary,
            # not at the older parent snapshot time.
            if next_tick is not None:
                t_dec = float(decision_state.simulator._time)
                t_tick = float(next_tick)
                if t_dec + 1e-9 < t_tick:
                    decision_state.simulator._set_time(t_tick)

            return decision_state, next_tick
        except Exception:
            return state, None

    def _adversary_replay_forbidden_stop_ids(
        self,
        node: MCTSNode,
        decision_state: VidurMCTSState,
        real_state: VidurMCTSState,
    ) -> Optional[Set[int]]:
        if node.player != "adversary" or node.parent is None or node.parent.player != "controller":
            return None

        flag, _ = self._is_missed_adv_tick(real_state)
        if not flag:
            return None

        get_src = getattr(self._env, "_v2_missed_adv_source", None)
        miss_src = int(get_src(real_state)) if callable(get_src) else 1
        if miss_src != 1:
            return None

        try:
            # IDs visible in replay basis (pre-controller)
            replay_ids = set(
                int(k) for k in self._env._build_request_lookup(
                    decision_state.simulator, state=decision_state
                ).keys()
            )
            # IDs truly alive after controller action (post-controller real state)
            post_ids = set(
                int(k) for k in self._env._build_request_lookup(
                    real_state.simulator, state=real_state
                ).keys()
            )
            blocked = replay_ids - post_ids
            return blocked if blocked else None
        except Exception:
            return None



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
        # TODO : Do we really need these two modes ?
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
        # TODO : Do we really need these 2 modes or not ?
        sim = state.simulator
        if hasattr(sim, "snapshot_state_fast"):
            node.cached_sim_snapshot = sim.snapshot_state_fast()
        else:
            node.cached_sim_snapshot = sim.snapshot_state()
        node.cached_stats = state.stats.clone()


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
            debug=do_debug,
            debug_out_path="simulator_output/mcts_dnn_logs/infer_root_debug.txt" if do_debug else None,
        )

        # self._perf["nn_build_inputs"] += time.perf_counter() - t_nn

        if do_debug:
            self._did_root_infer_debug = True

        # Override action mask from env (convert list -> tensor [1, A])
        if action_mask is not None:
            inputs = replace(inputs, action_mask=action_mask)


        # t_nn = time.perf_counter()
        value, priors = dnn_model.infer_from_inputs(inputs, player, device=device)
        # self._perf["nn_infer"] += time.perf_counter() - t_nn
        return float(value), list(priors)


    def _actions_and_mask(self, state: VidurMCTSState, player: str, *, forbidden_stop_ids: Optional[Set[int]] = None) -> Tuple[List[Optional[object]], torch.Tensor]:
        if player == "controller":
            actions_by_index, mask = self._env.sample_controller_actions(state)
        else:
            actions_by_index, mask = self._env.sample_adversary_actions(state, forbidden_stop_ids=forbidden_stop_ids)

        # mask must be 1D bool, length == len(actions_by_index)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        else:
            mask = mask.to(dtype=torch.bool)

        return actions_by_index, mask

    # TODO: Let the environment produce the action space with fixed indexing & MORE IMPORTANTLY, we need to have cost and violations here added for each child node created!
    def _next_player(self, player: str) -> str:
        return "controller" if player == "adversary" else "adversary"


    # TODO: This is the duplication of the existing function above,  _store_node_snapshot, remove one mode of it
    def _snapshot_state_and_stats(self, state: VidurMCTSState) -> tuple[Any, Any]:
        sim = state.simulator
        if hasattr(sim, "snapshot_state_fast"):
            snap = sim.snapshot_state_fast()
        else:
            snap = sim.snapshot_state()
        stats = state.stats.clone()
        return snap, stats

    def _bootstrap_value_from_model(
        self,
        dnn_model: Any,
        child_state: VidurMCTSState,
        next_player: str,
        model_version: int,
        use_model_bootstrap: bool,
    ) -> float:
        if (not bool(use_model_bootstrap)) or int(model_version) <= 0:
            return 0.0
        v, _ = self._nn_value_and_priors(
            dnn_model=dnn_model,
            state=child_state,
            player=next_player,
            action_mask=None,   # compatible with current infer path
        )
        return float(v)


    def _compose_q_from_state(
        self,
        *,
        leaf_state: VidurMCTSState,
        parent_cost: float,
        parent_time: float,
        next_player: str,
        dnn_model: Any,
        model_version: int,
        use_model_bootstrap: bool,
    ) -> tuple[float, float, float, float, float, float]:
        leaf_cost = float(self._state_cost(leaf_state))
        reward = float(self._transition_reward(parent_cost, leaf_cost))
        leaf_time = float(leaf_state.simulator._time)
        discount_time = getattr(leaf_state.stats, "transition_discount_time", None)
        if discount_time is None:
            discount_time = leaf_time

        discount = float(self._time_discount(float(discount_time), parent_time))

        bootstrap = self._bootstrap_value_from_model(
            dnn_model=dnn_model,
            child_state=leaf_state,
            next_player=next_player,
            model_version=int(model_version),
            use_model_bootstrap=bool(use_model_bootstrap),
        )

        q = float(reward + discount * bootstrap)
        return q, reward, discount, bootstrap, leaf_cost, leaf_time


    def _canonicalize_action_indices(
        self,
        *,
        player: str,
        actions_by_index: list[Union[AdversaryAction, ControllerAction] | None],
        valid_indices: list[int],
    ) -> tuple[Dict[int, int], Dict[int, List[int]], List[int]]:
        alias_to_canon: Dict[int, int] = {}
        canon_to_aliases: Dict[int, List[int]] = {}
        canonical_indices: List[int] = []

        if player == "controller":
            sig_to_canon: Dict[tuple, int] = {}
            for idx in valid_indices:
                act = actions_by_index[idx]
                if not isinstance(act, ControllerAction):
                    alias_to_canon[idx] = idx
                    canon_to_aliases.setdefault(idx, []).append(idx)
                    canonical_indices.append(idx)
                    continue

                sig = self._controller_action_key(act)
                canon = sig_to_canon.get(sig)
                if canon is None:
                    sig_to_canon[sig] = idx
                    alias_to_canon[idx] = idx
                    canon_to_aliases[idx] = [idx]
                    canonical_indices.append(idx)
                else:
                    alias_to_canon[idx] = canon
                    canon_to_aliases[canon].append(idx)
        else:
            sig_to_canon: Dict[tuple, int] = {}
            for idx in valid_indices:
                act = actions_by_index[idx]
                if not isinstance(act, AdversaryAction):
                    alias_to_canon[idx] = idx
                    canon_to_aliases.setdefault(idx, []).append(idx)
                    canonical_indices.append(idx)
                    continue

                sig = self._adversary_action_key(act)
                canon = sig_to_canon.get(sig)
                if canon is None:
                    sig_to_canon[sig] = idx
                    alias_to_canon[idx] = idx
                    canon_to_aliases[idx] = [idx]
                    canonical_indices.append(idx)
                else:
                    alias_to_canon[idx] = canon
                    canon_to_aliases[canon].append(idx)

        return alias_to_canon, canon_to_aliases, canonical_indices

    def _evaluate_adversary_action_q_two_step(
        self,
        *,
        decision_snapshot: Any,
        decision_stats: Any,
        parent_cost: float,
        parent_time: float,
        adv_action: AdversaryAction,
        dnn_model: Any,
        model_version: int,
        use_model_bootstrap: bool,
    ) -> tuple[float, float, float, float, float, float]:
        # Step 1: apply adversary action. This should not advance time.
        adv_child_state = self._scratch_restore(decision_snapshot, decision_stats)
        adv_child_state = self._env.apply_adversary_action_only(
            adv_child_state,
            adv_action,
            inplace=True,
        )

        # Step 2: expand controller actions on the adversary-produced state.
        controller_actions_by_index, controller_mask_t = self._actions_and_mask(
            adv_child_state,
            "controller",
            forbidden_stop_ids=None,
        )
        controller_valid_mask = [bool(x) for x in controller_mask_t.tolist()]
        controller_valid_indices = [
            i
            for i, ok in enumerate(controller_valid_mask)
            if ok and controller_actions_by_index[i] is not None
        ]

        # Fallback: if controller has no valid action, just value the current controller state.
        if not controller_valid_indices:
            return self._compose_q_from_state(
                leaf_state=adv_child_state,
                parent_cost=parent_cost,
                parent_time=parent_time,
                next_player="controller",
                dnn_model=dnn_model,
                model_version=int(model_version),
                use_model_bootstrap=bool(use_model_bootstrap),
            )

        _, _, controller_canonical_indices = self._canonicalize_action_indices(
            player="controller",
            actions_by_index=controller_actions_by_index,
            valid_indices=controller_valid_indices,
        )

        adv_child_snapshot, adv_child_stats = self._snapshot_state_and_stats(adv_child_state)

        best_controller_idx: int | None = None
        best_controller_q: float | None = None
        best_controller_tuple: tuple[float, float, float, float, float, float] | None = None

        for cidx in controller_canonical_indices:
            ctrl_action = controller_actions_by_index[cidx]
            if ctrl_action is None:
                continue

            controller_leaf_state = self._scratch_restore(adv_child_snapshot, adv_child_stats)
            controller_leaf_state = self._env.apply_controller_action_only(
                controller_leaf_state,
                ctrl_action,
                inplace=True,
                fast_forward=False,
            )

            q_tuple = self._compose_q_from_state(
                leaf_state=controller_leaf_state,
                parent_cost=parent_cost,
                parent_time=parent_time,
                next_player="adversary",
                dnn_model=dnn_model,
                model_version=int(model_version),
                use_model_bootstrap=bool(use_model_bootstrap),
            )

            q = float(q_tuple[0])

            # Controller is maximizing controller-valued Q.
            if (
                best_controller_q is None
                or q > best_controller_q
                or (q == best_controller_q and (best_controller_idx is None or int(cidx) < int(best_controller_idx)))
            ):
                best_controller_idx = int(cidx)
                best_controller_q = q
                best_controller_tuple = q_tuple

        if best_controller_tuple is None:
            return self._compose_q_from_state(
                leaf_state=adv_child_state,
                parent_cost=parent_cost,
                parent_time=parent_time,
                next_player="controller",
                dnn_model=dnn_model,
                model_version=int(model_version),
                use_model_bootstrap=bool(use_model_bootstrap),
            )

        return best_controller_tuple


  
    def _evaluate_depth1_action_q(
        self,
        *,
        decision_snapshot: Any,
        decision_stats: Any,
        parent_player: str,
        parent_cost: float,
        parent_time: float,
        action: Union[AdversaryAction, ControllerAction],
        dnn_model: Any,
        model_version: int,
        use_model_bootstrap: bool,
    ) -> tuple[float, float, float, float, float, float]:
        state = self._scratch_restore(decision_snapshot, decision_stats)

        if parent_player == "adversary":
            state = self._env.apply_adversary_action_only(state, action, inplace=True)
        else:
            state = self._env.apply_controller_action_only(state, action, inplace=True, fast_forward=False)

        next_player = self._next_player(parent_player)
        return self._compose_q_from_state(
            leaf_state=state,
            parent_cost=parent_cost,
            parent_time=parent_time,
            next_player=next_player,
            dnn_model=dnn_model,
            model_version=int(model_version),
            use_model_bootstrap=bool(use_model_bootstrap),
        )



    
    # ------------- 


    def _search_dnn_depth1(
        self,
        dnn_model: Any,
        rootState: VidurMCTSState,
        root_player: str,
        *,
        model_version: int,
        use_model_bootstrap: bool,
        game_id: int,
        root_id: int,
        root_node_id_override: int | None,
        root_depth: int,
    ) -> DepthOneSearchResult:
        self.clear_search_state(drop_scratch=False)

        if root_node_id_override is None:
            root_node_id = self._next_node_id()
        else:
            root_node_id = int(root_node_id_override)
            self._node_counter = max(self._node_counter, root_node_id + 1)

        root = MCTSNode(
            player=str(root_player),
            node_id=int(root_node_id),
            depth=int(root_depth),
            parent=None,
        )
        self._root = root

        # align to decision state exactly like current pipeline
        decision_state, _ = self._decision_state(root, rootState, root_player)
        root_cost = float(self._state_cost(decision_state))
        root_time = float(decision_state.simulator._time)

        root.state_cost = root_cost
        root.sim_time = root_time
        self._store_node_snapshot(root, decision_state)

        forbidden_stop_ids = self._adversary_replay_forbidden_stop_ids(
            root, decision_state, rootState
        )
        actions_by_index, mask_t = self._actions_and_mask(
            decision_state,
            root.player,
            forbidden_stop_ids=forbidden_stop_ids,
        )
        valid_mask = [bool(x) for x in mask_t.tolist()]
        n_actions = len(actions_by_index)
        action_values = [float("-inf")] * n_actions

        valid_indices = [i for i, ok in enumerate(valid_mask) if ok and actions_by_index[i] is not None]
        root.num_valid_actions = int(len(valid_indices))
        next_player = self._next_player(root.player)
        used_bootstrap = bool(use_model_bootstrap)

        if not valid_indices:
            root.visits = 1
            root.value_sum = 0.0
            root.nn_value_controller = 0.0
            root.nn_valid_mask = list(valid_mask)
            root.nn_priors = [0.0] * n_actions
            root.nn_priors_after_threshold = [0.0] * n_actions
            return DepthOneSearchResult(
                root_node_id=int(root.node_id),
                root_player=str(root.player),
                next_player=str(next_player),
                best_action_index=None,
                best_action=None,
                best_action_value=0.0,
                action_values=action_values,
                valid_mask=valid_mask,
                used_bootstrap=used_bootstrap,

                best_reward=0.0,
                best_discount=1.0,
                best_bootstrap=0.0,
                best_child_cost=float(root_cost),
                best_child_time=float(root_time),

            )

        # controller dedup (evaluate canonical only, then fan out to aliases)

        alias_to_canon, canon_to_aliases, canonical_indices = self._canonicalize_action_indices(
            player=root.player,
            actions_by_index=actions_by_index,
            valid_indices=valid_indices,
        )


        root.action_alias_to_canonical = dict(alias_to_canon)
        root.canonical_to_action_aliases = {k: list(v) for k, v in canon_to_aliases.items()}

        # decision_snapshot, decision_stats = self._snapshot_state_and_stats(decision_state)

        decision_snapshot = root.cached_sim_snapshot
        decision_stats = root.cached_stats
        if decision_snapshot is None or decision_stats is None:
            decision_snapshot, decision_stats = self._snapshot_state_and_stats(decision_state)

        canonical_q: Dict[int, float] = {}
        canonical_q_tuple: Dict[int, tuple[float, float, float, float, float, float]] = {}

        for cidx in canonical_indices:
            action = actions_by_index[cidx]
            if action is None:
                continue

            if root.player == "adversary":
                q_tuple = self._evaluate_adversary_action_q_two_step(
                    decision_snapshot=decision_snapshot,
                    decision_stats=decision_stats,
                    parent_cost=root_cost,
                    parent_time=root_time,
                    adv_action=action,
                    dnn_model=dnn_model,
                    model_version=int(model_version),
                    use_model_bootstrap=bool(use_model_bootstrap),
                )
            else:
                q_tuple = self._evaluate_depth1_action_q(
                    decision_snapshot=decision_snapshot,
                    decision_stats=decision_stats,
                    parent_player=root.player,
                    parent_cost=root_cost,
                    parent_time=root_time,
                    action=action,
                    dnn_model=dnn_model,
                    model_version=int(model_version),
                    use_model_bootstrap=bool(use_model_bootstrap),
                )

            q, _reward, _disc, _boot, _child_cost, _child_time = q_tuple
            canonical_q[cidx] = float(q)
            canonical_q_tuple[cidx] = q_tuple


        for alias_idx, canon_idx in alias_to_canon.items():
            missing_q = float("inf") if root.player == "adversary" else float("-inf")
            q = canonical_q.get(canon_idx, missing_q)
            action_values[alias_idx] = float(q)

        best_idx = self._select_depth1_best_action_index(
            root_player=str(root.player),
            valid_indices=valid_indices,
            action_values=action_values,
        )

        best_canon_idx = alias_to_canon.get(int(best_idx), int(best_idx))
        best_tuple = canonical_q_tuple.get(best_canon_idx)
        if best_tuple is None:
            best_reward = 0.0
            best_discount = 1.0
            best_bootstrap = 0.0
            best_child_cost = root_cost
            best_child_time = root_time
        else:
            _q, best_reward, best_discount, best_bootstrap, best_child_cost, best_child_time = best_tuple


        best_val = float(action_values[best_idx])

        root.visits = 1
        root.value_sum = float(best_val)
        root.nn_value_controller = float(best_val)
        root.nn_valid_mask = list(valid_mask)
        # root.nn_priors = list(one_hot)
        # root.nn_priors_after_threshold = list(one_hot)

        return DepthOneSearchResult(
            root_node_id=int(root.node_id),
            root_player=str(root.player),
            next_player=str(next_player),
            best_action_index=int(best_idx),
            best_action=actions_by_index[best_idx],
            best_action_value=float(best_val),
            action_values=action_values,
            valid_mask=valid_mask,
            used_bootstrap=used_bootstrap,
            best_reward=float(best_reward),
            best_discount=float(best_discount),
            best_bootstrap=float(best_bootstrap),
            best_child_cost=float(best_child_cost),
            best_child_time=float(best_child_time),
        )


    @staticmethod
    def _select_depth1_best_action_index(
        *,
        root_player: str,
        valid_indices: Sequence[int],
        action_values: Sequence[float],
    ) -> int:
        valid = [int(i) for i in valid_indices]
        if not valid:
            raise ValueError("_select_depth1_best_action_index requires at least one valid action")

        player = str(root_player)
        if player == "controller":
            # Q is controller-valued. Controller should maximize it.
            return max(valid, key=lambda i: (float(action_values[i]), -int(i)))
        if player == "adversary":
            # Adversary chooses the action with minimum controller value.
            return min(valid, key=lambda i: (float(action_values[i]), int(i)))
        raise ValueError(f"unknown root_player={root_player!r}")

    def search_dnn(
        self,
        dnn_model: Any,
        rootState: VidurMCTSState,
        root_player: str,
        *,
        game_id: int,
        root_id: int,
        root_node_id_override: int | None,
        root_depth: int,
        model_version: int = 0,
        use_model_bootstrap: bool | None = None,
        one_step_value_mode: bool = True,
        root_phase: str = "train_root",
        cycle_label: str = "",
    ) -> DepthOneSearchResult:
        del root_phase, cycle_label
        if not bool(one_step_value_mode):
            raise RuntimeError("GV3 mctsDNN now supports only one_step_value_mode=True")
        if use_model_bootstrap is None:
            use_model_bootstrap = bool(int(model_version) > 0)
        return self._search_dnn_depth1(
            dnn_model=dnn_model,
            rootState=rootState,
            root_player=root_player,
            model_version=int(model_version),
            use_model_bootstrap=bool(use_model_bootstrap),
            game_id=int(game_id),
            root_id=int(root_id),
            root_node_id_override=root_node_id_override,
            root_depth=int(root_depth),
        )


    def _next_node_id(self) -> int:
        node_id = self._node_counter
        self._node_counter += 1
        return node_id


    # add near other methods in VidurMCTS
    def clear_search_state(self, *, drop_scratch: bool = True) -> None:
        root = self._root
        self._root = None
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
            n.parent_action_index = None
            n.cached_sim_snapshot = None
            n.cached_stats = None
            n.nn_priors = None
            n.nn_priors_after_threshold = None
            n.nn_valid_mask = None
            n.last_expand_children_created.clear()
            n.last_expand_dedup.clear()
            n.action_alias_to_canonical.clear()
            n.canonical_to_action_aliases.clear()
