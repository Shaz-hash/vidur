from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from .environment import VidurMCTSState
from .game_types import AdversaryAction, ControllerAction
from .launch_mcts_job import MCTSExploreConfig
from .virtual_environment import VirtualVidurMCTSEnvironment
from .logger.mcts_logger import MCTSCsvLogger

def resolve_root_dirichlet_alpha(
    fixed_alpha: float,
    total_concentration: float,
    canonical_action_count: int,
) -> float:
    """Resolve per-action alpha while preserving fixed-alpha compatibility."""
    count = int(canonical_action_count)
    total = float(total_concentration)
    if total > 0.0 and count > 0:
        return total / float(count)
    return float(fixed_alpha)




class MCTSConfig:
    mcts_iterations: int = 1000
    exploration_constant: float = 1.0
    rng : random.Random = random.Random(42)
    log_flag: bool = False # Whether to log the search for debugging or not ?
    log_path: Optional[Path] = None
    tree_log_path: Optional[Path] = None

    uct_c: float = 1.4
    # New prior controls
    use_policy_prior: bool = False
    puct_c: float = 1.0
    policy_prior_temperature: float = 1.0
    prior_min_prob: float = 1e-8

    # Optional root noise, AlphaZero-style. Keep off for evaluation.
    root_dirichlet_alpha: float = 0.0
    root_dirichlet_epsilon: float = 0.0
    root_dirichlet_total_concentration: float = 0.0

    # Models. These should predict raw action scores/logits.
    controller_prior_model: Any = None
    adversary_prior_model: Any = None

    # Feature builders.
    # Signature:
    #   fn(state, player, actions_by_index, canonical_indices) -> List[List[float]]
    #
    # Controller should return 269D rows = 226D state + 43D action.
    # Adversary should return 233D rows = 226D state + 7D action.
    controller_prior_feature_fn: Any = None
    adversary_prior_feature_fn: Any = None



@dataclass
class MCTSSearchResult:
    root_node_id: int
    root_player: str
    next_player: str
    best_action_index: Optional[int]
    best_action: Optional[Union[AdversaryAction, ControllerAction]]
    best_action_value: float
    action_values: List[float]
    valid_mask: List[bool]
    used_bootstrap: bool = False


@dataclass
class Node:

    player: str # Player who will take action on this node, either "controller" or "adversary"
    node_id : int
    depth: int = 0
    parent: Optional[Node] = None
    parent_action: Optional[Union[AdversaryAction, ControllerAction]] = None # action from parent to this node
    parent_action_index: Optional[int] = None

    action_priors: Dict[int, float] = field(default_factory=dict) # canon action based only

    reward: float = 0.0 # reward/cost experienced from taking parent_action to reach this node from parent node
    visits: int = 0 # number of times the parent applied this action to reach this node
    value_sum: float = 0.0 # sum of the boostrap value estimated obtained down this node. {Does not include the reward from parent action}
    
    state_cost: float = 0.0 # the total SLO cost experienced at the state of this node 
    sim_time : float = 0.0 # the current sim time at the state of this node

    ## Needed to normalise the exploitation factor b/w 0 and 1
    min_value: float = float("inf") 
    max_value: float = float("-inf")

    # Edge stats from parent -> this node
    edge_discount: float = 1.0 # Discounting factor, need to be replaced by the actual discounting factor based on the time difference between parent and this node

    children: Dict[int, Node] = field(default_factory=dict) # children of this node action_index -> child node mapping
    actions_by_index: List[Optional[Union[AdversaryAction, ControllerAction]]] = field(default_factory=list) # actions from this node ,action_index -> action mapping for the children of this node
    valid_mask : List[bool] = field(default_factory=list) # valid mask for the actions from this node, action_index -> bool mapping for the children of this node
    untried_action_indices: List[int] = field(default_factory=list) # List of action indices which are still untried from this node

    action_alias_to_canonical: Dict[int, int] = field(default_factory=dict) # mapping from action index of an action alias to the canonical action index in the children of this node
    canonical_to_action_aliases: Dict[int, List[int]] = field(default_factory=dict) # mapping from canonical action index in the children of this node to the list of action indices of its aliases

    ## Optional/ Might remove them / Essentially the state of the simulator at this node:
    cached_sim_snapshot: Optional[Any] = None # cached simulation snapshot at the state of this node, can be used to speed up simulations from this node
    cached_stats: Any = None # cached stats at the state of this node, can be used to speed up simulations from this node


    def mean_value(self) -> float:
        if self.visits == 0.0 :
            return 0.0
        return self.value_sum / self.visits

    
    def expanded(self)-> bool:
        return len(self.actions_by_index) > 0




class VidurMCTS:

    def __init__(
        self,
        env: VirtualVidurMCTSEnvironment,
        mctsConfig: MCTSConfig,
    ) -> None:
        
        self._env = env
        self._mctsConfig = mctsConfig
        self._root: Optional[Node] = None
        self._scratch_state: Optional[VidurMCTSState] = None
        self._node_id_counter = 0 # to assign unique ids to the nodes in the tree
        self._logger: Optional[MCTSCsvLogger] = None

        if bool(getattr(mctsConfig, "log_flag", False)):
            if mctsConfig.log_path is None or mctsConfig.tree_log_path is None:
                raise ValueError("MCTS logging requires mctsConfig.log_path and mctsConfig.tree_log_path")

            self._logger = MCTSCsvLogger(
                root_log_path=mctsConfig.log_path,
                child_log_path=mctsConfig.tree_log_path,
                flush_every=1,
            )

    # Helper functions :
    def close(self):
        # clears the scratch state, should ideally free up the memory
        self._root = None
        self._node_id_counter = 0
        if self._logger is not None:
            self._logger.close()
            self._logger = None

    def clear_search_state(self, *, drop_scratch: bool = True) -> None:
        # clears the scratch state, should ideally free up the memory
        self._root = None
        if drop_scratch:
            self._scratch_state = None

    def get_next_node_id(self) -> int:
        self._node_id_counter += 1
        return self._node_id_counter

    def get_next_player(self, node: Node) -> str:
        return "controller" if node.player == "adversary" else "adversary"

    def get_state_cost (self,state: VidurMCTSState) -> float:
        # TODO: Remove this dependency from the environment, this is making it slow
        violations, lateness = self._env.evaluate_objective(state)
        return float(violations) + float(lateness)

    def get_transition_reward(self, parent_node: Any, child_node: Any) -> float:
        # Accept either Node objects or precomputed costs.  The tree expansion
        # path computes the child cost before building the child node.
        parent_cost = getattr(parent_node, "state_cost", parent_node)
        child_cost = getattr(child_node, "state_cost", child_node)
        return float(parent_cost) - float(child_cost)
    

    def time_discount(self, child_final_time: float, parent_time: float) -> float:
        " Provides us with the discounting factor based on how much time as progressed due to action from parent to child "
        gamma = float(getattr(self._mctsConfig, "discount_factor", 0.995))
        denom = float(getattr(self._mctsConfig, "_discount_time_denom", 0.015725797204323228))
        denom = max(denom, 1e-9)

        dt = max(0.0, float(child_final_time) - float(parent_time))
        discount_factor = gamma ** (dt / denom)

        if discount_factor < 1e-6 or discount_factor > 1.0:
            raise ValueError(
                f"Unreasonable discount factor computed: gamma={gamma}, "
                f"parent_time={parent_time}, child_final_time={child_final_time}, "
                f"dt={dt}, denom={denom}, discount_factor={discount_factor}"
            )

        return discount_factor


    def _bootstrap_value(
        self,
        dnn_model: Any,
        state: VidurMCTSState,
        player: str,
        *,
        model_version: int,
        use_model_bootstrap: bool,
    ) -> float:
        if (not bool(use_model_bootstrap)) or int(model_version) <= 0:
            return 0.0

        # Reuse the same inference helper pattern as mctsDNN.py.
        from .DNN import infer as dnn_infer

        device = next(dnn_model.parameters()).device if hasattr(dnn_model, "parameters") else "cpu"
        inputs = dnn_infer.build_model_inputs(state, player, device)
        # No priors needed, only value function's result
        value, _ = dnn_model.infer_from_inputs(inputs, player, device=device)
        return float(value)

    def _softmax(self, scores: Sequence[float], *, temperature: float) -> List[float]:
        # Temp = 1.0 is normal softmax, Temp < 1.0 sharpens the distribution, Temp > 1.0 flattens it.
        if not scores:
            return []

        temp = max(float(temperature), 1e-8)
        scaled = [float(x) / temp for x in scores]
        m = max(scaled)
        exps = [math.exp(x - m) for x in scaled]
        z = sum(exps)

        if z <= 0.0 or not math.isfinite(z):
            return [1.0 / len(scores)] * len(scores)

        return [float(x / z) for x in exps]

    def _uniform_priors(self, canonical_indices: Sequence[int]) -> Dict[int, float]:
        if not canonical_indices:
            return {}
        p = 1.0 / float(len(canonical_indices))
        return {int(idx): p for idx in canonical_indices}

    def _prior_model_and_feature_fn(self, player: str) -> tuple[Any, Any]:
        if player == "controller":
            return (
                getattr(self._mctsConfig, "controller_prior_model", None),
                getattr(self._mctsConfig, "controller_prior_feature_fn", None),
            )

        if player == "adversary":
            return (
                getattr(self._mctsConfig, "adversary_prior_model", None),
                getattr(self._mctsConfig, "adversary_prior_feature_fn", None),
            )

        return None, None

    def _predict_prior_scores(self, model: Any, feature_rows: Sequence[Sequence[float]]) -> List[float]:
        if model is None or not feature_rows:
            return []

        if hasattr(model, "predict"):
            pred = model.predict(feature_rows)
        elif callable(model):
            pred = model(feature_rows)
        else:
            raise TypeError(f"Unsupported prior model type: {type(model).__name__}")

        # sklearn returns numpy arrays; wrappers may return lists.
        return [float(x) for x in pred]

    def _apply_root_dirichlet_noise(self, priors: Dict[int, float], *, is_root: bool,) -> Dict[int, float]:
        ## Alpha > 1 gives a more uniform distribution, while alpha < 1 gives a more peaked distribution.
        ## Epsilon controls the weight of the noise, with higher values leading to more exploration.
        if not is_root or not priors:
            return priors

        eps = float(getattr(self._mctsConfig, "root_dirichlet_epsilon", 0.0) or 0.0)
        alpha = resolve_root_dirichlet_alpha(
            float(getattr(self._mctsConfig, "root_dirichlet_alpha", 0.0) or 0.0),
            float(getattr(self._mctsConfig, "root_dirichlet_total_concentration", 0.0) or 0.0),
            len(priors),
        )
        if eps <= 0.0 or alpha <= 0.0:
            return priors

        rng = getattr(self._mctsConfig, "rng", random)
        keys = list(priors.keys())
        noise = [rng.gammavariate(alpha, 1.0) for _ in keys]
        total = sum(noise)
        if total <= 0.0:
            return priors

        noise = [x / total for x in noise]
        mixed = {
            idx: (1.0 - eps) * float(priors[idx]) + eps * float(n)
            for idx, n in zip(keys, noise)
        }

        z = sum(mixed.values())
        if z <= 0.0:
            return self._uniform_priors(keys)

        return {idx: float(p / z) for idx, p in mixed.items()}


    def _compute_policy_priors(
        self,
        node: Node,
        state: VidurMCTSState,
        canonical_indices: Sequence[int],
    ) -> Dict[int, float]:

        canonical_indices = [int(x) for x in canonical_indices]
        if not canonical_indices:
            return {}

        if not bool(getattr(self._mctsConfig, "use_policy_prior", False)):
            return self._uniform_priors(canonical_indices)

        model, feature_fn = self._prior_model_and_feature_fn(node.player)
        if model is None or feature_fn is None:
            print(f"No model reference given ! Error!")
            return self._uniform_priors(canonical_indices)

        feature_rows = feature_fn(
            state,
            node.player,
            node.actions_by_index,
            canonical_indices,
        )

        if len(feature_rows) != len(canonical_indices):
            raise RuntimeError(
                f"prior feature rows mismatch: got {len(feature_rows)}, "
                f"expected {len(canonical_indices)}"
            )

        scores = self._predict_prior_scores(model, feature_rows)
        if len(scores) != len(canonical_indices):
            raise RuntimeError(
                f"prior score rows mismatch: got {len(scores)}, "
                f"expected {len(canonical_indices)}"
            )

        probs = self._softmax(
            scores,
            temperature=float(getattr(self._mctsConfig, "policy_prior_temperature", 1.0) or 1.0),
        )

        min_prob = float(getattr(self._mctsConfig, "prior_min_prob", 1e-8) or 1e-8)
        priors = {
            int(idx): max(min_prob, float(prob))
            for idx, prob in zip(canonical_indices, probs)
        }

        z = sum(priors.values())
        if z <= 0.0:
            print(f"Warning: zero total prior probability, check model outputs and temperature setting. Defaulting to uniform priors. Error!")
            priors = self._uniform_priors(canonical_indices)
        else:
            priors = {idx: float(p / z) for idx, p in priors.items()}

        priors = self._apply_root_dirichlet_noise(
            priors,
            is_root=(node.parent is None),
        )
        return priors
    
    def _update_node_bounds(self, node: Node, value: float) -> None:
        "Updates the min and max value node has seens to ensure our normalisation stays b/w 0 and 1 for the exploitation term in UCT"
        value = float(value)
        node.min_value = min(float(node.min_value), value)
        node.max_value = max(float(node.max_value), value)


    def snapshot_state_and_stats(self, node: Node, state: VidurMCTSState):
        # snapshot the state and stats at this node for potential reuse in future simulations from this node
        sim = state.simulator
        # TODO: Ensure that we can completely rely on the fast version ? Unnecessary code here 
        snap = sim.snapshot_state_fast() if hasattr(sim, "snapshot_state_fast") else sim.snapshot_state()
        return snap, state.stats.clone()
    
    def store_node_snapshot(self, node: Node, state: VidurMCTSState):
        """
            Stores the simulator's snapshot , stats, and initialises the state cost and time
        """
        node.cached_sim_snapshot, node.cached_stats = self.snapshot_state_and_stats(node, state)
        node.state_cost = self.get_state_cost(state)
        node.sim_time = float(state.simulator._time)

    def scratch_restore(self, snapshot: Any, stats: Any):
        """
            Restore the passed snapshot onto the scratch state of the MCTS
        """

        if self._scratch_state is None:
            self._scratch_state = self._env.initial_state()

        # TODO: Need to investigate this part of the code 
        sim = self._scratch_state.simulator
        if isinstance(snapshot, dict) and snapshot.get("__mode__") == "mcts_fast" and hasattr(sim, "restore_state_fast"):
            sim.restore_state_fast(snapshot)
        else:
            sim.restore_state(snapshot)

        self._scratch_state.stats = stats.clone()
        return self._scratch_state


    def get_actions_and_mask (self, state: VidurMCTSState, player: str) -> Tuple[List[Optional[Union[AdversaryAction, ControllerAction]]], torch.Tensor]:
        """
            Samples the actions from a given state based on the player specified. And masks the actions which are not valid. 
            Rule e.g. : Mask away the actions which involves ids for the adversary which are not allowed due to their eviction earlier
        """
        # TODO: Refresh over these functions
        if player == "controller":
            actions, mask = self._env.sample_controller_actions(state)    
        else :
            actions, mask = self._env.sample_adversary_actions(state)

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        else:
            mask = mask.to(dtype=torch.bool)
        return actions, mask
    
   
    def get_controller_action_key(self, action: ControllerAction) -> tuple:

        """
            Get a unique key for a controller action based on its parameters, which can be used to identify the action in the tree and 
            also to avoid duplication of same actions
        """

        # alloc = tuple(sorted((int(rid), int(tok)) for rid, tok in (action.token_allocations or {}).items()))
        # evicted_ids = tuple(sorted(int(x) for x in (getattr(action, "_evicted_request_ids", ()) or ())))
        # strategy = str(action.strategy or "")
        # evict_rule = strategy.split("|", 1)[1] if strategy.startswith("GV2|") else strategy
        # mapping = tuple(int(x) for x in (action.mapping or ()))
        # return (alloc, evicted_ids) if evicted_ids else (alloc, evict_rule, mapping)

        token_alloc = tuple(sorted((int(rid), int(tok)) for rid, tok in (action.token_allocations or {}).items()))
        prefill_alloc = tuple(sorted((int(rid), int(tok)) for rid, tok in (action.prefill_allocations or {}).items()))
        decode_alloc = tuple(sorted((int(rid), int(tok)) for rid, tok in (action.decode_allocations or {}).items()))
        evicted_ids = tuple(sorted(int(x) for x in (getattr(action, "_evicted_request_ids", ()) or ())))

        return (token_alloc, prefill_alloc, decode_alloc, evicted_ids)



    def canonicalize_action_indices( self, * , player: str, actions_by_index: List[Optional[Union[AdversaryAction, ControllerAction]]], valid_indices: List[int],) -> Tuple[Dict[int, int], Dict[int, List[int]], List[int]]:
        
        alias_to_canon: Dict[int, int] = {}
        canon_to_aliases: Dict[int, List[int]] = {}
        canonical_indices: List[int] = []

        if player != "controller":
            for idx in valid_indices:
                alias_to_canon[idx] = idx
                canon_to_aliases[idx] = [idx]
                canonical_indices.append(idx)
            return alias_to_canon, canon_to_aliases, canonical_indices
        
        seen: Dict[tuple, int] = {}

        for idx in valid_indices:

            action = actions_by_index[idx]
            if not isinstance(action, ControllerAction):
                key = ("non-controller", idx)
                raise ValueError(f"Expected ControllerAction for player 'controller', got {type(action)} at index {idx}")
            else:
                key = self.get_controller_action_key(action)

            canon = seen.get(key)
            if canon is None:
                seen[key] = idx
                alias_to_canon[idx] = idx
                canon_to_aliases[idx] = [idx]
                canonical_indices.append(idx)
            else:
                alias_to_canon[idx] = canon
                canon_to_aliases[canon].append(idx)

        return alias_to_canon, canon_to_aliases, canonical_indices


    def apply_action(
        self,
        state: VidurMCTSState,
        player: str,
        action: Union[AdversaryAction, ControllerAction],
    ) -> VidurMCTSState:
        if player == "adversary":
            return self._env.apply_adversary_action_only(state, action, inplace=True)
        # In tree MCTS this is a macro-transition. If fast_forward=True advances decode-only
        # or idle time, the child state's sim_time includes that advancement, and edge_discount
        # should use transition_final_time rather than transition_discount_time.
        return self._env.apply_controller_action_only(state, action, inplace=True, fast_forward=True)


    def ensure_expanded(self, node: Node, state: VidurMCTSState) -> None:

        """
            Ensures that the given node is expanded, if not expands the node by sampling all of the actions from the environment and setting up the children of the node accordingly.
            Does not create child nodes, just sets up the action space 
        """
        if node.expanded():
            return

        actions, mask_t = self.get_actions_and_mask(state, node.player)
        valid_mask = [bool(x) for x in mask_t.tolist()]
        valid_indices = [i for i, ok in enumerate(valid_mask) if ok and actions[i] is not None]

        alias_to_canon, canon_to_aliases, canonical_indices = self.canonicalize_action_indices(
            player=node.player,
            actions_by_index=actions,
            valid_indices=valid_indices,
        )

        node.actions_by_index = actions
        node.valid_mask = valid_mask
        node.action_alias_to_canonical = alias_to_canon
        node.canonical_to_action_aliases = canon_to_aliases
        node.untried_action_indices = list(canonical_indices)

        node.action_priors = self._compute_policy_priors(
            node,
            state,
            canonical_indices,
        )


    def _expand_one_child(self, node: Node, state: VidurMCTSState, action_idx: Optional[int] = None,) -> Tuple[Node, VidurMCTSState]:

        """
            Selects the action provided from the node and applies it from its state and gets a child
            If not provided then we select the one with highest prob score , index as a tie breaker.
            If prior disabled then we just select a random action from the untried actions.
            Note : Here the discount factor includes the transition time from the Fast forward
        """
        self.ensure_expanded(node, state)
        if not node.untried_action_indices:
            raise RuntimeError("expand_one_child called with no untried actions")

        if action_idx is None:
            if bool(getattr(self._mctsConfig, "use_policy_prior", False)):
                action_idx = max(
                    node.untried_action_indices,
                    key=lambda idx: (
                        node.action_priors.get(int(idx), 0.0),
                        -int(idx),
                    ),
                )
            else:
                pos = self._mctsConfig.rng.randrange(len(node.untried_action_indices))
                action_idx = int(node.untried_action_indices[pos])

        action_idx = int(action_idx)
        if action_idx not in node.untried_action_indices:
            raise RuntimeError(f"action_idx={action_idx} is not untried")

        node.untried_action_indices.remove(action_idx)

        action = node.actions_by_index[action_idx]
        if action is None:
            raise RuntimeError(f"action_idx={action_idx} is None")

        child_state = self.apply_action(state, node.player, action)
        child = Node(
            player=self.get_next_player(node),
            node_id=self.get_next_node_id(),
            depth=node.depth + 1,
            parent=node,
            parent_action=action,
            parent_action_index=action_idx,
        )
        child_cost = self.get_state_cost(child_state)
        child.reward = self.get_transition_reward(node.state_cost, child_cost)

        
        child_time = float(child_state.simulator._time)

        # Tree MCTS values the macro-transition that actually lands at child_state.
        # Therefore discount through all time included in that child state, including
        # controller-action time plus any decode/jump fast-forward time.
        final_time = getattr(child_state.stats, "transition_final_time", None)
        if final_time is None:
            final_time = child_time

        child.edge_discount = self.time_discount(float(final_time), float(node.sim_time))


        self.store_node_snapshot(child, child_state)
        node.children[action_idx] = child
        return child, child_state
    

    def _normalise_child_value_for_selection(self, parent: Node, child: Node) -> float:
        """
        Returns exploitation in [0, 1].

        child.mean_value() is controller-valued.
        Controller wants high values.
        Adversary wants low values, so invert after normalization.
        """
        q = float(child.mean_value())

        lo = float(parent.min_value)
        hi = float(parent.max_value)

        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo + 1e-12:
            return 0.5

        norm = (q - lo) / (hi - lo)
        norm = max(0.0, min(1.0, norm))

        if parent.player == "adversary":
            norm = 1.0 - norm

        return norm

    def _puct_explore(self, parent: Node, action_idx: int, child_visits: int) -> float:
        prior = float(parent.action_priors.get(int(action_idx), 0.0))
        prior = max(0.0, min(1.0, prior))
        parent_visits = max(1, int(parent.visits))
        child_visits = max(0, int(child_visits))

        explore = prior * math.sqrt(parent_visits) / (1.0 + child_visits)

        return explore


    def _puct_score_child(self, parent: Node, action_idx: int, child: Node) -> Tuple[float, int]:
        exploit = self._normalise_child_value_for_selection(parent, child)
        
        c = float(getattr(self._mctsConfig, "puct_c", 1.0) or 1.0)
        explore = self._puct_explore(parent, action_idx, child.visits)

        score = exploit + c * explore
        if exploit < 0.0 or exploit > 1.0 or explore < 0.0:
            raise ValueError(f"Unreasonable exploit/explore value computed: {exploit}for Parent node_id={parent.node_id}, action_idx={action_idx}")

        return (float(score), -int(action_idx))


    def _puct_score_untried(self, parent: Node, action_idx: int) -> Tuple[float, int]:
        # Neutral exploitation because this action has no Q yet.
        exploit = 0.5
        c = float(getattr(self._mctsConfig, "puct_c", 1.0) or 1.0)
        explore = self._puct_explore(parent, action_idx, 0)
        if exploit < 0.0 or exploit > 1.0 or explore < 0.0 :
            raise ValueError(f"Unreasonable exploit/explore value computed: {exploit}/{explore} for Parent node_id={parent.node_id}, action_idx={action_idx}")
        score = exploit + c * explore
        return (float(score), -int(action_idx))


    def puct_select_child_or_untried(self, node: Node) -> tuple[str, int, Optional[Node]]:
        """
        Returns:
        ("child", action_idx, child_node) for an already expanded child
        ("untried", action_idx, None) for an unexpanded canonical action
        """
        candidates: list[tuple[Tuple[float, int], str, int, Optional[Node]]] = []

        for action_idx, child in node.children.items():
            candidates.append(
                (
                    self._puct_score_child(node, int(action_idx), child),
                    "child",
                    int(action_idx),
                    child,
                )
            )

        for action_idx in node.untried_action_indices:
            candidates.append(
                (
                    self._puct_score_untried(node, int(action_idx)),
                    "untried",
                    int(action_idx),
                    None,
                )
            )

        if not candidates:
            raise RuntimeError("puct_select_child_or_untried called with no candidates")

        _, kind, action_idx, child = max(candidates, key=lambda x: x[0])
        return kind, action_idx, child

    def uct_select_child(self, node: Node) -> Node:
        """
        UCT where exploitation and exploration are both roughly [0, 1].
        Exploitation is normalized using node.min_value / node.max_value.
        TODO: Need to ensure that Exploration is correctly values because it will reach more than 1.0 so might be helpful to shift towards puct
        """
        c = float(getattr(self._mctsConfig, "uct_c", getattr(self._mctsConfig, "pb_c_base", 1.4)) or 1.4)
        parent_visits = max(1, int(node.visits))

        def score(item: Tuple[int, Node]) -> Tuple[float, int]:
            action_idx, child = item

            if child.visits <= 0:
                return (float("inf"), -int(action_idx))

            exploit = self._normalise_child_value_for_selection(node, child)
            # explore = math.sqrt(math.log(parent_visits + 1.0) / max(1, int(child.visits)))
            # explore = min(1.0, explore)
            # We need exploration to be between 0 and 1  for the UCT aswell 
            explore = math.sqrt(
                math.log(parent_visits + 1.0)
                / (math.log(parent_visits + 1.0) + max(1, int(child.visits)))
            )

            if explore < 0.0 or explore > 1.0 or exploit < 0.0 or exploit > 1.0:
                raise ValueError(f"Unreasonable explore/exploit values computed: parent_visits={parent_visits}, child_visits={child.visits}, exploit={exploit}, explore={explore}") 
            return (exploit + c * explore, -int(action_idx))

        return max(node.children.items(), key=score)[1]


    def _rollout_value(
        self,
        state: VidurMCTSState,
        player: str,
        *,
        dnn_model: Any,
        model_version: int,
        use_model_bootstrap: bool,
    ) -> float:
        """
        No random rollout here.

        This returns the model bootstrap value for the current leaf state.
        If model_version == 0, bootstrap is disabled and value is 0.
        """
        return self._bootstrap_value(
            dnn_model=dnn_model,
            state=state,
            player=player,
            model_version=int(model_version),
            use_model_bootstrap=bool(use_model_bootstrap),
        )

    

    def _backpropagate(self, path: Sequence[Node], leaf_bootstrap_value: float) -> None:
        """
        Backpropagates Bellman-style controller-valued returns.

        For edge parent -> child:

            q(parent, child) = child.reward + child.edge_discount * value(child)

        The propagated value is always controller-valued.
        """
        value = float(leaf_bootstrap_value)
        if value > 0.0:
            raise ValueError(f"Unreasonable leaf bootstrap value {value} at leaf node_id={path[-1].node_id}, values should be negative or 0")

        for node in reversed(path):
            if node.parent is not None:
                node_edge_discount = float(getattr(node, "edge_discount", 1.0))
                if node_edge_discount > 1.0 or node_edge_discount < 0.0:
                    raise ValueError(f"Unreasonable edge discount {node_edge_discount} at node_id={node.node_id}, parent_id={node.parent.node_id if node.parent else None}")
                
                value = float(node.reward) + node_edge_discount * value ## --> value becomes Q(a,s') from V(s')

            node.visits += 1
            node.value_sum += float(value)
            assert node.value_sum <= 0.0, f"Unreasonable value_sum {node.value_sum} at node_id={node.node_id}, values should be negative or 0"

            # self._update_node_bounds(node, value)

            if node.parent is not None:
                self._update_node_bounds(node.parent, value)



    def _best_root_action_index(self, root: Node) -> Optional[int]:
        "# Robust child: choose most visited. Tie-break by value from root player's objective."
        if not root.children:
            return None

        sign = 1.0 if root.player == "controller" else -1.0

        def key(item: Tuple[int, Node]) -> Tuple[int, float, int]:
            action_idx, child = item
            return (child.visits, sign * child.mean_value(), -action_idx)

        return max(root.children.items(), key=key)[0]



    def search_dnn(
        self,
        dnn_model: Any,
        rootState: VidurMCTSState,
        root_player: str,
        *,
        game_id: int,
        root_id: int,
        root_node_id_override: Optional[int],
        root_depth: int,
        mcts_iter: Optional[int] = None,
        model_version: int = 0,
        use_model_bootstrap: Optional[bool] = None,
        one_step_value_mode: bool = False,
        root_phase: str = "MCTS_root",
        cycle_label: str = "",
    ) -> MCTSSearchResult:
        # del dnn_model, game_id, root_id, model_version, use_model_bootstrap, one_step_value_mode, root_phase, cycle_label

        self.clear_search_state(drop_scratch=False)

        iterations = int(mcts_iter or getattr(self._mctsConfig, "mcts_iterations", 1000) or 1000)

        root_node_id = self.get_next_node_id() if root_node_id_override is None else int(root_node_id_override)
        self._node_id_counter = max(self._node_id_counter, root_node_id + 1)

        root = Node(
            player=str(root_player),
            node_id=root_node_id,
            depth=int(root_depth),
        )
        self._root = root
        self.store_node_snapshot(root, rootState)
        self.ensure_expanded(root, rootState)

        if not root.untried_action_indices and not root.children:
            return MCTSSearchResult(
                root_node_id=root.node_id,
                root_player=root.player,
                next_player=self.get_next_player(root),
                best_action_index=None,
                best_action=None,
                best_action_value=0.0,
                action_values=[float("-inf")] * len(root.actions_by_index),
                valid_mask=list(root.valid_mask),
                used_bootstrap=False,
            )

        if use_model_bootstrap is None:
            use_model_bootstrap = bool(int(model_version) > 0)

        for _ in range(iterations):
            node = root
            state = self.scratch_restore(root.cached_sim_snapshot, root.cached_stats)
            path = [root]


            if bool(getattr(self._mctsConfig, "use_policy_prior", False)):
                while True:
                    if not node.expanded():
                        self.ensure_expanded(node, state)

                    if not node.children and not node.untried_action_indices:
                        break

                    kind, action_idx, selected_child = self.puct_select_child_or_untried(node)

                    if kind == "child":
                        node = selected_child
                        state = self.scratch_restore(node.cached_sim_snapshot, node.cached_stats)
                        path.append(node)
                        continue

                    # Chosen action is an unexpanded canonical action.
                    node, state = self._expand_one_child(node, state, action_idx=action_idx)
                    path.append(node)
                    break
            else:
                print("We are using PUCT without policy priors, this is not recommended ! Error!")
                while node.expanded() and not node.untried_action_indices and node.children:
                    node = self.uct_select_child(node)
                    state = self.scratch_restore(node.cached_sim_snapshot, node.cached_stats)
                    path.append(node)

                if not node.expanded():
                    self.ensure_expanded(node, state)

                if node.untried_action_indices:
                    node, state = self._expand_one_child(node, state)
                    path.append(node)


            leaf_value = self._rollout_value(
                state,
                node.player,
                dnn_model=dnn_model,
                model_version=int(model_version),
                use_model_bootstrap=bool(use_model_bootstrap),
            )

            self._backpropagate(path, leaf_value)


        best_idx = self._best_root_action_index(root)
        n_actions = len(root.actions_by_index)
        action_values = [float("-inf")] * n_actions

        for canon_idx, child in root.children.items():
            q = child.mean_value()
            aliases = root.canonical_to_action_aliases.get(canon_idx, [canon_idx])
            for alias_idx in aliases:
                action_values[alias_idx] = q

        best_action = root.actions_by_index[best_idx] if best_idx is not None else None
        best_value = action_values[best_idx] if best_idx is not None else 0.0

        result = MCTSSearchResult(
            root_node_id=root.node_id,
            root_player=root.player,
            next_player=self.get_next_player(root),
            best_action_index=best_idx,
            best_action=best_action,
            best_action_value=float(best_value),
            action_values=action_values,
            valid_mask=list(root.valid_mask),
            used_bootstrap=bool(use_model_bootstrap),
        )

        if self._logger is not None:
            state_desc = self._env.describe_state(rootState)

            self._logger.log_root_summary(
                game_id=int(game_id),
                root_id=int(root_id),
                root=root,
                result=result,
                root_state=rootState,
                state_desc=state_desc,
            )

            def describe_child_state(child: Node) -> dict:
                state = self.scratch_restore(child.cached_sim_snapshot, child.cached_stats)
                return self._env.describe_state(state)

            self._logger.log_root_children(
                game_id=int(game_id),
                root_id=int(root_id),
                root=root,
                describe_child_state_fn=describe_child_state,
            )
        return result
    
    # Optional alias if some caller uses camelCase.
    def searchDNN(self, *args: Any, **kwargs: Any) -> MCTSSearchResult:
        return self.search_dnn(*args, **kwargs)



