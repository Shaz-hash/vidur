from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from GV4_Engine.action_resolver import (
    CanonicalAdversaryAction,
    CanonicalControllerAction,
)
from GV4_Engine.state import GV4State, Player
from GV4_Engine.virtual_environment import GV4VirtualVidurMCTSEnvironment
from GV4_Engine.dnn_inference.dnn_features import GV4StateFeatures
from GV4_Engine.dnn_inference.inference import GV4DNNInference

from .Game_Version3.logger.mcts_logger import MCTSCsvLogger


GV4Action = CanonicalAdversaryAction | CanonicalControllerAction


def _player_name(player: Player) -> str:
    if player == Player.ADVERSARY:
        return "adversary"
    if player == Player.CONTROLLER:
        return "controller"
    raise RuntimeError("GV4 Python MCTS does not yet implement router turns")


@dataclass(frozen=True, slots=True)
class _GV4SnapshotStats:
    """Retain the existing logger fields without duplicating simulator state."""

    transition_discount_time: float
    transition_final_time: float

    def clone(self) -> "_GV4SnapshotStats":
        return self


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
    """TODO:
    Need to remove unnecessary params such as :
    - exploration_constant, uct_c, prior_min_prob
    """

    mcts_iterations: int = 1000
    # exploration_constant: float = 1.0
    rng: random.Random = random.Random(42)
    log_flag: bool = False  # Whether to log the search for debugging or not ?
    log_path: Optional[Path] = None
    tree_log_path: Optional[Path] = None
    # Optional diagnostic hook called once per completed MCTS simulation.
    # Signature: observer(iteration_index=..., path=..., root=...,
    #                     game_id=..., root_id=...)
    iteration_observer: Any = None

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


@dataclass
class MCTSSearchResult:
    root_node_id: int
    root_player: str
    next_player: str
    best_action_index: Optional[int]
    best_action: Optional[GV4Action]
    best_action_value: float
    action_values: List[float]
    valid_mask: List[bool]
    used_bootstrap: bool = False


@dataclass
class Node:

    player: str  # Player who will take action on this node, either "controller" or "adversary"
    node_id: int
    depth: int = 0
    parent: Optional[Node] = None
    parent_action: Optional[GV4Action] = None  # action from parent to this node
    parent_action_index: Optional[int] = None

    action_priors: Dict[int, float] = field(
        default_factory=dict
    )  # canon action based only

    reward: float = (
        0.0  # reward/cost experienced from taking parent_action to reach this node from parent node
    )
    visits: int = 0  # number of times the parent applied this action to reach this node
    value_sum: float = (
        0.0  # sum of the boostrap value estimated obtained down this node. {Does not include the reward from parent action}
    )

    state_cost: float = 0.0  # the total SLO cost experienced at the state of this node
    sim_time: float = 0.0  # the current sim time at the state of this node

    ## Needed to normalise the exploitation factor b/w 0 and 1
    min_value: float = float("inf")
    max_value: float = float("-inf")

    # Edge stats from parent -> this node
    edge_discount: float = (
        1.0  # Discounting factor, need to be replaced by the actual discounting factor based on the time difference between parent and this node
    )

    children: Dict[int, Node] = field(
        default_factory=dict
    )  # children of this node action_index -> child node mapping
    actions_by_index: List[Optional[GV4Action]] = field(
        default_factory=list
    )  # actions from this node ,action_index -> action mapping for the children of this node
    valid_mask: List[bool] = field(
        default_factory=list
    )  # valid mask for the actions from this node, action_index -> bool mapping for the children of this node
    untried_action_indices: List[int] = field(
        default_factory=list
    )  # List of action indices which are still untried from this node

    action_alias_to_canonical: Dict[int, int] = field(
        default_factory=dict
    )  # mapping from action index of an action alias to the canonical action index in the children of this node
    canonical_to_action_aliases: Dict[int, List[int]] = field(
        default_factory=dict
    )  # mapping from canonical action index in the children of this node to the list of action indices of its aliases

    ## Optional/ Might remove them / Essentially the state of the simulator at this node:
    cached_sim_snapshot: Optional[Any] = (
        None  # cached simulation snapshot at the state of this node, can be used to speed up simulations from this node
    )
    cached_stats: Any = (
        None  # cached stats at the state of this node, can be used to speed up simulations from this node
    )

    cached_dnn_features: Optional[GV4StateFeatures] = None

    def mean_value(self) -> float:
        if self.visits == 0.0:
            return 0.0
        return self.value_sum / self.visits

    def expanded(self) -> bool:
        return len(self.actions_by_index) > 0


class VidurMCTS:

    def __init__(
        self,
        env: GV4VirtualVidurMCTSEnvironment,
        mctsConfig: MCTSConfig,
    ) -> None:
        if env.config.topology.num_replicas != 1:
            raise NotImplementedError(
                "GV4 Python MCTS currently supports one replica; router actions "
                "must be connected before multi-replica search"
            )

        self._env = env
        self._mctsConfig = mctsConfig
        self._root: Optional[Node] = None
        self._scratch_state: Optional[GV4State] = None
        self._node_id_counter = 0  # to assign unique ids to the nodes in the tree
        self._logger: Optional[MCTSCsvLogger] = None
        self._dnn_inference: GV4DNNInference | None = None

        if bool(getattr(mctsConfig, "log_flag", False)):
            if mctsConfig.log_path is None or mctsConfig.tree_log_path is None:
                raise ValueError(
                    "MCTS logging requires mctsConfig.log_path and mctsConfig.tree_log_path"
                )

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

    def get_state_cost(self, state: GV4State) -> float:
        """GV4 stores the complete objective cost in the state itself."""

        return float(state.objective.total_cost)

    def get_transition_reward(self, parent_node: Any, child_node: Any) -> float:
        # Accept either Node objects or precomputed costs.  The tree expansion
        # path computes the child cost before building the child node.
        parent_cost = getattr(parent_node, "state_cost", parent_node)
        child_cost = getattr(child_node, "state_cost", child_node)
        return float(parent_cost) - float(child_cost)

    def time_discount(self, child_final_time: float, parent_time: float) -> float:
        """Use the same elapsed-time discount contract as GV4 transitions."""

        elapsed = max(0.0, float(child_final_time) - float(parent_time))
        return float(self._env.config.reward.discount_for_elapsed(elapsed))

    def _node_features(self, node: Node, state: GV4State) -> GV4StateFeatures:
        inference = self._dnn_inference
        if inference is None:
            raise RuntimeError("GV4 DNN inference is not configured")

        if node.cached_dnn_features is None:
            node.cached_dnn_features = inference.build_state_features(state)

        return node.cached_dnn_features

    def _bootstrap_value(
        self,
        state: GV4State,
        player: str,
        *,
        node: Node | None,
        use_model_bootstrap: bool,
    ) -> float:
        if not use_model_bootstrap:
            return 0.0

        inference = self._dnn_inference
        if inference is None:
            raise RuntimeError("value bootstrap requires GV4DNNInference")

        features = (
            self._node_features(node, state)
            if node is not None
            else inference.build_state_features(state)
        )
        return inference.predict_value(
            state,
            player=player,
            state_features=features,
        )

    def _softmax(self, scores: Sequence[float], *, temperature: float) -> List[float]:
        # Temp = 1.0 is normal softmax, Temp < 1.0 sharpens the distribution, Temp > 1.0 flattens it.
        values = [float(score) for score in scores]
        if not values:
            return []

        temp = max(float(temperature), 1e-8)
        scaled = [float(x) / temp for x in values]
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

    def _apply_root_dirichlet_noise(
        self,
        priors: Dict[int, float],
        *,
        is_root: bool,
    ) -> Dict[int, float]:
        ## Alpha > 1 gives a more uniform distribution, while alpha < 1 gives a more peaked distribution.
        ## Epsilon controls the weight of the noise, with higher values leading to more exploration.
        if not is_root or not priors:
            return priors

        eps = float(getattr(self._mctsConfig, "root_dirichlet_epsilon", 0.0) or 0.0)
        alpha = resolve_root_dirichlet_alpha(
            float(getattr(self._mctsConfig, "root_dirichlet_alpha", 0.0) or 0.0),
            float(
                getattr(self._mctsConfig, "root_dirichlet_total_concentration", 0.0)
                or 0.0
            ),
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
        state: GV4State,
        canonical_indices: Sequence[int],
    ) -> Dict[int, float]:

        canonical_indices = [int(x) for x in canonical_indices]
        if not canonical_indices:
            return {}

        if not bool(getattr(self._mctsConfig, "use_policy_prior", False)):

            uniform_priors = self._uniform_priors(canonical_indices)
            uniform_priors = self._apply_root_dirichlet_noise(
                uniform_priors, is_root=(node.parent is None)
            )
            return uniform_priors

        actions = []
        scores = []
        if node.player == "controller":
            for index in canonical_indices:
                action = node.actions_by_index[index]
                if not isinstance(action, CanonicalControllerAction):
                    raise RuntimeError("controller canonical action is missing")
                actions.append(action)

            scores = self._dnn_inference.predict_controller_logits(
                state,
                actions,
                state_features=self._node_features(node, state),
            )
        else:
            for index in canonical_indices:
                action = node.actions_by_index[index]
                if not isinstance(action, CanonicalAdversaryAction):
                    raise RuntimeError("adversary canonical action is missing")
                actions.append(action)

            scores = self._dnn_inference.predict_adversary_logits(
                state,
                actions,
                state_features=self._node_features(node, state),
            )

        probs = self._softmax(
            scores,
            temperature=float(
                getattr(self._mctsConfig, "policy_prior_temperature", 1.0) or 1.0
            ),
        )

        min_prob = float(getattr(self._mctsConfig, "prior_min_prob", 1e-8) or 1e-8)
        priors = {
            int(idx): max(min_prob, float(prob))
            for idx, prob in zip(canonical_indices, probs)
        }
        total = sum(priors.values())
        priors = {idx: probability / total for idx, probability in priors.items()}

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

    def snapshot_state_and_stats(
        self,
        node: Node,
        state: GV4State,
    ) -> tuple[GV4State, _GV4SnapshotStats]:
        """Clone only compact GV4 branch state; immutable config stays shared."""

        del node
        metadata = _GV4SnapshotStats(state.now, state.now)
        return state.clone(), metadata

    def store_node_snapshot(self, node: Node, state: GV4State) -> None:
        node.cached_sim_snapshot, node.cached_stats = self.snapshot_state_and_stats(
            node, state
        )
        node.state_cost = self.get_state_cost(state)
        node.sim_time = float(state.now)

    def scratch_restore(self, snapshot: Any, stats: Any) -> GV4State:
        del stats
        if not isinstance(snapshot, GV4State):
            raise TypeError("GV4 MCTS snapshot must be a GV4State")
        self._scratch_state = snapshot.clone()
        return self._scratch_state

    def get_actions_and_mask(
        self,
        state: GV4State,
        player: str,
    ) -> tuple[list[GV4Action | None], list[bool]]:
        expected_player = _player_name(state.next_player)
        if player != expected_player:
            raise RuntimeError(
                f"MCTS player {player!r} does not match GV4 state turn "
                f"{expected_player!r}"
            )

        if player == "controller":
            actions, mask = self._env.sample_controller_actions(state, replica_id=0)
        else:
            actions, mask = self._env.sample_adversary_actions(state)
        return list(actions), [bool(value) for value in mask]

    def action_alias_views(
        self,
        actions_by_index: Sequence[GV4Action | None],
        valid_indices: Sequence[int],
    ) -> tuple[dict[int, int], dict[int, list[int]], list[int]]:
        """Expose GV4's existing canonical groups in MCTS index form."""

        alias_to_representative: dict[int, int] = {}
        representative_to_aliases: dict[int, list[int]] = {}

        for raw_index in valid_indices:
            action = actions_by_index[raw_index]
            if action is None:
                continue

            representative = int(action.representative_raw_index)
            if representative in representative_to_aliases:
                continue

            aliases = [int(index) for index in action.equivalent_raw_indices]
            representative_to_aliases[representative] = aliases

            for alias in aliases:
                alias_to_representative[alias] = representative

        representatives = sorted(representative_to_aliases)
        return (
            alias_to_representative,
            representative_to_aliases,
            representatives,
        )

    def apply_action(
        self,
        state: GV4State,
        player: str,
        action: GV4Action,
    ) -> GV4State:
        expected_player = _player_name(state.next_player)
        if player != expected_player:
            raise RuntimeError(
                f"MCTS player {player!r} does not match GV4 state turn "
                f"{expected_player!r}"
            )
        if player == "adversary":
            if not isinstance(action, CanonicalAdversaryAction):
                raise TypeError("adversary node requires CanonicalAdversaryAction")
            return self._env.apply_adversary_action_only(state, action, inplace=True)
        if not isinstance(action, CanonicalControllerAction):
            raise TypeError("controller node requires CanonicalControllerAction")
        return self._env.apply_controller_action_only(
            state,
            action,
            inplace=True,
            fast_forward=True,
        )

    def ensure_expanded(self, node: Node, state: GV4State) -> None:
        """
        Ensures that the given node is expanded, if not expands the node by sampling all of the actions from the environment and setting up the children of the node accordingly.
        Does not create child nodes, just sets up the action space
        """
        if node.expanded():
            return

        actions, valid_mask = self.get_actions_and_mask(state, node.player)
        valid_indices = [
            i for i, ok in enumerate(valid_mask) if ok and actions[i] is not None
        ]

        alias_to_canon, canon_to_aliases, canonical_indices = self.action_alias_views(
            actions, valid_indices
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

    def _expand_one_child(
        self,
        node: Node,
        state: GV4State,
        action_idx: Optional[int] = None,
    ) -> tuple[Node, GV4State]:
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
            player=_player_name(child_state.next_player),
            node_id=self.get_next_node_id(),
            depth=node.depth + 1,
            parent=node,
            parent_action=action,
            parent_action_index=action_idx,
        )
        child_cost = self.get_state_cost(child_state)
        child.reward = self.get_transition_reward(node.state_cost, child_cost)

        child.edge_discount = self.time_discount(child_state.now, node.sim_time)
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

    def _puct_score_child(
        self, parent: Node, action_idx: int, child: Node
    ) -> Tuple[float, int]:
        exploit = self._normalise_child_value_for_selection(parent, child)

        c = float(getattr(self._mctsConfig, "puct_c", 1.0) or 1.0)
        explore = self._puct_explore(parent, action_idx, child.visits)

        score = exploit + c * explore
        if exploit < 0.0 or exploit > 1.0 or explore < 0.0:
            raise ValueError(
                f"Unreasonable exploit/explore value computed: {exploit}for Parent node_id={parent.node_id}, action_idx={action_idx}"
            )

        return (float(score), -int(action_idx))

    def _puct_score_untried(self, parent: Node, action_idx: int) -> Tuple[float, int]:
        # Neutral exploitation because this action has no Q yet.
        exploit = 0.5
        c = float(getattr(self._mctsConfig, "puct_c", 1.0) or 1.0)
        explore = self._puct_explore(parent, action_idx, 0)
        if exploit < 0.0 or exploit > 1.0 or explore < 0.0:
            raise ValueError(
                f"Unreasonable exploit/explore value computed: {exploit}/{explore} for Parent node_id={parent.node_id}, action_idx={action_idx}"
            )
        score = exploit + c * explore
        return (float(score), -int(action_idx))

    def puct_select_child_or_untried(
        self, node: Node
    ) -> tuple[str, int, Optional[Node]]:
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
        Will remove it in near future after testing
        UCT where exploitation and exploration are both roughly [0, 1].
        Exploitation is normalized using node.min_value / node.max_value.
        TODO: Need to ensure that Exploration is correctly values because it will reach more than 1.0 so might be helpful to shift towards puct
        """
        c = float(
            getattr(
                self._mctsConfig, "uct_c", getattr(self._mctsConfig, "pb_c_base", 1.4)
            )
            or 1.4
        )
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
                raise ValueError(
                    f"Unreasonable explore/exploit values computed: parent_visits={parent_visits}, child_visits={child.visits}, exploit={exploit}, explore={explore}"
                )
            return (exploit + c * explore, -int(action_idx))

        return max(node.children.items(), key=score)[1]

    def _rollout_value(
        self,
        state: GV4State,
        player: str,
        *,
        node: Node | None,
        model_version: int,
        use_model_bootstrap: bool,
    ) -> float:
        """
        No random rollout here.

        This returns the model bootstrap value for the current leaf state.
        If model_version == 0, bootstrap is disabled and value is 0.
        """
        del model_version
        return self._bootstrap_value(
            node=node,
            state=state,
            player=player,
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
            raise ValueError(
                f"Unreasonable leaf bootstrap value {value} at leaf node_id={path[-1].node_id}, values should be negative or 0"
            )

        for node in reversed(path):
            if node.parent is not None:
                node_edge_discount = float(getattr(node, "edge_discount", 1.0))
                if node_edge_discount > 1.0 or node_edge_discount < 0.0:
                    raise ValueError(
                        f"Unreasonable edge discount {node_edge_discount} at node_id={node.node_id}, parent_id={node.parent.node_id if node.parent else None}"
                    )

                value = (
                    float(node.reward) + node_edge_discount * value
                )  ## --> value becomes Q(a,s') from V(s')

            node.visits += 1
            node.value_sum += float(value)
            assert (
                node.value_sum <= 0.0
            ), f"Unreasonable value_sum {node.value_sum} at node_id={node.node_id}, values should be negative or 0"

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
        # dnn_model: Any,
        dnn_inference: GV4DNNInference | None,
        rootState: GV4State,
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
        rootState.assert_valid(self._env.config)
        expected_player = _player_name(rootState.next_player)
        if root_player != expected_player:
            raise ValueError(
                f"root_player={root_player!r} does not match state turn "
                f"{expected_player!r}"
            )

        self.clear_search_state(drop_scratch=False)

        iterations = int(
            mcts_iter or getattr(self._mctsConfig, "mcts_iterations", 1000) or 1000
        )

        root_node_id = (
            self.get_next_node_id()
            if root_node_id_override is None
            else int(root_node_id_override)
        )
        self._node_id_counter = max(self._node_id_counter, root_node_id + 1)

        root = Node(
            player=str(root_player),
            node_id=root_node_id,
            depth=int(root_depth),
        )
        self._root = root
        self.store_node_snapshot(root, rootState)

        self._dnn_inference = dnn_inference

        # Model version is artifact metadata, not a switch for value inference.
        # In particular, an explicitly requested version-0 bootstrap is valid.
        bootstrap_enabled = (
            int(model_version) > 0
            if use_model_bootstrap is None
            else bool(use_model_bootstrap)
        )

        if self._mctsConfig.use_policy_prior and dnn_inference is None:
            raise RuntimeError("policy priors require GV4DNNInference")

        if bootstrap_enabled and dnn_inference is None:
            raise RuntimeError("value bootstrap requires GV4DNNInference")

        if dnn_inference is not None and dnn_inference.config != self._env.config:
            raise RuntimeError(
                "MCTS and DNN inference use different GV4 configurations"
            )

        self.ensure_expanded(root, rootState)

        if not root.untried_action_indices and not root.children:
            return MCTSSearchResult(
                root_node_id=root.node_id,
                root_player=root.player,
                next_player=expected_player,
                best_action_index=None,
                best_action=None,
                best_action_value=0.0,
                action_values=[float("-inf")] * len(root.actions_by_index),
                valid_mask=list(root.valid_mask),
                used_bootstrap=False,
            )

        for iteration_index in range(1, iterations + 1):
            node = root
            state = self.scratch_restore(root.cached_sim_snapshot, root.cached_stats)
            path = [root]

            while True:
                if not node.expanded():
                    self.ensure_expanded(node, state)

                if not node.children and not node.untried_action_indices:
                    break

                kind, action_idx, selected_child = self.puct_select_child_or_untried(
                    node
                )

                if kind == "child":
                    node = selected_child
                    state = self.scratch_restore(
                        node.cached_sim_snapshot, node.cached_stats
                    )
                    path.append(node)
                    continue

                # Chosen action is an unexpanded canonical action.
                node, state = self._expand_one_child(node, state, action_idx=action_idx)
                path.append(node)
                break

            leaf_value = self._rollout_value(
                state,
                node.player,
                node=node,
                model_version=int(model_version),
                use_model_bootstrap=bool(bootstrap_enabled),
            )

            self._backpropagate(path, leaf_value)

            observer = getattr(self._mctsConfig, "iteration_observer", None)
            if observer is not None:
                observer(
                    iteration_index=iteration_index,
                    path=tuple(path),
                    root=root,
                    game_id=int(game_id),
                    root_id=int(root_id),
                )

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
        next_player = expected_player
        if best_idx is not None:
            best_child = root.children.get(best_idx)
            if best_child is not None:
                next_player = best_child.player

        result = MCTSSearchResult(
            root_node_id=root.node_id,
            root_player=root.player,
            next_player=next_player,
            best_action_index=best_idx,
            best_action=best_action,
            best_action_value=float(best_value),
            action_values=action_values,
            valid_mask=list(root.valid_mask),
            used_bootstrap=bool(bootstrap_enabled),
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
                state = self.scratch_restore(
                    child.cached_sim_snapshot, child.cached_stats
                )
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


GV4VidurMCTS = VidurMCTS
