"""PUCT MCTS with policy-guided, fixed-simulated-time leaf rollouts.

This isolated variant reuses the existing tree, transitions, rewards,
time-dependent discounts, and backup. Only leaf evaluation changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Optional, Sequence

from .environment import VidurMCTSState
from .mcts_value_prior import Node, VidurMCTS


@dataclass
class PolicyRolloutStats:
    leaf_evaluations: int = 0
    trajectories: int = 0
    actions: int = 0
    terminal_trajectories: int = 0
    bootstrap_calls: int = 0

    cutoff_leaf_evaluations: int = 0
    root_time: Optional[float] = None
    deadline: Optional[float] = None
    min_start_time: Optional[float] = None
    max_start_time: Optional[float] = None
    min_final_time: Optional[float] = None
    max_final_time: Optional[float] = None
    min_deadline: Optional[float] = None
    max_deadline: Optional[float] = None
    min_expansion_parent_time: Optional[float] = None
    max_expansion_parent_time: Optional[float] = None
    min_remaining_rollout_sec: Optional[float] = None
    max_remaining_rollout_sec: Optional[float] = None
    first_history_hash: Optional[int] = None
    first_history_actions: int = 0

@dataclass(frozen=True)
class _RolloutEdge:
    reward: float
    discount: float


class VidurMCTSPolicyRollout(VidurMCTS):
    """Replace immediate child V with equal-horizon policy continuations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rollout_root_time: Optional[float] = None
        self._pending_rollout_parent_time: Optional[float] = None
        self._pending_rollout_deadline: Optional[float] = None
        self.rollout_stats = PolicyRolloutStats()

    def clear_search_state(self, *, drop_scratch: bool = True) -> None:
        super().clear_search_state(drop_scratch=drop_scratch)
        self._pending_rollout_parent_time = None
        self._pending_rollout_deadline = None
        self.rollout_stats = PolicyRolloutStats(
            root_time=self._rollout_root_time,
        )

    def search_dnn(
        self,
        dnn_model: Any,
        rootState: VidurMCTSState,
        root_player: str,
        **kwargs: Any,
    ) -> Any:
        """Run continuations to a fixed deadline from each expansion parent."""
        self._rollout_root_time = float(rootState.simulator._time)
        return super().search_dnn(
            dnn_model,
            rootState,
            root_player,
            **kwargs,
        )

    def _expand_one_child(
        self,
        node: Node,
        state: VidurMCTSState,
        action_idx: Optional[int] = None,
    ) -> tuple[Node, VidurMCTSState]:
        """Anchor the rollout horizon before applying the selected action."""
        parent_time = float(state.simulator._time)
        child, child_state = super()._expand_one_child(
            node,
            state,
            action_idx=action_idx,
        )
        horizon = float(getattr(self._mctsConfig, "rollout_horizon_sec", 0.4))
        self._pending_rollout_parent_time = parent_time
        self._pending_rollout_deadline = (
            parent_time + horizon if horizon > 0.0 else None
        )
        return child, child_state

    @staticmethod
    def _update_range(
        stats: PolicyRolloutStats,
        minimum_name: str,
        maximum_name: str,
        value: float,
    ) -> None:
        current_min = getattr(stats, minimum_name)
        current_max = getattr(stats, maximum_name)
        setattr(
            stats,
            minimum_name,
            value if current_min is None else min(current_min, value),
        )
        setattr(
            stats,
            maximum_name,
            value if current_max is None else max(current_max, value),
        )

    def _sample_action(self, node: Node, rng: random.Random) -> int:
        indices = sorted(int(idx) for idx in node.untried_action_indices)
        if not indices:
            raise RuntimeError("empty rollout action set")
        if len(indices) == 1:
            return indices[0]
        probabilities = [
            max(0.0, float(node.action_priors.get(idx, 0.0))) for idx in indices
        ]
        total = sum(probabilities)
        probabilities = (
            [probability / total for probability in probabilities]
            if total > 0.0
            else [1.0 / len(indices)] * len(indices)
        )
        quantum = float(
            getattr(self._mctsConfig, "rollout_probability_quantum", 1e-6)
        )
        if quantum > 0.0:
            probabilities = [
                math.floor(probability / quantum + 0.5) * quantum
                for probability in probabilities
            ]
            quantized_total = sum(probabilities)
            if quantized_total > 0.0:
                probabilities = [value / quantized_total for value in probabilities]
        draw = float(rng.random())
        cumulative = 0.0
        for idx, probability in zip(indices, probabilities):
            cumulative += probability
            if draw < cumulative:
                return idx
        return indices[-1]

    def _policy_node(
        self,
        state: VidurMCTSState,
        player: str,
        parent_sentinel: Node,
    ) -> Node:
        node = Node(player=player, node_id=-1, parent=parent_sentinel)
        actions, mask_t = self.get_actions_and_mask(state, player)
        node.valid_mask = [bool(value) for value in mask_t.tolist()]
        valid = [
            idx
            for idx, enabled in enumerate(node.valid_mask)
            if enabled and actions[idx] is not None
        ]
        alias, aliases, canonical = self.canonicalize_action_indices(
            player=player,
            actions_by_index=actions,
            valid_indices=valid,
        )
        node.actions_by_index = actions
        node.action_alias_to_canonical = alias
        node.canonical_to_action_aliases = aliases
        node.untried_action_indices = list(canonical)
        tree_temperature = self._mctsConfig.policy_prior_temperature
        self._mctsConfig.policy_prior_temperature = float(
            getattr(self._mctsConfig, "rollout_policy_temperature", 1.0)
        )
        try:
            node.action_priors = self._compute_policy_priors(node, state, canonical)
        finally:
            self._mctsConfig.policy_prior_temperature = tree_temperature
        return node

    @staticmethod
    def _compose_return(edges: Sequence[_RolloutEdge], bootstrap: float) -> float:
        value = float(bootstrap)
        for edge in reversed(edges):
            value = edge.reward + edge.discount * value
        return value

    def _one_rollout(
        self,
        snapshot: Any,
        stats: Any,
        player: str,
        *,
        dnn_model: Any,
        model_version: int,
        rng: random.Random,
        use_model_bootstrap: bool,
        target_time: float,
        capture_history: bool = False,
    ) -> float:
        state = self.scratch_restore(snapshot, stats)
        self._update_range(
            self.rollout_stats,
            "min_start_time",
            "max_start_time",
            float(state.simulator._time),
        )
        max_actions = max(
            1, int(getattr(self._mctsConfig, "rollout_max_actions", 4096))
        )
        edges: list[_RolloutEdge] = []
        history_hash = 2_166_136_261
        sentinel = Node(player="rollout", node_id=-2)
        for _ in range(max_actions):
            if float(state.simulator._time) >= target_time:
                break
            policy_node = self._policy_node(state, player, sentinel)
            if not policy_node.untried_action_indices:
                self.rollout_stats.terminal_trajectories += 1
                break
            action_idx = self._sample_action(policy_node, rng)
            history_hash = ((history_hash ^ int(action_idx)) * 16_777_619) & 0xFFFFFFFF
            action = policy_node.actions_by_index[action_idx]
            if action is None:
                raise RuntimeError(f"empty rollout action {action_idx}")
            parent_cost = self.get_state_cost(state)
            parent_time = float(state.simulator._time)
            state = self.apply_action(state, player, action)
            final_time = getattr(
                state.stats, "transition_final_time", state.simulator._time
            )
            edges.append(
                _RolloutEdge(
                    self.get_transition_reward(
                        parent_cost, self.get_state_cost(state)
                    ),
                    self.time_discount(float(final_time), parent_time),
                )
            )
            self.rollout_stats.actions += 1
            player = "controller" if player == "adversary" else "adversary"
        else:
            raise RuntimeError(f"rollout exceeded {max_actions} actions")
        bootstrap = self._bootstrap_value(
            dnn_model,
            state,
            player,
            model_version=model_version,
            use_model_bootstrap=use_model_bootstrap,
        )
        self.rollout_stats.bootstrap_calls += 1
        if capture_history:
            self.rollout_stats.first_history_hash = history_hash
            self.rollout_stats.first_history_actions = len(edges)
        self._update_range(
            self.rollout_stats,
            "min_final_time",
            "max_final_time",
            float(state.simulator._time),
        )
        return self._compose_return(edges, bootstrap)

    def _rollout_value(
        self,
        state: VidurMCTSState,
        player: str,
        *,
        dnn_model: Any,
        model_version: int,
        use_model_bootstrap: bool,
    ) -> float:
        horizon = float(getattr(self._mctsConfig, "rollout_horizon_sec", 0.4))
        if horizon <= 0.0:
            self._pending_rollout_parent_time = None
            self._pending_rollout_deadline = None
            return super()._rollout_value(
                state,
                player,
                dnn_model=dnn_model,
                model_version=model_version,
                use_model_bootstrap=use_model_bootstrap,
            )
        expansion_parent_time = self._pending_rollout_parent_time
        target_time = self._pending_rollout_deadline
        self._pending_rollout_parent_time = None
        self._pending_rollout_deadline = None
        if target_time is None:
            # No action was expanded, so there is no sibling-action horizon to
            # compare. This occurs at a terminal tree node; bootstrap in place.
            expansion_parent_time = float(state.simulator._time)
            target_time = expansion_parent_time
        if expansion_parent_time is None:
            raise RuntimeError("rollout deadline is missing its expansion parent")
        self._update_range(
            self.rollout_stats,
            "min_expansion_parent_time",
            "max_expansion_parent_time",
            expansion_parent_time,
        )
        remaining_rollout = max(
            0.0,
            target_time - float(state.simulator._time),
        )
        self._update_range(
            self.rollout_stats,
            "min_remaining_rollout_sec",
            "max_remaining_rollout_sec",
            remaining_rollout,
        )
        self._update_range(
            self.rollout_stats,
            "min_deadline",
            "max_deadline",
            target_time,
        )

        count = max(1, int(getattr(self._mctsConfig, "rollout_count", 10)))
        snapshot, stats = self.snapshot_state_and_stats(
            Node(player=player, node_id=-1), state
        )
        leaf_index = self.rollout_stats.leaf_evaluations
        base_seed = int(getattr(self._mctsConfig, "rollout_seed", 0))
        values = [
            self._one_rollout(
                snapshot,
                stats,
                player,
                dnn_model=dnn_model,
                model_version=model_version,
                rng=random.Random(
                    (base_seed + leaf_index * 1_000_003 + rollout_index * 9_176)
                    & ((1 << 64) - 1)
                ),
                use_model_bootstrap=use_model_bootstrap,
                target_time=target_time,
                capture_history=(leaf_index == 0 and rollout_index == 0),
            )
            for rollout_index in range(count)
        ]
        self.rollout_stats.leaf_evaluations += 1
        self.rollout_stats.trajectories += count
        return sum(values) / len(values)


MCTSSearchWithPolicyRollouts = VidurMCTSPolicyRollout
