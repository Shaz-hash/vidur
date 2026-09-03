"""Structured, schema-checked replay recording for one GV4 game cycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .engine_runtime import (
    AppliedEdge,
    CanonicalActionRef,
    EngineMetadata,
    SearchResult,
)


REPLAY_SCHEMA_VERSION = "gv4_agz_replay_v1"

__all__ = [
    "GV4ReplayRecorder",
    "REPLAY_SCHEMA_VERSION",
    "ReplayRuntimeError",
    "ReplayWriteResult",
    "discounted_trajectory_targets",
]


class ReplayRuntimeError(RuntimeError):
    """Raised when a trajectory cannot form unambiguous training data."""


@dataclass(frozen=True, slots=True)
class ReplayWriteResult:
    state_path: Path
    action_path: Path
    manifest_path: Path
    state_rows: int
    action_rows: int
    state_sha256: str
    action_sha256: str


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    decision_index: int
    search: SearchResult
    selected_action: CanonicalActionRef
    started_at: float
    finished_at: float
    objective_before: float
    objective_after: float
    reward: float
    discount: float
    forced_edges: tuple[AppliedEdge, ...]


def discounted_trajectory_targets(
    rewards: Sequence[float],
    discounts: Sequence[float],
    bootstrap_value: float,
) -> list[float]:
    """Return Bellman targets using the exact discount stored on every edge."""

    if len(rewards) != len(discounts):
        raise ValueError("rewards and discounts must have equal length")
    running = float(bootstrap_value)
    if not math.isfinite(running):
        raise ValueError("bootstrap value must be finite")

    targets = [0.0] * len(rewards)
    for index in range(len(rewards) - 1, -1, -1):
        reward = float(rewards[index])
        discount = float(discounts[index])
        if not math.isfinite(reward):
            raise ValueError(f"reward at index {index} must be finite")
        if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError(f"discount at index {index} must be in [0, 1]")
        running = reward + discount * running
        targets[index] = running
    return targets


def _action_identity(action: CanonicalActionRef) -> dict[str, Any]:
    return {
        "player": action.player,
        "canonical_action_index": action.canonical_action_index,
        "representative_raw_index": action.representative_raw_index,
        "equivalent_raw_indices": list(action.equivalent_raw_indices),
    }


def _edge_audit(edge: AppliedEdge) -> dict[str, Any]:
    return {
        **_action_identity(edge.action),
        "transition_kind": edge.transition_kind,
        "started_at": edge.started_at,
        "finished_at": edge.finished_at,
        "reward": edge.reward,
        "discount": edge.discount,
        "objective_before": edge.objective_before,
        "objective_after": edge.objective_after,
        "next_player": edge.next_player,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReplayRuntimeError("replay metadata contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return str(value)


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    _json_safe(row),
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            _json_safe(value),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GV4ReplayRecorder:
    """Buffer one cycle, back up targets, then publish three atomic files."""

    __slots__ = (
        "output_dir",
        "game_id",
        "cycle_label",
        "engine_metadata",
        "feature_schema_version",
        "config_manifest_sha256",
        "model_versions",
        "game_seed",
        "history_hops",
        "epsilon",
        "overwrite",
        "_decisions",
        "_finished",
    )

    def __init__(
        self,
        output_dir: str | Path,
        *,
        game_id: int,
        cycle_label: str,
        engine_metadata: EngineMetadata,
        model_versions: Mapping[str, int] | None = None,
        game_seed: int = 0,
        history_hops: int = 0,
        overwrite: bool = False,
    ) -> None:
        if game_id < 0:
            raise ValueError("game_id must be nonnegative")
        if not cycle_label:
            raise ValueError("cycle_label cannot be empty")
        if game_seed < 0 or history_hops < 0:
            raise ValueError("game seed and history hops must be nonnegative")

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.game_id = int(game_id)
        self.cycle_label = str(cycle_label)
        self.engine_metadata = engine_metadata
        self.feature_schema_version = engine_metadata.feature_schema_version
        self.config_manifest_sha256 = engine_metadata.config_manifest_sha256
        self.model_versions = {
            str(role): int(version) for role, version in (model_versions or {}).items()
        }
        if any(version < 0 for version in self.model_versions.values()):
            raise ValueError("model versions must be nonnegative")
        self.game_seed = int(game_seed)
        self.history_hops = int(history_hops)
        self.epsilon = float(engine_metadata.time_epsilon)
        if not math.isfinite(self.epsilon) or self.epsilon < 0.0:
            raise ValueError("engine time epsilon must be finite and nonnegative")
        self.overwrite = bool(overwrite)
        self._decisions: list[_PendingDecision] = []
        self._finished = False

    def record_decision(
        self,
        *,
        decision_index: int,
        search: SearchResult,
        selected_action: CanonicalActionRef,
        selected_edge: AppliedEdge,
        forced_edges: Sequence[AppliedEdge] = (),
    ) -> None:
        """Record one searched edge plus any following forced one-action chain."""

        if self._finished:
            raise ReplayRuntimeError("cannot add a decision after finalization")
        if decision_index != len(self._decisions):
            raise ReplayRuntimeError("decision indices must be contiguous from zero")
        if search.root_player != selected_action.player:
            raise ReplayRuntimeError("selected action has the wrong root player")
        if selected_edge.action != selected_action:
            raise ReplayRuntimeError("selected edge does not match the selected action")
        search.stats_for(selected_action)
        if (
            search.state_features.schema_version != self.feature_schema_version
            or search.state_features.config_manifest_sha256
            != self.config_manifest_sha256
        ):
            raise ReplayRuntimeError("search features use a different schema or config")

        representatives: set[int] = set()
        for item in search.action_stats:
            representative = item.action.representative_raw_index
            if representative in representatives:
                raise ReplayRuntimeError("search repeats a canonical root action")
            representatives.add(representative)
            if item.action.player != search.root_player or item.visits < 0:
                raise ReplayRuntimeError("root action statistics are invalid")
            if not all(
                math.isfinite(value) for value in (item.value_sum, item.mean_value)
            ):
                raise ReplayRuntimeError("root action value is non-finite")
            if item.prior is not None and (
                not math.isfinite(item.prior) or item.prior < 0.0
            ):
                raise ReplayRuntimeError("root action prior is invalid")

        edges = (selected_edge, *tuple(forced_edges))
        reward = 0.0
        discount = 1.0
        previous_finish = selected_edge.started_at
        previous_objective = selected_edge.objective_before
        expected_player = search.root_player
        for edge in edges:
            if edge.player != expected_player:
                raise ReplayRuntimeError(
                    "composed replay edges have inconsistent turns"
                )
            if abs(edge.started_at - previous_finish) > self.epsilon:
                raise ReplayRuntimeError("composed replay edges are not contiguous")
            if abs(edge.objective_before - previous_objective) > self.epsilon:
                raise ReplayRuntimeError(
                    "composed replay objectives are not contiguous"
                )
            if (
                not all(
                    math.isfinite(value)
                    for value in (
                        edge.started_at,
                        edge.finished_at,
                        edge.objective_before,
                        edge.objective_after,
                        edge.reward,
                        edge.discount,
                    )
                )
                or not 0.0 <= edge.discount <= 1.0
            ):
                raise ReplayRuntimeError("edge reward or discount is invalid")
            if edge.finished_at + self.epsilon < edge.started_at:
                raise ReplayRuntimeError("replay edge moved time backwards")
            expected_reward = edge.objective_before - edge.objective_after
            if abs(edge.reward - expected_reward) > self.epsilon:
                raise ReplayRuntimeError("edge reward disagrees with objective delta")
            reward += discount * edge.reward
            discount *= edge.discount
            previous_finish = edge.finished_at
            previous_objective = edge.objective_after
            expected_player = edge.next_player

        final_edge = edges[-1]
        self._decisions.append(
            _PendingDecision(
                decision_index=decision_index,
                search=search,
                selected_action=selected_action,
                started_at=selected_edge.started_at,
                finished_at=final_edge.finished_at,
                objective_before=selected_edge.objective_before,
                objective_after=final_edge.objective_after,
                reward=reward,
                discount=discount,
                forced_edges=tuple(forced_edges),
            )
        )

    def finish_cycle(
        self,
        *,
        bootstrap_kind: str,
        bootstrap_value: float,
        end_reason: str,
        final_time: float,
    ) -> ReplayWriteResult:
        """Compute targets and atomically publish state, action, and manifest data."""

        if self._finished:
            raise ReplayRuntimeError("cycle was already finalized")
        if bootstrap_kind not in {"terminal_zero", "neutral_zero", "model"}:
            raise ValueError("unsupported bootstrap_kind")
        if bootstrap_kind == "terminal_zero" and bootstrap_value != 0.0:
            raise ValueError("terminal_zero bootstrap must have value zero")
        if not math.isfinite(bootstrap_value):
            raise ValueError("bootstrap_value must be finite")
        if not end_reason:
            raise ValueError("end_reason cannot be empty")
        if not math.isfinite(final_time) or final_time < 0.0:
            raise ValueError("final_time must be finite and nonnegative")
        if (
            self._decisions
            and final_time + self.epsilon < self._decisions[-1].finished_at
        ):
            raise ReplayRuntimeError("final time precedes the final replay decision")

        targets = discounted_trajectory_targets(
            [item.reward for item in self._decisions],
            [item.discount for item in self._decisions],
            bootstrap_value,
        )
        state_rows: list[dict[str, Any]] = []
        action_rows: list[dict[str, Any]] = []

        for decision, target in zip(self._decisions, targets):
            search = decision.search
            total_visits = sum(item.visits for item in search.action_stats)
            if len(search.action_stats) > 1 and total_visits <= 0:
                raise ReplayRuntimeError("branching search has no root visits")

            state_rows.append(
                {
                    "replay_schema_version": REPLAY_SCHEMA_VERSION,
                    "game_id": self.game_id,
                    "cycle_label": self.cycle_label,
                    "decision_index": decision.decision_index,
                    "root_node_id": search.root_node_id,
                    "player": search.root_player,
                    "started_at": decision.started_at,
                    "finished_at": decision.finished_at,
                    "objective_before": decision.objective_before,
                    "objective_after": decision.objective_after,
                    "reward": decision.reward,
                    "discount": decision.discount,
                    "target_value": target,
                    "root_value": search.root_value,
                    "best_action_value": search.best_action_value,
                    "selected_action": _action_identity(decision.selected_action),
                    "canonical_action_count": len(search.action_stats),
                    "raw_valid_mask": list(search.raw_valid_mask),
                    "state_features": asdict(search.state_features),
                    "used_bootstrap": search.used_bootstrap,
                    "used_rollout": search.used_rollout,
                    "search_diagnostics": search.diagnostics,
                    "forced_chain": [
                        _edge_audit(edge) for edge in decision.forced_edges
                    ],
                    "bootstrap_kind": bootstrap_kind,
                    "bootstrap_value": bootstrap_value,
                }
            )

            for item in search.action_stats:
                probability = item.visits / total_visits if total_visits > 0 else 0.0
                action_rows.append(
                    {
                        "replay_schema_version": REPLAY_SCHEMA_VERSION,
                        "game_id": self.game_id,
                        "cycle_label": self.cycle_label,
                        "decision_index": decision.decision_index,
                        **_action_identity(item.action),
                        "visits": item.visits,
                        "visit_probability": probability,
                        "value_sum": item.value_sum,
                        "mean_value": item.mean_value,
                        "prior": item.prior,
                        "selected": (
                            item.action.representative_raw_index
                            == decision.selected_action.representative_raw_index
                        ),
                        "action_features": asdict(item.features),
                    }
                )

        state_path = self.output_dir / "replay_states.jsonl"
        action_path = self.output_dir / "replay_actions.jsonl"
        manifest_path = self.output_dir / "replay_manifest.json"
        destinations = (state_path, action_path, manifest_path)
        if not self.overwrite and any(path.exists() for path in destinations):
            raise FileExistsError("replay output already exists")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_jsonl(state_path, state_rows)
        _atomic_jsonl(action_path, action_rows)
        state_sha256 = _sha256_file(state_path)
        action_sha256 = _sha256_file(action_path)
        _atomic_json(
            manifest_path,
            {
                "status": "complete",
                "replay_schema_version": REPLAY_SCHEMA_VERSION,
                "engine_metadata": asdict(self.engine_metadata),
                "feature_schema_version": self.feature_schema_version,
                "config_manifest_sha256": self.config_manifest_sha256,
                "game_id": self.game_id,
                "cycle_label": self.cycle_label,
                "game_seed": self.game_seed,
                "history_hops": self.history_hops,
                "model_versions": self.model_versions,
                "state_rows": len(state_rows),
                "action_rows": len(action_rows),
                "state_sha256": state_sha256,
                "action_sha256": action_sha256,
                "bootstrap_kind": bootstrap_kind,
                "bootstrap_value": bootstrap_value,
                "end_reason": end_reason,
                "final_time": final_time,
            },
        )
        self._finished = True
        return ReplayWriteResult(
            state_path=state_path,
            action_path=action_path,
            manifest_path=manifest_path,
            state_rows=len(state_rows),
            action_rows=len(action_rows),
            state_sha256=state_sha256,
            action_sha256=action_sha256,
        )
