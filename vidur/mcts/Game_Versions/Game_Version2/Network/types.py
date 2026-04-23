# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal


TaskType = Literal["selfplay", "arena"]


@dataclass(frozen=True)
class RemoteModelRef:
    weights_s3_key: str
    model_version: int
    controller_ts_s3_key: str = ""
    adversary_ts_s3_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RemoteModelRef":
        return cls(
            weights_s3_key=str(payload["weights_s3_key"]),
            model_version=int(payload["model_version"]),
            controller_ts_s3_key=str(payload.get("controller_ts_s3_key", "")),
            adversary_ts_s3_key=str(payload.get("adversary_ts_s3_key", "")),
        )


@dataclass(frozen=True)
class SelfplayTask:
    session_id: str
    task_id: str
    generation: int
    task_type: TaskType
    model_ref: RemoteModelRef
    output_dataset_rel_dir: str
    iter_log_relpath: str
    root_log_relpath: str
    game_id: int
    num_roots: int
    start_root_id: int
    start_root_depth: int
    start_player: str
    feature_version: int
    adv_iterations_per_root: int
    cont_iterations_per_root: int
    max_batch_size: int
    history_nontrivial_hops: int
    history_seed: int
    sample_from_mcts_policy: bool
    selfplay_policy_temperature: float
    action_seed_base: int
    max_forced_hops_per_root: int
    history_max_total_steps: int
    log_history_rows: bool
    task_seed: int

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["model_ref"] = self.model_ref.to_dict()
        return out

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SelfplayTask":
        return cls(
            session_id=str(payload["session_id"]),
            task_id=str(payload["task_id"]),
            generation=int(payload["generation"]),
            task_type="selfplay",
            model_ref=RemoteModelRef.from_dict(dict(payload["model_ref"])),
            output_dataset_rel_dir=str(payload["output_dataset_rel_dir"]),
            iter_log_relpath=str(payload["iter_log_relpath"]),
            root_log_relpath=str(payload["root_log_relpath"]),
            game_id=int(payload["game_id"]),
            num_roots=int(payload["num_roots"]),
            start_root_id=int(payload["start_root_id"]),
            start_root_depth=int(payload["start_root_depth"]),
            start_player=str(payload["start_player"]),
            feature_version=int(payload["feature_version"]),
            adv_iterations_per_root=int(payload["adv_iterations_per_root"]),
            cont_iterations_per_root=int(payload["cont_iterations_per_root"]),
            max_batch_size=int(payload["max_batch_size"]),
            history_nontrivial_hops=int(payload["history_nontrivial_hops"]),
            history_seed=int(payload["history_seed"]),
            sample_from_mcts_policy=bool(payload["sample_from_mcts_policy"]),
            selfplay_policy_temperature=float(payload["selfplay_policy_temperature"]),
            action_seed_base=int(payload["action_seed_base"]),
            max_forced_hops_per_root=int(payload["max_forced_hops_per_root"]),
            history_max_total_steps=int(payload["history_max_total_steps"]),
            log_history_rows=bool(payload["log_history_rows"]),
            task_seed=int(payload["task_seed"]),
        )


@dataclass(frozen=True)
class ArenaTask:
    session_id: str
    task_id: str
    generation: int
    task_type: TaskType
    candidate_model_ref: RemoteModelRef
    best_model_ref: RemoteModelRef
    arena_entries: List[Dict[str, Any]]
    arena_games_rel_dir: str
    adv_iterations_per_root: int
    cont_iterations_per_root: int
    arena_time_limit_sec: float
    arena_max_controller_cleanup_steps: int
    arena_max_total_turns: int
    feature_version: int
    tie_points: float
    task_seed: int
    action_seed_base: int

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["candidate_model_ref"] = self.candidate_model_ref.to_dict()
        out["best_model_ref"] = self.best_model_ref.to_dict()
        return out

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ArenaTask":
        return cls(
            session_id=str(payload["session_id"]),
            task_id=str(payload["task_id"]),
            generation=int(payload["generation"]),
            task_type="arena",
            candidate_model_ref=RemoteModelRef.from_dict(dict(payload["candidate_model_ref"])),
            best_model_ref=RemoteModelRef.from_dict(dict(payload["best_model_ref"])),
            arena_entries=[dict(x) for x in payload.get("arena_entries", [])],
            arena_games_rel_dir=str(payload["arena_games_rel_dir"]),
            adv_iterations_per_root=int(payload["adv_iterations_per_root"]),
            cont_iterations_per_root=int(payload["cont_iterations_per_root"]),
            arena_time_limit_sec=float(payload["arena_time_limit_sec"]),
            arena_max_controller_cleanup_steps=int(payload["arena_max_controller_cleanup_steps"]),
            arena_max_total_turns=int(payload["arena_max_total_turns"]),
            feature_version=int(payload["feature_version"]),
            tie_points=float(payload["tie_points"]),
            task_seed=int(payload["task_seed"]),
            action_seed_base=int(payload.get("action_seed_base", payload["task_seed"])),
        )


@dataclass(frozen=True)
class RemoteResult:
    session_id: str
    task_id: str
    generation: int
    task_type: TaskType
    ok: bool
    produced_files: List[str]
    num_roots: int = 0
    num_games: int = 0
    error: str = ""
    traceback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RemoteResult":
        return cls(
            session_id=str(payload["session_id"]),
            task_id=str(payload["task_id"]),
            generation=int(payload["generation"]),
            task_type=str(payload["task_type"]),
            ok=bool(payload["ok"]),
            produced_files=[str(x) for x in payload.get("produced_files", [])],
            num_roots=int(payload.get("num_roots", 0)),
            num_games=int(payload.get("num_games", 0)),
            error=str(payload.get("error", "")),
            traceback=str(payload.get("traceback", "")),
        )


def task_from_dict(payload: Dict[str, Any]) -> SelfplayTask | ArenaTask:
    task_type = str(payload.get("task_type", ""))
    if task_type == "selfplay":
        return SelfplayTask.from_dict(payload)
    if task_type == "arena":
        return ArenaTask.from_dict(payload)
    raise ValueError(f"Unknown task_type={task_type!r}")
