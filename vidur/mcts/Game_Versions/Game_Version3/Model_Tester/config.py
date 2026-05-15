from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Tuple

from ..config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG, MultipleProcessTrainingConfig


def _repo_root() -> Path:
    # .../vidur/vidur/mcts/Game_Versions/Game_Version3/Model_Tester/config.py -> /home/ubuntu/vidur
    return Path(__file__).resolve().parents[5]


def _under_repo(*parts: str) -> str:
    return str(_repo_root().joinpath(*parts))


@dataclass(frozen=True)
class TrivialControllerPolicyConfig:
    heuristic: str = "SJF"
    budget_tokens: int = 128
    eviction_rule: str = "evict_none"


@dataclass(frozen=True)
class TrivialAdversaryPolicyConfig:
    heuristic: str = "max_prefill_tokens"


@dataclass(frozen=True)
class ModelTesterConfig:
    # Experiment scope
    num_games: int = 70
    game_id_start: int = 12_000_000
    start_player: str = "adversary"
    start_root_depth: int = 0
    feature_version: int = 1

    # Model under test
    model_kind: str = "torch_checkpoint"  # "torch_checkpoint" | "classical_joblib"
    model_checkpoint_path: str = _under_repo(
        "simulator_output",
        "Game_Version3",
        "mcts_dnn_checkpoints",
        "best.pt",
    )

    # Outputs
    output_dir: str = _under_repo("simulator_output", "Game_Version3", "Model_Tester_Results")
    write_arena_game_logs: bool = True
    write_model_action_detail_logs: bool = False

    # History sampling
    history_seed: int = 2026
    history_hops_min: int = 0
    history_hops_max: int = 100
    history_hops_unique: bool = True
    history_hops_force_zero: bool = True  # ensures one game starts from initial state

    # Arena controls
    arena_time_limit_sec: float = 5.0
    arena_max_controller_cleanup_steps: int = 1024
    arena_max_total_turns: int = 4096

    # One-step bootstrap selector controls
    bootstrap_model_version: int = 1

    # Trivial baselines used in the arena comparisons.
    trivial_policy: TrivialControllerPolicyConfig = field(default_factory=TrivialControllerPolicyConfig)
    trivial_adversary_policy: TrivialAdversaryPolicyConfig = field(default_factory=TrivialAdversaryPolicyConfig)

    # Runtime
    environment_lang: str = "native"  # "python" | "native"
    use_virtual_env: bool = True
    model_device: str = "cpu"

    native_log_events: bool = True
    native_profile: bool = False
    native_log_flush_every: int = 1
    native_model_version: int = 777_001
    native_torchscript_dir: str = _under_repo(
        "simulator_output",
        "Game_Version3",
        "torchscript",
        "model_tester",
    )

    # Use same simulator CLI defaults as pipeline unless overridden.
    sim_cli_args: Tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_MULTIPROCESS_TRAINING_CONFIG.sim.cli_args)
    )

    def output_dir_path(self) -> Path:
        return Path(self.output_dir)

    def arena_games_dir_path(self) -> Path:
        return self.output_dir_path() / "arena_games"

    def arena_results_csv_path(self) -> Path:
        return self.output_dir_path() / "arena_results.csv"

    def mcts_iter_log_path(self) -> Path:
        return self.output_dir_path() / "mcts_iter.csv"

    def mcts_root_log_path(self) -> Path:
        return self.output_dir_path() / "mcts_root.csv"

    def to_pipeline_cfg(self) -> MultipleProcessTrainingConfig:
        base = DEFAULT_MULTIPROCESS_TRAINING_CONFIG
        sim_group = replace(base.sim, cli_args=tuple(self.sim_cli_args))
        run_group = replace(
            base.run,
            root_player=str(self.start_player),
            root_depth=int(self.start_root_depth),
            feature_version=int(self.feature_version),
        )
        model_group = replace(base.model, device=str(self.model_device))
        logging_group = replace(
            base.logging,
            mcts_iter_log=str(self.mcts_iter_log_path()),
            mcts_root_log=str(self.mcts_root_log_path()),
            flush_every=1,
        )
        return replace(
            base,
            sim=sim_group,
            run=run_group,
            model=model_group,
            logging=logging_group,
            environment_lang=str(self.environment_lang),
            use_virtual_env=bool(self.use_virtual_env),
            native_log_events=bool(self.native_log_events),
            native_profile=bool(self.native_profile),
            native_log_flush_every=int(self.native_log_flush_every),
            native_torchscript_dir=str(self.native_torchscript_dir),
        )

    def sample_history_hops(self) -> list[int]:
        n = int(self.num_games)
        lo = int(self.history_hops_min)
        hi = int(self.history_hops_max)
        if n <= 0:
            return []
        if lo > hi:
            raise ValueError("history_hops_min must be <= history_hops_max")

        rng = random.Random(int(self.history_seed))
        if bool(self.history_hops_unique):
            pop = list(range(lo, hi + 1))
            if n > len(pop):
                raise ValueError("num_games exceeds unique history-hop capacity")
            hops = rng.sample(pop, k=n)
        else:
            hops = [int(rng.randint(lo, hi)) for _ in range(n)]

        if bool(self.history_hops_force_zero):
            if not (lo <= 0 <= hi):
                raise ValueError("history hop range must include 0 when history_hops_force_zero=True")
            if 0 in hops:
                zero_idx = hops.index(0)
                hops[0], hops[zero_idx] = hops[zero_idx], hops[0]
            else:
                hops[0] = 0

        return [int(x) for x in hops]

    def validate(self) -> None:
        if self.num_games <= 0:
            raise ValueError("num_games must be > 0")
        if self.game_id_start < 0:
            raise ValueError("game_id_start must be >= 0")
        if self.start_player not in {"adversary", "controller"}:
            raise ValueError("start_player must be 'adversary' or 'controller'")
        if self.start_root_depth < 0:
            raise ValueError("start_root_depth must be >= 0")
        if self.feature_version <= 0:
            raise ValueError("feature_version must be > 0")

        if self.model_kind not in {"torch_checkpoint", "classical_joblib"}:
            raise ValueError("model_kind must be 'torch_checkpoint' or 'classical_joblib'")

        model_path = Path(self.model_checkpoint_path)
        if not model_path.exists():
            raise FileNotFoundError(f"model_checkpoint_path does not exist: {model_path}")
        if self.model_kind == "classical_joblib":
            if model_path.suffix != ".joblib":
                raise ValueError("classical_joblib model_checkpoint_path must point to a .joblib file")
            if self.environment_lang != "python":
                raise ValueError("classical_joblib tester runs must use environment_lang='python'")

        if self.arena_time_limit_sec <= 0.0:
            raise ValueError("arena_time_limit_sec must be > 0")
        if self.arena_max_controller_cleanup_steps <= 0:
            raise ValueError("arena_max_controller_cleanup_steps must be > 0")
        if self.arena_max_total_turns <= 0:
            raise ValueError("arena_max_total_turns must be > 0")

        if int(self.bootstrap_model_version) <= 0:
            raise ValueError("bootstrap_model_version must be > 0")

        if self.environment_lang not in {"python", "native"}:
            raise ValueError("environment_lang must be 'python' or 'native'")

        if self.trivial_policy.budget_tokens < 0:
            raise ValueError("trivial_policy.budget_tokens must be >= 0")
        if self.trivial_adversary_policy.heuristic not in {
            "noop",
            "first_valid",
            "first_valid_nonempty",
            "max_request_count",
            "max_prefill_tokens",
        }:
            raise ValueError(
                "trivial_adversary_policy.heuristic must be one of "
                "{noop, first_valid, first_valid_nonempty, max_request_count, max_prefill_tokens}"
            )

        gv2_cfg = self.to_pipeline_cfg().game_v2
        if self.trivial_policy.heuristic not in set(gv2_cfg.controller_action.ordering_heuristics):
            raise ValueError(
                "trivial_policy.heuristic must be one of "
                f"{gv2_cfg.controller_action.ordering_heuristics}"
            )
        if self.trivial_policy.eviction_rule not in set(gv2_cfg.controller_action.eviction_rule_names):
            raise ValueError(
                "trivial_policy.eviction_rule must be one of "
                f"{gv2_cfg.controller_action.eviction_rule_names}"
            )

        # validates hop range + unique constraints too
        _ = self.sample_history_hops()


DEFAULT_MODEL_TESTER_CONFIG = ModelTesterConfig()
