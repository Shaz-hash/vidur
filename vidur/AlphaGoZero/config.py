"""Configuration defaults for GV3 AlphaGoZero experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_OUTPUT_ROOT = REPO_ROOT / "simulator_output" / "GV3_Agent"
ALPHAGOZERO_OUTPUT_ROOT = SIM_OUTPUT_ROOT / "AlphaGoZero"

# Initial deployment uses the single classical 200k HGB family.
DEFAULT_VALUE_MODEL_PATH = (
    SIM_OUTPUT_ROOT
    / "bellman_multiserver_HGB"
    / "latest_version"
    / "worker1_200k_alpha2"
    / "hgb_sq_63leaf_1050iter_a2"
    / "Model_Version100"
    / "model.joblib"
)
DEFAULT_CONTROLLER_PRIOR_MODEL_PATH = (
    SIM_OUTPUT_ROOT
    / "bellman_multiserver_HGB"
    / "bellman_prior_controller_adv_train_eval_results"
    / "controller"
    / "hgb_policy_63leaf_1050iter"
    / "Model_Version1"
    / "model.joblib"
)
DEFAULT_ADVERSARY_PRIOR_MODEL_PATH = (
    SIM_OUTPUT_ROOT
    / "bellman_multiserver_HGB"
    / "bellman_prior_controller_adv_train_eval_results"
    / "adversary"
    / "hgb_policy_63leaf_1050iter"
    / "Model_Version1"
    / "model.joblib"
)

PROMOTION_WIN_RATE_THRESHOLD = 0.57
MAX_PROMOTIONS = 50
TRAIN_TRIGGER_NEW_STATES = 200_000
MIN_ADVERSARY_STATES_FOR_EVAL = 25_000
ADVERSARY_POLICY_SAMPLE_CAP = 50_000
TRAIN_SAMPLE_MIN_MID_REPLAY = 50_000
TRAIN_SAMPLE_MIN_LARGE_REPLAY = 100_000
XL_MAX_REPLAY_STATES = 5_000_000
XL_KEEP_ACCEPTED_SHARDS = False
XL_WRITE_AUDIT_REPLAY = False
XL_CONTROLLER_POLICY_SAMPLE_CAP = 200_000


@dataclass(frozen=True)
class Phase1SmokeConfig:
    """Single-machine sanity-test settings for native value+prior MCTS arena play."""

    output_root: Path = ALPHAGOZERO_OUTPUT_ROOT / "phase1_local_smoke"
    value_model_path: Path = DEFAULT_VALUE_MODEL_PATH
    controller_prior_model_path: Path = DEFAULT_CONTROLLER_PRIOR_MODEL_PATH
    adversary_prior_model_path: Path = DEFAULT_ADVERSARY_PRIOR_MODEL_PATH
    model_version: int = 100
    feature_dim: int = 226
    game_id: int = 17_000_000
    num_games: int = 1
    parallel_games: int = 1
    history_hops: int = 0
    mcts_iterations: int = 1_000
    arena_time_limit_sec: float = 5.0
    trivial_budget_tokens: int = 256
    puct_c: float = 1.5
    uct_c: float = 1.4
    policy_prior_temperature: float = 1.0
    prior_min_prob: float = 1e-8
    root_dirichlet_noise_enabled: bool = True
    root_dirichlet_alpha: float = 0.03
    root_dirichlet_epsilon: float = 0.25
    agz_sample_initial_moves: bool = True
    agz_sample_initial_move_count: int = 30
    agz_mcts_action_temperature: float = 1.0
    seed: int = 2026
    worker_threads: int = 1
