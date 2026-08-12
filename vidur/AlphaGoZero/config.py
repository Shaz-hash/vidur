"""Configuration defaults for GV3 AlphaGoZero experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_OUTPUT_ROOT = REPO_ROOT / "simulator_output" / "GV3_Agent"
ALPHAGOZERO_OUTPUT_ROOT = SIM_OUTPUT_ROOT / "AlphaGoZero"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    return float(raw)

def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return str(default)
    return str(raw).strip()


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

PROMOTION_WIN_RATE_THRESHOLD = _env_float("AGZ_PROMOTION_WIN_RATE_THRESHOLD", 0.55)
ROLE_PROMOTION_WIN_THRESHOLD = _env_int("AGZ_ROLE_PROMOTION_WIN_THRESHOLD", 55)
MAX_PROMOTIONS = _env_int("AGZ_MAX_PROMOTIONS", 50)
TRAIN_TRIGGER_NEW_STATES = _env_int("AGZ_TRAIN_TRIGGER_NEW_STATES", 600_000)
MIN_CONTROLLER_STATES_FOR_EVAL = _env_int("AGZ_MIN_CONTROLLER_STATES_FOR_EVAL", 250_000)
MIN_ADVERSARY_STATES_FOR_EVAL = _env_int("AGZ_MIN_ADVERSARY_STATES_FOR_EVAL", 25_000)
ADVERSARY_POLICY_SAMPLE_CAP = _env_int("AGZ_ADVERSARY_POLICY_SAMPLE_CAP", 250_000)
ADVERSARY_POLICY_ROOT_SAMPLE_TARGET = _env_int("AGZ_ADVERSARY_POLICY_ROOT_SAMPLE_TARGET", 0)
TRAIN_SAMPLE_MIN_MID_REPLAY = _env_int("AGZ_TRAIN_SAMPLE_MIN_MID_REPLAY", 50_000)
TRAIN_SAMPLE_MIN_LARGE_REPLAY = _env_int("AGZ_TRAIN_SAMPLE_MIN_LARGE_REPLAY", 100_000)
XL_MAX_REPLAY_STATES = _env_int("AGZ_XL_MAX_REPLAY_STATES", 35_000_000)
XL_KEEP_ACCEPTED_SHARDS = False
# Zero drains every durably published shard on each coordinator pass. Ingestion
# stays live during training/evaluation; pruning can safely catch up afterward.
XL_MAX_INGEST_SHARDS_PER_LOOP = _env_int("AGZ_XL_MAX_INGEST_SHARDS_PER_LOOP", 0)
XL_PAUSE_INGEST_WHILE_PRUNE_DEFERRED = bool(_env_int("AGZ_XL_PAUSE_INGEST_WHILE_PRUNE_DEFERRED", 0))
XL_WRITE_AUDIT_REPLAY = False
XL_CONTROLLER_POLICY_SAMPLE_CAP = _env_int("AGZ_CONTROLLER_POLICY_SAMPLE_CAP", 250_000)
XL_CONTROLLER_MAX_REPLAY_STATES = _env_int("AGZ_XL_CONTROLLER_MAX_REPLAY_STATES", 20_000_000)
XL_ADVERSARY_MAX_REPLAY_STATES = _env_int("AGZ_XL_ADVERSARY_MAX_REPLAY_STATES", 15_000_000)
POLICY_ROOT_OVERSAMPLE_FACTOR = _env_float("AGZ_POLICY_ROOT_OVERSAMPLE_FACTOR", 1.25)
POLICY_CACHE_BUILD_WORKERS = _env_int("AGZ_POLICY_CACHE_BUILD_WORKERS", min(60, max(1, os.cpu_count() or 1)))
POLICY_METRICS_WORKERS = _env_int(
    "AGZ_POLICY_METRICS_WORKERS",
    min(80, max(1, os.cpu_count() or 1)),
)
AGZ_REPLAY_SAMPLER = _env_str("AGZ_REPLAY_SAMPLER", "legacy_cached").lower()
AGZ_REPLAY_INDEX_BUILD_WORKERS = _env_int(
    "AGZ_REPLAY_INDEX_BUILD_WORKERS",
    min(60, max(1, os.cpu_count() or 1)),
)
AGZ_REPLAY_EXTRACTION_WORKERS = _env_int(
    "AGZ_REPLAY_EXTRACTION_WORKERS",
    min(16, max(1, os.cpu_count() or 1)),
)
AGZ_SPOT_CONTROLLER_RETRY_ATTEMPTS = _env_int(
    "AGZ_SPOT_CONTROLLER_RETRY_ATTEMPTS", 12
)
AGZ_SPOT_CONTROLLER_RETRY_INITIAL_SEC = _env_float(
    "AGZ_SPOT_CONTROLLER_RETRY_INITIAL_SEC", 1.0
)
AGZ_SPOT_CONTROLLER_RETRY_MAX_SEC = _env_float(
    "AGZ_SPOT_CONTROLLER_RETRY_MAX_SEC", 30.0
)
AGZ_SPOT_PUBLISH_RETRY_ATTEMPTS = _env_int(
    "AGZ_SPOT_PUBLISH_RETRY_ATTEMPTS", 8
)
AGZ_SPOT_WORKER_ERROR_BACKOFF_SEC = _env_float(
    "AGZ_SPOT_WORKER_ERROR_BACKOFF_SEC", 5.0
)
AGZ_SPOT_RESULT_RECONCILE_ENABLED = bool(
    _env_int("AGZ_SPOT_RESULT_RECONCILE_ENABLED", 1)
)
AGZ_EVAL_GAMES = _env_int("AGZ_EVAL_GAMES", 100)
AGZ_BENCHMARK_GAMES = _env_int("AGZ_BENCHMARK_GAMES", 50)
AGZ_MODEL_FAMILY = _env_str("AGZ_MODEL_FAMILY", "hgb").lower()
AGZ_VALUE_FEATURE_SCHEMA = _env_str("AGZ_VALUE_FEATURE_SCHEMA", "legacy_226").lower()
AGZ_POLICY_FEATURE_SCHEMA = _env_str("AGZ_POLICY_FEATURE_SCHEMA", "legacy_226").lower()
AGZ_DNN_EPOCHS = _env_int("AGZ_DNN_EPOCHS", 5)
AGZ_DNN_VALUE_BATCH_SIZE = _env_int("AGZ_DNN_VALUE_BATCH_SIZE", 4096)
AGZ_DNN_POLICY_ROOT_BATCH_SIZE = _env_int("AGZ_DNN_POLICY_ROOT_BATCH_SIZE", 256)
AGZ_DNN_INITIAL_LR = _env_float("AGZ_DNN_INITIAL_LR", 3e-4)
AGZ_DNN_UPDATE_LR = _env_float("AGZ_DNN_UPDATE_LR", 1e-4)
AGZ_DNN_TORCH_THREADS_PER_MODEL = _env_int("AGZ_DNN_TORCH_THREADS_PER_MODEL", 24)
AGZ_INITIAL_SAMPLE_MOVE_COUNT = _env_int("AGZ_INITIAL_SAMPLE_MOVE_COUNT", 20)
AGZ_MCTS_ITERATIONS = _env_int("AGZ_MCTS_ITERATIONS", 1_000)
AGZ_EVAL_MCTS_ITERATIONS = _env_int("AGZ_EVAL_MCTS_ITERATIONS", 1_000)
AGZ_SELFPLAY_PUCT_C = _env_float("AGZ_SELFPLAY_PUCT_C", 2.5)
AGZ_EVAL_PUCT_C = _env_float("AGZ_EVAL_PUCT_C", 1.0)
AGZ_SJF_PUCT_C = _env_float("AGZ_SJF_PUCT_C", AGZ_EVAL_PUCT_C)
AGZ_SELFPLAY_ARENA_TIME_LIMIT_SEC = _env_float("AGZ_SELFPLAY_ARENA_TIME_LIMIT_SEC", 20.0)
AGZ_REPLAY_SAMPLE_WINDOW_SEC = _env_float("AGZ_REPLAY_SAMPLE_WINDOW_SEC", 0.0)
AGZ_MAX_ADVERSARY_VALUE_STATES = _env_int("AGZ_MAX_ADVERSARY_VALUE_STATES", 300_000)
AGZ_ROOT_DIRICHLET_ALPHA = _env_float("AGZ_ROOT_DIRICHLET_ALPHA", 0.05)
AGZ_ROOT_DIRICHLET_TOTAL_CONCENTRATION = _env_float(
    "AGZ_ROOT_DIRICHLET_TOTAL_CONCENTRATION", 0.0
)
AGZ_ROOT_DIRICHLET_EPSILON = _env_float("AGZ_ROOT_DIRICHLET_EPSILON", 0.25)
AGZ_DISCOUNT_FACTOR = _env_float("AGZ_DISCOUNT_FACTOR", 0.995)
AGZ_NATIVE_SEARCH_MODE = _env_str("AGZ_NATIVE_SEARCH_MODE", "full_tree")
AGZ_ROLLOUT_COUNT = _env_int("AGZ_ROLLOUT_COUNT", 10)
AGZ_EVAL_ROLLOUT_COUNT = _env_int("AGZ_EVAL_ROLLOUT_COUNT", AGZ_ROLLOUT_COUNT)
AGZ_SJF_ROLLOUT_COUNT = _env_int("AGZ_SJF_ROLLOUT_COUNT", AGZ_EVAL_ROLLOUT_COUNT)
AGZ_ROLLOUT_PARALLEL_THREADS = _env_int("AGZ_ROLLOUT_PARALLEL_THREADS", 1)
AGZ_EVAL_ROLLOUT_PARALLEL_THREADS = _env_int(
    "AGZ_EVAL_ROLLOUT_PARALLEL_THREADS",
    AGZ_ROLLOUT_PARALLEL_THREADS,
)
AGZ_SJF_ROLLOUT_PARALLEL_THREADS = _env_int(
    "AGZ_SJF_ROLLOUT_PARALLEL_THREADS",
    AGZ_EVAL_ROLLOUT_PARALLEL_THREADS,
)
AGZ_ROLLOUT_HORIZON_SEC = _env_float("AGZ_ROLLOUT_HORIZON_SEC", 0.4)
AGZ_ADAPTIVE_ROLLOUT_HORIZON = bool(
    _env_int("AGZ_ADAPTIVE_ROLLOUT_HORIZON", 0)
)
AGZ_ROLLOUT_MAX_HORIZON_SEC = _env_float(
    "AGZ_ROLLOUT_MAX_HORIZON_SEC",
    AGZ_ROLLOUT_HORIZON_SEC,
)
AGZ_ROLLOUT_VALUE_ERROR_THRESHOLD = _env_float(
    "AGZ_ROLLOUT_VALUE_ERROR_THRESHOLD",
    0.09,
)
AGZ_ROLLOUT_HORIZON_TICK_SEC = _env_float(
    "AGZ_ROLLOUT_HORIZON_TICK_SEC",
    0.2,
)
AGZ_ROLLOUT_REFERENCE_STEP_SEC = _env_float(
    "AGZ_ROLLOUT_REFERENCE_STEP_SEC",
    0.015725797204323228,
)
AGZ_ROLLOUT_POLICY_TEMPERATURE = _env_float("AGZ_ROLLOUT_POLICY_TEMPERATURE", 1.0)
AGZ_ROLLOUT_PROBABILITY_QUANTUM = _env_float("AGZ_ROLLOUT_PROBABILITY_QUANTUM", 1e-6)
AGZ_ROLLOUT_MAX_ACTIONS = _env_int("AGZ_ROLLOUT_MAX_ACTIONS", 4096)


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
    mcts_iterations: int = AGZ_MCTS_ITERATIONS
    discount_factor: float = AGZ_DISCOUNT_FACTOR
    arena_time_limit_sec: float = AGZ_SELFPLAY_ARENA_TIME_LIMIT_SEC
    replay_sample_window_sec: float = AGZ_REPLAY_SAMPLE_WINDOW_SEC
    trivial_budget_tokens: int = 256
    puct_c: float = AGZ_SELFPLAY_PUCT_C
    uct_c: float = 1.4
    policy_prior_temperature: float = 1.0
    prior_min_prob: float = 1e-8
    root_dirichlet_noise_enabled: bool = True
    root_dirichlet_alpha: float = AGZ_ROOT_DIRICHLET_ALPHA
    root_dirichlet_total_concentration: float = AGZ_ROOT_DIRICHLET_TOTAL_CONCENTRATION
    root_dirichlet_epsilon: float = AGZ_ROOT_DIRICHLET_EPSILON
    agz_sample_initial_moves: bool = True
    agz_sample_initial_move_count: int = AGZ_INITIAL_SAMPLE_MOVE_COUNT
    agz_mcts_action_temperature: float = 1.0
    seed: int = 2026
    native_search_mode: str = AGZ_NATIVE_SEARCH_MODE
    rollout_count: int = AGZ_ROLLOUT_COUNT
    rollout_parallel_threads: int = AGZ_ROLLOUT_PARALLEL_THREADS
    rollout_horizon_sec: float = AGZ_ROLLOUT_HORIZON_SEC
    rollout_policy_temperature: float = AGZ_ROLLOUT_POLICY_TEMPERATURE
    rollout_probability_quantum: float = AGZ_ROLLOUT_PROBABILITY_QUANTUM
    rollout_max_actions: int = AGZ_ROLLOUT_MAX_ACTIONS
    worker_threads: int = 1
