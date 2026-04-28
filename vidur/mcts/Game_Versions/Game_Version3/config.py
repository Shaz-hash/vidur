# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence
import math
import os
import random
from pathlib import Path



## GLOBAL FUNCTIONS 

def _vidur_repo_dir() -> Path:
    # /home/.../Vidur/vidur
    return Path(__file__).resolve().parents[4]

def _under_vidur(*parts: str) -> str:
    return str(_vidur_repo_dir().joinpath(*parts))

def _default_model_device() -> str:
    # try:
    #     import torch
    #     return "cuda" if torch.cuda.is_available() else "cpu"
    # except Exception:
    #     return "cpu"
    return "cpu"


def _default_train_num_threads() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, int(cpu_count)))


def _default_sim_cli_args() -> Tuple[str, ...]:
    return (
        "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
        "--replica_config_device", "a100",
        "--replica_config_network_device", "a100_dgx",
        "--cluster_config_num_replicas", "1",
        "--replica_config_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "1",
        "--global_scheduler_config_type", "round_robin",
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_batch_size_cap", "512",
        "--execution_time_predictor_config_type", "random_forest",
        "--random_forest_execution_time_predictor_config_prediction_max_tokens_per_request", "8192",
        "--random_forest_execution_time_predictor_config_prediction_max_batch_size", "256",
        "--random_forest_execution_time_predictor_config_prediction_max_prefill_chunk_size", "4096",
        "--random_forest_execution_time_predictor_config_cache_dir", _under_vidur("cache"),
        "--no-snapshot_rng_state",
    )


# add near global helpers
def _sample_hops(
    *,
    count: int,
    min_hop: int,
    max_hop: int,
    seed: int,
    ensure_zero: bool,
    unique: bool,
) -> Tuple[int, ...]:
    if count < 0:
        raise ValueError("count must be >= 0")
    if min_hop > max_hop:
        raise ValueError("min_hop cannot exceed max_hop")
    if count == 0:
        return tuple()

    rng = random.Random(int(seed))
    lo, hi = int(min_hop), int(max_hop)

    if unique:
        pop = list(range(lo, hi + 1))
        if count > len(pop):
            raise ValueError(f"Cannot sample {count} unique hops from [{lo}, {hi}]")
        if ensure_zero:
            if not (lo <= 0 <= hi):
                raise ValueError("ensure_zero=True requires 0 in hop range")
            if count == 1:
                return (0,)
            pop.remove(0)
            out = [0] + rng.sample(pop, k=count - 1)
            rng.shuffle(out)
            return tuple(int(x) for x in out)
        return tuple(int(x) for x in rng.sample(pop, k=count))

    out = [int(rng.randint(lo, hi)) for _ in range(count)]
    if ensure_zero:
        if not (lo <= 0 <= hi):
            raise ValueError("ensure_zero=True requires 0 in hop range")
        out[0] = 0
    return tuple(out)



@dataclass(frozen=True)
class CostConfig:
    # c_i = 0 (all deadlines met)
    # c_i = 1 + lateness_capped_sec (violation path)
    # c_i = 3 (drop/evict path)
    violation_base_cost: float = 1.0
    drop_cost: float = 3.0
    lateness_cap_sec: float = 2.0
    auto_drop_lateness_sec: float = 2.0


@dataclass(frozen=True)
class TimingConfig:
    # Adversary decision grid.
    adversary_tick_sec: float = 0.2

    # Sliding launch-rate window.
    launch_window_sec: float = 1.0
    max_requests_per_launch_window: int = 7

    # Numeric stability.
    eps: float = 1e-9
    time_round_digits: int = 10

    # If controller does strict no-op while prefill exists and no decode exists,
    # jump to next adversary tick.
    controller_noop_prefill_only_jump_to_next_adv_tick: bool = True


@dataclass(frozen=True)
class CreditConfig:
    # Prefill credits:
    # each adversary decision mints this many credits;
    # each minted prefill-credit lot expires after prefill_credit_expiry_sec.
    prefill_credit_mint_per_adv_tick: int = 1024
    prefill_credit_expiry_sec: float = 1.0

    # Decode credits:
    # each request transitioning prefill->decode adds this many credits.
    decode_credit_mint_per_prefill_complete: int = 216

    # Hard guard: decode credits cannot go below zero.
    enforce_nonnegative_decode_credits: bool = True


@dataclass(frozen=True)
class RequestConfig:
    # Hard per-request caps.
    max_prefill_tokens_per_request: int = 4096
    max_decode_tokens_per_request: int = 864
    min_decode_tokens_per_request: int = 1

    # Discrete prefill menu for adversary template generation.
    allowed_prefill_tokens: Tuple[int, ...] = (
        128, 256, 512, 1024, 1536, 2048, 3072, 4096
    )

    # Long-run decode average target (used for credit semantics/debug checks).
    target_decode_tokens_per_request_avg: int = 216

    # Operational target prefill average in 1s window (used for debugging/monitoring).
    target_prefill_tokens_per_request_avg_window: int = 1024


@dataclass(frozen=True)
class AdversaryActionConfig:
    # Head 1: launch count (0..max_launch_count_per_tick)
    max_launch_count_per_tick: int = 7

    # Head 3: decode stop rules.
    stop_rule_names: Tuple[str, ...] = (
        "stop_none",
        "stop_longest_decode",
        "stop_shortest_decode",
        "stop_all_decodes_over_512",
        "stop_all_decodes_over_216",
    )

    # If True, invalid sampled actions should be filtered out, not auto-clamped.
    strict_masking: bool = True


@dataclass(frozen=True)
class ControllerActionConfig:
    # Head 1: admission/eviction rule options.
    eviction_rule_names: Tuple[str, ...] = (
        "evict_none",
        "evict_largest_prefill",
        "evict_earliest_prefill_deadline",
        "evict_prefill_missed_deadline",
        "evict_prefill_lateness_over_0p5",
        "evict_longest_decode",
        "evict_decode_lateness_over_0p5",
        "evict_prefill_highest_lateness",
        "evict_decode_highest_lateness",
    )

    # Head 2: prefill budget menu.
    prefill_budget_options: Tuple[int, ...] = (
        0, 128, 256, 512, 1024, 1536, 2048, 3072, 4096
    )

    # Head 3: ordering heuristic.
    ordering_heuristics: Tuple[str, ...] = (
        "SJF",
        "EDF",
        "LST",
        "LJF",
    )



@dataclass(frozen=True)
class FeatureConfig:
    # Tensor shapes
    n_prefill_req: int = 10
    d_prefill_req: int = 10
    n_decode_req: int = 50
    d_decode_req: int = 13
    d_global: int = 24

    # Per-request normalization
    prefill_total_den: float = 4096.0
    prefill_remaining_den: float = 4096.0
    decode_total_den: float = 864.0
    decode_remaining_den: float = 864.0
    decode_processed_den: float = 864.0
    age_den_sec: float = 5.0
    lateness_den_sec: float = 2.0
    slack_den_sec: float = 2.0
    prefill_slo_den_sec: float = 2.0
    decode_slo_den_sec: float = 0.2
    objective_cost_den: float = 50.0
    total_lateness_den: float = 50.0

    # Global normalization
    system_load_den: float = 120.0
    active_prefill_count_den: float = 20.0
    active_decode_count_den: float = 100.0
    active_total_count_den: float = 120.0
    total_remaining_prefill_den: float = 20.0 * 4096.0
    total_remaining_decode_den: float = 100.0 * 864.0
    total_decode_generated_active_den: float = 100.0 * 864.0
    violated_count_den: float = 100.0
    prefill_near_drop_den: float = 20.0
    decode_near_drop_den: float = 100.0
    recent_launch_count_den: float = 7.0
    recent_launch_prefill_den: float = 7.0 * 1024.0
    decode_credit_den: float = 100.0 * 216.0

    # Near-drop bands
    near_drop_lateness_low_sec: float = 0.5
    near_drop_lateness_high_sec: float = 1.5

    # EWMA over launch counts in last 1s
    launch_ewma_alpha: float = 0.37
    launch_ewma_window_sec: float = 1.0

    # decode sampling --> used in case decodes become more than 50
    decode_sample_seed_offset: int = 1337

    

    # Features Validation
    def validate(self) -> None:
        if self.n_prefill_req <= 0 or self.n_decode_req <= 0:
            raise ValueError("feature request slots must be > 0")
        if self.d_prefill_req <= 0 or self.d_decode_req <= 0:
            raise ValueError("feature widths must be > 0")
        if self.d_global <= 0:
            raise ValueError("d_global must be > 0")
        if self.launch_ewma_alpha <= 0.0:
            raise ValueError("launch_ewma_alpha must be > 0")
        for name, value in (
            ("prefill_total_den", self.prefill_total_den),
            ("prefill_remaining_den", self.prefill_remaining_den),
            ("decode_total_den", self.decode_total_den),
            ("decode_remaining_den", self.decode_remaining_den),
            ("decode_processed_den", self.decode_processed_den),
            ("age_den_sec", self.age_den_sec),
            ("lateness_den_sec", self.lateness_den_sec),
            ("slack_den_sec", self.slack_den_sec),
            ("prefill_slo_den_sec", self.prefill_slo_den_sec),
            ("decode_slo_den_sec", self.decode_slo_den_sec),
            ("objective_cost_den", self.objective_cost_den),
            ("total_lateness_den", self.total_lateness_den),
            ("system_load_den", self.system_load_den),
            ("active_prefill_count_den", self.active_prefill_count_den),
            ("active_decode_count_den", self.active_decode_count_den),
            ("active_total_count_den", self.active_total_count_den),
            ("total_remaining_prefill_den", self.total_remaining_prefill_den),
            ("total_remaining_decode_den", self.total_remaining_decode_den),
            ("total_decode_generated_active_den", self.total_decode_generated_active_den),
            ("violated_count_den", self.violated_count_den),
            ("prefill_near_drop_den", self.prefill_near_drop_den),
            ("decode_near_drop_den", self.decode_near_drop_den),
            ("recent_launch_count_den", self.recent_launch_count_den),
            ("recent_launch_prefill_den", self.recent_launch_prefill_den),
            ("decode_credit_den", self.decode_credit_den),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if not (0.0 < self.near_drop_lateness_low_sec <= self.near_drop_lateness_high_sec):
            raise ValueError("near-drop lateness thresholds must satisfy 0 < low <= high")



@dataclass(frozen=True)
class TrainerHyperParams:
    lr: float = 2e-5
    weight_decay: float = 1e-4
    policy_weight: float = 0.0
    value_weight: float = 1.0
    value_loss_alpha: float = 50.0
    value_only: bool = True
    grad_clip_norm: float = 5.0
    checkpoint_every: int = 200
    eval_every: int = 200
    invalid_logit: float = -1e9

    def validate(self) -> None:
        if self.lr <= 0.0:
            raise ValueError("trainer.lr must be > 0")
        if self.weight_decay < 0.0:
            raise ValueError("trainer.weight_decay must be >= 0")
        if self.policy_weight < 0.0 or self.value_weight < 0.0:
            raise ValueError("trainer loss weights must be >= 0")
        if self.value_loss_alpha <= 0.0:
            raise ValueError("trainer.value_loss_alpha must be > 0")
        if self.grad_clip_norm < 0.0:
            raise ValueError("trainer.grad_clip_norm must be >= 0")


"""
    Config Settings for the Game SLO constraints etc:
"""

@dataclass(frozen=True)
class LegacyMCTSBridgeConfig:
    # Values needed by existing MCTSConstraintConfig / prefill profile plumbing.
    prefill_profile_path: str = "simulator_output/prefill_profile.csv"
    prefill_slowdown: float = 3.0
    decode_slos: Tuple[float, ...] = (50.0,)
    
    # used by mcts_tests.py
    enable_prefill_profile_time_check_in_tests: bool = True

    # Optional explicit override for profile lookup step.
    # If None, we derive from allowed_prefill_tokens via gcd.
    interval_request_size_override: Optional[int] = 128

    def validate(self) -> None:
        if not str(self.prefill_profile_path):
            raise ValueError("legacy_mcts.prefill_profile_path cannot be empty")
        if self.prefill_slowdown <= 0.0:
            raise ValueError("legacy_mcts.prefill_slowdown must be > 0")
        if not self.decode_slos:
            raise ValueError("legacy_mcts.decode_slos cannot be empty")
        if self.interval_request_size_override is not None and self.interval_request_size_override <= 0:
            raise ValueError("legacy_mcts.interval_request_size_override must be > 0")



@dataclass(frozen=True)
class MCTSSearchConfig:
    prior_value_mode: str = "model"  # "model" | "uniform"
    root_dirichlet_noise_enabled: bool = True
    root_dirichlet_alpha: float = 0.6
    root_dirichlet_epsilon: float = 0.30

    # PUCT constants.
    pb_c_base: float = 1500.0
    pb_c_init: float = 1.25

    # Time-discount config
    discount_factor: float = 0.98
    # If None: use profile-derived _prefill_step_time (current behavior).
    # If set: force this denominator in seconds for discount normalization.
    discount_time_denominator_sec: Optional[float] = 0.015725797204323228

    # Reward shaping config (already used in mctsDNN via getattr defaults)
    reward_knee: float = 25.0
    reward_max_penalty: float = 40.0
    # If None: mctsDNN computes 1/(reward_max_penalty - reward_knee)
    reward_tail_alpha: Optional[float] = None

    def validate(self) -> None:
        if self.prior_value_mode not in {"model", "uniform"}:
            raise ValueError("prior_value_mode must be 'model' or 'uniform'")
        if self.root_dirichlet_alpha < 0.0:
            raise ValueError("root_dirichlet_alpha must be >= 0")
        if not (0.0 <= self.root_dirichlet_epsilon <= 1.0):
            raise ValueError("root_dirichlet_epsilon must be in [0,1]")
        if self.pb_c_base <= 0.0:
            raise ValueError("pb_c_base must be > 0")
        if self.pb_c_init < 0.0:
            raise ValueError("pb_c_init must be >= 0")

        if not (0.0 < self.discount_factor <= 1.0):
            raise ValueError("discount_factor must be in (0, 1]")
        if self.discount_time_denominator_sec is not None and self.discount_time_denominator_sec <= 0.0:
            raise ValueError("discount_time_denominator_sec must be > 0 when set")

        if self.reward_knee < 0.0:
            raise ValueError("reward_knee must be >= 0")
        if self.reward_max_penalty <= 0.0:
            raise ValueError("reward_max_penalty must be > 0")
        if self.reward_tail_alpha is not None and self.reward_tail_alpha <= 0.0:
            raise ValueError("reward_tail_alpha must be > 0 when set")


@dataclass(frozen=True)
class ReproducibilityConfig:
    # Single seed to control model init + stochastic choices for reproducible traces.
    global_seed: int = 6
    # If True, request deterministic torch kernels where supported.
    # Keep False by default to avoid CuBLAS deterministic runtime constraints on CUDA.
    torch_deterministic: bool = False

    def validate(self) -> None:
        if self.global_seed < 0:
            raise ValueError("global_seed must be >= 0")




@dataclass(frozen=True)
class HistoryRootConfig:
    # Number of branching history actions before each train root.
    nontrivial_hops: int = 0

    # Single global history seed (no game_id/root_id derivation).
    seed: int = 0

    # Hard cap on total applied history actions (forced + branching).
    max_total_steps: int = 20000

    # Cap for forced-chain advance before each train root.
    max_forced_hops_per_root: int = 1024

    # Whether to emit history-* rows into logs.
    log_history_rows: bool = True

    def validate(self) -> None:
        if self.nontrivial_hops < 0:
            raise ValueError("history_root.nontrivial_hops must be >= 0")
        if self.seed < 0:
            raise ValueError("history_root.seed must be >= 0")
        if self.max_total_steps <= 0:
            raise ValueError("history_root.max_total_steps must be > 0")
        if self.max_forced_hops_per_root <= 0:
            raise ValueError("history_root.max_forced_hops_per_root must be > 0")


# ------------------------------
# MultiProcess Config For The Game Version 2
# -----------------------------

@dataclass(frozen=True)
class SimulationCLIGroup:
    cli_args: Tuple[str, ...] = field(default_factory=_default_sim_cli_args)


@dataclass(frozen=True)
class ModelGroup:
    device: str = field(default_factory=_default_model_device)
    # device: str = "cuda:0"


@dataclass(frozen=True)
class LoggingGroup:
    mcts_iter_log: str = _under_vidur("simulator_output", "Game_Version3", "mcts_dnn_logs", "mcts_iter.csv")
    mcts_root_log: str = _under_vidur("simulator_output", "Game_Version3", "mcts_dnn_logs", "mcts_root.csv")
    flush_every: int = 1


@dataclass(frozen=True)
class DatasetGroup:
    out_dir: str = _under_vidur("simulator_output", "Game_Version3", "mcts_dnn_dataset")
    shard_size: int = 512


@dataclass(frozen=True)
class RunGroup:
    game_id: int = 0
    root_id: int = 0
    root_depth: int = 0
    root_player: str = "adversary"
    iterations: int = 4000
    feature_version: int = 1


## CONFIG SETTINGS FOR THE EVALUATION GAMES 
@dataclass(frozen=True)
class EvaluationGroup:
    enabled: bool = False

    num_games: int = 70
    game_id_offset: int = 900_000
    random_seed_base: int = 12345

    unique_history_hops: bool = True

    # history hops sampled uniformly in [0, max_history_hops]
    max_history_hops: int = 100
    ensure_zero_hop_game: bool = True

    # NEW stop criterion
    arena_time_limit_sec: float = 5.0

    # arena MCTS search budgets
    arena_iters_adversary: int = 2000
    arena_iters_controller: int = 2000

    # safety guards only
    arena_max_total_turns_safety: int = 4096
    arena_max_controller_cleanup_steps_safety: int = 1024

    # winner rule
    arena_win_threshold: float = 0.70
    tie_points: float = 0.5

    # start state settings
    start_player: str = "adversary"
    start_root_depth: int = 0
    feature_version: int = 1


@dataclass(frozen=True)
class EvaluationLoggingGroup:
    metrics_csv: str = _under_vidur(
        "simulator_output", "Game_Version3", "mcts_dnn_logs", "eval_metrics.csv"
    )



@dataclass(frozen=True)
class MultipleProcessTrainingConfig:

    sim: SimulationCLIGroup = field(default_factory=SimulationCLIGroup)
    game_v2: GameVersion2Config = field(default_factory=lambda: DEFAULT_GAME_V2_CONFIG)
    model: ModelGroup = field(default_factory=ModelGroup)
    logging: LoggingGroup = field(default_factory=LoggingGroup)
    dataset: DatasetGroup = field(default_factory=DatasetGroup)
    run: RunGroup = field(default_factory=RunGroup)
    evaluation: EvaluationGroup = field(default_factory=EvaluationGroup)
    evaluation_logging: EvaluationLoggingGroup = field(default_factory=EvaluationLoggingGroup)


    ## Environment :
    environment_lang: str = "python"   # "python" | "native"
    
    # Special Native Settings , works if native is enabled
    native_log_events: bool = True
    native_profile: bool = False
    native_log_flush_every: int = 1
    native_torchscript_dir: str = _under_vidur("simulator_output", "Game_Version3", "torchscript")



    num_processes: int = 60
    max_concurrent_selfplay_workers: int = 60
    max_workers_per_interval: int = 3
    selfplay_dynamic_chunk_roots: int = 128
    selfplay_zero_progress_interval_patience: int = 4
    selfplay_launch_rss_limit_gb: float = 120.0
    selfplay_launch_poll_sec: float = 2.0
    num_generations: int = 20
    roots_per_generation: int = 40000
    sample_cycles_per_generation: int = 8

    adv_iterations_per_root: int = 4000
    cont_iterations_per_root: int = 4000
    max_batch_size: int = 256

    train_steps_per_generation: int = 136
    train_batch_size: int = 256
    train_target_epochs_per_generation: float = 10.0
    train_progress_print_every_steps: int = 100
    train_num_threads: int = field(default_factory=_default_train_num_threads)

    history_seed: int = 0
    history_hops_min: int = 0
    history_hops_max: int = 199
    history_hops_unique_per_generation: bool = False
    history_hops_force_zero: bool = True
    history_hop_interval_width: int = 5
    # Optional explicit per-worker (min_hop, max_hop). If empty, intervals are auto-built.
    history_hops_per_worker: Tuple[Tuple[int, int], ...] = tuple()
    max_forced_hops_per_root: int = 1024
    history_max_total_steps: int = 20000
    history_root_batch_size: int = 64
    history_allow_duplicate_root_fallback: bool = False
    virtual_simulator_max_free_request_pool: int = 300
    log_history_rows: bool = True
    eval_split_ratio: float = 0.1
    eval_split_seed_base: int = 0

    sample_from_mcts_policy: bool = True
    selfplay_policy_temperature: float = 2.0 # increase the temperature for more exploration in the sampled actions from the mcts policy during self-play, which can lead to more diverse training data and potentially better generalization of the trained model. Tune this parameter based on the desired level of exploration vs exploitation in the self-play data generation.
    action_seed_base: int = 4 ## used for the dirichlet noise + potential future stochasticity in action sampling in root creation for training

    replay_capacity_samples: int = 400000
    replay_max_cached_shards: int = 5000
    replay_preload_all_shards_on_reuse: bool = True
    replay_preload_all_shards_each_generation: bool = True
    replay_preload_max_shards: int = 5000
    replay_seed: int = 2026

    checkpoints_dir: str = _under_vidur("simulator_output", "Game_Version3", "mcts_dnn_checkpoints")
    train_metrics_csv: str = _under_vidur("simulator_output", "Game_Version3", "mcts_dnn_logs", "train_metrics.csv")
    
    use_virtual_env: bool = True
    worker_result_timeout_sec: int = 7200
    reuse_dataset_generation: int = -1
    reuse_dataset_generation_only_once: bool = False
    reuse_dataset_reset_replay_each_generation: bool = True

    def selfplay_hop_ranges_for_generation(self, generation: int) -> Tuple[Tuple[int, int], ...]:
        del generation
        if len(self.history_hops_per_worker) > 0:
            if len(self.history_hops_per_worker) != int(self.num_processes):
                raise ValueError(
                    "history_hops_per_worker length must equal num_processes "
                    f"({len(self.history_hops_per_worker)} != {self.num_processes})"
                )
            return tuple((int(lo), int(hi)) for lo, hi in self.history_hops_per_worker)

        width = max(1, int(self.history_hop_interval_width))
        start = int(self.history_hops_min)
        out = []
        for wid in range(int(self.num_processes)):
            lo = start + wid * width
            hi = lo + width - 1
            out.append((int(lo), int(hi)))
        return tuple(out)

    # Backward-compatible helper for old callsites.
    def selfplay_hops_for_generation(self, generation: int) -> Tuple[int, ...]:
        rng = random.Random(int(self.history_seed) + int(generation))
        ranges = self.selfplay_hop_ranges_for_generation(generation)
        hops = [int(rng.randint(int(lo), int(hi))) for lo, hi in ranges]
        if bool(self.history_hops_force_zero) and hops:
            lo0, hi0 = ranges[0]
            hops[0] = 0 if int(lo0) <= 0 <= int(hi0) else int(lo0)
        return tuple(hops)

    def arena_hops_for_generation(self, generation: int, num_games: int) -> Tuple[int, ...]:
        return _sample_hops(
            count=int(num_games),
            min_hop=0,
            max_hop=int(self.evaluation.max_history_hops),
            seed=int(self.evaluation.random_seed_base) + int(generation),
            ensure_zero=bool(self.evaluation.ensure_zero_hop_game),
            unique=bool(self.evaluation.unique_history_hops),
        )

    def validate(self) -> None:
        self.game_v2.validate()

        if self.environment_lang not in {"python", "native"}:
            raise ValueError("environment_lang must be 'python' or 'native'")
        if self.native_log_flush_every <= 0:
            raise ValueError("native_log_flush_every must be > 0")
            
        if self.num_processes <= 0:
            raise ValueError("num_processes must be > 0")
        if int(self.max_concurrent_selfplay_workers) <= 0:
            raise ValueError("max_concurrent_selfplay_workers must be > 0")
        if int(self.max_concurrent_selfplay_workers) > int(self.num_processes):
            raise ValueError("max_concurrent_selfplay_workers cannot exceed num_processes")
        if int(self.max_workers_per_interval) <= 0:
            raise ValueError("max_workers_per_interval must be > 0")
        if int(self.max_workers_per_interval) > int(self.max_concurrent_selfplay_workers):
            raise ValueError("max_workers_per_interval cannot exceed max_concurrent_selfplay_workers")
        if int(self.selfplay_dynamic_chunk_roots) <= 0:
            raise ValueError("selfplay_dynamic_chunk_roots must be > 0")
        if int(self.selfplay_zero_progress_interval_patience) <= 0:
            raise ValueError("selfplay_zero_progress_interval_patience must be > 0")
        if float(self.selfplay_launch_rss_limit_gb) <= 0.0:
            raise ValueError("selfplay_launch_rss_limit_gb must be > 0")
        if float(self.selfplay_launch_poll_sec) <= 0.0:
            raise ValueError("selfplay_launch_poll_sec must be > 0")
        if self.num_generations <= 0:
            raise ValueError("num_generations must be > 0")
        if self.roots_per_generation <= 0:
            raise ValueError("roots_per_generation must be > 0")
        if self.sample_cycles_per_generation <= 0:
            raise ValueError("sample_cycles_per_generation must be > 0")

        if self.adv_iterations_per_root <= 0 or self.cont_iterations_per_root <= 0:
            raise ValueError("adv_iterations_per_root and cont_iterations_per_root must be > 0")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")

        if self.train_steps_per_generation <= 0:
            raise ValueError("train_steps_per_generation must be > 0")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be > 0")
        if float(self.train_target_epochs_per_generation) <= 0.0:
            raise ValueError("train_target_epochs_per_generation must be > 0")
        if int(self.train_progress_print_every_steps) <= 0:
            raise ValueError("train_progress_print_every_steps must be > 0")
        if int(self.train_num_threads) <= 0:
            raise ValueError("train_num_threads must be > 0")

        if self.history_seed < 0:
            raise ValueError("history_seed must be >= 0")
        if self.history_hop_interval_width <= 0:
            raise ValueError("history_hop_interval_width must be > 0")
        if len(self.history_hops_per_worker) > 0:
            if len(self.history_hops_per_worker) != self.num_processes:
                raise ValueError(
                    "history_hops_per_worker length must equal num_processes "
                    f"({len(self.history_hops_per_worker)} != {self.num_processes})"
                )
            for i, pair in enumerate(self.history_hops_per_worker):
                if len(pair) != 2:
                    raise ValueError(f"history_hops_per_worker[{i}] must be (min_hop, max_hop)")
                lo, hi = int(pair[0]), int(pair[1])
                if lo < 0:
                    raise ValueError(f"history_hops_per_worker[{i}] min_hop must be >= 0")
                if hi < lo:
                    raise ValueError(f"history_hops_per_worker[{i}] max_hop must be >= min_hop")
        if self.max_forced_hops_per_root <= 0:
            raise ValueError("max_forced_hops_per_root must be > 0")
        if self.history_max_total_steps <= 0:
            raise ValueError("history_max_total_steps must be > 0")
        if self.history_root_batch_size <= 0:
            raise ValueError("history_root_batch_size must be > 0")
        if self.virtual_simulator_max_free_request_pool < 0:
            raise ValueError("virtual_simulator_max_free_request_pool must be >= 0")
        if not (0.0 <= float(self.eval_split_ratio) <= 1.0):
            raise ValueError("eval_split_ratio must be in [0, 1]")
        if int(self.eval_split_seed_base) < 0:
            raise ValueError("eval_split_seed_base must be >= 0")

        if self.replay_capacity_samples <= 0:
            raise ValueError("replay_capacity_samples must be > 0")
        if self.replay_max_cached_shards <= 0:
            raise ValueError("replay_max_cached_shards must be > 0")
        if self.replay_preload_max_shards <= 0:
            raise ValueError("replay_preload_max_shards must be > 0")
        if self.replay_seed < 0:
            raise ValueError("replay_seed must be >= 0")

        if self.logging.flush_every <= 0:
            raise ValueError("logging.flush_every must be > 0")
        if self.dataset.shard_size <= 0:
            raise ValueError("dataset.shard_size must be > 0")
        if self.run.iterations <= 0:
            raise ValueError("run.iterations must be > 0")
        if self.run.feature_version <= 0:
            raise ValueError("run.feature_version must be > 0")

        if self.worker_result_timeout_sec <= 0:
            raise ValueError("worker_result_timeout_sec must be > 0")
        if int(self.reuse_dataset_generation) < -1:
            raise ValueError("reuse_dataset_generation must be >= -1")


        if self.history_hops_min < 0:
            raise ValueError("history_hops_min must be >= 0")
        if self.history_hops_max < self.history_hops_min:
            raise ValueError("history_hops_max must be >= history_hops_min")
        if self.history_hops_force_zero and len(self.history_hops_per_worker) == 0 and not (self.history_hops_min <= 0 <= self.history_hops_max):
            raise ValueError("history hop range must include 0 when history_hops_force_zero=True")

        if self.evaluation.unique_history_hops:
            if self.evaluation.num_games > (self.evaluation.max_history_hops + 1):
                raise ValueError("evaluation.num_games exceeds unique arena history-hop capacity")

        # Evaluation config validation
        if self.evaluation.enabled:
            if self.evaluation.num_games <= 0:
                raise ValueError("evaluation.num_games must be > 0")
            if self.evaluation.game_id_offset < 0:
                raise ValueError("evaluation.game_id_offset must be >= 0")
            if self.evaluation.random_seed_base < 0:
                raise ValueError("evaluation.random_seed_base must be >= 0")
            if self.evaluation.max_history_hops < 0:
                raise ValueError("evaluation.max_history_hops must be >= 0")
            if self.evaluation.arena_time_limit_sec <= 0.0:
                raise ValueError("evaluation.arena_time_limit_sec must be > 0")
            if self.evaluation.arena_iters_adversary <= 0 or self.evaluation.arena_iters_controller <= 0:
                raise ValueError("evaluation arena iterations must be > 0")
            if self.evaluation.arena_max_total_turns_safety <= 0:
                raise ValueError("evaluation.arena_max_total_turns_safety must be > 0")
            if self.evaluation.arena_max_controller_cleanup_steps_safety <= 0:
                raise ValueError("evaluation.arena_max_controller_cleanup_steps_safety must be > 0")
            if not (0.0 <= self.evaluation.arena_win_threshold <= 1.0):
                raise ValueError("evaluation.arena_win_threshold must be in [0, 1]")
            if not (0.0 <= self.evaluation.tie_points <= 1.0):
                raise ValueError("evaluation.tie_points must be in [0, 1]")
            if self.evaluation.start_player not in ("adversary", "controller"):
                raise ValueError("evaluation.start_player must be 'adversary' or 'controller'")
            if self.evaluation.feature_version <= 0:
                raise ValueError("evaluation.feature_version must be > 0")



@dataclass(frozen=True)
class GameVersion2Config:

    def derived_interval_request_size(self) -> int:
        if self.legacy_mcts.interval_request_size_override is not None:
            return int(self.legacy_mcts.interval_request_size_override)

        vals = [int(x) for x in self.request.allowed_prefill_tokens]
        step = vals[0]
        for v in vals[1:]:
            step = math.gcd(step, v)
        return max(1, int(step))

    def derived_min_request_tokens(self) -> int:
        return int(min(self.request.allowed_prefill_tokens))

    def derived_max_request_tokens(self) -> int:
        return int(self.request.max_prefill_tokens_per_request)


    timing: TimingConfig = field(default_factory=TimingConfig)
    credits: CreditConfig = field(default_factory=CreditConfig)
    request: RequestConfig = field(default_factory=RequestConfig)
    adversary_action: AdversaryActionConfig = field(default_factory=AdversaryActionConfig)
    controller_action: ControllerActionConfig = field(default_factory=ControllerActionConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    trainer: TrainerHyperParams = field(default_factory=TrainerHyperParams)
    mcts_search: MCTSSearchConfig = field(default_factory=MCTSSearchConfig)
    legacy_mcts: LegacyMCTSBridgeConfig = field(default_factory=LegacyMCTSBridgeConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    history_root: HistoryRootConfig = field(default_factory=HistoryRootConfig)

    # Optional debugging guards.
    enable_debug_asserts: bool = True
    fail_fast_on_invalid_state: bool = True


    def validate(self) -> None:
        # Timing
        if self.timing.adversary_tick_sec <= 0.0:
            raise ValueError("adversary_tick_sec must be > 0")
        if self.timing.launch_window_sec <= 0.0:
            raise ValueError("launch_window_sec must be > 0")
        if self.timing.max_requests_per_launch_window <= 0:
            raise ValueError("max_requests_per_launch_window must be > 0")
        if self.timing.eps <= 0.0:
            raise ValueError("eps must be > 0")

        # Credits
        if self.credits.prefill_credit_mint_per_adv_tick <= 0:
            raise ValueError("prefill_credit_mint_per_adv_tick must be > 0")
        if self.credits.prefill_credit_expiry_sec <= 0.0:
            raise ValueError("prefill_credit_expiry_sec must be > 0")
        if self.credits.decode_credit_mint_per_prefill_complete <= 0:
            raise ValueError("decode_credit_mint_per_prefill_complete must be > 0")

        # Requests
        if self.request.max_prefill_tokens_per_request <= 0:
            raise ValueError("max_prefill_tokens_per_request must be > 0")
        if self.request.max_decode_tokens_per_request <= 0:
            raise ValueError("max_decode_tokens_per_request must be > 0")
        if self.request.min_decode_tokens_per_request <= 0:
            raise ValueError("min_decode_tokens_per_request must be > 0")
        if self.request.min_decode_tokens_per_request > self.request.max_decode_tokens_per_request:
            raise ValueError("min_decode_tokens_per_request cannot exceed max_decode_tokens_per_request")
        if not self.request.allowed_prefill_tokens:
            raise ValueError("allowed_prefill_tokens cannot be empty")
        if any(t <= 0 for t in self.request.allowed_prefill_tokens):
            raise ValueError("all allowed_prefill_tokens must be > 0")
        if any(t > self.request.max_prefill_tokens_per_request for t in self.request.allowed_prefill_tokens):
            raise ValueError("allowed_prefill_tokens cannot exceed max_prefill_tokens_per_request")

        # Adversary action
        if self.adversary_action.max_launch_count_per_tick < 0:
            raise ValueError("max_launch_count_per_tick must be >= 0")
        if self.adversary_action.max_launch_count_per_tick > self.timing.max_requests_per_launch_window:
            raise ValueError(
                "max_launch_count_per_tick cannot exceed max_requests_per_launch_window"
            )
        if not self.adversary_action.stop_rule_names:
            raise ValueError("stop_rule_names cannot be empty")

        # Controller action
        if not self.controller_action.eviction_rule_names:
            raise ValueError("eviction_rule_names cannot be empty")
        if not self.controller_action.prefill_budget_options:
            raise ValueError("prefill_budget_options cannot be empty")
        if any(b < 0 for b in self.controller_action.prefill_budget_options):
            raise ValueError("prefill_budget_options cannot contain negative values")
        if any(b > self.request.max_prefill_tokens_per_request for b in self.controller_action.prefill_budget_options):
            raise ValueError("prefill_budget_options cannot exceed max_prefill_tokens_per_request")
        if not self.controller_action.ordering_heuristics:
            raise ValueError("ordering_heuristics cannot be empty")

        # Cost
        if self.cost.violation_base_cost < 0.0:
            raise ValueError("violation_base_cost must be >= 0")
        if self.cost.drop_cost < 0.0:
            raise ValueError("drop_cost must be >= 0")
        if self.cost.lateness_cap_sec <= 0.0:
            raise ValueError("lateness_cap_sec must be > 0")
        if self.cost.auto_drop_lateness_sec <= 0.0:
            raise ValueError("auto_drop_lateness_sec must be > 0")
        if self.cost.drop_cost < (self.cost.violation_base_cost + self.cost.lateness_cap_sec):
            # You can relax this if you want drop == max violation cost.
            raise ValueError("drop_cost should be >= violation_base_cost + lateness_cap_sec")

        # Features 
        self.features.validate()
        self.trainer.validate()


        # Constraints
        self.legacy_mcts.validate()
        if self.derived_min_request_tokens() > self.derived_max_request_tokens():
            raise ValueError("derived min_request_tokens cannot exceed derived max_request_tokens")

        # MCTS Search
        self.mcts_search.validate()
        self.reproducibility.validate()


        # History Generation
        self.history_root.validate()





DEFAULT_GAME_V2_CONFIG = GameVersion2Config()
DEFAULT_GAME_V2_CONFIG.validate()

# TODO : Call these lines inside the GAME_V2_CONFIG
DEFAULT_MULTIPROCESS_TRAINING_CONFIG = MultipleProcessTrainingConfig()
DEFAULT_MULTIPROCESS_TRAINING_CONFIG.validate()
