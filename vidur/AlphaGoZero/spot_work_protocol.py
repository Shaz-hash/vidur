"""Durable pull-based work queue for ephemeral AlphaGoZero Spot workers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from vidur.AlphaGoZero.config import AGZ_SPOT_RESULT_RECONCILE_ENABLED
from vidur.AlphaGoZero.durable_transfer import (
    atomic_write_json,
    sha256_file,
    utc_now,
    verify_sha256sums,
)


SCHEMA_VERSION = 3
SELFPLAY_CONFIG_SCHEMA_VERSION = 1
DEFAULT_WORKER_HEARTBEAT_TIMEOUT_SEC = 300
GIB = 1024**3
MODEL_PATH_KEYS = (
    "controller_value_model_path",
    "adversary_value_model_path",
    "controller_prior_model_path",
    "adversary_prior_model_path",
)

SELFPLAY_CONFIG_INT_FIELDS = (
    "feature_dim",
    "iterations",
    "history_hops",
    "trivial_budget_tokens",
    "agz_sample_initial_move_count",
    "seed",
    "worker_threads",
    "rollout_count",
    "rollout_parallel_threads",
    "rollout_max_actions",
    "buffer_threshold",
    "parent_state_id",
    "parent_state_count",
)
SELFPLAY_CONFIG_FLOAT_FIELDS = (
    "discount_factor",
    "arena_time_limit_sec",
    "replay_sample_window_sec",
    "puct_c",
    "uct_c",
    "policy_prior_temperature",
    "prior_min_prob",
    "root_dirichlet_alpha",
    "root_dirichlet_total_concentration",
    "root_dirichlet_epsilon",
    "agz_mcts_action_temperature",
    "rollout_horizon_sec",
    "rollout_policy_temperature",
    "rollout_probability_quantum",
)
SELFPLAY_CONFIG_BOOL_FIELDS = (
    "root_dirichlet_noise_enabled",
    "agz_sample_initial_moves",
)
SELFPLAY_CONFIG_STR_FIELDS = (
    "native_search_mode",
    "parent_dataset_dir",
    "parent_root_player_filter",
)
SELFPLAY_CONFIG_FIELDS = (
    *SELFPLAY_CONFIG_INT_FIELDS,
    *SELFPLAY_CONFIG_FLOAT_FIELDS,
    *SELFPLAY_CONFIG_BOOL_FIELDS,
    *SELFPLAY_CONFIG_STR_FIELDS,
)


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_selfplay_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate the complete coordinator-owned self-play config."""

    expected = {"config_schema_version", *SELFPLAY_CONFIG_FIELDS}
    received = set(raw)
    missing = sorted(expected - received)
    unknown = sorted(received - expected)
    if missing or unknown:
        raise ValueError(
            f"invalid self-play config fields: missing={missing}, unknown={unknown}"
        )
    config: dict[str, Any] = {
        "config_schema_version": int(raw["config_schema_version"])
    }
    config.update({key: int(raw[key]) for key in SELFPLAY_CONFIG_INT_FIELDS})
    config.update({key: float(raw[key]) for key in SELFPLAY_CONFIG_FLOAT_FIELDS})
    config.update({key: bool(raw[key]) for key in SELFPLAY_CONFIG_BOOL_FIELDS})
    config.update({key: str(raw[key]) for key in SELFPLAY_CONFIG_STR_FIELDS})

    if config["config_schema_version"] != SELFPLAY_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported self-play config schema {config['config_schema_version']}"
        )
    for key in (
        "feature_dim",
        "iterations",
        "trivial_budget_tokens",
        "worker_threads",
        "rollout_parallel_threads",
        "rollout_max_actions",
        "buffer_threshold",
    ):
        if config[key] <= 0:
            raise ValueError(f"self-play config {key} must be positive")
    for key in (
        "history_hops",
        "agz_sample_initial_move_count",
        "seed",
        "rollout_count",
        "parent_state_count",
    ):
        if config[key] < 0:
            raise ValueError(f"self-play config {key} must be non-negative")
    if config["parent_state_id"] < -1:
        raise ValueError("self-play config parent_state_id must be >= -1")
    if not 0.0 < config["discount_factor"] <= 1.0:
        raise ValueError("self-play config discount_factor must be in (0, 1]")
    for key in (
        "arena_time_limit_sec",
        "puct_c",
        "uct_c",
        "policy_prior_temperature",
        "root_dirichlet_alpha",
        "agz_mcts_action_temperature",
        "rollout_policy_temperature",
        "rollout_probability_quantum",
    ):
        if config[key] <= 0.0:
            raise ValueError(f"self-play config {key} must be positive")
    if config["root_dirichlet_total_concentration"] < 0.0:
        raise ValueError(
            "self-play config root_dirichlet_total_concentration must be non-negative"
        )
    if config["replay_sample_window_sec"] < 0.0:
        raise ValueError("self-play config replay_sample_window_sec must be non-negative")
    if config["rollout_horizon_sec"] < 0.0:
        raise ValueError("self-play config rollout_horizon_sec must be non-negative")
    for key in ("prior_min_prob", "root_dirichlet_epsilon"):
        if not 0.0 <= config[key] <= 1.0:
            raise ValueError(f"self-play config {key} must be in [0, 1]")
    if config["native_search_mode"] not in {"full_tree", "full_tree_rollout"}:
        raise ValueError("unsupported self-play native_search_mode")
    if config["native_search_mode"] == "full_tree_rollout" and (
        config["rollout_count"] <= 0 or config["rollout_horizon_sec"] <= 0.0
    ):
        raise ValueError("rollout search requires positive count and horizon")
    if config["parent_root_player_filter"] not in {"controller", "adversary", "any"}:
        raise ValueError("unsupported self-play parent_root_player_filter")
    if config["parent_state_count"] > 0 and not config["parent_dataset_dir"]:
        raise ValueError("parent_dataset_dir is required when parent_state_count is positive")
    return config


def selfplay_config_sha256(config: dict[str, Any]) -> str:
    return _canonical_json_sha256(validate_selfplay_config(dict(config)))


def default_selfplay_config() -> dict[str, Any]:
    """Capture coordinator process settings as the persisted fleet source of truth."""

    from vidur.AlphaGoZero.config import Phase1SmokeConfig

    cfg = Phase1SmokeConfig()
    return validate_selfplay_config(
        {
            "config_schema_version": SELFPLAY_CONFIG_SCHEMA_VERSION,
            "feature_dim": cfg.feature_dim,
            "iterations": cfg.mcts_iterations,
            "discount_factor": cfg.discount_factor,
            "history_hops": cfg.history_hops,
            "arena_time_limit_sec": cfg.arena_time_limit_sec,
            "replay_sample_window_sec": cfg.replay_sample_window_sec,
            "trivial_budget_tokens": cfg.trivial_budget_tokens,
            "puct_c": cfg.puct_c,
            "uct_c": cfg.uct_c,
            "policy_prior_temperature": cfg.policy_prior_temperature,
            "prior_min_prob": cfg.prior_min_prob,
            "root_dirichlet_total_concentration": cfg.root_dirichlet_total_concentration,
            "root_dirichlet_noise_enabled": cfg.root_dirichlet_noise_enabled,
            "root_dirichlet_alpha": cfg.root_dirichlet_alpha,
            "root_dirichlet_epsilon": cfg.root_dirichlet_epsilon,
            "agz_sample_initial_moves": cfg.agz_sample_initial_moves,
            "agz_sample_initial_move_count": cfg.agz_sample_initial_move_count,
            "agz_mcts_action_temperature": cfg.agz_mcts_action_temperature,
            "seed": cfg.seed,
            "worker_threads": cfg.worker_threads,
            "native_search_mode": cfg.native_search_mode,
            "rollout_count": cfg.rollout_count,
            "rollout_parallel_threads": cfg.rollout_parallel_threads,
            "rollout_horizon_sec": cfg.rollout_horizon_sec,
            "rollout_policy_temperature": cfg.rollout_policy_temperature,
            "rollout_probability_quantum": cfg.rollout_probability_quantum,
            "rollout_max_actions": cfg.rollout_max_actions,
            "buffer_threshold": int(os.environ.get("AGZ_BUFFER_THRESHOLD", "4000")),
            "parent_dataset_dir": os.environ.get("AGZ_PARENT_DATASET_DIR", ""),
            "parent_state_id": int(os.environ.get("AGZ_PARENT_STATE_ID", "-1")),
            "parent_state_count": int(os.environ.get("AGZ_PARENT_STATE_COUNT", "0")),
            "parent_root_player_filter": os.environ.get(
                "AGZ_PARENT_ROOT_PLAYER_FILTER", "any"
            ),
        }
    )


def spot_root(experiment_root: Path) -> Path:
    return Path(experiment_root) / "spot_work"


def _layout(experiment_root: Path) -> dict[str, Path]:
    root = spot_root(experiment_root)
    paths = {
        "root": root,
        "control": root / "control",
        "queued": root / "queued",
        "leased": root / "leased",
        "selfplay_leased": root / "selfplay_leased",
        "completed": root / "completed",
        "failed": root / "failed",
        "results": root / "results",
        "result_uploading": root / "result_uploading",
        "result_rejected": root / "result_rejected",
        "batches": root / "batches",
        "workers": root / "workers",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _env_nonnegative_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    value = int(raw) if raw else int(default)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _worker_heartbeat_timeout_sec(value: Any) -> int:
    return min(DEFAULT_WORKER_HEARTBEAT_TIMEOUT_SEC, max(60, int(value)))


def _effective_lease_sec(requested_sec: Any, heartbeat_timeout_sec: Any) -> int:
    return min(
        max(60, int(requested_sec)),
        _worker_heartbeat_timeout_sec(heartbeat_timeout_sec),
    )


def configure_scheduler(
    experiment_root: Path,
    *,
    selfplay_total_parallel_games: int,
    eval_total_parallel_games: int,
    selfplay_lease_sec: int = DEFAULT_WORKER_HEARTBEAT_TIMEOUT_SEC,
    worker_heartbeat_timeout_sec: int = DEFAULT_WORKER_HEARTBEAT_TIMEOUT_SEC,
    selfplay_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist coordinator-owned fleet limits used by every pull request."""

    paths = _layout(experiment_root)
    resolved_selfplay = validate_selfplay_config(
        dict(selfplay_config) if selfplay_config is not None else default_selfplay_config()
    )
    heartbeat_timeout_sec = _worker_heartbeat_timeout_sec(
        worker_heartbeat_timeout_sec
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "selfplay_total_parallel_games": max(0, int(selfplay_total_parallel_games)),
        "eval_total_parallel_games": max(0, int(eval_total_parallel_games)),
        "selfplay_lease_sec": _effective_lease_sec(
            selfplay_lease_sec, heartbeat_timeout_sec
        ),
        "worker_heartbeat_timeout_sec": heartbeat_timeout_sec,
        "selfplay_config": resolved_selfplay,
        "selfplay_config_sha256": selfplay_config_sha256(resolved_selfplay),
        "updated_at_utc": utc_now(),
    }
    with _queue_lock(experiment_root):
        atomic_write_json(paths["control"] / "scheduler_config.json", config)
        maximum_expiry = time.time() + heartbeat_timeout_sec
        for directory in (paths["leased"], paths["selfplay_leased"]):
            for lease_path in directory.glob("*.json"):
                lease = _load_json(lease_path)
                lease["worker_heartbeat_timeout_sec"] = heartbeat_timeout_sec
                lease["lease_expires_epoch"] = min(
                    float(lease.get("lease_expires_epoch", maximum_expiry)),
                    maximum_expiry,
                )
                atomic_write_json(lease_path, lease)
    return config


def _scheduler_config(paths: dict[str, Path]) -> dict[str, Any]:
    path = paths["control"] / "scheduler_config.json"
    if path.is_file():
        config = _load_json(path)
        resolved = validate_selfplay_config(dict(config["selfplay_config"]))
        expected_hash = selfplay_config_sha256(resolved)
        if str(config.get("selfplay_config_sha256", "")) != expected_hash:
            raise ValueError("scheduler self-play configuration fingerprint mismatch")
        heartbeat_timeout_sec = _worker_heartbeat_timeout_sec(
            config.get(
                "worker_heartbeat_timeout_sec",
                _env_nonnegative_int(
                    "AGZ_SPOT_WORKER_HEARTBEAT_TIMEOUT_SEC",
                    DEFAULT_WORKER_HEARTBEAT_TIMEOUT_SEC,
                ),
            )
        )
        config["worker_heartbeat_timeout_sec"] = heartbeat_timeout_sec
        config["selfplay_lease_sec"] = _effective_lease_sec(
            config.get("selfplay_lease_sec", DEFAULT_WORKER_HEARTBEAT_TIMEOUT_SEC),
            heartbeat_timeout_sec,
        )
        config["selfplay_config"] = resolved
        return config
    resolved = default_selfplay_config()
    heartbeat_timeout_sec = _worker_heartbeat_timeout_sec(
        _env_nonnegative_int(
            "AGZ_SPOT_WORKER_HEARTBEAT_TIMEOUT_SEC",
            DEFAULT_WORKER_HEARTBEAT_TIMEOUT_SEC,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        # Zero keeps backwards compatibility and means no fleet-wide cap.
        "selfplay_total_parallel_games": _env_nonnegative_int(
            "AGZ_SPOT_SELFPLAY_TOTAL_PARALLEL_GAMES", 0
        ),
        "eval_total_parallel_games": _env_nonnegative_int(
            "AGZ_SPOT_EVAL_TOTAL_PARALLEL_GAMES", 0
        ),
        "selfplay_lease_sec": _effective_lease_sec(
            _env_nonnegative_int(
                "AGZ_SPOT_SELFPLAY_LEASE_SEC",
                DEFAULT_WORKER_HEARTBEAT_TIMEOUT_SEC,
            ),
            heartbeat_timeout_sec,
        ),
        "worker_heartbeat_timeout_sec": heartbeat_timeout_sec,
        "selfplay_config": resolved,
        "selfplay_config_sha256": selfplay_config_sha256(resolved),
    }


def _worker_capacity(
    *,
    capacity: int,
    cpu_count: int | None,
    reserved_cpu_count: int,
    available_memory_bytes: int | None,
    cpu_threads_per_game: int,
    memory_bytes_per_game: int,
) -> dict[str, int]:
    advertised = max(0, int(capacity))
    threads = max(1, int(cpu_threads_per_game))
    bytes_per_game = max(1, int(memory_bytes_per_game))

    if cpu_count is None:
        cpu_slots = advertised
        normalized_cpu_count = advertised * threads
        normalized_reserved = 0
    else:
        normalized_cpu_count = max(0, int(cpu_count))
        normalized_reserved = min(
            normalized_cpu_count, max(0, int(reserved_cpu_count))
        )
        cpu_slots = max(0, normalized_cpu_count - normalized_reserved) // threads

    if available_memory_bytes is None:
        memory_slots = advertised
        normalized_available_memory = advertised * bytes_per_game
    else:
        normalized_available_memory = max(0, int(available_memory_bytes))
        memory_slots = normalized_available_memory // bytes_per_game

    effective = min(advertised, cpu_slots, memory_slots)
    return {
        "advertised_capacity": advertised,
        "cpu_count": normalized_cpu_count,
        "reserved_cpu_count": normalized_reserved,
        "cpu_threads_per_game": threads,
        "cpu_slots": int(cpu_slots),
        "available_memory_bytes": normalized_available_memory,
        "memory_bytes_per_game": bytes_per_game,
        "memory_slots": int(memory_slots),
        "effective_capacity": max(0, int(effective)),
    }


@contextmanager
def _queue_lock(experiment_root: Path) -> Iterable[None]:
    paths = _layout(experiment_root)
    lock_path = paths["control"] / "queue.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _task_path(directory: Path, task_id: str) -> Path:
    return Path(directory) / f"{task_id}.json"


def _validate_task_result(
    task: dict[str, Any],
    result_dir: Path,
) -> tuple[bool, list[str], dict[str, Any]]:
    ok, errors = verify_sha256sums(result_dir)
    missing = [
        str(filename)
        for filename in task.get("expected_files", [])
        if not (result_dir / str(filename)).is_file()
    ]
    if missing:
        errors.extend(f"missing {filename}" for filename in missing)
    manifest_path = result_dir / "spot_result_manifest.json"
    if not manifest_path.is_file():
        errors.append("missing spot_result_manifest.json")
        return False, errors, {}
    try:
        manifest = _load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid spot_result_manifest.json: {exc}")
        return False, errors, {}
    for key in ("task_id", "batch_id", "block_name"):
        if str(manifest.get(key, "")) != str(task.get(key, "")):
            errors.append(
                f"result manifest {key}={manifest.get(key)!r}, "
                f"expected={task.get(key)!r}"
            )
    try:
        game_id_matches = int(manifest.get("game_id", -1)) == int(
            task.get("game_id", -2)
        )
    except (TypeError, ValueError):
        game_id_matches = False
    if not game_id_matches:
        errors.append(
            f"result manifest game_id={manifest.get('game_id')!r}, "
            f"expected={task.get('game_id')!r}"
        )
    return bool(ok and not errors), errors, manifest


def _mark_task_completed(
    paths: dict[str, Path],
    task: dict[str, Any],
    result_dir: Path,
    *,
    manifest: dict[str, Any],
    reconciled: bool,
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    completed = dict(task)
    completed["worker_id"] = str(
        completed.get("worker_id") or manifest.get("worker_id", "")
    )
    completed["lease_token"] = str(
        completed.get("lease_token") or manifest.get("lease_token", "")
    )
    completed["status"] = "completed"
    completed["completed_at_utc"] = utc_now()
    completed["result_dir"] = str(result_dir)
    completed["reconciled_from_published_result"] = bool(reconciled)
    atomic_write_json(_task_path(paths["completed"], task_id), completed)
    for state in ("leased", "queued", "failed"):
        _task_path(paths[state], task_id).unlink(missing_ok=True)
    return completed


def _reconcile_published_results(paths: dict[str, Path]) -> list[str]:
    """Commit valid orphaned results before expired work can be reissued."""

    if not AGZ_SPOT_RESULT_RECONCILE_ENABLED:
        return []
    reconciled: list[str] = []
    for result_dir in sorted(paths["results"].iterdir()):
        if not result_dir.is_dir():
            continue
        task_id = result_dir.name
        if _task_path(paths["completed"], task_id).is_file():
            continue
        source_path: Path | None = None
        source_state = ""
        for state in ("leased", "queued", "failed"):
            candidate = _task_path(paths[state], task_id)
            if candidate.is_file():
                source_path = candidate
                source_state = state
                break
        if source_path is None:
            continue
        task = _load_json(source_path)
        valid, _errors, manifest = _validate_task_result(task, result_dir)
        if not valid:
            continue
        if source_state == "leased" and (
            str(task.get("worker_id", "")) != str(manifest.get("worker_id", ""))
            or str(task.get("lease_token", ""))
            != str(manifest.get("lease_token", ""))
        ):
            continue
        _mark_task_completed(
            paths,
            task,
            result_dir,
            manifest=manifest,
            reconciled=True,
        )
        reconciled.append(task_id)
    return reconciled


def _command_value(command: list[str], flag: str) -> str:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"missing command argument {flag}") from exc


def _set_command_value(command: list[str], flag: str, value: str | int) -> None:
    try:
        command[command.index(flag) + 1] = str(value)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"missing command argument {flag}") from exc


def _command_bool_value(
    command: list[str],
    flag: str,
    *,
    default: bool,
) -> bool:
    """Read an argparse.BooleanOptionalAction value from a command."""

    positive = flag
    negative = f"--no-{flag.removeprefix('--')}"
    positive_index = max(
        (index for index, value in enumerate(command) if value == positive),
        default=-1,
    )
    negative_index = max(
        (index for index, value in enumerate(command) if value == negative),
        default=-1,
    )
    if positive_index < 0 and negative_index < 0:
        return bool(default)
    return positive_index > negative_index


def _planned_history_hops(command: list[str]) -> list[int] | None:
    """Plan a block's history hops exactly as the native arena launcher does.

    Older or third-party commands without explicit hop bounds retain the legacy
    offset behavior. Production AlphaGoZero arena commands always include the
    required bounds and seed, so their one-game Spot tasks can be pinned here.
    """

    required = (
        "--num-games",
        "--history-hops-min",
        "--history-hops-max",
        "--history-seed",
    )
    if any(flag not in command for flag in required):
        return None
    required_booleans = (
        ("--history-hops-unique", "--no-history-hops-unique"),
        ("--history-hops-force-zero", "--no-history-hops-force-zero"),
    )
    if any(not any(flag in command for flag in pair) for pair in required_booleans):
        return None

    num_games = int(_command_value(command, "--num-games"))
    base_offset = (
        int(_command_value(command, "--history-hops-offset"))
        if "--history-hops-offset" in command
        else 0
    )
    offset = max(0, base_offset)
    planned_count = num_games + offset
    minimum = int(_command_value(command, "--history-hops-min"))
    maximum = int(_command_value(command, "--history-hops-max"))
    if num_games <= 0:
        return []
    if minimum > maximum:
        raise ValueError("history_hops_min must be <= history_hops_max")

    rng = random.Random(int(_command_value(command, "--history-seed")))
    unique = _command_bool_value(
        command, "--history-hops-unique", default=True
    )
    prefix_stable = _command_bool_value(
        command, "--history-hops-prefix-stable", default=False
    )
    if unique:
        population = list(range(minimum, maximum + 1))
        if planned_count > len(population):
            raise ValueError("num_games exceeds unique history-hop capacity")
        if prefix_stable:
            rng.shuffle(population)
            hops = population[:planned_count]
        else:
            hops = rng.sample(population, k=planned_count)
    else:
        hops = [rng.randint(minimum, maximum) for _ in range(planned_count)]

    force_zero = _command_bool_value(
        command, "--history-hops-force-zero", default=False
    )
    if force_zero:
        if not minimum <= 0 <= maximum:
            raise ValueError(
                "history hop range must include 0 when history_hops_force_zero=True"
            )
        if 0 in hops:
            zero_index = hops.index(0)
            hops[0], hops[zero_index] = hops[zero_index], hops[0]
        else:
            hops[0] = 0

    return [int(hop) for hop in hops[offset:]]


def _pin_task_history_hop(command: list[str], hop: int) -> None:
    """Make a one-game command independent of worker-side offset handling."""

    _set_command_value(command, "--history-hops-min", int(hop))
    _set_command_value(command, "--history-hops-max", int(hop))
    if "--history-hops-offset" in command:
        _set_command_value(command, "--history-hops-offset", 0)
    else:
        command.extend(["--history-hops-offset", "0"])


def _model_version_from_command(command: list[str], flag: str) -> int:
    path = _command_value(command, flag)
    matches = re.findall(r"(?:Model_Version|model_v)(\d+)", path, flags=re.IGNORECASE)
    if matches:
        return int(matches[-1])
    return int(_command_value(command, "--model-version"))


def required_model_paths(commands: Iterable[Iterable[str]]) -> list[Path]:
    flags = {
        "--model-path",
        "--controller-prior-model-path",
        "--adversary-prior-model-path",
        "--role-controller-value-model-path",
        "--role-controller-prior-model-path",
        "--role-adversary-value-model-path",
        "--role-adversary-prior-model-path",
    }
    paths: set[Path] = set()
    for raw_command in commands:
        command = list(raw_command)
        for index, item in enumerate(command[:-1]):
            if item in flags:
                paths.add(Path(command[index + 1]))
    return sorted(paths)


def model_artifacts(primary_paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Describe complete model directories, including native exports and metadata."""

    directories = sorted({Path(path).resolve().parent for path in primary_paths})
    artifacts: list[dict[str, Any]] = []
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            files.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
        if not files:
            raise RuntimeError(f"model artifact directory is empty: {directory}")
        artifacts.append({"source_dir": str(directory), "files": files})
    return artifacts


def current_model_assignment(
    experiment_root: Path,
    *,
    capacity: int,
    selfplay_config: dict[str, Any],
    selfplay_config_fingerprint: str,
) -> dict[str, Any]:
    current = Path(experiment_root) / "models" / "current_model.json"
    if not current.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": "idle",
            "reason": "current_model_missing",
            "retry_after_sec": 5.0,
        }
    bundle = _load_json(current)
    primary_paths = [Path(str(bundle[key])) for key in MODEL_PATH_KEYS]
    missing = [str(path) for path in primary_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"current model bundle is incomplete: {missing}")
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "selfplay",
        "assignment_id": f"selfplay-{uuid.uuid4().hex}",
        "games": max(1, int(capacity)),
        "model_bundle": bundle,
        "model_artifacts": model_artifacts(primary_paths),
        "controller_model_version": int(bundle["controller_model_version"]),
        "adversary_model_version": int(bundle["adversary_model_version"]),
        "selfplay_config": validate_selfplay_config(dict(selfplay_config)),
        "selfplay_config_sha256": str(selfplay_config_fingerprint),
        "issued_at_utc": utc_now(),
    }


def _active_eval(paths: dict[str, Path]) -> dict[str, Any] | None:
    path = paths["control"] / "eval_active.json"
    return _load_json(path) if path.is_file() else None


def begin_eval_batch(
    experiment_root: Path,
    *,
    block_commands: dict[str, list[str]],
    expected_games_by_block: dict[str, int],
    lease_sec: int = 3600,
    batch_id: str | None = None,
) -> dict[str, Any]:
    if not block_commands:
        raise ValueError("evaluation batch has no commands")
    paths = _layout(experiment_root)
    resolved_batch_id = str(batch_id or f"eval-{int(time.time())}-{uuid.uuid4().hex[:10]}")
    primary_paths = required_model_paths(block_commands.values())
    artifacts = model_artifacts(primary_paths)
    task_ids: list[str] = []
    tasks_by_block: dict[str, list[str]] = {}

    with _queue_lock(experiment_root):
        active = _active_eval(paths)
        if active is not None and active.get("batch_id") != resolved_batch_id:
            raise RuntimeError(f"another Spot evaluation batch is active: {active.get('batch_id')}")
        for block_name, original_command in block_commands.items():
            command = list(original_command)
            num_games = int(_command_value(command, "--num-games"))
            planned_history_hops = _planned_history_hops(command)
            expected = int(expected_games_by_block[block_name])
            if num_games != expected:
                raise ValueError(
                    f"block {block_name} command games={num_games} expected={expected}"
                )
            base_game_id = int(_command_value(command, "--game-id-start"))
            tasks_by_block[block_name] = []
            for offset in range(num_games):
                task_id = f"{resolved_batch_id}__{block_name}__{offset:06d}"
                task_command = list(command)
                _set_command_value(task_command, "--output-dir", "{WORK_DIR}")
                _set_command_value(task_command, "--game-id-start", base_game_id + offset)
                _set_command_value(task_command, "--num-games", 1)
                _set_command_value(task_command, "--num-parallel-games", 1)
                if planned_history_hops is not None:
                    _pin_task_history_hop(task_command, planned_history_hops[offset])
                else:
                    if "--history-hops-offset" in task_command:
                        _set_command_value(task_command, "--history-hops-offset", offset)
                    else:
                        task_command.extend(["--history-hops-offset", str(offset)])
                    if "--history-hops-prefix-stable" not in task_command:
                        task_command.append("--history-hops-prefix-stable")
                task = {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "eval",
                    "task_id": task_id,
                    "batch_id": resolved_batch_id,
                    "block_name": str(block_name),
                    "game_offset": int(offset),
                    "game_id": int(base_game_id + offset),
                    "model_version": int(_command_value(task_command, "--model-version")),
                    "controller_model_version": _model_version_from_command(
                        task_command,
                        "--role-controller-value-model-path",
                    ),
                    "adversary_model_version": _model_version_from_command(
                        task_command,
                        "--role-adversary-value-model-path",
                    ),
                    "command": task_command,
                    "command_sha256": _canonical_json_sha256(task_command),
                    "model_artifacts": artifacts,
                    "expected_files": [
                        "arena_results.csv",
                        "planned_games.csv",
                        "job_status.csv",
                    ],
                    "lease_sec": max(60, int(lease_sec)),
                    "attempt": 0,
                    "created_at_utc": utc_now(),
                    "created_at_epoch": time.time(),
                }
                atomic_write_json(_task_path(paths["queued"], task_id), task)
                task_ids.append(task_id)
                tasks_by_block[block_name].append(task_id)

        batch = {
            "schema_version": SCHEMA_VERSION,
            "batch_id": resolved_batch_id,
            "status": "active",
            "task_ids": task_ids,
            "tasks_by_block": tasks_by_block,
            "expected_games_by_block": {
                name: int(count) for name, count in expected_games_by_block.items()
            },
            "created_at_utc": utc_now(),
        }
        atomic_write_json(paths["batches"] / f"{resolved_batch_id}.json", batch)
        atomic_write_json(paths["control"] / "eval_active.json", batch)
    return batch


def _reclaim_expired(paths: dict[str, Path], now_epoch: float) -> int:
    reclaimed = 0
    for lease_path in sorted(paths["leased"].glob("*.json")):
        task = _load_json(lease_path)
        if float(task.get("lease_expires_epoch", 0.0)) > float(now_epoch):
            continue
        task["last_expired_worker_id"] = str(task.get("worker_id", ""))
        task["last_expired_lease_token"] = str(task.get("lease_token", ""))
        task["last_expired_at_utc"] = utc_now()
        for key in ("worker_id", "lease_token", "leased_at_utc", "lease_expires_epoch"):
            task.pop(key, None)
        atomic_write_json(_task_path(paths["queued"], str(task["task_id"])), task)
        lease_path.unlink(missing_ok=True)
        reclaimed += 1
    for lease_path in sorted(paths["selfplay_leased"].glob("*.json")):
        lease = _load_json(lease_path)
        if float(lease.get("lease_expires_epoch", 0.0)) > float(now_epoch):
            continue
        lease_path.unlink(missing_ok=True)
        reclaimed += 1
    return reclaimed


def _active_selfplay_slots(paths: dict[str, Path]) -> int:
    return sum(
        max(0, int(_load_json(path).get("slots", 0)))
        for path in paths["selfplay_leased"].glob("*.json")
    )


def _global_available(limit: int, active: int, requested: int) -> int:
    if int(limit) <= 0:
        return max(0, int(requested))
    return max(0, min(int(requested), int(limit) - int(active)))


def request_work(
    experiment_root: Path,
    *,
    worker_id: str,
    capacity: int,
    cpu_count: int | None = None,
    reserved_cpu_count: int = 0,
    available_memory_bytes: int | None = None,
    cpu_threads_per_game: int = 1,
    memory_bytes_per_game: int = GIB,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    paths = _layout(experiment_root)
    now = float(time.time() if now_epoch is None else now_epoch)
    with _queue_lock(experiment_root):
        reconciled = _reconcile_published_results(paths)
        reclaimed = _reclaim_expired(paths, now)
        # A worker requests again only after its previous bounded self-play wave
        # has ended. Releasing here also supports schema-v1 workers.
        previous_selfplay = (
            paths["selfplay_leased"] / f"{worker_id}.json"
        )
        previous_selfplay_slots = 0
        if previous_selfplay.is_file():
            previous_selfplay_slots = int(
                _load_json(previous_selfplay).get("slots", 0)
            )
            previous_selfplay.unlink(missing_ok=True)

        config = _scheduler_config(paths)
        queued = sorted(paths["queued"].glob("*.json"))
        controlled_threads = max(
            1,
            int(config["selfplay_config"]["worker_threads"]),
            int(config["selfplay_config"]["rollout_parallel_threads"]),
        )
        if queued:
            first_command = list(_load_json(queued[0]).get("command", []))
            eval_threads: list[int] = []
            for flag in ("--worker-threads", "--rollout-parallel-threads"):
                try:
                    eval_threads.append(int(_command_value(first_command, flag)))
                except ValueError:
                    eval_threads.append(1)
            controlled_threads = max(1, *eval_threads)
        resources = _worker_capacity(
            capacity=capacity,
            cpu_count=cpu_count,
            reserved_cpu_count=reserved_cpu_count,
            available_memory_bytes=available_memory_bytes,
            cpu_threads_per_game=controlled_threads,
            memory_bytes_per_game=memory_bytes_per_game,
        )
        worker = {
            "worker_id": str(worker_id),
            "capacity": int(resources["effective_capacity"]),
            "resources": resources,
            "last_request_at_utc": utc_now(),
            "last_request_at_epoch": now,
            "reclaimed_expired_assignments": int(reclaimed),
            "reconciled_published_results": len(reconciled),
            "previous_selfplay_slots_released": int(previous_selfplay_slots),
        }
        atomic_write_json(paths["workers"] / f"{worker_id}.json", worker)

        resource_capacity = int(resources["effective_capacity"])
        if resource_capacity <= 0:
            return {
                "schema_version": SCHEMA_VERSION,
                "mode": "idle",
                "reason": "insufficient_worker_resources",
                "resources": resources,
                "retry_after_sec": 5.0,
            }

        if queued:
            active_eval = sum(1 for _ in paths["leased"].glob("*.json"))
            eval_limit = int(config.get("eval_total_parallel_games", 0))
            grant = _global_available(eval_limit, active_eval, resource_capacity)
            if grant <= 0:
                return {
                    "schema_version": SCHEMA_VERSION,
                    "mode": "idle",
                    "reason": "eval_global_capacity_full",
                    "resources": resources,
                    "global_parallel_limit": eval_limit,
                    "active_parallel_games": active_eval,
                    "retry_after_sec": 2.0,
                }
            assignments: list[dict[str, Any]] = []
            heartbeat_timeout_sec = int(config["worker_heartbeat_timeout_sec"])
            for queue_path in queued[:grant]:
                task = _load_json(queue_path)
                token = uuid.uuid4().hex
                task["attempt"] = int(task.get("attempt", 0)) + 1
                task["worker_id"] = str(worker_id)
                task["lease_token"] = token
                task["leased_at_utc"] = utc_now()
                effective_lease_sec = _effective_lease_sec(
                    task.get("lease_sec", heartbeat_timeout_sec),
                    heartbeat_timeout_sec,
                )
                task["effective_lease_sec"] = effective_lease_sec
                task["worker_heartbeat_timeout_sec"] = heartbeat_timeout_sec
                task["lease_expires_epoch"] = now + effective_lease_sec
                leased_path = _task_path(paths["leased"], str(task["task_id"]))
                atomic_write_json(leased_path, task)
                queue_path.unlink()
                assignments.append(task)
            return {
                "schema_version": SCHEMA_VERSION,
                "mode": "eval",
                "assignments": assignments,
                "parallel_games": len(assignments),
                "resources": resources,
                "global_parallel_limit": eval_limit,
                "active_parallel_games_before_grant": active_eval,
                "issued_at_utc": utc_now(),
            }
        if _active_eval(paths) is not None:
            return {
                "schema_version": SCHEMA_VERSION,
                "mode": "idle",
                "reason": "evaluation_tasks_leased",
                "resources": resources,
                "retry_after_sec": 2.0,
            }

        active_selfplay = _active_selfplay_slots(paths)
        selfplay_limit = int(config.get("selfplay_total_parallel_games", 0))
        grant = _global_available(selfplay_limit, active_selfplay, resource_capacity)
        if grant <= 0:
            return {
                "schema_version": SCHEMA_VERSION,
                "mode": "idle",
                "reason": "selfplay_global_capacity_full",
                "resources": resources,
                "global_parallel_limit": selfplay_limit,
                "active_parallel_games": active_selfplay,
                "retry_after_sec": 2.0,
            }

        resolved_selfplay = validate_selfplay_config(
            dict(config["selfplay_config"])
        )
        resolved_fingerprint = selfplay_config_sha256(resolved_selfplay)
        if str(config.get("selfplay_config_sha256", "")) != resolved_fingerprint:
            raise ValueError("scheduler self-play configuration fingerprint mismatch")
        assignment = current_model_assignment(
            experiment_root,
            capacity=grant,
            selfplay_config=resolved_selfplay,
            selfplay_config_fingerprint=resolved_fingerprint,
        )
        if assignment.get("mode") != "selfplay":
            assignment["resources"] = resources
            return assignment

        lease_token = uuid.uuid4().hex
        heartbeat_timeout_sec = int(config["worker_heartbeat_timeout_sec"])
        lease_sec = _effective_lease_sec(
            config.get("selfplay_lease_sec", heartbeat_timeout_sec),
            heartbeat_timeout_sec,
        )
        lease = {
            "schema_version": SCHEMA_VERSION,
            "worker_id": str(worker_id),
            "assignment_id": str(assignment["assignment_id"]),
            "lease_token": lease_token,
            "slots": int(grant),
            "selfplay_config_sha256": resolved_fingerprint,
            "leased_at_utc": utc_now(),
            "lease_expires_epoch": now + lease_sec,
            "worker_heartbeat_timeout_sec": heartbeat_timeout_sec,
        }
        atomic_write_json(paths["selfplay_leased"] / f"{worker_id}.json", lease)
        assignment.update(
            {
                "parallel_games": int(grant),
                "lease_token": lease_token,
                "lease_sec": lease_sec,
                "resources": resources,
                "global_parallel_limit": selfplay_limit,
                "active_parallel_games_before_grant": active_selfplay,
            }
        )
        return assignment


def heartbeat_selfplay_assignment(
    experiment_root: Path,
    *,
    worker_id: str,
    assignment_id: str,
    lease_token: str,
    extend_sec: int,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    paths = _layout(experiment_root)
    now = float(time.time() if now_epoch is None else now_epoch)
    with _queue_lock(experiment_root):
        lease_path = paths["selfplay_leased"] / f"{worker_id}.json"
        if not lease_path.is_file():
            return {"extended": False, "reason": "lease_missing"}
        lease = _load_json(lease_path)
        if (
            lease.get("assignment_id") != assignment_id
            or lease.get("lease_token") != lease_token
        ):
            return {"extended": False, "reason": "stale_lease"}
        config = _scheduler_config(paths)
        effective_lease_sec = _effective_lease_sec(
            extend_sec, config["worker_heartbeat_timeout_sec"]
        )
        lease["lease_expires_epoch"] = now + effective_lease_sec
        lease["worker_heartbeat_timeout_sec"] = int(
            config["worker_heartbeat_timeout_sec"]
        )
        lease["heartbeat_at_utc"] = utc_now()
        active_eval = _active_eval(paths)
        preempt_requested = active_eval is not None
        if preempt_requested:
            lease.setdefault("preempt_requested_at_utc", utc_now())
            lease["preempt_batch_id"] = str(active_eval.get("batch_id", ""))
        atomic_write_json(lease_path, lease)
        worker_path = paths["workers"] / f"{worker_id}.json"
        if worker_path.is_file():
            worker = _load_json(worker_path)
            worker["last_heartbeat_at_utc"] = utc_now()
            worker["last_heartbeat_at_epoch"] = now
            atomic_write_json(worker_path, worker)
        return {
            "extended": True,
            "slots": int(lease.get("slots", 0)),
            "lease_sec": effective_lease_sec,
            "preempt_requested": bool(preempt_requested),
            "preempt_reason": "evaluation_active" if preempt_requested else "",
            "eval_batch_id": (
                str(active_eval.get("batch_id", "")) if active_eval is not None else ""
            ),
        }


def release_selfplay_assignment(
    experiment_root: Path,
    *,
    worker_id: str,
    assignment_id: str,
    lease_token: str,
    status: str,
) -> dict[str, Any]:
    paths = _layout(experiment_root)
    with _queue_lock(experiment_root):
        lease_path = paths["selfplay_leased"] / f"{worker_id}.json"
        if not lease_path.is_file():
            return {"released": False, "reason": "lease_missing"}
        lease = _load_json(lease_path)
        if (
            lease.get("assignment_id") != assignment_id
            or lease.get("lease_token") != lease_token
        ):
            return {"released": False, "reason": "stale_lease"}
        slots = int(lease.get("slots", 0))
        lease_path.unlink()
        worker_path = paths["workers"] / f"{worker_id}.json"
        if worker_path.is_file():
            worker = _load_json(worker_path)
            worker["last_assignment_status"] = str(status)
            worker["last_assignment_finished_at_utc"] = utc_now()
            atomic_write_json(worker_path, worker)
        return {"released": True, "slots": slots}



def scheduler_status(
    experiment_root: Path,
    *,
    worker_freshness_sec: int = 300,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    paths = _layout(experiment_root)
    now = float(time.time() if now_epoch is None else now_epoch)
    with _queue_lock(experiment_root):
        reconciled = _reconcile_published_results(paths)
        reclaimed = _reclaim_expired(paths, now)
        config = _scheduler_config(paths)
        workers: list[dict[str, Any]] = []
        registered_capacity = 0
        for path in sorted(paths["workers"].glob("*.json")):
            worker = _load_json(path)
            last_seen = float(
                worker.get(
                    "last_heartbeat_at_epoch",
                    worker.get("last_request_at_epoch", 0.0),
                )
            )
            if now - last_seen > max(1, int(worker_freshness_sec)):
                continue
            workers.append(worker)
            registered_capacity += max(0, int(worker.get("capacity", 0)))

        active_selfplay = _active_selfplay_slots(paths)
        active_eval = sum(1 for _ in paths["leased"].glob("*.json"))
        selfplay_target = int(config.get("selfplay_total_parallel_games", 0))
        eval_target = int(config.get("eval_total_parallel_games", 0))
        return {
            "schema_version": SCHEMA_VERSION,
            "scheduler_config": config,
            "live_workers": len(workers),
            "registered_worker_capacity": int(registered_capacity),
            "selfplay_active_parallel_games": int(active_selfplay),
            "selfplay_parallel_shortfall": (
                max(0, selfplay_target - active_selfplay)
                if selfplay_target > 0
                else 0
            ),
            "eval_active_parallel_games": int(active_eval),
            "eval_parallel_shortfall": (
                max(0, eval_target - active_eval) if eval_target > 0 else 0
            ),
            "queued_eval_games": sum(1 for _ in paths["queued"].glob("*.json")),
            "reclaimed_expired_assignments": int(reclaimed),
            "reconciled_published_results": len(reconciled),
            "reconciled_task_ids": reconciled,
            "checked_at_utc": utc_now(),
        }


def heartbeat_tasks(
    experiment_root: Path,
    *,
    worker_id: str,
    leases: dict[str, str],
    extend_sec: int,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    paths = _layout(experiment_root)
    now = float(time.time() if now_epoch is None else now_epoch)
    extended: list[str] = []
    rejected: list[str] = []
    with _queue_lock(experiment_root):
        config = _scheduler_config(paths)
        effective_lease_sec = _effective_lease_sec(
            extend_sec, config["worker_heartbeat_timeout_sec"]
        )
        for task_id, token in leases.items():
            path = _task_path(paths["leased"], task_id)
            if not path.is_file():
                rejected.append(task_id)
                continue
            task = _load_json(path)
            if task.get("worker_id") != worker_id or task.get("lease_token") != token:
                rejected.append(task_id)
                continue
            task["effective_lease_sec"] = effective_lease_sec
            task["worker_heartbeat_timeout_sec"] = int(
                config["worker_heartbeat_timeout_sec"]
            )
            task["lease_expires_epoch"] = now + effective_lease_sec
            task["heartbeat_at_utc"] = utc_now()
            atomic_write_json(path, task)
            extended.append(task_id)
        worker_path = paths["workers"] / f"{worker_id}.json"
        if worker_path.is_file():
            worker = _load_json(worker_path)
            worker["last_heartbeat_at_utc"] = utc_now()
            worker["last_heartbeat_at_epoch"] = now
            atomic_write_json(worker_path, worker)
    return {
        "extended": extended,
        "rejected": rejected,
        "lease_sec": effective_lease_sec,
    }


def complete_task(
    experiment_root: Path,
    *,
    worker_id: str,
    task_id: str,
    lease_token: str,
) -> dict[str, Any]:
    paths = _layout(experiment_root)
    completed_path = _task_path(paths["completed"], task_id)
    with _queue_lock(experiment_root):
        if completed_path.is_file():
            return {"accepted": True, "idempotent": True}
        lease_path = _task_path(paths["leased"], task_id)
        if not lease_path.is_file():
            _reconcile_published_results(paths)
            if completed_path.is_file():
                return {"accepted": True, "idempotent": True}
            return {"accepted": False, "reason": "lease_missing"}
        task = _load_json(lease_path)
        if task.get("worker_id") != worker_id or task.get("lease_token") != lease_token:
            return {"accepted": False, "reason": "stale_lease"}
        result_dir = paths["results"] / task_id
        ok, errors, manifest = _validate_task_result(task, result_dir)
        if not ok:
            return {
                "accepted": False,
                "reason": "invalid_result",
                "checksum_errors": errors,
            }
        if (
            str(manifest.get("worker_id", "")) != worker_id
            or str(manifest.get("lease_token", "")) != lease_token
        ):
            return {"accepted": False, "reason": "result_lease_mismatch"}
        _mark_task_completed(
            paths, task, result_dir, manifest=manifest, reconciled=False
        )
        return {"accepted": True, "idempotent": False, "result_dir": str(result_dir)}


def fail_task(
    experiment_root: Path,
    *,
    worker_id: str,
    task_id: str,
    lease_token: str,
    error: str,
    max_attempts: int = 5,
) -> dict[str, Any]:
    paths = _layout(experiment_root)
    with _queue_lock(experiment_root):
        lease_path = _task_path(paths["leased"], task_id)
        if not lease_path.is_file():
            return {"accepted": False, "reason": "lease_missing"}
        task = _load_json(lease_path)
        if task.get("worker_id") != worker_id or task.get("lease_token") != lease_token:
            return {"accepted": False, "reason": "stale_lease"}
        task["last_error"] = str(error)[-8000:]
        task["last_failed_at_utc"] = utc_now()
        task.pop("worker_id", None)
        task.pop("lease_token", None)
        task.pop("leased_at_utc", None)
        task.pop("lease_expires_epoch", None)
        lease_path.unlink()
        if int(task.get("attempt", 0)) >= max(1, int(max_attempts)):
            task["status"] = "failed"
            atomic_write_json(_task_path(paths["failed"], task_id), task)
            return {"accepted": True, "requeued": False}
        atomic_write_json(_task_path(paths["queued"], task_id), task)
        return {"accepted": True, "requeued": True}


def batch_status(experiment_root: Path, batch_id: str) -> dict[str, Any]:
    paths = _layout(experiment_root)
    batch_path = paths["batches"] / f"{batch_id}.json"
    if not batch_path.is_file():
        raise FileNotFoundError(batch_path)
    with _queue_lock(experiment_root):
        reconciled = _reconcile_published_results(paths)
        batch = _load_json(batch_path)
        task_ids = [str(task_id) for task_id in batch["task_ids"]]
        counts = {}
        for state in ("queued", "leased", "completed", "failed"):
            counts[state] = sum(
                _task_path(paths[state], task_id).is_file() for task_id in task_ids
            )
        return {
            **batch,
            "counts": counts,
            "reconciled_task_ids": [
                task_id for task_id in reconciled if task_id in task_ids
            ],
        }


def wait_for_batch(
    experiment_root: Path,
    *,
    batch_id: str,
    timeout_sec: int = 0,
    poll_sec: float = 2.0,
) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        status = batch_status(experiment_root, batch_id)
        if int(status["counts"]["failed"]) > 0:
            raise RuntimeError(f"Spot evaluation batch failed: {status}")
        if int(status["counts"]["completed"]) == len(status["task_ids"]):
            return status
        if int(timeout_sec) > 0 and time.monotonic() - started >= int(timeout_sec):
            raise TimeoutError(f"Spot evaluation batch timed out: {status}")
        time.sleep(max(0.05, float(poll_sec)))


def finish_eval_batch(experiment_root: Path, *, batch_id: str) -> dict[str, Any]:
    paths = _layout(experiment_root)
    status = batch_status(experiment_root, batch_id)
    if int(status["counts"]["completed"]) != len(status["task_ids"]):
        raise RuntimeError(f"cannot finish incomplete evaluation batch: {status}")
    with _queue_lock(experiment_root):
        batch_path = paths["batches"] / f"{batch_id}.json"
        batch = _load_json(batch_path)
        batch["status"] = "complete"
        batch["completed_at_utc"] = utc_now()
        atomic_write_json(batch_path, batch)
        active_path = paths["control"] / "eval_active.json"
        if active_path.is_file() and _load_json(active_path).get("batch_id") == batch_id:
            active_path.unlink()
        return batch


def cancel_eval_batch(experiment_root: Path, *, batch_id: str, reason: str) -> None:
    paths = _layout(experiment_root)
    with _queue_lock(experiment_root):
        batch_path = paths["batches"] / f"{batch_id}.json"
        if batch_path.is_file():
            batch = _load_json(batch_path)
            batch["status"] = "cancelled"
            batch["cancel_reason"] = str(reason)
            batch["cancelled_at_utc"] = utc_now()
            atomic_write_json(batch_path, batch)
        active_path = paths["control"] / "eval_active.json"
        if active_path.is_file() and _load_json(active_path).get("batch_id") == batch_id:
            active_path.unlink()


def reset_spot_queue(experiment_root: Path) -> None:
    root = spot_root(experiment_root)
    if root.exists():
        shutil.rmtree(root)
    _layout(experiment_root)
