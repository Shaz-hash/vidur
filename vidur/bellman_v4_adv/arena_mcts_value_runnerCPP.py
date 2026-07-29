"""Run GV3 Model_Tester arena games with native C++ shared-root MCTS.

This runner keeps the Python ``Model_Tester`` arena harness and logging, but
replaces model action selection with ``Game_Version3_Cpp`` MCTS.  The C++ search
uses the native 226D HGB runtime as the leaf bootstrap and can optionally score
canonical root priors with controller/adversary HGB policy heads.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from vidur.mcts.Game_Versions.Game_Version3.Model_Tester import runner as tester_runner
from vidur.mcts.Game_Versions.Game_Version3.Model_Tester.config import (
    DEFAULT_MODEL_TESTER_CONFIG,
)
from vidur.mcts.Game_Versions.Game_Version3.DNN.native_selfplay import (
    _cfg_payload,
    attach_execution_predictor_payload,
)
from vidur.mcts.Game_Versions.Game_Version3.tests import native_logger_tests as nlt
from vidur.AlphaGoZero.replay_runtime import AlphaGoZeroReplayRecorder
from vidur.AlphaGoZero.config import AGZ_DISCOUNT_FACTOR, AGZ_VALUE_FEATURE_SCHEMA
from vidur.AlphaGoZero.markov_value_features import (
    MARKOV_VALUE_SCHEMA,
    build_markov_value_features,
)


_RUNTIME_ARGS: argparse.Namespace | None = None
_NATIVE_MODULE: Any | None = None
_NATIVE_RUNTIME: Any | None = None
_NATIVE_CONTROLLER_PRIOR_RUNTIME: Any | None = None
_NATIVE_ADVERSARY_PRIOR_RUNTIME: Any | None = None
_NATIVE_VALUE_RUNTIME_BY_PLAYER: dict[str, Any] = {}
_NATIVE_MODEL_EXPORT_PATH: Path | None = None
_NATIVE_CONTROLLER_PRIOR_EXPORT_PATH: Path | None = None
_NATIVE_ADVERSARY_PRIOR_EXPORT_PATH: Path | None = None
_CFG_PAYLOAD_CACHE: dict[int, dict[str, Any]] = {}
_AGZ_REPLAY_RECORDER: AlphaGoZeroReplayRecorder | None = None
_NATIVE_INT_SEED_MOD = 2_147_483_647


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cpp_dir() -> Path:
    return _repo_root() / "vidur" / "Game_Version3_Cpp"


def _native_module_path() -> Path | None:
    candidates = sorted(_cpp_dir().glob("mcts_native_gv2*.so"))
    return candidates[0] if candidates else None


def _import_native_cpp(*, force_build: bool = False) -> Any:
    global _NATIVE_MODULE
    if _NATIVE_MODULE is not None and not force_build:
        return _NATIVE_MODULE

    cpp_dir = _cpp_dir()
    build_dir = cpp_dir / "build"
    existing = _native_module_path()
    if force_build or existing is None:
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "cmake",
                "-S",
                str(cpp_dir),
                "-B",
                str(build_dir),
                f"-DPython_EXECUTABLE={sys.executable}",
            ],
            cwd=str(_repo_root()),
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(build_dir), "-j", str(max(1, min(16, os.cpu_count() or 1)))],
            cwd=str(_repo_root()),
            check=True,
        )
        existing = _native_module_path()
        if existing is None:
            raise FileNotFoundError(f"native module was not produced in {cpp_dir}")

    if str(cpp_dir) not in sys.path:
        sys.path.insert(0, str(cpp_dir))
    sys.modules.pop("mcts_native_gv2", None)
    _NATIVE_MODULE = importlib.import_module("mcts_native_gv2")
    return _NATIVE_MODULE


def _limit_native_threads(num_threads: int) -> None:
    threads = max(1, int(num_threads))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=threads)
    except Exception:
        pass


def _native_int_seed(seed: int) -> int:
    """Keep seeds inside pybind's signed C++ int conversion range."""
    return int(seed) % _NATIVE_INT_SEED_MOD


def _export_hgb_to_native_text(
    model: Any,
    out_path: Path,
    *,
    feature_dim_override: int | None = None,
    model_tag_override: str | None = None,
) -> Path:
    """Export sklearn HistGradientBoostingRegressor internals for C++ traversal."""
    import numpy as np

    hgb = getattr(model, "hgb", model)
    predictors = list(getattr(hgb, "_predictors", []) or [])
    if not predictors:
        raise TypeError("expected sklearn HistGradientBoostingRegressor with _predictors")
    baseline = float(np.ravel(getattr(hgb, "_baseline_prediction"))[0])
    feature_dim = int(feature_dim_override if feature_dim_override is not None else getattr(model, "feature_dim", 226))
    model_tag = str(model_tag_override if model_tag_override is not None else getattr(model, "model_tag", "v4_adv_hgb"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write("hgb226_v1\n")
        f.write(f"model_tag\t{model_tag}\n")
        f.write(f"feature_dim\t{feature_dim}\n")
        f.write(f"baseline\t{baseline:.17g}\n")
        for tree_idx, pred in enumerate(predictors):
            tree = pred[0]
            nodes = tree.nodes
            if "is_categorical" in nodes.dtype.names and bool(np.any(nodes["is_categorical"])):
                raise RuntimeError("native HGB export does not support categorical splits")
            f.write(f"tree\t{tree_idx}\t{len(nodes)}\n")
            for n in nodes:
                f.write(
                    "node"
                    f"\t{float(n['value']):.17g}"
                    f"\t{int(n['feature_idx'])}"
                    f"\t{float(n['num_threshold']):.17g}"
                    f"\t{1 if bool(n['missing_go_to_left']) else 0}"
                    f"\t{int(n['left'])}"
                    f"\t{int(n['right'])}"
                    f"\t{1 if bool(n['is_leaf']) else 0}\n"
                )
            f.write("end_tree\n")
    tmp.replace(out_path)
    return out_path


def _load_value_runtime_from_joblib(
    *,
    native: Any,
    model_path: Path,
    export_path: Path,
    feature_dim: int,
    model_tag: str,
) -> tuple[Any, Path, Any, bool]:
    model = joblib.load(model_path)
    from vidur.AlphaGoZero.dnn_models import export_dnn_to_native, is_dnn_model

    if is_dnn_model(model):
        native_export = export_dnn_to_native(
            model,
            export_path,
            model_tag=str(model_tag),
        )
        runtime = native.NewFeatures226HGBRuntime()
        runtime.load_model_export(str(native_export))
        return runtime, native_export, model, False

    wrapped = False
    if not callable(getattr(model, "infer_from_inputs", None)):
        from vidur.bellman_v4_adv.v4_adv_hgb_wrapper import V4AdvHGBWrapper

        model = V4AdvHGBWrapper(model, feature_dim=int(feature_dim), model_tag=str(model_tag))
        wrapped = True
    native_export = _export_hgb_to_native_text(
        model,
        export_path,
        feature_dim_override=int(feature_dim),
        model_tag_override=str(model_tag),
    )
    runtime = native.NewFeatures226HGBRuntime()
    runtime.load_model_export(str(native_export))
    return runtime, native_export, model, wrapped


def _load_prior_runtime_from_joblib(
    *,
    native: Any,
    model_path: Path,
    export_path: Path,
    feature_dim: int,
    model_tag: str,
) -> tuple[Any, Path]:
    model = joblib.load(model_path)
    from vidur.AlphaGoZero.dnn_models import export_dnn_to_native, is_dnn_model

    if is_dnn_model(model):
        native_export = export_dnn_to_native(
            model,
            export_path,
            model_tag=str(model_tag),
        )
        runtime = native.NativeHGBModelRuntime()
        runtime.load_model_export(str(native_export))
        return runtime, native_export

    native_export = _export_hgb_to_native_text(
        model,
        export_path,
        feature_dim_override=int(feature_dim),
        model_tag_override=str(model_tag),
    )
    runtime = native.NativeHGBModelRuntime()
    runtime.load_model_export(str(native_export))
    return runtime, native_export


def _prepare_native_runtime(args: argparse.Namespace) -> None:
    global _NATIVE_RUNTIME, _NATIVE_MODEL_EXPORT_PATH
    global _NATIVE_CONTROLLER_PRIOR_RUNTIME, _NATIVE_ADVERSARY_PRIOR_RUNTIME
    global _NATIVE_CONTROLLER_PRIOR_EXPORT_PATH, _NATIVE_ADVERSARY_PRIOR_EXPORT_PATH
    global _NATIVE_VALUE_RUNTIME_BY_PLAYER
    native = _import_native_cpp(force_build=bool(args.force_build_native))
    model_path = Path(args.model_path).expanduser()
    export_path = (
        Path(args.native_model_export_path).expanduser()
        if args.native_model_export_path
        else Path(args.output_dir).expanduser() / "native_model" / "v4_adv_hgb_native_export.tsv"
    )
    runtime, native_export, model, wrapped = _load_value_runtime_from_joblib(
        native=native,
        model_path=model_path,
        export_path=export_path,
        feature_dim=int(args.feature_dim),
        model_tag=f"wrapped:{model_path.name}",
    )
    setattr(args, "harness_model_path", str(model_path))
    if wrapped:
        # Model_Tester still loads a classical_joblib model before this runner
        # overrides action selection with C++ MCTS. Raw HGB models therefore
        # need a lightweight wrapper for the harness side.
        wrapper_path = Path(args.output_dir).expanduser() / "arena_model" / "v4_adv_hgb_wrapper.joblib"
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, wrapper_path, compress=3)
        setattr(args, "harness_model_path", str(wrapper_path))

    _NATIVE_MODEL_EXPORT_PATH = native_export
    _NATIVE_RUNTIME = runtime

    _NATIVE_CONTROLLER_PRIOR_RUNTIME = None
    _NATIVE_ADVERSARY_PRIOR_RUNTIME = None
    _NATIVE_VALUE_RUNTIME_BY_PLAYER = {}
    _NATIVE_CONTROLLER_PRIOR_EXPORT_PATH = None
    _NATIVE_ADVERSARY_PRIOR_EXPORT_PATH = None

    controller_prior_path = str(getattr(args, "controller_prior_model_path", "") or "")
    adversary_prior_path = str(getattr(args, "adversary_prior_model_path", "") or "")
    if controller_prior_path and adversary_prior_path:
        controller_export = (
            Path(args.controller_prior_native_export_path).expanduser()
            if str(getattr(args, "controller_prior_native_export_path", "") or "")
            else Path(args.output_dir).expanduser() / "native_model" / "controller_prior_hgb_native_export.tsv"
        )
        adversary_export = (
            Path(args.adversary_prior_native_export_path).expanduser()
            if str(getattr(args, "adversary_prior_native_export_path", "") or "")
            else Path(args.output_dir).expanduser() / "native_model" / "adversary_prior_hgb_native_export.tsv"
        )

        controller_runtime, _NATIVE_CONTROLLER_PRIOR_EXPORT_PATH = _load_prior_runtime_from_joblib(
            native=native,
            model_path=Path(controller_prior_path).expanduser(),
            export_path=controller_export,
            feature_dim=269,
            model_tag=f"controller_prior:{Path(controller_prior_path).name}",
        )
        adversary_runtime, _NATIVE_ADVERSARY_PRIOR_EXPORT_PATH = _load_prior_runtime_from_joblib(
            native=native,
            model_path=Path(adversary_prior_path).expanduser(),
            export_path=adversary_export,
            feature_dim=233,
            model_tag=f"adversary_prior:{Path(adversary_prior_path).name}",
        )
        _NATIVE_CONTROLLER_PRIOR_RUNTIME = controller_runtime
        _NATIVE_ADVERSARY_PRIOR_RUNTIME = adversary_runtime

    role_ctrl_value = str(getattr(args, "role_controller_value_model_path", "") or "")
    role_adv_value = str(getattr(args, "role_adversary_value_model_path", "") or "")
    role_ctrl_prior = str(getattr(args, "role_controller_prior_model_path", "") or "")
    role_adv_prior = str(getattr(args, "role_adversary_prior_model_path", "") or "")
    if role_ctrl_value and role_adv_value and role_ctrl_prior and role_adv_prior:
        role_dir = Path(args.output_dir).expanduser() / "native_model" / "role_bundles"
        ctrl_value_runtime, _, _, _ = _load_value_runtime_from_joblib(
            native=native,
            model_path=Path(role_ctrl_value).expanduser(),
            export_path=role_dir / "controller_agent_value_hgb_native_export.tsv",
            feature_dim=int(args.feature_dim),
            model_tag=f"controller_agent_value:{Path(role_ctrl_value).name}",
        )
        adv_value_runtime, _, _, _ = _load_value_runtime_from_joblib(
            native=native,
            model_path=Path(role_adv_value).expanduser(),
            export_path=role_dir / "adversary_agent_value_hgb_native_export.tsv",
            feature_dim=int(args.feature_dim),
            model_tag=f"adversary_agent_value:{Path(role_adv_value).name}",
        )
        _NATIVE_VALUE_RUNTIME_BY_PLAYER = {
            "controller": ctrl_value_runtime,
            "adversary": adv_value_runtime,
        }
        controller_runtime, _NATIVE_CONTROLLER_PRIOR_EXPORT_PATH = _load_prior_runtime_from_joblib(
            native=native,
            model_path=Path(role_ctrl_prior).expanduser(),
            export_path=role_dir / "controller_agent_prior_hgb_native_export.tsv",
            feature_dim=269,
            model_tag=f"controller_agent_prior:{Path(role_ctrl_prior).name}",
        )
        adversary_runtime, _NATIVE_ADVERSARY_PRIOR_EXPORT_PATH = _load_prior_runtime_from_joblib(
            native=native,
            model_path=Path(role_adv_prior).expanduser(),
            export_path=role_dir / "adversary_agent_prior_hgb_native_export.tsv",
            feature_dim=233,
            model_tag=f"adversary_agent_prior:{Path(role_adv_prior).name}",
        )
        _NATIVE_CONTROLLER_PRIOR_RUNTIME = controller_runtime
        _NATIVE_ADVERSARY_PRIOR_RUNTIME = adversary_runtime

def _get_cfg_payload(args: argparse.Namespace, cfg: Any, bundle: Any) -> dict[str, Any]:
    key = id(bundle.simulator)
    cached = _CFG_PAYLOAD_CACHE.get(key)
    if cached is not None:
        return cached

    pipeline_cfg = cfg.to_pipeline_cfg() if callable(getattr(cfg, "to_pipeline_cfg", None)) else cfg
    payload = _cfg_payload(pipeline_cfg, torchscript_model_spec="")
    # Override the engine default only for this explicitly configured run.
    payload["discount_factor"] = float(args.discount_factor)
    attach_execution_predictor_payload(payload, bundle.simulator)
    payload["use_model_bootstrap"] = bool(int(args.model_version) > 0) and not bool(args.disable_model_bootstrap)
    payload["native_search_mode"] = str(args.native_search_mode)
    payload["rollout_count"] = int(args.rollout_count)
    payload["rollout_parallel_threads"] = int(args.rollout_parallel_threads)
    payload["rollout_horizon_sec"] = float(args.rollout_horizon_sec)
    payload["rollout_policy_temperature"] = float(args.rollout_policy_temperature)
    payload["rollout_probability_quantum"] = float(args.rollout_probability_quantum)
    payload["rollout_max_actions"] = int(args.rollout_max_actions)
    payload["use_policy_prior"] = bool(_NATIVE_CONTROLLER_PRIOR_RUNTIME is not None and _NATIVE_ADVERSARY_PRIOR_RUNTIME is not None)
    payload["root_dirichlet_noise_enabled"] = bool(args.root_dirichlet_noise_enabled)
    payload["root_dirichlet_alpha"] = float(args.root_dirichlet_alpha)
    payload["root_dirichlet_epsilon"] = float(args.root_dirichlet_epsilon)
    payload["pb_c_base"] = float(pipeline_cfg.game_v2.mcts_search.pb_c_base)
    payload["pb_c_init"] = float(pipeline_cfg.game_v2.mcts_search.pb_c_init)
    payload["uct_c"] = float(args.uct_c)
    payload["puct_c"] = float(args.puct_c)
    payload["policy_prior_temperature"] = float(args.policy_prior_temperature)
    payload["prior_min_prob"] = float(args.prior_min_prob)
    payload["max_forced_hops"] = int(getattr(pipeline_cfg, "max_forced_hops_per_root", 0) or 0)
    _CFG_PAYLOAD_CACHE[key] = payload
    return payload




def _build_state_features_for_replay(*, args: argparse.Namespace, cfg: Any, bundle: Any, state: Any, root_id: int) -> tuple[list[float], dict[str, str | int]]:
    state_payload = nlt._native_state_payload(bundle.env, state)
    legacy_features: list[float] = []
    if _NATIVE_RUNTIME is not None:
        try:
            payload = dict(_get_cfg_payload(args, cfg, bundle))
            out = _NATIVE_RUNTIME.build_features_from_state(state_payload, payload, int(root_id))
            values = [float(x) for x in list(out.get("features", []) or [])]
            legacy_features = values if len(values) == 226 else []
        except Exception:
            legacy_features = []

    schema = str(AGZ_VALUE_FEATURE_SCHEMA).strip().lower()
    if schema == "legacy_226":
        return legacy_features, {}
    if schema != MARKOV_VALUE_SCHEMA:
        raise RuntimeError(f"unsupported AGZ_VALUE_FEATURE_SCHEMA={schema!r}")
    try:
        value_fields = build_markov_value_features(state_payload).replay_fields()
    except Exception as exc:
        raise RuntimeError(f"failed to encode {MARKOV_VALUE_SCHEMA} replay state") from exc
    return legacy_features, value_fields


def _build_action_features_for_replay(*, bundle: Any, state: Any, player: str, action: Any, canon_idx: int) -> list[float]:
    try:
        if str(player) == "controller":
            from vidur.bellman_v4_adv_2000k_multiprocess.multi_server_hgb_controller_prior_feature_bulding import (
                ACTION_FEATURE_NAMES as _CTRL_ACTION_FEATURE_NAMES,
                build_action_feature_dict as _build_controller_action_features,
            )

            fd = _build_controller_action_features(bundle, state, action)
            return [float(fd.get(name, 0.0)) for name in _CTRL_ACTION_FEATURE_NAMES]
        if str(player) == "adversary":
            from vidur.bellman_v4_adv_2000k_multiprocess.multi_server_hgb_adversary_prior_feature_building import (
                ACTION_FEATURE_NAMES as _ADV_ACTION_FEATURE_NAMES,
                STOP_RULE_NAMES as _ADV_STOP_RULE_NAMES,
                build_action_feature_dict as _build_adversary_action_features,
            )

            stop_rules = list(getattr(bundle.pipeline_cfg.game_v2.adversary_action, "stop_rule_names", _ADV_STOP_RULE_NAMES))
            fd = _build_adversary_action_features(action, canon_action_index=int(canon_idx), stop_rules=stop_rules)
            return [float(fd.get(name, 0.0)) for name in _ADV_ACTION_FEATURE_NAMES]
    except Exception:
        return []
    return []

def _state_from_parent_record(env: Any, record: dict[str, Any]) -> Any:
    clone_fn = getattr(env, "clone_state_from_snapshot", None)
    if callable(clone_fn):
        return clone_fn(record["simulator_snapshot"], record["stats"])
    state = env.initial_state()
    state.simulator.restore_state(record["simulator_snapshot"])
    state.stats = record["stats"].clone()
    return state


def _load_parent_record(dataset_dir: str, *, state_id: int, root_player_filter: str) -> tuple[int, dict[str, Any]]:
    from vidur.Game_Version3.ModelSearchBed.analysis_testing.rootChildGenerationAdv import load_samples

    sid = int(state_id)
    rows = list(
        load_samples(
            str(dataset_dir),
            root_player_filter=str(root_player_filter),
            parent_state_id_start=int(sid),
            parent_state_id_end=int(sid) + 1,
        )
    )
    if not rows:
        raise RuntimeError(
            f"parent state {sid} not found in {dataset_dir} with filter={root_player_filter}"
        )
    return int(rows[0][0]), dict(rows[0][1])


def _make_parent_base_snapshot_hook(old_make_base_snapshot: Any, args: argparse.Namespace):
    def _make_base_snapshot_from_parent(**kwargs: Any) -> tuple[Any, Any, str, int]:
        dataset_dir = str(getattr(args, "parent_dataset_dir", "") or "")
        if not dataset_dir:
            return old_make_base_snapshot(**kwargs)
        parent_state_id = int(getattr(args, "parent_state_id", -1))
        if parent_state_id < 0:
            return old_make_base_snapshot(**kwargs)
        bundle = kwargs["bundle"]
        cfg = kwargs["cfg"]
        record_sid, record = _load_parent_record(
            dataset_dir,
            state_id=int(parent_state_id),
            root_player_filter=str(getattr(args, "parent_root_player_filter", "any") or "any"),
        )
        state = _state_from_parent_record(bundle.env, record)
        root_player = str(record.get("root_player", getattr(cfg, "start_player", "adversary")) or "adversary")
        root_depth = int(record.get("root_depth", getattr(cfg, "start_root_depth", 0)) or 0)
        player, expanded = tester_runner._align_player_to_valid_actions(
            bundle=bundle,
            state=state,
            player=root_player,
            pending_adv_pre_ctrl_snapshot=record.get("pre_controller_snapshot"),
            pending_adv_pre_ctrl_stats=record.get("pre_controller_stats"),
        )
        if not expanded.valid_indices:
            raise RuntimeError(f"parent state {record_sid} has no valid actions after alignment")
        return state.simulator.snapshot_state(), state.stats.clone(), str(player), int(root_depth)
    return _make_base_snapshot_from_parent


def _safe_label(value: Any) -> str:
    text = str(value or "").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _mcts_visit_log_paths(
    args: argparse.Namespace,
    *,
    game_id: int,
    cycle_label: str,
    phase: str,
    turn: int,
    player: str,
) -> tuple[Path, Path]:
    base = Path(getattr(args, "mcts_visit_log_dir", "") or Path(args.output_dir) / "mcts_visit_logs")
    stem = (
        f"game_{int(game_id)}_"
        f"{_safe_label(cycle_label)}_"
        f"turn_{int(turn):06d}_"
        f"{_safe_label(phase)}_"
        f"{_safe_label(player)}"
    )
    return base / f"{stem}_root.csv", base / f"{stem}_children.csv"


def _set_runtime_selection_context(**kwargs: Any) -> None:
    args = _RUNTIME_ARGS
    if args is None:
        return
    game_id_raw = kwargs.get("game_id")
    turn_raw = kwargs.get("turn")
    depth_raw = kwargs.get("depth")
    game_id = int(game_id_raw) if game_id_raw is not None else 0
    turn = int(turn_raw) if turn_raw is not None else 0
    depth = int(depth_raw) if depth_raw is not None else 0

    setattr(args, "current_game_id", int(game_id))
    setattr(args, "current_turn", int(turn))
    setattr(args, "current_phase", str(kwargs.get("phase", "")))
    setattr(args, "current_cycle_label", str(kwargs.get("cycle_label", "")))
    setattr(args, "current_root_depth", int(depth))
    # C++ pybind signatures use int32; keep per-turn root ids bounded.
    setattr(args, "current_root_id", int(game_id % 100_000) * 10_000 + int(turn))
    setattr(args, "current_player", str(kwargs.get("player", "")))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _runtime_policy_prior_temperature() -> float:
    args = _RUNTIME_ARGS
    if args is None:
        return 1.0
    return float(getattr(args, "policy_prior_temperature", 1.0) or 1.0)


def _sample_count_key(*, game_id: int, cycle_label: str) -> tuple[int, str]:
    return (int(game_id), str(cycle_label))


def _get_agz_sample_count(args: argparse.Namespace, *, game_id: int, cycle_label: str) -> int:
    counts = getattr(args, "_agz_sample_counts", None)
    if not isinstance(counts, dict):
        counts = {}
        setattr(args, "_agz_sample_counts", counts)
    return int(counts.get(_sample_count_key(game_id=game_id, cycle_label=cycle_label), 0))


def _set_agz_sample_count(args: argparse.Namespace, *, game_id: int, cycle_label: str, value: int) -> None:
    counts = getattr(args, "_agz_sample_counts", None)
    if not isinstance(counts, dict):
        counts = {}
        setattr(args, "_agz_sample_counts", counts)
    counts[_sample_count_key(game_id=game_id, cycle_label=cycle_label)] = int(value)


def _sample_row_by_visit_temperature(
    rows: list[dict[str, Any]],
    *,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot sample from empty row set")
    temp = max(float(temperature), 1e-8)
    weights: list[float] = []
    for row in rows:
        visits = max(0.0, float(row.get("actual_iters", 0) or 0))
        weights.append(visits ** (1.0 / temp))
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        weights = [1.0 for _ in rows]
        total = float(len(rows))
    rng = random.Random(int(seed))
    threshold = rng.random() * total
    running = 0.0
    for row, weight in zip(rows, weights):
        running += float(weight)
        if running >= threshold:
            return row
    return rows[-1]


def _select_model_mcts_cpp_action(
    *,
    bundle: Any,
    cfg: Any,
    expanded: Any,
    model: Any,
) -> tuple[Any | None, dict[str, Any]]:
    del model
    args = _RUNTIME_ARGS
    if args is None:
        raise RuntimeError("CPP arena MCTS runtime args were not initialised")
    if _NATIVE_RUNTIME is None:
        raise RuntimeError("native HGB runtime was not initialised")

    player = str(expanded.player)
    value_runtime = _NATIVE_VALUE_RUNTIME_BY_PLAYER.get(str(player), _NATIVE_RUNTIME)
    valid_indices = [int(x) for x in list(expanded.valid_indices)]
    actions_by_index = list(expanded.actions_by_index)
    if not valid_indices:
        return None, {
            "selection_mode": f"model_mcts_cpp_{player}_no_valid_action",
            "valid_action_count": 0,
            "canonical_action_count": 0,
            "iterations_requested": 0,
            "iterations_used": 0,
            "policy_prior_temperature": _runtime_policy_prior_temperature(),
        }

    if len(valid_indices) == 1:
        only_idx = int(valid_indices[0])
        return actions_by_index[only_idx], {
            "selection_mode": f"model_mcts_cpp_{player}_single_valid_action",
            "valid_action_count": int(len(valid_indices)),
            "canonical_action_count": 1,
            "iterations_requested": 0,
            "iterations_used": 0,
            "chosen_q_value": None,
            "chosen_reward": None,
            "chosen_discount": None,
            "chosen_bootstrap": None,
            "chosen_child_cost": None,
            "candidate_ranking_mode": "single",
            "candidate_top5_action_reprs": [repr(actions_by_index[only_idx])],
            "candidate_top5_q_values": [],
            "candidate_top5_rewards": [],
            "candidate_top5_discounts": [],
            "candidate_top5_bootstraps": [],
            "candidate_top5_child_costs": [],
            "candidate_top5_visits": [0],
            "candidate_top5_priors": [],
            "policy_prior_temperature": _runtime_policy_prior_temperature(),
            "candidate_ranked_rows": [],
        }

    game_id = int(getattr(args, "current_game_id", 0))
    root_id = int(getattr(args, "current_root_id", 0))
    root_depth = int(getattr(args, "current_root_depth", 0))
    cycle_label = str(getattr(args, "current_cycle_label", ""))
    turn = int(getattr(args, "current_turn", -1))
    phase = str(getattr(args, "current_phase", "arena_step"))
    iterations = max(int(args.shared_root_mcts_iterations), len(valid_indices))

    payload = dict(_get_cfg_payload(args, cfg, bundle))
    root_log_path = ""
    child_log_path = ""
    log_events = bool(getattr(args, "write_mcts_visit_logs", False))
    if log_events:
        root_path, child_path = _mcts_visit_log_paths(
            args,
            game_id=int(game_id),
            cycle_label=str(cycle_label),
            phase=str(phase),
            turn=int(turn),
            player=str(player),
        )
        root_log_path = str(root_path)
        child_log_path = str(child_path)

    native = _import_native_cpp(force_build=False)
    if _NATIVE_CONTROLLER_PRIOR_RUNTIME is not None and _NATIVE_ADVERSARY_PRIOR_RUNTIME is not None:
        native_out = native.search_mcts_hgb226_value_prior_hgb(
            value_runtime,
            _NATIVE_CONTROLLER_PRIOR_RUNTIME,
            _NATIVE_ADVERSARY_PRIOR_RUNTIME,
            int(args.model_version),
            nlt._native_state_payload(bundle.env, expanded.search_state),
            payload,
            int(iterations),
            str(player),
            int(root_id),
            int(root_depth),
            int(game_id),
            int(root_id),
            _native_int_seed(int(args.seed) + int(root_id) + int(turn) + (0 if player == "controller" else 1_000_000)),
            bool(log_events),
            False,
            root_log_path,
            child_log_path,
        )
    else:
        native_out = native.search_mcts_hgb226(
            value_runtime,
            int(args.model_version),
            nlt._native_state_payload(bundle.env, expanded.search_state),
            payload,
            int(iterations),
            str(player),
            int(root_id),
            int(root_depth),
            int(game_id),
            int(root_id),
            _native_int_seed(int(args.seed) + int(root_id) + int(turn) + (0 if player == "controller" else 1_000_000)),
            bool(log_events),
            False,
            root_log_path,
            child_log_path,
        )

    q_values = [float(x) for x in list(native_out.get("root_action_values", []) or [])]
    rewards = [float(x) for x in list(native_out.get("root_action_rewards", []) or [])]
    discounts = [float(x) for x in list(native_out.get("root_action_discounts", []) or [])]
    bootstraps = [float(x) for x in list(native_out.get("root_action_bootstraps", []) or [])]
    child_cost_by_idx = {
        int(c.get("index", -1)): float(c.get("state_cost", 0.0) or 0.0)
        for c in list(native_out.get("children", []) or [])
        if int(c.get("index", -1)) >= 0
    }
    child_visits_by_idx = {
        int(c.get("index", -1)): int(c.get("visits", 0) or 0)
        for c in list(native_out.get("children", []) or [])
        if int(c.get("index", -1)) >= 0
    }
    child_node_by_idx = {
        int(c.get("index", -1)): int(c.get("node_id", -1) or -1)
        for c in list(native_out.get("children", []) or [])
        if int(c.get("index", -1)) >= 0
    }
    child_prior_by_idx = {
        int(c.get("index", -1)): float(c.get("prior", 0.0) or 0.0)
        for c in list(native_out.get("children", []) or [])
        if int(c.get("index", -1)) >= 0
    }
    alias_to_canon = {int(k): int(v) for k, v in dict(native_out.get("action_alias_to_canonical", {}) or {}).items()}
    canonical_indices = sorted({int(alias_to_canon.get(i, i)) for i in valid_indices})
    mcts_probs = [float(x) for x in list(native_out.get("mcts_root_prior", []) or [])]
    root_visits = int(native_out.get("root_visits", 0) or 0)
    root_value_sum = float(native_out.get("root_value_sum", 0.0) or 0.0)
    mcts_root_value = float(root_value_sum / root_visits) if root_visits > 0 else 0.0
    model_value_at_state = float(native_out.get("root_nn_value_controller", 0.0) or 0.0)
    state_features_for_replay, value_features_for_replay = _build_state_features_for_replay(
        args=args,
        cfg=cfg,
        bundle=bundle,
        state=expanded.search_state,
        root_id=int(root_id),
    )

    rows: list[dict[str, Any]] = []
    for canon_idx in canonical_indices:
        action = actions_by_index[canon_idx] if 0 <= canon_idx < len(actions_by_index) else None
        q = q_values[canon_idx] if 0 <= canon_idx < len(q_values) else float("nan")
        action_features_for_replay = _build_action_features_for_replay(
            bundle=bundle,
            state=expanded.search_state,
            player=str(player),
            action=action,
            canon_idx=int(canon_idx),
        )
        rows.append(
            {
                "canon_idx": int(canon_idx),
                "action_repr": repr(action),
                "action_features": action_features_for_replay,
                "q_value": float(q) if _finite(q) else (float("-inf") if player == "controller" else float("inf")),
                "reward": rewards[canon_idx] if 0 <= canon_idx < len(rewards) else 0.0,
                "discount": discounts[canon_idx] if 0 <= canon_idx < len(discounts) else 1.0,
                "bootstrap_value": bootstraps[canon_idx] if 0 <= canon_idx < len(bootstraps) else 0.0,
                "child_cost": child_cost_by_idx.get(int(canon_idx), 0.0),
                "actual_iters": child_visits_by_idx.get(int(canon_idx), 0),
                "prior": child_prior_by_idx.get(int(canon_idx), 0.0),
                "mcts_prob": mcts_probs[canon_idx] if 0 <= canon_idx < len(mcts_probs) else 0.0,
                "child_node_id": child_node_by_idx.get(int(canon_idx), ""),
            }
        )

    visited_rows = [
        r for r in rows
        if math.isfinite(float(r["q_value"])) and int(r.get("actual_iters", 0)) > 0
    ]
    if not visited_rows:
        return None, {
            "selection_mode": f"model_mcts_cpp_{player}_no_visited_children",
            "valid_action_count": int(len(valid_indices)),
            "canonical_action_count": int(len(canonical_indices)),
            "iterations_requested": int(iterations),
            "iterations_used": int(native_out.get("root_visits", 0) or 0),
            "policy_prior_temperature": _runtime_policy_prior_temperature(),
        }

    sign = 1.0 if player == "controller" else -1.0
    # Q-based final arena selection is intentionally disabled here.  The native
    # search still uses Q during UCT; after iterations, the arena action follows
    # the robust MCTS policy: most visits, then player-direction Q tie-break.
    # Old behavior:
    # reverse = player == "controller"
    # rows_sorted = sorted(
    #     finite_rows,
    #     key=lambda r: ((-float(r["q_value"])) if reverse else float(r["q_value"]), int(r["canon_idx"])),
    # )
    rows_sorted = sorted(
        visited_rows,
        key=lambda r: (
            -int(r.get("actual_iters", 0)),
            -sign * float(r["q_value"]),
            int(r["canon_idx"]),
        ),
    )
    sample_count_before = _get_agz_sample_count(args, game_id=game_id, cycle_label=cycle_label)
    use_temperature_sample = (
        bool(getattr(args, "agz_sample_initial_moves", False))
        and int(len(canonical_indices)) > 1
        and int(sample_count_before) < int(getattr(args, "agz_sample_initial_move_count", 30) or 30)
    )
    action_temperature = (
        float(getattr(args, "agz_mcts_action_temperature", 1.0) or 1.0)
        if use_temperature_sample
        else 0.0
    )
    if use_temperature_sample:
        sample_seed = (
            int(args.seed)
            + int(root_id) * 9973
            + int(turn) * 37
            + (17 if player == "controller" else 31)
        )
        best = _sample_row_by_visit_temperature(
            visited_rows,
            temperature=float(action_temperature),
            seed=int(sample_seed),
        )
        _set_agz_sample_count(
            args,
            game_id=game_id,
            cycle_label=cycle_label,
            value=int(sample_count_before) + 1,
        )
    else:
        best = rows_sorted[0]
    best_idx = int(best["canon_idx"])
    top5 = rows_sorted[:5]
    if use_temperature_sample:
        selection_mode = (
            "model_mcts_cpp_shared_root_ctrl_visit_sample_temp"
            if player == "controller"
            else "model_mcts_cpp_shared_root_adv_visit_sample_temp"
        )
    else:
        selection_mode = "model_mcts_cpp_shared_root_ctrl_most_visits" if player == "controller" else "model_mcts_cpp_shared_root_adv_most_visits"
    return actions_by_index[best_idx], {
        "selection_mode": selection_mode,
        "valid_action_count": int(len(valid_indices)),
        "canonical_action_count": int(len(canonical_indices)),
        "iterations_requested": int(iterations),
        "iterations_used": int(native_out.get("root_visits", 0) or 0),
        "chosen_q_value": float(best["q_value"]),
        "chosen_reward": float(best["reward"]),
        "chosen_discount": float(best["discount"]),
        "chosen_bootstrap": float(best["bootstrap_value"]),
        "chosen_child_cost": float(best["child_cost"]),
        "model_value_at_state": float(model_value_at_state),
        "mcts_root_value": float(mcts_root_value),
        "candidate_ranking_mode": "visits_desc_q_desc" if player == "controller" else "visits_desc_q_asc",
        "candidate_top5_action_reprs": [str(r["action_repr"]) for r in top5],
        "candidate_top5_q_values": [float(r["q_value"]) for r in top5],
        "candidate_top5_rewards": [float(r["reward"]) for r in top5],
        "candidate_top5_discounts": [float(r["discount"]) for r in top5],
        "candidate_top5_bootstraps": [float(r["bootstrap_value"]) for r in top5],
        "candidate_top5_child_costs": [float(r["child_cost"]) for r in top5],
        "candidate_top5_visits": [int(r["actual_iters"]) for r in top5],
        "candidate_top5_priors": [float(r["prior"]) for r in top5],
        "candidate_top5_mcts_probs": [float(r["mcts_prob"]) for r in top5],
        "policy_prior_temperature": _runtime_policy_prior_temperature(),
        "mcts_action_temperature": float(action_temperature),
        "mcts_action_sample_count": int(sample_count_before) + (1 if use_temperature_sample else 0),
        "state_features": [float(x) for x in state_features_for_replay],
        **value_features_for_replay,
        "policy_rows": [
            {
                "canon_action_index": int(r["canon_idx"]),
                "action_repr": str(r["action_repr"]),
                "visit_count": int(r["actual_iters"]),
                "mcts_visit_prob": float(r["mcts_prob"]),
                "model_prior": float(r["prior"]),
                "q_value": float(r["q_value"]),
                "immediate_reward": float(r["reward"]),
                "discount": float(r["discount"]),
                "bootstrap_value": float(r["bootstrap_value"]),
                "child_cost": float(r["child_cost"]),
                "action_features": [float(x) for x in list(r.get("action_features", []) or [])],
            }
            for r in rows
        ],
        "candidate_ranked_rows": [
            {
                "rank": int(rank),
                "action_repr": str(r["action_repr"]),
                "q_value": float(r["q_value"]),
                "immediate_reward": float(r["reward"]),
                "discount": float(r["discount"]),
                "bootstrap_value": float(r["bootstrap_value"]),
                "child_cost": float(r["child_cost"]),
                "mcts_child_visits": int(r["actual_iters"]),
                "mcts_child_prior": float(r["prior"]),
                "mcts_child_prob": float(r["mcts_prob"]),
                "mcts_child_node_id": r.get("child_node_id", ""),
            }
            for rank, r in enumerate(rows_sorted, start=1)
        ],
    }


def _agz_transition_record_hook(**kwargs: Any) -> None:
    recorder = _AGZ_REPLAY_RECORDER
    if recorder is None:
        return
    recorder.record_transition(**kwargs)


def _agz_final_bootstrap_value(*, cfg: Any, bundle: Any, state: Any, game_id: int, turns: int) -> float:
    if _NATIVE_RUNTIME is None:
        return 0.0
    args = _RUNTIME_ARGS
    if args is None:
        return 0.0
    payload = dict(_get_cfg_payload(args, cfg, bundle))
    out = _NATIVE_RUNTIME.infer_from_state(
        nlt._native_state_payload(bundle.env, state),
        payload,
        int(game_id % 100_000) * 10_000 + int(turns),
    )
    return float(out.get("value", out.get("raw_value", 0.0)) or 0.0)


def _agz_cycle_end_hook(**kwargs: Any) -> None:
    recorder = _AGZ_REPLAY_RECORDER
    if recorder is None:
        return
    terminal_value = _agz_final_bootstrap_value(
        cfg=kwargs["cfg"],
        bundle=kwargs["bundle"],
        state=kwargs["state"],
        game_id=int(kwargs.get("game_id", 0)),
        turns=int(kwargs.get("turns", 0)),
    )
    n = recorder.finish_cycle(
        game_id=int(kwargs.get("game_id", 0)),
        cycle_label=str(kwargs.get("cycle_label", "")),
        terminal_bootstrap_value=float(terminal_value),
    )
    if n > 0:
        print(
            f"[arena-mcts-cpp-game] agz_replay_targets={n} terminal_bootstrap={terminal_value}",
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Model_Tester arena games with native C++ HGB226 MCTS action selection.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-version", type=int, required=True)
    parser.add_argument("--feature-dim", type=int, default=226)
    parser.add_argument("--disable-model-bootstrap", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--game-id-start", type=int, default=12_000_000)
    parser.add_argument("--num-games", type=int, default=1)
    parser.add_argument("--num-parallel-games", type=int, default=5)
    parser.add_argument("--launcher-poll-sec", type=float, default=10.0)
    parser.add_argument("--launcher-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--keep-job-model-artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep per-job native_model/ and arena_model/ scratch directories. "
            "By default the launcher deletes them after copying arena CSVs upward."
        ),
    )
    parser.add_argument("--shared-root-mcts-iterations", type=int, default=1_000)
    parser.add_argument("--discount-factor", type=float, default=AGZ_DISCOUNT_FACTOR)
    parser.add_argument("--write-mcts-visit-logs", action="store_true")
    parser.add_argument("--mcts-visit-log-dir", default="")
    parser.add_argument("--native-model-export-path", default="")
    parser.add_argument("--controller-prior-model-path", default="")
    parser.add_argument("--adversary-prior-model-path", default="")
    parser.add_argument("--controller-prior-native-export-path", default="")
    parser.add_argument("--adversary-prior-native-export-path", default="")
    parser.add_argument("--role-controller-value-model-path", default="")
    parser.add_argument("--role-controller-prior-model-path", default="")
    parser.add_argument("--role-adversary-value-model-path", default="")
    parser.add_argument("--role-adversary-prior-model-path", default="")
    parser.add_argument("--force-build-native", action="store_true")
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--native-search-mode", choices=("full_tree", "full_tree_rollout"), default="full_tree")
    parser.add_argument("--rollout-count", type=int, default=10)
    parser.add_argument("--rollout-parallel-threads", type=int, default=1)
    parser.add_argument("--rollout-horizon-sec", type=float, default=0.4)
    parser.add_argument("--rollout-policy-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-probability-quantum", type=float, default=1e-6)
    parser.add_argument("--rollout-max-actions", type=int, default=4096)
    parser.add_argument("--trivial-budget-tokens", type=int, default=512)
    parser.add_argument("--skip-model-ctrl-cycle", action="store_true")
    parser.add_argument("--only-model-ctrl-cycle", action="store_true")
    parser.add_argument("--arena-time-limit-sec", type=float, default=5.0)
    parser.add_argument("--arena-max-total-turns", type=int, default=4096)
    parser.add_argument("--arena-max-controller-cleanup-steps", type=int, default=1024)
    parser.add_argument("--history-hops-min", type=int, default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_min)
    parser.add_argument("--history-hops-max", type=int, default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_max)
    parser.add_argument("--history-seed", type=int, default=DEFAULT_MODEL_TESTER_CONFIG.history_seed)
    parser.add_argument(
        "--history-hops-offset",
        type=int,
        default=0,
        help="Skip this many planned history-hop entries before assigning games.",
    )
    parser.add_argument(
        "--history-hops-unique",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_unique,
    )
    parser.add_argument(
        "--history-hops-force-zero",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MODEL_TESTER_CONFIG.history_hops_force_zero,
    )
    parser.add_argument(
        "--history-hops-prefix-stable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--uct-c", type=float, default=1.4)
    parser.add_argument("--puct-c", type=float, default=1.0)
    parser.add_argument("--policy-prior-temperature", type=float, default=1.0)
    parser.add_argument("--prior-min-prob", type=float, default=1e-8)
    parser.add_argument("--root-dirichlet-noise-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.03)
    parser.add_argument("--root-dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--agz-sample-initial-moves", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--agz-sample-initial-move-count", type=int, default=30)
    parser.add_argument("--agz-mcts-action-temperature", type=float, default=1.0)
    parser.add_argument("--no-arena-game-logs", action="store_true")
    parser.add_argument("--write-model-action-detail-logs", action="store_true")
    parser.add_argument("--agz-replay-target-csv", default="")
    parser.add_argument("--agz-replay-min-canonical-actions", type=int, default=2)
    parser.add_argument(
        "--agz-replay-sample-window-sec",
        type=float,
        default=0.0,
        help="Emit only replay states within this many simulated seconds of the history root; 0 emits all states.",
    )
    parser.add_argument("--parent-dataset-dir", default="")
    parser.add_argument("--parent-state-id", type=int, default=-1)
    parser.add_argument("--parent-root-player-filter", choices=("controller", "adversary", "any"), default="any")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _planned_history_hops(args: argparse.Namespace) -> list[int]:
    n = int(args.num_games)
    offset = max(0, int(getattr(args, "history_hops_offset", 0) or 0))
    planned_n = n + offset
    lo = int(args.history_hops_min)
    hi = int(args.history_hops_max)
    if n <= 0:
        return []
    if lo > hi:
        raise ValueError("history_hops_min must be <= history_hops_max")

    rng = random.Random(int(args.history_seed))
    if bool(args.history_hops_unique):
        population = list(range(lo, hi + 1))
        if planned_n > len(population):
            raise ValueError("num_games exceeds unique history-hop capacity")
        if bool(getattr(args, "history_hops_prefix_stable", False)):
            rng.shuffle(population)
            hops = population[:planned_n]
        else:
            hops = rng.sample(population, k=planned_n)
    else:
        hops = [int(rng.randint(lo, hi)) for _ in range(planned_n)]

    if bool(args.history_hops_force_zero):
        if not (lo <= 0 <= hi):
            raise ValueError("history hop range must include 0 when history_hops_force_zero=True")
        if 0 in hops:
            zero_idx = hops.index(0)
            hops[0], hops[zero_idx] = hops[zero_idx], hops[0]
        else:
            hops[0] = 0

    return [int(x) for x in hops[offset:]]


def _worker_command(args: argparse.Namespace, *, game_id: int, hop: int, job_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vidur.bellman_v4_adv.arena_mcts_value_runnerCPP",
        "--launcher-worker",
        "--model-path",
        str(args.model_path),
        "--model-version",
        str(int(args.model_version)),
        "--feature-dim",
        str(int(args.feature_dim)),
        "--output-dir",
        str(job_dir),
        "--game-id-start",
        str(int(game_id)),
        "--num-games",
        "1",
        "--num-parallel-games",
        "1",
        "--shared-root-mcts-iterations",
        str(int(args.shared_root_mcts_iterations)),
        "--discount-factor",
        str(float(args.discount_factor)),
        "--worker-threads",
        str(int(args.worker_threads)),
        "--trivial-budget-tokens",
        str(int(args.trivial_budget_tokens)),
        "--arena-time-limit-sec",
        str(float(args.arena_time_limit_sec)),
        "--arena-max-total-turns",
        str(int(args.arena_max_total_turns)),
        "--arena-max-controller-cleanup-steps",
        str(int(args.arena_max_controller_cleanup_steps)),
        "--history-hops-min",
        str(int(hop)),
        "--history-hops-max",
        str(int(hop)),
        "--no-history-hops-force-zero",
        "--seed",
        str(int(args.seed)),
        "--uct-c",
        str(float(args.uct_c)),
        "--puct-c",
        str(float(args.puct_c)),
        "--policy-prior-temperature",
        str(float(args.policy_prior_temperature)),
        "--prior-min-prob",
        str(float(args.prior_min_prob)),
        "--root-dirichlet-noise-enabled" if bool(args.root_dirichlet_noise_enabled) else "--no-root-dirichlet-noise-enabled",
        "--root-dirichlet-alpha",
        str(float(args.root_dirichlet_alpha)),
        "--root-dirichlet-epsilon",
        str(float(args.root_dirichlet_epsilon)),
        "--agz-sample-initial-moves" if bool(args.agz_sample_initial_moves) else "--no-agz-sample-initial-moves",
        "--agz-sample-initial-move-count",
        str(int(args.agz_sample_initial_move_count)),
        "--agz-mcts-action-temperature",
        str(float(args.agz_mcts_action_temperature)),
        "--native-search-mode",
        str(args.native_search_mode),
        "--rollout-count",
        str(int(args.rollout_count)),
        "--rollout-parallel-threads",
        str(int(args.rollout_parallel_threads)),
        "--rollout-horizon-sec",
        str(float(args.rollout_horizon_sec)),
        "--rollout-policy-temperature",
        str(float(args.rollout_policy_temperature)),
        "--rollout-probability-quantum",
        str(float(args.rollout_probability_quantum)),
        "--rollout-max-actions",
        str(int(args.rollout_max_actions)),
    ]
    if bool(args.skip_model_ctrl_cycle):
        cmd.append("--skip-model-ctrl-cycle")
    if bool(args.only_model_ctrl_cycle):
        cmd.append("--only-model-ctrl-cycle")
    if bool(args.disable_model_bootstrap):
        cmd.append("--disable-model-bootstrap")
    if bool(args.write_mcts_visit_logs):
        cmd.append("--write-mcts-visit-logs")
    if str(args.mcts_visit_log_dir or ""):
        cmd.extend(["--mcts-visit-log-dir", str(args.mcts_visit_log_dir)])
    if str(args.native_model_export_path or ""):
        export_name = Path(str(args.native_model_export_path)).name
        cmd.extend(["--native-model-export-path", str(job_dir / "native_model" / export_name)])
    if str(args.controller_prior_model_path or ""):
        cmd.extend(["--controller-prior-model-path", str(args.controller_prior_model_path)])
    if str(args.adversary_prior_model_path or ""):
        cmd.extend(["--adversary-prior-model-path", str(args.adversary_prior_model_path)])
    if str(args.controller_prior_native_export_path or ""):
        export_name = Path(str(args.controller_prior_native_export_path)).name
        cmd.extend(["--controller-prior-native-export-path", str(job_dir / "native_model" / export_name)])
    if str(args.adversary_prior_native_export_path or ""):
        export_name = Path(str(args.adversary_prior_native_export_path)).name
        cmd.extend(["--adversary-prior-native-export-path", str(job_dir / "native_model" / export_name)])
    for attr, flag in (
        ("role_controller_value_model_path", "--role-controller-value-model-path"),
        ("role_controller_prior_model_path", "--role-controller-prior-model-path"),
        ("role_adversary_value_model_path", "--role-adversary-value-model-path"),
        ("role_adversary_prior_model_path", "--role-adversary-prior-model-path"),
    ):
        value = str(getattr(args, attr, "") or "")
        if value:
            cmd.extend([flag, value])
    if bool(args.no_arena_game_logs):
        cmd.append("--no-arena-game-logs")
    if bool(args.write_model_action_detail_logs):
        cmd.append("--write-model-action-detail-logs")
    if str(getattr(args, "parent_dataset_dir", "") or ""):
        cmd.extend(["--parent-dataset-dir", str(args.parent_dataset_dir)])
        cmd.extend(["--parent-state-id", str(int(args.parent_state_id))])
        cmd.extend(["--parent-root-player-filter", str(args.parent_root_player_filter)])
    if str(args.agz_replay_target_csv or ""):
        cmd.extend(["--agz-replay-target-csv", str(job_dir / Path(str(args.agz_replay_target_csv)).name)])
        cmd.extend(["--agz-replay-min-canonical-actions", str(int(args.agz_replay_min_canonical_actions))])
        cmd.extend(["--agz-replay-sample-window-sec", str(float(args.agz_replay_sample_window_sec))])
    return cmd


def _write_planned_games(output_dir: Path, jobs: list[dict[str, Any]]) -> None:
    with (output_dir / "planned_games.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game_index", "game_id", "history_hops", "job_output_dir"])
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "game_index": int(job["game_index"]),
                    "game_id": int(job["game_id"]),
                    "history_hops": int(job["history_hops"]),
                    "job_output_dir": str(job["job_output_dir"]),
                }
            )


def _write_job_status(output_dir: Path, jobs: list[dict[str, Any]]) -> None:
    fields = [
        "timestamp",
        "game_index",
        "game_id",
        "history_hops",
        "status",
        "pid",
        "returncode",
        "elapsed_sec",
        "job_output_dir",
        "log_file",
        "arena_results_csv",
        "model_artifacts_cleaned",
    ]
    with (output_dir / "job_status.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow({k: job.get(k, "") for k in fields})


def _copy_job_arena_logs(job: dict[str, Any], output_dir: Path) -> None:
    dst_dir = output_dir / "arena_games"
    dst_dir.mkdir(parents=True, exist_ok=True)
    job_games = Path(job["job_output_dir"]) / "arena_games"
    if not job_games.exists():
        return
    for src in sorted(job_games.glob("*.csv")):
        if src.name.endswith("_model_action_details.csv"):
            continue
        shutil.copy2(src, dst_dir / src.name)


def _cleanup_job_model_artifacts(job: dict[str, Any], *, keep: bool) -> None:
    if bool(keep):
        job["model_artifacts_cleaned"] = "kept"
        return
    job_dir = Path(job["job_output_dir"])
    removed: list[str] = []
    for name in ("native_model", "arena_model"):
        path = job_dir / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(name)
    job["model_artifacts_cleaned"] = ",".join(removed) if removed else "none"


def _merge_job_results(output_dir: Path) -> int:
    result_files = sorted((output_dir / "jobs").glob("game_*_hop_*/arena_results.csv"))
    out_path = output_dir / "arena_results.csv"
    if not result_files:
        return 0
    merged = 0
    with out_path.open("w", newline="") as fout:
        writer: csv.DictWriter[str] | None = None
        for path in result_files:
            with path.open(newline="") as fin:
                reader = csv.DictReader(fin)
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=list(reader.fieldnames or []))
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    merged += 1
    return merged


def _cleanup_process_group(proc: subprocess.Popen[Any] | None, pid: int) -> None:
    if proc is None or pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
        time.sleep(0.5)
    except ProcessLookupError:
        pass
    except Exception:
        pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        pass
    try:
        proc.wait(timeout=1)
    except Exception:
        pass


def _run_parallel_launcher(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser()
    jobs_dir = output_dir / "jobs"
    logs_dir = output_dir / "launcher_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "arena_games").mkdir(parents=True, exist_ok=True)

    if bool(args.force_build_native):
        _import_native_cpp(force_build=True)

    hops = _planned_history_hops(args)
    jobs: list[dict[str, Any]] = []
    for game_index, hop in enumerate(hops):
        game_id = int(args.game_id_start) + int(game_index)
        job_dir = jobs_dir / f"game_{game_id}_hop_{int(hop)}"
        log_file = logs_dir / f"game_{game_id}_hop_{int(hop)}.log"
        jobs.append(
            {
                "timestamp": "",
                "game_index": int(game_index),
                "game_id": int(game_id),
                "history_hops": int(hop),
                "status": "pending",
                "pid": "",
                "returncode": "",
                "elapsed_sec": "",
                "job_output_dir": str(job_dir),
                "log_file": str(log_file),
                "arena_results_csv": str(job_dir / "arena_results.csv"),
                "process": None,
                "log_handle": None,
                "start_time": None,
            }
        )

    _write_planned_games(output_dir, jobs)
    _write_job_status(output_dir, jobs)

    max_parallel = max(1, int(args.num_parallel_games))
    poll_sec = max(0.25, float(args.launcher_poll_sec))
    print(
        f"[native-cpp-launcher] out={output_dir} games={len(jobs)} "
        f"parallel={max_parallel} time_limit={float(args.arena_time_limit_sec)}s",
        flush=True,
    )

    next_idx = 0
    running: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    start_all = time.time()

    try:
        while completed + failed < len(jobs):
            while next_idx < len(jobs) and len(running) < max_parallel:
                job = jobs[next_idx]
                job_dir = Path(job["job_output_dir"])
                job_dir.mkdir(parents=True, exist_ok=True)
                log_path = Path(job["log_file"])
                log_handle = log_path.open("w")
                cmd = _worker_command(
                    args,
                    game_id=int(job["game_id"]),
                    hop=int(job["history_hops"]),
                    job_dir=job_dir,
                )
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(_repo_root()),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    start_new_session=True,
                )
                job["process"] = proc
                job["log_handle"] = log_handle
                job["pid"] = int(proc.pid)
                job["status"] = "running"
                job["timestamp"] = _utc_now()
                job["start_time"] = time.time()
                running.append(job)
                print(
                    "[native-cpp-launcher] started game={} hop={} pid={}".format(
                        job["game_id"], job["history_hops"], proc.pid
                    ),
                    flush=True,
                )
                next_idx += 1

            time.sleep(poll_sec)
            still_running: list[dict[str, Any]] = []
            for job in running:
                proc = job.get("process")
                if proc is None:
                    continue
                rc = proc.poll()
                if rc is None:
                    job["elapsed_sec"] = f"{time.time() - float(job['start_time']):.3f}"
                    still_running.append(job)
                    continue

                job["returncode"] = int(rc)
                job["elapsed_sec"] = f"{time.time() - float(job['start_time']):.3f}"
                job["timestamp"] = _utc_now()
                try:
                    job["log_handle"].close()
                except Exception:
                    pass
                _cleanup_process_group(proc, int(job.get("pid") or 0))
                job["process"] = None
                job["log_handle"] = None
                gc.collect()

                if int(rc) == 0:
                    job["status"] = "ok"
                    completed += 1
                    _copy_job_arena_logs(job, output_dir)
                    _cleanup_job_model_artifacts(
                        job,
                        keep=bool(getattr(args, "keep_job_model_artifacts", False)),
                    )
                    print(
                        f"[native-cpp-launcher] done game={job['game_id']} "
                        f"hop={job['history_hops']} rc=0",
                        flush=True,
                    )
                else:
                    job["status"] = "failed"
                    failed += 1
                    print(
                        f"[native-cpp-launcher] FAILED game={job['game_id']} "
                        f"hop={job['history_hops']} rc={rc}",
                        flush=True,
                    )

            running = still_running
            merged = _merge_job_results(output_dir)
            _write_job_status(output_dir, jobs)
            print(
                "[native-cpp-launcher] progress done={} failed={} running={} pending={} merged={} elapsed={}s".format(
                    completed,
                    failed,
                    len(running),
                    len(jobs) - next_idx,
                    merged,
                    int(time.time() - start_all),
                ),
                flush=True,
            )
    finally:
        for job in running:
            try:
                handle = job.get("log_handle")
                if handle is not None:
                    handle.close()
            except Exception:
                pass
            _cleanup_process_group(job.get("process"), int(job.get("pid") or 0))
            job["process"] = None
            job["log_handle"] = None
        _merge_job_results(output_dir)
        _write_job_status(output_dir, jobs)
        gc.collect()

    if failed:
        raise RuntimeError(f"{failed} native C++ arena jobs failed; see {logs_dir}")
    print(f"[native-cpp-launcher] complete failures=0 out={output_dir}", flush=True)


def _run_single_game(args: argparse.Namespace) -> None:
    global _RUNTIME_ARGS, _AGZ_REPLAY_RECORDER
    _RUNTIME_ARGS = args
    _limit_native_threads(int(args.worker_threads))
    _prepare_native_runtime(args)

    trivial_policy = replace(
        DEFAULT_MODEL_TESTER_CONFIG.trivial_policy,
        budget_tokens=int(args.trivial_budget_tokens),
    )
    cfg = replace(
        DEFAULT_MODEL_TESTER_CONFIG,
        model_kind="classical_joblib",
        model_checkpoint_path=str(getattr(args, "harness_model_path", args.model_path)),
        output_dir=str(args.output_dir),
        num_games=int(args.num_games),
        game_id_start=int(args.game_id_start),
        environment_lang="python",
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.history_seed),
        history_hops_unique=bool(args.history_hops_unique),
        history_hops_force_zero=bool(args.history_hops_force_zero),
        arena_time_limit_sec=float(args.arena_time_limit_sec),
        arena_max_total_turns=int(args.arena_max_total_turns),
        arena_max_controller_cleanup_steps=int(args.arena_max_controller_cleanup_steps),
        bootstrap_model_version=int(args.model_version),
        write_arena_game_logs=not bool(args.no_arena_game_logs),
        write_model_action_detail_logs=bool(args.write_model_action_detail_logs),
        arena_num_processes=1,
        arena_worker_threads=1,
        arena_mp_start_method="spawn",
        trivial_policy=trivial_policy,
        skip_model_ctrl_cycle=bool(args.skip_model_ctrl_cycle),
        only_model_ctrl_cycle=bool(args.only_model_ctrl_cycle),
    )

    if str(args.agz_replay_target_csv or ""):
        replay_path = Path(str(args.agz_replay_target_csv)).expanduser()
        if replay_path.exists():
            replay_path.unlink()
        native_backup_fn = getattr(_NATIVE_MODULE, "discounted_trajectory_targets", None)
        _AGZ_REPLAY_RECORDER = AlphaGoZeroReplayRecorder(
            replay_path,
            min_canonical_actions=int(args.agz_replay_min_canonical_actions),
            discount_factor=float(args.discount_factor),
            sample_window_sec=float(args.agz_replay_sample_window_sec),
            target_backup_fn=native_backup_fn,
            target_backup_backend="native" if native_backup_fn is not None else "python",
        )
        print(
            "[arena-mcts-cpp-game] "
            f"agz_replay_sample_window_sec={float(args.agz_replay_sample_window_sec)} "
            f"target_backup_backend={'native' if native_backup_fn is not None else 'python'}",
            flush=True,
        )
    else:
        _AGZ_REPLAY_RECORDER = None

    old_selector = tester_runner._select_model_depth1_action
    old_context_hook = getattr(tester_runner, "_model_action_selection_context_hook", None)
    old_agz_transition_hook = getattr(tester_runner, "_agz_transition_record_hook", None)
    old_agz_cycle_end_hook = getattr(tester_runner, "_agz_cycle_end_hook", None)
    old_make_base_snapshot = getattr(tester_runner, "_make_base_snapshot", None)
    try:
        tester_runner._select_model_depth1_action = _select_model_mcts_cpp_action
        tester_runner._model_action_selection_context_hook = _set_runtime_selection_context
        tester_runner._agz_transition_record_hook = _agz_transition_record_hook
        tester_runner._agz_cycle_end_hook = _agz_cycle_end_hook
        if old_make_base_snapshot is not None:
            tester_runner._make_base_snapshot = _make_parent_base_snapshot_hook(old_make_base_snapshot, args)
        out_csv = tester_runner.run_model_vs_trivial_tester(cfg)
        print(f"[arena-mcts-cpp-game] completed {out_csv}")
        if _NATIVE_MODEL_EXPORT_PATH is not None:
            print(f"[arena-mcts-cpp-game] native_model_export={_NATIVE_MODEL_EXPORT_PATH}")
    finally:
        tester_runner._select_model_depth1_action = old_selector
        tester_runner._model_action_selection_context_hook = old_context_hook
        tester_runner._agz_transition_record_hook = old_agz_transition_hook
        tester_runner._agz_cycle_end_hook = old_agz_cycle_end_hook
        if old_make_base_snapshot is not None:
            tester_runner._make_base_snapshot = old_make_base_snapshot
        _AGZ_REPLAY_RECORDER = None
        _CFG_PAYLOAD_CACHE.clear()
        gc.collect()


def main() -> None:
    args = _parse_args()
    if int(args.num_games) > 1 and not bool(args.launcher_worker):
        _run_parallel_launcher(args)
        return
    _run_single_game(args)


if __name__ == "__main__":
    main()
