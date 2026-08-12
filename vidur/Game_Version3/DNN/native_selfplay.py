from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .dnn_spec import DEFAULT_DNN_SPEC
from .replay_write import ReplayWriter, RootSample


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "vidur" / "Game_Version3").is_dir():
            return parent
    return p.parents[3]


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _repo_root() / p


def _read_prefill_profile(path: str | Path) -> tuple[list[int], list[float]]:
    p = _resolve_repo_path(path)
    if not p.exists():
        return [], []

    tokens: list[int] = []
    times: list[float] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tokens.append(int(float(row.get("prefill_tokens", ""))))
                times.append(float(row.get("prefill_time_seconds", "")))
            except Exception:
                continue
    return tokens, times


def _execution_predictor_payload(source: Any) -> dict[str, Any]:
    """Return native-loadable sklearn predictor tables from a simulator/predictor.

    Native timing must mirror Python timing for Bellman target parity.  The
    sklearn predictor stores dense prediction caches as memmaps; those arrays
    are small enough for GV3 and can be passed through pybind once per task.
    """
    predictor = getattr(source, "_execution_time_predictor", source)
    predictions = getattr(predictor, "_predictions", None)
    if not predictions:
        return {}

    component_tables: dict[str, dict[str, Any]] = {}
    for name, table in dict(predictions).items():
        mmap = getattr(table, "mmap", None)
        if mmap is None:
            continue
        component_tables[str(name)] = {
            "kind": str(getattr(table, "kind", "")),
            "max_tokens": int(getattr(table, "max_tokens", 0)),
            "max_batch_size": int(getattr(table, "max_batch_size", 0)),
            "kv_gran": int(getattr(table, "kv_gran", 1)),
            "prefill_gran": int(getattr(table, "prefill_gran", 1)),
            "shape": [int(x) for x in tuple(getattr(mmap, "shape", ()))],
            "values": [float(x) for x in mmap.reshape(-1).tolist()],
        }

    if not component_tables:
        return {}

    pred_cfg = getattr(predictor, "_config", None)
    replica_cfg = getattr(predictor, "_replica_config", None)
    model_cfg = getattr(predictor, "_model_config", None)
    runtime_cfg = {
        "num_layers_per_pipeline_stage": int(
            getattr(predictor, "_num_layers_per_pipeline_stage", 1)
        ),
        "tensor_parallel_size": int(getattr(replica_cfg, "tensor_parallel_size", 1)),
        "num_pipeline_stages": int(getattr(replica_cfg, "num_pipeline_stages", 1)),
        "post_attn_norm": bool(getattr(model_cfg, "post_attn_norm", True)),
        "skip_cpu_overhead_modeling": bool(
            getattr(pred_cfg, "skip_cpu_overhead_modeling", False)
        ),
        "attention_prefill_batching_overhead_fraction": float(
            getattr(predictor, "_attention_prefill_batching_overhead_fraction", 0.0)
        ),
        "attention_decode_batching_overhead_fraction": float(
            getattr(predictor, "_attention_decode_batching_overhead_fraction", 0.0)
        ),
        "nccl_cpu_launch_overhead_ms": float(
            getattr(pred_cfg, "nccl_cpu_launch_overhead_ms", 0.0)
        ),
        "nccl_cpu_skew_overhead_per_device_ms": float(
            getattr(pred_cfg, "nccl_cpu_skew_overhead_per_device_ms", 0.0)
        ),
    }
    return {
        "execution_predictor_runtime_config": runtime_cfg,
        "execution_predictor_component_tables": component_tables,
    }


def attach_execution_predictor_payload(payload: dict[str, Any], source: Any) -> dict[str, Any]:
    payload.update(_execution_predictor_payload(source))
    return payload


class _FixedPlayerTorchScriptWrapper(nn.Module):
    def __init__(self, model: nn.Module, player: str) -> None:
        super().__init__()
        self.model = model
        self.player = str(player)

    def forward(
        self,
        prefill_req_features: torch.Tensor,
        decode_req_features: torch.Tensor,
        global_features: torch.Tensor,
        prefill_req_mask: torch.Tensor,
        decode_req_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(
            player=self.player,
            prefill_req_features=prefill_req_features,
            decode_req_features=decode_req_features,
            global_features=global_features,
            prefill_req_mask=prefill_req_mask,
            decode_req_mask=decode_req_mask,
            action_mask=action_mask,
        )


def _weights_fingerprint(weights_path: Path) -> str:
    try:
        st = weights_path.stat()
        return f"{int(st.st_size)}_{int(st.st_mtime_ns)}"
    except Exception:
        return "unknown"


def export_torchscript_pair(
    *,
    model: nn.Module,
    model_version: int,
    weights_path: str | Path,
    out_dir: str | Path,
) -> str:
    """Export fixed-player TorchScript modules and return native model spec."""
    weights = Path(weights_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fp = _weights_fingerprint(weights)
    prefix = f"gv3_v{int(model_version):06d}_{fp}"
    controller_path = out / f"{prefix}_controller.pt"
    adversary_path = out / f"{prefix}_adversary.pt"

    if controller_path.exists() and adversary_path.exists():
        return f"{controller_path}||{adversary_path}"

    was_training = bool(model.training)
    model.eval()
    device = next(model.parameters()).device
    spec = getattr(model, "spec", DEFAULT_DNN_SPEC)

    examples = {
        "controller": (
            torch.zeros((1, int(spec.n_prefill_req), int(spec.d_prefill_req)), dtype=torch.float32, device=device),
            torch.zeros((1, int(spec.n_decode_req), int(spec.d_decode_req)), dtype=torch.float32, device=device),
            torch.zeros((1, int(spec.d_global)), dtype=torch.float32, device=device),
            torch.zeros((1, int(spec.n_prefill_req)), dtype=torch.bool, device=device),
            torch.zeros((1, int(spec.n_decode_req)), dtype=torch.bool, device=device),
            torch.ones((1, int(spec.num_actions_controller)), dtype=torch.bool, device=device),
        ),
        "adversary": (
            torch.zeros((1, int(spec.n_prefill_req), int(spec.d_prefill_req)), dtype=torch.float32, device=device),
            torch.zeros((1, int(spec.n_decode_req), int(spec.d_decode_req)), dtype=torch.float32, device=device),
            torch.zeros((1, int(spec.d_global)), dtype=torch.float32, device=device),
            torch.zeros((1, int(spec.n_prefill_req)), dtype=torch.bool, device=device),
            torch.zeros((1, int(spec.n_decode_req)), dtype=torch.bool, device=device),
            torch.ones((1, int(spec.num_actions_adversary)), dtype=torch.bool, device=device),
        ),
    }

    paths = {"controller": controller_path, "adversary": adversary_path}
    with torch.inference_mode():
        for player, example in examples.items():
            path = paths[player]
            if path.exists():
                continue
            wrapper = _FixedPlayerTorchScriptWrapper(model, player).eval()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                traced = torch.jit.trace(wrapper, example, strict=False)
            traced = torch.jit.freeze(traced.eval())
            tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
            traced.save(str(tmp))
            os.replace(tmp, path)

    if was_training:
        model.train()
    return f"{controller_path}||{adversary_path}"


def _cfg_payload(
    cfg: Any,
    *,
    torchscript_model_spec: str,
    initial_history_signatures: list[str] | None = None,
) -> dict[str, Any]:
    gv2 = cfg.game_v2
    features = gv2.features
    profile_tokens, profile_times = _read_prefill_profile(gv2.legacy_mcts.prefill_profile_path)
    # Python GV3 uses PrefillProfile.load_or_generate(..., slowdown=...) for
    # adversary prefill SLO lookup and controller LST ETA.  The CSV stores raw
    # predictor timings, so the native payload must carry the slowed values.
    slowdown = float(gv2.legacy_mcts.prefill_slowdown or 1.0)
    if slowdown != 1.0:
        profile_times = [float(t) * slowdown for t in profile_times]
    search = gv2.mcts_search

    payload = {
        "n_prefill_req": int(features.n_prefill_req),
        "d_prefill_req": int(features.d_prefill_req),
        "n_decode_req": int(features.n_decode_req),
        "d_decode_req": int(features.d_decode_req),
        "d_global": int(features.d_global),
        "controller_action_space_size": int(DEFAULT_DNN_SPEC.num_actions_controller),
        "adversary_action_space_size": int(DEFAULT_DNN_SPEC.num_actions_adversary),
        "prefill_total_den": float(features.prefill_total_den),
        "prefill_remaining_den": float(features.prefill_remaining_den),
        "decode_total_den": float(features.decode_total_den),
        "decode_remaining_den": float(features.decode_remaining_den),
        "decode_processed_den": float(features.decode_processed_den),
        "age_den_sec": float(features.age_den_sec),
        "lateness_den_sec": float(features.lateness_den_sec),
        "slack_den_sec": float(features.slack_den_sec),
        "prefill_slo_den_sec": float(features.prefill_slo_den_sec),
        "decode_slo_den_sec": float(features.decode_slo_den_sec),
        "objective_cost_den": float(features.objective_cost_den),
        "total_lateness_den": float(features.total_lateness_den),
        "system_load_den": float(features.system_load_den),
        "active_prefill_count_den": float(features.active_prefill_count_den),
        "active_decode_count_den": float(features.active_decode_count_den),
        "active_total_count_den": float(features.active_total_count_den),
        "total_remaining_prefill_den": float(features.total_remaining_prefill_den),
        "total_remaining_decode_den": float(features.total_remaining_decode_den),
        "total_decode_generated_active_den": float(features.total_decode_generated_active_den),
        "violated_count_den": float(features.violated_count_den),
        "prefill_near_drop_den": float(features.prefill_near_drop_den),
        "decode_near_drop_den": float(features.decode_near_drop_den),
        "recent_launch_count_den": float(features.recent_launch_count_den),
        "recent_launch_prefill_den": float(features.recent_launch_prefill_den),
        "decode_credit_den": float(features.decode_credit_den),
        "near_drop_lateness_low_sec": float(features.near_drop_lateness_low_sec),
        "near_drop_lateness_high_sec": float(features.near_drop_lateness_high_sec),
        "launch_ewma_alpha": float(features.launch_ewma_alpha),
        "launch_ewma_window_sec": float(features.launch_ewma_window_sec),
        "decode_sample_seed_offset": int(features.decode_sample_seed_offset),
        "adversary_tick_sec": float(gv2.timing.adversary_tick_sec),
        "launch_window_sec": float(gv2.timing.launch_window_sec),
        "max_requests_per_launch_window": int(gv2.timing.max_requests_per_launch_window),
        "prefill_window_cap_tokens": int(
            gv2.request.target_prefill_tokens_per_request_avg_window
            * gv2.timing.max_requests_per_launch_window
        ),
        "max_prefill_tokens_per_request": int(gv2.request.max_prefill_tokens_per_request),
        "max_decode_tokens_per_request": int(gv2.request.max_decode_tokens_per_request),
        "min_decode_tokens_per_request": int(gv2.request.min_decode_tokens_per_request),
        "decode_slo_time_default": float((gv2.legacy_mcts.decode_slos or (50.0,))[0]) / 1000.0,
        "auto_drop_lateness_sec": float(gv2.cost.auto_drop_lateness_sec),
        "drop_cost": float(gv2.cost.drop_cost),
        "controller_noop_prefill_only_jump_to_next_adv_tick": bool(
            gv2.timing.controller_noop_prefill_only_jump_to_next_adv_tick
        ),
        "enforce_nonnegative_decode_credits": bool(gv2.credits.enforce_nonnegative_decode_credits),
        "decode_credit_mint_per_prefill_complete": int(gv2.credits.decode_credit_mint_per_prefill_complete),
        "discount_factor": float(search.discount_factor),
        "discount_time_denominator_sec": float(search.discount_time_denominator_sec or 0.015725797204323228),
        "reward_knee": float(search.reward_knee),
        "reward_max_penalty": float(search.reward_max_penalty),
        "reward_tail_alpha": float(search.reward_tail_alpha or (1.0 / 15.0)),
        "root_dirichlet_noise_enabled": bool(search.root_dirichlet_noise_enabled),
        "root_dirichlet_alpha": float(search.root_dirichlet_alpha),
        "root_dirichlet_total_concentration": float(search.root_dirichlet_total_concentration),
        "root_dirichlet_epsilon": float(search.root_dirichlet_epsilon),
        "prefill_profile_tokens": profile_tokens,
        "prefill_profile_times": profile_times,
        "torchscript_model_spec": str(torchscript_model_spec),
        "initial_history_signatures": list(initial_history_signatures or []),
    }
    return payload


def _empty_initial_state_payload() -> dict[str, Any]:
    return {
        "sim_time": 0.0,
        "next_request_id": 0,
        "requests": [],
        "stats": {
            "slo_violations": 0,
            "slo_lateness_sum": 0.0,
            "recent_arrivals": [],
            "active_request_ids": [],
            "completed_request_ids": [],
            "dropped_request_ids": [],
            "stopped_decode_request_ids": [],
            "violated_request_ids": [],
            "decode_tokens_counted_by_id": {},
            "per_request_prefill_lateness_by_id": {},
            "per_request_decode_lateness_by_id": {},
            "decode_next_deadline_by_id": {},
            "next_adv_tick": 0.0,
            "last_adv_tick": -1.0,
            "decode_credit_balance": 0,
            "decode_credit_available": 0,
        },
    }


def _tensor_1d(values: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(list(values or []), dtype=dtype)


def _tensor_2d(values: Any, rows: int, cols: int, *, dtype: torch.dtype) -> torch.Tensor:
    t = torch.tensor(list(values or []), dtype=dtype)
    expected = int(rows) * int(cols)
    if int(t.numel()) != expected:
        raise ValueError(f"native tensor has {int(t.numel())} values, expected {expected} ({rows}x{cols})")
    return t.reshape(int(rows), int(cols))


def _sanitize_feature_matrix(x: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
    if mask is not None:
        x = x * mask.to(dtype=x.dtype).view(-1, 1)
    return x


def _sanitize_feature_vector(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _native_sample_to_root_sample(obj: dict[str, Any]) -> RootSample:
    inputs = dict(obj["inputs"])
    prefill_n = int(inputs.get("prefill_req_n", DEFAULT_DNN_SPEC.n_prefill_req))
    prefill_d = int(inputs.get("prefill_req_d", DEFAULT_DNN_SPEC.d_prefill_req))
    decode_n = int(inputs.get("decode_req_n", DEFAULT_DNN_SPEC.n_decode_req))
    decode_d = int(inputs.get("decode_req_d", DEFAULT_DNN_SPEC.d_decode_req))
    req_n = int(inputs.get("req_n", prefill_n + decode_n))
    req_d = int(inputs.get("req_d", max(prefill_d, decode_d)))

    targets = dict(obj["targets"])
    prefill_mask = _tensor_1d(inputs.get("prefill_req_mask"), dtype=torch.bool)
    decode_mask = _tensor_1d(inputs.get("decode_req_mask"), dtype=torch.bool)
    req_mask = _tensor_1d(inputs.get("req_mask"), dtype=torch.bool)
    prefill_features = _sanitize_feature_matrix(
        _tensor_2d(inputs.get("prefill_req_features"), prefill_n, prefill_d, dtype=torch.float32),
        mask=prefill_mask,
    )
    decode_features = _sanitize_feature_matrix(
        _tensor_2d(inputs.get("decode_req_features"), decode_n, decode_d, dtype=torch.float32),
        mask=decode_mask,
    )
    req_features = _sanitize_feature_matrix(
        _tensor_2d(inputs.get("req_features"), req_n, req_d, dtype=torch.float32),
        mask=req_mask,
    )
    return {
        "feature_version": int(obj["feature_version"]),
        "game_id": int(obj["game_id"]),
        "root_id": int(obj["root_id"]),
        "root_node_id": int(obj["root_node_id"]),
        "root_depth": int(obj["root_depth"]),
        "player": str(obj["player"]),
        "inputs": {
            "prefill_req_features": prefill_features,
            "decode_req_features": decode_features,
            "global_features": _sanitize_feature_vector(_tensor_1d(inputs.get("global_features"), dtype=torch.float32)),
            "prefill_req_mask": prefill_mask,
            "decode_req_mask": decode_mask,
            "action_mask": _tensor_1d(inputs.get("action_mask"), dtype=torch.bool),
            "req_features": req_features,
            "req_mask": req_mask,
        },
        "targets": {
            "policy": _tensor_1d(targets.get("policy"), dtype=torch.float32),
            "value": torch.tensor(float(targets.get("value", 0.0)), dtype=torch.float32),
        },
        "meta": dict(obj.get("meta", {}) or {}),
    }


def run_native_selfplay_to_writers(
    *,
    cfg: Any,
    task: dict[str, Any],
    model: nn.Module,
    writer_train: ReplayWriter,
    writer_eval: ReplayWriter,
) -> dict[str, Any]:
    import vidur.mcts.mcts_native_gv2 as native

    model_version = int(task.get("model_version", 0))
    bootstrap_generation = int(task.get("generation", model_version))
    use_model_bootstrap = bool(int(bootstrap_generation) > 0 and int(model_version) > 0)
    torchscript_spec = ""
    runtime = native.NativeTorchScriptInferRuntimeGV2(str(cfg.model.device))
    if use_model_bootstrap:
        torchscript_spec = export_torchscript_pair(
            model=model,
            model_version=model_version,
            weights_path=Path(task["weights_path"]),
            out_dir=Path(cfg.native_torchscript_dir),
        )
        runtime.load_models({model_version: torchscript_spec})

    initial_history_signatures = [
        str(sig)
        for sig in list(task.get("history_seen_signatures", []) or [])
        if isinstance(sig, str)
    ]
    cfg_payload = _cfg_payload(
        cfg,
        torchscript_model_spec=torchscript_spec,
        initial_history_signatures=initial_history_signatures,
    )
    cfg_payload["use_model_bootstrap"] = bool(use_model_bootstrap)
    for task_key, payload_key in (
        ("native_history_trace_log_path", "native_history_trace_log_path"),
        ("history_trace_log_path", "native_history_trace_log_path"),
        ("native_frontier_log_path", "native_frontier_log_path"),
        ("frontier_log_path", "native_frontier_log_path"),
        ("native_depth1_search_log_path", "native_depth1_search_log_path"),
        ("depth1_search_log_path", "native_depth1_search_log_path"),
        ("native_depth1_details_log_path", "native_depth1_details_log_path"),
        ("depth1_details_log_path", "native_depth1_details_log_path"),
    ):
        if task.get(task_key):
            cfg_payload[payload_key] = str(task[task_key])
    native_out = native.generate_selfplay_samples_torchscript(
        runtime,
        model_version,
        _empty_initial_state_payload(),
        cfg_payload,
        int(task["game_id"]),
        int(task["num_roots"]),
        int(task["start_root_id"]),
        int(task["start_root_depth"]),
        str(task["start_player"]),
        int(task["feature_version"]),
        int(task["adv_iterations_per_root"]),
        int(task["cont_iterations_per_root"]),
        int(task["history_hops_min"]),
        int(task["history_hops_max"]),
        int(task["history_seed"]),
        int(task["history_max_total_steps"]),
        int(task["max_forced_hops_per_root"]),
        float(task.get("eval_split_ratio", 0.0)),
        int(task.get("eval_split_seed", 0)),
        int(task.get("action_seed_base", 0)),
        bool(task.get("allow_duplicate_history_fallback", True)),
    )

    history_signatures = list(native_out.get("history_signatures", []) or [])
    for idx, raw in enumerate(list(native_out.get("samples", []))):
        sample = _native_sample_to_root_sample(dict(raw))
        if idx < len(history_signatures):
            sample.setdefault("meta", {})["history_signature"] = history_signatures[idx]
        is_eval = bool(sample.get("meta", {}).get("is_eval", False))
        if is_eval:
            writer_eval.add(sample)
        else:
            writer_train.add(sample)

    stats = dict(native_out.get("stats", {}) or {})
    stats["history_signatures"] = history_signatures
    return stats
