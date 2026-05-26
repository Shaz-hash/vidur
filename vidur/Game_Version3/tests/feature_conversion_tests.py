from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch

from ..DNN.infer import build_model_inputs
from ..DNN.selfPlay import SelfPlayRunner
from ..config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG
from ..mctsDNN import VidurMCTS
from ..multiProcessUtils import _build_env_and_simulator, _set_global_seeds


META_DECODE_CREDIT_BAL = -9_100_005


PREFILL_FEATURE_LABELS = [
    "remaining_prefill_norm",
    "total_prefill_norm",
    "processed_prefill_frac",
    "age_norm",
    "prefill_lateness_norm",
    "prefill_slack_centered",
    "prefill_slo_norm",
    "violated_bit",
    "prefill_late_gt_0p5",
    "prefill_late_ge_1p5",
]

DECODE_FEATURE_LABELS = [
    "remaining_decode_norm",
    "total_decode_norm",
    "processed_decode_norm",
    "processed_decode_frac",
    "age_norm",
    "total_lateness_norm",
    "decode_slack_centered",
    "decode_slo_norm",
    "violated_bit",
    "lateness_gt_0p5",
    "lateness_ge_1p5",
    "processed_decode_gt_216",
    "processed_decode_gt_512",
]

GLOBAL_FEATURE_LABELS = [
    "player_is_controller",
    "player_is_adversary",
    "objective_cost_norm",
    "slo_violations_norm",
    "slo_lateness_sum_norm",
    "active_prefill_count_norm",
    "active_decode_count_norm",
    "active_total_count_norm",
    "total_remaining_prefill_norm",
    "total_remaining_decode_norm",
    "total_decode_processed_active_norm",
    "violated_active_count_norm",
    "prefill_late_0p5_1p5_count_norm",
    "prefill_late_ge_1p5_count_norm",
    "decode_late_0p5_1p5_count_norm",
    "decode_late_ge_1p5_count_norm",
    "recent_launch_count_norm",
    "recent_launch_prefill_norm",
    "remaining_launch_request_headroom_norm",
    "remaining_launch_prefill_headroom_norm",
    "launch_ewma_norm",
    "decode_credit_norm",
    "has_prefill_bit",
    "has_decode_bit",
]


class _NoopWriter:
    def add(self, sample: Any) -> None:
        del sample

    def close(self) -> None:
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_output_dir() -> Path:
    return _repo_root() / "simulator_output" / "Game_Version3" / "tests"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _norm01(x: float, denom: float) -> float:
    if denom <= 0.0:
        return 0.0
    return _clip(float(x) / float(denom), 0.0, 1.0)


def _centered01(x: float, radius: float) -> float:
    if radius <= 0.0:
        return 0.5
    clipped = _clip(float(x), -float(radius), float(radius))
    return (clipped + float(radius)) / (2.0 * float(radius))


def _arrived_at(req: Any) -> float:
    return _safe_float(getattr(req, "_arrived_at", getattr(req, "arrived_at", 0.0)), 0.0)


def _prefill_remaining(req: Any) -> int:
    total = _safe_int(getattr(req, "num_prefill_tokens", 0), 0)
    done = _safe_int(getattr(req, "num_processed_prefill_tokens", 0), 0)
    return max(0, total - done)


def _decode_remaining(req: Any) -> int:
    total = _safe_int(getattr(req, "_num_decode_tokens", getattr(req, "num_decode_tokens", 0)), 0)
    done = _safe_int(getattr(req, "num_processed_decode_tokens", 0), 0)
    return max(0, total - done)


def _prefill_deadline(req: Any) -> float | None:
    slo = getattr(req, "_prefill_slo_time", None)
    if slo is None:
        return None
    arrived = getattr(req, "_arrived_at", getattr(req, "arrived_at", 0.0))
    queued = getattr(req, "queued_at", arrived)
    return _safe_float(queued) + _safe_float(slo)


def _decode_deadline(req: Any, state: Any) -> float | None:
    rid = _safe_int(getattr(req, "id", -1), -1)
    stats = getattr(state, "stats", None)
    if stats is None:
        return None
    deadline = _safe_float(getattr(stats, "decode_next_deadline_by_id", {}).get(rid, 0.0), 0.0)
    if deadline > 0.0:
        return deadline
    decode_slo = _safe_float(getattr(req, "_decode_slo_time", getattr(req, "decode_slo_time", 0.0)), 0.0)
    prefill_completed_at = _safe_float(getattr(req, "_prefill_completed_at", 0.0), 0.0)
    if prefill_completed_at > 0.0 and decode_slo > 0.0:
        return prefill_completed_at + decode_slo
    return None


def _is_prefill_request(req: Any) -> bool:
    return (
        not bool(getattr(req, "completed", False))
        and not bool(getattr(req, "_is_prefill_complete", getattr(req, "is_prefill_complete", False)))
        and _prefill_remaining(req) > 0
    )


def _is_decode_request(req: Any) -> bool:
    if bool(getattr(req, "completed", False)):
        return False
    if not bool(getattr(req, "_is_prefill_complete", getattr(req, "is_prefill_complete", False))):
        return False
    return _decode_remaining(req) > 0


def _is_violated(state: Any, req: Any) -> bool:
    rid = _safe_int(getattr(req, "id", -1), -1)
    violated = set(getattr(getattr(state, "stats", None), "violated_request_ids", set()) or set())
    return rid in violated


def _request_total_lateness(state: Any, req: Any, sim_time: float) -> float:
    rid = _safe_int(getattr(req, "id", -1), -1)
    stats = getattr(state, "stats", None)
    if stats is None:
        return 0.0
    pref = float(getattr(stats, "per_request_prefill_lateness", {}).get(rid, 0.0))
    dec = float(getattr(stats, "per_request_decode_lateness", {}).get(rid, 0.0))
    if pref <= 0.0:
        deadline = _prefill_deadline(req)
        if deadline is not None:
            pref = max(0.0, float(sim_time) - float(deadline))
    return max(0.0, pref + dec)


def _recent_launch_summary(state: Any, sim_time: float, *, window_sec: float, alpha: float) -> tuple[float, float, float]:
    launch_count = 0.0
    launch_prefill = 0.0
    ewma = 0.0
    for item in list(getattr(getattr(state, "stats", None), "recent_arrivals", []) or []):
        ts = None
        cnt = 0
        prefill = 0
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            ts = float(item[0])
            cnt = int(item[1])
            prefill = int(item[2])
        elif isinstance(item, dict):
            if "timestamp" in item:
                ts = float(item["timestamp"])
            elif "time" in item:
                ts = float(item["time"])
            cnt = int(item.get("count", item.get("requests", 0)))
            prefill = int(item.get("prefill_tokens", item.get("tokens", 0)))
        elif isinstance(item, (int, float)):
            ts = float(item)
            cnt = 1

        if ts is None:
            continue
        dt = max(0.0, float(sim_time) - float(ts))
        if dt > float(window_sec):
            continue
        launch_count += max(0.0, float(cnt))
        launch_prefill += max(0.0, float(prefill))
        ewma += max(0.0, float(cnt)) * math.exp(-float(alpha) * dt)

    return launch_count, launch_prefill, ewma


def _decode_credit_available(state: Any) -> float:
    counted = getattr(getattr(state, "stats", None), "decode_tokens_counted", {}) or {}
    raw = _safe_float(counted.get(META_DECODE_CREDIT_BAL, 0.0), 0.0)
    return max(0.0, raw)


def _request_lookup(env: Any, state: Any) -> dict[int, Any]:
    if hasattr(env, "_build_request_lookup"):
        return dict(env._build_request_lookup(state.simulator, state=state))
    req_map = getattr(env, "_req_map", lambda sim: {})(state.simulator)
    return {int(k): v for k, v in dict(req_map).items() if not bool(getattr(v, "completed", False))}


def _expected_features(env: Any, state: Any, player: str, cfg: Any) -> dict[str, Any]:
    F = cfg.features
    spec_cfg = cfg
    sim_time = _safe_float(getattr(state.simulator, "_time", 0.0), 0.0)
    stats = getattr(state, "stats", None)
    lookup = _request_lookup(env, state)
    reqs = list(lookup.values())

    prefill_reqs = [r for r in reqs if _is_prefill_request(r)]
    decode_reqs = [r for r in reqs if _is_decode_request(r)]

    def prefill_key(req: Any) -> tuple[float, float, int]:
        deadline = _prefill_deadline(req)
        time_left = float("inf") if deadline is None else float(deadline) - float(sim_time)
        late = _request_total_lateness(state, req, sim_time)
        rid = _safe_int(getattr(req, "id", 0), 0)
        return time_left, -late, rid

    def decode_key(req: Any) -> tuple[int, float, int, int, int]:
        late = _request_total_lateness(state, req, sim_time)
        processed = _safe_int(getattr(req, "num_processed_decode_tokens", 0), 0)
        remaining = _decode_remaining(req)
        rid = _safe_int(getattr(req, "id", 0), 0)
        violated = 1 if _is_violated(state, req) else 0
        return -violated, -late, -processed, -remaining, rid

    prefill_reqs.sort(key=prefill_key)
    decode_reqs.sort(key=decode_key)

    prefill_feat = torch.zeros((1, int(F.n_prefill_req), int(F.d_prefill_req)), dtype=torch.float32)
    decode_feat = torch.zeros((1, int(F.n_decode_req), int(F.d_decode_req)), dtype=torch.float32)
    prefill_mask = torch.zeros((1, int(F.n_prefill_req)), dtype=torch.bool)
    decode_mask = torch.zeros((1, int(F.n_decode_req)), dtype=torch.bool)

    prefill_debug: list[dict[str, Any]] = []
    decode_debug: list[dict[str, Any]] = []

    for i, req in enumerate(prefill_reqs[: int(F.n_prefill_req)]):
        rid = _safe_int(getattr(req, "id", -1), -1)
        total_prefill = _safe_int(getattr(req, "num_prefill_tokens", 0), 0)
        rem_prefill = _prefill_remaining(req)
        done_prefill = max(0, total_prefill - rem_prefill)
        age = max(0.0, sim_time - _arrived_at(req))
        prefill_late = _safe_float(getattr(stats, "per_request_prefill_lateness", {}).get(rid, 0.0), 0.0)
        deadline = _prefill_deadline(req)
        slack = 0.0 if deadline is None else float(deadline) - sim_time
        prefill_slo = _safe_float(getattr(req, "_prefill_slo_time", getattr(req, "prefill_slo_time", 0.0)), 0.0)
        violated_bit = 1.0 if _is_violated(state, req) else 0.0
        processed_frac = 0.0 if total_prefill <= 0 else _clip(float(done_prefill) / float(max(1, total_prefill)), 0.0, 1.0)

        values = [
            _norm01(rem_prefill, F.prefill_remaining_den),
            _norm01(total_prefill, F.prefill_total_den),
            processed_frac,
            _norm01(age, F.age_den_sec),
            _norm01(prefill_late, F.lateness_den_sec),
            _centered01(slack, F.slack_den_sec),
            _norm01(prefill_slo, F.prefill_slo_den_sec),
            violated_bit,
            1.0 if prefill_late > float(F.near_drop_lateness_low_sec) else 0.0,
            1.0 if prefill_late >= float(F.near_drop_lateness_high_sec) else 0.0,
        ]
        prefill_feat[0, i, :] = torch.tensor(values, dtype=torch.float32)
        prefill_mask[0, i] = True
        prefill_debug.append(
            {
                "slot": int(i),
                "request_id": int(rid),
                "remaining_prefill": int(rem_prefill),
                "total_prefill": int(total_prefill),
                "done_prefill": int(done_prefill),
                "age": float(age),
                "deadline": "" if deadline is None else float(deadline),
                "slack": float(slack),
                "prefill_late": float(prefill_late),
            }
        )

    for i, req in enumerate(decode_reqs[: int(F.n_decode_req)]):
        rid = _safe_int(getattr(req, "id", -1), -1)
        total_decode = _safe_int(getattr(req, "_num_decode_tokens", getattr(req, "num_decode_tokens", 0)), 0)
        rem_decode = _decode_remaining(req)
        done_decode = max(0, total_decode - rem_decode)
        age = max(0.0, sim_time - _arrived_at(req))
        late = _request_total_lateness(state, req, sim_time)
        decode_deadline = _decode_deadline(req, state)
        decode_slack = 0.0 if decode_deadline is None else float(decode_deadline) - sim_time
        decode_slo = _safe_float(getattr(req, "_decode_slo_time", getattr(req, "decode_slo_time", 0.0)), 0.0)
        violated_bit = 1.0 if _is_violated(state, req) else 0.0
        processed_frac = 0.0 if total_decode <= 0 else _clip(float(done_decode) / float(max(1, total_decode)), 0.0, 1.0)

        values = [
            _norm01(rem_decode, F.decode_remaining_den),
            _norm01(total_decode, F.decode_total_den),
            _norm01(done_decode, F.decode_processed_den),
            processed_frac,
            _norm01(age, F.age_den_sec),
            _norm01(late, F.lateness_den_sec),
            _centered01(decode_slack, F.slack_den_sec),
            _norm01(decode_slo, F.decode_slo_den_sec),
            violated_bit,
            1.0 if late > float(F.near_drop_lateness_low_sec) else 0.0,
            1.0 if late >= float(F.near_drop_lateness_high_sec) else 0.0,
            1.0 if done_decode > 216 else 0.0,
            1.0 if done_decode > 512 else 0.0,
        ]
        decode_feat[0, i, :] = torch.tensor(values, dtype=torch.float32)
        decode_mask[0, i] = True
        decode_debug.append(
            {
                "slot": int(i),
                "request_id": int(rid),
                "remaining_decode": int(rem_decode),
                "total_decode": int(total_decode),
                "done_decode": int(done_decode),
                "age": float(age),
                "deadline": "" if decode_deadline is None else float(decode_deadline),
                "slack": float(decode_slack),
                "lateness": float(late),
            }
        )

    violated_ids = set(getattr(getattr(state, "stats", None), "violated_request_ids", set()) or set())
    active_ids = {_safe_int(getattr(r, "id", -1), -1) for r in reqs}
    num_violated_active = len([rid for rid in active_ids if rid in violated_ids])

    p_late_05_15 = 0
    p_late_15 = 0
    for req in prefill_reqs:
        rid = _safe_int(getattr(req, "id", -1), -1)
        late = _safe_float(getattr(stats, "per_request_prefill_lateness", {}).get(rid, 0.0), 0.0)
        if late > float(F.near_drop_lateness_low_sec) and late < float(F.near_drop_lateness_high_sec):
            p_late_05_15 += 1
        elif late >= float(F.near_drop_lateness_high_sec):
            p_late_15 += 1

    d_late_05_15 = 0
    d_late_15 = 0
    for req in decode_reqs:
        late = _request_total_lateness(state, req, sim_time)
        if late > float(F.near_drop_lateness_low_sec) and late < float(F.near_drop_lateness_high_sec):
            d_late_05_15 += 1
        elif late >= float(F.near_drop_lateness_high_sec):
            d_late_15 += 1

    num_prefill = len(prefill_reqs)
    num_decode = len(decode_reqs)
    num_active = len(reqs)
    total_remaining_prefill = sum(_prefill_remaining(r) for r in prefill_reqs)
    total_remaining_decode = sum(_decode_remaining(r) for r in decode_reqs)
    total_decode_generated_active = sum(_safe_int(getattr(r, "num_processed_decode_tokens", 0), 0) for r in decode_reqs)
    slo_violations = _safe_int(getattr(stats, "slo_violations", 0), 0)
    slo_lateness_sum = _safe_float(getattr(stats, "slo_lateness_sum", 0.0), 0.0)
    objective_cost = float(slo_violations) + float(slo_lateness_sum)
    max_requests_window = float(max(1, int(spec_cfg.timing.max_requests_per_launch_window)))
    max_prefill_window = float(
        max(
            1,
            int(spec_cfg.request.target_prefill_tokens_per_request_avg_window)
            * int(spec_cfg.timing.max_requests_per_launch_window),
        )
    )
    launch_count, launch_prefill, ewma = _recent_launch_summary(
        state,
        sim_time,
        window_sec=float(F.launch_ewma_window_sec),
        alpha=float(F.launch_ewma_alpha),
    )
    remaining_launch_request_headroom = max(0.0, max_requests_window - float(launch_count))
    remaining_launch_prefill_headroom = max(0.0, max_prefill_window - float(launch_prefill))
    decode_credit = _decode_credit_available(state)

    global_values = [
        1.0 if player == "controller" else 0.0,
        1.0 if player == "adversary" else 0.0,
        _norm01(objective_cost, F.objective_cost_den),
        _norm01(slo_violations, F.violated_count_den),
        _norm01(slo_lateness_sum, F.total_lateness_den),
        _norm01(num_prefill, F.active_prefill_count_den),
        _norm01(num_decode, F.active_decode_count_den),
        _norm01(num_active, F.active_total_count_den),
        _norm01(total_remaining_prefill, F.total_remaining_prefill_den),
        _norm01(total_remaining_decode, F.total_remaining_decode_den),
        _norm01(total_decode_generated_active, F.total_decode_generated_active_den),
        _norm01(num_violated_active, F.violated_count_den),
        _norm01(p_late_05_15, F.prefill_near_drop_den),
        _norm01(p_late_15, F.prefill_near_drop_den),
        _norm01(d_late_05_15, F.decode_near_drop_den),
        _norm01(d_late_15, F.decode_near_drop_den),
        _norm01(launch_count, F.recent_launch_count_den),
        _norm01(launch_prefill, F.recent_launch_prefill_den),
        _norm01(remaining_launch_request_headroom, max_requests_window),
        _norm01(remaining_launch_prefill_headroom, max_prefill_window),
        _norm01(ewma, F.recent_launch_count_den),
        _norm01(decode_credit, F.decode_credit_den),
        1.0 if num_prefill > 0 else 0.0,
        1.0 if num_decode > 0 else 0.0,
    ]
    global_feat = torch.tensor([global_values], dtype=torch.float32)

    return {
        "prefill_req_features": prefill_feat,
        "decode_req_features": decode_feat,
        "global_features": global_feat,
        "prefill_req_mask": prefill_mask,
        "decode_req_mask": decode_mask,
        "prefill_debug": prefill_debug,
        "decode_debug": decode_debug,
        "counts": {
            "num_prefill": int(num_prefill),
            "num_decode": int(num_decode),
            "num_active": int(num_active),
            "slo_violations": int(slo_violations),
            "objective_cost": float(objective_cost),
        },
    }


def _make_action_mask_fn(env: Any) -> Callable[[Any, str, int, torch.device], torch.Tensor]:
    def action_mask_fn(state: Any, player: str, n: int, device: torch.device) -> torch.Tensor:
        probe = state.fork(flag=False)
        if player == "controller":
            actions_by_index, mask = env.sample_controller_actions(probe)
        else:
            actions_by_index, mask = env.sample_adversary_actions(probe)
        mask_list = [
            bool(ok) and idx < len(actions_by_index) and actions_by_index[idx] is not None
            for idx, ok in enumerate(mask)
        ]
        if len(mask_list) != int(n):
            raise RuntimeError(f"mask length {len(mask_list)} != expected {int(n)} for player={player}")
        return torch.tensor([mask_list], dtype=torch.bool, device=device)

    return action_mask_fn


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return math.inf
    return float(torch.max(torch.abs(a.detach().cpu().to(torch.float32) - b.detach().cpu().to(torch.float32))).item())


def _json(values: Any) -> str:
    return json.dumps(values, sort_keys=True)


def _tensor_row(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit GV3 model feature conversion on history-generated roots.")
    parser.add_argument("--num-roots", type=int, default=32)
    parser.add_argument("--history-hops-min", type=int, default=0)
    parser.add_argument("--history-hops-max", type=int, default=32)
    parser.add_argument("--history-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    summary_csv = output_dir / "feature_conversion_summary.csv"
    prefill_csv = output_dir / "feature_conversion_prefill.csv"
    decode_csv = output_dir / "feature_conversion_decode.csv"
    global_csv = output_dir / "feature_conversion_global.csv"

    cfg = replace(
        DEFAULT_MULTIPROCESS_TRAINING_CONFIG,
        environment_lang="python",
        use_virtual_env=True,
        model=replace(DEFAULT_MULTIPROCESS_TRAINING_CONFIG.model, device="cpu"),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
    )
    cfg.validate()
    _set_global_seeds(
        int(cfg.game_v2.reproducibility.global_seed),
        torch_deterministic=bool(cfg.game_v2.reproducibility.torch_deterministic),
    )

    _simulator, env, _constraints, explore_cfg = _build_env_and_simulator(cfg, use_virtual_env=True)
    mcts = VidurMCTS(env=env, explore_cfg=explore_cfg, rng=random.Random(int(args.history_seed)), verbose=False)
    runner = SelfPlayRunner(
        env=env,
        mcts=mcts,
        model=None,
        writer=_NoopWriter(),
        eval_writer=None,
        device_for_features=torch.device("cpu"),
        game_v2_cfg=cfg.game_v2,
    )

    summary_rows: list[dict[str, Any]] = []
    prefill_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    root_count = 0
    action_mask_fn = _make_action_mask_fn(env)
    tol = float(args.tolerance)

    for prepared_batch in runner._iter_prepared_history_root_batches(
        game_id=0,
        num_roots=int(args.num_roots),
        start_root_id=0,
        start_root_depth=0,
        start_player="adversary",
        initial_state=None,
        history_nontrivial_hops=int(args.history_hops_min),
        history_hops_min=int(args.history_hops_min),
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.history_seed),
        history_max_total_steps=int(cfg.history_max_total_steps),
        log_history_rows=False,
        root_batch_size=int(args.batch_size),
        allow_duplicate_history_fallback=True,
    ):
        for pr in prepared_batch:
            root_state = pr.root_state
            root_player = str(pr.root_player)
            root_depth = int(pr.root_depth)
            root_state, root_player, root_depth = runner._advance_to_branching_root(
                root_state,
                root_player,
                root_depth,
                max_hops=int(cfg.max_forced_hops_per_root),
            )
            if root_player == "adversary":
                root_state, _ = runner._build_root_decision_state_for_adversary(
                    current_state=root_state,
                    pre_controller_snapshot=pr.pre_controller_snapshot,
                    pre_controller_stats=pr.pre_controller_stats,
                )

            inputs = build_model_inputs(
                root_state,
                root_player,
                torch.device("cpu"),
                build_action_mask_flag=True,
                action_mask_fn=action_mask_fn,
            )
            expected = _expected_features(env, root_state, root_player, cfg.game_v2)

            p_diff = _max_abs_diff(inputs.prefill_req_features, expected["prefill_req_features"])
            d_diff = _max_abs_diff(inputs.decode_req_features, expected["decode_req_features"])
            g_diff = _max_abs_diff(inputs.global_features, expected["global_features"])
            p_mask_match = bool(torch.equal(inputs.prefill_req_mask.cpu(), expected["prefill_req_mask"].cpu()))
            d_mask_match = bool(torch.equal(inputs.decode_req_mask.cpu(), expected["decode_req_mask"].cpu()))
            action_mask_expected = action_mask_fn(
                root_state,
                root_player,
                int(inputs.action_mask.shape[-1]),
                torch.device("cpu"),
            )
            action_mask_match = bool(torch.equal(inputs.action_mask.cpu(), action_mask_expected.cpu()))
            legacy_feat_expected = torch.cat(
                [
                    torch.nn.functional.pad(
                        expected["prefill_req_features"],
                        (0, int(inputs.req_features.shape[-1]) - int(expected["prefill_req_features"].shape[-1])),
                    ),
                    torch.nn.functional.pad(
                        expected["decode_req_features"],
                        (0, int(inputs.req_features.shape[-1]) - int(expected["decode_req_features"].shape[-1])),
                    ),
                ],
                dim=1,
            )
            legacy_mask_expected = torch.cat([expected["prefill_req_mask"], expected["decode_req_mask"]], dim=1)
            legacy_feat_diff = _max_abs_diff(inputs.req_features, legacy_feat_expected)
            legacy_mask_match = bool(torch.equal(inputs.req_mask.cpu(), legacy_mask_expected.cpu()))

            passed = (
                p_diff <= tol
                and d_diff <= tol
                and g_diff <= tol
                and legacy_feat_diff <= tol
                and p_mask_match
                and d_mask_match
                and action_mask_match
                and legacy_mask_match
            )
            if not passed:
                failures.append(f"root_id={int(pr.root_id)} player={root_player}")

            counts = expected["counts"]
            summary_rows.append(
                {
                    "root_id": int(pr.root_id),
                    "root_player": root_player,
                    "root_depth": int(root_depth),
                    "history_hops": int(pr.history_hops),
                    "sim_time": float(getattr(root_state.simulator, "_time", 0.0)),
                    "num_prefill": int(counts["num_prefill"]),
                    "num_decode": int(counts["num_decode"]),
                    "num_active": int(counts["num_active"]),
                    "slo_violations": int(counts["slo_violations"]),
                    "objective_cost": float(counts["objective_cost"]),
                    "prefill_max_abs_diff": p_diff,
                    "decode_max_abs_diff": d_diff,
                    "global_max_abs_diff": g_diff,
                    "legacy_max_abs_diff": legacy_feat_diff,
                    "prefill_mask_match": str(p_mask_match).lower(),
                    "decode_mask_match": str(d_mask_match).lower(),
                    "action_mask_match": str(action_mask_match).lower(),
                    "legacy_mask_match": str(legacy_mask_match).lower(),
                    "passed": str(passed).lower(),
                }
            )

            for item in expected["prefill_debug"]:
                slot = int(item["slot"])
                actual_vec = _tensor_row(inputs.prefill_req_features[0, slot])
                expected_vec = _tensor_row(expected["prefill_req_features"][0, slot])
                prefill_rows.append(
                    {
                        "root_id": int(pr.root_id),
                        "slot": slot,
                        "request_id": int(item["request_id"]),
                        "root_player": root_player,
                        "sim_time": float(getattr(root_state.simulator, "_time", 0.0)),
                        "remaining_prefill": int(item["remaining_prefill"]),
                        "total_prefill": int(item["total_prefill"]),
                        "done_prefill": int(item["done_prefill"]),
                        "age": float(item["age"]),
                        "deadline": item["deadline"],
                        "slack": float(item["slack"]),
                        "prefill_late": float(item["prefill_late"]),
                        "labels_json": _json(PREFILL_FEATURE_LABELS),
                        "actual_features_json": _json(actual_vec),
                        "expected_features_json": _json(expected_vec),
                        "max_abs_diff": max(abs(a - b) for a, b in zip(actual_vec, expected_vec)),
                    }
                )

            for item in expected["decode_debug"]:
                slot = int(item["slot"])
                actual_vec = _tensor_row(inputs.decode_req_features[0, slot])
                expected_vec = _tensor_row(expected["decode_req_features"][0, slot])
                decode_rows.append(
                    {
                        "root_id": int(pr.root_id),
                        "slot": slot,
                        "request_id": int(item["request_id"]),
                        "root_player": root_player,
                        "sim_time": float(getattr(root_state.simulator, "_time", 0.0)),
                        "remaining_decode": int(item["remaining_decode"]),
                        "total_decode": int(item["total_decode"]),
                        "done_decode": int(item["done_decode"]),
                        "age": float(item["age"]),
                        "deadline": item["deadline"],
                        "slack": float(item["slack"]),
                        "lateness": float(item["lateness"]),
                        "labels_json": _json(DECODE_FEATURE_LABELS),
                        "actual_features_json": _json(actual_vec),
                        "expected_features_json": _json(expected_vec),
                        "max_abs_diff": max(abs(a - b) for a, b in zip(actual_vec, expected_vec)),
                    }
                )

            actual_global = _tensor_row(inputs.global_features[0])
            expected_global = _tensor_row(expected["global_features"][0])
            for idx, (actual_value, expected_value) in enumerate(zip(actual_global, expected_global)):
                global_rows.append(
                    {
                        "root_id": int(pr.root_id),
                        "root_player": root_player,
                        "sim_time": float(getattr(root_state.simulator, "_time", 0.0)),
                        "feature_index": int(idx),
                        "feature_label": GLOBAL_FEATURE_LABELS[int(idx)],
                        "actual_value": float(actual_value),
                        "expected_value": float(expected_value),
                        "abs_diff": abs(float(actual_value) - float(expected_value)),
                    }
                )

            root_count += 1

    _write_csv(
        summary_csv,
        [
            "root_id",
            "root_player",
            "root_depth",
            "history_hops",
            "sim_time",
            "num_prefill",
            "num_decode",
            "num_active",
            "slo_violations",
            "objective_cost",
            "prefill_max_abs_diff",
            "decode_max_abs_diff",
            "global_max_abs_diff",
            "legacy_max_abs_diff",
            "prefill_mask_match",
            "decode_mask_match",
            "action_mask_match",
            "legacy_mask_match",
            "passed",
        ],
        summary_rows,
    )
    _write_csv(
        prefill_csv,
        [
            "root_id",
            "slot",
            "request_id",
            "root_player",
            "sim_time",
            "remaining_prefill",
            "total_prefill",
            "done_prefill",
            "age",
            "deadline",
            "slack",
            "prefill_late",
            "labels_json",
            "actual_features_json",
            "expected_features_json",
            "max_abs_diff",
        ],
        prefill_rows,
    )
    _write_csv(
        decode_csv,
        [
            "root_id",
            "slot",
            "request_id",
            "root_player",
            "sim_time",
            "remaining_decode",
            "total_decode",
            "done_decode",
            "age",
            "deadline",
            "slack",
            "lateness",
            "labels_json",
            "actual_features_json",
            "expected_features_json",
            "max_abs_diff",
        ],
        decode_rows,
    )
    _write_csv(
        global_csv,
        [
            "root_id",
            "root_player",
            "sim_time",
            "feature_index",
            "feature_label",
            "actual_value",
            "expected_value",
            "abs_diff",
        ],
        global_rows,
    )

    if root_count != int(args.num_roots):
        raise RuntimeError(f"expected {int(args.num_roots)} roots, checked {root_count}")
    if failures:
        raise RuntimeError(
            f"feature conversion failed for {len(failures)} roots: {failures[:20]} "
            f"(wrote {summary_csv})"
        )

    print(
        "feature conversion tests passed: "
        f"roots={root_count}, summary_csv={summary_csv}, "
        f"prefill_csv={prefill_csv}, decode_csv={decode_csv}, global_csv={global_csv}"
    )


if __name__ == "__main__":
    main()
