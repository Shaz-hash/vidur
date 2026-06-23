from __future__ import annotations

import argparse
import csv
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ..DNN.history_root import HistoryRootGenerator
from ..DNN.selfPlay import SelfPlayRunner
from ..config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG
from ..logger.mctsDNN_logger import DNNMCTSIterationLogger
from ..mctsDNN import VidurMCTS
from ..multiProcessUtils import _build_env_and_simulator, _set_global_seeds
from .feature_conversion_tests import _make_action_mask_fn
from .frontier_feature_trace_logger import (
    build_frontier_state_row,
    build_production_feature_row,
    compare_feature_rows,
    read_csv,
    write_csv,
)
from .history_node_tests import _validate_trace_csv


FRONTIER_FIELDS = [
    "root_id",
    "root_player",
    "root_depth",
    "history_hops",
    "history_log_node_id",
    "sim_time",
    "objective_cost",
    "slo_violations",
    "slo_lateness_sum",
    "active_request_ids_json",
    "completed_request_ids_json",
    "dropped_request_ids_json",
    "stopped_decode_request_ids_json",
    "violated_request_ids_json",
    "per_request_prefill_lateness_json",
    "per_request_decode_lateness_json",
    "decode_next_deadline_by_id_json",
    "decode_tokens_counted_by_id_json",
    "recent_arrivals_json",
    "pending_adv_tick",
    "last_adv_tick",
    "requests_json",
]


FEATURE_FIELDS = [
    "root_id",
    "root_player",
    "root_depth",
    "history_hops",
    "prefill_req_features_json",
    "decode_req_features_json",
    "global_features_json",
    "prefill_req_mask_json",
    "decode_req_mask_json",
    "req_features_json",
    "req_mask_json",
    "action_mask_json",
]


COMPARE_FIELDS = [
    "root_id",
    "root_player",
    "root_depth",
    "history_hops",
    "passed",
    "max_feature_diff",
    "prefill_diff",
    "decode_diff",
    "global_diff",
    "req_diff",
    "prefill_mask_match",
    "decode_mask_match",
    "req_mask_match",
    "feature_ids_subset_active",
    "feature_ids_disjoint_terminal",
    "active_request_ids_json",
    "terminal_request_ids_json",
    "prefill_feature_request_ids_json",
    "decode_feature_request_ids_json",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _default_output_dir() -> Path:
    return _repo_root() / "simulator_output" / "Game_Version3" / "tests" / "frontier_feature_trace_tests"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GV3 Python frontier traces, log production features, "
            "and independently recompute expected features from serialized frontier rows."
        )
    )
    parser.add_argument("--num-roots", type=int, default=32)
    parser.add_argument("--history-hops-min", type=int, default=0)
    parser.add_argument("--history-hops-max", type=int, default=8)
    parser.add_argument("--history-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    return parser.parse_args()


def _make_runner(*, cfg: Any, env: Any, explore_cfg: Any, history_seed: int, out_dir: Path) -> SelfPlayRunner:
    mcts = VidurMCTS(
        env=env,
        explore_cfg=explore_cfg,
        rng=random.Random(int(history_seed)),
        log_path=str(out_dir / "python_mcts_iter.csv"),
        tree_log_path=str(out_dir / "python_mcts_root.csv"),
        logger_flush_every=1,
        verbose=False,
        complete_log=False,
    )
    return SelfPlayRunner(
        env=env,
        mcts=mcts,
        model=None,
        writer=None,
        eval_writer=None,
        device_for_features=torch.device(str(cfg.model.device)),
        game_v2_cfg=cfg.game_v2,
    )


def _feature_state_for_record(
    *,
    runner: SelfPlayRunner,
    record: dict[str, Any],
    max_forced_hops: int,
) -> tuple[Any, str, int]:
    state = record["root_state"]
    player = str(record["root_player"])
    depth = int(record["root_depth"])
    if player == "adversary":
        state, _forbidden = runner._build_root_decision_state_for_adversary(
            current_state=state,
            pre_controller_snapshot=record.get("pre_controller_snapshot"),
            pre_controller_stats=record.get("pre_controller_stats"),
        )
    state, player, depth = runner._advance_to_branching_root(
        state,
        player,
        depth,
        max_hops=int(max_forced_hops),
    )
    return state, player, int(depth)


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _blank_row(fieldnames: list[str]) -> dict[str, Any]:
    return {field: "" for field in fieldnames}


def _ancestor_chain(shared_rows_by_node_id: dict[int, dict[str, str]], node_id: Any) -> list[dict[str, str]]:
    if node_id in (None, ""):
        return []

    chain: list[dict[str, str]] = []
    seen: set[int] = set()
    cur = int(node_id)
    while cur in shared_rows_by_node_id and cur not in seen:
        seen.add(cur)
        row = shared_rows_by_node_id[cur]
        chain.append(row)
        parent = str(row.get("parent_node_id", "") or "").strip()
        if parent == "":
            break
        cur = int(parent)
    chain.reverse()
    return chain


def _expand_final_root_iter_csv(
    *,
    shared_trace_csv: Path,
    final_trace_csv: Path,
    frontier_rows: list[dict[str, Any]],
) -> None:
    shared_rows, fieldnames = _read_rows(shared_trace_csv)
    shared_by_node_id = {
        int(row["node_id"]): row
        for row in shared_rows
        if str(row.get("node_id", "")).strip() != ""
    }

    expanded: list[dict[str, Any]] = []
    for frontier in frontier_rows:
        root_id = int(frontier["root_id"])
        chain = _ancestor_chain(shared_by_node_id, frontier.get("history_log_node_id", ""))
        node_remap = {
            int(row["node_id"]): int(idx)
            for idx, row in enumerate(chain)
        }

        for idx, row in enumerate(chain):
            out = dict(row)
            out["root_id"] = int(root_id)
            out["node_id"] = int(idx)
            parent = str(row.get("parent_node_id", "") or "").strip()
            out["parent_node_id"] = "" if parent == "" else int(node_remap[int(parent)])
            expanded.append(out)

        frontier_row = _blank_row(fieldnames)
        frontier_node_id = len(chain)
        parent_node_id = "" if not chain else int(frontier_node_id - 1)
        frontier_row.update(
            {
                "game_id": 0,
                "root_id": int(root_id),
                "sim_iteration": -1,
                "root_depth": int(frontier.get("root_depth", 0) or 0),
                "root_node_id": 0,
                "root_player": str(frontier.get("root_player", "")),
                "phase": "internal:frontier_state",
                "node_depth": int(frontier.get("root_depth", 0) or 0),
                "parent_node_id": parent_node_id,
                "node_id": int(frontier_node_id),
                "player_acted_to_create_this_node": "frontier_state",
                "player_to_act_in_this_node": str(frontier.get("root_player", "")),
                "action_index": -1,
                "action_repr": "FrontierState()",
                "prior": 1.0,
                "reward": 0.0,
                "nn_called": "false",
                "num_valid_actions": "",
                "unique_actions": "",
                "nn_value_controller": "",
                "objective_cost": frontier.get("objective_cost", ""),
                "sim_time": frontier.get("sim_time", ""),
                "decision_state_time": frontier.get("sim_time", ""),
                "start_time": frontier.get("sim_time", ""),
                "end_time": frontier.get("sim_time", ""),
                "stage_total_time": 0.0,
                "requests_in_system": len(
                    read_csv_list(frontier.get("active_request_ids_json", "[]"))
                ),
                "requests_generated": "",
                "requests_completed": "",
                "slo_violations": frontier.get("slo_violations", ""),
                "total_lateness": frontier.get("slo_lateness_sum", ""),
                "avg_lateness": frontier.get("slo_lateness_sum", ""),
                "state_active_ids": frontier.get("active_request_ids_json", "[]"),
                "state_waiting_ids": frontier.get("active_request_ids_json", "[]"),
                "state_completed_request_ids": frontier.get("completed_request_ids_json", "[]"),
                "state_dropped_request_ids": frontier.get("dropped_request_ids_json", "[]"),
                "state_stopped_decode_request_ids": frontier.get("stopped_decode_request_ids_json", "[]"),
                "state_pending_adv_tick": frontier.get("pending_adv_tick", "false"),
                "state_last_adv_tick": frontier.get("last_adv_tick", ""),
                "state_decode_credit_balance": "",
                "state_decode_tokens_counted_by_id": frontier.get("decode_tokens_counted_by_id_json", "{}"),
                "state_violated_request_ids": frontier.get("violated_request_ids_json", "[]"),
                "state_per_request_prefill_lateness_by_id": frontier.get("per_request_prefill_lateness_json", "{}"),
                "state_per_request_decode_lateness_by_id": frontier.get("per_request_decode_lateness_json", "{}"),
            }
        )
        expanded.append(frontier_row)

    final_trace_csv.parent.mkdir(parents=True, exist_ok=True)
    with final_trace_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in expanded:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv_list(raw: Any) -> list[Any]:
    if raw in (None, ""):
        return []
    if isinstance(raw, list):
        return raw
    try:
        import json

        return list(json.loads(str(raw)))
    except Exception:
        return []


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_trace_csv = out_dir / "shared_history_iter.csv"
    trace_csv = out_dir / "final_root_iter.csv"
    frontier_csv = out_dir / "frontiers.csv"
    feature_csv = out_dir / "frontier_feature.csv"
    compare_csv = out_dir / "frontier_feature_compare.csv"

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
    runner = _make_runner(cfg=cfg, env=env, explore_cfg=explore_cfg, history_seed=int(args.history_seed), out_dir=out_dir)
    history = HistoryRootGenerator(env=env)
    trace_logger = DNNMCTSIterationLogger(shared_trace_csv, flush_every=1)

    records: list[dict[str, Any]] = []
    try:
        for batch in history.generate_roots_batch_iter(
            initial_state=None,
            start_player="adversary",
            start_depth=0,
            num_roots=int(args.num_roots),
            game_id=0,
            start_root_id=0,
            nontrivial_hops=int(args.history_hops_min),
            min_history_hops=int(args.history_hops_min),
            max_history_hops=int(args.history_hops_max),
            seed=int(args.history_seed),
            max_total_steps=int(cfg.history_max_total_steps),
            batch_size=int(args.batch_size),
            allow_duplicate_fallback=True,
            history_trace_logger=trace_logger,
        ):
            records.extend(batch)
    finally:
        trace_logger.close()

    if len(records) != int(args.num_roots):
        raise RuntimeError(f"expected {int(args.num_roots)} roots, generated {len(records)}")

    frontier_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    action_mask_fn = _make_action_mask_fn(env)

    for record in records:
        state, player, depth = _feature_state_for_record(
            runner=runner,
            record=record,
            max_forced_hops=int(getattr(cfg, "max_forced_hops_per_root", 1024)),
        )
        root_id = int(record["root_id"])
        history_hops = int(record.get("history_hops", 0))
        frontier_rows.append(
            build_frontier_state_row(
                env=env,
                state=state,
                root_id=root_id,
                root_player=player,
                root_depth=int(depth),
                history_hops=history_hops,
                history_log_node_id=record.get("history_log_node_id"),
            )
        )
        feature_rows.append(
            build_production_feature_row(
                env=env,
                state=state,
                root_id=root_id,
                root_player=player,
                root_depth=int(depth),
                history_hops=history_hops,
                action_mask_fn=action_mask_fn,
            )
        )

    write_csv(frontier_csv, FRONTIER_FIELDS, frontier_rows)
    write_csv(feature_csv, FEATURE_FIELDS, feature_rows)
    _expand_final_root_iter_csv(
        shared_trace_csv=shared_trace_csv,
        final_trace_csv=trace_csv,
        frontier_rows=frontier_rows,
    )

    compare_rows, failures = compare_feature_rows(
        frontier_rows=read_csv(frontier_csv),
        feature_rows=read_csv(feature_csv),
        cfg=cfg.game_v2,
        tolerance=float(args.tolerance),
    )
    write_csv(compare_csv, COMPARE_FIELDS, compare_rows)

    trace_chains, adv_actions_checked = _validate_trace_csv(trace_csv)

    if failures:
        raise RuntimeError(
            f"frontier trace feature invariant failed for roots {failures[:20]} "
            f"(wrote {compare_csv})"
        )

    max_diff = max((float(row["max_feature_diff"]) for row in compare_rows), default=0.0)
    print(
        "frontier trace feature tests passed: "
        f"roots={len(records)}, trace_chains={trace_chains}, adv_actions_checked={adv_actions_checked}, "
        f"max_feature_diff={max_diff:.3g}, shared_trace_csv={shared_trace_csv}, trace_csv={trace_csv}, "
        f"frontier_csv={frontier_csv}, feature_csv={feature_csv}, compare_csv={compare_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
