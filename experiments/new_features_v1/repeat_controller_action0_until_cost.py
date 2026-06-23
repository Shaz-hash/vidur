"""Repeat controller action index 0 from a stored GV3 root until cost changes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vidur.mcts.Game_Versions.Game_Version3.ModelSearchBed.self_model_test import (  # noqa: E402
    RootStateLoader,
    build_self_model_test_config,
)

DEFAULT_MODEL_DIR = Path(
    "/home/shazer/Desktop/Research/Vidur/vidur-classical-search/"
    "simulator_output/GV3_Agent/BellmanConvergence/cached_state_local_v1/"
    "V1_models/hgb_sq_31leaf_700iter"
)
DEFAULT_D1_DIR = Path(
    "/home/shazer/Desktop/Research/Vidur/vidur/"
    "simulator_output/GV3_Agent/model_search_roots_controller_350k_abs1_ratio40"
)


def _manifest_offsets(dataset_dir: Path, *, root_player_filter: str) -> list[tuple[int, int, str]]:
    manifest = dataset_dir / "manifest.jsonl"
    out: list[tuple[int, int, str]] = []
    cursor = 0
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            n = int((entry.get("player_counts") or {}).get(root_player_filter, entry.get("num_records", 0)))
            out.append((cursor, cursor + n, str(entry["shard_path"])))
            cursor += n
    return out


def _locate_psid(dataset_dir: Path, psid: int, *, root_player_filter: str) -> tuple[str, int]:
    for start, end, shard in _manifest_offsets(dataset_dir, root_player_filter=root_player_filter):
        if start <= int(psid) < end:
            return shard, int(psid) - start
    raise IndexError(f"psid={psid} not found in {dataset_dir}")


def _load_record(dataset_dir: Path, psid: int, *, root_player_filter: str) -> tuple[dict[str, Any], str, int]:
    shard_rel, idx = _locate_psid(dataset_dir, psid, root_player_filter=root_player_filter)
    records = torch.load(dataset_dir / shard_rel, map_location="cpu", weights_only=False)
    return records[idx], shard_rel, idx


def _build_loader(dataset_dir: Path, model_dir: Path) -> RootStateLoader:
    cfg = build_self_model_test_config(
        dataset_dir=dataset_dir,
        output_dir=model_dir,
        num_roots=1,
        max_candidate_roots=1,
        candidate_batch_size=1,
        root_player_filter="controller",
        allow_dataset_generation=False,
        eval_ratio=0.5,
        split_seed=12345,
        seed=2027,
    )
    return RootStateLoader(cfg)


def _objective_cost(env: Any, state: Any) -> tuple[float, int, float]:
    violations, lateness = env.evaluate_objective(state)
    return float(violations) + float(lateness), int(violations), float(lateness)


def _compact(text: Any, max_len: int = 500) -> str:
    s = " ".join(repr(text).split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def run(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir).expanduser()
    dataset_dir = Path(args.dataset_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    record, shard_rel, in_shard_idx = _load_record(
        dataset_dir,
        int(args.psid),
        root_player_filter=str(args.root_player_filter),
    )
    loader = _build_loader(dataset_dir, model_dir)
    rows: list[dict[str, Any]] = []
    try:
        env = loader.env
        state = loader(record)
        initial_time = float(state.simulator._time)
        initial_cost, initial_violations, initial_lateness = _objective_cost(env, state)

        first_nonzero: dict[str, Any] | None = None
        for depth in range(1, int(args.max_depth) + 1):
            parent_time = float(state.simulator._time)
            parent_cost, parent_violations, parent_lateness = _objective_cost(env, state)
            actions, mask = env.sample_controller_actions(state)
            mask_list = mask.tolist() if hasattr(mask, "tolist") else list(mask)
            if len(actions) <= int(args.action_index) or not bool(mask_list[int(args.action_index)]) or actions[int(args.action_index)] is None:
                rows.append(
                    {
                        "depth": depth,
                        "status": "action_invalid",
                        "action_index": int(args.action_index),
                        "parent_time": parent_time,
                        "parent_cost": parent_cost,
                        "parent_violations": parent_violations,
                        "parent_lateness": parent_lateness,
                    }
                )
                break

            action = actions[int(args.action_index)]
            state = env.apply_controller_action_only(state, action, inplace=True, fast_forward=True)
            child_time = float(state.simulator._time)
            child_cost, child_violations, child_lateness = _objective_cost(env, state)
            transition_cost = child_cost - parent_cost
            transition_reward = parent_cost - child_cost
            desc = env.describe_state(state)
            row = {
                "depth": depth,
                "status": "ok",
                "action_index": int(args.action_index),
                "action_repr": _compact(action),
                "parent_time": parent_time,
                "child_time": child_time,
                "time_delta": child_time - parent_time,
                "parent_cost": parent_cost,
                "child_cost": child_cost,
                "transition_cost": transition_cost,
                "transition_reward": transition_reward,
                "parent_violations": parent_violations,
                "child_violations": child_violations,
                "parent_lateness": parent_lateness,
                "child_lateness": child_lateness,
                "active_ids": json.dumps(desc.get("active_request_ids", [])),
                "completed_ids": json.dumps(desc.get("completed_request_ids", [])),
                "dropped_ids": json.dumps(desc.get("dropped_request_ids", [])),
                "violated_ids": json.dumps(desc.get("violated_request_ids", [])),
                "decode_credit_balance": desc.get("decode_credit_balance", ""),
                "decode_tokens_counted_by_id": json.dumps(desc.get("decode_tokens_counted_by_id", {}), sort_keys=True),
                "per_request_decode_lateness_by_id": json.dumps(desc.get("per_request_decode_lateness_by_id", {}), sort_keys=True),
                "per_request_prefill_lateness_by_id": json.dumps(desc.get("per_request_prefill_lateness_by_id", {}), sort_keys=True),
            }
            rows.append(row)
            if abs(float(transition_cost)) > float(args.eps):
                first_nonzero = row
                break

        out_csv = model_dir / f"repeat_action{int(args.action_index)}_psid_{int(args.psid)}_until_cost.csv"
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "psid": int(args.psid),
            "dataset_dir": str(dataset_dir),
            "shard_path": shard_rel,
            "in_shard_idx": int(in_shard_idx),
            "root_id": int(record.get("root_id", args.psid)),
            "root_player": str(record.get("root_player", "")),
            "root_depth": int(record.get("root_depth", 0)),
            "history_hops": int(record.get("history_hops", 0)),
            "initial_time": initial_time,
            "initial_cost": initial_cost,
            "initial_violations": initial_violations,
            "initial_lateness": initial_lateness,
            "action_index_repeated": int(args.action_index),
            "eps": float(args.eps),
            "max_depth": int(args.max_depth),
            "steps_executed": len([r for r in rows if r.get("status") == "ok"]),
            "first_nonzero_transition_cost_depth": None if first_nonzero is None else int(first_nonzero["depth"]),
            "first_nonzero_transition_cost": None if first_nonzero is None else float(first_nonzero["transition_cost"]),
            "first_nonzero_child_cost": None if first_nonzero is None else float(first_nonzero["child_cost"]),
            "first_nonzero_child_time": None if first_nonzero is None else float(first_nonzero["child_time"]),
            "first_nonzero_action_repr": None if first_nonzero is None else str(first_nonzero["action_repr"]),
            "log_csv": str(out_csv),
        }
        out_json = model_dir / f"repeat_action{int(args.action_index)}_psid_{int(args.psid)}_summary.json"
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        loader.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psid", type=int, default=117694)
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=2000)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--root-player-filter", default="controller")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
