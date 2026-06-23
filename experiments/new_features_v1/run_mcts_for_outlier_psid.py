"""Run vanilla GV3 MCTS for one stored root-state psid.

This diagnostic loads one root record from the fixed ModelSearchBed root dataset,
rebuilds the simulator state, runs ``Game_Version3/mcts.py`` for the requested
number of iterations, and writes MCTS root/child CSV logs into the model output
directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import types
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vidur.mcts.Game_Versions.Game_Version3.mcts import MCTSConfig, VidurMCTS  # noqa: E402
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
DEFAULT_D2_DIR = Path(
    "/home/shazer/Desktop/Research/Vidur/vidur/"
    "simulator_output/GV3_Agent/model_search_roots_controller_extra_400k_abs1_ratio40_unique_from_292k_hops0_320"
)


def _clean_row(row: dict[str, Any]) -> dict[str, str]:
    return {str(k).strip(): (str(v).strip() if v is not None else "") for k, v in row.items()}


def _read_outlier_row(model_dir: Path, psid: int) -> dict[str, str] | None:
    path = model_dir / "bellman_outliers_inspection.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f, skipinitialspace=True):
            row = _clean_row(raw)
            if int(row.get("psid", -1)) == int(psid):
                return row
    return None


def _manifest_offsets(dataset_dir: Path, *, root_player_filter: str) -> list[tuple[int, int, str]]:
    manifest = dataset_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    out: list[tuple[int, int, str]] = []
    cursor = 0
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            n_kept = int((entry.get("player_counts") or {}).get(root_player_filter, entry.get("num_records", 0)))
            out.append((cursor, cursor + n_kept, str(entry["shard_path"])))
            cursor += n_kept
    return out


def _locate_psid(psid: int, dataset_dir: Path, *, root_player_filter: str) -> tuple[str, int]:
    for start, end, shard_path in _manifest_offsets(dataset_dir, root_player_filter=root_player_filter):
        if start <= int(psid) < end:
            return shard_path, int(psid) - start
    raise IndexError(f"psid={psid} not found in {dataset_dir}")


def _load_record(dataset_dir: Path, psid: int, *, root_player_filter: str) -> tuple[dict[str, Any], str, int]:
    shard_rel, in_shard_idx = _locate_psid(psid, dataset_dir, root_player_filter=root_player_filter)
    shard_path = dataset_dir / shard_rel
    records = torch.load(shard_path, map_location="cpu", weights_only=False)
    record = records[int(in_shard_idx)]
    return record, shard_rel, int(in_shard_idx)


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


def _patch_runtime_mcts_issues(mcts: VidurMCTS) -> None:
    """Keep this diagnostic isolated from the source tree.

    Current local ``mcts.py`` has two mechanical issues while it is under
    development: search_dnn expects ``_node_counter`` and transition reward is
    called with numeric costs. Patch those on this instance only.
    """

    if not hasattr(mcts, "_node_counter"):
        setattr(mcts, "_node_counter", getattr(mcts, "_node_id_counter", 0))

    def get_transition_reward(self: VidurMCTS, parent_cost: float, child_cost: float) -> float:
        return float(parent_cost) - float(child_cost)

    mcts.get_transition_reward = types.MethodType(get_transition_reward, mcts)


def run(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir).expanduser()
    dataset_dir = Path(args.dataset_dir).expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    record, shard_rel, in_shard_idx = _load_record(
        dataset_dir,
        int(args.psid),
        root_player_filter=str(args.root_player_filter),
    )
    snapshot = record.get("simulator_snapshot") or {}
    loaded_sim_time = float(snapshot.get("time", 0.0))

    outlier_row = _read_outlier_row(model_dir, int(args.psid))
    if outlier_row is not None:
        csv_sim_time = float(outlier_row["sim_time"])
        sim_time_abs_diff = abs(loaded_sim_time - csv_sim_time)
    else:
        csv_sim_time = None
        sim_time_abs_diff = None

    loader = _build_loader(dataset_dir, model_dir)
    try:
        state = loader(record)
        state_sim_time = float(state.simulator._time)

        root_log_path = model_dir / f"mcts_psid_{int(args.psid)}_root_summary.csv"
        child_log_path = model_dir / f"mcts_psid_{int(args.psid)}_children.csv"
        result_path = model_dir / f"mcts_psid_{int(args.psid)}_result.json"

        mcts_cfg = MCTSConfig()
        mcts_cfg.log_flag = True
        mcts_cfg.log_path = root_log_path
        mcts_cfg.tree_log_path = child_log_path
        mcts_cfg.rng = random.Random(int(args.seed))
        mcts_cfg.num_simulations = int(args.iterations)
        mcts_cfg.uct_c = float(args.uct_c)
        mcts_cfg.discount_factor = float(args.discount_factor)
        mcts_cfg._discount_time_denom = float(args.discount_time_denom)

        mcts = VidurMCTS(env=loader.env, mctsConfig=mcts_cfg)
        _patch_runtime_mcts_issues(mcts)
        try:
            result = mcts.search_dnn(
                dnn_model=None,
                rootState=state,
                root_player=str(record.get("root_player", args.root_player_filter)),
                game_id=int(args.game_id),
                root_id=int(record.get("root_id", args.psid)),
                root_node_id_override=None,
                root_depth=int(record.get("root_depth", 0)),
                mcts_iter=int(args.iterations),
                model_version=0,
                use_model_bootstrap=False,
                root_phase="outlier_psid_diagnostic",
                cycle_label=f"psid_{int(args.psid)}",
            )
        finally:
            mcts.close()

        summary = {
            "psid": int(args.psid),
            "dataset_dir": str(dataset_dir),
            "shard_path": shard_rel,
            "in_shard_idx": int(in_shard_idx),
            "root_id": int(record.get("root_id", args.psid)),
            "root_player": str(record.get("root_player", "")),
            "root_depth": int(record.get("root_depth", 0)),
            "history_hops": int(record.get("history_hops", 0)),
            "record_sim_time": loaded_sim_time,
            "state_sim_time": state_sim_time,
            "outlier_csv_sim_time": csv_sim_time,
            "outlier_csv_sim_time_abs_diff": sim_time_abs_diff,
            "iterations": int(args.iterations),
            "best_action_index": result.best_action_index,
            "best_action_repr": repr(result.best_action) if result.best_action is not None else "",
            "best_action_value": float(result.best_action_value),
            "root_log_path": str(root_log_path),
            "child_log_path": str(child_log_path),
        }
        result_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        loader.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psid", type=int, default=117694)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--root-player-filter", default="controller")
    parser.add_argument("--game-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--uct-c", type=float, default=1.4)
    parser.add_argument("--discount-factor", type=float, default=0.98)
    parser.add_argument("--discount-time-denom", type=float, default=0.015725797204323228)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
