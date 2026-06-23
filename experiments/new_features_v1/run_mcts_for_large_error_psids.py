"""Run MCTS for all large-error train/eval psids and plot value violins."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
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

SUMMARY_FIELDS = [
    "psid",
    "split",
    "y_true",
    "y_pred",
    "abs_err",
    "dataset",
    "local_psid",
    "shard_path",
    "in_shard_idx",
    "root_id",
    "root_player",
    "root_depth",
    "history_hops",
    "record_sim_time",
    "state_sim_time",
    "iterations",
    "mcts_best_action_index",
    "mcts_best_action_value",
    "mcts_best_action_repr",
    "root_log_path",
    "child_log_path",
    "status",
    "error",
    "elapsed_sec",
]


def _clean_row(row: dict[str, Any]) -> dict[str, str]:
    return {str(k).strip(): (str(v).strip() if v is not None else "") for k, v in row.items()}


def _read_large_error_rows(model_dir: Path, *, threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split, filename in (("train", "train_results.csv"), ("eval", "eval_results.csv")):
        path = model_dir / filename
        with path.open(newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f, skipinitialspace=True):
                row = _clean_row(raw)
                abs_err = float(row["abs_err"])
                if abs_err < float(threshold):
                    continue
                out.append(
                    {
                        "psid": int(row["psid"]),
                        "split": split,
                        "y_true": float(row["y_true"]),
                        "y_pred": float(row["y_pred"]),
                        "abs_err": abs_err,
                    }
                )
    out.sort(key=lambda r: (str(r["split"]), int(r["psid"])))
    return out


def _manifest_offsets(dataset_dir: Path, *, root_player_filter: str) -> tuple[list[tuple[int, int, str]], int]:
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
            n = int((entry.get("player_counts") or {}).get(root_player_filter, entry.get("num_records", 0)))
            out.append((cursor, cursor + n, str(entry["shard_path"])))
            cursor += n
    return out, cursor


def _locate_local_psid(local_psid: int, offsets: list[tuple[int, int, str]]) -> tuple[str, int]:
    for start, end, shard in offsets:
        if start <= int(local_psid) < end:
            return shard, int(local_psid) - start
    raise IndexError(f"local_psid={local_psid} not found in manifest offsets")


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
    if not hasattr(mcts, "_node_counter"):
        setattr(mcts, "_node_counter", getattr(mcts, "_node_id_counter", 0))

    def get_transition_reward(self: VidurMCTS, parent_cost: float, child_cost: float) -> float:
        return float(parent_cost) - float(child_cost)

    mcts.get_transition_reward = types.MethodType(get_transition_reward, mcts)


def _read_completed(summary_path: Path) -> set[int]:
    if not summary_path.exists():
        return set()
    done: set[int] = set()
    with summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add(int(row["psid"]))
    return done


def _append_summary(summary_path: Path, row: dict[str, Any]) -> None:
    exists = summary_path.exists()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def _load_record(
    *,
    dataset_dir: Path,
    shard_cache: dict[tuple[str, str], list[dict[str, Any]]],
    tag: str,
    shard_rel: str,
    in_shard_idx: int,
) -> dict[str, Any]:
    key = (tag, shard_rel)
    records = shard_cache.get(key)
    if records is None:
        records = torch.load(dataset_dir / shard_rel, map_location="cpu", weights_only=False)
        shard_cache.clear()
        shard_cache[key] = records
    return records[int(in_shard_idx)]


def _run_one(
    *,
    item: dict[str, Any],
    n_d1: int,
    offsets_by_tag: dict[str, list[tuple[int, int, str]]],
    dataset_dirs: dict[str, Path],
    loaders: dict[str, RootStateLoader],
    shard_cache: dict[tuple[str, str], list[dict[str, Any]]],
    logs_dir: Path,
    iterations: int,
    seed: int,
    uct_c: float,
    discount_factor: float,
    discount_time_denom: float,
) -> dict[str, Any]:
    psid = int(item["psid"])
    if psid < int(n_d1):
        tag = "d1"
        local_psid = psid
    else:
        tag = "d2"
        local_psid = psid - int(n_d1)

    shard_rel, in_shard_idx = _locate_local_psid(local_psid, offsets_by_tag[tag])
    record = _load_record(
        dataset_dir=dataset_dirs[tag],
        shard_cache=shard_cache,
        tag=tag,
        shard_rel=shard_rel,
        in_shard_idx=in_shard_idx,
    )
    loader = loaders[tag]
    state = loader(record)

    split_dir = logs_dir / str(item["split"])
    split_dir.mkdir(parents=True, exist_ok=True)
    root_log_path = split_dir / f"psid_{psid}_root_summary.csv"
    child_log_path = split_dir / f"psid_{psid}_children.csv"

    mcts_cfg = MCTSConfig()
    mcts_cfg.log_flag = True
    mcts_cfg.log_path = root_log_path
    mcts_cfg.tree_log_path = child_log_path
    mcts_cfg.rng = random.Random(int(seed) + psid)
    mcts_cfg.num_simulations = int(iterations)
    mcts_cfg.uct_c = float(uct_c)
    mcts_cfg.discount_factor = float(discount_factor)
    mcts_cfg._discount_time_denom = float(discount_time_denom)

    mcts = VidurMCTS(env=loader.env, mctsConfig=mcts_cfg)
    _patch_runtime_mcts_issues(mcts)
    try:
        result = mcts.search_dnn(
            dnn_model=None,
            rootState=state,
            root_player=str(record.get("root_player", "controller")),
            game_id=0,
            root_id=int(record.get("root_id", psid)),
            root_node_id_override=None,
            root_depth=int(record.get("root_depth", 0)),
            mcts_iter=int(iterations),
            model_version=0,
            use_model_bootstrap=False,
            root_phase="large_error_mcts_diagnostic",
            cycle_label=f"psid_{psid}",
        )
    finally:
        mcts.close()

    snap = record.get("simulator_snapshot") or {}
    return {
        **item,
        "dataset": tag,
        "local_psid": int(local_psid),
        "shard_path": shard_rel,
        "in_shard_idx": int(in_shard_idx),
        "root_id": int(record.get("root_id", psid)),
        "root_player": str(record.get("root_player", "")),
        "root_depth": int(record.get("root_depth", 0)),
        "history_hops": int(record.get("history_hops", 0)),
        "record_sim_time": float(snap.get("time", 0.0)),
        "state_sim_time": float(state.simulator._time),
        "iterations": int(iterations),
        "mcts_best_action_index": "" if result.best_action_index is None else int(result.best_action_index),
        "mcts_best_action_value": float(result.best_action_value),
        "mcts_best_action_repr": repr(result.best_action) if result.best_action is not None else "",
        "root_log_path": str(root_log_path),
        "child_log_path": str(child_log_path),
        "status": "ok",
        "error": "",
    }


def _plot_summary(summary_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, Any]] = []
    with summary_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No successful rows in {summary_path}")

    def vals(split: str, key: str) -> list[float]:
        return [float(r[key]) for r in rows if r["split"] == split]

    data = [
        vals("train", "y_pred"),
        vals("eval", "y_pred"),
        vals("train", "mcts_best_action_value"),
        vals("eval", "mcts_best_action_value"),
    ]
    labels = [
        f"y_pred\ntrain\nn={len(data[0])}",
        f"y_pred\neval\nn={len(data[1])}",
        f"MCTS value\ntrain\nn={len(data[2])}",
        f"MCTS value\neval\nn={len(data[3])}",
    ]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    parts = ax.violinplot(
        data,
        positions=[1, 2, 4, 5],
        showmeans=True,
        showmedians=True,
        showextrema=True,
        widths=0.82,
    )
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#1f2933")
        body.set_alpha(0.72)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#1f2933")
            parts[key].set_linewidth(1.0)
    ax.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.set_xticks([1, 2, 4, 5])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Controller-perspective value")
    ax.set_title("Large-error samples: model y_pred vs 1000-iteration MCTS value")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-d1", type=Path, default=DEFAULT_D1_DIR)
    parser.add_argument("--dataset-d2", type=Path, default=DEFAULT_D2_DIR)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--root-player-filter", default="controller")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--uct-c", type=float, default=1.4)
    parser.add_argument("--discount-factor", type=float, default=0.98)
    parser.add_argument("--discount-time-denom", type=float, default=0.015725797204323228)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else model_dir / f"mcts_abs_err_ge{float(args.threshold):g}_{int(args.iterations)}"
    logs_dir = out_dir / "logs"
    summary_path = out_dir / "mcts_large_error_summary.csv"
    plot_path = out_dir / "violin_y_pred_vs_mcts_value.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = _read_large_error_rows(model_dir, threshold=float(args.threshold))
    if int(args.limit) > 0:
        items = items[: int(args.limit)]
    completed = _read_completed(summary_path) if bool(args.resume) else set()

    dataset_dirs = {"d1": Path(args.dataset_d1).expanduser(), "d2": Path(args.dataset_d2).expanduser()}
    offsets_by_tag: dict[str, list[tuple[int, int, str]]] = {}
    counts_by_tag: dict[str, int] = {}
    for tag, dataset_dir in dataset_dirs.items():
        offsets_by_tag[tag], counts_by_tag[tag] = _manifest_offsets(dataset_dir, root_player_filter=str(args.root_player_filter))

    loaders = {tag: _build_loader(dataset_dir, model_dir) for tag, dataset_dir in dataset_dirs.items()}
    shard_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    print(json.dumps({
        "model_dir": str(model_dir),
        "output_dir": str(out_dir),
        "threshold": float(args.threshold),
        "iterations": int(args.iterations),
        "num_large_error_rows": len(items),
        "completed_existing": len(completed),
        "dataset_counts": counts_by_tag,
    }, indent=2, sort_keys=True), flush=True)

    start_all = time.perf_counter()
    ok = 0
    failed = 0
    skipped = 0
    try:
        for pos, item in enumerate(items, start=1):
            psid = int(item["psid"])
            if psid in completed:
                skipped += 1
                continue
            t0 = time.perf_counter()
            try:
                row = _run_one(
                    item=item,
                    n_d1=int(counts_by_tag["d1"]),
                    offsets_by_tag=offsets_by_tag,
                    dataset_dirs=dataset_dirs,
                    loaders=loaders,
                    shard_cache=shard_cache,
                    logs_dir=logs_dir,
                    iterations=int(args.iterations),
                    seed=int(args.seed),
                    uct_c=float(args.uct_c),
                    discount_factor=float(args.discount_factor),
                    discount_time_denom=float(args.discount_time_denom),
                )
                row["elapsed_sec"] = time.perf_counter() - t0
                ok += 1
            except Exception as exc:
                row = {
                    **item,
                    "iterations": int(args.iterations),
                    "status": "error",
                    "error": repr(exc),
                    "elapsed_sec": time.perf_counter() - t0,
                }
                failed += 1
            _append_summary(summary_path, row)
            if (ok + failed) % 25 == 0 or pos == len(items):
                elapsed = time.perf_counter() - start_all
                done = ok + failed + skipped
                rate = (ok + failed) / max(1e-9, elapsed)
                remaining = len(items) - done
                eta = remaining / max(1e-9, rate) if rate > 0 else None
                print(
                    json.dumps({
                        "processed_position": pos,
                        "total": len(items),
                        "ok": ok,
                        "failed": failed,
                        "skipped": skipped,
                        "elapsed_sec": elapsed,
                        "rate_per_sec": rate,
                        "eta_sec": eta,
                    }, sort_keys=True),
                    flush=True,
                )
    finally:
        for loader in loaders.values():
            loader.close()

    _plot_summary(summary_path, plot_path)
    print(json.dumps({
        "done": True,
        "summary_path": str(summary_path),
        "plot_path": str(plot_path),
        "ok_new": ok,
        "failed_new": failed,
        "skipped_existing": skipped,
        "elapsed_sec": time.perf_counter() - start_all,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
