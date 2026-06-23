"""Smoke checks for AlphaGoZero memory/disk resource fixes.

This test avoids native arena execution. It validates the Python-side contracts
that keep replay memory and disk bounded:
- XL ingests to prunable per-shard partitions, not monolithic replay CSVs.
- XL keeps accepted shard manifests only by default.
- Trainer sampling reads bounded samples from partitions.
- Worker deletes per-game run directories after merging replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from vidur.AlphaGoZero import agz_train_eval_promote as trainer
from vidur.AlphaGoZero import xl_coordinator as xl
from vidur.AlphaGoZero.durable_transfer import (
    atomic_write_json,
    build_shard_manifest,
    write_sha256sums,
)
from vidur.AlphaGoZero.worker_daemon import (
    WorkerState,
    _cleanup_game_output,
    _merge_game_into_active,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _replay_rows(base_gid: int) -> list[dict[str, object]]:
    features = json.dumps([0.01] * 226)
    return [
        {
            "game_id": base_gid,
            "turn_number": 0,
            "depth_number": 0,
            "player": "controller",
            "root_player": "controller",
            "target_value": 0.25,
            "feature_complete": 1,
            "state_features_json": features,
        },
        {
            "game_id": base_gid,
            "turn_number": 1,
            "depth_number": 0,
            "player": "adversary",
            "root_player": "adversary",
            "target_value": -0.5,
            "feature_complete": 1,
            "state_features_json": features,
        },
    ]


def _policy_rows(base_gid: int) -> list[dict[str, object]]:
    ctrl = json.dumps([0.02] * 43)
    adv = json.dumps([0.03] * 7)
    rows: list[dict[str, object]] = []
    for idx, visits in enumerate((7, 3, 1)):
        rows.append(
            {
                "game_id": base_gid,
                "turn_number": 0,
                "depth_number": 0,
                "player": "controller",
                "canon_action_index": idx,
                "visit_count": visits,
                "mcts_visit_prob": visits / 11.0,
                "action_features_json": ctrl,
            }
        )
    for idx, visits in enumerate((5, 2)):
        rows.append(
            {
                "game_id": base_gid,
                "turn_number": 1,
                "depth_number": 0,
                "player": "adversary",
                "canon_action_index": idx,
                "visit_count": visits,
                "mcts_visit_prob": visits / 7.0,
                "action_features_json": adv,
            }
        )
    return rows


def _make_shard(root: Path, *, worker_id: str, shard_id: str, base_gid: int) -> Path:
    shard = root / "incoming" / worker_id / shard_id
    shard.mkdir(parents=True, exist_ok=True)
    _write_csv(
        shard / "replay_target_runtime.csv",
        [
            "game_id",
            "turn_number",
            "depth_number",
            "player",
            "root_player",
            "target_value",
            "feature_complete",
            "state_features_json",
        ],
        _replay_rows(base_gid),
    )
    _write_csv(
        shard / "replay_policy_rows.csv",
        [
            "game_id",
            "turn_number",
            "depth_number",
            "player",
            "canon_action_index",
            "visit_count",
            "mcts_visit_prob",
            "action_features_json",
        ],
        _policy_rows(base_gid),
    )
    manifest = build_shard_manifest(
        shard_dir=shard,
        shard_id=shard_id,
        worker_id=worker_id,
        model_version=100,
        games_executed=1,
    )
    atomic_write_json(shard / "shard_manifest.json", manifest)
    write_sha256sums(shard, include_manifest=True)
    return shard


def _check_xl_ingest_and_streaming(root: Path) -> None:
    state = {"accepted_shards": [], "max_replay_states": 2}
    shard1 = _make_shard(root, worker_id="worker1", shard_id="worker1_000000", base_gid=1)
    assert xl._ingest_shard(root, shard1, state)
    assert not shard1.exists(), "full incoming shard should be removed after accepted ingest"
    assert not (root / "global_replay" / "replay_policy_rows.csv").exists(), "new ingest should not write monolithic policy replay"
    assert not (root / "accepted" / "worker1" / "worker1_000000" / "replay_target_runtime.csv").exists(), "accepted dir should keep manifests only"

    shard2 = _make_shard(root, worker_id="worker1", shard_id="worker1_000001", base_gid=2)
    assert xl._ingest_shard(root, shard2, state)
    partitions = sorted((root / "global_replay" / "partitions").glob("*/*/partition_manifest.json"))
    assert len(partitions) == 1, f"expected pruning to keep one partition, found {len(partitions)}"
    assert int(state["feature_states"]) == 2, state

    replay_paths = trainer._feature_replay_paths(root)
    policy_paths = trainer._policy_replay_paths(root)
    sampled, ctrl_roots, adv_roots, counts = trainer._stream_sample_state_rows(
        replay_paths,
        seed=123,
        max_value_rows=10,
        max_controller_policy_roots=10,
        max_adversary_policy_roots=10,
    )
    assert counts["states"] == 2, counts
    assert len(sampled) == 2
    Xc, yc, pc, offc = trainer._collect_policy_training_arrays(
        policy_paths,
        ctrl_roots,
        player="controller",
        action_dim=43,
    )
    Xa, ya, pa, offa = trainer._collect_policy_training_arrays(
        policy_paths,
        adv_roots,
        player="adversary",
        action_dim=7,
    )
    assert Xc.shape == (3, 269), Xc.shape
    assert Xa.shape == (2, 233), Xa.shape
    assert yc.shape[0] == pc.shape[0] == offc[-1][1]
    assert ya.shape[0] == pa.shape[0] == offa[-1][1]


def _check_worker_cleanup(root: Path) -> None:
    worker_root = root / "worker"
    game_out = worker_root / "runs" / "game_1"
    _write_csv(
        game_out / "replay_target_runtime.csv",
        [
            "game_id",
            "turn_number",
            "depth_number",
            "player",
            "root_player",
            "target_value",
            "feature_complete",
            "state_features_json",
        ],
        _replay_rows(10),
    )
    _write_csv(
        game_out / "replay_policy_rows.csv",
        [
            "game_id",
            "turn_number",
            "depth_number",
            "player",
            "canon_action_index",
            "visit_count",
            "mcts_visit_prob",
            "action_features_json",
        ],
        _policy_rows(10),
    )
    args = SimpleNamespace(
        output_root=worker_root,
        keep_game_runs=False,
        worker_id="worker_smoke",
        run_id="resource_smoke",
    )
    st = WorkerState(model_version=100)
    _merge_game_into_active(
        args,
        game_id=10,
        game_out=game_out,
        replay_csv=game_out / "replay_target_runtime.csv",
        st=st,
        model_version=100,
    )
    _cleanup_game_output(args, game_out)
    assert not game_out.exists(), "worker game run directory should be deleted by default"
    assert (worker_root / "active" / "replay_target_runtime.csv").exists()
    assert int(st.states) == 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-dir", action="store_true")
    args = parser.parse_args()
    if args.keep_dir:
        root = Path(tempfile.mkdtemp(prefix="agz_resource_smoke_")) / "agz_smoke"
        _check_xl_ingest_and_streaming(root / "xl")
        _check_worker_cleanup(root / "worker_check")
        print(f"ok smoke_resource_fixes root={root}")
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="agz_resource_smoke_"))
        try:
            root = tmp_dir / "agz_smoke"
            _check_xl_ingest_and_streaming(root / "xl")
            _check_worker_cleanup(root / "worker_check")
            print(f"ok smoke_resource_fixes root={root}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
