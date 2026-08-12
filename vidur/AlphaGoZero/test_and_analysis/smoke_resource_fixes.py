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
    CurrentModelPaths,
    WorkerState,
    _cleanup_game_output,
    _merge_game_into_active,
    _ready_shards_in_upload_order,
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


def _make_shard(
    root: Path,
    *,
    worker_id: str,
    shard_id: str,
    base_gid: int,
    controller_model_version: int = 100,
    adversary_model_version: int = 100,
    replay_rows: list[dict[str, object]] | None = None,
    policy_rows: list[dict[str, object]] | None = None,
) -> Path:
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
        replay_rows if replay_rows is not None else _replay_rows(base_gid),
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
        policy_rows if policy_rows is not None else _policy_rows(base_gid),
    )
    manifest = build_shard_manifest(
        shard_dir=shard,
        shard_id=shard_id,
        worker_id=worker_id,
        model_version=max(int(controller_model_version), int(adversary_model_version)),
        controller_model_version=int(controller_model_version),
        adversary_model_version=int(adversary_model_version),
        games_executed=1,
    )
    atomic_write_json(shard / "shard_manifest.json", manifest)
    write_sha256sums(shard, include_manifest=True)
    return shard


def _check_training_replay_snapshot(root: Path) -> None:
    partitions = root / "global_replay" / "partitions" / "worker1"

    complete = partitions / "complete__controller"
    complete.mkdir(parents=True)
    (complete / "replay_target_runtime_feature_complete.csv").write_text("state\n", encoding="utf-8")
    (complete / "replay_policy_rows.csv").write_text("policy\n", encoding="utf-8")
    atomic_write_json(complete / "partition_manifest.json", {"partition_role": "controller"})

    publishing = partitions / "publishing__controller"
    publishing.mkdir(parents=True)
    publishing_state = publishing / "replay_target_runtime_feature_complete.csv"
    publishing_policy = publishing / "replay_policy_rows.csv"
    publishing_state.write_text("state\n", encoding="utf-8")

    malformed = partitions / "malformed__controller"
    malformed.mkdir(parents=True)
    (malformed / "replay_target_runtime_feature_complete.csv").write_text("state\n", encoding="utf-8")
    atomic_write_json(malformed / "partition_manifest.json", {"partition_role": "controller"})

    feature_paths, policy_paths = trainer._replay_path_snapshot(root)
    assert feature_paths == [complete / "replay_target_runtime_feature_complete.csv"]
    assert policy_paths == [complete / "replay_policy_rows.csv"]

    publishing_policy.write_text("policy\n", encoding="utf-8")
    feature_paths, policy_paths = trainer._replay_path_snapshot(root)
    assert publishing_state not in feature_paths, "files without the final manifest are not committed"
    assert publishing_policy not in policy_paths

    atomic_write_json(publishing / "partition_manifest.json", {"partition_role": "controller"})
    feature_paths, policy_paths = trainer._replay_path_snapshot(root)
    assert feature_paths == [
        complete / "replay_target_runtime_feature_complete.csv",
        publishing_state,
    ]
    assert policy_paths == [
        complete / "replay_policy_rows.csv",
        publishing_policy,
    ]

    legacy = root / "legacy" / "global_replay"
    legacy.mkdir(parents=True)
    legacy_feature = legacy / "replay_target_runtime_feature_complete.csv"
    legacy_policy = legacy / "replay_policy_rows.csv"
    legacy_feature.write_text("state\n", encoding="utf-8")
    legacy_policy.write_text("policy\n", encoding="utf-8")
    assert trainer._replay_path_snapshot(root / "legacy") == ([legacy_feature], [legacy_policy])


def _check_xl_ingest_and_streaming(root: Path) -> None:
    _check_training_replay_snapshot(root / "snapshot_contract")
    state = {"accepted_shards": [], "max_replay_states": 100}
    shard1 = _make_shard(root, worker_id="worker1", shard_id="worker1_000000", base_gid=1)
    assert xl._ingest_shard(root, shard1, state)
    assert not shard1.exists(), "full incoming shard should be removed after accepted ingest"
    assert not (root / "global_replay" / "replay_policy_rows.csv").exists(), "new ingest should not write monolithic policy replay"
    assert not (root / "accepted" / "worker1" / "worker1_000000" / "replay_target_runtime.csv").exists(), "accepted dir should keep manifests only"

    state["controller_promoted_model_version"] = 101
    state["adversary_promoted_model_version"] = 100
    state["controller_promoted_model_history"] = [100, 101]
    state["adversary_promoted_model_history"] = [100]
    shard2 = _make_shard(root, worker_id="worker1", shard_id="worker1_000001", base_gid=2, controller_model_version=101, adversary_model_version=100)
    assert xl._ingest_shard(root, shard2, state)
    partitions = sorted((root / "global_replay" / "partitions").glob("*/*/partition_manifest.json"))
    assert len(partitions) == 4, f"expected two role partitions per shard, found {len(partitions)}"
    partition_manifests = [json.loads(path.read_text(encoding="utf-8")) for path in partitions]
    assert {str(item["partition_role"]) for item in partition_manifests} == {"controller", "adversary"}
    assert int(state["feature_states"]) == 4, state
    with (root / "replay_distribution.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "replay_distribution.csv should contain role distribution rows"
    last_distribution = rows[-1]
    assert "current_promoted_controller_model_version" in last_distribution
    assert "current_promoted_adversary_model_version" in last_distribution
    assert "other_controller_model_versions" in last_distribution

    replay_paths = trainer._feature_replay_paths(root)
    policy_paths = trainer._policy_replay_paths(root)
    sampled, ctrl_roots, adv_roots, counts = trainer._stream_sample_state_rows(
        replay_paths,
        seed=123,
        max_value_rows=10,
        max_controller_policy_roots=10,
        max_adversary_policy_roots=10,
    )
    assert counts["states"] == 4, counts
    assert len(sampled) == 4
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
    assert Xc.shape == (6, 269), Xc.shape
    assert Xa.shape == (4, 233), Xa.shape
    assert yc.shape[0] == pc.shape[0] == offc[-1][1]
    assert ya.shape[0] == pa.shape[0] == offa[-1][1]


def _custom_role_rows(
    base_gid: int,
    *,
    controller_count: int,
    adversary_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    features = json.dumps([0.01] * 226)
    ctrl_action = json.dumps([0.02] * 43)
    adv_action = json.dumps([0.03] * 7)
    replay: list[dict[str, object]] = []
    policy: list[dict[str, object]] = []
    for role, count, offset, action in (
        ("controller", controller_count, 0, ctrl_action),
        ("adversary", adversary_count, 10_000, adv_action),
    ):
        for index in range(count):
            game_id = base_gid + offset + index
            replay.append({
                "game_id": game_id,
                "turn_number": 0,
                "depth_number": 0,
                "player": role,
                "root_player": role,
                "target_value": 0.25 if role == "controller" else -0.5,
                "feature_complete": 1,
                "state_features_json": features,
            })
            policy.append({
                "game_id": game_id,
                "turn_number": 0,
                "depth_number": 0,
                "player": role,
                "canon_action_index": 0,
                "visit_count": 1,
                "mcts_visit_prob": 1.0,
                "action_features_json": action,
            })
    return replay, policy


def _check_simple_fifo_ingestion(root: Path) -> None:
    state = {
        "accepted_shards": [],
        "max_replay_states": 7,
        "controller_max_replay_states": 4,
        "adversary_max_replay_states": 3,
    }

    def ingest(
        shard_id: str,
        base_gid: int,
        controller_version: int,
        adversary_version: int,
        controller_count: int,
        adversary_count: int,
    ) -> dict[str, object]:
        replay, policy = _custom_role_rows(
            base_gid,
            controller_count=controller_count,
            adversary_count=adversary_count,
        )
        shard = _make_shard(
            root,
            worker_id="worker1",
            shard_id=shard_id,
            base_gid=base_gid,
            controller_model_version=controller_version,
            adversary_model_version=adversary_version,
            replay_rows=replay,
            policy_rows=policy,
        )
        assert xl._ingest_shard(root, shard, state)
        return json.loads(
            (root / "accepted" / "worker1" / shard_id / "accepted_manifest.json").read_text(
                encoding="utf-8"
            )
        )

    first = ingest("fifo_1", 1_000, 1, 1, 2, 1)
    second = ingest("fifo_2", 2_000, 999, 888, 3, 2)
    third = ingest("fifo_3", 3_000, 50, 40, 1, 2)

    for accepted in (first, second, third):
        assert accepted["admission"]["controller_reason"] == "admit_fifo"
        assert accepted["admission"]["adversary_reason"] == "admit_fifo"
        assert int(accepted["admission"]["dropped_controller_states"]) == 0
        assert int(accepted["admission"]["dropped_adversary_states"]) == 0

    entries = xl._partition_entries(root)
    retained = {
        (str(entry["data"]["partition_role"]), str(entry["data"]["source_shard_id"]))
        for entry in entries
    }
    assert ("controller", "fifo_1") not in retained
    assert ("controller", "fifo_2") in retained
    assert ("controller", "fifo_3") in retained
    assert ("adversary", "fifo_1") not in retained
    assert ("adversary", "fifo_2") not in retained
    assert ("adversary", "fifo_3") in retained
    assert int(state["feature_controller"]) == 4
    assert int(state["feature_adversary"]) == 2
    with (root / "replay_distribution.csv").open(encoding="utf-8", newline="") as handle:
        distribution = list(csv.DictReader(handle))[-1]
    assert distribution["other_controller_model_versions"] == "50,999"
    assert distribution["other_controller_model_states"] == "1,3"
    assert distribution["other_adversary_model_versions"] == "40"
    assert distribution["other_adversary_model_states"] == "2"


def _check_worker_fifo_upload_order(root: Path) -> None:
    ready = root / "ready"
    for shard_id, controller_version, adversary_version in (
        ("001_stale", 102, 107),
        ("002_one_current", 102, 108),
        ("003_both_current", 112, 108),
    ):
        shard = ready / shard_id
        shard.mkdir(parents=True, exist_ok=True)
        atomic_write_json(shard / "shard_manifest.json", {
            "shard_id": shard_id,
            "model_version": max(controller_version, adversary_version),
            "controller_model_version": controller_version,
            "adversary_model_version": adversary_version,
        })
    current = CurrentModelPaths(
        controller_value_model_path=Path("controller_value.joblib"),
        adversary_value_model_path=Path("adversary_value.joblib"),
        controller_prior_model_path=Path("controller_prior.joblib"),
        adversary_prior_model_path=Path("adversary_prior.joblib"),
        model_version=112,
        controller_model_version=112,
        adversary_model_version=108,
    )
    ordered = _ready_shards_in_upload_order(SimpleNamespace(run_id="test"), ready, current)
    assert [path.name for path in ordered] == [
        "001_stale",
        "002_one_current",
        "003_both_current",
    ]


def _check_parallel_role_partition_migration(root: Path) -> None:
    source = root / "global_replay" / "partitions" / "worker1" / "legacy_1"
    replay_rows = _replay_rows(50_000)
    policy_rows = _policy_rows(50_000)
    _write_csv(
        source / "replay_target_runtime_feature_complete.csv",
        list(replay_rows[0]),
        replay_rows,
    )
    _write_csv(source / "replay_policy_rows.csv", list(policy_rows[0]), policy_rows)
    atomic_write_json(source / "partition_manifest.json", {
        "schema_version": 3,
        "shard_id": "legacy_1",
        "worker_id": "worker1",
        "model_version": 100,
        "controller_model_version": 100,
        "adversary_model_version": 100,
        "raw_counts": {"states": 2, "controller": 1, "adversary": 1},
        "feature_counts": {"states": 2, "controller": 1, "adversary": 1},
    })
    state: dict[str, object] = {}
    assert xl._migrate_mixed_replay_partitions(root, state, workers=2) == 1
    assert not source.exists()
    assert (source.parent / "legacy_1__controller" / "partition_manifest.json").exists()
    assert (source.parent / "legacy_1__adversary" / "partition_manifest.json").exists()
    assert int(state["feature_controller"]) == 1
    assert int(state["feature_adversary"]) == 1


def _check_training_gate_consumes_only_launch_replay(root: Path) -> None:
    atomic_write_json(root / "xl_state.json", {
        "feature_states": 1_000,
        "lifetime_admitted_states": 1_000,
        "new_states_since_last_training": 1_000,
    })
    atomic_write_json(root / "models" / "Model_Version101" / "candidate_manifest.json", {
        "model_version": 101,
        "eval_status": "complete",
    })
    atomic_write_json(root / "training" / "current_training.json", {
        "expected_candidate_model_version_min": 101,
        "gate": {
            "new_states_since_last_training": 600,
            "lifetime_admitted_states": 600,
            "total_states": 600,
        },
    })
    assert xl._finalize_completed_training_cycle_if_needed(root, root / "training")
    state = json.loads((root / "xl_state.json").read_text(encoding="utf-8"))
    assert int(state["new_states_since_last_training"]) == 400
    assert int(state["states_consumed_by_last_training_cycle"]) == 600
    assert int(state["last_training_gate_lifetime_admitted_states"]) == 600
    assert int(state["last_candidate_model_version"]) == 101

    # A stale trainer-side completion write cannot make the next gate consume
    # the entire retained replay again; only the lifetime delta is eligible.
    state["new_states_since_last_training"] = 1_000
    atomic_write_json(root / "xl_state.json", state)
    gate = xl.write_training_gate_status(root)
    assert int(gate["new_states_since_last_training"]) == 400


def _check_policy_caps_are_not_training_minima(root: Path) -> None:
    original = (
        xl.MIN_CONTROLLER_STATES_FOR_EVAL,
        xl.MIN_ADVERSARY_STATES_FOR_EVAL,
        xl.XL_CONTROLLER_POLICY_SAMPLE_CAP,
        xl.ADVERSARY_POLICY_SAMPLE_CAP,
        xl.TRAIN_TRIGGER_NEW_STATES,
        xl.TRAIN_SAMPLE_MIN_LARGE_REPLAY,
    )
    try:
        xl.MIN_CONTROLLER_STATES_FOR_EVAL = 100_000
        xl.MIN_ADVERSARY_STATES_FOR_EVAL = 50_000
        xl.XL_CONTROLLER_POLICY_SAMPLE_CAP = 500_000
        xl.ADVERSARY_POLICY_SAMPLE_CAP = 300_000
        xl.TRAIN_TRIGGER_NEW_STATES = 50_000
        xl.TRAIN_SAMPLE_MIN_LARGE_REPLAY = 100_000
        atomic_write_json(root / "xl_state.json", {
            "feature_states": 150_000,
            "feature_controller": 100_000,
            "feature_adversary": 50_000,
            "lifetime_admitted_states": 150_000,
            "new_states_since_last_training": 150_000,
        })
        gate = xl.write_training_gate_status(root)
        assert gate["can_train"] is True
        assert int(gate["controller_policy_sample_count"]) == 100_000
        assert int(gate["controller_policy_sample_requirement"]) == 100_000
        assert int(gate["controller_policy_sample_cap"]) == 500_000
        assert int(gate["adversary_policy_sample_count"]) == 50_000
        assert int(gate["adversary_policy_sample_requirement"]) == 50_000
        assert int(gate["adversary_policy_sample_cap"]) == 300_000
        assert trainer._available_policy_root_requirement(73_000, 500_000) == 73_000
        assert trainer._available_policy_root_requirement(700_000, 500_000) == 500_000
    finally:
        (
            xl.MIN_CONTROLLER_STATES_FOR_EVAL,
            xl.MIN_ADVERSARY_STATES_FOR_EVAL,
            xl.XL_CONTROLLER_POLICY_SAMPLE_CAP,
            xl.ADVERSARY_POLICY_SAMPLE_CAP,
            xl.TRAIN_TRIGGER_NEW_STATES,
            xl.TRAIN_SAMPLE_MIN_LARGE_REPLAY,
        ) = original


def _check_completed_cycle_recovery_without_pid(root: Path) -> None:
    atomic_write_json(root / "xl_state.json", {
        "feature_states": 1_000,
        "lifetime_admitted_states": 1_000,
        "new_states_since_last_training": 1_000,
    })
    atomic_write_json(root / "models" / "Model_Version101" / "candidate_manifest.json", {
        "model_version": 101,
        "eval_status": "complete",
    })
    atomic_write_json(root / "training" / "current_training.json", {
        "expected_candidate_model_version_min": 101,
        "lifetime_admitted_states_at_launch": 600,
        "gate": {"new_states_since_last_training": 600},
    })

    first = xl.maybe_launch_training(root, {"can_train": False})
    assert first["training_cycle_recovered_without_pid"] is True
    state = json.loads((root / "xl_state.json").read_text(encoding="utf-8"))
    assert int(state["new_states_since_last_training"]) == 400

    second = xl.maybe_launch_training(root, {"can_train": False})
    assert second == {"training_launched": False, "training_running_pid": 0}


def _check_failed_eval_recovery_selection(root: Path) -> None:
    atomic_write_json(root / "xl_state.json", {
        "last_training_cycle_completed_model_version": 123,
    })
    for version, status in ((124, "failed"), (125, "complete"), (133, "failed")):
        atomic_write_json(
            root / "models" / f"Model_Version{version}" / "candidate_manifest.json",
            {"model_version": version, "eval_status": status},
        )
    assert xl._latest_failed_candidate_version(root) == 133
    atomic_write_json(root / "xl_state.json", {
        "last_training_cycle_completed_model_version": 133,
    })
    assert xl._latest_failed_candidate_version(root) == 0


def _check_eval_retry_preserves_original_gate(root: Path) -> None:
    train_dir = root / "training"
    atomic_write_json(train_dir / "current_training.json", {
        "expected_candidate_model_version_min": 133,
        "gate": {"new_states_since_last_training": 600_000},
    })
    original_popen = xl.subprocess.Popen
    try:
        xl.subprocess.Popen = lambda *args, **kwargs: SimpleNamespace(pid=12345)
        result = xl._launch_existing_candidate_eval(
            root,
            train_dir,
            {"new_states_since_last_training": 1_200_000},
            133,
        )
    finally:
        xl.subprocess.Popen = original_popen
    current = json.loads((train_dir / "current_training.json").read_text(encoding="utf-8"))
    assert result["evaluation_retry_launched"] is True
    assert int(current["gate"]["new_states_since_last_training"]) == 600_000


def _check_role_only_promotion(root: Path) -> None:
    current = trainer.ModelBundle(
        model_version=100,
        controller_model_version=100,
        adversary_model_version=100,
        controller_value_model_path=root / "v100" / "controller_value.joblib",
        adversary_value_model_path=root / "v100" / "adversary_value.joblib",
        controller_prior_model_path=root / "v100" / "controller_prior.joblib",
        adversary_prior_model_path=root / "v100" / "adversary_prior.joblib",
    )
    candidate = trainer.ModelBundle(
        model_version=101,
        controller_model_version=101,
        adversary_model_version=101,
        controller_value_model_path=root / "v101" / "controller_value.joblib",
        adversary_value_model_path=root / "v101" / "adversary_value.joblib",
        controller_prior_model_path=root / "v101" / "controller_prior.joblib",
        adversary_prior_model_path=root / "v101" / "adversary_prior.joblib",
    )
    promoted = trainer._promote_candidate_roles(
        root,
        current=current,
        candidate=candidate,
        promote_controller=True,
        promote_adversary=False,
    )
    assert int(promoted.controller_model_version) == 101
    assert int(promoted.adversary_model_version) == 100
    current_model = json.loads((root / "models" / "current_model.json").read_text(encoding="utf-8"))
    assert int(current_model["controller_model_version"]) == 101
    assert int(current_model["adversary_model_version"]) == 100
    assert not (root / "xl_state.json").exists(), "trainer must not rewrite coordinator replay state"


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
        iterations=4_000,
        puct_c=2.5,
        root_dirichlet_alpha=0.1,
        root_dirichlet_total_concentration=0.0,
        root_dirichlet_epsilon=0.35,
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
        _check_simple_fifo_ingestion(root / "simple_fifo")
        _check_worker_fifo_upload_order(root / "upload_order")
        _check_parallel_role_partition_migration(root / "partition_migration")
        _check_training_gate_consumes_only_launch_replay(root / "gate_accounting")
        _check_policy_caps_are_not_training_minima(root / "adaptive_policy_gate")
        _check_completed_cycle_recovery_without_pid(root / "cycle_recovery")
        _check_failed_eval_recovery_selection(root / "failed_eval_selection")
        _check_eval_retry_preserves_original_gate(root / "eval_retry_gate")
        _check_role_only_promotion(root / "role_promotion")
        _check_worker_cleanup(root / "worker_check")
        print(f"ok smoke_resource_fixes root={root}")
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="agz_resource_smoke_"))
        try:
            root = tmp_dir / "agz_smoke"
            _check_xl_ingest_and_streaming(root / "xl")
            _check_simple_fifo_ingestion(root / "simple_fifo")
            _check_worker_fifo_upload_order(root / "upload_order")
            _check_parallel_role_partition_migration(root / "partition_migration")
            _check_training_gate_consumes_only_launch_replay(root / "gate_accounting")
            _check_policy_caps_are_not_training_minima(root / "adaptive_policy_gate")
            _check_completed_cycle_recovery_without_pid(root / "cycle_recovery")
            _check_failed_eval_recovery_selection(root / "failed_eval_selection")
            _check_eval_retry_preserves_original_gate(root / "eval_retry_gate")
            _check_role_only_promotion(root / "role_promotion")
            _check_worker_cleanup(root / "worker_check")
            print(f"ok smoke_resource_fixes root={root}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
