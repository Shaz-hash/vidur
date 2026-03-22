# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
RUN COMMAND:
PYTHONPATH=/home/shazer/Desktop/Research/Vidur/vidur \
/home/shazer/Desktop/Research/Vidur/vidur/.venv/bin/python3 \
-m vidur.mcts.linear.sampler.run \
--target-unique-states 1000 \
--workers 8 \
--start-depth 0 \
--max-branching 10 \
--max-forced-hops 20000 \
--history-hops 0,10,20,30,40,50,60,70 \
--out-dir simulator_output/linear_sampler
"""


from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from vidur.mcts.linear.sampler import sampler_parallel
from vidur.mcts.linear.sampler.compat_export import export_compat_csvs
from vidur.mcts.linear.sampler.config import (
    SamplerCollectionSettings,
    SamplerOutputSettings,
    SamplerRunConfig,
)
from vidur.mcts.linear.sampler.run import build_parser, main as sampler_main
from vidur.mcts.linear.sampler.sampler_parallel import run_parallel_sampler
from vidur.mcts.linear.sampler.sampler_worker import SamplerWorkerResult, _WorkerSampler
from vidur.mcts.linear.sampler.state_schema import (
    ActionRow,
    AnchorSampleRow,
    NodeRow,
    RequestRow,
    StateRow,
    TransitionRequestDeltaRow,
    TransitionRow,
    encode_int_list,
)
from vidur.mcts.linear.sampler.storage import (
    MergeOutput,
    WorkerShardPaths,
    merge_worker_shards,
    write_parquet_records,
    write_worker_shard,
)


def _mk_state(state_id: str, waiting_ids: list[int], worker_id: int = 0) -> StateRow:
    return StateRow(
        state_id=state_id,
        worker_id=worker_id,
        round_idx=0,
        root_seed=123,
        history_hop=0,
        player_to_act="controller",
        branching_depth=0,
        sim_time=0.0,
        requests_in_system=len(waiting_ids),
        requests_generated=len(waiting_ids),
        requests_completed=0,
        slo_violations=0,
        total_lateness=0.0,
        total_cost=0.0,
        completed_request_ids_json="[]",
        waiting_request_ids_json=encode_int_list(waiting_ids),
        terminal=False,
    )


def _mk_req(state_id: str, rid: int) -> RequestRow:
    return RequestRow(
        state_id=state_id,
        request_id=rid,
        prefill_tokens_total=1024,
        decode_tokens_total=1000,
        prefill_tokens_remaining=1024,
        decode_tokens_remaining=1000,
        arrived_at=0.0,
        queued_at=0.0,
        prefill_complete=False,
        completed=False,
        prefill_slo=0.3,
        decode_slo=0.05,
        prefill_deadline=0.3,
        decode_deadline=0.0,
        prefill_lateness_now=0.0,
        decode_lateness_now=0.0,
        prefill_violated_now=False,
        decode_violated_now=False,
        total_lateness_now=0.0,
        violated_now=False,
    )


def test_merge_dedup(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"

    s1 = _mk_state("s1", [0], worker_id=0)
    s2 = _mk_state("s2", [1], worker_id=0)
    s2b = _mk_state("s2", [1], worker_id=1)

    a1 = ActionRow("a1", "s1", "controller", 0, "k0", "[0]", "{}", "act0")
    a2 = ActionRow("a2", "s2", "controller", 1, "k1", "[1]", "{}", "act1")

    t1 = TransitionRow(
        transition_id="t1",
        state_id="s1",
        action_id="a1",
        next_state_id="s2",
        actor="controller",
        sim_time_before=0.0,
        sim_time_after_action=0.1,
        sim_time_after_advance=0.1,
        delta_time_action=0.1,
        delta_time_advance=0.0,
        delta_time=0.1,
        cost_s=1.0,
        cost_next=0.5,
        reward=0.5,
        branching_depth=0,
        next_branching_depth=1,
        terminal=False,
        adversary_prefill_deadlines_by_id_json="{}",
    )
    n0 = NodeRow("n0", 0, 0, 0, "tr0", 0, 0, "", "s1", "", 0)
    n1 = NodeRow("n1", 0, 0, 0, "tr0", 0, 1, "n0", "s2", "a1", 1)
    d1 = TransitionRequestDeltaRow("t1", 0, 0.0, 0.0, 0.0, 0)

    shard0 = write_worker_shard(
        out_dir=shard_dir,
        worker_id=0,
        round_idx=0,
        compression="zstd",
        states=[s1, s2],
        requests=[_mk_req("s1", 0), _mk_req("s2", 1)],
        actions=[a1, a2],
        transitions=[t1],
        nodes=[n0, n1],
        transition_deltas=[d1],
        anchors=[
            AnchorSampleRow(
                anchor_state_id="s1",
                worker_id=0,
                round_idx=0,
                history_hop=0,
                player_to_act="controller",
                branching_depth=0,
                trace_len=1,
                attempt_index=1,
            )
        ],
    )
    shard1 = write_worker_shard(
        out_dir=shard_dir,
        worker_id=1,
        round_idx=0,
        compression="zstd",
        states=[s2b],
        requests=[_mk_req("s2", 1)],
        actions=[a2],
        transitions=[],
        nodes=[],
        transition_deltas=[],
        anchors=[],
    )

    merged = merge_worker_shards(
        out_dir=tmp_path,
        round_idx=0,
        shards=[shard0, shard1],
        compression="zstd",
    )
    assert merged.unique_states == 2
    assert Path(merged.merged_state_path).exists()
    assert Path(merged.manifest_path).exists()


def test_compat_export_smoke(tmp_path: Path) -> None:
    merged = tmp_path / "merged"
    merged.mkdir(parents=True, exist_ok=True)

    states = [
        _mk_state("s_root", [0], worker_id=0).as_record(),
        StateRow(
            state_id="s_child",
            worker_id=0,
            round_idx=0,
            root_seed=123,
            history_hop=0,
            player_to_act="adversary",
            branching_depth=1,
            sim_time=0.1,
            requests_in_system=1,
            requests_generated=1,
            requests_completed=0,
            slo_violations=0,
            total_lateness=0.0,
            total_cost=0.0,
            completed_request_ids_json="[]",
            waiting_request_ids_json=encode_int_list([0]),
            terminal=False,
        ).as_record(),
    ]
    actions = [
        ActionRow(
            action_id="a01",
            state_id="s_root",
            actor="controller",
            canonical_index=0,
            canonical_key="k0",
            alias_indices_json="[0]",
            action_json='{"type":"controller"}',
            action_repr="ControllerAction(token_budget=1024)",
        ).as_record()
    ]
    transitions = [
        TransitionRow(
            transition_id="t01",
            state_id="s_root",
            action_id="a01",
            next_state_id="s_child",
            actor="controller",
            sim_time_before=0.0,
            sim_time_after_action=0.1,
            sim_time_after_advance=0.1,
            delta_time_action=0.1,
            delta_time_advance=0.0,
            delta_time=0.1,
            cost_s=0.0,
            cost_next=0.0,
            reward=0.0,
            branching_depth=0,
            next_branching_depth=1,
            terminal=False,
            adversary_prefill_deadlines_by_id_json="{}",
        ).as_record()
    ]
    nodes = [
        NodeRow("n0", 0, 0, 42, "trace0", 0, 0, "", "s_root", "", 0).as_record(),
        NodeRow("n1", 0, 0, 42, "trace0", 0, 1, "n0", "s_child", "a01", 1).as_record(),
    ]

    write_parquet_records(merged / "states.parquet", states, compression="zstd")
    write_parquet_records(merged / "actions.parquet", actions, compression="zstd")
    write_parquet_records(merged / "transitions.parquet", transitions, compression="zstd")
    write_parquet_records(merged / "nodes.parquet", nodes, compression="zstd")

    out = export_compat_csvs(merged_dir=merged, out_dir=tmp_path / "compat")
    assert Path(out.mcts_iter_compat_csv).exists()
    assert Path(out.mcts_root_compat_csv).exists()

    with Path(out.mcts_iter_compat_csv).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["game_id"] == "42"


def test_sampler_cli_defaults_disable_compat() -> None:
    args = build_parser().parse_args([])
    assert args.max_trace_length == 10
    assert args.export_compat_root is False
    assert args.export_compat_iter is False


def test_sampler_main_passes_export_toggles(tmp_path: Path, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(cfg: SamplerRunConfig, *, export_compat_root: bool, export_compat_iter: bool):
        captured["cfg"] = cfg
        captured["export_compat_root"] = export_compat_root
        captured["export_compat_iter"] = export_compat_iter
        return (
            sampler_parallel.SamplerRoundResult(
                round_idx=0,
                worker_results=tuple(),
                merge_output=MergeOutput(
                    merged_state_path=str(tmp_path / "states.parquet"),
                    merged_request_path=str(tmp_path / "requests.parquet"),
                    merged_action_path=str(tmp_path / "actions.parquet"),
                    merged_transition_path=str(tmp_path / "transitions.parquet"),
                    merged_node_path=str(tmp_path / "nodes.parquet"),
                    merged_transition_delta_path=str(tmp_path / "transition_request_deltas.parquet"),
                    merged_anchor_path=str(tmp_path / "anchors.parquet"),
                    manifest_path=str(tmp_path / "manifest.json"),
                    unique_states=0,
                    unique_anchor_samples=0,
                    controller_lp_samples=0,
                    adversary_lp_samples=0,
                ),
                compat_output=sampler_parallel.CompatExportPaths(
                    mcts_root_compat_csv="",
                    mcts_iter_compat_csv="",
                ),
            ),
        )

    monkeypatch.setattr("vidur.mcts.linear.sampler.run.run_parallel_sampler", _fake_run)
    sampler_main(
        [
            "--target-unique-states",
            "5",
            "--workers",
            "1",
            "--max-trace-length",
            "7",
            "--out-dir",
            str(tmp_path / "out"),
            "--export-compat-root",
        ]
    )

    assert captured["export_compat_root"] is True
    assert captured["export_compat_iter"] is False
    assert int(captured["cfg"].collection.max_trace_length) == 7


def test_run_parallel_uses_unique_anchor_target_and_disables_compat_by_default(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    per_worker_targets: list[int] = []
    compat_calls: list[int] = []

    dummy_shard = WorkerShardPaths(
        worker_id=0,
        round_idx=0,
        state_path=str(tmp_path / "dummy_states.parquet"),
        request_path=str(tmp_path / "dummy_requests.parquet"),
        action_path=str(tmp_path / "dummy_actions.parquet"),
        transition_path=str(tmp_path / "dummy_transitions.parquet"),
        node_path=str(tmp_path / "dummy_nodes.parquet"),
        transition_delta_path=str(tmp_path / "dummy_deltas.parquet"),
        anchor_path=str(tmp_path / "dummy_anchors.parquet"),
    )

    def _fake_collect(_cfg: SamplerRunConfig, round_idx: int, *, per_worker_target: int = 0):
        per_worker_targets.append(int(per_worker_target))
        if round_idx == 0:
            return [
                SamplerWorkerResult(0, 0, 0, True, 0, 2, ("sA", "sB"), dummy_shard, ""),
                SamplerWorkerResult(1, 0, 0, True, 0, 1, ("sB",), dummy_shard, ""),
            ]
        return [
            SamplerWorkerResult(0, 1, 0, True, 0, 1, ("sC",), dummy_shard, ""),
            SamplerWorkerResult(1, 1, 0, True, 0, 1, ("sC",), dummy_shard, ""),
        ]

    def _fake_merge(**_: Any) -> MergeOutput:
        return MergeOutput(
            merged_state_path=str(tmp_path / "states.parquet"),
            merged_request_path=str(tmp_path / "requests.parquet"),
            merged_action_path=str(tmp_path / "actions.parquet"),
            merged_transition_path=str(tmp_path / "transitions.parquet"),
            merged_node_path=str(tmp_path / "nodes.parquet"),
            merged_transition_delta_path=str(tmp_path / "transition_request_deltas.parquet"),
            merged_anchor_path=str(tmp_path / "anchors.parquet"),
            manifest_path=str(tmp_path / "manifest.json"),
            unique_states=123,
            unique_anchor_samples=0,
            controller_lp_samples=0,
            adversary_lp_samples=0,
        )

    def _fake_export(**_: Any):
        compat_calls.append(1)
        return sampler_parallel.CompatExportPaths("", "")

    monkeypatch.setattr("vidur.mcts.linear.sampler.sampler_parallel._collect_round", _fake_collect)
    monkeypatch.setattr("vidur.mcts.linear.sampler.sampler_parallel.merge_worker_shards", _fake_merge)
    monkeypatch.setattr("vidur.mcts.linear.sampler.sampler_parallel.export_compat_csvs", _fake_export)

    cfg = SamplerRunConfig(
        collection=SamplerCollectionSettings(
            target_unique_states=3,
            workers=2,
            max_rounds=4,
            shard_unique_states_per_worker=10,
        ),
        output=SamplerOutputSettings(out_dir=str(tmp_path / "out")),
    )

    out = run_parallel_sampler(cfg)
    assert len(out) == 2
    assert per_worker_targets == [2, 1]
    assert compat_calls == []
    assert not (tmp_path / "out" / "round_000" / "compat").exists()


class _DummyStats:
    def clone(self) -> "_DummyStats":
        return _DummyStats()


class _DummySim:
    def snapshot_state(self) -> dict[str, int]:
        return {"ok": 1}


class _DummyState:
    def __init__(self) -> None:
        self.simulator = _DummySim()
        self.stats = _DummyStats()


def test_walk_random_trace_respects_trace_len() -> None:
    sampler = object.__new__(_WorkerSampler)
    sampler.rng = random.Random(123)

    applied: list[str] = []
    sampler._clone_from_snapshot = lambda _snap, _stats: _DummyState()
    sampler._advance_to_branching_or_terminal = lambda state, player: (state, player, False)
    sampler._canonical_actions = lambda _state, _player: [(0, [0], "a0", "k0"), (1, [1], "a1", "k1")]

    def _apply(state: _DummyState, player: str, action: str):
        applied.append(action)
        return state, player

    sampler._apply_action = _apply

    root = _DummyState()
    _state, _player, depth, terminal = sampler._walk_random_trace_to_anchor(
        root,
        "controller",
        trace_len=6,
    )
    assert depth == 6
    assert len(applied) == 6
    assert terminal is False
