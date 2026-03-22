from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vidur.mcts.linear.evaluator.feature_adapter import (
    extract_lp_baseline_v1_feature_map,
    extract_lp_baseline_v1_vector,
)
from vidur.mcts.linear.evaluator.weights import (
    load_weights_from_json,
    load_weights_from_lp_solution_dir,
)


@dataclass
class DummyReq:
    id: int
    completed: bool
    _is_prefill_complete: bool
    _num_prefill_tokens: int
    num_processed_prefill_tokens: int
    _num_decode_tokens: int
    num_processed_decode_tokens: int
    _prefill_slo_time: float
    _decode_slo_time: float
    arrived_at: float
    queued_at: float


@dataclass
class DummyStats:
    decode_next_deadline_by_id: dict[int, float]


@dataclass
class DummySim:
    _time: float


@dataclass
class DummyState:
    simulator: DummySim
    stats: DummyStats


class DummyEnv:
    def __init__(self, reqs: list[DummyReq]) -> None:
        self._reqs = reqs

    def _build_request_lookup(self, simulator, state=None):
        return {int(r.id): r for r in self._reqs}

    def describe_state(self, state):
        return {"requests_in_system": len(self._reqs)}


def test_load_weights_from_lp_solution_dir(tmp_path: Path) -> None:
    out = tmp_path / "lp"
    out.mkdir(parents=True, exist_ok=True)
    (out / "lp_feature_names.json").write_text(
        json.dumps({"feature_names": ["bias", "x"]}),
        encoding="utf-8",
    )
    np.savez(out / "lp_solution.npz", weights=np.asarray([1.5, -2.0], dtype=np.float64))

    loaded = load_weights_from_lp_solution_dir(str(out))
    assert loaded.feature_names == ["bias", "x"]
    assert loaded.weights.shape == (2,)
    assert float(loaded.weights[0]) == 1.5


def test_load_weights_from_json_with_names_file(tmp_path: Path) -> None:
    w_path = tmp_path / "weights.json"
    n_path = tmp_path / "lp_feature_names.json"
    w_path.write_text(
        json.dumps({"weights_by_feature": {"bias": 3.0, "x": -1.25}}),
        encoding="utf-8",
    )
    n_path.write_text(json.dumps({"feature_names": ["x", "bias"]}), encoding="utf-8")

    loaded = load_weights_from_json(str(w_path), feature_names_json=str(n_path))
    assert loaded.feature_names == ["x", "bias"]
    assert np.allclose(loaded.weights, np.asarray([-1.25, 3.0], dtype=np.float64))


def test_lp_feature_adapter_baseline_v1_shape_and_values() -> None:
    reqs = [
        DummyReq(
            id=2,
            completed=False,
            _is_prefill_complete=False,
            _num_prefill_tokens=3072,
            num_processed_prefill_tokens=0,
            _num_decode_tokens=5000,
            num_processed_decode_tokens=0,
            _prefill_slo_time=0.3,
            _decode_slo_time=0.05,
            arrived_at=0.0,
            queued_at=0.0,
        ),
        DummyReq(
            id=5,
            completed=False,
            _is_prefill_complete=True,
            _num_prefill_tokens=3072,
            num_processed_prefill_tokens=3072,
            _num_decode_tokens=100,
            num_processed_decode_tokens=0,
            _prefill_slo_time=0.3,
            _decode_slo_time=0.05,
            arrived_at=0.0,
            queued_at=0.0,
        ),
    ]
    env = DummyEnv(reqs)
    state = DummyState(
        simulator=DummySim(_time=0.25),
        stats=DummyStats(decode_next_deadline_by_id={5: 0.2}),
    )

    fmap = extract_lp_baseline_v1_feature_map(env, state)
    names = sorted(fmap.keys())
    vec, fmap2 = extract_lp_baseline_v1_vector(env, state, feature_names=names)
    assert vec.shape == (len(names),)
    assert len(fmap2) == 7
    assert abs(float(fmap2["bias"]) - 1.0) < 1e-12
    assert abs(float(fmap2["current_burst_size_norm"]) - (2.0 / 6.0)) < 1e-12
    assert abs(float(fmap2["current_burst_remaining_chunks_norm"]) - (6.0 / 36.0)) < 1e-12
    assert abs(float(fmap2["current_burst_deadline_urgency_norm"]) - 20.0) < 1e-12
    assert abs(float(fmap2["current_burst_lateness_sec_norm"]) - 0.0) < 1e-12
    assert abs(float(fmap2["decode_active_norm"]) - (1.0 / 200.0)) < 1e-12
    assert abs(float(fmap2["decode_violated_count_norm"]) - (1.0 / 200.0)) < 1e-12
