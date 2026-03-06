from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vidur.mcts.linear.features import extract_features


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


@dataclass
class DummyStats:
    slo_violations: int = 0
    slo_lateness_sum: float = 0.0
    decode_next_deadline_by_id: dict[int, float] = None

    def __post_init__(self) -> None:
        if self.decode_next_deadline_by_id is None:
            self.decode_next_deadline_by_id = {}


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
        return {r.id: r for r in self._reqs}


def test_feature_vector_shape_and_buckets() -> None:
    reqs = [
        DummyReq(
            id=1,
            completed=False,
            _is_prefill_complete=False,
            _num_prefill_tokens=3072,
            num_processed_prefill_tokens=0,
            _num_decode_tokens=5000,
            num_processed_decode_tokens=0,
            _prefill_slo_time=0.3,
            _decode_slo_time=0.05,
            arrived_at=0.0,
        ),
        DummyReq(
            id=2,
            completed=False,
            _is_prefill_complete=True,
            _num_prefill_tokens=3072,
            num_processed_prefill_tokens=3072,
            _num_decode_tokens=100,
            num_processed_decode_tokens=0,
            _prefill_slo_time=0.3,
            _decode_slo_time=0.05,
            arrived_at=0.0,
        ),
    ]

    env = DummyEnv(reqs)
    state = DummyState(
        simulator=DummySim(_time=0.1),
        stats=DummyStats(slo_violations=3, slo_lateness_sum=1.5, decode_next_deadline_by_id={2: 0.2}),
    )

    f = extract_features(env, state)
    assert f.shape == (30,)
    assert f.dtype == np.float32

    # decode count=1 => bucket_1_2 should be on, bucket_0 should be off
    decode_bucket_0 = f[23]
    decode_bucket_1_2 = f[24]
    assert float(decode_bucket_0) == 0.0
    assert float(decode_bucket_1_2) == 1.0

    # prefill count=1 => prefill_bucket_1_plus on
    assert float(f[28]) == 0.0
    assert float(f[29]) == 1.0
