# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from vidur.config import SimulationConfig
from vidur.entities.batch import Batch
from vidur.entities.batch_stage import BatchStage
from vidur.entities.request import Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.metrics.noop_metrics_store import NoOpClusterMetricsStore
from vidur.types.replica_id import ReplicaId

_SNAP_VERSION_VSIM = 1


def _encode_numpy_state(state: tuple) -> tuple:
    algo, keys, pos, has_gauss, cached_gauss = state
    try:
        keys_list = keys.tolist()
    except AttributeError:
        keys_list = keys
    return (algo, keys_list, pos, has_gauss, cached_gauss)


def _decode_numpy_state(state: tuple) -> tuple:
    algo, keys_list, pos, has_gauss, cached_gauss = state
    keys_arr = np.array(keys_list, dtype=np.uint32)
    return (algo, keys_arr, pos, has_gauss, cached_gauss)


class _VirtualWaitingQueue:
    def __init__(self) -> None:
        self._request_queue: deque[Request] = deque()
        self._num_prefill_tokens = 0

    def clear(self) -> None:
        self._request_queue.clear()
        self._num_prefill_tokens = 0

    def push(self, request: Request) -> None:
        self._request_queue.append(request)
        self._num_prefill_tokens += int(getattr(request, "num_prefill_tokens", 0))

    def pop(self) -> Request:
        req = self._request_queue.popleft()
        self._num_prefill_tokens -= int(getattr(req, "num_prefill_tokens", 0))
        return req

    def peek(self) -> Request:
        return self._request_queue[0]

    def to_list(self) -> List[Request]:
        return list(self._request_queue)

    def extend(self, requests: Iterable[Request]) -> None:
        for req in requests:
            self.push(req)

    def __len__(self) -> int:
        return len(self._request_queue)


@dataclass
class _VirtualReplicaSchedulerConfig:
    chunk_size: int


class _VirtualReplicaScheduler:
    def __init__(self, replica_id: ReplicaId, chunk_size: int) -> None:
        self._replica_id = replica_id
        self.replica_id = replica_id
        self._config = _VirtualReplicaSchedulerConfig(chunk_size=int(max(1, chunk_size)))
        self._waiting_queue = _VirtualWaitingQueue()
        self._running: List[Request] = []
        self._requests: Dict[int, Request] = {}
        self._token_budget_overrides: Dict[int, int] = {}

    def add_request(self, request: Request) -> None:
        self._requests[int(request.id)] = request
        self._waiting_queue.push(request)

    def set_token_budget_overrides(self, overrides: Dict[int, int]) -> None:
        self._token_budget_overrides = {
            int(rid): max(0, int(tokens)) for rid, tokens in dict(overrides).items()
        }


class _VirtualScheduler:
    def __init__(self, replica_scheduler: _VirtualReplicaScheduler) -> None:
        self._request_queue: List[Request] = []
        self._replica_schedulers: Dict[ReplicaId, _VirtualReplicaScheduler] = {
            replica_scheduler.replica_id: replica_scheduler
        }

    def get_replica_scheduler(self, replica_id: Any) -> _VirtualReplicaScheduler:
        if replica_id in self._replica_schedulers:
            return self._replica_schedulers[replica_id]
        return next(iter(self._replica_schedulers.values()))

    def is_empty(self) -> bool:
        rs = next(iter(self._replica_schedulers.values()))
        return len(rs._waiting_queue) == 0 and len(rs._running) == 0

    def add_request(self, request: Request) -> None:
        rs = next(iter(self._replica_schedulers.values()))
        rs.add_request(request)


class VirtualSimulator:
    """
    Lightweight simulator state container for MCTS rollouts.
    It intentionally keeps the same core fields used by mcts/infer:
    `_time`, `_scheduler`, `snapshot_state`, `restore_state`, `fork`.
    """

    def __init__(
        self,
        config: SimulationConfig,
        register_atexit: bool = False,
        execution_time_predictor: Any = None,
    ) -> None:
        del register_atexit
        self._config = config
        self._time = 0.0
        self._event_queue: List[Any] = []
        self._cluster_metric_store = NoOpClusterMetricsStore()

        # if execution_time_predictor is None:
        #     execution_time_predictor = ExecutionTimePredictorRegistry.get(
        #         config.execution_time_predictor_config.get_type(),
        #         predictor_config=config.execution_time_predictor_config,
        #     )
        if execution_time_predictor is None:
            execution_time_predictor = ExecutionTimePredictorRegistry.get(
                config.execution_time_predictor_config.get_type(),
                predictor_config=config.execution_time_predictor_config,
                replica_config=config.cluster_config.replica_config,
                cache_config=config.cluster_config.cache_config,
            )

        if execution_time_predictor is None:
            raise ValueError("VirtualSimulator requires a valid execution_time_predictor")
        self._execution_time_predictor = execution_time_predictor

        n_replicas = int(getattr(config.cluster_config, "num_replicas", 1))
        n_stages = int(getattr(config.cluster_config.replica_config, "num_pipeline_stages", 1))
        if n_replicas != 1:
            raise ValueError(
                f"VirtualSimulator V1 supports single-replica only, got num_replicas={n_replicas}"
            )
        if n_stages != 1:
            raise ValueError(
                f"VirtualSimulator V1 supports single-stage only, got num_pipeline_stages={n_stages}"
            )

        replica_sched_cfg = getattr(config.cluster_config, "replica_scheduler_config", None)
        chunk_size = int(getattr(replica_sched_cfg, "chunk_size", 512) or 512)
        self._replica_id = ReplicaId(0)
        self._scheduler = _VirtualScheduler(_VirtualReplicaScheduler(self._replica_id, chunk_size))

        self._restore_pool_requests_by_id: dict[int, Request] = {}
        self._restore_pool_requests_free: list[Request] = []

    @property
    def replica_id(self) -> ReplicaId:
        return self._replica_id

    def _set_time(self, time: float) -> None:
        self._time = float(time)

    def _add_event(self, event: object) -> None:
        # Minimal compatibility: handle RequestArrivalEvent-like payloads.
        req = getattr(event, "_request", None)
        if isinstance(req, Request):
            req.assign_replica(self._replica_id)
            self._scheduler.add_request(req)
            return
        self._event_queue.append(event)

    def _primary_replica_scheduler(self) -> _VirtualReplicaScheduler:
        return next(iter(self._scheduler._replica_schedulers.values()))

    def snapshot_state(self) -> Dict[str, Any]:
        rs = self._primary_replica_scheduler()
        request_states: Dict[int, dict] = {
            int(rid): req.snapshot_state() for rid, req in rs._requests.items()
        }
        waiting_ids = [int(req.id) for req in rs._waiting_queue.to_list()]
        running_ids = [int(req.id) for req in rs._running]

        py_state = None
        np_state = None
        if bool(getattr(self._config, "snapshot_rng_state", True)):
            py_state = random.getstate()
            np_state = _encode_numpy_state(np.random.get_state())

        return {
            "__v__": _SNAP_VERSION_VSIM,
            "time": float(self._time),
            "request_states": request_states,
            "waiting_ids": waiting_ids,
            "running_ids": running_ids,
            "overrides": {int(k): int(v) for k, v in rs._token_budget_overrides.items()},
            "entity_counters": {
                "Request": int(getattr(Request, "_id", -1)),
                "Batch": int(getattr(Batch, "_id", -1)),
                "BatchStage": int(getattr(BatchStage, "_id", -1)),
            },
            "python_random_state": py_state,
            "numpy_random_state": np_state,
            "event_queue": [],
        }

    def restore_state(self, snapshot: Dict[str, Any]) -> None:

        mode = str(snapshot.get("__mode__", "full"))
        if mode == "mcts_fast":
            self.restore_state_fast(snapshot)
            return        

        if int(snapshot.get("__v__", -1)) != _SNAP_VERSION_VSIM:
            raise ValueError("VirtualSimulator snapshot version mismatch")

        rs = self._primary_replica_scheduler()
        request_states: Dict[int, dict] = {
            int(rid): state for rid, state in dict(snapshot.get("request_states", {})).items()
        }
        live_ids = set(request_states.keys())

        for rid in list(self._restore_pool_requests_by_id.keys()):
            if rid not in live_ids:
                self._restore_pool_requests_free.append(self._restore_pool_requests_by_id.pop(rid))

        request_lookup: Dict[int, Request] = {}
        for rid, state in request_states.items():
            if rid in self._restore_pool_requests_by_id:
                req = self._restore_pool_requests_by_id[rid]
                req.restore_state(state)
            elif self._restore_pool_requests_free:
                req = self._restore_pool_requests_free.pop()
                req.restore_state(state)
                self._restore_pool_requests_by_id[rid] = req
            else:
                req = Request.from_snapshot(state)
                self._restore_pool_requests_by_id[rid] = req
            request_lookup[rid] = req

        rs._requests = request_lookup

        rs._waiting_queue.clear()
        for rid in snapshot.get("waiting_ids", []):
            req = request_lookup.get(int(rid))
            if req is not None:
                rs._waiting_queue.push(req)

        rs._running = []
        for rid in snapshot.get("running_ids", []):
            req = request_lookup.get(int(rid))
            if req is not None:
                rs._running.append(req)

        rs._token_budget_overrides = {
            int(k): int(v) for k, v in dict(snapshot.get("overrides", {})).items()
        }

        self._scheduler._request_queue = []
        self._event_queue = list(snapshot.get("event_queue", []))
        self._time = float(snapshot.get("time", 0.0))

        counters = dict(snapshot.get("entity_counters", {}))
        if "Request" in counters:
            Request._id = int(counters["Request"])
        if "Batch" in counters:
            Batch._id = int(counters["Batch"])
        if "BatchStage" in counters:
            BatchStage._id = int(counters["BatchStage"])

        if bool(getattr(self._config, "snapshot_rng_state", True)):
            py_state = snapshot.get("python_random_state", None)
            np_state = snapshot.get("numpy_random_state", None)
            if py_state is not None:
                random.setstate(py_state)
            if np_state is not None:
                np.random.set_state(_decode_numpy_state(np_state))

    def fork(self, flag: bool | None = None) -> "VirtualSimulator":
        del flag
        child = VirtualSimulator(
            self._config,
            register_atexit=False,
            execution_time_predictor=self._execution_time_predictor,
        )
        child.restore_state(self.snapshot_state())
        return child



# inside class VirtualSimulator

    def _restore_request_lookup_from_states(self, request_states: Dict[int, dict]) -> Dict[int, Request]:
        live_ids = set(request_states.keys())

        for rid in list(self._restore_pool_requests_by_id.keys()):
            if rid not in live_ids:
                self._restore_pool_requests_free.append(self._restore_pool_requests_by_id.pop(rid))

        request_lookup: Dict[int, Request] = {}
        for rid, state in request_states.items():
            if rid in self._restore_pool_requests_by_id:
                req = self._restore_pool_requests_by_id[rid]
                req.restore_state(state)
            elif self._restore_pool_requests_free:
                req = self._restore_pool_requests_free.pop()
                req.restore_state(state)
                self._restore_pool_requests_by_id[rid] = req
            else:
                req = Request.from_snapshot(state)
                self._restore_pool_requests_by_id[rid] = req
            request_lookup[rid] = req

        return request_lookup


    def snapshot_state_fast(self) -> Dict[str, Any]:
        rs = self._primary_replica_scheduler()
        request_states: Dict[int, dict] = {
            int(rid): req.snapshot_state() for rid, req in rs._requests.items()
        }
        return {
            "__v__": _SNAP_VERSION_VSIM,
            "__mode__": "mcts_fast",
            "time": float(self._time),
            "request_states": request_states,
            "overrides": {int(k): int(v) for k, v in rs._token_budget_overrides.items()},
        }


    def restore_state_fast(self, snapshot: Dict[str, Any]) -> None:
        if int(snapshot.get("__v__", -1)) != _SNAP_VERSION_VSIM:
            raise ValueError("VirtualSimulator snapshot version mismatch")

        rs = self._primary_replica_scheduler()
        request_states: Dict[int, dict] = {
            int(rid): state for rid, state in dict(snapshot.get("request_states", {})).items()
        }

        rs._requests = self._restore_request_lookup_from_states(request_states)

        # Fast mode: no waiting/running reconstruction needed for virtual MCTS path
        rs._waiting_queue.clear()
        rs._running = []
        rs._token_budget_overrides = {
            int(k): int(v) for k, v in dict(snapshot.get("overrides", {})).items()
        }

        self._scheduler._request_queue = []
        self._event_queue = []
        self._time = float(snapshot.get("time", 0.0))
