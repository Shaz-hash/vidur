import atexit
import heapq
import json
import random
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple  # add Tuple

import numpy as np
import wandb

from vidur.config import SimulationConfig
from vidur.entities import Cluster
from vidur.entities.batch import Batch
from vidur.entities.batch_stage import BatchStage
from vidur.entities.execution_time import ExecutionTime
from vidur.entities.request import Request
from vidur.events import BaseEvent, RequestArrivalEvent
from vidur.events.batch_end_event import BatchEndEvent
from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent
from vidur.events.batch_stage_end_event import BatchStageEndEvent
from vidur.events.global_schedule_event import GlobalScheduleEvent
from vidur.events.prefill_end_event import PrefillEndEvent
from vidur.events.replica_schedule_event import ReplicaScheduleEvent
from vidur.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
from vidur.events.request_end_event import RequestEndEvent
from vidur.logger import init_logger
from vidur.metrics.cluster_metrics_store import ClusterMetricsStore
from vidur.request_generator import RequestGeneratorRegistry
from vidur.scheduler import GlobalSchedulerRegistry
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.replica_id import ReplicaId
from vidur.utils.json_encoder import JsonEncoder
from vidur.utils.snapshot_utils import clone_mutable


# near the top
_SNAP_VERSION_SIM = 1

logger = init_logger(__name__)


@dataclass
class SimulatorSnapshot:
    __v__: int
    time: float
    time_limit_reached: bool
    event_queue: List[dict]
    # scheduler_state can be a dataclass (e.g., VLLMV1ReplicaSchedulerSnapshot),
    # so keep it as Any to avoid forcing dicts here.
    scheduler_state: Any
    request_generator_state: dict
    entity_counters: Dict[str, int]
    base_event_counter: int
    request_states: Dict[int, dict]
    batch_states: Dict[int, dict]
    batch_stage_states: Dict[int, dict]
    python_random_state: tuple
    numpy_random_state: tuple  # we’ll make this JSON-safe below if you persist



def _encode_numpy_state(state: tuple) -> tuple:
    # ('MT19937', ndarray, int, int, float) -> ('MT19937', list, int, int, float)
    algo, keys, pos, has_gauss, cached_gauss = state
    try:
        keys_list = keys.tolist()  # ndarray -> list
    except AttributeError:
        keys_list = keys
    return (algo, keys_list, pos, has_gauss, cached_gauss)

def _decode_numpy_state(state: tuple) -> tuple:
    import numpy as _np
    algo, keys_list, pos, has_gauss, cached_gauss = state
    keys_arr = _np.array(keys_list, dtype=_np.uint32)
    return (algo, keys_arr, pos, has_gauss, cached_gauss)





class Simulator:
    def __init__(self, config: SimulationConfig, register_atexit: bool = True) -> None:
        self._config: SimulationConfig = config

        self._time = 0
        self._time_limit_reached = False
        self._time_limit = self._config.time_limit
        if not self._time_limit:
            self._time_limit = float("inf")

        self._event_queue: List[BaseEvent] = []

        self._event_trace = []
        self._event_chrome_trace = []

        self._cluster = Cluster(
            cluster_config=self._config.cluster_config,
            metrics_config=self._config.metrics_config,
        )
        self._cluster_metric_store = ClusterMetricsStore(
            simulation_config=self._config,
            replicas=self._cluster.replicas,
        )
        self._request_generator = RequestGeneratorRegistry.get(
            self._config.request_generator_config.get_type(),
            self._config.request_generator_config,
        )
        self._scheduler = GlobalSchedulerRegistry.get(
            self._config.cluster_config.global_scheduler_config.get_type(),
            self._config,
            self._cluster.replicas,
        )

        self._init_event_queue()
        if register_atexit:
            atexit.register(self._write_output)

    def run(self) -> None:
        logger.info(f"Starting simulation with cluster: {self._cluster}")

        while not self._time_limit_reached and (
            self._event_queue
            or self._request_generator.get_next_request_arrival_time() is not None
        ):
            next_event_time = self._event_queue[0]._time if self._event_queue else None
            next_request_arrival_time = (
                self._request_generator.get_next_request_arrival_time()
            )
            if (next_request_arrival_time is not None) and (
                next_event_time is None or next_request_arrival_time <= next_event_time
            ):
                self._add_event(
                    RequestArrivalEvent(
                        next_request_arrival_time,
                        self._request_generator.get_next_request(),
                    )
                )
                continue

            event = self._event_queue[0]
            heapq.heappop(self._event_queue)
            self._set_time(event._time)
            new_events = event.handle_event(self._scheduler, self._cluster_metric_store)
            self._add_events(new_events)

            if self._config.metrics_config.write_json_trace:
                self._event_trace.append(event.to_dict())

            if self._config.metrics_config.enable_chrome_trace:
                chrome_trace = event.to_chrome_trace()
                if chrome_trace:
                    self._event_chrome_trace.append(chrome_trace)

        assert self._scheduler.is_empty() or self._time_limit_reached

        logger.info(f"Simulation ended at: {self._time}s")

    def _write_output(self) -> None:
        logger.info("Writing output")

        self._cluster_metric_store.plot(self._time)
        logger.info("Metrics written")

        if self._config.metrics_config.write_json_trace:
            self._write_event_trace()
            logger.info("Json event trace written")

        if self._config.metrics_config.enable_chrome_trace:
            self._write_chrome_trace()
            logger.info("Chrome event trace written")

    def _add_event(self, event: BaseEvent) -> None:
        heapq.heappush(self._event_queue, event)

    def _add_events(self, events: List[BaseEvent]) -> None:
        for event in events:
            self._add_event(event)

    def _init_event_queue(self) -> None:
        first_request = self._request_generator.get_next_request()
        if first_request:
            self._add_event(
                RequestArrivalEvent(first_request.arrived_at, first_request)
            )

    def _set_time(self, time: float) -> None:
        self._time = time
        if self._time > self._time_limit:
            logger.info(
                f"Time limit reached: {self._time_limit}s terminating the simulation."
            )
            self._time_limit_reached = True

    def _write_event_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/event_trace.json"
        with open(trace_file, "w") as f:
            json.dump(self._event_trace, f, cls=JsonEncoder)

    def _write_chrome_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/chrome_trace.json"

        chrome_trace = {"traceEvents": self._event_chrome_trace}

        with open(trace_file, "w") as f:
            json.dump(chrome_trace, f, cls=JsonEncoder)

        if wandb.run:
            zip_file_path = f"{self._config.output_dir}/chrome_trace.zip"
            with zipfile.ZipFile(
                zip_file_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                zf.writestr(
                    "chrome_trace.json",
                    json.dumps(chrome_trace, cls=JsonEncoder),
                )
            wandb.save(zip_file_path, policy="now")

    def snapshot_state(self) -> SimulatorSnapshot:
        request_states: Dict[int, dict] = {}
        batch_states: Dict[int, dict] = {}
        batch_stage_states: Dict[int, dict] = {}

        def track_request(request: Request) -> None:
            if request.id not in request_states:
                request_states[request.id] = request.snapshot_state()

        def track_batch(batch: Batch) -> None:
            if batch.id not in batch_states:
                batch_states[batch.id] = batch.snapshot_state()
                for req in batch.requests:
                    track_request(req)

        def track_batch_stage(batch_stage: BatchStage) -> None:
            if batch_stage.id not in batch_stage_states:
                batch_stage_states[batch_stage.id] = batch_stage.snapshot_state()
                for req in batch_stage.requests:
                    track_request(req)

        scheduler_snapshot = clone_mutable(self._scheduler.snapshot_state())

        # handle dataclass or dict uniformly
        if hasattr(scheduler_snapshot, "request_states"):
            req_states = getattr(scheduler_snapshot, "request_states")
        elif isinstance(scheduler_snapshot, dict):
            req_states = scheduler_snapshot.get("request_states", {})
        else:
            req_states = {}

        for req_id, state in getattr(req_states, "items", lambda: [])():
            request_states.setdefault(req_id, clone_mutable(state))


        # for req_id, state in scheduler_snapshot.get("request_states", {}).items():
        #     request_states.setdefault(req_id, clone_mutable(state))

        # Capture live requests/batches/stages from schedulers.
        for replica_scheduler in self._scheduler._replica_schedulers.values():
            if hasattr(replica_scheduler, "_requests"):
                for request in replica_scheduler._requests.values():
                    track_request(request)
            if hasattr(replica_scheduler, "_running"):
                for request in getattr(replica_scheduler, "_running", []):
                    track_request(request)
            if hasattr(replica_scheduler, "_waiting_queue"):
                waiting_queue = getattr(replica_scheduler, "_waiting_queue")
                if hasattr(waiting_queue, "to_list"):
                    for request in waiting_queue.to_list():
                        track_request(request)
                elif hasattr(waiting_queue, "__iter__"):
                    for item in waiting_queue:
                        if isinstance(item, Request):
                            track_request(item)
                        elif hasattr(item, "request"):
                            track_request(item.request)
            if hasattr(replica_scheduler, "_replica_stage_schedulers"):
                for stage_scheduler in replica_scheduler._replica_stage_schedulers.values():
                    for batch in getattr(stage_scheduler, "_batch_queue", []):
                        track_batch(batch)

        # Capture pending requests in global queue.
        for request in self._scheduler._request_queue:
            track_request(request)

        event_snapshots: List[dict] = []
        for event in self._event_queue:
            event_snapshots.append(
                self._snapshot_event(event, track_request, track_batch, track_batch_stage)
            )

        # Capture generator queue requests.
        if hasattr(self._request_generator, "requests"):
            for request in getattr(self._request_generator, "requests"):
                track_request(request)

        request_generator_state = clone_mutable(
            self._request_generator.snapshot_state()
        )

        entity_counters = {
            "Request": Request._id,
            "Batch": Batch._id,
            "BatchStage": BatchStage._id,
            "ExecutionTime": ExecutionTime._id,
        }

        snapshot = SimulatorSnapshot(
            __v__=_SNAP_VERSION_SIM,
            time=self._time,
            time_limit_reached=self._time_limit_reached,
            event_queue=event_snapshots,
            scheduler_state=scheduler_snapshot,
            request_generator_state=request_generator_state,
            entity_counters=entity_counters,
            base_event_counter=BaseEvent._id,
            request_states={k: clone_mutable(v) for k, v in request_states.items()},
            batch_states={k: clone_mutable(v) for k, v in batch_states.items()},
            batch_stage_states={k: clone_mutable(v) for k, v in batch_stage_states.items()},
            python_random_state=random.getstate(),
            numpy_random_state=_encode_numpy_state(np.random.get_state()),
        )
        return snapshot

    def restore_state(self, snapshot: SimulatorSnapshot) -> None:
        assert int(snapshot.__v__) == _SNAP_VERSION_SIM, "Simulator snapshot version mismatch"

        self._time = float(snapshot.time)
        self._time_limit_reached = bool(snapshot.time_limit_reached)

        random.setstate(snapshot.python_random_state)
        np.random.set_state(_decode_numpy_state(snapshot.numpy_random_state))

        # Restore entity counters first
        Request._id = snapshot.entity_counters.get("Request", Request._id)
        Batch._id = snapshot.entity_counters.get("Batch", Batch._id)
        BatchStage._id = snapshot.entity_counters.get("BatchStage", BatchStage._id)
        ExecutionTime._id = snapshot.entity_counters.get("ExecutionTime", ExecutionTime._id)
        BaseEvent._id = snapshot.base_event_counter

        # Rebuild objects
        request_lookup: Dict[int, Request] = {}
        for request_id, state in snapshot.request_states.items():
            req = Request.__new__(Request)
            req.restore_state(state)
            # Optional: assert id matches key
            assert req._id == int(request_id), "Request id mismatch during restore"
            request_lookup[int(request_id)] = req

        batch_lookup: Dict[int, Batch] = {}
        for batch_id, state in snapshot.batch_states.items():
            batch = Batch.restore_state(state, request_lookup)
            assert batch._id == int(batch_id), "Batch id mismatch during restore"
            batch_lookup[int(batch_id)] = batch

        batch_stage_lookup: Dict[int, BatchStage] = {}
        for stage_id, state in snapshot.batch_stage_states.items():
            stg = BatchStage.restore_state(state, request_lookup)
            assert stg._id == int(stage_id), "BatchStage id mismatch during restore"
            batch_stage_lookup[int(stage_id)] = stg

        # Request generator + scheduler
        self._request_generator.restore_state(snapshot.request_generator_state, request_lookup)
        self._scheduler.restore_state(snapshot.scheduler_state, request_lookup, batch_lookup)

        # Sync replica metric store aliases with scheduler aliases so restored
        # events that carry legacy replica ids still resolve correctly.
        replica_alias_map = getattr(self._scheduler, "_replica_alias", None)
        register_alias_fn = getattr(self._cluster_metric_store, "register_replica_alias", None)
        if isinstance(replica_alias_map, dict) and callable(register_alias_fn):
            for alias_key, scheduler in replica_alias_map.items():
                replica_identifier = getattr(
                    scheduler,
                    "replica_id",
                    getattr(scheduler, "_replica_id", None),
                )
                try:
                    register_alias_fn(alias_key, replica_identifier)
                except Exception:
                    # Fall back to registering without explicit replica id.
                    register_alias_fn(alias_key, None)

        # Events
        self._event_queue = [
            self._restore_event(es, request_lookup, batch_lookup, batch_stage_lookup)
            for es in snapshot.event_queue
        ]
        heapq.heapify(self._event_queue)

        # Optional: sanity checks
        # - all batch.request ids exist in request_lookup
        for b in batch_lookup.values():
            for r in b.requests:
                assert r.id in request_lookup, "Batch references unknown Request"

        # - event queue is consistent
        assert all(hasattr(e, "_priority_number") for e in self._event_queue)


    def fork(self) -> "Simulator":
        snapshot = self.snapshot_state()
        forked = Simulator(self._config, register_atexit=False)
        forked.restore_state(snapshot)
        return forked

    def _snapshot_event(
        self,
        event: BaseEvent,
        track_request,
        track_batch,
        track_batch_stage,
    ) -> dict:
        event_type = event.__class__.__name__
        data: Dict[str, Any] = {"time": event.time}

        if event_type == "RequestArrivalEvent":
            data["request_id"] = event._request.id  # type: ignore[attr-defined]
            track_request(event._request)  # type: ignore[attr-defined]
        elif event_type == "GlobalScheduleEvent":
            pass
        elif event_type == "ReplicaScheduleEvent":
            data["replica_id"] = event._replica_id.id  # type: ignore[attr-defined]
        elif event_type == "BatchStageArrivalEvent":
            batch = event._batch  # type: ignore[attr-defined]
            data["replica_id"] = event._replica_id.id  # type: ignore[attr-defined]
            data["stage_id"] = event._stage_id  # type: ignore[attr-defined]
            data["batch_id"] = batch.id
            track_batch(batch)
        elif event_type == "ReplicaStageScheduleEvent":
            data["replica_id"] = event._replica_id.id  # type: ignore[attr-defined]
            data["stage_id"] = event._stage_id  # type: ignore[attr-defined]
        elif event_type == "BatchStageEndEvent":
            batch = event._batch  # type: ignore[attr-defined]
            batch_stage = event._batch_stage  # type: ignore[attr-defined]
            data["replica_id"] = event._replica_id.id  # type: ignore[attr-defined]
            data["stage_id"] = event._stage_id  # type: ignore[attr-defined]
            data["is_last_stage"] = event._is_last_stage  # type: ignore[attr-defined]
            data["batch_id"] = batch.id
            data["batch_stage_id"] = batch_stage.id
            track_batch(batch)
            track_batch_stage(batch_stage)
        elif event_type == "BatchEndEvent":
            batch = event._batch  # type: ignore[attr-defined]
            data["replica_id"] = event._replica_id.id  # type: ignore[attr-defined]
            data["batch_id"] = batch.id
            track_batch(batch)
        elif event_type == "PrefillEndEvent":
            data["request_id"] = event._request.id  # type: ignore[attr-defined]
            track_request(event._request)  # type: ignore[attr-defined]
        elif event_type == "RequestEndEvent":
            data["request_id"] = event._request.id  # type: ignore[attr-defined]
            track_request(event._request)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unsupported event type for snapshot: {event_type}")

        return {"type": event_type, "data": data, "event_id": event.id}

    def _restore_event(
        self,
        snapshot: dict,
        request_lookup: Dict[int, Request],
        batch_lookup: Dict[int, Batch],
        batch_stage_lookup: Dict[int, BatchStage],
    ) -> BaseEvent:
        event_type = snapshot["type"]
        data = snapshot["data"]
        event_id = snapshot["event_id"]

        if event_type == "RequestArrivalEvent":
            event = RequestArrivalEvent(data["time"], request_lookup[data["request_id"]])
        elif event_type == "GlobalScheduleEvent":
            event = GlobalScheduleEvent(data["time"])
        elif event_type == "ReplicaScheduleEvent":
            event = ReplicaScheduleEvent(data["time"], ReplicaId(data["replica_id"]))
        elif event_type == "BatchStageArrivalEvent":
            event = BatchStageArrivalEvent(
                data["time"],
                ReplicaId(data["replica_id"]),
                data["stage_id"],
                batch_lookup[data["batch_id"]],
            )
        elif event_type == "ReplicaStageScheduleEvent":
            event = ReplicaStageScheduleEvent(
                data["time"], ReplicaId(data["replica_id"]), data["stage_id"]
            )
        elif event_type == "BatchStageEndEvent":
            event = BatchStageEndEvent(
                data["time"],
                ReplicaId(data["replica_id"]),
                data["stage_id"],
                data["is_last_stage"],
                batch_lookup[data["batch_id"]],
                batch_stage_lookup[data["batch_stage_id"]],
            )
        elif event_type == "BatchEndEvent":
            event = BatchEndEvent(
                data["time"],
                ReplicaId(data["replica_id"]),
                batch_lookup[data["batch_id"]],
            )
        elif event_type == "PrefillEndEvent":
            event = PrefillEndEvent(data["time"], request_lookup[data["request_id"]])
        elif event_type == "RequestEndEvent":
            event = RequestEndEvent(data["time"], request_lookup[data["request_id"]])
        else:
            raise ValueError(f"Unsupported event type during restore: {event_type}")

        event._id = event_id
        event._priority_number = event._get_priority_number()
        return event
