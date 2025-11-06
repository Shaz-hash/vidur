from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

from vidur.entities.base_entity import BaseEntity
from vidur.entities.execution_time import ExecutionTime
from vidur.entities.request import Request
from vidur.logger import init_logger
from vidur.types.replica_id import ReplicaId


logger = init_logger(__name__)


_SNAP_VERSION_BATCH_STAGE = 1


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper



@dataclass(frozen=True)
class BatchStageSnapshot:
    __v__: int
    id: int
    batch_id: int
    replica_id: int
    stage_id: int
    # ExecutionTime encoded as a plain dict
    execution_time: Dict[str, Any]
    request_ids: List[int]      # order matters
    num_tokens: List[int]       # aligns with request_ids
    scheduled: bool
    scheduled_at: Optional[float]
    completed_at: Optional[float]


class BatchStage(BaseEntity):
    def __init__(
        self,
        batch_id: int,
        replica_id: ReplicaId,
        stage_id: int,
        execution_time: ExecutionTime,
        requests: List[Request],
        # num_tokens: List[Request],
        num_tokens: List[int],  # <-- FIXED TYPE
    ) -> None:
        self._id = BatchStage.generate_id()

        self._requests = requests
        self._num_tokens = num_tokens
        self._batch_id = batch_id
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._execution_time = execution_time

        self._total_execution_time = self._execution_time.total_time
        self._model_execution_time = self._execution_time.model_time

        self._scheduled_at = None
        self._completed_at = None
        self._scheduled = False

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_scheduled
    def completed_at(self) -> float:
        return self._completed_at

    @property
    def execution_time(self) -> float:
        return self._total_execution_time

    @property
    def model_execution_time(self) -> float:
        return self._model_execution_time

    @property
    def replica_id(self) -> ReplicaId:
        return self._replica_id

    @property
    def stage_id(self) -> int:
        return self._stage_id

    @property
    def request_ids(self) -> List[int]:
        return [request.id for request in self._requests]

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def size(self) -> int:
        return len(self._requests)

    def on_schedule(
        self,
        time: float,
    ) -> None:
        self._scheduled_at = time
        self._scheduled = True

        for request in self._requests:
            request.on_batch_stage_schedule(time)

    def on_stage_end(
        self,
        time: float,
    ) -> None:
        assert (
            time == self._scheduled_at + self._total_execution_time
        ), f"{time} != {self._scheduled_at} + {self._total_execution_time}"

        self._completed_at = time

        for request in self._requests:
            request.on_batch_stage_end(
                time, self._total_execution_time, self._model_execution_time
            )

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "size": self.size,
            "execution_time": self._execution_time.to_dict(),
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "replica_id": self._replica_id,
            "batch_id": self._batch_id,
            "stage_id": self._stage_id,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
        }

    def to_chrome_trace(self, time: int) -> dict:
        return {
            "name": f"{self.request_ids}",
            "ph": "X",
            "ts": (time - self._total_execution_time) * 1e6,
            "dur": self._total_execution_time * 1e6,
            "pid": str(self._replica_id),
            "tid": self._stage_id,
            "args": {
                "batch_id": self._batch_id,
                "batch_size": self.size,
                "request_ids": self.request_ids,
                "num_tokens": self._num_tokens,
                "execution_time": self._execution_time.to_dict(),
                "requests": [request.to_dict() for request in self._requests],
            },
        }

    # --- Snapshot helpers -------------------------------------------------
    def snapshot_state(self) -> Dict[str, Any]:
        """Return a JSON-friendly, minimal snapshot of this stage."""
        snap = BatchStageSnapshot(
            __v__=_SNAP_VERSION_BATCH_STAGE,
            id=int(self._id),
            batch_id=int(self._batch_id),
            replica_id=int(getattr(self._replica_id, "id", self._replica_id)),
            stage_id=int(self._stage_id),
            execution_time=dict(self._execution_time.__dict__),
            request_ids=[r.id for r in self._requests],
            num_tokens=list(self._num_tokens),
            scheduled=bool(self._scheduled),
            scheduled_at=float(self._scheduled_at) if self._scheduled_at is not None else None,
            completed_at=float(self._completed_at) if self._completed_at is not None else None,
        )
        # Already primitives via .to_dict and lists of ints
        return asdict(snap)

    @classmethod
    def from_snapshot(cls, snap: Dict[str, Any], request_lookup: Dict[int, Request]) -> "BatchStage":
        assert int(snap["__v__"]) == _SNAP_VERSION_BATCH_STAGE, "BatchStage snapshot version mismatch"
        req_ids = list(snap["request_ids"])
        toks    = list(snap["num_tokens"])
        assert len(req_ids) == len(toks), "BatchStage snapshot: request_ids/num_tokens length mismatch"

        # Rebuild ExecutionTime without calling its constructor (to avoid extra logic).
        et_dict = dict(snap["execution_time"])
        et = ExecutionTime.__new__(ExecutionTime)
        et.__dict__ = et_dict
        # keep global id counter monotonic if ExecutionTime uses BaseEntity
        if hasattr(ExecutionTime, "_id") and hasattr(et, "_id"):
            ExecutionTime._id = max(ExecutionTime._id, et._id)

        req_objs = [request_lookup[rid] for rid in snap["request_ids"]]

        obj = cls(
            batch_id=int(snap["batch_id"]),
            replica_id=int(snap["replica_id"]),
            stage_id=int(snap["stage_id"]),
            execution_time=et,
            requests=req_objs,
            num_tokens=list(snap["num_tokens"]),
        )
        # identity + flags/timestamps (don’t call lifecycle hooks)
        obj._id = int(snap["id"])
        type(obj)._id = max(type(obj)._id, obj._id)

        obj._scheduled = bool(snap["scheduled"])
        obj._scheduled_at = snap.get("scheduled_at", None)
        obj._completed_at = snap.get("completed_at", None)

        # Derived fields from execution_time
        obj._total_execution_time = et.total_time
        obj._model_execution_time = et.model_time
        return obj

    @classmethod
    def restore_state(cls, snap: Dict[str, Any], request_lookup: Dict[int, Request]) -> "BatchStage":
        """Kept for API symmetry with other entities."""
        return cls.from_snapshot(snap, request_lookup)
