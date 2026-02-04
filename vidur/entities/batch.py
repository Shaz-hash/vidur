# --- tidy imports ---
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from vidur.entities.base_entity import BaseEntity
from vidur.entities.request import Request
from vidur.logger import init_logger
from vidur.types.replica_id import ReplicaId
from vidur.utils.snapshot_utils import to_primitive_tree


_SNAP_VERSION_BATCH = 1

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


"""For Simulator Snapshot : """
@dataclass(frozen=True)
class BatchSnapshot:
    __v__: int
    id: int
    replica_id: int
    request_ids: List[int]          # order matters
    num_tokens: List[int]           # order matches request_ids
    scheduled: bool
    completed: bool
    scheduled_at: Optional[float]
    completed_at: Optional[float]
    # Optional debug/metrics fields if you sometimes attach them:
    kv_free_blocks_before: Optional[int] = None
    kv_free_blocks_after: Optional[int] = None
    request_ids_in_batch: Optional[str] = None
    num_requests_not_selected: Optional[int] = None
    num_requests_initiated_not_completed: Optional[int] = None
    num_decode_phase_total: Optional[int] = None
    num_prefill_queue_total: Optional[int] = None
    num_not_initiated_in_queue: Optional[int] = None
    total_requests_batch_start: Optional[int] = None



class Batch(BaseEntity):
    def __init__(
        self,
        replica_id: ReplicaId,
        requests: List[Request],
        num_tokens: List[int],
    ) -> None:
        self._id = Batch.generate_id()
        self._replica_id = replica_id

        self._requests = requests
        self._num_tokens = num_tokens
        self._num_tokens_dict = {
            request.id: num_tokens[i] for i, request in enumerate(self._requests)
        }
        self._total_num_tokens = sum(num_tokens)
        self._num_prefill_tokens = sum(
            [
                (t if not r.is_prefill_complete else 0)
                for r, t in zip(self.requests, self._num_tokens)
            ]
        )
        decode_context_sizes = [
            r.num_processed_tokens for r in self.requests if r.is_prefill_complete
        ]
        self._decode_context_sum = sum(decode_context_sizes)
        self._decode_context_spread = max(decode_context_sizes, default=0) - min(
            decode_context_sizes, default=0
        )
        self._decode_context_iqr = int(
            (
                np.percentile(decode_context_sizes, 75)
                - np.percentile(decode_context_sizes, 25)
            )
            if len(decode_context_sizes) > 0
            else 0
        )

        self._scheduled_at = None
        self._completed_at = None
        self._scheduled = False
        self._completed = False

    @property
    def replica_id(self) -> ReplicaId:
        return self._replica_id

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens

    @property
    def num_tokens_dict(self) -> dict:
        return self._num_tokens_dict

    @property
    def total_num_tokens(self) -> int:
        return self._total_num_tokens

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self.total_num_tokens - self.num_prefill_tokens

    @property
    def num_completed_prefills(self) -> int:
        return len(self.completed_prefills)

    @property
    def decode_context_sum(self) -> int:
        return self._decode_context_sum

    @property
    def decode_context_spread(self) -> int:
        return self._decode_context_spread

    @property
    def decode_context_iqr(self) -> float:
        return self._decode_context_iqr

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def size(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def request_ids(self) -> List[int]:
        return [request.id for request in self._requests]

    @property
    def all_requests_completed(self) -> bool:
        return all([request.completed for request in self._requests])

    def on_schedule(
        self,
        time: float,
    ) -> None:
        self._scheduled_at = time
        self._scheduled = True

        for request in self._requests:
            request.on_batch_schedule(time)

    def on_batch_end(self, time: float):
        self._completed = True
        self._completed_at = time

        for request, num_tokens in zip(self._requests, self._num_tokens):
            request.on_batch_end(time, num_tokens)

    @property
    def preempted_requests(self) -> List[Request]:
        return [request for request in self._requests if request.preempted]

    @property
    def completed_requests(self) -> List[Request]:
        return [request for request in self._requests if request.completed]

    @property
    def completed_prefills(self) -> List[Request]:
        assert self.completed
        return [
            request
            for request in self._requests
            if request.prefill_completed_at == self.completed_at
        ]

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "size": self.size,
            "replica_id": self._replica_id,
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
            "num_prefill_tokens": self.num_prefill_tokens,
            "num_decode_tokens": self.num_decode_tokens,
        }

    # --- Snapshot helpers -------------------------------------------------

    # --- in Batch.snapshot_state ---
    # def snapshot_state(self) -> Dict[str, Any]:
    #     # Safely normalize replica id to a plain int for the snapshot
    #     rid = int(getattr(self._replica_id, "id", self._replica_id))
    #     snap = BatchSnapshot(
    #         __v__=_SNAP_VERSION_BATCH,
    #         id=int(self._id),
    #         replica_id=rid,                               # <-- was int(self._replica_id)
    #         request_ids=[r.id for r in self._requests],
    #         num_tokens=list(self._num_tokens),
    #         scheduled=bool(self._scheduled),
    #         completed=bool(self._completed),
    #         scheduled_at=float(self._scheduled_at) if self._scheduled_at is not None else None,
    #         completed_at=float(self._completed_at) if self._completed_at is not None else None,
    #         kv_free_blocks_before=getattr(self, "kv_free_blocks_before", None),
    #         kv_free_blocks_after=getattr(self, "kv_free_blocks_after", None),
    #         request_ids_in_batch=getattr(self, "request_ids_in_batch", None),
    #         num_requests_not_selected=getattr(self, "num_requests_not_selected", None),
    #         num_requests_initiated_not_completed=getattr(self, "num_requests_initiated_not_completed", None),
    #         num_decode_phase_total=getattr(self, "num_decode_phase_total", None),
    #         num_prefill_queue_total=getattr(self, "num_prefill_queue_total", None),
    #         num_not_initiated_in_queue=getattr(self, "num_not_initiated_in_queue", None),
    #         total_requests_batch_start=getattr(self, "total_requests_batch_start", None),
    #     )
    #     return to_primitive_tree(asdict(snap))

    def snapshot_state(self) -> Dict[str, Any]:
        return {
            "__v__": _SNAP_VERSION_BATCH,
            "id": int(self._id),
            "replica_id": int(getattr(self._replica_id, "id", self._replica_id)),
            "request_ids": [int(r.id) for r in self._requests],
            "num_tokens": [int(x) for x in self._num_tokens],
            "scheduled": bool(self._scheduled),
            "completed": bool(self._completed),
            "scheduled_at": float(self._scheduled_at) if self._scheduled_at is not None else None,
            "completed_at": float(self._completed_at) if self._completed_at is not None else None,

            # Optional debug/metrics annotations (present only if scheduler attached them)
            "kv_free_blocks_before": getattr(self, "kv_free_blocks_before", None),
            "kv_free_blocks_after": getattr(self, "kv_free_blocks_after", None),
            "request_ids_in_batch": getattr(self, "request_ids_in_batch", None),
            "num_requests_not_selected": getattr(self, "num_requests_not_selected", None),
            "num_requests_initiated_not_completed": getattr(self, "num_requests_initiated_not_completed", None),
            "num_decode_phase_total": getattr(self, "num_decode_phase_total", None),
            "num_prefill_queue_total": getattr(self, "num_prefill_queue_total", None),
            "num_not_initiated_in_queue": getattr(self, "num_not_initiated_in_queue", None),
            "total_requests_batch_start": getattr(self, "total_requests_batch_start", None),
        }


    

    # --- in Batch.from_snapshot ---
    @classmethod
    # def from_snapshot(cls, snap: Dict[str, Any], request_lookup: Dict[int, Request]) -> "Batch":
    #     assert int(snap["__v__"]) == _SNAP_VERSION_BATCH, "Batch snapshot version mismatch"

    #     req_ids = list(snap["request_ids"])
    #     num_tokens = list(snap["num_tokens"])
    #     assert len(req_ids) == len(num_tokens), "request_ids/num_tokens length mismatch"

    #     req_objs: List[Request] = []
    #     for rid in req_ids:
    #         assert isinstance(rid, int), "request_ids must be ints"
    #         assert rid in request_lookup, f"unknown request id {rid} in BatchSnapshot"
    #         req_objs.append(request_lookup[rid])

    #     # Re-wrap the stored int back into a ReplicaId for the constructor
    #     replica_id_obj = ReplicaId(int(snap["replica_id"]))   # <-- was int(...)

    #     batch = cls(
    #         replica_id=replica_id_obj,
    #         requests=req_objs,
    #         num_tokens=num_tokens,
    #     )
    #     batch._id = int(snap["id"])
    #     batch._scheduled = bool(snap["scheduled"])
    #     batch._completed = bool(snap["completed"])
    #     batch._scheduled_at = snap.get("scheduled_at", None)
    #     batch._completed_at = snap.get("completed_at", None)

    #     if "kv_free_blocks_before" in snap:
    #         batch.kv_free_blocks_before = snap["kv_free_blocks_before"]
    #     if "kv_free_blocks_after" in snap:
    #         batch.kv_free_blocks_after = snap["kv_free_blocks_after"]
    #     if "request_ids_in_batch" in snap:
    #         batch.request_ids_in_batch = snap["request_ids_in_batch"]
    #     if "num_requests_not_selected" in snap:
    #         batch.num_requests_not_selected = snap["num_requests_not_selected"]
    #     if "num_requests_initiated_not_completed" in snap:
    #         batch.num_requests_initiated_not_completed = snap["num_requests_initiated_not_completed"]
    #     if "num_decode_phase_total" in snap:
    #         batch.num_decode_phase_total = snap["num_decode_phase_total"]
    #     if "num_prefill_queue_total" in snap:
    #         batch.num_prefill_queue_total = snap["num_prefill_queue_total"]
    #     if "num_not_initiated_in_queue" in snap:
    #         batch.num_not_initiated_in_queue = snap["num_not_initiated_in_queue"]
    #     if "total_requests_batch_start" in snap:
    #         batch.total_requests_batch_start = snap["total_requests_batch_start"]

    #     return batch


    @classmethod
    def from_snapshot(cls, snap: Dict[str, Any], request_lookup: Dict[int, Request]) -> "Batch":
        b = cls.__new__(cls)

        b._id = int(snap["id"])
        b._replica_id = ReplicaId(int(snap["replica_id"]))

        b._requests = [request_lookup[int(rid)] for rid in snap["request_ids"]]
        b._num_tokens = [int(x) for x in snap["num_tokens"]]
        b._num_tokens_dict = {r.id: b._num_tokens[i] for i, r in enumerate(b._requests)}

        b._total_num_tokens = int(sum(b._num_tokens))
        b._num_prefill_tokens = int(
            sum((t if not r.is_prefill_complete else 0) for r, t in zip(b._requests, b._num_tokens))
        )

        decode_ctx = [int(r.num_processed_tokens) for r in b._requests if r.is_prefill_complete]
        b._decode_context_sum = int(sum(decode_ctx))
        b._decode_context_spread = int((max(decode_ctx) - min(decode_ctx)) if decode_ctx else 0)
        b._decode_context_iqr = 0  # keep cheap unless you actually use it

        b._scheduled = bool(snap["scheduled"])
        b._completed = bool(snap["completed"])
        b._scheduled_at = snap.get("scheduled_at", None)
        b._completed_at = snap.get("completed_at", None)

        # Optional debug/metrics annotations
        if "kv_free_blocks_before" in snap:
            b.kv_free_blocks_before = snap.get("kv_free_blocks_before", None)
        if "kv_free_blocks_after" in snap:
            b.kv_free_blocks_after = snap.get("kv_free_blocks_after", None)
        if "request_ids_in_batch" in snap:
            b.request_ids_in_batch = snap.get("request_ids_in_batch", None)
        if "num_requests_not_selected" in snap:
            b.num_requests_not_selected = snap.get("num_requests_not_selected", None)
        if "num_requests_initiated_not_completed" in snap:
            b.num_requests_initiated_not_completed = snap.get("num_requests_initiated_not_completed", None)
        if "num_decode_phase_total" in snap:
            b.num_decode_phase_total = snap.get("num_decode_phase_total", None)
        if "num_prefill_queue_total" in snap:
            b.num_prefill_queue_total = snap.get("num_prefill_queue_total", None)
        if "num_not_initiated_in_queue" in snap:
            b.num_not_initiated_in_queue = snap.get("num_not_initiated_in_queue", None)
        if "total_requests_batch_start" in snap:
            b.total_requests_batch_start = snap.get("total_requests_batch_start", None)

        return b



    @classmethod
    def restore_state(cls, snap: Dict[str, Any], request_lookup: Dict[int, Request]) -> "Batch":
        """Kept for API compatibility: construct via from_snapshot()."""
        return cls.from_snapshot(snap, request_lookup)




    # def snapshot_state(self) -> dict:
    #     """Capture a lightweight copy of the mutable batch state."""
    #     state = {key: clone_mutable(value) for key, value in self.__dict__.items()}
    #     state["_requests"] = [request.id for request in self._requests]
    #     return state

    # @classmethod
    # def restore_state(cls, snapshot: dict, request_lookup: dict[int, Request]) -> "Batch":
    #     """Rebuild a ``Batch`` instance from a snapshot and request map."""
    #     request_objs = [request_lookup[req_id] for req_id in snapshot["_requests"]]

    #     batch = cls(snapshot["_replica_id"], request_objs, snapshot["_num_tokens"])
    #     for key, value in snapshot.items():
    #         if key == "_requests":
    #             setattr(batch, key, list(request_objs))
    #         else:
    #             setattr(batch, key, clone_mutable(value))

    #     batch._id = snapshot["_id"]
    #     type(batch)._id = max(type(batch)._id, batch._id)
    #     return batch
