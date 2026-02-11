from typing import List, Optional, Tuple

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger
from vidur.types.replica_id import ReplicaId
# from vidur.utils.snapshot_utils import clone_mutable


from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
from vidur.utils.snapshot_utils import to_primitive_tree  # the primitive normalizer we discussed

_SNAP_VERSION_REQ = 1  # bump if you ever change fields/semantics


@dataclass(frozen=True)
class RequestSnapshot:
    __v__: int
    id: int
    arrived_at: float
    queued_at: float
    replica_id: Optional[int]

    # size + progress
    num_prefill_tokens: int
    num_prefill_tokens_cached: int
    num_decode_tokens: int
    num_processed_tokens: int

    # prefix caching aux
    block_hash_ids: Optional[List[int]]
    block_size: Optional[int]

    # flags
    scheduled: bool
    preempted: bool
    completed: bool
    is_prefill_complete: bool

    # counts
    num_restarts: int

    # timing / metrics
    scheduled_at: float
    preempted_time: float
    completed_at: float
    prefill_completed_at: float
    scheduling_delay: float
    execution_time: float
    model_execution_time: float
    latest_stage_scheduled_at: float
    latest_stage_completed_at: float
    latest_iteration_scheduled_at: float
    latest_iteration_completed_at: float
    latest_iteration_scheduling_delay: float

    # SLO / session
    prefill_slo_time: Optional[float]
    decode_slo_time: float
    completion_slo_time: float
    session_id: Optional[int]




logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Request has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_has_slo(func):
    def wrapper(self, *args, **kwargs):
        if not self._prefill_slo_time:
            raise ValueError("Request has no slo set")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Request has not been completed yet")
        return func(self, *args, **kwargs)

    return wrapper


class Request(BaseEntity):
    def __init__(
        self,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        block_hash_ids: Optional[List[int]],
        block_size: Optional[int],
        session_id: Optional[int] = None,
    ):
        if block_hash_ids is not None:
            last_block_size = (
                num_prefill_tokens
                + num_decode_tokens
                - (block_size * len(block_hash_ids))
            )
            assert (
                last_block_size >= 0 and last_block_size < block_size
            ), f"{last_block_size} is not in the range [0, {block_size})"

        self._id = Request.generate_id()
        self._arrived_at = arrived_at
        self._queued_at = arrived_at
        self._replica_id = None
        self._num_prefill_tokens = num_prefill_tokens
        self._num_prefill_tokens_cached = 0
        self._num_decode_tokens = num_decode_tokens
        self._num_processed_tokens = 0
        self._block_hash_ids = block_hash_ids
        self._block_size = block_size
        self._session_id = session_id

        self._scheduled_at = 0
        self._execution_time = 0
        self._model_execution_time = 0
        self._scheduling_delay = 0
        self._preempted_time = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        self._prefill_slo_time = None
        self._decode_slo_time = -1.0
        self._completion_slo_time = -1.0
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0

        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts = 0

        self._decode_next_deadline = None   # float or None
        self._decode_tokens_counted = 0     # how many decode tokens we've already accounted for in lateness


    @property
    def replica_id(self):
        return self._replica_id

    @property
    def size(self) -> Tuple[int, int]:
        return (self._num_prefill_tokens, self._num_decode_tokens)

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_scheduled
    def latest_stage_scheduled_at(self) -> float:
        return self._latest_stage_scheduled_at

    @property
    @check_scheduled
    def latest_stage_completed_at(self) -> float:
        return self._latest_stage_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduled_at(self) -> float:
        return self._latest_iteration_scheduled_at

    @property
    @check_scheduled
    def latest_iteration_completed_at(self) -> float:
        return self._latest_iteration_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduling_delay(self) -> float:
        return self._latest_iteration_scheduling_delay

    @property
    @check_scheduled
    def prefill_completed_at(self) -> float:
        return self._prefill_completed_at

    @property
    @check_scheduled
    def scheduling_delay(self) -> float:
        return self._scheduling_delay

    @property
    @check_scheduled
    def preempted_time(self) -> float:
        return self._preempted_time

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    @check_scheduled
    def e2e_time(self) -> float:
        return self._completed_at - self._arrived_at

    @property
    @check_scheduled
    def e2e_time_normalized(self) -> float:
        return self.e2e_time / self.num_decode_tokens

    @property
    @check_scheduled
    def execution_time(self) -> float:
        return self._execution_time

    @property
    @check_scheduled
    def execution_time_normalized(self) -> float:
        return self._execution_time / self.num_decode_tokens

    @property
    @check_scheduled
    def model_execution_time(self) -> float:
        return self._model_execution_time

    @property
    @check_scheduled
    def model_execution_time_normalized(self) -> float:
        return self._model_execution_time / self.num_decode_tokens

    @property
    def arrived_at(self) -> float:
        return self._arrived_at

    @property
    def queued_at(self):
        return self._queued_at

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_prefill_tokens_cached(self) -> int:
        return self._num_prefill_tokens_cached

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def pd_ratio(self) -> float:
        return self._num_prefill_tokens / self._num_decode_tokens

    @property
    def num_processed_tokens(self) -> int:
        return self._num_processed_tokens

    @property
    def total_tokens(self) -> int:
        return self._num_prefill_tokens + self._num_decode_tokens

    @property
    def num_processed_prefill_tokens(self) -> int:
        return min(self._num_processed_tokens, self._num_prefill_tokens)

    @property
    def num_processed_decode_tokens(self) -> int:
        return max(self._num_processed_tokens - self._num_prefill_tokens, 0)

    @property
    def block_hash_ids(self) -> Optional[List[int]]:
        return self._block_hash_ids

    @property
    def block_size(self) -> Optional[int]:
        return self._block_size

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def preempted(self) -> bool:
        return self._preempted and not self._completed

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def num_restarts(self) -> int:
        return self._num_restarts

    @property
    def is_prefill_complete(self) -> bool:
        return self._is_prefill_complete

    @property
    @check_has_slo
    def prefill_deadline_at(self) -> float:
        return self.queued_at + self._prefill_slo_time

    @property
    @check_has_slo
    def prefill_slo_time(self) -> float:
        return self._prefill_slo_time

    @prefill_slo_time.setter
    def prefill_slo_time(self, value: float):
        self._prefill_slo_time = value

    @property
    def decode_slo_time(self) -> float:
        return self._decode_slo_time

    @decode_slo_time.setter
    def decode_slo_time(self, value: float) -> None:
        self._decode_slo_time = value

    @property
    def completion_slo_time(self) -> float:
        return self._completion_slo_time

    @completion_slo_time.setter
    def completion_slo_time(self, value: float) -> None:
        self._completion_slo_time = value

    @property
    def has_started_decode(self) -> bool:
        return self._num_processed_tokens > self._num_prefill_tokens + 1

    @property
    def session_id(self) -> Optional[int]:
        return self._session_id

    def assign_replica(
        self,
        replica_id: ReplicaId,
    ) -> None:
        assert not self._scheduled, f"Request {self._id} already scheduled."
        self._replica_id = replica_id

    def on_cache_hit(self, num_tokens_cached: int):
        assert not self._scheduled, f"Request {self._id} already scheduled."
        assert (
            self._num_processed_tokens == 0
        ), f"Request {self._id} has already processed tokens."
        assert (
            num_tokens_cached <= self._num_prefill_tokens
        ), f"Request {self._id} has {num_tokens_cached} cached tokens, but only {self._num_prefill_tokens} prefill tokens."
        self._num_processed_tokens = num_tokens_cached
        self._num_prefill_tokens_cached = num_tokens_cached

    def restart(self):
        logger.debug(f"Restarting request {self._id}")

        # when we restart the request, we can process all the previously
        # decoded tokens in parallel (i.e., we can prefill all the tokens)
        total_tokens = self._num_prefill_tokens + self._num_decode_tokens
        self._num_prefill_tokens = self._num_processed_tokens
        self._num_decode_tokens = total_tokens - self._num_prefill_tokens

        self._num_processed_tokens = 0
        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        # reset decode tracking
        self._decode_next_deadline = None
        self._decode_tokens_counted = 0

        self._num_restarts += 1

    def on_batch_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_iteration_scheduled_at = time
        self._latest_iteration_scheduling_delay = (
            time - self._latest_iteration_completed_at
        )

        if self._scheduled:
            return

        if self._num_restarts > 0:
            self._scheduled = True
            return
        # First time scheduling
        self._scheduled_at = time
        self._scheduling_delay = time - self._arrived_at
        self._scheduled = True

    def on_batch_end(
        self,
        time: float,
        num_tokens_processed: int,
    ) -> None:
        self._num_processed_tokens += num_tokens_processed
        self._latest_iteration_completed_at = time

        assert self._num_processed_tokens <= self.total_tokens

        if self._num_processed_tokens == self._num_prefill_tokens:
            self._is_prefill_complete = True
            # we get one decode token when the prefill processing completes
            # if self._num_decode_tokens > 0:
            #     self._num_processed_tokens += 1

            # we must record the prefill completion time only in the first time
            # in the subsequent restarts, we keep adding the previously decoded
            # tokens to the prefill tokens - that is irrelevant to the original prefill
            if self._prefill_completed_at == 0:
                self._prefill_completed_at = time
                
            # Initialize decode lateness tracking when prefill first completes
            if getattr(self, "_decode_next_deadline", None) is None and self._decode_slo_time >= 0:
                self._decode_next_deadline = self._prefill_completed_at + self._decode_slo_time
                self._decode_tokens_counted = 0

        # check if request is completed
        if self._num_processed_tokens == self.total_tokens:
            self._completed_at = time
            self._completed = True
            logger.debug(f"Request {self._id} completed at {self._completed_at}")

    def on_batch_stage_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_stage_scheduled_at = time
        if self._latest_stage_completed_at == 0:
            self._preempted_time = 0
        else:
            self._preempted_time += time - self._latest_stage_completed_at
        self._preempted = False

    def on_batch_stage_end(
        self,
        time: float,
        execution_time: float,
        model_execution_time: float,
    ) -> None:
        self._execution_time += execution_time
        self._model_execution_time += model_execution_time
        self._latest_stage_completed_at = time
        self._preempted = True

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "num_processed_tokens": self._num_processed_tokens,
            "arrived_at": self._arrived_at,
            "queued_at": self._queued_at,
            # "prefill_deadline_at": self.prefill_deadline_at,
            "scheduled_at": self._scheduled_at,
            "prefill_completed_at": self._prefill_completed_at,
            "completed_at": self._completed_at,
            "scheduling_delay": self._scheduling_delay,
            "preempted_time": self._preempted_time,
            "execution_time": self._execution_time,
            "model_execution_time": self._model_execution_time,
            "latest_stage_scheduled_at": self._latest_stage_scheduled_at,
            "latest_stage_completed_at": self._latest_stage_completed_at,
            "latest_iteration_scheduled_at": self._latest_iteration_scheduled_at,
            "latest_iteration_completed_at": self._latest_iteration_completed_at,
            "scheduled": self._scheduled,
            "preempted": self._preempted,
            "completed": self._completed,
            "num_restarts": self._num_restarts,
            "decode_slo_time": self._decode_slo_time,
            "completion_slo_time": self._completion_slo_time,
        }

    # --- Snapshot helpers -------------------------------------------------
    # def snapshot_state(self) -> dict:
    #     """Capture a lightweight copy of the mutable request state."""
    #     return {key: clone_mutable(value) for key, value in self.__dict__.items()}

    # def restore_state(self, state: dict) -> None:
    #     """Restore the request to a previous state captured via ``snapshot_state``."""
    #     for key, value in state.items():
    #         setattr(self, key, clone_mutable(value))

    # --- Snapshot helpers (REPLACE THE OLD ONES) --------------------------

    # def snapshot_state(self) -> Dict[str, Any]:
    #     """Return a JSON-friendly, minimal snapshot of this Request's logical state."""
    #     # replica_id=int(self._replica_id) if self._replica_id is not None else None,
    #     rid = None
    #     if self._replica_id is not None:
    #         # prefer .id if present; otherwise fall back
    #         rid = int(getattr(self._replica_id, "id", self._replica_id))

    #     snap = RequestSnapshot(
    #         __v__=_SNAP_VERSION_REQ,
    #         id=int(self._id),
    #         arrived_at=float(self._arrived_at),
    #         queued_at=float(self._queued_at),
    #         replica_id = rid,
    #         num_prefill_tokens=int(self._num_prefill_tokens),
    #         num_prefill_tokens_cached=int(self._num_prefill_tokens_cached),
    #         num_decode_tokens=int(self._num_decode_tokens),
    #         num_processed_tokens=int(self._num_processed_tokens),

    #         block_hash_ids=list(self._block_hash_ids) if self._block_hash_ids is not None else None,
    #         block_size=int(self._block_size) if self._block_size is not None else None,

    #         scheduled=bool(self._scheduled),
    #         preempted=bool(self._preempted),
    #         completed=bool(self._completed),
    #         is_prefill_complete=bool(self._is_prefill_complete),

    #         num_restarts=int(self._num_restarts),

    #         scheduled_at=float(self._scheduled_at),
    #         preempted_time=float(self._preempted_time),
    #         completed_at=float(self._completed_at),
    #         prefill_completed_at=float(self._prefill_completed_at),
    #         scheduling_delay=float(self._scheduling_delay),
    #         execution_time=float(self._execution_time),
    #         model_execution_time=float(self._model_execution_time),
    #         latest_stage_scheduled_at=float(self._latest_stage_scheduled_at),
    #         latest_stage_completed_at=float(self._latest_stage_completed_at),
    #         latest_iteration_scheduled_at=float(self._latest_iteration_scheduled_at),
    #         latest_iteration_completed_at=float(self._latest_iteration_completed_at),
    #         latest_iteration_scheduling_delay=float(self._latest_iteration_scheduling_delay),

    #         prefill_slo_time=float(self._prefill_slo_time) if self._prefill_slo_time is not None else None,
    #         decode_slo_time=float(self._decode_slo_time),
    #         completion_slo_time=float(self._completion_slo_time),
    #         session_id=int(self._session_id) if self._session_id is not None else None,
    #     )
    #     # normalize/validate primitives for safety
    #     return to_primitive_tree(asdict(snap))

    def snapshot_state(self) -> Dict[str, Any]:
        rid = None
        if self._replica_id is not None:
            rid = int(getattr(self._replica_id, "id", self._replica_id))

        return {
            "__v__": _SNAP_VERSION_REQ,
            "id": int(self._id),
            "arrived_at": float(self._arrived_at),
            "queued_at": float(self._queued_at),
            "replica_id": rid,
            "num_prefill_tokens": int(self._num_prefill_tokens),
            "num_prefill_tokens_cached": int(self._num_prefill_tokens_cached),
            "num_decode_tokens": int(self._num_decode_tokens),
            "num_processed_tokens": int(self._num_processed_tokens),
            "block_hash_ids": list(self._block_hash_ids) if self._block_hash_ids is not None else None,
            "block_size": int(self._block_size) if self._block_size is not None else None,
            "scheduled": bool(self._scheduled),
            "preempted": bool(self._preempted),
            "completed": bool(self._completed),
            "is_prefill_complete": bool(self._is_prefill_complete),
            "num_restarts": int(self._num_restarts),
            "scheduled_at": float(self._scheduled_at),
            "preempted_time": float(self._preempted_time),
            "completed_at": float(self._completed_at),
            "prefill_completed_at": float(self._prefill_completed_at),
            "scheduling_delay": float(self._scheduling_delay),
            "execution_time": float(self._execution_time),
            "model_execution_time": float(self._model_execution_time),
            "latest_stage_scheduled_at": float(self._latest_stage_scheduled_at),
            "latest_stage_completed_at": float(self._latest_stage_completed_at),
            "latest_iteration_scheduled_at": float(self._latest_iteration_scheduled_at),
            "latest_iteration_completed_at": float(self._latest_iteration_completed_at),
            "latest_iteration_scheduling_delay": float(self._latest_iteration_scheduling_delay),
            "prefill_slo_time": float(self._prefill_slo_time) if self._prefill_slo_time is not None else None,
            "decode_slo_time": float(self._decode_slo_time),
            "completion_slo_time": float(self._completion_slo_time),
            "session_id": int(self._session_id) if self._session_id is not None else None,
        }



    @staticmethod
    def from_snapshot(s: Dict[str, Any]) -> "Request":
        """Construct a new Request object from a snapshot dict."""
        assert int(s["__v__"]) == _SNAP_VERSION_REQ, "Request snapshot version mismatch"

        req = Request.__new__(Request)

        req._arrived_at = float(s["arrived_at"])
        req._num_prefill_tokens = int(s["num_prefill_tokens"])
        req._num_decode_tokens = int(s["num_decode_tokens"])
        req._block_hash_ids = (
            list(s["block_hash_ids"]) if s.get("block_hash_ids") is not None else None
        )
        req._block_size = int(s["block_size"]) if s.get("block_size") is not None else None
        req._session_id = int(s["session_id"]) if s.get("session_id") is not None else None

        # Set identity & basic timeline first
        req._id = int(s["id"])
        req._queued_at = float(s["queued_at"])

        # Replica note: assign directly to bypass the "not scheduled" assert
        # req._replica_id = int(s["replica_id"]) if s.get("replica_id") is not None else None

        from vidur.types.replica_id import ReplicaId
        req._replica_id = (
            ReplicaId(int(s["replica_id"])) if s.get("replica_id") is not None else None
        )

        # Sizes & progress
        req._num_prefill_tokens = int(s["num_prefill_tokens"])
        req._num_prefill_tokens_cached = int(s["num_prefill_tokens_cached"])
        req._num_decode_tokens = int(s["num_decode_tokens"])
        req._num_processed_tokens = int(s["num_processed_tokens"])

        # Flags & counters
        req._scheduled = bool(s["scheduled"])
        req._preempted = bool(s["preempted"])
        req._completed = bool(s["completed"])
        req._is_prefill_complete = bool(s["is_prefill_complete"])
        req._num_restarts = int(s["num_restarts"])

        # Timing/metrics
        req._scheduled_at = float(s["scheduled_at"])
        req._preempted_time = float(s["preempted_time"])
        req._completed_at = float(s["completed_at"])
        req._prefill_completed_at = float(s["prefill_completed_at"])
        req._scheduling_delay = float(s["scheduling_delay"])
        req._execution_time = float(s["execution_time"])
        req._model_execution_time = float(s["model_execution_time"])
        req._latest_stage_scheduled_at = float(s["latest_stage_scheduled_at"])
        req._latest_stage_completed_at = float(s["latest_stage_completed_at"])
        req._latest_iteration_scheduled_at = float(s["latest_iteration_scheduled_at"])
        req._latest_iteration_completed_at = float(s["latest_iteration_completed_at"])
        req._latest_iteration_scheduling_delay = float(s["latest_iteration_scheduling_delay"])

        # SLO
        req._prefill_slo_time = float(s["prefill_slo_time"]) if s.get("prefill_slo_time") is not None else None
        req._decode_slo_time = float(s.get("decode_slo_time", -1.0))
        req._completion_slo_time = float(s.get("completion_slo_time", -1.0))

        # NEW: ensure decode tracking fields exist on restored requests
        req._decode_next_deadline = None
        req._decode_tokens_counted = 0

        return req

    # def restore_state(self, s: Dict[str, Any]) -> None:
    #     """Mutate this Request to match a snapshot."""
    #     rebuilt = Request.from_snapshot(s)
    #     # Copy fields over (keeps object identity stable if external maps hold this instance)
    #     self.__dict__.update(rebuilt.__dict__)



    def restore_state(self, s: Dict[str, Any]) -> None:
        assert int(s["__v__"]) == _SNAP_VERSION_REQ, "Request snapshot version mismatch"

        self._id = int(s["id"])
        self._arrived_at = float(s["arrived_at"])
        self._queued_at = float(s["queued_at"])

        rid = s.get("replica_id", None)
        self._replica_id = ReplicaId(int(rid)) if rid is not None else None

        self._num_prefill_tokens = int(s["num_prefill_tokens"])
        self._num_prefill_tokens_cached = int(s.get("num_prefill_tokens_cached", 0))
        self._num_decode_tokens = int(s["num_decode_tokens"])
        self._num_processed_tokens = int(s["num_processed_tokens"])

        self._block_hash_ids = list(s["block_hash_ids"]) if s.get("block_hash_ids") is not None else None
        self._block_size = int(s["block_size"]) if s.get("block_size") is not None else None
        self._session_id = int(s["session_id"]) if s.get("session_id") is not None else None

        self._scheduled = bool(s["scheduled"])
        self._preempted = bool(s["preempted"])
        self._completed = bool(s["completed"])
        self._is_prefill_complete = bool(s["is_prefill_complete"])
        self._num_restarts = int(s.get("num_restarts", 0))

        self._scheduled_at = float(s.get("scheduled_at", 0.0))
        self._preempted_time = float(s.get("preempted_time", 0.0))
        self._completed_at = float(s.get("completed_at", 0.0))
        self._prefill_completed_at = float(s.get("prefill_completed_at", 0.0))

        self._scheduling_delay = float(s.get("scheduling_delay", 0.0))
        self._execution_time = float(s.get("execution_time", 0.0))
        self._model_execution_time = float(s.get("model_execution_time", 0.0))
        self._latest_stage_scheduled_at = float(s.get("latest_stage_scheduled_at", 0.0))
        self._latest_stage_completed_at = float(s.get("latest_stage_completed_at", 0.0))
        self._latest_iteration_scheduled_at = float(s.get("latest_iteration_scheduled_at", 0.0))
        self._latest_iteration_completed_at = float(s.get("latest_iteration_completed_at", 0.0))
        self._latest_iteration_scheduling_delay = float(s.get("latest_iteration_scheduling_delay", 0.0))

        self._prefill_slo_time = float(s["prefill_slo_time"]) if s.get("prefill_slo_time") is not None else None
        self._decode_slo_time = float(s.get("decode_slo_time", -1.0))
        self._completion_slo_time = float(s.get("completion_slo_time", -1.0))

        # keep decode tracking fields present (env uses stats.*, but keep invariants)
        self._decode_next_deadline = None
        self._decode_tokens_counted = 0
