# from typing import List, Tuple

# from vidur.entities import Request
# from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
# from vidur.types.replica_id import ReplicaId


# class RoundRobinGlobalScheduler(BaseGlobalScheduler):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._request_counter = 0
#         self._replica_id_list = sorted(self._replicas.keys())

#     def schedule(self) -> List[Tuple[ReplicaId, Request]]:
#         self.sort_requests()

#         request_mapping = []
#         while self._request_queue:
#             request = self._request_queue.pop(0)
#             replica_id = self._replica_id_list[
#                 self._request_counter % self._num_replicas
#             ]
#             self._request_counter += 1
#             request_mapping.append((replica_id, request))

#         return request_mapping

#     def _snapshot_extra_state(self) -> dict:
#         return {"request_counter": self._request_counter}

#     def _restore_extra_state(self, snapshot: dict) -> None:
#         self._request_counter = snapshot.get("request_counter", 0)



from typing import List, Tuple, Dict, Any

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.replica_id import ReplicaId

_RR_SNAP_VERSION = 1

class RoundRobinGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Deterministic order across runs
        self._replica_id_list = sorted(self._replicas.keys())
        self._request_counter = 0

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        self.sort_requests()

        request_mapping: List[Tuple[ReplicaId, Request]] = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = self._replica_id_list[self._request_counter % self._num_replicas]
            self._request_counter += 1
            request_mapping.append((replica_id, request))

        return request_mapping

    # --- Snapshot hooks (called by BaseGlobalScheduler) -------------------
    def _snapshot_extra_state(self) -> Dict[str, Any]:
        # store the pointer and (optionally) the exact replica order we used
        return {
            "__v__": _RR_SNAP_VERSION,
            "request_counter": int(self._request_counter),
            "replica_ids": [str(rid) for rid in self._replica_id_list],
        }

    def _restore_extra_state(self, snapshot: Dict[str, Any]) -> None:
        v = int(snapshot.get("__v__", 1))
        if v != _RR_SNAP_VERSION:
            raise ValueError(
                f"RoundRobin scheduler snapshot version mismatch: got {v}, expected {_RR_SNAP_VERSION}"
            )

        self._request_counter = int(snapshot.get("request_counter", 0))

        # Rebuild deterministic order from live replicas, then optionally sanity-check
        current_order = [str(rid) for rid in sorted(self._replicas.keys())]
        saved_order = snapshot.get("replica_ids")
        if isinstance(saved_order, list) and saved_order != current_order:
            # If cluster layout changed, we keep the current order but reset the pointer
            # to avoid skew; alternatively, you could map indices. Choose what fits your use case.
            self._request_counter = 0
        self._replica_id_list = sorted(self._replicas.keys())
