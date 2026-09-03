import heapq
from collections import deque
from typing import Deque, Dict, List, Tuple, Optional

from vidur.entities.request import Request
from vidur.scheduler.request_queue.base_request_queue import BaseRequestQueue
from vidur.scheduler.request_queue.prioritised_request import PrioritizedRequest


_SNAP_VERSION_FCFS = 1


class FCFSRequestQueue(BaseRequestQueue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_queue: List[PrioritizedRequest] = []
        self._num_prefill_tokens = 0

    def _get_prioritized_request(self, request: Request) -> PrioritizedRequest:
        return PrioritizedRequest(request, request.arrived_at)

    def push(self, request):
        heapq.heappush(self._request_queue, self._get_prioritized_request(request))
        self._num_prefill_tokens += request.num_prefill_tokens

    def pop(self):
        request = heapq.heappop(self._request_queue).request
        self._num_prefill_tokens -= request.num_prefill_tokens
        return request

    def peek(self):
        return self._request_queue[0].request

    def to_list(self):
        return [
            prioritized_request.request for prioritized_request in self._request_queue
        ]

    def __len__(self):
        return len(self._request_queue)

    def get_num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    def sort(self, requests: Deque[Request]) -> Deque[Request]:
        return deque(sorted(requests, key=lambda x: (x.arrived_at, x.id)))

    # # --- Snapshot helpers -------------------------------------------------
    # def snapshot_state(self) -> dict:
    #     items: List[Tuple[int, float]] = [
    #         (prioritized_request.request.id, prioritized_request.priority)
    #         for prioritized_request in self._request_queue
    #     ]
    #     return {
    #         "items": items,
    #         "num_prefill_tokens": self._num_prefill_tokens,
    #     }

    # def restore_state(self, snapshot: dict, request_lookup: Dict[int, Request]) -> None:
    #     self._request_queue = [
    #         PrioritizedRequest(request_lookup[req_id], priority)
    #         for req_id, priority in snapshot.get("items", [])
    #     ]
    #     heapq.heapify(self._request_queue)
    #     self._num_prefill_tokens = snapshot.get("num_prefill_tokens", 0)

    # --- Snapshot helpers -------------------------------------------------
    def snapshot_state(self) -> dict:
        # Minimal, JSON-safe snapshot. We only store request IDs.
        return {
            "__v__": _SNAP_VERSION_FCFS,
            "request_ids": [pr.request.id for pr in self._request_queue],
        }

    def restore_state(self, snapshot: dict, request_lookup: Dict[int, Request]) -> None:
        assert int(snapshot.get("__v__", 0)) == _SNAP_VERSION_FCFS, "FCFS snapshot version mismatch"

        req_ids = list(snapshot.get("request_ids", []))
        # Rebuild heap from requests (recompute priorities)
        items: List[PrioritizedRequest] = []
        for rid in req_ids:
            if rid not in request_lookup:
                raise KeyError(f"FCFS restore: unknown request id {rid}")
            req = request_lookup[rid]
            items.append(self._get_prioritized_request(req))

        self._request_queue = items
        heapq.heapify(self._request_queue)

        # Recompute num_prefill_tokens to avoid drift
        self._num_prefill_tokens = sum(pr.request.num_prefill_tokens for pr in self._request_queue)


