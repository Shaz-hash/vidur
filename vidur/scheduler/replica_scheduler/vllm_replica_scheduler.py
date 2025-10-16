from math import ceil

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.types.request_queue_type import RequestQueueType


class VLLMReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            self._request_queue._config.get_type() == RequestQueueType.FCFS
        ), "VLLM scheduler only supports FCFS request queues"

        self._num_running_batches = 0
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
        self._max_batch_size = self._config.batch_size_cap
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        print("VLLM Batch Size here : ", self._max_batch_size , self._max_micro_batch_size)
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

    # --- NEW: small helper to read current free KV blocks
    def _kv_free_blocks(self) -> int:
        # num_blocks and _num_allocated_blocks are already tracked by BaseReplicaScheduler
        return self._config.num_blocks - self._num_allocated_blocks





    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

        # --- NEW: annotate "after" snapshot on the batch
        try:
            batch.kv_free_blocks_after = self._kv_free_blocks()
        except Exception:
            # leave silently if something changes in allocator plumbing
            pass

    




    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._replica_config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        # vllm requires at least one block to be available
        return self._config.num_blocks - self._num_allocated_blocks >= 1
    





    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._replica_config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = (
            self._allocation_map[request.id] * self._replica_config.block_size
        )
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)
        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        num_batch_tokens = 0

        while len(self._request_queue):
            request = self._request_queue[0]

            next_num_tokens = self._get_request_next_num_tokens(request)

            if not self._can_allocate_request(request):
                break

            new_num_batch_tokens = num_batch_tokens + next_num_tokens
            if new_num_batch_tokens > self._config.max_tokens_in_batch:
                break

            if len(self._allocation_map) == self._max_batch_size:
                break

            if len(requests) == self._max_micro_batch_size:
                break

            request = self._request_queue.popleft()

            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            num_batch_tokens += next_num_tokens

        if requests:

            batch = Batch(self._replica_id, requests, num_tokens)

            # --- NEW: annotate "before" KV and the request IDs chosen for this batch
            try:
                batch.kv_free_blocks_before = self._kv_free_blocks()
            except Exception:
                batch.kv_free_blocks_before = None

            # join as semicolon-separated string; friendlier for CSV
            batch.request_ids_in_batch = ";".join(str(r.id) for r in requests)

            return batch

        # Safer to sort preempted_requests to maintain FIFO order
        self._preempted_requests.sort(key=lambda r: r.arrived_at)
        # all preempted_requests will have prefill completed
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                break

            request = self._preempted_requests.popleft()

            while not self._can_allocate_request(request):
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop()
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue.appendleft(victim_request)
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue.appendleft(request)
                    break
            else:
                self._allocate_request(request)
                next_num_tokens = self._get_request_next_num_tokens(request)
                requests.append(request)
                num_tokens.append(next_num_tokens)

        if not requests:
            return

        batch = Batch(self._replica_id, requests, num_tokens)

        # --- NEW: annotate "before" KV and request IDs for the preempted path too
        try:
            batch.kv_free_blocks_before = self._kv_free_blocks()
        except Exception:
            batch.kv_free_blocks_before = None

        batch.request_ids_in_batch = ";".join(str(r.id) for r in requests)

        return batch
