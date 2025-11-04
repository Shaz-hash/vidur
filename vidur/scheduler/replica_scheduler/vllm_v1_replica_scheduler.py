from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Set

from vidur.entities.batch import Batch, Request
from vidur.kv_cache.replica_kv_cache_manager import (
    ReplicaKVCacheManager,
    ReplicaKVCacheManagerSnapshot,
)
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.replica_scheduler_output import (
    ReplicaSchedulerOutput,
)
from vidur.scheduler.replica_stage_scheduler.replica_stage_scheduler import (
    ReplicaStageSchedulerSnapshot,
)
from vidur.types.request_queue_type import RequestQueueType


_SNAP_VERSION_VLLM_V1 = 1


@dataclass(frozen=True)
class VLLMV1ReplicaSchedulerSnapshot:
    __v__: int
    # id -> Request.snapshot_state() dict (JSON-safe)
    request_states: Dict[int, dict]
    # waiting queue snapshot (JSON-safe dict)
    waiting_queue_state: dict
    # order matters for running list
    running_request_ids: List[int]
    # JSON-safe: sets as lists (we’ll cast back to sets on restore)
    scheduled_req_ids: List[int]
    ever_scheduled: List[int]
    # KV snapshot (already JSON-safe dataclass/dict)
    kv_cache_state: ReplicaKVCacheManagerSnapshot
    # stage_id -> snapshot (JSON-safe dataclass/dict)
    replica_stage_states: Dict[int, ReplicaStageSchedulerSnapshot]
    num_running_batches: int
    finished_req_ids: List[int]


class VLLMV1ReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            self._waiting_queue._config.get_type() == RequestQueueType.FCFS
        ), "VLLM_v1 scheduler only supports FCFS request queues"
        assert (
            self._num_stages == 1
        ), "VLLM_v1 scheduler doesn't support pipeline parallelism"

        # Scheduling constraints
        self._max_batch_size = self._config.batch_size_cap
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages

        # Create the KV Cache manager
        # NOTE TO MYSELF : 
        # This part is KV memory manager. Some params allow for the prefix caching which in our case will be default 
        self._kv_cache_manager = ReplicaKVCacheManager(
            block_size=self._cache_config.block_size,
            num_gpu_blocks=self._cache_config.num_blocks,
            enable_caching=self._cache_config.enable_prefix_caching,
            caching_hash_algo=self._cache_config.prefix_caching_hash_algo,
            num_preallocate_tokens=self._cache_config.num_preallocate_tokens,
        )

        # req_id -> Request
        self._requests: Dict[str, Request] = {}
        # self._waiting_queue has been initialized in the parent class
        self._running: List[Request] = []
        # The requests that have been scheduled and are being executed
        # by the executor.
        self.scheduled_req_ids: set[str] = set()

        # new--> Tracks whether a request has *ever* been scheduled at least once
        self._ever_scheduled: set[str] = set()

        print("VLLM SCHEDULER CALLED!")

        self._token_budget_overrides: Dict[int, int] = {}


    def _kv_free_blocks(self) -> int:
        """
        Return how many KV cache blocks are currently free on this replica.
        Prefer a manager API if it exists; otherwise derive from usage/num_gpu_blocks.
        """
        kvm = self._kv_cache_manager
        if hasattr(kvm, "free_blocks"):
            try:
                return int(kvm.free_blocks())
            except Exception:
                pass
        if hasattr(kvm, "num_gpu_blocks") and hasattr(kvm, "usage"):
            used = int(round(kvm.usage * kvm.num_gpu_blocks))
            return int(kvm.num_gpu_blocks - used)
        # As a last resort, fall back to BaseReplicaScheduler counters if you keep them:
        if hasattr(self, "_cache_config") and hasattr(self._cache_config, "num_blocks") and hasattr(self, "_num_allocated_blocks"):
            return int(self._cache_config.num_blocks - self._num_allocated_blocks)
        return -1  # sentinel if nothing available

    def _snapshot_batch_start_counters(self) -> dict:
        # 1) Requests still waiting after selection
        try:
            waiting_count = len(self._waiting_queue)
        except TypeError:
            # if not len()-able, adapt to your queue API:
            waiting_count = self._waiting_queue.size()  # or similar

        # 2) Initiated but not completed: those already running and not completed
        initiated_not_completed = sum(1 for r in self._running if not r.completed)

        # 3) Classify decode vs prefill across ALL live requests
        all_known = list(self._requests.values())

        def is_decode(r: Request) -> bool:
            if getattr(r, "completed", False):
                return False
            return bool(getattr(r, "has_started_decode", False) or getattr(r, "is_prefill_complete", False))

        def is_initiated(r: Request) -> bool:
            return (r.id in self.scheduled_req_ids) or getattr(r, "num_processed_tokens", 0) > 0 or (r in self._running)

        num_decode_phase_total = sum(1 for r in all_known if is_decode(r))
        num_prefill_queue_total = sum(1 for r in all_known if (not is_decode(r)) and (not r.completed))

        # 4) Not initiated but still in waiting queue
        try:
            waiting_iterable = list(self._waiting_queue)
        except TypeError:
            waiting_iterable = getattr(self._waiting_queue, "_items", [])  # adjust to your queue impl
        num_not_initiated_in_queue = sum(1 for r in waiting_iterable if not is_initiated(r))

        # 5) Total live requests at batch start (exclude completed)
        total_requests_batch_start = sum(1 for r in all_known if not r.completed)

        return {
            "num_requests_not_selected": waiting_count,
            "num_requests_initiated_not_completed": initiated_not_completed,
            "num_decode_phase_total": num_decode_phase_total,
            "num_prefill_queue_total": num_prefill_queue_total,
            "num_not_initiated_in_queue": num_not_initiated_in_queue,
            "total_requests_batch_start": total_requests_batch_start,
        }





    @property
    def memory_usage_percent(self) -> float:
        return self._kv_cache_manager.usage * 100

    def get_cached_prefill_length(self, request: Request) -> int:
        _, num_computed_tokens = self._kv_cache_manager.get_computed_blocks(request)
        return num_computed_tokens

    def add_request(self, request: Request):
        request.assign_replica(self._replica_id)
        self._waiting_queue.push(request)
        self._requests[request.id] = request

    def set_token_budget_overrides(self, overrides: Dict[int, int]):
        self._token_budget_overrides = {
            int(rid): max(0, int(tokens)) for rid, tokens in overrides.items()
        }

    def _get_request_next_num_tokens(self, request: Request, token_budget: int) -> int:
        assert not request.completed

        override = self._token_budget_overrides.get(request.id)
        if override is not None:
            override = max(0, min(int(override), token_budget))
            if request.is_prefill_complete:
                return min(override, 1)
            remaining_prefill = request.num_prefill_tokens - request.num_processed_tokens
            return max(0, min(override, remaining_prefill))

        # Calculate `next_num_tokens`
        if request.is_prefill_complete:
            next_num_tokens = 1
        else:
            next_num_tokens = request.num_prefill_tokens - request.num_processed_tokens
        # Pass through the token budget
        next_num_tokens = min(next_num_tokens, token_budget)
        # No negative answer
        next_num_tokens = max(0, next_num_tokens)
        return next_num_tokens

    def _get_next_batch(self, current_time: float) -> ReplicaSchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_reqs: List[Request] = []
        preempted_reqs: List[Request] = []
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self._config.chunk_size

        # First, schedule the RUNNING requests
        req_index = 0
        while req_index < len(self._running) and token_budget > 0:
            request: Request = self._running[req_index]


            if request.id in self.scheduled_req_ids:
                req_index += 1
                continue

            # Calculate compute to do for the request
            num_new_tokens = self._get_request_next_num_tokens(request, token_budget)
            assert (
                num_new_tokens > 0
            ), "num_new_tokens should be as token_budget > 0 and request is incomplete"

            # Try to allocate memory for the request
            while True:
                new_blocks = self._kv_cache_manager.allocate_slots(
                    request, num_new_tokens
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    preempted_req: Request = self._running.pop()  # from last
                    self._kv_cache_manager.free(preempted_req)
                    preempted_req.restart()
                    self._waiting_queue.push(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_reqs.append(request)
            self.scheduled_req_ids.add(request.id)

            ## To flag the requests that have been scheduled so far in the history :
            # new-->
            # self._ever_scheduled.add(request.id)   

            #  # --> new :
            # if not hasattr(self, "_ever_scheduled"):
            #     self._ever_scheduled = set()
            self._ever_scheduled.add(request.id)


            num_scheduled_tokens[request.id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1
            if request.id in self._token_budget_overrides:
                remaining = self._token_budget_overrides[request.id] - num_new_tokens
                if remaining <= 0:
                    self._token_budget_overrides.pop(request.id, None)
                else:
                    self._token_budget_overrides[request.id] = remaining

        # Use a temporary deque to collect requests that need to be skipped
        # and put back at the head of the waiting queue later
        skipped_waiting_requests: Deque[Request] = deque()

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while len(self._waiting_queue) and token_budget > 0:
                if len(self._running) >= self._max_micro_batch_size:
                    break

                request = self._waiting_queue.peek()

                # Get already-cached tokens. `computed` means `cached` here.
                computed_blocks, num_computed_tokens = (
                    self._kv_cache_manager.get_computed_blocks(request)
                )

                # Guard against restored snapshots that leave an extra cached block.
                if num_computed_tokens > request.num_prefill_tokens:
                    overflow = num_computed_tokens - request.num_prefill_tokens
                    blocks_to_drop = (overflow + self._cache_config.block_size - 1) // self._cache_config.block_size
                    for _ in range(blocks_to_drop):
                        if not computed_blocks:
                            break
                        computed_blocks.pop()
                        num_computed_tokens -= self._cache_config.block_size
                    num_computed_tokens = max(num_computed_tokens, request.num_prefill_tokens)


                # Number of tokens to be scheduled.
                # Using `request.num_prefill_tokens` is fine even for restarted requests
                # because done decode tokens have been added to prefill tokens.
                num_new_tokens = request.num_prefill_tokens - num_computed_tokens
                if num_new_tokens == 0:
                    # This happens when prompt length is divisible by the block
                    # size and all blocks are cached. Now we force to recompute
                    # the last block. Note that we have to re-compute an entire
                    # block because allocate_slots() assumes num_computed_tokens
                    # is always a multiple of the block size. This limitation
                    # can potentially be removed in the future to slightly
                    # improve the performance.
                    num_computed_tokens -= self._cache_config.block_size
                    num_new_tokens = self._cache_config.block_size
                    computed_blocks.pop()
                num_new_tokens = min(num_new_tokens, token_budget)
                override = self._token_budget_overrides.get(request.id)
                if override is not None:
                    override = max(0, min(int(override), token_budget))
                    remaining_prefill = request.num_prefill_tokens - num_computed_tokens
                    num_new_tokens = min(override, remaining_prefill)
                assert (
                    num_new_tokens > 0
                ), f"num_new_tokens should be greater than 0 but got {num_new_tokens}"

                new_blocks = self._kv_cache_manager.allocate_slots(
                    request, num_new_tokens, computed_blocks
                )
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                self._waiting_queue.pop()
                req_index += 1
                self._running.append(request)
                self.scheduled_req_ids.add(request.id)
                scheduled_reqs.append(request)
                #new-->
                self._ever_scheduled.add(request.id)  
                assert not request.scheduled     
                num_scheduled_tokens[request.id] = num_new_tokens
                token_budget -= num_new_tokens
                # Update the number of processed tokens for the request
                request.on_cache_hit(num_computed_tokens)
                if request.id in self._token_budget_overrides:
                    remaining = self._token_budget_overrides[request.id] - num_new_tokens
                    if remaining <= 0:
                        self._token_budget_overrides.pop(request.id, None)
                    else:
                        self._token_budget_overrides[request.id] = remaining

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self._waiting_queue.extend(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self._config.chunk_size
        assert token_budget >= 0
        assert len(self._running) <= self._max_micro_batch_size
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_reqs) <= len(self._running)

        scheduler_output = ReplicaSchedulerOutput(
            (
                Batch(
                    self._replica_id,
                    scheduled_reqs,
                    [num_scheduled_tokens[request.id] for request in scheduled_reqs],
                )
                if scheduled_reqs
                else None
            ),
            [],
        )
        # If a real batch was created, annotate it:
        if scheduler_output.batch is not None:
            try:
                # print("here!")
                snapshot = self._snapshot_batch_start_counters()
                scheduler_output.batch.kv_free_blocks_before = self._kv_free_blocks()
                scheduler_output.batch.num_requests_not_selected           = snapshot["num_requests_not_selected"]
                scheduler_output.batch.num_requests_initiated_not_completed = snapshot["num_requests_initiated_not_completed"]
                scheduler_output.batch.num_decode_phase_total              = snapshot["num_decode_phase_total"]
                scheduler_output.batch.num_prefill_queue_total             = snapshot["num_prefill_queue_total"]
                scheduler_output.batch.num_not_initiated_in_queue          = snapshot["num_not_initiated_in_queue"]
                scheduler_output.batch.total_requests_batch_start          = snapshot["total_requests_batch_start"]

                # print("scheduler_output KV BLOCKS : ", scheduler_output.batch.kv_free_blocks_before)
            except Exception:
                scheduler_output.batch.kv_free_blocks_before = None
            scheduler_output.batch.request_ids_in_batch = ";".join(
                str(r.id) for r in scheduled_reqs
            )
        




        # TODO(nitin): Immediately updating num_processed_tokens for the request is important for
        #  sequence pipeline parallelism and multi-step scheduling.
        # However, this is not done here to protect the invariant that num_processed_tokens is updated only after batch end.
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        # for req_id, num_scheduled_token in num_scheduled_tokens.items():
        #     self._requests[req_id].num_processed_tokens += num_scheduled_token

        self.finished_req_ids = set()
        return scheduler_output

    def is_empty(self) -> bool:
        return len(self._waiting_queue) + len(self._running) == 0

    def on_batch_end(self, batch: Batch) -> None:


        #--> new
        # ---- Build per-request rows BEFORE we mutate state / free KV ----
        rows = []
        added_ids = {r.id for r in batch.requests}

        # If you also want to include requests that completed in *this* batch,
        # collect them before freeing; otherwise the registry below is enough.
        all_live = list(self._requests.values())

        for req in all_live:
            # KV context length so far
            _, num_computed_tokens = self._kv_cache_manager.get_computed_blocks(req)

            in_decode = bool(getattr(req, "has_started_decode", False) or getattr(req, "is_prefill_complete", False))
            slo_type = "TBT" if in_decode else "TTFT"

            prefill_len = req.num_prefill_tokens
            prefill_remaining = 0 if in_decode else max(prefill_len - num_computed_tokens, 0)

            rows.append({
                "batch_id": batch.id,
                "request_id": req.id,
                "slo_type": slo_type,
                "slo_remaining_ms": 0,  # stub for now
                "only_in_queue": (req.id not in self._ever_scheduled),
                "added_in_last_batch": (req.id in added_ids),
                "request_type_after_batch": ("decode" if in_decode else "prefill"),
                "kv_context_len_tokens": num_computed_tokens,
                "prefill_len_tokens": prefill_len,
                "prefill_remaining_tokens": prefill_remaining,
                "TBT SLO": req.decode_slo_time,
                "TTC SLO": req.completion_slo_time,
            })

        # Stash on batch for the metrics store to consume
        batch._request_details_rows = rows

        if not hasattr(batch, "_request_details_rows"):
            batch._request_details_rows = []


        try:
            batch.kv_free_blocks_after = self._kv_free_blocks()
        except Exception:
            batch.kv_free_blocks_after = None

        self._num_running_batches -= 1
        new_running: List[Request] = []

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self._running:
            req_id = request.id
            num_tokens_scheduled = batch.num_tokens_dict.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue
            elif request.completed:
                self._free_request(request)
            else:
                new_running.append(request)
            self.scheduled_req_ids.remove(req_id)
        self._running = new_running

    def _free_request(self, request: Request) -> None:
        assert request.completed
        self._kv_cache_manager.free(request)
        self._kv_cache_manager.free_block_hashes(request)
        del self._requests[request.id]

    # --- Snapshot helpers -------------------------------------------------
    def snapshot_state(self) -> VLLMV1ReplicaSchedulerSnapshot:
        # Per-request snapshots (only those known to this replica)
        request_states = {
            int(request_id): req.snapshot_state()
            for request_id, req in self._requests.items()
        }

        waiting_queue_state = (
            self._waiting_queue.snapshot_state()
            if hasattr(self._waiting_queue, "snapshot_state")
            else {}
        )

        stage_states = {
            int(stage_id): stage_scheduler.snapshot_state()
            for stage_id, stage_scheduler in self._replica_stage_schedulers.items()
        }

        return VLLMV1ReplicaSchedulerSnapshot(
            __v__=_SNAP_VERSION_VLLM_V1,
            request_states=request_states,
            waiting_queue_state=waiting_queue_state,
            running_request_ids=[int(r.id) for r in self._running],
            scheduled_req_ids=[int(x) for x in self.scheduled_req_ids],
            ever_scheduled=[int(x) for x in self._ever_scheduled],
            kv_cache_state=self._kv_cache_manager.snapshot_state(),
            replica_stage_states=stage_states,
            num_running_batches=int(self._num_running_batches),
            finished_req_ids=[int(x) for x in getattr(self, "finished_req_ids", set())],
        )

    def restore_state(
        self,
        snapshot: VLLMV1ReplicaSchedulerSnapshot,
        request_lookup: Dict[int, Request],
        batch_lookup: Dict[int, Batch],
    ) -> None:
        # Version check
        assert (
            int(snapshot.__v__) == _SNAP_VERSION_VLLM_V1
        ), "VLLM v1 scheduler snapshot version mismatch"

        # 1) Rebuild request registry and restore each Request’s internal state
        rebuilt: Dict[int, Request] = {}
        for req_id, req_state in snapshot.request_states.items():
            req_obj = request_lookup[req_id]  # authoritative object from the sim
            req_obj.restore_state(req_state)
            rebuilt[int(req_id)] = req_obj
        self._requests = rebuilt

        # 2) Waiting queue
        if hasattr(self._waiting_queue, "restore_state"):
            self._waiting_queue.restore_state(
                snapshot.waiting_queue_state, request_lookup
            )

        # 3) Running list (preserve order)
        self._running = [request_lookup[rid] for rid in snapshot.running_request_ids]

        # 4) Sets and counters
        self.scheduled_req_ids = set(int(x) for x in snapshot.scheduled_req_ids)
        self._ever_scheduled = set(int(x) for x in snapshot.ever_scheduled)
        self._num_running_batches = int(snapshot.num_running_batches)
        self.finished_req_ids = set(int(x) for x in snapshot.finished_req_ids)

        # 5) Stage schedulers (now include active_batch_id in their snapshot)
        for stage_id, stage_scheduler in self._replica_stage_schedulers.items():
            stage_snapshot = snapshot.replica_stage_states.get(int(stage_id))
            if stage_snapshot:
                stage_scheduler.restore_state(stage_snapshot, batch_lookup)

        # 6) KV cache
        self._kv_cache_manager.restore_state(snapshot.kv_cache_state)

        # 7) Align request KV/bookkeeping to restored cache state
        for req_id, request in self._requests.items():
            cached_blocks, cached_tokens = self._kv_cache_manager.get_computed_blocks(
                request
            )
            if cached_tokens < 0:
                cached_tokens = 0
            if cached_tokens % self._cache_config.block_size != 0:
                cached_tokens = (
                    cached_tokens // self._cache_config.block_size
                ) * self._cache_config.block_size
            if cached_tokens > request.num_prefill_tokens:
                cached_tokens = request.num_prefill_tokens

            request._num_prefill_tokens_cached = cached_tokens
            if request._num_processed_tokens < cached_tokens:
                request._num_processed_tokens = cached_tokens

        # (Optional) quick sanity:
        assert all(
            r.id in self._requests for r in self._running
        ), "Running list references unknown requests"
