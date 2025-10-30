from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from vidur.entities import Batch, BatchStage, ExecutionTime, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.types.replica_id import ReplicaId


_SNAP_VERSION_STAGE = 1


@dataclass(frozen=True)
class ReplicaStageSchedulerSnapshot:
    __v__: int
    batch_ids: List[int]                 # queued (FIFO)
    active_batch_id: Optional[int]       # currently running batch, if any
    is_busy: bool                        # redundant if active_batch_id present, but kept for readability


class ReplicaStageScheduler:
    def __init__(
        self,
        replica_id: ReplicaId,
        stage_id: int,
        is_last_stage: bool,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage
        self._execution_time_predictor = execution_time_predictor

        self._batch_queue = []
        self._is_busy = False

        self._active_batch: Optional[Batch] = None   # <-- NEW: track in-flight batch

    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    def is_empty(self) -> bool:
        return len(self._batch_queue) == 0

    def add_batch(self, batch: Batch) -> None:
        self._batch_queue.append(batch)

    def on_stage_end(self) -> None:
        self._is_busy = False
        self._active_batch = None                  # <-- clear active on completion

    def on_schedule(self) -> Tuple[Batch, BatchStage, ExecutionTime]:
        if self._is_busy or not self._batch_queue:
            return None, None, None

        self._is_busy = True
        batch = self._batch_queue.pop(0)
        self._active_batch = batch                 # <-- set active when scheduled
        execution_time = self._execution_time_predictor.get_batch_execution_time(
            batch,
            self._stage_id,
        )
        batch_stage = BatchStage(
            batch.id,
            self._replica_id,
            self._stage_id,
            execution_time,
            batch.requests,
            batch.num_tokens,
        )

        return batch, batch_stage, execution_time

    # --- Snapshot helpers -------------------------------------------------
    def snapshot_state(self) -> ReplicaStageSchedulerSnapshot:
        return ReplicaStageSchedulerSnapshot(
            __v__=_SNAP_VERSION_STAGE,
            batch_ids=[b.id for b in self._batch_queue],
            active_batch_id=(self._active_batch.id if self._active_batch is not None else None),
            is_busy=bool(self._is_busy),
        )

    def restore_state(
        self,
        snapshot: ReplicaStageSchedulerSnapshot,
        batch_lookup: Dict[int, Batch],
    ) -> None:
        assert int(snapshot.__v__) == _SNAP_VERSION_STAGE, "ReplicaStage snapshot version mismatch"

        # restore queued FIFO
        self._batch_queue = [batch_lookup[bid] for bid in snapshot.batch_ids]

        # restore active/in-flight
        active_id = snapshot.active_batch_id
        self._active_batch = batch_lookup[active_id] if active_id is not None else None

        # derive busy flag from active batch to avoid inconsistencies
        self._is_busy = (self._active_batch is not None)