from typing import List

from vidur.entities import Batch
from vidur.events import BaseEvent
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
from vidur.types.replica_id import ReplicaId


class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: ReplicaId, batch: Batch):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch

    def handle_event(
        self, global_scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.prefill_end_event import PrefillEndEvent
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent
        from vidur.events.request_end_event import RequestEndEvent

        self._batch.on_batch_end(self.time)
        global_scheduler.on_batch_end(self._batch)
        replica_scheduler = global_scheduler.get_replica_scheduler(self._replica_id)
        replica_scheduler.on_batch_end(self._batch)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )

        # Attempt to pull requests for the next batch
        next_events = [ReplicaScheduleEvent(self.time, self._replica_id)]
        for request in self._batch.completed_prefills:
            next_events.append(PrefillEndEvent(self.time, request))
        for request in self._batch.completed_requests:
            next_events.append(RequestEndEvent(self.time, request))

        return next_events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "batch_id": self._batch.id,
        }
