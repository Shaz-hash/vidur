import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Request
from vidur.entities.batch import Batch
from vidur.entities.request import Request as RequestEntity
from vidur.entities.replica import Replica
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.kv_cache.disk_kv_cache_manager import DiskKVCacheManager
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from vidur.scheduler.replica_stage_scheduler.replica_stage_scheduler import (
    ReplicaStageScheduler,
)
from vidur.types.replica_id import ReplicaId
from vidur.utils.slo_manager import SLOManager


# near the imports
_SNAP_VERSION_GLOBAL_SCHED = 1


class BaseGlobalScheduler(ABC):
    def __init__(
        self,
        config: SimulationConfig,
        replicas: Dict[ReplicaId, Replica],
        execution_time_predictor=None,
    ):
        self._config = config
        self._replicas = replicas
        self._num_replicas = len(replicas)
        self._random_number_generator = random.Random(
            config.cluster_config.global_scheduler_config.seed
        )

        if execution_time_predictor is None:
            execution_time_predictor = ExecutionTimePredictorRegistry.get(
                config.execution_time_predictor_config.get_type(),
                predictor_config=config.execution_time_predictor_config,
                replica_config=config.cluster_config.replica_config,
                cache_config=config.cluster_config.cache_config,
            )
        self._execution_time_predictor = execution_time_predictor
        self._replica_schedulers: Dict[ReplicaId, BaseReplicaScheduler] = {
            replica_id: ReplicaSchedulerRegistry.get_from_str(
                self._config.cluster_config.replica_scheduler_config.get_type(),
                replica_config=self._config.cluster_config.replica_config,
                replica_scheduler_config=self._config.cluster_config.replica_scheduler_config,
                request_generator_config=self._config.request_generator_config,
                cache_config=self._config.cluster_config.cache_config,
                request_queue_config=self._config.cluster_config.request_queue_config,
                replica=replica,
                execution_time_predictor=self._execution_time_predictor,
            )
            for replica_id, replica in self._replicas.items()
        }

        if self._config.cluster_config.cache_config.enable_disk_caching:
            disk_kv_cache_manager = DiskKVCacheManager(
                block_size=self._config.cluster_config.cache_config.block_size,
                num_gpu_blocks=self._config.cluster_config.cache_config.disk_num_blocks,
                enable_caching=self._config.cluster_config.cache_config.enable_prefix_caching,
                caching_hash_algo=self._config.cluster_config.cache_config.prefix_caching_hash_algo,
                num_preallocate_tokens=self._config.cluster_config.cache_config.num_preallocate_tokens,
            )
            for _, replica in self._replica_schedulers.items():
                replica.set_disk_kv_cache(disk_kv_cache_manager)

        self._request_queue: List[Request] = []
        self._slo_manager = SLOManager(self._config.slo_config)

        # --- NEW: build a robust alias map for schedulers ---
        self._build_replica_alias_map()

    @staticmethod
    def _replica_key(rid) -> str:
        """Stable string key for a replica id (prefer explicit integer payload)."""
        if hasattr(rid, "id"):
            return str(int(rid.id))
        if hasattr(rid, "_id"):
            return str(int(rid._id))
        return str(rid)

    def _build_replica_alias_map(self) -> None:
        """Create a mapping from multiple possible serialized keys -> scheduler."""
        self._replica_alias: Dict[str, BaseReplicaScheduler] = {}
        for idx, (rid, sched) in enumerate(self._replica_schedulers.items()):
            # Preferred key
            self._replica_alias[self._replica_key(rid)] = sched
            # Common alternates
            self._replica_alias[str(rid)] = sched
            if hasattr(rid, "id"):
                self._replica_alias[str(int(rid.id))] = sched
            if hasattr(rid, "_id"):
                self._replica_alias[str(int(rid._id))] = sched
            # Index fallback (handles snapshots that used "0","1",...)
            self._replica_alias[str(idx)] = sched

    def _resolve_scheduler(self, replica_id: ReplicaId) -> BaseReplicaScheduler:
        """Resolve possibly-mismatched ReplicaId to a known scheduler."""
        # Direct dict hit
        if replica_id in self._replica_schedulers:
            return self._replica_schedulers[replica_id]

        # Alias hits (string forms)
        candidates = [
            self._replica_key(replica_id),
            str(replica_id),
        ]
        if hasattr(replica_id, "id"):
            candidates.append(str(int(replica_id.id)))
        if hasattr(replica_id, "_id"):
            candidates.append(str(int(replica_id._id)))

        for k in candidates:
            if k in self._replica_alias:
                return self._replica_alias[k]

        # Single-replica fallback: map anything to the only scheduler
        if len(self._replica_schedulers) == 1:
            return next(iter(self._replica_schedulers.values()))

        # As a last resort, try indexing by the integer value if it parses
        try:
            idx = int(getattr(replica_id, "id", replica_id))
            k = str(idx)
            if k in self._replica_alias:
                return self._replica_alias[k]
        except Exception:
            pass

        # Give a helpful error
        known = sorted(self._replica_alias.keys())
        raise KeyError(
            f"Unknown replica_id {replica_id}. Known aliases: {known[:10]}..."
        )

    def sort_requests(self) -> None:
        self._request_queue.sort(key=lambda x: (x.arrived_at, x.id))

    def add_request(self, request: Request) -> None:
        # This is the first instance the request comes into contact with the system
        self._slo_manager.set_slos(request)
        self._request_queue.append(request)

    def on_batch_end(self, batch: Batch) -> None:
        pass

    def on_prefill_end(self, request: Request) -> None:
        pass

    def on_request_end(self, request: Request) -> None:
        pass

    def get_replica_scheduler(self, replica_id: ReplicaId) -> BaseReplicaScheduler:
        return self._resolve_scheduler(replica_id)

    def get_replica_stage_scheduler(
        self, replica_id: ReplicaId, stage_id: int
    ) -> ReplicaStageScheduler:
        return self._resolve_scheduler(replica_id).get_replica_stage_scheduler(
            stage_id
        )

    def is_empty(self) -> bool:
        return len(self._request_queue) == 0 and all(
            replica_scheduler.is_empty()
            for replica_scheduler in self._replica_schedulers.values()
        )

    @abstractmethod
    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        pass

    # --- Snapshot helpers -------------------------------------------------
    def snapshot_state(self) -> dict:
        request_states: Dict[int, dict] = {}
        request_queue_ids: List[int] = []

        # Global queue: preserve order and capture states
        for request in self._request_queue:
            request_queue_ids.append(request.id)
            if request.id not in request_states:
                request_states[request.id] = request.snapshot_state()

        # Per-replica snapshots; allow dataclass or dict
        replica_snapshots: Dict[str, object] = {}
        for replica_id, replica_scheduler in self._replica_schedulers.items():
            if not hasattr(replica_scheduler, "snapshot_state"):
                raise NotImplementedError(
                    f"{replica_scheduler.__class__.__name__} does not implement snapshot_state()"
                )
            rsnap = replica_scheduler.snapshot_state()
            # pull request_states without assuming type
            rsnap_req_states = None
            if hasattr(rsnap, "request_states"):
                rsnap_req_states = getattr(rsnap, "request_states")
            elif isinstance(rsnap, dict):
                rsnap_req_states = rsnap.get("request_states")

            if isinstance(rsnap_req_states, dict):
                for rid, rstate in rsnap_req_states.items():
                    # don't clobber if we already took a state for this request
                    request_states.setdefault(rid, rstate)

            # key by string to be stable across restore
            # replica_snapshots[str(replica_id)] = rsnap
            replica_snapshots[self._replica_key(replica_id)] = rsnap

        return {
            "__v__": _SNAP_VERSION_GLOBAL_SCHED,
            "rng_state": self._random_number_generator.getstate(),
            "request_queue": request_queue_ids,
            "replica_schedulers": replica_snapshots,
            "request_states": request_states,
            "extra_state": self._snapshot_extra_state(),
        }



    def restore_state(self, snapshot: dict, request_lookup, batch_lookup) -> None:
        if "__v__" in snapshot:
            assert int(snapshot["__v__"]) == _SNAP_VERSION_GLOBAL_SCHED, \
                "BaseGlobalScheduler snapshot version mismatch"

        # RNG + global queue
        self._random_number_generator.setstate(snapshot["rng_state"])
        self._request_queue = [request_lookup[rid] for rid in snapshot["request_queue"]]

        # Build a robust alias map for schedulers present in THIS process
        by_key: Dict[str, BaseReplicaScheduler] = {}

        def add_alias(k: object, sched: BaseReplicaScheduler) -> None:
            try:
                by_key[str(k)] = sched
            except Exception:
                pass

        for idx, (rid, sched) in enumerate(self._replica_schedulers.items()):
            add_alias(self._replica_key(rid), sched)   # preferred ("1", "2", ...)
            add_alias(rid, sched)                      # "ReplicaId(id=1)"
            if hasattr(rid, "id"):   add_alias(int(rid.id), sched)   # "1"
            if hasattr(rid, "_id"):  add_alias(int(rid._id), sched)  # "1"
            add_alias(idx, sched)                      # index fallback: "0","1",...

        # Restore each replica scheduler; accept multiple key encodings
        for key, rsnap in snapshot["replica_schedulers"].items():
            target = by_key.get(key)
            if target is None:
                # final numeric fallback: e.g. "0" -> index 0
                try:
                    idx = int(key)
                    target = by_key.get(str(idx))
                except Exception:
                    target = None

            if target is None:
                # single-replica fallback: map anything to the only scheduler
                if len(self._replica_schedulers) == 1:
                    target = next(iter(self._replica_schedulers.values()))
                else:
                    sample = sorted(list(by_key.keys()))[:10]
                    raise KeyError(f"Replica key '{key}' not found. Known keys (sample): {sample}")

            restore_fn = getattr(target, "restore_state", None)
            if restore_fn is None:
                raise NotImplementedError(f"{target.__class__.__name__} lacks restore_state()")
            try:
                restore_fn(rsnap, request_lookup, batch_lookup)
            except TypeError:
                try:
                    restore_fn(rsnap, request_lookup)
                except TypeError:
                    restore_fn(rsnap)

        # Restore any subclass-specific fields (e.g., RoundRobin counter)
        self._restore_extra_state(snapshot.get("extra_state", {}))

        # Now refresh the instance-wide alias map used by get_replica_* calls
        self._build_replica_alias_map()



    def _snapshot_extra_state(self) -> dict:
        return {}

    def _restore_extra_state(self, snapshot: dict) -> None:
        pass
