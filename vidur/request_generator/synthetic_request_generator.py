# from collections import deque
# from typing import Optional

from collections import deque
from typing import Optional, Dict, Any, Deque, List

from vidur.config import SyntheticRequestGeneratorConfig
from vidur.entities import Request
from vidur.request_generator.base_request_generator import BaseRequestGenerator
from vidur.request_generator.request_interval_generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from vidur.request_generator.request_length_generator_registry import (
    RequestLengthGeneratorRegistry,
)


_SNAP_VERSION_SYNTHETIC_EXTRA = 1

class SyntheticRequestGenerator(BaseRequestGenerator):
    def __init__(self, config: SyntheticRequestGeneratorConfig):
        super().__init__(config)

        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            self._config.length_generator_config.get_type(),
            self._config.length_generator_config,
            self._random_number_generator,
        )
        self.request_interval_generator = RequestIntervalGeneratorRegistry.get(
            self._config.interval_generator_config.get_type(),
            self._config.interval_generator_config,
            self._random_number_generator,
        )
        self.requests = deque()
        self.last_arrived_at = 0
        self.num_requests_generated = 0

    # Attempt to generate a new request and append it to the queue of requests
    def _generate_next_request(self) -> None:
        if self._config.num_requests is not None:
            if self.num_requests_generated >= self._config.num_requests:
                return

        if self._config.duration is not None:
            if self.last_arrived_at >= self._config.duration:
                return

        inter_request_time = (
            self.request_interval_generator.get_next_inter_request_time()
        )
        assert isinstance(inter_request_time, float)
        arrived_at = self.last_arrived_at + inter_request_time
        request_length_output = self.request_length_generator.get_next_num_tokens()

        self.last_arrived_at = arrived_at
        self.num_requests_generated += 1
        self.requests.append(
            Request(
                arrived_at=arrived_at,
                num_prefill_tokens=request_length_output.num_prefill_tokens,
                num_decode_tokens=request_length_output.num_decode_tokens,
                block_hash_ids=request_length_output.block_hash_ids,
                block_size=request_length_output.block_size,
                session_id=request_length_output.session_id,
            )
        )

    def get_next_request_arrival_time(self) -> Optional[float]:
        if len(self.requests) == 0:
            self._generate_next_request()

        return self.requests[0].arrived_at if len(self.requests) > 0 else None

    def get_next_request(self) -> Optional[Request]:
        if len(self.requests) == 0:
            self._generate_next_request()

        return self.requests.popleft() if len(self.requests) > 0 else None

    def _snapshot_extra_state(self) -> dict:
            extra: Dict[str, Any] = {
                "__v__": _SNAP_VERSION_SYNTHETIC_EXTRA,
                "requests": [req.id for req in self.requests],
                "last_arrived_at": float(self.last_arrived_at),
                "num_requests_generated": int(self.num_requests_generated),
                # Optional identity for safety:
                "interval_gen_type": type(self.request_interval_generator).__name__,
                "length_gen_type": type(self.request_length_generator).__name__,
            }
            if hasattr(self.request_interval_generator, "snapshot_state"):
                extra["interval_gen"] = self.request_interval_generator.snapshot_state()
            if hasattr(self.request_length_generator, "snapshot_state"):
                extra["length_gen"] = self.request_length_generator.snapshot_state()
            return extra

    def _restore_extra_state(self, snapshot: dict, request_lookup: Dict[int, Request]) -> None:
        v = int(snapshot.get("__v__", 1))
        if v != _SNAP_VERSION_SYNTHETIC_EXTRA:
            # add migrations here when schema changes
            raise ValueError(f"SyntheticRequestGenerator extra snapshot version mismatch: got {v}, expected {_SNAP_VERSION_SYNTHETIC_EXTRA}")

        self.last_arrived_at = float(snapshot.get("last_arrived_at", 0.0))
        self.num_requests_generated = int(snapshot.get("num_requests_generated", 0))
        self.requests = deque(request_lookup[rid] for rid in snapshot.get("requests", []))

        # Optional: sanity check generator identities to catch config drift
        if "interval_gen_type" in snapshot:
            expect = snapshot["interval_gen_type"]
            actual = type(self.request_interval_generator).__name__
            if actual != expect and hasattr(self.request_interval_generator, "restore_state"):
                # You can choose to warn or raise; raising is safest for determinism.
                raise ValueError(f"Interval generator type mismatch: saved {expect}, got {actual}")
        if "length_gen_type" in snapshot:
            expect = snapshot["length_gen_type"]
            actual = type(self.request_length_generator).__name__
            if actual != expect and hasattr(self.request_length_generator, "restore_state"):
                raise ValueError(f"Length generator type mismatch: saved {expect}, got {actual}")

        if "interval_gen" in snapshot and hasattr(self.request_interval_generator, "restore_state"):
            self.request_interval_generator.restore_state(snapshot["interval_gen"])
        if "length_gen" in snapshot and hasattr(self.request_length_generator, "restore_state"):
            self.request_length_generator.restore_state(snapshot["length_gen"])
