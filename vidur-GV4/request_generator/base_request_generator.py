import copy
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import numpy as np

from vidur.config import BaseRequestGeneratorConfig
from vidur.entities import Request

_SNAP_VERSION_REQGEN = 1


def _encode_rng_state(gen: np.random.Generator) -> Dict[str, Any]:
    """
    Convert numpy Generator bit_generator.state into a JSON-friendly dict.
    (For most bit generators this is already plain ints, but this keeps us safe
    if a future backend adds arrays.)
    """
    def to_prim(x):
        try:
            import numpy as _np
            if isinstance(x, _np.ndarray):
                return x.tolist()
        except Exception:
            pass
        if isinstance(x, dict):
            return {k: to_prim(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [to_prim(v) for v in x]
        return x

    raw = gen.bit_generator.state
    return to_prim(copy.deepcopy(raw))


def _decode_rng_state(gen: np.random.Generator, state: Dict[str, Any]) -> None:
    """
    Restore the bit_generator.state from the JSON-friendly dict.
    Numpy accepts plain Python containers for the state.
    """
    gen.bit_generator.state = copy.deepcopy(state)


class BaseRequestGenerator(ABC):
    def __init__(self, config: BaseRequestGeneratorConfig):
        self._config = config
        self._random_number_generator = np.random.default_rng(config.seed)

    @abstractmethod
    def get_next_request_arrival_time(self) -> Optional[float]:
        pass

    @abstractmethod
    def get_next_request(self) -> Optional[Request]:
        pass

    # --- Snapshot helpers -------------------------------------------------
    def snapshot_state(self) -> dict:
        return {
            "__v__": _SNAP_VERSION_REQGEN,
            "rng_state": _encode_rng_state(self._random_number_generator),
            "extra_state": self._snapshot_extra_state(),
        }

    def restore_state(self, snapshot: dict, request_lookup: Dict[int, Request]) -> None:
        # versioning: accept older snapshots (no version) by default
        snap_ver = int(snapshot.get("__v__", 1))
        if snap_ver != _SNAP_VERSION_REQGEN:
            # If you ever change formats, handle migrations here.
            pass

        if "rng_state" in snapshot:
            _decode_rng_state(self._random_number_generator, snapshot["rng_state"])

        self._restore_extra_state(snapshot.get("extra_state", {}), request_lookup)

    def _snapshot_extra_state(self) -> dict:
        """
        Subclasses override to persist their own queue/backlog state.
        Must return only JSON-safe primitives (dict/list/str/int/float/bool/None).
        """
        return {}

    def _restore_extra_state(self, snapshot: dict, request_lookup: Dict[int, Request]) -> None:
        """
        Subclasses override to rebuild from snapshot. Use request_lookup to resolve IDs to Request objects
        if you store request IDs in extra_state.
        """
        pass