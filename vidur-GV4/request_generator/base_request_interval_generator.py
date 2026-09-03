import copy
from abc import ABC, abstractmethod
from typing import Generator

from vidur.config import BaseRequestIntervalGeneratorConfig


class BaseRequestIntervalGenerator(ABC):
    def __init__(
        self,
        config: BaseRequestIntervalGeneratorConfig,
        random_number_generator: Generator,
    ):
        self._config = config
        self._random_number_generator = random_number_generator

    @abstractmethod
    def get_next_inter_request_time(self) -> float:
        pass

    def snapshot_state(self) -> dict:
        return {
            "rng_state": copy.deepcopy(self._random_number_generator.bit_generator.state),
            "extra_state": self._snapshot_extra_state(),
        }

    def restore_state(self, snapshot: dict) -> None:
        self._random_number_generator.bit_generator.state = copy.deepcopy(
            snapshot["rng_state"]
        )
        self._restore_extra_state(snapshot.get("extra_state", {}))

    def _snapshot_extra_state(self) -> dict:
        return {}

    def _restore_extra_state(self, snapshot: dict) -> None:
        return
