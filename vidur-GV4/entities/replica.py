from vidur.config import ReplicaConfig
from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger
from vidur.types.replica_id import ReplicaId

logger = init_logger(__name__)


class Replica(BaseEntity):
    def __init__(
        self,
        replica_config: ReplicaConfig,
    ) -> None:
        self._id = ReplicaId(Replica.generate_id())
        self._replica_config = replica_config

    @property
    def id(self) -> ReplicaId:
        return self._id

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "num_pipeline_stages": self._replica_config.num_pipeline_stages,
            "num_tensor_parallel_workers": self._replica_config.tensor_parallel_size,
        }
