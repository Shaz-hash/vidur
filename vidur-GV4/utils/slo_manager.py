from vidur.config.config import SloConfig
from vidur.entities.request import Request


class SLOManager:
    def __init__(self, slo_config: SloConfig) -> None:
        self._config = slo_config

    def set_slos(self, request: Request):
        # Default placeholders when SLOs are not provided
        request.decode_slo_time = (
            float(self._config.decode_time_slo)
            if self._config.decode_time_slo is not None
            else -1.0
        )
        request.completion_slo_time = (
            float(self._config.completion_time_slo)
            if self._config.completion_time_slo is not None
            else -1.0
        )

        if self._config.prefill_e2e_time_normalized is not None:
            request.prefill_slo_time = (
                request.num_prefill_tokens * self._config.prefill_e2e_time_normalized
            )
        if self._config.prefill_e2e_time_min is not None:
            if hasattr(request, "prefill_slo_time"):
                request.prefill_slo_time = max(
                    request.prefill_slo_time, self._config.prefill_e2e_time_min
                )
            else:
                request.prefill_slo_time = self._config.prefill_e2e_time_min
