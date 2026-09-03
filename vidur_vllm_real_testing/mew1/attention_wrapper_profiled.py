"""Ray-importable attention wrapper with Vidur's standalone timer installed."""

import sarathi.metrics.cuda_timer

from vidur.profiling.common.cuda_timer import CudaTimer


sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from vidur.profiling.attention.attention_wrapper import (  # noqa: E402
    AttentionWrapper as _AttentionWrapper,
)
import sarathi.model_executor.attention.base_attention_wrapper as _base_attention  # noqa: E402


# Handle both fresh and previously imported Sarathi modules in Ray workers.
_base_attention.CudaTimer = CudaTimer


class AttentionWrapper(_AttentionWrapper):
    """Attention profiler actor that records into Vidur's TimerStatsStore."""

