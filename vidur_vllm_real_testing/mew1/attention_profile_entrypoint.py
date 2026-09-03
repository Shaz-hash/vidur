"""Run Vidur attention profiling with Vidur's CUDA-event timer installed."""

import sarathi.metrics.cuda_timer
import sarathi.model_executor.attention.base_attention_wrapper

from vidur.profiling.common.cuda_timer import CudaTimer


# Sarathi's default timer requires a running server MetricsStore. Profiling uses
# Vidur's standalone TimerStatsStore instead, as the MLP profiler already does.
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer
sarathi.model_executor.attention.base_attention_wrapper.CudaTimer = CudaTimer

from vidur.profiling.attention.main import main  # noqa: E402


if __name__ == "__main__":
    main()
