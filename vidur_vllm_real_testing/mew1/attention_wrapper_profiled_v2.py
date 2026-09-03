"""Ray-importable attention profiler compatible with FlashInfer 0.2."""

import sarathi.metrics.cuda_timer

from vidur.profiling.common.cuda_timer import CudaTimer


sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from vidur.profiling.attention.attention_wrapper import (  # noqa: E402
    AttentionWrapper as _AttentionWrapper,
)
import sarathi.model_executor.attention.base_attention_wrapper as _base_attention  # noqa: E402
import sarathi.model_executor.attention.flashinfer_attention_wrapper as _sarathi_flashinfer  # noqa: E402
from flashinfer import (  # noqa: E402
    append_paged_kv_cache as _append_paged_kv_cache,
    get_batch_indices_positions,
    get_seq_lens,
)


_base_attention.CudaTimer = CudaTimer


def _append_paged_kv_cache_compat(
    append_key,
    append_value,
    append_indptr,
    paged_kv_cache,
    kv_indices,
    kv_indptr,
    kv_last_page_len,
    kv_layout="NHD",
):
    """Translate Sarathi's pre-0.2 append call to FlashInfer 0.2."""
    page_size = paged_kv_cache.shape[2] if kv_layout == "NHD" else paged_kv_cache.shape[3]
    sequence_lengths = get_seq_lens(kv_indptr, kv_last_page_len, page_size)
    batch_indices, positions = get_batch_indices_positions(
        append_indptr, sequence_lengths, append_key.shape[0]
    )
    return _append_paged_kv_cache(
        append_key,
        append_value,
        batch_indices,
        positions,
        paged_kv_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        kv_layout=kv_layout,
    )


_sarathi_flashinfer.append_paged_kv_cache = _append_paged_kv_cache_compat


class AttentionWrapper(_AttentionWrapper):
    """Attention profiler actor with standalone timing and API compatibility."""

