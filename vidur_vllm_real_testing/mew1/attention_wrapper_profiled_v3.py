"""Ray-safe Vidur attention profiler for Sarathi with FlashInfer 0.2."""

import sarathi.metrics.cuda_timer
import torch

from vidur.profiling.common.cuda_timer import CudaTimer


sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from vidur.profiling.attention.attention_wrapper import (  # noqa: E402
    AttentionWrapper as _AttentionWrapper,
)
import sarathi.model_executor.attention.base_attention_wrapper as _base_attention  # noqa: E402
import sarathi.model_executor.attention.flashinfer_attention_wrapper as _sarathi_flashinfer  # noqa: E402
from flashinfer import append_paged_kv_cache as _append_paged_kv_cache  # noqa: E402


_base_attention.CudaTimer = CudaTimer
_latest_batch_indices = None
_latest_positions = None
_original_begin_forward = _sarathi_flashinfer.FlashinferAttentionWrapper.begin_forward


def _begin_forward_compat(self, sequence_metadata):
    """Precompute FlashInfer 0.2 append indices outside timed regions."""
    global _latest_batch_indices, _latest_positions

    result = _original_begin_forward(self, sequence_metadata)
    append_indptr = self.append_qo_indptr_tensor
    if append_indptr is None:
        _latest_batch_indices = None
        _latest_positions = None
        return result

    append_lengths = append_indptr[1:] - append_indptr[:-1]
    batch_size = append_lengths.numel()
    _latest_batch_indices = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=append_indptr.device),
        append_lengths,
    )
    page_counts = (
        self.append_kv_page_indptr_tensor[1:]
        - self.append_kv_page_indptr_tensor[:-1]
    )
    sequence_lengths = (
        (page_counts - 1) * self.block_size + self.append_kv_last_page_len_tensor
    )
    sequence_starts = sequence_lengths - append_lengths
    token_offsets = torch.arange(
        append_indptr[-1], dtype=torch.int32, device=append_indptr.device
    ) - torch.repeat_interleave(append_indptr[:-1], append_lengths)
    _latest_positions = (
        torch.repeat_interleave(sequence_starts, append_lengths) + token_offsets
    )
    return result


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
    del append_indptr
    if _latest_batch_indices is None or _latest_positions is None:
        raise RuntimeError("FlashInfer append indices were not initialized")
    return _append_paged_kv_cache(
        append_key,
        append_value,
        _latest_batch_indices,
        _latest_positions,
        paged_kv_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        kv_layout=kv_layout,
    )


_sarathi_flashinfer.FlashinferAttentionWrapper.begin_forward = _begin_forward_compat
_sarathi_flashinfer.append_paged_kv_cache = _append_paged_kv_cache_compat


class AttentionWrapper(_AttentionWrapper):
    """Attention profiler actor with standalone timing and API compatibility."""

