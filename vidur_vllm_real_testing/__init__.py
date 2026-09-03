"""Local contracts and preparation tools for real-vLLM GV3 testing."""

from .canonicalization import (
    CanonicalizationConfig,
    CanonicalizationError,
    PrefillProfile,
    canonicalize_decode_slo,
    canonicalize_decode_tokens,
    canonicalize_prefill_slo,
    canonicalize_prefill_tokens,
)
from .trace_contract import CanonicalTraceRequest, RawTraceRequest

__all__ = [
    "CanonicalTraceRequest",
    "CanonicalizationConfig",
    "CanonicalizationError",
    "PrefillProfile",
    "RawTraceRequest",
    "canonicalize_decode_slo",
    "canonicalize_decode_tokens",
    "canonicalize_prefill_slo",
    "canonicalize_prefill_tokens",
]
