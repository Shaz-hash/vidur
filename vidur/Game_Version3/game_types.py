from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AdversaryRequestSpec:
    prefill_tokens: int
    decode_tokens: int
    prefill_slo: float
    decode_slo: float


@dataclass
class AdversaryAction:
    requests: List[AdversaryRequestSpec] = field(default_factory=list)
    stop_decode_ids: List[int] = field(default_factory=list)


@dataclass
class ControllerAction:
    token_budget: int
    selected_request_ids: Optional[List[int]] = None
    token_allocations: Dict[int, int] = field(default_factory=dict)
    prefill_allocations: Dict[int, int] = field(default_factory=dict)
    decode_allocations: Dict[int, int] = field(default_factory=dict)
    heuristic: Optional[str] = None
    strategy: Optional[str] = None
    mapping: Optional[Tuple[int, ...]] = None
    _evicted_request_ids: List[int] = field(default_factory=list, repr=False)
