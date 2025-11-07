from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence




@dataclass
class RequestSLOOptions:
    """Discrete SLO options available to the adversary."""

    prefill_slos: Sequence[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])
    decode_slos: Sequence[float] = field(default_factory=lambda: [1, 2, 5])


@dataclass
class MCTSConstraintConfig:
    """Bounds that govern admissible controller/adversary actions."""

    maximum_qps: int = 12
    min_request_tokens: int = 64
    max_request_tokens: Optional[int] = None
    interval_request_size: int = 512
    request_slo_options: RequestSLOOptions = field(default_factory=RequestSLOOptions)
    prefill_slowdown: float = 3.0
    prefill_profile_path: Optional[str] = None


@dataclass
class MCTSExploreConfig:
    """Parameters that shape the Monte-Carlo Tree Search behaviour."""

    simulation_depth: int = 4
    simulation_random_tries: int = 1
    exploration_constant: float = 4.0
    max_branching: int = 25  # Cap number of candidate actions per node 
    controller_budget_combs: int = 10 # for each controller's selected total_token_budget, selected_ids , have atleast upto these number of compositions of token budget distribution
