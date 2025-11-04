from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence




@dataclass
class RequestSLOOptions:
    """Discrete SLO options available to the adversary."""

    prefill_slos: Sequence[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])
    decode_slos: Sequence[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])
    completion_slos: Sequence[float] = field(default_factory=lambda: [5.0, 10.0, 20.0])


@dataclass
class MCTSConstraintConfig:
    """Bounds that govern admissible controller/adversary actions."""

    maximum_qps: int = 4
    min_request_tokens: int = 64
    max_request_tokens: int = 2048
    interval_request_size: int = 64
    request_slo_options: RequestSLOOptions = field(default_factory=RequestSLOOptions)


@dataclass
class MCTSExploreConfig:
    """Parameters that shape the Monte-Carlo Tree Search behaviour."""

    simulation_depth: int = 8
    simulation_random_tries: int = 16
    exploration_constant: float = 1.4
    max_branching: int = 1000  # Cap number of candidate actions per node 
    controller_budget_combs: int = 50 # for each controller's selected total_token_budget, selected_ids , have atleast upto these number of compositions of token budget distribution

