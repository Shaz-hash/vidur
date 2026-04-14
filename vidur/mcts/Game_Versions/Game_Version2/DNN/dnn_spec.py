from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import DEFAULT_GAME_V2_CONFIG, GameVersion2Config
from ..player_sample_actions import (
    adversary_action_space_size,
    controller_action_space_size,
)

Player = str  # "controller" | "adversary"


@dataclass(frozen=True)
class DNNGameSpec:
    # -------- Feature schema --------
    n_prefill_req: int
    d_prefill_req: int
    n_decode_req: int
    d_decode_req: int
    d_global: int

    # -------- Flat policy dims (for MCTS/replay compatibility) --------
    num_actions_controller: int
    num_actions_adversary: int

    # -------- Factorized head cardinalities --------
    adv_launch_size: int      # 0..max_launch_count_per_tick  => max+1
    adv_template_size: int    # len(allowed_prefill_tokens)
    adv_stop_size: int        # len(stop rules)

    ctrl_evict_size: int      # len(eviction rules)
    ctrl_budget_size: int     # len(prefill budgets)
    ctrl_heur_size: int       # len(heuristics)

    # -------- Embedding/trunk --------
    d_req_emb: int = 16
    d_global_emb: int = 16
    d_trunk: int = 64
    d_cond_emb: int = 16

    # -------- Value scaling --------
    v_max: float = 0.0
    v_min: float = -50.0
    v_linear_min: float = -48.0
    v_norm_min: float = -1.0
    v_norm_max: float = 0.0
    v_linear_norm_min: float = -0.98
    v_tail_compress_power: float = 2.0

    @property
    def n_req_total(self) -> int:
        return int(self.n_prefill_req + self.n_decode_req)

    def validate(self) -> None:
        if self.n_prefill_req <= 0:
            raise ValueError("n_prefill_req must be > 0")
        if self.d_prefill_req <= 0:
            raise ValueError("d_prefill_req must be > 0")
        if self.n_decode_req <= 0:
            raise ValueError("n_decode_req must be > 0")
        if self.d_decode_req <= 0:
            raise ValueError("d_decode_req must be > 0")
        if self.d_global <= 0:
            raise ValueError("d_global must be > 0")

        if self.num_actions_controller <= 0 or self.num_actions_adversary <= 0:
            raise ValueError("flat action dims must be > 0")
        if self.adv_launch_size <= 0 or self.adv_template_size <= 0 or self.adv_stop_size <= 0:
            raise ValueError("adversary factor sizes must be > 0")
        if self.ctrl_evict_size <= 0 or self.ctrl_budget_size <= 0 or self.ctrl_heur_size <= 0:
            raise ValueError("controller factor sizes must be > 0")

        # consistency checks with flattening contracts
        expected_adv = self.adv_stop_size + (self.adv_launch_size - 1) * self.adv_template_size * self.adv_stop_size
        if expected_adv != self.num_actions_adversary:
            raise ValueError(
                f"adv flat mismatch: expected {expected_adv}, got {self.num_actions_adversary}"
            )

        expected_ctrl = self.ctrl_evict_size * self.ctrl_budget_size * self.ctrl_heur_size
        if expected_ctrl != self.num_actions_controller:
            raise ValueError(
                f"ctrl flat mismatch: expected {expected_ctrl}, got {self.num_actions_controller}"
            )

    # -------------------------
    # Flat indexing helpers
    # -------------------------
    def adv_flat_index(self, launch_idx: int, template_idx: int, stop_idx: int) -> int:
        """
        launch_idx in [0, adv_launch_size-1]
        template_idx in [0, adv_template_size-1]
        stop_idx in [0, adv_stop_size-1]
        Flatten order matches sampler:
          launch=0: stop only
          launch>=1: for launch, for template, for stop
        """
        l = int(launch_idx)
        t = int(template_idx)
        s = int(stop_idx)

        if l < 0 or l >= self.adv_launch_size:
            raise IndexError("launch_idx out of range")
        if s < 0 or s >= self.adv_stop_size:
            raise IndexError("stop_idx out of range")
        if t < 0 or t >= self.adv_template_size:
            raise IndexError("template_idx out of range")

        if l == 0:
            return s
        base = self.adv_stop_size
        return base + (l - 1) * (self.adv_template_size * self.adv_stop_size) + t * self.adv_stop_size + s

    def ctrl_flat_index(self, evict_idx: int, budget_idx: int, heur_idx: int) -> int:
        e = int(evict_idx)
        b = int(budget_idx)
        h = int(heur_idx)

        if e < 0 or e >= self.ctrl_evict_size:
            raise IndexError("evict_idx out of range")
        if b < 0 or b >= self.ctrl_budget_size:
            raise IndexError("budget_idx out of range")
        if h < 0 or h >= self.ctrl_heur_size:
            raise IndexError("heur_idx out of range")

        return (e * self.ctrl_budget_size + b) * self.ctrl_heur_size + h



def make_dnn_spec(
    *,
    cfg: Optional[GameVersion2Config] = None,
) -> DNNGameSpec:
    c = cfg or DEFAULT_GAME_V2_CONFIG
    c.validate()

    spec = DNNGameSpec(
        n_prefill_req=int(c.features.n_prefill_req),
        d_prefill_req=int(c.features.d_prefill_req),
        n_decode_req=int(c.features.n_decode_req),
        d_decode_req=int(c.features.d_decode_req),
        d_global=int(c.features.d_global),

        num_actions_controller=int(controller_action_space_size(c)),
        num_actions_adversary=int(adversary_action_space_size(c)),

        adv_launch_size=int(c.adversary_action.max_launch_count_per_tick) + 1,
        adv_template_size=len(c.request.allowed_prefill_tokens),
        adv_stop_size=len(c.adversary_action.stop_rule_names),

        ctrl_evict_size=len(c.controller_action.eviction_rule_names),
        ctrl_budget_size=len(c.controller_action.prefill_budget_options),
        ctrl_heur_size=len(c.controller_action.ordering_heuristics),
    )
    spec.validate()
    return spec





DEFAULT_DNN_SPEC: DNNGameSpec = make_dnn_spec()


def num_actions_for_player(player: Player, *, spec: DNNGameSpec = DEFAULT_DNN_SPEC) -> int:
    if player == "controller":
        return int(spec.num_actions_controller)
    if player == "adversary":
        return int(spec.num_actions_adversary)
    raise ValueError(f"Unknown player={player!r}")
