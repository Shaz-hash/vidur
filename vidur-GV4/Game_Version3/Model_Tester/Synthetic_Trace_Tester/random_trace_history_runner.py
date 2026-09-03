from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...DNN.history_root import HistoryRootGenerator
from ...config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG, GameVersion2Config
from ...game_types import AdversaryAction, AdversaryRequestSpec
from ...logger.mctsDNN_logger import DNNMCTSIterationLogger
from ...multiProcessUtils import _build_env_and_simulator, _set_global_seeds
from .trace_loader import load_trace_csv, rebase_trace
from .trace_token_policy import effective_trace_request
from .trace_types import EffectiveTraceRequest, RandomTraceHistoryResult, TraceRequest


@dataclass(frozen=True)
class RandomTraceHistoryConfig:
    trace_csv: Path
    output_csv: Path
    summary_csv: Path
    time_limit_sec: float = 20.0
    max_steps: int = 4096
    max_trace_rows: int = 10_000
    seed: int = 2026
    token_policy: str = "clip"
    rebase_to_zero: bool = True


def _mask_to_list(mask: Any) -> list[bool]:
    if isinstance(mask, torch.Tensor):
        return [bool(x) for x in mask.to(dtype=torch.bool).cpu().tolist()]
    return [bool(x) for x in list(mask)]


class RandomTraceHistoryRunner:
    def __init__(self, cfg: RandomTraceHistoryConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(int(cfg.seed))
        self.pipeline_cfg = DEFAULT_MULTIPROCESS_TRAINING_CONFIG
        self.gv2: GameVersion2Config = self.pipeline_cfg.game_v2
        self.gv2.validate()

        _set_global_seeds(int(cfg.seed), torch_deterministic=False)
        self.simulator, self.env, self.constraints, _explore = _build_env_and_simulator(
            self.pipeline_cfg,
            use_virtual_env=True,
        )

        self.trace_rows: list[TraceRequest] = load_trace_csv(
            cfg.trace_csv,
            limit=int(cfg.max_trace_rows),
        )
        if bool(cfg.rebase_to_zero):
            self.trace_rows = rebase_trace(self.trace_rows)

        self.trace_ptr = 0
        self.pending: list[EffectiveTraceRequest] = []
        self.released_rows = 0
        self.launched_rows = 0

        self.iter_logger = DNNMCTSIterationLogger(cfg.output_csv, flush_every=1)
        self.history_logger = HistoryRootGenerator(env=self.env, iter_logger=self.iter_logger)

    def close(self) -> None:
        self.iter_logger.close()
        sim = getattr(self, "simulator", None)
        for method in ("shutdown", "close", "stop"):
            fn = getattr(sim, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break

    def _decode_slo_time(self) -> float:
        decode_slos = tuple(float(x) for x in (self.gv2.legacy_mcts.decode_slos or (50.0,)))
        return float(decode_slos[0]) / 1000.0 if decode_slos else 0.05

    def _action_index_for_adversary(self, *, launch_count: int, prefill_tokens: int | None) -> int:
        stop_rules = list(self.gv2.adversary_action.stop_rule_names)
        templates = sorted(int(x) for x in self.gv2.request.allowed_prefill_tokens)
        n_stop = len(stop_rules)
        stop_none_idx = stop_rules.index("stop_none") if "stop_none" in stop_rules else 0

        count = int(launch_count)
        if count <= 0:
            return int(stop_none_idx)

        if prefill_tokens is None:
            raise ValueError("prefill_tokens is required when launch_count > 0")
        template_idx = templates.index(int(prefill_tokens))
        return int(n_stop + (count - 1) * len(templates) * n_stop + template_idx * n_stop + stop_none_idx)

    def _release_due_trace_rows(self, tick: float) -> None:
        max_prefill = int(self.gv2.request.max_prefill_tokens_per_request)
        max_decode = int(self.gv2.request.max_decode_tokens_per_request)
        min_decode = int(self.gv2.request.min_decode_tokens_per_request)
        allowed = tuple(int(x) for x in self.gv2.request.allowed_prefill_tokens)

        while self.trace_ptr < len(self.trace_rows):
            row = self.trace_rows[self.trace_ptr]
            if float(row.arrived_at) > float(tick) + float(self.gv2.timing.eps):
                break
            self.pending.append(
                effective_trace_request(
                    row,
                    policy=str(self.cfg.token_policy),
                    allowed_prefill_tokens=allowed,
                    max_prefill_tokens=max_prefill,
                    min_decode_tokens=min_decode,
                    max_decode_tokens=max_decode,
                )
            )
            self.trace_ptr += 1
            self.released_rows += 1

    def _make_trace_adversary_action(self, state: Any) -> tuple[AdversaryAction, int, int]:
        tick = float(self.env._v2_current_adv_tick(state))
        self._release_due_trace_rows(tick)

        if not self.pending:
            idx = self._action_index_for_adversary(launch_count=0, prefill_tokens=None)
            return AdversaryAction(requests=[], stop_decode_ids=[]), int(idx), 0

        launches = self.env._v2_prune_recent_launches(state, tick)
        used_count, used_prefill = self.env._v2_window_usage(tick, launches)

        req_cap = int(self.gv2.timing.max_requests_per_launch_window)
        prefill_cap = int(self.gv2.request.target_prefill_tokens_per_request_avg_window) * req_cap
        per_tick_cap = int(self.gv2.adversary_action.max_launch_count_per_tick)

        available_count = max(0, min(per_tick_cap, req_cap - int(used_count)))
        if available_count <= 0:
            idx = self._action_index_for_adversary(launch_count=0, prefill_tokens=None)
            return AdversaryAction(requests=[], stop_decode_ids=[]), int(idx), 0

        available_prefill = max(0, prefill_cap - int(used_prefill))
        selected_indices: list[int] = []
        selected_prefill = 0
        for i, row in enumerate(self.pending):
            next_prefill = selected_prefill + int(row.effective_prefill_tokens)
            if next_prefill > int(available_prefill):
                break
            selected_indices.append(i)
            selected_prefill = int(next_prefill)
            if len(selected_indices) >= int(available_count):
                break

        if not selected_indices:
            idx = self._action_index_for_adversary(launch_count=0, prefill_tokens=None)
            return AdversaryAction(requests=[], stop_decode_ids=[]), int(idx), 0

        selected = [self.pending[i] for i in selected_indices]
        selected_set = set(selected_indices)
        self.pending = [row for i, row in enumerate(self.pending) if i not in selected_set]

        decode_slo = float(self._decode_slo_time())
        action = AdversaryAction(
            requests=[
                AdversaryRequestSpec(
                    prefill_tokens=int(row.effective_prefill_tokens),
                    decode_tokens=int(row.effective_decode_tokens),
                    prefill_slo=float(self.env._prefill_profile.lookup(int(row.effective_prefill_tokens))),
                    decode_slo=float(decode_slo),
                )
                for row in selected
            ],
            stop_decode_ids=[],
        )
        # Synthetic trace launches can have arbitrary prefill/decode values and
        # mixed request sizes, so they are intentionally not representable by a
        # GV3 adversary template action index.
        idx = -1
        self.launched_rows += int(len(selected))
        return action, int(idx), int(len(selected))

    def _valid_actions(self, state: Any, player: str) -> tuple[list[Any | None], list[int], list[bool]]:
        if str(player) == "controller":
            actions, mask = self.env.sample_controller_actions(state)
        else:
            actions, mask = self.env.sample_adversary_actions(state)
        mask_l = _mask_to_list(mask)
        valid = [i for i, ok in enumerate(mask_l) if ok and i < len(actions) and actions[i] is not None]
        return list(actions), valid, mask_l

    def _run_one_step(
        self,
        *,
        state: Any,
        player: str,
        depth: int,
        node_id: int,
        parent_node_id: int | None,
        game_id: int,
        root_id: int,
    ) -> tuple[Any, str, int, int, int | None, bool]:
        sim_before = float(state.simulator._time)
        player_before = str(player)
        depth_before = int(depth)

        if player_before == "adversary":
            _actions, valid, _mask = self._valid_actions(state, player_before)
            n_valid = len(valid)
            action, action_index, _created = self._make_trace_adversary_action(state)
            decision_time = float(self.env._v2_current_adv_tick(state))
            state = self.env.apply_adversary_action_only(state, action, inplace=True)
            next_player = "controller"
        else:
            actions, valid, _mask = self._valid_actions(state, player_before)
            if not valid:
                return state, player_before, int(depth), int(node_id), parent_node_id, False
            n_valid = len(valid)
            action_index = int(self.rng.choice(valid))
            action = actions[action_index]
            if action is None:
                return state, player_before, int(depth), int(node_id), parent_node_id, False
            decision_time = sim_before
            state = self.env.apply_controller_action_only(state, action, inplace=True)
            next_player = "adversary"

        depth_after = int(depth) + 1
        phase = self.history_logger._phase_label(player_before, action, int(n_valid))
        self.history_logger._log_step(
            game_id=int(game_id),
            root_id=int(root_id),
            root_depth=int(depth_before),
            node_depth=int(depth_after),
            node_id=int(node_id),
            parent_node_id=parent_node_id,
            player_acted=player_before,
            next_player=next_player,
            action_index=int(action_index),
            action=action,
            n_valid=int(n_valid),
            phase=str(phase),
            decision_state_time=float(decision_time),
            start_time=float(sim_before),
            end_time=float(state.simulator._time),
            state_after=state,
        )
        return state, next_player, int(depth_after), int(node_id) + 1, int(node_id), True

    def run(self) -> RandomTraceHistoryResult:
        self.cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.summary_csv.parent.mkdir(parents=True, exist_ok=True)

        state = self.env.initial_state()
        player = "adversary"
        depth = 0
        node_id = 0
        parent_node_id: int | None = None
        game_id = 1
        root_id = 0
        turns = 0
        end_reason = "max_steps"

        while turns < int(self.cfg.max_steps):
            now = float(state.simulator._time)
            if now >= float(self.cfg.time_limit_sec) - float(self.gv2.timing.eps):
                end_reason = "time_limit"
                break

            state, player, depth, node_id, parent_node_id, progressed = self._run_one_step(
                state=state,
                player=player,
                depth=depth,
                node_id=node_id,
                parent_node_id=parent_node_id,
                game_id=game_id,
                root_id=root_id,
            )
            if not progressed:
                end_reason = "no_valid_action"
                break
            turns += 1

            if (
                self.trace_ptr >= len(self.trace_rows)
                and not self.pending
                and not state.stats.active_request_ids
                and float(state.simulator._time) > float(self.cfg.time_limit_sec)
            ):
                end_reason = "trace_exhausted"
                break

        violations, lateness = self.env.evaluate_objective(state)
        result = RandomTraceHistoryResult(
            trace_csv=self.cfg.output_csv,
            summary_csv=self.cfg.summary_csv,
            input_trace_csv=self.cfg.trace_csv,
            rows_read=len(self.trace_rows),
            rows_released=int(self.released_rows),
            rows_launched=int(self.launched_rows),
            rows_pending=len(self.pending),
            turns=int(turns),
            final_sim_time=float(state.simulator._time),
            requests_generated=int(state.stats.requests_generated),
            requests_completed=int(state.stats.requests_completed),
            slo_violations=int(violations),
            total_lateness=float(lateness),
            end_reason=str(end_reason),
        )
        self._write_summary(result)
        return result

    def _write_summary(self, result: RandomTraceHistoryResult) -> None:
        fields = [
            "trace_csv",
            "input_trace_csv",
            "rows_read",
            "rows_released",
            "rows_launched",
            "rows_pending",
            "turns",
            "final_sim_time",
            "requests_generated",
            "requests_completed",
            "slo_violations",
            "total_lateness",
            "end_reason",
        ]
        with result.summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "trace_csv": str(result.trace_csv),
                    "input_trace_csv": str(result.input_trace_csv),
                    "rows_read": int(result.rows_read),
                    "rows_released": int(result.rows_released),
                    "rows_launched": int(result.rows_launched),
                    "rows_pending": int(result.rows_pending),
                    "turns": int(result.turns),
                    "final_sim_time": float(result.final_sim_time),
                    "requests_generated": int(result.requests_generated),
                    "requests_completed": int(result.requests_completed),
                    "slo_violations": int(result.slo_violations),
                    "total_lateness": float(result.total_lateness),
                    "end_reason": str(result.end_reason),
                }
            )


def run_random_trace_history(cfg: RandomTraceHistoryConfig) -> RandomTraceHistoryResult:
    runner = RandomTraceHistoryRunner(cfg)
    try:
        return runner.run()
    finally:
        runner.close()
