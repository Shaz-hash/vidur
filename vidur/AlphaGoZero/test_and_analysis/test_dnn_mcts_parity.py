"""Deterministic same-root Python/native MCTS parity for AGZ residual DNNs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import torch

from vidur.AlphaGoZero.dnn_models import (
    MarkovPolicyRankDeepSet,
    MarkovValueDeepSet,
    PolicyRankMLP,
    ValueResidualMLP,
    export_dnn_to_native,
)
from vidur.bellman_v4_adv import arena_mcts_value_runnerCPP as cpp_runner
from vidur.bellman_v4_adv.build_state_local_features_adv import (
    extract_features_one_record,
)
from vidur.Game_Version3.DNN import infer as infer_module
from vidur.Game_Version3.DNN.eval_utils import NoopReplayWriter
from vidur.Game_Version3.DNN.native_selfplay import (
    _cfg_payload,
    attach_execution_predictor_payload,
)
from vidur.Game_Version3.DNN.selfPlay import SelfPlayRunner
from vidur.Game_Version3.mcts_value_prior import (
    MCTSConfig,
    VidurMCTS,
)
from vidur.Game_Version3.mcts_value_prior_rollout import VidurMCTSPolicyRollout
from vidur.Game_Version3.tests import native_logger_tests as nlt


ALIGNMENT_DIR = Path(__file__).resolve().parents[2] / "tests" / "native_allignment_tests"
if str(ALIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ALIGNMENT_DIR))
from common import import_native_cpp, make_args, prepare_python_roots  # noqa: E402


class TracingVidurMCTS(VidurMCTS):
    """Test-only recorder for the exact path and return sent to backup."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.iteration_trace: list[dict[str, Any]] = []
        self.selection_trace: list[dict[str, Any]] = []

    def puct_select_child_or_untried(self, node: Any) -> Any:
        candidates: list[dict[str, Any]] = []
        for action_index, child in node.children.items():
            score = self._puct_score_child(node, int(action_index), child)
            candidates.append(
                {
                    "kind": "child",
                    "action_index": int(action_index),
                    "score": float(score[0]),
                    "prior": float(node.action_priors.get(int(action_index), 0.0)),
                    "visits": int(child.visits),
                    "q": float(child.mean_value()),
                }
            )
        for action_index in node.untried_action_indices:
            score = self._puct_score_untried(node, int(action_index))
            candidates.append(
                {
                    "kind": "untried",
                    "action_index": int(action_index),
                    "score": float(score[0]),
                    "prior": float(node.action_priors.get(int(action_index), 0.0)),
                    "visits": 0,
                    "q": 0.0,
                }
            )
        selected = super().puct_select_child_or_untried(node)
        path: list[int] = []
        cursor = node
        while cursor.parent is not None:
            path.append(int(cursor.parent_action_index))
            cursor = cursor.parent
        path.reverse()
        ranked = sorted(
            candidates,
            key=lambda item: (float(item["score"]), -int(item["action_index"])),
            reverse=True,
        )
        self.selection_trace.append(
            {
                "sim_iteration": len(self.iteration_trace),
                "path": path,
                "node_id": int(node.node_id),
                "player": str(node.player),
                "visits": int(node.visits),
                "min_value": float(node.min_value),
                "max_value": float(node.max_value),
                "selected_kind": str(selected[0]),
                "selected_action_index": int(selected[1]),
                "top_candidates": ranked[:8],
            }
        )
        return selected

    def _backpropagate(self, path: Any, leaf_bootstrap_value: float) -> None:
        root_value_sum_before = float(path[0].value_sum)
        edges = [
            {
                "node_id": int(node.node_id),
                "action_index": int(node.parent_action_index),
                "player": str(node.player),
                "reward": float(node.reward),
                "edge_discount": float(node.edge_discount),
                "state_cost": float(node.state_cost),
                "sim_time": float(node.sim_time),
            }
            for node in path[1:]
        ]
        super()._backpropagate(path, leaf_bootstrap_value)
        self.iteration_trace.append(
            {
                "sim_iteration": len(self.iteration_trace),
                "leaf_bootstrap": float(leaf_bootstrap_value),
                "root_return_delta": float(path[0].value_sum) - root_value_sum_before,
                "edges": edges,
            }
        )


class TracingPolicyRolloutMCTS(VidurMCTSPolicyRollout, TracingVidurMCTS):
    pass


def _assert_rollout_deadline_stats(
    stats: dict[str, Any],
    *,
    root_time: float,
    horizon: float,
    native: bool,
) -> None:
    """Verify equal rollout deadlines from each expansion parent."""
    if horizon <= 0.0:
        return
    prefix = "rollout_" if native else ""
    terminal_key = "rollout_terminals" if native else "terminal_trajectories"
    actual_root_time = float(stats[f"{prefix}root_time"])
    min_start_time = float(stats[f"{prefix}min_start_time"])
    max_start_time = float(stats[f"{prefix}max_start_time"])
    min_final_time = float(stats[f"{prefix}min_final_time"])
    max_final_time = float(stats[f"{prefix}max_final_time"])
    min_deadline = float(stats[f"{prefix}min_deadline"])
    max_deadline = float(stats[f"{prefix}max_deadline"])
    min_parent_time = float(stats[f"{prefix}min_expansion_parent_time"])
    max_parent_time = float(stats[f"{prefix}max_expansion_parent_time"])
    min_remaining = float(stats[f"{prefix}min_remaining_rollout_sec"])
    max_remaining = float(stats[f"{prefix}max_remaining_rollout_sec"])
    terminal_trajectories = int(stats[terminal_key])
    values = (
        actual_root_time,
        min_start_time,
        max_start_time,
        min_final_time,
        max_final_time,
        min_deadline,
        max_deadline,
        min_parent_time,
        max_parent_time,
        min_remaining,
        max_remaining,
    )
    if not all(math.isfinite(value) for value in values):
        raise AssertionError(f"non-finite rollout deadline stats: {stats}")
    tolerance = 1e-8
    if abs(actual_root_time - root_time) > tolerance:
        raise AssertionError(f"rollout root time mismatch: {stats}")
    if abs(min_deadline - (min_parent_time + horizon)) > tolerance:
        raise AssertionError(
            f"minimum expansion-parent deadline mismatch: {stats}"
        )
    if abs(max_deadline - (max_parent_time + horizon)) > tolerance:
        raise AssertionError(
            f"maximum expansion-parent deadline mismatch: {stats}"
        )
    if min_remaining < -tolerance or max_remaining > horizon + tolerance:
        raise AssertionError(f"remaining rollout is outside [0, horizon]: {stats}")
    if terminal_trajectories == 0 and min_final_time < min_deadline - tolerance:
        raise AssertionError(f"shortest rollout stopped before its deadline: {stats}")
    if terminal_trajectories == 0 and max_final_time < max_deadline - tolerance:
        raise AssertionError(f"deepest rollout stopped before its deadline: {stats}")


def _load(path: Path, expected: type[Any] | tuple[type[Any], ...]) -> Any:
    model = joblib.load(path)
    if not isinstance(model, expected):
        expected_names = (
            ", ".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise TypeError(f"{path} has {type(model).__name__}, expected {expected_names}")
    return model.cpu().eval()


def _state_features(state: Any, player: str) -> list[float]:
    inputs = infer_module.build_model_inputs(state, player, torch.device("cpu"))
    extras = getattr(inputs, "extras", None)
    if not extras:
        raise RuntimeError("Python MCTS input extras are not enabled")
    values = np.asarray(
        extract_features_one_record(
            {
                "simulator_snapshot": extras.get("simulator_snapshot") or {},
                "stats": extras.get("stats"),
                "root_id": -1,
            }
        ),
        dtype=np.float32,
    ).reshape(-1)
    if values.size != 226:
        raise RuntimeError(f"state feature count is {values.size}, expected 226")
    return values.tolist()


def _feature_builder(bundle: Any, expected_action_dim: int):
    def build(
        state: Any,
        player: str,
        actions_by_index: list[Any],
        canonical_indices: list[int],
    ) -> list[list[float]]:
        state_features = _state_features(state, player)
        rows: list[list[float]] = []
        for canon_idx in canonical_indices:
            action = actions_by_index[int(canon_idx)]
            action_features = cpp_runner._build_action_features_for_replay(
                bundle=bundle,
                state=state,
                player=player,
                action=action,
                canon_idx=int(canon_idx),
            )
            if len(action_features) != int(expected_action_dim):
                raise RuntimeError(
                    f"{player} action feature count is {len(action_features)}, "
                    f"expected {expected_action_dim}"
                )
            rows.append(state_features + action_features)
        return rows

    return build


def run(args: argparse.Namespace) -> dict[str, Any]:
    native = import_native_cpp(force_build=False)
    infer_module.enable_inputs_extras()

    value_types = (ValueResidualMLP, MarkovValueDeepSet)
    controller_value = _load(args.controller_value_model, value_types)
    adversary_value = _load(args.adversary_value_model, value_types)
    policy_types = (PolicyRankMLP, MarkovPolicyRankDeepSet)
    controller_policy = _load(args.controller_policy_model, policy_types)
    adversary_policy = _load(args.adversary_policy_model, policy_types)
    markov_policy = isinstance(controller_policy, MarkovPolicyRankDeepSet) or isinstance(
        adversary_policy, MarkovPolicyRankDeepSet
    )
    if markov_policy and not bool(args.native_only):
        raise ValueError(
            "structured Markov policy MCTS parity currently requires --native-only; "
            "direct Python/native policy and feature parity is checked separately"
        )

    root_args = make_args(
        "agz_dnn_mcts_alignment",
        num_roots=max(8, int(args.source_roots)),
        history_hops_min=0,
        history_hops_max=int(args.history_hops_max),
        history_seed=int(args.seed),
        frontier_parity_roots=1,
        model_version=100,
        checkpoint_path=str(args.controller_value_model),
    )
    pipeline_cfg, simulator, env, explore_cfg, roots = prepare_python_roots(root_args)
    requested_discount = float(args.discount_factor)
    pipeline_cfg = replace(
        pipeline_cfg,
        game_v2=replace(
            pipeline_cfg.game_v2,
            mcts_search=replace(
                pipeline_cfg.game_v2.mcts_search,
                discount_factor=requested_discount,
            ),
        ),
    )
    setattr(explore_cfg, "discount_factor", requested_discount)
    discount = float(pipeline_cfg.game_v2.mcts_search.discount_factor)
    if abs(discount - requested_discount) > 1e-12:
        raise AssertionError(
            f"pipeline discount={discount} does not match requested {requested_discount}"
        )
    original_build_model_inputs = infer_module.build_model_inputs

    def _build_model_inputs_with_markov(
        state: Any,
        player: str,
        device: Any,
        *positional: Any,
        **keyword: Any,
    ) -> Any:
        inputs = original_build_model_inputs(
            state, player, device, *positional, **keyword
        )
        if inputs.extras is None:
            raise RuntimeError("Markov parity requires ModelInputs.extras")
        inputs.extras["markov_state_payload"] = nlt._native_state_payload(env, state)
        return inputs

    infer_module.build_model_inputs = _build_model_inputs_with_markov


    root_runner = SelfPlayRunner(
        env=env,
        mcts=nlt.VidurMCTS(env=env, explore_cfg=explore_cfg),
        model=controller_value,
        writer=NoopReplayWriter(),
        device_for_features=torch.device("cpu"),
        game_v2_cfg=pipeline_cfg.game_v2,
    )

    selected = None
    matching_roots_seen = 0
    for prepared in roots:
        state = prepared.root_state
        player = str(prepared.root_player)
        depth = int(prepared.root_depth)
        state, player, depth = root_runner._advance_to_branching_root(
            state,
            player,
            depth,
            max_hops=int(pipeline_cfg.max_forced_hops_per_root),
        )
        if player == "adversary":
            state, _ = root_runner._build_root_decision_state_for_adversary(
                current_state=state,
                pre_controller_snapshot=prepared.pre_controller_snapshot,
                pre_controller_stats=prepared.pre_controller_stats,
            )
        actions, mask = (
            env.sample_controller_actions(state)
            if player == "controller"
            else env.sample_adversary_actions(state)
        )
        valid_count = sum(
            bool(mask[idx]) and actions[idx] is not None for idx in range(len(actions))
        )
        if valid_count > 1 and (
            str(args.root_player) == "any" or player == str(args.root_player)
        ):
            if matching_roots_seen < int(args.skip_matching_roots):
                matching_roots_seen += 1
                continue
            selected = (prepared, state, player, depth, valid_count)
            break
    if selected is None:
        raise RuntimeError(f"no multi-action {args.root_player} root found")
    prepared, state, player, depth, valid_count = selected

    bundle = SimpleNamespace(env=env, pipeline_cfg=pipeline_cfg)
    config = MCTSConfig()
    config.rng = random.Random(int(args.seed) + int(prepared.root_id))
    config.mcts_iterations = int(args.iterations)
    config.num_simulations = int(args.iterations)
    config.discount_factor = float(args.discount_factor)
    config._discount_time_denom = float(
        pipeline_cfg.game_v2.mcts_search.discount_time_denominator_sec
        or 0.015725797204323228
    )
    config.uct_c = float(args.uct_c)
    config.puct_c = float(args.puct_c)
    config.use_policy_prior = True
    config.policy_prior_temperature = 1.0
    config.prior_min_prob = 1e-8
    config.root_dirichlet_alpha = 0.0
    config.root_dirichlet_epsilon = 0.0
    config.controller_prior_model = controller_policy
    config.adversary_prior_model = adversary_policy
    config.controller_prior_feature_fn = _feature_builder(bundle, 43)
    config.adversary_prior_feature_fn = _feature_builder(bundle, 7)
    config.log_flag = False
    config.rollout_count = int(args.rollout_count)
    config.rollout_seed = int(args.seed) + int(prepared.root_id)
    config.rollout_horizon_sec = float(args.rollout_horizon_sec)
    config.rollout_policy_temperature = float(args.rollout_policy_temperature)
    config.rollout_probability_quantum = float(args.rollout_probability_quantum)
    config.rollout_max_actions = int(args.rollout_max_actions)

    value_model = controller_value if player == "controller" else adversary_value
    mcts_class = (
        TracingPolicyRolloutMCTS if int(args.rollout_count) > 0 else TracingVidurMCTS
    )
    python_mcts = None
    python_result = None
    python_root = None
    python_elapsed = 0.0
    if not bool(args.native_only):
        python_mcts = mcts_class(env=env, mctsConfig=config)
        python_start = time.perf_counter()
        python_result = python_mcts.search_dnn(
            dnn_model=value_model,
            rootState=state.fork(flag=False),
            root_player=player,
            game_id=0,
            root_id=int(prepared.root_id),
            root_node_id_override=prepared.root_node_id_override,
            root_depth=int(depth),
            mcts_iter=int(args.iterations),
            model_version=100,
            use_model_bootstrap=True,
            one_step_value_mode=False,
        )
        python_elapsed = time.perf_counter() - python_start
        python_root = python_mcts._root
        if python_root is None:
            raise RuntimeError("Python MCTS did not retain its root")
    infer_module.build_model_inputs = original_build_model_inputs

    tmp = Path(args.output_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    value_export = export_dnn_to_native(value_model, tmp / "value.tsv")
    controller_export = export_dnn_to_native(controller_policy, tmp / "controller.tsv")
    adversary_export = export_dnn_to_native(adversary_policy, tmp / "adversary.tsv")
    value_runtime = native.NewFeatures226HGBRuntime()
    value_runtime.load_model_export(str(value_export))
    controller_runtime = native.NativeHGBModelRuntime()
    controller_runtime.load_model_export(str(controller_export))
    adversary_runtime = native.NativeHGBModelRuntime()
    adversary_runtime.load_model_export(str(adversary_export))

    payload = _cfg_payload(pipeline_cfg, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    payload["discount_factor"] = float(args.discount_factor)
    payload["use_model_bootstrap"] = True
    payload["use_policy_prior"] = True
    payload["native_search_mode"] = str(args.native_search_mode or (
        "full_tree_rollout" if int(args.rollout_count) > 0 else "full_tree"
    ))
    payload["rollout_count"] = int(args.rollout_count)
    payload["rollout_parallel_threads"] = int(args.rollout_parallel_threads)
    payload["rollout_policy_parallel_threads"] = int(args.rollout_policy_parallel_threads)
    payload["rollout_horizon_sec"] = float(args.rollout_horizon_sec)
    payload["rollout_policy_temperature"] = float(args.rollout_policy_temperature)
    payload["rollout_probability_quantum"] = float(args.rollout_probability_quantum)
    payload["rollout_max_actions"] = int(args.rollout_max_actions)
    payload["rollout_optimized_execution"] = bool(args.rollout_optimized_execution)
    payload["capture_rollout_trace"] = bool(args.capture_rollout_trace)
    payload["root_dirichlet_noise_enabled"] = False
    payload["pb_c_base"] = float(pipeline_cfg.game_v2.mcts_search.pb_c_base)
    payload["pb_c_init"] = float(pipeline_cfg.game_v2.mcts_search.pb_c_init)
    payload["uct_c"] = float(args.uct_c)
    payload["puct_c"] = float(args.puct_c)
    payload["policy_prior_temperature"] = 1.0
    payload["prior_min_prob"] = 1e-8
    payload["max_forced_hops"] = int(pipeline_cfg.max_forced_hops_per_root)

    native_start = time.perf_counter()
    native_search = (
        native.search_mcts_hgb226
        if str(args.native_search_mode or "") == "gv3_depth_one"
        else native.search_mcts_hgb226_value_prior_hgb
    )
    native_runtimes = (
        (value_runtime,)
        if str(args.native_search_mode or "") == "gv3_depth_one"
        else (value_runtime, controller_runtime, adversary_runtime)
    )
    if str(args.native_search_mode or "") == "gv3_depth_one":
        payload["use_policy_prior"] = False
    native_result = native_search(
        *native_runtimes,
        100,
        nlt._native_state_payload(env, state),
        payload,
        int(args.iterations),
        player,
        int(prepared.root_node_id_override or prepared.root_id),
        int(depth),
        0,
        int(prepared.root_id),
        int(args.seed) + int(prepared.root_id),
        bool(args.trace),
        False,
        str(args.native_iter_log or ""),
        str(args.native_root_log or ""),
    )
    native_elapsed = time.perf_counter() - native_start
    rollout_root_time = float(state.simulator._time)
    native_rollout_stats = dict(native_result.get("perf", {}))

    if bool(args.native_only):
        native_root_visits = int(native_result.get("root_visits", 0))
        result = {
            "iterations": int(args.iterations),
            "root_id": int(prepared.root_id),
            "root_player": player,
            "native_elapsed_s": float(native_elapsed),
            "native_root_visits": native_root_visits,
            "native_best_action_index": int(native_result.get("best_action_index", -1)),
            "native_root_value": (
                float(native_result.get("root_value_sum", 0.0)) / native_root_visits
                if native_root_visits
                else 0.0
            ),
            "native_visits": {
                int(child["index"]): int(child.get("visits", 0))
                for child in native_result.get("children", [])
            },
            "native_child_q": {int(child["index"]): float(child.get("value_sum", 0.0)) / max(1, int(child.get("visits", 0))) for child in native_result.get("children", [])},
            "native_priors": {int(child["index"]): float(child.get("prior", 0.0)) for child in native_result.get("children", [])},
            "native_rewards": {int(child["index"]): float(child.get("reward", 0.0)) for child in native_result.get("children", [])},
            "native_action_reprs": {int(child["index"]): repr(actions[int(child["index"])]) for child in native_result.get("children", [])},
            "rollout_horizon_from_each_leaf": float(args.rollout_horizon_sec),
            "native_rollout_stats": native_rollout_stats,
            "native_rollout_trace_steps": list(native_result.get("rollout_trace_steps", [])),
        }
        (tmp / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        if native_root_visits != int(args.iterations):
            raise AssertionError(result)
        if int(args.rollout_count) > 0:
            _assert_rollout_deadline_stats(
                native_rollout_stats,
                root_time=rollout_root_time,
                horizon=float(args.rollout_horizon_sec),
                native=True,
            )
        return result

    hgb_elapsed: float | None = None
    if (
        args.hgb_value_model is not None
        and args.hgb_controller_policy_model is not None
        and args.hgb_adversary_policy_model is not None
    ):
        hgb_value_runtime, _, _, _ = cpp_runner._load_value_runtime_from_joblib(
            native=native,
            model_path=args.hgb_value_model,
            export_path=tmp / "hgb_value.tsv",
            feature_dim=226,
            model_tag="same_root_hgb_value",
        )
        hgb_controller_runtime, _ = cpp_runner._load_prior_runtime_from_joblib(
            native=native,
            model_path=args.hgb_controller_policy_model,
            export_path=tmp / "hgb_controller.tsv",
            feature_dim=269,
            model_tag="same_root_hgb_controller",
        )
        hgb_adversary_runtime, _ = cpp_runner._load_prior_runtime_from_joblib(
            native=native,
            model_path=args.hgb_adversary_policy_model,
            export_path=tmp / "hgb_adversary.tsv",
            feature_dim=233,
            model_tag="same_root_hgb_adversary",
        )
        hgb_start = time.perf_counter()
        hgb_result = native.search_mcts_hgb226_value_prior_hgb(
            hgb_value_runtime,
            hgb_controller_runtime,
            hgb_adversary_runtime,
            100,
            nlt._native_state_payload(env, state),
            payload,
            int(args.iterations),
            player,
            int(prepared.root_node_id_override or prepared.root_id),
            int(depth),
            0,
            int(prepared.root_id),
            int(args.seed) + int(prepared.root_id),
            False,
            False,
            "",
            "",
        )
        hgb_elapsed = time.perf_counter() - hgb_start
        if int(hgb_result.get("root_visits", 0)) != int(args.iterations):
            raise AssertionError("HGB same-root benchmark did not complete all iterations")

    python_visits = {
        int(idx): int(child.visits) for idx, child in python_root.children.items()
    }
    native_visits = {
        int(child["index"]): int(child.get("visits", 0))
        for child in native_result.get("children", [])
    }
    native_children = {
        int(child["index"]): child for child in native_result.get("children", [])
    }
    python_priors = {
        int(idx): float(python_root.action_priors.get(int(idx), 0.0))
        for idx in python_visits
    }
    native_priors = {
        idx: float(child.get("prior", 0.0)) for idx, child in native_children.items()
    }
    python_rewards = {
        int(idx): float(child.reward) for idx, child in python_root.children.items()
    }
    native_rewards = {
        idx: float(child.get("reward", 0.0)) for idx, child in native_children.items()
    }
    python_discounts = {
        int(idx): float(child.edge_discount) for idx, child in python_root.children.items()
    }
    native_discounts = {
        idx: float(child.get("edge_discount", 0.0))
        for idx, child in native_children.items()
    }
    python_child_q = {
        int(idx): float(child.mean_value())
        for idx, child in python_root.children.items()
    }
    native_child_q = {
        idx: (
            float(child.get("value_sum", 0.0)) / int(child.get("visits", 0))
            if int(child.get("visits", 0))
            else 0.0
        )
        for idx, child in native_children.items()
    }
    all_indices = sorted(set(python_visits) | set(native_visits))
    visit_l1 = sum(
        abs(python_visits.get(idx, 0) - native_visits.get(idx, 0))
        for idx in all_indices
    ) / float(args.iterations)
    python_best = int(
        (python_root.action_alias_to_canonical or {}).get(
            int(python_result.best_action_index),
            int(python_result.best_action_index),
        )
    )
    native_best_alias = int(native_result.get("best_action_index", -1))
    native_best = int(
        dict(native_result.get("action_alias_to_canonical", {})).get(
            native_best_alias,
            native_best_alias,
        )
    )
    native_root_visits = int(native_result.get("root_visits", 0))
    native_root_value = (
        float(native_result.get("root_value_sum", 0.0)) / native_root_visits
        if native_root_visits
        else 0.0
    )
    python_rollout_stats = (
        vars(python_mcts.rollout_stats)
        if hasattr(python_mcts, "rollout_stats") else {}
    )
    rollout_first_history_matches = (
        int(python_rollout_stats.get("first_history_hash") or 0)
        == int(native_rollout_stats.get("rollout_first_history_hash") or 0)
        and int(python_rollout_stats.get("first_history_actions") or 0)
        == int(native_rollout_stats.get("rollout_first_history_actions") or 0)
    )
    result = {
        "iterations": int(args.iterations),
        "discount_factor": float(args.discount_factor),
        "root_id": int(prepared.root_id),
        "root_player": player,
        "valid_action_count": int(valid_count),
        "python_root_visits": int(python_root.visits),
        "native_root_visits": native_root_visits,
        "python_root_value": float(python_root.mean_value()),
        "native_root_value": native_root_value,
        "root_value_abs_error": abs(float(python_root.mean_value()) - native_root_value),
        "python_best_canonical": python_best,
        "native_best_canonical": native_best,
        "best_canonical_matches": bool(python_best == native_best),
        "visit_l1_fraction": float(visit_l1),
        "python_elapsed_s": float(python_elapsed),
        "native_elapsed_s": float(native_elapsed),
        "rollout_count": int(args.rollout_count),
        "rollout_horizon_sec": float(args.rollout_horizon_sec),
        "python_rollout_stats": python_rollout_stats,
        "rollout_horizon_from_each_leaf": float(args.rollout_horizon_sec),
        "native_rollout_stats": native_rollout_stats,
            "native_rollout_trace_steps": list(native_result.get("rollout_trace_steps", [])),
        "rollout_first_history_matches": rollout_first_history_matches,
        "native_to_python_time_ratio": float(native_elapsed / python_elapsed),
        "hgb_native_elapsed_s": hgb_elapsed,
        "dnn_to_hgb_native_time_ratio": (
            float(native_elapsed / hgb_elapsed) if hgb_elapsed else None
        ),
        "python_priors": python_priors,
        "native_priors": native_priors,
        "python_rewards": python_rewards,
        "native_rewards": native_rewards,
        "python_discounts": python_discounts,
        "native_discounts": native_discounts,
        "python_child_q": python_child_q,
        "native_child_q": native_child_q,
        "python_visits": python_visits,
        "native_visits": native_visits,
        "python_iteration_trace": (
            python_mcts.iteration_trace if bool(args.trace) else []
        ),
        "python_selection_trace": (
            python_mcts.selection_trace if bool(args.trace) else []
        ),
        "native_iteration_trace": (
            list(native_result.get("iter_events", [])) if bool(args.trace) else []
        ),
    }
    (tmp / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if int(python_root.visits) != int(args.iterations):
        raise AssertionError(result)
    if native_root_visits != int(args.iterations):
        raise AssertionError(result)
    if python_best != native_best:
        raise AssertionError(result)
    if visit_l1 > float(args.visit_l1_tolerance):
        raise AssertionError(result)
    if result["root_value_abs_error"] > float(args.value_tolerance):
        raise AssertionError(result)
    if int(args.rollout_count) > 0:
        _assert_rollout_deadline_stats(
            python_rollout_stats,
            root_time=rollout_root_time,
            horizon=float(args.rollout_horizon_sec),
            native=False,
        )
        _assert_rollout_deadline_stats(
            native_rollout_stats,
            root_time=rollout_root_time,
            horizon=float(args.rollout_horizon_sec),
            native=True,
        )
        if not rollout_first_history_matches:
            raise AssertionError(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-value-model", type=Path, required=True)
    parser.add_argument("--adversary-value-model", type=Path, required=True)
    parser.add_argument("--controller-policy-model", type=Path, required=True)
    parser.add_argument("--adversary-policy-model", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rollout-count", type=int, default=0)
    parser.add_argument("--rollout-parallel-threads", type=int, default=10)
    parser.add_argument("--rollout-policy-parallel-threads", type=int, default=0)
    parser.add_argument("--rollout-horizon-sec", type=float, default=0.4)
    parser.add_argument("--rollout-policy-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-probability-quantum", type=float, default=1e-6)
    parser.add_argument("--rollout-max-actions", type=int, default=4096)
    parser.add_argument("--rollout-optimized-execution", action="store_true")
    parser.add_argument("--capture-rollout-trace", action="store_true")
    parser.add_argument("--native-iter-log", type=Path)
    parser.add_argument("--native-root-log", type=Path)
    parser.add_argument(
        "--native-search-mode",
        choices=("full_tree", "full_tree_rollout", "gv3_depth_one"),
    )
    parser.add_argument("--discount-factor", type=float, default=0.995)
    parser.add_argument("--uct-c", type=float, default=1.4)
    parser.add_argument("--puct-c", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--source-roots", type=int, default=32)
    parser.add_argument("--history-hops-max", type=int, default=20)
    parser.add_argument("--root-player", choices=("any", "controller", "adversary"), default="controller")
    parser.add_argument("--skip-matching-roots", type=int, default=0)
    parser.add_argument("--visit-l1-tolerance", type=float, default=0.05)
    parser.add_argument("--value-tolerance", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--hgb-value-model", type=Path)
    parser.add_argument("--hgb-controller-policy-model", type=Path)
    parser.add_argument("--hgb-adversary-policy-model", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
