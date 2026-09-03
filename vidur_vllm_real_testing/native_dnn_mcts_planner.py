"""Concrete promoted Markov-DNN/native-MCTS planner for real vLLM scheduling.

The pybind search function retains ``hgb`` in its historical name, but the
three runtimes loaded here are strictly validated AlphaGoZero DNN exports.
No sklearn/HGB checkpoint is accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import threading
from typing import Any, Mapping

from .scheduler_contract import LiveStateSnapshot


_TRUE_VALUES = {"1", "true", "yes", "on"}
_SEED_MODULUS = 2_147_483_647
_VALUE_ARCHITECTURE = "agz_markov_value_deepset_v2"
_POLICY_ARCHITECTURE = "agz_markov_policy_deepset_v3"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in _TRUE_VALUES


def _metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing promoted DNN metadata: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"DNN metadata must be an object: {path}")
    return raw


def _native_export_header(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing native DNN export: {path}")
    header: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        magic = handle.readline().strip()
        if magic != "agz_dnn_v2":
            raise TypeError(
                f"{path} is not an AlphaGoZero DNN v2 export; observed {magic!r}"
            )
        header["magic"] = magic
        for line in handle:
            if line.startswith("tensor\t"):
                break
            key, separator, value = line.rstrip("\n").partition("\t")
            if separator:
                header[key] = value
    return header


def _require_dnn_artifact(
    directory: Path,
    *,
    model_kind: str,
    architecture: str,
    role: str,
) -> Path:
    metadata = _metadata(directory / "metadata.json")
    expected = {
        "model_kind": model_kind,
        "architecture_version": architecture,
        "feature_schema": "markov_v2",
        "role": role,
    }
    for key, value in expected.items():
        if str(metadata.get(key, "")) != value:
            raise TypeError(
                f"{directory}: expected DNN metadata {key}={value!r}, "
                f"observed {metadata.get(key)!r}"
            )

    export = directory / "native_model.tsv"
    header = _native_export_header(export)
    header_expected = {
        "model_kind": model_kind,
        "architecture": architecture,
        "feature_schema": "markov_v2",
        "role": role,
    }
    for key, value in header_expected.items():
        if header.get(key) != value:
            raise TypeError(
                f"{export}: expected DNN header {key}={value!r}, "
                f"observed {header.get(key)!r}"
            )
    return export


def _import_native_extension() -> Any:
    try:
        return importlib.import_module("mcts_native_gv2")
    except ImportError:
        pass

    repo = Path(__file__).resolve().parents[1]
    cpp_dir = repo / "vidur" / "Game_Version3_Cpp"
    extension = sorted(cpp_dir.glob("mcts_native_gv2*.so"))
    if not extension:
        raise FileNotFoundError(
            f"native GV3 extension is absent from {cpp_dir}; build it for {sys.executable}"
        )
    if str(cpp_dir) not in sys.path:
        sys.path.insert(0, str(cpp_dir))
    return importlib.import_module("mcts_native_gv2")


def _limit_cpu_threads(threads: int) -> None:
    value = str(max(1, int(threads)))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(value))
    except Exception:
        pass


def _build_native_cfg(config: "NativeDNNMCTSPlannerConfig") -> dict[str, Any]:
    from vidur.Game_Version3.DNN.native_selfplay import (
        _cfg_payload,
        attach_execution_predictor_payload,
    )
    from vidur.Game_Version3.Model_Tester.config import DEFAULT_MODEL_TESTER_CONFIG
    from vidur.Game_Version3.multiProcessUtils import _build_env_and_simulator

    pipeline_cfg = DEFAULT_MODEL_TESTER_CONFIG.to_pipeline_cfg()
    simulator, _env, _constraints, _explore = _build_env_and_simulator(
        pipeline_cfg,
        use_virtual_env=True,
    )
    payload = _cfg_payload(pipeline_cfg, torchscript_model_spec="")
    attach_execution_predictor_payload(payload, simulator)
    gv3 = pipeline_cfg.game_v2
    payload.update(
        {
            "discount_factor": float(config.discount_factor),
            "use_model_bootstrap": True,
            "use_policy_prior": True,
            "native_search_mode": str(config.search_mode),
            "rollout_count": int(config.rollout_count),
            "rollout_parallel_threads": int(config.rollout_parallel_threads),
            "rollout_policy_parallel_threads": int(config.rollout_policy_threads),
            "rollout_horizon_sec": float(config.rollout_horizon_s),
            "rollout_policy_temperature": float(config.rollout_policy_temperature),
            "rollout_probability_quantum": float(config.rollout_probability_quantum),
            "rollout_max_actions": int(config.rollout_max_actions),
            "rollout_optimized_execution": True,
            "capture_rollout_trace": False,
            "capture_root_puct_trace": False,
            "root_dirichlet_noise_enabled": False,
            "root_dirichlet_alpha": 0.0,
            "root_dirichlet_total_concentration": 0.0,
            "root_dirichlet_epsilon": 0.0,
            "puct_c": float(config.puct_c),
            "uct_c": float(config.uct_c),
            "pb_c_base": float(gv3.mcts_search.pb_c_base),
            "pb_c_init": float(gv3.mcts_search.pb_c_init),
            "policy_prior_temperature": float(config.policy_prior_temperature),
            "prior_min_prob": float(config.prior_min_prob),
            "max_forced_hops": int(pipeline_cfg.max_forced_hops_per_root),
            "value_feature_schema": "markov_v2",
            "policy_feature_schema": "markov_v2",
            "controller_prefill_budget_options": [
                int(value) for value in gv3.controller_action.prefill_budget_options
            ],
            "controller_eviction_rule_names": list(
                gv3.controller_action.eviction_rule_names
            ),
            "controller_ordering_heuristics": list(
                gv3.controller_action.ordering_heuristics
            ),
            "adversary_allowed_prefill_tokens": [
                int(value) for value in gv3.request.allowed_prefill_tokens
            ],
            "adversary_stop_rule_names": list(gv3.adversary_action.stop_rule_names),
        }
    )
    return payload


@dataclass(frozen=True)
class NativeDNNMCTSPlannerConfig:
    model_bundle: Path
    iterations: int = 2000
    discount_factor: float = 0.98
    puct_c: float = 0.5
    uct_c: float = 1.0
    search_mode: str = "full_tree_rollout"
    rollout_count: int = 1
    rollout_parallel_threads: int = 1
    rollout_policy_threads: int = 1
    rollout_horizon_s: float = 3.0
    rollout_policy_temperature: float = 1.0
    rollout_probability_quantum: float = 1e-6
    rollout_max_actions: int = 4096
    policy_prior_temperature: float = 1.0
    prior_min_prob: float = 1e-8
    native_threads: int = 1
    seed: int = 2026

    @classmethod
    def from_environment(cls) -> "NativeDNNMCTSPlannerConfig":
        bundle = os.environ.get("VIDUR_VLLM_GV3_MODEL_BUNDLE", "").strip()
        if not bundle:
            raise RuntimeError(
                "VIDUR_VLLM_GV3_MODEL_BUNDLE must name a promoted native DNN bundle"
            )
        return cls(
            model_bundle=Path(bundle).expanduser(),
            iterations=int(os.environ.get("VIDUR_VLLM_GV3_MCTS_ITERATIONS", "2000")),
            discount_factor=float(os.environ.get("VIDUR_VLLM_GV3_DISCOUNT_FACTOR", "0.98")),
            puct_c=float(os.environ.get("VIDUR_VLLM_GV3_PUCT_C", "0.5")),
            uct_c=float(os.environ.get("VIDUR_VLLM_GV3_UCT_C", "1.0")),
            search_mode=os.environ.get(
                "VIDUR_VLLM_GV3_NATIVE_SEARCH_MODE", "full_tree_rollout"
            ).strip(),
            rollout_count=int(os.environ.get("VIDUR_VLLM_GV3_ROLLOUT_COUNT", "1")),
            rollout_parallel_threads=int(
                os.environ.get("VIDUR_VLLM_GV3_ROLLOUT_THREADS", "1")
            ),
            rollout_policy_threads=int(
                os.environ.get("VIDUR_VLLM_GV3_ROLLOUT_POLICY_THREADS", "1")
            ),
            rollout_horizon_s=float(
                os.environ.get("VIDUR_VLLM_GV3_ROLLOUT_HORIZON_S", "3.0")
            ),
            rollout_policy_temperature=float(
                os.environ.get("VIDUR_VLLM_GV3_ROLLOUT_TEMPERATURE", "1.0")
            ),
            rollout_probability_quantum=float(
                os.environ.get("VIDUR_VLLM_GV3_ROLLOUT_PROBABILITY_QUANTUM", "1e-6")
            ),
            rollout_max_actions=int(
                os.environ.get("VIDUR_VLLM_GV3_ROLLOUT_MAX_ACTIONS", "4096")
            ),
            policy_prior_temperature=float(
                os.environ.get("VIDUR_VLLM_GV3_POLICY_PRIOR_TEMPERATURE", "1.0")
            ),
            prior_min_prob=float(
                os.environ.get("VIDUR_VLLM_GV3_PRIOR_MIN_PROB", "1e-8")
            ),
            native_threads=int(os.environ.get("VIDUR_VLLM_GV3_NATIVE_THREADS", "1")),
            seed=int(os.environ.get("VIDUR_VLLM_GV3_SEED", "2026")),
        )

    def validate(self) -> None:
        if self.iterations <= 0:
            raise ValueError("MCTS iterations must be positive")
        if not 0.0 < self.discount_factor <= 1.0:
            raise ValueError("discount factor must be in (0, 1]")
        if self.puct_c < 0.0 or self.uct_c < 0.0:
            raise ValueError("search constants must be nonnegative")
        if self.search_mode not in {"full_tree", "full_tree_rollout"}:
            raise ValueError("native search mode must be full_tree or full_tree_rollout")
        if self.rollout_count < 0 or self.rollout_horizon_s < 0.0:
            raise ValueError("rollout count and horizon must be nonnegative")
        if self.search_mode == "full_tree_rollout" and self.rollout_count <= 0:
            raise ValueError("full_tree_rollout requires at least one rollout")
        if self.native_threads <= 0:
            raise ValueError("native thread count must be positive")


@dataclass(frozen=True)
class NativeControllerDecision:
    token_budget: int
    selected_request_ids: list[int]
    token_allocations: dict[int, int]
    prefill_allocations: dict[int, int]
    decode_allocations: dict[int, int]
    heuristic: str | None
    strategy: str | None
    mapping: tuple[int, ...] | None
    evicted_request_ids: list[int]
    selection: Mapping[str, Any]


def _int_map(raw: object, *, field: str) -> dict[int, int]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"native action field {field} must be an object")
    result = {int(key): int(value) for key, value in raw.items()}
    if any(key < 0 or value <= 0 for key, value in result.items()):
        raise ValueError(f"native action field {field} contains invalid allocation")
    return result


def _action_from_json(
    action_json: str,
    *,
    state_payload: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> NativeControllerDecision:
    raw = json.loads(action_json)
    if not isinstance(raw, Mapping) or raw.get("type") != "controller":
        raise TypeError("native MCTS did not return a controller action")

    prefill = _int_map(raw.get("prefill_allocations", {}), field="prefill_allocations")
    decode = _int_map(raw.get("decode_allocations", {}), field="decode_allocations")
    token_allocations = _int_map(raw.get("token_allocations", {}), field="token_allocations")
    if set(prefill) & set(decode):
        raise ValueError("native controller action allocates prefill and decode to one request")
    combined = dict(prefill)
    combined.update(decode)
    if combined != token_allocations:
        raise ValueError("native controller token allocations do not match phase allocations")

    requests = {
        int(item["request_id"]): item
        for item in list(state_payload.get("requests", ()))
        if isinstance(item, Mapping)
    }
    for request_id, tokens in prefill.items():
        request = requests.get(request_id)
        if request is None or bool(request.get("completed", False)):
            raise ValueError(f"native action references inactive prefill request {request_id}")
        remaining = max(
            0,
            int(request.get("num_prefill_tokens", 0))
            - int(request.get("num_processed_prefill_tokens", 0)),
        )
        if tokens > remaining:
            raise ValueError(
                f"native action over-allocates prefill request {request_id}: {tokens}>{remaining}"
            )
    for request_id, tokens in decode.items():
        request = requests.get(request_id)
        if request is None or bool(request.get("completed", False)):
            raise ValueError(f"native action references inactive decode request {request_id}")
        if not bool(request.get("is_prefill_complete", False)) or tokens != 1:
            raise ValueError(f"native action has invalid decode allocation for {request_id}")

    token_budget = int(raw.get("token_budget", 0))
    if token_budget != sum(token_allocations.values()):
        raise ValueError("native action token budget does not equal allocated tokens")
    selected = [int(value) for value in list(raw.get("selected_request_ids", ()))]
    if set(selected) != set(token_allocations):
        raise ValueError("native selected request IDs do not match token allocations")
    evicted = [int(value) for value in list(raw.get("evicted_request_ids", ()))]
    if any(request_id not in requests for request_id in evicted):
        raise ValueError("native action evicts an unknown request")
    mapping_raw = raw.get("mapping")
    mapping = None if mapping_raw is None else tuple(int(value) for value in mapping_raw)
    return NativeControllerDecision(
        token_budget=token_budget,
        selected_request_ids=selected,
        token_allocations=token_allocations,
        prefill_allocations=prefill,
        decode_allocations=decode,
        heuristic=(None if raw.get("heuristic") in {None, ""} else str(raw["heuristic"])),
        strategy=(None if raw.get("strategy") in {None, ""} else str(raw["strategy"])),
        mapping=mapping,
        evicted_request_ids=evicted,
        selection=dict(selection),
    )


class PromotedNativeDNNMCTSPlanner:
    """State planner backed by promoted Markov DNNs and native GV3 MCTS."""

    def __init__(
        self,
        config: NativeDNNMCTSPlannerConfig,
        *,
        native: Any | None = None,
        cfg_payload: dict[str, Any] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        bundle = config.model_bundle.resolve()
        manifest = _metadata(bundle / "current_model.json")
        if manifest.get("model_family") != "dnn" or manifest.get("native_ready") is not True:
            raise TypeError(f"{bundle} is not a native-ready promoted DNN bundle")
        if manifest.get("target_perspective") != "controller":
            raise TypeError("promoted value bundle must use the controller target perspective")
        self.controller_model_version = int(manifest["controller_model_version"])
        self.adversary_model_version = int(manifest["adversary_model_version"])

        value_export = _require_dnn_artifact(
            bundle / "controller_value",
            model_kind="value_dnn",
            architecture=_VALUE_ARCHITECTURE,
            role="controller",
        )
        controller_export = _require_dnn_artifact(
            bundle / "controller_prior",
            model_kind="policy_dnn",
            architecture=_POLICY_ARCHITECTURE,
            role="controller",
        )
        adversary_export = _require_dnn_artifact(
            bundle / "adversary_prior",
            model_kind="policy_dnn",
            architecture=_POLICY_ARCHITECTURE,
            role="adversary",
        )

        _limit_cpu_threads(config.native_threads)
        self.native = native if native is not None else _import_native_extension()
        self.value_runtime = self.native.NewFeatures226HGBRuntime()
        self.controller_prior_runtime = self.native.NativeHGBModelRuntime()
        self.adversary_prior_runtime = self.native.NativeHGBModelRuntime()
        self.value_runtime.load_model_export(str(value_export))
        self.controller_prior_runtime.load_model_export(str(controller_export))
        self.adversary_prior_runtime.load_model_export(str(adversary_export))
        if not bool(self.value_runtime.loaded):
            raise RuntimeError("promoted controller value DNN failed to load natively")
        for role, runtime in (
            ("controller", self.controller_prior_runtime),
            ("adversary", self.adversary_prior_runtime),
        ):
            if not bool(runtime.loaded) or not bool(runtime.is_markov_policy):
                raise RuntimeError(f"promoted {role} policy is not a native Markov DNN")

        self.cfg_payload = dict(cfg_payload) if cfg_payload is not None else _build_native_cfg(config)
        self._decision_index = 0
        self._lock = threading.Lock()
        self.last_search: dict[str, Any] | None = None

    @classmethod
    def from_environment(cls) -> "PromotedNativeDNNMCTSPlanner":
        return cls(NativeDNNMCTSPlannerConfig.from_environment())

    def _search_seed(self, fingerprint: str, decision_index: int) -> int:
        encoded = f"{self.config.seed}:{decision_index}:{fingerprint}".encode("utf-8")
        return int.from_bytes(hashlib.blake2s(encoded, digest_size=4).digest(), "big") % _SEED_MODULUS

    @staticmethod
    def _winning_action(native_out: Mapping[str, Any]) -> tuple[int, str, dict[str, Any]]:
        children = [dict(row) for row in list(native_out.get("children", ())) if isinstance(row, Mapping)]
        q_values = [float(value) for value in list(native_out.get("root_action_values", ()))]
        if not children:
            raise RuntimeError("native MCTS returned no controller children")

        def rank(row: Mapping[str, Any]) -> tuple[int, float, int]:
            index = int(row.get("index", -1))
            q_value = q_values[index] if 0 <= index < len(q_values) else -math.inf
            return (-int(row.get("visits", 0)), -q_value, index)

        visited = [row for row in children if int(row.get("visits", 0)) > 0]
        if not visited:
            raise RuntimeError("native MCTS returned no visited controller child")
        winner = sorted(visited, key=rank)[0]
        index = int(winner["index"])
        action_json = str(winner.get("parent_action_json", "") or "")
        if not action_json:
            representations = list(native_out.get("root_action_reprs", ()))
            if 0 <= index < len(representations):
                action_json = str(representations[index] or "")
        if not action_json:
            raise RuntimeError(f"native MCTS omitted action JSON for child {index}")
        q_value = q_values[index] if 0 <= index < len(q_values) else -math.inf
        selection = {
            "action_index": index,
            "visits": int(winner.get("visits", 0)),
            "q_value": float(q_value),
            "prior": float(winner.get("prior", 0.0)),
            "root_visits": int(native_out.get("root_visits", 0)),
            "root_value": (
                float(native_out.get("root_value_sum", 0.0))
                / max(1, int(native_out.get("root_visits", 0)))
            ),
        }
        return index, action_json, selection

    def plan_state(
        self,
        state_payload: Mapping[str, Any],
        snapshot: LiveStateSnapshot,
    ) -> NativeControllerDecision:
        del snapshot
        with self._lock:
            decision_index = self._decision_index
            self._decision_index += 1
            root_id = 950_000_000 + decision_index
            fingerprint = hashlib.sha256(
                json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            iterations = max(1, int(self.config.iterations))
            native_out = self.native.search_mcts_hgb226_value_prior_hgb(
                self.value_runtime,
                self.controller_prior_runtime,
                self.adversary_prior_runtime,
                int(self.controller_model_version),
                dict(state_payload),
                dict(self.cfg_payload),
                iterations,
                "controller",
                root_id,
                decision_index,
                950_000_000,
                root_id,
                self._search_seed(fingerprint, decision_index),
                False,
                False,
                "",
                "",
            )
            _index, action_json, selection = self._winning_action(native_out)
            selection.update(
                {
                    "controller_model_version": self.controller_model_version,
                    "adversary_model_version": self.adversary_model_version,
                    "iterations_requested": iterations,
                    "state_hash": fingerprint,
                }
            )
            decision = _action_from_json(
                action_json,
                state_payload=state_payload,
                selection=selection,
            )
            self.last_search = dict(selection)
            return decision

    def __call__(
        self,
        state_payload: Mapping[str, Any],
        snapshot: LiveStateSnapshot,
    ) -> NativeControllerDecision:
        return self.plan_state(state_payload, snapshot)

