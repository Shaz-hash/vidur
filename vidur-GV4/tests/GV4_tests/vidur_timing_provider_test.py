"""Focused tests for the GV4-to-Vidur timing adapter."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


GV4_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GV4_ROOT.parent))
sys.path.insert(0, str(GV4_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from GV4_Engine.action_resolver import (  # noqa: E402
    ControllerTransitionKind,
    ResolvedControllerAction,
)
from GV4_Engine.config import (  # noqa: E402
    GV4EngineConfig,
    KVCacheConfig,
    ModelConfig,
    TopologyConfig,
    VidurPredictorConfig,
)
from GV4_Engine.state import (  # noqa: E402
    BatchAllocation,
    GV4State,
    Player,
    RequestLifecycle,
    RequestState,
)
from GV4_Engine.vidur_timing_provider import (  # noqa: E402
    VidurTimingProvider,
    VidurTimingProviderError,
    build_vidur_predictor,
    ensure_prefill_profile,
    prefill_profile_path,
)
from GV4_Engine.virtual_environment import (  # noqa: E402
    GV4VirtualVidurMCTSEnvironment,
)
from state_test import make_config  # noqa: E402


@dataclass(frozen=True)
class FakeExecutionTime:
    model_time: float
    pipeline_parallel_communication_time: float


class FakePredictor:
    def __init__(self, timings: tuple[FakeExecutionTime, ...]) -> None:
        self.timings = timings
        self.calls: list[tuple[int, tuple[tuple[int, int, bool], ...]]] = []

    def get_execution_time(self, requests, pipeline_stage):
        shape = tuple(
            (
                request.num_processed_tokens,
                request.num_tokens_to_process,
                request.is_prefill_complete,
            )
            for request in requests
        )
        self.calls.append((pipeline_stage, shape))
        return self.timings[pipeline_stage]


class TokenPredictor:
    """Return token-dependent TP-inclusive service and one PP transfer."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def get_execution_time(self, requests, pipeline_stage):
        tokens = requests[0].num_tokens_to_process
        self.calls.append((pipeline_stage, tokens))
        service_seconds = tokens * (pipeline_stage + 1) * 1e-6
        pp_seconds = 0.0002 if pipeline_stage == 0 else 0.0
        return FakeExecutionTime(
            model_time=service_seconds + pp_seconds,
            pipeline_parallel_communication_time=pp_seconds * 1e3,
        )


def batch_action(*allocations: BatchAllocation) -> ResolvedControllerAction:
    return ResolvedControllerAction(
        raw_action_index=0,
        replica_id=0,
        preemption_rule="preempt_none",
        eviction_rule="evict_none",
        prefill_budget=sum(
            item.prefill_tokens + item.recompute_tokens for item in allocations
        ),
        ordering_heuristic="SJF",
        transition_kind=ControllerTransitionKind.BATCH,
        evicted_request_ids=(),
        preempted_request_ids=(),
        pending_preemption_request_ids=(),
        allocations=tuple(allocations),
        released_kv_blocks=0,
        preempted_kv_blocks=0,
        reserved_kv_blocks=sum(item.new_kv_blocks for item in allocations),
        rank_kv_delta=(),
    )


class VidurTimingProviderTest(unittest.TestCase):
    def test_environment_can_be_created_from_one_config(self) -> None:
        config = make_config()
        predictor = FakePredictor((FakeExecutionTime(0.01, 0.0),))

        with (
            patch(
                "GV4_Engine.vidur_timing_provider.build_vidur_predictor",
                return_value=predictor,
            ),
            patch(
                "GV4_Engine.vidur_timing_provider.ensure_prefill_profile",
                return_value=None,
            ),
        ):
            environment = GV4VirtualVidurMCTSEnvironment.from_config(config)

        self.assertIs(environment.config, config)
        self.assertEqual(environment.initial_state().now, 0.0)

    def test_builder_maps_one_gv4_config_into_vidur_configs(self) -> None:
        model = ModelConfig(
            model_id="meta-llama/Meta-Llama-3-8B",
            model_revision="test-revision-001",
        )
        topology = TopologyConfig.contiguous(
            num_replicas=1,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            num_layers=model.num_layers,
        )
        config = GV4EngineConfig(
            model=model,
            topology=topology,
            vidur_predictor=VidurPredictorConfig(
                device="h100",
                network_device="h100_dgx",
                cache_dir="/tmp/gv4-builder-test",
                cache_mode="require_cache",
            ),
            kv_cache=KVCacheConfig(
                kv_budget_bytes_per_rank=(1 << 30,) * topology.total_ranks,
                block_size_tokens=16,
                memory_safety_margin_fraction=0.0,
            ),
        )
        sentinel = object()

        from vidur.execution_time_predictor import ExecutionTimePredictorRegistry

        with patch.object(
            ExecutionTimePredictorRegistry,
            "get",
            return_value=sentinel,
        ) as registry_get:
            result = build_vidur_predictor(config)

        self.assertIs(result, sentinel)
        keyword = registry_get.call_args.kwargs
        self.assertEqual(keyword["replica_config"].model_name, model.model_id)
        self.assertEqual(keyword["replica_config"].tensor_parallel_size, 2)
        self.assertEqual(keyword["replica_config"].num_pipeline_stages, 2)
        self.assertEqual(keyword["replica_config"].device, "h100")
        self.assertEqual(keyword["replica_config"].network_device, "h100_dgx")
        self.assertEqual(keyword["predictor_config"].cache_mode.name, "REQUIRE_CACHE")
        self.assertEqual(
            keyword["predictor_config"].cache_dir,
            config.vidur_predictor.cache_dir,
        )
        self.assertEqual(keyword["cache_config"].block_size, 16)

    def test_generates_validates_and_reuses_tp2_pp2_prefill_profile(self) -> None:
        with TemporaryDirectory() as directory:
            base = make_config(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
            )
            config = replace(
                base,
                vidur_predictor=replace(
                    base.vidur_predictor,
                    cache_dir=directory,
                    prefill_profile_step_tokens=128,
                ),
            )
            predictor = TokenPredictor()

            profile = ensure_prefill_profile(config, predictor)

            self.assertEqual(
                profile.path.name,
                "prefill_profile_TP2_PP2_test-gpu_test-topology.csv",
            )
            self.assertEqual(profile.path, prefill_profile_path(config))
            with profile.path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(
                    reader.fieldnames,
                    [
                        "prefill_request_size_tokens",
                        "stage_0_computation_time_sec",
                        "stage_0_1_pp_communication_time_sec",
                        "stage_1_computation_time_sec",
                        "end_to_end_prefill_time_sec",
                    ],
                )

            self.assertEqual(len(rows), 32)
            self.assertEqual(int(rows[0]["prefill_request_size_tokens"]), 128)
            self.assertEqual(int(rows[-1]["prefill_request_size_tokens"]), 4096)
            self.assertAlmostEqual(
                float(rows[0]["end_to_end_prefill_time_sec"]),
                0.000128 + 0.0002 + 0.000256,
            )
            self.assertEqual(len(predictor.calls), 64)

            calls_before_reuse = len(predictor.calls)
            reused = ensure_prefill_profile(config, predictor)
            self.assertEqual(reused, profile)
            self.assertEqual(len(predictor.calls), calls_before_reuse)

            provider = VidurTimingProvider(config, predictor, prefill_profile=profile)
            self.assertAlmostEqual(
                provider.estimate_prefill_time(129),
                0.000256 + 0.0002 + 0.000512,
            )
            self.assertEqual(len(predictor.calls), calls_before_reuse)

            profile.path.write_text("invalid-profile\n", encoding="utf-8")
            regenerated = ensure_prefill_profile(config, predictor)
            self.assertEqual(regenerated.path, profile.path)
            self.assertEqual(len(predictor.calls), calls_before_reuse + 64)

    def test_maps_batch_splits_pp_time_and_caches_result(self) -> None:
        config = make_config(pipeline_parallel_size=2)
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        state.requests.extend(
            (
                RequestState(
                    request_id=0,
                    owner_replica_id=0,
                    lifecycle=RequestLifecycle.WAITING_PREFILL,
                    arrival_time=0.0,
                    prefill_deadline=1.0,
                    decode_token_slo_sec=0.05,
                    original_prefill_tokens=128,
                    original_decode_tokens=16,
                    committed_prefill_tokens=64,
                ),
                RequestState(
                    request_id=1,
                    owner_replica_id=0,
                    lifecycle=RequestLifecycle.WAITING_DECODE,
                    arrival_time=0.0,
                    prefill_deadline=1.0,
                    decode_token_slo_sec=0.05,
                    original_prefill_tokens=128,
                    original_decode_tokens=16,
                    decode_credit_minted=True,
                    committed_prefill_tokens=128,
                    committed_decode_tokens=7,
                ),
            )
        )
        predictor = FakePredictor(
            (
                FakeExecutionTime(
                    model_time=0.025, pipeline_parallel_communication_time=5.0
                ),
                FakeExecutionTime(
                    model_time=0.030, pipeline_parallel_communication_time=0.0
                ),
            )
        )
        provider = VidurTimingProvider(config, predictor)
        action = batch_action(
            BatchAllocation(0, prefill_tokens=64),
            BatchAllocation(1, decode_tokens=1),
        )

        first = provider(state, action)
        second = provider(state, action)

        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0][0], 0.020)
        self.assertAlmostEqual(first[0][1], 0.030)
        self.assertAlmostEqual(first[1][0], 0.005)
        self.assertEqual(provider.cache_entries, 1)
        self.assertEqual(len(predictor.calls), 2)
        self.assertEqual(
            predictor.calls[0][1],
            ((64, 64, False), (135, 1, True)),
        )

    def test_rejects_prefill_shape_beyond_profile_boundary(self) -> None:
        config = make_config(pipeline_parallel_size=2)
        state = GV4State.initial(config, next_player=Player.CONTROLLER)
        for request_id in range(2):
            state.requests.append(
                RequestState(
                    request_id=request_id,
                    owner_replica_id=0,
                    lifecycle=RequestLifecycle.WAITING_PREFILL,
                    arrival_time=0.0,
                    prefill_deadline=1.0,
                    decode_token_slo_sec=0.05,
                    original_prefill_tokens=4096,
                    original_decode_tokens=16,
                )
            )
        predictor = FakePredictor(
            (
                FakeExecutionTime(0.1, 1.0),
                FakeExecutionTime(0.1, 0.0),
            )
        )
        provider = VidurTimingProvider(config, predictor)

        with self.assertRaisesRegex(
            VidurTimingProviderError, "aggregate prefill chunk"
        ):
            provider(
                state,
                batch_action(
                    BatchAllocation(0, prefill_tokens=4096),
                    BatchAllocation(1, prefill_tokens=4096),
                ),
            )

        self.assertFalse(predictor.calls)


if __name__ == "__main__":
    unittest.main()
