from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from vidur_vllm_real_testing.canonicalization import (
    CanonicalizationConfig,
    PREFILL_ROUNDING_CEILING,
    PrefillProfile,
    canonicalize_prefill_tokens,
)
from vidur_vllm_real_testing.gv3_live_adapter import (
    BatchExecutionObservation,
    BatchRequestProgress,
    GV3PersistentAdapter,
)
from vidur_vllm_real_testing.scheduler_contract import (
    LiveRequestSnapshot,
    LiveStateSnapshot,
    RequestPhase,
)
from vidur_vllm_real_testing.task_1_1_2_runner import validate_prepared_trace
from vidur_vllm_real_testing.task_1_1_2_trace import (
    DEFAULT_TOKENIZER_DIR,
    build_template_rows,
    prepare_task_1_1_2_trace,
)
from vidur_vllm_real_testing.test_traces_on_GPU_with_AlphaGOZERO_models_config import (
    AlphaGoZeroGPUTraceConfig,
    GV3_IN_DISTRIBUTION_PREFILL_TOKENS,
    TraceType,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = (
    REPO_ROOT
    / "vidur/AlphaGoZero/new_vidur_cache_experiment/flash-infer_prefill_profile.csv"
)


def _request() -> LiveRequestSnapshot:
    return LiveRequestSnapshot(
        request_id="req-0",
        phase=RequestPhase.PREFILL,
        arrival_time_s=0.0,
        actual_prefill_tokens=128,
        actual_prefill_remaining=128,
        canonical_prefill_tokens=128,
        canonical_prefill_remaining=128,
        actual_decode_tokens=216,
        actual_decode_remaining=216,
        canonical_decode_tokens=216,
        canonical_decode_remaining=216,
        actual_prefill_slo_s=0.1,
        canonical_prefill_slo_s=0.1,
        actual_decode_slo_s=0.05,
        canonical_decode_slo_s=0.05,
        num_computed_tokens=0,
        num_output_tokens=0,
        queue_name="waiting",
    )


class Task112Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = PrefillProfile.load(PROFILE)

    def config(self, trace_type: TraceType) -> AlphaGoZeroGPUTraceConfig:
        return AlphaGoZeroGPUTraceConfig(
            trace_type=trace_type,
            static_trace_length_s=0.1,
            prefill_profile_path=str(PROFILE),
        )

    def test_config_freezes_task_1_1_1_runtime_and_search(self) -> None:
        config = self.config(TraceType.IN_DISTRIBUTION)
        config.validate()
        self.assertEqual(config.vllm_version, "0.26.0")
        self.assertEqual(config.flashinfer_version, "0.6.14")
        self.assertEqual(config.cuda_home, "/usr/local/cuda-13.0")
        self.assertEqual((config.tensor_parallel_size, config.pipeline_parallel_size), (1, 1))
        self.assertEqual(config.mcts_iterations, 2000)
        self.assertEqual(config.rollout_horizon_s, 3.0)
        self.assertEqual(config.discount_factor, 0.98)
        self.assertEqual(config.puct_c, 0.5)
        self.assertEqual(config.timing_scope, "transformer_blocks")
        self.assertEqual(config.decode_tokens_per_request, 864)
        self.assertIn("hf_cache", config.remote_hf_home)
        self.assertIsNone(config.to_dict()["calibration"])

    def test_ceiling_mode_maps_157_to_256_without_changing_default(self) -> None:
        ceiling = CanonicalizationConfig(prefill_rounding=PREFILL_ROUNDING_CEILING)
        self.assertEqual(canonicalize_prefill_tokens(157, ceiling), 256)
        self.assertEqual(canonicalize_prefill_tokens(157), 128)

    def test_in_distribution_rows_use_only_gv3_support(self) -> None:
        rows = build_template_rows(
            self.config(TraceType.IN_DISTRIBUTION),
            profile=self.profile,
        )
        self.assertGreater(len(rows), 1)
        self.assertEqual({float(row["arrived_at_s"]) for row in rows}, {0.0})
        self.assertTrue(
            all(
                int(row["num_prefill_tokens"])
                in GV3_IN_DISTRIBUTION_PREFILL_TOKENS
                for row in rows
            )
        )

    def test_out_distribution_rows_are_non_grid_and_ceiling_canonicalized(self) -> None:
        config = self.config(TraceType.OUT_DISTRIBUTION)
        rows = build_template_rows(config, profile=self.profile)
        self.assertTrue(any(int(row["num_prefill_tokens"]) % 128 for row in rows))
        canonicalization = CanonicalizationConfig(
            prefill_rounding=PREFILL_ROUNDING_CEILING
        )
        for row in rows:
            actual = int(row["num_prefill_tokens"])
            self.assertEqual(
                canonicalize_prefill_tokens(actual, canonicalization),
                ((actual + 127) // 128) * 128,
            )

    def test_full_trace_preparation_verifies_prompts_and_simultaneous_arrivals(self) -> None:
        config = replace(
            self.config(TraceType.OUT_DISTRIBUTION),
            # A 128-token canonical group keeps this integration test small.
            trace_seed=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = prepare_task_1_1_2_trace(
                config,
                output_root=directory,
                tokenizer_dir=DEFAULT_TOKENIZER_DIR,
            )
            result = validate_prepared_trace(
                config,
                prepared_root=artifact.root,
                tokenizer_dir=DEFAULT_TOKENIZER_DIR,
            )
            self.assertEqual(result["request_count"], artifact.request_count)
            self.assertGreater(result["rounded_rows"], 0)
            self.assertGreater(result["simultaneous_arrival_groups"], 0)

    def test_canonical_clock_ignores_snapshot_and_planner_wall_time(self) -> None:
        planner_times: list[float] = []

        def planner(payload: object, live: LiveStateSnapshot) -> object:
            assert isinstance(payload, dict)
            planner_times.append(float(payload["sim_time"]))
            return {
                "state_fingerprint": live.fingerprint,
                "policy": "test-controller",
                "allocations": [
                    {"request_id": "req-0", "phase": "prefill", "num_tokens": 128}
                ],
            }

        adapter = GV3PersistentAdapter(controller=planner)
        row = _request()
        snapshot = LiveStateSnapshot.build(
            (row,),
            max_num_scheduled_tokens=8192,
            captured_monotonic_s=9_999.0,
        )
        adapter.plan(snapshot)
        self.assertEqual(planner_times, [0.0])
        adapter.on_batch_completed(
            BatchExecutionObservation(
                scheduled_monotonic_s=10.0,
                completed_monotonic_s=1000.0,
                duration_s=0.025,
                scheduled_tokens_by_request={"req-0": 128},
                request_progress=(
                    BatchRequestProgress(
                        request_id="req-0",
                        phase_before=RequestPhase.PREFILL,
                        num_computed_tokens_before=0,
                        num_output_tokens_before=0,
                        num_computed_tokens_after=128,
                        num_output_tokens_after=0,
                        finished_after=False,
                    ),
                ),
            )
        )
        self.assertAlmostEqual(adapter.sim_time, 0.025)


if __name__ == "__main__":
    unittest.main()
