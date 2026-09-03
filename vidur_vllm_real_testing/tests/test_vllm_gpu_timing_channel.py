from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vidur_vllm_real_testing.vllm_gpu_timing_channel import (
    GPUForwardTimingReader,
    append_gpu_forward_timing,
)


class GPUForwardTimingChannelTests(unittest.TestCase):
    def test_round_trip_requires_exact_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.jsonl"
            path.touch()
            append_gpu_forward_timing(
                path,
                scheduled_tokens={"a": 1024, "b": 1024, "c": 1024},
                gpu_forward_s=0.123,
            )
            reader = GPUForwardTimingReader(path)
            self.assertEqual(
                reader.read_for_batch({"a": 1024, "b": 1024, "c": 1024}),
                0.123,
            )

    def test_internal_warmup_records_are_skipped_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.jsonl"
            path.touch()
            append_gpu_forward_timing(
                path,
                scheduled_tokens={"_warmup_0_": 2, "_warmup_1_": 2},
                gpu_forward_s=0.050,
            )
            append_gpu_forward_timing(
                path,
                scheduled_tokens={"a": 1024},
                gpu_forward_s=0.123,
            )
            reader = GPUForwardTimingReader(path)
            self.assertEqual(reader.read_for_batch({"a": 1024}), 0.123)

    def test_mismatched_batch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.jsonl"
            path.touch()
            append_gpu_forward_timing(
                path,
                scheduled_tokens={"a": 1024},
                gpu_forward_s=0.123,
            )
            with self.assertRaisesRegex(RuntimeError, "batch mismatch"):
                GPUForwardTimingReader(path).read_for_batch({"b": 1024})

    def test_missing_record_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.jsonl"
            path.touch()
            with self.assertRaisesRegex(RuntimeError, "missing GPU-forward"):
                GPUForwardTimingReader(path).read_for_batch({"a": 1024})


if __name__ == "__main__":
    unittest.main()
