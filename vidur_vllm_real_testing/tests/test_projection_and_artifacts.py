from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from vidur_vllm_real_testing.action_projection import project_token_allocations
from vidur_vllm_real_testing.canonicalization import CanonicalizationError
from vidur_vllm_real_testing.verify_artifacts import verify


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "vidur_vllm_real_testing"


class ProjectionAndArtifactTests(unittest.TestCase):
    def test_canonical_chunk_is_truncated_only_at_actual_tail(self) -> None:
        projected = project_token_allocations(
            {"request-764": 128, "request-decode": 1},
            {"request-764": 124, "request-decode": 400},
        )
        by_id = {row.request_id: row for row in projected}
        self.assertEqual(by_id["request-764"].actual_allocation, 124)
        self.assertTrue(by_id["request-764"].truncated_to_actual_tail)
        self.assertEqual(by_id["request-decode"].actual_allocation, 1)
        self.assertFalse(by_id["request-decode"].truncated_to_actual_tail)

    def test_stale_request_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(CanonicalizationError, "no live vLLM request"):
            project_token_allocations({"stale": 128}, {"live": 128})

    def test_checked_in_trace_artifacts_verify(self) -> None:
        verify(
            manifest_path=PACKAGE_DIR / "traces" / "gv3_legal_20s_manifest.json",
            source_path=PACKAGE_DIR / "traces" / "gv3_legal_20s_raw.csv",
            canonical_path=PACKAGE_DIR / "traces" / "gv3_legal_20s_canonical.csv",
            prefill_profile_path=REPO_ROOT / "simulator_output" / "prefill_profile.csv",
            decode_profile_path=REPO_ROOT / "simulator_output" / "decode_profile.csv",
        )

    def test_modified_trace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            changed = Path(temp_dir) / "changed.csv"
            source = PACKAGE_DIR / "traces" / "gv3_legal_20s_canonical.csv"
            changed.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify(
                    manifest_path=PACKAGE_DIR / "traces" / "gv3_legal_20s_manifest.json",
                    source_path=PACKAGE_DIR / "traces" / "gv3_legal_20s_raw.csv",
                    canonical_path=changed,
                    prefill_profile_path=REPO_ROOT / "simulator_output" / "prefill_profile.csv",
                    decode_profile_path=REPO_ROOT / "simulator_output" / "decode_profile.csv",
                )


if __name__ == "__main__":
    unittest.main()
