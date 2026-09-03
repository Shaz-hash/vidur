from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile
import unittest

from vidur_vllm_real_testing.feature_audit import build_static_feature_audit


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "vidur_vllm_real_testing"


class FeatureAuditTests(unittest.TestCase):
    def test_splitwise_static_features_use_canonical_not_actual_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "audit.csv"
            count = build_static_feature_audit(
                canonical_trace_path=(
                    PACKAGE_DIR / "traces" / "splitwise_conv_20s_english_canonical.csv"
                ),
                output_path=output,
            )
            self.assertEqual(count, 31)
            with output.open("r", newline="", encoding="utf-8") as handle:
                first = next(csv.DictReader(handle))
            self.assertEqual(int(first["actual_prefill_tokens"]), 374)
            self.assertEqual(int(first["canonical_prefill_tokens"]), 384)
            self.assertAlmostEqual(
                float(first["markov_prefill_total_div_4096"]), 384 / 4096, places=7
            )
            self.assertAlmostEqual(
                float(first["markov_decode_total_div_864"]), 44 / 864, places=7
            )
            self.assertAlmostEqual(
                float(first["markov_asinh_decode_slo_div_0p05s"]),
                math.asinh(1.0),
                places=7,
            )


if __name__ == "__main__":
    unittest.main()
