from __future__ import annotations

import unittest
from pathlib import Path

from vidur_vllm_real_testing.canonicalization import (
    CanonicalizationError,
    PrefillProfile,
    canonicalize_decode_slo,
    canonicalize_decode_tokens,
    canonicalize_prefill_slo,
    canonicalize_prefill_tokens,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFILL_PROFILE = REPO_ROOT / "simulator_output" / "prefill_profile.csv"


class CanonicalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = PrefillProfile.load(PREFILL_PROFILE)

    def test_prefill_uses_half_up_not_bankers_rounding(self) -> None:
        expected = {
            1: 128,
            64: 128,
            191: 128,
            192: 256,
            764: 768,
            4032: 4096,
            4096: 4096,
        }
        for actual, canonical in expected.items():
            with self.subTest(actual=actual):
                self.assertEqual(canonicalize_prefill_tokens(actual), canonical)

    def test_prefill_rejects_out_of_domain_values(self) -> None:
        for value in (0, -1, 4097):
            with self.subTest(value=value):
                with self.assertRaises(CanonicalizationError):
                    canonicalize_prefill_tokens(value)

    def test_decode_is_identity_inside_strict_bounds(self) -> None:
        for value in (1, 216, 864):
            self.assertEqual(canonicalize_decode_tokens(value), value)
        for value in (0, 865):
            with self.assertRaises(CanonicalizationError):
                canonicalize_decode_tokens(value)

    def test_profile_and_slo_match_gv3(self) -> None:
        profile_time, prefill_slo = canonicalize_prefill_slo(768, self.profile)
        self.assertAlmostEqual(profile_time, 0.07016849423102292)
        self.assertAlmostEqual(prefill_slo, 3.0 * profile_time)
        self.assertEqual(canonicalize_decode_slo(0.04), 0.05)

    def test_profile_hash_is_frozen_experiment_hash(self) -> None:
        self.assertEqual(
            self.profile.sha256,
            "47f2ed85aca4a5ec47eacae8425d38284e76759b75f5c069852d393ecc62329b",
        )


if __name__ == "__main__":
    unittest.main()
