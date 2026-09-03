from __future__ import annotations

import json
from pathlib import Path
import unittest

from vidur_vllm_real_testing.prompt_materialization import verify_prompt_artifacts


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "vidur_vllm_real_testing"
TOKENIZER_DIR = PACKAGE_DIR / "tokenizer" / "llama3_8b"


class ReadyEnglishTraceTests(unittest.TestCase):
    def test_ready_traces_pin_and_verify_every_prompt(self) -> None:
        cases = (
            ("splitwise_conv_20s", 31),
            ("gv3_legal_20s", 79),
        )
        for name, expected_count in cases:
            with self.subTest(trace=name):
                trace = PACKAGE_DIR / "traces" / f"{name}_english_raw.csv"
                manifest_path = PACKAGE_DIR / "traces" / f"{name}_english_manifest.json"
                count = verify_prompt_artifacts(
                    raw_trace_path=trace,
                    tokenizer_dir=TOKENIZER_DIR,
                )
                self.assertEqual(count, expected_count)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                prompts = manifest["prompt_artifacts"]
                self.assertEqual(prompts["request_count"], expected_count)
                self.assertTrue(prompts["add_special_tokens"])
                self.assertEqual(prompts["bos_token_id"], 128000)
                self.assertEqual(len(prompts["aggregate_prompt_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
