from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from vidur_vllm_real_testing.container_smoke import run_smoke
from vidur_vllm_real_testing.tokenizer_compatibility import verify_model_tokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "vidur_vllm_real_testing"
TOKENIZER_DIR = PACKAGE_DIR / "tokenizer" / "llama3_8b"
SPLITWISE_TRACE = PACKAGE_DIR / "traces" / "splitwise_conv_20s_english_raw.csv"


class TokenizerCompatibilityTests(unittest.TestCase):
    def test_bundled_model_tokenizer_matches_every_prompt(self) -> None:
        report = verify_model_tokenizer(
            model_tokenizer=str(TOKENIZER_DIR),
            pinned_tokenizer_dir=TOKENIZER_DIR,
            trace_paths=[SPLITWISE_TRACE],
        )
        self.assertEqual(report["status"], "compatible")
        self.assertEqual(report["prompt_count"], 31)
        vllm_available = importlib.util.find_spec("vllm") is not None
        self.assertEqual(report["vllm_checked"], vllm_available)
        if vllm_available:
            self.assertIsNotNone(report["vllm_version"])

    def test_tokenizer_hash_mismatch_fails_before_prompt_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer_root = Path(temporary)
            for file_name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                (tokenizer_root / file_name).write_bytes((TOKENIZER_DIR / file_name).read_bytes())
            tokenizer_path = tokenizer_root / "tokenizer.json"
            tokenizer_path.write_bytes(tokenizer_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "tokenizer.json SHA256 mismatch"):
                verify_model_tokenizer(
                    model_tokenizer=str(tokenizer_root),
                    pinned_tokenizer_dir=TOKENIZER_DIR,
                    trace_paths=[SPLITWISE_TRACE],
                )

    def test_full_container_contract_passes_without_vllm_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "report.json"
            report = run_smoke(
                package_root=PACKAGE_DIR,
                model_tokenizer=str(TOKENIZER_DIR),
                revision=None,
                require_vllm=False,
                local_files_only=True,
                report_path=report_path,
            )
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ready")
            self.assertEqual(persisted["trace_counts"]["gv3_legal_20s"], 79)


if __name__ == "__main__":
    unittest.main()
