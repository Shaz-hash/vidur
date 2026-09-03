from __future__ import annotations

from pathlib import Path
import unittest

from vidur_vllm_real_testing.prompt_materialization import (
    LLAMA3_BOS_TOKEN_ID,
    generate_english_prompt,
    load_pinned_tokenizer,
    verify_prompt_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "vidur_vllm_real_testing"
TOKENIZER_DIR = PACKAGE_DIR / "tokenizer" / "llama3_8b"


class PromptMaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer, cls.manifest = load_pinned_tokenizer(TOKENIZER_DIR)

    def test_all_gv3_sizes_generate_exact_english_tokens(self) -> None:
        for index, size in enumerate((128, 256, 512, 1024, 1536, 2048, 3072, 4096)):
            with self.subTest(size=size):
                text, token_ids = generate_english_prompt(
                    self.tokenizer, target_tokens=size, request_index=index
                )
                self.assertEqual(len(token_ids), size)
                self.assertEqual(token_ids[0], LLAMA3_BOS_TOKEN_ID)
                self.assertEqual(
                    self.tokenizer.encode(text, add_special_tokens=True).ids,
                    token_ids,
                )
                self.assertIn("technical response", text)

    def test_full_arena_english_trace_retokenizes_exactly(self) -> None:
        count = verify_prompt_artifacts(
            raw_trace_path=PACKAGE_DIR / "traces" / "gv3_legal_20s_english_raw.csv",
            tokenizer_dir=TOKENIZER_DIR,
        )
        self.assertEqual(count, 79)


if __name__ == "__main__":
    unittest.main()
