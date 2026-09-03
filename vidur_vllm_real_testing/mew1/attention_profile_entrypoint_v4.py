"""Run Vidur attention profiling through the FlashInfer 0.2 wrapper."""

from vidur.profiling.attention import main as attention_main
from vidur_vllm_real_testing.mew1.attention_wrapper_profiled_v3 import AttentionWrapper


attention_main.AttentionWrapper = AttentionWrapper


if __name__ == "__main__":
    attention_main.main()
