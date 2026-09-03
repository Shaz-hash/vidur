import json
import logging
from typing import Generator

import numpy as np
import pandas as pd

from vidur.config import TraceRequestLengthGeneratorConfig
from vidur.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
    RequestLengthGeneratorOutput,
)

logger = logging.getLogger(__name__)


class TraceRequestLengthGenerator(BaseRequestLengthGenerator):

    def __init__(
        self,
        config: TraceRequestLengthGeneratorConfig,
        random_number_generator: Generator,
    ):
        super().__init__(config, random_number_generator)

        self.trace_df = pd.read_csv(config.trace_file)

        # scale prefill and decode tokens
        self.trace_df["num_prefill_tokens"] = (
            self.trace_df["num_prefill_tokens"] * config.prefill_scale_factor
        )
        self.trace_df["num_decode_tokens"] = (
            self.trace_df["num_decode_tokens"] * config.decode_scale_factor
        )

        # make sure all the prefill and decode counts are integers
        self.trace_df["num_prefill_tokens"] = self.trace_df[
            "num_prefill_tokens"
        ].astype(int)
        self.trace_df["num_decode_tokens"] = self.trace_df["num_decode_tokens"].astype(
            int
        )

        # assert that the total number of tokens does not exceed the max tokens
        assert (config.max_tokens is None) or all(
            self.trace_df["num_prefill_tokens"] + self.trace_df["num_decode_tokens"]
            <= config.max_tokens
        )
        assert all(self.trace_df["num_prefill_tokens"] > 0)
        assert all(self.trace_df["num_decode_tokens"] > 0)

        # Preprocess block_hash_ids, block_size
        if "block_hash_ids" in self.trace_df.columns:
            self.trace_df["block_hash_ids"] = self.trace_df["block_hash_ids"].apply(
                json.loads
            )
        else:
            self.trace_df["block_hash_ids"] = None

        if "block_size" not in self.trace_df.columns:
            self.trace_df["block_size"] = None

        # Shim session_id to None if not present
        if "session_id" not in self.trace_df.columns:
            self.trace_df["session_id"] = None
        else:
            self.trace_df["session_id"] = self.trace_df["session_id"].astype(int)

        # compute pd ratio and log the 25, 50, 75, 90, 95, 99 percentiles
        pd_ratio = (
            self.trace_df["num_prefill_tokens"] / self.trace_df["num_decode_tokens"]
        )
        logger.info(
            f"Loaded request length trace file {config.trace_file} with {len(self.trace_df)} requests"
        )
        pd_distribution = pd_ratio.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        logger.debug(f"Prompt/decode token ratio stats\n: {pd_distribution}")

        # randomly shuffle the df based on the seed
        if not self._config.preserve_request_order:
            self.trace_df = self.trace_df.sample(frac=1, random_state=self._config.seed)
        self.next_request_idx = 0

    def get_next_num_tokens(self) -> RequestLengthGeneratorOutput:
        if self.next_request_idx >= len(self.trace_df):
            self.next_request_idx = 0

        row = self.trace_df.iloc[self.next_request_idx]
        self.next_request_idx += 1

        return RequestLengthGeneratorOutput(
            num_prefill_tokens=row["num_prefill_tokens"],
            num_decode_tokens=row["num_decode_tokens"],
            block_hash_ids=row["block_hash_ids"],
            block_size=row["block_size"],
            session_id=row["session_id"],
        )

    def _snapshot_extra_state(self) -> dict:
        return {"next_request_idx": int(self.next_request_idx)}

    def _restore_extra_state(self, snapshot: dict) -> None:
        self.next_request_idx = int(snapshot.get("next_request_idx", self.next_request_idx))
