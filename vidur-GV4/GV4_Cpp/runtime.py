"""Small adapters between the immutable Python config and native GV4 objects."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from GV4_Engine.action_resolver import ControllerTransitionKind
from GV4_Engine.config import GV4EngineConfig

from . import gv4_native as native


def config_from_python(config: GV4EngineConfig) -> native.Config:
    """Copy the simulation-relevant fields from one validated Python manifest."""

    config.validate()
    if config.topology.num_replicas != 1:
        raise NotImplementedError("native GV4 v1 supports exactly one replica")

    result = native.Config()
    topology = config.topology
    placement = topology.replica_placements[0]
    result.tensor_parallel_size = topology.tensor_parallel_size
    result.pipeline_parallel_size = topology.pipeline_parallel_size
    result.rank_ids = list(placement.rank_ids)
    capacities = config.rank_kv_block_capacities()
    result.rank_kv_capacity_blocks = [capacities[index] for index in placement.rank_ids]

    result.block_size_tokens = config.kv_cache.block_size_tokens
    result.max_batch_tokens = config.scheduler.max_batch_tokens
    result.max_sequences = config.scheduler.max_sequences
    result.max_prefill_chunk_tokens = config.scheduler.max_prefill_chunk_tokens
    result.max_inflight_microbatches = config.scheduler.max_inflight_microbatches
    result.inter_stage_queue_capacity = config.scheduler.inter_stage_queue_capacity
    result.request_preemption_enabled = config.scheduler.request_preemption_enabled

    timing = config.timing
    result.adversary_tick_sec = timing.adversary_tick_sec
    result.launch_window_sec = timing.launch_window_sec
    result.max_requests_per_launch_window = timing.max_requests_per_launch_window
    result.epsilon = timing.epsilon
    result.time_round_digits = timing.time_round_digits
    result.max_zero_time_transitions_per_boundary = (
        timing.max_zero_time_transitions_per_boundary
    )

    result.decode_credit_mint = (
        config.credits.decode_credit_mint_per_prefill_completion
    )
    request = config.request
    result.max_prefill_tokens_per_request = request.max_prefill_tokens_per_request
    result.min_decode_tokens_per_request = request.min_decode_tokens_per_request
    result.max_decode_tokens_per_request = request.max_decode_tokens_per_request
    result.target_decode_tokens_average = (
        request.target_decode_tokens_per_request_average
    )
    result.target_prefill_tokens_window_average = (
        request.target_prefill_tokens_per_request_window_average
    )

    result.prefill_slowdown_factor = config.slo.prefill_slowdown_factor
    result.decode_token_slo_sec = config.slo.decode_token_slo_sec
    result.violation_base_cost = config.cost.violation_base_cost
    result.lateness_cap_sec = config.cost.lateness_cap_sec
    result.terminal_drop_cost = config.cost.terminal_drop_cost
    result.automatic_drop_lateness_sec = config.cost.automatic_drop_lateness_sec
    result.discount_factor = config.reward.discount_factor
    result.discount_reference_step_sec = config.reward.discount_reference_step_sec

    controller = native.ControllerActionConfig()
    controller.preemption_rules = list(
        config.controller_actions.preemption_rule_names
    )
    controller.eviction_rules = list(config.controller_actions.eviction_rule_names)
    controller.prefill_budgets = list(config.controller_actions.prefill_budget_options)
    controller.ordering_heuristics = list(
        config.controller_actions.ordering_heuristics
    )
    result.controller_actions = controller

    adversary = native.AdversaryActionConfig()
    adversary.max_launch_count_per_tick = (
        config.adversary_actions.max_launch_count_per_tick
    )
    adversary.prefill_templates = list(
        config.adversary_actions.prefill_token_templates
    )
    adversary.stop_rules = list(config.adversary_actions.stop_rule_names)
    result.adversary_actions = adversary

    result.max_requests = config.layout.max_requests
    result.max_launch_history_entries = config.layout.max_launch_history_entries
    result.global_seed = config.global_seed
    result.enable_debug_asserts = config.enable_debug_asserts
    result.state_schema_version = config.layout.state_schema_version
    result.feature_schema_version = config.layout.feature_schema_version
    result.manifest_sha256 = config.manifest_sha256()
    result.validate()
    return result


class NativeTimingAdapter:
    """Present a native action with the small Python timing-provider protocol."""

    __slots__ = ("provider",)

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def __call__(
        self,
        state: native.State,
        action: native.ResolvedControllerAction,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        # Providers only require these fields. Reusing native allocations and
        # requests avoids rebuilding a Python state for every timing lookup.
        action_view = SimpleNamespace(
            transition_kind=ControllerTransitionKind.BATCH,
            allocations=action.allocations,
            replica_id=action.replica_id,
        )
        service, communication = self.provider(state, action_view)
        return tuple(service), tuple(communication)

    def estimate_prefill_time(self, tokens: int) -> float:
        return float(self.provider.estimate_prefill_time(tokens))


def environment_from_python(
    config: GV4EngineConfig,
    timing_provider: Any,
) -> native.Environment:
    """Construct a native environment using the same predictor object as Python."""

    timing = NativeTimingAdapter(timing_provider)
    return native.Environment(
        config_from_python(config),
        timing,
        timing.estimate_prefill_time,
    )
