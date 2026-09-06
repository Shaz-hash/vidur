#include "gv4/config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace gv4 {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

template <typename T>
void require_unique(const std::vector<T>& values, const char* message) {
    std::unordered_set<T> seen;
    for (const T& value : values) require(seen.insert(value).second, message);
}

}  // namespace

int ControllerActionConfig::raw_action_count() const {
    return static_cast<int>(preemption_rules.size() * eviction_rules.size() *
                            prefill_budgets.size() * ordering_heuristics.size());
}

std::tuple<std::string, std::string, int, std::string>
ControllerActionConfig::components(int raw_index) const {
    if (raw_index < 0 || raw_index >= raw_action_count()) {
        throw std::out_of_range("controller raw action index is out of range");
    }
    const int heuristics = static_cast<int>(ordering_heuristics.size());
    const int budgets = static_cast<int>(prefill_budgets.size());
    const int heuristic_index = raw_index % heuristics;
    const int rule_budget_index = raw_index / heuristics;
    const int budget_index = rule_budget_index % budgets;
    const int preemption_eviction_index = rule_budget_index / budgets;
    const int eviction_count = static_cast<int>(eviction_rules.size());
    const int eviction_index = preemption_eviction_index % eviction_count;
    const int preemption_index = preemption_eviction_index / eviction_count;
    return {preemption_rules[preemption_index], eviction_rules[eviction_index],
            prefill_budgets[budget_index],
            ordering_heuristics[heuristic_index]};
}

int AdversaryActionConfig::raw_action_count() const {
    return static_cast<int>(stop_rules.size()) *
           (1 + max_launch_count_per_tick * static_cast<int>(prefill_templates.size()));
}

std::tuple<int, int, std::string>
AdversaryActionConfig::components(int raw_index) const {
    if (raw_index < 0 || raw_index >= raw_action_count()) {
        throw std::out_of_range("adversary raw action index is out of range");
    }
    const int stop_count = static_cast<int>(stop_rules.size());
    if (raw_index < stop_count) return {0, 0, stop_rules[raw_index]};
    const int launch_template_index = (raw_index - stop_count) / stop_count;
    const int stop_index = (raw_index - stop_count) % stop_count;
    const int template_count = static_cast<int>(prefill_templates.size());
    const int launch_count = launch_template_index / template_count + 1;
    const int template_index = launch_template_index % template_count;
    return {launch_count, prefill_templates[template_index], stop_rules[stop_index]};
}

void Config::validate() const {
    require(tensor_parallel_size > 0, "tensor_parallel_size must be positive");
    require(pipeline_parallel_size > 0, "pipeline_parallel_size must be positive");
    require(static_cast<int>(rank_ids.size()) ==
                tensor_parallel_size * pipeline_parallel_size,
            "single-replica rank count must equal TP * PP");
    require(rank_ids.size() == rank_kv_capacity_blocks.size(),
            "rank capacity count differs from rank count");
    for (std::size_t index = 0; index < rank_ids.size(); ++index) {
        require(rank_ids[index] == static_cast<int>(index),
                "native v1 requires contiguous rank IDs");
        require(rank_kv_capacity_blocks[index] > 0,
                "every rank must have positive KV capacity");
    }
    require(block_size_tokens > 0, "block_size_tokens must be positive");
    require(max_batch_tokens > 0 && max_sequences > 0,
            "batch limits must be positive");
    require(max_prefill_chunk_tokens > 0 &&
                max_prefill_chunk_tokens <= max_batch_tokens,
            "invalid prefill chunk limit");
    require(max_inflight_microbatches > 0,
            "max_inflight_microbatches must be positive");
    require(inter_stage_queue_capacity >= max_inflight_microbatches,
            "inter-stage capacity requires explicit backpressure");
    require(std::isfinite(adversary_tick_sec) && adversary_tick_sec > 0.0,
            "adversary tick must be finite and positive");
    require(std::isfinite(launch_window_sec) && launch_window_sec > 0.0,
            "launch window must be finite and positive");
    require(max_requests_per_launch_window > 0, "launch request cap must be positive");
    require(std::isfinite(epsilon) && epsilon > 0.0, "epsilon must be positive");
    require(time_round_digits >= 0 && time_round_digits <= 15,
            "time_round_digits must be in [0, 15]");
    require(max_zero_time_transitions_per_boundary > 0,
            "zero-time transition guard must be positive");
    require(decode_credit_mint > 0 && decode_credit_mint == target_decode_tokens_average,
            "decode mint must equal target decode average");
    require(min_decode_tokens_per_request > 0 &&
                min_decode_tokens_per_request <= max_decode_tokens_per_request,
            "invalid decode request range");
    require(max_prefill_tokens_per_request > 0,
            "max prefill tokens must be positive");
    require(max_requests > 0 && max_launch_history_entries > 0,
            "native state bounds must be positive");
    require(std::isfinite(prefill_slowdown_factor) && prefill_slowdown_factor > 0.0,
            "prefill slowdown must be positive");
    require(std::isfinite(decode_token_slo_sec) && decode_token_slo_sec > 0.0,
            "decode SLO must be positive");
    require(discount_factor > 0.0 && discount_factor <= 1.0,
            "discount factor must be in (0, 1]");
    require(discount_reference_step_sec > 0.0,
            "discount reference step must be positive");
    require(!controller_actions.eviction_rules.empty() &&
                controller_actions.eviction_rules.front() == "evict_none",
            "controller rules must begin with evict_none");
    require(!controller_actions.preemption_rules.empty() &&
                controller_actions.preemption_rules.front() == "preempt_none",
            "controller preemption rules must begin with preempt_none");
    require(!controller_actions.prefill_budgets.empty() &&
                controller_actions.prefill_budgets.front() == 0,
            "controller budgets must begin with zero");
    require(!controller_actions.ordering_heuristics.empty(),
            "controller heuristics cannot be empty");
    require(std::is_sorted(controller_actions.prefill_budgets.begin(),
                           controller_actions.prefill_budgets.end()),
            "controller budgets must be sorted");
    require_unique(controller_actions.preemption_rules,
                   "duplicate preemption rule");
    require_unique(controller_actions.eviction_rules, "duplicate eviction rule");
    require_unique(controller_actions.prefill_budgets, "duplicate prefill budget");
    require_unique(controller_actions.ordering_heuristics,
                   "duplicate ordering heuristic");
    require(!adversary_actions.stop_rules.empty() &&
                adversary_actions.stop_rules.front() == "stop_none",
            "adversary rules must begin with stop_none");
    require(adversary_actions.max_launch_count_per_tick >= 0 &&
                adversary_actions.max_launch_count_per_tick <=
                    max_requests_per_launch_window,
            "invalid per-tick launch cap");
    require_unique(adversary_actions.prefill_templates,
                   "duplicate adversary prefill template");
    require_unique(adversary_actions.stop_rules, "duplicate stop rule");
}

int Config::logical_kv_capacity_blocks() const {
    if (rank_kv_capacity_blocks.empty()) return 0;
    return *std::min_element(rank_kv_capacity_blocks.begin(),
                             rank_kv_capacity_blocks.end());
}

double Config::discount_for_elapsed(double elapsed_sec) const {
    if (!std::isfinite(elapsed_sec) || elapsed_sec < 0.0) {
        throw std::invalid_argument("elapsed_sec must be finite and nonnegative");
    }
    return std::pow(discount_factor, elapsed_sec / discount_reference_step_sec);
}

double round_time(double value, int digits) {
    if (!std::isfinite(value)) throw std::invalid_argument("cannot round non-finite time");
    const double scale = std::pow(10.0, static_cast<double>(digits));
    return std::nearbyint(value * scale) / scale;
}

}  // namespace gv4
