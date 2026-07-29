#pragma once

#include "gv2_types.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace mcts_native_gv2 {

template <typename T>
struct SampledActionSet {
    std::vector<T> actions;
    std::vector<uint8_t> mask;  // 0/1
    // Populated by compact rollout sampling; ordinary sampling uses position.
    std::vector<int> original_indices;
};

struct AdversarySamplerConfig {
    int max_launch_count_per_tick = 7;
    std::vector<int> allowed_prefill_tokens = {128, 256, 512, 1024, 1536, 2048, 3072, 4096};
    std::vector<std::string> stop_rule_names = {
        "stop_none",
        "stop_longest_decode",
        "stop_shortest_decode",
        "stop_all_decodes_over_512",
        "stop_all_decodes_over_216",
    };
    bool strict_masking = true;

    double launch_window_sec = 1.0;
    int max_requests_per_launch_window = 7;
    int prefill_window_cap_tokens = 7 * 1024;

    int max_decode_tokens_per_request = 864;
    double default_decode_slo_time = 0.05;

    // Optional template-specific prefill SLO overrides (seconds).
    std::unordered_map<int, double> prefill_slo_by_tokens;
};

struct ControllerSamplerConfig {
    std::vector<std::string> eviction_rule_names = {
        "evict_none",
        "evict_largest_prefill",
        "evict_earliest_prefill_deadline",
        "evict_prefill_missed_deadline",
        "evict_prefill_lateness_over_0p5",
        "evict_longest_decode",
        "evict_decode_lateness_over_0p5",
        "evict_prefill_highest_lateness",
        "evict_decode_highest_lateness",
    };
    std::vector<int> prefill_budget_options = {0, 128, 256, 512, 1024, 1536, 2048, 3072, 4096};
    std::vector<std::string> ordering_heuristics = {"SJF", "EDF", "LST", "LJF"};

    bool strict_masking = true;
    double eps = 1e-9;

    // Python GV3 LST uses PrefillProfile.lookup(remaining_prefill).  When this
    // profile is populated, native uses the same nearest-entry lookup.
    std::vector<int> prefill_profile_tokens;
    std::vector<double> prefill_profile_times;

    // Fallback only, used when no profile was supplied.
    double prefill_eta_tokens_per_sec = 4096.0;

    // Average decode credit available (when nonnegative mode is enabled).
    bool enforce_nonnegative_decode_credits = true;
};

int adversary_action_space_size(const AdversarySamplerConfig& cfg);
int controller_action_space_size(const ControllerSamplerConfig& cfg);

SampledActionSet<AdversaryAction> sample_adversary_actions_gv2(
    const SimState& state,
    const AdversarySamplerConfig& cfg,
    double decision_tick,
    const std::unordered_set<int>& forbidden_stop_ids = {},
    bool compact_valid_only = false,
    bool compact_request_materialization = false);

SampledActionSet<ControllerAction> sample_controller_actions_gv2(
    const SimState& state,
    const ControllerSamplerConfig& cfg,
    int decode_credit_balance,
    bool compact_valid_only = false,
    bool canonical_compact_only = false);

SampledActionSet<RolloutControllerAction> sample_controller_rollout_actions_gv2(
    const SimState& state,
    const ControllerSamplerConfig& cfg,
    int decode_credit_balance);

// Kept for backward compatibility in early native wiring.
SampledActionSet<AdversaryAction> sample_adversary_actions_simple(
    const SimState& state,
    int action_space_size);

SampledActionSet<ControllerAction> sample_controller_actions_simple(
    const SimState& state,
    int action_space_size);

}  // namespace mcts_native_gv2
