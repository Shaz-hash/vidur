#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace gv4 {

struct ControllerActionConfig {
    std::vector<std::string> preemption_rules{
        "preempt_none", "preempt_min_recompute", "preempt_largest_kv",
        "preempt_max_recovery_slack", "preempt_best_relief_cost"};
    std::vector<std::string> eviction_rules{
        "evict_none", "evict_largest_prefill", "evict_earliest_prefill_deadline",
        "evict_prefill_missed_deadline", "evict_prefill_lateness_over_0p5",
        "evict_longest_decode", "evict_decode_lateness_over_0p5",
        "evict_prefill_highest_lateness", "evict_decode_highest_lateness"};
    std::vector<int> prefill_budgets{0, 128, 256, 512, 1024, 1536, 2048, 3072, 4096};
    std::vector<std::string> ordering_heuristics{"SJF", "EDF", "LST", "LJF"};

    [[nodiscard]] int raw_action_count() const;
    [[nodiscard]] std::tuple<std::string, std::string, int, std::string>
    components(int raw_index) const;
};

struct AdversaryActionConfig {
    int max_launch_count_per_tick = 7;
    std::vector<int> prefill_templates{128, 256, 512, 1024, 1536, 2048, 3072, 4096};
    std::vector<std::string> stop_rules{
        "stop_none", "stop_longest_decode", "stop_shortest_decode",
        "stop_all_decodes_over_512", "stop_all_decodes_over_216"};

    [[nodiscard]] int raw_action_count() const;
    [[nodiscard]] std::tuple<int, int, std::string> components(int raw_index) const;
};

struct Config {
    // GV4 native v1 intentionally accepts one replica only.
    int tensor_parallel_size = 2;
    int pipeline_parallel_size = 2;
    std::vector<int> rank_ids{0, 1, 2, 3};
    std::vector<int> rank_kv_capacity_blocks;

    int block_size_tokens = 16;
    int max_batch_tokens = 4608;
    int max_sequences = 256;
    int max_prefill_chunk_tokens = 4096;
    int max_inflight_microbatches = 2;
    int inter_stage_queue_capacity = 2;
    bool request_preemption_enabled = true;

    double adversary_tick_sec = 0.2;
    double launch_window_sec = 1.0;
    int max_requests_per_launch_window = 7;
    double epsilon = 1e-6;
    int time_round_digits = 10;
    int max_zero_time_transitions_per_boundary = 512;

    int decode_credit_mint = 216;
    int max_prefill_tokens_per_request = 4096;
    int min_decode_tokens_per_request = 1;
    int max_decode_tokens_per_request = 864;
    int target_decode_tokens_average = 216;
    int target_prefill_tokens_window_average = 1024;

    double prefill_slowdown_factor = 3.0;
    double decode_token_slo_sec = 0.050;
    double violation_base_cost = 1.0;
    double lateness_cap_sec = 2.0;
    double terminal_drop_cost = 3.0;
    double automatic_drop_lateness_sec = 2.0;
    double discount_factor = 0.98;
    double discount_reference_step_sec = 0.015725797204323228;

    ControllerActionConfig controller_actions;
    AdversaryActionConfig adversary_actions;

    int max_requests = 512;
    int max_launch_history_entries = 64;
    std::uint64_t global_seed = 6;
    bool enable_debug_asserts = true;
    std::string state_schema_version = "gv4_state_v5";
    std::string feature_schema_version = "gv4_markov_v5";
    std::string manifest_sha256;

    void validate() const;
    [[nodiscard]] int logical_kv_capacity_blocks() const;
    [[nodiscard]] double discount_for_elapsed(double elapsed_sec) const;
};

[[nodiscard]] double round_time(double value, int digits);

}  // namespace gv4
