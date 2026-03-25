#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mcts_native {

struct AllocationEntry {
    int request_id = -1;
    int tokens = 0;
};

struct ControllerRequestStateNative {
    int request_id = -1;
    bool prefill_done = false;
    int remaining_prefill = 0;
    int remaining_decode = 0;
    double arrived_at = 0.0;
    double prefill_slo = 0.0;
    int num_processed_tokens = 0;
};

struct ControllerActionSpecNative {
    int token_budget = 0;
    std::vector<int> selected_request_ids;
    std::vector<AllocationEntry> token_allocations;
    std::vector<AllocationEntry> prefill_allocations;
    std::vector<AllocationEntry> decode_allocations;
    std::string heuristic;
    std::string strategy;
    bool valid = false;
};

struct AdversaryRequestSpecNative {
    int prefill_tokens = 0;
    int decode_tokens = 0;
    double prefill_slo = 0.0;
    double decode_slo = 0.0;
};

struct AdversaryActionSpecNative {
    std::vector<AdversaryRequestSpecNative> requests;
    std::vector<int> stop_decode_ids;
    bool valid = false;
};

struct ControllerSampleOutput {
    std::vector<ControllerActionSpecNative> actions;
    std::vector<int> mask; // 0/1
};

struct AdversarySampleOutput {
    std::vector<AdversaryActionSpecNative> actions;
    std::vector<int> mask; // 0/1
};

struct PrefillRecord {
    int rid = -1;
    int rem_pref = 0;
    double edf_key = 0.0;
    double lst_key = 0.0;
};

struct NativeRequestState {
    int request_id = -1;
    double arrived_at = 0.0;
    double queued_at = 0.0;

    int num_prefill_tokens = 0;
    int num_processed_prefill_tokens = 0;
    int num_decode_tokens = 0;
    int num_processed_decode_tokens = 0;

    bool prefill_done = false;
    bool completed = false;
    double prefill_completed_at = 0.0;

    double prefill_slo = 0.0;
    double decode_slo = 0.0;
};

struct NativeGameStats {
    int requests_generated = 0;
    int requests_completed = 0;
    int slo_violations = 0;
    double slo_lateness_sum = 0.0;
    int maximum_qps = 5;
    double last_prefill_batch_time = -1.0;  // < 0 means unset

    std::vector<double> recent_arrivals;

    std::unordered_map<int, double> per_request_prefill_lateness;
    std::unordered_map<int, double> per_request_decode_lateness;
    std::unordered_map<int, int> decode_tokens_counted;
    std::unordered_map<int, double> decode_next_deadline_by_id;

    std::unordered_set<int> prefill_lateness_finalized;
    std::unordered_set<int> violated_request_ids;
    std::unordered_set<int> active_request_ids;
    std::unordered_set<int> completed_request_ids;
};

struct NativeSimState {
    double sim_time = 0.0;
    int next_request_id = 0;
    std::vector<NativeRequestState> requests;
    NativeGameStats stats;
};

struct NativeRuntimeConfig {
    int interval_request_size = 512;
    int max_request_tokens = 3072;
    int adversary_num_actions = 6;
    int adversary_fixed_decode_tokens = 5000;
    double prefill_slowdown = 3.0;
    double default_decode_slo = 0.05;

    std::vector<int> prefill_profile_tokens;
    std::vector<double> prefill_profile_times;
};

} // namespace mcts_native
