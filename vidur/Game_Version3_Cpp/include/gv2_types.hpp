#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcts_native_gv2 {

// Frozen native contract version for GV2 phase-1.
inline constexpr const char* kGV2NativeContractVersion = "gv2_native_contract_v1";

struct RequestState {
    int request_id = -1;
    double arrived_at = 0.0;
    double queued_at = 0.0;

    int num_prefill_tokens = 0;
    int num_processed_prefill_tokens = 0;

    int num_decode_tokens = 0;
    int num_processed_decode_tokens = 0;

    // Request-level SLOs and deadline state.
    double prefill_slo_time = 0.0;
    double decode_slo_time = 0.0;
    double completion_slo_time = -1.0;

    double prefill_deadline = -1.0;
    double decode_next_deadline = -1.0;
    double prefill_completed_at = -1.0;
    double completed_at = -1.0;

    // Request-level lateness accumulators.
    double prefill_lateness = 0.0;  // monotone max
    double decode_lateness = 0.0;   // cumulative

    bool is_prefill_complete = false;
    bool completed = false;
    bool dropped = false;
    bool stopped_decode = false;
    bool violated = false;
    // Present only to reproduce Python HGB snapshot features. These records
    // must not participate in engine dynamics or active-id rebuilds.
    bool feature_only = false;

    int remaining_prefill() const {
        const int rem = num_prefill_tokens - num_processed_prefill_tokens;
        return rem > 0 ? rem : 0;
    }

    int remaining_decode() const {
        const int rem = num_decode_tokens - num_processed_decode_tokens;
        return rem > 0 ? rem : 0;
    }

    bool prefill_done() const { return is_prefill_complete || remaining_prefill() == 0; }
    bool decode_active() const { return !feature_only && prefill_done() && !completed && remaining_decode() > 0; }
    bool prefill_active() const { return !feature_only && !completed && !prefill_done() && remaining_prefill() > 0; }
};

struct LaunchWindowEntry {
    double timestamp = 0.0;
    int count = 0;
    int prefill_tokens = 0;
};

struct GameStats {
    int requests_generated = 0;
    int requests_completed = 0;

    int slo_violations = 0;
    double slo_lateness_sum = 0.0;
    std::vector<double> recent_arrivals;
    std::vector<LaunchWindowEntry> recent_launches;

    std::vector<int> active_request_ids;
    std::vector<int> completed_request_ids;
    std::vector<int> dropped_request_ids;
    std::vector<int> stopped_decode_request_ids;
    std::vector<int> violated_request_ids;

    std::unordered_map<int, double> per_request_prefill_lateness_by_id;
    std::unordered_map<int, double> per_request_decode_lateness_by_id;
    std::unordered_map<int, double> decode_next_deadline_by_id;

    std::unordered_map<int, int> decode_tokens_counted_by_id;

    int decode_credit_balance = 0;
    int decode_credit_available = 0;

    bool pending_adv_tick = false;
    double last_adv_tick = -1.0;
    double next_adv_tick = -1.0;
    int missed_adv_source = 0;
    double transition_discount_time = -1.0;
    double transition_final_time = -1.0;

    // Requests whose prefill lateness already transitioned to finalized.
    std::vector<int> prefill_lateness_finalized_ids;
};

struct SimState {
    double sim_time = 0.0;
    double decision_state_time = 0.0;
    int next_request_id = 0;
    std::vector<RequestState> requests;
    GameStats stats;
};

struct AdversaryRequestSpec {
    int prefill_tokens = 0;
    int decode_tokens = 0;
    double prefill_slo = 0.0;
    double decode_slo = 0.0;
};

struct AdversaryAction {
    std::vector<AdversaryRequestSpec> requests;
    std::vector<int> stop_decode_ids;
    bool valid = false;
};

struct ControllerAction {
    int token_budget = 0;
    std::vector<int> selected_request_ids;
    std::vector<int> evicted_request_ids;
    std::unordered_map<int, int> token_allocations;
    std::unordered_map<int, int> prefill_allocations;
    std::unordered_map<int, int> decode_allocations;
    std::string heuristic;
    std::string strategy;
    std::array<int, 3> mapping = {-1, -1, -1};
    bool has_mapping = false;
    bool valid = false;
};

struct IterEvent {
    int sim_iteration = 0;
    int selected_action_index = -1;      // root selected action for this simulation
    int selected_child_node_id = -1;     // root selected child node-id

    int action_index = -1;               // incoming edge index for leaf node
    int leaf_node_id = -1;
    int parent_node_id = -1;
    int leaf_depth = 0;

    double sim_time_before = 0.0;
    double sim_time_after = 0.0;
    double decision_state_time = 0.0;
    double leaf_state_cost = 0.0;
    double prior = 0.0;
    double reward = 0.0;

    int decode_credit_balance = 0;
    int num_valid_actions = 0;
    int unique_actions = 0;
    int root_visits_after = 0;
    double root_value_sum_after = 0.0;
    double root_mean_value_after = 0.0;
    int selected_child_visits_after = 0;
    double selected_child_value_sum_after = 0.0;
    double selected_child_mean_value_after = 0.0;
    double selected_child_prior = 0.0;
    bool nn_called = false;
    bool has_nn_value_controller = false;
    double nn_value_controller = 0.0;

    int requests_in_system = 0;
    int requests_generated = 0;
    int requests_completed = 0;
    int slo_violations = 0;
    double total_lateness = 0.0;
    double avg_lateness = 0.0;
    bool state_pending_adv_tick = false;
    bool has_state_last_adv_tick = false;
    double state_last_adv_tick = 0.0;

    std::string player_to_act;
    std::string player_acted_to_create_this_node;
    std::string phase;
    std::string action_repr;
    std::string active_request_ids_json;
    std::string waiting_request_ids_json;
    std::string completed_request_ids_json;
    std::string dropped_request_ids_json;
    std::string stopped_decode_request_ids_json;
    std::string violated_request_ids_json;
    std::string decode_tokens_counted_by_id_json;
    std::string per_request_prefill_lateness_by_id_json;
    std::string per_request_decode_lateness_by_id_json;
    std::string root_valid_mask_json;
    std::string root_nn_priors_json;
    std::string root_nn_priors_after_threshold_json;
    std::string root_mcts_prior_json;

    // action payloads (mirror python logger columns)
    std::string adversary_requests_json;
    std::string adversary_prefill_slos_json;
    std::string adversary_prefill_deadlines_by_id_json;
    std::string adversary_decode_slos_json;
    bool has_controller_token_budget = false;
    int controller_token_budget = 0;
    std::string controller_selected_ids_json;
    std::string controller_allocations_json;
    std::string controller_prefill_allocations_json;
    std::string controller_decode_allocations_json;
    int controller_prefill_total = 0;
    int controller_decode_total = 0;
    std::string controller_heuristic;
    std::string controller_strategy;
};

struct ChildSummary {
    int index = -1;
    int node_id = -1;
    int depth = 0;
    std::string player;

    double prior = 0.0;
    double reward = 0.0;
    double edge_discount = 1.0;
    int visits = 0;
    double value_sum = 0.0;
    double sim_time = 0.0;
    double state_cost = 0.0;
    int num_valid_actions = 0;

    std::string parent_action_json;
};

struct SearchOutput {
    std::string contract_version = kGV2NativeContractVersion;
    double decision_state_time = 0.0;
    SimState root_state_echo;

    int root_visits = 0;
    double root_value_sum = 0.0;
    double root_state_cost = 0.0;
    double root_sim_time = 0.0;
    int root_num_valid_actions = 0;

    double root_nn_value_controller = 0.0;
    std::vector<double> root_nn_priors;
    std::vector<double> root_nn_priors_after_threshold;
    std::vector<uint8_t> root_nn_valid_mask;

    std::unordered_map<int, int> action_alias_to_canonical;
    std::unordered_map<int, std::vector<int>> canonical_to_action_aliases;

    std::vector<ChildSummary> children;
    std::vector<double> mcts_root_prior;
    std::vector<IterEvent> iter_events;

    std::unordered_map<std::string, double> perf;

    // Root tensors used by native inference/search. These mirror
    // NativeInferInputsGV2 without depending on gv2_infer_runtime.hpp here.
    std::vector<float> root_global_features;
    std::vector<uint8_t> root_action_mask;
    std::vector<float> root_prefill_req_features;
    std::vector<float> root_decode_req_features;
    std::vector<uint8_t> root_prefill_req_mask;
    std::vector<uint8_t> root_decode_req_mask;
    int root_prefill_req_n = 0;
    int root_prefill_req_d = 0;
    int root_decode_req_n = 0;
    int root_decode_req_d = 0;
    std::vector<float> root_req_features;
    std::vector<uint8_t> root_req_mask;
    int root_req_n = 0;
    int root_req_d = 0;

    int best_action_index = -1;
    std::vector<double> root_action_values;
    std::vector<double> root_action_rewards;
    std::vector<double> root_action_discounts;
    std::vector<double> root_action_bootstraps;
    std::vector<std::string> root_action_reprs;
    std::vector<int> root_action_leaf_prefill_counts;
    std::vector<int> root_action_leaf_decode_counts;
    std::vector<int> root_action_leaf_decode_credit_balances;
};

std::vector<double> normalize_masked(
    const std::vector<double>& raw,
    const std::vector<uint8_t>& mask);

int argmax_masked(const std::vector<double>& values, const std::vector<uint8_t>& mask);

}  // namespace mcts_native_gv2
