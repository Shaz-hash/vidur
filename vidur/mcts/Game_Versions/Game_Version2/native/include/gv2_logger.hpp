#pragma once

#include "gv2_types.hpp"

#include <fstream>
#include <memory>
#include <string>

namespace mcts_native_gv2 {

struct NativeIterLogRow {
    int game_id = 0;
    int root_id = 0;
    int sim_iteration = 0;

    int root_depth = 0;
    int root_node_id = 0;
    std::string root_player;

    std::string phase;
    int node_depth = 0;
    bool has_parent_node_id = false;
    int parent_node_id = 0;
    int node_id = 0;
    std::string player_acted_to_create_this_node;
    std::string player_to_act_in_this_node;

    int action_index = -1;
    std::string action_repr;
    double prior = 0.0;
    std::string model_prior_json;
    std::string normalized_prior_json;
    double reward = 0.0;

    bool nn_called = false;
    int num_valid_actions = 0;
    int unique_actions = 0;
    bool has_nn_value_controller = false;
    double nn_value_controller = 0.0;

    double objective_cost = 0.0;
    double sim_time = 0.0;
    double decision_state_time = 0.0;
    double start_time = 0.0;
    double end_time = 0.0;
    double stage_total_time = 0.0;

    int requests_in_system = 0;
    int requests_generated = 0;
    int requests_completed = 0;
    int slo_violations = 0;
    double total_lateness = 0.0;
    double avg_lateness = 0.0;
    std::string state_active_ids;
    std::string state_waiting_ids;
    std::string state_completed_request_ids;
    std::string state_dropped_request_ids;
    std::string state_stopped_decode_request_ids;
    bool state_pending_adv_tick = false;
    bool has_state_last_adv_tick = false;
    double state_last_adv_tick = 0.0;
    int state_decode_credit_balance = 0;
    std::string state_decode_tokens_counted_by_id;
    std::string state_violated_request_ids;
    std::string state_per_request_prefill_lateness_by_id;
    std::string state_per_request_decode_lateness_by_id;

    std::string adversary_requests;
    std::string adversary_prefill_slos;
    std::string adversary_prefill_deadlines_by_id;
    std::string adversary_decode_slos;
    bool has_controller_token_budget = false;
    int controller_token_budget = 0;
    std::string controller_selected_ids;
    std::string controller_allocations;
    std::string controller_prefill_allocations;
    std::string controller_decode_allocations;
    int controller_prefill_total = 0;
    int controller_decode_total = 0;
    std::string controller_heuristic;
    std::string controller_strategy;
};

struct NativeRootLogRow {
    int game_id = 0;
    int root_id = 0;
    int root_depth = 0;
    int root_node_id = 0;
    std::string root_player;
    int num_simulations = 0;

    double model_root_value_controller = 0.0;
    std::string model_root_prior_json;
    std::string normalized_root_prior_json;
    std::string valid_action_mask_json;
    double mcts_root_value_controller = 0.0;
    std::string mcts_root_prior_json;
    int best_action_index = -1;
    double best_action_mcts_prob = 0.0;
    double best_action_model_prob = 0.0;
    std::string best_action_repr;
    std::string best_action_json;
    std::string phase;
    std::string cycle_label;

    double sim_time = 0.0;
    double decision_state_time = 0.0;
    bool state_pending_adv_tick = false;
    bool has_state_last_adv_tick = false;
    double state_last_adv_tick = 0.0;
    std::string state_active_ids;
    std::string state_completed_request_ids;
    int state_decode_credit_balance = 0;
    std::string state_decode_tokens_counted_by_id;
    int slo_violations = 0;
    double total_lateness = 0.0;
    double total_cost = 0.0;
};

class NativeIterCsvLogger {
public:
    explicit NativeIterCsvLogger(std::string path, int flush_every = 1);
    void write(const NativeIterLogRow& row);

private:
    std::string path_;
    int flush_every_ = 1;
    int rows_since_flush_ = 0;
    bool header_written_ = false;
    std::ofstream* out_ = nullptr;
    std::unique_ptr<std::ofstream> owned_out_;
    void ensure_header();
    void maybe_flush();
};

class NativeRootCsvLogger {
public:
    explicit NativeRootCsvLogger(std::string path, int flush_every = 1);
    void write(const NativeRootLogRow& row);

private:
    std::string path_;
    int flush_every_ = 1;
    int rows_since_flush_ = 0;
    bool header_written_ = false;
    std::ofstream* out_ = nullptr;
    std::unique_ptr<std::ofstream> owned_out_;
    void ensure_header();
    void maybe_flush();
};

}  // namespace mcts_native_gv2
