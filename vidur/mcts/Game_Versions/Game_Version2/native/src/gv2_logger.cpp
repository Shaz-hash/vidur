#include "gv2_logger.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mcts_native_gv2 {
namespace {

void ensure_parent_dir(const std::string& path) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
}

bool file_has_content(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good() && f.peek() != std::ifstream::traits_type::eof();
}

std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

const char* bool_str(bool v) {
    return v ? "true" : "false";
}

}  // namespace

NativeIterCsvLogger::NativeIterCsvLogger(std::string path, int flush_every)
    : path_(std::move(path)),
      flush_every_(std::max(1, flush_every)) {}

void NativeIterCsvLogger::ensure_header() {
    if (header_written_) return;
    ensure_parent_dir(path_);
    owned_out_ = std::make_unique<std::ofstream>(path_, std::ios::out | std::ios::app);
    if (!owned_out_ || !owned_out_->good()) {
        throw std::runtime_error("Failed to open native iter log: " + path_);
    }
    out_ = owned_out_.get();

    if (!file_has_content(path_)) {
        (*out_)
            << "game_id,root_id,sim_iteration,"
               "root_depth,root_node_id,root_player,"
               "phase,node_depth,parent_node_id,node_id,"
               "player_acted_to_create_this_node,player_to_act_in_this_node,"
               "action_index,action_repr,prior,model_prior_json,normalized_prior_json,reward,"
               "nn_called,num_valid_actions,unique_actions,nn_value_controller,"
               "objective_cost,sim_time,decision_state_time,start_time,end_time,stage_total_time,"
               "requests_in_system,requests_generated,requests_completed,slo_violations,total_lateness,avg_lateness,"
               "state_active_ids,state_waiting_ids,state_completed_request_ids,state_dropped_request_ids,"
               "state_stopped_decode_request_ids,state_pending_adv_tick,state_last_adv_tick,"
               "state_decode_credit_balance,state_decode_tokens_counted_by_id,state_violated_request_ids,"
               "state_per_request_prefill_lateness_by_id,state_per_request_decode_lateness_by_id,"
               "adversary_requests,adversary_prefill_slos,adversary_prefill_deadlines_by_id,adversary_decode_slos,"
               "controller_token_budget,controller_selected_ids,controller_allocations,"
               "controller_prefill_allocations,controller_decode_allocations,controller_prefill_total,"
               "controller_decode_total,controller_heuristic,controller_strategy\n";
        out_->flush();
    }
    header_written_ = true;
}

void NativeIterCsvLogger::maybe_flush() {
    rows_since_flush_ += 1;
    if (out_ != nullptr && rows_since_flush_ >= flush_every_) {
        out_->flush();
        rows_since_flush_ = 0;
    }
}

void NativeIterCsvLogger::write(const NativeIterLogRow& row) {
    ensure_header();
    (*out_) << row.game_id << ","
            << row.root_id << ","
            << row.sim_iteration << ","
            << row.root_depth << ","
            << row.root_node_id << ","
            << csv_escape(row.root_player) << ","
            << csv_escape(row.phase) << ","
            << row.node_depth << ","
            << (row.has_parent_node_id ? std::to_string(row.parent_node_id) : std::string()) << ","
            << row.node_id << ","
            << csv_escape(row.player_acted_to_create_this_node) << ","
            << csv_escape(row.player_to_act_in_this_node) << ","
            << row.action_index << ","
            << csv_escape(row.action_repr) << ","
            << row.prior << ","
            << csv_escape(row.model_prior_json) << ","
            << csv_escape(row.normalized_prior_json) << ","
            << row.reward << ","
            << bool_str(row.nn_called) << ","
            << row.num_valid_actions << ","
            << row.unique_actions << ","
            << (row.has_nn_value_controller ? std::to_string(row.nn_value_controller) : std::string()) << ","
            << row.objective_cost << ","
            << row.sim_time << ","
            << row.decision_state_time << ","
            << row.start_time << ","
            << row.end_time << ","
            << row.stage_total_time << ","
            << row.requests_in_system << ","
            << row.requests_generated << ","
            << row.requests_completed << ","
            << row.slo_violations << ","
            << row.total_lateness << ","
            << row.avg_lateness << ","
            << csv_escape(row.state_active_ids) << ","
            << csv_escape(row.state_waiting_ids) << ","
            << csv_escape(row.state_completed_request_ids) << ","
            << csv_escape(row.state_dropped_request_ids) << ","
            << csv_escape(row.state_stopped_decode_request_ids) << ","
            << bool_str(row.state_pending_adv_tick) << ","
            << (row.has_state_last_adv_tick ? std::to_string(row.state_last_adv_tick) : std::string()) << ","
            << row.state_decode_credit_balance << ","
            << csv_escape(row.state_decode_tokens_counted_by_id) << ","
            << csv_escape(row.state_violated_request_ids) << ","
            << csv_escape(row.state_per_request_prefill_lateness_by_id) << ","
            << csv_escape(row.state_per_request_decode_lateness_by_id) << ","
            << csv_escape(row.adversary_requests) << ","
            << csv_escape(row.adversary_prefill_slos) << ","
            << csv_escape(row.adversary_prefill_deadlines_by_id) << ","
            << csv_escape(row.adversary_decode_slos) << ","
            << (row.has_controller_token_budget ? std::to_string(row.controller_token_budget) : std::string()) << ","
            << csv_escape(row.controller_selected_ids) << ","
            << csv_escape(row.controller_allocations) << ","
            << csv_escape(row.controller_prefill_allocations) << ","
            << csv_escape(row.controller_decode_allocations) << ","
            << row.controller_prefill_total << ","
            << row.controller_decode_total << ","
            << csv_escape(row.controller_heuristic) << ","
            << csv_escape(row.controller_strategy)
            << "\n";
    maybe_flush();
}

NativeRootCsvLogger::NativeRootCsvLogger(std::string path, int flush_every)
    : path_(std::move(path)),
      flush_every_(std::max(1, flush_every)) {}

void NativeRootCsvLogger::ensure_header() {
    if (header_written_) return;
    ensure_parent_dir(path_);
    owned_out_ = std::make_unique<std::ofstream>(path_, std::ios::out | std::ios::app);
    if (!owned_out_ || !owned_out_->good()) {
        throw std::runtime_error("Failed to open native root log: " + path_);
    }
    out_ = owned_out_.get();

    if (!file_has_content(path_)) {
        (*out_)
            << "game_id,root_id,root_depth,root_node_id,root_player,num_simulations,"
               "model_root_value_controller,model_root_prior_json,normalized_root_prior_json,valid_action_mask_json,"
               "mcts_root_value_controller,mcts_root_prior_json,best_action_index,best_action_mcts_prob,best_action_model_prob,"
               "best_action_repr,best_action_json,phase,cycle_label,sim_time,decision_state_time,"
               "state_pending_adv_tick,state_last_adv_tick,state_active_ids,state_completed_request_ids,"
               "state_decode_credit_balance,state_decode_tokens_counted_by_id,slo_violations,total_lateness,total_cost\n";
        out_->flush();
    }
    header_written_ = true;
}

void NativeRootCsvLogger::maybe_flush() {
    rows_since_flush_ += 1;
    if (out_ != nullptr && rows_since_flush_ >= flush_every_) {
        out_->flush();
        rows_since_flush_ = 0;
    }
}

void NativeRootCsvLogger::write(const NativeRootLogRow& row) {
    ensure_header();
    (*out_) << row.game_id << ","
            << row.root_id << ","
            << row.root_depth << ","
            << row.root_node_id << ","
            << csv_escape(row.root_player) << ","
            << row.num_simulations << ","
            << row.model_root_value_controller << ","
            << csv_escape(row.model_root_prior_json) << ","
            << csv_escape(row.normalized_root_prior_json) << ","
            << csv_escape(row.valid_action_mask_json) << ","
            << row.mcts_root_value_controller << ","
            << csv_escape(row.mcts_root_prior_json) << ","
            << row.best_action_index << ","
            << row.best_action_mcts_prob << ","
            << row.best_action_model_prob << ","
            << csv_escape(row.best_action_repr) << ","
            << csv_escape(row.best_action_json) << ","
            << csv_escape(row.phase) << ","
            << csv_escape(row.cycle_label) << ","
            << row.sim_time << ","
            << row.decision_state_time << ","
            << bool_str(row.state_pending_adv_tick) << ","
            << (row.has_state_last_adv_tick ? std::to_string(row.state_last_adv_tick) : std::string()) << ","
            << csv_escape(row.state_active_ids) << ","
            << csv_escape(row.state_completed_request_ids) << ","
            << row.state_decode_credit_balance << ","
            << csv_escape(row.state_decode_tokens_counted_by_id) << ","
            << row.slo_violations << ","
            << row.total_lateness << ","
            << row.total_cost
            << "\n";
    maybe_flush();
}

NativeConfigJsonLogger::NativeConfigJsonLogger(std::string path)
    : path_(std::move(path)) {}

void NativeConfigJsonLogger::write_once(const std::string& json_payload) {
    if (path_.empty()) return;
    ensure_parent_dir(path_);
    if (file_has_content(path_)) return;

    std::ofstream out(path_, std::ios::out | std::ios::trunc);
    if (!out.good()) {
        throw std::runtime_error("Failed to open native config log: " + path_);
    }
    out << json_payload;
    out.flush();
}

std::string default_native_config_json_path(
    const std::string& iter_log_path,
    const std::string& root_log_path) {
    const std::string& base = root_log_path.empty() ? iter_log_path : root_log_path;
    if (base.empty()) return "";

    std::filesystem::path p(base);
    std::filesystem::path dir = p.has_parent_path() ? p.parent_path() : std::filesystem::path(".");
    std::string stem = p.stem().string();
    if (stem.empty()) {
        return (dir / "native_config.json").string();
    }
    return (dir / (stem + ".config.json")).string();
}

}  // namespace mcts_native_gv2
