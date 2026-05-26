#include "gv3_native_selfplay.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <unordered_set>

#include "gv2_logger.hpp"

namespace mcts_native_gv2 {
namespace {

struct AnyActions {
    SampledActionSet<ControllerAction> controller;
    SampledActionSet<AdversaryAction> adversary;
    std::vector<int> valid;
};

struct HistoryTraceStep {
    SimState state;
    int node_id = 0;
    int parent_node_id = -1;
    int depth = 0;
    int history_hops = 0;
    int action_index = -1;
    std::string player_acted;
    std::string player_to_act;
    std::string action_repr;
    bool action_is_controller = false;
    ControllerAction controller_action;
    AdversaryAction adversary_action;
};

struct HistoryNode {
    SimState state;
    SimState pre_controller_state;
    bool has_pre_controller_state = false;
    std::string player = "adversary";
    int depth = 0;
    int history_hops = 0;
    int node_id = 0;
    int parent_node_id = -1;
    std::vector<int> untried_action_indices;
    std::vector<HistoryTraceStep> trace;
};

struct HistorySession {
    SimState start_state;
    std::string start_player = "adversary";
    int start_depth = 0;
    int min_history_hops = 0;
    int max_history_hops = 0;
    int max_total_steps = 20000;
    int target_roots = 0;
    int next_root_id = 0;
    std::mt19937 rng;
    int anchor_budget = 0;
    int max_iterations = 0;
    int emitted_roots = 0;
    int anchors_built = 0;
    int iterations = 0;
    int next_node_id = 1;
    bool exhausted = false;
    bool allow_duplicate_fallback = true;
    bool record_trace = false;
    std::unordered_set<std::string> seen_signatures;
    std::vector<HistoryNode> active_path;
};

double round9(double x) {
    constexpr double k = 1000000000.0;
    return std::round(x * k) / k;
}

std::string ids_key(const std::vector<int>& ids) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) oss << ",";
        oss << ids[i];
    }
    return oss.str();
}

void ensure_parent_dir(const std::string& path) {
    if (path.empty()) return;
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
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

std::string json_int_vec(const std::vector<int>& xs) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << xs[i];
    }
    oss << "]";
    return oss.str();
}

std::string json_i32_i32_map(const std::unordered_map<int, int>& mp) {
    std::vector<int> keys;
    keys.reserve(mp.size());
    for (const auto& kv : mp) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    std::ostringstream oss;
    oss << "{";
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i > 0) oss << ",";
        const auto it = mp.find(keys[i]);
        if (it == mp.end()) continue;
        oss << "\"" << keys[i] << "\":" << it->second;
    }
    oss << "}";
    return oss.str();
}

std::string json_i32_f64_map(const std::unordered_map<int, double>& mp) {
    std::vector<int> keys;
    keys.reserve(mp.size());
    for (const auto& kv : mp) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "{";
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i > 0) oss << ",";
        const auto it = mp.find(keys[i]);
        if (it == mp.end()) continue;
        oss << "\"" << keys[i] << "\":" << it->second;
    }
    oss << "}";
    return oss.str();
}

std::string json_recent_launches_for_features(const GameStats& stats) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    if (!stats.recent_launches.empty()) {
        for (std::size_t i = 0; i < stats.recent_launches.size(); ++i) {
            if (i > 0) oss << ",";
            const auto& e = stats.recent_launches[i];
            oss << "{"
                << "\"timestamp\":" << e.timestamp << ","
                << "\"count\":" << e.count << ","
                << "\"prefill_tokens\":" << e.prefill_tokens
                << "}";
        }
    } else {
        for (std::size_t i = 0; i < stats.recent_arrivals.size(); ++i) {
            if (i > 0) oss << ",";
            oss << stats.recent_arrivals[i];
        }
    }
    oss << "]";
    return oss.str();
}

std::string json_request_snapshots_for_features(const SimState& state) {
    std::vector<const RequestState*> reqs;
    reqs.reserve(state.requests.size());
    for (const auto& req : state.requests) reqs.push_back(&req);
    std::sort(reqs.begin(), reqs.end(), [](const RequestState* a, const RequestState* b) {
        if (a == nullptr || b == nullptr) return a < b;
        return a->request_id < b->request_id;
    });

    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (std::size_t i = 0; i < reqs.size(); ++i) {
        if (i > 0) oss << ",";
        const RequestState& req = *reqs[i];
        oss << "{"
            << "\"request_id\":" << req.request_id << ","
            << "\"arrived_at\":" << req.arrived_at << ","
            << "\"queued_at\":" << req.queued_at << ","
            << "\"num_prefill_tokens\":" << req.num_prefill_tokens << ","
            << "\"num_processed_prefill_tokens\":" << req.num_processed_prefill_tokens << ","
            << "\"num_decode_tokens\":" << req.num_decode_tokens << ","
            << "\"num_processed_decode_tokens\":" << req.num_processed_decode_tokens << ","
            << "\"is_prefill_complete\":" << (req.is_prefill_complete ? "true" : "false") << ","
            << "\"completed\":" << (req.completed ? "true" : "false") << ","
            << "\"dropped\":" << (req.dropped ? "true" : "false") << ","
            << "\"stopped_decode\":" << (req.stopped_decode ? "true" : "false") << ","
            << "\"violated\":" << (req.violated ? "true" : "false") << ","
            << "\"prefill_slo_time\":" << req.prefill_slo_time << ","
            << "\"decode_slo_time\":" << req.decode_slo_time << ","
            << "\"prefill_deadline\":" << req.prefill_deadline << ","
            << "\"decode_next_deadline\":" << req.decode_next_deadline << ","
            << "\"prefill_completed_at\":" << req.prefill_completed_at << ","
            << "\"completed_at\":" << req.completed_at << ","
            << "\"prefill_lateness\":" << req.prefill_lateness << ","
            << "\"decode_lateness\":" << req.decode_lateness
            << "}";
    }
    oss << "]";
    return oss.str();
}

std::string controller_action_to_repr_local(const ControllerAction& a) {
    auto map_repr = [](const std::unordered_map<int, int>& mp) {
        std::vector<int> keys;
        keys.reserve(mp.size());
        for (const auto& kv : mp) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        std::ostringstream oss;
        oss << "{";
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (i > 0) oss << ", ";
            const auto it = mp.find(keys[i]);
            if (it == mp.end()) continue;
            oss << keys[i] << ": " << it->second;
        }
        oss << "}";
        return oss.str();
    };
    std::ostringstream oss;
    oss << "ControllerAction("
        << "token_budget=" << a.token_budget
        << ", selected_request_ids=" << json_int_vec(a.selected_request_ids)
        << ", token_allocations=" << map_repr(a.token_allocations)
        << ", prefill_allocations=" << map_repr(a.prefill_allocations)
        << ", decode_allocations=" << map_repr(a.decode_allocations)
        << ", heuristic='" << a.heuristic << "'"
        << ", strategy='" << a.strategy << "'";
    if (a.has_mapping) {
        oss << ", mapping=(" << a.mapping[0] << ", " << a.mapping[1] << ", " << a.mapping[2] << ")";
    }
    oss << ")";
    return oss.str();
}

std::string adversary_action_to_repr_local(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "AdversaryAction(requests=[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ", ";
        const auto& r = a.requests[i];
        oss << "AdversaryRequestSpec(prefill_tokens=" << r.prefill_tokens
            << ", decode_tokens=" << r.decode_tokens
            << ", prefill_slo=" << r.prefill_slo
            << ", decode_slo=" << r.decode_slo
            << ")";
    }
    oss << "], stop_decode_ids=" << json_int_vec(a.stop_decode_ids) << ")";
    return oss.str();
}

int prefill_total(const ControllerAction& a) {
    int total = 0;
    for (const auto& kv : a.prefill_allocations) total += kv.second;
    return total;
}

int decode_total(const ControllerAction& a) {
    int total = 0;
    for (const auto& kv : a.decode_allocations) total += kv.second;
    return total;
}

const char* bool_text(bool v) {
    return v ? "true" : "false";
}

std::string float_or_blank(double v) {
    if (!std::isfinite(v)) return "";
    std::ostringstream oss;
    oss << std::setprecision(10) << v;
    return oss.str();
}

void truncate_if_path(const std::string& path) {
    if (path.empty()) return;
    ensure_parent_dir(path);
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    out.close();
}

std::vector<int> waiting_prefill_ids(const SimState& state) {
    std::vector<int> ids;
    for (const auto& req : state.requests) {
        if (req.prefill_active()) ids.push_back(req.request_id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

int requests_in_system(const SimState& state) {
    int n = 0;
    for (const auto& req : state.requests) {
        if (!req.completed) ++n;
    }
    return n;
}

std::string adversary_requests_json_local(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& r = a.requests[i];
        oss << "{"
            << "\"prefill_tokens\":" << r.prefill_tokens << ","
            << "\"decode_tokens\":" << r.decode_tokens << ","
            << "\"prefill_slo\":" << r.prefill_slo << ","
            << "\"decode_slo\":" << r.decode_slo
            << "}";
    }
    oss << "]";
    return oss.str();
}

std::string adversary_prefill_slos_json_local(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17) << "[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ",";
        oss << a.requests[i].prefill_slo;
    }
    oss << "]";
    return oss.str();
}

std::string adversary_decode_slos_json_local(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17) << "[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ",";
        oss << a.requests[i].decode_slo;
    }
    oss << "]";
    return oss.str();
}

std::string adversary_prefill_deadlines_json_local(
    const AdversaryAction& action,
    const SimState& post_state) {
    if (action.requests.empty() || post_state.requests.empty()) return "{}";

    std::vector<int> ids;
    ids.reserve(post_state.requests.size());
    std::unordered_map<int, const RequestState*> by_id;
    by_id.reserve(post_state.requests.size());
    for (const auto& req : post_state.requests) {
        ids.push_back(req.request_id);
        by_id[req.request_id] = &req;
    }
    std::sort(ids.begin(), ids.end());

    const int k = std::min(static_cast<int>(action.requests.size()), static_cast<int>(ids.size()));
    std::unordered_map<int, double> deadlines;
    deadlines.reserve(static_cast<std::size_t>(std::max(0, k)));
    for (int i = static_cast<int>(ids.size()) - k; i < static_cast<int>(ids.size()); ++i) {
        if (i < 0) continue;
        const int rid = ids[static_cast<std::size_t>(i)];
        const auto it = by_id.find(rid);
        if (it == by_id.end() || it->second == nullptr) continue;
        deadlines[rid] = it->second->prefill_deadline;
    }
    return json_i32_f64_map(deadlines);
}

double objective_cost(const SimState& state) {
    return static_cast<double>(state.stats.slo_violations) + state.stats.slo_lateness_sum;
}

double next_decision_time_for_history_row(
    const SimState& state,
    const std::string& player_to_act) {
    if (player_to_act == "adversary" && state.stats.next_adv_tick >= 0.0) {
        return state.stats.next_adv_tick;
    }
    return state.sim_time;
}

NativeIterLogRow make_history_log_row_base(
    const NativeSelfplayConfigGV3& cfg,
    int root_id,
    int root_node_id,
    const std::string& root_player,
    const SimState& state,
    int node_depth,
    int node_id) {
    NativeIterLogRow row;
    row.game_id = cfg.game_id;
    row.root_id = root_id;
    row.sim_iteration = 0;
    row.root_depth = node_depth;
    row.root_node_id = root_node_id;
    row.root_player = root_player;
    row.phase = "history";
    row.node_depth = node_depth;
    row.node_id = node_id;
    row.objective_cost = objective_cost(state);
    row.sim_time = state.sim_time;
    row.decision_state_time = state.decision_state_time;
    row.start_time = state.sim_time;
    row.end_time = state.sim_time;
    row.stage_total_time = 0.0;
    row.requests_in_system = requests_in_system(state);
    row.requests_generated = state.stats.requests_generated;
    row.requests_completed = state.stats.requests_completed;
    row.slo_violations = state.stats.slo_violations;
    row.total_lateness = state.stats.slo_lateness_sum;
    row.avg_lateness = (state.stats.slo_violations > 0)
        ? (state.stats.slo_lateness_sum / static_cast<double>(state.stats.slo_violations))
        : 0.0;
    row.state_active_ids = json_int_vec(state.stats.active_request_ids);
    row.state_waiting_ids = json_int_vec(waiting_prefill_ids(state));
    row.state_completed_request_ids = json_int_vec(state.stats.completed_request_ids);
    row.state_dropped_request_ids = json_int_vec(state.stats.dropped_request_ids);
    row.state_stopped_decode_request_ids = json_int_vec(state.stats.stopped_decode_request_ids);
    row.state_pending_adv_tick = state.stats.pending_adv_tick;
    row.has_state_last_adv_tick = state.stats.last_adv_tick >= 0.0;
    row.state_last_adv_tick = state.stats.last_adv_tick;
    row.state_decode_credit_balance = state.stats.decode_credit_balance;
    row.state_decode_tokens_counted_by_id = json_i32_i32_map(state.stats.decode_tokens_counted_by_id);
    row.state_violated_request_ids = json_int_vec(state.stats.violated_request_ids);
    row.state_per_request_prefill_lateness_by_id =
        json_i32_f64_map(state.stats.per_request_prefill_lateness_by_id);
    row.state_per_request_decode_lateness_by_id =
        json_i32_f64_map(state.stats.per_request_decode_lateness_by_id);
    return row;
}

void fill_action_payload(NativeIterLogRow* row, const HistoryTraceStep& step) {
    if (row == nullptr) return;
    if (step.action_is_controller) {
        const ControllerAction& a = step.controller_action;
        row->has_controller_token_budget = true;
        row->controller_token_budget = a.token_budget;
        row->controller_selected_ids = json_int_vec(a.selected_request_ids);
        row->controller_allocations = json_i32_i32_map(a.token_allocations);
        row->controller_prefill_allocations = json_i32_i32_map(a.prefill_allocations);
        row->controller_decode_allocations = json_i32_i32_map(a.decode_allocations);
        row->controller_prefill_total = prefill_total(a);
        row->controller_decode_total = decode_total(a);
        row->controller_heuristic = a.heuristic;
        row->controller_strategy = a.strategy;
    } else {
        const AdversaryAction& a = step.adversary_action;
        row->adversary_requests = adversary_requests_json_local(a);
        row->adversary_prefill_slos = adversary_prefill_slos_json_local(a);
        row->adversary_prefill_deadlines_by_id =
            adversary_prefill_deadlines_json_local(a, step.state);
        row->adversary_decode_slos = adversary_decode_slos_json_local(a);
    }
}

void write_history_trace(
    NativeIterCsvLogger* logger,
    const NativeSelfplayConfigGV3& cfg,
    const HistoryNode& node,
    int root_id) {
    if (logger == nullptr) return;
    const int root_node_id = node.node_id;

    int sim_iteration = 1;
    for (const HistoryTraceStep& step : node.trace) {
        NativeIterLogRow row = make_history_log_row_base(
            cfg,
            root_id,
            root_node_id,
            cfg.start_player,
            step.state,
            step.depth,
            step.node_id);
        row.sim_iteration = sim_iteration++;
        row.root_depth = node.depth;
        row.has_parent_node_id = true;
        row.parent_node_id = step.parent_node_id;
        row.player_acted_to_create_this_node = step.player_acted;
        row.player_to_act_in_this_node = step.player_to_act;
        row.action_index = step.action_index;
        row.action_repr = step.action_repr;
        row.decision_state_time = next_decision_time_for_history_row(
            step.state,
            step.player_to_act);
        fill_action_payload(&row, step);
        logger->write(row);
    }
}

void write_frontier_header(std::ofstream& out) {
    out << "root_id,root_player,root_depth,history_hops,history_log_node_id,"
           "history_signature_json,strict_signature_json,duplicate_history_signature,"
           "duplicate_strict_signature,valid_action_count,frontier_kind,"
           "sim_time,objective_cost,slo_violations,total_lateness,"
           "decode_credit_balance,state_active_ids,state_completed_request_ids,"
           "slo_lateness_sum,active_request_ids_json,completed_request_ids_json,"
           "dropped_request_ids_json,stopped_decode_request_ids_json,violated_request_ids_json,"
           "per_request_prefill_lateness_json,per_request_decode_lateness_json,"
           "decode_next_deadline_by_id_json,decode_tokens_counted_by_id_json,"
           "recent_arrivals_json,pending_adv_tick,last_adv_tick,requests_json\n";
}

void write_depth1_search_header(std::ofstream& out) {
    out << "root_id,root_player,root_depth,history_hops,best_action_index,best_valid,"
           "best_q_value,best_discount,best_bootstrap_value,best_reward_cost,"
           "action_repr,expected_action_index,expected_q_value,selection_passed\n";
}

void write_depth1_details_header(std::ofstream& out) {
    out << "root_id,root_player,root_depth,history_hops,action_index,valid,"
           "action_q_value,action_discount,action_bootstrap_value,action_reward_cost,"
           "action_repr,rank,selected_by_mcts,expected_best\n";
}

std::string node_signature(const HistoryNode& node);
std::string root_signature(const SimState& state, const std::string& player, int depth);

void write_frontier_row(
    std::ofstream* out,
    std::unordered_set<std::string>* seen_history,
    std::unordered_set<std::string>* seen_strict,
    int root_id,
    const HistoryNode& node,
    const SimState& search_state,
    int valid_count) {
    if (out == nullptr || !out->good()) return;
    const std::string hist_sig = node_signature(node);
    const std::string strict_sig = root_signature(search_state, node.player, node.depth);
    const bool dup_hist = seen_history != nullptr && !seen_history->insert(hist_sig).second;
    const bool dup_strict = seen_strict != nullptr && !seen_strict->insert(strict_sig).second;
    const std::string kind = (valid_count <= 0) ? "terminal" : ((valid_count == 1) ? "forced" : "branching");
    std::unordered_map<int, double> decode_deadlines = search_state.stats.decode_next_deadline_by_id;
    decode_deadlines[-9100001] = search_state.stats.next_adv_tick;
    decode_deadlines[-9100002] = search_state.stats.last_adv_tick;
    decode_deadlines[-9100006] = static_cast<double>(search_state.stats.missed_adv_source);

    std::unordered_map<int, int> decode_counted = search_state.stats.decode_tokens_counted_by_id;
    decode_counted[-9100005] = search_state.stats.decode_credit_balance;

    (*out) << root_id << ","
           << csv_escape(node.player) << ","
           << node.depth << ","
           << node.history_hops << ","
           << node.node_id << ","
           << csv_escape(hist_sig) << ","
           << csv_escape(strict_sig) << ","
           << bool_text(dup_hist) << ","
           << bool_text(dup_strict) << ","
           << valid_count << ","
           << kind << ","
           << float_or_blank(search_state.sim_time) << ","
           << float_or_blank(objective_cost(search_state)) << ","
           << search_state.stats.slo_violations << ","
           << float_or_blank(search_state.stats.slo_lateness_sum) << ","
           << search_state.stats.decode_credit_balance << ","
           << csv_escape(json_int_vec(search_state.stats.active_request_ids)) << ","
           << csv_escape(json_int_vec(search_state.stats.completed_request_ids)) << ","
           << float_or_blank(search_state.stats.slo_lateness_sum) << ","
           << csv_escape(json_int_vec(search_state.stats.active_request_ids)) << ","
           << csv_escape(json_int_vec(search_state.stats.completed_request_ids)) << ","
           << csv_escape(json_int_vec(search_state.stats.dropped_request_ids)) << ","
           << csv_escape(json_int_vec(search_state.stats.stopped_decode_request_ids)) << ","
           << csv_escape(json_int_vec(search_state.stats.violated_request_ids)) << ","
           << csv_escape(json_i32_f64_map(search_state.stats.per_request_prefill_lateness_by_id)) << ","
           << csv_escape(json_i32_f64_map(search_state.stats.per_request_decode_lateness_by_id)) << ","
           << csv_escape(json_i32_f64_map(decode_deadlines)) << ","
           << csv_escape(json_i32_i32_map(decode_counted)) << ","
           << csv_escape(json_recent_launches_for_features(search_state.stats)) << ","
           << bool_text(search_state.stats.pending_adv_tick) << ","
           << float_or_blank(search_state.stats.last_adv_tick) << ","
           << csv_escape(json_request_snapshots_for_features(search_state)) << "\n";
}

int expected_best_index(const SearchOutput& search, const std::string& player) {
    int expected = -1;
    double best = (player == "controller")
        ? -std::numeric_limits<double>::infinity()
        : std::numeric_limits<double>::infinity();
    const int n = static_cast<int>(search.root_nn_valid_mask.size());
    for (int idx = 0; idx < n; ++idx) {
        const bool valid = bool(search.root_nn_valid_mask[static_cast<std::size_t>(idx)]);
        if (!valid) continue;
        const double q = (idx < static_cast<int>(search.root_action_values.size()))
            ? search.root_action_values[static_cast<std::size_t>(idx)]
            : std::numeric_limits<double>::quiet_NaN();
        if (!std::isfinite(q)) continue;
        const bool better = (player == "controller")
            ? (q > best || (q == best && idx < expected))
            : (q < best || (q == best && idx < expected));
        if (expected < 0 || better) {
            expected = idx;
            best = q;
        }
    }
    return expected;
}

double value_at(const std::vector<double>& xs, int idx) {
    if (idx < 0 || idx >= static_cast<int>(xs.size())) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return xs[static_cast<std::size_t>(idx)];
}

std::string repr_at(const std::vector<std::string>& xs, int idx) {
    if (idx < 0 || idx >= static_cast<int>(xs.size())) return "";
    return xs[static_cast<std::size_t>(idx)];
}

std::unordered_map<int, int> action_ranks(const SearchOutput& search, const std::string& player) {
    std::vector<std::pair<int, double>> valid;
    const int n = static_cast<int>(search.root_nn_valid_mask.size());
    valid.reserve(static_cast<std::size_t>(n));
    for (int idx = 0; idx < n; ++idx) {
        if (!search.root_nn_valid_mask[static_cast<std::size_t>(idx)]) continue;
        const double q = value_at(search.root_action_values, idx);
        if (std::isfinite(q)) valid.emplace_back(idx, q);
    }
    std::sort(valid.begin(), valid.end(), [&](const auto& a, const auto& b) {
        if (a.second == b.second) return a.first < b.first;
        return (player == "controller") ? (a.second > b.second) : (a.second < b.second);
    });
    std::unordered_map<int, int> ranks;
    ranks.reserve(valid.size());
    for (std::size_t i = 0; i < valid.size(); ++i) {
        ranks[valid[i].first] = static_cast<int>(i) + 1;
    }
    return ranks;
}

void write_depth1_rows(
    std::ofstream* search_out,
    std::ofstream* details_out,
    int root_id,
    const HistoryNode& node,
    const SearchOutput& search) {
    const int expected_idx = expected_best_index(search, node.player);
    const int selected_idx = search.best_action_index;
    const bool selected_valid =
        selected_idx >= 0 &&
        selected_idx < static_cast<int>(search.root_nn_valid_mask.size()) &&
        bool(search.root_nn_valid_mask[static_cast<std::size_t>(selected_idx)]);
    const bool passed = expected_idx >= 0 && selected_idx == expected_idx;

    if (details_out != nullptr && details_out->good()) {
        const auto ranks = action_ranks(search, node.player);
        const int n = static_cast<int>(search.root_nn_valid_mask.size());
        for (int idx = 0; idx < n; ++idx) {
            const bool valid = bool(search.root_nn_valid_mask[static_cast<std::size_t>(idx)]);
            const auto rank_it = ranks.find(idx);
            (*details_out) << root_id << ","
                           << csv_escape(node.player) << ","
                           << node.depth << ","
                           << node.history_hops << ","
                           << idx << ","
                           << bool_text(valid) << ","
                           << (valid ? float_or_blank(value_at(search.root_action_values, idx)) : "") << ","
                           << (valid ? float_or_blank(value_at(search.root_action_discounts, idx)) : "") << ","
                           << (valid ? float_or_blank(value_at(search.root_action_bootstraps, idx)) : "") << ","
                           << (valid ? float_or_blank(value_at(search.root_action_rewards, idx)) : "") << ","
                           << csv_escape(valid ? repr_at(search.root_action_reprs, idx) : "") << ","
                           << (rank_it == ranks.end() ? std::string() : std::to_string(rank_it->second)) << ","
                           << bool_text(idx == selected_idx) << ","
                           << bool_text(idx == expected_idx)
                           << "\n";
        }
    }

    if (search_out != nullptr && search_out->good()) {
        (*search_out) << root_id << ","
                      << csv_escape(node.player) << ","
                      << node.depth << ","
                      << node.history_hops << ","
                      << (selected_idx < 0 ? std::string() : std::to_string(selected_idx)) << ","
                      << bool_text(selected_valid) << ","
                      << (selected_valid ? float_or_blank(value_at(search.root_action_values, selected_idx)) : "") << ","
                      << (selected_valid ? float_or_blank(value_at(search.root_action_discounts, selected_idx)) : "") << ","
                      << (selected_valid ? float_or_blank(value_at(search.root_action_bootstraps, selected_idx)) : "") << ","
                      << (selected_valid ? float_or_blank(value_at(search.root_action_rewards, selected_idx)) : "") << ","
                      << csv_escape(selected_valid ? repr_at(search.root_action_reprs, selected_idx) : "") << ","
                      << (expected_idx < 0 ? std::string() : std::to_string(expected_idx)) << ","
                      << (expected_idx < 0 ? std::string() : float_or_blank(value_at(search.root_action_values, expected_idx))) << ","
                      << bool_text(passed)
                      << "\n";
    }
}

std::string node_signature(const HistoryNode& node) {
    std::ostringstream oss;
    oss.precision(17);
    oss << node.player << "|"
        << node.history_hops << "|"
        << round9(node.state.sim_time) << "|"
        << (node.state.stats.pending_adv_tick ? 1 : 0) << "|"
        << node.state.stats.decode_credit_balance << "|"
        << ids_key(node.state.stats.active_request_ids) << "|"
        << ids_key(node.state.stats.completed_request_ids);
    return oss.str();
}

std::string root_signature(const SimState& state, const std::string& player, int depth) {
    std::ostringstream oss;
    oss.precision(17);
    oss << player << "|"
        << depth << "|"
        << round9(state.sim_time) << "|"
        << (state.stats.pending_adv_tick ? 1 : 0) << "|"
        << state.stats.decode_credit_balance << "|"
        << ids_key(state.stats.active_request_ids) << "|"
        << ids_key(state.stats.completed_request_ids);
    return oss.str();
}

AnyActions sample_actions(GV2VirtualEnvironment& env, const SimState& state, const std::string& player) {
    AnyActions out;
    if (player == "controller") {
        out.controller = env.sample_controller_actions(state);
        const int n = static_cast<int>(out.controller.actions.size());
        out.valid.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (i >= static_cast<int>(out.controller.mask.size())) continue;
            if (!out.controller.mask[static_cast<std::size_t>(i)]) continue;
            if (!out.controller.actions[static_cast<std::size_t>(i)].valid) continue;
            out.valid.push_back(i);
        }
    } else {
        out.adversary = env.sample_adversary_actions(state);
        const int n = static_cast<int>(out.adversary.actions.size());
        out.valid.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (i >= static_cast<int>(out.adversary.mask.size())) continue;
            if (!out.adversary.mask[static_cast<std::size_t>(i)]) continue;
            if (!out.adversary.actions[static_cast<std::size_t>(i)].valid) continue;
            out.valid.push_back(i);
        }
    }
    return out;
}

std::string next_player(const std::string& player) {
    return (player == "adversary") ? "controller" : "adversary";
}

void rebuild_ids_local(SimState* state) {
    if (state == nullptr) return;
    state->stats.active_request_ids.clear();
    state->stats.completed_request_ids.clear();
    state->stats.dropped_request_ids.clear();
    state->stats.stopped_decode_request_ids.clear();
    state->stats.violated_request_ids.clear();
    for (const auto& req : state->requests) {
        if (req.completed) {
            state->stats.completed_request_ids.push_back(req.request_id);
            if (req.dropped) state->stats.dropped_request_ids.push_back(req.request_id);
            if (req.stopped_decode) state->stats.stopped_decode_request_ids.push_back(req.request_id);
            if (req.violated) state->stats.violated_request_ids.push_back(req.request_id);
        } else if (req.prefill_active() || req.decode_active()) {
            state->stats.active_request_ids.push_back(req.request_id);
            if (req.violated) state->stats.violated_request_ids.push_back(req.request_id);
        }
    }
    auto sort_unique = [](std::vector<int>* xs) {
        std::sort(xs->begin(), xs->end());
        xs->erase(std::unique(xs->begin(), xs->end()), xs->end());
    };
    sort_unique(&state->stats.active_request_ids);
    sort_unique(&state->stats.completed_request_ids);
    sort_unique(&state->stats.dropped_request_ids);
    sort_unique(&state->stats.stopped_decode_request_ids);
    sort_unique(&state->stats.violated_request_ids);
}

void apply_action(
    GV2VirtualEnvironment& env,
    HistoryNode* node,
    const AnyActions& sampled,
    int action_index,
    int child_node_id,
    bool record_trace) {
    if (node == nullptr) return;
    const std::string acted = node->player;
    const int parent_node_id = node->node_id;
    bool has_pre_controller = false;
    SimState pre_controller;
    HistoryTraceStep trace_step;
    if (record_trace) {
        trace_step.parent_node_id = parent_node_id;
        trace_step.node_id = child_node_id;
        trace_step.depth = node->depth + 1;
        trace_step.history_hops = node->history_hops;
        trace_step.action_index = action_index;
        trace_step.player_acted = acted;
        trace_step.player_to_act = next_player(acted);
        trace_step.action_is_controller = acted == "controller";
        if (trace_step.action_is_controller) {
            trace_step.controller_action =
                sampled.controller.actions[static_cast<std::size_t>(action_index)];
            trace_step.action_repr = controller_action_to_repr_local(trace_step.controller_action);
        } else {
            trace_step.adversary_action =
                sampled.adversary.actions[static_cast<std::size_t>(action_index)];
            trace_step.action_repr = adversary_action_to_repr_local(trace_step.adversary_action);
        }
    }

    if (acted == "controller") {
        pre_controller = node->state;
        has_pre_controller = true;
        env.apply_controller_action_inplace(
            node->state,
            sampled.controller.actions[static_cast<std::size_t>(action_index)]);
    } else {
        env.apply_adversary_action_inplace(
            node->state,
            sampled.adversary.actions[static_cast<std::size_t>(action_index)]);
    }

    if (acted == "controller" && node->state.stats.missed_adv_source == 1 && has_pre_controller) {
        node->pre_controller_state = std::move(pre_controller);
        node->has_pre_controller_state = true;
    } else {
        node->has_pre_controller_state = false;
    }
    node->player = next_player(acted);
    node->depth += 1;
    node->parent_node_id = parent_node_id;
    node->node_id = child_node_id;

    if (record_trace) {
        trace_step.state = node->state;
        node->trace.push_back(std::move(trace_step));
    }
}

int advance_to_branching(
    GV2VirtualEnvironment& env,
    HistoryNode* node,
    int max_forced_steps,
    int* next_node_id,
    bool record_trace) {
    if (node == nullptr) return 0;
    int forced = 0;
    const int limit = std::max(0, max_forced_steps);
    while (forced < limit) {
        AnyActions sampled = sample_actions(env, node->state, node->player);
        if (sampled.valid.size() != 1u) break;
        const int child_node_id = (next_node_id != nullptr) ? (*next_node_id)++ : (node->node_id + 1);
        apply_action(env, node, sampled, sampled.valid[0], child_node_id, record_trace);
        ++forced;
    }
    return forced;
}

HistoryNode roll_to_target_hops(
    GV2VirtualEnvironment& env,
    const SimState& initial_state,
    const std::string& start_player,
    int start_depth,
    int target_hops,
    std::mt19937& rng,
    int max_total_steps,
    int* next_node_id,
    bool record_trace) {
    HistoryNode node;
    node.state = initial_state;
    node.player = start_player;
    node.depth = start_depth;
    node.node_id = 0;

    const int step_cap = std::max(1, max_total_steps);
    const int target = std::max(0, target_hops);
    int steps = 0;
    node.history_hops = 0;

    while (node.history_hops < target && steps < step_cap) {
        steps += advance_to_branching(
            env,
            &node,
            std::min(2000, step_cap - steps),
            next_node_id,
            record_trace);
        if (steps >= step_cap) break;

        AnyActions sampled = sample_actions(env, node.state, node.player);
        if (sampled.valid.empty()) break;

        int idx = sampled.valid[0];
        bool nontrivial = false;
        if (sampled.valid.size() > 1u) {
            std::uniform_int_distribution<int> pick(0, static_cast<int>(sampled.valid.size()) - 1);
            idx = sampled.valid[static_cast<std::size_t>(pick(rng))];
            nontrivial = true;
        }

        const int child_node_id = (next_node_id != nullptr) ? (*next_node_id)++ : (node.node_id + 1);
        apply_action(env, &node, sampled, idx, child_node_id, record_trace);
        ++steps;
        if (nontrivial) {
            node.history_hops += 1;
            if (record_trace && !node.trace.empty()) {
                node.trace.back().history_hops = node.history_hops;
            }
        }
    }

    if (steps < step_cap) {
        (void)advance_to_branching(
            env,
            &node,
            std::min(2000, step_cap - steps),
            next_node_id,
            record_trace);
    }
    return node;
}

void prepare_untried_actions(
    GV2VirtualEnvironment& env,
    HistoryNode* node,
    std::mt19937& rng,
    int max_history_hops,
    int remaining_need) {
    if (node == nullptr) return;
    node->untried_action_indices.clear();
    if (node->history_hops >= max_history_hops) return;

    AnyActions sampled = sample_actions(env, node->state, node->player);
    if (sampled.valid.size() <= 1u) return;

    std::vector<int> choices = sampled.valid;
    std::shuffle(choices.begin(), choices.end(), rng);
    int cap = static_cast<int>(choices.size());
    if (remaining_need > 0) cap = std::min(cap, std::max(1, remaining_need));
    choices.resize(static_cast<std::size_t>(cap));
    node->untried_action_indices = std::move(choices);
}

HistoryNode* make_anchor(HistorySession* session, GV2VirtualEnvironment& env) {
    if (session == nullptr) return nullptr;
    if (session->anchors_built >= session->anchor_budget ||
        session->iterations >= session->max_iterations) {
        session->exhausted = true;
        return nullptr;
    }

    HistoryNode anchor = roll_to_target_hops(
        env,
        session->start_state,
        session->start_player,
        session->start_depth,
        session->min_history_hops,
        session->rng,
        session->max_total_steps,
        &session->next_node_id,
        session->record_trace);
    session->anchors_built += 1;
    session->iterations += 1;
    prepare_untried_actions(
        env,
        &anchor,
        session->rng,
        session->max_history_hops,
        session->target_roots - session->emitted_roots);
    if (!anchor.untried_action_indices.empty()) {
        session->active_path.push_back(std::move(anchor));
        return &session->active_path.back();
    }
    session->active_path.push_back(std::move(anchor));
    return &session->active_path.back();
}

HistoryNode* next_unique_node(HistorySession* session, GV2VirtualEnvironment& env) {
    if (session == nullptr) return nullptr;

    while (!session->exhausted) {
        if (session->iterations >= session->max_iterations) {
            session->exhausted = true;
            return nullptr;
        }

        if (session->active_path.empty()) {
            HistoryNode* anchor = make_anchor(session, env);
            if (anchor == nullptr) return nullptr;
            const std::string sig = node_signature(*anchor);
            if (session->seen_signatures.insert(sig).second) return anchor;
            continue;
        }

        HistoryNode& node = session->active_path.back();
        if (node.untried_action_indices.empty()) {
            session->active_path.pop_back();
            continue;
        }

        const int idx = node.untried_action_indices.back();
        node.untried_action_indices.pop_back();
        AnyActions sampled = sample_actions(env, node.state, node.player);
        if (std::find(sampled.valid.begin(), sampled.valid.end(), idx) == sampled.valid.end()) {
            continue;
        }

        HistoryNode child = node;
        child.untried_action_indices.clear();
        apply_action(
            env,
            &child,
            sampled,
            idx,
            session->next_node_id++,
            session->record_trace);
        child.history_hops = node.history_hops + 1;
        if (session->record_trace && !child.trace.empty()) {
            child.trace.back().history_hops = child.history_hops;
        }
        (void)advance_to_branching(
            env,
            &child,
            std::min(2000, session->max_total_steps),
            &session->next_node_id,
            session->record_trace);
        prepare_untried_actions(
            env,
            &child,
            session->rng,
            session->max_history_hops,
            session->target_roots - session->emitted_roots);
        session->iterations += 1;

        session->active_path.push_back(std::move(child));
        HistoryNode& stored_child = session->active_path.back();
        const std::string sig = node_signature(stored_child);
        if (session->seen_signatures.insert(sig).second) return &stored_child;
    }

    return nullptr;
}

HistoryNode next_root_node(HistorySession* session, GV2VirtualEnvironment& env) {
    HistoryNode* node = next_unique_node(session, env);
    if (node != nullptr) return *node;

    if (session == nullptr || !session->allow_duplicate_fallback) {
        return HistoryNode{};
    }
    std::uniform_int_distribution<int> hops_pick(session->min_history_hops, session->max_history_hops);
    return roll_to_target_hops(
        env,
        session->start_state,
        session->start_player,
        session->start_depth,
        hops_pick(session->rng),
        session->rng,
        session->max_total_steps,
        &session->next_node_id,
        session->record_trace);
}

SimState decision_state_for_root(const HistoryNode& node) {
    if (node.player != "adversary" || !node.has_pre_controller_state || node.state.stats.missed_adv_source != 1) {
        SimState out = node.state;
        out.decision_state_time = next_decision_time_for_history_row(out, node.player);
        return out;
    }

    SimState out = node.pre_controller_state;
    const double tick = node.state.stats.next_adv_tick;
    if (tick >= 0.0 && out.sim_time + 1e-9 < tick) {
        out.sim_time = tick;
    }
    out.decision_state_time = (tick >= 0.0) ? tick : out.sim_time;

    std::unordered_set<int> live;
    live.reserve(node.state.requests.size());
    for (const auto& req : node.state.requests) {
        if (!req.completed) live.insert(req.request_id);
    }
    for (auto& req : out.requests) {
        if (live.find(req.request_id) == live.end()) {
            req.completed = true;
            req.num_prefill_tokens = req.num_processed_prefill_tokens;
            req.num_decode_tokens = req.num_processed_decode_tokens;
        }
    }
    rebuild_ids_local(&out);
    return out;
}

NativeRootSampleGV3 make_sample_from_search(
    const NativeSelfplayConfigGV3& cfg,
    const SearchOutput& search,
    const HistoryNode& node,
    int root_id,
    int model_version,
    bool is_eval) {
    NativeRootSampleGV3 sample;
    sample.feature_version = cfg.feature_version;
    sample.game_id = cfg.game_id;
    sample.root_id = root_id;
    sample.root_node_id = root_id;
    sample.root_depth = node.depth;
    sample.player = node.player;

    sample.global_features = search.root_global_features;
    sample.action_mask = search.root_action_mask;
    sample.prefill_req_features = search.root_prefill_req_features;
    sample.decode_req_features = search.root_decode_req_features;
    sample.prefill_req_mask = search.root_prefill_req_mask;
    sample.decode_req_mask = search.root_decode_req_mask;
    sample.prefill_req_n = search.root_prefill_req_n;
    sample.prefill_req_d = search.root_prefill_req_d;
    sample.decode_req_n = search.root_decode_req_n;
    sample.decode_req_d = search.root_decode_req_d;
    sample.req_features = search.root_req_features;
    sample.req_mask = search.root_req_mask;
    sample.req_n = search.root_req_n;
    sample.req_d = search.root_req_d;

    sample.policy = search.mcts_root_prior;
    sample.value = (search.root_visits > 0)
        ? (search.root_value_sum / static_cast<double>(search.root_visits))
        : 0.0;
    sample.best_action_index = search.best_action_index;
    sample.used_bootstrap = bool(cfg.use_model_bootstrap && model_version > 0);
    sample.model_version = model_version;
    sample.history_hops = node.history_hops;
    sample.is_eval = is_eval;
    return sample;
}

}  // namespace

NativeSelfplayResultGV3 generate_native_selfplay_samples_gv3(
    const NativeSelfplayConfigGV3& cfg,
    GV2VirtualEnvironment& env,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version) {
    NativeSelfplayResultGV3 result;
    if (cfg.num_roots <= 0) return result;

    HistorySession session;
    session.start_state = cfg.has_initial_state ? cfg.initial_state : SimState{};
    session.start_player = cfg.start_player;
    session.start_depth = cfg.start_root_depth;
    session.min_history_hops = std::min(cfg.history_hops_min, cfg.history_hops_max);
    session.max_history_hops = std::max(cfg.history_hops_min, cfg.history_hops_max);
    session.max_total_steps = std::max(1, cfg.history_max_total_steps);
    session.target_roots = std::max(0, cfg.num_roots);
    session.next_root_id = cfg.start_root_id;
    session.rng.seed(static_cast<uint32_t>(std::max(0, cfg.history_seed)));
    session.anchor_budget = std::max(4, session.target_roots * 3);
    session.max_iterations = std::max(200, session.target_roots * 50);
    session.allow_duplicate_fallback = cfg.allow_duplicate_history_fallback;
    session.record_trace = !cfg.history_trace_log_path.empty();
    for (const auto& sig : cfg.initial_seen_signatures) {
        if (!sig.empty()) session.seen_signatures.insert(sig);
    }

    std::unique_ptr<NativeIterCsvLogger> history_logger;
    if (!cfg.history_trace_log_path.empty()) {
        truncate_if_path(cfg.history_trace_log_path);
        history_logger = std::make_unique<NativeIterCsvLogger>(cfg.history_trace_log_path, 1);
    }

    std::unique_ptr<std::ofstream> frontier_log;
    if (!cfg.frontier_log_path.empty()) {
        ensure_parent_dir(cfg.frontier_log_path);
        frontier_log = std::make_unique<std::ofstream>(
            cfg.frontier_log_path,
            std::ios::out | std::ios::trunc);
        if (frontier_log->good()) write_frontier_header(*frontier_log);
    }

    std::unique_ptr<std::ofstream> depth1_search_log;
    if (!cfg.depth1_search_log_path.empty()) {
        ensure_parent_dir(cfg.depth1_search_log_path);
        depth1_search_log = std::make_unique<std::ofstream>(
            cfg.depth1_search_log_path,
            std::ios::out | std::ios::trunc);
        if (depth1_search_log->good()) write_depth1_search_header(*depth1_search_log);
    }

    std::unique_ptr<std::ofstream> depth1_details_log;
    if (!cfg.depth1_details_log_path.empty()) {
        ensure_parent_dir(cfg.depth1_details_log_path);
        depth1_details_log = std::make_unique<std::ofstream>(
            cfg.depth1_details_log_path,
            std::ios::out | std::ios::trunc);
        if (depth1_details_log->good()) write_depth1_details_header(*depth1_details_log);
    }

    std::mt19937 eval_rng(static_cast<uint32_t>(std::max(0, cfg.eval_split_seed)));
    std::uniform_real_distribution<double> eval_pick(0.0, 1.0);
    const double eval_ratio = std::max(0.0, std::min(1.0, cfg.eval_split_ratio));

    std::unordered_set<std::string> root_sigs;
    std::unordered_set<std::string> logged_history_sigs;
    std::unordered_set<std::string> logged_strict_sigs;
    int controller_train = 0;
    int controller_eval = 0;
    int adversary_train = 0;
    int adversary_eval = 0;

    result.samples.reserve(static_cast<std::size_t>(session.target_roots));
    while (session.emitted_roots < session.target_roots) {
        HistoryNode node = next_root_node(&session, env);
        const int root_id = session.next_root_id++;
        session.emitted_roots += 1;

        (void)advance_to_branching(
            env,
            &node,
            std::max(0, cfg.max_forced_hops_per_root),
            &session.next_node_id,
            session.record_trace);
        const SimState search_state = decision_state_for_root(node);
        root_sigs.insert(root_signature(search_state, node.player, node.depth));
        result.history_signatures.push_back(node_signature(node));

        SearchInput in = cfg.search_template;
        in.root_state = search_state;
        in.root_player = node.player;
        in.root_depth = node.depth;
        in.root_id = root_id;
        in.root_node_id = root_id;
        in.game_id = cfg.game_id;
        in.iterations = (node.player == "adversary")
            ? cfg.adv_iterations_per_root
            : cfg.cont_iterations_per_root;
        in.seed = cfg.action_seed_base + root_id;
        in.use_model_bootstrap = bool(cfg.use_model_bootstrap && model_version > 0);

        SearchOutput search = run_search_torchscript_with_env(in, env, infer_runtime, model_version);
        write_history_trace(history_logger.get(), cfg, node, root_id);
        write_frontier_row(
            frontier_log.get(),
            &logged_history_sigs,
            &logged_strict_sigs,
            root_id,
            node,
            search_state,
            search.root_num_valid_actions);
        write_depth1_rows(
            depth1_search_log.get(),
            depth1_details_log.get(),
            root_id,
            node,
            search);
        const bool is_eval = eval_pick(eval_rng) < eval_ratio;
        NativeRootSampleGV3 sample = make_sample_from_search(
            cfg,
            search,
            node,
            root_id,
            model_version,
            is_eval);
        sample.root_node_id = in.root_node_id;

        if (sample.player == "controller") {
            if (is_eval) ++controller_eval;
            else ++controller_train;
        } else {
            if (is_eval) ++adversary_eval;
            else ++adversary_train;
        }
        result.samples.push_back(std::move(sample));
    }

    result.stats["num_roots_requested"] = cfg.num_roots;
    result.stats["num_roots_generated"] = static_cast<int>(result.samples.size());
    result.stats["num_unique_roots"] = static_cast<int>(root_sigs.size());
    result.stats["controller_train_samples"] = controller_train;
    result.stats["controller_eval_samples"] = controller_eval;
    result.stats["adversary_train_samples"] = adversary_train;
    result.stats["adversary_eval_samples"] = adversary_eval;
    result.stats["train_samples_total"] = controller_train + adversary_train;
    result.stats["eval_samples_total"] = controller_eval + adversary_eval;
    result.stats["num_history_roots_emitted"] = static_cast<int>(result.history_signatures.size());
    return result;
}

}  // namespace mcts_native_gv2
