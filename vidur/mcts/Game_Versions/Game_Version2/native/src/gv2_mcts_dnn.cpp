#include "gv2_mcts_dnn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace mcts_native_gv2 {
namespace {

template <typename T>
T clampv(T x, T lo, T hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
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

std::string json_u8_vec(const std::vector<uint8_t>& xs) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << static_cast<int>(xs[i]);
    }
    oss << "]";
    return oss.str();
}

std::string json_f64_vec(const std::vector<double>& xs) {
    std::ostringstream oss;
    oss << std::setprecision(17);
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

std::string controller_action_to_json(const ControllerAction& a) {
    std::vector<int> ids;
    ids.reserve(a.token_allocations.size());
    for (const auto& kv : a.token_allocations) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());

    auto map_json = [](const std::unordered_map<int, int>& mp) {
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
    };

    std::ostringstream oss;
    oss << "{";
    oss << "\"type\":\"controller\",";
    oss << "\"token_budget\":" << a.token_budget << ",";
    oss << "\"selected_request_ids\":" << json_int_vec(a.selected_request_ids) << ",";
    oss << "\"token_allocations\":" << map_json(a.token_allocations) << ",";
    oss << "\"prefill_allocations\":" << map_json(a.prefill_allocations) << ",";
    oss << "\"decode_allocations\":" << map_json(a.decode_allocations) << ",";
    oss << "\"heuristic\":\"" << a.heuristic << "\",";
    oss << "\"strategy\":\"" << a.strategy << "\",";
    oss << "\"mapping\":[" << a.mapping[0] << "," << a.mapping[1] << "," << a.mapping[2] << "]";
    oss << "}";
    return oss.str();
}

std::string controller_action_to_repr(const ControllerAction& a) {
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

std::string adversary_action_to_json(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "{";
    oss << "\"type\":\"adversary\",";
    oss << "\"requests\":[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& r = a.requests[i];
        oss << "{";
        oss << "\"prefill_tokens\":" << r.prefill_tokens << ",";
        oss << "\"decode_tokens\":" << r.decode_tokens << ",";
        oss << "\"prefill_slo\":" << r.prefill_slo << ",";
        oss << "\"decode_slo\":" << r.decode_slo;
        oss << "}";
    }
    oss << "],";
    oss << "\"stop_decode_ids\":" << json_int_vec(a.stop_decode_ids);
    oss << "}";
    return oss.str();
}

std::string adversary_action_to_repr(const AdversaryAction& a) {
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

std::string adversary_requests_json(const AdversaryAction& a) {
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

std::string adversary_prefill_slos_json(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ",";
        oss << a.requests[i].prefill_slo;
    }
    oss << "]";
    return oss.str();
}

std::string adversary_decode_slos_json(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ",";
        oss << a.requests[i].decode_slo;
    }
    oss << "]";
    return oss.str();
}

double state_cost(const SimState& s) {
    return static_cast<double>(s.stats.slo_violations) + static_cast<double>(s.stats.slo_lateness_sum);
}

double transition_reward(double parent_cost, double child_cost, const SearchInput& in) {
    const double delta = std::max(0.0, child_cost - parent_cost);
    const double knee = in.reward_knee;
    const double max_penalty = in.reward_max_penalty;
    if (max_penalty <= knee) {
        return -std::min(delta, max_penalty);
    }

    const double headroom = max_penalty - knee;
    const double alpha = in.reward_tail_alpha;
    double penalty = 0.0;
    if (delta <= knee) {
        penalty = delta;
    } else {
        penalty = knee + headroom * std::tanh(alpha * (delta - knee));
    }
    return -penalty;
}

double time_discount(double t_child, double t_parent, const SearchInput& in) {
    const double gamma = clampv(in.discount_factor, 1e-9, 1.0);
    const double denom = std::max(1e-9, in.prefill_step_time);
    const double dt = std::max(0.0, t_child - t_parent);
    return std::pow(gamma, dt / denom);
}

std::string controller_action_key(const ControllerAction& a) {
    std::vector<std::pair<int, int>> alloc;
    alloc.reserve(a.token_allocations.size());
    for (const auto& kv : a.token_allocations) {
        alloc.emplace_back(kv.first, kv.second);
    }
    std::sort(alloc.begin(), alloc.end());

    std::ostringstream oss;
    oss << "alloc:";
    for (std::size_t i = 0; i < alloc.size(); ++i) {
        if (i > 0) oss << "|";
        oss << alloc[i].first << ":" << alloc[i].second;
    }
    oss << "|strategy:" << a.strategy;
    oss << "|mapping:" << a.mapping[0] << "," << a.mapping[1] << "," << a.mapping[2];
    return oss.str();
}

struct MinMaxStats {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    void update(double v) {
        minimum = std::min(minimum, v);
        maximum = std::max(maximum, v);
    }
};

struct TreeNode {
    std::string player;
    int node_id = 0;
    int depth = 0;
    TreeNode* parent = nullptr;

    bool has_parent_action = false;
    bool parent_action_is_controller = false;
    ControllerAction parent_controller_action;
    AdversaryAction parent_adversary_action;
    int parent_action_index = -1;

    double prior = 0.0;
    double reward = 0.0;
    int visits = 0;
    double value_sum = 0.0;
    double state_cost = 0.0;
    double sim_time = 0.0;
    int num_valid_actions = 0;

    bool has_nn_value = false;
    double nn_value_controller = 0.0;
    std::vector<double> nn_priors;
    std::vector<double> nn_priors_after_threshold;
    std::vector<uint8_t> nn_valid_mask;
    std::unordered_map<int, int> action_alias_to_canonical;
    std::unordered_map<int, std::vector<int>> canonical_to_action_aliases;

    double last_decision_state_time = 0.0;

    bool has_snapshot = false;
    SimState cached_state;

    std::unordered_map<int, std::unique_ptr<TreeNode>> children;

    bool expanded() const { return !children.empty(); }
    double mean_value() const { return (visits > 0) ? (value_sum / static_cast<double>(visits)) : 0.0; }
};

struct SelectionResult {
    int action_index = -1;
    TreeNode* child = nullptr;
};

std::pair<bool, double> is_missed_adv_tick(const SimState& state, const GV2EnvConfig& cfg) {
    const double tick = (cfg.adversary_tick_sec > 0.0) ? cfg.adversary_tick_sec : 0.2;
    double next_tick = state.stats.next_adv_tick;
    if (next_tick < 0.0) {
        const double q = std::floor((state.sim_time + cfg.eps) / tick);
        next_tick = q * tick;
    }
    const bool missed = state.sim_time > (next_tick + cfg.eps);
    return {missed, next_tick};
}

std::unordered_set<int> live_request_ids(const SimState& s) {
    std::unordered_set<int> ids;
    ids.reserve(s.requests.size());
    for (const auto& r : s.requests) {
        if (!r.completed) ids.insert(r.request_id);
    }
    return ids;
}

class SearchRunner {
public:
    SearchRunner(
        const SearchInput& in,
        NativeTorchScriptInferRuntimeGV2& infer_runtime,
        int model_version)
        : in_(in),
          env_(in.env_cfg, in.sim_cfg),
          infer_runtime_(infer_runtime),
          model_version_(model_version),
          rng_(static_cast<uint32_t>(std::max(0, in.seed))),
          next_node_id_(std::max(1, in.root_node_id + 1)) {
        if (!in.predictor_csv_path.empty()) {
            (void)env_.load_predictor_csv(in.predictor_csv_path);
        }
        if (!in.sim_cfg.prefill_profile_tokens.empty() &&
            in.sim_cfg.prefill_profile_tokens.size() == in.sim_cfg.prefill_profile_times.size()) {
            env_.set_prefill_profile(in.sim_cfg.prefill_profile_tokens, in.sim_cfg.prefill_profile_times);
        }
    }

    SearchOutput run() {
        using clock = std::chrono::steady_clock;
        const auto t_total_begin = clock::now();

        SearchOutput out;
        out.contract_version = kGV2NativeContractVersion;
        out.decision_state_time = in_.decision_state_time;
        out.root_state_echo = in_.root_state;

        auto root = std::make_unique<TreeNode>();
        root->player = in_.root_player;
        root->node_id = in_.root_node_id;
        root->depth = in_.root_depth;
        root->parent = nullptr;
        root->sim_time = in_.root_state.sim_time;
        root->state_cost = state_cost(in_.root_state);
        root->has_snapshot = true;
        root->cached_state = in_.root_state;

        const auto t_root_expand_begin = clock::now();
        SimState root_state_work = root->cached_state;
        const auto root_expand = expand_node(root.get(), root_state_work);
        const auto t_root_expand_end = clock::now();

        const int root_valid_actions = std::get<2>(root_expand);
        out.root_num_valid_actions = root_valid_actions;
        out.root_state_cost = root->state_cost;
        out.root_sim_time = root->sim_time;

        const bool root_nn_called = std::get<1>(root_expand);
        apply_root_dirichlet_noise(root.get(), root_nn_called, root_valid_actions);


        // Root NN payload; for forced/terminal roots keep deterministic fallback.
        if (root->has_nn_value) {
            out.root_nn_value_controller = root->nn_value_controller;
            out.root_nn_priors = root->nn_priors;
            out.root_nn_priors_after_threshold = root->nn_priors_after_threshold;
            out.root_nn_valid_mask = root->nn_valid_mask;
        } else {
            out.root_nn_value_controller = 0.0;
            out.root_nn_valid_mask = root->nn_valid_mask.empty() ? in_.action_mask : root->nn_valid_mask;
            if (out.root_nn_valid_mask.empty()) {
                out.root_nn_valid_mask.assign(8, 1u);
            }
            out.root_nn_priors.assign(out.root_nn_valid_mask.size(), 0.0);
            out.root_nn_priors_after_threshold = normalize_masked(out.root_nn_priors, out.root_nn_valid_mask);
        }

        out.action_alias_to_canonical = root->action_alias_to_canonical;
        out.canonical_to_action_aliases = root->canonical_to_action_aliases;

        const int iterations = std::max(0, in_.iterations);
        for (int iter = 0; iter < iterations; ++iter) {
            run_one_simulation(iter, root.get(), &out);
        }

        out.root_visits = root->visits;
        out.root_value_sum = root->value_sum;
        out.root_state_cost = root->state_cost;
        out.root_sim_time = root->sim_time;

        // Root children summaries
        {
            std::vector<int> child_indices;
            child_indices.reserve(root->children.size());
            for (const auto& kv : root->children) child_indices.push_back(kv.first);
            std::sort(child_indices.begin(), child_indices.end());

            out.children.reserve(child_indices.size());
            for (int idx : child_indices) {
                const auto it = root->children.find(idx);
                if (it == root->children.end()) continue;
                const TreeNode* c = it->second.get();

                ChildSummary cs;
                cs.index = idx;
                cs.node_id = c->node_id;
                cs.depth = c->depth;
                cs.player = c->player;
                cs.prior = c->prior;
                cs.reward = c->reward;
                cs.visits = c->visits;
                cs.value_sum = c->value_sum;
                cs.sim_time = c->sim_time;
                cs.state_cost = c->state_cost;
                cs.num_valid_actions = c->num_valid_actions;
                if (c->has_parent_action) {
                    cs.parent_action_json = c->parent_action_is_controller
                        ? controller_action_to_json(c->parent_controller_action)
                        : adversary_action_to_json(c->parent_adversary_action);
                }
                out.children.push_back(std::move(cs));
            }
        }

        out.mcts_root_prior = compute_root_prior(*root, out.root_nn_valid_mask);

        const auto t_total_end = clock::now();
        out.perf["total_sec"] =
            std::chrono::duration_cast<std::chrono::duration<double>>(t_total_end - t_total_begin).count();
        out.perf["root_expand_sec"] =
            std::chrono::duration_cast<std::chrono::duration<double>>(t_root_expand_end - t_root_expand_begin).count();
        out.perf["selection_sec"] = perf_selection_sec_;
        out.perf["restore_sec"] = perf_restore_sec_;
        out.perf["forced_chain_sec"] = perf_forced_sec_;
        out.perf["expand_sec"] = perf_expand_sec_;
        out.perf["backprop_sec"] = perf_backprop_sec_;
        out.perf["infer_total_sec"] = perf_infer_sec_;
        out.perf["infer_calls"] = static_cast<double>(perf_infer_calls_);
        out.perf["selection_steps"] = static_cast<double>(perf_selection_steps_);
        out.perf["forced_steps"] = static_cast<double>(perf_forced_steps_);
        out.perf["minmax_min"] = std::isfinite(min_max_.minimum) ? min_max_.minimum : 0.0;
        out.perf["minmax_max"] = std::isfinite(min_max_.maximum) ? min_max_.maximum : 0.0;
        return out;
    }

private:
    struct ForcedStepLog {
        TreeNode* node = nullptr;
        SimState state_snapshot;
        int num_valid_actions = 0;
        int unique_actions = 0;
    };

    const SearchInput& in_;
    GV2VirtualEnvironment env_;
    NativeTorchScriptInferRuntimeGV2& infer_runtime_;
    int model_version_ = 0;
    std::mt19937 rng_;
    int next_node_id_ = 1;
    MinMaxStats min_max_;

    // Perf counters
    double perf_selection_sec_ = 0.0;
    double perf_restore_sec_ = 0.0;
    double perf_forced_sec_ = 0.0;
    double perf_expand_sec_ = 0.0;
    double perf_backprop_sec_ = 0.0;
    double perf_infer_sec_ = 0.0;
    int perf_infer_calls_ = 0;
    int perf_selection_steps_ = 0;
    int perf_forced_steps_ = 0;

    std::string build_adversary_prefill_deadlines_json(
        const AdversaryAction& action,
        const SimState& state) const {
        if (action.requests.empty() || state.requests.empty()) return "{}";

        std::unordered_map<int, const RequestState*> by_id;
        by_id.reserve(state.requests.size());
        std::vector<int> ids;
        ids.reserve(state.requests.size());
        for (const auto& req : state.requests) {
            by_id[req.request_id] = &req;
            ids.push_back(req.request_id);
        }
        std::sort(ids.begin(), ids.end());

        const int k = std::min(
            static_cast<int>(action.requests.size()),
            static_cast<int>(ids.size()));
        if (k <= 0) return "{}";

        std::unordered_map<int, double> deadlines;
        deadlines.reserve(static_cast<std::size_t>(k));
        for (int i = static_cast<int>(ids.size()) - k; i < static_cast<int>(ids.size()); ++i) {
            const int rid = ids[static_cast<std::size_t>(i)];
            const auto it = by_id.find(rid);
            if (it == by_id.end() || it->second == nullptr) continue;
            const RequestState& req = *it->second;
            deadlines[rid] = req.queued_at + req.prefill_slo_time;
        }
        return json_i32_f64_map(deadlines);
    }

    NativeInferInputsGV2 build_infer_inputs(
        const SimState& state,
        const std::vector<uint8_t>& action_mask) const {
        NativeFeatureBuildConfigGV2 feat_cfg;
        if (in_.root_infer_inputs.prefill_req_n > 0) {
            feat_cfg.n_prefill_req = in_.root_infer_inputs.prefill_req_n;
        }
        if (in_.root_infer_inputs.prefill_req_d > 0) {
            feat_cfg.d_prefill_req = in_.root_infer_inputs.prefill_req_d;
        }
        if (in_.root_infer_inputs.decode_req_n > 0) {
            feat_cfg.n_decode_req = in_.root_infer_inputs.decode_req_n;
        }
        if (in_.root_infer_inputs.decode_req_d > 0) {
            feat_cfg.d_decode_req = in_.root_infer_inputs.decode_req_d;
        }
        if (!in_.root_infer_inputs.global_features.empty()) {
            feat_cfg.d_global = static_cast<int>(in_.root_infer_inputs.global_features.size());
        } else if (!in_.global_features.empty()) {
            feat_cfg.d_global = static_cast<int>(in_.global_features.size());
        }
        feat_cfg.auto_drop_lateness_sec = std::max(1e-9, in_.env_cfg.auto_drop_lateness_sec);
        feat_cfg.launch_ewma_norm_den = std::max(1, in_.env_cfg.max_requests_per_launch_window);
        return infer_runtime_.build_inputs_from_state(state, action_mask, feat_cfg, &in_.root_infer_inputs);
    }

    std::pair<SimState, double> decision_state_for_node(TreeNode* node, const SimState& real_state) const {
        if (node == nullptr) return {real_state, real_state.sim_time};
        if (node->parent == nullptr) return {real_state, real_state.sim_time};
        if (node->player != "adversary") return {real_state, real_state.sim_time};
        if (node->parent->player != "controller") return {real_state, real_state.sim_time};

        const auto missed = is_missed_adv_tick(real_state, env_.cfg());
        if (!missed.first) {
            return {real_state, missed.second};
        }
        const int src = real_state.stats.missed_adv_source;
        if (src == 2) {
            return {real_state, missed.second};
        }
        if (!node->parent->has_snapshot) {
            return {real_state, missed.second};
        }

        SimState ds = node->parent->cached_state;
        if (ds.sim_time + env_.cfg().eps < missed.second) {
            ds.sim_time = missed.second;
        }
        return {std::move(ds), missed.second};
    }

    std::unordered_set<int> replay_forbidden_stop_ids(
        TreeNode* node,
        const SimState& decision_state,
        const SimState& real_state) const {
        std::unordered_set<int> out;
        if (node == nullptr || node->parent == nullptr) return out;
        if (node->player != "adversary") return out;
        if (node->parent->player != "controller") return out;

        const auto missed = is_missed_adv_tick(real_state, env_.cfg());
        if (!missed.first) return out;
        if (real_state.stats.missed_adv_source != 1) return out;

        const auto replay_ids = live_request_ids(decision_state);
        const auto post_ids = live_request_ids(real_state);
        for (int rid : replay_ids) {
            if (post_ids.find(rid) == post_ids.end()) out.insert(rid);
        }
        return out;
    }

    std::optional<double> child_q_controller(const TreeNode& parent, const TreeNode& child) const {
        if (child.visits <= 0) return std::nullopt;
        const bool parent_is_branching =
            (parent.num_valid_actions > 1) || (parent.children.size() > 1);
        const double disc = parent_is_branching
            ? time_discount(child.sim_time, parent.sim_time, in_)
            : 1.0;
        return child.reward + disc * child.mean_value();
    }

    std::pair<std::optional<double>, std::optional<double>> parent_child_q_bounds(
        const TreeNode& parent) const {
        std::optional<double> q_min;
        std::optional<double> q_max;
        for (const auto& kv : parent.children) {
            const auto q = child_q_controller(parent, *kv.second);
            if (!q.has_value()) continue;
            if (!q_min.has_value()) {
                q_min = q;
                q_max = q;
            } else {
                q_min = std::min(*q_min, *q);
                q_max = std::max(*q_max, *q);
            }
        }
        return {q_min, q_max};
    }

    double normalize_local_q(
        double q_controller,
        const std::optional<double>& q_min,
        const std::optional<double>& q_max) const {
        if (!q_min.has_value() || !q_max.has_value()) return 0.0;
        const double den = *q_max - *q_min;
        if (den <= 1e-8) return 0.0;
        return clampv((q_controller - *q_min) / den, 0.0, 1.0);
    }

    double puct_score(
        const TreeNode& parent,
        const TreeNode& child,
        const std::optional<double>& q_min,
        const std::optional<double>& q_max) const {
        const double pb_c_base = std::max(1.0, in_.pb_c_base);
        const double pb_c_init = in_.pb_c_init;
        double pb_c = std::log((static_cast<double>(parent.visits) + pb_c_base + 1.0) / pb_c_base) + pb_c_init;
        pb_c *= std::sqrt(static_cast<double>(parent.visits) + 1.0) /
                (static_cast<double>(child.visits) + 1.0);

        const double prior_score = pb_c * child.prior;
        const auto q = child_q_controller(parent, child);
        double value_score = 0.0;
        if (q.has_value()) {
            const double qn = normalize_local_q(*q, q_min, q_max);
            value_score = (parent.player == "controller") ? qn : -qn;
        }
        return prior_score + value_score;
    }

    void apply_root_dirichlet_noise(TreeNode* root, bool nn_called, int num_valid_actions) {
        if (root == nullptr) return;
        if (!in_.root_dirichlet_noise_enabled) return;
        if (!nn_called) return;
        if (num_valid_actions <= 1) return;
        if (root->children.empty()) return;

        const double alpha = in_.root_dirichlet_alpha;
        double eps = in_.root_dirichlet_epsilon;

        if (alpha <= 0.0 || eps <= 0.0) return;
        eps = clampv(eps, 0.0, 1.0);

        std::vector<int> child_indices;
        child_indices.reserve(root->children.size());
        for (const auto& kv : root->children) child_indices.push_back(kv.first);
        std::sort(child_indices.begin(), child_indices.end());

        const int n = static_cast<int>(child_indices.size());
        if (n <= 1) return;

        std::gamma_distribution<double> gamma(alpha, 1.0);
        std::vector<double> noise_raw(static_cast<std::size_t>(n), 0.0);
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            const double g = gamma(rng_);
            noise_raw[static_cast<std::size_t>(i)] = g;
            s += g;
        }

        std::vector<double> noise(static_cast<std::size_t>(n), 0.0);
        if (s <= 1e-12) {
            const double u = 1.0 / static_cast<double>(n);
            std::fill(noise.begin(), noise.end(), u);
        } else {
            const double inv_s = 1.0 / s;
            for (int i = 0; i < n; ++i) {
                noise[static_cast<std::size_t>(i)] = noise_raw[static_cast<std::size_t>(i)] * inv_s;
            }
        }

        std::unordered_map<int, double> mixed;
        mixed.reserve(child_indices.size());
        for (int j = 0; j < n; ++j) {
            const int idx = child_indices[static_cast<std::size_t>(j)];
            auto it = root->children.find(idx);
            if (it == root->children.end() || it->second == nullptr) continue;
            const double p = std::max(0.0, it->second->prior);
            mixed[idx] = (1.0 - eps) * p + eps * noise[static_cast<std::size_t>(j)];
        }

        double z = 0.0;
        for (const auto& kv : mixed) z += kv.second;

        if (z <= 1e-12) {
            const double u = 1.0 / static_cast<double>(n);
            for (int idx : child_indices) {
                auto it = root->children.find(idx);
                if (it != root->children.end() && it->second != nullptr) {
                    it->second->prior = u;
                }
            }
        } else {
            const double inv_z = 1.0 / z;
            for (int idx : child_indices) {
                auto it = root->children.find(idx);
                if (it != root->children.end() && it->second != nullptr) {
                    it->second->prior = std::max(0.0, mixed[idx]) * inv_z;
                }
            }
        }

        // Keep alias-level debug priors aligned with noisy root child priors.
        if (!root->nn_priors_after_threshold.empty()) {
            const int a = static_cast<int>(root->nn_priors_after_threshold.size());
            std::vector<double> noisy_full(static_cast<std::size_t>(a), 0.0);

            if (!root->canonical_to_action_aliases.empty()) {
                for (const auto& kv : root->canonical_to_action_aliases) {
                    const int canon = kv.first;
                    auto itc = root->children.find(canon);
                    if (itc == root->children.end() || itc->second == nullptr) continue;

                    std::vector<int> alias_ids;
                    alias_ids.reserve(kv.second.size());
                    for (int x : kv.second) {
                        if (x >= 0 && x < a) alias_ids.push_back(x);
                    }
                    if (alias_ids.empty()) continue;

                    const double share = itc->second->prior / static_cast<double>(alias_ids.size());
                    for (int ai : alias_ids) {
                        noisy_full[static_cast<std::size_t>(ai)] = share;
                    }
                }
            } else {
                for (const auto& kv : root->children) {
                    const int idx = kv.first;
                    if (idx >= 0 && idx < a && kv.second != nullptr) {
                        noisy_full[static_cast<std::size_t>(idx)] = kv.second->prior;
                    }
                }
            }

            root->nn_priors_after_threshold = std::move(noisy_full);
        }
    }



    SelectionResult select_child(TreeNode* node) {
        SelectionResult out;
        if (node == nullptr || node->children.empty()) return out;
        if (node->children.size() == 1u) {
            out.action_index = node->children.begin()->first;
            out.child = node->children.begin()->second.get();
            return out;
        }

        const auto bounds = parent_child_q_bounds(*node);
        double best_score = -std::numeric_limits<double>::infinity();
        std::vector<int> best_actions;
        best_actions.reserve(node->children.size());

        for (const auto& kv : node->children) {
            const double score = puct_score(*node, *kv.second, bounds.first, bounds.second);
            if (best_actions.empty() || score > best_score + 1e-12) {
                best_score = score;
                best_actions.clear();
                best_actions.push_back(kv.first);
            } else if (std::abs(score - best_score) <= 1e-12) {
                best_actions.push_back(kv.first);
            }
        }

        std::uniform_int_distribution<int> pick(0, static_cast<int>(best_actions.size()) - 1);
        out.action_index = best_actions[static_cast<std::size_t>(pick(rng_))];
        const auto it = node->children.find(out.action_index);
        out.child = (it == node->children.end()) ? nullptr : it->second.get();
        return out;
    }

    void set_child_action_from_controller(TreeNode* child, const ControllerAction& action) const {
        child->has_parent_action = true;
        child->parent_action_is_controller = true;
        child->parent_controller_action = action;
    }

    void set_child_action_from_adversary(TreeNode* child, const AdversaryAction& action) const {
        child->has_parent_action = true;
        child->parent_action_is_controller = false;
        child->parent_adversary_action = action;
    }

    std::tuple<double, bool, int> expand_node(TreeNode* node, const SimState& real_state) {
        node->state_cost = state_cost(real_state);
        node->sim_time = real_state.sim_time;

        const auto decision = decision_state_for_node(node, real_state);
        SimState decision_state = decision.first;
        node->last_decision_state_time = decision.second;

        if (node->player == "controller") {
            const auto sampled = env_.sample_controller_actions(decision_state);
            const int n = static_cast<int>(sampled.actions.size());
            std::vector<int> valid;
            valid.reserve(n);
            for (int i = 0; i < n; ++i) {
                if (i >= static_cast<int>(sampled.mask.size()) || !sampled.mask[static_cast<std::size_t>(i)]) continue;
                if (!sampled.actions[static_cast<std::size_t>(i)].valid) continue;
                valid.push_back(i);
            }
            node->num_valid_actions = static_cast<int>(valid.size());
            node->nn_valid_mask = sampled.mask;

            if (valid.empty()) {
                return {0.0, false, 0};
            }
            if (valid.size() == 1u) {
                const int idx = valid[0];
                auto it = node->children.find(idx);
                if (it == node->children.end()) {
                    auto child = std::make_unique<TreeNode>();
                    child->player = "adversary";
                    child->node_id = next_node_id_++;
                    child->depth = node->depth + 1;
                    child->parent = node;
                    child->parent_action_index = idx;
                    child->prior = 1.0;
                    child->reward = 0.0;
                    child->sim_time = node->sim_time;
                    set_child_action_from_controller(child.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                    node->children.emplace(idx, std::move(child));
                }
                return {0.0, false, 1};
            }

            const auto t_infer_begin = std::chrono::steady_clock::now();
            NativeInferInputsGV2 infer_inputs = build_infer_inputs(decision_state, sampled.mask);
            auto infer_out = infer_runtime_.infer_from_inputs(infer_inputs, node->player, model_version_);
            const auto t_infer_end = std::chrono::steady_clock::now();
            perf_infer_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t_infer_end - t_infer_begin).count();
            perf_infer_calls_ += 1;

            double model_value = infer_out.first;
            std::vector<double> priors = std::move(infer_out.second);
            if (priors.size() < sampled.mask.size()) priors.resize(sampled.mask.size(), 0.0);
            if (priors.size() > sampled.mask.size()) priors.resize(sampled.mask.size());

            node->has_nn_value = true;
            node->nn_value_controller = model_value;
            node->nn_priors = priors;

            std::unordered_map<int, double> valid_prior_by_idx;
            valid_prior_by_idx.reserve(valid.size());
            for (int idx : valid) {
                valid_prior_by_idx[idx] = std::max(0.0, priors[static_cast<std::size_t>(idx)]);
            }
            double s_valid = 0.0;
            for (const auto& kv : valid_prior_by_idx) s_valid += kv.second;
            if (s_valid <= 0.0) {
                const double u = 1.0 / static_cast<double>(valid.size());
                for (int idx : valid) valid_prior_by_idx[idx] = u;
            } else {
                const double inv = 1.0 / s_valid;
                for (int idx : valid) valid_prior_by_idx[idx] *= inv;
            }

            std::unordered_map<std::string, int> sig_to_canon;
            std::unordered_map<int, std::vector<int>> canon_to_alias;
            std::unordered_map<int, int> alias_to_canon;
            for (int idx : valid) {
                const auto& act = sampled.actions[static_cast<std::size_t>(idx)];
                const std::string sig = controller_action_key(act);
                auto it = sig_to_canon.find(sig);
                if (it == sig_to_canon.end()) {
                    sig_to_canon.emplace(sig, idx);
                    canon_to_alias[idx] = {idx};
                    alias_to_canon[idx] = idx;
                } else {
                    canon_to_alias[it->second].push_back(idx);
                    alias_to_canon[idx] = it->second;
                }
            }

            node->action_alias_to_canonical = alias_to_canon;
            node->canonical_to_action_aliases = canon_to_alias;

            std::unordered_map<int, double> canonical_prior;
            canonical_prior.reserve(canon_to_alias.size());
            for (const auto& kv : canon_to_alias) {
                double p = 0.0;
                for (int alias : kv.second) {
                    const auto itp = valid_prior_by_idx.find(alias);
                    if (itp != valid_prior_by_idx.end()) p += itp->second;
                }
                canonical_prior[kv.first] = p;
            }
            double s_canon = 0.0;
            for (const auto& kv : canonical_prior) s_canon += kv.second;
            if (s_canon > 0.0) {
                const double inv = 1.0 / s_canon;
                for (auto& kv : canonical_prior) kv.second *= inv;
            } else {
                const double u = 1.0 / static_cast<double>(canonical_prior.size());
                for (auto& kv : canonical_prior) kv.second = u;
            }

            std::vector<double> priors_thr(priors.size(), 0.0);
            for (const auto& kv : valid_prior_by_idx) priors_thr[static_cast<std::size_t>(kv.first)] = kv.second;
            node->nn_priors_after_threshold = std::move(priors_thr);

            for (const auto& kv : canonical_prior) {
                const int idx = kv.first;
                auto it = node->children.find(idx);
                if (it == node->children.end()) {
                    auto child = std::make_unique<TreeNode>();
                    child->player = "adversary";
                    child->node_id = next_node_id_++;
                    child->depth = node->depth + 1;
                    child->parent = node;
                    child->parent_action_index = idx;
                    child->prior = kv.second;
                    child->reward = 0.0;
                    child->sim_time = node->sim_time;
                    set_child_action_from_controller(child.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                    node->children.emplace(idx, std::move(child));
                } else {
                    it->second->prior = kv.second;
                    if (!it->second->has_parent_action) {
                        set_child_action_from_controller(it->second.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                    }
                }
            }
            return {model_value, true, static_cast<int>(valid.size())};
        }

        // Adversary branch
        const auto forbidden = replay_forbidden_stop_ids(node, decision_state, real_state);
        const auto sampled = env_.sample_adversary_actions(decision_state, forbidden);
        const int n = static_cast<int>(sampled.actions.size());
        std::vector<int> valid;
        valid.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (i >= static_cast<int>(sampled.mask.size()) || !sampled.mask[static_cast<std::size_t>(i)]) continue;
            if (!sampled.actions[static_cast<std::size_t>(i)].valid) continue;
            valid.push_back(i);
        }
        node->num_valid_actions = static_cast<int>(valid.size());
        node->nn_valid_mask = sampled.mask;

        if (valid.empty()) return {0.0, false, 0};
        if (valid.size() == 1u) {
            const int idx = valid[0];
            auto it = node->children.find(idx);
            if (it == node->children.end()) {
                auto child = std::make_unique<TreeNode>();
                child->player = "controller";
                child->node_id = next_node_id_++;
                child->depth = node->depth + 1;
                child->parent = node;
                child->parent_action_index = idx;
                child->prior = 1.0;
                child->reward = 0.0;
                child->sim_time = node->sim_time;
                set_child_action_from_adversary(child.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                node->children.emplace(idx, std::move(child));
            }
            return {0.0, false, 1};
        }

        const auto t_infer_begin = std::chrono::steady_clock::now();
        NativeInferInputsGV2 infer_inputs = build_infer_inputs(decision_state, sampled.mask);
        auto infer_out = infer_runtime_.infer_from_inputs(infer_inputs, node->player, model_version_);
        const auto t_infer_end = std::chrono::steady_clock::now();
        perf_infer_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t_infer_end - t_infer_begin).count();
        perf_infer_calls_ += 1;

        double model_value = infer_out.first;
        std::vector<double> priors = std::move(infer_out.second);
        if (priors.size() < sampled.mask.size()) priors.resize(sampled.mask.size(), 0.0);
        if (priors.size() > sampled.mask.size()) priors.resize(sampled.mask.size());

        node->has_nn_value = true;
        node->nn_value_controller = model_value;
        node->nn_priors = priors;
        node->action_alias_to_canonical.clear();
        node->canonical_to_action_aliases.clear();

        std::unordered_map<int, double> valid_prior_by_idx;
        valid_prior_by_idx.reserve(valid.size());
        for (int idx : valid) {
            valid_prior_by_idx[idx] = std::max(0.0, priors[static_cast<std::size_t>(idx)]);
        }
        double s_valid = 0.0;
        for (const auto& kv : valid_prior_by_idx) s_valid += kv.second;
        if (s_valid <= 0.0) {
            const double u = 1.0 / static_cast<double>(valid.size());
            for (int idx : valid) valid_prior_by_idx[idx] = u;
        } else {
            const double inv = 1.0 / s_valid;
            for (int idx : valid) valid_prior_by_idx[idx] *= inv;
        }

        std::vector<double> priors_thr(priors.size(), 0.0);
        for (const auto& kv : valid_prior_by_idx) priors_thr[static_cast<std::size_t>(kv.first)] = kv.second;
        node->nn_priors_after_threshold = std::move(priors_thr);

        for (int idx : valid) {
            auto it = node->children.find(idx);
            if (it == node->children.end()) {
                auto child = std::make_unique<TreeNode>();
                child->player = "controller";
                child->node_id = next_node_id_++;
                child->depth = node->depth + 1;
                child->parent = node;
                child->parent_action_index = idx;
                child->prior = valid_prior_by_idx[idx];
                child->reward = 0.0;
                child->sim_time = node->sim_time;
                set_child_action_from_adversary(child.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                node->children.emplace(idx, std::move(child));
            } else {
                it->second->prior = valid_prior_by_idx[idx];
                if (!it->second->has_parent_action) {
                    set_child_action_from_adversary(it->second.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                }
            }
        }
        return {model_value, true, static_cast<int>(valid.size())};
    }

    SimState restore_state_for_node(TreeNode* node) {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        if (node == nullptr) return in_.root_state;
        if (node->has_snapshot) {
            const auto t1 = clock::now();
            perf_restore_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
            return node->cached_state;
        }
        if (node->parent == nullptr) {
            throw std::runtime_error("restore_state_for_node: root node missing snapshot");
        }

        SimState state = restore_state_for_node(node->parent);
        const double parent_cost = node->parent->state_cost;

        if (!node->has_parent_action) {
            throw std::runtime_error("restore_state_for_node: missing parent action");
        }
        if (node->parent->player == "controller") {
            env_.apply_controller_action_inplace(state, node->parent_controller_action);
        } else {
            env_.apply_adversary_action_inplace(state, node->parent_adversary_action);
        }

        const double child_cost = state_cost(state);
        node->reward = transition_reward(parent_cost, child_cost, in_);
        node->state_cost = child_cost;
        node->sim_time = state.sim_time;
        node->cached_state = state;
        node->has_snapshot = true;

        const auto t1 = clock::now();
        perf_restore_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        return state;
    }

    std::pair<TreeNode*, SimState> advance_through_single_child_chain(
        TreeNode* node,
        SimState state,
        std::vector<TreeNode*>* search_path,
        std::vector<ForcedStepLog>* forced_logs) {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();

        int hops = 0;
        while (hops < std::max(1, in_.max_forced_hops)) {
            const auto decision = decision_state_for_node(node, state);
            SimState decision_state = decision.first;
            node->last_decision_state_time = decision.second;

            if (node->player == "controller") {
                const auto sampled = env_.sample_controller_actions(decision_state);
                std::vector<int> valid;
                for (int i = 0; i < static_cast<int>(sampled.actions.size()); ++i) {
                    if (i >= static_cast<int>(sampled.mask.size()) || !sampled.mask[static_cast<std::size_t>(i)]) continue;
                    if (!sampled.actions[static_cast<std::size_t>(i)].valid) continue;
                    valid.push_back(i);
                }
                node->num_valid_actions = static_cast<int>(valid.size());
                if (valid.size() != 1u) break;

                if (forced_logs != nullptr) {
                    ForcedStepLog item;
                    item.node = node;
                    item.state_snapshot = state;
                    item.num_valid_actions = static_cast<int>(valid.size());
                    item.unique_actions = static_cast<int>(node->children.size());
                    forced_logs->push_back(std::move(item));
                }

                const int idx = valid[0];
                TreeNode* child = nullptr;
                auto it = node->children.find(idx);
                if (it == node->children.end()) {
                    auto c = std::make_unique<TreeNode>();
                    c->player = "adversary";
                    c->node_id = next_node_id_++;
                    c->depth = node->depth + 1;
                    c->parent = node;
                    c->parent_action_index = idx;
                    c->prior = 1.0;
                    c->reward = 0.0;
                    c->sim_time = node->sim_time;
                    set_child_action_from_controller(c.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                    child = c.get();
                    node->children.emplace(idx, std::move(c));
                } else {
                    child = it->second.get();
                    if (!child->has_parent_action) {
                        set_child_action_from_controller(child, sampled.actions[static_cast<std::size_t>(idx)]);
                    }
                }

                const double parent_cost = node->state_cost;
                env_.apply_controller_action_inplace(state, sampled.actions[static_cast<std::size_t>(idx)]);
                const double child_cost = state_cost(state);

                child->reward = transition_reward(parent_cost, child_cost, in_);
                child->state_cost = child_cost;
                child->sim_time = state.sim_time;
                child->cached_state = state;
                child->has_snapshot = true;

                node = child;
                search_path->push_back(node);
                hops += 1;
                perf_forced_steps_ += 1;
                continue;
            }

            const auto forbidden = replay_forbidden_stop_ids(node, decision_state, state);
            const auto sampled = env_.sample_adversary_actions(decision_state, forbidden);
            std::vector<int> valid;
            for (int i = 0; i < static_cast<int>(sampled.actions.size()); ++i) {
                if (i >= static_cast<int>(sampled.mask.size()) || !sampled.mask[static_cast<std::size_t>(i)]) continue;
                if (!sampled.actions[static_cast<std::size_t>(i)].valid) continue;
                valid.push_back(i);
            }
            node->num_valid_actions = static_cast<int>(valid.size());
            if (valid.size() != 1u) break;

            if (forced_logs != nullptr) {
                ForcedStepLog item;
                item.node = node;
                item.state_snapshot = state;
                item.num_valid_actions = static_cast<int>(valid.size());
                item.unique_actions = static_cast<int>(node->children.size());
                forced_logs->push_back(std::move(item));
            }

            const int idx = valid[0];
            TreeNode* child = nullptr;
            auto it = node->children.find(idx);
            if (it == node->children.end()) {
                auto c = std::make_unique<TreeNode>();
                c->player = "controller";
                c->node_id = next_node_id_++;
                c->depth = node->depth + 1;
                c->parent = node;
                c->parent_action_index = idx;
                c->prior = 1.0;
                c->reward = 0.0;
                c->sim_time = node->sim_time;
                set_child_action_from_adversary(c.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                child = c.get();
                node->children.emplace(idx, std::move(c));
            } else {
                child = it->second.get();
                if (!child->has_parent_action) {
                    set_child_action_from_adversary(child, sampled.actions[static_cast<std::size_t>(idx)]);
                }
            }

            const double parent_cost = node->state_cost;
            env_.apply_adversary_action_inplace(state, sampled.actions[static_cast<std::size_t>(idx)]);
            const double child_cost = state_cost(state);

            child->reward = transition_reward(parent_cost, child_cost, in_);
            child->state_cost = child_cost;
            child->sim_time = state.sim_time;
            child->cached_state = state;
            child->has_snapshot = true;

            node = child;
            search_path->push_back(node);
            hops += 1;
            perf_forced_steps_ += 1;
        }

        if (!node->has_snapshot) {
            node->cached_state = state;
            node->has_snapshot = true;
        }

        const auto t1 = clock::now();
        perf_forced_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        return {node, std::move(state)};
    }

    void backpropagate(const std::vector<TreeNode*>& search_path, double leaf_value) {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();

        double value = leaf_value;
        for (auto it = search_path.rbegin(); it != search_path.rend(); ++it) {
            TreeNode* node = *it;
            node->value_sum += value;
            node->visits += 1;

            TreeNode* parent = node->parent;
            if (parent == nullptr) break;

            const bool parent_is_branching =
                (parent->num_valid_actions > 1) || (parent->children.size() > 1);
            const double disc = time_discount(node->sim_time, parent->sim_time, in_);
            const double reward_used = parent_is_branching ? node->reward : 0.0;
            if (parent_is_branching) {
                min_max_.update(reward_used + disc * node->mean_value());
            }
            value = reward_used + disc * value;
        }

        const auto t1 = clock::now();
        perf_backprop_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    }

    std::vector<double> compute_root_prior(
        const TreeNode& root,
        const std::vector<uint8_t>& valid_mask) const {
        const int n = static_cast<int>(valid_mask.size());
        std::vector<double> visit_mass(static_cast<std::size_t>(n), 0.0);

        if (!root.canonical_to_action_aliases.empty()) {
            for (const auto& kv : root.canonical_to_action_aliases) {
                const auto itc = root.children.find(kv.first);
                const double visits = (itc == root.children.end()) ? 0.0 : static_cast<double>(itc->second->visits);
                const std::vector<int>& aliases = kv.second;
                if (aliases.empty()) continue;
                const double share = visits / static_cast<double>(aliases.size());
                for (int a : aliases) {
                    if (a >= 0 && a < n) visit_mass[static_cast<std::size_t>(a)] += share;
                }
            }
        } else {
            for (const auto& kv : root.children) {
                const int idx = kv.first;
                if (idx >= 0 && idx < n) {
                    visit_mass[static_cast<std::size_t>(idx)] = static_cast<double>(kv.second->visits);
                }
            }
        }

        double total = 0.0;
        for (int i = 0; i < n; ++i) {
            if (valid_mask[static_cast<std::size_t>(i)]) total += visit_mass[static_cast<std::size_t>(i)];
        }
        if (total > 0.0) {
            for (int i = 0; i < n; ++i) {
                if (valid_mask[static_cast<std::size_t>(i)]) {
                    visit_mass[static_cast<std::size_t>(i)] /= total;
                } else {
                    visit_mass[static_cast<std::size_t>(i)] = 0.0;
                }
            }
            return visit_mass;
        }

        int valid_count = 0;
        for (int i = 0; i < n; ++i) {
            if (valid_mask[static_cast<std::size_t>(i)]) valid_count += 1;
        }
        if (valid_count <= 0) return visit_mass;
        const double u = 1.0 / static_cast<double>(valid_count);
        for (int i = 0; i < n; ++i) {
            visit_mass[static_cast<std::size_t>(i)] = valid_mask[static_cast<std::size_t>(i)] ? u : 0.0;
        }
        return visit_mass;
    }

    void run_one_simulation(int sim_iteration, TreeNode* root, SearchOutput* out) {
        using clock = std::chrono::steady_clock;

        // 1) Selection
        const auto t_sel_begin = clock::now();
        TreeNode* node = root;
        std::vector<TreeNode*> search_path;
        search_path.push_back(node);
        int root_selected_action = -1;
        int root_selected_child_node_id = -1;

        while (node->expanded()) {
            SelectionResult sel = select_child(node);
            if (sel.child == nullptr) break;
            if (search_path.size() == 1u) {
                root_selected_action = sel.action_index;
                root_selected_child_node_id = sel.child->node_id;
            }
            node = sel.child;
            search_path.push_back(node);
            perf_selection_steps_ += 1;
        }
        const auto t_sel_end = clock::now();
        perf_selection_sec_ +=
            std::chrono::duration_cast<std::chrono::duration<double>>(t_sel_end - t_sel_begin).count();

        // 2) Restore to selected node
        SimState state = restore_state_for_node(node);
        // 3) Forced single-child chain
        std::vector<ForcedStepLog> forced_logs;
        auto forced = advance_through_single_child_chain(node, state, &search_path, &forced_logs);
        TreeNode* leaf_node = forced.first;
        SimState leaf_state = std::move(forced.second);

        // 4) Expand leaf
        const auto t_expand_begin = clock::now();
        const auto leaf_expand = expand_node(leaf_node, leaf_state);
        const auto t_expand_end = clock::now();
        perf_expand_sec_ +=
            std::chrono::duration_cast<std::chrono::duration<double>>(t_expand_end - t_expand_begin).count();

        const double leaf_value = std::get<0>(leaf_expand);
        const bool leaf_nn_called = std::get<1>(leaf_expand);
        const int leaf_num_valid = std::get<2>(leaf_expand);
        const bool parent_multi = (leaf_node->parent == nullptr) || (leaf_node->parent->children.size() > 1u);
        std::string phase = "terminal";
        if (leaf_num_valid == 1) {
            phase = parent_multi ? "single-child" : "trivial-single-child";
        } else if (leaf_num_valid > 1) {
            phase = parent_multi ? "multiple-child" : "trivial-multiple-child";
        }

        // 5) Backprop
        backpropagate(search_path, leaf_value);

        // Emit forced-step rows before the main leaf row (matches Python logger semantics).
        for (const auto& flog : forced_logs) {
            TreeNode* n = flog.node;
            if (n == nullptr) continue;
            const SimState& st = flog.state_snapshot;
            const bool parent_multi = (n->parent == nullptr) || (n->parent->children.size() > 1u);
            std::string forced_phase = "terminal";
            if (flog.num_valid_actions == 1) {
                forced_phase = parent_multi ? "single-child" : "trivial-single-child";
            } else if (flog.num_valid_actions > 1) {
                forced_phase = parent_multi ? "multiple-child" : "trivial-multiple-child";
            }

            IterEvent ev;
            ev.sim_iteration = sim_iteration;
            ev.selected_action_index = -1;
            ev.selected_child_node_id = -1;
            ev.action_index = n->parent_action_index;
            ev.leaf_node_id = n->node_id;
            ev.parent_node_id = (n->parent == nullptr) ? -1 : n->parent->node_id;
            ev.leaf_depth = n->depth;
            ev.sim_time_before = st.sim_time;
            ev.sim_time_after = st.sim_time;
            ev.decision_state_time = n->last_decision_state_time;
            ev.leaf_state_cost = n->state_cost;
            ev.prior = n->prior;
            ev.reward = n->reward;
            ev.decode_credit_balance = st.stats.decode_credit_balance;
            ev.num_valid_actions = flog.num_valid_actions;
            ev.unique_actions = flog.unique_actions;
            ev.root_visits_after = root->visits;
            ev.root_value_sum_after = root->value_sum;
            ev.root_mean_value_after = root->mean_value();
            ev.nn_called = false;
            ev.has_nn_value_controller = false;
            ev.nn_value_controller = 0.0;
            ev.player_to_act = n->player;
            ev.player_acted_to_create_this_node =
                (n->parent == nullptr) ? std::string("root_no_parent") : n->parent->player;
            ev.phase = "forced_step:" + forced_phase;
            ev.requests_in_system = static_cast<int>(st.stats.active_request_ids.size());
            ev.requests_generated = st.stats.requests_generated;
            ev.requests_completed = st.stats.requests_completed;
            ev.slo_violations = st.stats.slo_violations;
            ev.total_lateness = st.stats.slo_lateness_sum;
            ev.avg_lateness = (st.stats.slo_violations > 0)
                ? (st.stats.slo_lateness_sum / static_cast<double>(st.stats.slo_violations))
                : 0.0;
            ev.state_pending_adv_tick = st.stats.pending_adv_tick;
            ev.has_state_last_adv_tick = true;
            ev.state_last_adv_tick = st.stats.last_adv_tick;
            ev.active_request_ids_json = json_int_vec(st.stats.active_request_ids);
            ev.waiting_request_ids_json = json_int_vec(st.stats.active_request_ids);
            ev.completed_request_ids_json = json_int_vec(st.stats.completed_request_ids);
            ev.dropped_request_ids_json = json_int_vec(st.stats.dropped_request_ids);
            ev.stopped_decode_request_ids_json = json_int_vec(st.stats.stopped_decode_request_ids);
            ev.violated_request_ids_json = json_int_vec(st.stats.violated_request_ids);
            ev.decode_tokens_counted_by_id_json = json_i32_i32_map(st.stats.decode_tokens_counted_by_id);
            ev.per_request_prefill_lateness_by_id_json = json_i32_f64_map(st.stats.per_request_prefill_lateness_by_id);
            ev.per_request_decode_lateness_by_id_json = json_i32_f64_map(st.stats.per_request_decode_lateness_by_id);

            const std::vector<double> root_prior_iter = compute_root_prior(*root, out->root_nn_valid_mask);
            ev.root_valid_mask_json = json_u8_vec(out->root_nn_valid_mask);
            ev.root_nn_priors_json = "[]";
            ev.root_nn_priors_after_threshold_json = "[]";
            ev.root_mcts_prior_json = json_f64_vec(root_prior_iter);

            if (n->has_parent_action) {
                if (n->parent_action_is_controller) {
                    const auto& a = n->parent_controller_action;
                    ev.action_repr = controller_action_to_repr(a);
                    ev.has_controller_token_budget = true;
                    ev.controller_token_budget = a.token_budget;
                    ev.controller_selected_ids_json = json_int_vec(a.selected_request_ids);
                    ev.controller_allocations_json = json_i32_i32_map(a.token_allocations);
                    ev.controller_prefill_allocations_json = json_i32_i32_map(a.prefill_allocations);
                    ev.controller_decode_allocations_json = json_i32_i32_map(a.decode_allocations);
                    ev.controller_heuristic = a.heuristic;
                    ev.controller_strategy = a.strategy;
                    for (const auto& kv : a.prefill_allocations) ev.controller_prefill_total += kv.second;
                    for (const auto& kv : a.decode_allocations) ev.controller_decode_total += kv.second;
                } else {
                    const auto& a = n->parent_adversary_action;
                    ev.action_repr = adversary_action_to_repr(a);
                    ev.adversary_requests_json = adversary_requests_json(a);
                    ev.adversary_prefill_slos_json = adversary_prefill_slos_json(a);
                    ev.adversary_decode_slos_json = adversary_decode_slos_json(a);
                    ev.adversary_prefill_deadlines_by_id_json = build_adversary_prefill_deadlines_json(a, st);
                }
            }

            out->iter_events.push_back(std::move(ev));
        }

        // Iter event
        IterEvent ev;
        ev.sim_iteration = sim_iteration;
        ev.selected_action_index = root_selected_action;
        ev.selected_child_node_id = root_selected_child_node_id;
        ev.action_index = leaf_node->parent_action_index;
        ev.leaf_node_id = leaf_node->node_id;
        ev.parent_node_id = (leaf_node->parent == nullptr) ? -1 : leaf_node->parent->node_id;
        ev.leaf_depth = leaf_node->depth;
        // Canonical Python iter logger semantics:
        // leaf expand rows use state_snapshot.sim_time as both start/end unless
        // explicitly overridden. Keep native canonical rows aligned so extracted-style
        // reordering remains causal.
        ev.sim_time_before = leaf_state.sim_time;
        ev.sim_time_after = leaf_state.sim_time;
        ev.decision_state_time = leaf_node->last_decision_state_time;
        ev.leaf_state_cost = leaf_node->state_cost;
        ev.prior = leaf_node->prior;
        ev.reward = leaf_node->reward;
        ev.decode_credit_balance = leaf_state.stats.decode_credit_balance;
        ev.num_valid_actions = leaf_num_valid;
        ev.unique_actions = static_cast<int>(leaf_node->children.size());
        ev.root_visits_after = root->visits;
        ev.root_value_sum_after = root->value_sum;
        ev.root_mean_value_after = root->mean_value();
        ev.nn_called = leaf_nn_called;
        ev.has_nn_value_controller = leaf_node->has_nn_value;
        ev.nn_value_controller = leaf_node->nn_value_controller;
        ev.player_to_act = leaf_node->player;
        ev.player_acted_to_create_this_node =
            (leaf_node->parent == nullptr) ? std::string("root_no_parent") : leaf_node->parent->player;
        ev.phase = phase;
        ev.requests_in_system = static_cast<int>(leaf_state.stats.active_request_ids.size());
        ev.requests_generated = leaf_state.stats.requests_generated;
        ev.requests_completed = leaf_state.stats.requests_completed;
        ev.slo_violations = leaf_state.stats.slo_violations;
        ev.total_lateness = leaf_state.stats.slo_lateness_sum;
        ev.avg_lateness = (leaf_state.stats.slo_violations > 0)
            ? (leaf_state.stats.slo_lateness_sum / static_cast<double>(leaf_state.stats.slo_violations))
            : 0.0;
        ev.state_pending_adv_tick = leaf_state.stats.pending_adv_tick;
        ev.has_state_last_adv_tick = true;
        ev.state_last_adv_tick = leaf_state.stats.last_adv_tick;
        ev.active_request_ids_json = json_int_vec(leaf_state.stats.active_request_ids);
        ev.waiting_request_ids_json = json_int_vec(leaf_state.stats.active_request_ids);
        ev.completed_request_ids_json = json_int_vec(leaf_state.stats.completed_request_ids);
        ev.dropped_request_ids_json = json_int_vec(leaf_state.stats.dropped_request_ids);
        ev.stopped_decode_request_ids_json = json_int_vec(leaf_state.stats.stopped_decode_request_ids);
        ev.violated_request_ids_json = json_int_vec(leaf_state.stats.violated_request_ids);
        ev.decode_tokens_counted_by_id_json = json_i32_i32_map(leaf_state.stats.decode_tokens_counted_by_id);
        ev.per_request_prefill_lateness_by_id_json = json_i32_f64_map(leaf_state.stats.per_request_prefill_lateness_by_id);
        ev.per_request_decode_lateness_by_id_json = json_i32_f64_map(leaf_state.stats.per_request_decode_lateness_by_id);

        if (root_selected_action >= 0) {
            const auto it = root->children.find(root_selected_action);
            if (it != root->children.end()) {
                const TreeNode* c = it->second.get();
                ev.selected_child_visits_after = c->visits;
                ev.selected_child_value_sum_after = c->value_sum;
                ev.selected_child_mean_value_after = c->mean_value();
                ev.selected_child_prior = c->prior;
            }
        }

        const std::vector<double> root_prior_iter = compute_root_prior(*root, out->root_nn_valid_mask);
        ev.root_valid_mask_json = json_u8_vec(out->root_nn_valid_mask);
        ev.root_nn_priors_json = json_f64_vec(out->root_nn_priors);
        ev.root_nn_priors_after_threshold_json = json_f64_vec(out->root_nn_priors_after_threshold);
        ev.root_mcts_prior_json = json_f64_vec(root_prior_iter);

        if (leaf_node->has_parent_action) {
            if (leaf_node->parent_action_is_controller) {
                const auto& a = leaf_node->parent_controller_action;
                ev.action_repr = controller_action_to_repr(a);
                ev.has_controller_token_budget = true;
                ev.controller_token_budget = a.token_budget;
                ev.controller_selected_ids_json = json_int_vec(a.selected_request_ids);
                ev.controller_allocations_json = json_i32_i32_map(a.token_allocations);
                ev.controller_prefill_allocations_json = json_i32_i32_map(a.prefill_allocations);
                ev.controller_decode_allocations_json = json_i32_i32_map(a.decode_allocations);
                ev.controller_heuristic = a.heuristic;
                ev.controller_strategy = a.strategy;
                for (const auto& kv : a.prefill_allocations) ev.controller_prefill_total += kv.second;
                for (const auto& kv : a.decode_allocations) ev.controller_decode_total += kv.second;
            } else {
                const auto& a = leaf_node->parent_adversary_action;
                ev.action_repr = adversary_action_to_repr(a);
                ev.adversary_requests_json = adversary_requests_json(a);
                ev.adversary_prefill_slos_json = adversary_prefill_slos_json(a);
                ev.adversary_decode_slos_json = adversary_decode_slos_json(a);
                ev.adversary_prefill_deadlines_by_id_json =
                    build_adversary_prefill_deadlines_json(a, leaf_state);
            }
        }

        out->iter_events.push_back(std::move(ev));
    }
};

}  // namespace

SearchOutput run_search_torchscript(
    const SearchInput& in,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version) {
    SearchRunner runner(in, infer_runtime, model_version);
    return runner.run();
}

}  // namespace mcts_native_gv2
