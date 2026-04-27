#include "gv3_native_selfplay.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <unordered_set>

namespace mcts_native_gv2 {
namespace {

struct AnyActions {
    SampledActionSet<ControllerAction> controller;
    SampledActionSet<AdversaryAction> adversary;
    std::vector<int> valid;
};

struct HistoryNode {
    SimState state;
    SimState pre_controller_state;
    bool has_pre_controller_state = false;
    std::string player = "adversary";
    int depth = 0;
    int history_hops = 0;
    std::vector<int> untried_action_indices;
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
    bool exhausted = false;
    bool allow_duplicate_fallback = true;
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
    int action_index) {
    if (node == nullptr) return;
    const std::string acted = node->player;
    bool has_pre_controller = false;
    SimState pre_controller;

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
}

int advance_to_branching(
    GV2VirtualEnvironment& env,
    HistoryNode* node,
    int max_forced_steps) {
    if (node == nullptr) return 0;
    int forced = 0;
    const int limit = std::max(0, max_forced_steps);
    while (forced < limit) {
        AnyActions sampled = sample_actions(env, node->state, node->player);
        if (sampled.valid.size() != 1u) break;
        apply_action(env, node, sampled, sampled.valid[0]);
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
    int max_total_steps) {
    HistoryNode node;
    node.state = initial_state;
    node.player = start_player;
    node.depth = start_depth;

    const int step_cap = std::max(1, max_total_steps);
    const int target = std::max(0, target_hops);
    int steps = 0;
    int hops = 0;

    while (hops < target && steps < step_cap) {
        steps += advance_to_branching(env, &node, std::min(2000, step_cap - steps));
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

        apply_action(env, &node, sampled, idx);
        ++steps;
        if (nontrivial) ++hops;
    }

    if (steps < step_cap) {
        (void)advance_to_branching(env, &node, std::min(2000, step_cap - steps));
    }
    node.history_hops = hops;
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
        session->max_total_steps);
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
        apply_action(env, &child, sampled, idx);
        child.history_hops = node.history_hops + 1;
        (void)advance_to_branching(env, &child, std::min(2000, session->max_total_steps));
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
        session->max_total_steps);
}

SimState decision_state_for_root(const HistoryNode& node) {
    if (node.player != "adversary" || !node.has_pre_controller_state || node.state.stats.missed_adv_source != 1) {
        return node.state;
    }

    SimState out = node.pre_controller_state;
    const double tick = node.state.stats.next_adv_tick;
    if (tick >= 0.0 && out.sim_time + 1e-9 < tick) {
        out.sim_time = tick;
    }

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
    sample.used_bootstrap = model_version > 0;
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
    for (const auto& sig : cfg.initial_seen_signatures) {
        if (!sig.empty()) session.seen_signatures.insert(sig);
    }

    std::mt19937 eval_rng(static_cast<uint32_t>(std::max(0, cfg.eval_split_seed)));
    std::uniform_real_distribution<double> eval_pick(0.0, 1.0);
    const double eval_ratio = std::max(0.0, std::min(1.0, cfg.eval_split_ratio));

    std::unordered_set<std::string> root_sigs;
    int controller_train = 0;
    int controller_eval = 0;
    int adversary_train = 0;
    int adversary_eval = 0;

    result.samples.reserve(static_cast<std::size_t>(session.target_roots));
    while (session.emitted_roots < session.target_roots) {
        HistoryNode node = next_root_node(&session, env);
        const int root_id = session.next_root_id++;
        session.emitted_roots += 1;

        (void)advance_to_branching(env, &node, std::max(0, cfg.max_forced_hops_per_root));
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

        SearchOutput search = run_search_torchscript_with_env(in, env, infer_runtime, model_version);
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
