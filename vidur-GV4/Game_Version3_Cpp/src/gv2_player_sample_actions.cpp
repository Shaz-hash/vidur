#include "gv2_player_sample_actions.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <type_traits>

namespace mcts_native_gv2 {

namespace {

struct ReqView {
    int rid = -1;
    bool prefill_done = false;
    int rem_prefill = 0;
    int rem_decode = 0;
    int decode_processed = 0;
    double arrived_at = 0.0;
    double prefill_slo = 0.0;
    double prefill_deadline = -1.0;
    double prefill_lateness = 0.0;
    double decode_lateness = 0.0;
    double total_lateness = 0.0;
};

inline int remaining_prefill(const RequestState& r) {
    return std::max(0, int(r.num_prefill_tokens) - int(r.num_processed_prefill_tokens));
}

inline int remaining_decode(const RequestState& r) {
    return std::max(0, int(r.num_decode_tokens) - int(r.num_processed_decode_tokens));
}

struct ReqViewTable {
    std::vector<ReqView> values;
    std::vector<int> index_by_rid;

    bool empty() const { return values.empty(); }
    std::size_t size() const { return values.size(); }

    const ReqView& at(int rid) const {
        assert(rid >= 0 && rid < static_cast<int>(index_by_rid.size()));
        const int index = index_by_rid[static_cast<std::size_t>(rid)];
        assert(index >= 0 && index < static_cast<int>(values.size()));
        return values[static_cast<std::size_t>(index)];
    }
};

const ReqViewTable& build_req_views(const SimState& state) {
    thread_local ReqViewTable out;
    out.values.clear();
    out.values.reserve(state.requests.size());

    const double sim_time = double(state.sim_time);
    int max_rid = -1;
    for (const auto& r : state.requests) {
        if (r.feature_only) continue;
        if (r.completed) continue;

        const bool prefill_done = r.prefill_done();
        const int rem_pref = remaining_prefill(r);
        const int rem_dec = remaining_decode(r);

        if ((!prefill_done && rem_pref <= 0) || (prefill_done && rem_dec <= 0)) {
            continue;
        }

        ReqView v;
        v.rid = r.request_id;
        v.prefill_done = prefill_done;
        v.rem_prefill = rem_pref;
        v.rem_decode = rem_dec;
        v.decode_processed = int(r.num_processed_decode_tokens);
        v.arrived_at = r.arrived_at;
        v.prefill_slo = r.prefill_slo_time;

        const double fallback_pref_deadline =
            v.arrived_at + std::max(0.0, v.prefill_slo);
        v.prefill_deadline = r.prefill_deadline > 0.0
            ? r.prefill_deadline
            : fallback_pref_deadline;

        const auto it_pref =
            state.stats.per_request_prefill_lateness_by_id.find(r.request_id);
        const auto it_dec =
            state.stats.per_request_decode_lateness_by_id.find(r.request_id);
        if (it_pref != state.stats.per_request_prefill_lateness_by_id.end()) {
            v.prefill_lateness = double(it_pref->second);
        } else if (prefill_done) {
            v.prefill_lateness = std::max(0.0, double(r.prefill_lateness));
        } else {
            v.prefill_lateness = std::max(0.0, sim_time - v.prefill_deadline);
        }
        v.decode_lateness =
            it_dec != state.stats.per_request_decode_lateness_by_id.end()
            ? double(it_dec->second)
            : 0.0;
        v.total_lateness =
            std::max(0.0, v.prefill_lateness) +
            std::max(0.0, v.decode_lateness);

        max_rid = std::max(max_rid, v.rid);
        out.values.push_back(v);
    }

    out.index_by_rid.assign(
        static_cast<std::size_t>(std::max(0, max_rid + 1)), -1);
    for (int index = 0; index < static_cast<int>(out.values.size()); ++index) {
        const int rid = out.values[static_cast<std::size_t>(index)].rid;
        if (rid >= 0) {
            out.index_by_rid[static_cast<std::size_t>(rid)] = index;
        }
    }
    return out;
}

std::pair<int, int> window_usage(
    const GameStats& stats,
    double anchor_time,
    double launch_window_sec,
    double eps) {
    const double lo = anchor_time - launch_window_sec;
    const double hi = anchor_time + eps;

    int total_count = 0;
    int total_prefill = 0;

    if (!stats.recent_launches.empty()) {
        for (const auto& x : stats.recent_launches) {
            if ((x.timestamp + eps) < lo) continue;
            if (x.timestamp > hi) continue;
            total_count += std::max(0, x.count);
            total_prefill += std::max(0, x.prefill_tokens);
        }
        return {total_count, total_prefill};
    }

    for (double ts : stats.recent_arrivals) {
        if ((ts + eps) < lo) continue;
        if (ts > hi) continue;
        total_count += 1;
    }
    return {total_count, total_prefill};
}

std::vector<int> decode_active_ids(const ReqViewTable& req_views) {
    std::vector<int> out;
    out.reserve(req_views.size());
    for (const ReqView& rv : req_views.values) {
        if (rv.prefill_done && rv.rem_decode > 0) out.push_back(rv.rid);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<int> stop_ids_for_rule(
    const std::string& rule,
    const ReqViewTable& req_views,
    const std::unordered_set<int>& forbidden_stop_ids) {
    std::vector<int> decode_ids = decode_active_ids(req_views);
    decode_ids.erase(
        std::remove_if(
            decode_ids.begin(),
            decode_ids.end(),
            [&](int rid) { return forbidden_stop_ids.find(rid) != forbidden_stop_ids.end(); }),
        decode_ids.end());

    if (decode_ids.empty()) return {};
    if (rule == "stop_none") return {};

    if (rule == "stop_longest_decode") {
        const int best = *std::max_element(
            decode_ids.begin(),
            decode_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.decode_processed != vb.decode_processed) {
                    return va.decode_processed < vb.decode_processed;
                }
                return a > b;  // tie: smaller rid wins => invert for max_element
            });
        return {best};
    }

    if (rule == "stop_shortest_decode") {
        const int best = *std::min_element(
            decode_ids.begin(),
            decode_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.decode_processed != vb.decode_processed) {
                    return va.decode_processed < vb.decode_processed;
                }
                return a < b;
            });
        return {best};
    }

    if (rule == "stop_all_decodes_over_512") {
        std::vector<int> out;
        for (int rid : decode_ids) {
            if (req_views.at(rid).decode_processed > 512) out.push_back(rid);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    if (rule == "stop_all_decodes_over_216") {
        std::vector<int> out;
        for (int rid : decode_ids) {
            if (req_views.at(rid).decode_processed > 216) out.push_back(rid);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    return {};
}

double prefill_slo_for_tokens(const AdversarySamplerConfig& cfg, int prefill_tokens) {
    const auto it = cfg.prefill_slo_by_tokens.find(int(prefill_tokens));
    if (it != cfg.prefill_slo_by_tokens.end()) return std::max(0.0, double(it->second));
    return 0.1;
}

double prefill_eta_for_tokens(const ControllerSamplerConfig& cfg, int prefill_tokens) {
    const std::size_t n = std::min(cfg.prefill_profile_tokens.size(), cfg.prefill_profile_times.size());
    if (n > 0) {
        std::size_t best = 0;
        long long best_dist = std::llabs(static_cast<long long>(cfg.prefill_profile_tokens[0]) -
                                         static_cast<long long>(prefill_tokens));
        for (std::size_t i = 1; i < n; ++i) {
            const long long dist = std::llabs(static_cast<long long>(cfg.prefill_profile_tokens[i]) -
                                              static_cast<long long>(prefill_tokens));
            if (dist < best_dist) {
                best = i;
                best_dist = dist;
            }
        }
        return std::max(0.0, double(cfg.prefill_profile_times[best]));
    }
    return double(std::max(0, prefill_tokens)) /
           std::max(1.0, cfg.prefill_eta_tokens_per_sec);
}

void write_eviction_targets(
    const std::string& rule,
    const ReqViewTable& req_views,
    const std::vector<int>& prefill_ids,
    const std::vector<int>& decode_ids,
    double eps,
    std::vector<int>* out) {
    out->clear();
    if (rule == "evict_none") return;

    if (rule == "evict_largest_prefill") {
        if (prefill_ids.empty()) return;
        out->push_back(*std::max_element(
            prefill_ids.begin(), prefill_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.rem_prefill != vb.rem_prefill) {
                    return va.rem_prefill < vb.rem_prefill;
                }
                return a > b;
            }));
        return;
    }

    if (rule == "evict_earliest_prefill_deadline") {
        if (prefill_ids.empty()) return;
        out->push_back(*std::min_element(
            prefill_ids.begin(), prefill_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.prefill_deadline != vb.prefill_deadline) {
                    return va.prefill_deadline < vb.prefill_deadline;
                }
                return a < b;
            }));
        return;
    }

    if (rule == "evict_prefill_missed_deadline" ||
        rule == "evict_prefill_lateness_over_0p5") {
        const double threshold = rule == "evict_prefill_missed_deadline"
            ? eps : 0.5;
        for (int rid : prefill_ids) {
            if (req_views.at(rid).prefill_lateness > threshold) {
                out->push_back(rid);
            }
        }
        std::sort(out->begin(), out->end());
        return;
    }

    if (rule == "evict_longest_decode") {
        if (decode_ids.empty()) return;
        out->push_back(*std::max_element(
            decode_ids.begin(), decode_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.decode_processed != vb.decode_processed) {
                    return va.decode_processed < vb.decode_processed;
                }
                return a > b;
            }));
        return;
    }

    if (rule == "evict_decode_lateness_over_0p5") {
        for (int rid : decode_ids) {
            if (req_views.at(rid).total_lateness > 0.5) {
                out->push_back(rid);
            }
        }
        std::sort(out->begin(), out->end());
        return;
    }

    if (rule == "evict_prefill_highest_lateness") {
        if (prefill_ids.empty()) return;
        const int rid = *std::max_element(
            prefill_ids.begin(), prefill_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.prefill_lateness != vb.prefill_lateness) {
                    return va.prefill_lateness < vb.prefill_lateness;
                }
                return a > b;
            });
        if (req_views.at(rid).prefill_lateness > eps) out->push_back(rid);
        return;
    }

    if (rule == "evict_decode_highest_lateness") {
        if (decode_ids.empty()) return;
        const int rid = *std::max_element(
            decode_ids.begin(), decode_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.total_lateness != vb.total_lateness) {
                    return va.total_lateness < vb.total_lateness;
                }
                return a > b;
            });
        if (req_views.at(rid).total_lateness > eps) out->push_back(rid);
    }
}

struct ControllerBranchWorkspace {
    bool valid = true;
    std::vector<int> evict_ids;
    std::vector<int> canonical_evict_ids;
    std::vector<int> decode_ids;
    int total_prefill = 0;
    std::uint64_t canonical_fingerprint_base = 0;
    std::vector<std::vector<int>> ordered_by_heur;
};

template <typename T>
SampledActionSet<T> resize_sampled(const SampledActionSet<T>& in, int action_space_size) {
    const int n = std::max(1, action_space_size);
    SampledActionSet<T> out;
    out.actions.resize(static_cast<std::size_t>(n));
    out.mask.assign(static_cast<std::size_t>(n), 0u);

    const int m = std::min<int>(n, static_cast<int>(in.actions.size()));
    for (int i = 0; i < m; ++i) {
        out.actions[static_cast<std::size_t>(i)] = in.actions[static_cast<std::size_t>(i)];
        out.mask[static_cast<std::size_t>(i)] =
            (i < static_cast<int>(in.mask.size())) ? in.mask[static_cast<std::size_t>(i)] : uint8_t{0};
    }

    if (m == 0) {
        out.mask[0] = 1u;
    }
    return out;
}

}  // namespace

int adversary_action_space_size(const AdversarySamplerConfig& cfg) {
    const int n_stop = std::max(1, int(cfg.stop_rule_names.size()));
    const int n_templates = std::max(1, int(cfg.allowed_prefill_tokens.size()));
    const int n_counts = std::max(0, int(cfg.max_launch_count_per_tick));
    return n_stop + (n_counts * n_templates * n_stop);
}

int controller_action_space_size(const ControllerSamplerConfig& cfg) {
    return std::max(1, int(cfg.eviction_rule_names.size())) *
           std::max(1, int(cfg.prefill_budget_options.size())) *
           std::max(1, int(cfg.ordering_heuristics.size()));
}

SampledActionSet<AdversaryAction> sample_adversary_actions_gv2(
    const SimState& state,
    const AdversarySamplerConfig& cfg,
    double decision_tick,
    const std::unordered_set<int>& forbidden_stop_ids,
    bool compact_valid_only,
    bool compact_request_materialization) {
    const auto& req_views = build_req_views(state);

    std::unordered_map<std::string, std::vector<int>> stop_ids_by_rule;
    stop_ids_by_rule.reserve(cfg.stop_rule_names.size());
    for (const auto& rule : cfg.stop_rule_names) {
        stop_ids_by_rule[rule] = stop_ids_for_rule(rule, req_views, forbidden_stop_ids);
    }

    const int n = adversary_action_space_size(cfg);
    SampledActionSet<AdversaryAction> out;
    if (compact_valid_only) {
        const std::size_t initial_capacity = static_cast<std::size_t>(
            std::min(n, 64));
        out.actions.reserve(initial_capacity);
        out.mask.reserve(initial_capacity);
        out.original_indices.reserve(initial_capacity);
    } else {
        out.actions.resize(static_cast<std::size_t>(n));
        out.mask.assign(static_cast<std::size_t>(n), 0u);
    }
    const auto store_action = [&](int action_index, AdversaryAction action, bool valid) {
        if (compact_valid_only) {
            if (!valid) return;
            out.actions.push_back(std::move(action));
            out.mask.push_back(1u);
            out.original_indices.push_back(action_index);
            return;
        }
        out.actions[static_cast<std::size_t>(action_index)] = std::move(action);
        out.mask[static_cast<std::size_t>(action_index)] = valid ? uint8_t{1} : uint8_t{0};
    };

    const double eps = 1e-9;
    const bool strict_pre_tick = (double(state.sim_time) + eps) < double(decision_tick);

    const auto usage = window_usage(
        state.stats,
        double(decision_tick),
        std::max(0.0, cfg.launch_window_sec),
        eps);
    const int used_count = usage.first;
    const int used_prefill = usage.second;

    const int req_cap = std::max(0, cfg.max_requests_per_launch_window);
    const int prefill_cap = std::max(0, cfg.prefill_window_cap_tokens);
    const int decode_tokens_per_new_req = std::max(1, cfg.max_decode_tokens_per_request);

    int idx = 0;

    // launch_count == 0 branch
    for (const auto& stop_rule : cfg.stop_rule_names) {
        const auto it = stop_ids_by_rule.find(stop_rule);
        const std::vector<int> stop_ids = (it != stop_ids_by_rule.end()) ? it->second : std::vector<int>{};

        bool valid = true;
        if (strict_pre_tick) {
            valid = (stop_rule == "stop_none");
        }
        if (cfg.strict_masking && stop_rule != "stop_none" && stop_ids.empty()) {
            valid = false;
        }

        if (compact_valid_only && !valid) {
            ++idx;
            continue;
        }
        AdversaryAction a;
        a.requests.clear();
        a.stop_decode_ids = stop_ids;
        a.valid = valid;

        store_action(idx, std::move(a), valid);
        ++idx;
    }

    // launch_count >= 1
    for (int launch_count = 1; launch_count <= std::max(0, cfg.max_launch_count_per_tick); ++launch_count) {
        for (int prefill_tokens : cfg.allowed_prefill_tokens) {
            const int new_prefill_total = launch_count * std::max(0, prefill_tokens);
            bool launch_valid = !strict_pre_tick;
            if (used_count + launch_count > req_cap) launch_valid = false;
            if (used_prefill + new_prefill_total > prefill_cap) launch_valid = false;

            for (const auto& stop_rule : cfg.stop_rule_names) {
                const auto it = stop_ids_by_rule.find(stop_rule);
                const std::vector<int> stop_ids = (it != stop_ids_by_rule.end()) ? it->second : std::vector<int>{};

                bool valid = launch_valid;
                if (cfg.strict_masking && stop_rule != "stop_none" && stop_ids.empty()) {
                    valid = false;
                }

                if (compact_valid_only && !valid) {
                    ++idx;
                    continue;
                }
                AdversaryAction a;
                if (valid) {
                    const double prefill_slo =
                        prefill_slo_for_tokens(cfg, prefill_tokens);
                    const double decode_slo =
                        std::max(0.0, cfg.default_decode_slo_time);
                    if (compact_request_materialization) {
                        a.compact_request_count = launch_count;
                        a.compact_prefill_tokens = std::max(1, prefill_tokens);
                        a.compact_decode_tokens = decode_tokens_per_new_req;
                        a.compact_prefill_slo = prefill_slo;
                        a.compact_decode_slo = decode_slo;
                        a.compact_requests = true;
                    } else {
                        a.requests.reserve(static_cast<std::size_t>(launch_count));
                        for (int i = 0; i < launch_count; ++i) {
                            AdversaryRequestSpec spec;
                            spec.prefill_tokens = std::max(1, prefill_tokens);
                            spec.decode_tokens = decode_tokens_per_new_req;
                            spec.prefill_slo = prefill_slo;
                            spec.decode_slo = decode_slo;
                            a.requests.push_back(spec);
                        }
                    }
                }
                a.stop_decode_ids = (valid || stop_rule != "stop_none") ? stop_ids : std::vector<int>{};
                a.valid = valid;

                store_action(idx, std::move(a), valid);
                ++idx;
            }
        }
    }

    assert(idx == n);
    return out;
}

template <typename Action>
SampledActionSet<Action> sample_controller_actions_impl(
    const SimState& state,
    const ControllerSamplerConfig& cfg,
    int decode_credit_balance,
    bool compact_valid_only,
    bool canonical_compact_only) {
    assert(!canonical_compact_only || compact_valid_only);
    const auto& req_views = build_req_views(state);

    const int n = controller_action_space_size(cfg);
    SampledActionSet<Action> out;
    if (compact_valid_only) {
        const std::size_t initial_capacity = static_cast<std::size_t>(
            std::min(n, 64));
        out.actions.reserve(initial_capacity);
        out.mask.reserve(initial_capacity);
        out.original_indices.reserve(initial_capacity);
    } else {
        out.actions.resize(static_cast<std::size_t>(n));
        out.mask.assign(static_cast<std::size_t>(n), 0u);
    }
    const auto store_action = [&](int action_index, Action action, bool valid) {
        if (compact_valid_only) {
            if (!valid) return;
            out.actions.push_back(std::move(action));
            out.mask.push_back(1u);
            out.original_indices.push_back(action_index);
            return;
        }
        out.actions[static_cast<std::size_t>(action_index)] = std::move(action);
        out.mask[static_cast<std::size_t>(action_index)] = valid ? uint8_t{1} : uint8_t{0};
    };

    if (cfg.eviction_rule_names.empty() || cfg.prefill_budget_options.empty() || cfg.ordering_heuristics.empty()) {
        if (compact_valid_only || !out.actions.empty()) {
            Action noop;
            noop.valid = true;
            noop.token_budget = 0;
            if constexpr (std::is_same_v<Action, ControllerAction>) {
                noop.heuristic.clear();
                noop.strategy = "GV2|evict_none";
            }
            noop.mapping = {0, 0, 0};
            noop.has_mapping = true;
            store_action(0, std::move(noop), true);
        }
        return out;
    }

    const std::string canonical_heuristic = cfg.ordering_heuristics.front();

    // No-request fast path.
    if (req_views.empty()) {
        Action noop;
        noop.valid = true;
        noop.token_budget = 0;
        if constexpr (std::is_same_v<Action, ControllerAction>) {
            noop.strategy = "GV2|evict_none";
        }
        noop.mapping = {0, 0, 0};
        noop.has_mapping = true;

        store_action(0, std::move(noop), true);
        return out;
    }

    thread_local std::vector<int> active_prefill_ids;
    thread_local std::vector<int> active_decode_ids;
    active_prefill_ids.clear();
    active_decode_ids.clear();
    active_prefill_ids.reserve(req_views.size());
    active_decode_ids.reserve(req_views.size());
    for (const ReqView& request : req_views.values) {
        if (!request.prefill_done && request.rem_prefill > 0) {
            active_prefill_ids.push_back(request.rid);
        } else if (request.prefill_done && request.rem_decode > 0) {
            active_decode_ids.push_back(request.rid);
        }
    }
    std::sort(active_prefill_ids.begin(), active_prefill_ids.end());
    std::sort(active_decode_ids.begin(), active_decode_ids.end());

    thread_local std::vector<std::vector<int>> active_ordered_by_heur;
    active_ordered_by_heur.resize(cfg.ordering_heuristics.size());
    for (std::size_t heuristic_index = 0;
         heuristic_index < cfg.ordering_heuristics.size();
         ++heuristic_index) {
        const auto& heuristic =
            cfg.ordering_heuristics[heuristic_index];
        auto& ordered = active_ordered_by_heur[heuristic_index];
        ordered = active_prefill_ids;
        if (heuristic == "SJF") {
            std::sort(
                ordered.begin(),
                ordered.end(),
                [&](int left, int right) {
                    const int left_remaining = req_views.at(left).rem_prefill;
                    const int right_remaining = req_views.at(right).rem_prefill;
                    if (left_remaining != right_remaining) {
                        return left_remaining < right_remaining;
                    }
                    return left < right;
                });
        } else if (heuristic == "EDF") {
            std::sort(
                ordered.begin(),
                ordered.end(),
                [&](int left, int right) {
                    const double left_deadline =
                        req_views.at(left).prefill_deadline;
                    const double right_deadline =
                        req_views.at(right).prefill_deadline;
                    if (left_deadline != right_deadline) {
                        return left_deadline < right_deadline;
                    }
                    return left < right;
                });
        } else if (heuristic == "LST") {
            std::sort(
                ordered.begin(),
                ordered.end(),
                [&](int left, int right) {
                    const ReqView& left_request = req_views.at(left);
                    const ReqView& right_request = req_views.at(right);
                    const double left_slack =
                        (left_request.prefill_slo -
                         std::max(0.0, state.sim_time - left_request.arrived_at)) -
                        prefill_eta_for_tokens(cfg, left_request.rem_prefill);
                    const double right_slack =
                        (right_request.prefill_slo -
                         std::max(0.0, state.sim_time - right_request.arrived_at)) -
                        prefill_eta_for_tokens(cfg, right_request.rem_prefill);
                    if (left_slack != right_slack) {
                        return left_slack < right_slack;
                    }
                    return left < right;
                });
        } else if (heuristic == "LJF") {
            std::sort(
                ordered.begin(),
                ordered.end(),
                [&](int left, int right) {
                    const int left_remaining = req_views.at(left).rem_prefill;
                    const int right_remaining = req_views.at(right).rem_prefill;
                    if (left_remaining != right_remaining) {
                        return left_remaining > right_remaining;
                    }
                    return left < right;
                });
        }
    }

    thread_local std::vector<ControllerBranchWorkspace> branch_cache;
    branch_cache.resize(cfg.eviction_rule_names.size());

    for (std::size_t e_idx = 0; e_idx < cfg.eviction_rule_names.size(); ++e_idx) {
        const std::string& ev_rule = cfg.eviction_rule_names[e_idx];
        ControllerBranchWorkspace& b = branch_cache[e_idx];
        b.valid = true;
        b.total_prefill = 0;
        b.canonical_evict_ids.clear();
        b.decode_ids.clear();
        b.ordered_by_heur.resize(active_ordered_by_heur.size());
        for (auto& ordered : b.ordered_by_heur) ordered.clear();
        write_eviction_targets(
            ev_rule,
            req_views,
            active_prefill_ids,
            active_decode_ids,
            cfg.eps,
            &b.evict_ids);
        b.canonical_evict_ids = b.evict_ids;
        std::sort(
            b.canonical_evict_ids.begin(),
            b.canonical_evict_ids.end());
        b.canonical_evict_ids.erase(
            std::unique(
                b.canonical_evict_ids.begin(),
                b.canonical_evict_ids.end()),
            b.canonical_evict_ids.end());

        b.valid = !(cfg.strict_masking &&
            ev_rule != "evict_none" && b.evict_ids.empty());
        if (!b.valid) {
            continue;
        }

        b.decode_ids.reserve(active_decode_ids.size());
        for (int request_id : active_prefill_ids) {
            if (std::find(
                    b.evict_ids.begin(),
                    b.evict_ids.end(),
                    request_id) != b.evict_ids.end()) {
                continue;
            }
            const int remaining = req_views.at(request_id).rem_prefill;
            b.total_prefill += remaining;
        }
        for (int request_id : active_decode_ids) {
            if (std::find(
                    b.evict_ids.begin(),
                    b.evict_ids.end(),
                    request_id) == b.evict_ids.end()) {
                b.decode_ids.push_back(request_id);
            }
        }

        const int max_decode = cfg.enforce_nonnegative_decode_credits
            ? std::max(0, decode_credit_balance)
            : static_cast<int>(b.decode_ids.size());
        if (static_cast<int>(b.decode_ids.size()) > max_decode) {
            b.decode_ids.resize(static_cast<std::size_t>(max_decode));
        }
        b.canonical_fingerprint_base = 1469598103934665603ULL;
        const auto mix_branch_fingerprint = [&](std::uint64_t value) {
            b.canonical_fingerprint_base ^= value;
            b.canonical_fingerprint_base *= 1099511628211ULL;
        };
        mix_branch_fingerprint(b.canonical_evict_ids.size());
        for (int request_id : b.canonical_evict_ids) {
            mix_branch_fingerprint(static_cast<std::uint32_t>(request_id));
        }
        mix_branch_fingerprint(b.decode_ids.size());
        for (int request_id : b.decode_ids) {
            mix_branch_fingerprint(static_cast<std::uint32_t>(request_id));
        }
        for (std::size_t heuristic_index = 0;
             heuristic_index < active_ordered_by_heur.size();
             ++heuristic_index) {
            const auto& active_ordered =
                active_ordered_by_heur[heuristic_index];
            auto& ordered = b.ordered_by_heur[heuristic_index];
            ordered.reserve(active_ordered.size());
            for (int request_id : active_ordered) {
                if (std::find(
                        b.evict_ids.begin(),
                        b.evict_ids.end(),
                        request_id) == b.evict_ids.end()) {
                    ordered.push_back(request_id);
                }
            }
        }
    }

    struct CanonicalControllerActionKey {
        InlineVector<int, 16> evicted;
        InlineVector<int, 32> decode;
        InlineVector<std::pair<int, int>, 16> prefill;
    };
    std::vector<CanonicalControllerActionKey> canonical_action_keys;
    std::vector<std::uint64_t> canonical_action_fingerprints;
    if (canonical_compact_only) {
        canonical_action_keys.reserve(static_cast<std::size_t>(n));
        canonical_action_fingerprints.reserve(static_cast<std::size_t>(n));
    }

    int idx = 0;
    for (int e_idx = 0; e_idx < static_cast<int>(cfg.eviction_rule_names.size()); ++e_idx) {
        const std::string& ev_rule = cfg.eviction_rule_names[static_cast<std::size_t>(e_idx)];
        const ControllerBranchWorkspace& b = branch_cache[static_cast<std::size_t>(e_idx)];

        for (int b_idx = 0; b_idx < static_cast<int>(cfg.prefill_budget_options.size()); ++b_idx) {
            const int budget = cfg.prefill_budget_options[static_cast<std::size_t>(b_idx)];

            for (int h_idx = 0; h_idx < static_cast<int>(cfg.ordering_heuristics.size()); ++h_idx) {
                const std::string& heur = cfg.ordering_heuristics[static_cast<std::size_t>(h_idx)];

                bool valid = b.valid;
                if (budget == 0 && heur != canonical_heuristic) valid = false;
                if (budget < 0) valid = false;
                if (budget > b.total_prefill) valid = false;
                if (b.total_prefill == 0 && budget > 0) valid = false;

                if (valid && budget == 0 && b.decode_ids.empty()) {
                    valid = false;
                }
                if (!valid) {
                    ++idx;
                    continue;
                }

                InlineVector<std::pair<int, int>, 16> prefill_items;
                if (valid && budget > 0) {
                    int remaining = budget;
                    const std::vector<int>& ordered_pref =
                        b.ordered_by_heur[static_cast<std::size_t>(h_idx)];

                    prefill_items.reserve(ordered_pref.size());
                    for (int rid : ordered_pref) {
                        if (remaining <= 0) break;
                        const int cap = std::max(
                            0,
                            req_views.at(rid).rem_prefill);
                        if (cap <= 0) continue;
                        const int alloc = std::min(cap, remaining);
                        if (alloc <= 0) continue;
                        prefill_items.emplace_back(rid, alloc);
                        remaining -= alloc;
                    }
                }

                if (canonical_compact_only) {
                    std::uint64_t fingerprint =
                        b.canonical_fingerprint_base;
                    const auto mix = [&](std::uint64_t value) {
                        fingerprint ^= value;
                        fingerprint *= 1099511628211ULL;
                    };
                    std::uint64_t prefill_sum = 0;
                    std::uint64_t prefill_xor = 0;
                    for (const auto& item : prefill_items) {
                        std::uint64_t item_hash =
                            (static_cast<std::uint64_t>(
                                static_cast<std::uint32_t>(item.first)) << 32U) |
                            static_cast<std::uint32_t>(item.second);
                        item_hash ^= item_hash >> 30U;
                        item_hash *= 0xbf58476d1ce4e5b9ULL;
                        item_hash ^= item_hash >> 27U;
                        item_hash *= 0x94d049bb133111ebULL;
                        item_hash ^= item_hash >> 31U;
                        prefill_sum += item_hash;
                        prefill_xor ^= item_hash;
                    }
                    mix(prefill_items.size());
                    mix(prefill_sum);
                    mix(prefill_xor);
                    bool duplicate = false;
                    for (std::size_t previous_index = 0;
                         previous_index < canonical_action_keys.size();
                         ++previous_index) {
                        if (canonical_action_fingerprints[previous_index] !=
                            fingerprint) {
                            continue;
                        }
                        const auto& previous =
                            canonical_action_keys[previous_index];
                        duplicate =
                            previous.evicted.size() ==
                                b.canonical_evict_ids.size() &&
                            std::equal(
                                previous.evicted.begin(),
                                previous.evicted.end(),
                                b.canonical_evict_ids.begin()) &&
                            previous.decode.size() == b.decode_ids.size() &&
                            std::equal(
                                previous.decode.begin(),
                                previous.decode.end(),
                                b.decode_ids.begin()) &&
                            previous.prefill.size() == prefill_items.size() &&
                            std::all_of(
                                prefill_items.begin(),
                                prefill_items.end(),
                                [&](const auto& item) {
                                    return std::find(
                                        previous.prefill.begin(),
                                        previous.prefill.end(),
                                        item) != previous.prefill.end();
                                });
                        if (duplicate) break;
                    }
                    if (duplicate) {
                        ++idx;
                        continue;
                    }
                    CanonicalControllerActionKey key;
                    key.evicted.assign(
                        b.canonical_evict_ids.begin(),
                        b.canonical_evict_ids.end());
                    key.decode.assign(b.decode_ids.begin(), b.decode_ids.end());
                    key.prefill = prefill_items;
                    canonical_action_keys.push_back(std::move(key));
                    canonical_action_fingerprints.push_back(fingerprint);
                }

                int total_prefill_alloc = 0;
                for (const auto& item : prefill_items) {
                    total_prefill_alloc += std::max(0, item.second);
                }

                Action action;
                action.token_budget =
                    total_prefill_alloc + static_cast<int>(b.decode_ids.size());
                action.evicted_request_ids.assign(
                    b.evict_ids.begin(), b.evict_ids.end());
                if constexpr (std::is_same_v<Action, RolloutControllerAction>) {
                    action.compact_prefill_allocations =
                        std::move(prefill_items);
                    action.compact_decode_request_ids = b.decode_ids;
                    action.compact_allocations = true;
                } else if (canonical_compact_only) {
                    action.compact_prefill_allocations =
                        std::move(prefill_items);
                    action.compact_decode_request_ids = b.decode_ids;
                    action.compact_allocations = true;
                } else {
                    action.prefill_allocations.reserve(prefill_items.size());
                    for (const auto& item : prefill_items) {
                        action.prefill_allocations.emplace(item.first, item.second);
                    }
                    action.decode_allocations.reserve(b.decode_ids.size());
                    for (int rid : b.decode_ids) {
                        action.decode_allocations.emplace(rid, 1);
                    }
                    action.token_allocations = action.decode_allocations;
                    for (const auto& kv : action.prefill_allocations) {
                        action.token_allocations[kv.first] = kv.second;
                    }
                    action.selected_request_ids.reserve(
                        action.token_allocations.size());
                    for (const auto& kv : action.token_allocations) {
                        action.selected_request_ids.push_back(kv.first);
                    }
                    std::sort(
                        action.selected_request_ids.begin(),
                        action.selected_request_ids.end());
                    action.heuristic = (budget > 0) ? heur : std::string{};
                    action.strategy = std::string("GV2|") + ev_rule;
                }
                action.mapping = {e_idx, b_idx, h_idx};
                action.has_mapping = true;
                action.valid = valid;

                store_action(idx, std::move(action), valid);
                ++idx;
            }
        }
    }

    assert(idx == n);
    return out;
}

SampledActionSet<ControllerAction> sample_controller_actions_gv2(
    const SimState& state,
    const ControllerSamplerConfig& cfg,
    int decode_credit_balance,
    bool compact_valid_only,
    bool canonical_compact_only) {
    return sample_controller_actions_impl<ControllerAction>(
        state,
        cfg,
        decode_credit_balance,
        compact_valid_only,
        canonical_compact_only);
}

SampledActionSet<RolloutControllerAction> sample_controller_rollout_actions_gv2(
    const SimState& state,
    const ControllerSamplerConfig& cfg,
    int decode_credit_balance) {
    return sample_controller_actions_impl<RolloutControllerAction>(
        state, cfg, decode_credit_balance, true, true);
}

SampledActionSet<AdversaryAction> sample_adversary_actions_simple(
    const SimState& state,
    int action_space_size) {
    AdversarySamplerConfig cfg;
    const double decision_tick = std::max(double(state.sim_time), double(state.stats.next_adv_tick));
    const auto full = sample_adversary_actions_gv2(state, cfg, decision_tick, {});
    return resize_sampled(full, action_space_size);
}

SampledActionSet<ControllerAction> sample_controller_actions_simple(
    const SimState& state,
    int action_space_size) {
    ControllerSamplerConfig cfg;
    const auto full = sample_controller_actions_gv2(
        state,
        cfg,
        int(std::max(0, state.stats.decode_credit_balance)));
    return resize_sampled(full, action_space_size);
}

}  // namespace mcts_native_gv2
