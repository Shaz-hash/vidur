#include "gv2_player_sample_actions.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>

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

std::unordered_map<int, ReqView> build_req_views(const SimState& state) {
    std::unordered_map<int, ReqView> out;
    out.reserve(state.requests.size());

    const double sim_time = double(state.sim_time);
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

        const double fallback_pref_deadline = v.arrived_at + std::max(0.0, v.prefill_slo);
        v.prefill_deadline = (r.prefill_deadline > 0.0) ? r.prefill_deadline : fallback_pref_deadline;

        const auto it_pref = state.stats.per_request_prefill_lateness_by_id.find(r.request_id);
        const auto it_dec = state.stats.per_request_decode_lateness_by_id.find(r.request_id);
        if (it_pref != state.stats.per_request_prefill_lateness_by_id.end()) {
            v.prefill_lateness = double(it_pref->second);
        } else if (prefill_done) {
            v.prefill_lateness = std::max(0.0, double(r.prefill_lateness));
        } else {
            v.prefill_lateness = std::max(0.0, sim_time - v.prefill_deadline);
        }
        v.decode_lateness = (it_dec != state.stats.per_request_decode_lateness_by_id.end())
            ? double(it_dec->second)
            : 0.0;
        v.total_lateness = std::max(0.0, v.prefill_lateness) + std::max(0.0, v.decode_lateness);

        out[v.rid] = v;
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

std::vector<int> decode_active_ids(const std::unordered_map<int, ReqView>& req_views) {
    std::vector<int> out;
    out.reserve(req_views.size());
    for (const auto& kv : req_views) {
        const ReqView& rv = kv.second;
        if (rv.prefill_done && rv.rem_decode > 0) out.push_back(rv.rid);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<int> stop_ids_for_rule(
    const std::string& rule,
    const std::unordered_map<int, ReqView>& req_views,
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

std::vector<int> eviction_targets(
    const std::string& rule,
    const std::unordered_map<int, ReqView>& req_views,
    const std::unordered_set<int>& violated_ids,
    double eps) {
    std::vector<int> prefill_ids;
    std::vector<int> decode_ids;
    prefill_ids.reserve(req_views.size());
    decode_ids.reserve(req_views.size());

    for (const auto& kv : req_views) {
        const ReqView& rv = kv.second;
        if (!rv.prefill_done && rv.rem_prefill > 0) prefill_ids.push_back(rv.rid);
        else if (rv.prefill_done && rv.rem_decode > 0) decode_ids.push_back(rv.rid);
    }

    std::sort(prefill_ids.begin(), prefill_ids.end());
    std::sort(decode_ids.begin(), decode_ids.end());

    if (rule == "evict_none") return {};

    if (rule == "evict_largest_prefill") {
        if (prefill_ids.empty()) return {};
        const int rid = *std::max_element(
            prefill_ids.begin(),
            prefill_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.rem_prefill != vb.rem_prefill) return va.rem_prefill < vb.rem_prefill;
                return a > b;
            });
        return {rid};
    }

    if (rule == "evict_earliest_prefill_deadline") {
        if (prefill_ids.empty()) return {};
        const int rid = *std::min_element(
            prefill_ids.begin(),
            prefill_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.prefill_deadline != vb.prefill_deadline) return va.prefill_deadline < vb.prefill_deadline;
                return a < b;
            });
        return {rid};
    }

    if (rule == "evict_prefill_missed_deadline") {
        std::vector<int> out;
        for (int rid : prefill_ids) {
            if (req_views.at(rid).prefill_lateness > eps) out.push_back(rid);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    if (rule == "evict_prefill_lateness_over_0p5") {
        std::vector<int> out;
        for (int rid : prefill_ids) {
            if (req_views.at(rid).prefill_lateness > 0.5) out.push_back(rid);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    if (rule == "evict_longest_decode") {
        if (decode_ids.empty()) return {};
        const int rid = *std::max_element(
            decode_ids.begin(),
            decode_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.decode_processed != vb.decode_processed) return va.decode_processed < vb.decode_processed;
                return a > b;
            });
        return {rid};
    }

    if (rule == "evict_decode_lateness_over_0p5") {
        std::vector<int> out;
        for (int rid : decode_ids) {
            if (req_views.at(rid).total_lateness > 0.5) out.push_back(rid);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    if (rule == "evict_prefill_highest_lateness") {
        if (prefill_ids.empty()) return {};
        const int rid = *std::max_element(
            prefill_ids.begin(),
            prefill_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.prefill_lateness != vb.prefill_lateness) return va.prefill_lateness < vb.prefill_lateness;
                return a > b;
            });
        return req_views.at(rid).prefill_lateness > eps ? std::vector<int>{rid} : std::vector<int>{};
    }

    if (rule == "evict_decode_highest_lateness") {
        if (decode_ids.empty()) return {};
        const int rid = *std::max_element(
            decode_ids.begin(),
            decode_ids.end(),
            [&](int a, int b) {
                const ReqView& va = req_views.at(a);
                const ReqView& vb = req_views.at(b);
                if (va.total_lateness != vb.total_lateness) return va.total_lateness < vb.total_lateness;
                return a > b;
            });
        return req_views.at(rid).total_lateness > eps ? std::vector<int>{rid} : std::vector<int>{};
    }

    (void)violated_ids;
    return {};
}

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
    const std::unordered_set<int>& forbidden_stop_ids) {
    const auto req_views = build_req_views(state);

    std::unordered_map<std::string, std::vector<int>> stop_ids_by_rule;
    stop_ids_by_rule.reserve(cfg.stop_rule_names.size());
    for (const auto& rule : cfg.stop_rule_names) {
        stop_ids_by_rule[rule] = stop_ids_for_rule(rule, req_views, forbidden_stop_ids);
    }

    const int n = adversary_action_space_size(cfg);
    SampledActionSet<AdversaryAction> out;
    out.actions.resize(static_cast<std::size_t>(n));
    out.mask.assign(static_cast<std::size_t>(n), 0u);

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

        AdversaryAction a;
        a.requests.clear();
        a.stop_decode_ids = stop_ids;
        a.valid = valid;

        out.actions[static_cast<std::size_t>(idx)] = std::move(a);
        out.mask[static_cast<std::size_t>(idx)] = valid ? uint8_t{1} : uint8_t{0};
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

                AdversaryAction a;
                if (valid) {
                    const double prefill_slo = prefill_slo_for_tokens(cfg, prefill_tokens);
                    a.requests.reserve(static_cast<std::size_t>(launch_count));
                    for (int i = 0; i < launch_count; ++i) {
                        AdversaryRequestSpec spec;
                        spec.prefill_tokens = std::max(1, prefill_tokens);
                        spec.decode_tokens = decode_tokens_per_new_req;
                        spec.prefill_slo = prefill_slo;
                        spec.decode_slo = std::max(0.0, cfg.default_decode_slo_time);
                        a.requests.push_back(spec);
                    }
                }
                a.stop_decode_ids = (valid || stop_rule != "stop_none") ? stop_ids : std::vector<int>{};
                a.valid = valid;

                out.actions[static_cast<std::size_t>(idx)] = std::move(a);
                out.mask[static_cast<std::size_t>(idx)] = valid ? uint8_t{1} : uint8_t{0};
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
    int decode_credit_balance) {
    const auto req_views = build_req_views(state);

    const int n = controller_action_space_size(cfg);
    SampledActionSet<ControllerAction> out;
    out.actions.resize(static_cast<std::size_t>(n));
    out.mask.assign(static_cast<std::size_t>(n), 0u);

    if (cfg.eviction_rule_names.empty() || cfg.prefill_budget_options.empty() || cfg.ordering_heuristics.empty()) {
        if (!out.actions.empty()) {
            ControllerAction noop;
            noop.valid = true;
            noop.token_budget = 0;
            noop.heuristic.clear();
            noop.strategy = "GV2|evict_none";
            noop.mapping = {0, 0, 0};
            noop.has_mapping = true;
            out.actions[0] = std::move(noop);
            out.mask[0] = 1u;
        }
        return out;
    }

    const std::string canonical_heuristic = cfg.ordering_heuristics.front();

    // No-request fast path.
    if (req_views.empty()) {
        ControllerAction noop;
        noop.valid = true;
        noop.token_budget = 0;
        noop.strategy = "GV2|evict_none";
        noop.mapping = {0, 0, 0};
        noop.has_mapping = true;

        out.actions[0] = std::move(noop);
        out.mask[0] = 1u;
        return out;
    }

    struct BranchCache {
        bool valid = true;
        std::vector<int> evict_ids;
        std::vector<int> decode_ids;
        std::vector<int> prefill_ids;
        std::unordered_map<int, int> rem_pref_by_id;
        int total_prefill = 0;
        std::unordered_map<std::string, std::vector<int>> ordered_by_heur;
    };

    std::unordered_set<int> violated_ids(
        state.stats.violated_request_ids.begin(),
        state.stats.violated_request_ids.end());

    std::unordered_map<std::string, BranchCache> branch_cache;
    branch_cache.reserve(cfg.eviction_rule_names.size());

    for (const auto& ev_rule : cfg.eviction_rule_names) {
        BranchCache b;
        b.evict_ids = eviction_targets(ev_rule, req_views, violated_ids, cfg.eps);

        std::unordered_set<int> evict_set(b.evict_ids.begin(), b.evict_ids.end());

        for (const auto& kv : req_views) {
            const ReqView& rv = kv.second;
            if (evict_set.find(rv.rid) != evict_set.end()) continue;

            if (!rv.prefill_done && rv.rem_prefill > 0) {
                b.prefill_ids.push_back(rv.rid);
                b.rem_pref_by_id[rv.rid] = rv.rem_prefill;
                b.total_prefill += rv.rem_prefill;
            } else if (rv.prefill_done && rv.rem_decode > 0) {
                b.decode_ids.push_back(rv.rid);
            }
        }

        std::sort(b.prefill_ids.begin(), b.prefill_ids.end());
        std::sort(b.decode_ids.begin(), b.decode_ids.end());

        b.valid = true;
        if (cfg.strict_masking && ev_rule != "evict_none" && b.evict_ids.empty()) {
            b.valid = false;
        }

        for (const auto& heur : cfg.ordering_heuristics) {
            std::vector<int> ordered = b.prefill_ids;

            if (heur == "SJF") {
                std::sort(
                    ordered.begin(),
                    ordered.end(),
                    [&](int a, int c) {
                        const int ra = b.rem_pref_by_id.at(a);
                        const int rc = b.rem_pref_by_id.at(c);
                        if (ra != rc) return ra < rc;
                        return a < c;
                    });
            } else if (heur == "EDF") {
                std::sort(
                    ordered.begin(),
                    ordered.end(),
                    [&](int a, int c) {
                        const double da = req_views.at(a).prefill_deadline;
                        const double dc = req_views.at(c).prefill_deadline;
                        if (da != dc) return da < dc;
                        return a < c;
                    });
            } else if (heur == "LST") {
                std::sort(
                    ordered.begin(),
                    ordered.end(),
                    [&](int a, int c) {
                        const ReqView& va = req_views.at(a);
                        const ReqView& vc = req_views.at(c);
                        const double eta_a = prefill_eta_for_tokens(cfg, va.rem_prefill);
                        const double eta_c = prefill_eta_for_tokens(cfg, vc.rem_prefill);
                        const double slack_a = (va.prefill_slo - std::max(0.0, state.sim_time - va.arrived_at)) - eta_a;
                        const double slack_c = (vc.prefill_slo - std::max(0.0, state.sim_time - vc.arrived_at)) - eta_c;
                        if (slack_a != slack_c) return slack_a < slack_c;
                        return a < c;
                    });
            } else if (heur == "LJF") {
                std::sort(
                    ordered.begin(),
                    ordered.end(),
                    [&](int a, int c) {
                        const int ra = b.rem_pref_by_id.at(a);
                        const int rc = b.rem_pref_by_id.at(c);
                        if (ra != rc) return ra > rc;
                        return a < c;
                    });
            }

            b.ordered_by_heur[heur] = std::move(ordered);
        }

        branch_cache[ev_rule] = std::move(b);
    }

    int idx = 0;
    for (int e_idx = 0; e_idx < static_cast<int>(cfg.eviction_rule_names.size()); ++e_idx) {
        const std::string& ev_rule = cfg.eviction_rule_names[static_cast<std::size_t>(e_idx)];
        const BranchCache& b = branch_cache.at(ev_rule);

        for (int b_idx = 0; b_idx < static_cast<int>(cfg.prefill_budget_options.size()); ++b_idx) {
            const int budget = cfg.prefill_budget_options[static_cast<std::size_t>(b_idx)];

            for (int h_idx = 0; h_idx < static_cast<int>(cfg.ordering_heuristics.size()); ++h_idx) {
                const std::string& heur = cfg.ordering_heuristics[static_cast<std::size_t>(h_idx)];

                bool valid = b.valid;
                if (budget == 0 && heur != canonical_heuristic) valid = false;
                if (budget < 0) valid = false;
                if (budget > b.total_prefill) valid = false;
                if (b.total_prefill == 0 && budget > 0) valid = false;

                const int max_decode = cfg.enforce_nonnegative_decode_credits
                    ? std::max(0, decode_credit_balance)
                    : static_cast<int>(b.decode_ids.size());
                std::vector<int> decode_ids_limited = b.decode_ids;
                if (static_cast<int>(decode_ids_limited.size()) > max_decode) {
                    decode_ids_limited.resize(static_cast<std::size_t>(max_decode));
                }

                std::unordered_map<int, int> decode_alloc;
                decode_alloc.reserve(decode_ids_limited.size());
                for (int rid : decode_ids_limited) {
                    decode_alloc[rid] = 1;
                }

                std::unordered_map<int, int> prefill_alloc;
                if (valid && budget > 0) {
                    int remaining = budget;
                    const auto it = b.ordered_by_heur.find(heur);
                    const std::vector<int>& ordered_pref = (it != b.ordered_by_heur.end())
                        ? it->second
                        : b.prefill_ids;

                    for (int rid : ordered_pref) {
                        if (remaining <= 0) break;
                        const auto rem_it = b.rem_pref_by_id.find(rid);
                        if (rem_it == b.rem_pref_by_id.end()) continue;
                        const int cap = std::max(0, rem_it->second);
                        if (cap <= 0) continue;
                        const int alloc = std::min(cap, remaining);
                        if (alloc <= 0) continue;
                        prefill_alloc[rid] = alloc;
                        remaining -= alloc;
                    }
                }

                std::unordered_map<int, int> token_alloc = decode_alloc;
                for (const auto& kv : prefill_alloc) {
                    token_alloc[kv.first] = kv.second;
                }

                if (valid && token_alloc.empty()) {
                    valid = false;
                }

                std::vector<int> selected_ids;
                selected_ids.reserve(token_alloc.size());
                for (const auto& kv : token_alloc) selected_ids.push_back(kv.first);
                std::sort(selected_ids.begin(), selected_ids.end());

                ControllerAction action;
                action.token_budget = 0;
                for (const auto& kv : token_alloc) action.token_budget += std::max(0, kv.second);
                action.selected_request_ids = std::move(selected_ids);
                action.evicted_request_ids = b.evict_ids;
                action.token_allocations = std::move(token_alloc);
                action.prefill_allocations = std::move(prefill_alloc);
                action.decode_allocations = std::move(decode_alloc);
                action.heuristic = (budget > 0) ? heur : std::string{};
                action.strategy = std::string("GV2|") + ev_rule;
                action.mapping = {e_idx, b_idx, h_idx};
                action.has_mapping = true;
                action.valid = valid;

                out.actions[static_cast<std::size_t>(idx)] = std::move(action);
                out.mask[static_cast<std::size_t>(idx)] = valid ? uint8_t{1} : uint8_t{0};
                ++idx;
            }
        }
    }

    assert(idx == n);
    return out;
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
