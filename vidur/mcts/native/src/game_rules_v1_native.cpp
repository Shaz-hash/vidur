#include "game_rules_native.hpp"
#include "native_sim.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace mcts_native {

namespace {

static int nonneg_i(int x) { return x < 0 ? 0 : x; }

static double send_interval(const NativeRuntimeConfig& cfg) {
    return std::max(1e-9, cfg.adversary_send_interval_sec);
}
static double window_sec(const NativeRuntimeConfig& cfg) {
    return (cfg.arrival_window_sec > 0.0) ? cfg.arrival_window_sec : send_interval(cfg);
}
static double nearest_prefill_estimate(
    int prefill_tokens,
    const std::vector<int>& profile_tokens,
    const std::vector<double>& profile_times
) {
    if (profile_tokens.empty() || profile_times.empty()) return 0.0;
    size_t best = 0;
    int best_diff = std::abs(profile_tokens[0] - prefill_tokens);
    for (size_t i = 1; i < profile_tokens.size(); ++i) {
        int d = std::abs(profile_tokens[i] - prefill_tokens);
        if (d < best_diff) { best_diff = d; best = i; }
    }
    return profile_times[best];
}

static ControllerActionSpecNative make_zero_action() {
    ControllerActionSpecNative a;
    a.token_budget = 0;
    a.valid = true;
    return a;
}

static ControllerActionSpecNative build_controller_action(
    const std::vector<int>& ordered_indices,
    int prefill_budget,
    const char* heur_name,
    const std::vector<PrefillRecord>& prefill_records,
    const std::vector<AllocationEntry>& decode_template,
    int decode_budget
) {
    ControllerActionSpecNative out;
    out.valid = true;
    out.heuristic = heur_name;
    out.strategy = "Fixed";

    out.decode_allocations = decode_template;
    out.token_allocations = decode_template;
    out.selected_request_ids.reserve(decode_template.size() + ordered_indices.size());
    for (const auto& e : decode_template) out.selected_request_ids.push_back(e.request_id);

    int remaining_budget = nonneg_i(prefill_budget);
    int used_prefill = 0;

    for (int idx : ordered_indices) {
        if (remaining_budget <= 0) break;
        const PrefillRecord& r = prefill_records[(size_t)idx];
        int cap = nonneg_i(r.rem_pref);
        if (cap <= 0) continue;

        int alloc = (cap < remaining_budget) ? cap : remaining_budget;
        if (alloc <= 0) continue;

        out.prefill_allocations.push_back({r.rid, alloc});
        out.token_allocations.push_back({r.rid, alloc});
        out.selected_request_ids.push_back(r.rid);

        remaining_budget -= alloc;
        used_prefill += alloc;
    }

    out.token_budget = decode_budget + used_prefill;
    return out;
}

static ControllerSampleOutput sample_controller_actions_v1(
    const std::vector<ControllerRequestStateNative>& request_states,
    double sim_time,
    const std::vector<int>& budgets,
    const NativeRuntimeConfig& cfg
) {
    ControllerSampleOutput out;
    const int num_heur = 4;
    const int num_budgets = (int)budgets.size();
    const int num_actions = num_heur * num_budgets;

    if (num_actions <= 0) {
        out.actions.resize(1);
        out.mask.resize(1, 0);
        out.actions[0] = make_zero_action();
        out.mask[0] = 1;
        return out;
    }

    out.actions.resize((size_t)num_actions);
    out.mask.assign((size_t)num_actions, 0);

    if (request_states.empty()) {
        out.actions[0] = make_zero_action();
        out.mask[0] = 1;
        return out;
    }

    std::vector<PrefillRecord> prefill_records;
    prefill_records.reserve(request_states.size());

    std::vector<int> decode_candidates;
    decode_candidates.reserve(request_states.size());

    std::vector<AllocationEntry> decode_template;
    decode_template.reserve(request_states.size());

    int total_remaining_prefill = 0;

    for (const auto& rs : request_states) {
        const int rid = rs.request_id;
        if (!rs.prefill_done) {
            int rem_pref = nonneg_i(rs.remaining_prefill);
            if (rem_pref > 0) {
                total_remaining_prefill += rem_pref;

                const double edf_key = rs.arrived_at + rs.prefill_slo;
                const double remaining_slo = rs.prefill_slo - std::max(0.0, sim_time - rs.arrived_at);
                const double est = nearest_prefill_estimate(
                    rem_pref, cfg.prefill_profile_tokens, cfg.prefill_profile_times
                );
                const double lst_key = remaining_slo - est;

                prefill_records.push_back(PrefillRecord{rid, rem_pref, edf_key, lst_key});
            }
        } else {
            if (nonneg_i(rs.remaining_decode) > 0) {
                decode_candidates.push_back(rid);
                decode_template.push_back({rid, 1});
            }
        }
    }

    if (total_remaining_prefill == 0) {
        ControllerActionSpecNative a;
        a.valid = true;
        a.heuristic = "SJF";
        a.strategy = "Fixed";
        a.decode_allocations = decode_template;
        a.token_allocations = decode_template;
        a.selected_request_ids = decode_candidates;
        a.token_budget = (int)decode_template.size();

        out.actions[0] = std::move(a);
        out.mask[0] = 1;
        return out;
    }

    const int decode_budget = (int)decode_candidates.size();
    const int n = (int)prefill_records.size();

    std::vector<int> ordered_sjf((size_t)n);
    std::iota(ordered_sjf.begin(), ordered_sjf.end(), 0);

    auto ordered_edf = ordered_sjf;
    auto ordered_lst = ordered_sjf;

    std::stable_sort(ordered_sjf.begin(), ordered_sjf.end(), [&](int a, int b) {
        const auto& ra = prefill_records[(size_t)a];
        const auto& rb = prefill_records[(size_t)b];
        if (ra.rem_pref != rb.rem_pref) return ra.rem_pref < rb.rem_pref;
        return ra.rid < rb.rid;
    });
    std::stable_sort(ordered_edf.begin(), ordered_edf.end(), [&](int a, int b) {
        const auto& ra = prefill_records[(size_t)a];
        const auto& rb = prefill_records[(size_t)b];
        if (ra.edf_key != rb.edf_key) return ra.edf_key < rb.edf_key;
        return ra.rid < rb.rid;
    });
    std::stable_sort(ordered_lst.begin(), ordered_lst.end(), [&](int a, int b) {
        const auto& ra = prefill_records[(size_t)a];
        const auto& rb = prefill_records[(size_t)b];
        if (ra.lst_key != rb.lst_key) return ra.lst_key < rb.lst_key;
        return ra.rid < rb.rid;
    });

    auto ordered_ljf = ordered_sjf;
    std::reverse(ordered_ljf.begin(), ordered_ljf.end());

    const std::vector<int>* orders[4] = {&ordered_sjf, &ordered_edf, &ordered_lst, &ordered_ljf};
    const char* names[4] = {"SJF", "EDF", "LST", "LJF"};

    bool any_valid = false;
    for (int b_idx = 0; b_idx < num_budgets; ++b_idx) {
        const int budget = nonneg_i(budgets[(size_t)b_idx]);
        const bool budget_valid = (total_remaining_prefill > 0) ? (budget <= total_remaining_prefill) : (b_idx == 0);

        for (int h_idx = 0; h_idx < 4; ++h_idx) {
            const int idx = b_idx * 4 + h_idx;
            if (!budget_valid) {
                out.mask[(size_t)idx] = 0;
                continue;
            }
            out.actions[(size_t)idx] = build_controller_action(
                *orders[h_idx], budget, names[h_idx], prefill_records, decode_template, decode_budget
            );
            out.mask[(size_t)idx] = 1;
            any_valid = true;
        }
    }

    if (!any_valid) {
        out.actions[0] = make_zero_action();
        out.mask[0] = 1;
    }
    return out;
}

} // namespace

NativeGameVersionId resolve_game_version_id(const std::string& name) {
    std::string s = name;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "game_version_1" || s == "v1" || s == "version_1" || s == "1") {
        return NativeGameVersionId::V1;
    }
    return NativeGameVersionId::V1; // safe default; optionally hard-fail
}

bool can_adversary_send(double sim_time, double last_prefill_batch_time, const NativeRuntimeConfig& cfg) {
    if (last_prefill_batch_time < 0.0) return true;
    return sim_time >= last_prefill_batch_time + send_interval(cfg) - 1e-9;
}

double compute_adversary_arrival_time(double sim_time, double last_prefill_batch_time, bool has_requests, const NativeRuntimeConfig& cfg) {
    const double first = cfg.first_arrival_floor ? std::floor(sim_time) : sim_time;
    if (!has_requests) return first;
    if (last_prefill_batch_time < 0.0) return first;
    return last_prefill_batch_time + send_interval(cfg);
}

double next_last_prefill_batch_time(double sim_time, double last_prefill_batch_time, bool has_requests, const NativeRuntimeConfig& cfg) {
    if (!has_requests) return last_prefill_batch_time;
    return compute_adversary_arrival_time(sim_time, last_prefill_batch_time, true, cfg);
}

double next_adversary_release_time(double last_prefill_batch_time, const NativeRuntimeConfig& cfg) {
    if (last_prefill_batch_time < 0.0) return -1.0;
    return last_prefill_batch_time + send_interval(cfg);
}

double arrival_window_start(double sim_time, const NativeRuntimeConfig& cfg) {
    return sim_time - window_sec(cfg);
}

AdversarySampleOutput sample_adversary_actions_for_rules(const NativeSimState& state, const NativeRuntimeConfig& cfg) {
    AdversarySampleOutput out;
    const int n = std::max(1, cfg.adversary_num_actions);
    out.actions.resize((size_t)n);
    out.mask.assign((size_t)n, 0);

    if (!can_adversary_send(state.sim_time, state.stats.last_prefill_batch_time, cfg)) {
        out.actions[0].valid = true;
        out.mask[0] = 1;
        return out;
    }

    const double prefill_slo = nearest_prefill_estimate(
        cfg.max_request_tokens, cfg.prefill_profile_tokens, cfg.prefill_profile_times
    );

    for (int i = 0; i < n; ++i) {
        AdversaryActionSpecNative a;
        a.valid = true;
        a.requests.reserve((size_t)(i + 1));
        for (int k = 0; k < i + 1; ++k) {
            AdversaryRequestSpecNative r;
            r.prefill_tokens = cfg.max_request_tokens;
            r.decode_tokens = cfg.adversary_fixed_decode_tokens;
            r.prefill_slo = prefill_slo;
            r.decode_slo = cfg.default_decode_slo;
            a.requests.push_back(r);
        }
        out.actions[(size_t)i] = std::move(a);
        out.mask[(size_t)i] = 1;
    }
    return out;
}

ControllerSampleOutput sample_controller_actions_for_rules(
    const NativeSimState& state,
    const NativeRuntimeConfig& cfg
) {
    std::vector<int> budgets;
    budgets.reserve(6);
    const int step = std::max(1, cfg.interval_request_size);
    for (int i = 1; i <= 6; ++i) budgets.push_back(step * i);

    const auto view = NativeSim::controller_view_from_state(state);

    switch (static_cast<NativeGameVersionId>(cfg.game_version_id)) {
        case NativeGameVersionId::V1:
        default:
            return sample_controller_actions_v1(view, state.sim_time, budgets, cfg);
    }
}


} // namespace mcts_native
