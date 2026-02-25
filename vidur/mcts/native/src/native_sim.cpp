#include "native_sim.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcts_native {

namespace {

static int remaining_prefill(const NativeRequestState& r) {
    return std::max(0, r.num_prefill_tokens - r.num_processed_prefill_tokens);
}

static int remaining_decode(const NativeRequestState& r) {
    return std::max(0, r.num_decode_tokens - r.num_processed_decode_tokens);
}

static bool is_pending(const NativeRequestState& r) {
    return (!r.completed) && ((remaining_prefill(r) > 0 && !r.prefill_done) || (r.prefill_done && remaining_decode(r) > 0));
}

static NativeRequestState* find_req_mut(NativeSimState& s, int rid) {
    for (auto& r : s.requests) {
        if (r.request_id == rid) return &r;
    }
    return nullptr;
}

static const NativeRequestState* find_req(const NativeSimState& s, int rid) {
    for (const auto& r : s.requests) {
        if (r.request_id == rid) return &r;
    }
    return nullptr;
}

static std::unordered_map<int, int> alloc_vec_to_map(const std::vector<AllocationEntry>& v) {
    std::unordered_map<int, int> out;
    out.reserve(v.size());
    for (const auto& e : v) {
        out[e.request_id] = std::max(0, e.tokens);
    }
    return out;
}

static void maybe_fast_forward_decode_only_to_next_adv_second(NativeSimState& state) {
    const double last = state.stats.last_prefill_batch_time;
    if (last < 0.0) return;

    const double target = last + 1.0;
    if (state.sim_time >= target - 1e-9) return;

    bool any_decode = false;
    for (const auto& r : state.requests) {
        if (r.completed) continue;
        if (remaining_prefill(r) > 0 && !r.prefill_done) return;  // cannot fast-forward with prefill pending
        if (r.prefill_done && remaining_decode(r) > 0) any_decode = true;
    }
    if (!any_decode) return;

    state.sim_time = target;
    for (const auto& r : state.requests) {
        if (r.completed || !r.prefill_done || remaining_decode(r) <= 0 || r.decode_slo < 0.0) continue;
        state.stats.decode_next_deadline_by_id[r.request_id] = target + r.decode_slo;
    }
}

static void update_requests_and_stats(
    NativeSimState& state,
    const std::unordered_map<int, int>& batch_tokens_by_id
) {
    const double sim_time = state.sim_time;
    auto& stats = state.stats;

    std::unordered_set<int> ids = stats.active_request_ids;
    for (const auto& kv : batch_tokens_by_id) ids.insert(kv.first);

    for (int rid : ids) {
        NativeRequestState* req = find_req_mut(state, rid);
        if (req == nullptr) {
            stats.active_request_ids.erase(rid);
            continue;
        }

        // Prefill lateness (monotone max, finalized once prefill complete)
        if (stats.prefill_lateness_finalized.find(rid) == stats.prefill_lateness_finalized.end()) {
            const double prefill_slo = req->prefill_slo;
            if (prefill_slo >= 0.0) {
                const double deadline = req->arrived_at + prefill_slo;
                const double actual = (req->prefill_done && req->prefill_completed_at > 0.0)
                    ? req->prefill_completed_at
                    : sim_time;
                const double prefill_late = std::max(0.0, actual - deadline);
                const double prev = stats.per_request_prefill_lateness.count(rid)
                    ? stats.per_request_prefill_lateness[rid]
                    : 0.0;
                if (prefill_late > prev) {
                    stats.slo_lateness_sum += (prefill_late - prev);
                    stats.per_request_prefill_lateness[rid] = prefill_late;
                }
                if (req->prefill_done) {
                    stats.prefill_lateness_finalized.insert(rid);
                }
            }
        }

        // Decode token lateness
        if (req->decode_slo >= 0.0 && req->num_decode_tokens > 0 && req->prefill_done && req->prefill_completed_at > 0.0) {
            if (!stats.decode_next_deadline_by_id.count(rid)) {
                stats.decode_next_deadline_by_id[rid] = req->prefill_completed_at + req->decode_slo;
            }
            const int done = req->num_processed_decode_tokens;
            const int counted = stats.decode_tokens_counted.count(rid) ? stats.decode_tokens_counted[rid] : 0;
            int new_tokens = done - counted;
            while (new_tokens > 0) {
                const double deadline = stats.decode_next_deadline_by_id[rid];
                const double token_late = std::max(0.0, sim_time - deadline);
                stats.per_request_decode_lateness[rid] += token_late;
                stats.slo_lateness_sum += token_late;
                stats.decode_tokens_counted[rid] = stats.decode_tokens_counted[rid] + 1;
                stats.decode_next_deadline_by_id[rid] = deadline + req->decode_slo;
                new_tokens -= 1;
            }
        }

        const double total_lateness =
            (stats.per_request_prefill_lateness.count(rid) ? stats.per_request_prefill_lateness[rid] : 0.0) +
            (stats.per_request_decode_lateness.count(rid) ? stats.per_request_decode_lateness[rid] : 0.0);
        if (total_lateness > 0.0 && stats.violated_request_ids.find(rid) == stats.violated_request_ids.end()) {
            stats.violated_request_ids.insert(rid);
            stats.slo_violations += 1;
        }

        if (req->completed) {
            if (stats.completed_request_ids.find(rid) == stats.completed_request_ids.end()) {
                stats.completed_request_ids.insert(rid);
                stats.requests_completed += 1;
            }
            stats.active_request_ids.erase(rid);
        } else if (is_pending(*req)) {
            stats.active_request_ids.insert(rid);
        } else {
            stats.active_request_ids.erase(rid);
        }
    }

    // Keep state compact: remove completed requests from live vector.
    state.requests.erase(
        std::remove_if(
            state.requests.begin(),
            state.requests.end(),
            [](const NativeRequestState& r) { return r.completed; }
        ),
        state.requests.end()
    );
}

} // namespace

NativeSimState NativeSim::apply_adversary_action(
    const NativeSimState& in_state,
    const AdversaryActionSpecNative& action,
    const NativeRuntimeConfig& cfg
) {
    NativeSimState out = in_state;
    apply_adversary_action_inplace(out, action, cfg);
    return out;
}

NativeSimState NativeSim::apply_controller_action(
    const NativeSimState& in_state,
    const ControllerActionSpecNative& action,
    const NativeRuntimeConfig& cfg,
    NativePredictor& predictor
) {
    NativeSimState out = in_state;
    apply_controller_action_inplace(out, action, cfg, predictor);
    return out;
}

void NativeSim::apply_adversary_action_inplace(
    NativeSimState& state,
    const AdversaryActionSpecNative& action,
    const NativeRuntimeConfig& cfg
) {
    if (action.requests.empty() && action.stop_decode_ids.empty()) return;

    const double time_now = state.sim_time;
    double arrival_time = std::floor(time_now);
    if (!action.requests.empty()) {
        if (state.stats.last_prefill_batch_time < 0.0) {
            arrival_time = std::floor(time_now);
        } else {
            arrival_time = state.stats.last_prefill_batch_time + 1.0;
        }
        state.stats.last_prefill_batch_time = arrival_time;
    }

    for (const auto& spec : action.requests) {
        NativeRequestState req;
        req.request_id = state.next_request_id++;
        req.arrived_at = arrival_time;
        req.queued_at = arrival_time;
        req.num_prefill_tokens = std::max(0, spec.prefill_tokens);
        req.num_processed_prefill_tokens = 0;
        req.num_decode_tokens = std::max(0, spec.decode_tokens);
        req.num_processed_decode_tokens = 0;
        req.prefill_done = false;
        req.completed = false;
        req.prefill_completed_at = 0.0;
        req.prefill_slo = (spec.prefill_slo > 0.0) ? spec.prefill_slo : 0.0;
        req.decode_slo = (spec.decode_slo >= 0.0) ? spec.decode_slo : cfg.default_decode_slo;

        state.requests.push_back(req);
        state.stats.active_request_ids.insert(req.request_id);
        state.stats.requests_generated += 1;
        state.stats.recent_arrivals.push_back(arrival_time);
    }

    for (int rid : action.stop_decode_ids) {
        NativeRequestState* req = find_req_mut(state, rid);
        if (req == nullptr) continue;
        req->num_decode_tokens = std::max(0, req->num_processed_decode_tokens);
    }

    for (int rid : action.stop_decode_ids) {
        const NativeRequestState* req = find_req(state, rid);
        if (req == nullptr || req->completed || !is_pending(*req)) state.stats.active_request_ids.erase(rid);
        else state.stats.active_request_ids.insert(rid);
    }

    const double window_start = time_now - 1.0;
    std::vector<double> recent;
    recent.reserve(state.stats.recent_arrivals.size());
    for (double t : state.stats.recent_arrivals) {
        if (t >= window_start) recent.push_back(t);
    }
    state.stats.recent_arrivals.swap(recent);
}

void NativeSim::apply_controller_action_inplace(
    NativeSimState& state,
    const ControllerActionSpecNative& action,
    const NativeRuntimeConfig& cfg,
    NativePredictor& predictor
) {
    const auto prefill_alloc_in = alloc_vec_to_map(action.prefill_allocations);
    const auto decode_alloc_in = alloc_vec_to_map(action.decode_allocations);
    const auto token_alloc_in = alloc_vec_to_map(action.token_allocations);

    std::vector<int> selected_ids;
    selected_ids.reserve(action.selected_request_ids.size() + token_alloc_in.size() + prefill_alloc_in.size() + decode_alloc_in.size());
    if (!action.selected_request_ids.empty()) {
        selected_ids = action.selected_request_ids;
    } else {
        for (const auto& kv : token_alloc_in) selected_ids.push_back(kv.first);
        for (const auto& kv : prefill_alloc_in) selected_ids.push_back(kv.first);
        for (const auto& kv : decode_alloc_in) selected_ids.push_back(kv.first);
        std::sort(selected_ids.begin(), selected_ids.end());
        selected_ids.erase(std::unique(selected_ids.begin(), selected_ids.end()), selected_ids.end());
    }

    std::vector<ControllerRequestStateNative> pred_reqs;
    std::vector<int> pred_tokens;
    pred_reqs.reserve(selected_ids.size());
    pred_tokens.reserve(selected_ids.size());

    std::unordered_map<int, int> prefill_used;
    std::unordered_map<int, int> decode_used;
    std::unordered_map<int, int> batch_tokens_by_id;

    bool has_prefill = false;
    for (int rid : selected_ids) {
        if (state.stats.active_request_ids.find(rid) == state.stats.active_request_ids.end()) continue;
        NativeRequestState* req = find_req_mut(state, rid);
        if (req == nullptr) continue;

        const int rem_pref = remaining_prefill(*req);
        const int rem_dec = remaining_decode(*req);

        int pre_tok = prefill_alloc_in.count(rid) ? prefill_alloc_in.at(rid) : 0;
        int dec_tok = decode_alloc_in.count(rid) ? decode_alloc_in.at(rid) : 0;
        if (pre_tok == 0 && dec_tok == 0 && token_alloc_in.count(rid)) {
            const int base = token_alloc_in.at(rid);
            if (req->prefill_done) dec_tok = base;
            else pre_tok = base;
        }

        pre_tok = std::min(std::max(0, pre_tok), rem_pref);
        if (!req->prefill_done) dec_tok = 0;
        dec_tok = std::min(std::max(0, dec_tok), rem_dec);
        const int total = pre_tok + dec_tok;
        if (total <= 0) continue;

        if (pre_tok > 0) {
            has_prefill = true;
            prefill_used[rid] = pre_tok;
        }
        if (dec_tok > 0) decode_used[rid] = dec_tok;
        batch_tokens_by_id[rid] = total;

        ControllerRequestStateNative prs;
        prs.request_id = rid;
        prs.prefill_done = req->prefill_done;
        prs.remaining_prefill = rem_pref;
        prs.remaining_decode = rem_dec;
        prs.arrived_at = req->arrived_at;
        prs.prefill_slo = req->prefill_slo;
        prs.num_processed_tokens = req->num_processed_prefill_tokens + req->num_processed_decode_tokens;
        pred_reqs.push_back(prs);
        pred_tokens.push_back(total);
    }

    if (!pred_reqs.empty()) {
        const double start_time = state.sim_time;
        const auto [stage_total_time, stage_model_time] = predictor.lookup_batch_time(
            pred_reqs,
            pred_tokens,
            cfg.prefill_profile_tokens,
            cfg.prefill_profile_times
        );
        (void)stage_model_time;

        const double end_time = start_time + std::max(0.0, stage_total_time);
        state.sim_time = end_time;

        for (const auto& kv : batch_tokens_by_id) {
            NativeRequestState* req = find_req_mut(state, kv.first);
            if (req == nullptr) continue;

            const int pre_inc = prefill_used.count(kv.first) ? prefill_used[kv.first] : 0;
            const int dec_inc = decode_used.count(kv.first) ? decode_used[kv.first] : 0;

            if (pre_inc > 0) {
                req->num_processed_prefill_tokens = std::min(
                    req->num_prefill_tokens,
                    req->num_processed_prefill_tokens + pre_inc
                );
            }
            if (!req->prefill_done && req->num_processed_prefill_tokens >= req->num_prefill_tokens) {
                req->prefill_done = true;
                req->prefill_completed_at = end_time;
            }
            if (dec_inc > 0 && req->prefill_done) {
                req->num_processed_decode_tokens = std::min(
                    req->num_decode_tokens,
                    req->num_processed_decode_tokens + dec_inc
                );
            }
            if (req->prefill_done && req->num_processed_decode_tokens >= req->num_decode_tokens) {
                req->completed = true;
            }
        }
    }

    if (!has_prefill) {
        maybe_fast_forward_decode_only_to_next_adv_second(state);
    }

    update_requests_and_stats(state, batch_tokens_by_id);
}

double NativeSim::evaluate_objective_cost(const NativeSimState& state) {
    return (double)state.stats.slo_violations + state.stats.slo_lateness_sum;
}

std::vector<ControllerRequestStateNative> NativeSim::controller_view_from_state(const NativeSimState& state) {
    std::vector<ControllerRequestStateNative> out;
    out.reserve(state.stats.active_request_ids.size());
    for (int rid : state.stats.active_request_ids) {
        const NativeRequestState* req = find_req(state, rid);
        if (req == nullptr || req->completed) continue;
        if (!is_pending(*req)) continue;
        ControllerRequestStateNative r;
        r.request_id = req->request_id;
        r.prefill_done = req->prefill_done;
        r.remaining_prefill = remaining_prefill(*req);
        r.remaining_decode = remaining_decode(*req);
        r.arrived_at = req->arrived_at;
        r.prefill_slo = req->prefill_slo;
        r.num_processed_tokens = req->num_processed_prefill_tokens + req->num_processed_decode_tokens;
        out.push_back(r);
    }
    return out;
}

std::string NativeSim::snapshot(const NativeSimState& in_state) {
    std::ostringstream oss;
    oss.precision(17);
    oss << in_state.sim_time << ';' << in_state.next_request_id << ';' << in_state.requests.size();
    for (const auto& r : in_state.requests) {
        oss << ';'
            << r.request_id << ','
            << r.arrived_at << ','
            << r.queued_at << ','
            << r.num_prefill_tokens << ','
            << r.num_processed_prefill_tokens << ','
            << r.num_decode_tokens << ','
            << r.num_processed_decode_tokens << ','
            << (r.prefill_done ? 1 : 0) << ','
            << (r.completed ? 1 : 0) << ','
            << r.prefill_completed_at << ','
            << r.prefill_slo << ','
            << r.decode_slo;
    }
    return oss.str();
}

NativeSimState NativeSim::restore(const std::string& payload) {
    NativeSimState out;
    if (payload.empty()) return out;

    std::vector<std::string> fields;
    std::string cur;
    for (char c : payload) {
        if (c == ';') {
            fields.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    fields.push_back(cur);
    if (fields.size() < 3) return out;

    try {
        out.sim_time = std::stod(fields[0]);
        out.next_request_id = std::stoi(fields[1]);
        const int n = std::stoi(fields[2]);
        out.requests.reserve((size_t)std::max(0, n));
        for (int i = 0; i < n; ++i) {
            const size_t idx = (size_t)(3 + i);
            if (idx >= fields.size()) break;
            std::vector<std::string> p;
            std::string tok;
            for (char c : fields[idx]) {
                if (c == ',') {
                    p.push_back(tok);
                    tok.clear();
                } else {
                    tok.push_back(c);
                }
            }
            p.push_back(tok);
            if (p.size() < 12) continue;
            NativeRequestState r;
            r.request_id = std::stoi(p[0]);
            r.arrived_at = std::stod(p[1]);
            r.queued_at = std::stod(p[2]);
            r.num_prefill_tokens = std::stoi(p[3]);
            r.num_processed_prefill_tokens = std::stoi(p[4]);
            r.num_decode_tokens = std::stoi(p[5]);
            r.num_processed_decode_tokens = std::stoi(p[6]);
            r.prefill_done = (std::stoi(p[7]) != 0);
            r.completed = (std::stoi(p[8]) != 0);
            r.prefill_completed_at = std::stod(p[9]);
            r.prefill_slo = std::stod(p[10]);
            r.decode_slo = std::stod(p[11]);
            out.requests.push_back(r);
            if (!r.completed && is_pending(r)) out.stats.active_request_ids.insert(r.request_id);
            if (r.completed) out.stats.completed_request_ids.insert(r.request_id);
        }
    } catch (...) {
        return NativeSimState{};
    }
    return out;
}

} // namespace mcts_native
