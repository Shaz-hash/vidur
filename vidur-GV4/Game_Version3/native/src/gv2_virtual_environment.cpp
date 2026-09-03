#include "gv2_virtual_environment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

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
    double total_lateness = 0.0;
};

std::unordered_map<int, ReqView> build_req_views(const SimState& state) {
    std::unordered_map<int, ReqView> out;
    out.reserve(state.requests.size());

    for (const auto& r : state.requests) {
        if (r.completed) continue;

        ReqView v;
        v.rid = r.request_id;
        v.prefill_done = r.prefill_done();
        v.rem_prefill = r.remaining_prefill();
        v.rem_decode = r.remaining_decode();
        v.decode_processed = int(r.num_processed_decode_tokens);
        v.arrived_at = r.arrived_at;
        v.prefill_slo = r.prefill_slo_time;
        v.prefill_deadline = (r.prefill_deadline > 0.0) ? r.prefill_deadline : (r.arrived_at + r.prefill_slo_time);

        const auto it_pref = state.stats.per_request_prefill_lateness_by_id.find(r.request_id);
        const auto it_dec = state.stats.per_request_decode_lateness_by_id.find(r.request_id);
        v.prefill_lateness = (it_pref != state.stats.per_request_prefill_lateness_by_id.end())
            ? it_pref->second
            : std::max(0.0, double(state.sim_time) - v.prefill_deadline);
        const double dec_lateness = (it_dec != state.stats.per_request_decode_lateness_by_id.end())
            ? it_dec->second
            : double(r.decode_lateness);
        v.total_lateness = std::max(0.0, v.prefill_lateness) + std::max(0.0, dec_lateness);

        if ((!v.prefill_done && v.rem_prefill <= 0) || (v.prefill_done && v.rem_decode <= 0)) continue;
        out[v.rid] = v;
    }

    return out;
}

std::vector<int> eviction_targets_for_rule(
    const std::string& rule,
    const std::unordered_map<int, ReqView>& views,
    double eps) {
    std::vector<int> prefill_ids;
    std::vector<int> decode_ids;
    prefill_ids.reserve(views.size());
    decode_ids.reserve(views.size());

    for (const auto& kv : views) {
        const ReqView& v = kv.second;
        if (!v.prefill_done && v.rem_prefill > 0) prefill_ids.push_back(v.rid);
        else if (v.prefill_done && v.rem_decode > 0) decode_ids.push_back(v.rid);
    }

    std::sort(prefill_ids.begin(), prefill_ids.end());
    std::sort(decode_ids.begin(), decode_ids.end());

    if (rule == "evict_none") return {};

    if (rule == "evict_largest_prefill") {
        if (prefill_ids.empty()) return {};
        const int rid = *std::max_element(
            prefill_ids.begin(), prefill_ids.end(), [&](int a, int b) {
                const ReqView& va = views.at(a);
                const ReqView& vb = views.at(b);
                if (va.rem_prefill != vb.rem_prefill) return va.rem_prefill < vb.rem_prefill;
                return a > b;
            });
        return {rid};
    }

    if (rule == "evict_earliest_prefill_deadline") {
        if (prefill_ids.empty()) return {};
        const int rid = *std::min_element(
            prefill_ids.begin(), prefill_ids.end(), [&](int a, int b) {
                const ReqView& va = views.at(a);
                const ReqView& vb = views.at(b);
                if (va.prefill_deadline != vb.prefill_deadline) return va.prefill_deadline < vb.prefill_deadline;
                return a < b;
            });
        return {rid};
    }

    if (rule == "evict_prefill_missed_deadline") {
        std::vector<int> out;
        for (int rid : prefill_ids) {
            if (views.at(rid).prefill_lateness > eps) out.push_back(rid);
        }
        return out;
    }

    if (rule == "evict_prefill_lateness_over_0p5") {
        std::vector<int> out;
        for (int rid : prefill_ids) {
            if (views.at(rid).prefill_lateness > 0.5) out.push_back(rid);
        }
        return out;
    }

    if (rule == "evict_longest_decode") {
        if (decode_ids.empty()) return {};
        const int rid = *std::max_element(
            decode_ids.begin(), decode_ids.end(), [&](int a, int b) {
                const ReqView& va = views.at(a);
                const ReqView& vb = views.at(b);
                if (va.decode_processed != vb.decode_processed) return va.decode_processed < vb.decode_processed;
                return a > b;
            });
        return {rid};
    }

    if (rule == "evict_decode_lateness_over_0p5") {
        std::vector<int> out;
        for (int rid : decode_ids) {
            if (views.at(rid).total_lateness > 0.5) out.push_back(rid);
        }
        return out;
    }

    if (rule == "evict_prefill_highest_lateness") {
        if (prefill_ids.empty()) return {};
        const int rid = *std::max_element(
            prefill_ids.begin(), prefill_ids.end(), [&](int a, int b) {
                const ReqView& va = views.at(a);
                const ReqView& vb = views.at(b);
                if (va.prefill_lateness != vb.prefill_lateness) return va.prefill_lateness < vb.prefill_lateness;
                return a > b;
            });
        return views.at(rid).prefill_lateness > eps ? std::vector<int>{rid} : std::vector<int>{};
    }

    if (rule == "evict_decode_highest_lateness") {
        if (decode_ids.empty()) return {};
        const int rid = *std::max_element(
            decode_ids.begin(), decode_ids.end(), [&](int a, int b) {
                const ReqView& va = views.at(a);
                const ReqView& vb = views.at(b);
                if (va.total_lateness != vb.total_lateness) return va.total_lateness < vb.total_lateness;
                return a > b;
            });
        return views.at(rid).total_lateness > eps ? std::vector<int>{rid} : std::vector<int>{};
    }

    return {};
}

}  // namespace

GV2VirtualEnvironment::GV2VirtualEnvironment(GV2EnvConfig cfg)
    : cfg_(std::move(cfg)),
      virtual_sim_([this]() {
          VirtualSimulatorConfig sim_cfg;
          sim_cfg.adversary_tick_sec = cfg_.adversary_tick_sec;
          return sim_cfg;
      }()) {}

GV2VirtualEnvironment::GV2VirtualEnvironment(GV2EnvConfig cfg, VirtualSimulatorConfig sim_cfg)
    : cfg_(std::move(cfg)), virtual_sim_(std::move(sim_cfg)) {
    if (virtual_sim_.cfg().adversary_tick_sec <= 0.0) {
        VirtualSimulatorConfig patched = virtual_sim_.cfg();
        patched.adversary_tick_sec = cfg_.adversary_tick_sec;
        virtual_sim_.set_config(std::move(patched));
    }
}

const GV2EnvConfig& GV2VirtualEnvironment::cfg() const { return cfg_; }

VirtualSimulatorGV2& GV2VirtualEnvironment::virtual_simulator() { return virtual_sim_; }

const VirtualSimulatorGV2& GV2VirtualEnvironment::virtual_simulator() const { return virtual_sim_; }

bool GV2VirtualEnvironment::load_predictor_csv(const std::string& path) {
    return virtual_sim_.load_predictor_csv(path);
}

void GV2VirtualEnvironment::set_prefill_profile(std::vector<int> tokens, std::vector<double> times) {
    virtual_sim_.set_prefill_profile(std::move(tokens), std::move(times));
}

SampledActionSet<AdversaryAction> GV2VirtualEnvironment::sample_adversary_actions(
    const SimState& state,
    const std::unordered_set<int>& forbidden_stop_ids) const {
    AdversarySamplerConfig scfg = cfg_.adversary_sampler;
    scfg.launch_window_sec = cfg_.launch_window_sec;
    scfg.max_requests_per_launch_window = cfg_.max_requests_per_launch_window;
    scfg.prefill_window_cap_tokens = cfg_.prefill_window_cap_tokens;
    scfg.max_decode_tokens_per_request = cfg_.max_decode_tokens_per_request;
    scfg.default_decode_slo_time = cfg_.decode_slo_time_default;
    for (int tokens : scfg.allowed_prefill_tokens) {
        if (scfg.prefill_slo_by_tokens.find(tokens) == scfg.prefill_slo_by_tokens.end()) {
            scfg.prefill_slo_by_tokens[tokens] = std::max(0.0, virtual_sim_.prefill_profile_lookup(tokens));
        }
    }

    const double decision_tick = (state.stats.next_adv_tick >= 0.0)
        ? state.stats.next_adv_tick
        : quantize_down(state.sim_time);

    return sample_adversary_actions_gv2(state, scfg, decision_tick, forbidden_stop_ids);
}

SampledActionSet<ControllerAction> GV2VirtualEnvironment::sample_controller_actions(const SimState& state) const {
    SimState tmp = state;
    init_clock_if_needed(tmp);
    if (has_pending_adv_tick(tmp)) {
        const int n = controller_action_space_size(cfg_.controller_sampler);
        SampledActionSet<ControllerAction> out;
        out.actions.resize(static_cast<std::size_t>(n));
        out.mask.assign(static_cast<std::size_t>(n), 0u);
        if (!out.actions.empty()) {
            ControllerAction noop;
            noop.token_budget = 0;
            noop.strategy = "GV2|evict_none";
            noop.mapping = {0, 0, 0};
            noop.has_mapping = true;
            noop.valid = true;
            out.actions[0] = std::move(noop);
            out.mask[0] = 1u;
        }
        return out;
    }

    ControllerSamplerConfig scfg = cfg_.controller_sampler;
    scfg.enforce_nonnegative_decode_credits = cfg_.enforce_nonnegative_decode_credits;
    scfg.prefill_profile_tokens = virtual_sim_.cfg().prefill_profile_tokens;
    scfg.prefill_profile_times = virtual_sim_.cfg().prefill_profile_times;
    return sample_controller_actions_gv2(tmp, scfg, std::max(0, tmp.stats.decode_credit_balance));
}

double GV2VirtualEnvironment::round_time(double t) const {
    const int digits = std::max(0, cfg_.time_round_digits);
    const double scale = std::pow(10.0, double(digits));
    return std::round(t * scale) / scale;
}

double GV2VirtualEnvironment::quantize_down(double t) const {
    const double tick = (cfg_.adversary_tick_sec > 0.0) ? cfg_.adversary_tick_sec : 0.2;
    const double q = std::floor((double(t) + cfg_.eps) / tick);
    return round_time(q * tick);
}

void GV2VirtualEnvironment::init_clock_if_needed(SimState& state) const {
    if (state.stats.next_adv_tick < 0.0) {
        state.stats.next_adv_tick = quantize_down(state.sim_time);
    }
    if (state.stats.last_adv_tick < -1.0) {
        state.stats.last_adv_tick = -1.0;
    }
    state.stats.pending_adv_tick = state.stats.next_adv_tick <= (state.sim_time + cfg_.eps);
}

double GV2VirtualEnvironment::current_adv_tick(SimState& state) const {
    init_clock_if_needed(state);
    return round_time(state.stats.next_adv_tick);
}

double GV2VirtualEnvironment::next_adv_tick_state(SimState& state) const {
    init_clock_if_needed(state);
    return round_time(state.stats.next_adv_tick);
}

bool GV2VirtualEnvironment::has_pending_adv_tick(SimState& state) const {
    init_clock_if_needed(state);
    const bool pending = state.stats.next_adv_tick <= (state.sim_time + cfg_.eps);
    state.stats.pending_adv_tick = pending;
    return pending;
}

bool GV2VirtualEnvironment::id_in_sorted(const std::vector<int>& sorted_ids, int rid) {
    return std::binary_search(sorted_ids.begin(), sorted_ids.end(), rid);
}

void GV2VirtualEnvironment::add_sorted_unique(std::vector<int>* sorted_ids, int rid) {
    if (sorted_ids == nullptr) return;
    auto it = std::lower_bound(sorted_ids->begin(), sorted_ids->end(), rid);
    if (it == sorted_ids->end() || *it != rid) {
        sorted_ids->insert(it, rid);
    }
}

void GV2VirtualEnvironment::remove_if_present(std::vector<int>* sorted_ids, int rid) {
    if (sorted_ids == nullptr) return;
    auto it = std::lower_bound(sorted_ids->begin(), sorted_ids->end(), rid);
    if (it != sorted_ids->end() && *it == rid) {
        sorted_ids->erase(it);
    }
}

bool GV2VirtualEnvironment::has_active_prefill(const SimState& s) {
    for (const auto& r : s.requests) {
        if (r.prefill_active()) return true;
    }
    return false;
}

bool GV2VirtualEnvironment::has_active_decode(const SimState& s) {
    for (const auto& r : s.requests) {
        if (r.decode_active()) return true;
    }
    return false;
}

bool GV2VirtualEnvironment::is_controller_strict_noop(const ControllerAction& a) {
    return a.token_budget == 0 &&
           a.selected_request_ids.empty() &&
           a.token_allocations.empty() &&
           a.prefill_allocations.empty() &&
           a.decode_allocations.empty();
}

double GV2VirtualEnvironment::next_adv_tick(double sim_time, double tick_sec, double eps) {
    const double tick = tick_sec > 0.0 ? tick_sec : 0.2;
    const double q = std::floor((sim_time + eps) / tick);
    const double t = (q + 1.0) * tick;
    return t > sim_time ? t : (sim_time + tick);
}

RequestState* GV2VirtualEnvironment::find_request(SimState& s, int request_id) {
    for (auto& r : s.requests) {
        if (r.request_id == request_id) return &r;
    }
    return nullptr;
}

const RequestState* GV2VirtualEnvironment::find_request_const(const SimState& s, int request_id) {
    for (const auto& r : s.requests) {
        if (r.request_id == request_id) return &r;
    }
    return nullptr;
}

void GV2VirtualEnvironment::prune_recent_launches(SimState& state, double anchor_time) const {
    const double lo = anchor_time - std::max(0.0, cfg_.launch_window_sec);
    std::vector<LaunchWindowEntry> keep;
    keep.reserve(state.stats.recent_launches.size());
    for (const auto& x : state.stats.recent_launches) {
        if (x.timestamp + cfg_.eps < lo) continue;
        keep.push_back(x);
    }
    state.stats.recent_launches.swap(keep);

    std::vector<double> keep_arrivals;
    keep_arrivals.reserve(state.stats.recent_arrivals.size());
    for (double ts : state.stats.recent_arrivals) {
        if (ts + cfg_.eps < lo) continue;
        keep_arrivals.push_back(ts);
    }
    state.stats.recent_arrivals.swap(keep_arrivals);
}

std::pair<int, int> GV2VirtualEnvironment::window_usage(const SimState& state, double anchor_time) const {
    const double lo = anchor_time - std::max(0.0, cfg_.launch_window_sec);
    const double hi = anchor_time + cfg_.eps;

    int cnt = 0;
    int pref = 0;

    if (!state.stats.recent_launches.empty()) {
        for (const auto& x : state.stats.recent_launches) {
            if (x.timestamp + cfg_.eps < lo) continue;
            if (x.timestamp > hi) continue;
            cnt += std::max(0, x.count);
            pref += std::max(0, x.prefill_tokens);
        }
        return {cnt, pref};
    }

    for (double ts : state.stats.recent_arrivals) {
        if (ts + cfg_.eps < lo) continue;
        if (ts > hi) continue;
        cnt += 1;
    }
    return {cnt, pref};
}

void GV2VirtualEnvironment::append_launch_event(
    SimState& state,
    double ts,
    int count,
    int prefill_tokens) const {
    LaunchWindowEntry e;
    e.timestamp = round_time(ts);
    e.count = std::max(0, count);
    e.prefill_tokens = std::max(0, prefill_tokens);
    state.stats.recent_launches.push_back(e);
    for (int i = 0; i < e.count; ++i) {
        state.stats.recent_arrivals.push_back(e.timestamp);
    }
}

void GV2VirtualEnvironment::drop_request(
    SimState& state,
    int rid,
    DecodeCreditLedger* ledger) const {
    RequestState* req = find_request(state, rid);
    if (req == nullptr) return;
    if (req->completed && req->dropped) return;

    auto& stats = state.stats;

    const double prev_pref = [&]() {
        const auto it = stats.per_request_prefill_lateness_by_id.find(rid);
        if (it == stats.per_request_prefill_lateness_by_id.end()) return 0.0;
        const double v = std::max(0.0, it->second);
        stats.per_request_prefill_lateness_by_id.erase(it);
        return v;
    }();
    const double prev_dec = [&]() {
        const auto it = stats.per_request_decode_lateness_by_id.find(rid);
        if (it == stats.per_request_decode_lateness_by_id.end()) return 0.0;
        const double v = std::max(0.0, it->second);
        stats.per_request_decode_lateness_by_id.erase(it);
        return v;
    }();

    const double prev_total = std::max(0.0, prev_pref + prev_dec);
    if (prev_total > 0.0) {
        stats.slo_lateness_sum = std::max(0.0, stats.slo_lateness_sum - prev_total);
    }

    if (id_in_sorted(stats.violated_request_ids, rid)) {
        remove_if_present(&stats.violated_request_ids, rid);
        if (stats.slo_violations > 0) stats.slo_violations -= 1;
    }

    if (!req->prefill_done()) {
        req->is_prefill_complete = true;
        req->prefill_completed_at = state.sim_time;
    }
    req->num_prefill_tokens = std::max(0, req->num_processed_prefill_tokens);
    req->num_decode_tokens = std::max(0, req->num_processed_decode_tokens);
    // Keep request flags consistent with dropped-terminal semantics:
    // dropped requests must not remain in violated/per-request lateness state.
    req->prefill_lateness = 0.0;
    req->decode_lateness = 0.0;
    req->violated = false;
    req->completed = true;
    req->dropped = true;
    req->completed_at = state.sim_time;

    if (!id_in_sorted(stats.completed_request_ids, rid)) {
        add_sorted_unique(&stats.completed_request_ids, rid);
        stats.requests_completed += 1;
    }
    remove_if_present(&stats.active_request_ids, rid);
    add_sorted_unique(&stats.dropped_request_ids, rid);

    if (ledger != nullptr) {
        (void)ledger->reclaim_on_drop(rid, cfg_.decode_credit_mint_per_prefill_complete);
    }

    stats.decode_next_deadline_by_id.erase(rid);
    remove_if_present(&stats.prefill_lateness_finalized_ids, rid);

    // Dropped request contributes terminal objective cost only.
    stats.slo_lateness_sum += std::max(0.0, cfg_.drop_cost);
}

void GV2VirtualEnvironment::apply_controller_eviction_rule(
    SimState& state,
    const ControllerAction& action,
    DecodeCreditLedger* ledger) const {
    std::string rule = "evict_none";

    if (action.has_mapping && !cfg_.controller_sampler.eviction_rule_names.empty()) {
        const int idx = action.mapping[0];
        if (idx >= 0 && idx < static_cast<int>(cfg_.controller_sampler.eviction_rule_names.size())) {
            rule = cfg_.controller_sampler.eviction_rule_names[static_cast<std::size_t>(idx)];
        }
    } else if (!action.strategy.empty()) {
        const std::string prefix = "GV2|";
        if (action.strategy.rfind(prefix, 0) == 0) {
            rule = action.strategy.substr(prefix.size());
        }
    }

    if (rule == "evict_none") return;

    const auto views = build_req_views(state);
    const auto targets = eviction_targets_for_rule(rule, views, cfg_.eps);

    for (int rid : targets) {
        drop_request(state, rid, ledger);
    }

    if (cfg_.enforce_nonnegative_decode_credits) {
        finalize_decodes_to_credit_budget(state, ledger);
    }
}

void GV2VirtualEnvironment::enforce_decode_caps(SimState& state) const {
    const int cap = std::max(1, cfg_.max_decode_tokens_per_request);
    for (auto& req : state.requests) {
        if (req.completed) continue;
        const int done = std::max(0, req.num_processed_decode_tokens);
        if (done < cap) continue;

        req.num_decode_tokens = std::min(req.num_decode_tokens, cap);
        if (done >= req.num_decode_tokens && req.prefill_done()) {
            req.completed = true;
            req.completed_at = state.sim_time;
        }
    }
}

void GV2VirtualEnvironment::finalize_decodes_to_credit_budget(
    SimState& state,
    DecodeCreditLedger* ledger) const {
    if (!cfg_.enforce_nonnegative_decode_credits || ledger == nullptr) {
        return;
    }

    enforce_decode_caps(state);

    const int bal = std::max(0, ledger->available_balance(true));

    std::vector<std::pair<int, int>> decode_active;  // (processed_decode_tokens, rid)
    decode_active.reserve(state.requests.size());

    for (const auto& req : state.requests) {
        if (req.completed) continue;
        if (!req.prefill_done()) continue;

        const int done = std::max(0, req.num_processed_decode_tokens);
        const int decode_goal = std::min(std::max(0, req.num_decode_tokens), cfg_.max_decode_tokens_per_request);
        if (done < decode_goal) {
            decode_active.emplace_back(done, req.request_id);
        }
    }

    const int overflow = static_cast<int>(decode_active.size()) - bal;
    if (overflow <= 0) return;

    std::sort(
        decode_active.begin(),
        decode_active.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second > b.second;
        });

    for (int i = 0; i < overflow; ++i) {
        const int rid = decode_active[static_cast<std::size_t>(i)].second;
        RequestState* req = find_request(state, rid);
        if (req == nullptr) continue;

        req->num_decode_tokens = std::max(0, req->num_processed_decode_tokens);
        req->completed = true;
        req->stopped_decode = true;
        req->completed_at = state.sim_time;

        add_sorted_unique(&state.stats.completed_request_ids, rid);
        add_sorted_unique(&state.stats.stopped_decode_request_ids, rid);
        remove_if_present(&state.stats.active_request_ids, rid);

        state.stats.decode_next_deadline_by_id.erase(rid);
        ledger->erase_request(rid);
    }
}

void GV2VirtualEnvironment::apply_batch_progress(
    SimState& state,
    const ControllerBatchPlan& plan,
    double batch_start,
    double batch_end,
    DecodeCreditLedger* ledger) const {
    auto& stats = state.stats;

    for (int rid : plan.request_ids) {
        RequestState* req = find_request(state, rid);
        if (req == nullptr || req->completed) continue;

        const bool prefill_complete_before = req->prefill_done();

        const int pre_add_req = [&]() {
            const auto it = plan.prefill_alloc.find(rid);
            return (it == plan.prefill_alloc.end()) ? 0 : std::max(0, it->second);
        }();
        if (pre_add_req > 0 && !req->prefill_done()) {
            const int applied = std::min(pre_add_req, req->remaining_prefill());
            if (applied > 0) {
                req->num_processed_prefill_tokens += applied;
            }
            if (req->remaining_prefill() <= 0) {
                req->is_prefill_complete = true;
                if (req->prefill_completed_at < 0.0) req->prefill_completed_at = batch_end;
            }
        }

        const bool prefill_complete_after = req->prefill_done();
        const bool became_prefill_complete = (!prefill_complete_before && prefill_complete_after);
        if (became_prefill_complete &&
            req->remaining_decode() > 0 &&
            !req->completed &&
            ledger != nullptr) {
            (void)ledger->mint_on_prefill_complete_once(
                rid,
                std::max(0, cfg_.decode_credit_mint_per_prefill_complete));
        }

        const int dec_add_req = [&]() {
            const auto it = plan.decode_alloc.find(rid);
            return (it == plan.decode_alloc.end()) ? 0 : std::max(0, it->second);
        }();

        if (dec_add_req > 0 && req->prefill_done() && !req->completed) {
            const int capped = std::min(dec_add_req, req->remaining_decode());
            int allowed = 0;
            if (ledger != nullptr) {
                if (cfg_.enforce_nonnegative_decode_credits) {
                    allowed = ledger->consume_decode(rid, capped, true);
                } else {
                    allowed = capped;
                    ledger->record_decode_without_spend(rid, allowed);
                }
            } else {
                allowed = capped;
            }

            if (allowed > 0) {
                req->num_processed_decode_tokens += allowed;

                if (req->decode_slo_time >= 0.0 && req->num_decode_tokens > 0) {
                    auto it_dl = stats.decode_next_deadline_by_id.find(rid);
                    if (it_dl == stats.decode_next_deadline_by_id.end()) {
                        const double base = (req->prefill_completed_at >= 0.0)
                            ? req->prefill_completed_at
                            : batch_start;
                        stats.decode_next_deadline_by_id[rid] = base + req->decode_slo_time;
                        it_dl = stats.decode_next_deadline_by_id.find(rid);
                    }

                    const double deadline = it_dl->second;
                    const double token_late = std::max(0.0, batch_end - deadline);
                    const double inc = token_late * static_cast<double>(allowed);

                    stats.per_request_decode_lateness_by_id[rid] =
                        stats.per_request_decode_lateness_by_id[rid] + inc;
                    stats.slo_lateness_sum += inc;
                    req->decode_lateness = stats.per_request_decode_lateness_by_id[rid];

                    stats.decode_next_deadline_by_id[rid] = batch_end + req->decode_slo_time;
                    req->decode_next_deadline = stats.decode_next_deadline_by_id[rid];
                }
            }
        }
    }
}

void GV2VirtualEnvironment::refresh_request_and_stats_post_step(
    SimState& state,
    DecodeCreditLedger* ledger) const {
    auto& stats = state.stats;
    const double sim_time = state.sim_time;

    std::unordered_set<int> finalized_prefill(
        stats.prefill_lateness_finalized_ids.begin(),
        stats.prefill_lateness_finalized_ids.end());

    for (auto& req : state.requests) {
        const int rid = req.request_id;

        if (!req.completed) {
            if (req.prefill_slo_time >= 0.0) {
                req.prefill_deadline = req.arrived_at + req.prefill_slo_time;
                if (finalized_prefill.find(rid) == finalized_prefill.end()) {
                    const double actual = (req.prefill_done() && req.prefill_completed_at >= 0.0)
                        ? req.prefill_completed_at
                        : sim_time;
                    const double prefill_late = std::max(0.0, actual - req.prefill_deadline);
                    const double prev = [&]() {
                        const auto it = stats.per_request_prefill_lateness_by_id.find(rid);
                        return (it == stats.per_request_prefill_lateness_by_id.end()) ? 0.0 : std::max(0.0, it->second);
                    }();
                    if (prefill_late > prev + cfg_.eps) {
                        stats.slo_lateness_sum += (prefill_late - prev);
                        stats.per_request_prefill_lateness_by_id[rid] = prefill_late;
                    }
                    req.prefill_lateness = std::max(prev, prefill_late);
                    if (req.prefill_done()) {
                        finalized_prefill.insert(rid);
                    }
                } else {
                    const auto it = stats.per_request_prefill_lateness_by_id.find(rid);
                    if (it != stats.per_request_prefill_lateness_by_id.end()) {
                        req.prefill_lateness = std::max(0.0, it->second);
                    }
                }
            }

            if (req.prefill_done() && req.num_decode_tokens > 0 && req.decode_slo_time >= 0.0) {
                if (req.prefill_completed_at < 0.0) req.prefill_completed_at = sim_time;
                if (stats.decode_next_deadline_by_id.find(rid) == stats.decode_next_deadline_by_id.end()) {
                    stats.decode_next_deadline_by_id[rid] = req.prefill_completed_at + req.decode_slo_time;
                }
                req.decode_next_deadline = stats.decode_next_deadline_by_id[rid];
            }

            const double pref_l = [&]() {
                const auto it = stats.per_request_prefill_lateness_by_id.find(rid);
                return (it == stats.per_request_prefill_lateness_by_id.end()) ? 0.0 : std::max(0.0, it->second);
            }();
            const double dec_l = [&]() {
                const auto it = stats.per_request_decode_lateness_by_id.find(rid);
                return (it == stats.per_request_decode_lateness_by_id.end()) ? 0.0 : std::max(0.0, it->second);
            }();

            req.prefill_lateness = pref_l;
            req.decode_lateness = dec_l;

            const double total_lateness = pref_l + dec_l;
            if (total_lateness >= std::max(0.0, cfg_.auto_drop_lateness_sec)) {
                drop_request(state, rid, ledger);
                continue;
            }

            if (total_lateness > cfg_.eps) {
                if (!id_in_sorted(stats.violated_request_ids, rid)) {
                    add_sorted_unique(&stats.violated_request_ids, rid);
                    stats.slo_violations += 1;
                }
                req.violated = true;
            }

            if (req.prefill_done() && req.remaining_decode() <= 0) {
                req.completed = true;
                req.completed_at = sim_time;
                add_sorted_unique(&stats.completed_request_ids, rid);
                remove_if_present(&stats.active_request_ids, rid);
                stats.decode_next_deadline_by_id.erase(rid);
                stats.requests_completed += 1;
            }
        }
    }

    stats.prefill_lateness_finalized_ids.assign(finalized_prefill.begin(), finalized_prefill.end());
    std::sort(stats.prefill_lateness_finalized_ids.begin(), stats.prefill_lateness_finalized_ids.end());

    finalize_decodes_to_credit_budget(state, ledger);

    if (ledger != nullptr) {
        ledger->write_back_stats(&stats, cfg_.enforce_nonnegative_decode_credits);
    } else {
        stats.decode_credit_available = cfg_.enforce_nonnegative_decode_credits
            ? std::max(0, stats.decode_credit_balance)
            : stats.decode_credit_balance;
    }

    rebuild_active_completed_ids(state);
    stats.pending_adv_tick = has_pending_adv_tick(state);
}

bool GV2VirtualEnvironment::maybe_fast_forward_decode_only_to_next_adv_tick(
    SimState& state,
    DecodeCreditLedger* ledger) const {
    bool progressed_any = false;
    constexpr int kMaxLoops = 10000;

    for (int loops = 0; loops < kMaxLoops; ++loops) {
        if (has_pending_adv_tick(state)) return progressed_any;
        if (has_active_prefill(state)) return progressed_any;

        std::vector<int> decode_ids;
        decode_ids.reserve(state.requests.size());
        const int cap = std::max(1, cfg_.max_decode_tokens_per_request);
        for (const auto& req : state.requests) {
            if (req.completed) continue;
            if (!req.prefill_done()) continue;
            if (req.remaining_decode() <= 0) continue;
            if (req.num_processed_decode_tokens >= cap) continue;
            decode_ids.push_back(req.request_id);
        }
        std::sort(decode_ids.begin(), decode_ids.end());

        if (cfg_.enforce_nonnegative_decode_credits) {
            const int bal = (ledger != nullptr)
                ? std::max(0, ledger->available_balance(true))
                : std::max(0, state.stats.decode_credit_balance);
            if (bal <= 0) {
                decode_ids.clear();
            } else if (static_cast<int>(decode_ids.size()) > bal) {
                decode_ids.resize(static_cast<std::size_t>(bal));
            }
        }

        if (decode_ids.empty()) {
            const double next_tick = next_adv_tick_state(state);
            if (state.sim_time + cfg_.eps < next_tick) {
                state.sim_time = next_tick;
                progressed_any = true;
                refresh_request_and_stats_post_step(state, ledger);
            }
            return progressed_any;
        }

        ControllerAction decode_action;
        decode_action.token_budget = static_cast<int>(decode_ids.size());
        decode_action.selected_request_ids = decode_ids;
        decode_action.strategy = "GV2|decode_only_ff";
        decode_action.valid = true;
        for (int rid : decode_ids) {
            decode_action.token_allocations[rid] = 1;
            decode_action.decode_allocations[rid] = 1;
        }

        const int decode_credit_limit = cfg_.enforce_nonnegative_decode_credits
            ? ((ledger != nullptr)
                ? std::max(0, ledger->available_balance(true))
                : std::max(0, state.stats.decode_credit_balance))
            : std::numeric_limits<int>::max();
        ControllerBatchPlan plan = virtual_sim_.build_controller_batch_plan(
            state,
            decode_action,
            cfg_.enforce_nonnegative_decode_credits,
            decode_credit_limit);
        if (plan.predictor_reqs.empty()) {
            return progressed_any;
        }

        const double batch_start = state.sim_time;
        (void)virtual_sim_.execute_controller_batch_timing(state, plan);
        const double batch_end = state.sim_time;
        apply_batch_progress(state, plan, batch_start, batch_end, ledger);
        refresh_request_and_stats_post_step(state, ledger);
        progressed_any = true;
    }

    return progressed_any;
}

void GV2VirtualEnvironment::apply_adversary_action_inplace(SimState& state, const AdversaryAction& action) const {
    init_clock_if_needed(state);
    state.stats.transition_discount_time = state.sim_time;
    state.stats.transition_final_time = state.sim_time;

    double time_now = state.sim_time;
    const double tick = current_adv_tick(state);

    const bool is_pre_tick = (time_now + cfg_.eps) < tick;
    const bool strict_noop = action.requests.empty() && action.stop_decode_ids.empty();

    if (strict_noop && is_pre_tick) {
        state.stats.decode_credit_available = cfg_.enforce_nonnegative_decode_credits
            ? std::max(0, state.stats.decode_credit_balance)
            : state.stats.decode_credit_balance;
        state.stats.pending_adv_tick = has_pending_adv_tick(state);
        state.stats.transition_discount_time = state.sim_time;
        state.stats.transition_final_time = state.sim_time;
        return;
    }

    if (time_now + cfg_.eps < tick) {
        state.sim_time = tick;
        time_now = tick;
    }

    prune_recent_launches(state, tick);
    const auto usage = window_usage(state, tick);

    const int req_cap = std::max(0, cfg_.max_requests_per_launch_window);
    const int prefill_cap = std::max(0, cfg_.prefill_window_cap_tokens);

    const int requested_count = static_cast<int>(action.requests.size());
    int requested_prefill = 0;
    for (const auto& spec : action.requests) {
        requested_prefill += std::max(0, spec.prefill_tokens);
    }

    const bool can_send =
        requested_count > 0 &&
        (usage.first + requested_count <= req_cap) &&
        (usage.second + requested_prefill <= prefill_cap);

    int created_count = 0;
    int created_prefill_total = 0;

    if (can_send) {
        const double arrival_time = tick;
        for (const auto& spec : action.requests) {
            RequestState r;
            r.request_id = state.next_request_id++;
            r.arrived_at = arrival_time;
            r.queued_at = arrival_time;
            r.num_prefill_tokens = std::max(1, std::min(int(spec.prefill_tokens), cfg_.max_prefill_tokens_per_request));
            r.num_decode_tokens = std::max(
                cfg_.min_decode_tokens_per_request,
                std::min(int(spec.decode_tokens), cfg_.max_decode_tokens_per_request));
            r.prefill_slo_time = std::max(0.0, spec.prefill_slo);
            r.decode_slo_time = std::max(0.0, spec.decode_slo);
            r.prefill_deadline = (r.prefill_slo_time > 0.0) ? (arrival_time + r.prefill_slo_time) : -1.0;
            r.decode_next_deadline = -1.0;
            r.prefill_completed_at = -1.0;
            r.completed_at = -1.0;
            r.is_prefill_complete = false;
            r.completed = false;

            state.requests.push_back(r);
            add_sorted_unique(&state.stats.active_request_ids, r.request_id);
            state.stats.requests_generated += 1;
            created_count += 1;
            created_prefill_total += r.num_prefill_tokens;
        }

        append_launch_event(state, arrival_time, created_count, created_prefill_total);
    }

    const double next_tick = round_time(tick + cfg_.adversary_tick_sec);
    state.stats.last_adv_tick = tick;
    state.stats.next_adv_tick = next_tick;
    state.stats.pending_adv_tick = false;
    state.stats.missed_adv_source = 0;

    for (int rid : action.stop_decode_ids) {
        RequestState* req = find_request(state, rid);
        if (req == nullptr) continue;
        if (req->completed) continue;
        if (!req->prefill_done()) continue;
        if (req->remaining_decode() <= 0) continue;

        req->num_decode_tokens = std::max(0, req->num_processed_decode_tokens);
        req->stopped_decode = true;
        if (req->remaining_decode() <= 0 && req->prefill_done()) {
            req->completed = true;
            req->completed_at = state.sim_time;
            add_sorted_unique(&state.stats.completed_request_ids, rid);
            add_sorted_unique(&state.stats.stopped_decode_request_ids, rid);
            remove_if_present(&state.stats.active_request_ids, rid);
            state.stats.decode_next_deadline_by_id.erase(rid);
            state.stats.requests_completed += 1;
        }
    }

    enforce_decode_caps(state);
    rebuild_active_completed_ids(state);

    state.stats.decode_credit_available = cfg_.enforce_nonnegative_decode_credits
        ? std::max(0, state.stats.decode_credit_balance)
        : state.stats.decode_credit_balance;
    state.stats.pending_adv_tick = has_pending_adv_tick(state);
    state.stats.transition_discount_time = state.sim_time;
    state.stats.transition_final_time = state.sim_time;
}

void GV2VirtualEnvironment::apply_controller_action_inplace(
    SimState& state,
    const ControllerAction& action,
    bool fast_forward) const {
    init_clock_if_needed(state);
    const double tick_before = next_adv_tick_state(state);
    state.stats.transition_discount_time = state.sim_time;
    state.stats.transition_final_time = state.sim_time;

    DecodeCreditLedger ledger(state.stats.decode_credit_balance);
    ledger.load_from_stats(state.stats);

    if (has_pending_adv_tick(state)) {
        refresh_request_and_stats_post_step(state, &ledger);
        const double action_time = state.sim_time;
        state.stats.transition_discount_time = action_time;
        state.stats.transition_final_time = action_time;
        return;
    }

    apply_controller_eviction_rule(state, action, &ledger);

    if (state.stats.active_request_ids.empty()) {
        refresh_request_and_stats_post_step(state, &ledger);
        const double action_end_time = state.sim_time;

        const bool pending_after_controller = has_pending_adv_tick(state);
        if (fast_forward && !pending_after_controller) {
            (void)maybe_fast_forward_decode_only_to_next_adv_tick(state, &ledger);
        }
        const double final_time = state.sim_time;
        state.stats.transition_discount_time = action_end_time;
        state.stats.transition_final_time = final_time;

        const int miss_src = (final_time > tick_before + cfg_.eps) ? 2 : 0;
        state.stats.missed_adv_source = miss_src;
        return;
    }

    const int decode_credit_limit = cfg_.enforce_nonnegative_decode_credits
        ? std::max(0, ledger.available_balance(true))
        : std::numeric_limits<int>::max();

    ControllerBatchPlan plan = virtual_sim_.build_controller_batch_plan(
        state,
        action,
        cfg_.enforce_nonnegative_decode_credits,
        decode_credit_limit);

    const double batch_start = state.sim_time;
    if (!plan.predictor_reqs.empty()) {
        (void)virtual_sim_.execute_controller_batch_timing(state, plan);
    }
    const double batch_end = state.sim_time;

    apply_batch_progress(state, plan, batch_start, batch_end, &ledger);
    refresh_request_and_stats_post_step(state, &ledger);

    const bool controller_crossed_adv_tick = (batch_end > tick_before + cfg_.eps);
    int miss_src = controller_crossed_adv_tick ? 1 : 0;
    const bool pending_after_controller = controller_crossed_adv_tick || has_pending_adv_tick(state);

    if (fast_forward && !pending_after_controller) {
        if (!has_active_prefill(state)) {
            (void)maybe_fast_forward_decode_only_to_next_adv_tick(state, &ledger);
        } else if (
            cfg_.controller_noop_prefill_only_jump_to_next_adv_tick &&
            is_controller_strict_noop(action) &&
            !has_active_decode(state)) {
            const double next_tick = next_adv_tick_state(state);
            if (state.sim_time + cfg_.eps < next_tick) {
                state.sim_time = next_tick;
                refresh_request_and_stats_post_step(state, &ledger);
            }
        }
    }

    const double final_time = state.sim_time;
    state.stats.transition_discount_time = batch_end;
    state.stats.transition_final_time = final_time;

    if (miss_src == 0 && final_time > tick_before + cfg_.eps) {
        miss_src = 2;
    }
    state.stats.missed_adv_source = miss_src;

    ledger.write_back_stats(&state.stats, cfg_.enforce_nonnegative_decode_credits);
    state.stats.pending_adv_tick = has_pending_adv_tick(state);
}

void GV2VirtualEnvironment::rebuild_active_completed_ids(SimState& s) {
    s.stats.active_request_ids.clear();
    s.stats.completed_request_ids.clear();
    s.stats.dropped_request_ids.clear();
    s.stats.stopped_decode_request_ids.clear();
    s.stats.violated_request_ids.clear();

    for (const auto& r : s.requests) {
        if (r.completed) {
            s.stats.completed_request_ids.push_back(r.request_id);
            if (r.dropped) s.stats.dropped_request_ids.push_back(r.request_id);
            if (r.stopped_decode) s.stats.stopped_decode_request_ids.push_back(r.request_id);
            if (r.violated) s.stats.violated_request_ids.push_back(r.request_id);
        } else if (r.prefill_active() || r.decode_active()) {
            s.stats.active_request_ids.push_back(r.request_id);
            if (r.violated) s.stats.violated_request_ids.push_back(r.request_id);
        }
    }

    std::sort(s.stats.active_request_ids.begin(), s.stats.active_request_ids.end());
    s.stats.active_request_ids.erase(
        std::unique(s.stats.active_request_ids.begin(), s.stats.active_request_ids.end()),
        s.stats.active_request_ids.end());

    std::sort(s.stats.completed_request_ids.begin(), s.stats.completed_request_ids.end());
    s.stats.completed_request_ids.erase(
        std::unique(s.stats.completed_request_ids.begin(), s.stats.completed_request_ids.end()),
        s.stats.completed_request_ids.end());

    std::sort(s.stats.dropped_request_ids.begin(), s.stats.dropped_request_ids.end());
    s.stats.dropped_request_ids.erase(
        std::unique(s.stats.dropped_request_ids.begin(), s.stats.dropped_request_ids.end()),
        s.stats.dropped_request_ids.end());

    std::sort(s.stats.stopped_decode_request_ids.begin(), s.stats.stopped_decode_request_ids.end());
    s.stats.stopped_decode_request_ids.erase(
        std::unique(s.stats.stopped_decode_request_ids.begin(), s.stats.stopped_decode_request_ids.end()),
        s.stats.stopped_decode_request_ids.end());

    std::sort(s.stats.violated_request_ids.begin(), s.stats.violated_request_ids.end());
    s.stats.violated_request_ids.erase(
        std::unique(s.stats.violated_request_ids.begin(), s.stats.violated_request_ids.end()),
        s.stats.violated_request_ids.end());

    if (s.stats.requests_completed < static_cast<int>(s.stats.completed_request_ids.size())) {
        s.stats.requests_completed = static_cast<int>(s.stats.completed_request_ids.size());
    }
}

std::pair<int, double> GV2VirtualEnvironment::evaluate_objective(const SimState& state) {
    return {state.stats.slo_violations, state.stats.slo_lateness_sum};
}

}  // namespace mcts_native_gv2
