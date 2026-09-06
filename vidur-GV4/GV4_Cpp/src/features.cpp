#include "gv4/features.hpp"

#include "gv4/engine.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace gv4 {
namespace {

constexpr double kWorkloadWindowMultiplier = 20.0;
constexpr RequestLifecycle kFeatureLifecycles[] = {
    RequestLifecycle::WaitingPrefill,
    RequestLifecycle::InflightPrefill,
    RequestLifecycle::WaitingDecode,
    RequestLifecycle::InflightDecode,
    RequestLifecycle::StopPending,
    RequestLifecycle::DropPending,
    RequestLifecycle::InflightRecompute,
    RequestLifecycle::PreemptPending,
};

std::vector<std::string> request_names(int stages) {
    std::vector<std::string> names{
        "decode_phase", "prefill_total", "prefill_committed", "prefill_remaining",
        "decode_total", "decode_committed", "decode_remaining", "committed_context",
        "kv_computed_context", "recompute_remaining",
        "arrival_age", "current_lateness", "prefill_deadline_delta",
        "decode_deadline_present", "decode_deadline_delta", "violated",
        "lifecycle_waiting_prefill", "lifecycle_inflight_prefill",
        "lifecycle_waiting_decode", "lifecycle_inflight_decode",
        "lifecycle_stop_pending", "lifecycle_drop_pending",
        "lifecycle_inflight_recompute", "lifecycle_preempt_pending",
        "reserved_tokens",
        "partial_block_used_fraction", "tokens_until_next_block_fraction",
        "has_inflight_work"};
    for (int stage = 0; stage < stages; ++stage) {
        names.push_back("active_pipeline_stage_" + std::to_string(stage));
    }
    names.insert(names.end(), {
        "pipeline_wait_or_transfer", "inflight_prefill_tokens",
        "inflight_decode_token", "inflight_recompute_tokens",
        "pending_stop", "pending_drop",
        "pending_termination_age"});
    return names;
}

std::vector<std::string> replica_names(int stages) {
    std::vector<std::string> names{
        "committed_logical_kv_blocks", "reserved_logical_kv_blocks",
        "free_logical_kv_blocks", "min_rank_free_fraction",
        "mean_rank_free_fraction", "max_rank_free_fraction",
        "inflight_microbatch_count"};
    for (int stage = 0; stage < stages; ++stage) {
        names.push_back("stage_" + std::to_string(stage) + "_free");
    }
    return names;
}

std::vector<std::string> microbatch_names(int stages) {
    std::vector<std::string> names;
    for (int stage = 0; stage < stages; ++stage) {
        names.push_back("active_stage_" + std::to_string(stage));
    }
    names.insert(names.end(), {
        "waiting_or_transfer", "prefill_request_count", "decode_request_count",
        "recompute_request_count", "prefill_tokens", "decode_tokens",
        "recompute_tokens", "prefill_reserved_kv_blocks",
        "decode_reserved_kv_blocks", "recompute_reserved_kv_blocks",
        "violated_prefill_request_count", "violated_decode_request_count",
        "violated_recompute_request_count"});
    return names;
}

FeatureLayout make_layout(const Config& config) {
    FeatureLayout result;
    result.schema_version = config.feature_schema_version;
    result.pipeline_stage_count = config.pipeline_parallel_size;
    result.global_names = {
        "active_request_count", "active_prefill_count", "active_decode_count",
        "remaining_prefill_tokens", "remaining_decode_tokens",
        "committed_prefill_tokens", "committed_decode_tokens",
        "committed_context_tokens", "decode_credit_balance",
        "decode_credits_reserved", "next_adversary_tick_delta",
        "logical_tokens_free_fraction", "waiting_prefill_count",
        "inflight_prefill_count", "waiting_decode_count", "inflight_decode_count",
        "stop_pending_count", "drop_pending_count", "inflight_recompute_count",
        "preempt_pending_count",
        "active_violation_fraction",
        "active_prefill_violation_fraction", "active_decode_violation_fraction",
        "waiting_prefill_violation_fraction", "inflight_prefill_violation_fraction",
        "waiting_decode_violation_fraction", "inflight_decode_violation_fraction",
        "stop_pending_violation_fraction", "drop_pending_violation_fraction",
        "inflight_recompute_violation_fraction",
        "preempt_pending_violation_fraction"};
    result.request_names = request_names(config.pipeline_parallel_size);
    result.launch_names = {"launch_age", "request_count", "prefill_tokens"};
    result.replica_names = replica_names(config.pipeline_parallel_size);
    result.microbatch_names = microbatch_names(config.pipeline_parallel_size);
    result.controller_header_names = {
        "transition_wait", "transition_evict_only", "transition_batch",
        "transition_preempt_only", "transition_evict_and_preempt",
        "evicted_prefill_count", "evicted_decode_count",
        "preempted_prefill_count", "preempted_decode_count",
        "preempted_inflight_count", "decode_request_count",
        "total_allocated_prefill_tokens",
        "total_allocated_recompute_tokens"};
    result.controller_request_names = {
        "allocated_prefill", "allocated_recompute", "evicted", "preempted",
        "allocated_prefill_tokens", "allocated_recompute_tokens",
        "remaining_prefill_tokens", "remaining_recompute_tokens",
        "total_prefill_tokens", "arrival_age", "current_lateness",
        "violated", "new_reserved_kv_blocks", "already_inflight"};
    result.adversary_header_names = {
        "launched_request_count", "prefill_tokens_per_launched_request",
        "total_launched_prefill_tokens", "stopped_decode_count",
        "stopped_inflight_decode_count"};
    result.adversary_request_names = {
        "decode_total", "decode_committed", "decode_remaining",
        "current_lateness", "decode_deadline_present", "decode_deadline_delta",
        "violated", "reserved_decode_token", "inflight"};
    return result;
}

FeatureScales make_scales(const Config& config) {
    FeatureScales scales;
    scales.window_request_cap = config.max_requests_per_launch_window;
    scales.window_prefill_cap =
        config.target_prefill_tokens_window_average * scales.window_request_cap;
    scales.active_request_scale =
        kWorkloadWindowMultiplier * scales.window_request_cap;
    scales.system_prefill_scale =
        kWorkloadWindowMultiplier * scales.window_prefill_cap;
    scales.system_decode_scale =
        scales.active_request_scale * config.target_decode_tokens_average;
    scales.request_prefill_scale = config.max_prefill_tokens_per_request;
    scales.request_decode_scale = config.max_decode_tokens_per_request;
    scales.decode_credit_scale = config.target_decode_tokens_average;
    scales.adversary_time_scale = config.adversary_tick_sec;
    scales.launch_age_scale = config.launch_window_sec;
    scales.lateness_scale = config.lateness_cap_sec;
    scales.block_token_scale = config.block_size_tokens;
    scales.controller_prefill_action_scale = *std::max_element(
        config.controller_actions.prefill_budgets.begin(),
        config.controller_actions.prefill_budgets.end());
    if (scales.controller_prefill_action_scale <= 0.0) {
        throw std::invalid_argument("feature schema needs a positive prefill action");
    }
    scales.controller_kv_block_scale = std::ceil(
        scales.controller_prefill_action_scale / scales.block_token_scale);
    scales.system_logical_blocks = config.logical_kv_capacity_blocks();
    scales.system_logical_tokens =
        scales.system_logical_blocks * scales.block_token_scale;
    return scales;
}

std::vector<float> vector32(const std::vector<double>& values) {
    std::vector<float> result;
    result.reserve(values.size());
    for (const double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("feature vector contains a non-finite value");
        }
        result.push_back(static_cast<float>(value));
    }
    return result;
}

FeatureMatrix matrix32(
    const std::vector<std::vector<double>>& rows,
    int columns) {
    FeatureMatrix result;
    result.rows = static_cast<int>(rows.size());
    result.columns = columns;
    result.values.reserve(rows.size() * static_cast<std::size_t>(columns));
    for (const auto& row : rows) {
        if (static_cast<int>(row.size()) != columns) {
            throw std::logic_error("feature row width does not match schema");
        }
        std::vector<float> converted = vector32(row);
        result.values.insert(result.values.end(), converted.begin(), converted.end());
    }
    return result;
}

double fraction(double numerator, double denominator) {
    return denominator == 0.0 ? 0.0 : numerator / denominator;
}

bool decode_phase(const RequestState& request) {
    if (request.lifecycle == RequestLifecycle::WaitingDecode ||
        request.lifecycle == RequestLifecycle::InflightDecode) {
        return true;
    }
    if (request.lifecycle == RequestLifecycle::InflightRecompute) {
        return request.is_decode_phase();
    }
    if (request.lifecycle == RequestLifecycle::PreemptPending) {
        return request.reserved_decode_tokens > 0 ||
               (request.reserved_recompute_tokens > 0 &&
                request.is_decode_phase());
    }
    if (request.lifecycle == RequestLifecycle::StopPending ||
        request.lifecycle == RequestLifecycle::DropPending) {
        return request.reserved_decode_tokens > 0 ||
               (request.reserved_recompute_tokens > 0 &&
                request.is_decode_phase());
    }
    return false;
}

double current_lateness(const RequestState& request, double now) {
    if (decode_phase(request)) {
        return request.prefill_lateness_sec + request.decode_lateness_sec;
    }
    return std::max({request.prefill_lateness_sec, now - request.prefill_deadline, 0.0});
}

int active_stage(const InflightMicrobatch& batch, double now, double epsilon) {
    for (std::size_t stage = 0; stage < batch.stage_start_times.size(); ++stage) {
        if (now + epsilon >= batch.stage_start_times[stage] &&
            now < batch.stage_finish_times[stage] - epsilon) {
            return static_cast<int>(stage);
        }
    }
    return kNoId;
}

int final_block_used(const RequestState& request, int block_size) {
    if (request.resident_tokens() == 0) {
        return 0;
    }
    const int remainder = request.resident_tokens() % block_size;
    return remainder == 0 ? block_size : remainder;
}

void check_state(const State& state, const Config& config) {
    state.validate(config);
    if (state.state_schema_version != config.state_schema_version ||
        state.config_manifest_sha256 != config.manifest_sha256) {
        throw std::invalid_argument("state does not match feature manifest");
    }
}

using BatchView = std::pair<const InflightMicrobatch*, int>;

std::vector<double> request_row(
    const RequestState& request,
    const State& state,
    const Config& config,
    const FeatureLayout& layout,
    const FeatureScales& scales,
    const BatchView* batch_view) {
    int stage = kNoId;
    if (request.has_inflight_work()) {
        if (batch_view == nullptr ||
            batch_view->first->microbatch_id != request.inflight_microbatch_id) {
            throw std::invalid_argument("in-flight request lacks its microbatch");
        }
        stage = batch_view->second;
    } else if (batch_view != nullptr) {
        throw std::invalid_argument("non-in-flight request has a microbatch");
    }

    const bool deadline_present = request.next_decode_deadline != kUnsetTime;
    const double deadline_delta = deadline_present
        ? std::asinh(
              (request.next_decode_deadline - state.now) /
              request.decode_token_slo_sec)
        : 0.0;
    const bool pending = request.lifecycle == RequestLifecycle::StopPending ||
                         request.lifecycle == RequestLifecycle::DropPending;
    double pending_age = 0.0;
    if (pending) {
        if (request.terminal_requested_at == kUnsetTime ||
            request.terminal_requested_at > state.now + config.epsilon) {
            throw std::invalid_argument("pending request has invalid terminal time");
        }
        pending_age = std::asinh(
            (state.now - request.terminal_requested_at) / scales.lateness_scale);
    }

    std::vector<double> row{
        static_cast<double>(decode_phase(request)),
        request.original_prefill_tokens / scales.request_prefill_scale,
        request.committed_prefill_tokens / scales.request_prefill_scale,
        request.remaining_prefill_tokens() / scales.request_prefill_scale,
        request.original_decode_tokens / scales.request_decode_scale,
        request.committed_decode_tokens / scales.request_decode_scale,
        request.remaining_decode_tokens() / scales.request_decode_scale,
        (request.committed_prefill_tokens + request.committed_decode_tokens) /
            (scales.request_prefill_scale + scales.request_decode_scale),
        request.kv_computed_tokens /
            (scales.request_prefill_scale + scales.request_decode_scale),
        request.remaining_recompute_tokens() /
            (scales.request_prefill_scale + scales.request_decode_scale),
        std::asinh((state.now - request.arrival_time) / scales.launch_age_scale),
        std::asinh(current_lateness(request, state.now) / scales.launch_age_scale),
        std::asinh((request.prefill_deadline - state.now) / scales.launch_age_scale),
        static_cast<double>(deadline_present),
        deadline_delta,
        static_cast<double>(request.violation_recorded)};
    for (const RequestLifecycle lifecycle : kFeatureLifecycles) {
        row.push_back(static_cast<double>(
            request.lifecycle == lifecycle));
    }
    const int used = final_block_used(request, config.block_size_tokens);
    const int until =
        used == 0 || used == config.block_size_tokens
            ? 0
            : config.block_size_tokens - used;
    row.push_back(
        (request.reserved_prefill_tokens + request.reserved_decode_tokens +
         request.reserved_recompute_tokens) /
        (scales.request_prefill_scale + scales.request_decode_scale));
    row.push_back(used / scales.block_token_scale);
    row.push_back(until / scales.block_token_scale);
    row.push_back(static_cast<double>(request.has_inflight_work()));
    for (int index = 0; index < layout.pipeline_stage_count; ++index) {
        row.push_back(static_cast<double>(stage == index));
    }
    row.push_back(static_cast<double>(request.has_inflight_work() && stage < 0));
    row.push_back(request.reserved_prefill_tokens / scales.request_prefill_scale);
    row.push_back(static_cast<double>(request.reserved_decode_tokens));
    row.push_back(
        request.reserved_recompute_tokens /
        (scales.request_prefill_scale + scales.request_decode_scale));
    row.push_back(static_cast<double>(request.lifecycle == RequestLifecycle::StopPending));
    row.push_back(static_cast<double>(request.lifecycle == RequestLifecycle::DropPending));
    row.push_back(pending_age);
    return row;
}

std::tuple<int, int, int> replica_usage(const ReplicaState& replica) {
    if (!std::all_of(
            replica.rank_kv_committed_blocks.begin(),
            replica.rank_kv_committed_blocks.end(),
            [&](int value) { return value == replica.rank_kv_committed_blocks[0]; }) ||
        !std::all_of(
            replica.rank_kv_reserved_blocks.begin(),
            replica.rank_kv_reserved_blocks.end(),
            [&](int value) { return value == replica.rank_kv_reserved_blocks[0]; })) {
        throw std::invalid_argument("logical KV is not mirrored across ranks");
    }
    return {
        replica.rank_kv_committed_blocks[0],
        replica.rank_kv_reserved_blocks[0],
        free_logical_blocks(replica)};
}

}  // namespace

FeatureBuilder::FeatureBuilder(Config config)
    : config_(std::move(config)),
      layout_(make_layout(config_)),
      scales_(make_scales(config_)) {
    config_.validate();
}

StateFeatures FeatureBuilder::build_state(const State& state) const {
    check_state(state, config_);
    std::vector<const RequestState*> live;
    live.reserve(state.requests.size());
    for (const RequestState& request : state.requests) {
        if (!is_terminal(request.lifecycle)) {
            if (request.arrival_time > state.now + config_.epsilon) {
                throw std::invalid_argument("request has a future arrival time");
            }
            live.push_back(&request);
        }
    }

    std::unordered_map<int, BatchView> by_request;
    std::unordered_map<int, int> batch_stage;
    for (const InflightMicrobatch& batch : state.replica.inflight_microbatches) {
        const int stage = active_stage(batch, state.now, config_.epsilon);
        batch_stage.emplace(batch.microbatch_id, stage);
        for (const BatchAllocation& allocation : batch.allocations) {
            if (!by_request.emplace(allocation.request_id, BatchView{&batch, stage}).second) {
                throw std::invalid_argument("request appears in two microbatches");
            }
        }
    }

    int prefills = 0;
    int decodes = 0;
    int violated = 0;
    int remaining_prefill = 0;
    int remaining_decode = 0;
    int committed_prefill = 0;
    int committed_decode = 0;
    int lifecycle_counts[8]{};
    int lifecycle_violations[8]{};
    int prefill_violations = 0;
    int decode_violations = 0;
    for (const RequestState* request : live) {
        const bool decode = decode_phase(*request);
        prefills += !decode;
        decodes += decode;
        remaining_prefill += request->remaining_prefill_tokens();
        remaining_decode += request->remaining_decode_tokens();
        committed_prefill += request->committed_prefill_tokens;
        committed_decode += request->committed_decode_tokens;
        const auto lifecycle_it = std::find(
            std::begin(kFeatureLifecycles), std::end(kFeatureLifecycles),
            request->lifecycle);
        if (lifecycle_it == std::end(kFeatureLifecycles)) {
            throw std::logic_error("active request has an unsupported lifecycle");
        }
        const int lifecycle = static_cast<int>(
            lifecycle_it - std::begin(kFeatureLifecycles));
        ++lifecycle_counts[lifecycle];
        if (request->violation_recorded) {
            ++violated;
            ++lifecycle_violations[lifecycle];
            prefill_violations += !decode;
            decode_violations += decode;
        }
    }

    const double count = static_cast<double>(live.size());
    std::vector<double> global{
        count / scales_.active_request_scale,
        prefills / scales_.active_request_scale,
        decodes / scales_.active_request_scale,
        remaining_prefill / scales_.system_prefill_scale,
        remaining_decode / scales_.system_decode_scale,
        committed_prefill / scales_.system_prefill_scale,
        committed_decode / scales_.system_decode_scale,
        (committed_prefill + committed_decode) / scales_.system_logical_tokens,
        std::asinh(state.decode_credits_available / scales_.decode_credit_scale),
        std::asinh(state.decode_credits_reserved / scales_.decode_credit_scale),
        std::asinh(
            (state.next_adversary_tick - state.now) / scales_.adversary_time_scale),
        (free_logical_blocks(state.replica) * scales_.block_token_scale) /
            scales_.system_logical_tokens};
    for (const int value : lifecycle_counts) {
        global.push_back(value / scales_.active_request_scale);
    }
    global.push_back(fraction(violated, count));
    global.push_back(fraction(prefill_violations, count));
    global.push_back(fraction(decode_violations, count));
    for (const int value : lifecycle_violations) {
        global.push_back(fraction(value, count));
    }

    std::vector<std::vector<double>> request_rows;
    request_rows.reserve(live.size());
    for (const RequestState* request : live) {
        const auto found = by_request.find(request->request_id);
        request_rows.push_back(request_row(
            *request,
            state,
            config_,
            layout_,
            scales_,
            found == by_request.end() ? nullptr : &found->second));
    }

    std::vector<std::vector<double>> launch_rows;
    const double cutoff = state.now - config_.launch_window_sec;
    double prior_time = -1.0;
    for (const LaunchRecord& record : state.launch_history) {
        if (record.launch_time < prior_time ||
            record.launch_time > state.now + config_.epsilon) {
            throw std::invalid_argument("launch history is not chronological");
        }
        prior_time = record.launch_time;
        if (record.launch_time <= cutoff + config_.epsilon) {
            continue;
        }
        launch_rows.push_back({
            std::asinh((state.now - record.launch_time) / scales_.launch_age_scale),
            record.request_count / scales_.window_request_cap,
            record.prefill_tokens / scales_.window_prefill_cap});
    }

    const auto [committed, reserved, free] = replica_usage(state.replica);
    std::vector<double> free_fractions;
    for (std::size_t rank = 0; rank < state.replica.rank_ids.size(); ++rank) {
        const int available = state.replica.rank_kv_capacity_blocks[rank] -
                              state.replica.rank_kv_committed_blocks[rank] -
                              state.replica.rank_kv_reserved_blocks[rank];
        free_fractions.push_back(
            static_cast<double>(available) /
            state.replica.rank_kv_capacity_blocks[rank]);
    }
    std::vector<bool> occupied(layout_.pipeline_stage_count, false);
    for (const auto& [id, stage] : batch_stage) {
        static_cast<void>(id);
        if (stage >= 0) {
            occupied[static_cast<std::size_t>(stage)] = true;
        }
    }
    std::vector<double> replica_row{
        committed / scales_.system_logical_blocks,
        reserved / scales_.system_logical_blocks,
        free / scales_.system_logical_blocks,
        *std::min_element(free_fractions.begin(), free_fractions.end()),
        0.0,
        *std::max_element(free_fractions.begin(), free_fractions.end()),
        state.replica.inflight_count() /
            static_cast<double>(config_.max_inflight_microbatches)};
    for (const double value : free_fractions) {
        replica_row[4] += value / free_fractions.size();
    }
    for (const bool value : occupied) {
        replica_row.push_back(static_cast<double>(!value));
    }

    std::vector<std::vector<double>> microbatch_rows;
    for (const InflightMicrobatch& batch : state.replica.inflight_microbatches) {
        const int stage = batch_stage.at(batch.microbatch_id);
        std::vector<double> row;
        for (int index = 0; index < layout_.pipeline_stage_count; ++index) {
            row.push_back(static_cast<double>(stage == index));
        }
        int prefill_count = 0;
        int decode_count = 0;
        int recompute_count = 0;
        int prefill_tokens = 0;
        int decode_tokens = 0;
        int recompute_tokens = 0;
        int prefill_blocks = 0;
        int decode_blocks = 0;
        int recompute_blocks = 0;
        int prefill_violated = 0;
        int decode_violated = 0;
        int recompute_violated = 0;
        for (const BatchAllocation& allocation : batch.allocations) {
            const bool prefill = allocation.prefill_tokens > 0;
            const bool decode = allocation.decode_tokens > 0;
            const bool recompute = allocation.recompute_tokens > 0;
            prefill_count += prefill;
            decode_count += decode;
            recompute_count += recompute;
            prefill_tokens += allocation.prefill_tokens;
            decode_tokens += allocation.decode_tokens;
            recompute_tokens += allocation.recompute_tokens;
            prefill_blocks += prefill ? allocation.new_kv_blocks : 0;
            decode_blocks += decode ? allocation.new_kv_blocks : 0;
            recompute_blocks += recompute ? allocation.new_kv_blocks : 0;
            const bool request_violated =
                state.request(allocation.request_id).violation_recorded;
            prefill_violated += prefill && request_violated;
            decode_violated += decode && request_violated;
            recompute_violated += recompute && request_violated;
        }
        row.insert(row.end(), {
            static_cast<double>(stage < 0),
            prefill_count / scales_.active_request_scale,
            decode_count / scales_.active_request_scale,
            recompute_count / scales_.active_request_scale,
            prefill_tokens / scales_.system_prefill_scale,
            decode_tokens / scales_.system_decode_scale,
            recompute_tokens / scales_.system_logical_tokens,
            prefill_blocks / scales_.system_logical_blocks,
            decode_blocks / scales_.system_logical_blocks,
            recompute_blocks / scales_.system_logical_blocks,
            prefill_violated / scales_.active_request_scale,
            decode_violated / scales_.active_request_scale,
            recompute_violated / scales_.active_request_scale});
        microbatch_rows.push_back(std::move(row));
    }

    StateFeatures result;
    result.schema_version = layout_.schema_version;
    result.config_manifest_sha256 = config_.manifest_sha256;
    result.global_features = vector32(global);
    result.request_rows = matrix32(
        request_rows, static_cast<int>(layout_.request_names.size()));
    result.request_replica_offsets = {0, static_cast<int>(request_rows.size())};
    result.launch_rows = matrix32(
        launch_rows, static_cast<int>(layout_.launch_names.size()));
    result.replica_rows = matrix32(
        {replica_row}, static_cast<int>(layout_.replica_names.size()));
    result.microbatch_rows = matrix32(
        microbatch_rows, static_cast<int>(layout_.microbatch_names.size()));
    result.microbatch_replica_offsets = {
        0, static_cast<int>(microbatch_rows.size())};
    return result;
}

ControllerActionFeatures FeatureBuilder::build_controller_action(
    const State& state,
    const CanonicalControllerAction& edge) const {
    check_state(state, config_);
    if (state.next_player != Player::Controller) {
        throw std::invalid_argument("controller features require controller turn");
    }
    const ResolvedControllerAction& action = edge.action;
    int evicted_prefill = 0;
    int evicted_decode = 0;
    int preempted_prefill = 0;
    int preempted_decode = 0;
    int preempted_inflight = 0;
    using Affected = std::tuple<
        const RequestState*, int, int, int, bool, bool>;
    std::map<int, Affected> affected;
    for (const int request_id : action.evicted_request_ids) {
        const RequestState& request = state.request(request_id);
        evicted_decode += decode_phase(request);
        evicted_prefill += !decode_phase(request);
        affected.emplace(
            request_id, Affected{&request, 0, 0, 0, true, false});
    }
    for (const int request_id : action.preempted_request_ids) {
        const RequestState& request = state.request(request_id);
        preempted_decode += decode_phase(request);
        preempted_prefill += !decode_phase(request);
        preempted_inflight += request.has_inflight_work();
        if (!affected.emplace(
                request_id,
                Affected{&request, 0, 0, 0, false, true}).second) {
            throw std::invalid_argument("action affects one request twice");
        }
    }
    int decode_allocations = 0;
    for (const BatchAllocation& allocation : action.allocations) {
        decode_allocations += allocation.decode_tokens > 0;
        if (allocation.prefill_tokens > 0 || allocation.recompute_tokens > 0) {
            const RequestState& request = state.request(allocation.request_id);
            if (!affected.emplace(
                    request.request_id,
                    Affected{
                        &request,
                        allocation.prefill_tokens,
                        allocation.recompute_tokens,
                        allocation.new_kv_blocks,
                        false,
                        false}).second) {
                throw std::invalid_argument("action affects one request twice");
            }
        }
    }
    std::vector<double> header;
    for (int kind = 0; kind < 5; ++kind) {
        header.push_back(static_cast<double>(
            static_cast<int>(action.transition_kind) == kind));
    }
    header.insert(header.end(), {
        evicted_prefill / scales_.active_request_scale,
        evicted_decode / scales_.active_request_scale,
        preempted_prefill / scales_.active_request_scale,
        preempted_decode / scales_.active_request_scale,
        preempted_inflight / scales_.active_request_scale,
        decode_allocations / scales_.active_request_scale,
        action.total_prefill_tokens() / scales_.controller_prefill_action_scale,
        action.total_recompute_tokens() /
            scales_.controller_prefill_action_scale});

    std::vector<std::vector<double>> rows;
    for (const auto& [request_id, values] : affected) {
        static_cast<void>(request_id);
        const auto [request, prefill, recompute, blocks, evicted, preempted] =
            values;
        rows.push_back({
            static_cast<double>(prefill > 0),
            static_cast<double>(recompute > 0),
            static_cast<double>(evicted),
            static_cast<double>(preempted),
            prefill / scales_.controller_prefill_action_scale,
            recompute / scales_.controller_prefill_action_scale,
            request->remaining_prefill_tokens() / scales_.request_prefill_scale,
            request->remaining_recompute_tokens() /
                (scales_.request_prefill_scale + scales_.request_decode_scale),
            request->original_prefill_tokens / scales_.request_prefill_scale,
            std::asinh((state.now - request->arrival_time) / scales_.launch_age_scale),
            std::asinh(current_lateness(*request, state.now) / scales_.launch_age_scale),
            static_cast<double>(request->violation_recorded),
            blocks / scales_.controller_kv_block_scale,
            static_cast<double>(request->has_inflight_work())});
    }
    return ControllerActionFeatures{
        vector32(header),
        matrix32(rows, static_cast<int>(layout_.controller_request_names.size()))};
}

AdversaryActionFeatures FeatureBuilder::build_adversary_action(
    const State& state,
    const CanonicalAdversaryAction& edge) const {
    check_state(state, config_);
    if (state.next_player != Player::Adversary) {
        throw std::invalid_argument("adversary features require adversary turn");
    }
    const ResolvedAdversaryAction& action = edge.action;
    std::vector<const RequestState*> stopped;
    for (const int request_id : action.stop_request_ids) {
        const RequestState& request = state.request(request_id);
        if (!decode_phase(request)) {
            throw std::invalid_argument("adversary stop target is not decode");
        }
        stopped.push_back(&request);
    }
    std::sort(stopped.begin(), stopped.end(), [](const auto* left, const auto* right) {
        return left->request_id < right->request_id;
    });
    const int inflight_stopped = static_cast<int>(std::count_if(
        stopped.begin(), stopped.end(), [](const RequestState* request) {
            return request->has_inflight_work();
        }));
    std::vector<double> header{
        action.launch_count / scales_.window_request_cap,
        action.prefill_tokens / scales_.request_prefill_scale,
        (action.launch_count * action.prefill_tokens) / scales_.window_prefill_cap,
        stopped.size() / scales_.window_request_cap,
        inflight_stopped / scales_.window_request_cap};
    std::vector<std::vector<double>> rows;
    for (const RequestState* request : stopped) {
        const bool deadline = request->next_decode_deadline != kUnsetTime;
        rows.push_back({
            request->original_decode_tokens / scales_.request_decode_scale,
            request->committed_decode_tokens / scales_.request_decode_scale,
            request->remaining_decode_tokens() / scales_.request_decode_scale,
            std::asinh(current_lateness(*request, state.now) / scales_.launch_age_scale),
            static_cast<double>(deadline),
            deadline
                ? std::asinh(
                      (request->next_decode_deadline - state.now) /
                      request->decode_token_slo_sec)
                : 0.0,
            static_cast<double>(request->violation_recorded),
            static_cast<double>(request->reserved_decode_tokens),
            static_cast<double>(request->has_inflight_work())});
    }
    return AdversaryActionFeatures{
        vector32(header),
        matrix32(rows, static_cast<int>(layout_.adversary_request_names.size()))};
}

}  // namespace gv4
