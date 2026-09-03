#include "gv4/engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace gv4 {
namespace {

void validate_if_enabled(const State& state, const Config& config) {
    if (config.enable_debug_asserts) state.validate(config);
}

TransitionOutcome outcome(
    const State& state,
    const Config& config,
    const std::string& transition_kind,
    double started_at,
    double objective_before) {
    const double elapsed = std::max(0.0, state.now - started_at);
    return {
        transition_kind,
        elapsed,
        objective_before,
        state.objective.total_cost,
        objective_before - state.objective.total_cost,
        config.discount_for_elapsed(elapsed),
    };
}

void refresh_objective(State& state, const Config& config) {
    ObjectiveState objective;
    objective.requests_generated = static_cast<int>(state.requests.size());
    for (const auto& request : state.requests) {
        if (request.lifecycle == RequestLifecycle::Completed) ++objective.requests_completed;
        else if (request.lifecycle == RequestLifecycle::Stopped) ++objective.requests_stopped;
        else if (request.lifecycle == RequestLifecycle::Dropped) ++objective.requests_dropped;

        const bool dropped = request.lifecycle == RequestLifecycle::DropPending ||
                             request.lifecycle == RequestLifecycle::Dropped;
        if (dropped) {
            objective.terminal_cost += config.terminal_drop_cost;
            objective.total_cost += config.terminal_drop_cost;
            continue;
        }
        objective.prefill_lateness_sec += request.prefill_lateness_sec;
        objective.decode_lateness_sec += request.decode_lateness_sec;
        if (request.violation_recorded) {
            ++objective.slo_violations;
            const double lateness = request.prefill_lateness_sec +
                                    request.decode_lateness_sec;
            objective.total_cost += config.violation_base_cost +
                                    std::min(lateness, config.lateness_cap_sec);
        }
    }
    state.objective = objective;
}

void record_prefill_lateness(
    RequestState& request, double at_time, const Config& config) {
    request.prefill_lateness_sec = std::max(
        request.prefill_lateness_sec,
        std::max(0.0, at_time - request.prefill_deadline));
    if (request.prefill_lateness_sec > config.epsilon) {
        request.violation_recorded = true;
    }
}

void prune_launch_history(State& state, const Config& config, double at_time) {
    const double cutoff = at_time - config.launch_window_sec;
    state.launch_history.erase(
        std::remove_if(
            state.launch_history.begin(), state.launch_history.end(),
            [&](const LaunchRecord& record) {
                return record.launch_time <= cutoff + config.epsilon;
            }),
        state.launch_history.end());
}

void release_request_blocks(State& state, RequestState& request) {
    if (request.has_inflight_work() || request.reserved_kv_blocks != 0) {
        throw std::logic_error("cannot release in-flight or reserved request KV");
    }
    const int released = request.committed_kv_blocks;
    for (int& value : state.replica.rank_kv_committed_blocks) {
        if (value < released) throw std::logic_error("rank has fewer blocks than request");
        value -= released;
    }
    request.committed_kv_blocks = 0;
}

void release_and_finish(
    State& state,
    RequestState& request,
    RequestLifecycle lifecycle,
    TerminalReason reason,
    double at_time) {
    if (request.has_inflight_work()) {
        throw std::logic_error("cannot physically remove in-flight request");
    }
    release_request_blocks(state, request);
    request.lifecycle = lifecycle;
    request.terminal_reason = reason;
    request.terminal_time = at_time;
    request.next_decode_deadline = kUnsetTime;
}

void mark_drop(
    State& state,
    RequestState& request,
    TerminalReason reason,
    double at_time) {
    if (request.lifecycle == RequestLifecycle::DropPending ||
        request.lifecycle == RequestLifecycle::Dropped) return;
    request.prefill_lateness_sec = 0.0;
    request.decode_lateness_sec = 0.0;
    request.violation_recorded = false;
    request.terminal_reason = reason;
    request.terminal_requested_at = at_time;
    if (request.has_inflight_work()) {
        request.lifecycle = RequestLifecycle::DropPending;
    } else {
        release_and_finish(
            state, request, RequestLifecycle::Dropped, reason, at_time);
    }
}

void mark_stop(
    State& state,
    RequestState& request,
    double at_time,
    TerminalReason reason = TerminalReason::AdversaryStop) {
    request.terminal_reason = reason;
    request.terminal_requested_at = at_time;
    if (request.has_inflight_work()) {
        request.lifecycle = RequestLifecycle::StopPending;
    } else {
        release_and_finish(
            state, request, RequestLifecycle::Stopped, reason, at_time);
    }
}

void stop_decodes_after_credit_exhaustion(State& state, double at_time) {
    if (state.decode_credits_available != 0) return;
    for (auto& request : state.requests) {
        if (request.lifecycle == RequestLifecycle::WaitingDecode ||
            request.lifecycle == RequestLifecycle::InflightDecode) {
            mark_stop(state, request, at_time, TerminalReason::DecodeCreditExhausted);
        }
    }
}

void finish_naturally(State& state, RequestState& request, double at_time) {
    release_and_finish(state, request, RequestLifecycle::Completed,
                       TerminalReason::NaturalCompletion, at_time);
}

void commit_batch_blocks(State& state, const InflightMicrobatch& batch) {
    int total_blocks = 0;
    for (const auto& allocation : batch.allocations) {
        auto& request = state.request(allocation.request_id);
        if (request.reserved_kv_blocks != allocation.new_kv_blocks) {
            throw std::logic_error("request block reservation differs from batch");
        }
        total_blocks += allocation.new_kv_blocks;
    }
    for (int reserved : state.replica.rank_kv_reserved_blocks) {
        if (reserved < total_blocks) {
            throw std::logic_error("rank has fewer reserved blocks than batch");
        }
    }
    for (const auto& allocation : batch.allocations) {
        auto& request = state.request(allocation.request_id);
        request.reserved_kv_blocks -= allocation.new_kv_blocks;
        request.committed_kv_blocks += allocation.new_kv_blocks;
    }
    for (std::size_t rank = 0;
         rank < state.replica.rank_kv_reserved_blocks.size(); ++rank) {
        state.replica.rank_kv_reserved_blocks[rank] -= total_blocks;
        state.replica.rank_kv_committed_blocks[rank] += total_blocks;
    }
}

void complete_microbatch(State& state, const Config& config, int microbatch_id) {
    const InflightMicrobatch* stored = state.replica.find_microbatch(microbatch_id);
    if (stored == nullptr || stored->completion_applied) {
        throw std::logic_error("microbatch completion is missing or already applied");
    }
    const InflightMicrobatch batch = *stored;
    const double completion_time = batch.final_completion_time();
    commit_batch_blocks(state, batch);
    if (batch.total_decode_tokens() > state.decode_credits_reserved) {
        throw std::logic_error("batch exceeds reserved decode credits");
    }
    state.decode_credits_reserved -= batch.total_decode_tokens();
    state.decode_tokens_committed_total += batch.total_decode_tokens();

    for (const auto& allocation : batch.allocations) {
        auto& request = state.request(allocation.request_id);
        const RequestLifecycle pending_lifecycle = request.lifecycle;
        request.committed_prefill_tokens += allocation.prefill_tokens;
        request.reserved_prefill_tokens -= allocation.prefill_tokens;
        request.committed_decode_tokens += allocation.decode_tokens;
        request.reserved_decode_tokens -= allocation.decode_tokens;
        request.inflight_microbatch_id = kNoId;

        if (pending_lifecycle == RequestLifecycle::DropPending) {
            release_and_finish(state, request, RequestLifecycle::Dropped,
                               request.terminal_reason, completion_time);
            continue;
        }
        if (pending_lifecycle == RequestLifecycle::StopPending) {
            if (allocation.decode_tokens && request.next_decode_deadline != kUnsetTime) {
                const double lateness = std::max(
                    0.0, completion_time - request.next_decode_deadline);
                request.decode_lateness_sec += lateness;
                request.violation_recorded = request.violation_recorded ||
                                             lateness > config.epsilon;
            }
            release_and_finish(state, request, RequestLifecycle::Stopped,
                               request.terminal_reason, completion_time);
            continue;
        }

        if (allocation.prefill_tokens > 0) {
            if (request.remaining_prefill_tokens() > 0) {
                request.lifecycle = RequestLifecycle::WaitingPrefill;
            } else {
                record_prefill_lateness(request, completion_time, config);
                request.lifecycle = RequestLifecycle::WaitingDecode;
                request.next_decode_deadline = round_time(
                    completion_time + request.decode_token_slo_sec,
                    config.time_round_digits);
                if (request.decode_credit_minted) {
                    throw std::logic_error("request minted decode credit twice");
                }
                request.decode_credit_minted = true;
                state.decode_credits_available += config.decode_credit_mint;
                state.decode_credits_minted_total += config.decode_credit_mint;
            }
        } else {
            if (request.next_decode_deadline == kUnsetTime) {
                throw std::logic_error("decode completion lacks deadline");
            }
            const double lateness = std::max(
                0.0, completion_time - request.next_decode_deadline);
            request.decode_lateness_sec += lateness;
            request.violation_recorded = request.violation_recorded ||
                                         lateness > config.epsilon;
            if (request.remaining_decode_tokens() > 0) {
                request.lifecycle = RequestLifecycle::WaitingDecode;
                request.next_decode_deadline = round_time(
                    completion_time + request.decode_token_slo_sec,
                    config.time_round_digits);
            } else {
                finish_naturally(state, request, completion_time);
            }
        }
    }

    auto& batches = state.replica.inflight_microbatches;
    const auto iterator = std::find_if(
        batches.begin(), batches.end(),
        [&](const InflightMicrobatch& item) {
            return item.microbatch_id == microbatch_id;
        });
    if (iterator == batches.end()) throw std::logic_error("completed batch disappeared");
    batches.erase(iterator);
    prune_launch_history(state, config, completion_time);
    refresh_objective(state, config);
}

void apply_automatic_drops(State& state, const Config& config, double at_time) {
    for (auto& request : state.requests) {
        if (request.lifecycle == RequestLifecycle::WaitingPrefill ||
            request.lifecycle == RequestLifecycle::InflightPrefill) {
            record_prefill_lateness(request, at_time, config);
        }
    }
    for (auto& request : state.requests) {
        if (is_terminal(request.lifecycle) ||
            request.lifecycle == RequestLifecycle::StopPending ||
            request.lifecycle == RequestLifecycle::DropPending) continue;
        if (request.prefill_lateness_sec + request.decode_lateness_sec >=
            config.automatic_drop_lateness_sec) {
            mark_drop(state, request, TerminalReason::AutomaticSloDrop, at_time);
        }
    }
    refresh_objective(state, config);
}

void advance_to_inplace(State& state, const Config& config, double target_time) {
    if (!std::isfinite(target_time) || target_time < 0.0) {
        throw std::invalid_argument("target time must be finite and nonnegative");
    }
    if (target_time + config.epsilon < state.now) {
        throw std::logic_error("cannot move simulator time backwards");
    }
    if (target_time > state.next_adversary_tick + config.epsilon) {
        throw std::logic_error("cannot advance past unprocessed adversary tick");
    }
    std::vector<std::pair<double, int>> completions;
    for (const auto& batch : state.replica.inflight_microbatches) {
        if (batch.final_completion_time() <= target_time + config.epsilon) {
            completions.emplace_back(batch.final_completion_time(), batch.microbatch_id);
        }
    }
    std::sort(completions.begin(), completions.end());
    for (const auto& [completion_time, microbatch_id] : completions) {
        state.now = completion_time;
        complete_microbatch(state, config, microbatch_id);
    }
    state.now = round_time(target_time, config.time_round_digits);
    prune_launch_history(state, config, state.now);
    apply_automatic_drops(state, config, state.now);
}

InflightMicrobatch build_microbatch_calendar(
    const ReplicaState& replica,
    int microbatch_id,
    int raw_action_index,
    int canonical_action_index,
    const std::vector<BatchAllocation>& allocations,
    double admitted_at,
    const BatchTiming& timing,
    const Config& config) {
    if (!can_admit_microbatch(replica, admitted_at, config)) {
        throw std::logic_error("pipeline cannot admit microbatch now");
    }
    if (microbatch_id < 0 || raw_action_index < 0 || canonical_action_index < 0) {
        throw std::logic_error("microbatch/action IDs must be nonnegative");
    }
    if (allocations.empty()) throw std::logic_error("microbatch cannot be empty");
    if (timing.stage_service_times.size() !=
            static_cast<std::size_t>(config.pipeline_parallel_size) ||
        timing.pp_communication_times.size() !=
            static_cast<std::size_t>(config.pipeline_parallel_size - 1)) {
        throw std::logic_error("timing width differs from PP topology");
    }
    InflightMicrobatch batch;
    batch.microbatch_id = microbatch_id;
    batch.raw_action_index = raw_action_index;
    batch.canonical_action_index = canonical_action_index;
    batch.allocations = allocations;
    double ready = admitted_at;
    for (int stage = 0; stage < config.pipeline_parallel_size; ++stage) {
        const double service = timing.stage_service_times[stage];
        if (!std::isfinite(service) || service <= 0.0) {
            throw std::logic_error("stage service time must be positive");
        }
        const double start = std::max(ready, replica.stage_tail_finish_times[stage]);
        const double finish = round_time(start + service, config.time_round_digits);
        if (finish <= start) throw std::logic_error("rounding removed stage duration");
        batch.stage_ready_times.push_back(ready);
        batch.stage_start_times.push_back(start);
        batch.stage_finish_times.push_back(finish);
        if (stage + 1 < config.pipeline_parallel_size) {
            const double communication = timing.pp_communication_times[stage];
            if (!std::isfinite(communication) || communication < 0.0) {
                throw std::logic_error("PP communication time is invalid");
            }
            ready = round_time(finish + communication, config.time_round_digits);
        }
    }
    return batch;
}

void reserve_batch_blocks(
    State& state,
    const std::vector<BatchAllocation>& allocations,
    const Config& config) {
    int total_blocks = 0;
    for (const auto& allocation : allocations) {
        auto& request = state.request(allocation.request_id);
        if (request.has_inflight_work() || request.reserved_kv_blocks != 0 ||
            is_terminal(request.lifecycle)) {
            throw std::logic_error("request cannot reserve batch blocks");
        }
        const int expected = additional_blocks_for_work(
            request, allocation.prefill_tokens, allocation.decode_tokens,
            config.block_size_tokens);
        if (expected != allocation.new_kv_blocks) {
            throw std::logic_error("allocation block demand is stale");
        }
        total_blocks += expected;
    }
    if (total_blocks > free_logical_blocks(state.replica)) {
        throw std::logic_error("batch exceeds KV capacity");
    }
    for (const auto& allocation : allocations) {
        state.request(allocation.request_id).reserved_kv_blocks +=
            allocation.new_kv_blocks;
    }
    for (int& value : state.replica.rank_kv_reserved_blocks) value += total_blocks;
}

const CanonicalControllerAction& current_controller_action(
    const State& state,
    const Config& config,
    const CanonicalControllerAction& stale,
    const PrefillTimeEstimator& estimator,
    ControllerActionSpace& storage) {
    storage = resolve_controller_actions(state, config, estimator);
    const int raw = stale.action.raw_action_index;
    if (raw < 0 || raw >= static_cast<int>(storage.raw_to_canonical.size()) ||
        storage.raw_to_canonical[raw] < 0) {
        throw std::logic_error("controller action is stale or masked");
    }
    const auto& current = storage.canonical_actions[storage.raw_to_canonical[raw]];
    if (!current.action.same_effect(stale.action)) {
        throw std::logic_error("controller action effects are stale");
    }
    return current;
}

const CanonicalAdversaryAction& current_adversary_action(
    const State& state,
    const Config& config,
    const CanonicalAdversaryAction& stale,
    AdversaryActionSpace& storage) {
    storage = resolve_adversary_actions(state, config);
    const int raw = stale.action.raw_action_index;
    if (raw < 0 || raw >= static_cast<int>(storage.raw_to_canonical.size()) ||
        storage.raw_to_canonical[raw] < 0) {
        throw std::logic_error("adversary action is stale or masked");
    }
    const auto& current = storage.canonical_actions[storage.raw_to_canonical[raw]];
    if (!current.action.same_effect(stale.action)) {
        throw std::logic_error("adversary action effects are stale");
    }
    return current;
}

bool has_active_prefill(const State& state) {
    for (const auto& request : state.requests) {
        if (!is_terminal(request.lifecycle) &&
            (request.remaining_prefill_tokens() > 0 ||
             request.reserved_prefill_tokens > 0)) return true;
    }
    return false;
}

bool has_inflight_decode(const State& state) {
    return std::any_of(
        state.requests.begin(), state.requests.end(),
        [](const RequestState& request) { return request.reserved_decode_tokens > 0; });
}

bool has_active_request(const State& state) {
    return std::any_of(
        state.requests.begin(), state.requests.end(),
        [](const RequestState& request) { return !is_terminal(request.lifecycle); });
}

}  // namespace

int blocks_for_tokens(int token_count, int block_size_tokens) {
    if (token_count < 0 || block_size_tokens <= 0) {
        throw std::invalid_argument("token count must be nonnegative and block size positive");
    }
    return (token_count + block_size_tokens - 1) / block_size_tokens;
}

int additional_blocks_for_work(
    const RequestState& request,
    int prefill_tokens,
    int decode_tokens,
    int block_size_tokens) {
    if (prefill_tokens < 0 || decode_tokens < 0 ||
        (prefill_tokens == 0 && decode_tokens == 0) ||
        (prefill_tokens > 0 && decode_tokens > 0)) {
        throw std::invalid_argument("allocation must contain one nonnegative work type");
    }
    if (prefill_tokens > request.remaining_prefill_tokens() ||
        decode_tokens > request.remaining_decode_tokens()) {
        throw std::logic_error("allocation exceeds remaining request work");
    }
    const int needed = blocks_for_tokens(
        request.resident_tokens() + prefill_tokens + decode_tokens,
        block_size_tokens);
    const int owned = request.committed_kv_blocks + request.reserved_kv_blocks;
    return std::max(0, needed - owned);
}

int free_logical_blocks(const ReplicaState& replica) {
    if (replica.rank_kv_capacity_blocks.empty()) {
        throw std::logic_error("replica has no ranks");
    }
    int least = std::numeric_limits<int>::max();
    for (std::size_t rank = 0; rank < replica.rank_kv_capacity_blocks.size(); ++rank) {
        const int free = replica.rank_kv_capacity_blocks[rank] -
                         replica.rank_kv_committed_blocks[rank] -
                         replica.rank_kv_reserved_blocks[rank];
        if (free < 0) throw std::logic_error("replica KV usage exceeds capacity");
        least = std::min(least, free);
    }
    return least;
}

bool can_admit_microbatch(
    const ReplicaState& replica, double admitted_at, const Config& config) {
    return replica.inflight_count() < config.max_inflight_microbatches &&
           replica.stage_tail_finish_times.front() <= admitted_at + config.epsilon;
}

double next_pipeline_admission_time(
    const ReplicaState& replica, double now, const Config& config) {
    double candidate = std::max(now, replica.stage_tail_finish_times.front());
    if (replica.inflight_count() == config.max_inflight_microbatches) {
        if (replica.inflight_microbatches.empty()) {
            throw std::logic_error("full in-flight count has no batch record");
        }
        candidate = std::max(
            candidate, replica.inflight_microbatches.front().final_completion_time());
    }
    return round_time(candidate, config.time_round_digits);
}

double next_internal_completion_time(const State& state) {
    double earliest = std::numeric_limits<double>::infinity();
    for (const auto& batch : state.replica.inflight_microbatches) {
        earliest = std::min(earliest, batch.final_completion_time());
    }
    return earliest;
}

double next_wait_boundary_time(const State& state, const Config& config) {
    double earliest = std::numeric_limits<double>::infinity();
    if (state.next_adversary_tick > state.now + config.epsilon) {
        earliest = std::min(earliest, state.next_adversary_tick);
    }
    const double stage_zero = state.replica.stage_tail_finish_times.front();
    if (stage_zero > state.now + config.epsilon) earliest = std::min(earliest, stage_zero);
    for (const auto& batch : state.replica.inflight_microbatches) {
        if (batch.final_completion_time() > state.now + config.epsilon) {
            earliest = std::min(earliest, batch.final_completion_time());
        }
    }
    if (!std::isfinite(earliest)) {
        throw std::logic_error("WAIT has no future enabling boundary");
    }
    return round_time(earliest, config.time_round_digits);
}

TransitionOutcome advance_to(State& state, const Config& config, double target_time) {
    validate_if_enabled(state, config);
    const double started_at = state.now;
    const double objective_before = state.objective.total_cost;
    advance_to_inplace(state, config, target_time);
    validate_if_enabled(state, config);
    return outcome(state, config, "ADVANCE", started_at, objective_before);
}

TransitionOutcome apply_controller_action(
    State& state,
    const Config& config,
    const CanonicalControllerAction& action,
    const BatchTiming& timing,
    const PrefillTimeEstimator& prefill_time_estimator) {
    validate_if_enabled(state, config);
    const double started_at = state.now;
    const double objective_before = state.objective.total_cost;
    ControllerActionSpace action_space;
    const auto& current = current_controller_action(
        state, config, action, prefill_time_estimator, action_space);
    const auto& resolved = current.action;

    InflightMicrobatch calendar;
    if (resolved.transition_kind == ControllerTransitionKind::Batch) {
        calendar = build_microbatch_calendar(
            state.replica, state.next_microbatch_id,
            resolved.raw_action_index, action.canonical_action_index,
            resolved.allocations, state.now, timing, config);
    } else if (!timing.stage_service_times.empty() ||
               !timing.pp_communication_times.empty()) {
        throw std::logic_error("non-batch action cannot carry timing");
    }

    for (int request_id : resolved.evicted_request_ids) {
        mark_drop(state, state.request(request_id),
                  TerminalReason::ControllerEviction, state.now);
    }

    if (resolved.transition_kind == ControllerTransitionKind::Batch) {
        reserve_batch_blocks(state, resolved.allocations, config);
        const int decode_tokens = resolved.total_decode_tokens();
        if (decode_tokens > state.decode_credits_available) {
            throw std::logic_error("batch exceeds available decode credits");
        }
        state.decode_credits_available -= decode_tokens;
        state.decode_credits_reserved += decode_tokens;
        for (const auto& allocation : resolved.allocations) {
            auto& request = state.request(allocation.request_id);
            request.reserved_prefill_tokens = allocation.prefill_tokens;
            request.reserved_decode_tokens = allocation.decode_tokens;
            request.inflight_microbatch_id = state.next_microbatch_id;
            request.lifecycle = allocation.prefill_tokens > 0
                ? RequestLifecycle::InflightPrefill
                : RequestLifecycle::InflightDecode;
        }
        state.replica.stage_tail_finish_times = calendar.stage_finish_times;
        std::fill(state.replica.stage_last_microbatch_ids.begin(),
                  state.replica.stage_last_microbatch_ids.end(),
                  state.next_microbatch_id);
        state.replica.inflight_microbatches.push_back(std::move(calendar));
        if (decode_tokens > 0 && state.decode_credits_available == 0) {
            stop_decodes_after_credit_exhaustion(state, state.now);
        }
        ++state.next_microbatch_id;
    } else if (resolved.transition_kind == ControllerTransitionKind::Wait) {
        advance_to_inplace(state, config, next_wait_boundary_time(state, config));
    }

    state.next_player = Player::Adversary;
    refresh_objective(state, config);
    validate_if_enabled(state, config);
    return outcome(state, config, transition_kind_name(resolved.transition_kind),
                   started_at, objective_before);
}

TransitionOutcome apply_adversary_action(
    State& state,
    const Config& config,
    const CanonicalAdversaryAction& action,
    const PrefillTimeEstimator& prefill_time_estimator) {
    validate_if_enabled(state, config);
    const double started_at = state.now;
    const double objective_before = state.objective.total_cost;
    AdversaryActionSpace action_space;
    const auto& current = current_adversary_action(state, config, action, action_space);
    const auto& resolved = current.action;

    if (state.now + config.epsilon < state.next_adversary_tick) {
        state.next_player = Player::Controller;
        validate_if_enabled(state, config);
        return outcome(state, config, "FORCED_NOOP", started_at, objective_before);
    }

    double prefill_duration = 0.0;
    if (resolved.launch_count > 0) {
        prefill_duration = prefill_time_estimator(resolved.prefill_tokens);
        if (!std::isfinite(prefill_duration) || prefill_duration <= 0.0) {
            throw std::logic_error("prefill estimator returned invalid duration");
        }
    }
    prune_launch_history(state, config, state.now);
    for (int request_id : resolved.stop_request_ids) {
        mark_stop(state, state.request(request_id), state.now);
    }
    if (resolved.launch_count > 0) {
        const double deadline = round_time(
            state.now + config.prefill_slowdown_factor * prefill_duration,
            config.time_round_digits);
        for (int index = 0; index < resolved.launch_count; ++index) {
            RequestState request;
            request.request_id = state.next_request_id;
            request.arrival_time = state.now;
            request.prefill_deadline = deadline;
            request.decode_token_slo_sec = config.decode_token_slo_sec;
            request.original_prefill_tokens = resolved.prefill_tokens;
            request.original_decode_tokens = config.max_decode_tokens_per_request;
            state.requests.push_back(request);
            ++state.next_request_id;
        }
        state.launch_history.push_back({
            state.now,
            resolved.launch_count,
            resolved.launch_count * resolved.prefill_tokens,
        });
    }
    state.next_adversary_tick = round_time(
        state.now + config.adversary_tick_sec, config.time_round_digits);
    state.next_player = Player::Controller;
    refresh_objective(state, config);
    validate_if_enabled(state, config);
    return outcome(state, config, "ADVERSARY", started_at, objective_before);
}

void fast_forward_decode_only_to_next_tick(
    State& state,
    const Config& config,
    const BatchTimingProvider& timing_provider,
    const PrefillTimeEstimator& prefill_time_estimator) {
    if (state.now + config.epsilon >= state.next_adversary_tick) {
        state.next_player = Player::Adversary;
        return;
    }
    if (has_active_prefill(state)) return;
    int zero_time_steps = 0;
    while (state.now + config.epsilon < state.next_adversary_tick) {
        if (state.next_player == Player::Adversary) {
            const auto actions = resolve_adversary_actions(state, config);
            const int canonical = actions.raw_to_canonical.at(0);
            if (canonical < 0) throw std::logic_error("forced adversary noop is masked");
            apply_adversary_action(
                state, config, actions.canonical_actions[canonical],
                [](int) { return 0.0; });
            ++zero_time_steps;
        }

        bool has_waiting_decode = false;
        for (const auto& request : state.requests) {
            if (request.lifecycle == RequestLifecycle::WaitingDecode) {
                has_waiting_decode = true;
                break;
            }
        }
        bool kv_blocked = false;
        bool admitted = false;
        if (has_waiting_decode && can_admit_microbatch(state.replica, state.now, config)) {
            const auto actions = resolve_controller_actions(
                state, config, prefill_time_estimator);
            const int canonical = actions.raw_to_canonical.at(0);
            if (canonical < 0) throw std::logic_error("decode-only raw action is masked");
            const auto& action = actions.canonical_actions[canonical];
            if (action.action.transition_kind == ControllerTransitionKind::Batch) {
                if (action.action.total_prefill_tokens() != 0) {
                    throw std::logic_error("decode fast-forward resolved prefill work");
                }
                apply_controller_action(
                    state, config, action, timing_provider(state, action.action),
                    prefill_time_estimator);
                admitted = true;
                ++zero_time_steps;
            } else {
                kv_blocked = true;
            }
        }

        if (!admitted) {
            if (kv_blocked) return;
            if (has_inflight_decode(state)) {
                advance_to_inplace(state, config, next_wait_boundary_time(state, config));
                zero_time_steps = 0;
            } else if (!has_active_request(state) &&
                       state.replica.inflight_microbatches.empty()) {
                advance_to_inplace(state, config, state.next_adversary_tick);
                state.next_player = Player::Adversary;
                return;
            } else {
                // Active terminal-pending/non-decode work cannot be forced safely.
                return;
            }
        }
        if (zero_time_steps > config.max_zero_time_transitions_per_boundary) {
            throw std::logic_error("too many zero-time fast-forward transitions");
        }
    }
    state.next_player = Player::Adversary;
}

Environment::Environment(
    Config config,
    BatchTimingProvider batch_timing_provider,
    PrefillTimeEstimator prefill_time_estimator)
    : config_(std::move(config)),
      batch_timing_provider_(std::move(batch_timing_provider)),
      prefill_time_estimator_(std::move(prefill_time_estimator)) {
    config_.validate();
    if (!batch_timing_provider_ || !prefill_time_estimator_) {
        throw std::invalid_argument("environment timing callbacks are required");
    }
}

State Environment::initial_state(double now, Player next_player) const {
    return State::initial(config_, now, next_player);
}

ControllerActionSpace Environment::sample_controller_actions(const State& state) const {
    return resolve_controller_actions(state, config_, prefill_time_estimator_);
}

AdversaryActionSpace Environment::sample_adversary_actions(const State& state) const {
    return resolve_adversary_actions(state, config_);
}

State Environment::apply_controller_action_only(
    const State& state,
    const CanonicalControllerAction& action,
    bool fast_forward) const {
    State result = state;
    BatchTiming timing;
    if (action.action.transition_kind == ControllerTransitionKind::Batch) {
        timing = batch_timing_provider_(result, action.action);
    }
    apply_controller_action(
        result, config_, action, timing, prefill_time_estimator_);
    if (fast_forward) {
        fast_forward_decode_only_to_next_tick(
            result, config_, batch_timing_provider_, prefill_time_estimator_);
    }
    return result;
}

State Environment::apply_adversary_action_only(
    const State& state,
    const CanonicalAdversaryAction& action) const {
    State result = state;
    apply_adversary_action(result, config_, action, prefill_time_estimator_);
    return result;
}

}  // namespace gv4
