#include "gv4/state.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace gv4 {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::logic_error(message);
}

void require_nonnegative(int value, const char* message) {
    require(value >= 0, message);
}

void require_time(double value, const char* message) {
    require(std::isfinite(value) && value >= 0.0, message);
}

}  // namespace

bool is_inflight(RequestLifecycle lifecycle) {
    return lifecycle == RequestLifecycle::InflightPrefill ||
           lifecycle == RequestLifecycle::InflightDecode ||
           lifecycle == RequestLifecycle::StopPending ||
           lifecycle == RequestLifecycle::DropPending;
}

bool is_terminal(RequestLifecycle lifecycle) {
    return lifecycle == RequestLifecycle::Completed ||
           lifecycle == RequestLifecycle::Stopped ||
           lifecycle == RequestLifecycle::Dropped;
}

const char* player_name(Player player) {
    switch (player) {
        case Player::Adversary: return "ADVERSARY";
        case Player::Controller: return "CONTROLLER";
    }
    throw std::logic_error("unknown player");
}

const char* lifecycle_name(RequestLifecycle lifecycle) {
    switch (lifecycle) {
        case RequestLifecycle::WaitingPrefill: return "WAITING_PREFILL";
        case RequestLifecycle::InflightPrefill: return "INFLIGHT_PREFILL";
        case RequestLifecycle::WaitingDecode: return "WAITING_DECODE";
        case RequestLifecycle::InflightDecode: return "INFLIGHT_DECODE";
        case RequestLifecycle::StopPending: return "STOP_PENDING";
        case RequestLifecycle::DropPending: return "DROP_PENDING";
        case RequestLifecycle::Completed: return "COMPLETED";
        case RequestLifecycle::Stopped: return "STOPPED";
        case RequestLifecycle::Dropped: return "DROPPED";
    }
    throw std::logic_error("unknown lifecycle");
}

const char* terminal_reason_name(TerminalReason reason) {
    switch (reason) {
        case TerminalReason::None: return "NONE";
        case TerminalReason::NaturalCompletion: return "NATURAL_COMPLETION";
        case TerminalReason::AdversaryStop: return "ADVERSARY_STOP";
        case TerminalReason::ControllerEviction: return "CONTROLLER_EVICTION";
        case TerminalReason::AutomaticSloDrop: return "AUTOMATIC_SLO_DROP";
        case TerminalReason::DecodeCreditExhausted: return "DECODE_CREDIT_EXHAUSTED";
    }
    throw std::logic_error("unknown terminal reason");
}

double InflightMicrobatch::final_completion_time() const {
    if (stage_finish_times.empty()) throw std::logic_error("microbatch has no stages");
    return stage_finish_times.back();
}

int InflightMicrobatch::total_prefill_tokens() const {
    int total = 0;
    for (const auto& allocation : allocations) total += allocation.prefill_tokens;
    return total;
}

int InflightMicrobatch::total_decode_tokens() const {
    int total = 0;
    for (const auto& allocation : allocations) total += allocation.decode_tokens;
    return total;
}

int RequestState::remaining_prefill_tokens() const {
    return original_prefill_tokens - committed_prefill_tokens - reserved_prefill_tokens;
}

int RequestState::remaining_decode_tokens() const {
    return original_decode_tokens - committed_decode_tokens - reserved_decode_tokens;
}

int RequestState::resident_tokens() const {
    return committed_prefill_tokens + reserved_prefill_tokens +
           committed_decode_tokens + reserved_decode_tokens;
}

bool RequestState::has_inflight_work() const {
    return inflight_microbatch_id != kNoId;
}

int ReplicaState::inflight_count() const {
    return static_cast<int>(inflight_microbatches.size());
}

int ReplicaState::pipeline_parallel_size() const {
    return static_cast<int>(stage_tail_finish_times.size());
}

InflightMicrobatch* ReplicaState::find_microbatch(int microbatch_id) {
    for (auto& batch : inflight_microbatches) {
        if (batch.microbatch_id == microbatch_id) return &batch;
    }
    return nullptr;
}

const InflightMicrobatch* ReplicaState::find_microbatch(int microbatch_id) const {
    for (const auto& batch : inflight_microbatches) {
        if (batch.microbatch_id == microbatch_id) return &batch;
    }
    return nullptr;
}

State State::initial(const Config& config, double now, Player next_player) {
    config.validate();
    require_time(now, "initial time must be finite and nonnegative");
    State state;
    state.state_schema_version = config.state_schema_version;
    state.config_manifest_sha256 = config.manifest_sha256;
    state.now = now;
    state.next_player = next_player;
    state.next_adversary_tick = now;
    state.rng_seed = config.global_seed;
    state.replica.replica_id = 0;
    state.replica.rank_ids = config.rank_ids;
    state.replica.rank_kv_capacity_blocks = config.rank_kv_capacity_blocks;
    state.replica.rank_kv_committed_blocks.assign(config.rank_ids.size(), 0);
    state.replica.rank_kv_reserved_blocks.assign(config.rank_ids.size(), 0);
    state.replica.stage_tail_finish_times.assign(config.pipeline_parallel_size, now);
    state.replica.stage_last_microbatch_ids.assign(config.pipeline_parallel_size, kNoId);
    if (config.enable_debug_asserts) state.validate(config);
    return state;
}

RequestState& State::request(int request_id) {
    if (request_id < 0 || request_id >= static_cast<int>(requests.size()) ||
        requests[request_id].request_id != request_id) {
        throw std::out_of_range("unknown request ID");
    }
    return requests[request_id];
}

const RequestState& State::request(int request_id) const {
    if (request_id < 0 || request_id >= static_cast<int>(requests.size()) ||
        requests[request_id].request_id != request_id) {
        throw std::out_of_range("unknown request ID");
    }
    return requests[request_id];
}

void State::validate(const Config& config) const {
    require(state_schema_version == config.state_schema_version,
            "state schema differs from config");
    require(config_manifest_sha256 == config.manifest_sha256,
            "state manifest differs from config");
    require_time(now, "state time must be finite and nonnegative");
    require_time(next_adversary_tick, "adversary tick must be finite and nonnegative");
    require(next_adversary_tick + config.epsilon >= now,
            "next adversary tick is behind state time");
    require(next_request_id == static_cast<int>(requests.size()),
            "next request ID differs from append-only request count");
    require(next_request_id <= config.max_requests, "request capacity exceeded");
    require_nonnegative(next_microbatch_id, "next microbatch ID is negative");
    require_nonnegative(decode_credits_available, "available decode credit is negative");
    require_nonnegative(decode_credits_reserved, "reserved decode credit is negative");
    require_nonnegative(decode_credits_minted_total, "minted decode credit is negative");
    require_nonnegative(decode_tokens_committed_total,
                        "committed decode token count is negative");

    require(static_cast<int>(launch_history.size()) <= config.max_launch_history_entries,
            "launch history capacity exceeded");
    double prior_launch = -1.0;
    for (const auto& record : launch_history) {
        require_time(record.launch_time, "launch time is invalid");
        require(record.request_count > 0 && record.prefill_tokens > 0,
                "launch record must be positive");
        require(record.launch_time >= prior_launch, "launch history is unordered");
        prior_launch = record.launch_time;
    }

    const auto& replica_state = replica;
    require(replica_state.replica_id == 0, "native v1 replica ID must be zero");
    require(replica_state.rank_ids == config.rank_ids, "replica ranks differ from config");
    require(replica_state.rank_kv_capacity_blocks == config.rank_kv_capacity_blocks,
            "replica KV capacities differ from config");
    const std::size_t rank_count = config.rank_ids.size();
    require(replica_state.rank_kv_committed_blocks.size() == rank_count &&
                replica_state.rank_kv_reserved_blocks.size() == rank_count,
            "replica rank ledgers have the wrong width");
    require(replica_state.stage_tail_finish_times.size() ==
                static_cast<std::size_t>(config.pipeline_parallel_size) &&
                replica_state.stage_last_microbatch_ids.size() ==
                    static_cast<std::size_t>(config.pipeline_parallel_size),
            "replica stage arrays have the wrong width");
    require(replica_state.inflight_count() <= config.max_inflight_microbatches,
            "in-flight microbatch capacity exceeded");
    for (std::size_t rank = 0; rank < rank_count; ++rank) {
        const int committed = replica_state.rank_kv_committed_blocks[rank];
        const int reserved = replica_state.rank_kv_reserved_blocks[rank];
        require(committed >= 0 && reserved >= 0 &&
                    committed + reserved <= replica_state.rank_kv_capacity_blocks[rank],
                "rank KV ledger is invalid");
    }

    std::unordered_map<int, const InflightMicrobatch*> batches;
    int previous_batch_id = -1;
    const InflightMicrobatch* prior_batch = nullptr;
    for (const auto& batch : replica_state.inflight_microbatches) {
        require(batch.microbatch_id > previous_batch_id,
                "in-flight microbatch IDs must be sorted and unique");
        previous_batch_id = batch.microbatch_id;
        require(batch.replica_id == 0 && !batch.completion_applied,
                "invalid in-flight microbatch ownership/state");
        require(!batch.allocations.empty(), "in-flight microbatch has no allocations");
        const std::size_t stages = static_cast<std::size_t>(config.pipeline_parallel_size);
        require(batch.stage_ready_times.size() == stages &&
                    batch.stage_start_times.size() == stages &&
                    batch.stage_finish_times.size() == stages,
                "microbatch stage-time width differs from config");
        int previous_request_id = -1;
        double previous_stage_finish = -1.0;
        for (const auto& allocation : batch.allocations) {
            require(allocation.request_id > previous_request_id,
                    "batch request IDs must be sorted and unique");
            require(allocation.prefill_tokens >= 0 && allocation.decode_tokens >= 0 &&
                        allocation.new_kv_blocks >= 0 && allocation.total_tokens() > 0 &&
                        !(allocation.prefill_tokens && allocation.decode_tokens),
                    "invalid batch allocation");
            previous_request_id = allocation.request_id;
        }
        for (std::size_t stage = 0; stage < stages; ++stage) {
            const double ready = batch.stage_ready_times[stage];
            const double start = batch.stage_start_times[stage];
            const double finish = batch.stage_finish_times[stage];
            require_time(ready, "invalid stage ready time");
            require_time(start, "invalid stage start time");
            require_time(finish, "invalid stage finish time");
            require(ready <= start && start < finish,
                    "stage calendar must satisfy ready <= start < finish");
            if (stage > 0) require(ready >= previous_stage_finish,
                                   "stage is ready before prior stage completion");
            previous_stage_finish = finish;
            require(finish <= replica_state.stage_tail_finish_times[stage],
                    "stage tail precedes scheduled work");
            if (prior_batch != nullptr) {
                require(start >= prior_batch->stage_finish_times[stage],
                        "FIFO stage batches overlap");
            }
        }
        prior_batch = &batch;
        require(batches.emplace(batch.microbatch_id, &batch).second,
                "duplicate microbatch ID");
    }
    if (!batches.empty()) require(next_microbatch_id > previous_batch_id,
                                  "next microbatch ID does not exceed active IDs");

    int request_committed_blocks = 0;
    int request_reserved_blocks = 0;
    int reserved_decode_tokens = 0;
    int minted_requests = 0;
    int committed_decode_tokens = 0;
    int completed = 0;
    int stopped = 0;
    int dropped = 0;
    int violations = 0;
    for (int request_id = 0; request_id < static_cast<int>(requests.size()); ++request_id) {
        const auto& item = requests[request_id];
        require(item.request_id == request_id, "requests are not indexed by ID");
        require(item.owner_replica_id == 0, "native v1 request owner must be replica zero");
        require_time(item.arrival_time, "invalid request arrival time");
        require_time(item.prefill_deadline, "invalid request prefill deadline");
        require(item.prefill_deadline >= item.arrival_time,
                "prefill deadline precedes arrival");
        require(item.decode_token_slo_sec > 0.0 && std::isfinite(item.decode_token_slo_sec),
                "invalid decode token SLO");
        require(item.original_prefill_tokens > 0 &&
                    item.original_prefill_tokens <= config.max_prefill_tokens_per_request,
                "invalid original prefill size");
        require(item.original_decode_tokens >= config.min_decode_tokens_per_request &&
                    item.original_decode_tokens <= config.max_decode_tokens_per_request,
                "invalid original decode size");
        require(item.remaining_prefill_tokens() >= 0 && item.remaining_decode_tokens() >= 0,
                "request work exceeds original token count");
        require(item.committed_kv_blocks >= 0 && item.reserved_kv_blocks >= 0,
                "request KV ownership is negative");
        require(is_inflight(item.lifecycle) == item.has_inflight_work(),
                "request lifecycle and batch link disagree");
        if (item.has_inflight_work()) {
            require(item.reserved_prefill_tokens > 0 || item.reserved_decode_tokens > 0,
                    "in-flight request has no reserved work");
        } else {
            require(item.reserved_prefill_tokens == 0 &&
                        item.reserved_decode_tokens == 0 &&
                        item.reserved_kv_blocks == 0,
                    "non-in-flight request retains reservations");
        }
        if (item.lifecycle == RequestLifecycle::WaitingPrefill) {
            require(item.remaining_prefill_tokens() > 0,
                    "waiting prefill has no remaining work");
        } else if (item.lifecycle == RequestLifecycle::InflightPrefill) {
            require(item.reserved_prefill_tokens > 0 && item.reserved_decode_tokens == 0,
                    "in-flight prefill reservation is invalid");
        } else if (item.lifecycle == RequestLifecycle::WaitingDecode) {
            require(item.remaining_prefill_tokens() == 0 &&
                        item.remaining_decode_tokens() > 0,
                    "waiting decode progress is invalid");
        } else if (item.lifecycle == RequestLifecycle::InflightDecode) {
            require(item.reserved_decode_tokens == 1 && item.reserved_prefill_tokens == 0,
                    "in-flight decode must reserve exactly one token");
        } else if (item.lifecycle == RequestLifecycle::Completed) {
            require(item.remaining_prefill_tokens() == 0 &&
                        item.remaining_decode_tokens() == 0,
                    "completed request retains work");
        }
        if (item.decode_credit_minted) {
            require(item.committed_prefill_tokens == item.original_prefill_tokens &&
                        item.reserved_prefill_tokens == 0,
                    "decode credit minted before prefill completion");
        } else {
            require(item.committed_decode_tokens == 0 && item.reserved_decode_tokens == 0,
                    "decode work exists before credit mint");
        }
        const bool active_decode = item.lifecycle == RequestLifecycle::WaitingDecode ||
                                   item.lifecycle == RequestLifecycle::InflightDecode ||
                                   item.reserved_decode_tokens > 0;
        if (active_decode) require(item.next_decode_deadline != kUnsetTime,
                                   "active decode lacks deadline");
        if (is_terminal(item.lifecycle)) {
            require(item.committed_kv_blocks == 0 && item.reserved_kv_blocks == 0,
                    "terminal request retains KV blocks");
            require(item.terminal_reason != TerminalReason::None &&
                        item.terminal_time != kUnsetTime,
                    "terminal request lacks terminal bookkeeping");
        }
        if (!is_terminal(item.lifecycle)) {
            const int minimum_blocks =
                (item.resident_tokens() + config.block_size_tokens - 1) /
                config.block_size_tokens;
            require(item.committed_kv_blocks + item.reserved_kv_blocks >= minimum_blocks,
                    "request owns fewer blocks than resident tokens");
        }

        request_committed_blocks += item.committed_kv_blocks;
        request_reserved_blocks += item.reserved_kv_blocks;
        reserved_decode_tokens += item.reserved_decode_tokens;
        minted_requests += item.decode_credit_minted ? 1 : 0;
        committed_decode_tokens += item.committed_decode_tokens;
        completed += item.lifecycle == RequestLifecycle::Completed ? 1 : 0;
        stopped += item.lifecycle == RequestLifecycle::Stopped ? 1 : 0;
        dropped += item.lifecycle == RequestLifecycle::Dropped ? 1 : 0;
        violations += item.violation_recorded ? 1 : 0;

        if (item.has_inflight_work()) {
            const auto batch_it = batches.find(item.inflight_microbatch_id);
            require(batch_it != batches.end(), "request references missing microbatch");
            const auto allocation_it = std::find_if(
                batch_it->second->allocations.begin(), batch_it->second->allocations.end(),
                [&](const BatchAllocation& allocation) {
                    return allocation.request_id == item.request_id;
                });
            require(allocation_it != batch_it->second->allocations.end(),
                    "request has no matching batch allocation");
            require(allocation_it->prefill_tokens == item.reserved_prefill_tokens &&
                        allocation_it->decode_tokens == item.reserved_decode_tokens &&
                        allocation_it->new_kv_blocks == item.reserved_kv_blocks,
                    "request reservations differ from batch allocation");
        }
    }

    for (std::size_t rank = 0; rank < rank_count; ++rank) {
        require(replica_state.rank_kv_committed_blocks[rank] == request_committed_blocks,
                "committed rank ledger disagrees with requests");
        require(replica_state.rank_kv_reserved_blocks[rank] == request_reserved_blocks,
                "reserved rank ledger disagrees with requests");
    }
    require(decode_credits_reserved == reserved_decode_tokens,
            "reserved decode credits disagree with requests");
    require(decode_credits_minted_total == minted_requests * config.decode_credit_mint,
            "minted decode total disagrees with requests");
    require(decode_tokens_committed_total == committed_decode_tokens,
            "committed decode total disagrees with requests");
    require(decode_credits_available + decode_credits_reserved +
                decode_tokens_committed_total == decode_credits_minted_total,
            "decode credit conservation failed");
    if (decode_credits_available == 0) {
        for (const auto& item : requests) {
            require(item.lifecycle != RequestLifecycle::WaitingDecode &&
                        item.lifecycle != RequestLifecycle::InflightDecode,
                    "zero available decode credit leaves active decode request");
        }
    }

    require(objective.requests_generated == static_cast<int>(requests.size()),
            "objective generated count disagrees with requests");
    require(objective.requests_completed == completed,
            "objective completed count disagrees with requests");
    require(objective.requests_stopped == stopped,
            "objective stopped count disagrees with requests");
    require(objective.requests_dropped == dropped,
            "objective dropped count disagrees with requests");
    require(objective.slo_violations == violations,
            "objective violation count disagrees with requests");
}

}  // namespace gv4
