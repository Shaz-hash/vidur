#pragma once

#include "gv4/config.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gv4 {

constexpr int kNoId = -1;
constexpr double kUnsetTime = -1.0;

enum class Player : int { Adversary = 0, Controller = 1 };

enum class RequestLifecycle : int {
    WaitingPrefill = 0,
    InflightPrefill = 1,
    WaitingDecode = 2,
    InflightDecode = 3,
    StopPending = 4,
    DropPending = 5,
    Completed = 6,
    Stopped = 7,
    Dropped = 8,
    InflightRecompute = 9,
    PreemptPending = 10,
};

enum class TerminalReason : int {
    None = 0,
    NaturalCompletion = 1,
    AdversaryStop = 2,
    ControllerEviction = 3,
    AutomaticSloDrop = 4,
    DecodeCreditExhausted = 5,
};

[[nodiscard]] bool is_inflight(RequestLifecycle lifecycle);
[[nodiscard]] bool is_terminal(RequestLifecycle lifecycle);
[[nodiscard]] const char* player_name(Player player);
[[nodiscard]] const char* lifecycle_name(RequestLifecycle lifecycle);
[[nodiscard]] const char* terminal_reason_name(TerminalReason reason);

struct LaunchRecord {
    double launch_time = 0.0;
    int request_count = 0;
    int prefill_tokens = 0;
};

struct BatchAllocation {
    int request_id = kNoId;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    int new_kv_blocks = 0;
    int recompute_tokens = 0;

    [[nodiscard]] int total_tokens() const {
        return prefill_tokens + decode_tokens + recompute_tokens;
    }
    [[nodiscard]] bool operator==(const BatchAllocation& other) const {
        return request_id == other.request_id &&
               prefill_tokens == other.prefill_tokens &&
               decode_tokens == other.decode_tokens &&
               new_kv_blocks == other.new_kv_blocks &&
               recompute_tokens == other.recompute_tokens;
    }
};

struct InflightMicrobatch {
    int microbatch_id = kNoId;
    int replica_id = 0;
    int raw_action_index = 0;
    int canonical_action_index = 0;
    std::vector<BatchAllocation> allocations;
    std::vector<double> stage_ready_times;
    std::vector<double> stage_start_times;
    std::vector<double> stage_finish_times;
    bool completion_applied = false;

    [[nodiscard]] double final_completion_time() const;
    [[nodiscard]] int total_prefill_tokens() const;
    [[nodiscard]] int total_decode_tokens() const;
    [[nodiscard]] int total_recompute_tokens() const;
};

struct RequestState {
    int request_id = kNoId;
    int owner_replica_id = 0;
    RequestLifecycle lifecycle = RequestLifecycle::WaitingPrefill;
    double arrival_time = 0.0;
    double prefill_deadline = 0.0;
    double decode_token_slo_sec = 0.05;
    int original_prefill_tokens = 0;
    int original_decode_tokens = 0;
    bool decode_credit_minted = false;
    int committed_prefill_tokens = 0;
    int reserved_prefill_tokens = 0;
    int committed_decode_tokens = 0;
    int reserved_decode_tokens = 0;
    int kv_computed_tokens = 0;
    int reserved_recompute_tokens = 0;
    int committed_kv_blocks = 0;
    int reserved_kv_blocks = 0;
    int inflight_microbatch_id = kNoId;
    double next_decode_deadline = kUnsetTime;
    double prefill_lateness_sec = 0.0;
    double decode_lateness_sec = 0.0;
    bool violation_recorded = false;
    TerminalReason terminal_reason = TerminalReason::None;
    double terminal_requested_at = kUnsetTime;
    double terminal_time = kUnsetTime;

    [[nodiscard]] int remaining_prefill_tokens() const;
    [[nodiscard]] int remaining_decode_tokens() const;
    [[nodiscard]] int logical_context_tokens() const;
    [[nodiscard]] int remaining_recompute_tokens() const;
    [[nodiscard]] bool is_decode_phase() const;
    [[nodiscard]] int resident_tokens() const;
    [[nodiscard]] bool has_inflight_work() const;
};

struct ObjectiveState {
    int requests_generated = 0;
    int requests_completed = 0;
    int requests_stopped = 0;
    int requests_dropped = 0;
    int slo_violations = 0;
    double prefill_lateness_sec = 0.0;
    double decode_lateness_sec = 0.0;
    double terminal_cost = 0.0;
    double total_cost = 0.0;
};

struct ReplicaState {
    int replica_id = 0;
    std::vector<int> rank_ids;
    std::vector<int> rank_kv_capacity_blocks;
    std::vector<int> rank_kv_committed_blocks;
    std::vector<int> rank_kv_reserved_blocks;
    std::vector<double> stage_tail_finish_times;
    std::vector<int> stage_last_microbatch_ids;
    std::vector<InflightMicrobatch> inflight_microbatches;

    [[nodiscard]] int inflight_count() const;
    [[nodiscard]] int pipeline_parallel_size() const;
    [[nodiscard]] InflightMicrobatch* find_microbatch(int microbatch_id);
    [[nodiscard]] const InflightMicrobatch* find_microbatch(int microbatch_id) const;
};

struct State {
    std::string state_schema_version;
    std::string config_manifest_sha256;
    double now = 0.0;
    Player next_player = Player::Adversary;
    double next_adversary_tick = 0.0;
    int next_request_id = 0;
    int next_microbatch_id = 0;
    std::uint64_t tie_break_counter = 0;
    std::uint64_t rng_seed = 0;
    std::uint64_t rng_counter = 0;
    std::vector<LaunchRecord> launch_history;
    int decode_credits_available = 0;
    int decode_credits_reserved = 0;
    int decode_credits_minted_total = 0;
    int decode_tokens_committed_total = 0;
    std::vector<RequestState> requests;
    ReplicaState replica;
    ObjectiveState objective;

    static State initial(const Config& config, double now = 0.0,
                         Player next_player = Player::Adversary);
    [[nodiscard]] RequestState& request(int request_id);
    [[nodiscard]] const RequestState& request(int request_id) const;
    void validate(const Config& config) const;
};

}  // namespace gv4
