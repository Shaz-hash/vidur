#pragma once

#include "gv4/actions.hpp"

#include <functional>
#include <string>
#include <vector>

namespace gv4 {

struct BatchTiming {
    std::vector<double> stage_service_times;
    std::vector<double> pp_communication_times;
};

using BatchTimingProvider =
    std::function<BatchTiming(const State&, const ResolvedControllerAction&)>;

struct TransitionOutcome {
    std::string transition_kind;
    double elapsed_sec = 0.0;
    double objective_before = 0.0;
    double objective_after = 0.0;
    double edge_reward = 0.0;
    double discount = 1.0;
};

[[nodiscard]] int blocks_for_tokens(int token_count, int block_size_tokens);
[[nodiscard]] int additional_blocks_for_work(
    const RequestState& request,
    int prefill_tokens,
    int decode_tokens,
    int recompute_tokens,
    int block_size_tokens);
[[nodiscard]] int preempt_request_blocks(State& state, RequestState& request);
[[nodiscard]] int free_logical_blocks(const ReplicaState& replica);

[[nodiscard]] bool can_admit_microbatch(
    const ReplicaState& replica,
    double admitted_at,
    const Config& config);
[[nodiscard]] double next_pipeline_admission_time(
    const ReplicaState& replica,
    double now,
    const Config& config);
[[nodiscard]] double next_wait_boundary_time(const State& state, const Config& config);
[[nodiscard]] double next_internal_completion_time(const State& state);

TransitionOutcome advance_to(State& state, const Config& config, double target_time);
TransitionOutcome apply_controller_action(
    State& state,
    const Config& config,
    const CanonicalControllerAction& action,
    const BatchTiming& timing,
    const PrefillTimeEstimator& prefill_time_estimator);
TransitionOutcome apply_adversary_action(
    State& state,
    const Config& config,
    const CanonicalAdversaryAction& action,
    const PrefillTimeEstimator& prefill_time_estimator);

void fast_forward_decode_only_to_next_tick(
    State& state,
    const Config& config,
    const BatchTimingProvider& timing_provider,
    const PrefillTimeEstimator& prefill_time_estimator);

class Environment {
public:
    Environment(Config config,
                BatchTimingProvider batch_timing_provider,
                PrefillTimeEstimator prefill_time_estimator);

    [[nodiscard]] const Config& config() const { return config_; }
    [[nodiscard]] State initial_state(
        double now = 0.0,
        Player next_player = Player::Adversary) const;
    [[nodiscard]] ControllerActionSpace sample_controller_actions(const State& state) const;
    [[nodiscard]] AdversaryActionSpace sample_adversary_actions(const State& state) const;
    [[nodiscard]] State apply_controller_action_only(
        const State& state,
        const CanonicalControllerAction& action,
        bool fast_forward = true) const;
    [[nodiscard]] State apply_adversary_action_only(
        const State& state,
        const CanonicalAdversaryAction& action) const;

private:
    Config config_;
    BatchTimingProvider batch_timing_provider_;
    PrefillTimeEstimator prefill_time_estimator_;
};

}  // namespace gv4
