#pragma once
#include <string>
#include "native_types.hpp"

namespace mcts_native {

enum class NativeGameVersionId : int {
    V1 = 1,
};

NativeGameVersionId resolve_game_version_id(const std::string& name);

bool can_adversary_send(
    double sim_time,
    double last_prefill_batch_time,
    const NativeRuntimeConfig& cfg
);

double compute_adversary_arrival_time(
    double sim_time,
    double last_prefill_batch_time,
    bool has_requests,
    const NativeRuntimeConfig& cfg
);

double next_last_prefill_batch_time(
    double sim_time,
    double last_prefill_batch_time,
    bool has_requests,
    const NativeRuntimeConfig& cfg
);

double next_adversary_release_time(
    double last_prefill_batch_time,
    const NativeRuntimeConfig& cfg
);

double arrival_window_start(double sim_time, const NativeRuntimeConfig& cfg);

AdversarySampleOutput sample_adversary_actions_for_rules(
    const NativeSimState& state,
    const NativeRuntimeConfig& cfg
);

ControllerSampleOutput sample_controller_actions_for_rules(
    const NativeSimState& state,
    const NativeRuntimeConfig& cfg
);
} // namespace mcts_native
