#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "native_predictor.hpp"
#include "native_types.hpp"

namespace mcts_native {

class NativeSim {
public:
    static NativeSimState apply_adversary_action(
        const NativeSimState& in_state,
        const AdversaryActionSpecNative& action,
        const NativeRuntimeConfig& cfg
    );
    static NativeSimState apply_controller_action(
        const NativeSimState& in_state,
        const ControllerActionSpecNative& action,
        const NativeRuntimeConfig& cfg,
        NativePredictor& predictor
    );

    static void apply_adversary_action_inplace(
        NativeSimState& state,
        const AdversaryActionSpecNative& action,
        const NativeRuntimeConfig& cfg
    );
    static void apply_controller_action_inplace(
        NativeSimState& state,
        const ControllerActionSpecNative& action,
        const NativeRuntimeConfig& cfg,
        NativePredictor& predictor
    );

    static double evaluate_objective_cost(const NativeSimState& state);
    static std::vector<ControllerRequestStateNative> controller_view_from_state(const NativeSimState& state);

    static std::string snapshot(const NativeSimState& in_state);
    static NativeSimState restore(const std::string& payload);
};

} // namespace mcts_native
