#pragma once

#include "gv4/features.hpp"

#include <functional>
#include <vector>

namespace gv4 {

using ValuePredictor = std::function<std::vector<float>(
    Player player,
    const std::vector<StateFeatures>& states)>;
using ControllerPolicyPredictor = std::function<std::vector<float>(
    const StateFeatures& state,
    const std::vector<ControllerActionFeatures>& actions)>;
using AdversaryPolicyPredictor = std::function<std::vector<float>(
    const StateFeatures& state,
    const std::vector<AdversaryActionFeatures>& actions)>;

class InferenceRuntime {
public:
    InferenceRuntime(
        Config config,
        ValuePredictor value_predictor = {},
        ControllerPolicyPredictor controller_policy_predictor = {},
        AdversaryPolicyPredictor adversary_policy_predictor = {});

    [[nodiscard]] const Config& config() const { return builder_.config(); }
    [[nodiscard]] const FeatureBuilder& builder() const { return builder_; }
    [[nodiscard]] StateFeatures build_state_features(const State& state) const;
    [[nodiscard]] std::vector<float> predict_values(
        const std::vector<State>& states,
        Player player) const;
    [[nodiscard]] float predict_value(const State& state, Player player) const;
    [[nodiscard]] std::vector<float> predict_controller_logits(
        const State& state,
        const std::vector<CanonicalControllerAction>& actions) const;
    [[nodiscard]] std::vector<float> predict_adversary_logits(
        const State& state,
        const std::vector<CanonicalAdversaryAction>& actions) const;

private:
    static void validate_predictions(
        const std::vector<float>& values,
        std::size_t expected,
        const char* label);

    FeatureBuilder builder_;
    ValuePredictor value_predictor_;
    ControllerPolicyPredictor controller_policy_predictor_;
    AdversaryPolicyPredictor adversary_policy_predictor_;
};

}  // namespace gv4
