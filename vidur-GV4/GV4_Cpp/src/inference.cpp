#include "gv4/inference.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace gv4 {

InferenceRuntime::InferenceRuntime(
    Config config,
    ValuePredictor value_predictor,
    ControllerPolicyPredictor controller_policy_predictor,
    AdversaryPolicyPredictor adversary_policy_predictor)
    : builder_(std::move(config)),
      value_predictor_(std::move(value_predictor)),
      controller_policy_predictor_(std::move(controller_policy_predictor)),
      adversary_policy_predictor_(std::move(adversary_policy_predictor)) {}

void InferenceRuntime::validate_predictions(
    const std::vector<float>& values,
    std::size_t expected,
    const char* label) {
    if (values.size() != expected) {
        throw std::runtime_error(std::string(label) + " returned the wrong value count");
    }
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string(label) + " returned a non-finite value");
        }
    }
}

StateFeatures InferenceRuntime::build_state_features(const State& state) const {
    return builder_.build_state(state);
}

std::vector<float> InferenceRuntime::predict_values(
    const std::vector<State>& states,
    Player player) const {
    if (states.empty()) {
        return {};
    }
    if (!value_predictor_) {
        throw std::runtime_error("no native GV4 value predictor is configured");
    }
    std::vector<StateFeatures> features;
    features.reserve(states.size());
    for (const State& state : states) {
        features.push_back(builder_.build_state(state));
    }
    std::vector<float> result = value_predictor_(player, features);
    validate_predictions(result, states.size(), "value model");
    return result;
}

float InferenceRuntime::predict_value(const State& state, Player player) const {
    return predict_values({state}, player).front();
}

std::vector<float> InferenceRuntime::predict_controller_logits(
    const State& state,
    const std::vector<CanonicalControllerAction>& actions) const {
    if (state.next_player != Player::Controller) {
        throw std::invalid_argument("controller policy called on another player's turn");
    }
    if (actions.empty()) {
        return {};
    }
    if (!controller_policy_predictor_) {
        throw std::runtime_error("no native GV4 controller policy is configured");
    }
    StateFeatures state_features = builder_.build_state(state);
    std::vector<ControllerActionFeatures> action_features;
    action_features.reserve(actions.size());
    for (const CanonicalControllerAction& action : actions) {
        action_features.push_back(builder_.build_controller_action(state, action));
    }
    std::vector<float> result =
        controller_policy_predictor_(state_features, action_features);
    validate_predictions(result, actions.size(), "controller policy model");
    return result;
}

std::vector<float> InferenceRuntime::predict_adversary_logits(
    const State& state,
    const std::vector<CanonicalAdversaryAction>& actions) const {
    if (state.next_player != Player::Adversary) {
        throw std::invalid_argument("adversary policy called on another player's turn");
    }
    if (actions.empty()) {
        return {};
    }
    if (!adversary_policy_predictor_) {
        throw std::runtime_error("no native GV4 adversary policy is configured");
    }
    StateFeatures state_features = builder_.build_state(state);
    std::vector<AdversaryActionFeatures> action_features;
    action_features.reserve(actions.size());
    for (const CanonicalAdversaryAction& action : actions) {
        action_features.push_back(builder_.build_adversary_action(state, action));
    }
    std::vector<float> result =
        adversary_policy_predictor_(state_features, action_features);
    validate_predictions(result, actions.size(), "adversary policy model");
    return result;
}

}  // namespace gv4
