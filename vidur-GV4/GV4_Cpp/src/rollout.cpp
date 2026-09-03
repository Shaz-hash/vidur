#include "gv4/mcts.hpp"

#include "gv4/inference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gv4 {
namespace {

class PythonRandomCompat {
public:
    explicit PythonRandomCompat(std::uint64_t seed) { seed_int(seed); }

    double random_double() {
        const std::uint32_t high = next_uint32() >> 5;
        const std::uint32_t low = next_uint32() >> 6;
        return (static_cast<double>(high) * 67108864.0 +
                static_cast<double>(low)) /
               9007199254740992.0;
    }

private:
    static constexpr int kStateSize = 624;
    static constexpr int kOffset = 397;
    static constexpr std::uint32_t kMatrixA = 0x9908b0dfU;
    static constexpr std::uint32_t kUpperMask = 0x80000000U;
    static constexpr std::uint32_t kLowerMask = 0x7fffffffU;

    std::array<std::uint32_t, kStateSize> state_{};
    int index_ = kStateSize + 1;

    void seed_int(std::uint64_t seed) {
        std::vector<std::uint32_t> key;
        if (seed == 0) {
            key.push_back(0U);
        } else {
            while (seed > 0) {
                key.push_back(static_cast<std::uint32_t>(seed & 0xffffffffULL));
                seed >>= 32;
            }
        }
        initialize_by_array(key);
    }

    void initialize(std::uint32_t seed) {
        state_[0] = seed;
        for (index_ = 1; index_ < kStateSize; ++index_) {
            state_[index_] = static_cast<std::uint32_t>(
                1812433253U *
                    (state_[index_ - 1] ^ (state_[index_ - 1] >> 30)) +
                static_cast<std::uint32_t>(index_));
        }
    }

    void initialize_by_array(const std::vector<std::uint32_t>& key) {
        initialize(19650218U);
        int state_index = 1;
        int key_index = 0;
        int remaining = std::max(kStateSize, static_cast<int>(key.size()));
        for (; remaining > 0; --remaining) {
            state_[state_index] = static_cast<std::uint32_t>(
                (state_[state_index] ^
                 ((state_[state_index - 1] ^
                   (state_[state_index - 1] >> 30)) *
                  1664525U)) +
                key[static_cast<std::size_t>(key_index)] +
                static_cast<std::uint32_t>(key_index));
            ++state_index;
            ++key_index;
            if (state_index >= kStateSize) {
                state_[0] = state_[kStateSize - 1];
                state_index = 1;
            }
            if (key_index >= static_cast<int>(key.size())) {
                key_index = 0;
            }
        }
        for (remaining = kStateSize - 1; remaining > 0; --remaining) {
            state_[state_index] = static_cast<std::uint32_t>(
                (state_[state_index] ^
                 ((state_[state_index - 1] ^
                   (state_[state_index - 1] >> 30)) *
                  1566083941U)) -
                static_cast<std::uint32_t>(state_index));
            ++state_index;
            if (state_index >= kStateSize) {
                state_[0] = state_[kStateSize - 1];
                state_index = 1;
            }
        }
        state_[0] = 0x80000000U;
    }

    std::uint32_t next_uint32() {
        static constexpr std::uint32_t matrix[2] = {0U, kMatrixA};
        if (index_ >= kStateSize) {
            int state_index = 0;
            for (; state_index < kStateSize - kOffset; ++state_index) {
                const std::uint32_t value =
                    (state_[state_index] & kUpperMask) |
                    (state_[state_index + 1] & kLowerMask);
                state_[state_index] =
                    state_[state_index + kOffset] ^ (value >> 1) ^
                    matrix[value & 1U];
            }
            for (; state_index < kStateSize - 1; ++state_index) {
                const std::uint32_t value =
                    (state_[state_index] & kUpperMask) |
                    (state_[state_index + 1] & kLowerMask);
                state_[state_index] =
                    state_[state_index + (kOffset - kStateSize)] ^
                    (value >> 1) ^ matrix[value & 1U];
            }
            const std::uint32_t value =
                (state_[kStateSize - 1] & kUpperMask) |
                (state_[0] & kLowerMask);
            state_[kStateSize - 1] =
                state_[kOffset - 1] ^ (value >> 1) ^ matrix[value & 1U];
            index_ = 0;
        }

        std::uint32_t value = state_[index_++];
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680U;
        value ^= (value << 15) & 0xefc60000U;
        value ^= value >> 18;
        return value;
    }
};

struct RolloutEdge {
    double reward = 0.0;
    double discount = 1.0;
};

int action_representative(const CanonicalAction& action) {
    return std::visit(
        [](const auto& value) { return value.representative_raw_index(); },
        action);
}

std::vector<int> action_aliases(const CanonicalAction& action) {
    return std::visit(
        [](const auto& value) { return value.equivalent_raw_indices; }, action);
}

State apply_rollout_action(
    const Environment& environment,
    const State& state,
    const CanonicalAction& action) {
    if (const auto* controller =
            std::get_if<CanonicalControllerAction>(&action)) {
        return environment.apply_controller_action_only(
            state, *controller, true);
    }
    return environment.apply_adversary_action_only(
        state, std::get<CanonicalAdversaryAction>(action));
}

void update_range(
    std::optional<double>& minimum,
    std::optional<double>& maximum,
    double value) {
    minimum = minimum.has_value() ? std::min(*minimum, value) : value;
    maximum = maximum.has_value() ? std::max(*maximum, value) : value;
}

int sample_action(
    const std::vector<MCTSActionEntry>& actions,
    double quantum,
    PythonRandomCompat& random) {
    if (actions.empty()) {
        throw std::logic_error("cannot sample an empty rollout action set");
    }
    if (actions.size() == 1) {
        return 0;
    }

    std::vector<double> probabilities;
    probabilities.reserve(actions.size());
    double total = 0.0;
    for (const MCTSActionEntry& action : actions) {
        const double probability = std::max(0.0, action.prior);
        probabilities.push_back(probability);
        total += probability;
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        const double uniform = 1.0 / static_cast<double>(actions.size());
        std::fill(probabilities.begin(), probabilities.end(), uniform);
    } else {
        for (double& probability : probabilities) {
            probability /= total;
        }
    }

    if (quantum > 0.0) {
        double quantized_total = 0.0;
        for (double& probability : probabilities) {
            probability = std::floor(probability / quantum + 0.5) * quantum;
            quantized_total += probability;
        }
        if (quantized_total > 0.0) {
            for (double& probability : probabilities) {
                probability /= quantized_total;
            }
        }
    }

    const double draw = random.random_double();
    double cumulative = 0.0;
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
        cumulative += probabilities[index];
        if (draw < cumulative) {
            return static_cast<int>(index);
        }
    }
    return static_cast<int>(actions.size() - 1);
}

}  // namespace

std::vector<MCTSActionEntry> UniformMCTS::legal_actions(
    const State& state) const {
    std::vector<MCTSActionEntry> result;
    if (state.next_player == Player::Controller) {
        ControllerActionSpace space = environment_.sample_controller_actions(state);
        result.reserve(space.canonical_actions.size());
        for (auto& action : space.canonical_actions) {
            CanonicalAction canonical{std::move(action)};
            result.push_back(MCTSActionEntry{
                action_representative(canonical),
                action_aliases(canonical),
                std::move(canonical),
                0.0});
        }
    } else {
        AdversaryActionSpace space = environment_.sample_adversary_actions(state);
        result.reserve(space.canonical_actions.size());
        for (auto& action : space.canonical_actions) {
            CanonicalAction canonical{std::move(action)};
            result.push_back(MCTSActionEntry{
                action_representative(canonical),
                action_aliases(canonical),
                std::move(canonical),
                0.0});
        }
    }
    std::sort(
        result.begin(), result.end(),
        [](const MCTSActionEntry& left, const MCTSActionEntry& right) {
            return left.representative_raw_index < right.representative_raw_index;
        });
    return result;
}

void UniformMCTS::assign_priors(
    const State& state,
    std::vector<MCTSActionEntry>& actions,
    double temperature,
    bool use_policy) const {
    if (actions.empty()) {
        return;
    }
    if (!use_policy) {
        const double uniform = 1.0 / static_cast<double>(actions.size());
        for (MCTSActionEntry& action : actions) {
            action.prior = uniform;
        }
        return;
    }
    if (inference_ == nullptr) {
        throw std::logic_error("policy priors require an inference runtime");
    }

    std::vector<float> logits;
    if (state.next_player == Player::Controller) {
        std::vector<CanonicalControllerAction> typed;
        typed.reserve(actions.size());
        for (const MCTSActionEntry& action : actions) {
            typed.push_back(std::get<CanonicalControllerAction>(action.action));
        }
        logits = inference_->predict_controller_logits(state, typed);
    } else {
        std::vector<CanonicalAdversaryAction> typed;
        typed.reserve(actions.size());
        for (const MCTSActionEntry& action : actions) {
            typed.push_back(std::get<CanonicalAdversaryAction>(action.action));
        }
        logits = inference_->predict_adversary_logits(state, typed);
    }

    const double safe_temperature = std::max(temperature, 1e-8);
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities;
    probabilities.reserve(logits.size());
    double softmax_total = 0.0;
    for (const float logit : logits) {
        const double probability =
            std::exp((static_cast<double>(logit) - maximum) / safe_temperature);
        probabilities.push_back(probability);
        softmax_total += probability;
    }
    if (!(softmax_total > 0.0) || !std::isfinite(softmax_total)) {
        throw std::runtime_error("policy softmax produced an invalid distribution");
    }

    double floored_total = 0.0;
    for (double& probability : probabilities) {
        probability = std::max(
            config_.prior_min_probability, probability / softmax_total);
        floored_total += probability;
    }
    for (std::size_t index = 0; index < actions.size(); ++index) {
        actions[index].prior = probabilities[index] / floored_total;
    }
}

double UniformMCTS::bootstrap_value(
    const State& state,
    Player player) const {
    if (!config_.use_model_bootstrap) {
        return 0.0;
    }
    if (inference_ == nullptr) {
        throw std::logic_error("model bootstrap requires an inference runtime");
    }
    const double value = inference_->predict_value(state, player);
    if (!std::isfinite(value) || value > 0.0) {
        throw std::runtime_error(
            "controller-valued bootstrap must be finite and nonpositive");
    }
    return value;
}

double UniformMCTS::one_rollout(
    const State& leaf_state,
    Player leaf_player,
    double target_time,
    std::uint64_t seed,
    bool capture_history) {
    State state = leaf_state;
    Player player = leaf_player;
    PythonRandomCompat random(seed);
    std::vector<RolloutEdge> edges;
    edges.reserve(static_cast<std::size_t>(config_.rollout_max_actions));
    std::uint32_t history_hash = 2166136261U;

    update_range(
        rollout_stats_.min_start_time,
        rollout_stats_.max_start_time,
        state.now);

    int action_count = 0;
    for (; action_count < config_.rollout_max_actions; ++action_count) {
        if (state.now >= target_time) {
            break;
        }
        if (state.next_player != player) {
            throw std::logic_error("rollout player does not match the state turn");
        }

        std::vector<MCTSActionEntry> actions = legal_actions(state);
        if (actions.empty()) {
            ++rollout_stats_.terminal_trajectories;
            break;
        }
        assign_priors(
            state,
            actions,
            config_.rollout_policy_temperature,
            config_.use_policy_prior);
        const int selected = sample_action(
            actions, config_.rollout_probability_quantum, random);
        const MCTSActionEntry& action =
            actions.at(static_cast<std::size_t>(selected));
        history_hash =
            (history_hash ^
             static_cast<std::uint32_t>(action.representative_raw_index)) *
            16777619U;

        const double parent_cost = state.objective.total_cost;
        const double parent_time = state.now;
        state = apply_rollout_action(environment_, state, action.action);
        edges.push_back(RolloutEdge{
            parent_cost - state.objective.total_cost,
            environment_.config().discount_for_elapsed(
                std::max(0.0, state.now - parent_time))});
        ++rollout_stats_.actions;
        player = state.next_player;
    }
    if (action_count == config_.rollout_max_actions && state.now < target_time) {
        throw std::runtime_error("rollout exceeded rollout_max_actions");
    }

    double value = bootstrap_value(state, player);
    ++rollout_stats_.bootstrap_calls;
    for (auto iterator = edges.rbegin(); iterator != edges.rend(); ++iterator) {
        value = iterator->reward + iterator->discount * value;
    }
    if (capture_history) {
        rollout_stats_.first_history_hash = history_hash;
        rollout_stats_.first_history_actions = static_cast<int>(edges.size());
    }
    update_range(
        rollout_stats_.min_final_time,
        rollout_stats_.max_final_time,
        state.now);
    return value;
}

double UniformMCTS::rollout_value(
    const State& leaf_state,
    Player leaf_player,
    double expansion_parent_time,
    double target_time) {
    update_range(
        rollout_stats_.min_expansion_parent_time,
        rollout_stats_.max_expansion_parent_time,
        expansion_parent_time);
    update_range(
        rollout_stats_.min_remaining_rollout_sec,
        rollout_stats_.max_remaining_rollout_sec,
        std::max(0.0, target_time - leaf_state.now));
    update_range(
        rollout_stats_.min_deadline,
        rollout_stats_.max_deadline,
        target_time);

    const std::uint64_t leaf_index =
        static_cast<std::uint64_t>(rollout_stats_.leaf_evaluations);
    double total = 0.0;
    for (int trajectory = 0; trajectory < config_.rollout_count; ++trajectory) {
        const std::uint64_t seed =
            config_.rollout_seed + leaf_index * 1000003ULL +
            static_cast<std::uint64_t>(trajectory) * 9176ULL;
        total += one_rollout(
            leaf_state,
            leaf_player,
            target_time,
            seed,
            leaf_index == 0 && trajectory == 0);
    }
    ++rollout_stats_.leaf_evaluations;
    rollout_stats_.trajectories += config_.rollout_count;
    return total / static_cast<double>(config_.rollout_count);
}

}  // namespace gv4
