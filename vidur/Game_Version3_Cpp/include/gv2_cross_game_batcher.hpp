#pragma once

#include "markov_value_features.hpp"

#include <string>
#include <vector>

namespace mcts_native_gv2 {

class NativeHGBModelRuntime;
class NewFeatures226HGBRuntime;
class VirtualSimulatorGV2;
struct SimState;

class CrossGameInferenceBatcher {
public:
    virtual ~CrossGameInferenceBatcher() = default;

    virtual std::vector<double> predict_markov_policy(
        const NativeHGBModelRuntime& runtime,
        const std::string& player,
        const std::vector<MarkovValueFeatures>& markov_states,
        const std::vector<float>& flat_actions,
        int num_rows,
        const std::vector<int>& group_offsets) = 0;

    virtual std::vector<double> infer_values(
        const NewFeatures226HGBRuntime& runtime,
        const std::vector<SimState>& states,
        const std::vector<const VirtualSimulatorGV2*>& simulators) = 0;
};

}  // namespace mcts_native_gv2
