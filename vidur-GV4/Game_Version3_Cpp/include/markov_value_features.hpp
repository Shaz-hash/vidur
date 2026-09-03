#pragma once

#include "gv2_types.hpp"

#include <vector>

namespace mcts_native_gv2 {

inline constexpr const char* kMarkovValueFeatureSchema = "markov_v2";
inline constexpr int kMarkovGlobalDim = 19;
inline constexpr int kMarkovRequestDim = 24;
inline constexpr int kMarkovLaunchDim = 3;

struct MarkovValueFeatures {
    std::vector<float> global_features;
    std::vector<float> request_features;
    std::vector<float> launch_features;
    std::vector<int> request_ids;
    int request_count = 0;
    int launch_count = 0;
};

MarkovValueFeatures build_markov_value_features(
    const SimState& state,
    double launch_window_sec = 1.0);

}  // namespace mcts_native_gv2
