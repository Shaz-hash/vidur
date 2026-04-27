#pragma once

#include "gv2_infer_runtime.hpp"
#include "gv2_mcts_dnn.hpp"
#include "gv2_types.hpp"
#include "gv2_virtual_environment.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace mcts_native_gv2 {

struct NativeRootSampleGV3 {
    int feature_version = 1;
    int game_id = 0;
    int root_id = 0;
    int root_node_id = 0;
    int root_depth = 0;
    std::string player = "adversary";

    std::vector<float> global_features;
    std::vector<uint8_t> action_mask;
    std::vector<float> prefill_req_features;
    std::vector<float> decode_req_features;
    std::vector<uint8_t> prefill_req_mask;
    std::vector<uint8_t> decode_req_mask;
    int prefill_req_n = 0;
    int prefill_req_d = 0;
    int decode_req_n = 0;
    int decode_req_d = 0;
    std::vector<float> req_features;
    std::vector<uint8_t> req_mask;
    int req_n = 0;
    int req_d = 0;

    std::vector<double> policy;
    double value = 0.0;

    int best_action_index = -1;
    bool used_bootstrap = false;
    int model_version = 0;
    int history_hops = 0;
    bool is_eval = false;
};

struct NativeSelfplayConfigGV3 {
    SearchInput search_template;
    SimState initial_state;
    bool has_initial_state = false;

    int game_id = 0;
    int num_roots = 0;
    int start_root_id = 0;
    int start_root_depth = 0;
    std::string start_player = "adversary";
    int feature_version = 1;

    int adv_iterations_per_root = 1;
    int cont_iterations_per_root = 1;
    int history_hops_min = 0;
    int history_hops_max = 0;
    int history_seed = 0;
    int history_max_total_steps = 20000;
    int max_forced_hops_per_root = 2000;
    bool allow_duplicate_history_fallback = true;
    std::vector<std::string> initial_seen_signatures;

    double eval_split_ratio = 0.0;
    int eval_split_seed = 0;
    int action_seed_base = 0;
};

struct NativeSelfplayResultGV3 {
    std::vector<NativeRootSampleGV3> samples;
    std::unordered_map<std::string, int> stats;
    std::vector<std::string> history_signatures;
};

NativeSelfplayResultGV3 generate_native_selfplay_samples_gv3(
    const NativeSelfplayConfigGV3& cfg,
    GV2VirtualEnvironment& env,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version);

}  // namespace mcts_native_gv2
