#pragma once

#include "gv2_infer_runtime.hpp"
#include "gv2_virtual_environment.hpp"
#include "gv2_types.hpp"

#include <string>
#include <vector>

namespace mcts_native_gv2 {

struct SearchInput {
    std::string contract_version = kGV2NativeContractVersion;
    SimState root_state;
    std::string root_player;
    int iterations = 0;

    int root_node_id = 0;
    int root_depth = 0;
    int game_id = 0;
    int root_id = 0;
    int seed = 0;

    // Search hyper-parameters (mirroring Python defaults unless overridden).
    int max_forced_hops = 2000;
    double pb_c_base = 5000.0;
    double pb_c_init = 0.75;
    double discount_factor = 0.98;
    double prefill_step_time = 0.015725797204323228;
    double reward_knee = 25.0;
    double reward_max_penalty = 40.0;
    double reward_tail_alpha = 1.0 / 15.0;

    // Environment/simulator config (optional overrides).
    GV2EnvConfig env_cfg;
    VirtualSimulatorConfig sim_cfg;
    std::string predictor_csv_path;

    std::vector<uint8_t> action_mask;
    std::vector<float> global_features;
    NativeInferInputsGV2 root_infer_inputs;
    bool reuse_root_infer_inputs = true;

    double decision_state_time = 0.0;
    std::string root_phase = "train_root";
    std::string cycle_label;
    bool log_events = true;
    bool profile = false;
};

SearchOutput run_search_torchscript(
    const SearchInput& in,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version);

}  // namespace mcts_native_gv2
