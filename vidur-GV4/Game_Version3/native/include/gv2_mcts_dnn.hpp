#pragma once

#include "gv2_infer_runtime.hpp"
#include "gv2_virtual_environment.hpp"
#include "gv2_types.hpp"

#include <string>
#include <vector>

namespace mcts_native_gv2 {
inline double resolve_root_dirichlet_alpha(
    double fixed_alpha,
    double total_concentration,
    int canonical_action_count) noexcept {
    if (total_concentration > 0.0 && canonical_action_count > 0) {
        return total_concentration / static_cast<double>(canonical_action_count);
    }
    return fixed_alpha;
}


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
    double pb_c_base = 1500.0; // Decrease for more exploration : normal value 5000
    double pb_c_init = 1.25; // Increase for more exploration at low visit counts : normal value 0.75
    double discount_factor = 0.995;
    double prefill_step_time = 0.015725797204323228;
    double reward_knee = 25.0;
    double reward_max_penalty = 40.0;
    double reward_tail_alpha = 1.0 / 15.0;

    // Root Dirichlet noise (AlphaZero style): root-only
    bool root_dirichlet_noise_enabled = false;
    double root_dirichlet_alpha = 0.1;
    double root_dirichlet_total_concentration = 0.0;
    double root_dirichlet_epsilon = 0.35;

    // Environment/simulator config (optional overrides).
    GV2EnvConfig env_cfg;
    VirtualSimulatorConfig sim_cfg;
    NativeFeatureBuildConfigGV2 feature_cfg;
    std::string predictor_csv_path;

    std::vector<uint8_t> action_mask;
    std::vector<float> global_features;
    NativeInferInputsGV2 root_infer_inputs;
    bool reuse_root_infer_inputs = true;

    double decision_state_time = 0.0;
    std::string root_phase = "train_root";
    std::string cycle_label;
    bool use_model_bootstrap = true;
    bool log_events = true;
    bool profile = false;
};

SearchOutput run_search_torchscript(
    const SearchInput& in,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version);

SearchOutput run_search_torchscript_with_env(
    const SearchInput& in,
    GV2VirtualEnvironment& env,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version);

}  // namespace mcts_native_gv2
