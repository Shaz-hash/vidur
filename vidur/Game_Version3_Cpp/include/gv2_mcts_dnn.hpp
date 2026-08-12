#pragma once

#include "gv2_infer_runtime.hpp"
#include "gv2_virtual_environment.hpp"
#include "new_features_226_inference.hpp"
#include "gv2_types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mcts_native_gv2 {
class CrossGameInferenceBatcher;
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
    double uct_c = 1.4;
    bool use_policy_prior = false;
    double puct_c = 2.5;
    double policy_prior_temperature = 1.0;
    double prior_min_prob = 1e-8;
    double pb_c_base = 1500.0; // Decrease for more exploration : normal value 5000
    double pb_c_init = 1.25; // Increase for more exploration at low visit counts : normal value 0.75
    double discount_factor = 0.995;
    double prefill_step_time = 0.015725797204323228;
    double reward_knee = 25.0;
    double reward_max_penalty = 40.0;
    double reward_tail_alpha = 1.0 / 15.0;

    // Optional policy-guided leaf continuations for full-tree search.
    int rollout_count = 10;
    int rollout_parallel_threads = 10;
    // Zero preserves the legacy behavior of using rollout_parallel_threads.
    int rollout_policy_parallel_threads = 0;

    double rollout_horizon_sec = 0.4;
    double rollout_policy_temperature = 1.0;
    double rollout_probability_quantum = 1e-6;
    int rollout_max_actions = 4096;
    bool capture_rollout_trace = false;
    bool capture_root_puct_trace = false;
    // Local A/B switch for semantics-preserving rollout execution changes.
    // Keep the reference path available until differential validation passes.
    bool rollout_optimized_execution = true;

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
    std::string search_mode = "depth_one";
    bool log_events = true;
    bool profile = false;

    // Test-only hook used by the independent-game batching benchmark. Normal
    // production searches leave this null and retain the synchronous path.
    CrossGameInferenceBatcher* cross_game_inference_batcher = nullptr;
};

struct CrossGameBatchStats {
    std::int64_t policy_requests = 0;
    std::int64_t policy_batches = 0;
    std::int64_t policy_action_rows = 0;
    std::int64_t value_requests = 0;
    std::int64_t value_batches = 0;
    std::int64_t value_states = 0;
    std::size_t max_policy_batch_requests = 0;
    std::size_t max_value_batch_requests = 0;
    double policy_inference_sec = 0.0;
    double value_inference_sec = 0.0;
};

struct CrossGameBatchResult {
    std::vector<SearchOutput> outputs;
    CrossGameBatchStats stats;
    double elapsed_sec = 0.0;
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

SearchOutput run_search_torchscript_value_prior(
    const SearchInput& in,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version);

SearchOutput run_search_torchscript_value_prior_with_env(
    const SearchInput& in,
    GV2VirtualEnvironment& env,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version);

SearchOutput run_search_hgb226(
    const SearchInput& in,
    NewFeatures226HGBRuntime& infer_runtime);

SearchOutput run_search_hgb226_with_env(
    const SearchInput& in,
    GV2VirtualEnvironment& env,
    NewFeatures226HGBRuntime& infer_runtime);

SearchOutput run_search_hgb226_value_prior(
    const SearchInput& in,
    NewFeatures226HGBRuntime& infer_runtime);

SearchOutput run_search_hgb226_value_prior_with_env(
    const SearchInput& in,
    GV2VirtualEnvironment& env,
    NewFeatures226HGBRuntime& infer_runtime);

SearchOutput run_search_hgb226_value_prior_with_env(
    const SearchInput& in,
    GV2VirtualEnvironment& env,
    NewFeatures226HGBRuntime& infer_runtime,
    NativeHGBModelRuntime& controller_prior_runtime,
    NativeHGBModelRuntime& adversary_prior_runtime);

CrossGameBatchResult run_search_hgb226_value_prior_cross_game_batch(
    const std::vector<SearchInput>& inputs,
    const GV2VirtualEnvironment& env,
    NewFeatures226HGBRuntime& infer_runtime,
    NativeHGBModelRuntime& controller_prior_runtime,
    NativeHGBModelRuntime& adversary_prior_runtime,
    int worker_threads,
    int inference_threads,
    int max_batch_requests,
    int max_batch_wait_us);

CrossGameBatchResult run_search_hgb226_value_prior_parallel_baseline(
    const std::vector<SearchInput>& inputs,
    const GV2VirtualEnvironment& env,
    NewFeatures226HGBRuntime& infer_runtime,
    NativeHGBModelRuntime& controller_prior_runtime,
    NativeHGBModelRuntime& adversary_prior_runtime,
    int worker_threads);

}  // namespace mcts_native_gv2
