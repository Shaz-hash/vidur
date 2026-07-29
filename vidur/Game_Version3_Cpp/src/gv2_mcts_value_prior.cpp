#include "gv2_mcts_dnn.hpp"

#include <cmath>

namespace mcts_native_gv2 {
namespace {

SearchInput as_value_prior_input(const SearchInput& in) {
    SearchInput out = in;
    if (out.search_mode != "full_tree_rollout") {
        out.search_mode = "full_tree";
    }
    out.use_policy_prior = true;
    if (!std::isfinite(out.puct_c) || out.puct_c == 0.0) {
        out.puct_c = 2.5;
    }
    if (!std::isfinite(out.policy_prior_temperature) || out.policy_prior_temperature <= 0.0) {
        out.policy_prior_temperature = 1.0;
    }
    if (!std::isfinite(out.prior_min_prob) || out.prior_min_prob < 0.0) {
        out.prior_min_prob = 1e-8;
    }
    return out;
}

}  // namespace

SearchOutput run_search_torchscript_value_prior(
    const SearchInput& in,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version) {
    SearchInput prior_in = as_value_prior_input(in);
    return run_search_torchscript(prior_in, infer_runtime, model_version);
}

SearchOutput run_search_torchscript_value_prior_with_env(
    const SearchInput& in,
    GV2VirtualEnvironment& env,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version) {
    SearchInput prior_in = as_value_prior_input(in);
    return run_search_torchscript_with_env(prior_in, env, infer_runtime, model_version);
}

SearchOutput run_search_hgb226_value_prior(
    const SearchInput& in,
    NewFeatures226HGBRuntime& infer_runtime) {
    SearchInput prior_in = as_value_prior_input(in);
    return run_search_hgb226(prior_in, infer_runtime);
}

SearchOutput run_search_hgb226_value_prior_with_env(
    const SearchInput& in,
    GV2VirtualEnvironment& env,
    NewFeatures226HGBRuntime& infer_runtime) {
    SearchInput prior_in = as_value_prior_input(in);
    return run_search_hgb226_with_env(prior_in, env, infer_runtime);
}

}  // namespace mcts_native_gv2
