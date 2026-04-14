#pragma once

#include <c10/core/Device.h>
#include <torch/script.h>

#include "gv2_types.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcts_native_gv2 {

struct NativeInferInputsGV2 {
    std::vector<float> global_features;
    std::vector<uint8_t> action_mask;

    // Preferred split features (match GV2 Python model forward path).
    std::vector<float> prefill_req_features;  // flattened [Np * Dp]
    std::vector<float> decode_req_features;   // flattened [Nd * Dd]
    std::vector<uint8_t> prefill_req_mask;    // [Np]
    std::vector<uint8_t> decode_req_mask;     // [Nd]
    int prefill_req_n = 0;
    int prefill_req_d = 0;
    int decode_req_n = 0;
    int decode_req_d = 0;

    // Backward-compat fallback features.
    std::vector<float> req_features;  // flattened [N * D]
    std::vector<uint8_t> req_mask;    // [N]
    int req_n = 0;
    int req_d = 0;
};

struct NativeFeatureBuildConfigGV2 {
    int n_prefill_req = 10;
    int d_prefill_req = 5;
    int n_decode_req = 50;
    int d_decode_req = 5;
    int d_global = 11;

    double prefill_remaining_den = 4096.0;
    double decode_remaining_den = 864.0;
    double age_den_sec = 2.0;
    double lateness_den_sec = 2.0;
    double slack_drop_den_sec = 2.0;

    double system_load_den = 60.0;
    double active_prefill_count_den = 10.0;
    double active_decode_count_den = 50.0;
    double total_remaining_prefill_den = 40960.0;
    double total_decode_generated_active_den = 43200.0;
    double violated_count_den = 100.0;
    double prefill_near_drop_den = 10.0;
    double decode_near_drop_den = 50.0;

    double near_drop_lateness_low_sec = 0.5;
    double near_drop_lateness_high_sec = 1.5;

    double launch_ewma_alpha = 0.37;
    double launch_ewma_window_sec = 1.0;
    int launch_ewma_norm_den = 7;

    int decode_sample_seed_offset = 1337;
    double auto_drop_lateness_sec = 2.0;
};

class NativeTorchScriptInferRuntimeGV2 {
public:
    NativeTorchScriptInferRuntimeGV2(std::string device, double v_min, double v_step);

    // model_version -> model spec
    // spec formats:
    // 1) "/abs/path/model.pt"  (same module for controller/adversary)
    // 2) "/abs/path/controller.pt||/abs/path/adversary.pt"
    void load_models(const std::unordered_map<int, std::string>& model_version_to_path);

    std::pair<double, std::vector<double>> infer_from_inputs(
        const std::vector<float>& global_features,
        const std::vector<uint8_t>& action_mask,
        const std::string& player,
        int model_version);
    std::pair<double, std::vector<double>> infer_from_inputs(
        const NativeInferInputsGV2& inputs,
        const std::string& player,
        int model_version);
    NativeInferInputsGV2 build_inputs_from_state(
        const SimState& state,
        const std::vector<uint8_t>& action_mask,
        const NativeFeatureBuildConfigGV2& cfg,
        const NativeInferInputsGV2* template_inputs = nullptr) const;

    const std::string& device() const;
    double v_min() const;
    double v_step() const;

private:
    struct ModelPair {
        torch::jit::script::Module controller;
        torch::jit::script::Module adversary;
    };

    static c10::Device parse_device(const std::string& device);
    static std::pair<std::string, std::string> parse_model_spec(const std::string& spec);
    static std::vector<double> masked_softmax(
        const std::vector<double>& logits,
        const std::vector<uint8_t>& action_mask);

    double decode_value_from_tensor(const torch::Tensor& value_raw) const;

    std::string device_str_;
    c10::Device device_;
    double v_min_;
    double v_step_;
    std::unordered_map<int, std::string> model_specs_;

    std::unordered_map<int, std::shared_ptr<ModelPair>> models_;
    std::mutex mu_;
};

}  // namespace mcts_native_gv2
