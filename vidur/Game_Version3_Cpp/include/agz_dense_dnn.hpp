#pragma once

#include "markov_value_features.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace mcts_native_gv2 {

class NativeDenseDNNModel {
public:
    bool load_model_export(const std::string& path);

    double predict_value(const std::vector<float>& state_features) const;
    double predict_markov_value(const MarkovValueFeatures& features) const;
    double predict_policy(const std::vector<float>& state_action_features) const;
    std::vector<double> predict_policy_batch_flat(
        const std::vector<float>& flat_features,
        int num_rows,
        int row_dim) const;
    std::vector<double> predict_policy_grouped_batch_flat(
        const std::vector<float>& flat_features,
        int num_rows,
        int row_dim,
        const std::vector<int>& group_offsets,
        int parallel_threads) const;
    std::vector<double> predict_policy_grouped_split_batch_flat(
        const std::vector<float>& flat_states,
        const std::vector<float>& flat_actions,
        int num_rows,
        const std::vector<int>& group_offsets,
        int parallel_threads) const;

    bool loaded() const;
    bool is_value() const;
    bool is_policy() const;
    bool is_markov_value() const;
    int feature_dim() const;
    int state_dim() const;
    int action_dim() const;
    const std::string& model_tag() const;
    const std::string& model_kind() const;

    const std::string& feature_schema() const;
    int global_dim() const;
    int request_dim() const;
    int launch_dim() const;
private:
    struct Tensor {
        std::vector<int> shape;
        std::vector<float> values;
        std::vector<float> transposed_values;
    };

    struct PolicyBatchTensorRefs {
        const Tensor* state_fc_weight = nullptr;
        const Tensor* state_fc_bias = nullptr;
        const Tensor* state_norm_weight = nullptr;
        const Tensor* state_norm_bias = nullptr;
        const Tensor* fusion_bias = nullptr;
        const Tensor* action_fc_weight = nullptr;
        const Tensor* action_fc_bias = nullptr;
        const Tensor* action_norm_weight = nullptr;
        const Tensor* action_norm_bias = nullptr;
        const Tensor* fusion_norm_weight = nullptr;
        const Tensor* fusion_norm_bias = nullptr;
        const Tensor* block_norm_weight = nullptr;
        const Tensor* block_norm_bias = nullptr;
        const Tensor* block_fc1_weight = nullptr;
        const Tensor* block_fc1_bias = nullptr;
        const Tensor* block_fc2_weight = nullptr;
        const Tensor* block_fc2_bias = nullptr;
        const Tensor* head_norm_weight = nullptr;
        const Tensor* head_norm_bias = nullptr;
        const Tensor* head1_weight = nullptr;
        const Tensor* head1_bias = nullptr;
        const Tensor* head2_weight = nullptr;
        const Tensor* head2_bias = nullptr;
    };

    struct PolicyActionTensorRefs {
        const Tensor* action_fc_weight;
        const Tensor* action_fc_bias;
        const Tensor* action_norm_weight;
        const Tensor* action_norm_bias;
        const Tensor* fusion_weight;
        const Tensor* fusion_norm_weight;
        const Tensor* fusion_norm_bias;
        const Tensor* block_norm_weight;
        const Tensor* block_norm_bias;
        const Tensor* block_fc1_weight;
        const Tensor* block_fc1_bias;
        const Tensor* block_fc2_weight;
        const Tensor* block_fc2_bias;
        const Tensor* head_norm_weight;
        const Tensor* head_norm_bias;
        const Tensor* head1_weight;
        const Tensor* head1_bias;
        const Tensor* head2_weight;
        const Tensor* head2_bias;
    };

    const Tensor& tensor(const std::string& name) const;
    static void linear_into(
        const float* input,
        int input_dim,
        const Tensor& weight,
        const Tensor& bias,
        float* output);
    static void linear_batch_into(
        const float* input,
        int rows,
        int input_dim,
        const Tensor& weight,
        const Tensor& bias,
        float* output);
    static void layer_norm_into(
        const float* input,
        int size,
        const Tensor& weight,
        const Tensor& bias,
        float epsilon,
        float* output);
    static void silu_inplace(float* values, int size);

    double value_from_state(const float* state) const;
    void policy_state_embedding_into(const float* state, float* output) const;
    double markov_value_from_features(const MarkovValueFeatures& features) const;
    double policy_from_embeddings(
        const float* state_embedding,
        const float* action) const;
    void policy_fusion_state_base_into(
        const float* state_embedding,
        float* output) const;
    double policy_from_fusion_state_base(
        const float* fusion_state_base,
        const float* action) const;
    PolicyActionTensorRefs policy_action_tensor_refs() const;
    double policy_from_fusion_state_base(
        const float* fusion_state_base,
        const float* action,
        const PolicyActionTensorRefs& refs) const;

    bool loaded_ = false;
    std::string model_kind_;
    std::string model_tag_;
    std::string role_;
    std::string architecture_;
    int feature_dim_ = 0;
    std::string feature_schema_;
    int state_dim_ = 226;
    int action_dim_ = 0;
    float value_min_ = -50.0f;
    int global_dim_ = 0;
    int request_dim_ = 0;
    int launch_dim_ = 0;
    float value_max_ = 0.0f;
    float layer_norm_eps_ = 1e-5f;
    std::unordered_map<std::string, Tensor> tensors_;
    Tensor policy_fusion_state_weight_;
    Tensor policy_fusion_action_weight_;
    Tensor policy_fusion_zero_bias_;
    PolicyBatchTensorRefs policy_batch_refs_;
};

}  // namespace mcts_native_gv2
