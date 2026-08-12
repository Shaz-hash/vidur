#include "agz_dense_dnn.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef GV2_HAVE_OPENBLAS_ILP64
extern "C" void cblas_sgemm64_(
    int layout,
    int transpose_a,
    int transpose_b,
    long long rows_a,
    long long columns_b,
    long long shared_dimension,
    float alpha,
    const float* a,
    long long leading_a,
    const float* b,
    long long leading_b,
    float beta,
    float* c,
    long long leading_c);
extern "C" void openblas_set_num_threads64_(int threads);
#endif




namespace mcts_native_gv2 {
namespace {

std::string trim_copy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> out;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delimiter)) out.push_back(item);
    if (!value.empty() && value.back() == delimiter) out.emplace_back();
    return out;
}

std::size_t element_count(const std::vector<int>& shape) {
    std::size_t count = 1;
    for (int dim : shape) {
        if (dim <= 0) throw std::runtime_error("DNN tensor has non-positive dimension");
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

struct PolicyActionFeatureKey {
    std::array<std::uint32_t, 43> bits{};
    std::uint8_t size = 0;

    bool operator==(const PolicyActionFeatureKey& other) const {
        return size == other.size &&
            std::equal(bits.begin(), bits.begin() + size, other.bits.begin());
    }
};

struct PolicyActionFeatureKeyHash {
    std::size_t operator()(const PolicyActionFeatureKey& key) const {
        std::uint64_t hash = 1469598103934665603ULL ^ key.size;
        const std::size_t pair_count =
            static_cast<std::size_t>(key.size) / 2U;
        for (std::size_t index = 0; index < pair_count; ++index) {
            std::uint64_t word = 0;
            std::memcpy(
                &word,
                key.bits.data() + index * 2U,
                sizeof(word));
            hash ^= word;
            hash *= 1099511628211ULL;
        }
        if ((key.size & 1U) != 0U) {
            hash ^= static_cast<std::uint64_t>(
                key.bits[static_cast<std::size_t>(key.size) - 1U]);
            hash *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(hash);
    }
};

PolicyActionFeatureKey policy_action_feature_key(
    const float* action,
    int action_dim) {
    if (action_dim < 0 || action_dim > 43) {
        throw std::runtime_error("invalid policy action key dimension");
    }
    PolicyActionFeatureKey key;
    key.size = static_cast<std::uint8_t>(action_dim);
    std::memcpy(
        key.bits.data(),
        action,
        static_cast<std::size_t>(action_dim) * sizeof(float));
    return key;
}

struct PolicySplitBatchWorkspace {
    using ActionCache = std::unordered_map<
        PolicyActionFeatureKey,
        std::array<float, 128>,
        PolicyActionFeatureKeyHash>;

    std::vector<std::array<float, 128>> fusion_state_bases;
    std::vector<float> state_embeddings;
    std::vector<int> row_groups;
    std::unordered_map<const NativeDenseDNNModel*, ActionCache>
        action_fusion_cache;
    std::vector<int> row_action_indices;
    std::vector<PolicyActionFeatureKey> unique_action_keys;
    std::vector<float> unique_actions;
    std::vector<float> action_embedding;
    std::vector<float> action_fusion;
    std::vector<const std::array<float, 128>*> row_action_bases;
    std::vector<const std::array<float, 128>*> unique_action_bases;
    std::vector<float> hidden;
    std::vector<float> normalized;
    std::vector<float> bottleneck;
    std::vector<float> residual;
    std::vector<float> head;
    std::vector<float> scalar;
};

}  // namespace

bool NativeDenseDNNModel::load_model_export(const std::string& path) {
    std::ifstream input(path);
    if (!input.good()) throw std::runtime_error("failed to open native DNN export: " + path);

    loaded_ = false;
    model_kind_.clear();
    model_tag_.clear();
    role_.clear();
    architecture_.clear();
    feature_schema_.clear();
    feature_dim_ = 0;
    state_dim_ = 226;
    action_dim_ = 0;
    value_min_ = -50.0f;
    value_max_ = 0.0f;
    layer_norm_eps_ = 1e-5f;
    global_dim_ = 0;
    request_dim_ = 0;
    launch_dim_ = 0;
    tensors_.clear();
    policy_fusion_state_weight_.shape.clear();
    policy_fusion_state_weight_.values.clear();
    policy_fusion_action_weight_.shape.clear();
    policy_fusion_action_weight_.values.clear();
    policy_fusion_zero_bias_.shape.clear();
    policy_fusion_zero_bias_.values.clear();
    policy_batch_refs_ = {};

    std::string line;
    if (!std::getline(input, line) ||
        (trim_copy(line) != "agz_dnn_v1" && trim_copy(line) != "agz_dnn_v2")) {
        throw std::runtime_error("invalid native DNN export header: " + path);
    }
    while (std::getline(input, line)) {
        line = trim_copy(line);
        if (line.empty() || line.front() == '#') continue;
        const auto columns = split(line, '\t');
        if (columns.empty()) continue;
        const std::string& kind = columns[0];
        if (kind == "model_kind" && columns.size() >= 2) {
            model_kind_ = columns[1];
        } else if (kind == "architecture" && columns.size() >= 2) {
            architecture_ = columns[1];
        } else if (kind == "model_tag" && columns.size() >= 2) {
            model_tag_ = columns[1];
        } else if (kind == "role" && columns.size() >= 2) {
            role_ = columns[1];
        } else if (kind == "feature_dim" && columns.size() >= 2) {
            feature_dim_ = std::stoi(columns[1]);
        } else if (kind == "state_dim" && columns.size() >= 2) {
            state_dim_ = std::stoi(columns[1]);
        } else if (kind == "action_dim" && columns.size() >= 2) {
            action_dim_ = std::stoi(columns[1]);
        } else if (kind == "feature_schema" && columns.size() >= 2) {
            feature_schema_ = columns[1];
        } else if (kind == "global_dim" && columns.size() >= 2) {
            global_dim_ = std::stoi(columns[1]);
        } else if (kind == "request_dim" && columns.size() >= 2) {
            request_dim_ = std::stoi(columns[1]);
        } else if (kind == "launch_dim" && columns.size() >= 2) {
            launch_dim_ = std::stoi(columns[1]);
        } else if (kind == "value_min" && columns.size() >= 2) {
            value_min_ = std::stof(columns[1]);
        } else if (kind == "value_max" && columns.size() >= 2) {
            value_max_ = std::stof(columns[1]);
        } else if (kind == "layer_norm_eps" && columns.size() >= 2) {
            layer_norm_eps_ = std::stof(columns[1]);
        } else if (kind == "tensor" && columns.size() >= 4) {
            Tensor tensor_value;
            for (const auto& raw : split(columns[2], ',')) {
                if (!raw.empty()) tensor_value.shape.push_back(std::stoi(raw));
            }
            for (const auto& raw : split(columns[3], ',')) {
                if (!raw.empty()) tensor_value.values.push_back(std::stof(raw));
            }
            if (tensor_value.values.size() != element_count(tensor_value.shape)) {
                throw std::runtime_error("native DNN tensor size mismatch: " + columns[1]);
            }
            tensors_.emplace(columns[1], std::move(tensor_value));
        }
    }

    for (auto& tensor_entry : tensors_) {
        auto& tensor_value = tensor_entry.second;
        if (tensor_value.shape.size() != 2) continue;
        const int output_dim = tensor_value.shape[0];
        const int input_dim = tensor_value.shape[1];
        tensor_value.transposed_values.resize(
            static_cast<std::size_t>(output_dim) * input_dim);
        for (int output_index = 0; output_index < output_dim; ++output_index) {
            for (int input_index = 0; input_index < input_dim; ++input_index) {
                tensor_value.transposed_values[
                    static_cast<std::size_t>(input_index) * output_dim +
                    output_index] = tensor_value.values[
                        static_cast<std::size_t>(output_index) * input_dim +
                        input_index];
            }
        }
    }

    const bool legacy_architecture = architecture_ == "agz_residual_mlp_v1";
    const bool markov_value_architecture =
        architecture_ == "agz_markov_value_deepset_v2";
    const bool markov_policy_architecture =
        architecture_ == "agz_markov_policy_deepset_v3";
    const bool markov_architecture =
        markov_value_architecture || markov_policy_architecture;
    if (!legacy_architecture && !markov_architecture) {
        throw std::runtime_error("unsupported native DNN architecture: " + architecture_);
    }
    if (!is_value() && !is_policy()) {
        throw std::runtime_error("unsupported native DNN model kind: " + model_kind_);
    }
    if (markov_architecture) {
        if (feature_schema_ != kMarkovValueFeatureSchema ||
            feature_dim_ != 0 || state_dim_ != 0 ||
            global_dim_ != kMarkovGlobalDim ||
            request_dim_ != kMarkovRequestDim ||
            launch_dim_ != kMarkovLaunchDim) {
            throw std::runtime_error("invalid native Markov DNN schema or dimensions");
        }
        if (markov_value_architecture &&
            (!is_value() || action_dim_ != 0)) {
            throw std::runtime_error("invalid native Markov value DNN dimensions");
        }
        if (markov_policy_architecture &&
            (!is_policy() || (action_dim_ != 43 && action_dim_ != 7))) {
            throw std::runtime_error("invalid native Markov policy DNN dimensions");
        }
        tensor("global_fc.weight");
        tensor("request_fc1.weight");
        tensor("launch_fc1.weight");
        tensor("head2.bias");
        if (markov_policy_architecture) tensor("state_fusion_fc.weight");
    } else {
        if (state_dim_ != 226) {
            throw std::runtime_error("native DNN state dimension must be 226");
        }
        if (is_value()) {
            if (feature_dim_ != state_dim_ || action_dim_ != 0) {
                throw std::runtime_error("invalid native value DNN dimensions");
            }
            tensor("stem.weight");
            tensor("head2.bias");
        } else {
            if (action_dim_ != 43 && action_dim_ != 7) {
                throw std::runtime_error("invalid native policy DNN action dimension");
            }
            if (feature_dim_ != state_dim_ + action_dim_) {
                throw std::runtime_error("invalid native policy DNN feature dimension");
            }
            tensor("state_fc.weight");
            tensor("head2.bias");
        }
    }
    if (is_policy()) {
        const auto& fusion_weight = tensor("fusion_fc.weight");
        policy_fusion_state_weight_.shape = {128, 192};
        policy_fusion_state_weight_.values.resize(128U * 192U);
        for (int row = 0; row < 128; ++row) {
            std::copy_n(
                fusion_weight.values.data() +
                    static_cast<std::size_t>(row) * 256U,
                192,
                policy_fusion_state_weight_.values.data() +
                    static_cast<std::size_t>(row) * 192U);
        }
        policy_fusion_state_weight_.transposed_values.resize(128U * 192U);
        for (int row = 0; row < 128; ++row) {
            for (int column = 0; column < 192; ++column) {
                policy_fusion_state_weight_.transposed_values[
                    static_cast<std::size_t>(column) * 128U + row] =
                    policy_fusion_state_weight_.values[
                        static_cast<std::size_t>(row) * 192U + column];
            }
        }
        policy_fusion_action_weight_.shape = {128, 64};
        policy_fusion_action_weight_.values.resize(128U * 64U);
        for (int row = 0; row < 128; ++row) {
            std::copy_n(
                fusion_weight.values.data() +
                    static_cast<std::size_t>(row) * 256U + 192U,
                64,
                policy_fusion_action_weight_.values.data() +
                    static_cast<std::size_t>(row) * 64U);
        }
        policy_fusion_action_weight_.transposed_values.resize(128U * 64U);
        for (int row = 0; row < 128; ++row) {
            for (int column = 0; column < 64; ++column) {
                policy_fusion_action_weight_.transposed_values[
                    static_cast<std::size_t>(column) * 128U + row] =
                    policy_fusion_action_weight_.values[
                        static_cast<std::size_t>(row) * 64U + column];
            }
        }
        policy_fusion_zero_bias_.shape = {128};
        policy_fusion_zero_bias_.values.assign(128U, 0.0f);
        policy_batch_refs_ = {};
        if (!markov_policy_architecture) {
            policy_batch_refs_.state_fc_weight = &tensor("state_fc.weight");
            policy_batch_refs_.state_fc_bias = &tensor("state_fc.bias");
            policy_batch_refs_.state_norm_weight = &tensor("state_norm.weight");
            policy_batch_refs_.state_norm_bias = &tensor("state_norm.bias");
        }
        policy_batch_refs_.fusion_bias = &tensor("fusion_fc.bias");
        policy_batch_refs_.action_fc_weight = &tensor("action_fc.weight");
        policy_batch_refs_.action_fc_bias = &tensor("action_fc.bias");
        policy_batch_refs_.action_norm_weight = &tensor("action_norm.weight");
        policy_batch_refs_.action_norm_bias = &tensor("action_norm.bias");
        policy_batch_refs_.fusion_norm_weight = &tensor("fusion_norm.weight");
        policy_batch_refs_.fusion_norm_bias = &tensor("fusion_norm.bias");
        policy_batch_refs_.block_norm_weight = &tensor("fusion_block.norm.weight");
        policy_batch_refs_.block_norm_bias = &tensor("fusion_block.norm.bias");
        policy_batch_refs_.block_fc1_weight = &tensor("fusion_block.fc1.weight");
        policy_batch_refs_.block_fc1_bias = &tensor("fusion_block.fc1.bias");
        policy_batch_refs_.block_fc2_weight = &tensor("fusion_block.fc2.weight");
        policy_batch_refs_.block_fc2_bias = &tensor("fusion_block.fc2.bias");
        policy_batch_refs_.head_norm_weight = &tensor("head_norm.weight");
        policy_batch_refs_.head_norm_bias = &tensor("head_norm.bias");
        policy_batch_refs_.head1_weight = &tensor("head1.weight");
        policy_batch_refs_.head1_bias = &tensor("head1.bias");
        policy_batch_refs_.head2_weight = &tensor("head2.weight");
        policy_batch_refs_.head2_bias = &tensor("head2.bias");
    }

    if (model_tag_.empty()) model_tag_ = model_kind_ + ":" + role_;
    loaded_ = true;
    return true;
}

const NativeDenseDNNModel::Tensor& NativeDenseDNNModel::tensor(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) throw std::runtime_error("missing native DNN tensor: " + name);
    return found->second;
}

void NativeDenseDNNModel::linear_into(
    const float* input,
    int input_dim,
    const Tensor& weight,
    const Tensor& bias,
    float* output) {
    if (weight.shape.size() != 2 || bias.shape.size() != 1) {
        throw std::runtime_error("invalid native DNN linear tensor rank");
    }
    const int output_dim = weight.shape[0];
    if (weight.shape[1] != input_dim || bias.shape[0] != output_dim) {
        throw std::runtime_error("invalid native DNN linear tensor dimensions");
    }
    for (int row = 0; row < output_dim; ++row) {
        const float* weights = weight.values.data() +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(input_dim);
        float total = bias.values[static_cast<std::size_t>(row)];
        for (int column = 0; column < input_dim; ++column) {
            total += weights[column] * input[column];
        }
        output[row] = total;
    }
}

void NativeDenseDNNModel::linear_batch_into(
    const float* input,
    int rows,
    int input_dim,
    const Tensor& weight,
    const Tensor& bias,
    float* output) {
    if (rows < 0 || weight.shape.size() != 2 || bias.shape.size() != 1 ||
        weight.shape[1] != input_dim || bias.shape[0] != weight.shape[0]) {
        throw std::runtime_error("invalid native DNN batched linear dimensions");
    }
    const int output_dim = weight.shape[0];
    if (rows == 0) return;
#ifdef GV2_HAVE_OPENBLAS_ILP64
    constexpr int kRowMajor = 101;
    constexpr int kNoTranspose = 111;
    cblas_sgemm64_(
        kRowMajor,
        kNoTranspose,
        kNoTranspose,
        static_cast<long long>(rows),
        static_cast<long long>(output_dim),
        static_cast<long long>(input_dim),
        1.0f,
        input,
        static_cast<long long>(input_dim),
        weight.transposed_values.data(),
        static_cast<long long>(output_dim),
        0.0f,
        output,
        static_cast<long long>(output_dim));
    for (int row = 0; row < rows; ++row) {
        float* destination =
            output + static_cast<std::size_t>(row) * output_dim;
        for (int column = 0; column < output_dim; ++column) {
            destination[column] +=
                bias.values[static_cast<std::size_t>(column)];
        }
    }
#else
    for (int row = 0; row < rows; ++row) {
        linear_into(
            input + static_cast<std::size_t>(row) * input_dim,
            input_dim,
            weight,
            bias,
            output + static_cast<std::size_t>(row) * output_dim);
    }
#endif
}
void NativeDenseDNNModel::layer_norm_into(
    const float* input,
    int size,
    const Tensor& weight,
    const Tensor& bias,
    float epsilon,
    float* output) {
    if (weight.shape.size() != 1 || bias.shape.size() != 1 ||
        static_cast<int>(weight.values.size()) != size ||
        static_cast<int>(bias.values.size()) != size) {
        throw std::runtime_error("invalid native DNN layer norm dimensions");
    }
    double mean = 0.0;
    for (int i = 0; i < size; ++i) mean += static_cast<double>(input[i]);
    mean /= static_cast<double>(size);
    double variance = 0.0;
    for (int i = 0; i < size; ++i) {
        const double centered = static_cast<double>(input[i]) - mean;
        variance += centered * centered;
    }
    variance /= static_cast<double>(size);
    const double inverse_std = 1.0 / std::sqrt(variance + static_cast<double>(epsilon));
    for (int i = 0; i < size; ++i) {
        const double normalized = (static_cast<double>(input[i]) - mean) * inverse_std;
        output[i] = static_cast<float>(
            normalized * static_cast<double>(weight.values[static_cast<std::size_t>(i)]) +
            static_cast<double>(bias.values[static_cast<std::size_t>(i)]));
    }
}

void NativeDenseDNNModel::silu_inplace(float* values, int size) {
    for (int i = 0; i < size; ++i) {
        const float x = values[i];
        if (x >= 0.0) {
            values[i] = x / (1.0f + std::exp(-x));
        } else {
            const float exp_x = std::exp(x);
            values[i] = x * exp_x / (1.0f + exp_x);
        }
    }
}

double NativeDenseDNNModel::value_from_state(const float* state) const {
    std::array<float, 192> hidden{};
    std::array<float, 192> normalized_buffer{};
    std::array<float, 192> residual{};
    std::array<float, 64> bottleneck{};
    std::array<float, 32> head{};
    float scalar = 0.0f;

    linear_into(state, 226, tensor("stem.weight"), tensor("stem.bias"), hidden.data());
    silu_inplace(hidden.data(), 192);
    for (const char* block : {"block1", "block2"}) {
        const std::string prefix(block);
        layer_norm_into(
            hidden.data(), 192,
            tensor(prefix + ".norm.weight"), tensor(prefix + ".norm.bias"),
            layer_norm_eps_, normalized_buffer.data());
        linear_into(
            normalized_buffer.data(), 192,
            tensor(prefix + ".fc1.weight"), tensor(prefix + ".fc1.bias"),
            bottleneck.data());
        silu_inplace(bottleneck.data(), 64);
        linear_into(
            bottleneck.data(), 64,
            tensor(prefix + ".fc2.weight"), tensor(prefix + ".fc2.bias"),
            residual.data());
        for (int i = 0; i < 192; ++i) hidden[static_cast<std::size_t>(i)] += residual[static_cast<std::size_t>(i)];
    }
    layer_norm_into(
        hidden.data(), 192,
        tensor("head_norm.weight"), tensor("head_norm.bias"),
        layer_norm_eps_, normalized_buffer.data());
    linear_into(
        normalized_buffer.data(), 192,
        tensor("head1.weight"), tensor("head1.bias"), head.data());
    silu_inplace(head.data(), 32);
    linear_into(head.data(), 32, tensor("head2.weight"), tensor("head2.bias"), &scalar);
    const double normalized = std::tanh(static_cast<double>(scalar));
    return static_cast<double>(value_min_) +
        0.5 * (normalized + 1.0) *
        static_cast<double>(value_max_ - value_min_);
}

double NativeDenseDNNModel::markov_value_from_features(
    const MarkovValueFeatures& features) const {
    if (static_cast<int>(features.global_features.size()) != global_dim_ ||
        features.request_count < 0 || features.launch_count < 0 ||
        static_cast<int>(features.request_features.size()) !=
            features.request_count * request_dim_ ||
        static_cast<int>(features.launch_features.size()) !=
            features.launch_count * launch_dim_) {
        throw std::runtime_error("native Markov DNN feature dimensions mismatch");
    }

    std::array<float, 64> global_embedding{};
    std::array<float, 64> request_hidden{};
    std::array<float, 64> request_embedding{};
    std::array<float, 64> request_sum{};
    std::array<float, 64> request_max{};
    std::array<float, 32> launch_hidden{};
    std::array<float, 32> launch_embedding{};
    std::array<float, 32> launch_sum{};
    std::array<float, 32> launch_max{};
    std::array<float, 256> fusion_input{};
    std::array<float, 192> hidden{};
    std::array<float, 192> normalized_buffer{};
    std::array<float, 192> residual{};
    std::array<float, 64> bottleneck{};
    std::array<float, 32> head{};
    float scalar = 0.0f;

    linear_into(
        features.global_features.data(), global_dim_,
        tensor("global_fc.weight"), tensor("global_fc.bias"),
        global_embedding.data());
    layer_norm_into(
        global_embedding.data(), 64,
        tensor("global_norm.weight"), tensor("global_norm.bias"),
        layer_norm_eps_, global_embedding.data());
    silu_inplace(global_embedding.data(), 64);

    request_max.fill(std::numeric_limits<float>::lowest());
    for (int row = 0; row < features.request_count; ++row) {
        const float* input = features.request_features.data() +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(request_dim_);
        linear_into(
            input, request_dim_,
            tensor("request_fc1.weight"), tensor("request_fc1.bias"),
            request_hidden.data());
        silu_inplace(request_hidden.data(), 64);
        linear_into(
            request_hidden.data(), 64,
            tensor("request_fc2.weight"), tensor("request_fc2.bias"),
            request_embedding.data());
        layer_norm_into(
            request_embedding.data(), 64,
            tensor("request_norm.weight"), tensor("request_norm.bias"),
            layer_norm_eps_, request_embedding.data());
        silu_inplace(request_embedding.data(), 64);
        for (int i = 0; i < 64; ++i) {
            request_sum[static_cast<std::size_t>(i)] +=
                request_embedding[static_cast<std::size_t>(i)];
            request_max[static_cast<std::size_t>(i)] = std::max(
                request_max[static_cast<std::size_t>(i)],
                request_embedding[static_cast<std::size_t>(i)]);
        }
    }
    if (features.request_count == 0) request_max.fill(0.0f);

    launch_max.fill(std::numeric_limits<float>::lowest());
    for (int row = 0; row < features.launch_count; ++row) {
        const float* input = features.launch_features.data() +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(launch_dim_);
        linear_into(
            input, launch_dim_,
            tensor("launch_fc1.weight"), tensor("launch_fc1.bias"),
            launch_hidden.data());
        silu_inplace(launch_hidden.data(), 32);
        linear_into(
            launch_hidden.data(), 32,
            tensor("launch_fc2.weight"), tensor("launch_fc2.bias"),
            launch_embedding.data());
        layer_norm_into(
            launch_embedding.data(), 32,
            tensor("launch_norm.weight"), tensor("launch_norm.bias"),
            layer_norm_eps_, launch_embedding.data());
        silu_inplace(launch_embedding.data(), 32);
        for (int i = 0; i < 32; ++i) {
            launch_sum[static_cast<std::size_t>(i)] +=
                launch_embedding[static_cast<std::size_t>(i)];
            launch_max[static_cast<std::size_t>(i)] = std::max(
                launch_max[static_cast<std::size_t>(i)],
                launch_embedding[static_cast<std::size_t>(i)]);
        }
    }
    if (features.launch_count == 0) launch_max.fill(0.0f);

    std::copy(global_embedding.begin(), global_embedding.end(), fusion_input.begin());
    std::copy(request_sum.begin(), request_sum.end(), fusion_input.begin() + 64);
    std::copy(request_max.begin(), request_max.end(), fusion_input.begin() + 128);
    std::copy(launch_sum.begin(), launch_sum.end(), fusion_input.begin() + 192);
    std::copy(launch_max.begin(), launch_max.end(), fusion_input.begin() + 224);
    linear_into(
        fusion_input.data(), 256,
        tensor("fusion_fc.weight"), tensor("fusion_fc.bias"),
        hidden.data());
    silu_inplace(hidden.data(), 192);
    for (const char* block : {"block1", "block2"}) {
        const std::string prefix(block);
        layer_norm_into(
            hidden.data(), 192,
            tensor(prefix + ".norm.weight"), tensor(prefix + ".norm.bias"),
            layer_norm_eps_, normalized_buffer.data());
        linear_into(
            normalized_buffer.data(), 192,
            tensor(prefix + ".fc1.weight"), tensor(prefix + ".fc1.bias"),
            bottleneck.data());
        silu_inplace(bottleneck.data(), 64);
        linear_into(
            bottleneck.data(), 64,
            tensor(prefix + ".fc2.weight"), tensor(prefix + ".fc2.bias"),
            residual.data());
        for (int i = 0; i < 192; ++i) {
            hidden[static_cast<std::size_t>(i)] += residual[static_cast<std::size_t>(i)];
        }
    }
    layer_norm_into(
        hidden.data(), 192,
        tensor("head_norm.weight"), tensor("head_norm.bias"),
        layer_norm_eps_, normalized_buffer.data());
    linear_into(
        normalized_buffer.data(), 192,
        tensor("head1.weight"), tensor("head1.bias"), head.data());
    silu_inplace(head.data(), 32);
    linear_into(head.data(), 32, tensor("head2.weight"), tensor("head2.bias"), &scalar);
    const double normalized = std::tanh(static_cast<double>(scalar));
    return static_cast<double>(value_min_) +
        0.5 * (normalized + 1.0) *
        static_cast<double>(value_max_ - value_min_);
}

void NativeDenseDNNModel::markov_policy_state_embedding_into(
    const MarkovValueFeatures& features,
    float* output) const {
    if (static_cast<int>(features.global_features.size()) != global_dim_ ||
        features.request_count < 0 || features.launch_count < 0 ||
        static_cast<int>(features.request_features.size()) !=
            features.request_count * request_dim_ ||
        static_cast<int>(features.launch_features.size()) !=
            features.launch_count * launch_dim_) {
        throw std::runtime_error("native Markov policy feature dimensions mismatch");
    }
    std::array<float, 32> global_embedding{};
    std::array<float, 32> request_hidden{};
    std::array<float, 32> request_embedding{};
    std::array<float, 32> request_sum{};
    std::array<float, 32> request_max{};
    std::array<float, 16> launch_hidden{};
    std::array<float, 16> launch_embedding{};
    std::array<float, 16> launch_sum{};
    std::array<float, 16> launch_max{};
    std::array<float, 128> fusion_input{};

    linear_into(
        features.global_features.data(), global_dim_,
        tensor("global_fc.weight"), tensor("global_fc.bias"),
        global_embedding.data());
    layer_norm_into(
        global_embedding.data(), 32,
        tensor("global_norm.weight"), tensor("global_norm.bias"),
        layer_norm_eps_, global_embedding.data());
    silu_inplace(global_embedding.data(), 32);

    request_max.fill(std::numeric_limits<float>::lowest());
    for (int row = 0; row < features.request_count; ++row) {
        const float* input = features.request_features.data() +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(request_dim_);
        linear_into(
            input, request_dim_,
            tensor("request_fc1.weight"), tensor("request_fc1.bias"),
            request_hidden.data());
        silu_inplace(request_hidden.data(), 32);
        linear_into(
            request_hidden.data(), 32,
            tensor("request_fc2.weight"), tensor("request_fc2.bias"),
            request_embedding.data());
        layer_norm_into(
            request_embedding.data(), 32,
            tensor("request_norm.weight"), tensor("request_norm.bias"),
            layer_norm_eps_, request_embedding.data());
        silu_inplace(request_embedding.data(), 32);
        for (int i = 0; i < 32; ++i) {
            request_sum[static_cast<std::size_t>(i)] +=
                request_embedding[static_cast<std::size_t>(i)];
            request_max[static_cast<std::size_t>(i)] = std::max(
                request_max[static_cast<std::size_t>(i)],
                request_embedding[static_cast<std::size_t>(i)]);
        }
    }
    if (features.request_count == 0) request_max.fill(0.0f);

    launch_max.fill(std::numeric_limits<float>::lowest());
    for (int row = 0; row < features.launch_count; ++row) {
        const float* input = features.launch_features.data() +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(launch_dim_);
        linear_into(
            input, launch_dim_,
            tensor("launch_fc1.weight"), tensor("launch_fc1.bias"),
            launch_hidden.data());
        silu_inplace(launch_hidden.data(), 16);
        linear_into(
            launch_hidden.data(), 16,
            tensor("launch_fc2.weight"), tensor("launch_fc2.bias"),
            launch_embedding.data());
        layer_norm_into(
            launch_embedding.data(), 16,
            tensor("launch_norm.weight"), tensor("launch_norm.bias"),
            layer_norm_eps_, launch_embedding.data());
        silu_inplace(launch_embedding.data(), 16);
        for (int i = 0; i < 16; ++i) {
            launch_sum[static_cast<std::size_t>(i)] +=
                launch_embedding[static_cast<std::size_t>(i)];
            launch_max[static_cast<std::size_t>(i)] = std::max(
                launch_max[static_cast<std::size_t>(i)],
                launch_embedding[static_cast<std::size_t>(i)]);
        }
    }
    if (features.launch_count == 0) launch_max.fill(0.0f);

    std::copy(global_embedding.begin(), global_embedding.end(), fusion_input.begin());
    std::copy(request_sum.begin(), request_sum.end(), fusion_input.begin() + 32);
    std::copy(request_max.begin(), request_max.end(), fusion_input.begin() + 64);
    std::copy(launch_sum.begin(), launch_sum.end(), fusion_input.begin() + 96);
    std::copy(launch_max.begin(), launch_max.end(), fusion_input.begin() + 112);
    linear_into(
        fusion_input.data(), 128,
        tensor("state_fusion_fc.weight"), tensor("state_fusion_fc.bias"),
        output);
    silu_inplace(output, 192);
}

void NativeDenseDNNModel::policy_state_embedding_into(
    const float* state,
    float* output) const {
    linear_into(
        state, 226,
        tensor("state_fc.weight"), tensor("state_fc.bias"), output);
    layer_norm_into(
        output, 192,
        tensor("state_norm.weight"), tensor("state_norm.bias"),
        layer_norm_eps_, output);
    silu_inplace(output, 192);
}

double NativeDenseDNNModel::policy_from_embeddings(
    const float* state_embedding,
    const float* action) const {
    std::array<float, 128> fusion_state_base{};
    policy_fusion_state_base_into(state_embedding, fusion_state_base.data());
    return policy_from_fusion_state_base(fusion_state_base.data(), action);
}

void NativeDenseDNNModel::policy_fusion_state_base_into(
    const float* state_embedding,
    float* output) const {
    const Tensor& weight = tensor("fusion_fc.weight");
    const Tensor& bias = tensor("fusion_fc.bias");
    if (weight.shape.size() != 2 || weight.shape[0] != 128 || weight.shape[1] != 256 ||
        bias.shape.size() != 1 || bias.shape[0] != 128) {
        throw std::runtime_error("invalid policy fusion tensor dimensions");
    }
    for (int row = 0; row < 128; ++row) {
        const float* weights = weight.values.data() +
            static_cast<std::size_t>(row) * 256U;
        float total = bias.values[static_cast<std::size_t>(row)];
        for (int column = 0; column < 192; ++column) {
            total += weights[column] * state_embedding[column];
        }
        output[row] = total;
    }
}

NativeDenseDNNModel::PolicyActionTensorRefs
NativeDenseDNNModel::policy_action_tensor_refs() const {
    return {
        &tensor("action_fc.weight"),
        &tensor("action_fc.bias"),
        &tensor("action_norm.weight"),
        &tensor("action_norm.bias"),
        &tensor("fusion_fc.weight"),
        &tensor("fusion_norm.weight"),
        &tensor("fusion_norm.bias"),
        &tensor("fusion_block.norm.weight"),
        &tensor("fusion_block.norm.bias"),
        &tensor("fusion_block.fc1.weight"),
        &tensor("fusion_block.fc1.bias"),
        &tensor("fusion_block.fc2.weight"),
        &tensor("fusion_block.fc2.bias"),
        &tensor("head_norm.weight"),
        &tensor("head_norm.bias"),
        &tensor("head1.weight"),
        &tensor("head1.bias"),
        &tensor("head2.weight"),
        &tensor("head2.bias"),
    };
}

double NativeDenseDNNModel::policy_from_fusion_state_base(
    const float* fusion_state_base,
    const float* action) const {
    return policy_from_fusion_state_base(
        fusion_state_base,
        action,
        policy_action_tensor_refs());
}

double NativeDenseDNNModel::policy_from_fusion_state_base(
    const float* fusion_state_base,
    const float* action,
    const PolicyActionTensorRefs& refs) const {
    std::array<float, 64> action_embedding{};
    std::array<float, 128> hidden{};
    std::array<float, 128> normalized_buffer{};
    std::array<float, 128> residual{};
    std::array<float, 32> bottleneck{};
    std::array<float, 64> head{};
    float scalar = 0.0f;

    linear_into(
        action, action_dim_,
        *refs.action_fc_weight, *refs.action_fc_bias,
        action_embedding.data());
    layer_norm_into(
        action_embedding.data(), 64,
        *refs.action_norm_weight, *refs.action_norm_bias,
        layer_norm_eps_, action_embedding.data());
    silu_inplace(action_embedding.data(), 64);
    const Tensor& fusion_weight = *refs.fusion_weight;
    for (int row = 0; row < 128; ++row) {
        const float* weights = fusion_weight.values.data() +
            static_cast<std::size_t>(row) * 256U + 192U;
        float total = fusion_state_base[row];
        for (int column = 0; column < 64; ++column) {
            total += weights[column] * action_embedding[static_cast<std::size_t>(column)];
        }
        hidden[static_cast<std::size_t>(row)] = total;
    }
    layer_norm_into(
        hidden.data(), 128,
        *refs.fusion_norm_weight, *refs.fusion_norm_bias,
        layer_norm_eps_, hidden.data());
    silu_inplace(hidden.data(), 128);
    layer_norm_into(
        hidden.data(), 128,
        *refs.block_norm_weight, *refs.block_norm_bias,
        layer_norm_eps_, normalized_buffer.data());
    linear_into(
        normalized_buffer.data(), 128,
        *refs.block_fc1_weight, *refs.block_fc1_bias,
        bottleneck.data());
    silu_inplace(bottleneck.data(), 32);
    linear_into(
        bottleneck.data(), 32,
        *refs.block_fc2_weight, *refs.block_fc2_bias,
        residual.data());
    for (int i = 0; i < 128; ++i) hidden[static_cast<std::size_t>(i)] += residual[static_cast<std::size_t>(i)];
    layer_norm_into(
        hidden.data(), 128,
        *refs.head_norm_weight, *refs.head_norm_bias,
        layer_norm_eps_, normalized_buffer.data());
    linear_into(
        normalized_buffer.data(), 128,
        *refs.head1_weight, *refs.head1_bias, head.data());
    silu_inplace(head.data(), 64);
    linear_into(head.data(), 64, *refs.head2_weight, *refs.head2_bias, &scalar);
    return static_cast<double>(scalar);
}

double NativeDenseDNNModel::predict_value(const std::vector<float>& state_features) const {
    if (!loaded_ || !is_value() || is_markov_value()) {
        throw std::runtime_error("native legacy value DNN is not loaded");
    }
    if (static_cast<int>(state_features.size()) != state_dim_) {
        throw std::runtime_error("native value DNN feature dimension mismatch");
    }
    return value_from_state(state_features.data());
}

double NativeDenseDNNModel::predict_markov_value(
    const MarkovValueFeatures& features) const {
    if (!loaded_ || !is_markov_value()) {
        throw std::runtime_error("native Markov value DNN is not loaded");
    }
    return markov_value_from_features(features);
}

double NativeDenseDNNModel::predict_policy(
    const std::vector<float>& state_action_features) const {
    if (!loaded_ || !is_policy()) throw std::runtime_error("native policy DNN is not loaded");
    if (static_cast<int>(state_action_features.size()) != feature_dim_) {
        throw std::runtime_error("native policy DNN feature dimension mismatch");
    }
    std::array<float, 192> state_embedding{};
    policy_state_embedding_into(state_action_features.data(), state_embedding.data());
    return policy_from_embeddings(
        state_embedding.data(),
        state_action_features.data() + static_cast<std::size_t>(state_dim_));
}

std::vector<double> NativeDenseDNNModel::predict_policy_batch_flat(
    const std::vector<float>& flat_features,
    int num_rows,
    int row_dim) const {
    if (!loaded_ || !is_policy()) throw std::runtime_error("native policy DNN is not loaded");
    if (num_rows < 0 || row_dim != feature_dim_) {
        throw std::runtime_error("native policy DNN batch dimensions mismatch");
    }
    const std::size_t expected =
        static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(row_dim);
    if (flat_features.size() != expected) {
        throw std::runtime_error("native policy DNN flat batch size mismatch");
    }
    std::vector<double> output(static_cast<std::size_t>(num_rows), 0.0);
    if (num_rows == 0) return output;

    const float* first = flat_features.data();
    std::array<float, 192> shared_state{};
    std::array<float, 128> shared_fusion_state_base{};
    policy_state_embedding_into(first, shared_state.data());
    policy_fusion_state_base_into(shared_state.data(), shared_fusion_state_base.data());
    bool all_states_match = true;
    for (int row = 1; row < num_rows && all_states_match; ++row) {
        const float* current = first +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(row_dim);
        all_states_match = std::equal(first, first + state_dim_, current);
    }

    #pragma omp parallel for if(num_rows >= 8) num_threads(3) schedule(static)
    for (int row = 0; row < num_rows; ++row) {
        const float* current = first +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(row_dim);
        if (all_states_match) {
            output[static_cast<std::size_t>(row)] =
                policy_from_fusion_state_base(
                    shared_fusion_state_base.data(), current + state_dim_);
        } else {
            std::array<float, 192> state_embedding{};
            std::array<float, 128> fusion_state_base{};
            policy_state_embedding_into(current, state_embedding.data());
            policy_fusion_state_base_into(state_embedding.data(), fusion_state_base.data());
            output[static_cast<std::size_t>(row)] =
                policy_from_fusion_state_base(fusion_state_base.data(), current + state_dim_);
        }
    }
    return output;
}

std::vector<double> NativeDenseDNNModel::predict_policy_grouped_batch_flat(
    const std::vector<float>& flat_features,
    int num_rows,
    int row_dim,
    const std::vector<int>& group_offsets,
    int parallel_threads) const {
    if (!loaded_ || !is_policy()) throw std::runtime_error("native policy DNN is not loaded");
    if (num_rows < 0 || row_dim != feature_dim_) {
        throw std::runtime_error("native policy DNN grouped batch dimensions mismatch");
    }
    const std::size_t expected =
        static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(row_dim);
    if (flat_features.size() != expected) {
        throw std::runtime_error("native policy DNN grouped flat batch size mismatch");
    }
    if (group_offsets.empty() || group_offsets.front() != 0 ||
        group_offsets.back() != num_rows) {
        throw std::runtime_error("native policy DNN grouped offsets do not span the batch");
    }
    for (std::size_t group = 1; group < group_offsets.size(); ++group) {
        if (group_offsets[group] < group_offsets[group - 1]) {
            throw std::runtime_error("native policy DNN grouped offsets are not monotonic");
        }
    }

    std::vector<double> output(static_cast<std::size_t>(num_rows), 0.0);
    if (num_rows == 0) return output;

    const int group_count = static_cast<int>(group_offsets.size()) - 1;
    const int threads = std::max(1, parallel_threads);
    std::vector<std::array<float, 128>> fusion_state_bases(
        static_cast<std::size_t>(group_count));
    std::vector<int> row_groups(static_cast<std::size_t>(num_rows), 0);

    for (int group = 0; group < group_count; ++group) {
        const int begin = group_offsets[static_cast<std::size_t>(group)];
        const int end = group_offsets[static_cast<std::size_t>(group + 1)];
        if (begin >= end) continue;
        const float* row = flat_features.data() +
            static_cast<std::size_t>(begin) * static_cast<std::size_t>(row_dim);
        std::array<float, 192> state_embedding{};
        policy_state_embedding_into(row, state_embedding.data());
        policy_fusion_state_base_into(
            state_embedding.data(),
            fusion_state_bases[static_cast<std::size_t>(group)].data());
        for (int row_index = begin; row_index < end; ++row_index) {
            row_groups[static_cast<std::size_t>(row_index)] = group;
        }
    }
    const PolicyActionTensorRefs refs = policy_action_tensor_refs();

    #pragma omp parallel for if(num_rows >= 8) num_threads(threads) schedule(static)
    for (int row_index = 0; row_index < num_rows; ++row_index) {
        const float* row = flat_features.data() +
            static_cast<std::size_t>(row_index) * static_cast<std::size_t>(row_dim);
        const int group = row_groups[static_cast<std::size_t>(row_index)];
        output[static_cast<std::size_t>(row_index)] =
            policy_from_fusion_state_base(
                fusion_state_bases[static_cast<std::size_t>(group)].data(),
                row + state_dim_,
                refs);
    }
    return output;
}

std::vector<double> NativeDenseDNNModel::predict_policy_grouped_split_batch_flat(
    const std::vector<float>& flat_states,
    const std::vector<float>& flat_actions,
    int num_rows,
    const std::vector<int>& group_offsets,
    int parallel_threads) const {
    if (!loaded_ || !is_policy() || is_markov_policy()) {
        throw std::runtime_error("native split policy DNN is not loaded");
    }
    if (num_rows < 0 || group_offsets.empty() || group_offsets.front() != 0 ||
        group_offsets.back() != num_rows) {
        throw std::runtime_error("native split policy DNN batch dimensions mismatch");
    }
    for (std::size_t group = 1; group < group_offsets.size(); ++group) {
        if (group_offsets[group] < group_offsets[group - 1]) {
            throw std::runtime_error("native split policy DNN offsets are not monotonic");
        }
    }
    const int group_count = static_cast<int>(group_offsets.size()) - 1;
    if (flat_states.size() !=
        static_cast<std::size_t>(group_count) * static_cast<std::size_t>(state_dim_) ||
        flat_actions.size() !=
        static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(action_dim_)) {
        throw std::runtime_error("native split policy DNN flat batch size mismatch");
    }

    std::vector<double> output(static_cast<std::size_t>(num_rows), 0.0);
    if (num_rows == 0) return output;

#if defined(__aarch64__)
    {
    // OpenBLAS' small, repeated SGEMMs regress on Neoverse-V2. The row kernel
    // is faster there and remains deterministic across rollout thread counts.
    const int threads = std::max(1, parallel_threads);
    std::vector<std::array<float, 128>> fusion_state_bases(
        static_cast<std::size_t>(group_count));
    std::vector<int> row_groups(static_cast<std::size_t>(num_rows), 0);
    for (int group = 0; group < group_count; ++group) {
        const float* state = flat_states.data() +
            static_cast<std::size_t>(group) * static_cast<std::size_t>(state_dim_);
        std::array<float, 192> state_embedding{};
        policy_state_embedding_into(state, state_embedding.data());
        policy_fusion_state_base_into(
            state_embedding.data(),
            fusion_state_bases[static_cast<std::size_t>(group)].data());
        for (int row_index = group_offsets[static_cast<std::size_t>(group)];
             row_index < group_offsets[static_cast<std::size_t>(group + 1)];
             ++row_index) {
            row_groups[static_cast<std::size_t>(row_index)] = group;
        }
    }
    const PolicyActionTensorRefs refs = policy_action_tensor_refs();
    #pragma omp parallel for if(num_rows >= 8) num_threads(threads) schedule(static)
    for (int row_index = 0; row_index < num_rows; ++row_index) {
        const int group = row_groups[static_cast<std::size_t>(row_index)];
        output[static_cast<std::size_t>(row_index)] =
            policy_from_fusion_state_base(
                fusion_state_bases[static_cast<std::size_t>(group)].data(),
                flat_actions.data() +
                    static_cast<std::size_t>(row_index) *
                    static_cast<std::size_t>(action_dim_),
                refs);
    }
    return output;
    }
#endif

#ifdef GV2_HAVE_OPENBLAS_ILP64
    const int requested_blas_threads = std::max(1, parallel_threads);
    thread_local int configured_blas_threads = -1;
    if (configured_blas_threads != requested_blas_threads) {
        openblas_set_num_threads64_(requested_blas_threads);
        configured_blas_threads = requested_blas_threads;
    }
#else
    (void)parallel_threads;
#endif

    thread_local PolicySplitBatchWorkspace workspace;
    auto& fusion_state_bases = workspace.fusion_state_bases;
    auto& state_embeddings = workspace.state_embeddings;
    auto& row_groups = workspace.row_groups;
    fusion_state_bases.resize(static_cast<std::size_t>(group_count));
    state_embeddings.resize(static_cast<std::size_t>(group_count) * 192U);
    row_groups.resize(static_cast<std::size_t>(num_rows));

    linear_batch_into(
        flat_states.data(), group_count, state_dim_,
        *policy_batch_refs_.state_fc_weight,
        *policy_batch_refs_.state_fc_bias,
        state_embeddings.data());
    for (int group = 0; group < group_count; ++group) {
        float* state_embedding = state_embeddings.data() +
            static_cast<std::size_t>(group) * 192U;
        layer_norm_into(
            state_embedding, 192,
            *policy_batch_refs_.state_norm_weight,
            *policy_batch_refs_.state_norm_bias,
            layer_norm_eps_, state_embedding);
        silu_inplace(state_embedding, 192);
    }
    linear_batch_into(
        state_embeddings.data(), group_count, 192,
        policy_fusion_state_weight_, *policy_batch_refs_.fusion_bias,
        fusion_state_bases.front().data());

    for (int group = 0; group < group_count; ++group) {
        for (int row_index = group_offsets[static_cast<std::size_t>(group)];
             row_index < group_offsets[static_cast<std::size_t>(group + 1)];
             ++row_index) {
            row_groups[static_cast<std::size_t>(row_index)] = group;
        }
    }

    const auto& refs = policy_batch_refs_;
    const auto& action_weight = *refs.action_fc_weight;
    const auto& action_bias = *refs.action_fc_bias;
    const auto& action_norm_weight = *refs.action_norm_weight;
    const auto& action_norm_bias = *refs.action_norm_bias;
    const auto& fusion_norm_weight = *refs.fusion_norm_weight;
    const auto& fusion_norm_bias = *refs.fusion_norm_bias;
    const auto& block_norm_weight = *refs.block_norm_weight;
    const auto& block_norm_bias = *refs.block_norm_bias;
    const auto& block_fc1_weight = *refs.block_fc1_weight;
    const auto& block_fc1_bias = *refs.block_fc1_bias;
    const auto& block_fc2_weight = *refs.block_fc2_weight;
    const auto& block_fc2_bias = *refs.block_fc2_bias;
    const auto& head_norm_weight = *refs.head_norm_weight;
    const auto& head_norm_bias = *refs.head_norm_bias;
    const auto& head1_weight = *refs.head1_weight;
    const auto& head1_bias = *refs.head1_bias;
    const auto& head2_weight = *refs.head2_weight;
    const auto& head2_bias = *refs.head2_bias;

    auto& action_cache = workspace.action_fusion_cache[this];
    if (action_cache.size() > 65536U) action_cache.clear();
    auto& row_action_indices = workspace.row_action_indices;
    auto& unique_action_keys = workspace.unique_action_keys;
    auto& unique_actions = workspace.unique_actions;
    auto& action_embedding = workspace.action_embedding;
    auto& action_fusion = workspace.action_fusion;
    auto& row_action_bases = workspace.row_action_bases;
    auto& unique_action_bases = workspace.unique_action_bases;
    auto& hidden = workspace.hidden;
    auto& normalized = workspace.normalized;
    auto& bottleneck = workspace.bottleneck;
    auto& residual = workspace.residual;
    auto& head = workspace.head;
    auto& scalar = workspace.scalar;
    row_action_indices.assign(static_cast<std::size_t>(num_rows), -1);
    row_action_bases.assign(static_cast<std::size_t>(num_rows), nullptr);
    unique_action_keys.clear();
    unique_actions.clear();
    hidden.resize(static_cast<std::size_t>(num_rows) * 128U);
    normalized.resize(static_cast<std::size_t>(num_rows) * 128U);
    bottleneck.resize(static_cast<std::size_t>(num_rows) * 32U);
    residual.resize(static_cast<std::size_t>(num_rows) * 128U);
    head.resize(static_cast<std::size_t>(num_rows) * 64U);
    scalar.resize(static_cast<std::size_t>(num_rows));

    for (int row = 0; row < num_rows; ++row) {
        const float* action = flat_actions.data() +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(action_dim_);
        PolicyActionFeatureKey key =
            policy_action_feature_key(action, action_dim_);
        const auto cached = action_cache.find(key);
        if (cached != action_cache.end()) {
            row_action_bases[static_cast<std::size_t>(row)] =
                &cached->second;
            continue;
        }
        int action_index = -1;
        for (int index = 0;
             index < static_cast<int>(unique_action_keys.size());
             ++index) {
            if (unique_action_keys[static_cast<std::size_t>(index)] == key) {
                action_index = index;
                break;
            }
        }
        if (action_index < 0) {
            action_index = static_cast<int>(unique_action_keys.size());
            unique_action_keys.push_back(std::move(key));
            unique_actions.insert(
                unique_actions.end(), action, action + action_dim_);
        }
        row_action_indices[static_cast<std::size_t>(row)] = action_index;
    }

    const int unique_action_count =
        static_cast<int>(unique_action_keys.size());
    action_embedding.resize(
        static_cast<std::size_t>(unique_action_count) * 64U);
    action_fusion.resize(
        static_cast<std::size_t>(unique_action_count) * 128U);
    unique_action_bases.assign(
        static_cast<std::size_t>(unique_action_count), nullptr);
    if (unique_action_count > 0) {
        linear_batch_into(
            unique_actions.data(), unique_action_count, action_dim_,
            action_weight, action_bias, action_embedding.data());
        for (int row = 0; row < unique_action_count; ++row) {
            float* values = action_embedding.data() +
                static_cast<std::size_t>(row) * 64U;
            layer_norm_into(
                values, 64, action_norm_weight, action_norm_bias,
                layer_norm_eps_, values);
            silu_inplace(values, 64);
        }
        linear_batch_into(
            action_embedding.data(), unique_action_count, 64,
            policy_fusion_action_weight_, policy_fusion_zero_bias_,
            action_fusion.data());
        for (int action_index = 0;
             action_index < unique_action_count;
             ++action_index) {
            std::array<float, 128> cached_base{};
            const float* base = action_fusion.data() +
                static_cast<std::size_t>(action_index) * 128U;
            std::copy(base, base + 128, cached_base.begin());
            const auto inserted = action_cache.emplace(
                unique_action_keys[static_cast<std::size_t>(action_index)],
                cached_base);
            unique_action_bases[static_cast<std::size_t>(action_index)] =
                &inserted.first->second;
        }
        for (int row = 0; row < num_rows; ++row) {
            const int action_index =
                row_action_indices[static_cast<std::size_t>(row)];
            if (action_index < 0) continue;
            row_action_bases[static_cast<std::size_t>(row)] =
                unique_action_bases[static_cast<std::size_t>(action_index)];
        }
    }
    for (int row = 0; row < num_rows; ++row) {
        float* values = hidden.data() + static_cast<std::size_t>(row) * 128U;
        const auto* action_base =
            row_action_bases[static_cast<std::size_t>(row)];
        if (action_base == nullptr) {
            throw std::runtime_error("native policy action fusion cache miss");
        }
        const auto& state_base =
            fusion_state_bases[static_cast<std::size_t>(
                row_groups[static_cast<std::size_t>(row)])];
        double mean = 0.0;
        for (int column = 0; column < 128; ++column) {
            const float combined =
                (*action_base)[static_cast<std::size_t>(column)] +
                state_base[static_cast<std::size_t>(column)];
            values[column] = combined;
            mean += static_cast<double>(combined);
        }
        mean /= 128.0;
        double variance = 0.0;
        for (int column = 0; column < 128; ++column) {
            const double centered =
                static_cast<double>(values[column]) - mean;
            variance += centered * centered;
        }
        variance /= 128.0;
        const double inverse_std = 1.0 / std::sqrt(
            variance + static_cast<double>(layer_norm_eps_));
        for (int column = 0; column < 128; ++column) {
            const double normalized_value =
                (static_cast<double>(values[column]) - mean) * inverse_std;
            values[column] = static_cast<float>(
                normalized_value * static_cast<double>(
                    fusion_norm_weight.values[
                        static_cast<std::size_t>(column)]) +
                static_cast<double>(
                    fusion_norm_bias.values[
                        static_cast<std::size_t>(column)]));
        }
        silu_inplace(values, 128);
        layer_norm_into(
            values, 128, block_norm_weight, block_norm_bias,
            layer_norm_eps_,
            normalized.data() + static_cast<std::size_t>(row) * 128U);
    }

    linear_batch_into(
        normalized.data(), num_rows, 128,
        block_fc1_weight, block_fc1_bias, bottleneck.data());
    for (int row = 0; row < num_rows; ++row) {
        silu_inplace(
            bottleneck.data() + static_cast<std::size_t>(row) * 32U, 32);
    }
    linear_batch_into(
        bottleneck.data(), num_rows, 32,
        block_fc2_weight, block_fc2_bias, residual.data());
    for (int row = 0; row < num_rows; ++row) {
        float* values = hidden.data() + static_cast<std::size_t>(row) * 128U;
        const float* update =
            residual.data() + static_cast<std::size_t>(row) * 128U;
        for (int column = 0; column < 128; ++column) {
            values[column] += update[column];
        }
        layer_norm_into(
            values, 128, head_norm_weight, head_norm_bias,
            layer_norm_eps_,
            normalized.data() + static_cast<std::size_t>(row) * 128U);
    }

    linear_batch_into(
        normalized.data(), num_rows, 128,
        head1_weight, head1_bias, head.data());
    for (int row = 0; row < num_rows; ++row) {
        silu_inplace(head.data() + static_cast<std::size_t>(row) * 64U, 64);
    }
    linear_batch_into(
        head.data(), num_rows, 64,
        head2_weight, head2_bias, scalar.data());
    for (int row = 0; row < num_rows; ++row) {
        output[static_cast<std::size_t>(row)] =
            static_cast<double>(scalar[static_cast<std::size_t>(row)]);
    }
    return output;
}

std::vector<double> NativeDenseDNNModel::predict_markov_policy_grouped_batch(
    const std::vector<MarkovValueFeatures>& states,
    const std::vector<float>& flat_actions,
    int num_rows,
    const std::vector<int>& group_offsets,
    int parallel_threads) const {
    if (!loaded_ || !is_markov_policy()) {
        throw std::runtime_error("native Markov policy DNN is not loaded");
    }
    if (num_rows < 0 || group_offsets.empty() || group_offsets.front() != 0 ||
        group_offsets.back() != num_rows ||
        states.size() + 1U != group_offsets.size() ||
        flat_actions.size() !=
            static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(action_dim_)) {
        throw std::runtime_error("native Markov policy grouped dimensions mismatch");
    }
    for (std::size_t group = 1; group < group_offsets.size(); ++group) {
        if (group_offsets[group] < group_offsets[group - 1]) {
            throw std::runtime_error("native Markov policy offsets are not monotonic");
        }
    }
    std::vector<double> output(static_cast<std::size_t>(num_rows), 0.0);
    if (num_rows == 0) return output;

    const int group_count = static_cast<int>(states.size());
    const int threads = std::max(1, parallel_threads);
    std::vector<std::array<float, 128>> fusion_state_bases(
        static_cast<std::size_t>(group_count));
    std::vector<int> row_groups(static_cast<std::size_t>(num_rows), 0);
    #pragma omp parallel for if(group_count > 1) \
        num_threads(threads) schedule(static)
    for (int group = 0; group < group_count; ++group) {
        std::array<float, 192> state_embedding{};
        markov_policy_state_embedding_into(
            states[static_cast<std::size_t>(group)], state_embedding.data());
        policy_fusion_state_base_into(
            state_embedding.data(),
            fusion_state_bases[static_cast<std::size_t>(group)].data());
        for (int row = group_offsets[static_cast<std::size_t>(group)];
             row < group_offsets[static_cast<std::size_t>(group + 1)];
             ++row) {
            row_groups[static_cast<std::size_t>(row)] = group;
        }
    }
    const PolicyActionTensorRefs refs = policy_action_tensor_refs();
    #pragma omp parallel for if(num_rows >= 8) num_threads(threads) schedule(static)
    for (int row = 0; row < num_rows; ++row) {
        const int group = row_groups[static_cast<std::size_t>(row)];
        output[static_cast<std::size_t>(row)] =
            policy_from_fusion_state_base(
                fusion_state_bases[static_cast<std::size_t>(group)].data(),
                flat_actions.data() +
                    static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(action_dim_),
                refs);
    }
    return output;
}

bool NativeDenseDNNModel::loaded() const { return loaded_; }
bool NativeDenseDNNModel::is_value() const { return model_kind_ == "value_dnn"; }
bool NativeDenseDNNModel::is_policy() const { return model_kind_ == "policy_dnn"; }
bool NativeDenseDNNModel::is_markov_value() const {
    return is_value() && architecture_ == "agz_markov_value_deepset_v2";
}
bool NativeDenseDNNModel::is_markov_policy() const {
    return is_policy() && architecture_ == "agz_markov_policy_deepset_v3";
}
int NativeDenseDNNModel::feature_dim() const { return feature_dim_; }
int NativeDenseDNNModel::state_dim() const { return state_dim_; }
int NativeDenseDNNModel::action_dim() const { return action_dim_; }
const std::string& NativeDenseDNNModel::model_tag() const { return model_tag_; }
const std::string& NativeDenseDNNModel::model_kind() const { return model_kind_; }
const std::string& NativeDenseDNNModel::feature_schema() const { return feature_schema_; }
int NativeDenseDNNModel::global_dim() const { return global_dim_; }
int NativeDenseDNNModel::request_dim() const { return request_dim_; }
int NativeDenseDNNModel::launch_dim() const { return launch_dim_; }

}  // namespace mcts_native_gv2
