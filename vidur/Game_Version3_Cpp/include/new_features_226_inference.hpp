#pragma once

#include "gv2_types.hpp"
#include "virtual_simulator.hpp"

#include <string>
#include <vector>

namespace mcts_native_gv2 {

struct NewFeatures226Result {
    std::vector<float> features;
    double raw_value = 0.0;
    double value = 0.0;
    double decode_time_at_max = 0.0;
};

class NewFeatures226HGBRuntime {
public:
    bool load_model_export(const std::string& path);

    std::vector<float> build_features(
        const SimState& state,
        const VirtualSimulatorGV2* simulator = nullptr,
        int root_id = -1,
        double* decode_time_at_max_out = nullptr) const;

    double predict_raw(const std::vector<float>& features) const;
    double infer_value(
        const SimState& state,
        const VirtualSimulatorGV2* simulator = nullptr,
        int root_id = -1) const;
    NewFeatures226Result infer_debug(
        const SimState& state,
        const VirtualSimulatorGV2* simulator = nullptr,
        int root_id = -1) const;

    int feature_dim() const;
    int num_trees() const;
    bool loaded() const;
    const std::string& model_tag() const;

private:
    struct Node {
        double value = 0.0;
        int feature_idx = -1;
        double threshold = 0.0;
        bool missing_go_to_left = false;
        int left = -1;
        int right = -1;
        bool is_leaf = false;
    };
    struct Tree {
        std::vector<Node> nodes;
    };

    static double tree_value(const Tree& tree, const std::vector<float>& features);

    std::string model_tag_;
    int feature_dim_ = 226;
    double baseline_ = 0.0;
    std::vector<Tree> trees_;
};

// Generic native HGB text-export runtime. This is intentionally feature-builder
// agnostic and is used for state-action policy/prior HGB models whose feature
// dimensions differ from the 226D value model.
class NativeHGBModelRuntime {
public:
    bool load_model_export(const std::string& path);

    double predict_raw(const std::vector<float>& features) const;
    std::vector<double> predict_raw_batch_flat(
        const std::vector<float>& flat_features,
        int num_rows,
        int row_dim) const;

    int feature_dim() const;
    int num_trees() const;
    bool loaded() const;
    const std::string& model_tag() const;

private:
    struct Node {
        double value = 0.0;
        int feature_idx = -1;
        double threshold = 0.0;
        bool missing_go_to_left = false;
        int left = -1;
        int right = -1;
        bool is_leaf = false;
    };
    struct Tree {
        std::vector<Node> nodes;
    };

    static double tree_value(const Tree& tree, const float* features, int feature_dim);

    std::string model_tag_;
    int feature_dim_ = 0;
    double baseline_ = 0.0;
    std::vector<Tree> trees_;
};

}  // namespace mcts_native_gv2
