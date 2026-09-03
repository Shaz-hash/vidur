#pragma once

#include "gv4/actions.hpp"

#include <string>
#include <vector>

namespace gv4 {

struct FeatureLayout {
    std::string schema_version;
    int pipeline_stage_count = 0;
    std::vector<std::string> global_names;
    std::vector<std::string> request_names;
    std::vector<std::string> launch_names;
    std::vector<std::string> replica_names;
    std::vector<std::string> microbatch_names;
    std::vector<std::string> controller_header_names;
    std::vector<std::string> controller_request_names;
    std::vector<std::string> adversary_header_names;
    std::vector<std::string> adversary_request_names;
};

struct FeatureScales {
    double window_request_cap = 0.0;
    double window_prefill_cap = 0.0;
    double active_request_scale = 0.0;
    double system_prefill_scale = 0.0;
    double system_decode_scale = 0.0;
    double request_prefill_scale = 0.0;
    double request_decode_scale = 0.0;
    double decode_credit_scale = 0.0;
    double adversary_time_scale = 0.0;
    double launch_age_scale = 0.0;
    double lateness_scale = 0.0;
    double block_token_scale = 0.0;
    double controller_prefill_action_scale = 0.0;
    double controller_kv_block_scale = 0.0;
    double system_logical_blocks = 0.0;
    double system_logical_tokens = 0.0;
};

struct FeatureMatrix {
    std::vector<float> values;
    int rows = 0;
    int columns = 0;
};

struct StateFeatures {
    std::string schema_version;
    std::string config_manifest_sha256;
    std::vector<float> global_features;
    FeatureMatrix request_rows;
    std::vector<int> request_replica_offsets;
    FeatureMatrix launch_rows;
    FeatureMatrix replica_rows;
    FeatureMatrix microbatch_rows;
    std::vector<int> microbatch_replica_offsets;
};

struct ControllerActionFeatures {
    std::vector<float> header;
    FeatureMatrix affected_request_rows;
};

struct AdversaryActionFeatures {
    std::vector<float> header;
    FeatureMatrix affected_request_rows;
};

class FeatureBuilder {
public:
    explicit FeatureBuilder(Config config);

    [[nodiscard]] const Config& config() const { return config_; }
    [[nodiscard]] const FeatureLayout& layout() const { return layout_; }
    [[nodiscard]] const FeatureScales& scales() const { return scales_; }
    [[nodiscard]] StateFeatures build_state(const State& state) const;
    [[nodiscard]] ControllerActionFeatures build_controller_action(
        const State& state,
        const CanonicalControllerAction& edge) const;
    [[nodiscard]] AdversaryActionFeatures build_adversary_action(
        const State& state,
        const CanonicalAdversaryAction& edge) const;

private:
    Config config_;
    FeatureLayout layout_;
    FeatureScales scales_;
};

}  // namespace gv4
