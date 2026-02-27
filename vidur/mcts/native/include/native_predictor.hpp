#pragma once

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "native_types.hpp"

namespace mcts_native {

struct NativePredictorKey {
    int total_tokens_rounded = 0;
    int batch_size = 0;
    int prefill_batch_size = 0;
    int decode_batch_size = 0;
    int decode_avg_kv_cache_size = 0;
    int prefill_agg_kv_cache_size = 0;
    int prefill_agg_chunk_size = 0;
};

struct NativePredictorValue {
    double total_time = 0.0;
    double model_time = 0.0;
};

struct NativePredictorTable {
    std::string kind;
    int max_tokens = 0;
    int max_batch_size = 0;
    int kv_gran = 1;
    int prefill_gran = 1;
    std::vector<int> shape;
    std::vector<double> values;
};

struct NativePredictorRuntimeConfig {
    int num_layers_per_pipeline_stage = 1;
    int tensor_parallel_size = 1;
    int num_pipeline_stages = 1;
    bool post_attn_norm = true;
    bool skip_cpu_overhead_modeling = false;
    double attention_prefill_batching_overhead_fraction = 0.0;
    double attention_decode_batching_overhead_fraction = 0.0;
    double nccl_cpu_launch_overhead_ms = 0.0;
    double nccl_cpu_skew_overhead_per_device_ms = 0.0;
};

class NativePredictor {
public:
    bool load_csv(const std::string& path);
    bool is_loaded() const;
    bool has_component_tables() const;

    void clear_component_tables();
    void set_runtime_config(const NativePredictorRuntimeConfig& cfg);
    void set_component_table(
        const std::string& name,
        const std::string& kind,
        int max_tokens,
        int max_batch_size,
        int kv_gran,
        int prefill_gran,
        std::vector<int> shape,
        std::vector<double> values
    );

    std::tuple<double, double> lookup_batch_time(
        const std::vector<ControllerRequestStateNative>& reqs,
        const std::vector<int>& token_alloc,
        const std::vector<int>& profile_tokens,
        const std::vector<double>& profile_times,
        double fallback_total = 0.001,
        double fallback_model = 0.0007
    );

    static NativePredictorKey build_key(
        const std::vector<ControllerRequestStateNative>& reqs,
        const std::vector<int>& token_alloc,
        int kv_granularity = 64,
        int prefill_chunk_granularity = 32
    );

private:
    static long long pack_key(const NativePredictorKey& k);
    static long long pack_component_key(int a, int b = 0);
    static int clamp_nonneg(int x);
    static int round_up(int x, int g);
    static double nearest_prefill_estimate(
        int tokens,
        const std::vector<int>& profile_tokens,
        const std::vector<double>& profile_times
    );
    static std::tuple<double, double> default_fallback_time(
        const std::vector<ControllerRequestStateNative>& reqs,
        const std::vector<int>& token_alloc,
        const std::vector<int>& profile_tokens,
        const std::vector<double>& profile_times,
        double fallback_total,
        double fallback_model
    );

    std::unordered_map<long long, NativePredictorValue> table_;
    std::unordered_map<std::string, NativePredictorTable> component_tables_;
    int kv_granularity_ = 64;
    int prefill_chunk_granularity_ = 32;
    NativePredictorRuntimeConfig runtime_cfg_;
};

} // namespace mcts_native
