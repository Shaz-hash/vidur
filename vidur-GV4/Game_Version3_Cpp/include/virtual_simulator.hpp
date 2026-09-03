#pragma once

#include "gv2_types.hpp"

#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace mcts_native_gv2 {

struct ControllerPredictorRequestState {
    int request_id = -1;
    bool prefill_done = false;
    int remaining_prefill = 0;
    int remaining_decode = 0;
    double arrived_at = 0.0;
    double prefill_slo = 0.0;
    int num_processed_tokens = 0;
};

struct PredictorKey {
    int total_tokens_rounded = 0;
    int batch_size = 0;
    int prefill_batch_size = 0;
    int decode_batch_size = 0;
    int decode_avg_kv_cache_size = 0;
    int prefill_agg_kv_cache_size = 0;
    int prefill_agg_chunk_size = 0;
};

struct PredictorValue {
    double total_time = 0.0;
    double model_time = 0.0;
};

struct PredictorTable {
    std::string kind;
    int max_tokens = 0;
    int max_batch_size = 0;
    int kv_gran = 1;
    int prefill_gran = 1;
    std::vector<int> shape;
    std::vector<double> values;
};

struct PredictorRuntimeConfig {
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

class NativeBatchTimePredictorGV2 {
public:
    bool load_csv(const std::string& path);
    bool is_loaded() const;
    bool has_component_tables() const;

    void clear_component_tables();
    void set_runtime_config(const PredictorRuntimeConfig& cfg);
    void set_component_table(
        const std::string& name,
        const std::string& kind,
        int max_tokens,
        int max_batch_size,
        int kv_gran,
        int prefill_gran,
        std::vector<int> shape,
        std::vector<double> values);

    std::tuple<double, double> lookup_batch_time(
        const InlineVector<ControllerPredictorRequestState, 32>& reqs,
        const InlineVector<int, 32>& token_alloc,
        const std::vector<int>& profile_tokens,
        const std::vector<double>& profile_times,
        double fallback_total = 0.001,
        double fallback_model = 0.0007) const;

    static PredictorKey build_key(
        const InlineVector<ControllerPredictorRequestState, 32>& reqs,
        const InlineVector<int, 32>& token_alloc,
        int kv_granularity = 64,
        int prefill_chunk_granularity = 32);

private:
    struct ComponentTableCache {
        bool initialized = false;
        const PredictorTable* attn_pre_proj = nullptr;
        const PredictorTable* attn_post_proj = nullptr;
        const PredictorTable* mlp_up_proj = nullptr;
        const PredictorTable* mlp_down_proj = nullptr;
        const PredictorTable* mlp_act = nullptr;
        const PredictorTable* input_layernorm = nullptr;
        const PredictorTable* add = nullptr;
        const PredictorTable* attn_rope = nullptr;
        const PredictorTable* attn_kv_cache_save = nullptr;
        const PredictorTable* attn_decode = nullptr;
        const PredictorTable* attn_prefill = nullptr;
        const PredictorTable* post_attention_layernorm = nullptr;
        const PredictorTable* schedule = nullptr;
        const PredictorTable* sampler_e2e = nullptr;
        const PredictorTable* prepare_inputs_e2e = nullptr;
        const PredictorTable* process_model_outputs = nullptr;
        const PredictorTable* ray_comm_time = nullptr;
        const PredictorTable* all_reduce = nullptr;
        const PredictorTable* send_recv = nullptr;
    };

    static long long pack_key(const PredictorKey& k);
    static int round_up(int x, int g);
    static double nearest_prefill_estimate(
        int tokens,
        const std::vector<int>& profile_tokens,
        const std::vector<double>& profile_times);
    static std::tuple<double, double> default_fallback_time(
        const InlineVector<ControllerPredictorRequestState, 32>& reqs,
        const InlineVector<int, 32>& token_alloc,
        const std::vector<int>& profile_tokens,
        const std::vector<double>& profile_times,
        double fallback_total,
        double fallback_model);
    void invalidate_component_table_cache();
    const ComponentTableCache& component_table_cache() const;

    std::unordered_map<long long, PredictorValue> table_;
    std::unordered_map<std::string, PredictorTable> component_tables_;
    mutable ComponentTableCache component_table_cache_;
    int kv_granularity_ = 64;
    int prefill_chunk_granularity_ = 32;
    PredictorRuntimeConfig runtime_cfg_;
};

struct VirtualSimulatorConfig {
    double adversary_tick_sec = 0.2;
    std::vector<int> prefill_profile_tokens;
    std::vector<double> prefill_profile_times;
    double fallback_total_time_sec = 0.001;
    double fallback_model_time_sec = 0.0007;
};

struct ControllerBatchPlan {
    bool strict_noop = false;
    InlineVector<int, 32> request_ids;
    InlineVector<int, 32> prefill_tokens;
    InlineVector<int, 32> decode_tokens;
    InlineVector<int, 32> num_tokens;
    InlineVector<ControllerPredictorRequestState, 32> predictor_reqs;
};

struct BatchExecutionResult {
    double stage_total_time_sec = 0.0;
    double stage_model_time_sec = 0.0;
    double start_time = 0.0;
    double end_time = 0.0;
    bool executed = false;
};

class VirtualSimulatorGV2 {
public:
    explicit VirtualSimulatorGV2(VirtualSimulatorConfig cfg = {});

    const VirtualSimulatorConfig& cfg() const;
    void set_config(VirtualSimulatorConfig cfg);

    void set_prefill_profile(std::vector<int> tokens, std::vector<double> times);
    double prefill_profile_lookup(int tokens) const;
    bool load_predictor_csv(const std::string& path);

    NativeBatchTimePredictorGV2& predictor();
    const NativeBatchTimePredictorGV2& predictor() const;

    double next_adv_tick(double sim_time) const;

    ControllerBatchPlan build_controller_batch_plan(
        const SimState& state,
        const ControllerAction& action,
        bool enforce_nonnegative_decode_credits,
        int decode_credit_available) const;

    BatchExecutionResult execute_controller_batch_timing(
        SimState& state,
        const ControllerBatchPlan& plan) const;

    bool maybe_fast_forward_decode_only_to_next_adv_tick(SimState& state) const;

    SimState fork_state(const SimState& in_state) const;
    std::string snapshot_state(const SimState& in_state) const;
    SimState restore_state(const std::string& payload) const;

private:
    static bool is_pending(const RequestState& r);
    static RequestState* find_request(SimState& state, int request_id);

    VirtualSimulatorConfig cfg_;
    NativeBatchTimePredictorGV2 predictor_;
};

}  // namespace mcts_native_gv2
