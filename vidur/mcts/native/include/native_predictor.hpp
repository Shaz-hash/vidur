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

class NativePredictor {
public:
    bool load_csv(const std::string& path);
    bool is_loaded() const;

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
    static int clamp_nonneg(int x);
    static int round_up(int x, int g);
    static double nearest_prefill_estimate(
        int tokens,
        const std::vector<int>& profile_tokens,
        const std::vector<double>& profile_times
    );

    std::unordered_map<long long, NativePredictorValue> table_;
    int kv_granularity_ = 64;
    int prefill_chunk_granularity_ = 32;
};

} // namespace mcts_native
