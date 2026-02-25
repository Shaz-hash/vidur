#include "native_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>

namespace mcts_native {

namespace {

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(line.size());
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (c == ',' && !in_quotes) {
            out.push_back(cur);
            cur.clear();
            continue;
        }
        cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

static int to_int_or(const std::string& s, int dflt) {
    try {
        return std::stoi(s);
    } catch (...) {
        return dflt;
    }
}

static double to_double_or(const std::string& s, double dflt) {
    try {
        return std::stod(s);
    } catch (...) {
        return dflt;
    }
}

} // namespace

int NativePredictor::clamp_nonneg(int x) {
    return (x < 0) ? 0 : x;
}

int NativePredictor::round_up(int x, int g) {
    const int gg = std::max(1, g);
    return ((x + gg - 1) / gg) * gg;
}

double NativePredictor::nearest_prefill_estimate(
    int tokens,
    const std::vector<int>& profile_tokens,
    const std::vector<double>& profile_times
) {
    const size_t n = std::min(profile_tokens.size(), profile_times.size());
    if (n == 0) return 0.0;
    size_t best = 0;
    long long best_dist = std::llabs((long long)profile_tokens[0] - (long long)tokens);
    for (size_t i = 1; i < n; ++i) {
        long long d = std::llabs((long long)profile_tokens[i] - (long long)tokens);
        if (d < best_dist) {
            best = i;
            best_dist = d;
        }
    }
    return profile_times[best];
}

long long NativePredictor::pack_key(const NativePredictorKey& k) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](int x) {
        h ^= (std::uint64_t)(std::uint32_t)x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 1099511628211ull;
    };
    mix(k.total_tokens_rounded);
    mix(k.batch_size);
    mix(k.prefill_batch_size);
    mix(k.decode_batch_size);
    mix(k.decode_avg_kv_cache_size);
    mix(k.prefill_agg_kv_cache_size);
    mix(k.prefill_agg_chunk_size);
    return (long long)h;
}

NativePredictorKey NativePredictor::build_key(
    const std::vector<ControllerRequestStateNative>& reqs,
    const std::vector<int>& token_alloc,
    int kv_granularity,
    int prefill_chunk_granularity
) {
    NativePredictorKey k;
    const int n = (int)std::min(reqs.size(), token_alloc.size());
    if (n <= 0) return k;

    int total_num_tokens = 0;
    for (int i = 0; i < n; ++i) total_num_tokens += std::max(0, token_alloc[(size_t)i]);
    k.total_tokens_rounded = ((total_num_tokens + 7) / 8) * 8;
    k.batch_size = n;

    int prefill_batch = 0;
    int decode_batch = 0;
    std::vector<int> decode_kv;
    decode_kv.reserve(n);
    int prefill_agg_kv = 0;
    long long prefill_agg_chunk_sq = 0;

    for (int i = 0; i < n; ++i) {
        const auto& r = reqs[(size_t)i];
        const int tok = std::max(0, token_alloc[(size_t)i]);
        if (r.prefill_done) {
            decode_batch += 1;
            decode_kv.push_back(std::max(0, r.num_processed_tokens));
        } else {
            prefill_batch += 1;
            prefill_agg_kv += std::max(0, r.num_processed_tokens);
            prefill_agg_chunk_sq += 1LL * tok * tok;
        }
    }
    k.prefill_batch_size = prefill_batch;
    k.decode_batch_size = decode_batch;

    const int kv_g = std::max(1, kv_granularity);
    const int prefill_g = std::max(1, prefill_chunk_granularity);

    if (!decode_kv.empty()) {
        const long long s = std::accumulate(decode_kv.begin(), decode_kv.end(), 0LL);
        const int avg = (int)std::llround((double)s / (double)decode_kv.size());
        k.decode_avg_kv_cache_size = round_up(avg, kv_g);
    } else {
        k.decode_avg_kv_cache_size = 0;
    }
    k.prefill_agg_kv_cache_size = round_up(prefill_agg_kv, kv_g);
    k.prefill_agg_chunk_size = round_up((int)std::llround(std::sqrt((double)prefill_agg_chunk_sq)), prefill_g);
    return k;
}

bool NativePredictor::load_csv(const std::string& path) {
    table_.clear();
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in.good()) return false;

    std::string header_line;
    if (!std::getline(in, header_line)) return false;
    auto header = split_csv_line(header_line);
    std::unordered_map<std::string, int> col;
    for (int i = 0; i < (int)header.size(); ++i) col[header[(size_t)i]] = i;

    const bool has_full =
        col.count("total_tokens_rounded") &&
        col.count("batch_size") &&
        col.count("prefill_batch_size") &&
        col.count("decode_batch_size") &&
        col.count("decode_avg_kv_cache_size") &&
        col.count("prefill_agg_kv_cache_size") &&
        col.count("prefill_agg_chunk_size") &&
        col.count("total_time") &&
        col.count("model_time");

    const bool has_simple = col.count("num_tokens") && col.count("prediction");

    if (!has_full && !has_simple) return false;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto row = split_csv_line(line);
        NativePredictorKey k;
        NativePredictorValue v;
        if (has_full) {
            auto get_i = [&](const char* name, int dflt = 0) {
                auto it = col.find(name);
                if (it == col.end() || it->second < 0 || it->second >= (int)row.size()) return dflt;
                return to_int_or(row[(size_t)it->second], dflt);
            };
            auto get_d = [&](const char* name, double dflt = 0.0) {
                auto it = col.find(name);
                if (it == col.end() || it->second < 0 || it->second >= (int)row.size()) return dflt;
                return to_double_or(row[(size_t)it->second], dflt);
            };
            k.total_tokens_rounded = get_i("total_tokens_rounded");
            k.batch_size = get_i("batch_size");
            k.prefill_batch_size = get_i("prefill_batch_size");
            k.decode_batch_size = get_i("decode_batch_size");
            k.decode_avg_kv_cache_size = get_i("decode_avg_kv_cache_size");
            k.prefill_agg_kv_cache_size = get_i("prefill_agg_kv_cache_size");
            k.prefill_agg_chunk_size = get_i("prefill_agg_chunk_size");
            v.total_time = get_d("total_time");
            v.model_time = get_d("model_time");
        } else {
            const int t = to_int_or(row[(size_t)col["num_tokens"]], 0);
            const double p = to_double_or(row[(size_t)col["prediction"]], 0.0);
            k.total_tokens_rounded = ((std::max(0, t) + 7) / 8) * 8;
            k.batch_size = 1;
            v.total_time = p;
            v.model_time = p;
        }
        table_[pack_key(k)] = v;
    }
    return !table_.empty();
}

bool NativePredictor::is_loaded() const {
    return !table_.empty();
}

std::tuple<double, double> NativePredictor::lookup_batch_time(
    const std::vector<ControllerRequestStateNative>& reqs,
    const std::vector<int>& token_alloc,
    const std::vector<int>& profile_tokens,
    const std::vector<double>& profile_times,
    double fallback_total,
    double fallback_model
) {
    const NativePredictorKey k = build_key(
        reqs,
        token_alloc,
        kv_granularity_,
        prefill_chunk_granularity_
    );
    const auto it = table_.find(pack_key(k));
    if (it != table_.end()) {
        return std::make_tuple(it->second.total_time, it->second.model_time);
    }

    // Heuristic fallback for fully-native mode when no exact row exists.
    int prefill_tokens = 0;
    int decode_count = 0;
    for (size_t i = 0; i < reqs.size() && i < token_alloc.size(); ++i) {
        if (reqs[i].prefill_done) decode_count += 1;
        else prefill_tokens += std::max(0, token_alloc[i]);
    }

    double prefill_time = 0.0;
    if (prefill_tokens > 0) {
        prefill_time = nearest_prefill_estimate(prefill_tokens, profile_tokens, profile_times);
    }
    double decode_time = (double)decode_count * 0.0009;
    double total = prefill_time + decode_time;
    if (total <= 0.0) total = fallback_total;
    double model = total * 0.7;
    if (model <= 0.0) model = fallback_model;
    return std::make_tuple(total, model);
}

} // namespace mcts_native
