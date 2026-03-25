#include "native_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

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

static bool has_exact_key(const NativePredictorTable& table, int a, int b) {
    if (table.kind == "num_tokens") {
        return a >= 1 && a <= table.max_tokens;
    }
    if (table.kind == "batch_size") {
        return a >= 1 && a <= table.max_batch_size;
    }
    if (table.kind == "decode") {
        return a >= 1 && a <= table.max_batch_size &&
               b >= 0 && b <= table.max_tokens &&
               table.kv_gran > 0 && (b % table.kv_gran) == 0;
    }
    if (table.kind == "prefill") {
        return b >= 0 && b <= table.max_tokens &&
               table.kv_gran > 0 && (b % table.kv_gran) == 0 &&
               a >= table.prefill_gran &&
               table.prefill_gran > 0 && (a % table.prefill_gran) == 0;
    }
    return false;
}

static std::pair<int, int> clamp_key(const NativePredictorTable& table, int a, int b) {
    if (table.kind == "num_tokens") {
        a = std::max(1, std::min(a, table.max_tokens));
        return {a, 0};
    }
    if (table.kind == "batch_size") {
        a = std::max(1, std::min(a, table.max_batch_size));
        return {a, 0};
    }
    if (table.kind == "decode") {
        a = std::max(1, std::min(a, table.max_batch_size));
        b = std::max(0, std::min(b, table.max_tokens));
        const int g = std::max(1, table.kv_gran);
        b = (b / g) * g;
        return {a, b};
    }
    if (table.kind == "prefill") {
        b = std::max(0, std::min(b, table.max_tokens));
        const int kg = std::max(1, table.kv_gran);
        b = (b / kg) * kg;
        const int pg = std::max(1, table.prefill_gran);
        int max_chunk = pg;
        if (table.shape.size() >= 2) {
            max_chunk = std::max(pg, pg * table.shape[1]);
        }
        a = std::max(pg, std::min(a, max_chunk));
        a = (a / pg) * pg;
        return {a, b};
    }
    return {a, b};
}

static bool key_to_index(const NativePredictorTable& table, int a, int b, size_t& idx) {
    if (table.kind == "num_tokens") {
        if (table.shape.size() != 1) return false;
        if (a < 1 || a > table.shape[0]) return false;
        idx = (size_t)(a - 1);
        return idx < table.values.size();
    }
    if (table.kind == "batch_size") {
        if (table.shape.size() != 1) return false;
        if (a < 1 || a > table.shape[0]) return false;
        idx = (size_t)(a - 1);
        return idx < table.values.size();
    }
    if (table.kind == "decode") {
        if (table.shape.size() != 2) return false;
        if (table.kv_gran <= 0) return false;
        const int row = a - 1;
        const int col = b / table.kv_gran;
        if (row < 0 || row >= table.shape[0]) return false;
        if (col < 0 || col >= table.shape[1]) return false;
        idx = (size_t)row * (size_t)table.shape[1] + (size_t)col;
        return idx < table.values.size();
    }
    if (table.kind == "prefill") {
        if (table.shape.size() != 2) return false;
        if (table.kv_gran <= 0 || table.prefill_gran <= 0) return false;
        const int row = b / table.kv_gran;
        const int col = (a / table.prefill_gran) - 1;
        if (row < 0 || row >= table.shape[0]) return false;
        if (col < 0 || col >= table.shape[1]) return false;
        idx = (size_t)row * (size_t)table.shape[1] + (size_t)col;
        return idx < table.values.size();
    }
    return false;
}

static double table_lookup_with_python_fallback(const NativePredictorTable& table, int a, int b) {
    std::pair<int, int> key = has_exact_key(table, a, b) ? std::make_pair(a, b) : clamp_key(table, a, b);
    size_t idx = 0;
    if (!key_to_index(table, key.first, key.second, idx)) {
        return 0.0;
    }
    return table.values[idx];
}

static bool has_name(
    const std::unordered_map<std::string, NativePredictorTable>& tables,
    const char* name
) {
    return tables.find(name) != tables.end();
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

long long NativePredictor::pack_component_key(int a, int b) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](int x) {
        h ^= (std::uint64_t)(std::uint32_t)x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 1099511628211ull;
    };
    mix(a);
    mix(b);
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
    return !table_.empty() || !component_tables_.empty();
}

bool NativePredictor::has_component_tables() const {
    return !component_tables_.empty();
}

void NativePredictor::clear_component_tables() {
    component_tables_.clear();
}

void NativePredictor::set_runtime_config(const NativePredictorRuntimeConfig& cfg) {
    runtime_cfg_ = cfg;
}

void NativePredictor::set_component_table(
    const std::string& name,
    const std::string& kind,
    int max_tokens,
    int max_batch_size,
    int kv_gran,
    int prefill_gran,
    std::vector<int> shape,
    std::vector<double> values
) {
    NativePredictorTable table;
    table.kind = kind;
    table.max_tokens = max_tokens;
    table.max_batch_size = max_batch_size;
    table.kv_gran = std::max(1, kv_gran);
    table.prefill_gran = std::max(1, prefill_gran);
    table.shape = std::move(shape);
    table.values = std::move(values);
    component_tables_[name] = std::move(table);
    kv_granularity_ = std::max(1, kv_gran);
    prefill_chunk_granularity_ = std::max(1, prefill_gran);
}

std::tuple<double, double> NativePredictor::default_fallback_time(
    const std::vector<ControllerRequestStateNative>& reqs,
    const std::vector<int>& token_alloc,
    const std::vector<int>& profile_tokens,
    const std::vector<double>& profile_times,
    double fallback_total,
    double fallback_model
) {
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

std::tuple<double, double> NativePredictor::lookup_batch_time(
    const std::vector<ControllerRequestStateNative>& reqs,
    const std::vector<int>& token_alloc,
    const std::vector<int>& profile_tokens,
    const std::vector<double>& profile_times,
    double fallback_total,
    double fallback_model
) {
    const auto heuristic_fallback = [&]() {
        return default_fallback_time(
            reqs,
            token_alloc,
            profile_tokens,
            profile_times,
            fallback_total,
            fallback_model
        );
    };

    if (!component_tables_.empty()) {
        bool can_use_component_tables = true;
        const char* required_tables[] = {
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "input_layernorm",
            "add",
            "attn_rope",
            "attn_kv_cache_save",
            "attn_decode",
            "attn_prefill",
        };
        for (const char* name : required_tables) {
            if (!has_name(component_tables_, name)) {
                can_use_component_tables = false;
                break;
            }
        }
        if (can_use_component_tables && runtime_cfg_.post_attn_norm && !has_name(component_tables_, "post_attention_layernorm")) {
            can_use_component_tables = false;
        }
        if (can_use_component_tables && !runtime_cfg_.skip_cpu_overhead_modeling) {
            const char* cpu_tables[] = {
                "schedule",
                "sampler_e2e",
                "prepare_inputs_e2e",
                "process_model_outputs",
                "ray_comm_time",
            };
            for (const char* name : cpu_tables) {
                if (!has_name(component_tables_, name)) {
                    can_use_component_tables = false;
                    break;
                }
            }
        }
        if (can_use_component_tables && runtime_cfg_.tensor_parallel_size > 1 && !has_name(component_tables_, "all_reduce")) {
            can_use_component_tables = false;
        }
        if (can_use_component_tables && runtime_cfg_.num_pipeline_stages > 1 && !has_name(component_tables_, "send_recv")) {
            can_use_component_tables = false;
        }

        if (can_use_component_tables) {
            const NativePredictorKey k = build_key(
                reqs,
                token_alloc,
                kv_granularity_,
                prefill_chunk_granularity_
            );
            int total_num_tokens = 0;
            for (int tok : token_alloc) total_num_tokens += std::max(0, tok);

            const auto lookup1 = [&](const char* name, int a) -> double {
                const auto it = component_tables_.find(name);
                if (it == component_tables_.end()) return 0.0;
                return table_lookup_with_python_fallback(it->second, a, 0);
            };
            const auto lookup2 = [&](const char* name, int a, int b) -> double {
                const auto it = component_tables_.find(name);
                if (it == component_tables_.end()) return 0.0;
                return table_lookup_with_python_fallback(it->second, a, b);
            };

            const double attn_pre_proj = lookup1("attn_pre_proj", k.total_tokens_rounded);
            const double attn_post_proj = lookup1("attn_post_proj", k.total_tokens_rounded);
            const double mlp_up_proj = lookup1("mlp_up_proj", k.total_tokens_rounded);
            const double mlp_down_proj = lookup1("mlp_down_proj", k.total_tokens_rounded);
            const double mlp_act = lookup1("mlp_act", k.total_tokens_rounded);
            const double attn_norm = lookup1("input_layernorm", k.total_tokens_rounded);
            const double mlp_norm = runtime_cfg_.post_attn_norm
                ? lookup1("post_attention_layernorm", k.total_tokens_rounded)
                : 0.0;
            const double add_time = lookup1("add", k.total_tokens_rounded);
            const double attn_rope = lookup1("attn_rope", k.total_tokens_rounded);
            const double attn_kv_cache_save = lookup1("attn_kv_cache_save", total_num_tokens);

            double attn_decode = 0.0;
            if (k.decode_batch_size > 0) {
                const double base = lookup2("attn_decode", k.decode_batch_size, k.decode_avg_kv_cache_size);
                attn_decode = base * (
                    1.0 + runtime_cfg_.attention_decode_batching_overhead_fraction * (k.decode_batch_size > 1 ? 1.0 : 0.0)
                );
            }

            double attn_prefill = 0.0;
            if (k.prefill_batch_size > 0) {
                const double base = lookup2("attn_prefill", k.prefill_agg_chunk_size, k.prefill_agg_kv_cache_size);
                attn_prefill = base * (
                    1.0 + runtime_cfg_.attention_prefill_batching_overhead_fraction * (k.prefill_batch_size > 1 ? 1.0 : 0.0)
                );
            }

            double tensor_parallel_comm = 0.0;
            if (runtime_cfg_.tensor_parallel_size > 1) {
                tensor_parallel_comm =
                    lookup1("all_reduce", k.total_tokens_rounded) +
                    runtime_cfg_.nccl_cpu_launch_overhead_ms +
                    runtime_cfg_.nccl_cpu_skew_overhead_per_device_ms * std::pow((double)runtime_cfg_.tensor_parallel_size, 1.25);
            }

            double pipeline_parallel_comm = 0.0;
            if (runtime_cfg_.num_pipeline_stages > 1) {
                pipeline_parallel_comm = lookup1("send_recv", k.total_tokens_rounded);
            }

            double cpu_overhead_ms = 0.0;
            if (!runtime_cfg_.skip_cpu_overhead_modeling) {
                cpu_overhead_ms += lookup1("schedule", k.batch_size);
                cpu_overhead_ms += lookup1("sampler_e2e", k.batch_size);
                cpu_overhead_ms += lookup1("prepare_inputs_e2e", k.batch_size);
                cpu_overhead_ms += lookup1("process_model_outputs", k.batch_size);
                cpu_overhead_ms += lookup1("ray_comm_time", k.batch_size);
            }

            const double attn_layer_ms =
                attn_pre_proj +
                attn_post_proj +
                attn_rope +
                attn_kv_cache_save +
                attn_decode +
                attn_prefill +
                tensor_parallel_comm +
                attn_norm;
            const double mlp_layer_ms =
                mlp_up_proj +
                mlp_down_proj +
                mlp_act +
                tensor_parallel_comm +
                mlp_norm;
            const double block_ms = attn_layer_ms + mlp_layer_ms + add_time;
            const double model_ms = block_ms * std::max(1, runtime_cfg_.num_layers_per_pipeline_stage) + pipeline_parallel_comm;
            const double total_ms = model_ms + cpu_overhead_ms;
            return std::make_tuple(total_ms * 1e-3, model_ms * 1e-3);
        }
    }

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

    return heuristic_fallback();
}

} // namespace mcts_native
