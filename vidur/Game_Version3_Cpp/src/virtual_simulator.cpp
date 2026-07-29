#include "virtual_simulator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>

namespace mcts_native_gv2 {

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
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

std::vector<std::string> split_sv(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(s.size());
    for (char c : s) {
        if (c == delim) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

int to_int_or(const std::string& s, int dflt) {
    try {
        return std::stoi(s);
    } catch (...) {
        return dflt;
    }
}

double to_double_or(const std::string& s, double dflt) {
    try {
        return std::stod(s);
    } catch (...) {
        return dflt;
    }
}

bool to_bool_or(const std::string& s, bool dflt) {
    if (s == "1" || s == "true" || s == "True") return true;
    if (s == "0" || s == "false" || s == "False") return false;
    return dflt;
}

std::string join_i32(const std::vector<int>& v) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) oss << ',';
        oss << v[i];
    }
    return oss.str();
}

std::string join_f64(const std::vector<double>& v) {
    std::ostringstream oss;
    oss.precision(17);
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) oss << ',';
        oss << v[i];
    }
    return oss.str();
}

template <typename T>
std::string join_map_i32(const std::unordered_map<int, T>& m) {
    std::vector<int> keys;
    keys.reserve(m.size());
    for (const auto& kv : m) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());

    std::ostringstream oss;
    oss.precision(17);
    bool first = true;
    for (int k : keys) {
        const auto it = m.find(k);
        if (it == m.end()) continue;
        if (!first) oss << ',';
        first = false;
        oss << k << ':' << it->second;
    }
    return oss.str();
}

std::vector<int> parse_i32_vec(const std::string& s) {
    std::vector<int> out;
    if (s.empty()) return out;
    for (const auto& tok : split_sv(s, ',')) {
        if (tok.empty()) continue;
        out.push_back(to_int_or(tok, 0));
    }
    return out;
}

std::vector<double> parse_f64_vec(const std::string& s) {
    std::vector<double> out;
    if (s.empty()) return out;
    for (const auto& tok : split_sv(s, ',')) {
        if (tok.empty()) continue;
        out.push_back(to_double_or(tok, 0.0));
    }
    return out;
}

template <typename T>
std::unordered_map<int, T> parse_map_i32(const std::string& s) {
    std::unordered_map<int, T> out;
    if (s.empty()) return out;
    for (const auto& item : split_sv(s, ',')) {
        if (item.empty()) continue;
        const auto pos = item.find(':');
        if (pos == std::string::npos) continue;
        const int k = to_int_or(item.substr(0, pos), 0);
        const std::string v = item.substr(pos + 1);
        if constexpr (std::is_same<T, int>::value) {
            out[k] = to_int_or(v, 0);
        } else {
            out[k] = static_cast<T>(to_double_or(v, 0.0));
        }
    }
    return out;
}

bool has_exact_key(const PredictorTable& table, int a, int b) {
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

std::pair<int, int> clamp_key(const PredictorTable& table, int a, int b) {
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

bool key_to_index(const PredictorTable& table, int a, int b, std::size_t& idx) {
    if (table.kind == "num_tokens") {
        if (table.shape.size() != 1) return false;
        if (a < 1 || a > table.shape[0]) return false;
        idx = static_cast<std::size_t>(a - 1);
        return idx < table.values.size();
    }
    if (table.kind == "batch_size") {
        if (table.shape.size() != 1) return false;
        if (a < 1 || a > table.shape[0]) return false;
        idx = static_cast<std::size_t>(a - 1);
        return idx < table.values.size();
    }
    if (table.kind == "decode") {
        if (table.shape.size() != 2) return false;
        if (table.kv_gran <= 0) return false;
        const int row = a - 1;
        const int col = b / table.kv_gran;
        if (row < 0 || row >= table.shape[0]) return false;
        if (col < 0 || col >= table.shape[1]) return false;
        idx = static_cast<std::size_t>(row) * static_cast<std::size_t>(table.shape[1]) +
              static_cast<std::size_t>(col);
        return idx < table.values.size();
    }
    if (table.kind == "prefill") {
        if (table.shape.size() != 2) return false;
        if (table.kv_gran <= 0 || table.prefill_gran <= 0) return false;
        const int row = b / table.kv_gran;
        const int col = (a / table.prefill_gran) - 1;
        if (row < 0 || row >= table.shape[0]) return false;
        if (col < 0 || col >= table.shape[1]) return false;
        idx = static_cast<std::size_t>(row) * static_cast<std::size_t>(table.shape[1]) +
              static_cast<std::size_t>(col);
        return idx < table.values.size();
    }
    return false;
}

double table_lookup_with_python_fallback(const PredictorTable& table, int a, int b) {
    const std::pair<int, int> key = has_exact_key(table, a, b)
        ? std::make_pair(a, b)
        : clamp_key(table, a, b);
    std::size_t idx = 0;
    if (!key_to_index(table, key.first, key.second, idx)) {
        return 0.0;
    }
    return table.values[idx];
}

bool has_name(const std::unordered_map<std::string, PredictorTable>& tables, const char* name) {
    return tables.find(name) != tables.end();
}

bool request_is_active(const RequestState& r) {
    if (r.completed) return false;
    return r.prefill_active() || r.decode_active();
}

double nearest_profile_lookup(
    int tokens,
    const std::vector<int>& profile_tokens,
    const std::vector<double>& profile_times) {
    const std::size_t n = std::min(profile_tokens.size(), profile_times.size());
    if (n == 0) return 0.0;
    std::size_t best = 0;
    long long best_dist = std::llabs(static_cast<long long>(profile_tokens[0]) -
                                     static_cast<long long>(tokens));
    for (std::size_t i = 1; i < n; ++i) {
        const long long dist = std::llabs(static_cast<long long>(profile_tokens[i]) -
                                          static_cast<long long>(tokens));
        if (dist < best_dist) {
            best = i;
            best_dist = dist;
        }
    }
    return profile_times[best];
}

}  // namespace

long long NativeBatchTimePredictorGV2::pack_key(const PredictorKey& k) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](int x) {
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) +
             0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 1099511628211ull;
    };
    mix(k.total_tokens_rounded);
    mix(k.batch_size);
    mix(k.prefill_batch_size);
    mix(k.decode_batch_size);
    mix(k.decode_avg_kv_cache_size);
    mix(k.prefill_agg_kv_cache_size);
    mix(k.prefill_agg_chunk_size);
    return static_cast<long long>(h);
}

int NativeBatchTimePredictorGV2::round_up(int x, int g) {
    const int gg = std::max(1, g);
    return ((x + gg - 1) / gg) * gg;
}

double NativeBatchTimePredictorGV2::nearest_prefill_estimate(
    int tokens,
    const std::vector<int>& profile_tokens,
    const std::vector<double>& profile_times) {
    const std::size_t n = std::min(profile_tokens.size(), profile_times.size());
    if (n == 0) return 0.0;
    std::size_t best = 0;
    long long best_dist = std::llabs(static_cast<long long>(profile_tokens[0]) -
                                     static_cast<long long>(tokens));
    for (std::size_t i = 1; i < n; ++i) {
        const long long d = std::llabs(static_cast<long long>(profile_tokens[i]) -
                                       static_cast<long long>(tokens));
        if (d < best_dist) {
            best = i;
            best_dist = d;
        }
    }
    return profile_times[best];
}

std::tuple<double, double> NativeBatchTimePredictorGV2::default_fallback_time(
    const InlineVector<ControllerPredictorRequestState, 32>& reqs,
    const InlineVector<int, 32>& token_alloc,
    const std::vector<int>& profile_tokens,
    const std::vector<double>& profile_times,
    double fallback_total,
    double fallback_model) {
    int prefill_tokens = 0;
    int decode_count = 0;
    for (std::size_t i = 0; i < reqs.size() && i < token_alloc.size(); ++i) {
        if (reqs[i].prefill_done) {
            decode_count += 1;
        } else {
            prefill_tokens += std::max(0, token_alloc[i]);
        }
    }

    double prefill_time = 0.0;
    if (prefill_tokens > 0) {
        prefill_time = nearest_prefill_estimate(prefill_tokens, profile_tokens, profile_times);
    }
    const double decode_time = static_cast<double>(decode_count) * 0.0009;
    double total = prefill_time + decode_time;
    if (total <= 0.0) total = fallback_total;
    double model = total * 0.7;
    if (model <= 0.0) model = fallback_model;
    return std::make_tuple(total, model);
}

PredictorKey NativeBatchTimePredictorGV2::build_key(
    const InlineVector<ControllerPredictorRequestState, 32>& reqs,
    const InlineVector<int, 32>& token_alloc,
    int kv_granularity,
    int prefill_chunk_granularity) {
    PredictorKey k;
    const int n = static_cast<int>(std::min(reqs.size(), token_alloc.size()));
    if (n <= 0) return k;

    int total_num_tokens = 0;
    for (int i = 0; i < n; ++i) total_num_tokens += std::max(0, token_alloc[static_cast<std::size_t>(i)]);
    k.total_tokens_rounded = ((total_num_tokens + 7) / 8) * 8;
    k.batch_size = n;

    int prefill_batch = 0;
    int decode_batch = 0;
    long long decode_kv_sum = 0;
    int prefill_agg_kv = 0;
    long long prefill_agg_chunk_sq = 0;

    for (int i = 0; i < n; ++i) {
        const auto& r = reqs[static_cast<std::size_t>(i)];
        const int tok = std::max(0, token_alloc[static_cast<std::size_t>(i)]);
        if (r.prefill_done) {
            decode_batch += 1;
            decode_kv_sum += std::max(0, r.num_processed_tokens);
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

    if (decode_batch > 0) {
        // Python uses int(np.mean(...)) before granularity rounding, i.e. floor
        // for these non-negative token counts. Using llround here changes
        // half cases like 128.5 -> 192 at granularity 64, which breaks parity.
        const int avg = static_cast<int>(static_cast<double>(decode_kv_sum) /
                                         static_cast<double>(decode_batch));
        k.decode_avg_kv_cache_size = round_up(avg, kv_g);
    }

    k.prefill_agg_kv_cache_size = round_up(prefill_agg_kv, kv_g);
    k.prefill_agg_chunk_size = round_up(
        static_cast<int>(std::llround(std::sqrt(static_cast<double>(prefill_agg_chunk_sq)))),
        prefill_g);
    return k;
}

bool NativeBatchTimePredictorGV2::load_csv(const std::string& path) {
    table_.clear();
    if (path.empty()) return false;

    std::ifstream in(path);
    if (!in.good()) return false;

    std::string header_line;
    if (!std::getline(in, header_line)) return false;
    const auto header = split_csv_line(header_line);

    std::unordered_map<std::string, int> col;
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        col[header[static_cast<std::size_t>(i)]] = i;
    }

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
        const auto row = split_csv_line(line);

        PredictorKey k;
        PredictorValue v;

        if (has_full) {
            auto get_i = [&](const char* name, int dflt = 0) {
                const auto it = col.find(name);
                if (it == col.end() || it->second < 0 || it->second >= static_cast<int>(row.size())) {
                    return dflt;
                }
                return to_int_or(row[static_cast<std::size_t>(it->second)], dflt);
            };
            auto get_d = [&](const char* name, double dflt = 0.0) {
                const auto it = col.find(name);
                if (it == col.end() || it->second < 0 || it->second >= static_cast<int>(row.size())) {
                    return dflt;
                }
                return to_double_or(row[static_cast<std::size_t>(it->second)], dflt);
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
            const int t = to_int_or(row[static_cast<std::size_t>(col["num_tokens"])], 0);
            const double p = to_double_or(row[static_cast<std::size_t>(col["prediction"])], 0.0);
            k.total_tokens_rounded = ((std::max(0, t) + 7) / 8) * 8;
            k.batch_size = 1;
            v.total_time = p;
            v.model_time = p;
        }

        table_[pack_key(k)] = v;
    }

    return !table_.empty();
}

bool NativeBatchTimePredictorGV2::is_loaded() const {
    return !table_.empty() || !component_tables_.empty();
}

bool NativeBatchTimePredictorGV2::has_component_tables() const {
    return !component_tables_.empty();
}

void NativeBatchTimePredictorGV2::clear_component_tables() {
    component_tables_.clear();
    invalidate_component_table_cache();
}

void NativeBatchTimePredictorGV2::set_runtime_config(const PredictorRuntimeConfig& cfg) {
    runtime_cfg_ = cfg;
}

void NativeBatchTimePredictorGV2::set_component_table(
    const std::string& name,
    const std::string& kind,
    int max_tokens,
    int max_batch_size,
    int kv_gran,
    int prefill_gran,
    std::vector<int> shape,
    std::vector<double> values) {
    PredictorTable table;
    table.kind = kind;
    table.max_tokens = max_tokens;
    table.max_batch_size = max_batch_size;
    table.kv_gran = std::max(1, kv_gran);
    table.prefill_gran = std::max(1, prefill_gran);
    table.shape = std::move(shape);
    table.values = std::move(values);
    component_tables_[name] = std::move(table);
    invalidate_component_table_cache();

    kv_granularity_ = std::max(1, kv_gran);
    prefill_chunk_granularity_ = std::max(1, prefill_gran);
}

void NativeBatchTimePredictorGV2::invalidate_component_table_cache() {
    component_table_cache_ = {};
}

const NativeBatchTimePredictorGV2::ComponentTableCache&
NativeBatchTimePredictorGV2::component_table_cache() const {
    if (component_table_cache_.initialized) {
        return component_table_cache_;
    }

    const auto find_table = [&](const char* name) -> const PredictorTable* {
        const auto it = component_tables_.find(name);
        return it == component_tables_.end() ? nullptr : &it->second;
    };
    auto& cache = component_table_cache_;
    cache.attn_pre_proj = find_table("attn_pre_proj");
    cache.attn_post_proj = find_table("attn_post_proj");
    cache.mlp_up_proj = find_table("mlp_up_proj");
    cache.mlp_down_proj = find_table("mlp_down_proj");
    cache.mlp_act = find_table("mlp_act");
    cache.input_layernorm = find_table("input_layernorm");
    cache.add = find_table("add");
    cache.attn_rope = find_table("attn_rope");
    cache.attn_kv_cache_save = find_table("attn_kv_cache_save");
    cache.attn_decode = find_table("attn_decode");
    cache.attn_prefill = find_table("attn_prefill");
    cache.post_attention_layernorm = find_table("post_attention_layernorm");
    cache.schedule = find_table("schedule");
    cache.sampler_e2e = find_table("sampler_e2e");
    cache.prepare_inputs_e2e = find_table("prepare_inputs_e2e");
    cache.process_model_outputs = find_table("process_model_outputs");
    cache.ray_comm_time = find_table("ray_comm_time");
    cache.all_reduce = find_table("all_reduce");
    cache.send_recv = find_table("send_recv");
    cache.initialized = true;
    return cache;
}

std::tuple<double, double> NativeBatchTimePredictorGV2::lookup_batch_time(
    const InlineVector<ControllerPredictorRequestState, 32>& reqs,
    const InlineVector<int, 32>& token_alloc,
    const std::vector<int>& profile_tokens,
    const std::vector<double>& profile_times,
    double fallback_total,
    double fallback_model) const {
    const auto heuristic_fallback = [&]() {
        return default_fallback_time(
            reqs,
            token_alloc,
            profile_tokens,
            profile_times,
            fallback_total,
            fallback_model);
    };

    if (!component_tables_.empty()) {
        const auto& tables = component_table_cache();
        bool can_use_component_tables =
            tables.attn_pre_proj &&
            tables.attn_post_proj &&
            tables.mlp_up_proj &&
            tables.mlp_down_proj &&
            tables.mlp_act &&
            tables.input_layernorm &&
            tables.add &&
            tables.attn_rope &&
            tables.attn_kv_cache_save &&
            tables.attn_decode &&
            tables.attn_prefill;
        if (can_use_component_tables && runtime_cfg_.post_attn_norm) {
            can_use_component_tables = tables.post_attention_layernorm;
        }
        if (can_use_component_tables && !runtime_cfg_.skip_cpu_overhead_modeling) {
            can_use_component_tables =
                tables.schedule &&
                tables.sampler_e2e &&
                tables.prepare_inputs_e2e &&
                tables.process_model_outputs &&
                tables.ray_comm_time;
        }
        if (can_use_component_tables && runtime_cfg_.tensor_parallel_size > 1) {
            can_use_component_tables = tables.all_reduce;
        }
        if (can_use_component_tables && runtime_cfg_.num_pipeline_stages > 1) {
            can_use_component_tables = tables.send_recv;
        }

        if (can_use_component_tables) {
            const PredictorKey k = build_key(reqs, token_alloc, kv_granularity_, prefill_chunk_granularity_);
            int total_num_tokens = 0;
            for (int tok : token_alloc) total_num_tokens += std::max(0, tok);

            const auto lookup1 = [&](const PredictorTable* table, int a) -> double {
                return table ? table_lookup_with_python_fallback(*table, a, 0) : 0.0;
            };
            const auto lookup2 = [&](const PredictorTable* table, int a, int b) -> double {
                return table ? table_lookup_with_python_fallback(*table, a, b) : 0.0;
            };

            const double attn_pre_proj = lookup1(tables.attn_pre_proj, k.total_tokens_rounded);
            const double attn_post_proj = lookup1(tables.attn_post_proj, k.total_tokens_rounded);
            const double mlp_up_proj = lookup1(tables.mlp_up_proj, k.total_tokens_rounded);
            const double mlp_down_proj = lookup1(tables.mlp_down_proj, k.total_tokens_rounded);
            const double mlp_act = lookup1(tables.mlp_act, k.total_tokens_rounded);
            const double attn_norm = lookup1(tables.input_layernorm, k.total_tokens_rounded);
            const double mlp_norm = runtime_cfg_.post_attn_norm
                ? lookup1(tables.post_attention_layernorm, k.total_tokens_rounded)
                : 0.0;
            const double add_time = lookup1(tables.add, k.total_tokens_rounded);
            const double attn_rope = lookup1(tables.attn_rope, k.total_tokens_rounded);
            const double attn_kv_cache_save = lookup1(tables.attn_kv_cache_save, total_num_tokens);

            double attn_decode = 0.0;
            if (k.decode_batch_size > 0) {
                const double base = lookup2(tables.attn_decode, k.decode_batch_size, k.decode_avg_kv_cache_size);
                attn_decode = base *
                    (1.0 + runtime_cfg_.attention_decode_batching_overhead_fraction *
                               (k.decode_batch_size > 1 ? 1.0 : 0.0));
            }

            double attn_prefill = 0.0;
            if (k.prefill_batch_size > 0) {
                const double base = lookup2(tables.attn_prefill, k.prefill_agg_chunk_size, k.prefill_agg_kv_cache_size);
                attn_prefill = base *
                    (1.0 + runtime_cfg_.attention_prefill_batching_overhead_fraction *
                               (k.prefill_batch_size > 1 ? 1.0 : 0.0));
            }

            double tensor_parallel_comm = 0.0;
            if (runtime_cfg_.tensor_parallel_size > 1) {
                tensor_parallel_comm =
                    lookup1(tables.all_reduce, k.total_tokens_rounded) +
                    runtime_cfg_.nccl_cpu_launch_overhead_ms +
                    runtime_cfg_.nccl_cpu_skew_overhead_per_device_ms *
                        std::pow(static_cast<double>(runtime_cfg_.tensor_parallel_size), 1.25);
            }

            double pipeline_parallel_comm = 0.0;
            if (runtime_cfg_.num_pipeline_stages > 1) {
                pipeline_parallel_comm = lookup1(tables.send_recv, k.total_tokens_rounded);
            }

            double cpu_overhead_ms = 0.0;
            if (!runtime_cfg_.skip_cpu_overhead_modeling) {
                cpu_overhead_ms += lookup1(tables.schedule, k.batch_size);
                cpu_overhead_ms += lookup1(tables.sampler_e2e, k.batch_size);
                cpu_overhead_ms += lookup1(tables.prepare_inputs_e2e, k.batch_size);
                cpu_overhead_ms += lookup1(tables.process_model_outputs, k.batch_size);
                cpu_overhead_ms += lookup1(tables.ray_comm_time, k.batch_size);
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
            const double model_ms =
                block_ms * std::max(1, runtime_cfg_.num_layers_per_pipeline_stage) +
                pipeline_parallel_comm;
            const double total_ms = model_ms + cpu_overhead_ms;
            return std::make_tuple(total_ms * 1e-3, model_ms * 1e-3);
        }
    }

    const PredictorKey k = build_key(reqs, token_alloc, kv_granularity_, prefill_chunk_granularity_);
    const auto it = table_.find(pack_key(k));
    if (it != table_.end()) {
        return std::make_tuple(it->second.total_time, it->second.model_time);
    }

    return heuristic_fallback();
}

VirtualSimulatorGV2::VirtualSimulatorGV2(VirtualSimulatorConfig cfg) : cfg_(std::move(cfg)) {}

const VirtualSimulatorConfig& VirtualSimulatorGV2::cfg() const { return cfg_; }

void VirtualSimulatorGV2::set_config(VirtualSimulatorConfig cfg) { cfg_ = std::move(cfg); }

void VirtualSimulatorGV2::set_prefill_profile(std::vector<int> tokens, std::vector<double> times) {
    cfg_.prefill_profile_tokens = std::move(tokens);
    cfg_.prefill_profile_times = std::move(times);
}

double VirtualSimulatorGV2::prefill_profile_lookup(int tokens) const {
    return nearest_profile_lookup(
        tokens,
        cfg_.prefill_profile_tokens,
        cfg_.prefill_profile_times);
}

bool VirtualSimulatorGV2::load_predictor_csv(const std::string& path) {
    return predictor_.load_csv(path);
}

NativeBatchTimePredictorGV2& VirtualSimulatorGV2::predictor() { return predictor_; }

const NativeBatchTimePredictorGV2& VirtualSimulatorGV2::predictor() const { return predictor_; }

double VirtualSimulatorGV2::next_adv_tick(double sim_time) const {
    const double tick = cfg_.adversary_tick_sec > 0.0 ? cfg_.adversary_tick_sec : 0.2;
    const double q = std::floor(sim_time / tick);
    const double t = (q + 1.0) * tick;
    return (t > sim_time) ? t : (sim_time + tick);
}

ControllerBatchPlan VirtualSimulatorGV2::build_controller_batch_plan(
    const SimState& state,
    const ControllerAction& action,
    bool enforce_nonnegative_decode_credits,
    int decode_credit_available) const {
    ControllerBatchPlan plan;

    plan.strict_noop =
        action.token_budget == 0 &&
        action.selected_request_ids.empty() &&
        action.token_allocations.empty() &&
        action.prefill_allocations.empty() &&
        action.decode_allocations.empty() &&
        !action.compact_allocations;

    InlineVector<int, 64> selected_ids;
    if (plan.strict_noop) {
        // No-op means no prefill admission but decode can still execute.
        for (const auto& request : state.requests) {
            if (request.decode_active()) {
                selected_ids.push_back(request.request_id);
            }
        }
    } else if (action.compact_allocations) {
        selected_ids.reserve(
            action.compact_prefill_allocations.size() +
            action.compact_decode_request_ids.size());
        for (const auto& item : action.compact_prefill_allocations) {
            selected_ids.push_back(item.first);
        }
        for (const int request_id : action.compact_decode_request_ids) {
            selected_ids.push_back(request_id);
        }
    } else if (!action.selected_request_ids.empty()) {
        selected_ids.assign(
            action.selected_request_ids.begin(),
            action.selected_request_ids.end());
    } else {
        selected_ids.reserve(
            action.token_allocations.size() +
            action.prefill_allocations.size() +
            action.decode_allocations.size());
        for (const auto& item : action.token_allocations) {
            selected_ids.push_back(item.first);
        }
        for (const auto& item : action.prefill_allocations) {
            selected_ids.push_back(item.first);
        }
        for (const auto& item : action.decode_allocations) {
            selected_ids.push_back(item.first);
        }
    }

    std::sort(selected_ids.begin(), selected_ids.end());

    int decode_credit_left = enforce_nonnegative_decode_credits
        ? std::max(0, decode_credit_available)
        : std::numeric_limits<int>::max();

    bool have_previous_id = false;
    int previous_id = 0;
    for (const int rid : selected_ids) {
        if (have_previous_id && rid == previous_id) continue;
        previous_id = rid;
        have_previous_id = true;
        const RequestState* req = nullptr;
        if (rid >= 0 && rid < static_cast<int>(state.requests.size())) {
            const auto& direct =
                state.requests[static_cast<std::size_t>(rid)];
            if (direct.request_id == rid) req = &direct;
        }
        if (req == nullptr) {
            for (const auto& request : state.requests) {
                if (request.request_id == rid) {
                    req = &request;
                    break;
                }
            }
        }
        if (req == nullptr || !request_is_active(*req)) continue;

        const bool prefill_done = req->prefill_done();
        const int rem_pref = req->remaining_prefill();
        const int rem_dec = req->remaining_decode();

        int pre_tok = 0;
        int dec_tok = plan.strict_noop ? 1 : 0;
        if (action.compact_allocations) {
            for (const auto& item : action.compact_prefill_allocations) {
                if (item.first == rid) {
                    pre_tok = std::max(0, item.second);
                    break;
                }
            }
            if (std::binary_search(
                    action.compact_decode_request_ids.begin(),
                    action.compact_decode_request_ids.end(),
                    rid)) {
                dec_tok = 1;
            }
        } else {
            const auto prefill_it = action.prefill_allocations.find(rid);
            if (prefill_it != action.prefill_allocations.end()) {
                pre_tok = std::max(0, prefill_it->second);
            }
            const auto decode_it = action.decode_allocations.find(rid);
            if (decode_it != action.decode_allocations.end()) {
                dec_tok = std::max(0, decode_it->second);
            }
        }

        if (pre_tok == 0 && dec_tok == 0) {
            const auto token_it = action.token_allocations.find(rid);
            if (token_it != action.token_allocations.end()) {
                const int base = std::max(0, token_it->second);
                if (prefill_done) dec_tok = base;
                else pre_tok = base;
            }
        }

        pre_tok = std::min(pre_tok, rem_pref);
        if (!prefill_done) dec_tok = 0;
        dec_tok = std::min(dec_tok, rem_dec);

        if (dec_tok > 0 && enforce_nonnegative_decode_credits) {
            if (decode_credit_left <= 0) {
                dec_tok = 0;
            } else {
                dec_tok = std::min(dec_tok, decode_credit_left);
                decode_credit_left -= dec_tok;
            }
        }

        const int total = pre_tok + dec_tok;
        if (total <= 0) continue;

        plan.request_ids.push_back(rid);
        plan.prefill_tokens.push_back(pre_tok);
        plan.decode_tokens.push_back(dec_tok);
        plan.num_tokens.push_back(total);

        ControllerPredictorRequestState prs;
        prs.request_id = rid;
        prs.prefill_done = prefill_done;
        prs.remaining_prefill = rem_pref;
        prs.remaining_decode = rem_dec;
        prs.arrived_at = req->arrived_at;
        prs.prefill_slo = req->prefill_slo_time;
        prs.num_processed_tokens =
            req->num_processed_prefill_tokens +
            req->num_processed_decode_tokens;
        plan.predictor_reqs.push_back(prs);
    }

    return plan;
}

BatchExecutionResult VirtualSimulatorGV2::execute_controller_batch_timing(
    SimState& state,
    const ControllerBatchPlan& plan) const {
    BatchExecutionResult out;
    if (plan.predictor_reqs.empty()) return out;

    out.start_time = state.sim_time;
    const auto pred = predictor_.lookup_batch_time(
        plan.predictor_reqs,
        plan.num_tokens,
        cfg_.prefill_profile_tokens,
        cfg_.prefill_profile_times,
        cfg_.fallback_total_time_sec,
        cfg_.fallback_model_time_sec);

    out.stage_total_time_sec = std::max(0.0, std::get<0>(pred));
    out.stage_model_time_sec = std::max(0.0, std::get<1>(pred));
    out.end_time = out.start_time + out.stage_total_time_sec;
    out.executed = true;

    state.sim_time = out.end_time;
    return out;
}

bool VirtualSimulatorGV2::is_pending(const RequestState& r) {
    if (r.completed) return false;
    if (r.prefill_active()) return true;
    return r.decode_active();
}

RequestState* VirtualSimulatorGV2::find_request(
    SimState& state,
    int request_id) {
    if (request_id >= 0 &&
        request_id < static_cast<int>(state.requests.size())) {
        auto& direct =
            state.requests[static_cast<std::size_t>(request_id)];
        if (direct.request_id == request_id) return &direct;
    }
    for (auto& r : state.requests) {
        if (r.request_id == request_id) return &r;
    }
    return nullptr;
}

bool VirtualSimulatorGV2::maybe_fast_forward_decode_only_to_next_adv_tick(SimState& state) const {
    bool any_prefill = false;
    bool any_decode = false;
    for (const auto& r : state.requests) {
        if (r.prefill_active()) {
            any_prefill = true;
            break;
        }
        if (r.decode_active()) any_decode = true;
    }
    if (any_prefill) return false;

    const double target = next_adv_tick(state.sim_time);
    if (state.sim_time >= target - 1e-9) return false;

    state.sim_time = target;
    if (any_decode) {
        for (auto& r : state.requests) {
            if (!r.decode_active()) continue;
            if (r.decode_slo_time > 0.0) {
                r.decode_next_deadline = target + r.decode_slo_time;
                state.stats.decode_next_deadline_by_id[r.request_id] = r.decode_next_deadline;
            }
        }
    }
    return true;
}

SimState VirtualSimulatorGV2::fork_state(const SimState& in_state) const {
    return in_state;
}

std::string VirtualSimulatorGV2::snapshot_state(const SimState& in_state) const {
    std::ostringstream oss;
    oss.precision(17);

    oss << "SIM\t" << in_state.sim_time << '\t'
        << in_state.decision_state_time << '\t'
        << in_state.next_request_id << '\n';

    oss << "REQS\t" << in_state.requests.size() << '\n';
    for (const auto& r : in_state.requests) {
        oss << "REQ\t"
            << r.request_id << '\t'
            << r.arrived_at << '\t'
            << r.queued_at << '\t'
            << r.num_prefill_tokens << '\t'
            << r.num_processed_prefill_tokens << '\t'
            << r.num_decode_tokens << '\t'
            << r.num_processed_decode_tokens << '\t'
            << r.prefill_slo_time << '\t'
            << r.decode_slo_time << '\t'
            << r.completion_slo_time << '\t'
            << r.prefill_deadline << '\t'
            << r.decode_next_deadline << '\t'
            << r.prefill_completed_at << '\t'
            << r.completed_at << '\t'
            << r.prefill_lateness << '\t'
            << r.decode_lateness << '\t'
            << (r.is_prefill_complete ? 1 : 0) << '\t'
            << (r.completed ? 1 : 0) << '\t'
            << (r.dropped ? 1 : 0) << '\t'
            << (r.stopped_decode ? 1 : 0) << '\t'
            << (r.violated ? 1 : 0)
            << '\n';
    }

    const auto& s = in_state.stats;
    oss << "STAT\trequests_generated\t" << s.requests_generated << '\n';
    oss << "STAT\trequests_completed\t" << s.requests_completed << '\n';
    oss << "STAT\tslo_violations\t" << s.slo_violations << '\n';
    oss << "STAT\tslo_lateness_sum\t" << s.slo_lateness_sum << '\n';
    oss << "STAT\tdecode_credit_balance\t" << s.decode_credit_balance << '\n';
    oss << "STAT\tdecode_credit_available\t" << s.decode_credit_available << '\n';
    oss << "STAT\tpending_adv_tick\t" << (s.pending_adv_tick ? 1 : 0) << '\n';
    oss << "STAT\tlast_adv_tick\t" << s.last_adv_tick << '\n';
    oss << "STAT\tnext_adv_tick\t" << s.next_adv_tick << '\n';
    oss << "STAT\tmissed_adv_source\t" << s.missed_adv_source << '\n';

    oss << "VECI\tactive_request_ids\t" << join_i32(s.active_request_ids) << '\n';
    oss << "VECI\tcompleted_request_ids\t" << join_i32(s.completed_request_ids) << '\n';
    oss << "VECI\tdropped_request_ids\t" << join_i32(s.dropped_request_ids) << '\n';
    oss << "VECI\tstopped_decode_request_ids\t" << join_i32(s.stopped_decode_request_ids) << '\n';
    oss << "VECI\tviolated_request_ids\t" << join_i32(s.violated_request_ids) << '\n';
    oss << "VECI\tprefill_lateness_finalized_ids\t" << join_i32(s.prefill_lateness_finalized_ids) << '\n';
    oss << "VECD\trecent_arrivals\t" << join_f64(s.recent_arrivals) << '\n';
    for (const auto& ev : s.recent_launches) {
        oss << "LAUNCH\t" << ev.timestamp << '\t' << ev.count << '\t' << ev.prefill_tokens << '\n';
    }

    oss << "MAPI\tdecode_tokens_counted_by_id\t" << join_map_i32(s.decode_tokens_counted_by_id) << '\n';
    oss << "MAPD\tper_request_prefill_lateness_by_id\t" << join_map_i32(s.per_request_prefill_lateness_by_id)
        << '\n';
    oss << "MAPD\tper_request_decode_lateness_by_id\t" << join_map_i32(s.per_request_decode_lateness_by_id)
        << '\n';
    oss << "MAPD\tdecode_next_deadline_by_id\t" << join_map_i32(s.decode_next_deadline_by_id) << '\n';

    return oss.str();
}

SimState VirtualSimulatorGV2::restore_state(const std::string& payload) const {
    SimState out;
    std::istringstream iss(payload);
    std::string line;

    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        const auto cols = split_sv(line, '\t');
        if (cols.empty()) continue;

        if (cols[0] == "SIM" && cols.size() >= 4) {
            out.sim_time = to_double_or(cols[1], out.sim_time);
            out.decision_state_time = to_double_or(cols[2], out.decision_state_time);
            out.next_request_id = to_int_or(cols[3], out.next_request_id);
            continue;
        }

        if (cols[0] == "REQ" && cols.size() >= 20) {
            RequestState r;
            r.request_id = to_int_or(cols[1], r.request_id);
            r.arrived_at = to_double_or(cols[2], r.arrived_at);
            r.queued_at = to_double_or(cols[3], r.queued_at);
            r.num_prefill_tokens = to_int_or(cols[4], r.num_prefill_tokens);
            r.num_processed_prefill_tokens = to_int_or(cols[5], r.num_processed_prefill_tokens);
            r.num_decode_tokens = to_int_or(cols[6], r.num_decode_tokens);
            r.num_processed_decode_tokens = to_int_or(cols[7], r.num_processed_decode_tokens);
            r.prefill_slo_time = to_double_or(cols[8], r.prefill_slo_time);
            r.decode_slo_time = to_double_or(cols[9], r.decode_slo_time);
            r.completion_slo_time = to_double_or(cols[10], r.completion_slo_time);
            r.prefill_deadline = to_double_or(cols[11], r.prefill_deadline);
            r.decode_next_deadline = to_double_or(cols[12], r.decode_next_deadline);
            if (cols.size() >= 22) {
                r.prefill_completed_at = to_double_or(cols[13], r.prefill_completed_at);
                r.completed_at = to_double_or(cols[14], r.completed_at);
                r.prefill_lateness = to_double_or(cols[15], r.prefill_lateness);
                r.decode_lateness = to_double_or(cols[16], r.decode_lateness);
                r.is_prefill_complete = to_bool_or(cols[17], r.is_prefill_complete);
                r.completed = to_bool_or(cols[18], r.completed);
                r.dropped = to_bool_or(cols[19], r.dropped);
                r.stopped_decode = to_bool_or(cols[20], r.stopped_decode);
                r.violated = to_bool_or(cols[21], r.violated);
            } else {
                r.prefill_lateness = to_double_or(cols[13], r.prefill_lateness);
                r.decode_lateness = to_double_or(cols[14], r.decode_lateness);
                r.is_prefill_complete = to_bool_or(cols[15], r.is_prefill_complete);
                r.completed = to_bool_or(cols[16], r.completed);
                r.dropped = to_bool_or(cols[17], r.dropped);
                r.stopped_decode = to_bool_or(cols[18], r.stopped_decode);
                r.violated = to_bool_or(cols[19], r.violated);
            }
            out.requests.push_back(std::move(r));
            continue;
        }

        if (cols[0] == "STAT" && cols.size() >= 3) {
            const std::string& key = cols[1];
            if (key == "requests_generated") out.stats.requests_generated = to_int_or(cols[2], 0);
            else if (key == "requests_completed") out.stats.requests_completed = to_int_or(cols[2], 0);
            else if (key == "slo_violations") out.stats.slo_violations = to_int_or(cols[2], 0);
            else if (key == "slo_lateness_sum") out.stats.slo_lateness_sum = to_double_or(cols[2], 0.0);
            else if (key == "decode_credit_balance") out.stats.decode_credit_balance = to_int_or(cols[2], 0);
            else if (key == "decode_credit_available") out.stats.decode_credit_available = to_int_or(cols[2], 0);
            else if (key == "pending_adv_tick") out.stats.pending_adv_tick = to_bool_or(cols[2], false);
            else if (key == "last_adv_tick") out.stats.last_adv_tick = to_double_or(cols[2], -1.0);
            else if (key == "next_adv_tick") out.stats.next_adv_tick = to_double_or(cols[2], -1.0);
            else if (key == "missed_adv_source") out.stats.missed_adv_source = to_int_or(cols[2], 0);
            continue;
        }

        if (cols[0] == "VECI" && cols.size() >= 3) {
            const std::vector<int> v = parse_i32_vec(cols[2]);
            const std::string& key = cols[1];
            if (key == "active_request_ids") out.stats.active_request_ids = v;
            else if (key == "completed_request_ids") out.stats.completed_request_ids = v;
            else if (key == "dropped_request_ids") out.stats.dropped_request_ids = v;
            else if (key == "stopped_decode_request_ids") out.stats.stopped_decode_request_ids = v;
            else if (key == "violated_request_ids") out.stats.violated_request_ids = v;
            else if (key == "prefill_lateness_finalized_ids") out.stats.prefill_lateness_finalized_ids = v;
            continue;
        }

        if (cols[0] == "VECD" && cols.size() >= 3) {
            if (cols[1] == "recent_arrivals") {
                out.stats.recent_arrivals = parse_f64_vec(cols[2]);
            }
            continue;
        }

        if (cols[0] == "MAPI" && cols.size() >= 3) {
            if (cols[1] == "decode_tokens_counted_by_id") {
                out.stats.decode_tokens_counted_by_id = parse_map_i32<int>(cols[2]);
            }
            continue;
        }

        if (cols[0] == "MAPD" && cols.size() >= 3) {
            if (cols[1] == "per_request_prefill_lateness_by_id") {
                out.stats.per_request_prefill_lateness_by_id = parse_map_i32<double>(cols[2]);
            } else if (cols[1] == "per_request_decode_lateness_by_id") {
                out.stats.per_request_decode_lateness_by_id = parse_map_i32<double>(cols[2]);
            } else if (cols[1] == "decode_next_deadline_by_id") {
                out.stats.decode_next_deadline_by_id = parse_map_i32<double>(cols[2]);
            }
            continue;
        }

        if (cols[0] == "LAUNCH" && cols.size() >= 4) {
            LaunchWindowEntry ev;
            ev.timestamp = to_double_or(cols[1], 0.0);
            ev.count = to_int_or(cols[2], 0);
            ev.prefill_tokens = to_int_or(cols[3], 0);
            out.stats.recent_launches.push_back(std::move(ev));
            continue;
        }
    }

    if (out.stats.decode_credit_available == 0 && out.stats.decode_credit_balance != 0) {
        out.stats.decode_credit_available = std::max(0, out.stats.decode_credit_balance);
    }

    return out;
}

}  // namespace mcts_native_gv2
