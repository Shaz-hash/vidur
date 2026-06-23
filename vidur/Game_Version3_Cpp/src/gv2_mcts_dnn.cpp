#include "gv2_mcts_dnn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace mcts_native_gv2 {
namespace {

template <typename T>
T clampv(T x, T lo, T hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

std::string json_int_vec(const std::vector<int>& xs) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << xs[i];
    }
    oss << "]";
    return oss.str();
}

std::string json_u8_vec(const std::vector<uint8_t>& xs) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << static_cast<int>(xs[i]);
    }
    oss << "]";
    return oss.str();
}

std::string json_f64_vec(const std::vector<double>& xs) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << xs[i];
    }
    oss << "]";
    return oss.str();
}

std::string json_i32_i32_map(const std::unordered_map<int, int>& mp) {
    std::vector<int> keys;
    keys.reserve(mp.size());
    for (const auto& kv : mp) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    std::ostringstream oss;
    oss << "{";
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i > 0) oss << ",";
        const auto it = mp.find(keys[i]);
        if (it == mp.end()) continue;
        oss << "\"" << keys[i] << "\":" << it->second;
    }
    oss << "}";
    return oss.str();
}

std::string json_i32_f64_map(const std::unordered_map<int, double>& mp) {
    std::vector<int> keys;
    keys.reserve(mp.size());
    for (const auto& kv : mp) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "{";
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (i > 0) oss << ",";
        const auto it = mp.find(keys[i]);
        if (it == mp.end()) continue;
        oss << "\"" << keys[i] << "\":" << it->second;
    }
    oss << "}";
    return oss.str();
}

std::string controller_action_to_json(const ControllerAction& a) {
    std::vector<int> ids;
    ids.reserve(a.token_allocations.size());
    for (const auto& kv : a.token_allocations) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());

    auto map_json = [](const std::unordered_map<int, int>& mp) {
        std::vector<int> keys;
        keys.reserve(mp.size());
        for (const auto& kv : mp) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        std::ostringstream oss;
        oss << "{";
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (i > 0) oss << ",";
            const auto it = mp.find(keys[i]);
            if (it == mp.end()) continue;
            oss << "\"" << keys[i] << "\":" << it->second;
        }
        oss << "}";
        return oss.str();
    };

    std::ostringstream oss;
    oss << "{";
    oss << "\"type\":\"controller\",";
    oss << "\"token_budget\":" << a.token_budget << ",";
    oss << "\"selected_request_ids\":" << json_int_vec(a.selected_request_ids) << ",";
    oss << "\"evicted_request_ids\":" << json_int_vec(a.evicted_request_ids) << ",";
    oss << "\"token_allocations\":" << map_json(a.token_allocations) << ",";
    oss << "\"prefill_allocations\":" << map_json(a.prefill_allocations) << ",";
    oss << "\"decode_allocations\":" << map_json(a.decode_allocations) << ",";
    oss << "\"heuristic\":\"" << a.heuristic << "\",";
    oss << "\"strategy\":\"" << a.strategy << "\",";
    oss << "\"mapping\":[" << a.mapping[0] << "," << a.mapping[1] << "," << a.mapping[2] << "]";
    oss << "}";
    return oss.str();
}

std::string controller_action_to_repr(const ControllerAction& a) {
    auto map_repr = [](const std::unordered_map<int, int>& mp) {
        std::vector<int> keys;
        keys.reserve(mp.size());
        for (const auto& kv : mp) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        std::ostringstream oss;
        oss << "{";
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (i > 0) oss << ", ";
            const auto it = mp.find(keys[i]);
            if (it == mp.end()) continue;
            oss << keys[i] << ": " << it->second;
        }
        oss << "}";
        return oss.str();
    };
    std::ostringstream oss;
    oss << "ControllerAction("
        << "token_budget=" << a.token_budget
        << ", selected_request_ids=" << json_int_vec(a.selected_request_ids)
        << ", evicted_request_ids=" << json_int_vec(a.evicted_request_ids)
        << ", token_allocations=" << map_repr(a.token_allocations)
        << ", prefill_allocations=" << map_repr(a.prefill_allocations)
        << ", decode_allocations=" << map_repr(a.decode_allocations)
        << ", heuristic='" << a.heuristic << "'"
        << ", strategy='" << a.strategy << "'";
    if (a.has_mapping) {
        oss << ", mapping=(" << a.mapping[0] << ", " << a.mapping[1] << ", " << a.mapping[2] << ")";
    }
    oss << ")";
    return oss.str();
}

std::string adversary_action_to_json(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "{";
    oss << "\"type\":\"adversary\",";
    oss << "\"requests\":[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& r = a.requests[i];
        oss << "{";
        oss << "\"prefill_tokens\":" << r.prefill_tokens << ",";
        oss << "\"decode_tokens\":" << r.decode_tokens << ",";
        oss << "\"prefill_slo\":" << r.prefill_slo << ",";
        oss << "\"decode_slo\":" << r.decode_slo;
        oss << "}";
    }
    oss << "],";
    oss << "\"stop_decode_ids\":" << json_int_vec(a.stop_decode_ids);
    oss << "}";
    return oss.str();
}

std::string adversary_action_to_repr(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "AdversaryAction(requests=[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ", ";
        const auto& r = a.requests[i];
        oss << "AdversaryRequestSpec(prefill_tokens=" << r.prefill_tokens
            << ", decode_tokens=" << r.decode_tokens
            << ", prefill_slo=" << r.prefill_slo
            << ", decode_slo=" << r.decode_slo
            << ")";
    }
    oss << "], stop_decode_ids=" << json_int_vec(a.stop_decode_ids) << ")";
    return oss.str();
}

std::string adversary_requests_json(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& r = a.requests[i];
        oss << "{"
            << "\"prefill_tokens\":" << r.prefill_tokens << ","
            << "\"decode_tokens\":" << r.decode_tokens << ","
            << "\"prefill_slo\":" << r.prefill_slo << ","
            << "\"decode_slo\":" << r.decode_slo
            << "}";
    }
    oss << "]";
    return oss.str();
}

std::string adversary_prefill_slos_json(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ",";
        oss << a.requests[i].prefill_slo;
    }
    oss << "]";
    return oss.str();
}

std::string adversary_decode_slos_json(const AdversaryAction& a) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "[";
    for (std::size_t i = 0; i < a.requests.size(); ++i) {
        if (i > 0) oss << ",";
        oss << a.requests[i].decode_slo;
    }
    oss << "]";
    return oss.str();
}

double state_cost(const SimState& s) {
    return static_cast<double>(s.stats.slo_violations) + static_cast<double>(s.stats.slo_lateness_sum);
}

double transition_reward(double parent_cost, double child_cost, const SearchInput& in) {
    const double delta = std::max(0.0, static_cast<double>(child_cost) - static_cast<double>(parent_cost));
    const double knee = static_cast<double>(in.reward_knee);
    const double max_penalty = static_cast<double>(in.reward_max_penalty);
    if (max_penalty <= knee) {
        return -std::min(delta, max_penalty);
    }
    const double headroom = max_penalty - knee;
    const double alpha = (std::isfinite(in.reward_tail_alpha) && in.reward_tail_alpha > 0.0)
        ? static_cast<double>(in.reward_tail_alpha)
        : (1.0 / std::max(1e-12, headroom));
    const double penalty = (delta <= knee)
        ? delta
        : knee + headroom * std::tanh(alpha * (delta - knee));
    return -penalty;
}

double time_discount(double t_child, double t_parent, const SearchInput& in) {
    const double gamma = clampv(in.discount_factor, 1e-9, 1.0);
    const double denom = std::max(1e-9, in.prefill_step_time);
    const double dt = std::max(0.0, t_child - t_parent);
    return std::pow(gamma, dt / denom);
}

std::string controller_action_key(const ControllerAction& a) {
    auto sorted_pairs = [](const std::unordered_map<int, int>& mp) {
        std::vector<std::pair<int, int>> out;
        out.reserve(mp.size());
        for (const auto& kv : mp) out.emplace_back(int(kv.first), int(kv.second));
        std::sort(out.begin(), out.end());
        return out;
    };
    auto sorted_ids = [](const std::vector<int>& ids) {
        std::vector<int> out;
        out.reserve(ids.size());
        for (int id : ids) out.push_back(int(id));
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    };

    std::ostringstream oss;
    const auto token_alloc = sorted_pairs(a.token_allocations);
    const auto prefill_alloc = sorted_pairs(a.prefill_allocations);
    const auto decode_alloc = sorted_pairs(a.decode_allocations);
    const auto evicted_ids = sorted_ids(a.evicted_request_ids);

    oss << "alloc:";
    for (std::size_t i = 0; i < token_alloc.size(); ++i) {
        if (i > 0) oss << "|";
        oss << token_alloc[i].first << ":" << token_alloc[i].second;
    }
    oss << ";prefill:";
    for (std::size_t i = 0; i < prefill_alloc.size(); ++i) {
        if (i > 0) oss << "|";
        oss << prefill_alloc[i].first << ":" << prefill_alloc[i].second;
    }
    oss << ";decode:";
    for (std::size_t i = 0; i < decode_alloc.size(); ++i) {
        if (i > 0) oss << "|";
        oss << decode_alloc[i].first << ":" << decode_alloc[i].second;
    }
    oss << ";evicted:";
    for (std::size_t i = 0; i < evicted_ids.size(); ++i) {
        if (i > 0) oss << "|";
        oss << evicted_ids[i];
    }
    return oss.str();
}

std::string adversary_action_key(const AdversaryAction& a) {
    std::vector<std::string> request_specs;
    request_specs.reserve(a.requests.size());
    for (const auto& req : a.requests) {
        std::ostringstream rs;
        rs << std::setprecision(17)
           << int(req.prefill_tokens) << ":"
           << int(req.decode_tokens) << ":"
           << double(req.prefill_slo) << ":"
           << double(req.decode_slo);
        request_specs.push_back(rs.str());
    }
    std::sort(request_specs.begin(), request_specs.end());

    std::vector<int> stop_ids;
    stop_ids.reserve(a.stop_decode_ids.size());
    for (int id : a.stop_decode_ids) stop_ids.push_back(int(id));
    std::sort(stop_ids.begin(), stop_ids.end());
    stop_ids.erase(std::unique(stop_ids.begin(), stop_ids.end()), stop_ids.end());

    std::ostringstream oss;
    oss << "requests:";
    for (std::size_t i = 0; i < request_specs.size(); ++i) {
        if (i > 0) oss << "|";
        oss << request_specs[i];
    }
    oss << ";stop:";
    for (std::size_t i = 0; i < stop_ids.size(); ++i) {
        if (i > 0) oss << "|";
        oss << stop_ids[i];
    }
    return oss.str();
}

struct MinMaxStats {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    void update(double v) {
        minimum = std::min(minimum, v);
        maximum = std::max(maximum, v);
    }
};


class PythonRandomCompat {
public:
    explicit PythonRandomCompat(std::uint64_t seed = 0) { seed_int(seed); }

    void seed_int(std::uint64_t seed) {
        std::vector<std::uint32_t> key;
        if (seed == 0) {
            key.push_back(0u);
        } else {
            while (seed > 0) {
                key.push_back(static_cast<std::uint32_t>(seed & 0xffffffffULL));
                seed >>= 32;
            }
        }
        init_by_array(key);
    }

    int randbelow(int n) {
        if (n <= 0) return 0;
        const int k = bit_length(static_cast<std::uint32_t>(n));
        std::uint32_t r = getrandbits(k);
        while (r >= static_cast<std::uint32_t>(n)) {
            r = getrandbits(k);
        }
        return static_cast<int>(r);
    }

private:
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr std::uint32_t MATRIX_A = 0x9908b0dfU;
    static constexpr std::uint32_t UPPER_MASK = 0x80000000U;
    static constexpr std::uint32_t LOWER_MASK = 0x7fffffffU;

    std::array<std::uint32_t, N> mt_{};
    int index_ = N + 1;

    static int bit_length(std::uint32_t x) {
        int k = 0;
        do {
            ++k;
            x >>= 1;
        } while (x != 0);
        return k;
    }

    void init_genrand(std::uint32_t s) {
        mt_[0] = s;
        for (index_ = 1; index_ < N; ++index_) {
            mt_[index_] = static_cast<std::uint32_t>(
                1812433253U * (mt_[index_ - 1] ^ (mt_[index_ - 1] >> 30)) + static_cast<std::uint32_t>(index_));
        }
    }

    void init_by_array(const std::vector<std::uint32_t>& init_key) {
        init_genrand(19650218U);
        int i = 1;
        int j = 0;
        int k = (N > static_cast<int>(init_key.size())) ? N : static_cast<int>(init_key.size());
        for (; k > 0; --k) {
            mt_[i] = static_cast<std::uint32_t>(
                (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525U)) +
                init_key[static_cast<std::size_t>(j)] + static_cast<std::uint32_t>(j));
            ++i;
            ++j;
            if (i >= N) {
                mt_[0] = mt_[N - 1];
                i = 1;
            }
            if (j >= static_cast<int>(init_key.size())) j = 0;
        }
        for (k = N - 1; k > 0; --k) {
            mt_[i] = static_cast<std::uint32_t>(
                (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941U)) -
                static_cast<std::uint32_t>(i));
            ++i;
            if (i >= N) {
                mt_[0] = mt_[N - 1];
                i = 1;
            }
        }
        mt_[0] = 0x80000000U;
    }

    std::uint32_t genrand_uint32() {
        static constexpr std::uint32_t mag01[2] = {0x0U, MATRIX_A};
        if (index_ >= N) {
            int kk = 0;
            for (; kk < N - M; ++kk) {
                const std::uint32_t y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
                mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            for (; kk < N - 1; ++kk) {
                const std::uint32_t y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
                mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            const std::uint32_t y = (mt_[N - 1] & UPPER_MASK) | (mt_[0] & LOWER_MASK);
            mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
            index_ = 0;
        }

        std::uint32_t y = mt_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }

    std::uint32_t getrandbits(int k) {
        if (k <= 0) return 0u;
        if (k >= 32) return genrand_uint32();
        return genrand_uint32() >> (32 - k);
    }
};

struct TreeNode {
    std::string player;
    int node_id = 0;
    int depth = 0;
    TreeNode* parent = nullptr;

    bool has_parent_action = false;
    bool parent_action_is_controller = false;
    ControllerAction parent_controller_action;
    AdversaryAction parent_adversary_action;
    int parent_action_index = -1;

    double prior = 0.0;
    double reward = 0.0;
    int visits = 0;
    double value_sum = 0.0;
    double state_cost = 0.0;
    double sim_time = 0.0;
    int num_valid_actions = 0;

    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    double edge_discount = 1.0;

    std::vector<ControllerAction> controller_actions_by_index;
    std::vector<AdversaryAction> adversary_actions_by_index;
    std::vector<uint8_t> valid_mask;
    std::vector<int> untried_action_indices;

    bool has_nn_value = false;
    double nn_value_controller = 0.0;
    std::vector<double> nn_priors;
    std::vector<double> nn_priors_after_threshold;
    std::vector<uint8_t> nn_valid_mask;
    std::unordered_map<int, double> action_priors;
    std::unordered_map<int, int> action_alias_to_canonical;
    std::unordered_map<int, std::vector<int>> canonical_to_action_aliases;

    double last_decision_state_time = 0.0;

    bool has_snapshot = false;
    SimState cached_state;

    std::unordered_map<int, std::unique_ptr<TreeNode>> children;

    bool expanded() const { return !children.empty(); }
    double mean_value() const { return (visits > 0) ? (value_sum / static_cast<double>(visits)) : 0.0; }
};

struct SelectionResult {
    int action_index = -1;
    TreeNode* child = nullptr;
};

constexpr double kActionSelectionTieEps = 1e-5;

std::pair<bool, double> is_missed_adv_tick(const SimState& state, const GV2EnvConfig& cfg) {
    const double tick = (cfg.adversary_tick_sec > 0.0) ? cfg.adversary_tick_sec : 0.2;
    double next_tick = state.stats.next_adv_tick;
    if (next_tick < 0.0) {
        const double q = std::floor((state.sim_time + cfg.eps) / tick);
        next_tick = q * tick;
    }
    const bool missed = state.sim_time > (next_tick + cfg.eps);
    return {missed, next_tick};
}

std::unordered_set<int> live_request_ids(const SimState& s) {
    std::unordered_set<int> ids;
    ids.reserve(s.requests.size());
    for (const auto& r : s.requests) {
        if (!r.completed) ids.insert(r.request_id);
    }
    return ids;
}

void copy_root_infer_inputs(SearchOutput* out, const NativeInferInputsGV2& inputs) {
    if (out == nullptr) return;
    out->root_global_features = inputs.global_features;
    out->root_action_mask = inputs.action_mask;
    out->root_prefill_req_features = inputs.prefill_req_features;
    out->root_decode_req_features = inputs.decode_req_features;
    out->root_prefill_req_mask = inputs.prefill_req_mask;
    out->root_decode_req_mask = inputs.decode_req_mask;
    out->root_prefill_req_n = inputs.prefill_req_n;
    out->root_prefill_req_d = inputs.prefill_req_d;
    out->root_decode_req_n = inputs.decode_req_n;
    out->root_decode_req_d = inputs.decode_req_d;
    out->root_req_features = inputs.req_features;
    out->root_req_mask = inputs.req_mask;
    out->root_req_n = inputs.req_n;
    out->root_req_d = inputs.req_d;
}

class SearchRunner {
public:
    SearchRunner(
        const SearchInput& in,
        NativeTorchScriptInferRuntimeGV2& infer_runtime,
        int model_version)
        : in_(in),
          env_(in.env_cfg, in.sim_cfg),
          torch_runtime_(&infer_runtime),
          hgb_runtime_(nullptr),
          model_version_(model_version),
          rng_(static_cast<uint32_t>(std::max(0, in.seed))),
          py_rng_(static_cast<std::uint64_t>(std::max(0, in.seed))),
          next_node_id_(std::max(1, in.root_node_id + 1)) {
        init_owned_env();
    }

    SearchRunner(
        const SearchInput& in,
        const GV2VirtualEnvironment& env,
        NativeTorchScriptInferRuntimeGV2& infer_runtime,
        int model_version)
        : in_(in),
          env_(env),
          torch_runtime_(&infer_runtime),
          hgb_runtime_(nullptr),
          model_version_(model_version),
          rng_(static_cast<uint32_t>(std::max(0, in.seed))),
          py_rng_(static_cast<std::uint64_t>(std::max(0, in.seed))),
          next_node_id_(std::max(1, in.root_node_id + 1)) {}

    SearchRunner(
        const SearchInput& in,
        NewFeatures226HGBRuntime& infer_runtime)
        : in_(in),
          env_(in.env_cfg, in.sim_cfg),
          torch_runtime_(nullptr),
          hgb_runtime_(&infer_runtime),
          controller_prior_runtime_(nullptr),
          adversary_prior_runtime_(nullptr),
          model_version_(0),
          rng_(static_cast<uint32_t>(std::max(0, in.seed))),
          py_rng_(static_cast<std::uint64_t>(std::max(0, in.seed))),
          next_node_id_(std::max(1, in.root_node_id + 1)) {
        init_owned_env();
    }

    SearchRunner(
        const SearchInput& in,
        const GV2VirtualEnvironment& env,
        NewFeatures226HGBRuntime& infer_runtime)
        : in_(in),
          env_(env),
          torch_runtime_(nullptr),
          hgb_runtime_(&infer_runtime),
          controller_prior_runtime_(nullptr),
          adversary_prior_runtime_(nullptr),
          model_version_(0),
          rng_(static_cast<uint32_t>(std::max(0, in.seed))),
          py_rng_(static_cast<std::uint64_t>(std::max(0, in.seed))),
          next_node_id_(std::max(1, in.root_node_id + 1)) {}

    SearchRunner(
        const SearchInput& in,
        const GV2VirtualEnvironment& env,
        NewFeatures226HGBRuntime& infer_runtime,
        NativeHGBModelRuntime& controller_prior_runtime,
        NativeHGBModelRuntime& adversary_prior_runtime)
        : in_(in),
          env_(env),
          torch_runtime_(nullptr),
          hgb_runtime_(&infer_runtime),
          controller_prior_runtime_(&controller_prior_runtime),
          adversary_prior_runtime_(&adversary_prior_runtime),
          model_version_(0),
          rng_(static_cast<uint32_t>(std::max(0, in.seed))),
          py_rng_(static_cast<std::uint64_t>(std::max(0, in.seed))),
          next_node_id_(std::max(1, in.root_node_id + 1)) {}

    void init_owned_env() {
        if (!in_.predictor_csv_path.empty()) {
            (void)env_.load_predictor_csv(in_.predictor_csv_path);
        }
        if (!in_.sim_cfg.prefill_profile_tokens.empty() &&
            in_.sim_cfg.prefill_profile_tokens.size() == in_.sim_cfg.prefill_profile_times.size()) {
            env_.set_prefill_profile(in_.sim_cfg.prefill_profile_tokens, in_.sim_cfg.prefill_profile_times);
        }
    }

    SearchOutput run() {
        if (in_.search_mode == "full_tree" ||
            in_.search_mode == "mcts_full_tree" ||
            in_.search_mode == "tree") {
            return run_full_tree_search();
        }
        return run_gv3_depth_one_search();
    }

private:
    struct ForcedStepLog {
        TreeNode* node = nullptr;
        SimState state_snapshot;
        int num_valid_actions = 0;
        int unique_actions = 0;
    };

    struct QEval {
        double q = 0.0;
        double reward = 0.0;
        double discount = 1.0;
        double bootstrap = 0.0;
        double leaf_cost = 0.0;
        double leaf_time = 0.0;
        int leaf_prefill_count = 0;
        int leaf_decode_count = 0;
        int leaf_decode_credit_balance = 0;
    };

    struct PendingBootstrap {
        int group_id = -1;
        int tie_index = -1;
        SimState leaf_state;
        std::string player_to_act;
        QEval q;
    };

    static std::string next_player(const std::string& player) {
        return (player == "adversary") ? "controller" : "adversary";
    }

    int action_space_size_for_player(const std::string& player) const {
        if (player == "controller") {
            return std::max(1, controller_action_space_size(in_.env_cfg.controller_sampler));
        }
        return std::max(1, adversary_action_space_size(in_.env_cfg.adversary_sampler));
    }

    std::vector<uint8_t> all_true_action_mask(const std::string& player) const {
        return std::vector<uint8_t>(
            static_cast<std::size_t>(action_space_size_for_player(player)),
            uint8_t{1});
    }

    static bool better_max_value(double candidate, double current, int candidate_idx, int current_idx) {
        if (current_idx < 0) return true;
        if (candidate > current + kActionSelectionTieEps) return true;
        if (std::abs(candidate - current) <= kActionSelectionTieEps && candidate_idx < current_idx) {
            return true;
        }
        return false;
    }

    static bool better_min_value(double candidate, double current, int candidate_idx, int current_idx) {
        if (current_idx < 0) return true;
        if (candidate < current - kActionSelectionTieEps) return true;
        if (std::abs(candidate - current) <= kActionSelectionTieEps && candidate_idx < current_idx) {
            return true;
        }
        return false;
    }

    static bool better_for_player(
        const std::string& player,
        double candidate,
        double current,
        int candidate_idx,
        int current_idx) {
        return (player == "controller")
            ? better_max_value(candidate, current, candidate_idx, current_idx)
            : better_min_value(candidate, current, candidate_idx, current_idx);
    }

    bool model_bootstrap_enabled() const {
        if (!in_.use_model_bootstrap) return false;
        if (hgb_runtime_ != nullptr) return hgb_runtime_->loaded();
        return torch_runtime_ != nullptr && model_version_ > 0;
    }

    std::pair<double, std::vector<double>> infer_value_and_priors(
        const SimState& state,
        const std::string& player,
        const std::vector<uint8_t>& action_mask) {
        if (hgb_runtime_ != nullptr) {
            const double value = hgb_runtime_->infer_value(state, &env_.virtual_simulator(), -1);
            return {value, {}};
        }
        if (torch_runtime_ == nullptr || model_version_ <= 0) {
            return {0.0, {}};
        }
        NativeInferInputsGV2 infer_inputs = build_infer_inputs(state, player, action_mask);
        return torch_runtime_->infer_from_inputs(infer_inputs, player, model_version_);
    }

    std::unordered_map<int, double> uniform_policy_priors_plain(
        const std::vector<int>& canonical_indices) const {
        std::unordered_map<int, double> priors;
        if (canonical_indices.empty()) return priors;
        const double p = 1.0 / static_cast<double>(canonical_indices.size());
        for (int idx : canonical_indices) priors[idx] = p;
        return priors;
    }

    std::vector<double> softmax_scores_plain(
        const std::vector<double>& scores,
        double temperature) const {
        std::vector<double> out;
        if (scores.empty()) return out;

        temperature = std::max(temperature, 1e-8);
        double max_scaled = -std::numeric_limits<double>::infinity();
        std::vector<double> scaled;
        scaled.reserve(scores.size());
        for (double x : scores) {
            const double s = x / temperature;
            scaled.push_back(s);
            max_scaled = std::max(max_scaled, s);
        }

        double z = 0.0;
        out.reserve(scores.size());
        for (double s : scaled) {
            const double e = std::exp(s - max_scaled);
            out.push_back(e);
            z += e;
        }

        if (!(z > 0.0) || !std::isfinite(z)) {
            const double p = 1.0 / static_cast<double>(scores.size());
            std::fill(out.begin(), out.end(), p);
            return out;
        }

        for (double& x : out) x /= z;
        return out;
    }

    void apply_root_dirichlet_noise_plain(TreeNode* node) {
        if (node == nullptr || node->parent != nullptr) return;
        if (!in_.root_dirichlet_noise_enabled) return;
        if (node->action_priors.size() <= 1) return;

        const double alpha = in_.root_dirichlet_alpha;
        double eps = in_.root_dirichlet_epsilon;
        if (alpha <= 0.0 || eps <= 0.0) return;
        eps = clampv(eps, 0.0, 1.0);

        std::vector<int> keys;
        keys.reserve(node->action_priors.size());
        for (const auto& kv : node->action_priors) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());

        std::gamma_distribution<double> gamma(alpha, 1.0);
        std::vector<double> noise(keys.size(), 0.0);
        double noise_sum = 0.0;
        for (double& x : noise) {
            x = gamma(rng_);
            noise_sum += x;
        }

        if (!(noise_sum > 0.0) || !std::isfinite(noise_sum)) {
            const double u = 1.0 / static_cast<double>(noise.size());
            std::fill(noise.begin(), noise.end(), u);
        } else {
            for (double& x : noise) x /= noise_sum;
        }

        double z = 0.0;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            const int idx = keys[i];
            const double old_p = clampv(node->action_priors[idx], 0.0, 1.0);
            const double mixed = (1.0 - eps) * old_p + eps * noise[i];
            node->action_priors[idx] = mixed;
            z += mixed;
        }

        if (z > 0.0 && std::isfinite(z)) {
            for (int idx : keys) node->action_priors[idx] /= z;
        }
    }

    void refresh_policy_prior_vectors_plain(
        TreeNode* node,
        const std::vector<double>& model_priors) const {
        if (node == nullptr) return;
        const int n = static_cast<int>(node->valid_mask.size());

        node->nn_priors.assign(static_cast<std::size_t>(std::max(0, n)), 0.0);
        if (static_cast<int>(model_priors.size()) == n) {
            for (int i = 0; i < n; ++i) {
                node->nn_priors[static_cast<std::size_t>(i)] = model_priors[static_cast<std::size_t>(i)];
            }
        }

        node->nn_priors_after_threshold.assign(static_cast<std::size_t>(std::max(0, n)), 0.0);
        for (const auto& kv : node->action_priors) {
            const int canon_idx = kv.first;
            const double prior = kv.second;
            std::vector<int> aliases;
            const auto ita = node->canonical_to_action_aliases.find(canon_idx);
            if (ita != node->canonical_to_action_aliases.end() && !ita->second.empty()) {
                aliases = ita->second;
            } else {
                aliases.push_back(canon_idx);
            }
            const double share = prior / static_cast<double>(std::max<std::size_t>(1, aliases.size()));
            for (int alias : aliases) {
                if (alias >= 0 && alias < n) {
                    node->nn_priors_after_threshold[static_cast<std::size_t>(alias)] = share;
                    if (node->nn_priors[static_cast<std::size_t>(alias)] == 0.0) {
                        node->nn_priors[static_cast<std::size_t>(alias)] = share;
                    }
                }
            }
        }
    }

    static double norm01_feature(double value, double denom) {
        if (!(denom > 0.0) || !std::isfinite(denom)) return 0.0;
        return clampv(value / denom, 0.0, 1.0);
    }

    static bool has_id(const std::vector<int>& ids, int rid) {
        return std::find(ids.begin(), ids.end(), rid) != ids.end();
    }

    static double get_lateness(const std::unordered_map<int, double>& by_id, int rid) {
        const auto it = by_id.find(rid);
        if (it == by_id.end() || !std::isfinite(it->second)) return 0.0;
        return std::max(0.0, it->second);
    }

    std::vector<const RequestState*> active_prefill_requests_for_policy(const SimState& state) const {
        std::unordered_set<int> active_ids;
        active_ids.reserve(state.stats.active_request_ids.size());
        for (int rid : state.stats.active_request_ids) active_ids.insert(rid);

        std::vector<const RequestState*> out;
        out.reserve(state.requests.size());
        for (const RequestState& r : state.requests) {
            if (!active_ids.empty() && active_ids.find(r.request_id) == active_ids.end()) continue;
            if (r.completed) continue;
            if (!r.prefill_done() && r.remaining_prefill() > 0) out.push_back(&r);
        }
        std::sort(out.begin(), out.end(), [](const RequestState* a, const RequestState* b) {
            return a->request_id < b->request_id;
        });
        return out;
    }

    std::vector<const RequestState*> active_decode_requests_for_policy(const SimState& state) const {
        std::unordered_set<int> active_ids;
        active_ids.reserve(state.stats.active_request_ids.size());
        for (int rid : state.stats.active_request_ids) active_ids.insert(rid);

        std::vector<const RequestState*> out;
        out.reserve(state.requests.size());
        for (const RequestState& r : state.requests) {
            if (!active_ids.empty() && active_ids.find(r.request_id) == active_ids.end()) continue;
            if (r.completed) continue;
            if (r.prefill_done() && r.remaining_decode() > 0) out.push_back(&r);
        }
        std::sort(out.begin(), out.end(), [](const RequestState* a, const RequestState* b) {
            return a->request_id < b->request_id;
        });
        return out;
    }

    double prefill_lateness_for_policy(const SimState& state, const RequestState& r) const {
        double late = get_lateness(state.stats.per_request_prefill_lateness_by_id, r.request_id);
        if (late <= 0.0) {
            late = std::max(0.0, state.sim_time - (r.arrived_at + r.prefill_slo_time));
        }
        return late;
    }

    std::vector<float> controller_action_features_for_policy(
        const SimState& state,
        const ControllerAction& action) const {
        constexpr double kMaxPrefillActionAlloc = 4096.0;
        constexpr double kMaxDecodeRequests = 100.0;
        constexpr int kMaxPrefillSlots = 7;

        const auto prefill_reqs = active_prefill_requests_for_policy(state);
        const auto decode_reqs = active_decode_requests_for_policy(state);
        std::unordered_set<int> prefill_ids;
        std::unordered_set<int> decode_ids;
        prefill_ids.reserve(prefill_reqs.size());
        decode_ids.reserve(decode_reqs.size());
        for (const RequestState* r : prefill_reqs) prefill_ids.insert(r->request_id);
        for (const RequestState* r : decode_reqs) decode_ids.insert(r->request_id);

        std::unordered_set<int> evicted;
        evicted.reserve(action.evicted_request_ids.size());
        for (int rid : action.evicted_request_ids) evicted.insert(rid);

        int evicted_prefill = 0;
        int evicted_decode = 0;
        int evicted_decode_late = 0;
        int evicted_prefill_late = 0;
        int evicted_prefill_missed = 0;
        for (int rid : evicted) {
            if (prefill_ids.find(rid) != prefill_ids.end()) {
                ++evicted_prefill;
                const RequestState* req = nullptr;
                for (const RequestState* r : prefill_reqs) {
                    if (r->request_id == rid) {
                        req = r;
                        break;
                    }
                }
                if (req != nullptr) {
                    if (prefill_lateness_for_policy(state, *req) > 0.5) ++evicted_prefill_late;
                    if (state.sim_time > req->arrived_at + req->prefill_slo_time) ++evicted_prefill_missed;
                }
            }
            if (decode_ids.find(rid) != decode_ids.end()) {
                ++evicted_decode;
                if (get_lateness(state.stats.per_request_decode_lateness_by_id, rid) > 0.5) {
                    ++evicted_decode_late;
                }
            }
        }

        int total_prefill_alloc = 0;
        for (const auto& kv : action.prefill_allocations) total_prefill_alloc += std::max(0, kv.second);
        int total_decode_alloc = 0;
        for (const auto& kv : action.decode_allocations) total_decode_alloc += std::max(0, kv.second);

        int highest_prefill_late = -1;
        double highest_prefill_value = -1.0;
        for (const RequestState* r : prefill_reqs) {
            const double late = prefill_lateness_for_policy(state, *r);
            if (highest_prefill_late < 0 ||
                late > highest_prefill_value ||
                (std::abs(late - highest_prefill_value) <= 1e-12 && r->request_id < highest_prefill_late)) {
                highest_prefill_late = r->request_id;
                highest_prefill_value = late;
            }
        }

        int highest_decode_late = -1;
        double highest_decode_value = -1.0;
        for (const RequestState* r : decode_reqs) {
            const double late = get_lateness(state.stats.per_request_decode_lateness_by_id, r->request_id);
            if (highest_decode_late < 0 ||
                late > highest_decode_value ||
                (std::abs(late - highest_decode_value) <= 1e-12 && r->request_id < highest_decode_late)) {
                highest_decode_late = r->request_id;
                highest_decode_value = late;
            }
        }

        std::vector<float> out;
        out.reserve(43);
        out.push_back(static_cast<float>(norm01_feature(total_prefill_alloc, kMaxPrefillActionAlloc)));
        out.push_back(static_cast<float>(norm01_feature(total_decode_alloc, kMaxDecodeRequests)));
        out.push_back(static_cast<float>(norm01_feature(static_cast<double>(action.prefill_allocations.size()), kMaxPrefillSlots)));
        out.push_back(static_cast<float>(norm01_feature(static_cast<double>(action.decode_allocations.size()), kMaxDecodeRequests)));
        out.push_back(static_cast<float>(norm01_feature(static_cast<double>(evicted_prefill), std::max(1.0, static_cast<double>(prefill_reqs.size())))));
        out.push_back(static_cast<float>(norm01_feature(static_cast<double>(evicted_decode), kMaxDecodeRequests)));
        out.push_back(total_prefill_alloc > 0 ? 1.0f : 0.0f);
        out.push_back(total_decode_alloc > 0 ? 1.0f : 0.0f);
        out.push_back(evicted.empty() ? 0.0f : 1.0f);
        out.push_back((total_prefill_alloc == 0 && total_decode_alloc == 0 && evicted.empty()) ? 1.0f : 0.0f);
        out.push_back(static_cast<float>(norm01_feature(static_cast<double>(evicted_decode_late), kMaxDecodeRequests)));
        out.push_back(static_cast<float>(norm01_feature(static_cast<double>(evicted_prefill_late), kMaxPrefillSlots)));
        out.push_back(static_cast<float>(norm01_feature(static_cast<double>(evicted_prefill_missed), kMaxPrefillSlots)));
        out.push_back((highest_prefill_late >= 0 && evicted.find(highest_prefill_late) != evicted.end()) ? 1.0f : 0.0f);
        out.push_back((highest_decode_late >= 0 && evicted.find(highest_decode_late) != evicted.end()) ? 1.0f : 0.0f);

        struct SlotItem {
            int rid = -1;
            int remaining = 0;
            double slack = 0.0;
            bool violated = false;
        };
        std::vector<SlotItem> slots;
        slots.reserve(prefill_reqs.size());
        for (const RequestState* r : prefill_reqs) {
            const double deadline = r->arrived_at + r->prefill_slo_time;
            slots.push_back(SlotItem{
                r->request_id,
                std::max(0, r->remaining_prefill()),
                std::max(0.0, deadline - state.sim_time),
                has_id(state.stats.violated_request_ids, r->request_id),
            });
        }
        const bool all_violated = !slots.empty() &&
            std::all_of(slots.begin(), slots.end(), [](const SlotItem& x) { return x.violated; });
        if (all_violated) {
            std::sort(slots.begin(), slots.end(), [](const SlotItem& a, const SlotItem& b) {
                return a.rid < b.rid;
            });
        } else {
            std::sort(slots.begin(), slots.end(), [](const SlotItem& a, const SlotItem& b) {
                if (std::abs(a.slack - b.slack) > 1e-12) return a.slack < b.slack;
                return a.rid < b.rid;
            });
        }

        for (int slot = 0; slot < kMaxPrefillSlots; ++slot) {
            const bool has_slot = slot < static_cast<int>(slots.size());
            const int rid = has_slot ? slots[static_cast<std::size_t>(slot)].rid : -1;
            const int remaining = has_slot ? std::max(1, slots[static_cast<std::size_t>(slot)].remaining) : 1;
            const auto alloc_it = action.prefill_allocations.find(rid);
            const int alloc = (alloc_it == action.prefill_allocations.end()) ? 0 : std::max(0, alloc_it->second);
            out.push_back(alloc > 0 ? 1.0f : 0.0f);
            out.push_back(static_cast<float>(norm01_feature(static_cast<double>(alloc), kMaxPrefillActionAlloc)));
            out.push_back(static_cast<float>(norm01_feature(static_cast<double>(alloc), static_cast<double>(remaining))));
            out.push_back((rid >= 0 && evicted.find(rid) != evicted.end()) ? 1.0f : 0.0f);
        }
        return out;
    }

    std::vector<float> adversary_action_features_for_policy(
        const AdversaryAction& action,
        int canon_action_index) const {
        constexpr double kMaxLaunchWindow = 7.0;
        constexpr double kMaxPrefillTokens = 4096.0;
        constexpr int kStopRuleCount = 5;

        double avg_prefill = 0.0;
        if (!action.requests.empty()) {
            for (const auto& req : action.requests) avg_prefill += static_cast<double>(req.prefill_tokens);
            avg_prefill /= static_cast<double>(action.requests.size());
        }
        const int stop_idx = ((canon_action_index % kStopRuleCount) + kStopRuleCount) % kStopRuleCount;

        std::vector<float> out;
        out.reserve(7);
        out.push_back(static_cast<float>(norm01_feature(static_cast<double>(action.requests.size()), kMaxLaunchWindow)));
        out.push_back(static_cast<float>(norm01_feature(avg_prefill, kMaxPrefillTokens)));
        for (int i = 0; i < kStopRuleCount; ++i) {
            out.push_back(i == stop_idx ? 1.0f : 0.0f);
        }
        return out;
    }

    bool hgb_policy_priors_available(const std::string& player) const {
        if (hgb_runtime_ == nullptr || !hgb_runtime_->loaded()) return false;
        if (player == "controller") {
            return controller_prior_runtime_ != nullptr && controller_prior_runtime_->loaded();
        }
        return adversary_prior_runtime_ != nullptr && adversary_prior_runtime_->loaded();
    }

    std::vector<double> hgb_policy_scores_plain(
        const TreeNode* node,
        const SimState& state,
        const std::vector<int>& canonical_indices) const {
        std::vector<double> scores;
        if (node == nullptr || canonical_indices.empty() || !hgb_policy_priors_available(node->player)) {
            return scores;
        }

        const NativeHGBModelRuntime* prior_runtime =
            (node->player == "controller") ? controller_prior_runtime_ : adversary_prior_runtime_;
        if (prior_runtime == nullptr || !prior_runtime->loaded()) return scores;

        const std::vector<float> state_features = hgb_runtime_->build_features(
            state,
            &env_.virtual_simulator(),
            -1,
            nullptr);
        const int row_dim = prior_runtime->feature_dim();
        const int action_dim = row_dim - static_cast<int>(state_features.size());
        if (row_dim <= 0 || action_dim <= 0) return scores;

        const int num_rows = static_cast<int>(canonical_indices.size());
        std::vector<float> flat_rows(
            static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(row_dim),
            0.0f);
        std::vector<uint8_t> valid_row(static_cast<std::size_t>(num_rows), 1);

        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
            const int canon_idx = canonical_indices[static_cast<std::size_t>(row_idx)];
            const std::size_t row_offset =
                static_cast<std::size_t>(row_idx) * static_cast<std::size_t>(row_dim);
            std::copy(state_features.begin(), state_features.end(), flat_rows.begin() + row_offset);

            std::vector<float> action_features;
            if (node->player == "controller") {
                if (canon_idx < 0 || canon_idx >= static_cast<int>(node->controller_actions_by_index.size())) {
                    valid_row[static_cast<std::size_t>(row_idx)] = 0;
                    continue;
                }
                action_features = controller_action_features_for_policy(
                    state,
                    node->controller_actions_by_index[static_cast<std::size_t>(canon_idx)]);
            } else {
                if (canon_idx < 0 || canon_idx >= static_cast<int>(node->adversary_actions_by_index.size())) {
                    valid_row[static_cast<std::size_t>(row_idx)] = 0;
                    continue;
                }
                action_features = adversary_action_features_for_policy(
                    node->adversary_actions_by_index[static_cast<std::size_t>(canon_idx)],
                    canon_idx);
            }

            if (static_cast<int>(action_features.size()) != action_dim) {
                throw std::runtime_error(
                    "native HGB policy action feature dimension mismatch: expected " +
                    std::to_string(action_dim) + ", got " + std::to_string(action_features.size()));
            }
            std::copy(
                action_features.begin(),
                action_features.end(),
                flat_rows.begin() + row_offset + state_features.size());
        }

        scores = prior_runtime->predict_raw_batch_flat(flat_rows, num_rows, row_dim);
        for (std::size_t i = 0; i < valid_row.size() && i < scores.size(); ++i) {
            if (!valid_row[i]) scores[i] = 0.0;
        }
        return scores;
    }

    void compute_policy_priors_plain(
        TreeNode* node,
        const SimState& state,
        const std::vector<int>& canonical_indices) {
        if (node == nullptr) return;

        node->action_priors = uniform_policy_priors_plain(canonical_indices);
        std::vector<double> model_priors;

        if (in_.use_policy_prior && hgb_policy_priors_available(node->player)) {
            const std::vector<double> scores = hgb_policy_scores_plain(node, state, canonical_indices);
            const std::vector<double> probs = softmax_scores_plain(scores, in_.policy_prior_temperature);
            if (probs.size() == canonical_indices.size()) {
                const double min_prob = std::max(0.0, in_.prior_min_prob);
                double z = 0.0;
                for (std::size_t i = 0; i < canonical_indices.size(); ++i) {
                    const double p = std::max(min_prob, probs[i]);
                    node->action_priors[canonical_indices[i]] = p;
                    z += p;
                }
                if (z > 0.0 && std::isfinite(z)) {
                    for (int idx : canonical_indices) node->action_priors[idx] /= z;
                } else {
                    node->action_priors = uniform_policy_priors_plain(canonical_indices);
                }

                model_priors.assign(node->valid_mask.size(), 0.0);
                for (int canon_idx : canonical_indices) {
                    double p = plain_action_prior(*node, canon_idx);
                    std::vector<int> aliases;
                    const auto ita = node->canonical_to_action_aliases.find(canon_idx);
                    if (ita != node->canonical_to_action_aliases.end() && !ita->second.empty()) {
                        aliases = ita->second;
                    } else {
                        aliases.push_back(canon_idx);
                    }
                    p /= static_cast<double>(std::max<std::size_t>(1, aliases.size()));
                    for (int alias : aliases) {
                        if (alias >= 0 && alias < static_cast<int>(model_priors.size())) {
                            model_priors[static_cast<std::size_t>(alias)] = p;
                        }
                    }
                }
            }
        } else if (in_.use_policy_prior && torch_runtime_ != nullptr && model_version_ > 0) {
            const auto infer_out = infer_value_and_priors(state, node->player, node->valid_mask);
            node->has_nn_value = true;
            node->nn_value_controller = infer_out.first;
            model_priors = infer_out.second;

            if (static_cast<int>(model_priors.size()) == static_cast<int>(node->valid_mask.size())) {
                std::vector<double> canonical_scores;
                canonical_scores.reserve(canonical_indices.size());
                for (int canon_idx : canonical_indices) {
                    double score = 0.0;
                    std::vector<int> aliases;
                    const auto ita = node->canonical_to_action_aliases.find(canon_idx);
                    if (ita != node->canonical_to_action_aliases.end() && !ita->second.empty()) {
                        aliases = ita->second;
                    } else {
                        aliases.push_back(canon_idx);
                    }
                    for (int alias : aliases) {
                        if (alias >= 0 && alias < static_cast<int>(model_priors.size())) {
                            const double p = model_priors[static_cast<std::size_t>(alias)];
                            if (std::isfinite(p)) score += p;
                        }
                    }
                    canonical_scores.push_back(score);
                }

                const std::vector<double> probs = softmax_scores_plain(
                    canonical_scores,
                    in_.policy_prior_temperature);
                if (probs.size() == canonical_indices.size()) {
                    const double min_prob = std::max(0.0, in_.prior_min_prob);
                    double z = 0.0;
                    for (std::size_t i = 0; i < canonical_indices.size(); ++i) {
                        const double p = std::max(min_prob, probs[i]);
                        node->action_priors[canonical_indices[i]] = p;
                        z += p;
                    }
                    if (z > 0.0 && std::isfinite(z)) {
                        for (int idx : canonical_indices) node->action_priors[idx] /= z;
                    } else {
                        node->action_priors = uniform_policy_priors_plain(canonical_indices);
                    }
                }
            }
        }

        apply_root_dirichlet_noise_plain(node);
        refresh_policy_prior_vectors_plain(node, model_priors);
    }

    std::vector<double> infer_values_for_states(
        const std::vector<SimState>& states,
        const std::string& player,
        const std::vector<uint8_t>& action_mask) {
        if (hgb_runtime_ != nullptr) {
            std::vector<double> out;
            out.reserve(states.size());
            for (const SimState& state : states) {
                out.push_back(hgb_runtime_->infer_value(state, &env_.virtual_simulator(), -1));
            }
            return out;
        }
        if (torch_runtime_ == nullptr || model_version_ <= 0) {
            return std::vector<double>(states.size(), 0.0);
        }
        std::vector<NativeInferInputsGV2> batch;
        batch.reserve(states.size());
        for (const SimState& state : states) {
            batch.push_back(build_infer_inputs(state, player, action_mask));
        }
        return torch_runtime_->infer_values_from_inputs_batch(batch, player, model_version_);
    }

    QEval compose_q_from_state(
        const SimState& leaf_state,
        double parent_cost,
        double parent_time) const {
        QEval out;
        out.leaf_cost = state_cost(leaf_state);
        out.reward = transition_reward(parent_cost, out.leaf_cost, in_);
        out.leaf_time = leaf_state.sim_time;
        for (const auto& req : leaf_state.requests) {
            if (req.prefill_active()) ++out.leaf_prefill_count;
            if (req.decode_active()) ++out.leaf_decode_count;
        }
        out.leaf_decode_credit_balance = leaf_state.stats.decode_credit_balance;
        const double discount_time = (leaf_state.stats.transition_discount_time >= 0.0)
            ? leaf_state.stats.transition_discount_time
            : leaf_state.sim_time;
        out.discount = time_discount(discount_time, parent_time, in_);
        out.bootstrap = 0.0;
        out.q = out.reward;
        return out;
    }

    void apply_batched_bootstrap(std::vector<PendingBootstrap>* pending) {
        if (pending == nullptr || pending->empty()) return;
        if (!model_bootstrap_enabled()) {
            for (PendingBootstrap& item : *pending) {
                item.q.bootstrap = 0.0;
                item.q.q = item.q.reward;
            }
            return;
        }

        constexpr std::size_t kMaxBootstrapBatch = 512;
        for (const std::string player : {"controller", "adversary"}) {
            std::vector<std::size_t> selected;
            selected.reserve(pending->size());
            for (std::size_t i = 0; i < pending->size(); ++i) {
                if ((*pending)[i].player_to_act == player) selected.push_back(i);
            }
            if (selected.empty()) continue;

            const std::vector<uint8_t> mask = all_true_action_mask(player);
            std::size_t pos = 0;
            while (pos < selected.size()) {
                const std::size_t end = std::min(selected.size(), pos + kMaxBootstrapBatch);
                std::vector<SimState> states;
                states.reserve(end - pos);
                const auto t_infer_begin = std::chrono::steady_clock::now();
                for (std::size_t j = pos; j < end; ++j) {
                    const PendingBootstrap& item = (*pending)[selected[j]];
                    states.push_back(item.leaf_state);
                }
                const std::vector<double> values = infer_values_for_states(states, player, mask);
                const auto t_infer_end = std::chrono::steady_clock::now();
                perf_infer_sec_ +=
                    std::chrono::duration_cast<std::chrono::duration<double>>(t_infer_end - t_infer_begin).count();
                perf_infer_calls_ += 1;

                if (values.size() != states.size()) {
                    throw std::runtime_error("batched bootstrap returned wrong number of values");
                }
                for (std::size_t k = 0; k < values.size(); ++k) {
                    PendingBootstrap& item = (*pending)[selected[pos + k]];
                    item.q.bootstrap = values[k];
                    item.q.q = item.q.reward + item.q.discount * item.q.bootstrap;
                }
                pos = end;
            }
        }
    }

    std::vector<int> valid_controller_indices(const SampledActionSet<ControllerAction>& sampled) const {
        std::vector<int> out;
        const int n = static_cast<int>(sampled.actions.size());
        out.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (i >= static_cast<int>(sampled.mask.size())) continue;
            if (!sampled.mask[static_cast<std::size_t>(i)]) continue;
            if (!sampled.actions[static_cast<std::size_t>(i)].valid) continue;
            out.push_back(i);
        }
        return out;
    }

    std::vector<int> valid_adversary_indices(const SampledActionSet<AdversaryAction>& sampled) const {
        std::vector<int> out;
        const int n = static_cast<int>(sampled.actions.size());
        out.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (i >= static_cast<int>(sampled.mask.size())) continue;
            if (!sampled.mask[static_cast<std::size_t>(i)]) continue;
            if (!sampled.actions[static_cast<std::size_t>(i)].valid) continue;
            out.push_back(i);
        }
        return out;
    }

    std::tuple<std::unordered_map<int, int>, std::unordered_map<int, std::vector<int>>, std::vector<int>>
    canonicalize_controller_indices(
        const SampledActionSet<ControllerAction>& sampled,
        const std::vector<int>& valid_indices) const {
        std::unordered_map<int, int> alias_to_canon;
        std::unordered_map<int, std::vector<int>> canon_to_aliases;
        std::vector<int> canonical_indices;
        std::unordered_map<std::string, int> sig_to_canon;

        alias_to_canon.reserve(valid_indices.size());
        canon_to_aliases.reserve(valid_indices.size());
        canonical_indices.reserve(valid_indices.size());
        sig_to_canon.reserve(valid_indices.size());

        for (int idx : valid_indices) {
            const ControllerAction& act = sampled.actions[static_cast<std::size_t>(idx)];
            const std::string sig = controller_action_key(act);
            const auto it = sig_to_canon.find(sig);
            if (it == sig_to_canon.end()) {
                sig_to_canon.emplace(sig, idx);
                alias_to_canon[idx] = idx;
                canon_to_aliases[idx] = {idx};
                canonical_indices.push_back(idx);
            } else {
                alias_to_canon[idx] = it->second;
                canon_to_aliases[it->second].push_back(idx);
            }
        }
        return {std::move(alias_to_canon), std::move(canon_to_aliases), std::move(canonical_indices)};
    }

    std::tuple<std::unordered_map<int, int>, std::unordered_map<int, std::vector<int>>, std::vector<int>>
    canonicalize_adversary_indices(
        const SampledActionSet<AdversaryAction>& sampled,
        const std::vector<int>& valid_indices) const {
        std::unordered_map<int, int> alias_to_canon;
        std::unordered_map<int, std::vector<int>> canon_to_aliases;
        std::vector<int> canonical_indices;
        std::unordered_map<std::string, int> sig_to_canon;
        alias_to_canon.reserve(valid_indices.size());
        canon_to_aliases.reserve(valid_indices.size());
        canonical_indices.reserve(valid_indices.size());
        sig_to_canon.reserve(valid_indices.size());
        for (int idx : valid_indices) {
            const AdversaryAction& act = sampled.actions[static_cast<std::size_t>(idx)];
            const std::string sig = adversary_action_key(act);
            const auto it = sig_to_canon.find(sig);
            if (it == sig_to_canon.end()) {
                sig_to_canon.emplace(sig, idx);
                alias_to_canon[idx] = idx;
                canon_to_aliases[idx] = {idx};
                canonical_indices.push_back(idx);
            } else {
                alias_to_canon[idx] = it->second;
                canon_to_aliases[it->second].push_back(idx);
            }
        }
        return {std::move(alias_to_canon), std::move(canon_to_aliases), std::move(canonical_indices)};
    }

    QEval evaluate_depth1_controller_action(
        const SimState& decision_state,
        double parent_cost,
        double parent_time,
        const ControllerAction& action) {
        SimState leaf = decision_state;
        env_.apply_controller_action_inplace(leaf, action, false);
        PendingBootstrap item;
        item.group_id = 0;
        item.tie_index = 0;
        item.player_to_act = "adversary";
        item.q = compose_q_from_state(leaf, parent_cost, parent_time);
        item.leaf_state = std::move(leaf);
        std::vector<PendingBootstrap> pending;
        pending.push_back(std::move(item));
        apply_batched_bootstrap(&pending);
        return pending.front().q;
    }

    QEval evaluate_depth1_adversary_action(
        const SimState& decision_state,
        double parent_cost,
        double parent_time,
        const AdversaryAction& action) {
        SimState leaf = decision_state;
        env_.apply_adversary_action_inplace(leaf, action);
        PendingBootstrap item;
        item.group_id = 0;
        item.tie_index = 0;
        item.player_to_act = "controller";
        item.q = compose_q_from_state(leaf, parent_cost, parent_time);
        item.leaf_state = std::move(leaf);
        std::vector<PendingBootstrap> pending;
        pending.push_back(std::move(item));
        apply_batched_bootstrap(&pending);
        return pending.front().q;
    }

    QEval evaluate_adversary_action_two_step(
        const SimState& decision_state,
        double parent_cost,
        double parent_time,
        const AdversaryAction& action) {
        SimState adv_child = decision_state;
        env_.apply_adversary_action_inplace(adv_child, action);

        const auto controller_sampled = env_.sample_controller_actions(adv_child);
        const std::vector<int> valid = valid_controller_indices(controller_sampled);
        if (valid.empty()) {
            PendingBootstrap item;
            item.group_id = 0;
            item.tie_index = -1;
            item.player_to_act = "controller";
            item.q = compose_q_from_state(adv_child, parent_cost, parent_time);
            item.leaf_state = std::move(adv_child);
            std::vector<PendingBootstrap> pending;
            pending.push_back(std::move(item));
            apply_batched_bootstrap(&pending);
            return pending.front().q;
        }

        auto [alias_to_canon, canon_to_aliases, canonical_indices] =
            canonicalize_controller_indices(controller_sampled, valid);
        (void)alias_to_canon;
        (void)canon_to_aliases;

        std::vector<PendingBootstrap> pending;
        pending.reserve(canonical_indices.size());
        for (int cidx : canonical_indices) {
            SimState leaf = adv_child;
            env_.apply_controller_action_inplace(
                leaf,
                controller_sampled.actions[static_cast<std::size_t>(cidx)],
                false);
            PendingBootstrap item;
            item.group_id = 0;
            item.tie_index = cidx;
            item.player_to_act = "adversary";
            item.q = compose_q_from_state(leaf, parent_cost, parent_time);
            item.leaf_state = std::move(leaf);
            pending.push_back(std::move(item));
        }

        apply_batched_bootstrap(&pending);
        const PendingBootstrap* best = nullptr;
        for (const PendingBootstrap& item : pending) {
            if (best == nullptr || better_max_value(item.q.q, best->q.q, item.tie_index, best->tie_index)) {
                best = &item;
            }
        }
        if (best != nullptr) return best->q;

        PendingBootstrap item;
        item.group_id = 0;
        item.tie_index = -1;
        item.player_to_act = "controller";
        item.q = compose_q_from_state(adv_child, parent_cost, parent_time);
        item.leaf_state = std::move(adv_child);
        pending.push_back(std::move(item));
        apply_batched_bootstrap(&pending);
        return pending.back().q;
    }

    void append_controller_child_summary(
        SearchOutput* out,
        int action_index,
        int node_id,
        const ControllerAction& action,
        const QEval& q,
        const std::string& child_player,
        int child_depth,
        bool is_best,
        double prior) const {
        ChildSummary cs;
        cs.index = action_index;
        cs.node_id = node_id;
        cs.depth = child_depth;
        cs.player = child_player;
        cs.prior = prior;
        cs.reward = q.reward;
        cs.edge_discount = q.discount;
        cs.visits = is_best ? 1 : 0;
        cs.value_sum = is_best ? q.q : 0.0;
        cs.sim_time = q.leaf_time;
        cs.state_cost = q.leaf_cost;
        cs.num_valid_actions = 0;
        cs.parent_action_json = controller_action_to_json(action);
        out->children.push_back(std::move(cs));
    }

    void append_adversary_child_summary(
        SearchOutput* out,
        int action_index,
        int node_id,
        const AdversaryAction& action,
        const QEval& q,
        const std::string& child_player,
        int child_depth,
        bool is_best,
        double prior) const {
        ChildSummary cs;
        cs.index = action_index;
        cs.node_id = node_id;
        cs.depth = child_depth;
        cs.player = child_player;
        cs.prior = prior;
        cs.reward = q.reward;
        cs.edge_discount = q.discount;
        cs.visits = is_best ? 1 : 0;
        cs.value_sum = is_best ? q.q : 0.0;
        cs.sim_time = q.leaf_time;
        cs.state_cost = q.leaf_cost;
        cs.num_valid_actions = 0;
        cs.parent_action_json = adversary_action_to_json(action);
        out->children.push_back(std::move(cs));
    }

    SearchOutput make_empty_depth_one_output(
        const SimState& decision_state,
        const std::vector<uint8_t>& valid_mask,
        double root_cost,
        double root_time,
        double total_sec) const {
        SearchOutput out;
        out.contract_version = kGV2NativeContractVersion;
        out.decision_state_time = decision_state.sim_time;
        out.root_state_echo = in_.root_state;
        out.root_visits = 1;
        out.root_value_sum = 0.0;
        out.root_state_cost = root_cost;
        out.root_sim_time = root_time;
        out.root_num_valid_actions = 0;
        out.root_nn_value_controller = 0.0;
        out.root_nn_valid_mask = valid_mask;
        out.root_nn_priors.assign(valid_mask.size(), 0.0);
        out.root_nn_priors_after_threshold.assign(valid_mask.size(), 0.0);
        out.mcts_root_prior.assign(valid_mask.size(), 0.0);
        out.perf["total_sec"] = total_sec;
        out.perf["root_expand_sec"] = total_sec;
        out.perf["infer_total_sec"] = perf_infer_sec_;
        out.perf["infer_calls"] = static_cast<double>(perf_infer_calls_);
        out.perf["gv3_depth_one"] = 1.0;
        return out;
    }

    SearchOutput run_gv3_depth_one_search() {
        using clock = std::chrono::steady_clock;
        const auto t_total_begin = clock::now();

        TreeNode root;
        root.player = in_.root_player;
        root.node_id = in_.root_node_id;
        root.depth = in_.root_depth;
        root.parent = nullptr;

        const auto decision = decision_state_for_node(&root, in_.root_state);
        const SimState decision_state = decision.first;
        const double root_cost = state_cost(decision_state);
        const double root_time = decision_state.sim_time;

        std::vector<uint8_t> valid_mask;
        std::vector<double> action_values;
        std::vector<int> valid_indices;
        std::unordered_map<int, int> alias_to_canon;
        std::unordered_map<int, std::vector<int>> canon_to_aliases;
        std::vector<int> canonical_indices;

        std::vector<ControllerAction> controller_actions;
        std::vector<AdversaryAction> adversary_actions;

        if (root.player == "controller") {
            const auto sampled = env_.sample_controller_actions(decision_state);
            controller_actions = sampled.actions;
            valid_mask = sampled.mask;
            valid_indices = valid_controller_indices(sampled);
            std::tie(alias_to_canon, canon_to_aliases, canonical_indices) =
                canonicalize_controller_indices(sampled, valid_indices);
        } else {
            const auto forbidden = replay_forbidden_stop_ids(&root, decision_state, in_.root_state);
            const auto sampled = env_.sample_adversary_actions(decision_state, forbidden);
            adversary_actions = sampled.actions;
            valid_mask = sampled.mask;
            valid_indices = valid_adversary_indices(sampled);
            std::tie(alias_to_canon, canon_to_aliases, canonical_indices) =
                canonicalize_adversary_indices(sampled, valid_indices);
        }

        NativeInferInputsGV2 root_inputs = build_infer_inputs(decision_state, root.player, valid_mask);

        const int n_actions = static_cast<int>(valid_mask.size());
        action_values.assign(static_cast<std::size_t>(n_actions), -std::numeric_limits<double>::infinity());
        std::vector<double> action_rewards(static_cast<std::size_t>(n_actions), 0.0);
        std::vector<double> action_discounts(static_cast<std::size_t>(n_actions), 0.0);
        std::vector<double> action_bootstraps(static_cast<std::size_t>(n_actions), 0.0);
        std::vector<std::string> action_reprs(static_cast<std::size_t>(n_actions));
        std::vector<int> action_leaf_prefill_counts(static_cast<std::size_t>(n_actions), 0);
        std::vector<int> action_leaf_decode_counts(static_cast<std::size_t>(n_actions), 0);
        std::vector<int> action_leaf_decode_credit_balances(static_cast<std::size_t>(n_actions), 0);
        std::unordered_map<int, QEval> canonical_q;
        canonical_q.reserve(canonical_indices.size());

        if (root.player == "controller") {
            for (int idx : valid_indices) {
                if (idx >= 0 && idx < static_cast<int>(controller_actions.size())) {
                    action_reprs[static_cast<std::size_t>(idx)] =
                        controller_action_to_repr(controller_actions[static_cast<std::size_t>(idx)]);
                }
            }
        } else {
            for (int idx : valid_indices) {
                if (idx >= 0 && idx < static_cast<int>(adversary_actions.size())) {
                    action_reprs[static_cast<std::size_t>(idx)] =
                        adversary_action_to_repr(adversary_actions[static_cast<std::size_t>(idx)]);
                }
            }
        }

        std::vector<PendingBootstrap> pending_bootstrap;
        if (root.player == "controller") {
            pending_bootstrap.reserve(canonical_indices.size());
            for (int cidx : canonical_indices) {
                SimState leaf = decision_state;
                env_.apply_controller_action_inplace(
                    leaf,
                    controller_actions[static_cast<std::size_t>(cidx)],
                    false);

                PendingBootstrap item;
                item.group_id = cidx;
                item.tie_index = cidx;
                item.player_to_act = "adversary";
                item.q = compose_q_from_state(leaf, root_cost, root_time);
                item.leaf_state = std::move(leaf);
                pending_bootstrap.push_back(std::move(item));
            }

            apply_batched_bootstrap(&pending_bootstrap);
            for (const PendingBootstrap& item : pending_bootstrap) {
                canonical_q[item.group_id] = item.q;
            }
        } else {
            std::unordered_map<int, std::vector<std::size_t>> pending_by_adversary_action;
            pending_by_adversary_action.reserve(canonical_indices.size());

            for (int aidx : canonical_indices) {
                SimState adv_child = decision_state;
                env_.apply_adversary_action_inplace(
                    adv_child,
                    adversary_actions[static_cast<std::size_t>(aidx)]);

                const auto controller_sampled = env_.sample_controller_actions(adv_child);
                const std::vector<int> valid = valid_controller_indices(controller_sampled);
                if (valid.empty()) {
                    PendingBootstrap item;
                    item.group_id = aidx;
                    item.tie_index = -1;
                    item.player_to_act = "controller";
                    item.q = compose_q_from_state(adv_child, root_cost, root_time);
                    item.leaf_state = std::move(adv_child);

                    const std::size_t pending_index = pending_bootstrap.size();
                    pending_bootstrap.push_back(std::move(item));
                    pending_by_adversary_action[aidx].push_back(pending_index);
                    continue;
                }

                auto controller_canon = canonicalize_controller_indices(controller_sampled, valid);
                const std::vector<int>& controller_canonical_indices = std::get<2>(controller_canon);
                for (int cidx : controller_canonical_indices) {
                    SimState leaf = adv_child;
                    env_.apply_controller_action_inplace(
                        leaf,
                        controller_sampled.actions[static_cast<std::size_t>(cidx)],
                        false);

                    PendingBootstrap item;
                    item.group_id = aidx;
                    item.tie_index = cidx;
                    item.player_to_act = "adversary";
                    item.q = compose_q_from_state(leaf, root_cost, root_time);
                    item.leaf_state = std::move(leaf);

                    const std::size_t pending_index = pending_bootstrap.size();
                    pending_bootstrap.push_back(std::move(item));
                    pending_by_adversary_action[aidx].push_back(pending_index);
                }
            }

            apply_batched_bootstrap(&pending_bootstrap);
            for (int aidx : canonical_indices) {
                const auto it = pending_by_adversary_action.find(aidx);
                if (it == pending_by_adversary_action.end() || it->second.empty()) {
                    continue;
                }

                const PendingBootstrap* best = nullptr;
                for (std::size_t pending_index : it->second) {
                    const PendingBootstrap& item = pending_bootstrap[pending_index];
                    if (best == nullptr ||
                        better_max_value(item.q.q, best->q.q, item.tie_index, best->tie_index)) {
                        best = &item;
                    }
                }
                if (best != nullptr) {
                    canonical_q[aidx] = best->q;
                }
            }
        }

        for (const auto& kv : alias_to_canon) {
            const int alias = kv.first;
            const int canon = kv.second;
            const auto it = canonical_q.find(canon);
            if (alias >= 0 && alias < n_actions && it != canonical_q.end()) {
                action_values[static_cast<std::size_t>(alias)] = it->second.q;
                action_rewards[static_cast<std::size_t>(alias)] = it->second.reward;
                action_discounts[static_cast<std::size_t>(alias)] = it->second.discount;
                action_bootstraps[static_cast<std::size_t>(alias)] = it->second.bootstrap;
                action_leaf_prefill_counts[static_cast<std::size_t>(alias)] = it->second.leaf_prefill_count;
                action_leaf_decode_counts[static_cast<std::size_t>(alias)] = it->second.leaf_decode_count;
                action_leaf_decode_credit_balances[static_cast<std::size_t>(alias)] =
                    it->second.leaf_decode_credit_balance;
            }
        }

        const auto t_eval_end = clock::now();
        if (valid_indices.empty()) {
            SearchOutput empty = make_empty_depth_one_output(
                decision_state,
                valid_mask,
                root_cost,
                root_time,
                std::chrono::duration_cast<std::chrono::duration<double>>(t_eval_end - t_total_begin).count());
            copy_root_infer_inputs(&empty, root_inputs);
            empty.root_action_values = std::move(action_values);
            empty.root_action_rewards = std::move(action_rewards);
            empty.root_action_discounts = std::move(action_discounts);
            empty.root_action_bootstraps = std::move(action_bootstraps);
            empty.root_action_reprs = std::move(action_reprs);
            empty.root_action_leaf_prefill_counts = std::move(action_leaf_prefill_counts);
            empty.root_action_leaf_decode_counts = std::move(action_leaf_decode_counts);
            empty.root_action_leaf_decode_credit_balances = std::move(action_leaf_decode_credit_balances);
            return empty;
        }

        int best_idx = -1;
        double best_value = (root.player == "controller")
            ? -std::numeric_limits<double>::infinity()
            : std::numeric_limits<double>::infinity();
        for (int idx : valid_indices) {
            const double v = (idx >= 0 && idx < n_actions)
                ? action_values[static_cast<std::size_t>(idx)]
                : ((root.player == "controller")
                    ? -std::numeric_limits<double>::infinity()
                    : std::numeric_limits<double>::infinity());
            if (better_for_player(root.player, v, best_value, idx, best_idx)) {
                best_idx = idx;
                best_value = v;
            }
        }
        if (best_idx < 0) best_value = 0.0;

        SearchOutput out;
        out.contract_version = kGV2NativeContractVersion;
        out.decision_state_time = decision_state.sim_time;
        out.root_state_echo = in_.root_state;
        out.root_visits = 1;
        out.root_value_sum = std::isfinite(best_value) ? best_value : 0.0;
        out.root_state_cost = root_cost;
        out.root_sim_time = root_time;
        out.root_num_valid_actions = static_cast<int>(valid_indices.size());
        out.root_nn_value_controller = out.root_value_sum;
        out.root_nn_valid_mask = valid_mask;
        out.root_nn_priors.assign(valid_mask.size(), 0.0);
        out.root_nn_priors_after_threshold.assign(valid_mask.size(), 0.0);
        copy_root_infer_inputs(&out, root_inputs);
        out.best_action_index = int(best_idx);
        out.root_action_values = action_values;
        out.root_action_rewards = action_rewards;
        out.root_action_discounts = action_discounts;
        out.root_action_bootstraps = action_bootstraps;
        out.root_action_reprs = action_reprs;
        out.root_action_leaf_prefill_counts = action_leaf_prefill_counts;
        out.root_action_leaf_decode_counts = action_leaf_decode_counts;
        out.root_action_leaf_decode_credit_balances = action_leaf_decode_credit_balances;
        out.action_alias_to_canonical = std::move(alias_to_canon);
        out.canonical_to_action_aliases = std::move(canon_to_aliases);
        out.mcts_root_prior.assign(valid_mask.size(), 0.0);
        if (best_idx >= 0 && best_idx < static_cast<int>(out.mcts_root_prior.size())) {
            out.mcts_root_prior[static_cast<std::size_t>(best_idx)] = 1.0;
        }

        int node_id = std::max(1, in_.root_node_id + 1);
        for (int idx : valid_indices) {
            const int canon = out.action_alias_to_canonical.count(idx) ? out.action_alias_to_canonical[idx] : idx;
            const auto itq = canonical_q.find(canon);
            QEval q = (itq == canonical_q.end()) ? QEval{} : itq->second;
            q.q = (idx >= 0 && idx < n_actions) ? action_values[static_cast<std::size_t>(idx)] : q.q;
            const bool is_best = idx == best_idx;
            const double prior = (idx >= 0 && idx < static_cast<int>(out.mcts_root_prior.size()))
                ? out.mcts_root_prior[static_cast<std::size_t>(idx)]
                : 0.0;
            if (root.player == "controller") {
                append_controller_child_summary(
                    &out,
                    idx,
                    node_id++,
                    controller_actions[static_cast<std::size_t>(idx)],
                    q,
                    "adversary",
                    in_.root_depth + 1,
                    is_best,
                    prior);
            } else {
                append_adversary_child_summary(
                    &out,
                    idx,
                    node_id++,
                    adversary_actions[static_cast<std::size_t>(idx)],
                    q,
                    "controller",
                    in_.root_depth + 1,
                    is_best,
                    prior);
            }
        }

        const auto t_total_end = clock::now();
        out.perf["total_sec"] =
            std::chrono::duration_cast<std::chrono::duration<double>>(t_total_end - t_total_begin).count();
        out.perf["root_expand_sec"] =
            std::chrono::duration_cast<std::chrono::duration<double>>(t_eval_end - t_total_begin).count();
        out.perf["infer_total_sec"] = perf_infer_sec_;
        out.perf["infer_calls"] = static_cast<double>(perf_infer_calls_);
        out.perf["gv3_depth_one"] = 1.0;
        out.perf["selection_sec"] = 0.0;
        out.perf["restore_sec"] = 0.0;
        out.perf["forced_chain_sec"] = 0.0;
        out.perf["expand_sec"] = 0.0;
        out.perf["backprop_sec"] = 0.0;
        out.perf["selection_steps"] = 0.0;
        out.perf["forced_steps"] = 0.0;
        return out;
    }


    bool plain_expanded(const TreeNode& node) const {
        return !node.valid_mask.empty() ||
               !node.controller_actions_by_index.empty() ||
               !node.adversary_actions_by_index.empty();
    }

    void ensure_expanded_plain(TreeNode* node, const SimState& state) {
        if (node == nullptr || plain_expanded(*node)) return;

        if (node->player == "controller") {
            const auto sampled = env_.sample_controller_actions(state);
            const auto valid = valid_controller_indices(sampled);
            auto canon = canonicalize_controller_indices(sampled, valid);

            node->controller_actions_by_index = sampled.actions;
            node->adversary_actions_by_index.clear();
            node->valid_mask = sampled.mask;
            node->nn_valid_mask = sampled.mask;
            node->num_valid_actions = static_cast<int>(valid.size());
            node->action_alias_to_canonical = std::move(std::get<0>(canon));
            node->canonical_to_action_aliases = std::move(std::get<1>(canon));
            node->untried_action_indices = std::move(std::get<2>(canon));
            compute_policy_priors_plain(node, state, node->untried_action_indices);
            return;
        }

        const auto sampled = env_.sample_adversary_actions(state);
        const auto valid = valid_adversary_indices(sampled);
        auto canon = canonicalize_adversary_indices(sampled, valid);

        node->adversary_actions_by_index = sampled.actions;
        node->controller_actions_by_index.clear();
        node->valid_mask = sampled.mask;
        node->nn_valid_mask = sampled.mask;
        node->num_valid_actions = static_cast<int>(valid.size());
        node->action_alias_to_canonical = std::move(std::get<0>(canon));
        node->canonical_to_action_aliases = std::move(std::get<1>(canon));
        node->untried_action_indices = std::move(std::get<2>(canon));
        compute_policy_priors_plain(node, state, node->untried_action_indices);
    }

    void store_plain_snapshot(TreeNode* node, const SimState& state) {
        node->cached_state = state;
        node->has_snapshot = true;
        node->state_cost = state_cost(state);
        node->sim_time = state.sim_time;
    }

    double plain_action_prior(const TreeNode& node, int action_idx) const {
        const auto it = node.action_priors.find(action_idx);
        if (it != node.action_priors.end()) {
            return clampv(it->second, 0.0, 1.0);
        }
        const std::size_t n = node.canonical_to_action_aliases.empty()
            ? (node.children.size() + node.untried_action_indices.size())
            : node.canonical_to_action_aliases.size();
        if (n == 0) return 0.0;
        return 1.0 / static_cast<double>(n);
    }

    std::pair<TreeNode*, SimState> expand_one_child_plain(
        TreeNode* node,
        SimState state,
        int requested_action_idx = -1) {
        ensure_expanded_plain(node, state);
        if (node == nullptr || node->untried_action_indices.empty()) {
            throw std::runtime_error("expand_one_child_plain called with no untried actions");
        }

        int action_idx = requested_action_idx;
        auto action_it = node->untried_action_indices.end();

        if (action_idx >= 0) {
            action_it = std::find(
                node->untried_action_indices.begin(),
                node->untried_action_indices.end(),
                action_idx);
            if (action_it == node->untried_action_indices.end()) {
                throw std::runtime_error("expand_one_child_plain requested action is not untried");
            }
        } else if (in_.use_policy_prior) {
            action_it = std::max_element(
                node->untried_action_indices.begin(),
                node->untried_action_indices.end(),
                [&](int a, int b) {
                    const double pa = plain_action_prior(*node, a);
                    const double pb = plain_action_prior(*node, b);
                    if (std::abs(pa - pb) <= kActionSelectionTieEps) return a > b;
                    return pa < pb;
                });
            action_idx = (action_it == node->untried_action_indices.end()) ? -1 : *action_it;
        } else {
            const int pos = py_rng_.randbelow(static_cast<int>(node->untried_action_indices.size()));
            action_it = node->untried_action_indices.begin() + pos;
            action_idx = *action_it;
        }

        if (action_it == node->untried_action_indices.end() || action_idx < 0) {
            throw std::runtime_error("expand_one_child_plain failed to choose an untried action");
        }
        node->untried_action_indices.erase(action_it);

        TreeNode child;
        child.player = next_player(node->player);
        child.node_id = next_node_id_++;
        child.depth = node->depth + 1;
        child.parent = node;
        child.parent_action_index = action_idx;
        child.prior = plain_action_prior(*node, action_idx);
        child.sim_time = node->sim_time;

        if (node->player == "controller") {
            if (action_idx < 0 || action_idx >= static_cast<int>(node->controller_actions_by_index.size())) {
                throw std::runtime_error("expand_one_child_plain controller action index out of range");
            }
            const ControllerAction& action = node->controller_actions_by_index[static_cast<std::size_t>(action_idx)];
            set_child_action_from_controller(&child, action);
            env_.apply_controller_action_inplace(state, action, true);
        } else {
            if (action_idx < 0 || action_idx >= static_cast<int>(node->adversary_actions_by_index.size())) {
                throw std::runtime_error("expand_one_child_plain adversary action index out of range");
            }
            const AdversaryAction& action = node->adversary_actions_by_index[static_cast<std::size_t>(action_idx)];
            set_child_action_from_adversary(&child, action);
            env_.apply_adversary_action_inplace(state, action);
        }

        const double child_cost = state_cost(state);
        child.reward = node->state_cost - child_cost;
        const double final_time = (state.stats.transition_final_time >= 0.0)
            ? state.stats.transition_final_time
            : state.sim_time;
        child.edge_discount = time_discount(final_time, node->sim_time, in_);
        store_plain_snapshot(&child, state);

        auto inserted = node->children.emplace(action_idx, std::make_unique<TreeNode>(std::move(child)));
        return {inserted.first->second.get(), std::move(state)};
    }

    double normalize_plain_child_value_for_selection(const TreeNode& parent, const TreeNode& child) const {
        const double lo = parent.min_value;
        const double hi = parent.max_value;
        if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo + 1e-12) return 0.5;
        double norm = clampv((child.mean_value() - lo) / (hi - lo), 0.0, 1.0);
        if (parent.player == "adversary") norm = 1.0 - norm;
        return norm;
    }

    TreeNode* select_child_plain(TreeNode* node) {
        if (node == nullptr || node->children.empty()) return nullptr;

        const int parent_visits = std::max(1, node->visits);
        const double log_term = std::log(static_cast<double>(parent_visits) + 1.0);
        double best_score = -std::numeric_limits<double>::infinity();
        int best_action = -1;
        TreeNode* best_child = nullptr;

        for (const auto& kv : node->children) {
            const int action_idx = kv.first;
            TreeNode* child = kv.second.get();
            double score = std::numeric_limits<double>::infinity();
            if (child->visits > 0) {
                const double exploit = normalize_plain_child_value_for_selection(*node, *child);
                const double explore = std::sqrt(
                    log_term / (log_term + static_cast<double>(std::max(1, child->visits))));
                score = exploit + std::max(0.0, in_.uct_c) * explore;
            }

            if (best_child == nullptr || score > best_score ||
                (score == best_score && action_idx < best_action)) {
                best_score = score;
                best_action = action_idx;
                best_child = child;
            }
        }
        return best_child;
    }

    struct PlainPuctSelection {
        bool is_child = false;
        int action_index = -1;
        TreeNode* child = nullptr;
    };

    double puct_explore_plain(const TreeNode& parent, int action_idx, int child_visits) const {
        const double prior = plain_action_prior(parent, action_idx);
        const int parent_visits = std::max(1, parent.visits);
        const int visits = std::max(0, child_visits);
        const double explore = prior * std::sqrt(static_cast<double>(parent_visits)) /
            (1.0 + static_cast<double>(visits));
        if (!std::isfinite(explore)) return 0.0;
        return explore;
    }

    double puct_score_child_plain(const TreeNode& parent, int action_idx, const TreeNode& child) const {
        const double exploit = normalize_plain_child_value_for_selection(parent, child);
        const double explore = puct_explore_plain(parent, action_idx, child.visits);
        if (exploit < 0.0 || exploit > 1.0 || explore < 0.0) {
            throw std::runtime_error("PUCT produced unreasonable exploit/explore value");
        }
        double c = in_.puct_c;
        if (!std::isfinite(c) || c == 0.0) c = 1.0;
        return exploit + c * explore;
    }

    double puct_score_untried_plain(const TreeNode& parent, int action_idx) const {
        const double exploit = 0.5;
        const double explore = puct_explore_plain(parent, action_idx, 0);
        if (explore < 0.0) {
            throw std::runtime_error("PUCT produced unreasonable untried explore value");
        }
        double c = in_.puct_c;
        if (!std::isfinite(c) || c == 0.0) c = 1.0;
        return exploit + c * explore;
    }

    PlainPuctSelection puct_select_child_or_untried_plain(TreeNode* node) const {
        if (node == nullptr) {
            throw std::runtime_error("puct_select_child_or_untried_plain called with null node");
        }

        bool have_best = false;
        double best_score = -std::numeric_limits<double>::infinity();
        int best_action = -1;
        PlainPuctSelection best;

        auto consider = [&](double score, bool is_child, int action_idx, TreeNode* child) {
            if (!have_best ||
                score > best_score + kActionSelectionTieEps ||
                (std::abs(score - best_score) <= kActionSelectionTieEps && action_idx < best_action)) {
                have_best = true;
                best_score = score;
                best_action = action_idx;
                best.is_child = is_child;
                best.action_index = action_idx;
                best.child = child;
            }
        };

        for (const auto& kv : node->children) {
            if (kv.second == nullptr) continue;
            consider(
                puct_score_child_plain(*node, kv.first, *kv.second),
                true,
                kv.first,
                kv.second.get());
        }

        for (int action_idx : node->untried_action_indices) {
            consider(
                puct_score_untried_plain(*node, action_idx),
                false,
                action_idx,
                nullptr);
        }

        if (!have_best) {
            throw std::runtime_error("puct_select_child_or_untried_plain called with no candidates");
        }
        return best;
    }

    double rollout_value_plain(const SimState& state, const std::string& player) {
        if (!model_bootstrap_enabled()) return 0.0;

        const std::vector<uint8_t> mask = all_true_action_mask(player);
        const auto t_infer_begin = std::chrono::steady_clock::now();
        auto infer_out = infer_value_and_priors(state, player, mask);
        const auto t_infer_end = std::chrono::steady_clock::now();
        perf_infer_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t_infer_end - t_infer_begin).count();
        perf_infer_calls_ += 1;
        return infer_out.first;
    }

    void update_plain_node_bounds(TreeNode* node, double value) {
        if (node == nullptr) return;
        node->min_value = std::min(node->min_value, value);
        node->max_value = std::max(node->max_value, value);
    }

    void backpropagate_plain(const std::vector<TreeNode*>& path, double leaf_value) {
        double value = leaf_value;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            TreeNode* node = *it;
            if (node->parent != nullptr) {
                value = node->reward + node->edge_discount * value;
            }

            node->visits += 1;
            node->value_sum += value;

            if (node->parent != nullptr) {
                update_plain_node_bounds(node->parent, value);
            }
        }
    }

    SearchOutput run_full_tree_search() {
        using clock = std::chrono::steady_clock;
        const auto t_total_begin = clock::now();

        TreeNode root;
        root.player = in_.root_player;
        root.node_id = in_.root_node_id;
        root.depth = in_.root_depth;
        store_plain_snapshot(&root, in_.root_state);
        next_node_id_ = std::max(next_node_id_, std::max(1, root.node_id + 1));
        ensure_expanded_plain(&root, in_.root_state);

        SearchOutput out;
        out.contract_version = kGV2NativeContractVersion;
        out.decision_state_time = in_.decision_state_time;
        out.root_state_echo = in_.root_state;
        out.root_state_cost = root.state_cost;
        out.root_sim_time = root.sim_time;
        out.root_num_valid_actions = root.num_valid_actions;
        out.root_nn_value_controller = root.has_nn_value ? root.nn_value_controller : 0.0;
        out.root_nn_valid_mask = root.valid_mask;
        out.root_nn_priors = root.nn_priors.empty()
            ? std::vector<double>(root.valid_mask.size(), 0.0)
            : root.nn_priors;
        out.root_nn_priors_after_threshold = root.nn_priors_after_threshold.empty()
            ? std::vector<double>(root.valid_mask.size(), 0.0)
            : root.nn_priors_after_threshold;
        out.action_alias_to_canonical = root.action_alias_to_canonical;
        out.canonical_to_action_aliases = root.canonical_to_action_aliases;
        NativeInferInputsGV2 root_inputs = build_infer_inputs(in_.root_state, root.player, out.root_nn_valid_mask);
        copy_root_infer_inputs(&out, root_inputs);

        if (root.untried_action_indices.empty() && root.children.empty()) {
            out.root_visits = 0;
            out.root_value_sum = 0.0;
            const int n_actions = static_cast<int>(out.root_nn_valid_mask.size());
            out.root_action_values.assign(static_cast<std::size_t>(std::max(0, n_actions)), -std::numeric_limits<double>::infinity());
            return out;
        }

        const int iterations = std::max(1, in_.iterations);
        for (int sim = 0; sim < iterations; ++sim) {
            (void)sim;
            TreeNode* node = &root;
            SimState state = root.cached_state;
            std::vector<TreeNode*> path;
            path.push_back(node);

            if (in_.use_policy_prior) {
                while (true) {
                    if (!plain_expanded(*node)) {
                        ensure_expanded_plain(node, state);
                    }

                    if (node->children.empty() && node->untried_action_indices.empty()) {
                        break;
                    }

                    PlainPuctSelection selected = puct_select_child_or_untried_plain(node);
                    if (selected.is_child) {
                        node = selected.child;
                        if (node == nullptr) break;
                        state = node->cached_state;
                        path.push_back(node);
                        continue;
                    }

                    auto expanded = expand_one_child_plain(node, state, selected.action_index);
                    node = expanded.first;
                    state = std::move(expanded.second);
                    path.push_back(node);
                    break;
                }
            } else {
                while (plain_expanded(*node) && node->untried_action_indices.empty() && !node->children.empty()) {
                    node = select_child_plain(node);
                    if (node == nullptr) break;
                    state = node->cached_state;
                    path.push_back(node);
                }
                if (node == nullptr) continue;

                if (!plain_expanded(*node)) {
                    ensure_expanded_plain(node, state);
                }

                if (!node->untried_action_indices.empty()) {
                    auto expanded = expand_one_child_plain(node, state);
                    node = expanded.first;
                    state = std::move(expanded.second);
                    path.push_back(node);
                }
            }

            if (node == nullptr) continue;
            const double leaf_value = rollout_value_plain(state, node->player);
            backpropagate_plain(path, leaf_value);
        }

        out.root_visits = root.visits;
        out.root_value_sum = root.value_sum;
        out.mcts_root_prior = compute_root_prior(root, out.root_nn_valid_mask);

        int n_actions = static_cast<int>(out.root_nn_valid_mask.size());
        if (n_actions <= 0) n_actions = action_space_size_for_player(root.player);
        out.root_action_values.assign(static_cast<std::size_t>(n_actions), -std::numeric_limits<double>::infinity());
        out.root_action_rewards.assign(static_cast<std::size_t>(n_actions), 0.0);
        out.root_action_discounts.assign(static_cast<std::size_t>(n_actions), 1.0);
        out.root_action_bootstraps.assign(static_cast<std::size_t>(n_actions), 0.0);
        out.root_action_reprs.assign(static_cast<std::size_t>(n_actions), std::string());
        out.root_action_leaf_prefill_counts.assign(static_cast<std::size_t>(n_actions), 0);
        out.root_action_leaf_decode_counts.assign(static_cast<std::size_t>(n_actions), 0);
        out.root_action_leaf_decode_credit_balances.assign(static_cast<std::size_t>(n_actions), 0);

        int best_idx = -1;
        int best_visits = -1;
        const double sign = (root.player == "controller") ? 1.0 : -1.0;
        double best_signed_value = -std::numeric_limits<double>::infinity();

        std::vector<int> child_indices;
        child_indices.reserve(root.children.size());
        for (const auto& kv : root.children) child_indices.push_back(kv.first);
        std::sort(child_indices.begin(), child_indices.end());

        for (int canon_idx : child_indices) {
            auto it = root.children.find(canon_idx);
            if (it == root.children.end() || it->second == nullptr) continue;
            TreeNode& child = *it->second;
            const double q = child.mean_value();

            std::string action_json;
            if (child.has_parent_action) {
                action_json = child.parent_action_is_controller
                    ? controller_action_to_json(child.parent_controller_action)
                    : adversary_action_to_json(child.parent_adversary_action);
            }

            std::vector<int> aliases;
            const auto ita = root.canonical_to_action_aliases.find(canon_idx);
            if (ita != root.canonical_to_action_aliases.end() && !ita->second.empty()) {
                aliases = ita->second;
            } else {
                aliases.push_back(canon_idx);
            }
            for (int alias : aliases) {
                if (alias < 0 || alias >= n_actions) continue;
                out.root_action_values[static_cast<std::size_t>(alias)] = q;
                out.root_action_rewards[static_cast<std::size_t>(alias)] = child.reward;
                out.root_action_discounts[static_cast<std::size_t>(alias)] = child.edge_discount;
                out.root_action_bootstraps[static_cast<std::size_t>(alias)] = child.mean_value();
                out.root_action_reprs[static_cast<std::size_t>(alias)] = action_json;
                out.root_action_leaf_decode_credit_balances[static_cast<std::size_t>(alias)] = child.cached_state.stats.decode_credit_balance;
            }

            ChildSummary cs;
            cs.index = canon_idx;
            cs.node_id = child.node_id;
            cs.depth = child.depth;
            cs.player = child.player;
            cs.prior = child.prior;
            cs.reward = child.reward;
            cs.edge_discount = child.edge_discount;
            cs.visits = child.visits;
            cs.value_sum = child.value_sum;
            cs.sim_time = child.sim_time;
            cs.state_cost = child.state_cost;
            cs.num_valid_actions = child.num_valid_actions;
            cs.parent_action_json = action_json;
            out.children.push_back(std::move(cs));

            const double signed_value = sign * q;
            if (child.visits > best_visits ||
                (child.visits == best_visits && signed_value > best_signed_value) ||
                (child.visits == best_visits && signed_value == best_signed_value && canon_idx < best_idx)) {
                best_idx = canon_idx;
                best_visits = child.visits;
                best_signed_value = signed_value;
            }
        }

        out.best_action_index = best_idx;
        const auto t_total_end = clock::now();
        out.perf["total_sec"] =
            std::chrono::duration_cast<std::chrono::duration<double>>(t_total_end - t_total_begin).count();
        out.perf["root_expand_sec"] = 0.0;
        out.perf["infer_total_sec"] = perf_infer_sec_;
        out.perf["infer_calls"] = static_cast<double>(perf_infer_calls_);
        out.perf["gv3_depth_one"] = 0.0;
        out.perf["selection_sec"] = 0.0;
        out.perf["restore_sec"] = 0.0;
        out.perf["forced_chain_sec"] = 0.0;
        out.perf["expand_sec"] = 0.0;
        out.perf["backprop_sec"] = 0.0;
        out.perf["selection_steps"] = 0.0;
        out.perf["forced_steps"] = 0.0;
        return out;
    }

    const SearchInput& in_;
    GV2VirtualEnvironment env_;
    NativeTorchScriptInferRuntimeGV2* torch_runtime_ = nullptr;
    NewFeatures226HGBRuntime* hgb_runtime_ = nullptr;
    NativeHGBModelRuntime* controller_prior_runtime_ = nullptr;
    NativeHGBModelRuntime* adversary_prior_runtime_ = nullptr;
    int model_version_ = 0;
    std::mt19937 rng_;
    PythonRandomCompat py_rng_;
    int next_node_id_ = 1;
    MinMaxStats min_max_;

    // Perf counters
    double perf_selection_sec_ = 0.0;
    double perf_restore_sec_ = 0.0;
    double perf_forced_sec_ = 0.0;
    double perf_expand_sec_ = 0.0;
    double perf_backprop_sec_ = 0.0;
    double perf_infer_sec_ = 0.0;
    int perf_infer_calls_ = 0;
    int perf_selection_steps_ = 0;
    int perf_forced_steps_ = 0;

    std::string build_adversary_prefill_deadlines_json(
        const AdversaryAction& action,
        const SimState& state) const {
        if (action.requests.empty() || state.requests.empty()) return "{}";

        std::unordered_map<int, const RequestState*> by_id;
        by_id.reserve(state.requests.size());
        std::vector<int> ids;
        ids.reserve(state.requests.size());
        for (const auto& req : state.requests) {
            by_id[req.request_id] = &req;
            ids.push_back(req.request_id);
        }
        std::sort(ids.begin(), ids.end());

        const int k = std::min(
            static_cast<int>(action.requests.size()),
            static_cast<int>(ids.size()));
        if (k <= 0) return "{}";

        std::unordered_map<int, double> deadlines;
        deadlines.reserve(static_cast<std::size_t>(k));
        for (int i = static_cast<int>(ids.size()) - k; i < static_cast<int>(ids.size()); ++i) {
            const int rid = ids[static_cast<std::size_t>(i)];
            const auto it = by_id.find(rid);
            if (it == by_id.end() || it->second == nullptr) continue;
            const RequestState& req = *it->second;
            deadlines[rid] = req.queued_at + req.prefill_slo_time;
        }
        return json_i32_f64_map(deadlines);
    }

    NativeInferInputsGV2 build_infer_inputs(
        const SimState& state,
        const std::string& player,
        const std::vector<uint8_t>& action_mask) const {
        NativeFeatureBuildConfigGV2 feat_cfg = in_.feature_cfg;
        if (in_.root_infer_inputs.prefill_req_n > 0) {
            feat_cfg.n_prefill_req = in_.root_infer_inputs.prefill_req_n;
        }
        if (in_.root_infer_inputs.prefill_req_d > 0) {
            feat_cfg.d_prefill_req = in_.root_infer_inputs.prefill_req_d;
        }
        if (in_.root_infer_inputs.decode_req_n > 0) {
            feat_cfg.n_decode_req = in_.root_infer_inputs.decode_req_n;
        }
        if (in_.root_infer_inputs.decode_req_d > 0) {
            feat_cfg.d_decode_req = in_.root_infer_inputs.decode_req_d;
        }
        if (!in_.root_infer_inputs.global_features.empty()) {
            feat_cfg.d_global = static_cast<int>(in_.root_infer_inputs.global_features.size());
        } else if (!in_.global_features.empty()) {
            feat_cfg.d_global = static_cast<int>(in_.global_features.size());
        }
        feat_cfg.auto_drop_lateness_sec = std::max(1e-9, in_.env_cfg.auto_drop_lateness_sec);
        feat_cfg.recent_launch_count_den = std::max(1.0, static_cast<double>(in_.env_cfg.max_requests_per_launch_window));
        feat_cfg.recent_launch_prefill_den = std::max(1.0, static_cast<double>(in_.env_cfg.prefill_window_cap_tokens));
        if (torch_runtime_ == nullptr) {
            NativeInferInputsGV2 empty;
            empty.action_mask = action_mask;
            return empty;
        }
        return torch_runtime_->build_inputs_from_state(state, player, action_mask, feat_cfg, &in_.root_infer_inputs);
    }

    std::pair<SimState, double> decision_state_for_node(TreeNode* node, const SimState& real_state) const {
        if (node == nullptr) return {real_state, real_state.sim_time};
        if (node->parent == nullptr) return {real_state, real_state.sim_time};
        if (node->player != "adversary") return {real_state, real_state.sim_time};
        if (node->parent->player != "controller") return {real_state, real_state.sim_time};

        const auto missed = is_missed_adv_tick(real_state, env_.cfg());
        if (!missed.first) {
            return {real_state, missed.second};
        }
        const int src = real_state.stats.missed_adv_source;
        if (src == 2) {
            return {real_state, missed.second};
        }
        if (!node->parent->has_snapshot) {
            return {real_state, missed.second};
        }

        SimState ds = node->parent->cached_state;
        if (ds.sim_time + env_.cfg().eps < missed.second) {
            ds.sim_time = missed.second;
        }
        return {std::move(ds), missed.second};
    }

    std::unordered_set<int> replay_forbidden_stop_ids(
        TreeNode* node,
        const SimState& decision_state,
        const SimState& real_state) const {
        std::unordered_set<int> out;
        if (node == nullptr || node->parent == nullptr) return out;
        if (node->player != "adversary") return out;
        if (node->parent->player != "controller") return out;

        const auto missed = is_missed_adv_tick(real_state, env_.cfg());
        if (!missed.first) return out;
        if (real_state.stats.missed_adv_source != 1) return out;

        const auto replay_ids = live_request_ids(decision_state);
        const auto post_ids = live_request_ids(real_state);
        for (int rid : replay_ids) {
            if (post_ids.find(rid) == post_ids.end()) out.insert(rid);
        }
        return out;
    }

    std::optional<double> child_q_controller(const TreeNode& parent, const TreeNode& child) const {
        if (child.visits <= 0) return std::nullopt;
        const bool parent_is_branching =
            (parent.num_valid_actions > 1) || (parent.children.size() > 1);
        const double disc = parent_is_branching
            ? time_discount(child.sim_time, parent.sim_time, in_)
            : 1.0;
        return child.reward + disc * child.mean_value();
    }

    std::pair<std::optional<double>, std::optional<double>> parent_child_q_bounds(
        const TreeNode& parent) const {
        std::optional<double> q_min;
        std::optional<double> q_max;
        for (const auto& kv : parent.children) {
            const auto q = child_q_controller(parent, *kv.second);
            if (!q.has_value()) continue;
            if (!q_min.has_value()) {
                q_min = q;
                q_max = q;
            } else {
                q_min = std::min(*q_min, *q);
                q_max = std::max(*q_max, *q);
            }
        }
        return {q_min, q_max};
    }

    double normalize_local_q(
        double q_controller,
        const std::optional<double>& q_min,
        const std::optional<double>& q_max) const {
        if (!q_min.has_value() || !q_max.has_value()) return 0.0;
        const double den = *q_max - *q_min;
        if (den <= 1e-8) return 0.0;
        return clampv((q_controller - *q_min) / den, 0.0, 1.0);
    }

    double uct_score(
        const TreeNode& parent,
        const TreeNode& child,
        const std::optional<double>& q_min,
        const std::optional<double>& q_max) const {
        if (child.visits <= 0) return std::numeric_limits<double>::infinity();

        double exploit = 0.5;
        const auto q = child_q_controller(parent, child);
        if (q.has_value() && q_min.has_value() && q_max.has_value()) {
            const double den = *q_max - *q_min;
            if (std::isfinite(den) && den > 1e-12) {
                exploit = clampv((*q - *q_min) / den, 0.0, 1.0);
                if (parent.player == "adversary") exploit = 1.0 - exploit;
            }
        }

        const int parent_visits = std::max(1, parent.visits);
        const double log_term = std::log(static_cast<double>(parent_visits) + 1.0);
        const double explore = std::sqrt(
            log_term / (log_term + static_cast<double>(std::max(1, child.visits))));
        return exploit + std::max(0.0, in_.uct_c) * explore;
    }

    void apply_root_dirichlet_noise(TreeNode* root, bool nn_called, int num_valid_actions) {
        if (root == nullptr) return;
        if (!in_.root_dirichlet_noise_enabled) return;
        if (!nn_called) return;
        if (num_valid_actions <= 1) return;
        if (root->children.empty()) return;

        const double alpha = in_.root_dirichlet_alpha;
        double eps = in_.root_dirichlet_epsilon;

        if (alpha <= 0.0 || eps <= 0.0) return;
        eps = clampv(eps, 0.0, 1.0);

        std::vector<int> child_indices;
        child_indices.reserve(root->children.size());
        for (const auto& kv : root->children) child_indices.push_back(kv.first);
        std::sort(child_indices.begin(), child_indices.end());

        const int n = static_cast<int>(child_indices.size());
        if (n <= 1) return;

        std::gamma_distribution<double> gamma(alpha, 1.0);
        std::vector<double> noise_raw(static_cast<std::size_t>(n), 0.0);
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            const double g = gamma(rng_);
            noise_raw[static_cast<std::size_t>(i)] = g;
            s += g;
        }

        std::vector<double> noise(static_cast<std::size_t>(n), 0.0);
        if (s <= 1e-12) {
            const double u = 1.0 / static_cast<double>(n);
            std::fill(noise.begin(), noise.end(), u);
        } else {
            const double inv_s = 1.0 / s;
            for (int i = 0; i < n; ++i) {
                noise[static_cast<std::size_t>(i)] = noise_raw[static_cast<std::size_t>(i)] * inv_s;
            }
        }

        std::unordered_map<int, double> mixed;
        mixed.reserve(child_indices.size());
        for (int j = 0; j < n; ++j) {
            const int idx = child_indices[static_cast<std::size_t>(j)];
            auto it = root->children.find(idx);
            if (it == root->children.end() || it->second == nullptr) continue;
            const double p = std::max(0.0, it->second->prior);
            mixed[idx] = (1.0 - eps) * p + eps * noise[static_cast<std::size_t>(j)];
        }

        double z = 0.0;
        for (const auto& kv : mixed) z += kv.second;

        if (z <= 1e-12) {
            const double u = 1.0 / static_cast<double>(n);
            for (int idx : child_indices) {
                auto it = root->children.find(idx);
                if (it != root->children.end() && it->second != nullptr) {
                    it->second->prior = u;
                }
            }
        } else {
            const double inv_z = 1.0 / z;
            for (int idx : child_indices) {
                auto it = root->children.find(idx);
                if (it != root->children.end() && it->second != nullptr) {
                    it->second->prior = std::max(0.0, mixed[idx]) * inv_z;
                }
            }
        }

        // Keep alias-level debug priors aligned with noisy root child priors.
        if (!root->nn_priors_after_threshold.empty()) {
            const int a = static_cast<int>(root->nn_priors_after_threshold.size());
            std::vector<double> noisy_full(static_cast<std::size_t>(a), 0.0);

            if (!root->canonical_to_action_aliases.empty()) {
                for (const auto& kv : root->canonical_to_action_aliases) {
                    const int canon = kv.first;
                    auto itc = root->children.find(canon);
                    if (itc == root->children.end() || itc->second == nullptr) continue;

                    std::vector<int> alias_ids;
                    alias_ids.reserve(kv.second.size());
                    for (int x : kv.second) {
                        if (x >= 0 && x < a) alias_ids.push_back(x);
                    }
                    if (alias_ids.empty()) continue;

                    const double share = itc->second->prior / static_cast<double>(alias_ids.size());
                    for (int ai : alias_ids) {
                        noisy_full[static_cast<std::size_t>(ai)] = share;
                    }
                }
            } else {
                for (const auto& kv : root->children) {
                    const int idx = kv.first;
                    if (idx >= 0 && idx < a && kv.second != nullptr) {
                        noisy_full[static_cast<std::size_t>(idx)] = kv.second->prior;
                    }
                }
            }

            root->nn_priors_after_threshold = std::move(noisy_full);
        }
    }



    SelectionResult select_child(TreeNode* node) {
        SelectionResult out;
        if (node == nullptr || node->children.empty()) return out;
        if (node->children.size() == 1u) {
            out.action_index = node->children.begin()->first;
            out.child = node->children.begin()->second.get();
            return out;
        }

        const auto bounds = parent_child_q_bounds(*node);
        double best_score = -std::numeric_limits<double>::infinity();
        int best_action = -1;

        for (const auto& kv : node->children) {
            const int action_idx = kv.first;
            const double score = uct_score(*node, *kv.second, bounds.first, bounds.second);
            if (best_action < 0 || score > best_score + 1e-12 ||
                (std::abs(score - best_score) <= 1e-12 && action_idx < best_action)) {
                best_score = score;
                best_action = action_idx;
            }
        }

        out.action_index = best_action;
        const auto it = node->children.find(out.action_index);
        out.child = (it == node->children.end()) ? nullptr : it->second.get();
        return out;
    }

    void set_child_action_from_controller(TreeNode* child, const ControllerAction& action) const {
        child->has_parent_action = true;
        child->parent_action_is_controller = true;
        child->parent_controller_action = action;
    }

    void set_child_action_from_adversary(TreeNode* child, const AdversaryAction& action) const {
        child->has_parent_action = true;
        child->parent_action_is_controller = false;
        child->parent_adversary_action = action;
    }

    std::tuple<double, bool, int> expand_node(TreeNode* node, const SimState& real_state) {
        node->state_cost = state_cost(real_state);
        node->sim_time = real_state.sim_time;

        const auto decision = decision_state_for_node(node, real_state);
        SimState decision_state = decision.first;
        node->last_decision_state_time = decision.second;

        if (node->player == "controller") {
            const auto sampled = env_.sample_controller_actions(decision_state);
            const int n = static_cast<int>(sampled.actions.size());
            std::vector<int> valid;
            valid.reserve(n);
            for (int i = 0; i < n; ++i) {
                if (i >= static_cast<int>(sampled.mask.size()) || !sampled.mask[static_cast<std::size_t>(i)]) continue;
                if (!sampled.actions[static_cast<std::size_t>(i)].valid) continue;
                valid.push_back(i);
            }
            node->num_valid_actions = static_cast<int>(valid.size());
            node->nn_valid_mask = sampled.mask;

            if (valid.empty()) {
                return {0.0, false, 0};
            }
            if (valid.size() == 1u) {
                const int idx = valid[0];
                auto it = node->children.find(idx);
                if (it == node->children.end()) {
                    auto child = std::make_unique<TreeNode>();
                    child->player = "adversary";
                    child->node_id = next_node_id_++;
                    child->depth = node->depth + 1;
                    child->parent = node;
                    child->parent_action_index = idx;
                    child->prior = 1.0;
                    child->reward = 0.0;
                    child->sim_time = node->sim_time;
                    set_child_action_from_controller(child.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                    node->children.emplace(idx, std::move(child));
                }
                return {0.0, false, 1};
            }

            double model_value = 0.0;
            bool nn_called = false;
            std::vector<double> priors(sampled.mask.size(), 0.0);
            if (model_bootstrap_enabled()) {
                const auto t_infer_begin = std::chrono::steady_clock::now();
                auto infer_out = infer_value_and_priors(decision_state, node->player, sampled.mask);
                const auto t_infer_end = std::chrono::steady_clock::now();
                perf_infer_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t_infer_end - t_infer_begin).count();
                perf_infer_calls_ += 1;
                model_value = infer_out.first;
                priors = std::move(infer_out.second);
                if (priors.size() < sampled.mask.size()) priors.resize(sampled.mask.size(), 0.0);
                if (priors.size() > sampled.mask.size()) priors.resize(sampled.mask.size());
                nn_called = true;
            }

            node->has_nn_value = nn_called;
            node->nn_value_controller = model_value;
            node->nn_priors = priors;

            std::unordered_map<int, double> valid_prior_by_idx;
            valid_prior_by_idx.reserve(valid.size());
            const double uniform_prior = 1.0 / static_cast<double>(valid.size());
            for (int idx : valid) {
                valid_prior_by_idx[idx] = uniform_prior;
            }

            std::unordered_map<std::string, int> sig_to_canon;
            std::unordered_map<int, std::vector<int>> canon_to_alias;
            std::unordered_map<int, int> alias_to_canon;
            for (int idx : valid) {
                const auto& act = sampled.actions[static_cast<std::size_t>(idx)];
                const std::string sig = controller_action_key(act);
                auto it = sig_to_canon.find(sig);
                if (it == sig_to_canon.end()) {
                    sig_to_canon.emplace(sig, idx);
                    canon_to_alias[idx] = {idx};
                    alias_to_canon[idx] = idx;
                } else {
                    canon_to_alias[it->second].push_back(idx);
                    alias_to_canon[idx] = it->second;
                }
            }

            node->action_alias_to_canonical = alias_to_canon;
            node->canonical_to_action_aliases = canon_to_alias;

            std::unordered_map<int, double> canonical_prior;
            canonical_prior.reserve(canon_to_alias.size());
            for (const auto& kv : canon_to_alias) {
                double p = 0.0;
                for (int alias : kv.second) {
                    const auto itp = valid_prior_by_idx.find(alias);
                    if (itp != valid_prior_by_idx.end()) p += itp->second;
                }
                canonical_prior[kv.first] = p;
            }
            double s_canon = 0.0;
            for (const auto& kv : canonical_prior) s_canon += kv.second;
            if (s_canon > 0.0) {
                const double inv = 1.0 / s_canon;
                for (auto& kv : canonical_prior) kv.second *= inv;
            } else {
                const double u = 1.0 / static_cast<double>(canonical_prior.size());
                for (auto& kv : canonical_prior) kv.second = u;
            }

            std::vector<double> priors_thr(priors.size(), 0.0);
            for (const auto& kv : valid_prior_by_idx) priors_thr[static_cast<std::size_t>(kv.first)] = kv.second;
            node->nn_priors_after_threshold = std::move(priors_thr);

            for (const auto& kv : canonical_prior) {
                const int idx = kv.first;
                auto it = node->children.find(idx);
                if (it == node->children.end()) {
                    auto child = std::make_unique<TreeNode>();
                    child->player = "adversary";
                    child->node_id = next_node_id_++;
                    child->depth = node->depth + 1;
                    child->parent = node;
                    child->parent_action_index = idx;
                    child->prior = kv.second;
                    child->reward = 0.0;
                    child->sim_time = node->sim_time;
                    set_child_action_from_controller(child.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                    node->children.emplace(idx, std::move(child));
                } else {
                    it->second->prior = kv.second;
                    if (!it->second->has_parent_action) {
                        set_child_action_from_controller(it->second.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                    }
                }
            }
            return {model_value, nn_called, static_cast<int>(valid.size())};
        }

        // Adversary branch
        const auto forbidden = replay_forbidden_stop_ids(node, decision_state, real_state);
        const auto sampled = env_.sample_adversary_actions(decision_state, forbidden);
        const int n = static_cast<int>(sampled.actions.size());
        std::vector<int> valid;
        valid.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (i >= static_cast<int>(sampled.mask.size()) || !sampled.mask[static_cast<std::size_t>(i)]) continue;
            if (!sampled.actions[static_cast<std::size_t>(i)].valid) continue;
            valid.push_back(i);
        }
        node->num_valid_actions = static_cast<int>(valid.size());
        node->nn_valid_mask = sampled.mask;

        if (valid.empty()) return {0.0, false, 0};
        if (valid.size() == 1u) {
            const int idx = valid[0];
            auto it = node->children.find(idx);
            if (it == node->children.end()) {
                auto child = std::make_unique<TreeNode>();
                child->player = "controller";
                child->node_id = next_node_id_++;
                child->depth = node->depth + 1;
                child->parent = node;
                child->parent_action_index = idx;
                child->prior = 1.0;
                child->reward = 0.0;
                child->sim_time = node->sim_time;
                set_child_action_from_adversary(child.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                node->children.emplace(idx, std::move(child));
            }
            return {0.0, false, 1};
        }

        double model_value = 0.0;
        bool nn_called = false;
        std::vector<double> priors(sampled.mask.size(), 0.0);
        if (model_bootstrap_enabled()) {
            const auto t_infer_begin = std::chrono::steady_clock::now();
                auto infer_out = infer_value_and_priors(decision_state, node->player, sampled.mask);
            const auto t_infer_end = std::chrono::steady_clock::now();
            perf_infer_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t_infer_end - t_infer_begin).count();
            perf_infer_calls_ += 1;
            model_value = infer_out.first;
            priors = std::move(infer_out.second);
            if (priors.size() < sampled.mask.size()) priors.resize(sampled.mask.size(), 0.0);
            if (priors.size() > sampled.mask.size()) priors.resize(sampled.mask.size());
            nn_called = true;
        }

        node->has_nn_value = nn_called;
        node->nn_value_controller = model_value;
        node->nn_priors = priors;
        node->action_alias_to_canonical.clear();
        node->canonical_to_action_aliases.clear();

        std::unordered_map<int, double> valid_prior_by_idx;
        valid_prior_by_idx.reserve(valid.size());
        const double uniform_prior = 1.0 / static_cast<double>(valid.size());
        for (int idx : valid) {
            valid_prior_by_idx[idx] = uniform_prior;
        }

        std::vector<double> priors_thr(priors.size(), 0.0);
        for (const auto& kv : valid_prior_by_idx) priors_thr[static_cast<std::size_t>(kv.first)] = kv.second;
        node->nn_priors_after_threshold = std::move(priors_thr);

        for (int idx : valid) {
            auto it = node->children.find(idx);
            if (it == node->children.end()) {
                auto child = std::make_unique<TreeNode>();
                child->player = "controller";
                child->node_id = next_node_id_++;
                child->depth = node->depth + 1;
                child->parent = node;
                child->parent_action_index = idx;
                child->prior = valid_prior_by_idx[idx];
                child->reward = 0.0;
                child->sim_time = node->sim_time;
                set_child_action_from_adversary(child.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                node->children.emplace(idx, std::move(child));
            } else {
                it->second->prior = valid_prior_by_idx[idx];
                if (!it->second->has_parent_action) {
                    set_child_action_from_adversary(it->second.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                }
            }
        }
        return {model_value, nn_called, static_cast<int>(valid.size())};
    }

    SimState restore_state_for_node(TreeNode* node) {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        if (node == nullptr) return in_.root_state;
        if (node->has_snapshot) {
            const auto t1 = clock::now();
            perf_restore_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
            return node->cached_state;
        }
        if (node->parent == nullptr) {
            throw std::runtime_error("restore_state_for_node: root node missing snapshot");
        }

        SimState state = restore_state_for_node(node->parent);
        const double parent_cost = node->parent->state_cost;

        if (!node->has_parent_action) {
            throw std::runtime_error("restore_state_for_node: missing parent action");
        }
        if (node->parent->player == "controller") {
            env_.apply_controller_action_inplace(state, node->parent_controller_action);
        } else {
            env_.apply_adversary_action_inplace(state, node->parent_adversary_action);
        }

        const double child_cost = state_cost(state);
        node->reward = transition_reward(parent_cost, child_cost, in_);
        node->state_cost = child_cost;
        node->sim_time = state.sim_time;
        node->cached_state = state;
        node->has_snapshot = true;

        const auto t1 = clock::now();
        perf_restore_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        return state;
    }

    std::pair<TreeNode*, SimState> advance_through_single_child_chain(
        TreeNode* node,
        SimState state,
        std::vector<TreeNode*>* search_path,
        std::vector<ForcedStepLog>* forced_logs) {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();

        int hops = 0;
        while (hops < std::max(1, in_.max_forced_hops)) {
            const auto decision = decision_state_for_node(node, state);
            SimState decision_state = decision.first;
            node->last_decision_state_time = decision.second;

            if (node->player == "controller") {
                const auto sampled = env_.sample_controller_actions(decision_state);
                std::vector<int> valid;
                for (int i = 0; i < static_cast<int>(sampled.actions.size()); ++i) {
                    if (i >= static_cast<int>(sampled.mask.size()) || !sampled.mask[static_cast<std::size_t>(i)]) continue;
                    if (!sampled.actions[static_cast<std::size_t>(i)].valid) continue;
                    valid.push_back(i);
                }
                node->num_valid_actions = static_cast<int>(valid.size());
                if (valid.size() != 1u) break;

                if (forced_logs != nullptr) {
                    ForcedStepLog item;
                    item.node = node;
                    item.state_snapshot = state;
                    item.num_valid_actions = static_cast<int>(valid.size());
                    item.unique_actions = static_cast<int>(node->children.size());
                    forced_logs->push_back(std::move(item));
                }

                const int idx = valid[0];
                TreeNode* child = nullptr;
                auto it = node->children.find(idx);
                if (it == node->children.end()) {
                    auto c = std::make_unique<TreeNode>();
                    c->player = "adversary";
                    c->node_id = next_node_id_++;
                    c->depth = node->depth + 1;
                    c->parent = node;
                    c->parent_action_index = idx;
                    c->prior = 1.0;
                    c->reward = 0.0;
                    c->sim_time = node->sim_time;
                    set_child_action_from_controller(c.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                    child = c.get();
                    node->children.emplace(idx, std::move(c));
                } else {
                    child = it->second.get();
                    if (!child->has_parent_action) {
                        set_child_action_from_controller(child, sampled.actions[static_cast<std::size_t>(idx)]);
                    }
                }

                const double parent_cost = node->state_cost;
                env_.apply_controller_action_inplace(state, sampled.actions[static_cast<std::size_t>(idx)]);
                const double child_cost = state_cost(state);

                child->reward = transition_reward(parent_cost, child_cost, in_);
                child->state_cost = child_cost;
                child->sim_time = state.sim_time;
                child->cached_state = state;
                child->has_snapshot = true;

                node = child;
                search_path->push_back(node);
                hops += 1;
                perf_forced_steps_ += 1;
                continue;
            }

            const auto forbidden = replay_forbidden_stop_ids(node, decision_state, state);
            const auto sampled = env_.sample_adversary_actions(decision_state, forbidden);
            std::vector<int> valid;
            for (int i = 0; i < static_cast<int>(sampled.actions.size()); ++i) {
                if (i >= static_cast<int>(sampled.mask.size()) || !sampled.mask[static_cast<std::size_t>(i)]) continue;
                if (!sampled.actions[static_cast<std::size_t>(i)].valid) continue;
                valid.push_back(i);
            }
            node->num_valid_actions = static_cast<int>(valid.size());
            if (valid.size() != 1u) break;

            if (forced_logs != nullptr) {
                ForcedStepLog item;
                item.node = node;
                item.state_snapshot = state;
                item.num_valid_actions = static_cast<int>(valid.size());
                item.unique_actions = static_cast<int>(node->children.size());
                forced_logs->push_back(std::move(item));
            }

            const int idx = valid[0];
            TreeNode* child = nullptr;
            auto it = node->children.find(idx);
            if (it == node->children.end()) {
                auto c = std::make_unique<TreeNode>();
                c->player = "controller";
                c->node_id = next_node_id_++;
                c->depth = node->depth + 1;
                c->parent = node;
                c->parent_action_index = idx;
                c->prior = 1.0;
                c->reward = 0.0;
                c->sim_time = node->sim_time;
                set_child_action_from_adversary(c.get(), sampled.actions[static_cast<std::size_t>(idx)]);
                child = c.get();
                node->children.emplace(idx, std::move(c));
            } else {
                child = it->second.get();
                if (!child->has_parent_action) {
                    set_child_action_from_adversary(child, sampled.actions[static_cast<std::size_t>(idx)]);
                }
            }

            const double parent_cost = node->state_cost;
            env_.apply_adversary_action_inplace(state, sampled.actions[static_cast<std::size_t>(idx)]);
            const double child_cost = state_cost(state);

            child->reward = transition_reward(parent_cost, child_cost, in_);
            child->state_cost = child_cost;
            child->sim_time = state.sim_time;
            child->cached_state = state;
            child->has_snapshot = true;

            node = child;
            search_path->push_back(node);
            hops += 1;
            perf_forced_steps_ += 1;
        }

        if (!node->has_snapshot) {
            node->cached_state = state;
            node->has_snapshot = true;
        }

        const auto t1 = clock::now();
        perf_forced_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        return {node, std::move(state)};
    }

    void backpropagate(const std::vector<TreeNode*>& search_path, double leaf_value) {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();

        double value = leaf_value;
        for (auto it = search_path.rbegin(); it != search_path.rend(); ++it) {
            TreeNode* node = *it;
            node->value_sum += value;
            node->visits += 1;

            TreeNode* parent = node->parent;
            if (parent == nullptr) break;

            const bool parent_is_branching =
                (parent->num_valid_actions > 1) || (parent->children.size() > 1);
            const double disc = time_discount(node->sim_time, parent->sim_time, in_);
            const double reward_used = parent_is_branching ? node->reward : 0.0;
            if (parent_is_branching) {
                min_max_.update(reward_used + disc * node->mean_value());
            }
            value = reward_used + disc * value;
        }

        const auto t1 = clock::now();
        perf_backprop_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    }

    std::vector<double> compute_root_prior(
        const TreeNode& root,
        const std::vector<uint8_t>& valid_mask) const {
        const int n = static_cast<int>(valid_mask.size());
        std::vector<double> visit_mass(static_cast<std::size_t>(n), 0.0);

        if (!root.canonical_to_action_aliases.empty()) {
            for (const auto& kv : root.canonical_to_action_aliases) {
                const auto itc = root.children.find(kv.first);
                const double visits = (itc == root.children.end()) ? 0.0 : static_cast<double>(itc->second->visits);
                const std::vector<int>& aliases = kv.second;
                if (aliases.empty()) continue;
                const double share = visits / static_cast<double>(aliases.size());
                for (int a : aliases) {
                    if (a >= 0 && a < n) visit_mass[static_cast<std::size_t>(a)] += share;
                }
            }
        } else {
            for (const auto& kv : root.children) {
                const int idx = kv.first;
                if (idx >= 0 && idx < n) {
                    visit_mass[static_cast<std::size_t>(idx)] = static_cast<double>(kv.second->visits);
                }
            }
        }

        double total = 0.0;
        for (int i = 0; i < n; ++i) {
            if (valid_mask[static_cast<std::size_t>(i)]) total += visit_mass[static_cast<std::size_t>(i)];
        }
        if (total > 0.0) {
            for (int i = 0; i < n; ++i) {
                if (valid_mask[static_cast<std::size_t>(i)]) {
                    visit_mass[static_cast<std::size_t>(i)] /= total;
                } else {
                    visit_mass[static_cast<std::size_t>(i)] = 0.0;
                }
            }
            return visit_mass;
        }

        int valid_count = 0;
        for (int i = 0; i < n; ++i) {
            if (valid_mask[static_cast<std::size_t>(i)]) valid_count += 1;
        }
        if (valid_count <= 0) return visit_mass;
        const double u = 1.0 / static_cast<double>(valid_count);
        for (int i = 0; i < n; ++i) {
            visit_mass[static_cast<std::size_t>(i)] = valid_mask[static_cast<std::size_t>(i)] ? u : 0.0;
        }
        return visit_mass;
    }

    void run_one_simulation(int sim_iteration, TreeNode* root, SearchOutput* out) {
        using clock = std::chrono::steady_clock;

        // 1) Selection
        const auto t_sel_begin = clock::now();
        TreeNode* node = root;
        std::vector<TreeNode*> search_path;
        search_path.push_back(node);
        int root_selected_action = -1;
        int root_selected_child_node_id = -1;

        while (node->expanded()) {
            SelectionResult sel = select_child(node);
            if (sel.child == nullptr) break;
            if (search_path.size() == 1u) {
                root_selected_action = sel.action_index;
                root_selected_child_node_id = sel.child->node_id;
            }
            node = sel.child;
            search_path.push_back(node);
            perf_selection_steps_ += 1;
        }
        const auto t_sel_end = clock::now();
        perf_selection_sec_ +=
            std::chrono::duration_cast<std::chrono::duration<double>>(t_sel_end - t_sel_begin).count();

        // 2) Restore to selected node
        SimState state = restore_state_for_node(node);
        // 3) Forced single-child chain
        std::vector<ForcedStepLog> forced_logs;
        auto forced = advance_through_single_child_chain(node, state, &search_path, &forced_logs);
        TreeNode* leaf_node = forced.first;
        SimState leaf_state = std::move(forced.second);

        // 4) Expand leaf
        const auto t_expand_begin = clock::now();
        const auto leaf_expand = expand_node(leaf_node, leaf_state);
        const auto t_expand_end = clock::now();
        perf_expand_sec_ +=
            std::chrono::duration_cast<std::chrono::duration<double>>(t_expand_end - t_expand_begin).count();

        const double leaf_value = std::get<0>(leaf_expand);
        const bool leaf_nn_called = std::get<1>(leaf_expand);
        const int leaf_num_valid = std::get<2>(leaf_expand);
        const bool parent_multi = (leaf_node->parent == nullptr) || (leaf_node->parent->children.size() > 1u);
        std::string phase = "terminal";
        if (leaf_num_valid == 1) {
            phase = parent_multi ? "single-child" : "trivial-single-child";
        } else if (leaf_num_valid > 1) {
            phase = parent_multi ? "multiple-child" : "trivial-multiple-child";
        }

        // 5) Backprop
        backpropagate(search_path, leaf_value);

        if (!in_.log_events) {
            return;
        }

        // Emit forced-step rows before the main leaf row (matches Python logger semantics).
        for (const auto& flog : forced_logs) {
            TreeNode* n = flog.node;
            if (n == nullptr) continue;
            const SimState& st = flog.state_snapshot;
            const bool parent_multi = (n->parent == nullptr) || (n->parent->children.size() > 1u);
            std::string forced_phase = "terminal";
            if (flog.num_valid_actions == 1) {
                forced_phase = parent_multi ? "single-child" : "trivial-single-child";
            } else if (flog.num_valid_actions > 1) {
                forced_phase = parent_multi ? "multiple-child" : "trivial-multiple-child";
            }

            IterEvent ev;
            ev.sim_iteration = sim_iteration;
            ev.selected_action_index = -1;
            ev.selected_child_node_id = -1;
            ev.action_index = n->parent_action_index;
            ev.leaf_node_id = n->node_id;
            ev.parent_node_id = (n->parent == nullptr) ? -1 : n->parent->node_id;
            ev.leaf_depth = n->depth;
            ev.sim_time_before = st.sim_time;
            ev.sim_time_after = st.sim_time;
            ev.decision_state_time = n->last_decision_state_time;
            ev.leaf_state_cost = n->state_cost;
            ev.prior = n->prior;
            ev.reward = n->reward;
            ev.decode_credit_balance = st.stats.decode_credit_balance;
            ev.num_valid_actions = flog.num_valid_actions;
            ev.unique_actions = flog.unique_actions;
            ev.root_visits_after = root->visits;
            ev.root_value_sum_after = root->value_sum;
            ev.root_mean_value_after = root->mean_value();
            ev.nn_called = false;
            ev.has_nn_value_controller = false;
            ev.nn_value_controller = 0.0;
            ev.player_to_act = n->player;
            ev.player_acted_to_create_this_node =
                (n->parent == nullptr) ? std::string("root_no_parent") : n->parent->player;
            ev.phase = "forced_step:" + forced_phase;
            ev.requests_in_system = static_cast<int>(st.stats.active_request_ids.size());
            ev.requests_generated = st.stats.requests_generated;
            ev.requests_completed = st.stats.requests_completed;
            ev.slo_violations = st.stats.slo_violations;
            ev.total_lateness = st.stats.slo_lateness_sum;
            ev.avg_lateness = (st.stats.slo_violations > 0)
                ? (st.stats.slo_lateness_sum / static_cast<double>(st.stats.slo_violations))
                : 0.0;
            ev.state_pending_adv_tick = st.stats.pending_adv_tick;
            ev.has_state_last_adv_tick = true;
            ev.state_last_adv_tick = st.stats.last_adv_tick;
            ev.active_request_ids_json = json_int_vec(st.stats.active_request_ids);
            ev.waiting_request_ids_json = json_int_vec(st.stats.active_request_ids);
            ev.completed_request_ids_json = json_int_vec(st.stats.completed_request_ids);
            ev.dropped_request_ids_json = json_int_vec(st.stats.dropped_request_ids);
            ev.stopped_decode_request_ids_json = json_int_vec(st.stats.stopped_decode_request_ids);
            ev.violated_request_ids_json = json_int_vec(st.stats.violated_request_ids);
            ev.decode_tokens_counted_by_id_json = json_i32_i32_map(st.stats.decode_tokens_counted_by_id);
            ev.per_request_prefill_lateness_by_id_json = json_i32_f64_map(st.stats.per_request_prefill_lateness_by_id);
            ev.per_request_decode_lateness_by_id_json = json_i32_f64_map(st.stats.per_request_decode_lateness_by_id);

            const std::vector<double> root_prior_iter = compute_root_prior(*root, out->root_nn_valid_mask);
            ev.root_valid_mask_json = json_u8_vec(out->root_nn_valid_mask);
            ev.root_nn_priors_json = "[]";
            ev.root_nn_priors_after_threshold_json = "[]";
            ev.root_mcts_prior_json = json_f64_vec(root_prior_iter);

            if (n->has_parent_action) {
                if (n->parent_action_is_controller) {
                    const auto& a = n->parent_controller_action;
                    ev.action_repr = controller_action_to_repr(a);
                    ev.has_controller_token_budget = true;
                    ev.controller_token_budget = a.token_budget;
                    ev.controller_selected_ids_json = json_int_vec(a.selected_request_ids);
                    ev.controller_allocations_json = json_i32_i32_map(a.token_allocations);
                    ev.controller_prefill_allocations_json = json_i32_i32_map(a.prefill_allocations);
                    ev.controller_decode_allocations_json = json_i32_i32_map(a.decode_allocations);
                    ev.controller_heuristic = a.heuristic;
                    ev.controller_strategy = a.strategy;
                    for (const auto& kv : a.prefill_allocations) ev.controller_prefill_total += kv.second;
                    for (const auto& kv : a.decode_allocations) ev.controller_decode_total += kv.second;
                } else {
                    const auto& a = n->parent_adversary_action;
                    ev.action_repr = adversary_action_to_repr(a);
                    ev.adversary_requests_json = adversary_requests_json(a);
                    ev.adversary_prefill_slos_json = adversary_prefill_slos_json(a);
                    ev.adversary_decode_slos_json = adversary_decode_slos_json(a);
                    ev.adversary_prefill_deadlines_by_id_json = build_adversary_prefill_deadlines_json(a, st);
                }
            }

            out->iter_events.push_back(std::move(ev));
        }

        // Iter event
        IterEvent ev;
        ev.sim_iteration = sim_iteration;
        ev.selected_action_index = root_selected_action;
        ev.selected_child_node_id = root_selected_child_node_id;
        ev.action_index = leaf_node->parent_action_index;
        ev.leaf_node_id = leaf_node->node_id;
        ev.parent_node_id = (leaf_node->parent == nullptr) ? -1 : leaf_node->parent->node_id;
        ev.leaf_depth = leaf_node->depth;
        // Canonical Python iter logger semantics:
        // leaf expand rows use state_snapshot.sim_time as both start/end unless
        // explicitly overridden. Keep native canonical rows aligned so extracted-style
        // reordering remains causal.
        ev.sim_time_before = leaf_state.sim_time;
        ev.sim_time_after = leaf_state.sim_time;
        ev.decision_state_time = leaf_node->last_decision_state_time;
        ev.leaf_state_cost = leaf_node->state_cost;
        ev.prior = leaf_node->prior;
        ev.reward = leaf_node->reward;
        ev.decode_credit_balance = leaf_state.stats.decode_credit_balance;
        ev.num_valid_actions = leaf_num_valid;
        ev.unique_actions = static_cast<int>(leaf_node->children.size());
        ev.root_visits_after = root->visits;
        ev.root_value_sum_after = root->value_sum;
        ev.root_mean_value_after = root->mean_value();
        ev.nn_called = leaf_nn_called;
        ev.has_nn_value_controller = leaf_node->has_nn_value;
        ev.nn_value_controller = leaf_node->nn_value_controller;
        ev.player_to_act = leaf_node->player;
        ev.player_acted_to_create_this_node =
            (leaf_node->parent == nullptr) ? std::string("root_no_parent") : leaf_node->parent->player;
        ev.phase = phase;
        ev.requests_in_system = static_cast<int>(leaf_state.stats.active_request_ids.size());
        ev.requests_generated = leaf_state.stats.requests_generated;
        ev.requests_completed = leaf_state.stats.requests_completed;
        ev.slo_violations = leaf_state.stats.slo_violations;
        ev.total_lateness = leaf_state.stats.slo_lateness_sum;
        ev.avg_lateness = (leaf_state.stats.slo_violations > 0)
            ? (leaf_state.stats.slo_lateness_sum / static_cast<double>(leaf_state.stats.slo_violations))
            : 0.0;
        ev.state_pending_adv_tick = leaf_state.stats.pending_adv_tick;
        ev.has_state_last_adv_tick = true;
        ev.state_last_adv_tick = leaf_state.stats.last_adv_tick;
        ev.active_request_ids_json = json_int_vec(leaf_state.stats.active_request_ids);
        ev.waiting_request_ids_json = json_int_vec(leaf_state.stats.active_request_ids);
        ev.completed_request_ids_json = json_int_vec(leaf_state.stats.completed_request_ids);
        ev.dropped_request_ids_json = json_int_vec(leaf_state.stats.dropped_request_ids);
        ev.stopped_decode_request_ids_json = json_int_vec(leaf_state.stats.stopped_decode_request_ids);
        ev.violated_request_ids_json = json_int_vec(leaf_state.stats.violated_request_ids);
        ev.decode_tokens_counted_by_id_json = json_i32_i32_map(leaf_state.stats.decode_tokens_counted_by_id);
        ev.per_request_prefill_lateness_by_id_json = json_i32_f64_map(leaf_state.stats.per_request_prefill_lateness_by_id);
        ev.per_request_decode_lateness_by_id_json = json_i32_f64_map(leaf_state.stats.per_request_decode_lateness_by_id);

        if (root_selected_action >= 0) {
            const auto it = root->children.find(root_selected_action);
            if (it != root->children.end()) {
                const TreeNode* c = it->second.get();
                ev.selected_child_visits_after = c->visits;
                ev.selected_child_value_sum_after = c->value_sum;
                ev.selected_child_mean_value_after = c->mean_value();
                ev.selected_child_prior = c->prior;
            }
        }

        const std::vector<double> root_prior_iter = compute_root_prior(*root, out->root_nn_valid_mask);
        ev.root_valid_mask_json = json_u8_vec(out->root_nn_valid_mask);
        ev.root_nn_priors_json = json_f64_vec(out->root_nn_priors);
        ev.root_nn_priors_after_threshold_json = json_f64_vec(out->root_nn_priors_after_threshold);
        ev.root_mcts_prior_json = json_f64_vec(root_prior_iter);

        if (leaf_node->has_parent_action) {
            if (leaf_node->parent_action_is_controller) {
                const auto& a = leaf_node->parent_controller_action;
                ev.action_repr = controller_action_to_repr(a);
                ev.has_controller_token_budget = true;
                ev.controller_token_budget = a.token_budget;
                ev.controller_selected_ids_json = json_int_vec(a.selected_request_ids);
                ev.controller_allocations_json = json_i32_i32_map(a.token_allocations);
                ev.controller_prefill_allocations_json = json_i32_i32_map(a.prefill_allocations);
                ev.controller_decode_allocations_json = json_i32_i32_map(a.decode_allocations);
                ev.controller_heuristic = a.heuristic;
                ev.controller_strategy = a.strategy;
                for (const auto& kv : a.prefill_allocations) ev.controller_prefill_total += kv.second;
                for (const auto& kv : a.decode_allocations) ev.controller_decode_total += kv.second;
            } else {
                const auto& a = leaf_node->parent_adversary_action;
                ev.action_repr = adversary_action_to_repr(a);
                ev.adversary_requests_json = adversary_requests_json(a);
                ev.adversary_prefill_slos_json = adversary_prefill_slos_json(a);
                ev.adversary_decode_slos_json = adversary_decode_slos_json(a);
                ev.adversary_prefill_deadlines_by_id_json =
                    build_adversary_prefill_deadlines_json(a, leaf_state);
            }
        }

        out->iter_events.push_back(std::move(ev));
    }
};

}  // namespace

SearchOutput run_search_torchscript(
    const SearchInput& in,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version) {
    SearchRunner runner(in, infer_runtime, model_version);
    return runner.run();
}

SearchOutput run_search_torchscript_with_env(
    const SearchInput& in,
    GV2VirtualEnvironment& env,
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version) {
    SearchRunner runner(in, env, infer_runtime, model_version);
    return runner.run();
}

SearchOutput run_search_hgb226(
    const SearchInput& in,
    NewFeatures226HGBRuntime& infer_runtime) {
    SearchRunner runner(in, infer_runtime);
    return runner.run();
}

SearchOutput run_search_hgb226_with_env(
    const SearchInput& in,
    GV2VirtualEnvironment& env,
    NewFeatures226HGBRuntime& infer_runtime) {
    SearchRunner runner(in, env, infer_runtime);
    return runner.run();
}

SearchOutput run_search_hgb226_value_prior_with_env(
    const SearchInput& in,
    GV2VirtualEnvironment& env,
    NewFeatures226HGBRuntime& infer_runtime,
    NativeHGBModelRuntime& controller_prior_runtime,
    NativeHGBModelRuntime& adversary_prior_runtime) {
    SearchInput prior_in = in;
    prior_in.search_mode = "full_tree";
    prior_in.use_policy_prior = true;
    if (!std::isfinite(prior_in.puct_c) || prior_in.puct_c == 0.0) {
        prior_in.puct_c = 1.0;
    }
    if (!std::isfinite(prior_in.policy_prior_temperature) ||
        prior_in.policy_prior_temperature <= 0.0) {
        prior_in.policy_prior_temperature = 1.0;
    }
    if (!std::isfinite(prior_in.prior_min_prob) || prior_in.prior_min_prob < 0.0) {
        prior_in.prior_min_prob = 1e-8;
    }
    SearchRunner runner(
        prior_in,
        env,
        infer_runtime,
        controller_prior_runtime,
        adversary_prior_runtime);
    return runner.run();
}

}  // namespace mcts_native_gv2
