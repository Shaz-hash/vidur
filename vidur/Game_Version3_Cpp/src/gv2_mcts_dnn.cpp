#include "gv2_mcts_dnn.hpp"
#include "gv2_cross_game_batcher.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace mcts_native_gv2 {
namespace {

std::uint64_t rollout_hash_bytes(
    const void* data,
    std::size_t size,
    std::uint64_t seed) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::uint64_t hash = seed;
    for (std::size_t idx = 0; idx < size; ++idx) {
        hash ^= static_cast<std::uint64_t>(bytes[idx]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t rollout_hash_fast(
    const void* data,
    std::size_t size,
    std::uint64_t seed) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::uint64_t hash =
        seed ^ (static_cast<std::uint64_t>(size) * 0x9e3779b185ebca87ULL);
    while (size >= sizeof(std::uint64_t)) {
        std::uint64_t word = 0;
        std::memcpy(&word, bytes, sizeof(word));
        word ^= word >> 33U;
        word *= 0xff51afd7ed558ccdULL;
        word ^= word >> 33U;
        hash ^= word;
        hash = ((hash << 27U) | (hash >> 37U)) *
            0x3c79ac492ba7b653ULL + 0x1c69b3f74ac4ae35ULL;
        bytes += sizeof(std::uint64_t);
        size -= sizeof(std::uint64_t);
    }
    std::uint64_t tail = 0;
    if (size > 0) std::memcpy(&tail, bytes, size);
    hash ^= tail * 0x9e3779b185ebca87ULL;
    hash ^= hash >> 33U;
    hash *= 0xc2b2ae3d27d4eb4fULL;
    hash ^= hash >> 29U;
    return hash;
}

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

std::uint64_t hash_controller_component(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t hash_controller_allocations(
    const std::unordered_map<int, int>& allocations,
    std::uint64_t salt) {
    std::uint64_t sum = 0;
    std::uint64_t mixed_xor = 0;
    for (const auto& allocation : allocations) {
        const std::uint64_t packed =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(allocation.first)) << 32U) |
            static_cast<std::uint32_t>(allocation.second);
        const std::uint64_t mixed = hash_controller_component(packed ^ salt);
        sum += mixed;
        mixed_xor ^= mixed;
    }
    return hash_controller_component(
        salt ^ static_cast<std::uint64_t>(allocations.size()) ^ sum ^
        (mixed_xor * 0x9e3779b97f4a7c15ULL));
}

std::uint64_t controller_action_hash(const ControllerAction& action) {
    std::uint64_t hash =
        hash_controller_allocations(action.token_allocations, 0x3c79ac492ba7b653ULL) ^
        hash_controller_allocations(action.prefill_allocations, 0x1c69b3f74ac4ae35ULL) ^
        hash_controller_allocations(action.decode_allocations, 0xd6e8feb86659fd93ULL);
    std::uint64_t evicted_sum = 0;
    std::uint64_t evicted_xor = 0;
    std::size_t unique_count = 0;
    for (std::size_t i = 0; i < action.evicted_request_ids.size(); ++i) {
        const int request_id = action.evicted_request_ids[i];
        bool seen = false;
        for (std::size_t j = 0; j < i; ++j) {
            if (action.evicted_request_ids[j] == request_id) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        const std::uint64_t mixed = hash_controller_component(
            static_cast<std::uint32_t>(request_id) ^ 0xa0761d6478bd642fULL);
        evicted_sum += mixed;
        evicted_xor ^= mixed;
        ++unique_count;
    }
    return hash_controller_component(
        hash ^ evicted_sum ^ (evicted_xor * 0xe7037ed1a0b428dbULL) ^
        static_cast<std::uint64_t>(unique_count));
}

bool controller_actions_equivalent(
    const ControllerAction& left,
    const ControllerAction& right) {
    if (left.token_allocations != right.token_allocations ||
        left.prefill_allocations != right.prefill_allocations ||
        left.decode_allocations != right.decode_allocations) {
        return false;
    }
    for (int request_id : left.evicted_request_ids) {
        if (std::find(
                right.evicted_request_ids.begin(),
                right.evicted_request_ids.end(),
                request_id) == right.evicted_request_ids.end()) {
            return false;
        }
    }
    for (int request_id : right.evicted_request_ids) {
        if (std::find(
                left.evicted_request_ids.begin(),
                left.evicted_request_ids.end(),
                request_id) == left.evicted_request_ids.end()) {
            return false;
        }
    }
    return true;
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

    double random_double() {
        const std::uint32_t a = genrand_uint32() >> 5;
        const std::uint32_t b = genrand_uint32() >> 6;
        return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b)) /
            9007199254740992.0;
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
    double rollout_reward_value_sum = 0.0;
    double rollout_bootstrap_value_sum = 0.0;
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
    std::vector<int> rollout_action_original_indices;

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

constexpr double kActionSelectionTieEps = 0.0;

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
            in_.search_mode == "tree" ||
            in_.search_mode == "full_tree_rollout") {
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
            if (in_.cross_game_inference_batcher != nullptr) {
                const std::vector<double> values =
                    in_.cross_game_inference_batcher->infer_values(
                        *hgb_runtime_,
                        {state},
                        {&env_.virtual_simulator()});
                if (values.size() != 1u) {
                    throw std::runtime_error(
                        "cross-game value batch returned wrong result count");
                }
                return {values.front(), {}};
            }
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

        const double fixed_alpha = in_.root_dirichlet_alpha;
        const double total_concentration = in_.root_dirichlet_total_concentration;
        double eps = in_.root_dirichlet_epsilon;
        if ((fixed_alpha <= 0.0 && total_concentration <= 0.0) || eps <= 0.0) return;
        eps = clampv(eps, 0.0, 1.0);

        std::vector<int> keys;
        keys.reserve(node->action_priors.size());
        for (const auto& kv : node->action_priors) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        const double alpha = resolve_root_dirichlet_alpha(
            fixed_alpha,
            total_concentration,
            static_cast<int>(keys.size()));
        if (alpha <= 0.0) return;

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

    struct ControllerPolicyFeatureSlot {
        int rid = -1;
        int remaining = 0;
        double slack = 0.0;
        bool violated = false;
    };

    struct ControllerPolicyFeatureContext {
        std::size_t prefill_count = 0;
        std::vector<int> prefill_ids;
        std::vector<int> decode_ids;
        std::vector<int> prefill_late_ids;
        std::vector<int> prefill_missed_ids;
        std::vector<int> decode_late_ids;
        int highest_prefill_late = -1;
        int highest_decode_late = -1;
        std::vector<ControllerPolicyFeatureSlot> slots;
        std::vector<const RequestState*> prefill_requests;
        std::vector<const RequestState*> decode_requests;
    };

    static bool policy_id_contains(
        const std::vector<int>& sorted_ids,
        int rid) {
        return std::binary_search(sorted_ids.begin(), sorted_ids.end(), rid);
    }

    const ControllerPolicyFeatureContext& build_controller_policy_feature_context(
        const SimState& state) const {
        thread_local ControllerPolicyFeatureContext context;
        context.prefill_count = 0;
        context.prefill_ids.clear();
        context.decode_ids.clear();
        context.prefill_late_ids.clear();
        context.prefill_missed_ids.clear();
        context.decode_late_ids.clear();
        context.highest_prefill_late = -1;
        context.highest_decode_late = -1;
        context.slots.clear();
        context.prefill_requests.clear();
        context.decode_requests.clear();

        const auto& active_ids = state.stats.active_request_ids;
        const bool active_ids_sorted =
            std::is_sorted(active_ids.begin(), active_ids.end());
        const auto is_active = [&](int rid) {
            if (active_ids.empty()) return true;
            if (active_ids_sorted) {
                return std::binary_search(
                    active_ids.begin(), active_ids.end(), rid);
            }
            return std::find(active_ids.begin(), active_ids.end(), rid) !=
                active_ids.end();
        };
        for (const RequestState& request : state.requests) {
            if (!is_active(request.request_id) || request.completed) continue;
            if (!request.prefill_done() && request.remaining_prefill() > 0) {
                context.prefill_requests.push_back(&request);
            } else if (request.prefill_done() && request.remaining_decode() > 0) {
                context.decode_requests.push_back(&request);
            }
        }
        const auto request_id_less = [](
            const RequestState* left,
            const RequestState* right) {
            return left->request_id < right->request_id;
        };
        std::sort(
            context.prefill_requests.begin(),
            context.prefill_requests.end(),
            request_id_less);
        std::sort(
            context.decode_requests.begin(),
            context.decode_requests.end(),
            request_id_less);

        const auto& prefill_reqs = context.prefill_requests;
        const auto& decode_reqs = context.decode_requests;
        context.prefill_count = prefill_reqs.size();
        context.prefill_ids.reserve(prefill_reqs.size());
        context.decode_ids.reserve(decode_reqs.size());
        context.prefill_late_ids.reserve(prefill_reqs.size());
        context.prefill_missed_ids.reserve(prefill_reqs.size());
        context.decode_late_ids.reserve(decode_reqs.size());
        context.slots.reserve(prefill_reqs.size());

        double highest_prefill_value = -1.0;
        for (const RequestState* request : prefill_reqs) {
            const int rid = request->request_id;
            context.prefill_ids.push_back(rid);
            const double late = prefill_lateness_for_policy(state, *request);
            if (late > 0.5) context.prefill_late_ids.push_back(rid);
            if (state.sim_time > request->arrived_at + request->prefill_slo_time) {
                context.prefill_missed_ids.push_back(rid);
            }
            if (context.highest_prefill_late < 0 ||
                late > highest_prefill_value ||
                (std::abs(late - highest_prefill_value) <= 1e-12 &&
                 rid < context.highest_prefill_late)) {
                context.highest_prefill_late = rid;
                highest_prefill_value = late;
            }
            const double deadline =
                request->arrived_at + request->prefill_slo_time;
            context.slots.push_back(ControllerPolicyFeatureSlot{
                rid,
                std::max(0, request->remaining_prefill()),
                std::max(0.0, deadline - state.sim_time),
                has_id(state.stats.violated_request_ids, rid),
            });
        }

        double highest_decode_value = -1.0;
        for (const RequestState* request : decode_reqs) {
            const int rid = request->request_id;
            context.decode_ids.push_back(rid);
            const double late = get_lateness(
                state.stats.per_request_decode_lateness_by_id, rid);
            if (late > 0.5) context.decode_late_ids.push_back(rid);
            if (context.highest_decode_late < 0 ||
                late > highest_decode_value ||
                (std::abs(late - highest_decode_value) <= 1e-12 &&
                 rid < context.highest_decode_late)) {
                context.highest_decode_late = rid;
                highest_decode_value = late;
            }
        }

        const bool all_violated = !context.slots.empty() &&
            std::all_of(
                context.slots.begin(),
                context.slots.end(),
                [](const ControllerPolicyFeatureSlot& item) {
                    return item.violated;
                });
        if (all_violated) {
            std::sort(
                context.slots.begin(),
                context.slots.end(),
                [](const ControllerPolicyFeatureSlot& left,
                   const ControllerPolicyFeatureSlot& right) {
                    return left.rid < right.rid;
                });
        } else {
            std::sort(
                context.slots.begin(),
                context.slots.end(),
                [](const ControllerPolicyFeatureSlot& left,
                   const ControllerPolicyFeatureSlot& right) {
                    if (std::abs(left.slack - right.slack) > 1e-12) {
                        return left.slack < right.slack;
                    }
                    return left.rid < right.rid;
                });
        }
        return context;
    }

    template <typename ControllerActionType>
    void write_controller_action_features_for_policy(
        const ControllerActionType& action,
        const ControllerPolicyFeatureContext& context,
        float* output) const {
        constexpr double kMaxPrefillActionAlloc = 4096.0;
        constexpr double kMaxDecodeRequests = 100.0;
        constexpr int kMaxPrefillSlots = 7;
        constexpr bool kRolloutAction =
            std::is_same_v<ControllerActionType, RolloutControllerAction>;

        thread_local std::vector<int> evicted;
        evicted.assign(
            action.evicted_request_ids.begin(),
            action.evicted_request_ids.end());
        std::sort(evicted.begin(), evicted.end());
        evicted.erase(
            std::unique(evicted.begin(), evicted.end()), evicted.end());

        int evicted_prefill = 0;
        int evicted_decode = 0;
        int evicted_decode_late = 0;
        int evicted_prefill_late = 0;
        int evicted_prefill_missed = 0;
        for (int rid : evicted) {
            if (policy_id_contains(context.prefill_ids, rid)) {
                ++evicted_prefill;
                if (policy_id_contains(context.prefill_late_ids, rid)) {
                    ++evicted_prefill_late;
                }
                if (policy_id_contains(context.prefill_missed_ids, rid)) {
                    ++evicted_prefill_missed;
                }
            }
            if (policy_id_contains(context.decode_ids, rid)) {
                ++evicted_decode;
                if (policy_id_contains(context.decode_late_ids, rid)) {
                    ++evicted_decode_late;
                }
            }
        }

        int total_prefill_alloc = 0;
        if constexpr (kRolloutAction) {
            for (const auto& item : action.compact_prefill_allocations) {
                total_prefill_alloc += std::max(0, item.second);
            }
        } else {
            if (action.compact_allocations) {
                for (const auto& item : action.compact_prefill_allocations) {
                    total_prefill_alloc += std::max(0, item.second);
                }
            } else {
                for (const auto& item : action.prefill_allocations) {
                    total_prefill_alloc += std::max(0, item.second);
                }
            }
        }
        int total_decode_alloc = 0;
        if constexpr (kRolloutAction) {
            total_decode_alloc = static_cast<int>(
                action.compact_decode_request_ids.size());
        } else {
            if (action.compact_allocations) {
                total_decode_alloc = static_cast<int>(
                    action.compact_decode_request_ids.size());
            } else {
                for (const auto& item : action.decode_allocations) {
                    total_decode_alloc += std::max(0, item.second);
                }
            }
        }
        std::size_t prefill_allocation_count = 0;
        std::size_t decode_allocation_count = 0;
        if constexpr (kRolloutAction) {
            prefill_allocation_count = action.compact_prefill_allocations.size();
            decode_allocation_count = action.compact_decode_request_ids.size();
        } else {
            prefill_allocation_count = action.compact_allocations
                ? action.compact_prefill_allocations.size()
                : action.prefill_allocations.size();
            decode_allocation_count = action.compact_allocations
                ? action.compact_decode_request_ids.size()
                : action.decode_allocations.size();
        }

        float* out = output;
        *out++ = static_cast<float>(norm01_feature(
            total_prefill_alloc, kMaxPrefillActionAlloc));
        *out++ = static_cast<float>(norm01_feature(
            total_decode_alloc, kMaxDecodeRequests));
        *out++ = static_cast<float>(norm01_feature(
            static_cast<double>(prefill_allocation_count),
            kMaxPrefillSlots));
        *out++ = static_cast<float>(norm01_feature(
            static_cast<double>(decode_allocation_count),
            kMaxDecodeRequests));
        *out++ = static_cast<float>(norm01_feature(
            static_cast<double>(evicted_prefill),
            std::max(1.0, static_cast<double>(context.prefill_count))));
        *out++ = static_cast<float>(norm01_feature(
            static_cast<double>(evicted_decode), kMaxDecodeRequests));
        *out++ = total_prefill_alloc > 0 ? 1.0f : 0.0f;
        *out++ = total_decode_alloc > 0 ? 1.0f : 0.0f;
        *out++ = evicted.empty() ? 0.0f : 1.0f;
        *out++ = (total_prefill_alloc == 0 &&
                   total_decode_alloc == 0 && evicted.empty())
            ? 1.0f
            : 0.0f;
        *out++ = static_cast<float>(norm01_feature(
            static_cast<double>(evicted_decode_late), kMaxDecodeRequests));
        *out++ = static_cast<float>(norm01_feature(
            static_cast<double>(evicted_prefill_late), kMaxPrefillSlots));
        *out++ = static_cast<float>(norm01_feature(
            static_cast<double>(evicted_prefill_missed), kMaxPrefillSlots));
        *out++ = (context.highest_prefill_late >= 0 &&
                  std::binary_search(evicted.begin(), evicted.end(), context.highest_prefill_late))
            ? 1.0f
            : 0.0f;
        *out++ = (context.highest_decode_late >= 0 &&
                  std::binary_search(evicted.begin(), evicted.end(), context.highest_decode_late))
            ? 1.0f
            : 0.0f;

        for (int slot = 0; slot < kMaxPrefillSlots; ++slot) {
            const bool has_slot =
                slot < static_cast<int>(context.slots.size());
            const int rid = has_slot
                ? context.slots[static_cast<std::size_t>(slot)].rid
                : -1;
            const int remaining = has_slot
                ? std::max(
                    1,
                    context.slots[static_cast<std::size_t>(slot)].remaining)
                : 1;
            int alloc = 0;
            if constexpr (kRolloutAction) {
                for (const auto& item : action.compact_prefill_allocations) {
                    if (item.first == rid) {
                        alloc = std::max(0, item.second);
                        break;
                    }
                }
            } else {
                if (action.compact_allocations) {
                    for (const auto& item : action.compact_prefill_allocations) {
                        if (item.first == rid) {
                            alloc = std::max(0, item.second);
                            break;
                        }
                    }
                } else {
                    const auto allocation = action.prefill_allocations.find(rid);
                    if (allocation != action.prefill_allocations.end()) {
                        alloc = std::max(0, allocation->second);
                    }
                }
            }
            *out++ = alloc > 0 ? 1.0f : 0.0f;
            *out++ = static_cast<float>(norm01_feature(
                static_cast<double>(alloc), kMaxPrefillActionAlloc));
            *out++ = static_cast<float>(norm01_feature(
                static_cast<double>(alloc), static_cast<double>(remaining)));
            *out++ = (rid >= 0 && std::binary_search(evicted.begin(), evicted.end(), rid))
                ? 1.0f
                : 0.0f;
        }
    }

    std::vector<float> controller_action_features_for_policy(
        const SimState& state,
        const ControllerAction& action) const {
        const ControllerPolicyFeatureContext& context =
            build_controller_policy_feature_context(state);
        std::vector<float> output(43, 0.0f);
        write_controller_action_features_for_policy(
            action, context, output.data());
        return output;
    }
    void write_adversary_action_features_for_policy(
        const AdversaryAction& action,
        int canon_action_index,
        float* output) const {
        constexpr double kMaxLaunchWindow = 7.0;
        constexpr double kMaxPrefillTokens = 4096.0;
        constexpr int kStopRuleCount = 5;

        const int request_count = action.compact_requests
            ? std::max(0, action.compact_request_count)
            : static_cast<int>(action.requests.size());
        double avg_prefill = action.compact_requests
            ? static_cast<double>(action.compact_prefill_tokens)
            : 0.0;
        if (!action.compact_requests && !action.requests.empty()) {
            for (const auto& request : action.requests) {
                avg_prefill += static_cast<double>(request.prefill_tokens);
            }
            avg_prefill /= static_cast<double>(action.requests.size());
        }
        const int stop_idx =
            ((canon_action_index % kStopRuleCount) + kStopRuleCount) %
            kStopRuleCount;
        output[0] = static_cast<float>(norm01_feature(
            static_cast<double>(request_count), kMaxLaunchWindow));
        output[1] = static_cast<float>(norm01_feature(
            avg_prefill, kMaxPrefillTokens));
        for (int index = 0; index < kStopRuleCount; ++index) {
            output[2 + index] = index == stop_idx ? 1.0f : 0.0f;
        }
    }

    std::vector<float> adversary_action_features_for_policy(
        const AdversaryAction& action,
        int canon_action_index) const {
        std::vector<float> output(7, 0.0f);
        write_adversary_action_features_for_policy(
            action, canon_action_index, output.data());
        return output;
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

        const bool markov_policy = prior_runtime->is_markov_policy();
        const MarkovValueFeatures markov_features = markov_policy
            ? build_markov_value_features(state)
            : MarkovValueFeatures{};
        const std::vector<float> state_features = markov_policy
            ? std::vector<float>{}
            : hgb_runtime_->build_features(
                state,
                &env_.virtual_simulator(),
                -1,
                nullptr);
        const int action_dim = markov_policy
            ? prior_runtime->action_dim()
            : prior_runtime->feature_dim() -
                static_cast<int>(state_features.size());
        const int row_dim = static_cast<int>(state_features.size()) + action_dim;
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

        std::string cache_key;
        bool cache_owner = false;
        if (rollout_policy_cache_active_ && !markov_policy) {
            const std::uint64_t metadata =
                (static_cast<std::uint64_t>(num_rows) << 32U) ^
                static_cast<std::uint64_t>(row_dim) ^
                (node->player == "controller" ? 0x434f4e54524f4c4cULL
                                               : 0x4144564552534152ULL);
            const std::size_t byte_count = flat_rows.size() * sizeof(float);
            const std::uint64_t hash1 = rollout_hash_bytes(
                flat_rows.data(), byte_count, 1469598103934665603ULL ^ metadata);
            const std::uint64_t hash2 = rollout_hash_bytes(
                flat_rows.data(), byte_count, 1099511628211ULL ^ ~metadata);
            cache_key.resize(2U * sizeof(std::uint64_t));
            std::memcpy(cache_key.data(), &hash1, sizeof(hash1));
            std::memcpy(cache_key.data() + sizeof(hash1), &hash2, sizeof(hash2));
            std::unique_lock<std::mutex> lock(rollout_policy_cache_mutex_);
            auto cached = rollout_policy_score_cache_.find(cache_key);
            if (cached != rollout_policy_score_cache_.end()) {
                ++perf_rollout_policy_cache_hits_;
                return cached->second;
            }
            while (rollout_policy_scores_inflight_.find(cache_key) !=
                   rollout_policy_scores_inflight_.end()) {
                rollout_policy_cache_cv_.wait(lock);
                cached = rollout_policy_score_cache_.find(cache_key);
                if (cached != rollout_policy_score_cache_.end()) {
                    ++perf_rollout_policy_cache_hits_;
                    return cached->second;
                }
            }
            rollout_policy_scores_inflight_.insert(cache_key);
            cache_owner = true;
            ++perf_rollout_policy_cache_misses_;
        }
        try {
            if (markov_policy) {
                scores = in_.cross_game_inference_batcher != nullptr
                    ? in_.cross_game_inference_batcher->predict_markov_policy(
                        *prior_runtime,
                        node->player,
                        {markov_features},
                        flat_rows,
                        num_rows,
                        {0, num_rows})
                    : prior_runtime->predict_markov_policy_grouped_batch(
                        {markov_features}, flat_rows, num_rows, {0, num_rows}, 1);
            } else {
                scores = prior_runtime->predict_raw_batch_flat(
                    flat_rows, num_rows, row_dim);
            }
        } catch (...) {
            if (cache_owner) {
                std::lock_guard<std::mutex> lock(rollout_policy_cache_mutex_);
                rollout_policy_scores_inflight_.erase(cache_key);
                rollout_policy_cache_cv_.notify_all();
            }
            throw;
        }
        for (std::size_t i = 0; i < valid_row.size() && i < scores.size(); ++i) {
            if (!valid_row[i]) scores[i] = 0.0;
        }
        if (rollout_policy_cache_active_ && !markov_policy) {
            std::lock_guard<std::mutex> lock(rollout_policy_cache_mutex_);
            rollout_policy_score_cache_.emplace(cache_key, scores);
            rollout_policy_scores_inflight_.erase(cache_key);
            rollout_policy_cache_cv_.notify_all();
        }
        return scores;
    }

    void compute_policy_priors_plain(
        TreeNode* node,
        const SimState& state,
        const std::vector<int>& canonical_indices,
        double temperature_override = -1.0,
        bool allow_root_noise = true) {
        if (node == nullptr) return;

        const double policy_temperature = temperature_override > 0.0
            ? temperature_override : in_.policy_prior_temperature;
        node->action_priors = uniform_policy_priors_plain(canonical_indices);
        std::vector<double> model_priors;

        if (in_.use_policy_prior && hgb_policy_priors_available(node->player)) {
            const std::vector<double> scores = hgb_policy_scores_plain(node, state, canonical_indices);
            const std::vector<double> probs = softmax_scores_plain(scores, policy_temperature);
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

                // Torch policy inference already returns masked probabilities.
                // Apply temperature in probability space; softmaxing these
                // probabilities again would flatten the policy a second time.
                const double inverse_temperature = 1.0 /
                    std::max(1e-6, policy_temperature);
                std::vector<double> probs(canonical_scores.size(), 0.0);
                double probability_sum = 0.0;
                for (std::size_t i = 0; i < canonical_scores.size(); ++i) {
                    const double probability = clampv(canonical_scores[i], 0.0, 1.0);
                    probs[i] = probability > 0.0
                        ? std::pow(probability, inverse_temperature)
                        : 0.0;
                    probability_sum += probs[i];
                }
                if (probability_sum > 0.0 && std::isfinite(probability_sum)) {
                    for (double& probability : probs) probability /= probability_sum;
                }
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

        if (allow_root_noise) apply_root_dirichlet_noise_plain(node);
        refresh_policy_prior_vectors_plain(node, model_priors);
    }

    std::vector<double> infer_values_for_states(
        const std::vector<SimState>& states,
        const std::string& player,
        const std::vector<uint8_t>& action_mask) {
        if (hgb_runtime_ != nullptr) {
            if (in_.cross_game_inference_batcher != nullptr) {
                return in_.cross_game_inference_batcher->infer_values(
                    *hgb_runtime_,
                    states,
                    std::vector<const VirtualSimulatorGV2*>(
                        states.size(), &env_.virtual_simulator()));
            }
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
        std::unordered_map<std::uint64_t, std::vector<int>> hash_to_canons;

        alias_to_canon.reserve(valid_indices.size());
        canon_to_aliases.reserve(valid_indices.size());
        canonical_indices.reserve(valid_indices.size());
        hash_to_canons.reserve(valid_indices.size());

        for (int idx : valid_indices) {
            const ControllerAction& act = sampled.actions[static_cast<std::size_t>(idx)];
            const std::uint64_t hash = controller_action_hash(act);
            std::vector<int>& candidates = hash_to_canons[hash];
            int canonical = -1;
            for (int candidate : candidates) {
                if (controller_actions_equivalent(
                        act, sampled.actions[static_cast<std::size_t>(candidate)])) {
                    canonical = candidate;
                    break;
                }
            }
            if (canonical < 0) {
                candidates.push_back(idx);
                alias_to_canon[idx] = idx;
                canon_to_aliases[idx] = {idx};
                canonical_indices.push_back(idx);
            } else {
                alias_to_canon[idx] = canonical;
                canon_to_aliases[canonical].push_back(idx);
            }
        }
        return {std::move(alias_to_canon), std::move(canon_to_aliases), std::move(canonical_indices)};
    }

    std::vector<int> canonicalize_controller_indices_only(
        const SampledActionSet<ControllerAction>& sampled,
        const std::vector<int>& valid_indices) const {
        std::vector<int> canonical_indices;
        std::unordered_map<std::uint64_t, std::vector<int>> hash_to_canons;
        canonical_indices.reserve(valid_indices.size());
        hash_to_canons.reserve(valid_indices.size());
        for (int idx : valid_indices) {
            const ControllerAction& action =
                sampled.actions[static_cast<std::size_t>(idx)];
            const std::uint64_t hash = controller_action_hash(action);
            std::vector<int>& candidates = hash_to_canons[hash];
            bool found = false;
            for (int candidate : candidates) {
                if (controller_actions_equivalent(
                        action,
                        sampled.actions[static_cast<std::size_t>(candidate)])) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                candidates.push_back(idx);
                canonical_indices.push_back(idx);
            }
        }
        return canonical_indices;
    }

    std::tuple<std::unordered_map<int, int>, std::unordered_map<int, std::vector<int>>, std::vector<int>>
    canonicalize_adversary_indices(
        const SampledActionSet<AdversaryAction>& sampled,
        const std::vector<int>& valid_indices) const {
        (void)sampled;
        std::unordered_map<int, int> alias_to_canon;
        std::unordered_map<int, std::vector<int>> canon_to_aliases;
        std::vector<int> canonical_indices;
        alias_to_canon.reserve(valid_indices.size());
        canon_to_aliases.reserve(valid_indices.size());
        canonical_indices.reserve(valid_indices.size());
        for (int idx : valid_indices) {
            alias_to_canon[idx] = idx;
            canon_to_aliases[idx] = {idx};
            canonical_indices.push_back(idx);
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
            auto sampled = env_.sample_controller_actions(state);
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

    void capture_root_puct_snapshot(
        const TreeNode& root,
        int completed_simulations,
        std::vector<RootPuctTraceStep>* trace) const {
        if (trace == nullptr) return;

        std::vector<int> action_indices;
        action_indices.reserve(root.action_priors.size());
        for (const auto& kv : root.action_priors) action_indices.push_back(kv.first);
        std::sort(action_indices.begin(), action_indices.end());

        double c = in_.puct_c;
        if (!std::isfinite(c) || c == 0.0) c = 1.0;
        const double bound_min = std::isfinite(root.min_value)
            ? root.min_value
            : std::numeric_limits<double>::quiet_NaN();
        const double bound_max = std::isfinite(root.max_value)
            ? root.max_value
            : std::numeric_limits<double>::quiet_NaN();

        for (int action_idx : action_indices) {
            const auto child_it = root.children.find(action_idx);
            const TreeNode* child = child_it != root.children.end()
                ? child_it->second.get()
                : nullptr;
            const bool visited = child != nullptr && child->visits > 0;
            const int visits = visited ? child->visits : 0;
            const double q_value = visited
                ? child->mean_value()
                : std::numeric_limits<double>::quiet_NaN();
            const double normalized_q = visited
                ? normalize_plain_child_value_for_selection(root, *child)
                : 0.5;
            const double exploration_raw =
                puct_explore_plain(root, action_idx, visits);

            RootPuctTraceStep step;
            step.sim_iteration = completed_simulations;
            step.action_index = action_idx;
            step.visited = visited;
            step.visits = visits;
            step.q_value = q_value;
            step.normalized_q = normalized_q;
            step.prior = plain_action_prior(root, action_idx);
            step.exploration_raw = exploration_raw;
            step.exploration_weighted = c * exploration_raw;
            step.puct_score = normalized_q + step.exploration_weighted;
            step.parent_min_value = bound_min;
            step.parent_max_value = bound_max;
            trace->push_back(std::move(step));
        }
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

    struct PolicyRolloutEdge {
        double reward = 0.0;
        double discount = 1.0;
    };

    struct RolloutValueParts {
        double reward_return = 0.0;
        double bootstrap_return = 0.0;

        double total() const { return reward_return + bootstrap_return; }
    };

    struct PolicyRolloutTrajectory {
        SimState state;
        std::string player;
        std::vector<PolicyRolloutEdge> edges;
        std::vector<int> action_history;
        std::vector<std::string> step_players;
        std::vector<std::string> step_action_categories;
        std::vector<double> step_sim_times_before;
        std::vector<double> step_sim_times_after;
        bool active = true;
        bool terminal = false;
        double policy_sec = 0.0;
        double transition_sec = 0.0;
        PythonRandomCompat rng;
    };

    static std::string controller_rollout_category(
        const ControllerAction& action) {
        int prefill_tokens = 0;
        int decode_tokens = 0;
        if (action.compact_allocations) {
            for (const auto& item : action.compact_prefill_allocations) {
                prefill_tokens += item.second;
            }
            decode_tokens = static_cast<int>(
                action.compact_decode_request_ids.size());
        } else {
            for (const auto& item : action.prefill_allocations) {
                prefill_tokens += item.second;
            }
            for (const auto& item : action.decode_allocations) {
                decode_tokens += item.second;
            }
        }
        if (prefill_tokens == 0 && decode_tokens > 0) return "decode_only";
        if (prefill_tokens == 128) return "prefill_128";
        if (prefill_tokens == 256) return "prefill_256";
        if (prefill_tokens == 512) return "prefill_512";
        if (prefill_tokens == 1024) return "prefill_1024";
        return prefill_tokens > 0 ? "prefill_other" : "other";
    }

    struct RolloutPreparedPolicy {
        TreeNode sentinel;
        TreeNode node;
        bool compact_mode = false;
        bool controller_player = false;
        std::vector<RolloutControllerAction> compact_controller_actions;
        std::vector<AdversaryAction> compact_adversary_actions;
        std::vector<int> compact_original_indices;
        std::vector<double> compact_priors;
        std::vector<int> canonical_indices;
        bool markov_policy = false;
        MarkovValueFeatures markov_features;
        std::vector<float> state_features;
        std::vector<float> flat_actions;
        std::vector<uint8_t> valid_rows;
        int action_dim = 0;
        int action_index = -1;
        double initialize_sec = 0.0;
        double state_features_sec = 0.0;
        double action_features_sec = 0.0;
    };

    struct RolloutPolicyBatchWorkspace {
        std::vector<std::size_t> selected;
        std::vector<float> flat_states;
        std::vector<MarkovValueFeatures> markov_states;
        std::vector<float> flat_actions;
        std::vector<uint8_t> valid_rows;
        std::vector<int> group_offsets;
        std::vector<std::uint64_t> cache_hashes;
    };

    struct FastRolloutPolicyCacheEntry {
        bool controller_player = false;
        int action_dim = 0;
        std::size_t canonical_count = 0;
        std::vector<float> state_features;
        std::vector<float> flat_actions;
        std::vector<uint8_t> valid_rows;
        std::vector<double> scores;
    };

    void reset_rollout_prepared_policy_plain(
        RolloutPreparedPolicy* prepared) {
        if (prepared == nullptr) return;
        prepared->node.player.clear();
        prepared->node.parent = nullptr;
        prepared->node.controller_actions_by_index.clear();
        prepared->node.adversary_actions_by_index.clear();
        prepared->node.valid_mask.clear();
        prepared->node.untried_action_indices.clear();
        prepared->node.rollout_action_original_indices.clear();
        prepared->node.action_priors.clear();
        prepared->node.action_alias_to_canonical.clear();
        prepared->node.canonical_to_action_aliases.clear();
        prepared->compact_mode = false;
        prepared->controller_player = false;
        prepared->compact_controller_actions.clear();
        prepared->compact_adversary_actions.clear();
        prepared->compact_original_indices.clear();
        prepared->compact_priors.clear();
        prepared->canonical_indices.clear();
        prepared->markov_policy = false;
        prepared->markov_features = MarkovValueFeatures{};
        prepared->state_features.clear();
        prepared->flat_actions.clear();
        prepared->valid_rows.clear();
        prepared->action_dim = 0;
        prepared->action_index = -1;
        prepared->initialize_sec = 0.0;
        prepared->state_features_sec = 0.0;
        prepared->action_features_sec = 0.0;
    }

    void initialize_compact_rollout_policy_plain(
        RolloutPreparedPolicy* prepared,
        const SimState& state,
        bool controller_player) {
        prepared->compact_mode = true;
        prepared->controller_player = controller_player;
        if (controller_player) {
            bool has_schedulable_request = false;
            for (const auto& request : state.requests) {
                if (request.feature_only || request.completed) continue;
                if ((!request.prefill_done() && request.remaining_prefill() > 0) ||
                    (request.prefill_done() && request.remaining_decode() > 0)) {
                    has_schedulable_request = true;
                    break;
                }
            }
            const bool pending_adversary_tick =
                state.stats.next_adv_tick >= 0.0 &&
                state.stats.next_adv_tick <= state.sim_time + env_.cfg().eps;
            if (pending_adversary_tick || !has_schedulable_request) {
                RolloutControllerAction noop;
                noop.token_budget = 0;
                noop.mapping = {0, 0, 0};
                noop.has_mapping = true;
                noop.valid = true;
                prepared->compact_controller_actions.push_back(std::move(noop));
                prepared->compact_original_indices.push_back(0);
            } else {
                auto sampled = env_.sample_controller_rollout_actions(state);
                prepared->compact_controller_actions = std::move(sampled.actions);
                prepared->compact_original_indices =
                    std::move(sampled.original_indices);
            }
            prepared->canonical_indices.resize(
                prepared->compact_controller_actions.size());
        } else {
            const bool strictly_before_adversary_tick =
                state.stats.next_adv_tick >= 0.0 &&
                state.sim_time + 1e-9 < state.stats.next_adv_tick;
            if (strictly_before_adversary_tick) {
                AdversaryAction noop;
                noop.valid = true;
                prepared->compact_adversary_actions.push_back(std::move(noop));
                prepared->compact_original_indices.push_back(0);
            } else {
                auto sampled = env_.sample_adversary_actions(
                    state, {}, true, true);
                prepared->compact_adversary_actions = std::move(sampled.actions);
                prepared->compact_original_indices =
                    std::move(sampled.original_indices);
            }
            prepared->canonical_indices.resize(
                prepared->compact_adversary_actions.size());
        }
        std::iota(
            prepared->canonical_indices.begin(),
            prepared->canonical_indices.end(),
            0);
        const std::size_t action_count = prepared->canonical_indices.size();
        prepared->compact_priors.assign(
            action_count,
            action_count > 0
                ? 1.0 / static_cast<double>(action_count)
                : 0.0);
    }

    bool prepare_rollout_policy_plain(
        RolloutPreparedPolicy* prepared,
        const PolicyRolloutTrajectory& trajectory) {
        if (prepared == nullptr || !trajectory.active) return false;
        const bool controller_player = trajectory.player == "controller";
        const bool detailed_perf = !in_.rollout_optimized_execution;
        const auto initialize_begin = detailed_perf
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        if (in_.rollout_optimized_execution) {
            initialize_compact_rollout_policy_plain(
                prepared, trajectory.state, controller_player);
        } else {
            prepared->node.player = trajectory.player;
            prepared->node.parent = &prepared->sentinel;
            initialize_rollout_policy_node_plain(
                &prepared->node, trajectory.state);
            prepared->canonical_indices =
                prepared->node.untried_action_indices;
            prepared->node.action_priors =
                uniform_policy_priors_plain(prepared->canonical_indices);
        }
        if (detailed_perf) {
            const auto initialize_end = std::chrono::steady_clock::now();
            prepared->initialize_sec = std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    initialize_end - initialize_begin).count();
        }
        if (prepared->canonical_indices.empty()) return false;
        // A forced policy decision has probability one regardless of model
        // logits or temperature.
        if (prepared->canonical_indices.size() == 1U) return false;

        if (!in_.use_policy_prior ||
            !hgb_policy_priors_available(trajectory.player)) {
            if (!prepared->compact_mode) {
                compute_policy_priors_plain(
                    &prepared->node,
                    trajectory.state,
                    prepared->canonical_indices,
                    std::max(1e-8, in_.rollout_policy_temperature),
                    false);
            }
            return false;
        }

        const NativeHGBModelRuntime* prior_runtime = controller_player
            ? controller_prior_runtime_
            : adversary_prior_runtime_;
        const auto state_features_begin = detailed_perf
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        prepared->markov_policy = prior_runtime->is_markov_policy();
        if (prepared->markov_policy) {
            prepared->markov_features =
                build_markov_value_features(trajectory.state);
            prepared->state_features.reserve(
                2U + prepared->markov_features.global_features.size() +
                prepared->markov_features.request_features.size() +
                prepared->markov_features.launch_features.size());
            prepared->state_features.push_back(
                static_cast<float>(prepared->markov_features.request_count));
            prepared->state_features.push_back(
                static_cast<float>(prepared->markov_features.launch_count));
            prepared->state_features.insert(
                prepared->state_features.end(),
                prepared->markov_features.global_features.begin(),
                prepared->markov_features.global_features.end());
            prepared->state_features.insert(
                prepared->state_features.end(),
                prepared->markov_features.request_features.begin(),
                prepared->markov_features.request_features.end());
            prepared->state_features.insert(
                prepared->state_features.end(),
                prepared->markov_features.launch_features.begin(),
                prepared->markov_features.launch_features.end());
        } else {
            hgb_runtime_->build_features_into(
                trajectory.state,
                &env_.virtual_simulator(),
                -1,
                &prepared->state_features,
                nullptr);
        }
        if (detailed_perf) {
            const auto state_features_end = std::chrono::steady_clock::now();
            prepared->state_features_sec = std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    state_features_end - state_features_begin).count();
        }
        const int row_dim = prior_runtime->feature_dim();
        prepared->action_dim = prepared->markov_policy
            ? prior_runtime->action_dim()
            : row_dim - static_cast<int>(prepared->state_features.size());
        if ((!prepared->markov_policy && row_dim <= 0) ||
            prepared->action_dim <= 0) {
            if (!prepared->compact_mode) {
                compute_policy_priors_plain(
                    &prepared->node,
                    trajectory.state,
                    prepared->canonical_indices,
                    std::max(1e-8, in_.rollout_policy_temperature),
                    false);
            }
            return false;
        }

        const int num_rows =
            static_cast<int>(prepared->canonical_indices.size());
        prepared->flat_actions.assign(
            static_cast<std::size_t>(num_rows) *
                static_cast<std::size_t>(prepared->action_dim),
            0.0f);
        prepared->valid_rows.assign(static_cast<std::size_t>(num_rows), 1);
        const auto action_features_begin = detailed_perf
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        const ControllerPolicyFeatureContext* controller_context = nullptr;
        if (controller_player) {
            if (prepared->action_dim != 43) {
                throw std::runtime_error(
                    "native rollout policy action feature dimension mismatch");
            }
            controller_context =
                &build_controller_policy_feature_context(trajectory.state);
        } else if (prepared->action_dim != 7) {
            throw std::runtime_error(
                "native rollout policy action feature dimension mismatch");
        }
        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
            const int canon_idx =
                prepared->canonical_indices[static_cast<std::size_t>(row_idx)];
            const std::size_t row_offset =
                static_cast<std::size_t>(row_idx) *
                static_cast<std::size_t>(prepared->action_dim);
            float* action_features =
                prepared->flat_actions.data() + row_offset;

            if (controller_player) {
                if (prepared->compact_mode) {
                    const auto& actions = prepared->compact_controller_actions;
                    if (canon_idx < 0 ||
                        canon_idx >= static_cast<int>(actions.size())) {
                        prepared->valid_rows[static_cast<std::size_t>(row_idx)] = 0;
                        continue;
                    }
                    write_controller_action_features_for_policy(
                        actions[static_cast<std::size_t>(canon_idx)],
                        *controller_context,
                        action_features);
                } else {
                    const auto& actions =
                        prepared->node.controller_actions_by_index;
                    if (canon_idx < 0 ||
                        canon_idx >= static_cast<int>(actions.size())) {
                        prepared->valid_rows[static_cast<std::size_t>(row_idx)] = 0;
                        continue;
                    }
                    write_controller_action_features_for_policy(
                        actions[static_cast<std::size_t>(canon_idx)],
                        *controller_context,
                        action_features);
                }
            } else {
                const auto& actions = prepared->compact_mode
                    ? prepared->compact_adversary_actions
                    : prepared->node.adversary_actions_by_index;
                if (canon_idx < 0 ||
                    canon_idx >= static_cast<int>(actions.size())) {
                    prepared->valid_rows[static_cast<std::size_t>(row_idx)] = 0;
                    continue;
                }
                const auto& original_indices = prepared->compact_mode
                    ? prepared->compact_original_indices
                    : prepared->node.rollout_action_original_indices;
                const int original_idx =
                    canon_idx < static_cast<int>(original_indices.size())
                        ? original_indices[static_cast<std::size_t>(canon_idx)]
                        : canon_idx;
                write_adversary_action_features_for_policy(
                    actions[static_cast<std::size_t>(canon_idx)],
                    original_idx,
                    action_features);
            }
        }
        if (detailed_perf) {
            const auto action_features_end = std::chrono::steady_clock::now();
            prepared->action_features_sec = std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    action_features_end - action_features_begin).count();
        }
        return true;
    }

    void apply_rollout_policy_scores_plain(
        RolloutPreparedPolicy* prepared,
        const double* scores,
        std::size_t score_count) {
        if (prepared == nullptr || scores == nullptr ||
            score_count != prepared->canonical_indices.size()) {
            throw std::runtime_error(
                "native grouped rollout policy score size mismatch");
        }

        if (prepared->compact_mode) {
            const double temperature = std::max(
                1e-8, in_.rollout_policy_temperature);
            double max_scaled = -std::numeric_limits<double>::infinity();
            for (std::size_t idx = 0; idx < score_count; ++idx) {
                max_scaled = std::max(
                    max_scaled, scores[idx] / temperature);
            }
            double softmax_total = 0.0;
            for (std::size_t idx = 0; idx < score_count; ++idx) {
                const int canon_idx = prepared->canonical_indices[idx];
                const double probability = std::exp(
                    scores[idx] / temperature - max_scaled);
                prepared->compact_priors[
                    static_cast<std::size_t>(canon_idx)] = probability;
                softmax_total += probability;
            }
            if (!(softmax_total > 0.0) ||
                !std::isfinite(softmax_total)) {
                const double uniform = 1.0 /
                    static_cast<double>(score_count);
                for (int canon_idx : prepared->canonical_indices) {
                    prepared->compact_priors[
                        static_cast<std::size_t>(canon_idx)] = uniform;
                }
            } else {
                for (int canon_idx : prepared->canonical_indices) {
                    prepared->compact_priors[
                        static_cast<std::size_t>(canon_idx)] /= softmax_total;
                }
            }

            const double min_prob = std::max(0.0, in_.prior_min_prob);
            double total = 0.0;
            for (int canon_idx : prepared->canonical_indices) {
                double& probability = prepared->compact_priors[
                    static_cast<std::size_t>(canon_idx)];
                probability = std::max(min_prob, probability);
                total += probability;
            }
            if (total > 0.0 && std::isfinite(total)) {
                for (int canon_idx : prepared->canonical_indices) {
                    prepared->compact_priors[
                        static_cast<std::size_t>(canon_idx)] /= total;
                }
            } else {
                const double uniform = 1.0 /
                    static_cast<double>(score_count);
                for (int canon_idx : prepared->canonical_indices) {
                    prepared->compact_priors[
                        static_cast<std::size_t>(canon_idx)] = uniform;
                }
            }
            return;
        }

        const std::vector<double> score_vector(
            scores, scores + score_count);
        const std::vector<double> probs = softmax_scores_plain(
            score_vector,
            std::max(1e-8, in_.rollout_policy_temperature));
        const double min_prob = std::max(0.0, in_.prior_min_prob);
        double total = 0.0;
        for (std::size_t idx = 0; idx < score_count; ++idx) {
            const double probability = std::max(min_prob, probs[idx]);
            prepared->node.action_priors[
                prepared->canonical_indices[idx]] = probability;
            total += probability;
        }
        if (total > 0.0 && std::isfinite(total)) {
            for (int canon_idx : prepared->canonical_indices) {
                prepared->node.action_priors[canon_idx] /= total;
            }
        } else {
            prepared->node.action_priors =
                uniform_policy_priors_plain(prepared->canonical_indices);
        }
    }

    void apply_rollout_policy_scores_plain(
        RolloutPreparedPolicy* prepared,
        const std::vector<double>& scores) {
        apply_rollout_policy_scores_plain(
            prepared, scores.data(), scores.size());
    }

    void score_grouped_rollout_policies_plain(
        std::vector<RolloutPreparedPolicy>* prepared,
        const std::vector<PolicyRolloutTrajectory>& trajectories,
        const std::string& player,
        int parallel_threads,
        RolloutPolicyBatchWorkspace* workspace) {
        if (prepared == nullptr || workspace == nullptr) return;
        const NativeHGBModelRuntime* prior_runtime =
            player == "controller"
                ? controller_prior_runtime_
                : adversary_prior_runtime_;
        if (prior_runtime == nullptr || !prior_runtime->loaded()) return;

        auto& selected = workspace->selected;
        auto& flat_states = workspace->flat_states;
        auto& markov_states = workspace->markov_states;
        auto& flat_actions = workspace->flat_actions;
        auto& valid_rows = workspace->valid_rows;
        auto& group_offsets = workspace->group_offsets;
        auto& cache_hashes = workspace->cache_hashes;
        std::vector<std::string> cache_keys;
        selected.clear();
        flat_states.clear();
        markov_states.clear();
        flat_actions.clear();
        valid_rows.clear();
        group_offsets.clear();
        cache_hashes.clear();
        group_offsets.push_back(0);
        int state_dim = 0;
        int action_dim = 0;
        int total_rows = 0;
        for (std::size_t idx = 0; idx < prepared->size(); ++idx) {
            auto& item = (*prepared)[idx];
            const bool player_matches = item.compact_mode
                ? ((player == "controller") == item.controller_player)
                : item.node.player == player;
            if (!trajectories[idx].active ||
                !player_matches ||
                item.flat_actions.empty()) {
                continue;
            }
            perf_rollout_policy_action_rows_ +=
                static_cast<std::int64_t>(item.canonical_indices.size());

            std::uint64_t fast_hash = 0;
            bool fast_cache_hit = false;
            if (in_.rollout_optimized_execution) {
                const bool controller_item = player == "controller";
                const std::uint64_t metadata[] = {
                    controller_item ? 1ULL : 0ULL,
                    static_cast<std::uint64_t>(item.state_features.size()),
                    static_cast<std::uint64_t>(item.flat_actions.size()),
                    static_cast<std::uint64_t>(item.valid_rows.size()),
                    static_cast<std::uint64_t>(item.action_dim),
                    static_cast<std::uint64_t>(item.canonical_indices.size()),
                };
                std::uint64_t hash1 = rollout_hash_fast(
                    metadata, sizeof(metadata), 0x243f6a8885a308d3ULL);
                hash1 = rollout_hash_fast(
                    item.state_features.data(),
                    item.state_features.size() * sizeof(float), hash1);
                hash1 = rollout_hash_fast(
                    item.flat_actions.data(),
                    item.flat_actions.size() * sizeof(float), hash1);
                hash1 = rollout_hash_fast(
                    item.valid_rows.data(), item.valid_rows.size(), hash1);
                fast_hash = hash1;

                const auto cached_bucket =
                    fast_rollout_policy_score_cache_.find(hash1);
                if (cached_bucket != fast_rollout_policy_score_cache_.end()) {
                    for (const auto& entry : cached_bucket->second) {
                        const bool exact_match =
                            entry.controller_player == controller_item &&
                            entry.action_dim == item.action_dim &&
                            entry.canonical_count == item.canonical_indices.size() &&
                            entry.state_features.size() == item.state_features.size() &&
                            entry.flat_actions.size() == item.flat_actions.size() &&
                            entry.valid_rows.size() == item.valid_rows.size() &&
                            entry.scores.size() == item.canonical_indices.size() &&
                            (item.state_features.empty() ||
                                std::memcmp(
                                    entry.state_features.data(),
                                    item.state_features.data(),
                                    item.state_features.size() * sizeof(float)) == 0) &&
                            (item.flat_actions.empty() ||
                                std::memcmp(
                                    entry.flat_actions.data(),
                                    item.flat_actions.data(),
                                    item.flat_actions.size() * sizeof(float)) == 0) &&
                            (item.valid_rows.empty() ||
                                std::memcmp(
                                    entry.valid_rows.data(),
                                    item.valid_rows.data(),
                                    item.valid_rows.size()) == 0);
                        if (!exact_match) continue;
                        apply_rollout_policy_scores_plain(&item, entry.scores);
                        ++perf_rollout_policy_cache_hits_;
                        fast_cache_hit = true;
                        break;
                    }
                }
                if (fast_cache_hit) continue;
                ++perf_rollout_policy_cache_misses_;
            }

            std::string cache_key;
            if (rollout_policy_cache_active_) {
                const std::uint64_t player_tag =
                    player == "controller" ? 0x434f4e54524f4c4cULL
                                           : 0x4144564552534152ULL;
                const std::uint64_t metadata[] = {
                    player_tag,
                    static_cast<std::uint64_t>(item.state_features.size()),
                    static_cast<std::uint64_t>(item.flat_actions.size()),
                    static_cast<std::uint64_t>(item.valid_rows.size()),
                    static_cast<std::uint64_t>(item.action_dim),
                    static_cast<std::uint64_t>(item.canonical_indices.size()),
                };
                cache_key.reserve(
                    sizeof(metadata) +
                    item.state_features.size() * sizeof(float) +
                    item.flat_actions.size() * sizeof(float) +
                    item.valid_rows.size());
                cache_key.append(
                    reinterpret_cast<const char*>(metadata), sizeof(metadata));
                cache_key.append(
                    reinterpret_cast<const char*>(item.state_features.data()),
                    item.state_features.size() * sizeof(float));
                cache_key.append(
                    reinterpret_cast<const char*>(item.flat_actions.data()),
                    item.flat_actions.size() * sizeof(float));
                cache_key.append(
                    reinterpret_cast<const char*>(item.valid_rows.data()),
                    item.valid_rows.size());
                const auto cached =
                    rollout_policy_score_cache_.find(cache_key);
                if (cached != rollout_policy_score_cache_.end()) {
                    apply_rollout_policy_scores_plain(&item, cached->second);
                    ++perf_rollout_policy_cache_hits_;
                    continue;
                }
                ++perf_rollout_policy_cache_misses_;
            }

            if (state_dim == 0 && !item.markov_policy) {
                state_dim = static_cast<int>(item.state_features.size());
                action_dim = item.action_dim;
            }
            if ((!item.markov_policy &&
                 static_cast<int>(item.state_features.size()) != state_dim) ||
                (action_dim != 0 && item.action_dim != action_dim)) {
                throw std::runtime_error(
                    "native grouped rollout policy split dimensions differ");
            }
            if (action_dim == 0) action_dim = item.action_dim;

            selected.push_back(idx);
            cache_keys.push_back(std::move(cache_key));
            if (in_.rollout_optimized_execution) {
                cache_hashes.push_back(fast_hash);
            }
            if (item.markov_policy) {
                markov_states.push_back(item.markov_features);
            } else {
                flat_states.insert(
                    flat_states.end(),
                    item.state_features.begin(),
                    item.state_features.end());
            }
            flat_actions.insert(
                flat_actions.end(),
                item.flat_actions.begin(),
                item.flat_actions.end());
            valid_rows.insert(
                valid_rows.end(),
                item.valid_rows.begin(),
                item.valid_rows.end());
            total_rows += static_cast<int>(item.canonical_indices.size());
            group_offsets.push_back(total_rows);
        }
        if (selected.empty()) return;

        perf_rollout_policy_scored_action_rows_ += total_rows;

        std::vector<double> scores = prior_runtime->is_markov_policy()
            ? (in_.cross_game_inference_batcher != nullptr
                ? in_.cross_game_inference_batcher->predict_markov_policy(
                    *prior_runtime,
                    player,
                    markov_states,
                    flat_actions,
                    total_rows,
                    group_offsets)
                : prior_runtime->predict_markov_policy_grouped_batch(
                    markov_states,
                    flat_actions,
                    total_rows,
                    group_offsets,
                    parallel_threads))
            : prior_runtime->predict_raw_grouped_split_batch_flat(
                flat_states,
                flat_actions,
                total_rows,
                group_offsets,
                parallel_threads);
        if (scores.size() != static_cast<std::size_t>(total_rows)) {
            throw std::runtime_error(
                "native grouped rollout policy output size mismatch");
        }
        for (std::size_t row = 0; row < valid_rows.size(); ++row) {
            if (!valid_rows[row]) scores[row] = 0.0;
        }
        for (std::size_t group = 0; group < selected.size(); ++group) {
            const int begin = group_offsets[group];
            const int end = group_offsets[group + 1];
            auto& selected_item = (*prepared)[selected[group]];
            apply_rollout_policy_scores_plain(
                &selected_item,
                scores.data() + begin,
                static_cast<std::size_t>(end - begin));
            if (in_.rollout_optimized_execution) {
                constexpr std::size_t kFastPolicyCacheMaxEntries = 65536U;
                if (fast_rollout_policy_cache_entry_count_ >=
                    kFastPolicyCacheMaxEntries) {
                    fast_rollout_policy_score_cache_.clear();
                    fast_rollout_policy_cache_entry_count_ = 0;
                }
                FastRolloutPolicyCacheEntry entry;
                entry.controller_player = player == "controller";
                entry.action_dim = selected_item.action_dim;
                entry.canonical_count = selected_item.canonical_indices.size();
                entry.state_features = std::move(selected_item.state_features);
                entry.flat_actions = std::move(selected_item.flat_actions);
                entry.valid_rows = std::move(selected_item.valid_rows);
                entry.scores.assign(scores.begin() + begin, scores.begin() + end);
                fast_rollout_policy_score_cache_[cache_hashes[group]]
                    .push_back(std::move(entry));
                ++fast_rollout_policy_cache_entry_count_;
            }
            if (rollout_policy_cache_active_) {
                std::vector<double> group_scores(
                    scores.begin() + begin, scores.begin() + end);
                rollout_policy_score_cache_.emplace(
                    std::move(cache_keys[group]), std::move(group_scores));
            }
        }
    }

    void initialize_rollout_policy_node_plain(TreeNode* node, const SimState& state) {
        if (node == nullptr || plain_expanded(*node)) return;
        if (node->player == "controller") {
            bool has_schedulable_request = false;
            for (const auto& request : state.requests) {
                if (request.feature_only || request.completed) continue;
                if ((!request.prefill_done() && request.remaining_prefill() > 0) ||
                    (request.prefill_done() && request.remaining_decode() > 0)) {
                    has_schedulable_request = true;
                    break;
                }
            }
            const bool pending_adversary_tick =
                state.stats.next_adv_tick >= 0.0 &&
                state.stats.next_adv_tick <=
                    (state.sim_time + env_.cfg().eps);
            if (pending_adversary_tick || !has_schedulable_request) {
                ControllerAction noop;
                noop.token_budget = 0;
                noop.strategy = "GV2|evict_none";
                noop.mapping = {0, 0, 0};
                noop.has_mapping = true;
                noop.valid = true;
                node->controller_actions_by_index.push_back(std::move(noop));
                node->valid_mask.push_back(1u);
                node->untried_action_indices.push_back(0);
                node->rollout_action_original_indices.push_back(0);
                return;
            }
            auto sampled = env_.sample_controller_actions(
                state, true, in_.rollout_optimized_execution);
            if (in_.rollout_optimized_execution) {
                node->untried_action_indices.reserve(sampled.actions.size());
                for (int index = 0;
                     index < static_cast<int>(sampled.actions.size());
                     ++index) {
                    node->untried_action_indices.push_back(index);
                }
            } else {
                const auto valid = valid_controller_indices(sampled);
                node->untried_action_indices =
                    canonicalize_controller_indices_only(sampled, valid);
            }
            node->controller_actions_by_index = std::move(sampled.actions);
            node->valid_mask = std::move(sampled.mask);
            node->rollout_action_original_indices =
                std::move(sampled.original_indices);
        } else {
            const bool strictly_before_adversary_tick =
                state.stats.next_adv_tick >= 0.0 &&
                (state.sim_time + 1e-9) < state.stats.next_adv_tick;
            if (strictly_before_adversary_tick) {
                AdversaryAction noop;
                noop.valid = true;
                node->adversary_actions_by_index.push_back(std::move(noop));
                node->valid_mask.push_back(1u);
                node->untried_action_indices.push_back(0);
                node->rollout_action_original_indices.push_back(0);
                return;
            }
            auto sampled = env_.sample_adversary_actions(
                state, {}, true, in_.rollout_optimized_execution);
            node->untried_action_indices =
                valid_adversary_indices(sampled);
            node->adversary_actions_by_index = std::move(sampled.actions);
            node->valid_mask = std::move(sampled.mask);
            node->rollout_action_original_indices =
                std::move(sampled.original_indices);
        }
    }

    void ensure_rollout_policy_node_plain(TreeNode* node, const SimState& state) {
        if (node == nullptr || plain_expanded(*node)) return;
        initialize_rollout_policy_node_plain(node, state);
        compute_policy_priors_plain(
            node,
            state,
            node->untried_action_indices,
            std::max(1e-8, in_.rollout_policy_temperature),
            false);
    }

    int sample_rollout_action_plain(const TreeNode& node, PythonRandomCompat* rng) {
        std::vector<int> indices = node.untried_action_indices;
        std::sort(indices.begin(), indices.end());
        if (indices.empty()) return -1;
        if (indices.size() == 1) return indices.front();

        double total = 0.0;
        for (int idx : indices) total += std::max(0.0, plain_action_prior(node, idx));
        std::vector<double> probabilities;
        probabilities.reserve(indices.size());
        if (!(total > 0.0) || !std::isfinite(total)) {
            probabilities.assign(indices.size(), 1.0 / static_cast<double>(indices.size()));
        } else {
            for (int idx : indices) {
                probabilities.push_back(std::max(0.0, plain_action_prior(node, idx)) / total);
            }
        }
        const double quantum = in_.rollout_probability_quantum;
        if (quantum > 0.0) {
            double quantized_total = 0.0;
            for (double& probability : probabilities) {
                probability = std::floor(probability / quantum + 0.5) * quantum;
                quantized_total += probability;
            }
            if (quantized_total > 0.0) {
                for (double& probability : probabilities) probability /= quantized_total;
            }
        }
        const double draw = rng->random_double();
        double cumulative = 0.0;
        for (std::size_t pos = 0; pos < indices.size(); ++pos) {
            cumulative += probabilities[pos];
            if (draw < cumulative) return indices[pos];
        }
        return indices.back();
    }

    int sample_rollout_action_plain(
        const RolloutPreparedPolicy& prepared,
        PythonRandomCompat* rng) {
        if (!prepared.compact_mode) {
            return sample_rollout_action_plain(prepared.node, rng);
        }
        std::vector<int> indices = prepared.canonical_indices;
        std::sort(indices.begin(), indices.end());
        if (indices.empty()) return -1;
        if (indices.size() == 1) return indices.front();

        double total = 0.0;
        for (int idx : indices) {
            total += std::max(
                0.0,
                prepared.compact_priors[static_cast<std::size_t>(idx)]);
        }
        std::vector<double> probabilities;
        probabilities.reserve(indices.size());
        if (!(total > 0.0) || !std::isfinite(total)) {
            probabilities.assign(
                indices.size(), 1.0 / static_cast<double>(indices.size()));
        } else {
            for (int idx : indices) {
                probabilities.push_back(
                    std::max(
                        0.0,
                        prepared.compact_priors[
                            static_cast<std::size_t>(idx)]) /
                    total);
            }
        }
        const double quantum = in_.rollout_probability_quantum;
        if (quantum > 0.0) {
            double quantized_total = 0.0;
            for (double& probability : probabilities) {
                probability = std::floor(probability / quantum + 0.5) * quantum;
                quantized_total += probability;
            }
            if (quantized_total > 0.0) {
                for (double& probability : probabilities) {
                    probability /= quantized_total;
                }
            }
        }
        const double draw = rng->random_double();
        double cumulative = 0.0;
        for (std::size_t pos = 0; pos < indices.size(); ++pos) {
            cumulative += probabilities[pos];
            if (draw < cumulative) return indices[pos];
        }
        return indices.back();
    }

    void advance_policy_rollout_plain(PolicyRolloutTrajectory* trajectory) {
        if (trajectory == nullptr || !trajectory->active) return;
        const auto policy_begin = std::chrono::steady_clock::now();
        TreeNode sentinel;
        TreeNode policy_node;
        policy_node.player = trajectory->player;
        policy_node.parent = &sentinel;
        ensure_rollout_policy_node_plain(&policy_node, trajectory->state);
        const int action_idx = sample_rollout_action_plain(policy_node, &trajectory->rng);
        const auto policy_end = std::chrono::steady_clock::now();
        trajectory->policy_sec += std::chrono::duration_cast<
            std::chrono::duration<double>>(policy_end - policy_begin).count();
        if (action_idx < 0) {
            trajectory->active = false;
            trajectory->terminal = true;
            return;
        }

        const double parent_cost = state_cost(trajectory->state);
        const double parent_time = trajectory->state.sim_time;
        const auto transition_begin = std::chrono::steady_clock::now();
        if (trajectory->player == "controller") {
            if (action_idx >= static_cast<int>(policy_node.controller_actions_by_index.size())) {
                throw std::runtime_error("rollout controller action index out of range");
            }
            env_.apply_controller_action_inplace(
                trajectory->state,
                policy_node.controller_actions_by_index[static_cast<std::size_t>(action_idx)],
                true);
        } else {
            if (action_idx >= static_cast<int>(policy_node.adversary_actions_by_index.size())) {
                throw std::runtime_error("rollout adversary action index out of range");
            }
            env_.apply_adversary_action_inplace(
                trajectory->state,
                policy_node.adversary_actions_by_index[static_cast<std::size_t>(action_idx)]);
        }

        const double final_time = trajectory->state.stats.transition_final_time >= 0.0
            ? trajectory->state.stats.transition_final_time
            : trajectory->state.sim_time;
        PolicyRolloutEdge edge;
        edge.reward = parent_cost - state_cost(trajectory->state);
        edge.discount = time_discount(final_time, parent_time, in_);
        trajectory->edges.push_back(edge);
        trajectory->action_history.push_back(action_idx);
        trajectory->player = next_player(trajectory->player);
        const auto transition_end = std::chrono::steady_clock::now();
        trajectory->transition_sec += std::chrono::duration_cast<
            std::chrono::duration<double>>(transition_end - transition_begin).count();
    }

    void apply_prepared_rollout_action_plain(
        PolicyRolloutTrajectory* trajectory,
        const RolloutPreparedPolicy& prepared,
        int action_idx) {
        if (trajectory == nullptr || !trajectory->active) return;
        if (action_idx < 0) {
            trajectory->active = false;
            trajectory->terminal = true;
            return;
        }

        const double parent_cost = state_cost(trajectory->state);
        const double parent_time = trajectory->state.sim_time;
        if (current_rollout_trace_enabled_) {
            trajectory->step_players.push_back(trajectory->player);
        }
        const bool detailed_perf = !in_.rollout_optimized_execution;
        const auto transition_begin = detailed_perf
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        if (trajectory->player == "controller") {
            if (prepared.compact_mode) {
                const auto& actions = prepared.compact_controller_actions;
                if (action_idx >= static_cast<int>(actions.size())) {
                    throw std::runtime_error(
                        "rollout controller action index out of range");
                }
                const auto& compact =
                    actions[static_cast<std::size_t>(action_idx)];
                ControllerAction action;
                action.token_budget = compact.token_budget;
                action.evicted_request_ids.assign(
                    compact.evicted_request_ids.begin(),
                    compact.evicted_request_ids.end());
                action.compact_prefill_allocations =
                    compact.compact_prefill_allocations;
                action.compact_decode_request_ids =
                    compact.compact_decode_request_ids;
                action.compact_allocations = compact.compact_allocations;
                action.mapping = compact.mapping;
                action.has_mapping = compact.has_mapping;
                action.valid = compact.valid;
                env_.apply_controller_action_inplace(
                    trajectory->state, action, true);
                if (current_rollout_trace_enabled_) {
                    trajectory->step_action_categories.push_back(
                        controller_rollout_category(action));
                }
            } else {
                const auto& actions =
                    prepared.node.controller_actions_by_index;
                if (action_idx >= static_cast<int>(actions.size())) {
                    throw std::runtime_error(
                        "rollout controller action index out of range");
                }
                const auto& action =
                    actions[static_cast<std::size_t>(action_idx)];
                env_.apply_controller_action_inplace(
                    trajectory->state, action, true);
                if (current_rollout_trace_enabled_) {
                    trajectory->step_action_categories.push_back(
                        controller_rollout_category(action));
                }
            }
        } else {
            const auto& actions = prepared.compact_mode
                ? prepared.compact_adversary_actions
                : prepared.node.adversary_actions_by_index;
            if (action_idx >= static_cast<int>(actions.size())) {
                throw std::runtime_error(
                    "rollout adversary action index out of range");
            }
            env_.apply_adversary_action_inplace(
                trajectory->state,
                actions[static_cast<std::size_t>(action_idx)]);
            if (current_rollout_trace_enabled_) {
                trajectory->step_action_categories.push_back("adversary");
            }
        }

        const double final_time =
            trajectory->state.stats.transition_final_time >= 0.0
                ? trajectory->state.stats.transition_final_time
                : trajectory->state.sim_time;
        PolicyRolloutEdge edge;
        edge.reward = parent_cost - state_cost(trajectory->state);
        edge.discount = time_discount(final_time, parent_time, in_);
        trajectory->edges.push_back(edge);
        if (current_rollout_trace_enabled_) {
            trajectory->step_sim_times_before.push_back(parent_time);
            trajectory->step_sim_times_after.push_back(trajectory->state.sim_time);
        }
        const auto& original_indices = prepared.compact_mode
            ? prepared.compact_original_indices
            : prepared.node.rollout_action_original_indices;
        const int original_action_idx =
            action_idx < static_cast<int>(original_indices.size())
                ? original_indices[static_cast<std::size_t>(action_idx)]
                : action_idx;
        trajectory->action_history.push_back(original_action_idx);
        trajectory->player = next_player(trajectory->player);
        if (detailed_perf) {
            const auto transition_end = std::chrono::steady_clock::now();
            trajectory->transition_sec += std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    transition_end - transition_begin).count();
        }
    }
    void copy_shared_rollout_transition_plain(
        PolicyRolloutTrajectory* target,
        const PolicyRolloutTrajectory& source) {
        if (target == nullptr || !target->active) return;
        const bool detailed_perf = !in_.rollout_optimized_execution;
        const auto copy_begin = detailed_perf
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        target->state = source.state;
        target->active = source.active;
        target->terminal = source.terminal;
        target->player = source.player;
        if (!source.edges.empty()) {
            target->edges.push_back(source.edges.back());
        }
        if (!source.action_history.empty()) {
            target->action_history.push_back(source.action_history.back());
        }
        if (current_rollout_trace_enabled_) {
            if (!source.step_players.empty()) {
                target->step_players.push_back(source.step_players.back());
            }
            if (!source.step_action_categories.empty()) {
                target->step_action_categories.push_back(
                    source.step_action_categories.back());
            }
            if (!source.step_sim_times_before.empty()) {
                target->step_sim_times_before.push_back(
                    source.step_sim_times_before.back());
            }
            if (!source.step_sim_times_after.empty()) {
                target->step_sim_times_after.push_back(
                    source.step_sim_times_after.back());
            }
        }
        if (detailed_perf) {
            const auto copy_end = std::chrono::steady_clock::now();
            target->transition_sec += std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    copy_end - copy_begin).count();
        }
    }

    struct IndependentRolloutPerf {
        double policy_sec = 0.0;
        double prepare_sec = 0.0;
        double initialize_sec = 0.0;
        double state_features_sec = 0.0;
        double action_features_sec = 0.0;
        double score_sec = 0.0;
        double sample_sec = 0.0;
        std::int64_t prepared_count = 0;
        std::int64_t action_rows = 0;
        std::int64_t scored_action_rows = 0;
        bool exceeded_max_actions = false;
    };

    void run_independent_rollout_trajectory_plain(
        PolicyRolloutTrajectory* trajectory,
        double target_time,
        int max_actions,
        int policy_parallel_threads,
        IndependentRolloutPerf* perf) {
        if (trajectory == nullptr || perf == nullptr) return;
        RolloutPreparedPolicy prepared;
        std::vector<int> group_offsets = {0, 0};
        for (int step = 0; step < max_actions; ++step) {
            if (!trajectory->active ||
                trajectory->state.sim_time >= target_time) {
                trajectory->active = false;
                break;
            }

            const auto policy_begin = std::chrono::steady_clock::now();
            reset_rollout_prepared_policy_plain(&prepared);
            prepare_rollout_policy_plain(&prepared, *trajectory);
            const auto prepare_end = std::chrono::steady_clock::now();
            perf->prepare_sec += std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    prepare_end - policy_begin).count();
            perf->initialize_sec += prepared.initialize_sec;
            perf->state_features_sec += prepared.state_features_sec;
            perf->action_features_sec += prepared.action_features_sec;
            ++perf->prepared_count;

            const auto score_begin = prepare_end;
            if (!prepared.flat_actions.empty()) {
                const NativeHGBModelRuntime* prior_runtime =
                    prepared.controller_player
                        ? controller_prior_runtime_
                        : adversary_prior_runtime_;
                if (prior_runtime == nullptr || !prior_runtime->loaded()) {
                    throw std::runtime_error(
                        "native independent rollout policy runtime is unavailable");
                }
                const int row_count = static_cast<int>(
                    prepared.canonical_indices.size());
                group_offsets[1] = row_count;
                std::vector<double> scores = prepared.markov_policy
                    ? (in_.cross_game_inference_batcher != nullptr
                        ? in_.cross_game_inference_batcher->predict_markov_policy(
                            *prior_runtime,
                            prepared.controller_player
                                ? "controller" : "adversary",
                            {prepared.markov_features},
                            prepared.flat_actions,
                            row_count,
                            group_offsets)
                        : prior_runtime->predict_markov_policy_grouped_batch(
                            {prepared.markov_features},
                            prepared.flat_actions,
                            row_count,
                            group_offsets,
                            std::max(1, policy_parallel_threads)))
                    : prior_runtime->predict_raw_grouped_split_batch_flat(
                        prepared.state_features,
                        prepared.flat_actions,
                        row_count,
                        group_offsets,
                        std::max(1, policy_parallel_threads));
                if (scores.size() != prepared.canonical_indices.size()) {
                    throw std::runtime_error(
                        "native independent rollout policy output size mismatch");
                }
                for (std::size_t row = 0;
                     row < prepared.valid_rows.size();
                     ++row) {
                    if (!prepared.valid_rows[row]) scores[row] = 0.0;
                }
                apply_rollout_policy_scores_plain(&prepared, scores);
                perf->action_rows += row_count;
                perf->scored_action_rows += row_count;
            }
            const auto score_end = std::chrono::steady_clock::now();
            perf->score_sec += std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    score_end - score_begin).count();

            const auto sample_begin = score_end;
            prepared.action_index = sample_rollout_action_plain(
                prepared, &trajectory->rng);
            const auto sample_end = std::chrono::steady_clock::now();
            perf->sample_sec += std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    sample_end - sample_begin).count();
            perf->policy_sec += std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    sample_end - policy_begin).count();
            apply_prepared_rollout_action_plain(
                trajectory, prepared, prepared.action_index);
            if (step + 1 == max_actions) {
                perf->exceeded_max_actions = true;
            }
        }
    }

    double policy_rollout_deadline_plain(
        double expansion_parent_time) const {
        return expansion_parent_time +
            std::max(0.0, in_.rollout_horizon_sec);
    }

    RolloutValueParts policy_rollout_value_plain(
        const SimState& leaf_state,
        const std::string& leaf_player,
        double expansion_parent_time,
        double target_time) {
        const int count = std::max(1, in_.rollout_count);
        const int policy_parallel_threads = std::max(
            1, in_.rollout_policy_parallel_threads > 0
                ? in_.rollout_policy_parallel_threads
                : in_.rollout_parallel_threads);
        const int trajectory_parallel_threads = std::max(
            1, std::min(
                count, std::max(1, in_.rollout_parallel_threads)));
        const int max_actions = std::max(1, in_.rollout_max_actions);
        perf_rollout_min_expansion_parent_time_ = std::min(
            perf_rollout_min_expansion_parent_time_, expansion_parent_time);
        perf_rollout_max_expansion_parent_time_ = std::max(
            perf_rollout_max_expansion_parent_time_, expansion_parent_time);
        const double remaining_rollout =
            std::max(0.0, target_time - leaf_state.sim_time);
        perf_rollout_min_remaining_sec_ = std::min(
            perf_rollout_min_remaining_sec_, remaining_rollout);
        perf_rollout_max_remaining_sec_ = std::max(
            perf_rollout_max_remaining_sec_, remaining_rollout);
        perf_rollout_min_start_time_ =
            std::min(perf_rollout_min_start_time_, leaf_state.sim_time);
        perf_rollout_max_start_time_ =
            std::max(perf_rollout_max_start_time_, leaf_state.sim_time);
        perf_rollout_min_deadline_ =
            std::min(perf_rollout_min_deadline_, target_time);
        perf_rollout_max_deadline_ =
            std::max(perf_rollout_max_deadline_, target_time);
        auto& trajectories = rollout_trajectories_workspace_;
        trajectories.resize(static_cast<std::size_t>(count));
        const std::uint64_t leaf_index =
            static_cast<std::uint64_t>(perf_rollout_leaf_count_++);
        const std::uint64_t base_seed =
            static_cast<std::uint64_t>(std::max(0, in_.seed));
        for (std::size_t idx = 0; idx < trajectories.size(); ++idx) {
            auto& trajectory = trajectories[idx];
            trajectory.state = leaf_state;
            trajectory.player = leaf_player;
            trajectory.edges.clear();
            trajectory.action_history.clear();
            trajectory.step_players.clear();
            trajectory.step_action_categories.clear();
            trajectory.step_sim_times_before.clear();
            trajectory.step_sim_times_after.clear();
            trajectory.active = true;
            trajectory.terminal = false;
            trajectory.policy_sec = 0.0;
            trajectory.transition_sec = 0.0;
            trajectory.action_history.reserve(static_cast<std::size_t>(max_actions));
            if (in_.capture_rollout_trace) {
                trajectory.step_players.reserve(static_cast<std::size_t>(max_actions));
                trajectory.step_action_categories.reserve(
                    static_cast<std::size_t>(max_actions));
                trajectory.step_sim_times_before.reserve(
                    static_cast<std::size_t>(max_actions));
                trajectory.step_sim_times_after.reserve(
                    static_cast<std::size_t>(max_actions));
            }
            trajectory.rng.seed_int(
                base_seed + leaf_index * 1000003ULL + idx * 9176ULL);
        }

        if (in_.rollout_optimized_execution &&
            trajectory_parallel_threads > 1) {
            std::vector<IndependentRolloutPerf> independent_perf(
                static_cast<std::size_t>(count));
            #pragma omp parallel for if(trajectory_parallel_threads > 1) \
                num_threads(trajectory_parallel_threads) schedule(static)
            for (int idx = 0; idx < count; ++idx) {
                run_independent_rollout_trajectory_plain(
                    &trajectories[static_cast<std::size_t>(idx)],
                    target_time,
                    max_actions,
                    policy_parallel_threads,
                    &independent_perf[static_cast<std::size_t>(idx)]);
            }

            bool exceeded_max_actions = false;
            double max_prepare_sec = 0.0;
            double max_score_sec = 0.0;
            double max_sample_sec = 0.0;
            for (std::size_t idx = 0; idx < independent_perf.size(); ++idx) {
                const auto& item = independent_perf[idx];
                trajectories[idx].policy_sec = item.policy_sec;
                max_prepare_sec = std::max(max_prepare_sec, item.prepare_sec);
                max_score_sec = std::max(max_score_sec, item.score_sec);
                max_sample_sec = std::max(max_sample_sec, item.sample_sec);
                perf_rollout_initialize_cpu_sec_ += item.initialize_sec;
                perf_rollout_state_features_cpu_sec_ += item.state_features_sec;
                perf_rollout_action_features_cpu_sec_ += item.action_features_sec;
                perf_rollout_prepared_unique_count_ += item.prepared_count;
                perf_rollout_policy_action_rows_ += item.action_rows;
                perf_rollout_policy_scored_action_rows_ +=
                    item.scored_action_rows;
                exceeded_max_actions =
                    exceeded_max_actions || item.exceeded_max_actions;
            }
            perf_rollout_policy_prepare_sec_ += max_prepare_sec;
            perf_rollout_policy_score_sec_ += max_score_sec;
            perf_rollout_policy_sample_sec_ += max_sample_sec;
            if (exceeded_max_actions) {
                throw std::runtime_error(
                    "policy rollout exceeded rollout_max_actions");
            }
        } else {

        auto& prepared = rollout_prepared_workspace_;
        prepared.resize(static_cast<std::size_t>(count));
        auto& batch_workspace = rollout_batch_workspace_;
        auto& prepared_owner = rollout_prepared_owner_workspace_;
        prepared_owner.assign(static_cast<std::size_t>(count), -1);
        bool any_active = true;
        bool exceeded_max_actions = false;
        for (int step = 0; step < max_actions; ++step) {
            any_active = false;
            for (auto& trajectory : trajectories) {
                if (trajectory.active &&
                    trajectory.state.sim_time < target_time) {
                    any_active = true;
                } else if (trajectory.active) {
                    trajectory.active = false;
                }
            }
            if (!any_active) break;

            std::fill(prepared_owner.begin(), prepared_owner.end(), -1);
            for (int idx = 0; idx < count; ++idx) {
                if (!trajectories[static_cast<std::size_t>(idx)].active) continue;
                int owner = idx;
                for (int previous = 0; previous < idx; ++previous) {
                    if (!trajectories[static_cast<std::size_t>(previous)].active) {
                        continue;
                    }
                    if (trajectories[static_cast<std::size_t>(idx)].action_history ==
                        trajectories[static_cast<std::size_t>(previous)].action_history) {
                        owner = prepared_owner[static_cast<std::size_t>(previous)];
                        break;
                    }
                }
                prepared_owner[static_cast<std::size_t>(idx)] = owner;
                if (owner == idx) {
                    ++perf_rollout_prepared_unique_count_;
                } else {
                    ++perf_rollout_prepared_reuse_count_;
                }
            }

            const auto policy_begin = std::chrono::steady_clock::now();
            #pragma omp parallel for if(trajectory_parallel_threads > 1) \
                num_threads(trajectory_parallel_threads) schedule(static)
            for (int idx = 0; idx < count; ++idx) {
                auto& item = prepared[static_cast<std::size_t>(idx)];
                reset_rollout_prepared_policy_plain(&item);
                if (trajectories[static_cast<std::size_t>(idx)].active &&
                    prepared_owner[static_cast<std::size_t>(idx)] == idx) {
                    prepare_rollout_policy_plain(
                        &item,
                        trajectories[static_cast<std::size_t>(idx)]);
                }
            }
            for (const auto& item : prepared) {
                perf_rollout_initialize_cpu_sec_ += item.initialize_sec;
                perf_rollout_state_features_cpu_sec_ += item.state_features_sec;
                perf_rollout_action_features_cpu_sec_ += item.action_features_sec;
            }

            const auto prepare_end = std::chrono::steady_clock::now();
            perf_rollout_policy_prepare_sec_ += std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    prepare_end - policy_begin).count();
            const auto score_begin = prepare_end;

            score_grouped_rollout_policies_plain(
                &prepared,
                trajectories,
                "controller",
                policy_parallel_threads,
                &batch_workspace);
            score_grouped_rollout_policies_plain(
                &prepared,
                trajectories,
                "adversary",
                policy_parallel_threads,
                &batch_workspace);
            const auto score_end = std::chrono::steady_clock::now();
            perf_rollout_policy_score_sec_ += std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    score_end - score_begin).count();
            const auto sample_begin = score_end;

            for (int idx = 0; idx < count; ++idx) {
                auto& trajectory =
                    trajectories[static_cast<std::size_t>(idx)];
                if (!trajectory.active) continue;
                auto& item = prepared[static_cast<std::size_t>(idx)];
                const int owner = prepared_owner[static_cast<std::size_t>(idx)];
                if (owner < 0) {
                    item.action_index = -1;
                } else {
                    item.action_index = sample_rollout_action_plain(
                        prepared[static_cast<std::size_t>(owner)],
                        &trajectory.rng);
                }
            }
            const auto policy_end = std::chrono::steady_clock::now();
            perf_rollout_policy_sample_sec_ += std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    policy_end - sample_begin).count();
            perf_rollout_policy_sec_ += std::chrono::duration_cast<
                std::chrono::duration<double>>(
                    policy_end - policy_begin).count();

            if (in_.rollout_optimized_execution) {
                auto& transition_owner = rollout_transition_owner_workspace_;
                transition_owner.assign(static_cast<std::size_t>(count), -1);
                for (int idx = 0; idx < count; ++idx) {
                    if (!trajectories[static_cast<std::size_t>(idx)].active) {
                        continue;
                    }
                    const int policy_owner =
                        prepared_owner[static_cast<std::size_t>(idx)];
                    const int action_index =
                        prepared[static_cast<std::size_t>(idx)].action_index;
                    int owner = idx;
                    if (policy_owner >= 0 && action_index >= 0) {
                        for (int previous = 0; previous < idx; ++previous) {
                            if (transition_owner[static_cast<std::size_t>(previous)] !=
                                previous) {
                                continue;
                            }
                            if (prepared_owner[static_cast<std::size_t>(previous)] ==
                                    policy_owner &&
                                prepared[static_cast<std::size_t>(previous)].action_index ==
                                    action_index) {
                                owner = previous;
                                break;
                            }
                        }
                    }
                    transition_owner[static_cast<std::size_t>(idx)] = owner;
                    if (owner != idx) ++perf_rollout_shared_transition_count_;
                }

                #pragma omp parallel for if(trajectory_parallel_threads > 1) \
                    num_threads(trajectory_parallel_threads) schedule(static)
                for (int idx = 0; idx < count; ++idx) {
                    auto& trajectory =
                        trajectories[static_cast<std::size_t>(idx)];
                    if (!trajectory.active ||
                        transition_owner[static_cast<std::size_t>(idx)] != idx) {
                        continue;
                    }
                    const int policy_owner =
                        prepared_owner[static_cast<std::size_t>(idx)];
                    if (policy_owner < 0) {
                        trajectory.active = false;
                        trajectory.terminal = true;
                        continue;
                    }
                    apply_prepared_rollout_action_plain(
                        &trajectory,
                        prepared[static_cast<std::size_t>(policy_owner)],
                        prepared[static_cast<std::size_t>(idx)].action_index);
                }

                #pragma omp parallel for if(trajectory_parallel_threads > 1) \
                    num_threads(trajectory_parallel_threads) schedule(static)
                for (int idx = 0; idx < count; ++idx) {
                    auto& trajectory =
                        trajectories[static_cast<std::size_t>(idx)];
                    if (!trajectory.active) continue;
                    const int owner =
                        transition_owner[static_cast<std::size_t>(idx)];
                    if (owner >= 0 && owner != idx) {
                        copy_shared_rollout_transition_plain(
                            &trajectory,
                            trajectories[static_cast<std::size_t>(owner)]);
                    }
                }
            } else {
                #pragma omp parallel for if(trajectory_parallel_threads > 1) \
                    num_threads(trajectory_parallel_threads) schedule(static)
                for (int idx = 0; idx < count; ++idx) {
                    auto& trajectory =
                        trajectories[static_cast<std::size_t>(idx)];
                    if (!trajectory.active) continue;
                    const auto& item =
                        prepared[static_cast<std::size_t>(idx)];
                    const int owner =
                        prepared_owner[static_cast<std::size_t>(idx)];
                    apply_prepared_rollout_action_plain(
                        &trajectory,
                        prepared[static_cast<std::size_t>(owner)],
                        item.action_index);
                }
            }

            if (step + 1 == max_actions) exceeded_max_actions = true;
        }
        if (exceeded_max_actions) {
            throw std::runtime_error("policy rollout exceeded rollout_max_actions");
        }
        }

        std::vector<double> bootstraps(trajectories.size(), 0.0);
        if (model_bootstrap_enabled()) {
            for (const std::string player : {"controller", "adversary"}) {
                std::vector<std::size_t> selected;
                std::vector<SimState> states;
                for (std::size_t idx = 0; idx < trajectories.size(); ++idx) {
                    if (trajectories[idx].player != player) continue;
                    selected.push_back(idx);
                    states.push_back(trajectories[idx].state);
                }
                if (states.empty()) continue;
                const auto infer_begin = std::chrono::steady_clock::now();
                const auto values = infer_values_for_states(
                    states, player, all_true_action_mask(player));
                const auto infer_end = std::chrono::steady_clock::now();
                perf_infer_sec_ += std::chrono::duration_cast<
                    std::chrono::duration<double>>(infer_end - infer_begin).count();
                ++perf_infer_calls_;
                if (values.size() != selected.size()) {
                    throw std::runtime_error("rollout bootstrap batch size mismatch");
                }
                for (std::size_t i = 0; i < values.size(); ++i) {
                    bootstraps[selected[i]] = values[i];
                }
            }
        }

        RolloutValueParts total;
        if (leaf_index == 0 && !trajectories.empty()) {
            std::uint32_t history_hash = 2166136261U;
            for (int action_index : trajectories.front().action_history) {
                history_hash = static_cast<std::uint32_t>(
                    (history_hash ^ static_cast<std::uint32_t>(action_index)) *
                    16777619U);
            }
            perf_rollout_first_history_hash_ = history_hash;
            perf_rollout_first_history_actions_ = static_cast<int>(
                trajectories.front().action_history.size());
        }
        for (std::size_t idx = 0; idx < trajectories.size(); ++idx) {
            perf_rollout_action_count_ +=
                static_cast<int>(trajectories[idx].edges.size());
            perf_rollout_policy_sec_ += trajectories[idx].policy_sec;
            perf_rollout_transition_sec_ += trajectories[idx].transition_sec;
            perf_rollout_min_final_time_ = std::min(
                perf_rollout_min_final_time_, trajectories[idx].state.sim_time);
            perf_rollout_max_final_time_ = std::max(
                perf_rollout_max_final_time_, trajectories[idx].state.sim_time);
            if (trajectories[idx].terminal) ++perf_rollout_terminal_count_;
            double reward_return = 0.0;
            double bootstrap_return = bootstraps[idx];
            for (auto edge = trajectories[idx].edges.rbegin();
                 edge != trajectories[idx].edges.rend(); ++edge) {
                reward_return = edge->reward + edge->discount * reward_return;
                bootstrap_return = edge->discount * bootstrap_return;
            }
            total.reward_return += reward_return;
            total.bootstrap_return += bootstrap_return;
        }
        perf_rollout_trajectory_count_ += count;
        if (current_rollout_trace_enabled_) {
            for (std::size_t idx = 0; idx < trajectories.size(); ++idx) {
                const auto& trajectory = trajectories[idx];
                double trajectory_reward_return = 0.0;
                double trajectory_bootstrap_return = bootstraps[idx];
                for (auto edge = trajectory.edges.rbegin();
                     edge != trajectory.edges.rend(); ++edge) {
                    trajectory_reward_return =
                        edge->reward +
                        edge->discount * trajectory_reward_return;
                    trajectory_bootstrap_return =
                        edge->discount * trajectory_bootstrap_return;
                }
                for (std::size_t path_idx =
                         current_rollout_trace_path_.size();
                     path_idx > 1; --path_idx) {
                    const TreeNode* child =
                        current_rollout_trace_path_[path_idx - 1];
                    if (child == nullptr) continue;
                    trajectory_reward_return =
                        child->reward +
                        child->edge_discount * trajectory_reward_return;
                    trajectory_bootstrap_return =
                        child->edge_discount * trajectory_bootstrap_return;
                }
                const double trajectory_total_return =
                    trajectory_reward_return +
                    trajectory_bootstrap_return;
                double cumulative_cost = 0.0;
                double running_discount = 1.0;
                int trace_step_number = 0;

                for (std::size_t path_idx = 1;
                     path_idx < current_rollout_trace_path_.size();
                     ++path_idx) {
                    const TreeNode* parent =
                        current_rollout_trace_path_[path_idx - 1];
                    const TreeNode* child =
                        current_rollout_trace_path_[path_idx];
                    if (parent == nullptr || child == nullptr) continue;
                    const double step_cost = -child->reward;
                    const double discounted_step_cost =
                        running_discount * step_cost;
                    cumulative_cost += discounted_step_cost;

                    RolloutTraceStep step;
                    step.sim_iteration = current_sim_iteration_;
                    step.leaf_evaluation_id = static_cast<int>(leaf_index);
                    step.rollout_id = static_cast<int>(idx);
                    step.root_action_index = current_root_action_index_;
                    step.step_number = trace_step_number++;
                    step.step_action_index = child->parent_action_index;
                    step.root_sim_time = current_root_sim_time_;
                    step.rollout_deadline = target_time;
                    step.step_sim_time_before = parent->sim_time;
                    step.step_sim_time_after = child->sim_time;
                    step.step_cost = step_cost;
                    step.discounted_step_cost = discounted_step_cost;
                    step.cumulative_discounted_cost = cumulative_cost;
                    step.trajectory_reward_return_from_root =
                        trajectory_reward_return;
                    step.trajectory_bootstrap_return_from_root =
                        trajectory_bootstrap_return;
                    step.trajectory_total_return_from_root =
                        trajectory_total_return;
                    step.root_action_category = current_root_action_category_;
                    step.step_phase = "tree_path";
                    step.step_player = parent->player;
                    step.step_action_category =
                        child->parent_action_is_controller
                            ? controller_rollout_category(
                                  child->parent_controller_action)
                            : "adversary";
                    rollout_trace_steps_.push_back(std::move(step));
                    running_discount *= child->edge_discount;
                }

                for (std::size_t edge_idx = 0;
                     edge_idx < trajectory.edges.size(); ++edge_idx) {
                    const auto& edge = trajectory.edges[edge_idx];
                    const double step_cost = -edge.reward;
                    const double discounted_step_cost =
                        running_discount * step_cost;
                    cumulative_cost += discounted_step_cost;

                    RolloutTraceStep step;
                    step.sim_iteration = current_sim_iteration_;
                    step.leaf_evaluation_id = static_cast<int>(leaf_index);
                    step.rollout_id = static_cast<int>(idx);
                    step.root_action_index = current_root_action_index_;
                    step.step_number = trace_step_number++;
                    step.step_action_index = trajectory.action_history[edge_idx];
                    step.root_sim_time = current_root_sim_time_;
                    step.rollout_deadline = target_time;
                    step.step_sim_time_before =
                        trajectory.step_sim_times_before[edge_idx];
                    step.step_sim_time_after =
                        trajectory.step_sim_times_after[edge_idx];
                    step.step_cost = step_cost;
                    step.discounted_step_cost = discounted_step_cost;
                    step.cumulative_discounted_cost = cumulative_cost;
                    step.trajectory_reward_return_from_root =
                        trajectory_reward_return;
                    step.trajectory_bootstrap_return_from_root =
                        trajectory_bootstrap_return;
                    step.trajectory_total_return_from_root =
                        trajectory_total_return;
                    step.root_action_category = current_root_action_category_;
                    step.step_phase = "policy_rollout";
                    step.step_player = trajectory.step_players[edge_idx];
                    step.step_action_category =
                        trajectory.step_action_categories[edge_idx];
                    rollout_trace_steps_.push_back(std::move(step));
                    running_discount *= edge.discount;
                }
            }
        }
        total.reward_return /= static_cast<double>(count);
        total.bootstrap_return /= static_cast<double>(count);
        return total;
    }

    RolloutValueParts rollout_value_plain(
        const SimState& state,
        const std::string& player,
        double expansion_parent_time,
        double target_time) {
        if (in_.search_mode == "full_tree_rollout" && in_.rollout_horizon_sec > 0.0) {
            return policy_rollout_value_plain(
                state, player, expansion_parent_time, target_time);
        }
        if (!model_bootstrap_enabled()) return {};

        const std::vector<uint8_t> mask = all_true_action_mask(player);
        const auto t_infer_begin = std::chrono::steady_clock::now();
        auto infer_out = infer_value_and_priors(state, player, mask);
        const auto t_infer_end = std::chrono::steady_clock::now();
        perf_infer_sec_ += std::chrono::duration_cast<std::chrono::duration<double>>(t_infer_end - t_infer_begin).count();
        perf_infer_calls_ += 1;
        return {0.0, infer_out.first};
    }

    void update_plain_node_bounds(TreeNode* node, double value) {
        if (node == nullptr) return;
        node->min_value = std::min(node->min_value, value);
        node->max_value = std::max(node->max_value, value);
    }

    void backpropagate_plain(
        const std::vector<TreeNode*>& path,
        RolloutValueParts leaf_value) {
        double reward_return = leaf_value.reward_return;
        double bootstrap_return = leaf_value.bootstrap_return;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            TreeNode* node = *it;
            if (node->parent != nullptr) {
                reward_return =
                    node->reward + node->edge_discount * reward_return;
                bootstrap_return = node->edge_discount * bootstrap_return;
            }
            const double value = reward_return + bootstrap_return;

            node->visits += 1;
            node->value_sum += value;
            node->rollout_reward_value_sum += reward_return;
            node->rollout_bootstrap_value_sum += bootstrap_return;

            if (node->parent != nullptr) {
                update_plain_node_bounds(node->parent, value);
            }
        }
    }

    SearchOutput run_full_tree_search() {
        using clock = std::chrono::steady_clock;
        const auto t_total_begin = clock::now();
        rollout_policy_score_cache_.clear();
        rollout_policy_scores_inflight_.clear();
        rollout_policy_cache_active_ = !in_.rollout_optimized_execution &&
            in_.search_mode == "full_tree_rollout" &&
            in_.rollout_horizon_sec > 0.0;

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

        if (in_.capture_root_puct_trace) {
            capture_root_puct_snapshot(root, 0, &out.root_puct_trace_steps);
        }

        const int iterations = std::max(1, in_.iterations);
        for (int sim = 0; sim < iterations; ++sim) {
            (void)sim;
            TreeNode* node = &root;
            SimState state = root.cached_state;
            std::vector<TreeNode*> path;
            path.push_back(node);
            double rollout_parent_time = state.sim_time;
            double rollout_deadline = state.sim_time;
            bool expanded_action = false;

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

                    rollout_parent_time = state.sim_time;
                    rollout_deadline =
                        policy_rollout_deadline_plain(rollout_parent_time);
                    auto expanded = expand_one_child_plain(
                        node, state, selected.action_index);
                    node = expanded.first;
                    state = std::move(expanded.second);
                    path.push_back(node);
                    expanded_action = true;
                    break;
                }
            } else {
                while (plain_expanded(*node) &&
                       node->untried_action_indices.empty() &&
                       !node->children.empty()) {
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
                    rollout_parent_time = state.sim_time;
                    rollout_deadline =
                        policy_rollout_deadline_plain(rollout_parent_time);
                    auto expanded = expand_one_child_plain(node, state);
                    node = expanded.first;
                    state = std::move(expanded.second);
                    path.push_back(node);
                    expanded_action = true;
                }
            }

            if (node == nullptr) continue;
            current_sim_iteration_ = sim;
            current_rollout_trace_enabled_ = false;
            current_rollout_trace_path_.clear();
            if (in_.capture_rollout_trace && expanded_action &&
                path.size() >= 2 && root.player == "controller") {
                TreeNode* root_child = path[1];
                if (root_child != nullptr &&
                    root_child->has_parent_action &&
                    root_child->parent_action_is_controller) {
                    const std::string category = controller_rollout_category(
                        root_child->parent_controller_action);
                    if (category == "decode_only" ||
                        category == "prefill_128" ||
                        category == "prefill_256" ||
                        category == "prefill_512") {
                        current_rollout_trace_enabled_ = true;
                        current_root_action_index_ =
                            root_child->parent_action_index;
                        current_root_action_category_ = category;
                        current_root_sim_time_ = root.sim_time;
                        current_rollout_trace_path_ = path;
                    }
                }
            }
            if (!expanded_action) {
                rollout_parent_time = state.sim_time;
                rollout_deadline = state.sim_time;
            }
            const RolloutValueParts leaf_value = rollout_value_plain(
                state, node->player, rollout_parent_time, rollout_deadline);
            backpropagate_plain(path, leaf_value);
            if (in_.capture_root_puct_trace) {
                capture_root_puct_snapshot(
                    root, sim + 1, &out.root_puct_trace_steps);
            }
        }

        out.root_visits = root.visits;
        out.root_value_sum = root.value_sum;
        out.rollout_trace_steps = std::move(rollout_trace_steps_);
        out.mcts_root_prior = compute_root_prior(root, out.root_nn_valid_mask);

        int n_actions = static_cast<int>(out.root_nn_valid_mask.size());
        if (n_actions <= 0) n_actions = action_space_size_for_player(root.player);
        out.root_action_values.assign(static_cast<std::size_t>(n_actions), -std::numeric_limits<double>::infinity());
        out.root_action_rewards.assign(static_cast<std::size_t>(n_actions), 0.0);
        out.root_action_discounts.assign(static_cast<std::size_t>(n_actions), 1.0);
        out.root_action_bootstraps.assign(static_cast<std::size_t>(n_actions), 0.0);
        out.root_action_rollout_reward_returns.assign(
            static_cast<std::size_t>(n_actions), 0.0);
        out.root_action_rollout_bootstrap_returns.assign(
            static_cast<std::size_t>(n_actions), 0.0);
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
                out.root_action_rollout_reward_returns[static_cast<std::size_t>(alias)] =
                    child.visits > 0
                        ? child.rollout_reward_value_sum /
                            static_cast<double>(child.visits)
                        : 0.0;
                out.root_action_rollout_bootstrap_returns[static_cast<std::size_t>(alias)] =
                    child.visits > 0
                        ? child.rollout_bootstrap_value_sum /
                            static_cast<double>(child.visits)
                        : 0.0;
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
        out.perf["rollout_enabled"] =
            in_.search_mode == "full_tree_rollout" ? 1.0 : 0.0;
        out.perf["rollout_root_time"] = in_.root_state.sim_time;
        out.perf["rollout_deadline"] = -1.0;
        out.perf["rollout_min_deadline"] =
            std::isfinite(perf_rollout_min_deadline_) ? perf_rollout_min_deadline_ : -1.0;
        out.perf["rollout_max_deadline"] =
            std::isfinite(perf_rollout_max_deadline_) ? perf_rollout_max_deadline_ : -1.0;
        out.perf["rollout_min_expansion_parent_time"] =
            std::isfinite(perf_rollout_min_expansion_parent_time_)
                ? perf_rollout_min_expansion_parent_time_ : -1.0;
        out.perf["rollout_max_expansion_parent_time"] =
            std::isfinite(perf_rollout_max_expansion_parent_time_)
                ? perf_rollout_max_expansion_parent_time_ : -1.0;
        out.perf["rollout_min_remaining_rollout_sec"] =
            std::isfinite(perf_rollout_min_remaining_sec_)
                ? perf_rollout_min_remaining_sec_ : -1.0;
        out.perf["rollout_max_remaining_rollout_sec"] =
            std::isfinite(perf_rollout_max_remaining_sec_)
                ? perf_rollout_max_remaining_sec_ : -1.0;
        out.perf["rollout_first_history_hash"] =
            static_cast<double>(perf_rollout_first_history_hash_);
        out.perf["rollout_first_history_actions"] =
            static_cast<double>(perf_rollout_first_history_actions_);
        out.perf["rollout_cutoff_leaf_evaluations"] =
            static_cast<double>(perf_rollout_cutoff_leaf_count_);
        out.perf["rollout_min_start_time"] =
            std::isfinite(perf_rollout_min_start_time_)
                ? perf_rollout_min_start_time_ : -1.0;
        out.perf["rollout_max_start_time"] =
            std::isfinite(perf_rollout_max_start_time_)
                ? perf_rollout_max_start_time_ : -1.0;
        out.perf["rollout_min_final_time"] =
            std::isfinite(perf_rollout_min_final_time_)
                ? perf_rollout_min_final_time_ : -1.0;
        out.perf["rollout_max_final_time"] =
            std::isfinite(perf_rollout_max_final_time_)
                ? perf_rollout_max_final_time_ : -1.0;
        out.perf["rollout_trajectories"] =
            static_cast<double>(perf_rollout_trajectory_count_);
        out.perf["rollout_actions"] = static_cast<double>(perf_rollout_action_count_);
        out.perf["rollout_prepared_unique"] =
            static_cast<double>(perf_rollout_prepared_unique_count_);
        out.perf["rollout_prepared_reused"] =
            static_cast<double>(perf_rollout_prepared_reuse_count_);
        out.perf["rollout_shared_transitions"] =
            static_cast<double>(perf_rollout_shared_transition_count_);
        out.perf["rollout_terminals"] = static_cast<double>(perf_rollout_terminal_count_);
        out.perf["rollout_policy_cache_hits"] =
            static_cast<double>(perf_rollout_policy_cache_hits_);
        out.perf["rollout_policy_cache_misses"] =
            static_cast<double>(perf_rollout_policy_cache_misses_);
        out.perf["rollout_policy_sec"] = perf_rollout_policy_sec_;
        out.perf["rollout_policy_prepare_sec"] =
            perf_rollout_policy_prepare_sec_;
        out.perf["rollout_initialize_cpu_sec"] =
            perf_rollout_initialize_cpu_sec_;
        out.perf["rollout_state_features_cpu_sec"] =
            perf_rollout_state_features_cpu_sec_;
        out.perf["rollout_action_features_cpu_sec"] =
            perf_rollout_action_features_cpu_sec_;
        out.perf["rollout_policy_score_sec"] = perf_rollout_policy_score_sec_;
        out.perf["rollout_policy_sample_sec"] =
            perf_rollout_policy_sample_sec_;
        out.perf["rollout_policy_action_rows"] =
            static_cast<double>(perf_rollout_policy_action_rows_);
        out.perf["rollout_policy_scored_action_rows"] =
            static_cast<double>(perf_rollout_policy_scored_action_rows_);
        out.perf["rollout_transition_sec"] = perf_rollout_transition_sec_;
        return out;
    }

    const SearchInput& in_;
    GV2VirtualEnvironment env_;
    NativeTorchScriptInferRuntimeGV2* torch_runtime_ = nullptr;
    NewFeatures226HGBRuntime* hgb_runtime_ = nullptr;
    NativeHGBModelRuntime* controller_prior_runtime_ = nullptr;
    NativeHGBModelRuntime* adversary_prior_runtime_ = nullptr;
    bool rollout_policy_cache_active_ = false;
    mutable std::unordered_map<std::string, std::vector<double>>
        rollout_policy_score_cache_;
    std::unordered_map<std::uint64_t, std::vector<FastRolloutPolicyCacheEntry>>
        fast_rollout_policy_score_cache_;
    std::size_t fast_rollout_policy_cache_entry_count_ = 0;
    mutable std::unordered_set<std::string> rollout_policy_scores_inflight_;
    mutable std::mutex rollout_policy_cache_mutex_;
    mutable std::condition_variable rollout_policy_cache_cv_;
    std::vector<PolicyRolloutTrajectory> rollout_trajectories_workspace_;
    std::vector<RolloutPreparedPolicy> rollout_prepared_workspace_;
    RolloutPolicyBatchWorkspace rollout_batch_workspace_;
    std::vector<int> rollout_prepared_owner_workspace_;
    std::vector<int> rollout_transition_owner_workspace_;
    std::vector<RolloutTraceStep> rollout_trace_steps_;
    std::vector<TreeNode*> current_rollout_trace_path_;
    bool current_rollout_trace_enabled_ = false;
    int current_sim_iteration_ = -1;
    int current_root_action_index_ = -1;
    std::string current_root_action_category_;
    double current_root_sim_time_ = 0.0;
    int model_version_ = 0;
    std::mt19937 rng_;
    PythonRandomCompat py_rng_;
    int next_node_id_ = 1;
    MinMaxStats min_max_;

    double perf_rollout_min_start_time_ =
        std::numeric_limits<double>::infinity();
    double perf_rollout_max_start_time_ =
        -std::numeric_limits<double>::infinity();
    double perf_rollout_min_final_time_ =
        std::numeric_limits<double>::infinity();
    double perf_rollout_max_final_time_ =
        -std::numeric_limits<double>::infinity();
    double perf_rollout_min_deadline_ =
        std::numeric_limits<double>::infinity();
    double perf_rollout_max_deadline_ =
        -std::numeric_limits<double>::infinity();
    double perf_rollout_min_expansion_parent_time_ =
        std::numeric_limits<double>::infinity();
    double perf_rollout_max_expansion_parent_time_ =
        -std::numeric_limits<double>::infinity();
    double perf_rollout_min_remaining_sec_ =
        std::numeric_limits<double>::infinity();
    double perf_rollout_max_remaining_sec_ =
        -std::numeric_limits<double>::infinity();
    std::uint32_t perf_rollout_first_history_hash_ = 0U;
    int perf_rollout_first_history_actions_ = 0;
    // Perf counters
    double perf_selection_sec_ = 0.0;
    double perf_restore_sec_ = 0.0;
    double perf_forced_sec_ = 0.0;
    std::int64_t perf_rollout_cutoff_leaf_count_ = 0;
    double perf_expand_sec_ = 0.0;
    double perf_backprop_sec_ = 0.0;
    double perf_infer_sec_ = 0.0;
    double perf_rollout_policy_sec_ = 0.0;
    double perf_rollout_policy_prepare_sec_ = 0.0;
    double perf_rollout_initialize_cpu_sec_ = 0.0;
    double perf_rollout_state_features_cpu_sec_ = 0.0;
    double perf_rollout_action_features_cpu_sec_ = 0.0;
    double perf_rollout_policy_score_sec_ = 0.0;
    double perf_rollout_policy_sample_sec_ = 0.0;
    double perf_rollout_transition_sec_ = 0.0;
    int perf_infer_calls_ = 0;
    int perf_selection_steps_ = 0;
    int perf_forced_steps_ = 0;
    std::int64_t perf_rollout_trajectory_count_ = 0;
    std::int64_t perf_rollout_leaf_count_ = 0;
    std::int64_t perf_rollout_action_count_ = 0;
    std::int64_t perf_rollout_prepared_unique_count_ = 0;
    std::int64_t perf_rollout_policy_action_rows_ = 0;
    std::int64_t perf_rollout_policy_scored_action_rows_ = 0;
    std::int64_t perf_rollout_prepared_reuse_count_ = 0;
    std::int64_t perf_rollout_shared_transition_count_ = 0;
    std::int64_t perf_rollout_terminal_count_ = 0;
    mutable std::int64_t perf_rollout_policy_cache_hits_ = 0;
    mutable std::int64_t perf_rollout_policy_cache_misses_ = 0;

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

        const double fixed_alpha = in_.root_dirichlet_alpha;
        const double total_concentration = in_.root_dirichlet_total_concentration;
        double eps = in_.root_dirichlet_epsilon;

        if ((fixed_alpha <= 0.0 && total_concentration <= 0.0) || eps <= 0.0) return;
        eps = clampv(eps, 0.0, 1.0);

        std::vector<int> child_indices;
        child_indices.reserve(root->children.size());
        for (const auto& kv : root->children) child_indices.push_back(kv.first);
        std::sort(child_indices.begin(), child_indices.end());

        const int n = static_cast<int>(child_indices.size());
        const double alpha = resolve_root_dirichlet_alpha(
            fixed_alpha,
            total_concentration,
            n);
        if (alpha <= 0.0) return;
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

            std::unordered_map<std::uint64_t, std::vector<int>> hash_to_canons;
            std::unordered_map<int, std::vector<int>> canon_to_alias;
            std::unordered_map<int, int> alias_to_canon;
            for (int idx : valid) {
                const auto& act = sampled.actions[static_cast<std::size_t>(idx)];
                const std::uint64_t hash = controller_action_hash(act);
                std::vector<int>& candidates = hash_to_canons[hash];
                int canonical = -1;
                for (int candidate : candidates) {
                    if (controller_actions_equivalent(
                            act, sampled.actions[static_cast<std::size_t>(candidate)])) {
                        canonical = candidate;
                        break;
                    }
                }
                if (canonical < 0) {
                    candidates.push_back(idx);
                    canon_to_alias[idx] = {idx};
                    alias_to_canon[idx] = idx;
                } else {
                    canon_to_alias[canonical].push_back(idx);
                    alias_to_canon[idx] = canonical;
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
    if (prior_in.search_mode != "full_tree_rollout") {
        prior_in.search_mode = "full_tree";
    }
    prior_in.use_policy_prior = true;
    if (!std::isfinite(prior_in.puct_c) || prior_in.puct_c == 0.0) {
        prior_in.puct_c = 2.5;
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
