#include "new_features_226_inference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mcts_native_gv2 {
namespace {

constexpr int kFeatureDim = 226;
constexpr int kGlobalDim = 23;
constexpr int kPrefillSlots = 7;
constexpr int kDecodeSlots = 7;
constexpr int kPrefillSlotDim = 25;
constexpr int kDecodeSlotDim = 4;
constexpr int kLateBuckets = 4;
constexpr int kSlackBuckets = 17;
constexpr int kMetaNextAdvTick = -9100001;
constexpr int kMetaDecodeCreditBalance = -9100005;

constexpr double kActivePrefillCountDen = 20.0;
constexpr double kActiveDecodeCountDen = 100.0;
constexpr double kActiveTotalCountDen = 120.0;
constexpr double kTotalRemainingPrefillDen = 20.0 * 4096.0;
constexpr double kTotalRemainingDecodeDen = 100.0 * 864.0;
constexpr double kTotalDecodeGeneratedActiveDen = 100.0 * 864.0;
constexpr double kViolatedCountDen = 100.0;
constexpr double kPrefillNearDropDen = 20.0;
constexpr double kDecodeNearDropDen = 100.0;
constexpr double kRecentLaunchCountDen = 7.0;
constexpr double kRecentLaunchPrefillDen = 1024.0 * 7.0;
constexpr double kDecodeCreditDen = 100.0 * 216.0;
constexpr double kNearDropLowSec = 0.5;
constexpr double kNearDropHighSec = 1.5;
constexpr double kLaunchEwmaWindowSec = 1.0;
constexpr double kLaunchEwmaAlpha = 0.37;
constexpr double kAdvTickSecDen = 0.2;
constexpr double kDecodeRemainingDen = 864.0;

const double kSlackEdges[] = {
    0.0133,
    0.0135,
    0.0137,
    0.0140,
    0.0145,
    0.015725797204323228,
    0.023274675327417962,
    0.031963770276289896,
    0.03888623299112536,
    0.06091750997076902,
    0.07016849423102292,
    0.08612442901238251,
    0.09850190759874299,
    0.19613847773632437,
    0.28408680179190937,
};

std::string trim_copy(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::vector<std::string> split_tab(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == '\t') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

double norm01(double x, double denom) {
    if (denom <= 0.0) return 0.0;
    return clamp(x / denom, 0.0, 1.0);
}

bool contains_id(const std::vector<int>& ids, int rid) {
    return std::find(ids.begin(), ids.end(), rid) != ids.end();
}

double map_get(const std::unordered_map<int, double>& m, int k, double dflt = 0.0) {
    const auto it = m.find(k);
    return it == m.end() ? dflt : it->second;
}

int map_get(const std::unordered_map<int, int>& m, int k, int dflt = 0) {
    const auto it = m.find(k);
    return it == m.end() ? dflt : it->second;
}

int slack_bucket_idx(double slack) {
    if (slack <= 0.0) return 0;
    for (int i = 0; i < static_cast<int>(sizeof(kSlackEdges) / sizeof(kSlackEdges[0])); ++i) {
        if (slack <= kSlackEdges[i]) return i + 1;
    }
    return kSlackBuckets - 1;
}

int lateness_bucket_idx(double late) {
    if (late < 0.5) return 0;
    if (late < 1.0) return 1;
    if (late < 1.5) return 2;
    return 3;
}

bool is_prefill_req(const RequestState& r) {
    if (r.completed) return false;
    if (r.prefill_done()) return false;
    return true;
}

bool is_decode_req(const RequestState& r) {
    if (r.completed) return false;
    if (!r.prefill_done()) return false;
    return r.remaining_decode() > 0;
}

struct FeatureBuildScratch {
    std::vector<const RequestState*> active_requests;
    std::vector<const RequestState*> prefill_requests;
    std::vector<const RequestState*> decode_requests;
    std::vector<const RequestState*> non_violated_decode;
    std::vector<const RequestState*> chosen_decode;
};

bool contains_sorted_or_linear(
    const std::vector<int>& values,
    bool values_are_sorted,
    int request_id) {
    return values_are_sorted
        ? std::binary_search(values.begin(), values.end(), request_id)
        : contains_id(values, request_id);
}

double decode_remaining(const RequestState& r) {
    return static_cast<double>(std::max(0, r.remaining_decode()));
}

double decode_done(const RequestState& r) {
    return static_cast<double>(std::max(0, r.num_processed_decode_tokens));
}

double prefill_lateness_for_feature(const SimState& state, const RequestState& r) {
    double late = std::max(0.0, map_get(state.stats.per_request_prefill_lateness_by_id, r.request_id, 0.0));
    if (late <= 0.0) {
        const double deadline = r.arrived_at + r.prefill_slo_time;
        late = std::max(0.0, state.sim_time - deadline);
    }
    return late;
}

double decode_time_at_max(
    const SimState& state,
    const VirtualSimulatorGV2* simulator,
    const std::vector<const RequestState*>& decode_reqs) {
    if (decode_reqs.empty()) return 0.0;
    const double next_adv_tick = (state.stats.next_adv_tick >= 0.0)
        ? state.stats.next_adv_tick
        : map_get(
            state.stats.decode_next_deadline_by_id,
            kMetaNextAdvTick,
            state.stats.next_adv_tick);
    if (next_adv_tick >= 0.0 && next_adv_tick <= state.sim_time + 1e-9) {
        return 0.0;
    }
    if (state.stats.pending_adv_tick) {
        return 0.0;
    }
    if (simulator == nullptr) {
        return static_cast<double>(decode_reqs.size()) * 0.0009;
    }

    ControllerAction noop;
    noop.token_budget = 0;
    noop.strategy = "GV2|evict_none";
    noop.mapping = {0, 0, 0};
    noop.has_mapping = true;
    noop.valid = true;

    const ControllerBatchPlan plan = simulator->build_controller_batch_plan(
        state,
        noop,
        true,
        std::max(0, state.stats.decode_credit_balance));
    if (plan.predictor_reqs.empty()) return 0.0;
    const auto& cfg = simulator->cfg();
    const auto prediction = simulator->predictor().lookup_batch_time(
        plan.predictor_reqs,
        plan.num_tokens,
        cfg.prefill_profile_tokens,
        cfg.prefill_profile_times,
        cfg.fallback_total_time_sec,
        cfg.fallback_model_time_sec);
    return std::max(0.0, std::get<0>(prediction));
}

std::uint32_t rotr32(std::uint32_t x, unsigned r) {
    return (x >> r) | (x << ((32u - r) & 31u));
}

std::uint32_t sha256_ch(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return (x & y) ^ (~x & z);
}

std::uint32_t sha256_maj(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

std::uint32_t sha256_bs0(std::uint32_t x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

std::uint32_t sha256_bs1(std::uint32_t x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

std::uint32_t sha256_ss0(std::uint32_t x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

std::uint32_t sha256_ss1(std::uint32_t x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

std::array<std::uint8_t, 32> sha256_bytes(const std::string& input) {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    std::uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    std::vector<std::uint8_t> msg(input.begin(), input.end());
    const std::uint64_t bit_len = static_cast<std::uint64_t>(msg.size()) * 8ull;
    msg.push_back(0x80u);
    while ((msg.size() % 64u) != 56u) msg.push_back(0u);
    for (int shift = 56; shift >= 0; shift -= 8) {
        msg.push_back(static_cast<std::uint8_t>((bit_len >> shift) & 0xffu));
    }

    for (std::size_t offset = 0; offset < msg.size(); offset += 64u) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const std::size_t j = offset + static_cast<std::size_t>(i) * 4u;
            w[i] = (static_cast<std::uint32_t>(msg[j]) << 24) |
                   (static_cast<std::uint32_t>(msg[j + 1]) << 16) |
                   (static_cast<std::uint32_t>(msg[j + 2]) << 8) |
                   static_cast<std::uint32_t>(msg[j + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            w[i] = sha256_ss1(w[i - 2]) + w[i - 7] + sha256_ss0(w[i - 15]) + w[i - 16];
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        std::uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t t1 = hh + sha256_bs1(e) + sha256_ch(e, f, g) + k[i] + w[i];
            const std::uint32_t t2 = sha256_bs0(a) + sha256_maj(a, b, c);
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
        out[static_cast<std::size_t>(4 * i + 0)] = static_cast<std::uint8_t>((h[i] >> 24) & 0xffu);
        out[static_cast<std::size_t>(4 * i + 1)] = static_cast<std::uint8_t>((h[i] >> 16) & 0xffu);
        out[static_cast<std::size_t>(4 * i + 2)] = static_cast<std::uint8_t>((h[i] >> 8) & 0xffu);
        out[static_cast<std::size_t>(4 * i + 3)] = static_cast<std::uint8_t>(h[i] & 0xffu);
    }
    return out;
}

std::uint64_t sha256_seed64(const std::string& seed_key) {
    const auto digest = sha256_bytes(seed_key);
    std::uint64_t seed = 0;
    for (int i = 0; i < 8; ++i) {
        seed = (seed << 8) | static_cast<std::uint64_t>(digest[static_cast<std::size_t>(i)]);
    }
    return seed;
}

std::uint32_t seedseq_hashmix(std::uint32_t value, std::uint32_t& hash_const) {
    value ^= hash_const;
    hash_const *= 0x931e8875u;
    value *= hash_const;
    value ^= value >> 16;
    return value;
}

std::uint32_t seedseq_mix(std::uint32_t x, std::uint32_t y) {
    std::uint32_t result = 0xca01f9ddu * x - 0x4973f715u * y;
    result ^= result >> 16;
    return result;
}

std::array<std::uint64_t, 4> numpy_seedsequence_generate_u64(std::uint64_t seed) {
    std::vector<std::uint32_t> entropy;
    if (seed == 0) {
        entropy.push_back(0u);
    } else {
        while (seed > 0) {
            entropy.push_back(static_cast<std::uint32_t>(seed & 0xffffffffu));
            seed >>= 32;
        }
    }

    std::array<std::uint32_t, 4> pool{};
    std::uint32_t hash_const = 0x43b0d7e5u;
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t value = (i < static_cast<int>(entropy.size())) ? entropy[static_cast<std::size_t>(i)] : 0u;
        pool[static_cast<std::size_t>(i)] = seedseq_hashmix(value, hash_const);
    }
    for (int i_src = 0; i_src < 4; ++i_src) {
        for (int i_dst = 0; i_dst < 4; ++i_dst) {
            if (i_src == i_dst) continue;
            pool[static_cast<std::size_t>(i_dst)] = seedseq_mix(
                pool[static_cast<std::size_t>(i_dst)],
                seedseq_hashmix(pool[static_cast<std::size_t>(i_src)], hash_const));
        }
    }
    for (std::size_t i_src = 4; i_src < entropy.size(); ++i_src) {
        for (int i_dst = 0; i_dst < 4; ++i_dst) {
            pool[static_cast<std::size_t>(i_dst)] = seedseq_mix(
                pool[static_cast<std::size_t>(i_dst)],
                seedseq_hashmix(entropy[i_src], hash_const));
        }
    }

    std::array<std::uint32_t, 8> words32{};
    hash_const = 0x8b51f9ddu;
    for (int i = 0; i < 8; ++i) {
        std::uint32_t data_val = pool[static_cast<std::size_t>(i % 4)];
        data_val ^= hash_const;
        hash_const *= 0x58f38dedu;
        data_val *= hash_const;
        data_val ^= data_val >> 16;
        words32[static_cast<std::size_t>(i)] = data_val;
    }

    std::array<std::uint64_t, 4> out{};
    for (int i = 0; i < 4; ++i) {
        out[static_cast<std::size_t>(i)] =
            static_cast<std::uint64_t>(words32[static_cast<std::size_t>(2 * i)]) |
            (static_cast<std::uint64_t>(words32[static_cast<std::size_t>(2 * i + 1)]) << 32);
    }
    return out;
}

std::uint64_t rotr64(std::uint64_t value, unsigned rot) {
    return (value >> rot) | (value << ((-rot) & 63u));
}

class NumpyPCG64 {
public:
    explicit NumpyPCG64(std::uint64_t seed) {
        const auto val = numpy_seedsequence_generate_u64(seed);
        const auto initstate = (static_cast<unsigned __int128>(val[0]) << 64) | val[1];
        const auto initseq = (static_cast<unsigned __int128>(val[2]) << 64) | val[3];
        state_ = 0;
        inc_ = (initseq << 1u) | 1u;
        step();
        state_ += initstate;
        step();
    }

    std::uint64_t next64() {
        step();
        const std::uint64_t hi = static_cast<std::uint64_t>(state_ >> 64);
        const std::uint64_t lo = static_cast<std::uint64_t>(state_);
        return rotr64(hi ^ lo, static_cast<unsigned>(hi >> 58u));
    }

    std::uint32_t next32() {
        if (has_uint32_) {
            has_uint32_ = false;
            return uinteger_;
        }
        const std::uint64_t next = next64();
        has_uint32_ = true;
        uinteger_ = static_cast<std::uint32_t>(next >> 32);
        return static_cast<std::uint32_t>(next & 0xffffffffu);
    }

private:
    void step() {
        static constexpr unsigned __int128 kMultiplier =
            (static_cast<unsigned __int128>(2549297995355413924ull) << 64) |
            static_cast<unsigned __int128>(4865540595714422341ull);
        state_ = state_ * kMultiplier + inc_;
    }

    unsigned __int128 state_ = 0;
    unsigned __int128 inc_ = 0;
    bool has_uint32_ = false;
    std::uint32_t uinteger_ = 0;
};

std::uint64_t gen_mask64(std::uint64_t max_value) {
    std::uint64_t mask = max_value;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    return mask;
}

std::uint64_t numpy_bounded_uint64(NumpyPCG64& rng, std::uint64_t range_inclusive) {
    if (range_inclusive == 0) return 0;
    if (range_inclusive <= 0xffffffffull) {
        if (range_inclusive == 0xffffffffull) return rng.next32();
        const std::uint64_t range_exclusive = range_inclusive + 1ull;
        std::uint64_t m = static_cast<std::uint64_t>(rng.next32()) * range_exclusive;
        std::uint32_t leftover = static_cast<std::uint32_t>(m & 0xffffffffu);
        if (leftover < range_exclusive) {
            const std::uint32_t threshold = static_cast<std::uint32_t>((0xffffffffull - range_inclusive) % range_exclusive);
            while (leftover < threshold) {
                m = static_cast<std::uint64_t>(rng.next32()) * range_exclusive;
                leftover = static_cast<std::uint32_t>(m & 0xffffffffu);
            }
        }
        return m >> 32;
    }

    if (range_inclusive == 0xffffffffffffffffull) return rng.next64();
    const unsigned __int128 range_exclusive = static_cast<unsigned __int128>(range_inclusive) + 1u;
    unsigned __int128 m = static_cast<unsigned __int128>(rng.next64()) * range_exclusive;
    std::uint64_t leftover = static_cast<std::uint64_t>(m);
    if (leftover < range_exclusive) {
        const std::uint64_t threshold = static_cast<std::uint64_t>((0xffffffffffffffffull - range_inclusive) % static_cast<std::uint64_t>(range_exclusive));
        while (leftover < threshold) {
            m = static_cast<unsigned __int128>(rng.next64()) * range_exclusive;
            leftover = static_cast<std::uint64_t>(m);
        }
    }
    return static_cast<std::uint64_t>(m >> 64);
}

std::vector<int> deterministic_subset_indices(int n_total, int k, const std::string& seed_key) {
    std::vector<int> idxs;
    idxs.reserve(static_cast<std::size_t>(std::max(0, n_total)));
    for (int i = 0; i < n_total; ++i) idxs.push_back(i);
    if (n_total <= k) return idxs;

    NumpyPCG64 rng(sha256_seed64(seed_key));
    std::vector<int> sample;
    sample.reserve(static_cast<std::size_t>(k));

    std::uint64_t set_size = static_cast<std::uint64_t>(1.2 * static_cast<double>(k));
    std::uint64_t mask = gen_mask64(set_size);
    set_size = mask + 1ull;
    std::vector<std::uint64_t> hash_set(static_cast<std::size_t>(set_size), std::numeric_limits<std::uint64_t>::max());
    const std::uint64_t empty = std::numeric_limits<std::uint64_t>::max();

    for (std::uint64_t j = static_cast<std::uint64_t>(n_total - k); j < static_cast<std::uint64_t>(n_total); ++j) {
        const std::uint64_t val = numpy_bounded_uint64(rng, j);
        std::uint64_t loc = val & mask;
        while (hash_set[static_cast<std::size_t>(loc)] != empty && hash_set[static_cast<std::size_t>(loc)] != val) {
            loc = (loc + 1ull) & mask;
        }
        if (hash_set[static_cast<std::size_t>(loc)] == empty) {
            hash_set[static_cast<std::size_t>(loc)] = val;
            sample.push_back(static_cast<int>(val));
        } else {
            loc = j & mask;
            while (hash_set[static_cast<std::size_t>(loc)] != empty) {
                loc = (loc + 1ull) & mask;
            }
            hash_set[static_cast<std::size_t>(loc)] = j;
            sample.push_back(static_cast<int>(j));
        }
    }

    // NumPy's choice(..., replace=False) shuffles the Floyd sample before the
    // Python feature builder sorts it; consuming the same RNG draws matters.
    for (int i = k - 1; i >= 1; --i) {
        const std::uint64_t j = numpy_bounded_uint64(rng, static_cast<std::uint64_t>(i));
        std::swap(sample[static_cast<std::size_t>(j)], sample[static_cast<std::size_t>(i)]);
    }
    std::sort(sample.begin(), sample.end());
    return sample;
}

std::string decode_seed_key(int root_id, double sim_time, const std::vector<const RequestState*>& reqs) {
    std::ostringstream oss;
    oss << root_id << "|";
    oss << std::fixed << std::setprecision(9) << sim_time << "|";
    for (std::size_t i = 0; i < reqs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << reqs[i]->request_id;
    }
    return oss.str();
}

}  // namespace

bool NewFeatures226HGBRuntime::load_model_export(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) {
        throw std::runtime_error("failed to open native HGB export: " + path);
    }

    std::string export_header;
    std::getline(in, export_header);
    if (trim_copy(export_header) == "agz_dnn_v1" ||
        trim_copy(export_header) == "agz_dnn_v2") {
        dnn_model_ = NativeDenseDNNModel();
        dnn_model_.load_model_export(path);
        if (!dnn_model_.is_value()) {
            throw std::runtime_error("value runtime received a non-value DNN export");
        }
        feature_dim_ = dnn_model_.feature_dim();
        model_tag_ = dnn_model_.model_tag();
        trees_.clear();
        return true;
    }
    in.clear(); in.seekg(0); dnn_model_ = NativeDenseDNNModel();
    model_tag_.clear();
    feature_dim_ = 226;
    baseline_ = 0.0;
    trees_.clear();

    std::string line;
    bool saw_header = false;
    Tree* cur_tree = nullptr;
    int expected_nodes = -1;
    while (std::getline(in, line)) {
        line = trim_copy(line);
        if (line.empty() || line[0] == '#') continue;
        const auto cols = split_tab(line);
        if (!saw_header) {
            if (cols.empty() || cols[0] != "hgb226_v1") {
                throw std::runtime_error("invalid native HGB export header in " + path);
            }
            saw_header = true;
            continue;
        }

        const std::string& kind = cols[0];
        if (kind == "model_tag" && cols.size() >= 2) {
            model_tag_ = cols[1];
        } else if (kind == "feature_dim" && cols.size() >= 2) {
            feature_dim_ = std::stoi(cols[1]);
        } else if (kind == "baseline" && cols.size() >= 2) {
            baseline_ = std::stod(cols[1]);
        } else if (kind == "tree" && cols.size() >= 3) {
            trees_.push_back(Tree{});
            cur_tree = &trees_.back();
            expected_nodes = std::stoi(cols[2]);
            cur_tree->nodes.reserve(static_cast<std::size_t>(std::max(0, expected_nodes)));
        } else if (kind == "node" && cols.size() >= 8) {
            if (cur_tree == nullptr) {
                throw std::runtime_error("native HGB export node before tree in " + path);
            }
            Node n;
            n.value = std::stod(cols[1]);
            n.feature_idx = std::stoi(cols[2]);
            n.threshold = std::stod(cols[3]);
            n.missing_go_to_left = std::stoi(cols[4]) != 0;
            n.left = std::stoi(cols[5]);
            n.right = std::stoi(cols[6]);
            n.is_leaf = std::stoi(cols[7]) != 0;
            cur_tree->nodes.push_back(n);
        } else if (kind == "end_tree") {
            if (cur_tree != nullptr && expected_nodes >= 0 &&
                static_cast<int>(cur_tree->nodes.size()) != expected_nodes) {
                throw std::runtime_error("native HGB export tree node count mismatch in " + path);
            }
            cur_tree = nullptr;
            expected_nodes = -1;
        }
    }

    if (!saw_header || trees_.empty()) {
        throw std::runtime_error("native HGB export did not contain trees: " + path);
    }
    if (feature_dim_ != kFeatureDim) {
        throw std::runtime_error("native HGB feature_dim must be 226, got " + std::to_string(feature_dim_));
    }
    if (model_tag_.empty()) model_tag_ = "v4_adv_hgb_native";
    return true;
}

std::vector<float> NewFeatures226HGBRuntime::build_features(
    const SimState& state,
    const VirtualSimulatorGV2* simulator,
    int root_id,
    double* decode_time_at_max_out) const {
    std::vector<float> output;
    build_features_into(
        state,
        simulator,
        root_id,
        &output,
        decode_time_at_max_out);
    return output;
}

void NewFeatures226HGBRuntime::build_features_into(
    const SimState& state,
    const VirtualSimulatorGV2* simulator,
    int root_id,
    std::vector<float>* output,
    double* decode_time_at_max_out) const {
    if (output == nullptr) {
        throw std::runtime_error("new_features_226 output is null");
    }
    const double sim_time = state.sim_time;

    thread_local FeatureBuildScratch scratch;
    auto& active_requests = scratch.active_requests;
    auto& prefill_reqs = scratch.prefill_requests;
    auto& decode_reqs = scratch.decode_requests;
    active_requests.clear();
    prefill_reqs.clear();
    decode_reqs.clear();
    active_requests.reserve(state.requests.size());
    prefill_reqs.reserve(state.requests.size());
    decode_reqs.reserve(state.requests.size());
    const bool active_ids_sorted = std::is_sorted(
        state.stats.active_request_ids.begin(),
        state.stats.active_request_ids.end());
    for (const auto& r : state.requests) {
        // Match build_state_local_features_adv.py: snapshot request records can
        // include finalized historical requests, so only ids in active_request_ids
        // are in-system feature candidates. An empty active set means no active
        // requests, not "all snapshot requests".
        if (!contains_sorted_or_linear(
                state.stats.active_request_ids,
                active_ids_sorted,
                r.request_id)) {
            continue;
        }
        active_requests.push_back(&r);
        if (is_prefill_req(r)) prefill_reqs.push_back(&r);
        else if (is_decode_req(r)) decode_reqs.push_back(&r);
    }

    const int num_prefill = static_cast<int>(prefill_reqs.size());
    const int num_decode = static_cast<int>(decode_reqs.size());
    const int num_active = static_cast<int>(active_requests.size());

    double total_remaining_prefill = 0.0;
    for (const RequestState* r : prefill_reqs) {
        total_remaining_prefill += static_cast<double>(std::max(0, r->remaining_prefill()));
    }
    double total_remaining_decode = 0.0;
    double total_decode_generated_active = 0.0;
    for (const RequestState* r : decode_reqs) {
        total_remaining_decode += decode_remaining(*r);
        total_decode_generated_active += decode_done(*r);
    }

    int num_violated_active = 0;
    int num_prefill_violated = 0;
    int num_decode_violated = 0;
    const bool violated_ids_sorted = std::is_sorted(
        state.stats.violated_request_ids.begin(),
        state.stats.violated_request_ids.end());
    const auto request_is_violated = [&](int request_id) {
        return contains_sorted_or_linear(
            state.stats.violated_request_ids,
            violated_ids_sorted,
            request_id);
    };
    for (const RequestState* r : active_requests) {
        if (request_is_violated(r->request_id)) ++num_violated_active;
    }
    for (const RequestState* r : prefill_reqs) {
        if (request_is_violated(r->request_id)) ++num_prefill_violated;
    }
    for (const RequestState* r : decode_reqs) {
        if (request_is_violated(r->request_id)) ++num_decode_violated;
    }

    int p_late_05_15 = 0;
    int p_late_15 = 0;
    for (const RequestState* r : prefill_reqs) {
        const double late = prefill_lateness_for_feature(state, *r);
        if (late > kNearDropLowSec && late < kNearDropHighSec) ++p_late_05_15;
        else if (late >= kNearDropHighSec) ++p_late_15;
    }

    int d_late_05_15 = 0;
    int d_late_15 = 0;
    for (const RequestState* r : decode_reqs) {
        const double late_p = std::max(0.0, map_get(state.stats.per_request_prefill_lateness_by_id, r->request_id, 0.0));
        const double late_d = std::max(0.0, map_get(state.stats.per_request_decode_lateness_by_id, r->request_id, 0.0));
        const double late = std::max(late_p, late_d);
        if (late > kNearDropLowSec && late < kNearDropHighSec) ++d_late_05_15;
        else if (late >= kNearDropHighSec) ++d_late_15;
    }

    double launch_count = 0.0;
    double launch_prefill = 0.0;
    double ewma = 0.0;
    bool have_latest_launch = false;
    double latest_launch = 0.0;
    if (!state.stats.recent_launches.empty()) {
        for (const auto& item : state.stats.recent_launches) {
            const double dt = std::max(0.0, sim_time - item.timestamp);
            if (dt > kLaunchEwmaWindowSec) continue;
            have_latest_launch = true;
            latest_launch = std::max(latest_launch, item.timestamp);
            launch_count += std::max(0, item.count);
            launch_prefill += std::max(0, item.prefill_tokens);
            ewma += std::max(0, item.count) * std::exp(-kLaunchEwmaAlpha * dt);
        }
    } else {
        for (double ts : state.stats.recent_arrivals) {
            const double dt = std::max(0.0, sim_time - ts);
            if (dt > kLaunchEwmaWindowSec) continue;
            have_latest_launch = true;
            latest_launch = std::max(latest_launch, ts);
            launch_count += 1.0;
            ewma += std::exp(-kLaunchEwmaAlpha * dt);
        }
    }

    const double delta_since_last_adv_launch =
        have_latest_launch ? std::max(0.0, sim_time - latest_launch) : kLaunchEwmaWindowSec;
    const double remaining_launch_request_headroom = std::max(0.0, 7.0 - launch_count);
    const double remaining_launch_prefill_headroom = std::max(0.0, 7168.0 - launch_prefill);

    const double decode_credit = std::max(0.0, static_cast<double>(state.stats.decode_credit_balance));

    double min_prefill_slack = std::numeric_limits<double>::infinity();
    for (const RequestState* r : prefill_reqs) {
        const double slack = (r->arrived_at + r->prefill_slo_time) - sim_time;
        min_prefill_slack = std::min(min_prefill_slack, slack);
    }

    const double decode_batch_time = decode_time_at_max(state, simulator, decode_reqs);
    if (decode_time_at_max_out != nullptr) *decode_time_at_max_out = decode_batch_time;

    double edf_minus_batch_norm = 1.0;
    if (!std::isinf(min_prefill_slack)) {
        edf_minus_batch_norm = clamp(min_prefill_slack - decode_batch_time, 0.0, 1.0);
    }

    int n_active_edf_margin_gt_batch = 0;
    for (const RequestState* r : prefill_reqs) {
        const double deadline = r->arrived_at + r->prefill_slo_time;
        if ((deadline - sim_time) - decode_batch_time > 0.0) ++n_active_edf_margin_gt_batch;
    }
    for (const RequestState* r : decode_reqs) {
        const double deadline = map_get(
            state.stats.decode_next_deadline_by_id,
            r->request_id,
            r->arrived_at + r->decode_slo_time);
        if ((deadline - sim_time) - decode_batch_time > 0.0) ++n_active_edf_margin_gt_batch;
    }

    const double next_adv_tick = (state.stats.next_adv_tick >= 0.0)
        ? state.stats.next_adv_tick
        : map_get(
            state.stats.decode_next_deadline_by_id,
            kMetaNextAdvTick,
            sim_time);
    const double delta_next_adv_tick = std::max(0.0, next_adv_tick - sim_time);

    output->assign(static_cast<std::size_t>(kFeatureDim), 0.0f);
    std::vector<float>& out = *output;
    std::size_t pos = 0;
    auto push = [&](double v) {
        if (pos >= out.size()) throw std::runtime_error("new_features_226 overflow");
        out[pos++] = static_cast<float>(v);
    };

    push(norm01(num_prefill, kActivePrefillCountDen));
    push(norm01(num_decode, kActiveDecodeCountDen));
    push(norm01(num_active, kActiveTotalCountDen));
    push(norm01(total_remaining_prefill, kTotalRemainingPrefillDen));
    push(norm01(total_remaining_decode, kTotalRemainingDecodeDen));
    push(norm01(total_decode_generated_active, kTotalDecodeGeneratedActiveDen));
    push(norm01(num_violated_active, kViolatedCountDen));
    push(norm01(p_late_05_15, kPrefillNearDropDen));
    push(norm01(p_late_15, kPrefillNearDropDen));
    push(norm01(d_late_05_15, kDecodeNearDropDen));
    push(norm01(d_late_15, kDecodeNearDropDen));
    push(norm01(launch_count, kRecentLaunchCountDen));
    push(norm01(launch_prefill, kRecentLaunchPrefillDen));
    push(norm01(remaining_launch_request_headroom, 7.0));
    push(norm01(remaining_launch_prefill_headroom, 7168.0));
    push(norm01(ewma, kRecentLaunchCountDen));
    push(norm01(decode_credit, kDecodeCreditDen));
    push(norm01(num_prefill_violated, kActivePrefillCountDen));
    push(norm01(num_decode_violated, kActiveDecodeCountDen));
    push(edf_minus_batch_norm);
    push(norm01(n_active_edf_margin_gt_batch, kActiveTotalCountDen));
    push(norm01(delta_next_adv_tick, kAdvTickSecDen));
    push(norm01(delta_since_last_adv_launch, kLaunchEwmaWindowSec));

    if (pos != static_cast<std::size_t>(kGlobalDim)) {
        throw std::runtime_error("new_features_226 global dim mismatch");
    }

    struct PrefillSlot {
        int rid = -1;
        double remaining = 0.0;
        double total = 0.0;
        bool violated = false;
        double slack_clamped = 0.0;
        double lateness = 0.0;
    };
    std::vector<PrefillSlot> enriched;
    enriched.reserve(prefill_reqs.size());
    for (const RequestState* r : prefill_reqs) {
        PrefillSlot p;
        p.rid = r->request_id;
        p.remaining = std::max(0, r->remaining_prefill());
        p.total = std::max(0, r->num_prefill_tokens);
        p.violated = request_is_violated(r->request_id);
        const double deadline = r->arrived_at + r->prefill_slo_time;
        p.slack_clamped = std::max(0.0, deadline - sim_time);
        p.lateness = std::max(
            std::max(0.0, map_get(state.stats.per_request_prefill_lateness_by_id, r->request_id, 0.0)),
            std::max(0.0, sim_time - deadline));
        enriched.push_back(p);
    }
    const bool all_violated = !enriched.empty() &&
        std::all_of(enriched.begin(), enriched.end(), [](const PrefillSlot& p) { return p.violated; });
    if (all_violated) {
        std::sort(enriched.begin(), enriched.end(), [](const PrefillSlot& a, const PrefillSlot& b) {
            return a.rid < b.rid;
        });
    } else {
        std::sort(enriched.begin(), enriched.end(), [](const PrefillSlot& a, const PrefillSlot& b) {
            if (a.slack_clamped != b.slack_clamped) return a.slack_clamped < b.slack_clamped;
            return a.rid < b.rid;
        });
    }

    for (int slot = 0; slot < kPrefillSlots; ++slot) {
        const std::size_t slot_start = pos;
        if (slot < static_cast<int>(enriched.size())) {
            const auto& p = enriched[static_cast<std::size_t>(slot)];
            out[slot_start + 0] = 1.0f;
            out[slot_start + 1] = static_cast<float>(norm01(p.remaining, 4096.0));
            out[slot_start + 2] = static_cast<float>(norm01(p.total, 4096.0));
            out[slot_start + 3] = p.violated ? 1.0f : 0.0f;
            const int lb = lateness_bucket_idx(p.lateness);
            out[slot_start + 4 + lb] = 1.0f;
            const int sb = slack_bucket_idx(p.slack_clamped);
            out[slot_start + 4 + kLateBuckets + sb] = 1.0f;
        }
        pos += kPrefillSlotDim;
    }

    auto& non_violated_decode = scratch.non_violated_decode;
    non_violated_decode.clear();
    non_violated_decode.reserve(decode_reqs.size());
    for (const RequestState* r : decode_reqs) {
        if (!request_is_violated(r->request_id)) {
            non_violated_decode.push_back(r);
        }
    }
    std::sort(non_violated_decode.begin(), non_violated_decode.end(), [](const RequestState* a, const RequestState* b) {
        return a->request_id < b->request_id;
    });
    auto& chosen_decode = scratch.chosen_decode;
    chosen_decode.clear();
    chosen_decode.reserve(kDecodeSlots);
    if (static_cast<int>(non_violated_decode.size()) > kDecodeSlots) {
        const auto idxs = deterministic_subset_indices(
            static_cast<int>(non_violated_decode.size()),
            kDecodeSlots,
            decode_seed_key(root_id, sim_time, non_violated_decode));
        for (int idx : idxs) chosen_decode.push_back(non_violated_decode[static_cast<std::size_t>(idx)]);
    } else {
        chosen_decode.insert(
            chosen_decode.end(),
            non_violated_decode.begin(),
            non_violated_decode.end());
    }

    for (int slot = 0; slot < kDecodeSlots; ++slot) {
        const std::size_t slot_start = pos;
        if (slot < static_cast<int>(chosen_decode.size())) {
            const RequestState* r = chosen_decode[static_cast<std::size_t>(slot)];
            const double rem = decode_remaining(*r);
            const double done = decode_done(*r);
            out[slot_start + 0] = 1.0f;
            out[slot_start + 1] = static_cast<float>(norm01(rem, kDecodeRemainingDen));
            out[slot_start + 2] = done > 216.0 ? 1.0f : 0.0f;
            out[slot_start + 3] = done > 512.0 ? 1.0f : 0.0f;
        }
        pos += kDecodeSlotDim;
    }

    if (pos != out.size()) {
        throw std::runtime_error("new_features_226 final dim mismatch");
    }
}

double NewFeatures226HGBRuntime::tree_value(const Tree& tree, const std::vector<float>& features) {
    if (tree.nodes.empty()) return 0.0;
    int idx = 0;
    int guard = 0;
    while (idx >= 0 && idx < static_cast<int>(tree.nodes.size())) {
        const Node& n = tree.nodes[static_cast<std::size_t>(idx)];
        if (n.is_leaf) return n.value;
        if (n.feature_idx < 0 || n.feature_idx >= static_cast<int>(features.size())) {
            throw std::runtime_error("native HGB split feature index out of range");
        }
        const double x = static_cast<double>(features[static_cast<std::size_t>(n.feature_idx)]);
        if (std::isnan(x)) {
            idx = n.missing_go_to_left ? n.left : n.right;
        } else if (x <= n.threshold) {
            idx = n.left;
        } else {
            idx = n.right;
        }
        if (++guard > 100000) throw std::runtime_error("native HGB tree traversal loop");
    }
    throw std::runtime_error("native HGB tree traversal left node range");
}

double NewFeatures226HGBRuntime::predict_raw(const std::vector<float>& features) const {
    if (dnn_model_.loaded()) {
        return dnn_model_.predict_value(features);
    }
    if (features.size() != static_cast<std::size_t>(feature_dim_)) {
        throw std::runtime_error("native HGB expected " + std::to_string(feature_dim_) +
                                 " features, got " + std::to_string(features.size()));
    }
    double total = baseline_;
    for (const auto& tree : trees_) total += tree_value(tree, features);
    return total;
}

double NewFeatures226HGBRuntime::infer_value(
    const SimState& state,
    const VirtualSimulatorGV2* simulator,
    int root_id) const {
    if (dnn_model_.is_markov_value()) {
        return std::min(
            dnn_model_.predict_markov_value(build_markov_value_features(state)),
            0.0);
    }
    const std::vector<float> f = build_features(state, simulator, root_id, nullptr);
    return std::min(predict_raw(f), 0.0);
}

NewFeatures226Result NewFeatures226HGBRuntime::infer_debug(
    const SimState& state,
    const VirtualSimulatorGV2* simulator,
    int root_id) const {
    NewFeatures226Result out;
    if (dnn_model_.is_markov_value()) {
        const MarkovValueFeatures features = build_markov_value_features(state);
        out.features = features.global_features;
        out.raw_value = dnn_model_.predict_markov_value(features);
    } else {
        out.features = build_features(state, simulator, root_id, &out.decode_time_at_max);
        out.raw_value = predict_raw(out.features);
    }
    out.value = std::min(out.raw_value, 0.0);
    return out;
}

int NewFeatures226HGBRuntime::feature_dim() const { return feature_dim_; }

int NewFeatures226HGBRuntime::num_trees() const {
    return static_cast<int>(trees_.size());
}

bool NewFeatures226HGBRuntime::loaded() const {
    return dnn_model_.loaded() || !trees_.empty();
}

const std::string& NewFeatures226HGBRuntime::model_tag() const {
    if (dnn_model_.loaded()) return dnn_model_.model_tag();
    return model_tag_;
}

bool NativeHGBModelRuntime::load_model_export(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) {
        throw std::runtime_error("failed to open native HGB export: " + path);
    }

    std::string export_header;
    std::getline(in, export_header);
    if (trim_copy(export_header) == "agz_dnn_v1" ||
        trim_copy(export_header) == "agz_dnn_v2") {
        dnn_model_ = NativeDenseDNNModel();
        dnn_model_.load_model_export(path);
        if (!dnn_model_.is_policy()) {
            throw std::runtime_error("policy runtime received a non-policy DNN export");
        }
        feature_dim_ = dnn_model_.feature_dim();
        model_tag_ = dnn_model_.model_tag();
        trees_.clear();
        return true;
    }
    in.clear(); in.seekg(0); dnn_model_ = NativeDenseDNNModel();
    model_tag_.clear();
    feature_dim_ = 0;
    baseline_ = 0.0;
    trees_.clear();

    std::string line;
    bool saw_header = false;
    Tree* cur_tree = nullptr;
    int expected_nodes = -1;
    while (std::getline(in, line)) {
        line = trim_copy(line);
        if (line.empty() || line[0] == '#') continue;
        const auto cols = split_tab(line);
        if (!saw_header) {
            if (cols.empty() || cols[0] != "hgb226_v1") {
                throw std::runtime_error("invalid native HGB export header in " + path);
            }
            saw_header = true;
            continue;
        }

        const std::string& kind = cols[0];
        if (kind == "model_tag" && cols.size() >= 2) {
            model_tag_ = cols[1];
        } else if (kind == "feature_dim" && cols.size() >= 2) {
            feature_dim_ = std::stoi(cols[1]);
        } else if (kind == "baseline" && cols.size() >= 2) {
            baseline_ = std::stod(cols[1]);
        } else if (kind == "tree" && cols.size() >= 3) {
            trees_.push_back(Tree{});
            cur_tree = &trees_.back();
            expected_nodes = std::stoi(cols[2]);
            cur_tree->nodes.reserve(static_cast<std::size_t>(std::max(0, expected_nodes)));
        } else if (kind == "node" && cols.size() >= 8) {
            if (cur_tree == nullptr) {
                throw std::runtime_error("native HGB export node before tree in " + path);
            }
            Node n;
            n.value = std::stod(cols[1]);
            n.feature_idx = std::stoi(cols[2]);
            n.threshold = std::stod(cols[3]);
            n.missing_go_to_left = std::stoi(cols[4]) != 0;
            n.left = std::stoi(cols[5]);
            n.right = std::stoi(cols[6]);
            n.is_leaf = std::stoi(cols[7]) != 0;
            cur_tree->nodes.push_back(n);
        } else if (kind == "end_tree") {
            if (cur_tree != nullptr && expected_nodes >= 0 &&
                static_cast<int>(cur_tree->nodes.size()) != expected_nodes) {
                throw std::runtime_error("native HGB export tree node count mismatch in " + path);
            }
            cur_tree = nullptr;
            expected_nodes = -1;
        }
    }

    if (!saw_header || trees_.empty()) {
        throw std::runtime_error("native HGB export did not contain trees: " + path);
    }
    if (feature_dim_ <= 0) {
        throw std::runtime_error("native HGB feature_dim must be positive, got " + std::to_string(feature_dim_));
    }
    if (model_tag_.empty()) model_tag_ = "native_hgb_model";
    return true;
}

double NativeHGBModelRuntime::tree_value(const Tree& tree, const float* features, int feature_dim) {
    if (tree.nodes.empty()) return 0.0;
    int idx = 0;
    int guard = 0;
    while (idx >= 0 && idx < static_cast<int>(tree.nodes.size())) {
        const Node& n = tree.nodes[static_cast<std::size_t>(idx)];
        if (n.is_leaf) return n.value;
        if (features == nullptr || n.feature_idx < 0 || n.feature_idx >= feature_dim) {
            throw std::runtime_error("native HGB split feature index out of range");
        }
        const double x = static_cast<double>(features[n.feature_idx]);
        if (std::isnan(x)) {
            idx = n.missing_go_to_left ? n.left : n.right;
        } else if (x <= n.threshold) {
            idx = n.left;
        } else {
            idx = n.right;
        }
        if (++guard > 100000) throw std::runtime_error("native HGB tree traversal loop");
    }
    throw std::runtime_error("native HGB tree traversal left node range");
}

double NativeHGBModelRuntime::predict_raw(const std::vector<float>& features) const {
    if (dnn_model_.loaded()) {
        return dnn_model_.predict_policy(features);
    }
    if (features.size() != static_cast<std::size_t>(feature_dim_)) {
        throw std::runtime_error("native HGB expected " + std::to_string(feature_dim_) +
                                 " features, got " + std::to_string(features.size()));
    }
    double total = baseline_;
    const float* row = features.empty() ? nullptr : features.data();
    for (const auto& tree : trees_) total += tree_value(tree, row, feature_dim_);
    return total;
}

std::vector<double> NativeHGBModelRuntime::predict_raw_batch_flat(
    const std::vector<float>& flat_features,
    int num_rows,
    int row_dim) const {
    if (dnn_model_.loaded()) {
        return dnn_model_.predict_policy_batch_flat(flat_features, num_rows, row_dim);
    }
    if (num_rows < 0) {
        throw std::runtime_error("native HGB batch num_rows must be non-negative");
    }
    if (row_dim != feature_dim_) {
        throw std::runtime_error("native HGB batch expected row_dim " + std::to_string(feature_dim_) +
                                 ", got " + std::to_string(row_dim));
    }
    const std::size_t expected = static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(row_dim);
    if (flat_features.size() != expected) {
        throw std::runtime_error("native HGB batch expected " + std::to_string(expected) +
                                 " flat features, got " + std::to_string(flat_features.size()));
    }

    std::vector<double> out(static_cast<std::size_t>(num_rows), baseline_);
    if (num_rows == 0) return out;

    const float* base = flat_features.data();
    for (const auto& tree : trees_) {
        for (int r = 0; r < num_rows; ++r) {
            const float* row = base + static_cast<std::size_t>(r) * static_cast<std::size_t>(row_dim);
            out[static_cast<std::size_t>(r)] += tree_value(tree, row, row_dim);
        }
    }
    return out;
}

std::vector<double> NativeHGBModelRuntime::predict_raw_grouped_batch_flat(
    const std::vector<float>& flat_features,
    int num_rows,
    int row_dim,
    const std::vector<int>& group_offsets,
    int parallel_threads) const {
    if (dnn_model_.loaded()) {
        return dnn_model_.predict_policy_grouped_batch_flat(
            flat_features,
            num_rows,
            row_dim,
            group_offsets,
            parallel_threads);
    }
    return predict_raw_batch_flat(flat_features, num_rows, row_dim);
}

std::vector<double> NativeHGBModelRuntime::predict_raw_grouped_split_batch_flat(
    const std::vector<float>& flat_states,
    const std::vector<float>& flat_actions,
    int num_rows,
    const std::vector<int>& group_offsets,
    int parallel_threads) const {
    if (dnn_model_.loaded()) {
        return dnn_model_.predict_policy_grouped_split_batch_flat(
            flat_states,
            flat_actions,
            num_rows,
            group_offsets,
            parallel_threads);
    }
    if (num_rows < 0 || group_offsets.empty() || group_offsets.front() != 0 ||
        group_offsets.back() != num_rows) {
        throw std::runtime_error("native HGB split batch dimensions mismatch");
    }
    const int group_count = static_cast<int>(group_offsets.size()) - 1;
    if (group_count == 0) return {};
    if (flat_states.size() % static_cast<std::size_t>(group_count) != 0 ||
        num_rows == 0 ||
        flat_actions.size() % static_cast<std::size_t>(num_rows) != 0) {
        throw std::runtime_error("native HGB split flat batch size mismatch");
    }
    const int state_dim = static_cast<int>(
        flat_states.size() / static_cast<std::size_t>(group_count));
    const int action_dim = static_cast<int>(
        flat_actions.size() / static_cast<std::size_t>(num_rows));
    const int row_dim = state_dim + action_dim;
    if (row_dim != feature_dim_) {
        throw std::runtime_error("native HGB split feature dimension mismatch");
    }

    std::vector<float> flat_features(
        static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(row_dim));
    for (int group = 0; group < group_count; ++group) {
        const int begin = group_offsets[static_cast<std::size_t>(group)];
        const int end = group_offsets[static_cast<std::size_t>(group + 1)];
        if (begin > end) {
            throw std::runtime_error("native HGB split offsets are not monotonic");
        }
        const float* state = flat_states.data() +
            static_cast<std::size_t>(group) * static_cast<std::size_t>(state_dim);
        for (int row = begin; row < end; ++row) {
            float* destination = flat_features.data() +
                static_cast<std::size_t>(row) * static_cast<std::size_t>(row_dim);
            std::copy(state, state + state_dim, destination);
            const float* action = flat_actions.data() +
                static_cast<std::size_t>(row) * static_cast<std::size_t>(action_dim);
            std::copy(action, action + action_dim, destination + state_dim);
        }
    }
    return predict_raw_batch_flat(flat_features, num_rows, row_dim);
}

std::vector<double> NativeHGBModelRuntime::predict_markov_policy_grouped_batch(
    const std::vector<MarkovValueFeatures>& states,
    const std::vector<float>& flat_actions,
    int num_rows,
    const std::vector<int>& group_offsets,
    int parallel_threads) const {
    if (!dnn_model_.is_markov_policy()) {
        throw std::runtime_error("native prior runtime is not a Markov policy");
    }
    return dnn_model_.predict_markov_policy_grouped_batch(
        states,
        flat_actions,
        num_rows,
        group_offsets,
        parallel_threads);
}

int NativeHGBModelRuntime::feature_dim() const { return feature_dim_; }

int NativeHGBModelRuntime::num_trees() const {
    return static_cast<int>(trees_.size());
}

bool NativeHGBModelRuntime::loaded() const {
    return dnn_model_.loaded() || !trees_.empty();
}

const std::string& NativeHGBModelRuntime::model_tag() const {
    if (dnn_model_.loaded()) return dnn_model_.model_tag();
    return model_tag_;
}

bool NativeHGBModelRuntime::is_markov_policy() const {
    return dnn_model_.is_markov_policy();
}

int NativeHGBModelRuntime::action_dim() const {
    return dnn_model_.loaded()
        ? dnn_model_.action_dim()
        : std::max(0, feature_dim_ - 226);
}

}  // namespace mcts_native_gv2
