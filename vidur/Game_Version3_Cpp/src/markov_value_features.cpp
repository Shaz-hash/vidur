#include "markov_value_features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace mcts_native_gv2 {
namespace {

constexpr double kActiveScale = 120.0;
constexpr double kPrefillRequestScale = 20.0;
constexpr double kDecodeRequestScale = 100.0;
constexpr double kPrefillTokens = 4096.0;
constexpr double kDecodeTokens = 864.0;
constexpr double kContextTokens = kPrefillTokens + kDecodeTokens;
constexpr double kDecodeCreditMint = 216.0;
constexpr double kAdversaryTickSec = 0.2;
constexpr double kLaunchRequestCap = 7.0;
constexpr double kLaunchPrefillCap = 7168.0;
constexpr double kPrefillTimeSec = 1.0;
constexpr double kDecodeTimeSec = 0.05;
constexpr int kMetaNextAdvTick = -9100001;
constexpr int kMetaDecodeCreditBalance = -9100005;
constexpr int kMetaMissedAdvSource = -9100006;

double checked_finite(double value, const std::string& label) {
    if (!std::isfinite(value)) throw std::runtime_error(label + " is not finite");
    return value;
}

float encoded(double value) {
    if (!std::isfinite(value)) throw std::runtime_error("Markov feature is not finite");
    return static_cast<float>(value);
}

double asinh_scaled(double value, double scale) {
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0) {
        throw std::runtime_error("invalid Markov asinh arguments");
    }
    return std::asinh(value / scale);
}

bool contains(const std::unordered_set<int>& values, int value) {
    return values.find(value) != values.end();
}

template <typename T>
const T* map_find(const std::unordered_map<int, T>& values, int key) {
    const auto found = values.find(key);
    return found == values.end() ? nullptr : &found->second;
}

std::unordered_set<int> id_set(const std::vector<int>& values) {
    return std::unordered_set<int>(values.begin(), values.end());
}

void append_request_row(
    std::vector<float>* output,
    bool decode_phase,
    int prefill_total,
    int prefill_processed,
    int prefill_remaining,
    int decode_total,
    int decode_processed,
    int decode_remaining,
    double sim_time,
    const RequestState& request,
    double prefill_deadline,
    bool decode_deadline_present,
    double decode_deadline,
    bool completion_present,
    double prefill_completed_at,
    double prefill_lateness,
    double decode_lateness,
    bool ledger_present,
    int ledger_counted,
    bool violated,
    bool finalized) {
    const double row[] = {
        decode_phase ? 1.0 : 0.0,
        prefill_total / kPrefillTokens,
        prefill_processed / kPrefillTokens,
        prefill_remaining / kPrefillTokens,
        decode_total / kDecodeTokens,
        decode_processed / kDecodeTokens,
        decode_remaining / kDecodeTokens,
        (prefill_processed + decode_processed) / kContextTokens,
        asinh_scaled(sim_time - request.arrived_at, kPrefillTimeSec),
        asinh_scaled(sim_time - request.queued_at, kPrefillTimeSec),
        asinh_scaled(request.prefill_slo_time, kPrefillTimeSec),
        asinh_scaled(prefill_deadline - sim_time, kPrefillTimeSec),
        asinh_scaled(request.decode_slo_time, kDecodeTimeSec),
        decode_deadline_present ? 1.0 : 0.0,
        decode_deadline_present
            ? asinh_scaled(decode_deadline - sim_time, kDecodeTimeSec)
            : 0.0,
        completion_present ? 1.0 : 0.0,
        completion_present
            ? asinh_scaled(sim_time - prefill_completed_at, kPrefillTimeSec)
            : 0.0,
        asinh_scaled(prefill_lateness, kPrefillTimeSec),
        asinh_scaled(decode_lateness, kPrefillTimeSec),
        ledger_counted / kDecodeTokens,
        ledger_present ? 1.0 : 0.0,
        violated ? 1.0 : 0.0,
        finalized ? 1.0 : 0.0,
        request.is_prefill_complete ? 1.0 : 0.0,
    };
    static_assert(sizeof(row) / sizeof(row[0]) == kMarkovRequestDim);
    for (double value : row) output->push_back(encoded(value));
}

}  // namespace

MarkovValueFeatures build_markov_value_features(
    const SimState& state,
    double launch_window_sec) {
    if (!std::isfinite(launch_window_sec) || launch_window_sec <= 0.0) {
        throw std::runtime_error("invalid Markov launch window");
    }
    const double sim_time = checked_finite(state.sim_time, "sim_time");

    std::unordered_map<int, const RequestState*> requests;
    requests.reserve(state.requests.size());
    for (const auto& request : state.requests) {
        if (!requests.emplace(request.request_id, &request).second) {
            throw std::runtime_error("duplicate Markov request id");
        }
    }

    std::vector<int> active_ids = state.stats.active_request_ids;
    std::sort(active_ids.begin(), active_ids.end());
    if (std::adjacent_find(active_ids.begin(), active_ids.end()) != active_ids.end()) {
        throw std::runtime_error("duplicate Markov active request id");
    }
    const auto violated_ids = id_set(state.stats.violated_request_ids);
    const auto finalized_ids = id_set(state.stats.prefill_lateness_finalized_ids);

    MarkovValueFeatures result;
    result.request_ids = active_ids;
    result.request_count = static_cast<int>(active_ids.size());
    result.request_features.reserve(
        active_ids.size() * static_cast<std::size_t>(kMarkovRequestDim));

    int prefill_count = 0;
    int decode_count = 0;
    long long remaining_prefill_total = 0;
    long long remaining_decode_total = 0;
    long long processed_prefill_total = 0;
    long long processed_decode_total = 0;
    int active_violated = 0;
    int active_finalized = 0;

    for (int request_id : active_ids) {
        const auto found = requests.find(request_id);
        if (found == requests.end()) {
            throw std::runtime_error("active Markov request has no record");
        }
        const RequestState& request = *found->second;
        if (request.completed || request.dropped || request.stopped_decode || request.feature_only) {
            throw std::runtime_error("active Markov request is terminal or feature-only");
        }
        const int prefill_total = request.num_prefill_tokens;
        const int prefill_processed = request.num_processed_prefill_tokens;
        const int decode_total = request.num_decode_tokens;
        const int decode_processed = request.num_processed_decode_tokens;
        if (prefill_total < 0 || prefill_processed < 0 || decode_total < 0 ||
            decode_processed < 0 || prefill_processed > prefill_total ||
            decode_processed > decode_total) {
            throw std::runtime_error("invalid Markov request token counts");
        }
        const int prefill_remaining = prefill_total - prefill_processed;
        const int decode_remaining = decode_total - decode_processed;
        if (request.is_prefill_complete != (prefill_remaining == 0)) {
            throw std::runtime_error("Markov prefill-complete flag mismatch");
        }
        const bool decode_phase = request.is_prefill_complete && decode_remaining > 0;
        if (!decode_phase && prefill_remaining <= 0) {
            throw std::runtime_error("active Markov request has no executable work");
        }
        prefill_count += decode_phase ? 0 : 1;
        decode_count += decode_phase ? 1 : 0;

        checked_finite(request.arrived_at, "arrived_at");
        checked_finite(request.queued_at, "queued_at");
        checked_finite(request.prefill_slo_time, "prefill_slo");
        checked_finite(request.decode_slo_time, "decode_slo");
        if (request.prefill_slo_time < 0.0 || request.decode_slo_time < 0.0) {
            throw std::runtime_error("negative Markov request SLO");
        }
        const double prefill_deadline = checked_finite(
            request.prefill_deadline >= 0.0
                ? request.prefill_deadline
                : request.arrived_at + request.prefill_slo_time,
            "prefill_deadline");

        const double* stats_deadline =
            map_find(state.stats.decode_next_deadline_by_id, request_id);
        const double decode_deadline = checked_finite(
            stats_deadline != nullptr ? *stats_deadline : request.decode_next_deadline,
            "decode_deadline");
        const bool decode_deadline_present = decode_deadline >= 0.0;
        if (decode_phase && !decode_deadline_present) {
            throw std::runtime_error("decode-phase Markov request has no deadline");
        }
        const double prefill_completed_at =
            checked_finite(request.prefill_completed_at, "prefill_completed_at");
        const bool completion_present = prefill_completed_at >= 0.0;
        if (decode_phase && !completion_present) {
            throw std::runtime_error("decode-phase Markov request has no prefill completion time");
        }

        const double* stats_prefill_lateness =
            map_find(state.stats.per_request_prefill_lateness_by_id, request_id);
        const double* stats_decode_lateness =
            map_find(state.stats.per_request_decode_lateness_by_id, request_id);
        const double prefill_lateness = checked_finite(
            stats_prefill_lateness != nullptr
                ? *stats_prefill_lateness
                : request.prefill_lateness,
            "prefill_lateness");
        const double decode_lateness = checked_finite(
            stats_decode_lateness != nullptr
                ? *stats_decode_lateness
                : request.decode_lateness,
            "decode_lateness");
        if (prefill_lateness < 0.0 || decode_lateness < 0.0) {
            throw std::runtime_error("negative Markov request lateness");
        }

        const int* ledger_value =
            map_find(state.stats.decode_tokens_counted_by_id, request_id);
        const bool ledger_present = ledger_value != nullptr;
        const int ledger_counted = ledger_present ? *ledger_value : 0;
        if (ledger_counted < 0) {
            throw std::runtime_error("negative Markov credit-ledger count");
        }
        const bool violated = request.violated || contains(violated_ids, request_id);
        const bool finalized = contains(finalized_ids, request_id);
        append_request_row(
            &result.request_features,
            decode_phase,
            prefill_total,
            prefill_processed,
            prefill_remaining,
            decode_total,
            decode_processed,
            decode_remaining,
            sim_time,
            request,
            prefill_deadline,
            decode_deadline_present,
            decode_deadline,
            completion_present,
            prefill_completed_at,
            prefill_lateness,
            decode_lateness,
            ledger_present,
            ledger_counted,
            violated,
            finalized);

        remaining_prefill_total += prefill_remaining;
        remaining_decode_total += decode_remaining;
        processed_prefill_total += prefill_processed;
        processed_decode_total += decode_processed;
        active_violated += violated ? 1 : 0;
        active_finalized += finalized ? 1 : 0;
    }

    long long launch_requests = 0;
    long long launch_prefill = 0;
    result.launch_count = 0;
    result.launch_features.reserve(
        state.stats.recent_launches.size() * static_cast<std::size_t>(kMarkovLaunchDim));
    for (const auto& launch : state.stats.recent_launches) {
        checked_finite(launch.timestamp, "launch timestamp");
        if (launch.count < 0 || launch.prefill_tokens < 0) {
            throw std::runtime_error("negative Markov launch payload");
        }
        const double age = sim_time - launch.timestamp;
        if (age < -1e-7) throw std::runtime_error("Markov launch is in the future");
        if (age > launch_window_sec + 1e-6) {
            continue;
        }
        ++result.launch_count;
        result.launch_features.push_back(encoded(asinh_scaled(age, launch_window_sec)));
        result.launch_features.push_back(encoded(launch.count / kLaunchRequestCap));
        result.launch_features.push_back(encoded(launch.prefill_tokens / kLaunchPrefillCap));
        launch_requests += launch.count;
        launch_prefill += launch.prefill_tokens;
    }

    double next_tick = state.stats.next_adv_tick;
    if (next_tick < 0.0) {
        const double* fallback =
            map_find(state.stats.decode_next_deadline_by_id, kMetaNextAdvTick);
        if (fallback == nullptr) throw std::runtime_error("Markov next_adv_tick is missing");
        next_tick = *fallback;
    }
    checked_finite(next_tick, "next_adv_tick");

    int credit = state.stats.decode_credit_balance;
    if (const int* fallback =
            map_find(state.stats.decode_tokens_counted_by_id, kMetaDecodeCreditBalance);
        credit == 0 && fallback != nullptr) {
        credit = *fallback;
    }
    const int usable_credit = std::max(0, state.stats.decode_credit_available);
    int missed_source = state.stats.missed_adv_source;
    if (const double* fallback =
            map_find(state.stats.decode_next_deadline_by_id, kMetaMissedAdvSource);
        missed_source == 0 && fallback != nullptr) {
        missed_source = static_cast<int>(*fallback);
    }
    if (missed_source < 0 || missed_source > 2) {
        throw std::runtime_error("unsupported Markov missed_adv_source");
    }

    const double global[] = {
        active_ids.size() / kActiveScale,
        prefill_count / kPrefillRequestScale,
        decode_count / kDecodeRequestScale,
        remaining_prefill_total / (kPrefillRequestScale * kPrefillTokens),
        remaining_decode_total / (kDecodeRequestScale * kDecodeTokens),
        processed_prefill_total / (kPrefillRequestScale * kPrefillTokens),
        processed_decode_total / (kDecodeRequestScale * kDecodeTokens),
        (processed_prefill_total + processed_decode_total) / (kActiveScale * kContextTokens),
        asinh_scaled(credit, kDecodeCreditMint),
        usable_credit / kDecodeCreditMint,
        asinh_scaled(next_tick - sim_time, kAdversaryTickSec),
        state.stats.pending_adv_tick ? 1.0 : 0.0,
        missed_source == 0 ? 1.0 : 0.0,
        missed_source == 1 ? 1.0 : 0.0,
        missed_source == 2 ? 1.0 : 0.0,
        launch_requests / kLaunchRequestCap,
        launch_prefill / kLaunchPrefillCap,
        active_violated / kDecodeRequestScale,
        active_finalized / kPrefillRequestScale,
    };
    static_assert(sizeof(global) / sizeof(global[0]) == kMarkovGlobalDim);
    result.global_features.reserve(kMarkovGlobalDim);
    for (double value : global) result.global_features.push_back(encoded(value));

    if (result.request_features.size() !=
            active_ids.size() * static_cast<std::size_t>(kMarkovRequestDim) ||
        result.launch_features.size() !=
            static_cast<std::size_t>(result.launch_count) * static_cast<std::size_t>(kMarkovLaunchDim)) {
        throw std::runtime_error("Markov feature shape mismatch");
    }
    return result;
}

}  // namespace mcts_native_gv2
