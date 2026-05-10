#include "gv2_infer_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace mcts_native_gv2 {
namespace {

std::string trim_copy(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    return f.good();
}

bool dict_get_tensor(
    const c10::impl::GenericDict& d,
    const std::vector<std::string>& keys,
    torch::Tensor* out) {
    if (out == nullptr) return false;
    for (const auto& wanted : keys) {
        for (const auto& item : d) {
            const c10::IValue& k = item.key();
            const c10::IValue& v = item.value();
            if (!k.isString() || !v.isTensor()) continue;
            if (k.toStringRef() == wanted) {
                *out = v.toTensor();
                return true;
            }
        }
    }
    return false;
}

std::pair<torch::Tensor, torch::Tensor> parse_forward_output(const c10::IValue& out_iv) {
    if (out_iv.isTuple()) {
        const auto elems = out_iv.toTuple()->elements();
        if (elems.size() >= 2 && elems[0].isTensor() && elems[1].isTensor()) {
            return {elems[0].toTensor(), elems[1].toTensor()};
        }
    }
    if (out_iv.isList()) {
        const auto elems = out_iv.toListRef();
        if (elems.size() >= 2 && elems[0].isTensor() && elems[1].isTensor()) {
            return {elems[0].toTensor(), elems[1].toTensor()};
        }
    }
    if (out_iv.isGenericDict()) {
        const auto d = out_iv.toGenericDict();
        torch::Tensor policy;
        torch::Tensor value;
        const bool has_policy = dict_get_tensor(
            d, {"policy_logits", "policy_logit", "policy", "logits"}, &policy);
        const bool has_value = dict_get_tensor(
            d, {"value_raw", "value_logits", "value", "v"}, &value);
        if (has_policy && has_value) {
            return {policy, value};
        }
    }
    throw std::runtime_error(
        "Unsupported TorchScript output. Expected tuple/list/dict with policy+value tensors.");
}

double clamp_double(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

double sigmoid_stable(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

double denormalize_value_model_scalar(double value_norm, double v_min) {
    constexpr double kVLinearMin = -48.0;
    constexpr double kVNormMin = -1.0;
    constexpr double kVNormMax = 0.0;
    constexpr double kVLinearNormMin = -0.98;
    constexpr double kVTailCompressPower = 2.0;
    constexpr double kVMax = 0.0;

    const double y = clamp_double(value_norm, kVNormMin, kVNormMax);
    const double linear_scale = std::abs(kVLinearNormMin) / std::abs(kVLinearMin);
    const double x_linear = y / linear_scale;

    double tail_real_span = kVLinearMin - v_min;
    if (tail_real_span <= 0.0) tail_real_span = 2.0;

    const double tail_norm_span = kVLinearNormMin - kVNormMin;
    const double t = clamp_double((kVLinearNormMin - y) / tail_norm_span, 0.0, 1.0);
    const double x_tail = kVLinearMin - tail_real_span * std::pow(t, 1.0 / kVTailCompressPower);

    const double x = (y >= kVLinearNormMin) ? x_linear : x_tail;
    return clamp_double(x, v_min, kVMax);
}

double safe_den(double x, double fallback) {
    return (std::isfinite(x) && x > 0.0) ? x : fallback;
}

int safe_pos_int(int x, int fallback) {
    return (x > 0) ? x : fallback;
}

double prefill_deadline_from_request(const RequestState& req) {
    if (std::isfinite(req.prefill_deadline) && req.prefill_deadline > 0.0) {
        return req.prefill_deadline;
    }
    if (std::isfinite(req.prefill_slo_time) && req.prefill_slo_time > 0.0) {
        return req.queued_at + req.prefill_slo_time;
    }
    return std::numeric_limits<double>::infinity();
}

std::optional<double> decode_deadline_from_request(const SimState& state, const RequestState& req) {
    const auto it = state.stats.decode_next_deadline_by_id.find(req.request_id);
    if (it != state.stats.decode_next_deadline_by_id.end() && it->second > 0.0) {
        return it->second;
    }
    if (req.prefill_completed_at > 0.0 && req.decode_slo_time > 0.0) {
        return req.prefill_completed_at + req.decode_slo_time;
    }
    return std::nullopt;
}

double stored_prefill_lateness(const SimState& state, const RequestState& req) {
    const auto it = state.stats.per_request_prefill_lateness_by_id.find(req.request_id);
    if (it == state.stats.per_request_prefill_lateness_by_id.end()) return 0.0;
    return std::max(0.0, it->second);
}

double request_total_lateness(const SimState& state, const RequestState& req, double sim_time) {
    double pref = 0.0;
    auto itp = state.stats.per_request_prefill_lateness_by_id.find(req.request_id);
    if (itp != state.stats.per_request_prefill_lateness_by_id.end()) {
        pref = std::max(0.0, itp->second);
    } else {
        const double dl = prefill_deadline_from_request(req);
        if (std::isfinite(dl)) {
            pref = std::max(0.0, sim_time - dl);
        }
    }

    double dec = 0.0;
    auto itd = state.stats.per_request_decode_lateness_by_id.find(req.request_id);
    if (itd != state.stats.per_request_decode_lateness_by_id.end()) {
        dec = std::max(0.0, itd->second);
    } else {
        dec = std::max(0.0, req.decode_lateness);
    }
    return std::max(0.0, pref + dec);
}

double clip(double x, double lo, double hi) {
    return clamp_double(x, lo, hi);
}

double norm01(double x, double denom) {
    if (denom <= 0.0) return 0.0;
    return clip(x / denom, 0.0, 1.0);
}

double centered01(double x, double radius) {
    if (radius <= 0.0) return 0.5;
    const double clipped = clip(x, -radius, radius);
    return (clipped + radius) / (2.0 * radius);
}

bool is_prefill_request(const RequestState& req) {
    return (!req.completed) && (!req.prefill_done());
}

bool is_decode_request(const RequestState& req) {
    if (req.completed) return false;
    if (!req.prefill_done()) return false;
    return req.remaining_decode() > 0;
}

std::size_t flatten_offset(int row, int col, int d) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(d) + static_cast<std::size_t>(col);
}

}  // namespace

NativeTorchScriptInferRuntimeGV2::NativeTorchScriptInferRuntimeGV2(
    std::string device, double v_min, double v_step)
    : device_str_(std::move(device)),
      device_(parse_device(device_str_)),
      v_min_(v_min),
      v_step_(v_step) {}

c10::Device NativeTorchScriptInferRuntimeGV2::parse_device(const std::string& device) {
    try {
        return c10::Device(device);
    } catch (const std::exception&) {
        throw;
    } catch (...) {
        throw std::runtime_error("Failed to parse torch device: " + device);
    }
}

std::pair<std::string, std::string> NativeTorchScriptInferRuntimeGV2::parse_model_spec(
    const std::string& spec) {
    const std::string s = trim_copy(spec);
    if (s.empty()) {
        throw std::runtime_error("Empty model path spec");
    }

    const std::string delim = "||";
    const std::size_t pos = s.find(delim);
    if (pos == std::string::npos) {
        return {s, s};  // shared model for both players
    }

    const std::string ctrl = trim_copy(s.substr(0, pos));
    const std::string adv = trim_copy(s.substr(pos + delim.size()));
    if (ctrl.empty() || adv.empty()) {
        throw std::runtime_error(
            "Invalid model spec '" + s +
            "'. Expected '/path/model.pt' or '/path/controller.pt||/path/adversary.pt'");
    }
    return {ctrl, adv};
}

NativeInferInputsGV2 NativeTorchScriptInferRuntimeGV2::build_inputs_from_state(
    const SimState& state,
    const std::string& player,
    const std::vector<uint8_t>& action_mask,
    const NativeFeatureBuildConfigGV2& cfg_in,
    const NativeInferInputsGV2* template_inputs) const {
    NativeFeatureBuildConfigGV2 cfg = cfg_in;

    if (template_inputs != nullptr) {
        cfg.n_prefill_req = safe_pos_int(template_inputs->prefill_req_n, cfg.n_prefill_req);
        cfg.d_prefill_req = safe_pos_int(template_inputs->prefill_req_d, cfg.d_prefill_req);
        cfg.n_decode_req = safe_pos_int(template_inputs->decode_req_n, cfg.n_decode_req);
        cfg.d_decode_req = safe_pos_int(template_inputs->decode_req_d, cfg.d_decode_req);
        if (!template_inputs->global_features.empty()) {
            cfg.d_global = static_cast<int>(template_inputs->global_features.size());
        }
    }

    cfg.n_prefill_req = safe_pos_int(cfg.n_prefill_req, 10);
    cfg.d_prefill_req = safe_pos_int(cfg.d_prefill_req, 10);
    cfg.n_decode_req = safe_pos_int(cfg.n_decode_req, 50);
    cfg.d_decode_req = safe_pos_int(cfg.d_decode_req, 13);
    cfg.d_global = safe_pos_int(cfg.d_global, 24);

    const double sim_time = state.sim_time;

    std::vector<const RequestState*> prefill_reqs;
    std::vector<const RequestState*> decode_reqs;
    prefill_reqs.reserve(state.requests.size());
    decode_reqs.reserve(state.requests.size());

    std::unordered_set<int> violated_ids;
    violated_ids.reserve(state.stats.violated_request_ids.size());
    for (int rid : state.stats.violated_request_ids) violated_ids.insert(rid);

    for (const auto& req : state.requests) {
        if (is_prefill_request(req)) prefill_reqs.push_back(&req);
        if (is_decode_request(req)) decode_reqs.push_back(&req);
    }
    const int num_active_all = static_cast<int>(prefill_reqs.size() + decode_reqs.size());
    std::unordered_set<int> active_ids_all;
    active_ids_all.reserve(static_cast<std::size_t>(num_active_all));
    for (const auto* req : prefill_reqs) active_ids_all.insert(req->request_id);
    for (const auto* req : decode_reqs) active_ids_all.insert(req->request_id);

    std::sort(prefill_reqs.begin(), prefill_reqs.end(), [&](const RequestState* a, const RequestState* b) {
        const double da = prefill_deadline_from_request(*a);
        const double db = prefill_deadline_from_request(*b);
        const double ta = std::isfinite(da) ? (da - sim_time) : std::numeric_limits<double>::infinity();
        const double tb = std::isfinite(db) ? (db - sim_time) : std::numeric_limits<double>::infinity();
        if (ta != tb) return ta < tb;
        const double la = request_total_lateness(state, *a, sim_time);
        const double lb = request_total_lateness(state, *b, sim_time);
        if (la != lb) return la > lb;
        return a->request_id < b->request_id;
    });

    std::sort(decode_reqs.begin(), decode_reqs.end(), [&](const RequestState* a, const RequestState* b) {
        const bool av = a->violated || (violated_ids.find(a->request_id) != violated_ids.end());
        const bool bv = b->violated || (violated_ids.find(b->request_id) != violated_ids.end());
        if (av != bv) return av > bv;
        const double la = request_total_lateness(state, *a, sim_time);
        const double lb = request_total_lateness(state, *b, sim_time);
        if (la != lb) return la > lb;
        if (a->num_processed_decode_tokens != b->num_processed_decode_tokens) {
            return a->num_processed_decode_tokens > b->num_processed_decode_tokens;
        }
        const int ar = a->remaining_decode();
        const int br = b->remaining_decode();
        if (ar != br) return ar > br;
        return a->request_id < b->request_id;
    });
    if (static_cast<int>(decode_reqs.size()) > cfg.n_decode_req) {
        decode_reqs.resize(static_cast<std::size_t>(cfg.n_decode_req));
    }

    NativeInferInputsGV2 out;
    out.action_mask = action_mask;

    out.prefill_req_n = cfg.n_prefill_req;
    out.prefill_req_d = cfg.d_prefill_req;
    out.decode_req_n = cfg.n_decode_req;
    out.decode_req_d = cfg.d_decode_req;
    out.req_n = cfg.n_prefill_req + cfg.n_decode_req;
    out.req_d = std::max(cfg.d_prefill_req, cfg.d_decode_req);

    out.prefill_req_features.assign(
        static_cast<std::size_t>(cfg.n_prefill_req) * static_cast<std::size_t>(cfg.d_prefill_req),
        0.0f);
    out.decode_req_features.assign(
        static_cast<std::size_t>(cfg.n_decode_req) * static_cast<std::size_t>(cfg.d_decode_req),
        0.0f);
    out.prefill_req_mask.assign(static_cast<std::size_t>(cfg.n_prefill_req), 0u);
    out.decode_req_mask.assign(static_cast<std::size_t>(cfg.n_decode_req), 0u);

    const double prefill_remaining_den = safe_den(cfg.prefill_remaining_den, 4096.0);
    const double prefill_total_den = safe_den(cfg.prefill_total_den, 4096.0);
    const double decode_remaining_den = safe_den(cfg.decode_remaining_den, 864.0);
    const double decode_total_den = safe_den(cfg.decode_total_den, 864.0);
    const double decode_processed_den = safe_den(cfg.decode_processed_den, 864.0);
    const double age_den = safe_den(cfg.age_den_sec, 5.0);
    const double late_den = safe_den(cfg.lateness_den_sec, 2.0);
    const double slack_den = safe_den(cfg.slack_den_sec, 2.0);
    const double prefill_slo_den = safe_den(cfg.prefill_slo_den_sec, 2.0);
    const double decode_slo_den = safe_den(cfg.decode_slo_den_sec, 0.2);

    const int n_prefill_fill = std::min<int>(cfg.n_prefill_req, static_cast<int>(prefill_reqs.size()));
    for (int i = 0; i < n_prefill_fill; ++i) {
        const RequestState& req = *prefill_reqs[static_cast<std::size_t>(i)];
        const int total_prefill = std::max(0, req.num_prefill_tokens);
        const int rem_prefill = req.remaining_prefill();
        const int done_prefill = std::max(0, total_prefill - rem_prefill);
        const double age = std::max(0.0, sim_time - req.arrived_at);
        const double late = stored_prefill_lateness(state, req);
        const double deadline = prefill_deadline_from_request(req);
        const double slack = std::isfinite(deadline) ? (deadline - sim_time) : 0.0;
        const double prefill_slo = std::max(0.0, req.prefill_slo_time);
        const double violated = (req.violated || (violated_ids.find(req.request_id) != violated_ids.end())) ? 1.0 : 0.0;
        const double processed_frac = (total_prefill <= 0)
            ? 0.0
            : clip(static_cast<double>(done_prefill) / static_cast<double>(std::max(1, total_prefill)), 0.0, 1.0);

        out.prefill_req_features[flatten_offset(i, 0, cfg.d_prefill_req)] =
            static_cast<float>(norm01(rem_prefill, prefill_remaining_den));
        if (cfg.d_prefill_req > 1) {
            out.prefill_req_features[flatten_offset(i, 1, cfg.d_prefill_req)] =
                static_cast<float>(norm01(total_prefill, prefill_total_den));
        }
        if (cfg.d_prefill_req > 2) {
            out.prefill_req_features[flatten_offset(i, 2, cfg.d_prefill_req)] =
                static_cast<float>(processed_frac);
        }
        if (cfg.d_prefill_req > 3) {
            out.prefill_req_features[flatten_offset(i, 3, cfg.d_prefill_req)] =
                static_cast<float>(norm01(age, age_den));
        }
        if (cfg.d_prefill_req > 4) {
            out.prefill_req_features[flatten_offset(i, 4, cfg.d_prefill_req)] =
                static_cast<float>(norm01(late, late_den));
        }
        if (cfg.d_prefill_req > 5) {
            out.prefill_req_features[flatten_offset(i, 5, cfg.d_prefill_req)] =
                static_cast<float>(centered01(slack, slack_den));
        }
        if (cfg.d_prefill_req > 6) {
            out.prefill_req_features[flatten_offset(i, 6, cfg.d_prefill_req)] =
                static_cast<float>(norm01(prefill_slo, prefill_slo_den));
        }
        if (cfg.d_prefill_req > 7) {
            out.prefill_req_features[flatten_offset(i, 7, cfg.d_prefill_req)] = static_cast<float>(violated);
        }
        if (cfg.d_prefill_req > 8) {
            out.prefill_req_features[flatten_offset(i, 8, cfg.d_prefill_req)] =
                static_cast<float>(late > cfg.near_drop_lateness_low_sec ? 1.0 : 0.0);
        }
        if (cfg.d_prefill_req > 9) {
            out.prefill_req_features[flatten_offset(i, 9, cfg.d_prefill_req)] =
                static_cast<float>(late >= cfg.near_drop_lateness_high_sec ? 1.0 : 0.0);
        }
        out.prefill_req_mask[static_cast<std::size_t>(i)] = 1u;
    }

    const int n_decode_fill = std::min<int>(cfg.n_decode_req, static_cast<int>(decode_reqs.size()));
    for (int i = 0; i < n_decode_fill; ++i) {
        const RequestState& req = *decode_reqs[static_cast<std::size_t>(i)];
        const int total_decode = std::max(0, req.num_decode_tokens);
        const int rem_decode = req.remaining_decode();
        const int done_decode = std::max(0, total_decode - rem_decode);
        const double age = std::max(0.0, sim_time - req.arrived_at);
        const double late = request_total_lateness(state, req, sim_time);
        const auto decode_deadline = decode_deadline_from_request(state, req);
        const double decode_slack = decode_deadline.has_value() ? (*decode_deadline - sim_time) : 0.0;
        const double decode_slo = std::max(0.0, req.decode_slo_time);
        const double violated = (req.violated || (violated_ids.find(req.request_id) != violated_ids.end())) ? 1.0 : 0.0;
        const double processed_frac = (total_decode <= 0)
            ? 0.0
            : clip(static_cast<double>(done_decode) / static_cast<double>(std::max(1, total_decode)), 0.0, 1.0);

        out.decode_req_features[flatten_offset(i, 0, cfg.d_decode_req)] =
            static_cast<float>(norm01(rem_decode, decode_remaining_den));
        if (cfg.d_decode_req > 1) {
            out.decode_req_features[flatten_offset(i, 1, cfg.d_decode_req)] =
                static_cast<float>(norm01(total_decode, decode_total_den));
        }
        if (cfg.d_decode_req > 2) {
            out.decode_req_features[flatten_offset(i, 2, cfg.d_decode_req)] =
                static_cast<float>(norm01(done_decode, decode_processed_den));
        }
        if (cfg.d_decode_req > 3) {
            out.decode_req_features[flatten_offset(i, 3, cfg.d_decode_req)] =
                static_cast<float>(processed_frac);
        }
        if (cfg.d_decode_req > 4) {
            out.decode_req_features[flatten_offset(i, 4, cfg.d_decode_req)] =
                static_cast<float>(norm01(age, age_den));
        }
        if (cfg.d_decode_req > 5) {
            out.decode_req_features[flatten_offset(i, 5, cfg.d_decode_req)] =
                static_cast<float>(norm01(late, late_den));
        }
        if (cfg.d_decode_req > 6) {
            out.decode_req_features[flatten_offset(i, 6, cfg.d_decode_req)] =
                static_cast<float>(centered01(decode_slack, slack_den));
        }
        if (cfg.d_decode_req > 7) {
            out.decode_req_features[flatten_offset(i, 7, cfg.d_decode_req)] =
                static_cast<float>(norm01(decode_slo, decode_slo_den));
        }
        if (cfg.d_decode_req > 8) {
            out.decode_req_features[flatten_offset(i, 8, cfg.d_decode_req)] = static_cast<float>(violated);
        }
        if (cfg.d_decode_req > 9) {
            out.decode_req_features[flatten_offset(i, 9, cfg.d_decode_req)] =
                static_cast<float>(late > cfg.near_drop_lateness_low_sec ? 1.0 : 0.0);
        }
        if (cfg.d_decode_req > 10) {
            out.decode_req_features[flatten_offset(i, 10, cfg.d_decode_req)] =
                static_cast<float>(late >= cfg.near_drop_lateness_high_sec ? 1.0 : 0.0);
        }
        if (cfg.d_decode_req > 11) {
            out.decode_req_features[flatten_offset(i, 11, cfg.d_decode_req)] =
                static_cast<float>(done_decode > 216 ? 1.0 : 0.0);
        }
        if (cfg.d_decode_req > 12) {
            out.decode_req_features[flatten_offset(i, 12, cfg.d_decode_req)] =
                static_cast<float>(done_decode > 512 ? 1.0 : 0.0);
        }
        out.decode_req_mask[static_cast<std::size_t>(i)] = 1u;
    }

    const int num_prefill = static_cast<int>(prefill_reqs.size());
    const int num_decode = static_cast<int>(decode_reqs.size());
    const int num_active = num_active_all;

    int total_remaining_prefill = 0;
    for (const auto* req : prefill_reqs) total_remaining_prefill += std::max(0, req->remaining_prefill());

    int total_remaining_decode = 0;
    for (const auto* req : decode_reqs) total_remaining_decode += std::max(0, req->remaining_decode());

    int total_decode_generated_active = 0;
    for (const auto* req : decode_reqs) total_decode_generated_active += std::max(0, req->num_processed_decode_tokens);

    int num_violated_active = 0;
    for (int rid : active_ids_all) {
        if (violated_ids.find(rid) != violated_ids.end()) ++num_violated_active;
    }

    int p_late_05_15 = 0;
    int p_late_15 = 0;
    for (const auto* req : prefill_reqs) {
        const double late = stored_prefill_lateness(state, *req);
        if (late > cfg.near_drop_lateness_low_sec && late < cfg.near_drop_lateness_high_sec) ++p_late_05_15;
        else if (late >= cfg.near_drop_lateness_high_sec) ++p_late_15;
    }

    int d_late_05_15 = 0;
    int d_late_15 = 0;
    for (const auto* req : decode_reqs) {
        const double late = request_total_lateness(state, *req, sim_time);
        if (late > cfg.near_drop_lateness_low_sec && late < cfg.near_drop_lateness_high_sec) ++d_late_05_15;
        else if (late >= cfg.near_drop_lateness_high_sec) ++d_late_15;
    }

    double launch_count = 0.0;
    double launch_prefill = 0.0;
    double ewma = 0.0;
    const double ewma_alpha = safe_den(cfg.launch_ewma_alpha, 0.37);
    const double ewma_window = safe_den(cfg.launch_ewma_window_sec, 1.0);

    if (!state.stats.recent_launches.empty()) {
        for (const auto& e : state.stats.recent_launches) {
            const double dt = std::max(0.0, sim_time - e.timestamp);
            if (dt > ewma_window) continue;
            const double cnt = static_cast<double>(std::max(0, e.count));
            const double pref = static_cast<double>(std::max(0, e.prefill_tokens));
            launch_count += cnt;
            launch_prefill += pref;
            ewma += cnt * std::exp(-ewma_alpha * dt);
        }
    } else {
        for (double ts : state.stats.recent_arrivals) {
            const double dt = std::max(0.0, sim_time - ts);
            if (dt > ewma_window) continue;
            launch_count += 1.0;
            ewma += std::exp(-ewma_alpha * dt);
        }
    }

    const double objective_cost = static_cast<double>(state.stats.slo_violations) +
                                  static_cast<double>(state.stats.slo_lateness_sum);
    const double decode_credit = std::max(0, state.stats.decode_credit_balance);
    const double max_launch_count = safe_den(cfg.recent_launch_count_den, 7.0);
    const double max_launch_prefill = safe_den(cfg.recent_launch_prefill_den, 7168.0);
    const double remaining_launch_request_headroom = std::max(0.0, max_launch_count - launch_count);
    const double remaining_launch_prefill_headroom = std::max(0.0, max_launch_prefill - launch_prefill);

    out.global_features.assign(static_cast<std::size_t>(cfg.d_global), 0.0f);
    auto set_global = [&](int idx, double v) {
        if (idx < 0 || idx >= cfg.d_global) return;
        out.global_features[static_cast<std::size_t>(idx)] = static_cast<float>(v);
    };

    set_global(0, player == "controller" ? 1.0 : 0.0);
    set_global(1, player == "adversary" ? 1.0 : 0.0);
    set_global(2, norm01(objective_cost, safe_den(cfg.objective_cost_den, 50.0)));
    set_global(3, norm01(state.stats.slo_violations, safe_den(cfg.violated_count_den, 100.0)));
    set_global(4, norm01(state.stats.slo_lateness_sum, safe_den(cfg.total_lateness_den, 50.0)));
    set_global(5, norm01(num_prefill, safe_den(cfg.active_prefill_count_den, 20.0)));
    set_global(6, norm01(num_decode, safe_den(cfg.active_decode_count_den, 100.0)));
    set_global(7, norm01(num_active, safe_den(cfg.active_total_count_den, 120.0)));
    set_global(8, norm01(total_remaining_prefill, safe_den(cfg.total_remaining_prefill_den, 81920.0)));
    set_global(9, norm01(total_remaining_decode, safe_den(cfg.total_remaining_decode_den, 86400.0)));
    set_global(10, norm01(total_decode_generated_active, safe_den(cfg.total_decode_generated_active_den, 86400.0)));
    set_global(11, norm01(num_violated_active, safe_den(cfg.violated_count_den, 100.0)));
    set_global(12, norm01(p_late_05_15, safe_den(cfg.prefill_near_drop_den, 20.0)));
    set_global(13, norm01(p_late_15, safe_den(cfg.prefill_near_drop_den, 20.0)));
    set_global(14, norm01(d_late_05_15, safe_den(cfg.decode_near_drop_den, 100.0)));
    set_global(15, norm01(d_late_15, safe_den(cfg.decode_near_drop_den, 100.0)));
    set_global(16, norm01(launch_count, max_launch_count));
    set_global(17, norm01(launch_prefill, max_launch_prefill));
    set_global(18, norm01(remaining_launch_request_headroom, max_launch_count));
    set_global(19, norm01(remaining_launch_prefill_headroom, max_launch_prefill));
    set_global(20, norm01(ewma, max_launch_count));
    set_global(21, norm01(decode_credit, safe_den(cfg.decode_credit_den, 21600.0)));
    set_global(22, num_prefill > 0 ? 1.0 : 0.0);
    set_global(23, num_decode > 0 ? 1.0 : 0.0);

    out.req_features.assign(static_cast<std::size_t>(out.req_n) * static_cast<std::size_t>(out.req_d), 0.0f);
    out.req_mask.assign(static_cast<std::size_t>(out.req_n), 0u);

    const int copy_pref_d = std::min(out.req_d, cfg.d_prefill_req);
    for (int i = 0; i < cfg.n_prefill_req; ++i) {
        if (!out.prefill_req_mask[static_cast<std::size_t>(i)]) continue;
        for (int d = 0; d < copy_pref_d; ++d) {
            out.req_features[flatten_offset(i, d, out.req_d)] =
                out.prefill_req_features[flatten_offset(i, d, cfg.d_prefill_req)];
        }
        out.req_mask[static_cast<std::size_t>(i)] = 1u;
    }
    const int copy_dec_d = std::min(out.req_d, cfg.d_decode_req);
    for (int i = 0; i < cfg.n_decode_req; ++i) {
        if (!out.decode_req_mask[static_cast<std::size_t>(i)]) continue;
        const int row = cfg.n_prefill_req + i;
        for (int d = 0; d < copy_dec_d; ++d) {
            out.req_features[flatten_offset(row, d, out.req_d)] =
                out.decode_req_features[flatten_offset(i, d, cfg.d_decode_req)];
        }
        out.req_mask[static_cast<std::size_t>(row)] = 1u;
    }

    return out;
}

void NativeTorchScriptInferRuntimeGV2::load_models(
    const std::unordered_map<int, std::string>& model_version_to_path) {
    std::lock_guard<std::mutex> lock(mu_);

    if (models_.size() > 32) {
        models_.clear();
        model_specs_.clear();
    }

    for (const auto& kv : model_version_to_path) {
        const int version = kv.first;
        if (version < 0) {
            throw std::runtime_error("Negative model version in load_models");
        }
        const std::string spec = trim_copy(kv.second);
        if (spec.empty()) {
            throw std::runtime_error("Empty model spec for version " + std::to_string(version));
        }

        const auto it_model = models_.find(version);
        const auto it_spec = model_specs_.find(version);
        if (it_model != models_.end() && it_spec != model_specs_.end() && it_spec->second == spec) {
            continue;  // already loaded with same source
        }
        if (it_model != models_.end()) {
            models_.erase(it_model);
            model_specs_.erase(version);
        }

        const auto [controller_path, adversary_path] = parse_model_spec(spec);
        if (!file_exists(controller_path)) {
            throw std::runtime_error(
                "Controller TorchScript file not found for version " + std::to_string(version) +
                ": " + controller_path);
        }
        if (!file_exists(adversary_path)) {
            throw std::runtime_error(
                "Adversary TorchScript file not found for version " + std::to_string(version) +
                ": " + adversary_path);
        }

        auto pair = std::make_shared<ModelPair>();
        try {
            pair->controller = torch::jit::load(controller_path, device_);
            pair->adversary = torch::jit::load(adversary_path, device_);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed loading TorchScript models for version " + std::to_string(version) +
                ": " + std::string(e.what()));
        }

        pair->controller.eval();
        pair->adversary.eval();

        models_.emplace(version, std::move(pair));
        model_specs_[version] = spec;
    }
}

std::vector<double> NativeTorchScriptInferRuntimeGV2::masked_softmax(
    const std::vector<double>& logits,
    const std::vector<uint8_t>& action_mask) {
    if (logits.size() != action_mask.size()) {
        throw std::runtime_error("masked_softmax: logits/action_mask size mismatch");
    }
    const std::size_t n = logits.size();
    std::vector<double> out(n, 0.0);
    if (n == 0) return out;

    std::size_t allowed_count = 0;
    std::size_t finite_count = 0;
    double max_logit = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        if (!action_mask[i]) continue;
        ++allowed_count;
        if (!std::isfinite(logits[i])) continue;
        ++finite_count;
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    if (allowed_count == 0) {
        const double u = 1.0 / static_cast<double>(n);
        std::fill(out.begin(), out.end(), u);
        return out;
    }
    if (finite_count == 0) {
        const double u = 1.0 / static_cast<double>(allowed_count);
        for (std::size_t i = 0; i < n; ++i) out[i] = action_mask[i] ? u : 0.0;
        return out;
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!action_mask[i] || !std::isfinite(logits[i])) continue;
        const double e = std::exp(logits[i] - max_logit);
        out[i] = e;
        sum += e;
    }

    if (sum <= 0.0 || !std::isfinite(sum)) {
        const double u = 1.0 / static_cast<double>(allowed_count);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = action_mask[i] ? u : 0.0;
        }
        return out;
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (!action_mask[i]) {
            out[i] = 0.0;
        } else {
            out[i] /= sum;
        }
    }
    return out;
}

double NativeTorchScriptInferRuntimeGV2::decode_value_from_tensor(const torch::Tensor& value_raw) const {
    torch::Tensor t = value_raw.detach()
                          .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64))
                          .reshape({-1})
                          .contiguous();

    if (t.numel() == 0) return 0.0;
    if (t.numel() == 1) {
        const double raw = t.item<double>();
        const double value_norm = -sigmoid_stable(raw);
        return denormalize_value_model_scalar(value_norm, v_min_);
    }

    // If value head is categorical logits over bins, map to expected scalar.
    if (v_step_ <= 0.0 || !std::isfinite(v_step_)) {
        const double raw = t[0].item<double>();
        const double value_norm = -sigmoid_stable(raw);
        return denormalize_value_model_scalar(value_norm, v_min_);
    }

    const double* logits = t.data_ptr<double>();
    const int64_t n = t.numel();
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int64_t i = 0; i < n; ++i) {
        if (std::isfinite(logits[i])) {
            max_logit = std::max(max_logit, logits[i]);
        }
    }
    if (!std::isfinite(max_logit)) {
        return v_min_;
    }

    double sum = 0.0;
    std::vector<double> probs(static_cast<std::size_t>(n), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isfinite(logits[i])) continue;
        const double e = std::exp(logits[i] - max_logit);
        probs[static_cast<std::size_t>(i)] = e;
        sum += e;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        return v_min_;
    }
    double expected_bin = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        expected_bin += static_cast<double>(i) * (probs[static_cast<std::size_t>(i)] / sum);
    }
    return v_min_ + expected_bin * v_step_;
}

std::vector<double> NativeTorchScriptInferRuntimeGV2::decode_values_from_tensor_batch(
    const torch::Tensor& value_raw,
    std::size_t batch_size) const {
    std::vector<double> out(batch_size, 0.0);
    if (batch_size == 0) return out;

    torch::Tensor t = value_raw.detach()
                          .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64))
                          .contiguous();

    if (t.numel() == 0) return out;

    if (t.dim() == 0 || batch_size == 1) {
        out[0] = decode_value_from_tensor(t);
        return out;
    }

    const int64_t b = static_cast<int64_t>(batch_size);
    if (t.size(0) == b) {
        torch::Tensor flat = t.reshape({b, -1}).contiguous();
        for (int64_t i = 0; i < b; ++i) {
            out[static_cast<std::size_t>(i)] = decode_value_from_tensor(flat[i]);
        }
        return out;
    }

    torch::Tensor flat = t.reshape({-1}).contiguous();
    const int64_t total = flat.numel();
    if (total == b) {
        const double* p = flat.data_ptr<double>();
        for (int64_t i = 0; i < b; ++i) {
            const double value_norm = -sigmoid_stable(p[i]);
            out[static_cast<std::size_t>(i)] = denormalize_value_model_scalar(value_norm, v_min_);
        }
        return out;
    }
    if (total > 0 && total % b == 0) {
        const int64_t width = total / b;
        torch::Tensor rows = flat.reshape({b, width}).contiguous();
        for (int64_t i = 0; i < b; ++i) {
            out[static_cast<std::size_t>(i)] = decode_value_from_tensor(rows[i]);
        }
        return out;
    }

    throw std::runtime_error(
        "decode_values_from_tensor_batch: value tensor shape is incompatible with batch size");
}

std::pair<double, std::vector<double>> NativeTorchScriptInferRuntimeGV2::infer_from_inputs(
    const std::vector<float>& global_features,
    const std::vector<uint8_t>& action_mask,
    const std::string& player,
    int model_version) {
    NativeInferInputsGV2 in;
    in.global_features = global_features;
    in.action_mask = action_mask;
    return infer_from_inputs(in, player, model_version);
}

std::pair<double, std::vector<double>> NativeTorchScriptInferRuntimeGV2::infer_from_inputs(
    const NativeInferInputsGV2& inputs,
    const std::string& player,
    int model_version) {
    if (inputs.action_mask.empty()) {
        throw std::runtime_error("infer_from_inputs: action_mask cannot be empty");
    }
    if (player != "controller" && player != "adversary") {
        throw std::runtime_error("infer_from_inputs: invalid player '" + player + "'");
    }

    std::shared_ptr<ModelPair> pair;
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = models_.find(model_version);
        if (it == models_.end()) {
            throw std::runtime_error(
                "infer_from_inputs: model version " + std::to_string(model_version) + " not loaded");
        }
        pair = it->second;
    }

    torch::jit::script::Module& module =
        (player == "controller") ? pair->controller : pair->adversary;

    const int64_t gdim = std::max<int64_t>(1, static_cast<int64_t>(inputs.global_features.size()));
    torch::Tensor global_cpu = torch::zeros(
        {1, gdim},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    if (!inputs.global_features.empty()) {
        std::memcpy(
            global_cpu.data_ptr<float>(),
            inputs.global_features.data(),
            inputs.global_features.size() * sizeof(float));
    }
    torch::Tensor global_t = global_cpu.to(device_);

    const int64_t adim = static_cast<int64_t>(inputs.action_mask.size());
    torch::Tensor action_mask_cpu_u8 = torch::zeros(
        {1, adim},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
    {
        uint8_t* m = action_mask_cpu_u8.data_ptr<uint8_t>();
        for (int64_t i = 0; i < adim; ++i) {
            m[i] = inputs.action_mask[static_cast<std::size_t>(i)] ? 1u : 0u;
        }
    }
    torch::Tensor action_mask_t = action_mask_cpu_u8.to(device_).to(torch::kBool);

    const bool has_split =
        inputs.prefill_req_n > 0 &&
        inputs.prefill_req_d > 0 &&
        inputs.decode_req_n > 0 &&
        inputs.decode_req_d > 0 &&
        static_cast<int>(inputs.prefill_req_features.size()) ==
            inputs.prefill_req_n * inputs.prefill_req_d &&
        static_cast<int>(inputs.decode_req_features.size()) ==
            inputs.decode_req_n * inputs.decode_req_d;

    const bool has_req =
        inputs.req_n > 0 &&
        inputs.req_d > 0 &&
        static_cast<int>(inputs.req_features.size()) == inputs.req_n * inputs.req_d;

    auto make_f32_3d = [&](const std::vector<float>& flat, int n, int d) -> torch::Tensor {
        torch::Tensor cpu = torch::zeros(
            {1, std::max(1, n), std::max(1, d)},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
        if (n > 0 && d > 0 && !flat.empty()) {
            std::memcpy(cpu.data_ptr<float>(), flat.data(), flat.size() * sizeof(float));
        }
        return cpu.to(device_);
    };
    auto make_u8_2d = [&](const std::vector<uint8_t>& flat, int n, bool default_true) -> torch::Tensor {
        torch::Tensor cpu = torch::zeros(
            {1, std::max(1, n)},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
        uint8_t* p = cpu.data_ptr<uint8_t>();
        if (n > 0 && static_cast<int>(flat.size()) == n) {
            std::memcpy(p, flat.data(), flat.size() * sizeof(uint8_t));
        } else if (default_true && n > 0) {
            std::fill(p, p + n, static_cast<uint8_t>(1));
        }
        return cpu.to(device_).to(torch::kBool);
    };

    torch::Tensor prefill_req_features_t = make_f32_3d(
        has_split ? inputs.prefill_req_features : std::vector<float>{}, has_split ? inputs.prefill_req_n : 1,
        has_split ? inputs.prefill_req_d : 1);
    torch::Tensor decode_req_features_t = make_f32_3d(
        has_split ? inputs.decode_req_features : std::vector<float>{}, has_split ? inputs.decode_req_n : 1,
        has_split ? inputs.decode_req_d : 1);
    torch::Tensor prefill_req_mask_t = make_u8_2d(
        has_split ? inputs.prefill_req_mask : std::vector<uint8_t>{}, has_split ? inputs.prefill_req_n : 1,
        true);
    torch::Tensor decode_req_mask_t = make_u8_2d(
        has_split ? inputs.decode_req_mask : std::vector<uint8_t>{}, has_split ? inputs.decode_req_n : 1,
        true);

    torch::Tensor req_features_t = make_f32_3d(
        has_req ? inputs.req_features : std::vector<float>{}, has_req ? inputs.req_n : 1,
        has_req ? inputs.req_d : 1);
    torch::Tensor req_mask_t = make_u8_2d(
        has_req ? inputs.req_mask : std::vector<uint8_t>{}, has_req ? inputs.req_n : 1, true);

    c10::IValue out_iv;
    std::string last_err;

    torch::InferenceMode infer_mode_guard(true);
    torch::NoGradGuard no_grad_guard;

    auto try_forward = [&](std::initializer_list<torch::jit::IValue> args) -> bool {
        try {
            std::vector<torch::jit::IValue> v;
            v.reserve(args.size());
            for (const auto& x : args) v.push_back(x);
            out_iv = module.forward(v);
            return true;
        } catch (const std::exception& e) {
            last_err = e.what();
            return false;
        }
    };

    bool ok = false;
    if (has_split) {
        ok = try_forward(
            {prefill_req_features_t, decode_req_features_t, global_t, prefill_req_mask_t,
             decode_req_mask_t, action_mask_t});
        if (!ok) ok = try_forward({prefill_req_features_t, decode_req_features_t, global_t, action_mask_t});
        if (!ok) ok = try_forward({prefill_req_features_t, decode_req_features_t, global_t});
    }
    if (!ok) ok = try_forward({req_features_t, global_t, req_mask_t, action_mask_t});
    if (!ok) ok = try_forward({req_features_t, global_t, action_mask_t});
    if (!ok) ok = try_forward({req_features_t, global_t});
    if (!ok) ok = try_forward({global_t, action_mask_t});
    if (!ok) ok = try_forward({global_t});
    if (!ok) {
        throw std::runtime_error(
            "TorchScript forward failed for all supported signatures. Last error: " + last_err);
    }

    auto [policy_logits, value_raw] = parse_forward_output(out_iv);

    torch::Tensor policy_cpu = policy_logits.detach()
                                   .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64))
                                   .reshape({-1})
                                   .contiguous();

    std::vector<double> logits(inputs.action_mask.size(), -1e30);
    const int64_t k = std::min<int64_t>(policy_cpu.numel(), static_cast<int64_t>(logits.size()));
    if (k > 0) {
        const double* p = policy_cpu.data_ptr<double>();
        for (int64_t i = 0; i < k; ++i) {
            logits[static_cast<std::size_t>(i)] = p[i];
        }
    }

    const std::vector<double> priors = masked_softmax(logits, inputs.action_mask);
    const double value = decode_value_from_tensor(value_raw);

    return {value, priors};
}

std::vector<double> NativeTorchScriptInferRuntimeGV2::infer_values_from_inputs_batch(
    const std::vector<NativeInferInputsGV2>& inputs,
    const std::string& player,
    int model_version) {
    if (inputs.empty()) return {};
    if (player != "controller" && player != "adversary") {
        throw std::runtime_error("infer_values_from_inputs_batch: invalid player '" + player + "'");
    }

    std::shared_ptr<ModelPair> pair;
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = models_.find(model_version);
        if (it == models_.end()) {
            throw std::runtime_error(
                "infer_values_from_inputs_batch: model version " + std::to_string(model_version) + " not loaded");
        }
        pair = it->second;
    }

    torch::jit::script::Module& module =
        (player == "controller") ? pair->controller : pair->adversary;

    const NativeInferInputsGV2& first = inputs.front();
    if (first.action_mask.empty()) {
        throw std::runtime_error("infer_values_from_inputs_batch: action_mask cannot be empty");
    }

    const int64_t batch = static_cast<int64_t>(inputs.size());
    const int64_t gdim = std::max<int64_t>(1, static_cast<int64_t>(first.global_features.size()));
    const int64_t adim = static_cast<int64_t>(first.action_mask.size());

    const bool has_split =
        first.prefill_req_n > 0 &&
        first.prefill_req_d > 0 &&
        first.decode_req_n > 0 &&
        first.decode_req_d > 0 &&
        static_cast<int>(first.prefill_req_features.size()) ==
            first.prefill_req_n * first.prefill_req_d &&
        static_cast<int>(first.decode_req_features.size()) ==
            first.decode_req_n * first.decode_req_d;

    const bool has_req =
        first.req_n > 0 &&
        first.req_d > 0 &&
        static_cast<int>(first.req_features.size()) == first.req_n * first.req_d;

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const NativeInferInputsGV2& in = inputs[i];
        if (static_cast<int64_t>(in.global_features.size()) != gdim) {
            throw std::runtime_error("infer_values_from_inputs_batch: inconsistent global feature width");
        }
        if (static_cast<int64_t>(in.action_mask.size()) != adim) {
            throw std::runtime_error("infer_values_from_inputs_batch: inconsistent action mask width");
        }
        if (has_split) {
            if (in.prefill_req_n != first.prefill_req_n ||
                in.prefill_req_d != first.prefill_req_d ||
                in.decode_req_n != first.decode_req_n ||
                in.decode_req_d != first.decode_req_d ||
                in.prefill_req_features.size() != first.prefill_req_features.size() ||
                in.decode_req_features.size() != first.decode_req_features.size() ||
                in.prefill_req_mask.size() != first.prefill_req_mask.size() ||
                in.decode_req_mask.size() != first.decode_req_mask.size()) {
                throw std::runtime_error("infer_values_from_inputs_batch: inconsistent split feature shape");
            }
        }
        if (has_req) {
            if (in.req_n != first.req_n ||
                in.req_d != first.req_d ||
                in.req_features.size() != first.req_features.size() ||
                in.req_mask.size() != first.req_mask.size()) {
                throw std::runtime_error("infer_values_from_inputs_batch: inconsistent req feature shape");
            }
        }
    }

    torch::Tensor global_cpu = torch::zeros(
        {batch, gdim},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    {
        float* dst = global_cpu.data_ptr<float>();
        for (int64_t b = 0; b < batch; ++b) {
            const auto& src = inputs[static_cast<std::size_t>(b)].global_features;
            if (!src.empty()) {
                std::memcpy(dst + b * gdim, src.data(), src.size() * sizeof(float));
            }
        }
    }
    torch::Tensor global_t = global_cpu.to(device_);

    torch::Tensor action_mask_cpu_u8 = torch::zeros(
        {batch, adim},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
    {
        uint8_t* dst = action_mask_cpu_u8.data_ptr<uint8_t>();
        for (int64_t b = 0; b < batch; ++b) {
            const auto& src = inputs[static_cast<std::size_t>(b)].action_mask;
            std::memcpy(dst + b * adim, src.data(), src.size() * sizeof(uint8_t));
        }
    }
    torch::Tensor action_mask_t = action_mask_cpu_u8.to(device_).to(torch::kBool);

    auto make_f32_3d_batch = [&](auto get_flat, int n, int d) -> torch::Tensor {
        const int64_t nn = std::max<int64_t>(1, n);
        const int64_t dd = std::max<int64_t>(1, d);
        torch::Tensor cpu = torch::zeros(
            {batch, nn, dd},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
        float* dst = cpu.data_ptr<float>();
        for (int64_t b = 0; b < batch; ++b) {
            const std::vector<float>& src = get_flat(inputs[static_cast<std::size_t>(b)]);
            if (!src.empty()) {
                std::memcpy(dst + b * nn * dd, src.data(), src.size() * sizeof(float));
            }
        }
        return cpu.to(device_);
    };

    auto make_u8_2d_batch = [&](auto get_mask, int n, bool default_true) -> torch::Tensor {
        const int64_t nn = std::max<int64_t>(1, n);
        torch::Tensor cpu = torch::zeros(
            {batch, nn},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
        uint8_t* dst = cpu.data_ptr<uint8_t>();
        for (int64_t b = 0; b < batch; ++b) {
            uint8_t* row = dst + b * nn;
            const std::vector<uint8_t>& src = get_mask(inputs[static_cast<std::size_t>(b)]);
            if (!src.empty()) {
                std::memcpy(row, src.data(), src.size() * sizeof(uint8_t));
            } else if (default_true) {
                std::fill(row, row + nn, static_cast<uint8_t>(1));
            }
        }
        return cpu.to(device_).to(torch::kBool);
    };

    torch::Tensor prefill_req_features_t = make_f32_3d_batch(
        [](const NativeInferInputsGV2& in) -> const std::vector<float>& { return in.prefill_req_features; },
        has_split ? first.prefill_req_n : 1,
        has_split ? first.prefill_req_d : 1);
    torch::Tensor decode_req_features_t = make_f32_3d_batch(
        [](const NativeInferInputsGV2& in) -> const std::vector<float>& { return in.decode_req_features; },
        has_split ? first.decode_req_n : 1,
        has_split ? first.decode_req_d : 1);
    torch::Tensor prefill_req_mask_t = make_u8_2d_batch(
        [](const NativeInferInputsGV2& in) -> const std::vector<uint8_t>& { return in.prefill_req_mask; },
        has_split ? first.prefill_req_n : 1,
        true);
    torch::Tensor decode_req_mask_t = make_u8_2d_batch(
        [](const NativeInferInputsGV2& in) -> const std::vector<uint8_t>& { return in.decode_req_mask; },
        has_split ? first.decode_req_n : 1,
        true);

    torch::Tensor req_features_t = make_f32_3d_batch(
        [](const NativeInferInputsGV2& in) -> const std::vector<float>& { return in.req_features; },
        has_req ? first.req_n : 1,
        has_req ? first.req_d : 1);
    torch::Tensor req_mask_t = make_u8_2d_batch(
        [](const NativeInferInputsGV2& in) -> const std::vector<uint8_t>& { return in.req_mask; },
        has_req ? first.req_n : 1,
        true);

    c10::IValue out_iv;
    std::string last_err;
    torch::InferenceMode infer_mode_guard(true);
    torch::NoGradGuard no_grad_guard;

    auto try_forward = [&](std::initializer_list<torch::jit::IValue> args) -> bool {
        try {
            std::vector<torch::jit::IValue> v;
            v.reserve(args.size());
            for (const auto& x : args) v.push_back(x);
            out_iv = module.forward(v);
            return true;
        } catch (const std::exception& e) {
            last_err = e.what();
            return false;
        }
    };

    bool ok = false;
    if (has_split) {
        ok = try_forward(
            {prefill_req_features_t, decode_req_features_t, global_t, prefill_req_mask_t,
             decode_req_mask_t, action_mask_t});
        if (!ok) ok = try_forward({prefill_req_features_t, decode_req_features_t, global_t, action_mask_t});
        if (!ok) ok = try_forward({prefill_req_features_t, decode_req_features_t, global_t});
    }
    if (!ok) ok = try_forward({req_features_t, global_t, req_mask_t, action_mask_t});
    if (!ok) ok = try_forward({req_features_t, global_t, action_mask_t});
    if (!ok) ok = try_forward({req_features_t, global_t});
    if (!ok) ok = try_forward({global_t, action_mask_t});
    if (!ok) ok = try_forward({global_t});
    if (!ok) {
        throw std::runtime_error(
            "TorchScript batch forward failed for all supported signatures. Last error: " + last_err);
    }

    auto [_policy_logits, value_raw] = parse_forward_output(out_iv);
    return decode_values_from_tensor_batch(value_raw, inputs.size());
}

const std::string& NativeTorchScriptInferRuntimeGV2::device() const { return device_str_; }
double NativeTorchScriptInferRuntimeGV2::v_min() const { return v_min_; }
double NativeTorchScriptInferRuntimeGV2::v_step() const { return v_step_; }

}  // namespace mcts_native_gv2
