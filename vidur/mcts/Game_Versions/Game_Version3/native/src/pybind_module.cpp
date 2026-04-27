#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gv2_infer_runtime.hpp"
#include "gv2_logger.hpp"
#include "gv2_mcts_dnn.hpp"
#include "gv2_types.hpp"
#include "gv3_native_selfplay.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using namespace mcts_native_gv2;

namespace {

constexpr int kPyMetaNextAdvTick = -9100001;
constexpr int kPyMetaLastAdvTick = -9100002;
constexpr int kPyMetaDecodeCreditBalance = -9100005;
constexpr int kPyMetaMissedAdvSource = -9100006;

std::vector<uint8_t> py_to_mask(const py::handle& obj) {
    std::vector<uint8_t> out;
    if (obj.is_none()) return out;
    for (auto item : obj) {
        if (py::isinstance<py::bool_>(item)) {
            out.push_back(py::cast<bool>(item) ? 1u : 0u);
        } else {
            out.push_back(py::cast<int>(item) ? 1u : 0u);
        }
    }
    return out;
}

std::vector<float> py_to_f32_vec(const py::handle& obj) {
    std::vector<float> out;
    if (obj.is_none()) return out;
    for (auto item : obj) out.push_back(py::cast<float>(item));
    return out;
}

std::vector<float> py_to_f32_flat(const py::handle& obj) {
    std::vector<float> out;
    if (obj.is_none()) return out;
    try {
        for (auto item : obj) {
            try {
                out.push_back(py::cast<float>(item));
                continue;
            } catch (const std::exception&) {
            }
            try {
                for (auto sub : item) {
                    out.push_back(py::cast<float>(sub));
                }
            } catch (const std::exception&) {
            }
        }
    } catch (const std::exception&) {
    }
    return out;
}

std::vector<uint8_t> py_to_mask_flat(const py::handle& obj) {
    std::vector<uint8_t> out;
    if (obj.is_none()) return out;
    try {
        for (auto item : obj) {
            try {
                if (py::isinstance<py::bool_>(item)) {
                    out.push_back(py::cast<bool>(item) ? 1u : 0u);
                } else {
                    out.push_back(py::cast<int>(item) ? 1u : 0u);
                }
                continue;
            } catch (const std::exception&) {
            }
            try {
                for (auto sub : item) {
                    if (py::isinstance<py::bool_>(sub)) {
                        out.push_back(py::cast<bool>(sub) ? 1u : 0u);
                    } else {
                        out.push_back(py::cast<int>(sub) ? 1u : 0u);
                    }
                }
            } catch (const std::exception&) {
            }
        }
    } catch (const std::exception&) {
    }
    return out;
}

std::vector<int> py_to_i32_vec(const py::handle& obj) {
    std::vector<int> out;
    if (obj.is_none()) return out;
    if (py::isinstance<py::str>(obj) || py::isinstance<py::bytes>(obj)) return out;
    try {
        for (auto item : obj) {
            try {
                out.push_back(py::cast<int>(item));
            } catch (const std::exception&) {
            }
        }
    } catch (const std::exception&) {
    }
    return out;
}

std::vector<double> py_to_f64_vec(const py::handle& obj) {
    std::vector<double> out;
    if (obj.is_none()) return out;
    if (py::isinstance<py::str>(obj) || py::isinstance<py::bytes>(obj)) return out;
    try {
        for (auto item : obj) {
            try {
                out.push_back(py::cast<double>(item));
            } catch (const std::exception&) {
            }
        }
    } catch (const std::exception&) {
    }
    return out;
}

std::vector<LaunchWindowEntry> py_to_launch_vec(const py::handle& obj) {
    std::vector<LaunchWindowEntry> out;
    if (obj.is_none()) return out;
    if (py::isinstance<py::str>(obj) || py::isinstance<py::bytes>(obj)) return out;
    try {
        for (auto item : obj) {
            LaunchWindowEntry e;
            bool ok = false;
            if (py::isinstance<py::tuple>(item) || py::isinstance<py::list>(item)) {
                try {
                    const py::sequence s = py::reinterpret_borrow<py::sequence>(item);
                    if (py::len(s) >= 3) {
                        e.timestamp = py::cast<double>(s[0]);
                        e.count = py::cast<int>(s[1]);
                        e.prefill_tokens = py::cast<int>(s[2]);
                        ok = true;
                    } else if (py::len(s) >= 1) {
                        e.timestamp = py::cast<double>(s[0]);
                        e.count = 1;
                        e.prefill_tokens = 0;
                        ok = true;
                    }
                } catch (const std::exception&) {
                }
            } else if (py::isinstance<py::dict>(item)) {
                try {
                    const py::dict d = py::reinterpret_borrow<py::dict>(item);
                    if (d.contains("timestamp")) e.timestamp = py::cast<double>(d["timestamp"]);
                    else if (d.contains("time")) e.timestamp = py::cast<double>(d["time"]);
                    e.count = d.contains("count") ? py::cast<int>(d["count"]) : 0;
                    if (e.count == 0 && d.contains("requests")) e.count = py::cast<int>(d["requests"]);
                    e.prefill_tokens = d.contains("prefill_tokens")
                        ? py::cast<int>(d["prefill_tokens"])
                        : 0;
                    if (e.prefill_tokens == 0 && d.contains("tokens")) {
                        e.prefill_tokens = py::cast<int>(d["tokens"]);
                    }
                    ok = true;
                } catch (const std::exception&) {
                }
            } else {
                try {
                    e.timestamp = py::cast<double>(item);
                    e.count = 1;
                    e.prefill_tokens = 0;
                    ok = true;
                } catch (const std::exception&) {
                }
            }
            if (ok) {
                e.count = std::max(0, e.count);
                e.prefill_tokens = std::max(0, e.prefill_tokens);
                out.push_back(e);
            }
        }
    } catch (const std::exception&) {
    }
    return out;
}

int py_key_to_int(const py::handle& key, bool* ok_out = nullptr) {
    if (ok_out) *ok_out = false;
    try {
        const int v = py::cast<int>(key);
        if (ok_out) *ok_out = true;
        return v;
    } catch (const std::exception&) {
    }
    try {
        const std::string s = py::cast<std::string>(key);
        const int v = std::stoi(s);
        if (ok_out) *ok_out = true;
        return v;
    } catch (const std::exception&) {
    }
    return 0;
}

template <typename V>
std::unordered_map<int, V> py_to_i32_map(const py::handle& obj) {
    std::unordered_map<int, V> out;
    if (obj.is_none() || !py::isinstance<py::dict>(obj)) return out;
    const py::dict d = py::reinterpret_borrow<py::dict>(obj);
    for (auto item : d) {
        bool ok = false;
        const int k = py_key_to_int(item.first, &ok);
        if (!ok) continue;
        try {
            out[k] = py::cast<V>(item.second);
        } catch (const std::exception&) {
        }
    }
    return out;
}

std::string json_i32_vec(const std::vector<int>& xs) {
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

std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c < 0x20u) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec << std::setfill(' ');
                } else {
                    oss << static_cast<char>(c);
                }
        }
    }
    return oss.str();
}

std::string json_string(const std::string& s) {
    return "\"" + json_escape(s) + "\"";
}

std::string json_string_vec(const std::vector<std::string>& xs) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < xs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << json_string(xs[i]);
    }
    oss << "]";
    return oss.str();
}

std::string build_native_config_json(
    const SearchInput& in,
    int model_version,
    const std::string& iter_log_path,
    const std::string& root_log_path) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "{\n";
    oss << "  \"contract_version\": " << json_string(in.contract_version) << ",\n";
    oss << "  \"model_version\": " << model_version << ",\n";
    oss << "  \"root\": {\n";
    oss << "    \"player\": " << json_string(in.root_player) << ",\n";
    oss << "    \"iterations\": " << in.iterations << ",\n";
    oss << "    \"game_id\": " << in.game_id << ",\n";
    oss << "    \"root_id\": " << in.root_id << ",\n";
    oss << "    \"root_node_id\": " << in.root_node_id << ",\n";
    oss << "    \"root_depth\": " << in.root_depth << ",\n";
    oss << "    \"seed\": " << in.seed << ",\n";
    oss << "    \"decision_state_time\": " << in.decision_state_time << ",\n";
    oss << "    \"root_phase\": " << json_string(in.root_phase) << ",\n";
    oss << "    \"cycle_label\": " << json_string(in.cycle_label) << "\n";
    oss << "  },\n";
    oss << "  \"search\": {\n";
    oss << "    \"max_forced_hops\": " << in.max_forced_hops << ",\n";
    oss << "    \"pb_c_base\": " << in.pb_c_base << ",\n";
    oss << "    \"pb_c_init\": " << in.pb_c_init << ",\n";
    oss << "    \"discount_factor\": " << in.discount_factor << ",\n";
    oss << "    \"prefill_step_time\": " << in.prefill_step_time << ",\n";
    oss << "    \"reward_knee\": " << in.reward_knee << ",\n";
    oss << "    \"reward_max_penalty\": " << in.reward_max_penalty << ",\n";
    oss << "    \"reward_tail_alpha\": " << in.reward_tail_alpha << ",\n";
    oss << "    \"reuse_root_infer_inputs\": " << (in.reuse_root_infer_inputs ? "true" : "false") << ",\n";
    oss << "    \"root_dirichlet_noise_enabled\": " << (in.root_dirichlet_noise_enabled ? "true" : "false") << ",\n";
    oss << "    \"root_dirichlet_alpha\": " << in.root_dirichlet_alpha << ",\n";
    oss << "    \"root_dirichlet_epsilon\": " << in.root_dirichlet_epsilon << "\n";
    oss << "  },\n";
    oss << "  \"environment\": {\n";
    oss << "    \"adversary_tick_sec\": " << in.env_cfg.adversary_tick_sec << ",\n";
    oss << "    \"launch_window_sec\": " << in.env_cfg.launch_window_sec << ",\n";
    oss << "    \"max_requests_per_launch_window\": " << in.env_cfg.max_requests_per_launch_window << ",\n";
    oss << "    \"prefill_window_cap_tokens\": " << in.env_cfg.prefill_window_cap_tokens << ",\n";
    oss << "    \"max_prefill_tokens_per_request\": " << in.env_cfg.max_prefill_tokens_per_request << ",\n";
    oss << "    \"max_decode_tokens_per_request\": " << in.env_cfg.max_decode_tokens_per_request << ",\n";
    oss << "    \"min_decode_tokens_per_request\": " << in.env_cfg.min_decode_tokens_per_request << ",\n";
    oss << "    \"decode_slo_time_default\": " << in.env_cfg.decode_slo_time_default << ",\n";
    oss << "    \"auto_drop_lateness_sec\": " << in.env_cfg.auto_drop_lateness_sec << ",\n";
    oss << "    \"drop_cost\": " << in.env_cfg.drop_cost << ",\n";
    oss << "    \"controller_noop_prefill_only_jump_to_next_adv_tick\": "
        << (in.env_cfg.controller_noop_prefill_only_jump_to_next_adv_tick ? "true" : "false") << ",\n";
    oss << "    \"enforce_nonnegative_decode_credits\": "
        << (in.env_cfg.enforce_nonnegative_decode_credits ? "true" : "false") << ",\n";
    oss << "    \"decode_credit_mint_per_prefill_complete\": " << in.env_cfg.decode_credit_mint_per_prefill_complete
        << ",\n";
    oss << "    \"adversary_sampler\": {\n";
    oss << "      \"max_launch_count_per_tick\": " << in.env_cfg.adversary_sampler.max_launch_count_per_tick << ",\n";
    oss << "      \"allowed_prefill_tokens\": " << json_i32_vec(in.env_cfg.adversary_sampler.allowed_prefill_tokens) << ",\n";
    oss << "      \"stop_rule_names\": " << json_string_vec(in.env_cfg.adversary_sampler.stop_rule_names) << ",\n";
    oss << "      \"strict_masking\": " << (in.env_cfg.adversary_sampler.strict_masking ? "true" : "false") << ",\n";
    oss << "      \"prefill_slo_by_tokens\": " << json_i32_f64_map(in.env_cfg.adversary_sampler.prefill_slo_by_tokens) << "\n";
    oss << "    },\n";
    oss << "    \"controller_sampler\": {\n";
    oss << "      \"eviction_rule_names\": " << json_string_vec(in.env_cfg.controller_sampler.eviction_rule_names) << ",\n";
    oss << "      \"prefill_budget_options\": " << json_i32_vec(in.env_cfg.controller_sampler.prefill_budget_options) << ",\n";
    oss << "      \"ordering_heuristics\": " << json_string_vec(in.env_cfg.controller_sampler.ordering_heuristics) << ",\n";
    oss << "      \"strict_masking\": " << (in.env_cfg.controller_sampler.strict_masking ? "true" : "false") << ",\n";
    oss << "      \"prefill_eta_tokens_per_sec\": " << in.env_cfg.controller_sampler.prefill_eta_tokens_per_sec << ",\n";
    oss << "      \"enforce_nonnegative_decode_credits\": "
        << (in.env_cfg.controller_sampler.enforce_nonnegative_decode_credits ? "true" : "false") << "\n";
    oss << "    }\n";
    oss << "  },\n";
    oss << "  \"simulator\": {\n";
    oss << "    \"predictor_csv_path\": " << json_string(in.predictor_csv_path) << ",\n";
    oss << "    \"adversary_tick_sec\": " << in.sim_cfg.adversary_tick_sec << ",\n";
    oss << "    \"prefill_profile_tokens\": " << json_i32_vec(in.sim_cfg.prefill_profile_tokens) << ",\n";
    oss << "    \"prefill_profile_times\": " << json_f64_vec(in.sim_cfg.prefill_profile_times) << ",\n";
    oss << "    \"fallback_total_time_sec\": " << in.sim_cfg.fallback_total_time_sec << ",\n";
    oss << "    \"fallback_model_time_sec\": " << in.sim_cfg.fallback_model_time_sec << "\n";
    oss << "  },\n";
    oss << "  \"inputs\": {\n";
    oss << "    \"action_mask_size\": " << in.action_mask.size() << ",\n";
    oss << "    \"global_feature_size\": " << in.global_features.size() << ",\n";
    oss << "    \"prefill_req_n\": " << in.root_infer_inputs.prefill_req_n << ",\n";
    oss << "    \"prefill_req_d\": " << in.root_infer_inputs.prefill_req_d << ",\n";
    oss << "    \"decode_req_n\": " << in.root_infer_inputs.decode_req_n << ",\n";
    oss << "    \"decode_req_d\": " << in.root_infer_inputs.decode_req_d << ",\n";
    oss << "    \"req_n\": " << in.root_infer_inputs.req_n << ",\n";
    oss << "    \"req_d\": " << in.root_infer_inputs.req_d << "\n";
    oss << "  },\n";
    oss << "  \"logging\": {\n";
    oss << "    \"log_events\": " << (in.log_events ? "true" : "false") << ",\n";
    oss << "    \"profile\": " << (in.profile ? "true" : "false") << ",\n";
    oss << "    \"iter_log_path\": " << json_string(iter_log_path) << ",\n";
    oss << "    \"root_log_path\": " << json_string(root_log_path) << "\n";
    oss << "  }\n";
    oss << "}\n";
    return oss.str();
}

void sort_unique(std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

template <typename T>
bool dict_try_get(const py::dict& d, std::initializer_list<const char*> keys, T* out) {
    for (const char* k : keys) {
        if (!d.contains(k)) continue;
        try {
            *out = py::cast<T>(d[k]);
            return true;
        } catch (const std::exception&) {
        }
    }
    return false;
}

template <typename T>
bool merged_try_get(
    const py::dict& primary,
    const py::dict* secondary,
    std::initializer_list<const char*> keys,
    T* out) {
    if (dict_try_get(primary, keys, out)) return true;
    if (secondary != nullptr) {
        if (dict_try_get(*secondary, keys, out)) return true;
    }
    return false;
}

template <typename T>
T merged_get_or(
    const py::dict& primary,
    const py::dict* secondary,
    std::initializer_list<const char*> keys,
    T default_value) {
    T out = default_value;
    (void)merged_try_get(primary, secondary, keys, &out);
    return out;
}

bool dict_or_nested_try_get(
    const py::dict& d,
    const char* nested_key,
    const char* key,
    py::object* out) {
    if (d.contains(key)) {
        *out = py::reinterpret_borrow<py::object>(d[key]);
        return true;
    }
    if (d.contains(nested_key) && py::isinstance<py::dict>(d[nested_key])) {
        const py::dict nested = py::reinterpret_borrow<py::dict>(d[nested_key]);
        if (nested.contains(key)) {
            *out = py::reinterpret_borrow<py::object>(nested[key]);
            return true;
        }
    }
    return false;
}

bool parse_python_request_counter(const py::dict& d, int* last_request_id) {
    py::object counters;
    if (!dict_or_nested_try_get(d, "simulator", "entity_counters", &counters)) return false;
    if (counters.is_none() || !py::isinstance<py::dict>(counters)) return false;
    const py::dict cd = py::reinterpret_borrow<py::dict>(counters);
    for (auto item : cd) {
        try {
            const std::string key = py::cast<std::string>(item.first);
            if (key != "Request") continue;
            *last_request_id = py::cast<int>(item.second);
            return true;
        } catch (const std::exception&) {
        }
    }
    return false;
}

void apply_python_gv3_meta(GameStats* out, double sim_time) {
    const auto next_it = out->decode_next_deadline_by_id.find(kPyMetaNextAdvTick);
    if (next_it != out->decode_next_deadline_by_id.end()) {
        out->next_adv_tick = double(next_it->second);
    }
    const auto last_it = out->decode_next_deadline_by_id.find(kPyMetaLastAdvTick);
    if (last_it != out->decode_next_deadline_by_id.end()) {
        out->last_adv_tick = double(last_it->second);
    }
    const auto missed_it = out->decode_next_deadline_by_id.find(kPyMetaMissedAdvSource);
    if (missed_it != out->decode_next_deadline_by_id.end()) {
        out->missed_adv_source = int(std::llround(double(missed_it->second)));
    }
    const auto credit_it = out->decode_tokens_counted_by_id.find(kPyMetaDecodeCreditBalance);
    if (credit_it != out->decode_tokens_counted_by_id.end()) {
        out->decode_credit_balance = int(credit_it->second);
        out->decode_credit_available = std::max(0, out->decode_credit_balance);
    }
    if (out->next_adv_tick >= 0.0) {
        out->pending_adv_tick = out->next_adv_tick <= (sim_time + 1e-9);
    }
}

void parse_stats_fields_into(
    const py::dict& d,
    const py::dict* stats_d,
    GameStats* out) {
    out->requests_generated = merged_get_or<int>(
        d, stats_d, {"requests_generated"}, out->requests_generated);
    out->requests_completed = merged_get_or<int>(
        d, stats_d, {"requests_completed"}, out->requests_completed);
    out->slo_violations = merged_get_or<int>(
        d, stats_d, {"slo_violations"}, out->slo_violations);
    out->slo_lateness_sum = merged_get_or<double>(
        d, stats_d, {"slo_lateness_sum", "total_lateness"}, out->slo_lateness_sum);
    out->decode_credit_balance = merged_get_or<int>(
        d, stats_d, {"decode_credit_balance"}, out->decode_credit_balance);
    out->decode_credit_available = merged_get_or<int>(
        d, stats_d, {"decode_credit_available"}, out->decode_credit_available);
    out->pending_adv_tick = merged_get_or<bool>(
        d, stats_d, {"pending_adv_tick"}, out->pending_adv_tick);
    out->last_adv_tick = merged_get_or<double>(
        d, stats_d, {"last_adv_tick"}, out->last_adv_tick);
    out->next_adv_tick = merged_get_or<double>(
        d, stats_d, {"next_adv_tick"}, out->next_adv_tick);
    out->missed_adv_source = merged_get_or<int>(
        d, stats_d, {"missed_adv_source"}, out->missed_adv_source);

    {
        bool have = false;
        std::vector<int> ids;
        if (d.contains("active_request_ids")) {
            have = true;
            ids = py_to_i32_vec(d["active_request_ids"]);
        } else if (stats_d != nullptr && stats_d->contains("active_request_ids")) {
            have = true;
            ids = py_to_i32_vec((*stats_d)["active_request_ids"]);
        }
        if (have) out->active_request_ids = std::move(ids);
    }
    {
        bool have = false;
        std::vector<int> ids;
        if (d.contains("completed_request_ids")) {
            have = true;
            ids = py_to_i32_vec(d["completed_request_ids"]);
        } else if (stats_d != nullptr && stats_d->contains("completed_request_ids")) {
            have = true;
            ids = py_to_i32_vec((*stats_d)["completed_request_ids"]);
        }
        if (have) out->completed_request_ids = std::move(ids);
    }
    {
        bool have = false;
        std::vector<int> ids;
        if (d.contains("dropped_request_ids")) {
            have = true;
            ids = py_to_i32_vec(d["dropped_request_ids"]);
        } else if (stats_d != nullptr && stats_d->contains("dropped_request_ids")) {
            have = true;
            ids = py_to_i32_vec((*stats_d)["dropped_request_ids"]);
        }
        if (have) out->dropped_request_ids = std::move(ids);
    }
    {
        bool have = false;
        std::vector<int> ids;
        if (d.contains("stopped_decode_request_ids")) {
            have = true;
            ids = py_to_i32_vec(d["stopped_decode_request_ids"]);
        } else if (stats_d != nullptr && stats_d->contains("stopped_decode_request_ids")) {
            have = true;
            ids = py_to_i32_vec((*stats_d)["stopped_decode_request_ids"]);
        }
        if (have) out->stopped_decode_request_ids = std::move(ids);
    }
    {
        bool have = false;
        std::vector<int> ids;
        if (d.contains("violated_request_ids")) {
            have = true;
            ids = py_to_i32_vec(d["violated_request_ids"]);
        } else if (stats_d != nullptr && stats_d->contains("violated_request_ids")) {
            have = true;
            ids = py_to_i32_vec((*stats_d)["violated_request_ids"]);
        }
        if (have) out->violated_request_ids = std::move(ids);
    }

    {
        bool have = false;
        std::vector<double> xs;
        std::vector<LaunchWindowEntry> launches;
        if (d.contains("recent_arrivals")) {
            have = true;
            xs = py_to_f64_vec(d["recent_arrivals"]);
            launches = py_to_launch_vec(d["recent_arrivals"]);
        } else if (stats_d != nullptr && stats_d->contains("recent_arrivals")) {
            have = true;
            xs = py_to_f64_vec((*stats_d)["recent_arrivals"]);
            launches = py_to_launch_vec((*stats_d)["recent_arrivals"]);
        }
        if (have) {
            out->recent_arrivals = std::move(xs);
            if (!launches.empty()) out->recent_launches = std::move(launches);
        }
    }

    {
        bool have = false;
        std::vector<LaunchWindowEntry> launches;
        if (d.contains("recent_launches")) {
            have = true;
            launches = py_to_launch_vec(d["recent_launches"]);
        } else if (stats_d != nullptr && stats_d->contains("recent_launches")) {
            have = true;
            launches = py_to_launch_vec((*stats_d)["recent_launches"]);
        }
        if (have) out->recent_launches = std::move(launches);
    }

    {
        bool have = false;
        std::vector<int> ids;
        if (d.contains("prefill_lateness_finalized_ids")) {
            have = true;
            ids = py_to_i32_vec(d["prefill_lateness_finalized_ids"]);
        } else if (stats_d != nullptr && stats_d->contains("prefill_lateness_finalized_ids")) {
            have = true;
            ids = py_to_i32_vec((*stats_d)["prefill_lateness_finalized_ids"]);
        }
        if (have) out->prefill_lateness_finalized_ids = std::move(ids);
    }

    if (d.contains("decode_tokens_counted_by_id")) {
        out->decode_tokens_counted_by_id = py_to_i32_map<int>(d["decode_tokens_counted_by_id"]);
    } else if (stats_d != nullptr && stats_d->contains("decode_tokens_counted_by_id")) {
        out->decode_tokens_counted_by_id = py_to_i32_map<int>((*stats_d)["decode_tokens_counted_by_id"]);
    }

    if (d.contains("per_request_prefill_lateness_by_id")) {
        out->per_request_prefill_lateness_by_id =
            py_to_i32_map<double>(d["per_request_prefill_lateness_by_id"]);
    } else if (stats_d != nullptr && stats_d->contains("per_request_prefill_lateness_by_id")) {
        out->per_request_prefill_lateness_by_id =
            py_to_i32_map<double>((*stats_d)["per_request_prefill_lateness_by_id"]);
    }

    if (d.contains("per_request_decode_lateness_by_id")) {
        out->per_request_decode_lateness_by_id =
            py_to_i32_map<double>(d["per_request_decode_lateness_by_id"]);
    } else if (stats_d != nullptr && stats_d->contains("per_request_decode_lateness_by_id")) {
        out->per_request_decode_lateness_by_id =
            py_to_i32_map<double>((*stats_d)["per_request_decode_lateness_by_id"]);
    }

    if (d.contains("decode_next_deadline_by_id")) {
        out->decode_next_deadline_by_id = py_to_i32_map<double>(d["decode_next_deadline_by_id"]);
    } else if (stats_d != nullptr && stats_d->contains("decode_next_deadline_by_id")) {
        out->decode_next_deadline_by_id = py_to_i32_map<double>((*stats_d)["decode_next_deadline_by_id"]);
    }

    sort_unique(out->active_request_ids);
    sort_unique(out->completed_request_ids);
    sort_unique(out->dropped_request_ids);
    sort_unique(out->stopped_decode_request_ids);
    sort_unique(out->violated_request_ids);
    sort_unique(out->prefill_lateness_finalized_ids);

    if (out->decode_credit_available == 0 && out->decode_credit_balance != 0) {
        out->decode_credit_available = std::max(0, out->decode_credit_balance);
    }
}

RequestState parse_request_state_payload(const py::handle& obj) {
    RequestState r;
    if (!py::isinstance<py::dict>(obj)) return r;
    const py::dict d = py::reinterpret_borrow<py::dict>(obj);

    r.request_id = merged_get_or<int>(d, nullptr, {"request_id", "id", "rid"}, r.request_id);
    r.arrived_at = merged_get_or<double>(d, nullptr, {"arrived_at"}, r.arrived_at);
    r.queued_at = merged_get_or<double>(d, nullptr, {"queued_at"}, r.queued_at);

    r.num_prefill_tokens = merged_get_or<int>(
        d, nullptr, {"num_prefill_tokens", "prefill_tokens"}, r.num_prefill_tokens);
    r.num_processed_prefill_tokens = merged_get_or<int>(
        d, nullptr, {"num_processed_prefill_tokens", "processed_prefill_tokens"}, r.num_processed_prefill_tokens);
    r.num_decode_tokens = merged_get_or<int>(
        d, nullptr, {"num_decode_tokens", "decode_tokens"}, r.num_decode_tokens);
    r.num_processed_decode_tokens = merged_get_or<int>(
        d, nullptr, {"num_processed_decode_tokens", "processed_decode_tokens"}, r.num_processed_decode_tokens);
    if (!d.contains("num_processed_prefill_tokens") &&
        !d.contains("processed_prefill_tokens") &&
        !d.contains("num_processed_decode_tokens") &&
        !d.contains("processed_decode_tokens") &&
        d.contains("num_processed_tokens")) {
        const int processed = std::max(0, py::cast<int>(d["num_processed_tokens"]));
        r.num_processed_prefill_tokens = std::min(processed, std::max(0, r.num_prefill_tokens));
        r.num_processed_decode_tokens = std::max(0, processed - std::max(0, r.num_prefill_tokens));
    }

    r.prefill_slo_time = merged_get_or<double>(d, nullptr, {"prefill_slo_time", "prefill_slo"}, r.prefill_slo_time);
    r.decode_slo_time = merged_get_or<double>(d, nullptr, {"decode_slo_time", "decode_slo"}, r.decode_slo_time);
    r.completion_slo_time = merged_get_or<double>(d, nullptr, {"completion_slo_time"}, r.completion_slo_time);

    r.prefill_deadline = merged_get_or<double>(d, nullptr, {"prefill_deadline"}, r.prefill_deadline);
    r.decode_next_deadline = merged_get_or<double>(
        d, nullptr, {"decode_next_deadline", "decode_deadline"}, r.decode_next_deadline);
    r.prefill_completed_at = merged_get_or<double>(
        d, nullptr, {"prefill_completed_at"}, r.prefill_completed_at);
    r.completed_at = merged_get_or<double>(d, nullptr, {"completed_at"}, r.completed_at);

    r.prefill_lateness = merged_get_or<double>(d, nullptr, {"prefill_lateness"}, r.prefill_lateness);
    r.decode_lateness = merged_get_or<double>(d, nullptr, {"decode_lateness"}, r.decode_lateness);

    r.is_prefill_complete = merged_get_or<bool>(
        d, nullptr, {"is_prefill_complete", "prefill_complete", "_is_prefill_complete"}, r.is_prefill_complete);
    r.completed = merged_get_or<bool>(d, nullptr, {"completed"}, r.completed);
    r.dropped = merged_get_or<bool>(d, nullptr, {"dropped"}, r.dropped);
    r.stopped_decode = merged_get_or<bool>(d, nullptr, {"stopped_decode"}, r.stopped_decode);
    r.violated = merged_get_or<bool>(d, nullptr, {"violated"}, r.violated);

    if (!r.is_prefill_complete && r.remaining_prefill() == 0) r.is_prefill_complete = true;
    return r;
}

void parse_request_collection(const py::dict& d, std::vector<RequestState>* out_requests) {
    auto parse_from = [out_requests](const py::handle& obj) {
        if (obj.is_none()) return;
        if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
            for (auto item : obj) {
                const RequestState r = parse_request_state_payload(item);
                if (r.request_id >= 0) out_requests->push_back(r);
            }
            return;
        }
        if (!py::isinstance<py::dict>(obj)) return;
        const py::dict mp = py::reinterpret_borrow<py::dict>(obj);
        for (auto item : mp) {
            RequestState r = parse_request_state_payload(item.second);
            if (r.request_id < 0) {
                bool ok = false;
                const int rid = py_key_to_int(item.first, &ok);
                if (!ok) continue;
                r.request_id = rid;
                if (py::isinstance<py::dict>(item.second)) {
                    const py::dict vv = py::reinterpret_borrow<py::dict>(item.second);
                    r.num_prefill_tokens = merged_get_or<int>(
                        vv, nullptr, {"num_prefill_tokens", "prefill_tokens"}, r.num_prefill_tokens);
                    r.num_processed_prefill_tokens = merged_get_or<int>(
                        vv, nullptr, {"num_processed_prefill_tokens", "processed_prefill_tokens"}, r.num_processed_prefill_tokens);
                    r.num_decode_tokens = merged_get_or<int>(
                        vv, nullptr, {"num_decode_tokens", "decode_tokens"}, r.num_decode_tokens);
                    r.num_processed_decode_tokens = merged_get_or<int>(
                        vv, nullptr, {"num_processed_decode_tokens", "processed_decode_tokens"}, r.num_processed_decode_tokens);
                    r.completed = merged_get_or<bool>(vv, nullptr, {"completed"}, r.completed);
                }
            }
            if (r.request_id >= 0) out_requests->push_back(std::move(r));
        }
    };

    if (d.contains("requests")) parse_from(d["requests"]);
    if (d.contains("request_states")) parse_from(d["request_states"]);
    if (d.contains("request_list")) parse_from(d["request_list"]);
    if (d.contains("request_lookup")) parse_from(d["request_lookup"]);
}

SimState parse_root_state_payload(const py::dict& d) {
    SimState s;
    py::dict stats_d;
    const py::dict* stats_ptr = nullptr;
    if (d.contains("stats") && py::isinstance<py::dict>(d["stats"])) {
        stats_d = py::cast<py::dict>(d["stats"]);
        stats_ptr = &stats_d;
    }

    s.sim_time = merged_get_or<double>(d, stats_ptr, {"sim_time", "time"}, s.sim_time);
    s.decision_state_time = merged_get_or<double>(
        d, stats_ptr, {"decision_state_time"}, s.sim_time);
    bool have_next_request_id = false;
    int next_request_id = s.next_request_id;
    if (merged_try_get<int>(d, stats_ptr, {"next_request_id"}, &next_request_id)) {
        have_next_request_id = true;
        s.next_request_id = next_request_id;
    }

    parse_stats_fields_into(d, stats_ptr, &s.stats);
    apply_python_gv3_meta(&s.stats, s.sim_time);
    parse_request_collection(d, &s.requests);

    std::unordered_map<int, std::size_t> rid_to_idx;
    rid_to_idx.reserve(s.requests.size());
    for (std::size_t i = 0; i < s.requests.size(); ++i) {
        rid_to_idx[s.requests[i].request_id] = i;
    }

    for (const auto& kv : s.stats.per_request_prefill_lateness_by_id) {
        const auto it = rid_to_idx.find(kv.first);
        if (it != rid_to_idx.end()) s.requests[it->second].prefill_lateness = kv.second;
    }
    for (const auto& kv : s.stats.per_request_decode_lateness_by_id) {
        const auto it = rid_to_idx.find(kv.first);
        if (it != rid_to_idx.end()) s.requests[it->second].decode_lateness = kv.second;
    }
    for (const auto& kv : s.stats.decode_next_deadline_by_id) {
        const auto it = rid_to_idx.find(kv.first);
        if (it != rid_to_idx.end()) s.requests[it->second].decode_next_deadline = kv.second;
    }
    {
        std::unordered_set<int> violated_set(
            s.stats.violated_request_ids.begin(),
            s.stats.violated_request_ids.end());
        std::unordered_set<int> dropped_set(
            s.stats.dropped_request_ids.begin(),
            s.stats.dropped_request_ids.end());
        std::unordered_set<int> stopped_set(
            s.stats.stopped_decode_request_ids.begin(),
            s.stats.stopped_decode_request_ids.end());
        for (auto& r : s.requests) {
            if (violated_set.find(r.request_id) != violated_set.end()) r.violated = true;
            if (dropped_set.find(r.request_id) != dropped_set.end()) r.dropped = true;
            if (stopped_set.find(r.request_id) != stopped_set.end()) r.stopped_decode = true;
        }
    }

    if (s.stats.active_request_ids.empty() && s.stats.completed_request_ids.empty() && !s.requests.empty()) {
        for (const auto& r : s.requests) {
            if (r.completed) {
                s.stats.completed_request_ids.push_back(r.request_id);
            } else if (r.prefill_active() || r.decode_active()) {
                s.stats.active_request_ids.push_back(r.request_id);
            }
        }
        sort_unique(s.stats.active_request_ids);
        sort_unique(s.stats.completed_request_ids);
    }
    if (!have_next_request_id) {
        int last_request_id = -1;
        if (parse_python_request_counter(d, &last_request_id)) {
            s.next_request_id = std::max(0, last_request_id + 1);
        } else {
            for (const auto& r : s.requests) {
                last_request_id = std::max(last_request_id, r.request_id);
            }
            s.next_request_id = std::max(0, last_request_id + 1);
        }
    }
    return s;
}

py::dict request_state_to_py_dict(const RequestState& r) {
    py::dict d;
    d["request_id"] = r.request_id;
    d["arrived_at"] = r.arrived_at;
    d["queued_at"] = r.queued_at;
    d["num_prefill_tokens"] = r.num_prefill_tokens;
    d["num_processed_prefill_tokens"] = r.num_processed_prefill_tokens;
    d["num_decode_tokens"] = r.num_decode_tokens;
    d["num_processed_decode_tokens"] = r.num_processed_decode_tokens;
    d["prefill_slo_time"] = r.prefill_slo_time;
    d["decode_slo_time"] = r.decode_slo_time;
    d["completion_slo_time"] = r.completion_slo_time;
    d["prefill_deadline"] = r.prefill_deadline;
    d["decode_next_deadline"] = r.decode_next_deadline;
    d["prefill_completed_at"] = r.prefill_completed_at;
    d["completed_at"] = r.completed_at;
    d["prefill_lateness"] = r.prefill_lateness;
    d["decode_lateness"] = r.decode_lateness;
    d["is_prefill_complete"] = r.is_prefill_complete;
    d["completed"] = r.completed;
    d["dropped"] = r.dropped;
    d["stopped_decode"] = r.stopped_decode;
    d["violated"] = r.violated;
    d["remaining_prefill"] = r.remaining_prefill();
    d["remaining_decode"] = r.remaining_decode();
    return d;
}

py::dict i32_i32_map_to_py(const std::unordered_map<int, int>& mp) {
    py::dict d;
    for (const auto& kv : mp) d[py::int_(kv.first)] = py::int_(kv.second);
    return d;
}

py::dict i32_f64_map_to_py(const std::unordered_map<int, double>& mp) {
    py::dict d;
    for (const auto& kv : mp) d[py::int_(kv.first)] = kv.second;
    return d;
}

py::dict game_stats_to_py_dict(const GameStats& s) {
    py::dict d;
    d["requests_generated"] = s.requests_generated;
    d["requests_completed"] = s.requests_completed;
    d["slo_violations"] = s.slo_violations;
    d["slo_lateness_sum"] = s.slo_lateness_sum;
    d["recent_arrivals"] = s.recent_arrivals;
    py::list recent_launches;
    for (const auto& e : s.recent_launches) {
        recent_launches.append(py::make_tuple(e.timestamp, e.count, e.prefill_tokens));
    }
    d["recent_launches"] = std::move(recent_launches);
    d["active_request_ids"] = s.active_request_ids;
    d["completed_request_ids"] = s.completed_request_ids;
    d["dropped_request_ids"] = s.dropped_request_ids;
    d["stopped_decode_request_ids"] = s.stopped_decode_request_ids;
    d["violated_request_ids"] = s.violated_request_ids;
    d["prefill_lateness_finalized_ids"] = s.prefill_lateness_finalized_ids;
    d["per_request_prefill_lateness_by_id"] = i32_f64_map_to_py(s.per_request_prefill_lateness_by_id);
    d["per_request_decode_lateness_by_id"] = i32_f64_map_to_py(s.per_request_decode_lateness_by_id);
    d["decode_next_deadline_by_id"] = i32_f64_map_to_py(s.decode_next_deadline_by_id);
    d["decode_tokens_counted_by_id"] = i32_i32_map_to_py(s.decode_tokens_counted_by_id);
    d["decode_credit_balance"] = s.decode_credit_balance;
    d["decode_credit_available"] = s.decode_credit_available;
    d["pending_adv_tick"] = s.pending_adv_tick;
    d["last_adv_tick"] = s.last_adv_tick;
    d["next_adv_tick"] = s.next_adv_tick;
    d["missed_adv_source"] = s.missed_adv_source;
    return d;
}

py::dict sim_state_to_py_dict(const SimState& s) {
    py::dict d;
    d["sim_time"] = s.sim_time;
    d["decision_state_time"] = s.decision_state_time;
    d["next_request_id"] = s.next_request_id;
    py::list reqs;
    for (const auto& r : s.requests) reqs.append(request_state_to_py_dict(r));
    d["requests"] = std::move(reqs);
    d["stats"] = game_stats_to_py_dict(s.stats);
    return d;
}

py::dict root_input_fields_to_py(
    const std::vector<float>& global_features,
    const std::vector<uint8_t>& action_mask,
    const std::vector<float>& prefill_req_features,
    const std::vector<float>& decode_req_features,
    const std::vector<uint8_t>& prefill_req_mask,
    const std::vector<uint8_t>& decode_req_mask,
    int prefill_req_n,
    int prefill_req_d,
    int decode_req_n,
    int decode_req_d,
    const std::vector<float>& req_features,
    const std::vector<uint8_t>& req_mask,
    int req_n,
    int req_d) {
    py::dict d;
    d["global_features"] = global_features;
    d["action_mask"] = action_mask;
    d["prefill_req_features"] = prefill_req_features;
    d["decode_req_features"] = decode_req_features;
    d["prefill_req_mask"] = prefill_req_mask;
    d["decode_req_mask"] = decode_req_mask;
    d["prefill_req_n"] = prefill_req_n;
    d["prefill_req_d"] = prefill_req_d;
    d["decode_req_n"] = decode_req_n;
    d["decode_req_d"] = decode_req_d;
    d["req_features"] = req_features;
    d["req_mask"] = req_mask;
    d["req_n"] = req_n;
    d["req_d"] = req_d;
    return d;
}

py::dict search_root_inputs_to_py(const SearchOutput& out) {
    return root_input_fields_to_py(
        out.root_global_features,
        out.root_action_mask,
        out.root_prefill_req_features,
        out.root_decode_req_features,
        out.root_prefill_req_mask,
        out.root_decode_req_mask,
        out.root_prefill_req_n,
        out.root_prefill_req_d,
        out.root_decode_req_n,
        out.root_decode_req_d,
        out.root_req_features,
        out.root_req_mask,
        out.root_req_n,
        out.root_req_d);
}

py::dict native_sample_to_py(const NativeRootSampleGV3& s) {
    py::dict d;
    d["feature_version"] = s.feature_version;
    d["game_id"] = s.game_id;
    d["root_id"] = s.root_id;
    d["root_node_id"] = s.root_node_id;
    d["root_depth"] = s.root_depth;
    d["player"] = s.player;
    d["inputs"] = root_input_fields_to_py(
        s.global_features,
        s.action_mask,
        s.prefill_req_features,
        s.decode_req_features,
        s.prefill_req_mask,
        s.decode_req_mask,
        s.prefill_req_n,
        s.prefill_req_d,
        s.decode_req_n,
        s.decode_req_d,
        s.req_features,
        s.req_mask,
        s.req_n,
        s.req_d);
    py::dict targets;
    targets["policy"] = s.policy;
    targets["value"] = s.value;
    d["targets"] = std::move(targets);
    py::dict meta;
    meta["best_action_index"] = s.best_action_index;
    meta["used_bootstrap"] = s.used_bootstrap;
    meta["model_version"] = s.model_version;
    meta["history_hops"] = s.history_hops;
    meta["is_eval"] = s.is_eval;
    d["meta"] = std::move(meta);
    return d;
}

int pick_default_action_space(const py::dict& cfg, const std::string& root_player) {
    if (root_player == "controller") {
        if (cfg.contains("controller_action_space_size")) {
            return std::max(1, py::cast<int>(cfg["controller_action_space_size"]));
        }
    } else {
        if (cfg.contains("adversary_action_space_size")) {
            return std::max(1, py::cast<int>(cfg["adversary_action_space_size"]));
        }
    }
    return 8;
}

void apply_feature_cfg_payload(const py::dict& cfg, NativeFeatureBuildConfigGV2* out) {
    if (out == nullptr) return;

    auto get_int = [&](std::initializer_list<const char*> keys, int dflt) {
        for (const char* k : keys) {
            if (!cfg.contains(k)) continue;
            try {
                return py::cast<int>(cfg[k]);
            } catch (const std::exception&) {
            }
        }
        return dflt;
    };
    auto get_double = [&](std::initializer_list<const char*> keys, double dflt) {
        for (const char* k : keys) {
            if (!cfg.contains(k)) continue;
            try {
                return py::cast<double>(cfg[k]);
            } catch (const std::exception&) {
            }
        }
        return dflt;
    };

    out->n_prefill_req = get_int({"n_prefill_req"}, out->n_prefill_req);
    out->d_prefill_req = get_int({"d_prefill_req"}, out->d_prefill_req);
    out->n_decode_req = get_int({"n_decode_req"}, out->n_decode_req);
    out->d_decode_req = get_int({"d_decode_req"}, out->d_decode_req);
    out->d_global = get_int({"d_global"}, out->d_global);

    out->prefill_total_den = get_double({"prefill_total_den"}, out->prefill_total_den);
    out->prefill_remaining_den = get_double({"prefill_remaining_den"}, out->prefill_remaining_den);
    out->decode_total_den = get_double({"decode_total_den"}, out->decode_total_den);
    out->decode_remaining_den = get_double({"decode_remaining_den"}, out->decode_remaining_den);
    out->decode_processed_den = get_double({"decode_processed_den"}, out->decode_processed_den);
    out->age_den_sec = get_double({"age_den_sec"}, out->age_den_sec);
    out->lateness_den_sec = get_double({"lateness_den_sec"}, out->lateness_den_sec);
    out->slack_den_sec = get_double({"slack_den_sec"}, out->slack_den_sec);
    out->prefill_slo_den_sec = get_double({"prefill_slo_den_sec"}, out->prefill_slo_den_sec);
    out->decode_slo_den_sec = get_double({"decode_slo_den_sec"}, out->decode_slo_den_sec);
    out->objective_cost_den = get_double({"objective_cost_den"}, out->objective_cost_den);
    out->total_lateness_den = get_double({"total_lateness_den"}, out->total_lateness_den);

    out->system_load_den = get_double({"system_load_den"}, out->system_load_den);
    out->active_prefill_count_den = get_double({"active_prefill_count_den"}, out->active_prefill_count_den);
    out->active_decode_count_den = get_double({"active_decode_count_den"}, out->active_decode_count_den);
    out->active_total_count_den = get_double({"active_total_count_den"}, out->active_total_count_den);
    out->total_remaining_prefill_den =
        get_double({"total_remaining_prefill_den"}, out->total_remaining_prefill_den);
    out->total_remaining_decode_den =
        get_double({"total_remaining_decode_den"}, out->total_remaining_decode_den);
    out->total_decode_generated_active_den =
        get_double({"total_decode_generated_active_den"}, out->total_decode_generated_active_den);
    out->violated_count_den = get_double({"violated_count_den"}, out->violated_count_den);
    out->prefill_near_drop_den = get_double({"prefill_near_drop_den"}, out->prefill_near_drop_den);
    out->decode_near_drop_den = get_double({"decode_near_drop_den"}, out->decode_near_drop_den);
    out->recent_launch_count_den = get_double({"recent_launch_count_den"}, out->recent_launch_count_den);
    out->recent_launch_prefill_den = get_double({"recent_launch_prefill_den"}, out->recent_launch_prefill_den);
    out->decode_credit_den = get_double({"decode_credit_den"}, out->decode_credit_den);

    out->near_drop_lateness_low_sec =
        get_double({"near_drop_lateness_low_sec"}, out->near_drop_lateness_low_sec);
    out->near_drop_lateness_high_sec =
        get_double({"near_drop_lateness_high_sec"}, out->near_drop_lateness_high_sec);
    out->launch_ewma_alpha = get_double({"launch_ewma_alpha"}, out->launch_ewma_alpha);
    out->launch_ewma_window_sec = get_double({"launch_ewma_window_sec"}, out->launch_ewma_window_sec);
    out->decode_sample_seed_offset = get_int({"decode_sample_seed_offset"}, out->decode_sample_seed_offset);
}

py::dict search_mcts_dnn_gv2_torchscript(
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version,
    py::dict root_state_payload,
    py::dict cfg_payload,
    int iterations,
    const std::string& root_player,
    int root_node_id,
    int root_depth,
    int game_id,
    int root_id,
    int seed,
    bool log_events,
    bool profile,
    const std::string& iter_log_path,
    const std::string& root_log_path) {
    SearchInput in;
    in.contract_version = kGV2NativeContractVersion;
    if (root_state_payload.contains("contract_version")) {
        try {
            in.contract_version = py::cast<std::string>(root_state_payload["contract_version"]);
        } catch (const std::exception&) {
        }
    }
    in.root_state = parse_root_state_payload(root_state_payload);
    in.root_player = root_player;
    in.iterations = iterations;
    in.root_node_id = root_node_id;
    in.root_depth = root_depth;
    in.game_id = game_id;
    in.root_id = root_id;
    in.seed = seed;
    in.log_events = log_events;
    in.profile = profile;
    apply_feature_cfg_payload(cfg_payload, &in.feature_cfg);

    auto get_int = [&](std::initializer_list<const char*> keys, int dflt) {
        for (const char* k : keys) {
            if (!cfg_payload.contains(k)) continue;
            try {
                return py::cast<int>(cfg_payload[k]);
            } catch (const std::exception&) {
            }
        }
        return dflt;
    };
    auto get_double = [&](std::initializer_list<const char*> keys, double dflt) {
        for (const char* k : keys) {
            if (!cfg_payload.contains(k)) continue;
            try {
                return py::cast<double>(cfg_payload[k]);
            } catch (const std::exception&) {
            }
        }
        return dflt;
    };
    auto get_bool = [&](std::initializer_list<const char*> keys, bool dflt) {
        for (const char* k : keys) {
            if (!cfg_payload.contains(k)) continue;
            try {
                return py::cast<bool>(cfg_payload[k]);
            } catch (const std::exception&) {
            }
        }
        return dflt;
    };
    auto get_str = [&](std::initializer_list<const char*> keys, const std::string& dflt) {
        for (const char* k : keys) {
            if (!cfg_payload.contains(k)) continue;
            try {
                return py::cast<std::string>(cfg_payload[k]);
            } catch (const std::exception&) {
            }
        }
        return dflt;
    };
    if (root_state_payload.contains("root_phase")) {
        try {
            in.root_phase = py::cast<std::string>(root_state_payload["root_phase"]);
        } catch (const std::exception&) {
        }
    } else if (cfg_payload.contains("root_phase")) {
        try {
            in.root_phase = py::cast<std::string>(cfg_payload["root_phase"]);
        } catch (const std::exception&) {
        }
    }
    if (root_state_payload.contains("cycle_label")) {
        try {
            in.cycle_label = py::cast<std::string>(root_state_payload["cycle_label"]);
        } catch (const std::exception&) {
        }
    } else if (cfg_payload.contains("cycle_label")) {
        try {
            in.cycle_label = py::cast<std::string>(cfg_payload["cycle_label"]);
        } catch (const std::exception&) {
        }
    }

    if (root_state_payload.contains("action_mask")) {
        in.action_mask = py_to_mask(root_state_payload["action_mask"]);
    }
    if (in.action_mask.empty()) {
        in.action_mask.assign(static_cast<std::size_t>(pick_default_action_space(cfg_payload, root_player)), 1u);
    }

    if (root_state_payload.contains("global_features")) {
        in.global_features = py_to_f32_vec(root_state_payload["global_features"]);
    }
    in.root_infer_inputs.global_features = in.global_features;
    in.root_infer_inputs.action_mask = in.action_mask;

    if (root_state_payload.contains("prefill_req_features")) {
        in.root_infer_inputs.prefill_req_features = py_to_f32_flat(root_state_payload["prefill_req_features"]);
    }
    if (root_state_payload.contains("decode_req_features")) {
        in.root_infer_inputs.decode_req_features = py_to_f32_flat(root_state_payload["decode_req_features"]);
    }
    if (root_state_payload.contains("prefill_req_mask")) {
        in.root_infer_inputs.prefill_req_mask = py_to_mask_flat(root_state_payload["prefill_req_mask"]);
    }
    if (root_state_payload.contains("decode_req_mask")) {
        in.root_infer_inputs.decode_req_mask = py_to_mask_flat(root_state_payload["decode_req_mask"]);
    }
    if (root_state_payload.contains("prefill_req_n")) {
        in.root_infer_inputs.prefill_req_n = py::cast<int>(root_state_payload["prefill_req_n"]);
    }
    if (root_state_payload.contains("prefill_req_d")) {
        in.root_infer_inputs.prefill_req_d = py::cast<int>(root_state_payload["prefill_req_d"]);
    }
    if (root_state_payload.contains("decode_req_n")) {
        in.root_infer_inputs.decode_req_n = py::cast<int>(root_state_payload["decode_req_n"]);
    }
    if (root_state_payload.contains("decode_req_d")) {
        in.root_infer_inputs.decode_req_d = py::cast<int>(root_state_payload["decode_req_d"]);
    }
    if (root_state_payload.contains("req_features")) {
        in.root_infer_inputs.req_features = py_to_f32_flat(root_state_payload["req_features"]);
    }
    if (root_state_payload.contains("req_mask")) {
        in.root_infer_inputs.req_mask = py_to_mask_flat(root_state_payload["req_mask"]);
    }
    if (root_state_payload.contains("req_n")) {
        in.root_infer_inputs.req_n = py::cast<int>(root_state_payload["req_n"]);
    }
    if (root_state_payload.contains("req_d")) {
        in.root_infer_inputs.req_d = py::cast<int>(root_state_payload["req_d"]);
    }

    if (root_state_payload.contains("decision_state_time")) {
        in.decision_state_time = py::cast<double>(root_state_payload["decision_state_time"]);
    } else {
        in.decision_state_time = in.root_state.decision_state_time;
        if (in.decision_state_time <= 0.0) in.decision_state_time = in.root_state.sim_time;
    }
    in.root_state.decision_state_time = in.decision_state_time;

    // Search knobs
    in.max_forced_hops = get_int({"max_forced_hops", "mcts_max_forced_hops"}, in.max_forced_hops);
    in.pb_c_base = get_double({"pb_c_base", "mcts_pb_c_base"}, in.pb_c_base);
    in.pb_c_init = get_double({"pb_c_init", "mcts_pb_c_init"}, in.pb_c_init);
    in.discount_factor = get_double({"discount_factor", "mcts_discount_factor"}, in.discount_factor);
    in.root_dirichlet_noise_enabled = get_bool(
        {"root_dirichlet_noise_enabled"},
        in.root_dirichlet_noise_enabled);
    in.root_dirichlet_alpha = get_double(
        {"root_dirichlet_alpha"},
        in.root_dirichlet_alpha);
    in.root_dirichlet_epsilon = get_double(
        {"root_dirichlet_epsilon"},
        in.root_dirichlet_epsilon);

    // in.prefill_step_time = get_double({"prefill_step_time"}, in.prefill_step_time);
    in.prefill_step_time = get_double({"discount_time_denominator_sec", "prefill_step_time"},in.prefill_step_time);
    in.reward_knee = get_double({"reward_knee"}, in.reward_knee);
    in.reward_max_penalty = get_double({"reward_max_penalty"}, in.reward_max_penalty);
    in.reward_tail_alpha = get_double({"reward_tail_alpha"}, in.reward_tail_alpha);
    in.reuse_root_infer_inputs = get_bool({"reuse_root_infer_inputs"}, in.reuse_root_infer_inputs);

    // Environment/simulator knobs (direct keys only; adapter can flatten before call)
    in.env_cfg.adversary_tick_sec = get_double({"adversary_tick_sec"}, in.env_cfg.adversary_tick_sec);
    in.env_cfg.launch_window_sec = get_double({"launch_window_sec"}, in.env_cfg.launch_window_sec);
    in.env_cfg.max_requests_per_launch_window = get_int(
        {"max_requests_per_launch_window"},
        in.env_cfg.max_requests_per_launch_window);
    in.env_cfg.prefill_window_cap_tokens = get_int(
        {"prefill_window_cap_tokens"},
        in.env_cfg.prefill_window_cap_tokens);
    in.env_cfg.max_prefill_tokens_per_request = get_int(
        {"max_prefill_tokens_per_request"},
        in.env_cfg.max_prefill_tokens_per_request);
    in.env_cfg.max_decode_tokens_per_request = get_int(
        {"max_decode_tokens_per_request"},
        in.env_cfg.max_decode_tokens_per_request);
    in.env_cfg.min_decode_tokens_per_request = get_int(
        {"min_decode_tokens_per_request"},
        in.env_cfg.min_decode_tokens_per_request);
    in.env_cfg.decode_slo_time_default = get_double(
        {"decode_slo_time_default"},
        in.env_cfg.decode_slo_time_default);
    in.env_cfg.auto_drop_lateness_sec = get_double(
        {"auto_drop_lateness_sec"},
        in.env_cfg.auto_drop_lateness_sec);
    in.env_cfg.drop_cost = get_double({"drop_cost"}, in.env_cfg.drop_cost);
    in.env_cfg.controller_noop_prefill_only_jump_to_next_adv_tick = get_bool(
        {"controller_noop_prefill_only_jump_to_next_adv_tick"},
        in.env_cfg.controller_noop_prefill_only_jump_to_next_adv_tick);
    in.env_cfg.enforce_nonnegative_decode_credits = get_bool(
        {"enforce_nonnegative_decode_credits"},
        in.env_cfg.enforce_nonnegative_decode_credits);
    in.env_cfg.decode_credit_mint_per_prefill_complete = get_int(
        {"decode_credit_mint_per_prefill_complete"},
        in.env_cfg.decode_credit_mint_per_prefill_complete);

    in.sim_cfg.adversary_tick_sec = in.env_cfg.adversary_tick_sec;
    in.predictor_csv_path = get_str(
        {"native_predictor_csv_path", "prefill_predictor_csv", "predictor_csv_path"},
        in.predictor_csv_path);

    if (cfg_payload.contains("prefill_profile_tokens")) {
        in.sim_cfg.prefill_profile_tokens = py_to_i32_vec(cfg_payload["prefill_profile_tokens"]);
    }
    if (cfg_payload.contains("prefill_profile_times")) {
        in.sim_cfg.prefill_profile_times = py_to_f64_vec(cfg_payload["prefill_profile_times"]);
    }
    in.sim_cfg.fallback_total_time_sec = get_double(
        {"fallback_total_time_sec"},
        in.sim_cfg.fallback_total_time_sec);
    in.sim_cfg.fallback_model_time_sec = get_double(
        {"fallback_model_time_sec"},
        in.sim_cfg.fallback_model_time_sec);

    if (log_events && (!iter_log_path.empty() || !root_log_path.empty())) {
        const std::string config_json_path = default_native_config_json_path(iter_log_path, root_log_path);
        if (!config_json_path.empty()) {
            NativeConfigJsonLogger config_logger(config_json_path);
            config_logger.write_once(build_native_config_json(in, model_version, iter_log_path, root_log_path));
        }
    }

    SearchOutput out = run_search_torchscript(in, infer_runtime, model_version);

    if (log_events) {
        const int native_flush_every = get_int({"native_log_flush_every"}, 1);
        if (!iter_log_path.empty()) {
            NativeIterCsvLogger iter_logger(iter_log_path, native_flush_every);
            for (const auto& ev : out.iter_events) {
                NativeIterLogRow row;
                row.game_id = game_id;
                row.root_id = root_id;
                row.root_depth = root_depth;
                row.root_node_id = root_node_id;
                row.root_player = root_player;
                row.sim_iteration = ev.sim_iteration;
                row.phase = ev.phase.empty() ? "native_iter" : ev.phase;
                row.node_depth = ev.leaf_depth;
                if (ev.parent_node_id >= 0) {
                    row.has_parent_node_id = true;
                    row.parent_node_id = ev.parent_node_id;
                }
                row.node_id = ev.leaf_node_id;
                row.player_acted_to_create_this_node = ev.player_acted_to_create_this_node;
                row.player_to_act_in_this_node = ev.player_to_act;
                row.action_index = ev.action_index;
                row.action_repr = ev.action_repr;
                row.prior = ev.prior;
                row.model_prior_json = ev.root_nn_priors_json;
                row.normalized_prior_json = ev.root_nn_priors_after_threshold_json;
                row.reward = ev.reward;
                row.nn_called = ev.nn_called;
                row.num_valid_actions = ev.num_valid_actions;
                row.unique_actions = ev.unique_actions;
                row.has_nn_value_controller = ev.has_nn_value_controller;
                row.nn_value_controller = ev.nn_value_controller;
                row.objective_cost = ev.leaf_state_cost;
                row.sim_time = ev.sim_time_after;
                row.decision_state_time = ev.decision_state_time;
                row.start_time = ev.sim_time_before;
                row.end_time = ev.sim_time_after;
                row.stage_total_time = ev.sim_time_after - ev.sim_time_before;
                row.requests_in_system = ev.requests_in_system;
                row.requests_generated = ev.requests_generated;
                row.requests_completed = ev.requests_completed;
                row.slo_violations = ev.slo_violations;
                row.total_lateness = ev.total_lateness;
                row.avg_lateness = ev.avg_lateness;
                row.state_active_ids = ev.active_request_ids_json;
                row.state_waiting_ids = ev.waiting_request_ids_json;
                row.state_completed_request_ids = ev.completed_request_ids_json;
                row.state_dropped_request_ids = ev.dropped_request_ids_json;
                row.state_stopped_decode_request_ids = ev.stopped_decode_request_ids_json;
                row.state_pending_adv_tick = ev.state_pending_adv_tick;
                row.has_state_last_adv_tick = ev.has_state_last_adv_tick;
                row.state_last_adv_tick = ev.state_last_adv_tick;
                row.state_decode_credit_balance = ev.decode_credit_balance;
                row.state_decode_tokens_counted_by_id = ev.decode_tokens_counted_by_id_json;
                row.state_violated_request_ids = ev.violated_request_ids_json;
                row.state_per_request_prefill_lateness_by_id = ev.per_request_prefill_lateness_by_id_json;
                row.state_per_request_decode_lateness_by_id = ev.per_request_decode_lateness_by_id_json;
                row.adversary_requests = ev.adversary_requests_json;
                row.adversary_prefill_slos = ev.adversary_prefill_slos_json;
                row.adversary_prefill_deadlines_by_id = ev.adversary_prefill_deadlines_by_id_json;
                row.adversary_decode_slos = ev.adversary_decode_slos_json;
                row.has_controller_token_budget = ev.has_controller_token_budget;
                row.controller_token_budget = ev.controller_token_budget;
                row.controller_selected_ids = ev.controller_selected_ids_json;
                row.controller_allocations = ev.controller_allocations_json;
                row.controller_prefill_allocations = ev.controller_prefill_allocations_json;
                row.controller_decode_allocations = ev.controller_decode_allocations_json;
                row.controller_prefill_total = ev.controller_prefill_total;
                row.controller_decode_total = ev.controller_decode_total;
                row.controller_heuristic = ev.controller_heuristic;
                row.controller_strategy = ev.controller_strategy;
                iter_logger.write(row);
            }
        }

        if (!root_log_path.empty()) {
            NativeRootCsvLogger root_logger(root_log_path, native_flush_every);
            int best_idx = argmax_masked(out.mcts_root_prior, out.root_nn_valid_mask);
            NativeRootLogRow row;
            row.game_id = game_id;
            row.root_id = root_id;
            row.root_depth = root_depth;
            row.root_node_id = root_node_id;
            row.root_player = root_player;
            row.num_simulations = iterations;
            row.model_root_value_controller = out.root_nn_value_controller;
            row.model_root_prior_json = json_f64_vec(out.root_nn_priors);
            row.normalized_root_prior_json = json_f64_vec(out.root_nn_priors_after_threshold);
            row.valid_action_mask_json = json_u8_vec(out.root_nn_valid_mask);
            row.mcts_root_value_controller = (out.root_visits > 0)
                ? (out.root_value_sum / static_cast<double>(out.root_visits))
                : 0.0;
            row.mcts_root_prior_json = json_f64_vec(out.mcts_root_prior);
            row.best_action_index = best_idx;
            if (best_idx >= 0 && best_idx < static_cast<int>(out.mcts_root_prior.size())) {
                row.best_action_mcts_prob = out.mcts_root_prior[static_cast<std::size_t>(best_idx)];
            }
            if (best_idx >= 0 && best_idx < static_cast<int>(out.root_nn_priors.size())) {
                row.best_action_model_prob = out.root_nn_priors[static_cast<std::size_t>(best_idx)];
            }
            if (best_idx >= 0) {
                for (const auto& c : out.children) {
                    if (c.index == best_idx) {
                        row.best_action_json = c.parent_action_json;
                        row.best_action_repr = c.parent_action_json;
                        break;
                    }
                }
            }
            row.sim_time = out.root_sim_time;
            row.decision_state_time = in.decision_state_time;
            row.state_pending_adv_tick = in.root_state.stats.pending_adv_tick;
            row.has_state_last_adv_tick = true;
            row.state_last_adv_tick = in.root_state.stats.last_adv_tick;
            row.state_active_ids = json_i32_vec(in.root_state.stats.active_request_ids);
            row.state_completed_request_ids = json_i32_vec(in.root_state.stats.completed_request_ids);
            row.state_decode_credit_balance = in.root_state.stats.decode_credit_balance;
            {
                std::unordered_set<int> active_set(
                    in.root_state.stats.active_request_ids.begin(),
                    in.root_state.stats.active_request_ids.end());
                std::unordered_map<int, int> decode_tokens_counted_filtered;
                for (const auto& kv : in.root_state.stats.decode_tokens_counted_by_id) {
                    const int rid = kv.first;
                    if (rid < 0) continue;  // strip meta keys like decode-credit ledger sentinel ids
                    if (active_set.find(rid) == active_set.end()) continue;
                    decode_tokens_counted_filtered[rid] = kv.second;
                }
                row.state_decode_tokens_counted_by_id = json_i32_i32_map(decode_tokens_counted_filtered);
            }
            row.slo_violations = in.root_state.stats.slo_violations;
            row.total_lateness = in.root_state.stats.slo_lateness_sum;
            row.total_cost = static_cast<double>(row.slo_violations) + row.total_lateness;
            row.phase = in.root_phase;
            row.cycle_label = in.cycle_label;
            root_logger.write(row);
        }
    }

    py::list py_children;
    for (const auto& c : out.children) {
        py::dict d;
        d["index"] = c.index;
        d["node_id"] = c.node_id;
        d["player"] = c.player;
        d["depth"] = c.depth;
        d["parent_action"] = py::none();  // adapter/Python side can fill real action object later
        d["prior"] = c.prior;
        d["reward"] = c.reward;
        d["visits"] = c.visits;
        d["value_sum"] = c.value_sum;
        d["sim_time"] = c.sim_time;
        d["state_cost"] = c.state_cost;
        d["num_valid_actions"] = c.num_valid_actions;
        d["parent_action_json"] = c.parent_action_json;
        py_children.append(std::move(d));
    }

    py::list py_iter_events;
    for (const auto& ev : out.iter_events) {
        py::dict d;
        d["sim_iteration"] = ev.sim_iteration;
        d["selected_action_index"] = ev.selected_action_index;
        d["selected_child_node_id"] = ev.selected_child_node_id;
        d["sim_time_before"] = ev.sim_time_before;
        d["sim_time_after"] = ev.sim_time_after;
        d["decision_state_time"] = ev.decision_state_time;
        d["leaf_state_cost"] = ev.leaf_state_cost;
        d["decode_credit_balance"] = ev.decode_credit_balance;
        d["num_valid_actions"] = ev.num_valid_actions;
        d["root_visits_after"] = ev.root_visits_after;
        d["root_value_sum_after"] = ev.root_value_sum_after;
        d["root_mean_value_after"] = ev.root_mean_value_after;
        d["selected_child_visits_after"] = ev.selected_child_visits_after;
        d["selected_child_value_sum_after"] = ev.selected_child_value_sum_after;
        d["selected_child_mean_value_after"] = ev.selected_child_mean_value_after;
        d["selected_child_prior"] = ev.selected_child_prior;
        d["leaf_node_id"] = ev.leaf_node_id;
        d["leaf_depth"] = ev.leaf_depth;
        d["player_to_act"] = ev.player_to_act;
        d["phase"] = ev.phase;
        d["active_request_ids_json"] = ev.active_request_ids_json;
        d["completed_request_ids_json"] = ev.completed_request_ids_json;
        d["root_valid_mask_json"] = ev.root_valid_mask_json;
        d["root_nn_priors_json"] = ev.root_nn_priors_json;
        d["root_nn_priors_after_threshold_json"] = ev.root_nn_priors_after_threshold_json;
        d["root_mcts_prior_json"] = ev.root_mcts_prior_json;
        py_iter_events.append(std::move(d));
    }

    py::dict py_alias_to_canonical;
    for (const auto& kv : out.action_alias_to_canonical) {
        py_alias_to_canonical[py::int_(kv.first)] = py::int_(kv.second);
    }

    py::dict py_canonical_to_aliases;
    for (const auto& kv : out.canonical_to_action_aliases) {
        py_canonical_to_aliases[py::int_(kv.first)] = py::cast(kv.second);
    }

    py::dict py_perf;
    for (const auto& kv : out.perf) py_perf[kv.first.c_str()] = kv.second;

    py::dict result;
    result["contract_version"] = out.contract_version;
    result["decision_state_time"] = out.decision_state_time;
    result["root_state_echo"] = sim_state_to_py_dict(out.root_state_echo);
    result["root_phase"] = in.root_phase;
    result["cycle_label"] = in.cycle_label;
    result["root_visits"] = out.root_visits;
    result["root_value_sum"] = out.root_value_sum;
    result["root_state_cost"] = out.root_state_cost;
    result["root_sim_time"] = out.root_sim_time;
    result["root_num_valid_actions"] = out.root_num_valid_actions;

    result["root_nn_value_controller"] = out.root_nn_value_controller;
    result["root_nn_priors"] = out.root_nn_priors;
    result["root_nn_priors_after_threshold"] = out.root_nn_priors_after_threshold;
    result["root_nn_valid_mask"] = out.root_nn_valid_mask;
    result["root_inputs"] = search_root_inputs_to_py(out);
    result["best_action_index"] = out.best_action_index;
    result["root_action_values"] = out.root_action_values;

    result["action_alias_to_canonical"] = py_alias_to_canonical;
    result["canonical_to_action_aliases"] = py_canonical_to_aliases;

    result["children"] = py_children;
    result["mcts_root_prior"] = out.mcts_root_prior;
    result["iter_events"] = py_iter_events;
    result["perf"] = py_perf;

    return result;
}

py::dict generate_selfplay_gv3_torchscript(
    NativeTorchScriptInferRuntimeGV2& infer_runtime,
    int model_version,
    py::dict initial_state_payload,
    py::dict cfg_payload,
    int game_id,
    int num_roots,
    int start_root_id,
    int start_root_depth,
    const std::string& start_player,
    int feature_version,
    int adv_iterations_per_root,
    int cont_iterations_per_root,
    int history_hops_min,
    int history_hops_max,
    int history_seed,
    int history_max_total_steps,
    int max_forced_hops_per_root,
    double eval_split_ratio,
    int eval_split_seed,
    int action_seed_base,
    bool allow_duplicate_history_fallback) {
    auto get_int = [&](std::initializer_list<const char*> keys, int dflt) {
        for (const char* k : keys) {
            if (!cfg_payload.contains(k)) continue;
            try {
                return py::cast<int>(cfg_payload[k]);
            } catch (const std::exception&) {
            }
        }
        return dflt;
    };
    auto get_double = [&](std::initializer_list<const char*> keys, double dflt) {
        for (const char* k : keys) {
            if (!cfg_payload.contains(k)) continue;
            try {
                return py::cast<double>(cfg_payload[k]);
            } catch (const std::exception&) {
            }
        }
        return dflt;
    };
    auto get_bool = [&](std::initializer_list<const char*> keys, bool dflt) {
        for (const char* k : keys) {
            if (!cfg_payload.contains(k)) continue;
            try {
                return py::cast<bool>(cfg_payload[k]);
            } catch (const std::exception&) {
            }
        }
        return dflt;
    };
    auto get_str = [&](std::initializer_list<const char*> keys, const std::string& dflt) {
        for (const char* k : keys) {
            if (!cfg_payload.contains(k)) continue;
            try {
                return py::cast<std::string>(cfg_payload[k]);
            } catch (const std::exception&) {
            }
        }
        return dflt;
    };

    NativeSelfplayConfigGV3 cfg;
    cfg.initial_state = parse_root_state_payload(initial_state_payload);
    cfg.has_initial_state = true;
    cfg.game_id = int(game_id);
    cfg.num_roots = int(num_roots);
    cfg.start_root_id = int(start_root_id);
    cfg.start_root_depth = int(start_root_depth);
    cfg.start_player = start_player.empty() ? std::string("adversary") : start_player;
    cfg.feature_version = int(feature_version);
    cfg.adv_iterations_per_root = int(adv_iterations_per_root);
    cfg.cont_iterations_per_root = int(cont_iterations_per_root);
    cfg.history_hops_min = int(history_hops_min);
    cfg.history_hops_max = int(history_hops_max);
    cfg.history_seed = int(history_seed);
    cfg.history_max_total_steps = int(history_max_total_steps);
    cfg.max_forced_hops_per_root = int(max_forced_hops_per_root);
    cfg.eval_split_ratio = double(eval_split_ratio);
    cfg.eval_split_seed = int(eval_split_seed);
    cfg.action_seed_base = int(action_seed_base);
    cfg.allow_duplicate_history_fallback = bool(allow_duplicate_history_fallback);
    if (cfg_payload.contains("initial_history_signatures")) {
        try {
            for (auto item : py::cast<py::iterable>(cfg_payload["initial_history_signatures"])) {
                cfg.initial_seen_signatures.push_back(py::cast<std::string>(item));
            }
        } catch (const std::exception&) {
            cfg.initial_seen_signatures.clear();
        }
    }

    SearchInput& tmpl = cfg.search_template;
    tmpl.contract_version = kGV2NativeContractVersion;
    tmpl.root_player = cfg.start_player;
    tmpl.game_id = cfg.game_id;
    tmpl.root_id = cfg.start_root_id;
    tmpl.root_depth = cfg.start_root_depth;
    tmpl.root_node_id = cfg.start_root_id;
    tmpl.seed = cfg.action_seed_base;
    tmpl.max_forced_hops = int(max_forced_hops_per_root);
    tmpl.pb_c_base = get_double({"pb_c_base", "mcts_pb_c_base"}, tmpl.pb_c_base);
    tmpl.pb_c_init = get_double({"pb_c_init", "mcts_pb_c_init"}, tmpl.pb_c_init);
    tmpl.discount_factor = get_double({"discount_factor", "mcts_discount_factor"}, tmpl.discount_factor);
    tmpl.prefill_step_time = get_double({"discount_time_denominator_sec", "prefill_step_time"}, tmpl.prefill_step_time);
    tmpl.reward_knee = get_double({"reward_knee"}, tmpl.reward_knee);
    tmpl.reward_max_penalty = get_double({"reward_max_penalty"}, tmpl.reward_max_penalty);
    tmpl.reward_tail_alpha = get_double({"reward_tail_alpha"}, tmpl.reward_tail_alpha);
    tmpl.reuse_root_infer_inputs = false;
    tmpl.log_events = false;
    tmpl.profile = false;
    apply_feature_cfg_payload(cfg_payload, &tmpl.feature_cfg);

    tmpl.env_cfg.adversary_tick_sec = get_double({"adversary_tick_sec"}, tmpl.env_cfg.adversary_tick_sec);
    tmpl.env_cfg.launch_window_sec = get_double({"launch_window_sec"}, tmpl.env_cfg.launch_window_sec);
    tmpl.env_cfg.max_requests_per_launch_window = get_int(
        {"max_requests_per_launch_window"},
        tmpl.env_cfg.max_requests_per_launch_window);
    tmpl.env_cfg.prefill_window_cap_tokens = get_int(
        {"prefill_window_cap_tokens"},
        tmpl.env_cfg.prefill_window_cap_tokens);
    tmpl.env_cfg.max_prefill_tokens_per_request = get_int(
        {"max_prefill_tokens_per_request"},
        tmpl.env_cfg.max_prefill_tokens_per_request);
    tmpl.env_cfg.max_decode_tokens_per_request = get_int(
        {"max_decode_tokens_per_request"},
        tmpl.env_cfg.max_decode_tokens_per_request);
    tmpl.env_cfg.min_decode_tokens_per_request = get_int(
        {"min_decode_tokens_per_request"},
        tmpl.env_cfg.min_decode_tokens_per_request);
    tmpl.env_cfg.decode_slo_time_default = get_double(
        {"decode_slo_time_default"},
        tmpl.env_cfg.decode_slo_time_default);
    tmpl.env_cfg.auto_drop_lateness_sec = get_double(
        {"auto_drop_lateness_sec"},
        tmpl.env_cfg.auto_drop_lateness_sec);
    tmpl.env_cfg.drop_cost = get_double({"drop_cost"}, tmpl.env_cfg.drop_cost);
    tmpl.env_cfg.controller_noop_prefill_only_jump_to_next_adv_tick = get_bool(
        {"controller_noop_prefill_only_jump_to_next_adv_tick"},
        tmpl.env_cfg.controller_noop_prefill_only_jump_to_next_adv_tick);
    tmpl.env_cfg.enforce_nonnegative_decode_credits = get_bool(
        {"enforce_nonnegative_decode_credits"},
        tmpl.env_cfg.enforce_nonnegative_decode_credits);
    tmpl.env_cfg.decode_credit_mint_per_prefill_complete = get_int(
        {"decode_credit_mint_per_prefill_complete"},
        tmpl.env_cfg.decode_credit_mint_per_prefill_complete);

    tmpl.sim_cfg.adversary_tick_sec = tmpl.env_cfg.adversary_tick_sec;
    if (cfg_payload.contains("prefill_profile_tokens")) {
        tmpl.sim_cfg.prefill_profile_tokens = py_to_i32_vec(cfg_payload["prefill_profile_tokens"]);
    }
    if (cfg_payload.contains("prefill_profile_times")) {
        tmpl.sim_cfg.prefill_profile_times = py_to_f64_vec(cfg_payload["prefill_profile_times"]);
    }
    tmpl.sim_cfg.fallback_total_time_sec = get_double(
        {"fallback_total_time_sec"},
        tmpl.sim_cfg.fallback_total_time_sec);
    tmpl.sim_cfg.fallback_model_time_sec = get_double(
        {"fallback_model_time_sec"},
        tmpl.sim_cfg.fallback_model_time_sec);
    tmpl.predictor_csv_path = get_str(
        {"native_predictor_csv_path", "prefill_predictor_csv", "predictor_csv_path"},
        tmpl.predictor_csv_path);

    GV2VirtualEnvironment env(tmpl.env_cfg, tmpl.sim_cfg);
    if (!tmpl.predictor_csv_path.empty()) {
        (void)env.load_predictor_csv(tmpl.predictor_csv_path);
    }
    if (!tmpl.sim_cfg.prefill_profile_tokens.empty() &&
        tmpl.sim_cfg.prefill_profile_tokens.size() == tmpl.sim_cfg.prefill_profile_times.size()) {
        env.set_prefill_profile(tmpl.sim_cfg.prefill_profile_tokens, tmpl.sim_cfg.prefill_profile_times);
    }

    NativeSelfplayResultGV3 native_result = generate_native_selfplay_samples_gv3(
        cfg,
        env,
        infer_runtime,
        int(model_version));

    py::list samples;
    for (const auto& sample : native_result.samples) {
        samples.append(native_sample_to_py(sample));
    }
    py::dict stats;
    for (const auto& kv : native_result.stats) {
        stats[kv.first.c_str()] = kv.second;
    }

    py::dict result;
    result["samples"] = std::move(samples);
    result["stats"] = std::move(stats);
    result["history_signatures"] = native_result.history_signatures;
    return result;
}

}  // namespace

PYBIND11_MODULE(mcts_native_gv2, m) {
    m.doc() = "Game Version 2 native MCTS phase-1 module";

    py::class_<NativeTorchScriptInferRuntimeGV2>(m, "NativeTorchScriptInferRuntimeGV2")
        .def(py::init<std::string, double, double>(),
             py::arg("device"),
             py::arg("v_min") = -50.0,
             py::arg("v_step") = 100.0)
        .def("load_models", [](NativeTorchScriptInferRuntimeGV2& self, const py::dict& d) {
            std::unordered_map<int, std::string> mm;
            for (auto item : d) {
                mm.emplace(py::cast<int>(item.first), py::cast<std::string>(item.second));
            }
            self.load_models(mm);
        })
        .def("infer_from_inputs",
             [](NativeTorchScriptInferRuntimeGV2& self,
                const py::dict& inputs,
                const std::string& player,
                int model_version) {
                NativeInferInputsGV2 in;

                if (inputs.contains("global_features")) {
                    in.global_features = py_to_f32_flat(inputs["global_features"]);
                }
                if (inputs.contains("action_mask")) {
                    in.action_mask = py_to_mask_flat(inputs["action_mask"]);
                }
                if (in.action_mask.empty()) in.action_mask.assign(8, 1u);

                auto get_i = [&](const char* k, int defv) -> int {
                    if (!inputs.contains(k)) return defv;
                    try {
                        return py::cast<int>(inputs[k]);
                    } catch (const std::exception&) {
                        return defv;
                    }
                };

                if (inputs.contains("prefill_req_features")) {
                    in.prefill_req_features = py_to_f32_flat(inputs["prefill_req_features"]);
                }
                if (inputs.contains("decode_req_features")) {
                    in.decode_req_features = py_to_f32_flat(inputs["decode_req_features"]);
                }
                if (inputs.contains("prefill_req_mask")) {
                    in.prefill_req_mask = py_to_mask_flat(inputs["prefill_req_mask"]);
                }
                if (inputs.contains("decode_req_mask")) {
                    in.decode_req_mask = py_to_mask_flat(inputs["decode_req_mask"]);
                }
                in.prefill_req_n = get_i("prefill_req_n", get_i("n_prefill_req", 0));
                in.prefill_req_d = get_i("prefill_req_d", get_i("d_prefill_req", 0));
                in.decode_req_n = get_i("decode_req_n", get_i("n_decode_req", 0));
                in.decode_req_d = get_i("decode_req_d", get_i("d_decode_req", 0));

                if (inputs.contains("req_features")) {
                    in.req_features = py_to_f32_flat(inputs["req_features"]);
                }
                if (inputs.contains("req_mask")) {
                    in.req_mask = py_to_mask_flat(inputs["req_mask"]);
                }
                in.req_n = get_i("req_n", get_i("n_req", 0));
                in.req_d = get_i("req_d", get_i("d_req", 0));

                auto out = self.infer_from_inputs(in, player, model_version);
                return py::make_tuple(out.first, out.second);
             },
             py::arg("inputs"),
             py::arg("player"),
             py::arg("model_version"))
        .def_property_readonly("device", &NativeTorchScriptInferRuntimeGV2::device)
        .def_property_readonly("v_min", &NativeTorchScriptInferRuntimeGV2::v_min)
        .def_property_readonly("v_step", &NativeTorchScriptInferRuntimeGV2::v_step);

    m.def(
        "search_mcts_dnn_torchscript",
        &search_mcts_dnn_gv2_torchscript,
        py::arg("infer_runtime"),
        py::arg("model_version"),
        py::arg("root_state_payload"),
        py::arg("cfg_payload"),
        py::arg("iterations"),
        py::arg("root_player"),
        py::arg("root_node_id"),
        py::arg("root_depth"),
        py::arg("game_id"),
        py::arg("root_id"),
        py::arg("seed"),
        py::arg("log_events") = true,
        py::arg("profile") = false,
        py::arg("iter_log_path") = "",
        py::arg("root_log_path") = ""
    );

    m.def(
        "generate_selfplay_samples_torchscript",
        &generate_selfplay_gv3_torchscript,
        py::arg("infer_runtime"),
        py::arg("model_version"),
        py::arg("initial_state_payload"),
        py::arg("cfg_payload"),
        py::arg("game_id"),
        py::arg("num_roots"),
        py::arg("start_root_id"),
        py::arg("start_root_depth"),
        py::arg("start_player"),
        py::arg("feature_version") = 1,
        py::arg("adv_iterations_per_root") = 1,
        py::arg("cont_iterations_per_root") = 1,
        py::arg("history_hops_min") = 0,
        py::arg("history_hops_max") = 0,
        py::arg("history_seed") = 0,
        py::arg("history_max_total_steps") = 20000,
        py::arg("max_forced_hops_per_root") = 2000,
        py::arg("eval_split_ratio") = 0.0,
        py::arg("eval_split_seed") = 0,
        py::arg("action_seed_base") = 0,
        py::arg("allow_duplicate_history_fallback") = true
    );
}
