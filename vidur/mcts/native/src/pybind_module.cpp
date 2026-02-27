#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "native_mcts.hpp"
#include "native_predictor.hpp"
#include "native_sim.hpp"
#include "native_types.hpp"
#include "infer_shared.hpp"

namespace py = pybind11;
using namespace mcts_native;

struct NativeActionSpace {};
struct NativeTreeNode;
struct NativeSearchCfg;

#ifdef _WIN32
#include <process.h>
static int native_getpid() { return _getpid(); }
#else
#include <unistd.h>
static int native_getpid() { return (int)::getpid(); }
#endif

static bool native_trace_enabled() {
    static bool enabled = [] {
        const char* v = std::getenv("VIDUR_NATIVE_TRACE");
        if (v == nullptr) return false;
        std::string s(v);
        std::transform(
            s.begin(),
            s.end(),
            s.begin(),
            [](unsigned char c) { return (char)std::tolower(c); }
        );
        return !(s.empty() || s == "0" || s == "false" || s == "off" || s == "no");
    }();
    return enabled;
}

static int native_trace_every() {
    static int every = [] {
        const char* v = std::getenv("VIDUR_NATIVE_TRACE_EVERY");
        if (v == nullptr) return 500;
        try {
            const int parsed = std::stoi(std::string(v));
            return std::max(1, parsed);
        } catch (...) {
            return 500;
        }
    }();
    return every;
}

static std::string native_trace_path() {
    const char* f = std::getenv("VIDUR_NATIVE_TRACE_FILE");
    if (f != nullptr && *f != '\0') return std::string(f);

    std::string dir = "/tmp";
    const char* d = std::getenv("VIDUR_NATIVE_TRACE_DIR");
    if (d != nullptr && *d != '\0') dir = std::string(d);
    try {
        std::filesystem::create_directories(dir);
    } catch (...) {
    }

    std::ostringstream oss;
    oss << dir << "/native_worker_" << native_getpid() << ".log";
    return oss.str();
}

static std::ofstream& native_trace_stream() {
    static std::ofstream ofs;
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        ofs.open(native_trace_path(), std::ios::app);
        if (ofs) {
            ofs << "=== native trace start pid=" << native_getpid() << " ===" << std::endl;
        }
    }
    return ofs;
}

static void native_trace(const std::string& msg) {
    if (!native_trace_enabled()) return;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    auto& ofs = native_trace_stream();
    if (ofs) {
        ofs << ms
            << " pid=" << native_getpid()
            << " tid=" << std::this_thread::get_id()
            << " " << msg << std::endl;
    } else {
        std::cerr << ms << " pid=" << native_getpid() << " " << msg << std::endl;
    }
}

static bool native_trace_should_log(long long count) {
    if (!native_trace_enabled()) return false;
    if (count <= 10) return true;
    const int every = native_trace_every();
    return every > 0 && (count % (long long)every) == 0;
}

static std::atomic<long long> g_native_ts_search_calls{0};
static std::atomic<long long> g_native_ts_infer_calls{0};

static inline int nonneg_i(int x) { return x < 0 ? 0 : x; }

static inline double now_sec() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

static double nearest_prefill_estimate(
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
            best_dist = d;
            best = i;
        }
    }
    return profile_times[best];
}

static ControllerActionSpecNative make_zero_action() {
    ControllerActionSpecNative a;
    a.token_budget = 0;
    a.valid = true;
    return a;
}

static ControllerActionSpecNative build_controller_action(
    const std::vector<int>& ordered_indices,
    int prefill_budget,
    const char* heur_name,
    const std::vector<PrefillRecord>& prefill_records,
    const std::vector<AllocationEntry>& decode_template,
    int decode_budget
) {
    ControllerActionSpecNative out;
    out.valid = true;
    out.heuristic = heur_name;
    out.strategy = "Fixed";

    out.decode_allocations = decode_template;
    out.token_allocations = decode_template;
    out.selected_request_ids.reserve(decode_template.size() + ordered_indices.size());
    for (const auto& e : decode_template) {
        out.selected_request_ids.push_back(e.request_id);
    }

    int remaining_budget = nonneg_i(prefill_budget);
    int used_prefill = 0;

    for (int idx : ordered_indices) {
        if (remaining_budget <= 0) break;
        const PrefillRecord& r = prefill_records[idx];
        int cap = nonneg_i(r.rem_pref);
        if (cap <= 0) continue;

        int alloc = (cap < remaining_budget) ? cap : remaining_budget;
        if (alloc <= 0) continue;

        out.prefill_allocations.push_back({r.rid, alloc});
        out.token_allocations.push_back({r.rid, alloc});
        out.selected_request_ids.push_back(r.rid);

        remaining_budget -= alloc;
        used_prefill += alloc;
    }

    out.token_budget = decode_budget + used_prefill;
    return out;
}

ControllerSampleOutput sample_controller_actions_native(
    const std::vector<ControllerRequestStateNative>& request_states,
    double sim_time,
    const std::vector<int>& budgets,
    const std::vector<int>& profile_tokens,
    const std::vector<double>& profile_times
) {
    ControllerSampleOutput out;
    const int num_heur = 4;
    const int num_budgets = (int)budgets.size();
    const int num_actions = num_heur * num_budgets;

    if (num_actions <= 0) {
        out.actions.resize(1);
        out.mask.resize(1, 0);
        out.actions[0] = make_zero_action();
        out.mask[0] = 1;
        return out;
    }

    out.actions.resize((size_t)num_actions);
    out.mask.assign((size_t)num_actions, 0);

    if (request_states.empty()) {
        out.actions[0] = make_zero_action();
        out.mask[0] = 1;
        return out;
    }

    std::vector<PrefillRecord> prefill_records;
    prefill_records.reserve(request_states.size());

    std::vector<int> decode_candidates;
    decode_candidates.reserve(request_states.size());

    std::vector<AllocationEntry> decode_template;
    decode_template.reserve(request_states.size());

    int total_remaining_prefill = 0;

    for (const auto& rs : request_states) {
        const int rid = rs.request_id;
        if (!rs.prefill_done) {
            int rem_pref = nonneg_i(rs.remaining_prefill);
            if (rem_pref > 0) {
                total_remaining_prefill += rem_pref;

                const double edf_key = rs.arrived_at + rs.prefill_slo;
                const double remaining_slo = rs.prefill_slo - std::max(0.0, sim_time - rs.arrived_at);
                const double est = nearest_prefill_estimate(rem_pref, profile_tokens, profile_times);
                const double lst_key = remaining_slo - est;

                prefill_records.push_back(PrefillRecord{rid, rem_pref, edf_key, lst_key});
            }
        } else {
            if (nonneg_i(rs.remaining_decode) > 0) {
                decode_candidates.push_back(rid);
                decode_template.push_back({rid, 1});
            }
        }
    }

    if (total_remaining_prefill == 0) {
        ControllerActionSpecNative a;
        a.valid = true;
        a.heuristic = "SJF";
        a.strategy = "Fixed";
        a.decode_allocations = decode_template;
        a.token_allocations = decode_template;
        a.selected_request_ids = decode_candidates;
        a.token_budget = (int)decode_template.size();

        out.actions[0] = std::move(a);
        out.mask[0] = 1;
        return out;
    }

    const int decode_budget = (int)decode_candidates.size();
    const int n = (int)prefill_records.size();

    std::vector<int> ordered_sjf((size_t)n);
    std::iota(ordered_sjf.begin(), ordered_sjf.end(), 0);

    auto ordered_edf = ordered_sjf;
    auto ordered_lst = ordered_sjf;

    std::stable_sort(
        ordered_sjf.begin(), ordered_sjf.end(),
        [&](int a, int b) { return prefill_records[a].rem_pref < prefill_records[b].rem_pref; }
    );
    std::stable_sort(
        ordered_edf.begin(), ordered_edf.end(),
        [&](int a, int b) { return prefill_records[a].edf_key < prefill_records[b].edf_key; }
    );
    std::stable_sort(
        ordered_lst.begin(), ordered_lst.end(),
        [&](int a, int b) { return prefill_records[a].lst_key < prefill_records[b].lst_key; }
    );

    auto ordered_ljf = ordered_sjf;
    std::reverse(ordered_ljf.begin(), ordered_ljf.end());

    const std::vector<int>* orders[4] = {&ordered_sjf, &ordered_edf, &ordered_lst, &ordered_ljf};
    const char* names[4] = {"SJF", "EDF", "LST", "LJF"};

    bool any_valid = false;
    for (int b_idx = 0; b_idx < num_budgets; ++b_idx) {
        const int budget = nonneg_i(budgets[(size_t)b_idx]);
        const bool budget_valid = (total_remaining_prefill > 0) ? (budget <= total_remaining_prefill) : (b_idx == 0);

        for (int h_idx = 0; h_idx < 4; ++h_idx) {
            const int idx = b_idx * 4 + h_idx;
            if (!budget_valid) {
                out.mask[(size_t)idx] = 0;
                continue;
            }
            out.actions[(size_t)idx] = build_controller_action(
                *orders[h_idx], budget, names[h_idx], prefill_records, decode_template, decode_budget
            );
            out.mask[(size_t)idx] = 1;
            any_valid = true;
        }
    }

    if (!any_valid) {
        out.actions[0] = make_zero_action();
        out.mask[0] = 1;
    }

    return out;
}

static py::dict alloc_to_pydict(const std::vector<AllocationEntry>& v) {
    py::dict d;
    for (const auto& e : v) d[py::int_(e.request_id)] = py::int_(e.tokens);
    return d;
}

py::tuple sample_controller_actions_pyready(
    const std::vector<ControllerRequestStateNative>& request_states,
    double sim_time,
    const std::vector<int>& budgets,
    const std::vector<int>& profile_tokens,
    const std::vector<double>& profile_times,
    int num_actions
) {
    ControllerSampleOutput out = sample_controller_actions_native(
        request_states, sim_time, budgets, profile_tokens, profile_times
    );

    py::object ControllerAction =
        py::module_::import("vidur.mcts.environment").attr("ControllerAction");

    py::list actions_by_index;
    py::list mask;

    for (int i = 0; i < num_actions; ++i) {
        actions_by_index.append(py::none());
        mask.append(py::bool_(false));
    }

    int n = std::min(num_actions, (int)out.mask.size());
    bool any_valid = false;

    for (int idx = 0; idx < n; ++idx) {
        if (!out.mask[(size_t)idx]) continue;
        const auto& a = out.actions[(size_t)idx];

        py::list selected;
        for (int rid : a.selected_request_ids) selected.append(py::int_(rid));
        py::object selected_obj = selected.size() ? py::object(selected) : py::none();

        py::object heur_obj = a.heuristic.empty()
            ? py::reinterpret_borrow<py::object>(py::none())
            : py::reinterpret_steal<py::object>(py::str(a.heuristic).release());
        py::object strat_obj = a.strategy.empty()
            ? py::reinterpret_borrow<py::object>(py::none())
            : py::reinterpret_steal<py::object>(py::str(a.strategy).release());

        py::object py_action = ControllerAction(
            py::arg("token_budget") = py::int_(a.token_budget),
            py::arg("selected_request_ids") = selected_obj,
            py::arg("token_allocations") = alloc_to_pydict(a.token_allocations),
            py::arg("prefill_allocations") = alloc_to_pydict(a.prefill_allocations),
            py::arg("decode_allocations") = alloc_to_pydict(a.decode_allocations),
            py::arg("heuristic") = heur_obj,
            py::arg("strategy") = strat_obj
        );

        actions_by_index[py::int_(idx)] = py_action;
        mask[py::int_(idx)] = py::bool_(true);
        any_valid = true;
    }

    if (!any_valid && num_actions > 0) {
        actions_by_index[py::int_(0)] = ControllerAction(
            py::arg("token_budget") = 0,
            py::arg("selected_request_ids") = py::none()
        );
        mask[py::int_(0)] = py::bool_(true);
    }

    return py::make_tuple(actions_by_index, mask);
}

py::tuple sample_adversary_actions_pyready(int num_actions) {
    py::object AdversaryAction = py::module_::import("vidur.mcts.environment").attr("AdversaryAction");
    py::list actions;
    py::list mask;
    for (int i = 0; i < num_actions; ++i) {
        actions.append(py::none());
        mask.append(py::bool_(false));
    }
    if (num_actions > 0) {
        actions[py::int_(0)] = AdversaryAction(
            py::arg("requests") = py::list(),
            py::arg("stop_decode_ids") = py::list()
        );
        mask[py::int_(0)] = py::bool_(true);
    }
    return py::make_tuple(actions, mask);
}

struct NativeTsModelPair {
    torch::jit::script::Module controller;
    torch::jit::script::Module adversary;
};

class NativeTorchScriptInferRuntime {
public:
    explicit NativeTorchScriptInferRuntime(
        const std::string& device = "cpu",
        double v_min = -50.0,
        double v_step = 0.125
    )
        : device_(parse_device(device)), v_min_(v_min), v_step_(v_step) {}

    void load_models(int model_version, const std::string& controller_path, const std::string& adversary_path) {
        if (models_.find(model_version) != models_.end()) return;
        // Bound per-worker model cache growth across generations.
        // We keep only a small number of versions because workers repeatedly
        // load new checkpoints in long self-play runs.
        if (models_.size() >= max_cached_model_versions_) {
            models_.clear();
        }
        NativeTsModelPair pair{
            torch::jit::load(controller_path, device_),
            torch::jit::load(adversary_path, device_),
        };
        pair.controller.eval();
        pair.adversary.eval();
        models_.emplace(model_version, std::move(pair));
    }

    py::tuple infer_from_inputs(const py::object& inputs, const std::string& player, int model_version) {
        torch::Tensor req_features = inputs.attr("req_features").cast<torch::Tensor>();
        torch::Tensor global_features = inputs.attr("global_features").cast<torch::Tensor>();
        c10::optional<torch::Tensor> req_mask = c10::nullopt;
        c10::optional<torch::Tensor> action_mask = c10::nullopt;

        py::object req_mask_obj = inputs.attr("req_mask");
        if (!req_mask_obj.is_none()) req_mask = req_mask_obj.cast<torch::Tensor>();
        py::object action_mask_obj = inputs.attr("action_mask");
        if (!action_mask_obj.is_none()) action_mask = action_mask_obj.cast<torch::Tensor>();

        auto out = infer_tensors(
            req_features,
            global_features,
            req_mask,
            action_mask,
            player,
            model_version
        );
        return py::make_tuple(out.first, out.second);
    }

    std::pair<double, std::vector<double>> infer_tensors(
        const torch::Tensor& req_features,
        const torch::Tensor& global_features,
        c10::optional<torch::Tensor> req_mask,
        c10::optional<torch::Tensor> action_mask,
        const std::string& player,
        int model_version
    ) {
        auto it = models_.find(model_version);
        if (it == models_.end()) {
            throw std::runtime_error("NativeTorchScriptInferRuntime: model version not loaded");
        }
        auto& mod = (player == "controller") ? it->second.controller : it->second.adversary;
        if (player != "controller" && player != "adversary") {
            throw std::runtime_error("NativeTorchScriptInferRuntime: invalid player");
        }

        const long long infer_call = ++g_native_ts_infer_calls;
        if (native_trace_should_log(infer_call)) {
            std::ostringstream oss;
            oss
                << "infer_tensors begin call=" << infer_call
                << " player=" << player
                << " model_version=" << model_version
                << " req_shape=[" << req_features.size(0) << "," << req_features.size(1) << "," << req_features.size(2) << "]"
                << " global_shape=[" << global_features.size(0) << "," << global_features.size(1) << "]";
            native_trace(oss.str());
        }

        c10::InferenceMode infer_guard(true);
        std::vector<torch::jit::IValue> in;
        in.reserve(4);
        in.emplace_back(req_features.to(device_));
        in.emplace_back(global_features.to(device_));
        if (req_mask.has_value()) {
            in.emplace_back(req_mask.value().to(device_));
        } else {
            in.emplace_back(torch::zeros({1, req_features.size(1)}, torch::TensorOptions().dtype(torch::kBool).device(device_)));
        }
        if (action_mask.has_value()) {
            in.emplace_back(action_mask.value().to(device_));
        } else {
            const int64_t a = (player == "controller") ? 24 : 6;
            in.emplace_back(torch::ones({1, a}, torch::TensorOptions().dtype(torch::kBool).device(device_)));
        }

        auto out_iv = mod.forward(in);
        auto out_t = out_iv.toTuple();
        torch::Tensor policy_logits = out_t->elements()[0].toTensor();
        torch::Tensor value_logits = out_t->elements()[1].toTensor();

        torch::Tensor priors_t = torch::softmax(policy_logits, -1)
            .squeeze(0)
            .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64))
            .contiguous();
        std::vector<double> priors((size_t)priors_t.numel(), 0.0);
        if (!priors.empty()) {
            auto p_acc = priors_t.data_ptr<double>();
            for (size_t i = 0; i < priors.size(); ++i) priors[i] = p_acc[i];
        }

        const int64_t bins = value_logits.size(-1);
        torch::Tensor support = torch::arange(
            0,
            bins,
            torch::TensorOptions().dtype(torch::kFloat32).device(value_logits.device())
        ) * (float)v_step_ + (float)v_min_;
        torch::Tensor probs = torch::softmax(value_logits, -1);
        torch::Tensor value_t = (probs * support).sum(-1).squeeze(0).to(torch::kCPU);
        const double value = value_t.item<double>();
        if (native_trace_should_log(infer_call)) {
            std::ostringstream oss;
            oss
                << "infer_tensors end call=" << infer_call
                << " value=" << value
                << " priors_size=" << priors.size();
            native_trace(oss.str());
        }
        return std::make_pair(value, std::move(priors));
    }

private:
    static torch::Device parse_device(const std::string& s) {
        try {
            torch::Device d(s);
            if (d.is_cuda() && !torch::cuda::is_available()) {
                std::ostringstream oss;
                oss << "NativeTorchScriptInferRuntime requested CUDA device '" << s
                    << "' but torch::cuda::is_available() is false";
                throw std::runtime_error(oss.str());
            }
            return d;
        } catch (const std::exception&) {
            throw;
        } catch (...) {
            throw std::runtime_error(
                "NativeTorchScriptInferRuntime failed to parse requested device: " + s
            );
        }
    }

    std::unordered_map<int, NativeTsModelPair> models_;
    torch::Device device_;
    double v_min_ = -50.0;
    double v_step_ = 0.125;
    size_t max_cached_model_versions_ = 3;
};

namespace {
constexpr uint8_t kInferCmdPing = 1;
constexpr uint8_t kInferCmdShutdown = 2;
constexpr uint8_t kInferCmdLoadModels = 3;
constexpr uint8_t kInferCmdInfer = 4;
}  // namespace

class NativeInferServiceRuntime {
public:
    explicit NativeInferServiceRuntime(
        const std::string& addr,
        double v_min = -50.0,
        double v_step = 0.125,
        int connect_timeout_ms = 2000,
        int request_timeout_ms = 30000
    )
        : addr_(addr),
          v_min_(v_min),
          v_step_(v_step),
          connect_timeout_ms_(std::max(100, connect_timeout_ms)),
          request_timeout_ms_(std::max(1000, request_timeout_ms)) {
        parse_addr(addr_, host_, port_);
    }

    ~NativeInferServiceRuntime() {
        close_shm();
        close_socket();
    }

    bool ping() {
        std::vector<uint8_t> req{(uint8_t)kInferCmdPing};
        std::vector<uint8_t> resp;
        request(req, resp);
        parse_ok_empty(resp, "ping");
        return true;
    }

    void shutdown() {
        std::vector<uint8_t> req{(uint8_t)kInferCmdShutdown};
        std::vector<uint8_t> resp;
        request(req, resp);
        parse_ok_empty(resp, "shutdown");
    }

    void load_models(int model_version, const std::string& controller_path, const std::string& adversary_path) {
        std::vector<uint8_t> req;
        req.reserve(1 + 4 + 8 + controller_path.size() + adversary_path.size());
        req.push_back((uint8_t)kInferCmdLoadModels);
        append_i32(req, model_version);
        append_str(req, controller_path);
        append_str(req, adversary_path);
        std::vector<uint8_t> resp;
        request(req, resp);
        parse_ok_empty(resp, "load_models");
    }

    py::tuple infer_from_inputs(const py::object& inputs, const std::string& player, int model_version) {
        torch::Tensor req_features = inputs.attr("req_features").cast<torch::Tensor>();
        torch::Tensor global_features = inputs.attr("global_features").cast<torch::Tensor>();
        c10::optional<torch::Tensor> req_mask = c10::nullopt;
        c10::optional<torch::Tensor> action_mask = c10::nullopt;

        py::object req_mask_obj = inputs.attr("req_mask");
        if (!req_mask_obj.is_none()) req_mask = req_mask_obj.cast<torch::Tensor>();
        py::object action_mask_obj = inputs.attr("action_mask");
        if (!action_mask_obj.is_none()) action_mask = action_mask_obj.cast<torch::Tensor>();

        auto out = infer_tensors(
            req_features,
            global_features,
            req_mask,
            action_mask,
            player,
            model_version
        );
        return py::make_tuple(out.first, out.second);
    }

    std::pair<double, std::vector<double>> infer_tensors(
        const torch::Tensor& req_features,
        const torch::Tensor& global_features,
        c10::optional<torch::Tensor> req_mask,
        c10::optional<torch::Tensor> action_mask,
        const std::string& player,
        int model_version
    ) {
        const bool is_controller = (player == "controller");
        if (!is_controller && player != "adversary") {
            throw std::runtime_error("NativeInferServiceRuntime: invalid player");
        }
        const int action_len = is_controller ? 24 : 6;

        torch::Tensor req_f = req_features.detach()
            .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32))
            .contiguous()
            .view({-1});
        torch::Tensor glob_f = global_features.detach()
            .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32))
            .contiguous()
            .view({-1});
        if (req_f.numel() != 60 || glob_f.numel() != 9) {
            std::ostringstream oss;
            oss << "NativeInferServiceRuntime: invalid feature sizes req=" << req_f.numel()
                << " global=" << glob_f.numel();
            throw std::runtime_error(oss.str());
        }

        torch::Tensor req_m;
        if (req_mask.has_value()) {
            req_m = req_mask.value().detach()
                .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8))
                .contiguous()
                .view({-1});
        } else {
            req_m = torch::ones({20}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        }
        if (req_m.numel() != 20) {
            std::ostringstream oss;
            oss << "NativeInferServiceRuntime: invalid req_mask size=" << req_m.numel();
            throw std::runtime_error(oss.str());
        }

        torch::Tensor act_m;
        if (action_mask.has_value()) {
            act_m = action_mask.value().detach()
                .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8))
                .contiguous()
                .view({-1});
        } else {
            act_m = torch::ones({action_len}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        }
        if (act_m.numel() != action_len) {
            std::ostringstream oss;
            oss << "NativeInferServiceRuntime: invalid action_mask size=" << act_m.numel()
                << " expected=" << action_len;
            throw std::runtime_error(oss.str());
        }

        const auto* req_ptr = req_f.data_ptr<float>();
        const auto* glob_ptr = glob_f.data_ptr<float>();
        const auto* req_m_ptr = req_m.data_ptr<uint8_t>();
        const auto* act_m_ptr = act_m.data_ptr<uint8_t>();

        std::lock_guard<std::mutex> lk(mu_);
        if (!ensure_shm_locked()) {
            throw std::runtime_error("NativeInferServiceRuntime infer: shared memory region not available");
        }
        infer_shared::Region* reg = shm_reg_;

        if (reg->magic != infer_shared::kMagic || reg->version != infer_shared::kVersion ||
            reg->slot_count != infer_shared::kSlots) {
            close_shm_locked();
            throw std::runtime_error("NativeInferServiceRuntime infer: shared memory region mismatch");
        }

        pthread_mutex_lock(&reg->mu);
        if (reg->shutdown != 0) {
            pthread_mutex_unlock(&reg->mu);
            throw std::runtime_error("NativeInferServiceRuntime infer: service is shutting down");
        }

        uint32_t slot_idx = 0;
        {
            const auto deadline = deadline_after_ms(request_timeout_ms_);
            while (reg->free_count == 0 && reg->shutdown == 0) {
                const int rc = pthread_cond_timedwait(&reg->cv_done, &reg->mu, &deadline);
                if (rc == ETIMEDOUT) {
                    pthread_mutex_unlock(&reg->mu);
                    throw std::runtime_error("NativeInferServiceRuntime infer: timeout waiting for free slot");
                }
                if (rc != 0) {
                    pthread_mutex_unlock(&reg->mu);
                    throw std::runtime_error("NativeInferServiceRuntime infer: wait for free slot failed");
                }
            }
            if (reg->shutdown != 0) {
                pthread_mutex_unlock(&reg->mu);
                throw std::runtime_error("NativeInferServiceRuntime infer: service is shutting down");
            }
            if (reg->free_count == 0) {
                pthread_mutex_unlock(&reg->mu);
                throw std::runtime_error("NativeInferServiceRuntime infer: no free slot available");
            }
            slot_idx = reg->free_ring[reg->free_head];
            reg->free_head = (reg->free_head + 1U) % infer_shared::kSlots;
            reg->free_count -= 1U;
        }

        if (slot_idx >= infer_shared::kSlots) {
            pthread_mutex_unlock(&reg->mu);
            throw std::runtime_error("NativeInferServiceRuntime infer: invalid slot index");
        }
        auto& slot = reg->slots[slot_idx];
        if (slot.state != infer_shared::SLOT_FREE) {
            pthread_mutex_unlock(&reg->mu);
            throw std::runtime_error("NativeInferServiceRuntime infer: slot state is not free");
        }

        uint32_t req_id = ++reg->request_seq;
        if (req_id == 0) req_id = ++reg->request_seq;
        slot.request_id = req_id;
        slot.model_version = model_version;
        slot.player = static_cast<uint8_t>(is_controller ? 0 : 1);
        slot.action_len = static_cast<uint16_t>(action_len);
        std::memcpy(slot.req_features, req_ptr, 60 * sizeof(float));
        std::memcpy(slot.global_features, glob_ptr, 9 * sizeof(float));
        std::memcpy(slot.req_mask, req_m_ptr, 20 * sizeof(uint8_t));
        std::memset(slot.action_mask, 0, sizeof(slot.action_mask));
        std::memcpy(slot.action_mask, act_m_ptr, static_cast<size_t>(action_len) * sizeof(uint8_t));
        slot.error_code = 0;
        std::memset(slot.error_msg, 0, sizeof(slot.error_msg));
        slot.state = infer_shared::SLOT_READY;

        reg->ready_ring[reg->ready_tail] = slot_idx;
        reg->ready_tail = (reg->ready_tail + 1U) % infer_shared::kSlots;
        reg->ready_count += 1U;
        pthread_cond_signal(&reg->cv_ready);

        {
            const auto deadline = deadline_after_ms(request_timeout_ms_);
            while (slot.state != infer_shared::SLOT_DONE && reg->shutdown == 0) {
                const int rc = pthread_cond_timedwait(&reg->cv_done, &reg->mu, &deadline);
                if (rc == ETIMEDOUT) break;
                if (rc != 0) {
                    break;
                }
            }
        }

        if (slot.state != infer_shared::SLOT_DONE) {
            if (slot.state == infer_shared::SLOT_READY) {
                slot.state = infer_shared::SLOT_FREE;
                slot.request_id = 0;
                slot.action_len = 0;
                reg->free_ring[reg->free_tail] = slot_idx;
                reg->free_tail = (reg->free_tail + 1U) % infer_shared::kSlots;
                reg->free_count += 1U;
                pthread_cond_signal(&reg->cv_done);
            }
            pthread_mutex_unlock(&reg->mu);
            throw std::runtime_error("NativeInferServiceRuntime infer: timeout waiting for response");
        }

        if (slot.request_id != req_id) {
            const uint32_t observed = slot.request_id;
            slot.state = infer_shared::SLOT_FREE;
            slot.request_id = 0;
            slot.action_len = 0;
            reg->free_ring[reg->free_tail] = slot_idx;
            reg->free_tail = (reg->free_tail + 1U) % infer_shared::kSlots;
            reg->free_count += 1U;
            pthread_cond_signal(&reg->cv_done);
            pthread_mutex_unlock(&reg->mu);
            std::ostringstream oss;
            oss << "NativeInferServiceRuntime infer: request id mismatch expected=" << req_id
                << " observed=" << observed;
            throw std::runtime_error(oss.str());
        }

        if (slot.error_code != 0) {
            std::string msg(slot.error_msg, strnlen(slot.error_msg, sizeof(slot.error_msg)));
            slot.state = infer_shared::SLOT_FREE;
            slot.request_id = 0;
            slot.action_len = 0;
            slot.error_code = 0;
            std::memset(slot.error_msg, 0, sizeof(slot.error_msg));
            reg->free_ring[reg->free_tail] = slot_idx;
            reg->free_tail = (reg->free_tail + 1U) % infer_shared::kSlots;
            reg->free_count += 1U;
            pthread_cond_signal(&reg->cv_done);
            pthread_mutex_unlock(&reg->mu);
            if (msg.empty()) msg = "unknown service error";
            throw std::runtime_error("NativeInferServiceRuntime infer error: " + msg);
        }

        const double value = static_cast<double>(slot.value);
        std::vector<double> priors(static_cast<size_t>(action_len), 0.0);
        for (int i = 0; i < action_len; ++i) {
            priors[static_cast<size_t>(i)] = static_cast<double>(slot.priors[static_cast<size_t>(i)]);
        }

        slot.state = infer_shared::SLOT_FREE;
        slot.request_id = 0;
        slot.action_len = 0;
        slot.error_code = 0;
        std::memset(slot.error_msg, 0, sizeof(slot.error_msg));
        reg->free_ring[reg->free_tail] = slot_idx;
        reg->free_tail = (reg->free_tail + 1U) % infer_shared::kSlots;
        reg->free_count += 1U;
        pthread_cond_signal(&reg->cv_done);
        pthread_mutex_unlock(&reg->mu);

        return std::make_pair(value, std::move(priors));
    }

private:
    static void parse_addr(const std::string& addr, std::string& host_out, int& port_out) {
        const auto pos = addr.rfind(':');
        if (pos == std::string::npos) {
            throw std::runtime_error("NativeInferServiceRuntime addr must be host:port");
        }
        host_out = addr.substr(0, pos);
        if (host_out.empty()) host_out = "127.0.0.1";
        port_out = std::stoi(addr.substr(pos + 1));
    }

    static void append_u16(std::vector<uint8_t>& out, uint16_t v) {
        out.push_back((uint8_t)(v & 0xFF));
        out.push_back((uint8_t)((v >> 8) & 0xFF));
    }

    static void append_u32(std::vector<uint8_t>& out, uint32_t v) {
        out.push_back((uint8_t)(v & 0xFF));
        out.push_back((uint8_t)((v >> 8) & 0xFF));
        out.push_back((uint8_t)((v >> 16) & 0xFF));
        out.push_back((uint8_t)((v >> 24) & 0xFF));
    }

    static void append_i32(std::vector<uint8_t>& out, int32_t v) {
        append_u32(out, (uint32_t)v);
    }

    static void append_str(std::vector<uint8_t>& out, const std::string& s) {
        append_u32(out, (uint32_t)s.size());
        out.insert(out.end(), s.begin(), s.end());
    }

    static void append_f32_bytes(std::vector<uint8_t>& out, const float* ptr, size_t n) {
        const uint8_t* b = reinterpret_cast<const uint8_t*>(ptr);
        out.insert(out.end(), b, b + n * sizeof(float));
    }

    static timespec deadline_after_ms(int timeout_ms) {
        timespec ts{};
        ::clock_gettime(CLOCK_REALTIME, &ts);
        const long add_ns = static_cast<long>(timeout_ms % 1000) * 1000000L;
        ts.tv_sec += timeout_ms / 1000;
        ts.tv_nsec += add_ns;
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000L;
        }
        return ts;
    }

    static std::runtime_error decode_error(const std::vector<uint8_t>& resp, const char* op) {
        if (resp.size() < 5) {
            std::ostringstream oss;
            oss << "NativeInferServiceRuntime " << op << ": malformed error response";
            return std::runtime_error(oss.str());
        }
        uint32_t n = (uint32_t)resp[1] |
                     ((uint32_t)resp[2] << 8) |
                     ((uint32_t)resp[3] << 16) |
                     ((uint32_t)resp[4] << 24);
        if (resp.size() < 5 + (size_t)n) {
            std::ostringstream oss;
            oss << "NativeInferServiceRuntime " << op << ": truncated error response";
            return std::runtime_error(oss.str());
        }
        std::string msg((const char*)resp.data() + 5, (size_t)n);
        std::ostringstream oss;
        oss << "NativeInferServiceRuntime " << op << " error: " << msg;
        return std::runtime_error(oss.str());
    }

    static void parse_ok_empty(const std::vector<uint8_t>& resp, const char* op) {
        if (resp.empty()) {
            std::ostringstream oss;
            oss << "NativeInferServiceRuntime " << op << ": empty response";
            throw std::runtime_error(oss.str());
        }
        if (resp[0] != 0) throw decode_error(resp, op);
    }

    bool read_exact_locked(void* dst, size_t n) {
        uint8_t* p = static_cast<uint8_t*>(dst);
        size_t off = 0;
        while (off < n) {
            const ssize_t r = ::recv(sock_fd_, p + off, n - off, 0);
            if (r <= 0) return false;
            off += (size_t)r;
        }
        return true;
    }

    bool write_exact_locked(const void* src, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(src);
        size_t off = 0;
        while (off < n) {
            const ssize_t w = ::send(sock_fd_, p + off, n - off, 0);
            if (w <= 0) return false;
            off += (size_t)w;
        }
        return true;
    }

    void close_socket_locked() {
        if (sock_fd_ >= 0) {
            ::close(sock_fd_);
            sock_fd_ = -1;
        }
    }

    void close_socket() {
        std::lock_guard<std::mutex> lk(mu_);
        close_socket_locked();
    }

    void close_shm_locked() {
        if (shm_reg_ != nullptr) {
            ::munmap(shm_reg_, sizeof(infer_shared::Region));
            shm_reg_ = nullptr;
        }
        if (shm_fd_ >= 0) {
            ::close(shm_fd_);
            shm_fd_ = -1;
        }
    }

    void close_shm() {
        std::lock_guard<std::mutex> lk(mu_);
        close_shm_locked();
    }

    bool ensure_shm_locked() {
        if (shm_reg_ != nullptr) return true;
        if (shm_name_.empty()) {
            try {
                shm_name_ = infer_shared::shm_name_from_addr(addr_);
            } catch (...) {
                return false;
            }
        }
        shm_fd_ = ::shm_open(shm_name_.c_str(), O_RDWR, 0);
        if (shm_fd_ < 0) return false;

        void* p = ::mmap(
            nullptr,
            sizeof(infer_shared::Region),
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            shm_fd_,
            0
        );
        if (p == MAP_FAILED) {
            ::close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }

        auto* reg = reinterpret_cast<infer_shared::Region*>(p);
        if (reg->magic != infer_shared::kMagic || reg->version != infer_shared::kVersion ||
            reg->slot_count != infer_shared::kSlots) {
            ::munmap(reg, sizeof(infer_shared::Region));
            ::close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }

        shm_reg_ = reg;
        return true;
    }

    bool connect_locked() {
        close_socket_locked();

        sock_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sock_fd_ < 0) return false;

        timeval snd_tv{};
        snd_tv.tv_sec = request_timeout_ms_ / 1000;
        snd_tv.tv_usec = (request_timeout_ms_ % 1000) * 1000;
        timeval rcv_tv = snd_tv;
        ::setsockopt(sock_fd_, SOL_SOCKET, SO_SNDTIMEO, &snd_tv, sizeof(snd_tv));
        ::setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &rcv_tv, sizeof(rcv_tv));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t)port_);
        if (::inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
            close_socket_locked();
            return false;
        }

        if (::connect(sock_fd_, (sockaddr*)&addr, sizeof(addr)) != 0) {
            close_socket_locked();
            return false;
        }
        return true;
    }

    bool request_locked_once(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp) {
        const uint32_t n = (uint32_t)req.size();
        uint8_t hdr[4] = {
            (uint8_t)(n & 0xFF),
            (uint8_t)((n >> 8) & 0xFF),
            (uint8_t)((n >> 16) & 0xFF),
            (uint8_t)((n >> 24) & 0xFF),
        };
        if (!write_exact_locked(hdr, 4)) return false;
        if (n > 0 && !write_exact_locked(req.data(), req.size())) return false;

        uint8_t rh[4];
        if (!read_exact_locked(rh, 4)) return false;
        const uint32_t rn = (uint32_t)rh[0] |
                            ((uint32_t)rh[1] << 8) |
                            ((uint32_t)rh[2] << 16) |
                            ((uint32_t)rh[3] << 24);
        resp.resize((size_t)rn);
        if (rn > 0 && !read_exact_locked(resp.data(), resp.size())) return false;
        return true;
    }

    void request(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp) {
        std::lock_guard<std::mutex> lk(mu_);
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (sock_fd_ < 0 && !connect_locked()) {
                continue;
            }
            if (request_locked_once(req, resp)) {
                return;
            }
            close_socket_locked();
        }
        throw std::runtime_error("NativeInferServiceRuntime request failed");
    }

    std::string addr_;
    std::string shm_name_;
    std::string host_;
    int port_ = 0;
    double v_min_ = -50.0;
    double v_step_ = 0.125;
    int connect_timeout_ms_ = 2000;
    int request_timeout_ms_ = 30000;
    int sock_fd_ = -1;
    int shm_fd_ = -1;
    infer_shared::Region* shm_reg_ = nullptr;
    std::mutex mu_;
};

struct NativeMinMaxStats {
    double maximum = -std::numeric_limits<double>::infinity();
    double minimum = std::numeric_limits<double>::infinity();

    void update(double v) {
        if (v > maximum) maximum = v;
        if (v < minimum) minimum = v;
    }
    double normalize(double v) const {
        if (maximum > minimum) {
            return (v - minimum) / (maximum - minimum);
        }
        return v;
    }
};

struct NativeSearchPerf {
    double total_sec = 0.0;
    double state_build_sec = 0.0;
    double predictor_load_sec = 0.0;
    double root_expand_sec = 0.0;
    double selection_sec = 0.0;
    double restore_sec = 0.0;
    double forced_chain_sec = 0.0;
    double forced_actions_mask_sec = 0.0;
    double forced_apply_sec = 0.0;
    double leaf_expand_sec = 0.0;
    double backprop_sec = 0.0;
    double actions_mask_total_sec = 0.0;
    double infer_total_sec = 0.0;
    double infer_build_sec = 0.0;
    double infer_forward_sec = 0.0;
    double expand_threshold_sec = 0.0;
    double expand_dedup_sec = 0.0;
    double expand_child_create_sec = 0.0;

    long long infer_calls = 0;
    long long expand_calls = 0;
    long long selection_steps = 0;
    long long forced_steps = 0;
    long long restore_missing_nodes_total = 0;
    long long expand_controller_actions_total = 0;
    long long expand_controller_alloc_pairs_total = 0;
    long long nodes_capacity_grows = 0;
};

static inline std::string f64(double v) {
    std::ostringstream oss;
    oss << std::setprecision(17) << v;
    return oss.str();
}

static std::string csv_escape(const std::string& s) {
    bool needs_quotes = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) return s;
    std::string out;
    out.reserve(s.size() + 8);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

class NativeIterCsvLogger {
public:
    explicit NativeIterCsvLogger(const std::string& path) {
        enabled_ = !path.empty();
        if (!enabled_) return;
        try {
            const std::filesystem::path p(path);
            std::error_code ec;
            const auto parent = p.parent_path();
            if (!parent.empty()) {
                std::filesystem::create_directories(parent, ec);
            }
            bool write_header = true;
            if (std::filesystem::exists(p, ec)) {
                write_header = std::filesystem::file_size(p, ec) == 0;
            }
            out_.open(path, std::ios::app);
            if (!out_) {
                enabled_ = false;
                return;
            }
            if (write_header) {
                out_ << "game_id,root_id,sim_iteration,root_depth,root_node_id,root_player,phase,node_depth,parent_node_id,node_id,player_acted_to_create_this_node,player_to_act_in_this_node,action_index,action_repr,prior,model_prior_json,normalized_prior_json,reward,nn_called,num_valid_actions,unique_actions,nn_value_controller,objective_cost,sim_time,requests_in_system,requests_generated,requests_completed,slo_violations,avg_lateness,state_waiting_ids,state_completed_request_ids,adversary_requests,adversary_prefill_slos,adversary_prefill_deadlines_by_id,adversary_decode_slos,controller_token_budget,controller_selected_ids,controller_allocations,controller_prefill_allocations,controller_decode_allocations,controller_prefill_total,controller_decode_total,controller_heuristic,controller_strategy\n";
            }
        } catch (...) {
            enabled_ = false;
        }
    }

    bool enabled() const { return enabled_; }

    void write_row(const std::vector<std::string>& cols) {
        if (!enabled_) return;
        for (size_t i = 0; i < cols.size(); ++i) {
            if (i) out_ << ',';
            out_ << csv_escape(cols[i]);
        }
        out_ << '\n';
    }

private:
    bool enabled_ = false;
    std::ofstream out_;
};

static std::string json_int_list(const std::vector<int>& v) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ", ";
        oss << v[i];
    }
    oss << ']';
    return oss.str();
}

static std::string json_float_list(const std::vector<double>& v) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ", ";
        oss << f64(v[i]);
    }
    oss << ']';
    return oss.str();
}

static std::string json_float_list_from_f32(const std::vector<float>& v) {
    std::vector<double> tmp;
    tmp.reserve(v.size());
    for (float x : v) tmp.push_back((double)x);
    return json_float_list(tmp);
}

static std::string py_map_repr_alloc(const std::vector<AllocationEntry>& v) {
    std::ostringstream oss;
    oss << '{';
    bool first = true;
    for (const auto& e : v) {
        if (!first) oss << ", ";
        first = false;
        oss << e.request_id << ": " << e.tokens;
    }
    oss << '}';
    return oss.str();
}

static std::string json_alloc_map(const std::vector<AllocationEntry>& v) {
    std::ostringstream oss;
    oss << '{';
    bool first = true;
    for (const auto& e : v) {
        if (!first) oss << ", ";
        first = false;
        oss << '"' << e.request_id << '"' << ": " << e.tokens;
    }
    oss << '}';
    return oss.str();
}

static std::string controller_action_repr_native(const ControllerActionSpecNative& a) {
    std::ostringstream oss;
    oss << "ControllerAction(token_budget=" << a.token_budget
        << ", selected_request_ids=" << json_int_list(a.selected_request_ids)
        << ", token_allocations=" << py_map_repr_alloc(a.token_allocations)
        << ", prefill_allocations=" << py_map_repr_alloc(a.prefill_allocations)
        << ", decode_allocations=" << py_map_repr_alloc(a.decode_allocations)
        << ", heuristic=";
    if (a.heuristic.empty()) oss << "None";
    else oss << '\'' << a.heuristic << '\'';
    oss << ", strategy=";
    if (a.strategy.empty()) oss << "None";
    else oss << '\'' << a.strategy << '\'';
    oss << ", mapping=None)";
    return oss.str();
}

static std::string adversary_action_repr_native(const AdversaryActionSpecNative& a) {
    std::ostringstream oss;
    oss << "AdversaryAction(requests=[";
    for (size_t i = 0; i < a.requests.size(); ++i) {
        if (i) oss << ", ";
        const auto& r = a.requests[i];
        oss << "AdversaryRequestSpec(prefill_tokens=" << r.prefill_tokens
            << ", decode_tokens=" << r.decode_tokens
            << ", prefill_slo=" << f64(r.prefill_slo)
            << ", decode_slo=" << f64(r.decode_slo)
            << ")";
    }
    oss << "], stop_decode_ids=" << json_int_list(a.stop_decode_ids) << ")";
    return oss.str();
}

static std::vector<int> sorted_ids(const std::unordered_set<int>& s) {
    std::vector<int> v;
    v.reserve(s.size());
    for (int x : s) v.push_back(x);
    std::sort(v.begin(), v.end());
    return v;
}

static std::string adversary_prefill_deadlines_json(
    const NativeSimState& state,
    const AdversaryActionSpecNative& a
) {
    if (a.requests.empty()) return "{}";
    std::vector<int> ids;
    ids.reserve(state.requests.size());
    for (const auto& r : state.requests) ids.push_back(r.request_id);
    if (ids.empty()) return "{}";
    std::sort(ids.begin(), ids.end());
    int k = (int)a.requests.size();
    if (k <= 0) return "{}";
    if (k > (int)ids.size()) k = (int)ids.size();
    const int start = (int)ids.size() - k;

    std::ostringstream oss;
    oss << '{';
    bool first = true;
    for (int i = start; i < (int)ids.size(); ++i) {
        const int rid = ids[(size_t)i];
        for (const auto& req : state.requests) {
            if (req.request_id != rid) continue;
            if (!first) oss << ", ";
            first = false;
            oss << '"' << rid << '"' << ": " << f64(req.queued_at + req.prefill_slo);
            break;
        }
    }
    oss << '}';
    return oss.str();
}

static inline void ensure_node_index(
    const std::vector<NativeTreeNode>& nodes,
    int idx,
    const char* where
);

static std::string iter_phase_label(bool parent_multi, int num_valid_actions) {
    if (num_valid_actions <= 0) return "terminal";
    if (num_valid_actions == 1) {
        return parent_multi ? "single-child" : "trivial-single-child";
    }
    return parent_multi ? "multiple-child" : "trivial-multiple-child";
}

static void write_native_iter_row(
    NativeIterCsvLogger& iter_logger,
    const std::vector<NativeTreeNode>& nodes,
    int node_idx,
    const NativeSimState& state,
    int game_id,
    int root_id,
    int sim_iteration,
    const NativeSearchCfg& cfg,
    const std::string& root_player,
    const std::string& phase,
    bool nn_called,
    int num_valid_actions,
    int unique_actions
);

static inline void ensure_node_capacity(
    std::vector<NativeTreeNode>& nodes,
    size_t additional_nodes,
    NativeSearchPerf* perf = nullptr
) {
    const size_t needed = nodes.size() + additional_nodes;
    if (nodes.capacity() >= needed) {
        return;
    }
    size_t grown = std::max(needed, nodes.capacity() > 0 ? nodes.capacity() * 2 : size_t(64));
    if (grown < 64) {
        grown = 64;
    }
    nodes.reserve(grown);
    if (perf != nullptr) {
        perf->nodes_capacity_grows += 1;
    }
}

struct NativeSearchCfg {
    int max_branching = 10;
    double controller_min_prior_threshold = 0.0;
    double adversary_min_prior_threshold = 0.0;
    bool root_dirichlet_noise_enabled = false;
    double root_dirichlet_alpha = 0.3;
    double root_dirichlet_epsilon = 0.25;
    double pb_c_base = 5000.0;
    double pb_c_init = 0.75;
    double discount_factor = 0.98;
    double prefill_step_time = 0.0388862329;
    double reward_knee = 25.0;
    double reward_max_penalty = 40.0;
    double reward_tail_alpha = 1.0 / 15.0;
    int seed = 0;
    int root_node_id = 0;
    int root_depth = 0;
};

struct NativeActionMask {
    py::list actions;
    std::vector<int> mask;
    std::vector<int> valid;
};

struct NativeTreeNode {
    std::string player;
    int node_id = 0;
    int depth = 0;
    int parent = -1;
    int parent_action_index = -1;
    py::object parent_action = py::none();
    bool parent_action_is_controller = false;
    ControllerActionSpecNative parent_controller_action;
    AdversaryActionSpecNative parent_adversary_action;
    double prior = 0.0;
    double reward = 0.0;
    int visits = 0;
    double value_sum = 0.0;
    double state_cost = 0.0;
    double sim_time = 0.0;

    bool has_nn_value = false;
    double nn_value_controller = 0.0;
    std::vector<double> nn_priors;
    std::vector<double> nn_priors_after_threshold;
    std::vector<int> nn_valid_mask;
    int num_valid_actions = 0;

    std::map<int, int> children;  // action index -> node index
    std::map<int, int> action_alias_to_canonical;
    std::map<int, std::vector<int>> canonical_to_action_aliases;

    bool has_state_snapshot = false;
    NativeSimState state_snapshot;
};

static inline void ensure_node_index(
    const std::vector<NativeTreeNode>& nodes,
    int idx,
    const char* where
) {
    if (idx < 0 || (size_t)idx >= nodes.size()) {
        std::ostringstream oss;
        oss
            << "invalid node index at " << where
            << " idx=" << idx
            << " node_count=" << nodes.size();
        throw std::runtime_error(oss.str());
    }
}

static void write_native_iter_row(
    NativeIterCsvLogger& iter_logger,
    const std::vector<NativeTreeNode>& nodes,
    int node_idx,
    const NativeSimState& state,
    int game_id,
    int root_id,
    int sim_iteration,
    const NativeSearchCfg& cfg,
    const std::string& root_player,
    const std::string& phase,
    bool nn_called,
    int num_valid_actions,
    int unique_actions
) {
    if (!iter_logger.enabled()) return;
    ensure_node_index(nodes, node_idx, "write_native_iter_row.node");
    const auto& node = nodes[(size_t)node_idx];
    const int parent_idx = node.parent;

    std::string parent_node_id_s;
    std::string action_index_s;
    std::string player_acted_to_create = "root_no_parent";
    std::string action_repr;
    std::string adversary_requests = "[]";
    std::string adversary_prefill_slos = "[]";
    std::string adversary_prefill_deadlines_by_id = "{}";
    std::string adversary_decode_slos = "[]";
    std::string controller_token_budget;
    std::string controller_selected_ids = "[]";
    std::string controller_allocations = "{}";
    std::string controller_prefill_allocations = "{}";
    std::string controller_decode_allocations = "{}";
    std::string controller_prefill_total = "0";
    std::string controller_decode_total = "0";
    std::string controller_heuristic;
    std::string controller_strategy;

    if (parent_idx >= 0) {
        ensure_node_index(nodes, parent_idx, "write_native_iter_row.parent");
        const auto& parent = nodes[(size_t)parent_idx];
        parent_node_id_s = std::to_string(parent.node_id);
        action_index_s = std::to_string(node.parent_action_index);
        player_acted_to_create = parent.player;

        if (node.parent_action_is_controller) {
            const auto& a = node.parent_controller_action;
            action_repr = controller_action_repr_native(a);
            controller_token_budget = std::to_string(a.token_budget);
            controller_selected_ids = json_int_list(a.selected_request_ids);
            controller_allocations = json_alloc_map(a.token_allocations);
            controller_prefill_allocations = json_alloc_map(a.prefill_allocations);
            controller_decode_allocations = json_alloc_map(a.decode_allocations);
            int prefill_total = 0;
            int decode_total = 0;
            for (const auto& e : a.prefill_allocations) prefill_total += e.tokens;
            for (const auto& e : a.decode_allocations) decode_total += e.tokens;
            controller_prefill_total = std::to_string(prefill_total);
            controller_decode_total = std::to_string(decode_total);
            controller_heuristic = a.heuristic;
            controller_strategy = a.strategy;
        } else {
            const auto& a = node.parent_adversary_action;
            action_repr = adversary_action_repr_native(a);
            {
                std::ostringstream reqs, pre_slos, dec_slos;
                reqs << '[';
                pre_slos << '[';
                dec_slos << '[';
                for (size_t i = 0; i < a.requests.size(); ++i) {
                    if (i) {
                        reqs << ", ";
                        pre_slos << ", ";
                        dec_slos << ", ";
                    }
                    const auto& r = a.requests[i];
                    reqs << "{\"prefill_tokens\":" << r.prefill_tokens
                         << ",\"decode_tokens\":" << r.decode_tokens
                         << ",\"prefill_slo\":" << f64(r.prefill_slo)
                         << ",\"decode_slo\":" << f64(r.decode_slo) << "}";
                    pre_slos << f64(r.prefill_slo);
                    dec_slos << f64(r.decode_slo);
                }
                reqs << ']';
                pre_slos << ']';
                dec_slos << ']';
                adversary_requests = reqs.str();
                adversary_prefill_slos = pre_slos.str();
                adversary_decode_slos = dec_slos.str();
            }
            adversary_prefill_deadlines_by_id = adversary_prefill_deadlines_json(state, a);
        }
    }

    std::string model_prior_json = "[]";
    std::string normalized_prior_json = "[]";
    std::string nn_value_controller_s;
    if (nn_called) {
        model_prior_json = json_float_list(node.nn_priors);
        normalized_prior_json = json_float_list(node.nn_priors_after_threshold);
        if (node.has_nn_value) {
            nn_value_controller_s = f64(node.nn_value_controller);
        }
    }

    const std::vector<int> waiting_ids = sorted_ids(state.stats.active_request_ids);
    const std::vector<int> completed_ids = sorted_ids(state.stats.completed_request_ids);

    std::vector<std::string> row;
    row.reserve(44);
    row.push_back(std::to_string(game_id));
    row.push_back(std::to_string(root_id));
    row.push_back(std::to_string(sim_iteration));
    row.push_back(std::to_string(cfg.root_depth));
    row.push_back(std::to_string(cfg.root_node_id));
    row.push_back(root_player);
    row.push_back(phase);
    row.push_back(std::to_string(node.depth));
    row.push_back(parent_node_id_s);
    row.push_back(std::to_string(node.node_id));
    row.push_back(player_acted_to_create);
    row.push_back(node.player);
    row.push_back(action_index_s);
    row.push_back(action_repr);
    row.push_back(f64(node.prior));
    row.push_back(model_prior_json);
    row.push_back(normalized_prior_json);
    row.push_back(f64(node.reward));
    row.push_back(nn_called ? "True" : "False");
    row.push_back(std::to_string(num_valid_actions));
    row.push_back(std::to_string(unique_actions));
    row.push_back(nn_value_controller_s);
    row.push_back(f64(node.state_cost));
    row.push_back(f64(state.sim_time));
    row.push_back(std::to_string((int)state.stats.active_request_ids.size()));
    row.push_back(std::to_string(state.stats.requests_generated));
    row.push_back(std::to_string(state.stats.requests_completed));
    row.push_back(std::to_string(state.stats.slo_violations));
    row.push_back(f64(state.stats.slo_lateness_sum));
    row.push_back(json_int_list(waiting_ids));
    row.push_back(json_int_list(completed_ids));
    row.push_back(adversary_requests);
    row.push_back(adversary_prefill_slos);
    row.push_back(adversary_prefill_deadlines_by_id);
    row.push_back(adversary_decode_slos);
    row.push_back(controller_token_budget);
    row.push_back(controller_selected_ids);
    row.push_back(controller_allocations);
    row.push_back(controller_prefill_allocations);
    row.push_back(controller_decode_allocations);
    row.push_back(controller_prefill_total);
    row.push_back(controller_decode_total);
    row.push_back(controller_heuristic);
    row.push_back(controller_strategy);

    iter_logger.write_row(row);
}

static std::string next_player(const std::string& player) {
    return (player == "adversary") ? "controller" : "adversary";
}

static py::list to_pybool_list(const std::vector<int>& mask) {
    py::list out;
    for (int x : mask) out.append(py::bool_(x != 0));
    return out;
}

static std::vector<int> mask_to_vec(const py::object& obj, size_t n_expected) {
    py::list mask_list;
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        mask_list = obj.cast<py::list>();
    } else if (py::hasattr(obj, "tolist")) {
        mask_list = obj.attr("tolist")().cast<py::list>();
    } else {
        throw std::runtime_error("mask must be list/tuple/tensor-like");
    }
    std::vector<int> out;
    out.reserve(n_expected > 0 ? n_expected : (size_t)py::len(mask_list));
    const size_t n = std::max(n_expected, (size_t)py::len(mask_list));
    for (size_t i = 0; i < n; ++i) {
        bool v = false;
        if (i < (size_t)py::len(mask_list)) {
            v = py::cast<bool>(mask_list[py::int_(i)]);
        }
        out.push_back(v ? 1 : 0);
    }
    return out;
}

static NativeActionMask actions_and_mask(
    const py::object& env,
    const py::object& state,
    const std::string& player,
    int max_branching
) {
    py::object out_obj;
    if (player == "controller") {
        out_obj = env.attr("sample_controller_actions")(state, py::int_(max_branching));
    } else {
        out_obj = env.attr("sample_adversary_actions")(state, py::int_(max_branching));
    }
    py::tuple tup = out_obj.cast<py::tuple>();
    py::list actions = tup[0].cast<py::list>();
    std::vector<int> mask = mask_to_vec(tup[1].cast<py::object>(), (size_t)py::len(actions));

    std::vector<int> valid;
    const size_t n = (size_t)py::len(actions);
    valid.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (i < mask.size() && mask[i] != 0 && !actions[py::int_(i)].is_none()) {
            valid.push_back((int)i);
        }
    }
    return NativeActionMask{actions, std::move(mask), std::move(valid)};
}

static double evaluate_state_cost(const py::object& env, const py::object& state) {
    py::tuple t = env.attr("evaluate_objective")(state).cast<py::tuple>();
    const double v = py::cast<double>(t[0]);
    const double l = py::cast<double>(t[1]);
    return v + l;
}

static double transition_reward(const NativeSearchCfg& cfg, double parent_cost, double child_cost) {
    const double delta = std::max(0.0, child_cost - parent_cost);
    const double knee = cfg.reward_knee;
    const double max_penalty = cfg.reward_max_penalty;
    if (max_penalty <= knee) {
        return -std::min(delta, max_penalty);
    }
    const double headroom = max_penalty - knee;
    const double alpha = cfg.reward_tail_alpha;
    double penalty = 0.0;
    if (delta <= knee) {
        penalty = delta;
    } else {
        penalty = knee + headroom * std::tanh(alpha * (delta - knee));
    }
    return -penalty;
}

static double time_discount(const NativeSearchCfg& cfg, double t_child, double t_parent) {
    const double gamma = cfg.discount_factor;
    const double denom = std::max(cfg.prefill_step_time, 1e-9);
    const double dt = std::max(0.0, t_child - t_parent);
    return std::pow(gamma, dt / denom);
}

static std::map<int, double> apply_min_prior_threshold(
    const std::map<int, double>& prior_by_idx,
    double min_prior,
    double eps = 1e-12,
    int max_iters = 64
) {
    if (prior_by_idx.empty()) return {};
    std::vector<int> keys;
    keys.reserve(prior_by_idx.size());
    for (const auto& kv : prior_by_idx) keys.push_back(kv.first);
    const int n = (int)keys.size();

    const double mp = (min_prior > 0.0 ? min_prior : 0.0);
    if (mp <= 0.0 || n <= 1) {
        double s = 0.0;
        for (int k : keys) s += std::max(0.0, prior_by_idx.at(k));
        if (s <= eps) {
            const double u = 1.0 / (double)n;
            std::map<int, double> out;
            for (int k : keys) out[k] = u;
            return out;
        }
        std::map<int, double> out;
        for (int k : keys) out[k] = std::max(0.0, prior_by_idx.at(k)) / s;
        return out;
    }

    if (mp * (double)n >= 1.0 - eps) {
        const double u = 1.0 / (double)n;
        std::map<int, double> out;
        for (int k : keys) out[k] = u;
        return out;
    }

    std::map<int, double> p0;
    double s0 = 0.0;
    for (int k : keys) {
        const double v = std::max(0.0, prior_by_idx.at(k));
        p0[k] = v;
        s0 += v;
    }
    if (s0 <= eps) {
        const double u = 1.0 / (double)n;
        for (int k : keys) p0[k] = u;
    } else {
        for (int k : keys) p0[k] /= s0;
    }

    std::map<int, double> q;
    for (int k : keys) q[k] = (p0[k] >= mp ? p0[k] : mp);

    for (int it = 0; it < max_iters; ++it) {
        double total = 0.0;
        for (int k : keys) total += q[k];
        const double over = total - 1.0;
        if (std::fabs(over) <= 1e-10) break;

        if (over > 0.0) {
            std::vector<int> adjustable;
            adjustable.reserve(keys.size());
            for (int k : keys) {
                if (q[k] > mp + eps) adjustable.push_back(k);
            }
            if (adjustable.empty()) {
                const double u = 1.0 / (double)n;
                std::map<int, double> out;
                for (int k : keys) out[k] = u;
                return out;
            }
            double wsum = 0.0;
            for (int k : adjustable) wsum += p0[k];
            if (wsum <= eps) {
                const double u = 1.0 / (double)n;
                std::map<int, double> out;
                for (int k : keys) out[k] = u;
                return out;
            }
            for (int k : adjustable) q[k] -= over * (p0[k] / wsum);
            for (int k : keys) if (q[k] < mp) q[k] = mp;
        } else {
            const double under = -over;
            double wsum = 0.0;
            for (int k : keys) wsum += p0[k];
            for (int k : keys) q[k] += under * (p0[k] / wsum);
        }
    }

    double total = 0.0;
    for (int k : keys) total += q[k];
    if (std::fabs(total - 1.0) > 1e-8) {
        int kmax = keys[0];
        for (int k : keys) if (q[k] > q[kmax]) kmax = k;
        q[kmax] = std::max(mp, q[kmax] + (1.0 - total));
    }
    return q;
}

static std::string controller_action_key(const py::object& action) {
    std::vector<std::pair<int, int>> pairs;
    try {
        py::object alloc_obj = action.attr("token_allocations");
        if (!alloc_obj.is_none()) {
            py::dict d = alloc_obj.cast<py::dict>();
            for (const auto& item : d) {
                pairs.emplace_back(py::cast<int>(item.first), py::cast<int>(item.second));
            }
        }
    } catch (...) {
    }
    std::sort(pairs.begin(), pairs.end());
    std::string out;
    out.reserve(pairs.size() * 16);
    for (const auto& p : pairs) {
        out += std::to_string(p.first);
        out.push_back(':');
        out += std::to_string(p.second);
        out.push_back(';');
    }
    return out;
}

static int select_child(
    const std::vector<NativeTreeNode>& nodes,
    int node_idx,
    const NativeSearchCfg& cfg,
    const NativeMinMaxStats& minmax,
    std::mt19937& rng
) {
    ensure_node_index(nodes, node_idx, "select_child(node)");
    const auto& node = nodes[(size_t)node_idx];
    if (node.children.empty()) return -1;
    if (node.children.size() == 1) return node.children.begin()->second;

    const bool parent_is_branching = (node.num_valid_actions > 1) || (node.children.size() > 1);
    double best = -std::numeric_limits<double>::infinity();
    std::vector<int> best_children;

    for (const auto& kv : node.children) {
        if (kv.second < 0 || (size_t)kv.second >= nodes.size()) {
            continue;
        }
        const auto& child = nodes[(size_t)kv.second];
        double pb_c = std::log((node.visits + cfg.pb_c_base + 1.0) / cfg.pb_c_base) + cfg.pb_c_init;
        pb_c *= std::sqrt((double)node.visits + 1.0) / ((double)child.visits + 1.0);
        double prior_score = pb_c * child.prior;

        double value_score = 0.0;
        if (child.visits > 0) {
            const double disc = parent_is_branching ? time_discount(cfg, child.sim_time, node.sim_time) : 1.0;
            const double q_controller = child.reward + disc * (child.value_sum / (double)child.visits);
            const double q_norm = minmax.normalize(q_controller);
            value_score = (node.player == "controller") ? q_norm : -q_norm;
        }
        const double u = prior_score + value_score;
        if (u > best + 1e-15) {
            best = u;
            best_children.clear();
            best_children.push_back(kv.second);
        } else if (std::fabs(u - best) <= 1e-15) {
            best_children.push_back(kv.second);
        }
    }

    if (best_children.empty()) return node.children.begin()->second;
    std::uniform_int_distribution<int> dist(0, (int)best_children.size() - 1);
    return best_children[(size_t)dist(rng)];
}

static int create_child(
    std::vector<NativeTreeNode>& nodes,
    int parent_idx,
    int action_index,
    const py::object& action_obj,
    double prior,
    int& next_node_id
) {
    NativeTreeNode child;
    const auto& parent = nodes[(size_t)parent_idx];
    child.player = next_player(parent.player);
    child.node_id = next_node_id++;
    child.depth = parent.depth + 1;
    child.parent = parent_idx;
    child.parent_action_index = action_index;
    child.parent_action = action_obj;
    child.prior = prior;
    nodes.push_back(std::move(child));
    int idx = (int)nodes.size() - 1;
    nodes[(size_t)parent_idx].children[action_index] = idx;
    return idx;
}

static void apply_edge_transition(
    std::vector<NativeTreeNode>& nodes,
    int parent_idx,
    int child_idx,
    py::object& state,
    const py::object& env,
    const NativeSearchCfg& cfg
) {
    auto& parent = nodes[(size_t)parent_idx];
    auto& child = nodes[(size_t)child_idx];
    const double parent_cost = parent.state_cost;
    if (parent.player == "adversary") {
        state = env.attr("apply_adversary_action_only")(
            state, child.parent_action, py::arg("inplace") = true
        );
    } else {
        state = env.attr("apply_controller_action_only")(
            state, child.parent_action, py::arg("inplace") = true
        );
    }
    const double child_cost = evaluate_state_cost(env, state);
    child.reward = transition_reward(cfg, parent_cost, child_cost);
    child.state_cost = child_cost;
    child.sim_time = py::cast<double>(state.attr("simulator").attr("_time"));
}

static std::tuple<double, bool, int> expand_node(
    std::vector<NativeTreeNode>& nodes,
    int node_idx,
    py::object& state,
    const py::object& env,
    const py::function& infer_cb,
    const NativeSearchCfg& cfg,
    std::mt19937& rng,
    int& next_node_id
) {
    auto am = actions_and_mask(env, state, nodes[(size_t)node_idx].player, cfg.max_branching);
    nodes[(size_t)node_idx].num_valid_actions = (int)am.valid.size();
    nodes[(size_t)node_idx].nn_valid_mask = am.mask;

    if (am.valid.empty()) {
        return std::make_tuple(0.0, false, 0);
    }

    if (am.valid.size() == 1) {
        const int idx = am.valid[0];
        if (nodes[(size_t)node_idx].children.find(idx) == nodes[(size_t)node_idx].children.end()) {
            create_child(
                nodes,
                node_idx,
                idx,
                am.actions[py::int_(idx)],
                1.0,
                next_node_id
            );
        }
        return std::make_tuple(0.0, false, 1);
    }

    py::tuple infer_out = infer_cb(
        state,
        py::str(nodes[(size_t)node_idx].player),
        to_pybool_list(am.mask)
    ).cast<py::tuple>();
    const double model_value = py::cast<double>(infer_out[0]);
    std::vector<double> priors = infer_out[1].cast<std::vector<double>>();
    const size_t a = (size_t)py::len(am.actions);
    if (priors.size() < a) priors.resize(a, 0.0);

    nodes[(size_t)node_idx].has_nn_value = true;
    nodes[(size_t)node_idx].nn_value_controller = model_value;
    nodes[(size_t)node_idx].nn_priors = priors;

    const bool is_controller_player = (nodes[(size_t)node_idx].player == "controller");
    const double min_p = is_controller_player
        ? cfg.controller_min_prior_threshold
        : cfg.adversary_min_prior_threshold;

    const double t_thr0 = now_sec();
    std::map<int, double> valid_prior_by_idx;
    for (int i : am.valid) {
        valid_prior_by_idx[i] = (i >= 0 && (size_t)i < priors.size()) ? priors[(size_t)i] : 0.0;
    }
    std::map<int, double> norm_prior = apply_min_prior_threshold(valid_prior_by_idx, min_p);

    nodes[(size_t)node_idx].nn_priors_after_threshold.assign(priors.size(), 0.0);
    for (const auto& kv : norm_prior) {
        if (kv.first >= 0 && (size_t)kv.first < nodes[(size_t)node_idx].nn_priors_after_threshold.size()) {
            nodes[(size_t)node_idx].nn_priors_after_threshold[(size_t)kv.first] = kv.second;
        }
    }

    nodes[(size_t)node_idx].action_alias_to_canonical.clear();
    nodes[(size_t)node_idx].canonical_to_action_aliases.clear();

    std::vector<int> canonical_indices;
    std::map<int, double> canonical_prior;

    if (is_controller_player) {
        std::map<std::string, int> sig_to_canon;
        for (int idx : am.valid) {
            py::object action_obj = am.actions[py::int_(idx)];
            if (action_obj.is_none()) continue;
            const std::string sig = controller_action_key(action_obj);
            auto it = sig_to_canon.find(sig);
            int canon = idx;
            if (it == sig_to_canon.end()) {
                sig_to_canon[sig] = canon;
                nodes[(size_t)node_idx].canonical_to_action_aliases[canon] = {idx};
            } else {
                canon = it->second;
                nodes[(size_t)node_idx].canonical_to_action_aliases[canon].push_back(idx);
            }
            nodes[(size_t)node_idx].action_alias_to_canonical[idx] = canon;
        }
        for (const auto& kv : nodes[(size_t)node_idx].canonical_to_action_aliases) {
            canonical_indices.push_back(kv.first);
            double psum = 0.0;
            for (int aidx : kv.second) {
                auto itp = norm_prior.find(aidx);
                if (itp != norm_prior.end()) psum += itp->second;
            }
            canonical_prior[kv.first] = psum;
        }
    } else {
        canonical_indices = am.valid;
        for (int idx : canonical_indices) {
            auto itp = norm_prior.find(idx);
            canonical_prior[idx] = (itp != norm_prior.end()) ? itp->second : 0.0;
        }
    }

    ensure_node_capacity(nodes, canonical_indices.size() + 4, nullptr);

    for (int idx : canonical_indices) {
        if (nodes[(size_t)node_idx].children.find(idx) != nodes[(size_t)node_idx].children.end()) continue;
        py::object action_obj = am.actions[py::int_(idx)];
        if (action_obj.is_none()) continue;
        create_child(
            nodes,
            node_idx,
            idx,
            action_obj,
            canonical_prior[idx],
            next_node_id
        );
    }

    return std::make_tuple(model_value, true, (int)am.valid.size());
}

static void maybe_add_root_dirichlet_noise(
    std::vector<NativeTreeNode>& nodes,
    int root_idx,
    bool nn_called,
    int num_valid_actions,
    const NativeSearchCfg& cfg,
    std::mt19937& rng
) {
    auto& root = nodes[(size_t)root_idx];
    if (!cfg.root_dirichlet_noise_enabled) return;
    if (!nn_called) return;
    if (num_valid_actions <= 1) return;
    if (root.children.size() <= 1) return;
    if (cfg.root_dirichlet_alpha <= 0.0 || cfg.root_dirichlet_epsilon <= 0.0) return;

    double eps = cfg.root_dirichlet_epsilon;
    if (eps < 0.0) eps = 0.0;
    if (eps > 1.0) eps = 1.0;
    std::gamma_distribution<double> gamma(cfg.root_dirichlet_alpha, 1.0);

    std::vector<int> child_ids;
    child_ids.reserve(root.children.size());
    for (const auto& kv : root.children) child_ids.push_back(kv.second);

    std::vector<double> noise(child_ids.size(), 0.0);
    double s = 0.0;
    for (size_t i = 0; i < child_ids.size(); ++i) {
        noise[i] = gamma(rng);
        s += noise[i];
    }
    if (s <= 1e-12) {
        for (double& x : noise) x = 1.0 / (double)noise.size();
    } else {
        for (double& x : noise) x /= s;
    }

    std::vector<double> mixed(child_ids.size(), 0.0);
    double z = 0.0;
    for (size_t i = 0; i < child_ids.size(); ++i) {
        auto& child = nodes[(size_t)child_ids[i]];
        mixed[i] = (1.0 - eps) * std::max(0.0, child.prior) + eps * noise[i];
        z += mixed[i];
    }
    if (z <= 1e-12) {
        const double u = 1.0 / (double)child_ids.size();
        for (int ci : child_ids) nodes[(size_t)ci].prior = u;
    } else {
        for (size_t i = 0; i < child_ids.size(); ++i) {
            nodes[(size_t)child_ids[i]].prior = mixed[i] / z;
        }
    }

    if (!root.nn_priors_after_threshold.empty()) {
        std::vector<double> noisy_full(root.nn_priors_after_threshold.size(), 0.0);
        if (!root.canonical_to_action_aliases.empty()) {
            for (const auto& kv : root.canonical_to_action_aliases) {
                const int canon = kv.first;
                auto it_child = root.children.find(canon);
                if (it_child == root.children.end()) continue;
                const auto& child = nodes[(size_t)it_child->second];
                const auto& aliases = kv.second;
                if (aliases.empty()) continue;
                const double share = child.prior / (double)aliases.size();
                for (int aidx : aliases) {
                    if (aidx >= 0 && (size_t)aidx < noisy_full.size()) noisy_full[(size_t)aidx] = share;
                }
            }
        } else {
            for (const auto& kv : root.children) {
                const int idx = kv.first;
                if (idx >= 0 && (size_t)idx < noisy_full.size()) {
                    noisy_full[(size_t)idx] = nodes[(size_t)kv.second].prior;
                }
            }
        }
        root.nn_priors_after_threshold = noisy_full;
    }
}

static py::dict search_mcts_dnn(
    py::object env,
    py::object root_state,
    std::string root_player,
    int iterations,
    py::function infer_cb,
    int max_branching,
    double controller_min_prior_threshold,
    double adversary_min_prior_threshold,
    bool root_dirichlet_noise_enabled,
    double root_dirichlet_alpha,
    double root_dirichlet_epsilon,
    double pb_c_base,
    double pb_c_init,
    double discount_factor,
    double prefill_step_time,
    double reward_knee,
    double reward_max_penalty,
    double reward_tail_alpha,
    int seed,
    int root_node_id,
    int root_depth,
    int game_id,
    int root_id,
    std::string iter_log_path,
    bool iter_complete_log
) {
    (void)game_id;
    (void)root_id;
    (void)iter_log_path;
    (void)iter_complete_log;
    NativeSearchCfg cfg;
    cfg.max_branching = max_branching;
    cfg.controller_min_prior_threshold = controller_min_prior_threshold;
    cfg.adversary_min_prior_threshold = adversary_min_prior_threshold;
    cfg.root_dirichlet_noise_enabled = root_dirichlet_noise_enabled;
    cfg.root_dirichlet_alpha = root_dirichlet_alpha;
    cfg.root_dirichlet_epsilon = root_dirichlet_epsilon;
    cfg.pb_c_base = pb_c_base;
    cfg.pb_c_init = pb_c_init;
    cfg.discount_factor = discount_factor;
    cfg.prefill_step_time = prefill_step_time;
    cfg.reward_knee = reward_knee;
    cfg.reward_max_penalty = reward_max_penalty;
    cfg.reward_tail_alpha = reward_tail_alpha;
    cfg.seed = seed;
    cfg.root_node_id = root_node_id;
    cfg.root_depth = root_depth;

    std::mt19937 rng((uint32_t)cfg.seed);
    NativeMinMaxStats minmax;

    std::vector<NativeTreeNode> nodes;
    {
        const size_t expected_nodes =
            (size_t)std::max(8, iterations + 8) * (size_t)std::max(2, max_branching);
        nodes.reserve(expected_nodes);
    }

    NativeTreeNode root;
    root.player = root_player;
    root.node_id = cfg.root_node_id;
    root.depth = cfg.root_depth;
    root.parent = -1;
    root.parent_action = py::none();
    root.state_cost = evaluate_state_cost(env, root_state);
    root.sim_time = py::cast<double>(root_state.attr("simulator").attr("_time"));
    nodes.push_back(std::move(root));

    int next_node_id = cfg.root_node_id + 1;
    py::object root_work;
    try {
        root_work = root_state.attr("fork")(py::arg("flag") = false);
    } catch (...) {
        root_work = root_state.attr("fork")();
    }
    auto root_expand = expand_node(nodes, 0, root_work, env, infer_cb, cfg, rng, next_node_id);
    maybe_add_root_dirichlet_noise(
        nodes,
        0,
        std::get<1>(root_expand),
        std::get<2>(root_expand),
        cfg,
        rng
    );

    for (int it = 0; it < iterations; ++it) {
        py::object state;
        try {
            state = root_state.attr("fork")(py::arg("flag") = false);
        } catch (...) {
            state = root_state.attr("fork")();
        }

        int current = 0;
        std::vector<int> search_path;
        search_path.reserve(32);
        search_path.push_back(current);

        while (!nodes[(size_t)current].children.empty()) {
            int child_idx = select_child(nodes, current, cfg, minmax, rng);
            if (child_idx < 0) break;
            apply_edge_transition(nodes, current, child_idx, state, env, cfg);
            current = child_idx;
            search_path.push_back(current);
        }

        while (true) {
            auto am = actions_and_mask(env, state, nodes[(size_t)current].player, cfg.max_branching);
            nodes[(size_t)current].num_valid_actions = (int)am.valid.size();
            if (am.valid.size() != 1) break;

            const int only_idx = am.valid[0];
            auto it_child = nodes[(size_t)current].children.find(only_idx);
            int child_idx = -1;
            if (it_child == nodes[(size_t)current].children.end()) {
                child_idx = create_child(
                    nodes,
                    current,
                    only_idx,
                    am.actions[py::int_(only_idx)],
                    1.0,
                    next_node_id
                );
            } else {
                child_idx = it_child->second;
            }
            apply_edge_transition(nodes, current, child_idx, state, env, cfg);
            current = child_idx;
            search_path.push_back(current);
        }

        auto leaf_expand = expand_node(nodes, current, state, env, infer_cb, cfg, rng, next_node_id);
        double value = std::get<0>(leaf_expand);

        for (int i = (int)search_path.size() - 1; i >= 0; --i) {
            const int node_idx = search_path[(size_t)i];
            auto& node = nodes[(size_t)node_idx];
            node.value_sum += value;
            node.visits += 1;

            if (node.parent < 0) continue;
            auto& parent = nodes[(size_t)node.parent];
            const bool parent_is_branching =
                (parent.num_valid_actions > 1) || (parent.children.size() > 1);
            const double disc =
                parent_is_branching ? time_discount(cfg, node.sim_time, parent.sim_time) : 1.0;
            const double reward_used = parent_is_branching ? node.reward : 0.0;

            if (parent_is_branching && node.visits > 0) {
                const double q = reward_used + disc * (node.value_sum / (double)node.visits);
                minmax.update(q);
            }
            value = reward_used + disc * value;
        }
    }

    const auto& root_final = nodes[0];
    py::dict out;
    out["root_node_id"] = py::int_(root_final.node_id);
    out["root_depth"] = py::int_(root_final.depth);
    out["root_player"] = py::str(root_final.player);
    out["root_visits"] = py::int_(root_final.visits);
    out["root_value_sum"] = py::float_(root_final.value_sum);
    out["root_state_cost"] = py::float_(root_final.state_cost);
    out["root_sim_time"] = py::float_(root_final.sim_time);
    out["root_num_valid_actions"] = py::int_(root_final.num_valid_actions);
    out["root_nn_value_controller"] = root_final.has_nn_value
        ? py::object(py::float_(root_final.nn_value_controller))
        : py::object(py::none());
    out["root_nn_priors"] = py::cast(root_final.nn_priors);
    out["root_nn_priors_after_threshold"] = py::cast(root_final.nn_priors_after_threshold);
    out["root_nn_valid_mask"] = py::cast(root_final.nn_valid_mask);
    out["action_alias_to_canonical"] = py::cast(root_final.action_alias_to_canonical);
    out["canonical_to_action_aliases"] = py::cast(root_final.canonical_to_action_aliases);

    py::list children;
    for (const auto& kv : root_final.children) {
        const int action_idx = kv.first;
        const auto& child = nodes[(size_t)kv.second];
        py::dict c;
        c["index"] = py::int_(action_idx);
        c["node_id"] = py::int_(child.node_id);
        c["depth"] = py::int_(child.depth);
        c["player"] = py::str(child.player);
        c["prior"] = py::float_(child.prior);
        c["reward"] = py::float_(child.reward);
        c["visits"] = py::int_(child.visits);
        c["value_sum"] = py::float_(child.value_sum);
        c["state_cost"] = py::float_(child.state_cost);
        c["sim_time"] = py::float_(child.sim_time);
        c["num_valid_actions"] = py::int_(child.num_valid_actions);
        c["parent_action"] = child.parent_action;
        c["parent_action_index"] = py::int_(child.parent_action_index);
        children.append(c);
    }
    out["children"] = children;
    out["node_count"] = py::int_(nodes.size());
    return out;
}

struct NativeActionMaskFull {
    std::vector<ControllerActionSpecNative> controller_actions;
    std::vector<AdversaryActionSpecNative> adversary_actions;
    std::vector<int> mask;
    std::vector<int> valid;
};

static std::string controller_action_signature(const ControllerActionSpecNative& a) {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(a.token_allocations.size());
    for (const auto& e : a.token_allocations) pairs.emplace_back(e.request_id, e.tokens);
    std::sort(pairs.begin(), pairs.end());
    std::string out;
    out.reserve(pairs.size() * 16);
    for (const auto& p : pairs) {
        out += std::to_string(p.first);
        out.push_back(':');
        out += std::to_string(p.second);
        out.push_back(';');
    }
    return out;
}

static py::object controller_action_to_py(const ControllerActionSpecNative& a) {
    py::object ControllerAction = py::module_::import("vidur.mcts.environment").attr("ControllerAction");
    py::list selected;
    for (int rid : a.selected_request_ids) selected.append(py::int_(rid));
    py::object selected_obj = selected.size() ? py::object(selected) : py::none();
    py::object heur_obj = a.heuristic.empty()
        ? py::reinterpret_borrow<py::object>(py::none())
        : py::reinterpret_steal<py::object>(py::str(a.heuristic).release());
    py::object strat_obj = a.strategy.empty()
        ? py::reinterpret_borrow<py::object>(py::none())
        : py::reinterpret_steal<py::object>(py::str(a.strategy).release());
    return ControllerAction(
        py::arg("token_budget") = py::int_(a.token_budget),
        py::arg("selected_request_ids") = selected_obj,
        py::arg("token_allocations") = alloc_to_pydict(a.token_allocations),
        py::arg("prefill_allocations") = alloc_to_pydict(a.prefill_allocations),
        py::arg("decode_allocations") = alloc_to_pydict(a.decode_allocations),
        py::arg("heuristic") = heur_obj,
        py::arg("strategy") = strat_obj
    );
}

static py::object adversary_action_to_py(const AdversaryActionSpecNative& a) {
    py::module_ env_mod = py::module_::import("vidur.mcts.environment");
    py::object AdversaryRequestSpec = env_mod.attr("AdversaryRequestSpec");
    py::object AdversaryAction = env_mod.attr("AdversaryAction");
    py::list reqs;
    for (const auto& r : a.requests) {
        reqs.append(AdversaryRequestSpec(
            py::arg("prefill_tokens") = py::int_(r.prefill_tokens),
            py::arg("decode_tokens") = py::int_(r.decode_tokens),
            py::arg("prefill_slo") = py::float_(r.prefill_slo),
            py::arg("decode_slo") = py::float_(r.decode_slo)
        ));
    }
    py::list stop_ids;
    for (int rid : a.stop_decode_ids) stop_ids.append(py::int_(rid));
    return AdversaryAction(py::arg("requests") = reqs, py::arg("stop_decode_ids") = stop_ids);
}

static std::unordered_set<int> py_int_set_to_cpp(const py::object& obj) {
    std::unordered_set<int> out;
    if (obj.is_none()) return out;
    py::iterable it = obj.cast<py::iterable>();
    for (auto item : it) out.insert(py::cast<int>(item));
    return out;
}

static std::unordered_map<int, double> py_map_i_d_to_cpp(const py::object& obj) {
    std::unordered_map<int, double> out;
    if (obj.is_none()) return out;
    py::dict d = obj.cast<py::dict>();
    out.reserve((size_t)py::len(d));
    for (const auto& kv : d) out[py::cast<int>(kv.first)] = py::cast<double>(kv.second);
    return out;
}

static std::unordered_map<int, int> py_map_i_i_to_cpp(const py::object& obj) {
    std::unordered_map<int, int> out;
    if (obj.is_none()) return out;
    py::dict d = obj.cast<py::dict>();
    out.reserve((size_t)py::len(d));
    for (const auto& kv : d) out[py::cast<int>(kv.first)] = py::cast<int>(kv.second);
    return out;
}

static double py_get_f(const py::object& obj, const char* name, double dflt) {
    try {
        if (py::hasattr(obj, name)) {
            py::object v = obj.attr(name);
            if (!v.is_none()) return py::cast<double>(v);
        }
    } catch (...) {
    }
    return dflt;
}

static int py_get_i(const py::object& obj, const char* name, int dflt) {
    try {
        if (py::hasattr(obj, name)) {
            py::object v = obj.attr(name);
            if (!v.is_none()) return py::cast<int>(v);
        }
    } catch (...) {
    }
    return dflt;
}

static bool py_get_b(const py::object& obj, const char* name, bool dflt) {
    try {
        if (py::hasattr(obj, name)) {
            py::object v = obj.attr(name);
            if (!v.is_none()) return py::cast<bool>(v);
        }
    } catch (...) {
    }
    return dflt;
}

static NativeRuntimeConfig runtime_cfg_from_env(const py::object& env) {
    NativeRuntimeConfig cfg;
    try {
        py::object c = env.attr("_constraints");
        cfg.interval_request_size = py_get_i(c, "interval_request_size", 512);
        cfg.max_request_tokens = py_get_i(c, "max_request_tokens", 3072);
        cfg.prefill_slowdown = py_get_f(c, "prefill_slowdown", 3.0);
        if (py::hasattr(c, "request_slo_options")) {
            py::object slo = c.attr("request_slo_options");
            if (py::hasattr(slo, "decode_slos")) {
                py::object ds = slo.attr("decode_slos");
                if (!ds.is_none() && py::len(ds) > 0) {
                    cfg.default_decode_slo = py::cast<double>(py::list(ds)[0]) / 1000.0;
                }
            }
        }
    } catch (...) {
    }

    try {
        py::dict p = env.attr("_prefill_profile").attr("entries").cast<py::dict>();
        std::vector<std::pair<int, double>> entries;
        entries.reserve((size_t)py::len(p));
        for (auto kv : p) {
            entries.emplace_back(py::cast<int>(kv.first), py::cast<double>(kv.second));
        }
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        cfg.prefill_profile_tokens.reserve(entries.size());
        cfg.prefill_profile_times.reserve(entries.size());
        for (const auto& e : entries) {
            cfg.prefill_profile_tokens.push_back(e.first);
            cfg.prefill_profile_times.push_back(e.second);
        }
    } catch (...) {
    }

    if (cfg.prefill_profile_tokens.empty()) {
        cfg.prefill_profile_tokens = {cfg.max_request_tokens};
        cfg.prefill_profile_times = {0.001};
    }
    return cfg;
}

static NativeSimState native_state_from_py(const py::object& env, const py::object& root_state) {
    NativeRuntimeConfig rcfg = runtime_cfg_from_env(env);
    NativeSimState out;

    py::object sim = root_state.attr("simulator");
    py::object stats = root_state.attr("stats");
    out.sim_time = py_get_f(sim, "_time", 0.0);
    out.next_request_id = 0;

    out.stats.requests_generated = py_get_i(stats, "requests_generated", 0);
    out.stats.requests_completed = py_get_i(stats, "requests_completed", 0);
    out.stats.slo_violations = py_get_i(stats, "slo_violations", 0);
    out.stats.slo_lateness_sum = py_get_f(stats, "slo_lateness_sum", 0.0);
    out.stats.maximum_qps = py_get_i(stats, "maximum_qps", py_get_i(env.attr("_constraints"), "maximum_qps", 5));
    {
        py::object last = py::none();
        try { last = stats.attr("last_prefill_batch_time"); } catch (...) {}
        out.stats.last_prefill_batch_time = last.is_none() ? -1.0 : py::cast<double>(last);
    }

    try {
        py::iterable arr = stats.attr("recent_arrivals").cast<py::iterable>();
        for (auto t : arr) out.stats.recent_arrivals.push_back(py::cast<double>(t));
    } catch (...) {
    }

    try { out.stats.per_request_prefill_lateness = py_map_i_d_to_cpp(stats.attr("per_request_prefill_lateness")); } catch (...) {}
    try { out.stats.per_request_decode_lateness = py_map_i_d_to_cpp(stats.attr("per_request_decode_lateness")); } catch (...) {}
    try { out.stats.decode_tokens_counted = py_map_i_i_to_cpp(stats.attr("decode_tokens_counted")); } catch (...) {}
    try { out.stats.decode_next_deadline_by_id = py_map_i_d_to_cpp(stats.attr("decode_next_deadline_by_id")); } catch (...) {}
    try { out.stats.prefill_lateness_finalized = py_int_set_to_cpp(stats.attr("prefill_lateness_finalized")); } catch (...) {}
    try { out.stats.violated_request_ids = py_int_set_to_cpp(stats.attr("violated_request_ids")); } catch (...) {}
    try { out.stats.active_request_ids = py_int_set_to_cpp(stats.attr("active_request_ids")); } catch (...) {}
    try { out.stats.completed_request_ids = py_int_set_to_cpp(stats.attr("completed_request_ids")); } catch (...) {}

    py::object rs = sim.attr("_scheduler").attr("get_replica_scheduler")(sim.attr("replica_id"));
    py::dict req_map = rs.attr("_requests").cast<py::dict>();
    out.requests.reserve((size_t)py::len(req_map));
    int max_seen_id = -1;
    for (auto kv : req_map) {
        py::object req = py::reinterpret_borrow<py::object>(kv.second);
        NativeRequestState r;
        r.request_id = py_get_i(req, "id", -1);
        max_seen_id = std::max(max_seen_id, r.request_id);
        r.arrived_at = py_get_f(req, "_arrived_at", py_get_f(req, "arrived_at", 0.0));
        r.queued_at = py_get_f(req, "queued_at", r.arrived_at);
        r.num_prefill_tokens = py_get_i(req, "num_prefill_tokens", 0);
        r.num_processed_prefill_tokens = py_get_i(req, "num_processed_prefill_tokens", 0);
        r.num_decode_tokens = py_get_i(req, "_num_decode_tokens", py_get_i(req, "num_decode_tokens", 0));
        r.num_processed_decode_tokens = py_get_i(req, "num_processed_decode_tokens", 0);
        r.prefill_done = py_get_b(req, "_is_prefill_complete", py_get_b(req, "is_prefill_complete", false));
        r.completed = py_get_b(req, "completed", false);
        r.prefill_completed_at = py_get_f(req, "_prefill_completed_at", py_get_f(req, "prefill_completed_at", 0.0));
        r.prefill_slo = py_get_f(req, "_prefill_slo_time", py_get_f(req, "prefill_slo_time", 0.0));
        r.decode_slo = py_get_f(req, "_decode_slo_time", py_get_f(req, "decode_slo_time", rcfg.default_decode_slo));
        if (!r.completed) out.requests.push_back(r);
    }
    out.next_request_id = std::max(out.next_request_id, max_seen_id + 1);

    if (out.stats.active_request_ids.empty()) {
        for (const auto& r : out.requests) {
            if (!r.completed && ((r.prefill_done && (r.num_decode_tokens > r.num_processed_decode_tokens)) ||
                                 (!r.prefill_done && (r.num_prefill_tokens > r.num_processed_prefill_tokens)))) {
                out.stats.active_request_ids.insert(r.request_id);
            }
        }
    }
    return out;
}

template <typename InferRuntimeT>
static std::tuple<double, std::vector<double>> infer_from_native_state_torchscript(
    const NativeSimState& state,
    const std::string& player,
    const std::vector<int>& mask,
    const NativeRuntimeConfig& runtime_cfg,
    InferRuntimeT& infer_runtime,
    int model_version,
    double* out_build_sec = nullptr,
    double* out_forward_sec = nullptr
) {
    const double t_build0 = now_sec();
    constexpr int N_REQ = 20;
    constexpr int D_REQ = 3;
    constexpr int D_GLOBAL = 9;
    constexpr int MAX_ACTIVE_REQUESTS = 200;
    constexpr int REMAINING_PREFILL_DENOM = 3072 * 10;
    constexpr float SLACK_CLIP = 5.0f;
    constexpr float EPS = 1e-9f;

    std::vector<const NativeRequestState*> active;
    active.reserve(state.requests.size());
    for (const auto& r : state.requests) {
        if (!r.completed) active.push_back(&r);
    }

    std::vector<const NativeRequestState*> prefill_reqs;
    prefill_reqs.reserve(active.size());
    for (const auto* r : active) {
        if (!r->prefill_done) prefill_reqs.push_back(r);
    }

    std::sort(prefill_reqs.begin(), prefill_reqs.end(), [&](const NativeRequestState* a, const NativeRequestState* b) {
        const double d1 = (a->prefill_slo > 0.0) ? (a->queued_at + a->prefill_slo) : std::numeric_limits<double>::infinity();
        const double d2 = (b->prefill_slo > 0.0) ? (b->queued_at + b->prefill_slo) : std::numeric_limits<double>::infinity();
        const double t1 = d1 - state.sim_time;
        const double t2 = d2 - state.sim_time;
        if (std::fabs(t1 - t2) > 1e-12) return t1 < t2;
        return a->request_id < b->request_id;
    });

    std::vector<float> req_feat((size_t)N_REQ * (size_t)D_REQ, 0.0f);
    std::vector<std::uint8_t> req_mask_vec((size_t)N_REQ, 0);

    const float max_prefill_tokens_f = (float)std::max(1, runtime_cfg.max_request_tokens);
    const float slowdown = (float)std::max(runtime_cfg.prefill_slowdown, 1e-9);

    for (int i = 0; i < N_REQ && i < (int)prefill_reqs.size(); ++i) {
        const auto* req = prefill_reqs[(size_t)i];
        const int total_pref = std::max(0, req->num_prefill_tokens);
        const int done_pref = std::max(0, req->num_processed_prefill_tokens);
        const int remaining_pref = std::max(0, total_pref - done_pref);

        const float rem_norm = (float)remaining_pref / max_prefill_tokens_f;
        const float cached_norm = (float)done_pref / max_prefill_tokens_f;

        float slack_ratio = 0.0f;
        const float slo = (float)req->prefill_slo;
        if (slo > 0.0f) {
            const float deadline = (float)(req->queued_at + req->prefill_slo);
            const float time_left = deadline - (float)state.sim_time;
            const float base_total_exec = slo / slowdown;
            const float frac_remaining = (float)remaining_pref / (float)std::max(1, total_pref);
            const float est_remaining_exec = base_total_exec * frac_remaining;
            const float slack = time_left - est_remaining_exec;
            slack_ratio = slack / slo;
            if (slack_ratio > SLACK_CLIP) slack_ratio = SLACK_CLIP;
            if (slack_ratio < -SLACK_CLIP) slack_ratio = -SLACK_CLIP;
        }

        req_feat[(size_t)i * 3 + 0] = rem_norm;
        req_feat[(size_t)i * 3 + 1] = cached_norm;
        req_feat[(size_t)i * 3 + 2] = slack_ratio;
        req_mask_vec[(size_t)i] = 1;
    }

    int num_prefill = (int)prefill_reqs.size();
    int num_decode = 0;
    for (const auto* r : active) {
        if (r->prefill_done && (r->num_decode_tokens > r->num_processed_decode_tokens)) num_decode += 1;
    }
    int missed_prefill = 0;
    for (const auto* r : prefill_reqs) {
        if (r->prefill_slo > 0.0 && state.sim_time > (r->queued_at + r->prefill_slo)) missed_prefill += 1;
    }
    const float denom_total = (float)std::max(1, (int)active.size());
    const float denom_prefill = (float)std::max(1, num_prefill);
    const float prefill_frac_total = (float)num_prefill / denom_total;
    const float decode_frac_total = (float)num_decode / denom_total;
    const float prefill_missed_frac = (float)missed_prefill / denom_prefill;
    float prefill_rate = (float)((state.stats.maximum_qps > 0) ? state.stats.maximum_qps : 5);
    if (prefill_rate <= 0.0f) prefill_rate = 5.0f;
    float backlog_over_rate = (float)num_prefill / std::max(1.0f, prefill_rate + 1.0f);
    backlog_over_rate = std::min(backlog_over_rate, 25.0f);

    const float prefill_over_200 = (float)num_prefill / (float)MAX_ACTIVE_REQUESTS;
    const float decode_over_200 = (float)num_decode / (float)MAX_ACTIVE_REQUESTS;

    int total_remaining_prefill = 0;
    for (const auto* r : prefill_reqs) total_remaining_prefill += std::max(0, r->num_prefill_tokens - r->num_processed_prefill_tokens);
    const float remaining_prefill_norm = (float)total_remaining_prefill / (float)REMAINING_PREFILL_DENOM;

    int num_total_violated_active = 0;
    int num_decode_violated_active = 0;
    for (const auto* r : active) {
        const bool violated = state.stats.violated_request_ids.find(r->request_id) != state.stats.violated_request_ids.end();
        if (!violated) continue;
        num_total_violated_active += 1;
        if (r->prefill_done && (r->num_decode_tokens > r->num_processed_decode_tokens)) num_decode_violated_active += 1;
    }
    const float decode_violated_over_200 = (float)num_decode_violated_active / (float)MAX_ACTIVE_REQUESTS;
    const float total_violated_over_200 = (float)num_total_violated_active / (float)MAX_ACTIVE_REQUESTS;

    std::vector<float> global_feat = {
        prefill_frac_total,
        decode_frac_total,
        prefill_missed_frac,
        backlog_over_rate,
        prefill_over_200,
        decode_over_200,
        decode_violated_over_200,
        total_violated_over_200,
        remaining_prefill_norm,
    };

    torch::Tensor req_features = torch::from_blob(
        req_feat.data(),
        {1, N_REQ, D_REQ},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
    torch::Tensor global_features = torch::from_blob(
        global_feat.data(),
        {1, D_GLOBAL},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
    torch::Tensor req_mask = torch::from_blob(
        req_mask_vec.data(),
        {1, N_REQ},
        torch::TensorOptions().dtype(torch::kBool)
    ).clone();

    std::vector<std::uint8_t> action_mask_vec(mask.size(), 0);
    for (size_t i = 0; i < mask.size(); ++i) action_mask_vec[i] = (mask[i] != 0) ? 1 : 0;
    torch::Tensor action_mask = torch::from_blob(
        action_mask_vec.data(),
        {1, (long long)mask.size()},
        torch::TensorOptions().dtype(torch::kBool)
    ).clone();

    const double t_build1 = now_sec();
    auto out = infer_runtime.infer_tensors(
        req_features,
        global_features,
        c10::optional<torch::Tensor>(req_mask),
        c10::optional<torch::Tensor>(action_mask),
        player,
        model_version
    );
    const double t_fwd1 = now_sec();
    if (out_build_sec != nullptr) *out_build_sec = (t_build1 - t_build0);
    if (out_forward_sec != nullptr) *out_forward_sec = (t_fwd1 - t_build1);
    return out;
}

static NativeActionMaskFull actions_and_mask_full_native(
    const NativeSimState& state,
    const std::string& player,
    const NativeRuntimeConfig& runtime_cfg
) {
    NativeActionMaskFull out;
    if (player == "controller") {
        const int step = std::max(1, runtime_cfg.interval_request_size);
        std::vector<int> budgets;
        budgets.reserve(6);
        for (int i = 1; i <= 6; ++i) budgets.push_back(step * i);
        auto view = NativeSim::controller_view_from_state(state);
        ControllerSampleOutput c = sample_controller_actions_native(
            view,
            state.sim_time,
            budgets,
            runtime_cfg.prefill_profile_tokens,
            runtime_cfg.prefill_profile_times
        );
        out.controller_actions = std::move(c.actions);
        out.mask = std::move(c.mask);
    } else {
        const int num_actions = std::max(1, runtime_cfg.adversary_num_actions);
        out.adversary_actions.resize((size_t)num_actions);
        out.mask.assign((size_t)num_actions, 0);

        const bool can_send = (state.stats.last_prefill_batch_time < 0.0)
            || (state.sim_time >= state.stats.last_prefill_batch_time + 1.0 - 1e-9);
        if (!can_send) {
            out.adversary_actions[0].valid = true;
            out.mask[0] = 1;
        } else {
            const double prefill_slo = nearest_prefill_estimate(
                runtime_cfg.max_request_tokens,
                runtime_cfg.prefill_profile_tokens,
                runtime_cfg.prefill_profile_times
            );
            for (int i = 0; i < num_actions; ++i) {
                AdversaryActionSpecNative a;
                a.valid = true;
                a.requests.reserve((size_t)(i + 1));
                for (int k = 0; k < i + 1; ++k) {
                    AdversaryRequestSpecNative r;
                    r.prefill_tokens = runtime_cfg.max_request_tokens;
                    r.decode_tokens = runtime_cfg.adversary_fixed_decode_tokens;
                    r.prefill_slo = prefill_slo;
                    r.decode_slo = runtime_cfg.default_decode_slo;
                    a.requests.push_back(r);
                }
                out.adversary_actions[(size_t)i] = std::move(a);
                out.mask[(size_t)i] = 1;
            }
        }
    }

    out.valid.reserve(out.mask.size());
    for (size_t i = 0; i < out.mask.size(); ++i) {
        if (out.mask[i] != 0) out.valid.push_back((int)i);
    }
    return out;
}

static int create_child_native(
    std::vector<NativeTreeNode>& nodes,
    int parent_idx,
    int action_index,
    const ControllerActionSpecNative* ctrl_action,
    const AdversaryActionSpecNative* adv_action,
    double prior,
    int& next_node_id
) {
    ensure_node_index(nodes, parent_idx, "create_child_native(parent)");
    NativeTreeNode child;
    const auto& parent = nodes[(size_t)parent_idx];
    child.player = next_player(parent.player);
    child.node_id = next_node_id++;
    child.depth = parent.depth + 1;
    child.parent = parent_idx;
    child.parent_action_index = action_index;
    child.prior = prior;
    child.parent_action = py::none();
    if (ctrl_action != nullptr) {
        child.parent_action_is_controller = true;
        child.parent_controller_action = *ctrl_action;
    } else if (adv_action != nullptr) {
        child.parent_action_is_controller = false;
        child.parent_adversary_action = *adv_action;
    }
    nodes.push_back(std::move(child));
    int idx = (int)nodes.size() - 1;
    nodes[(size_t)parent_idx].children[action_index] = idx;
    return idx;
}

static void apply_edge_transition_native(
    std::vector<NativeTreeNode>& nodes,
    int parent_idx,
    int child_idx,
    NativeSimState& state,
    const NativeSearchCfg& cfg,
    const NativeRuntimeConfig& runtime_cfg,
    NativePredictor& predictor
) {
    ensure_node_index(nodes, parent_idx, "apply_edge_transition_native(parent)");
    ensure_node_index(nodes, child_idx, "apply_edge_transition_native(child)");
    auto& parent = nodes[(size_t)parent_idx];
    auto& child = nodes[(size_t)child_idx];
    const double parent_cost = parent.state_cost;

    if (parent.player == "adversary") {
        if (child.parent_action_is_controller) {
            throw std::runtime_error("parent/child action type mismatch (expected adversary action)");
        }
        NativeSim::apply_adversary_action_inplace(state, child.parent_adversary_action, runtime_cfg);
    } else {
        if (!child.parent_action_is_controller) {
            throw std::runtime_error("parent/child action type mismatch (expected controller action)");
        }
        NativeSim::apply_controller_action_inplace(state, child.parent_controller_action, runtime_cfg, predictor);
    }

    const double child_cost = NativeSim::evaluate_objective_cost(state);
    child.reward = transition_reward(cfg, parent_cost, child_cost);
    child.state_cost = child_cost;
    child.sim_time = state.sim_time;
}

static NativeSimState restore_state_for_node_native(
    std::vector<NativeTreeNode>& nodes,
    int node_idx,
    const NativeSimState& root_native_state,
    const NativeSearchCfg& cfg,
    const NativeRuntimeConfig& runtime_cfg,
    NativePredictor& predictor,
    int* out_missing_nodes = nullptr
) {
    ensure_node_index(nodes, node_idx, "restore_state_for_node_native(target)");

    std::vector<int> missing;
    missing.reserve(64);

    int cur = node_idx;
    while (cur >= 0) {
        ensure_node_index(nodes, cur, "restore_state_for_node_native(walk)");
        if (nodes[(size_t)cur].has_state_snapshot) break;
        missing.push_back(cur);
        cur = nodes[(size_t)cur].parent;
    }

    NativeSimState state;
    if (cur >= 0) {
        state = nodes[(size_t)cur].state_snapshot;
    } else {
        state = root_native_state;
    }

    for (auto it = missing.rbegin(); it != missing.rend(); ++it) {
        const int idx = *it;
        ensure_node_index(nodes, idx, "restore_state_for_node_native(fill)");
        auto& node = nodes[(size_t)idx];

        if (node.parent < 0) {
            node.state_snapshot = state;
            node.has_state_snapshot = true;
            node.state_cost = NativeSim::evaluate_objective_cost(state);
            node.sim_time = state.sim_time;
            continue;
        }

        ensure_node_index(nodes, node.parent, "restore_state_for_node_native(parent)");
        apply_edge_transition_native(nodes, node.parent, idx, state, cfg, runtime_cfg, predictor);
        node.state_snapshot = state;
        node.has_state_snapshot = true;
    }

    if (out_missing_nodes != nullptr) {
        *out_missing_nodes = (int)missing.size();
    }

    return state;
}

template <typename InferRuntimeT>
static std::tuple<double, bool, int> expand_node_full_native(
    std::vector<NativeTreeNode>& nodes,
    int node_idx,
    NativeSimState& state,
    InferRuntimeT& infer_runtime,
    int model_version,
    const NativeSearchCfg& cfg,
    const NativeRuntimeConfig& runtime_cfg,
    NativePredictor& predictor,
    std::mt19937& rng,
    int& next_node_id,
    NativeSearchPerf* perf = nullptr
) {
    ensure_node_index(nodes, node_idx, "expand_node_full_native(node)");
    (void)predictor;
    (void)rng;
    if (perf != nullptr) perf->expand_calls += 1;
    const double t_am0 = now_sec();
    auto am = actions_and_mask_full_native(state, nodes[(size_t)node_idx].player, runtime_cfg);
    const double t_am1 = now_sec();
    if (perf != nullptr) perf->actions_mask_total_sec += (t_am1 - t_am0);
    nodes[(size_t)node_idx].num_valid_actions = (int)am.valid.size();
    nodes[(size_t)node_idx].nn_valid_mask = am.mask;

    if (am.valid.empty()) {
        return std::make_tuple(0.0, false, 0);
    }

    if (am.valid.size() == 1) {
        const int idx = am.valid[0];
        if (nodes[(size_t)node_idx].children.find(idx) == nodes[(size_t)node_idx].children.end()) {
            if (nodes[(size_t)node_idx].player == "controller") {
                create_child_native(
                    nodes,
                    node_idx,
                    idx,
                    &am.controller_actions[(size_t)idx],
                    nullptr,
                    1.0,
                    next_node_id
                );
            } else {
                create_child_native(
                    nodes,
                    node_idx,
                    idx,
                    nullptr,
                    &am.adversary_actions[(size_t)idx],
                    1.0,
                    next_node_id
                );
            }
        }
        return std::make_tuple(0.0, false, 1);
    }

    double infer_build_sec = 0.0;
    double infer_forward_sec = 0.0;
    auto infer_out = infer_from_native_state_torchscript(
        state,
        nodes[(size_t)node_idx].player,
        am.mask,
        runtime_cfg,
        infer_runtime,
        model_version,
        &infer_build_sec,
        &infer_forward_sec
    );
    if (perf != nullptr) {
        perf->infer_calls += 1;
        perf->infer_build_sec += infer_build_sec;
        perf->infer_forward_sec += infer_forward_sec;
        perf->infer_total_sec += (infer_build_sec + infer_forward_sec);
    }
    const double model_value = std::get<0>(infer_out);
    std::vector<double> priors = std::get<1>(std::move(infer_out));
    const size_t a = am.mask.size();
    if (priors.size() < a) priors.resize(a, 0.0);

    nodes[(size_t)node_idx].has_nn_value = true;
    nodes[(size_t)node_idx].nn_value_controller = model_value;
    nodes[(size_t)node_idx].nn_priors = priors;

    const bool is_controller_player = (nodes[(size_t)node_idx].player == "controller");
    const double min_p = is_controller_player
        ? cfg.controller_min_prior_threshold
        : cfg.adversary_min_prior_threshold;

    const double t_thr0 = now_sec();
    std::map<int, double> valid_prior_by_idx;
    for (int i : am.valid) {
        valid_prior_by_idx[i] = (i >= 0 && (size_t)i < priors.size()) ? priors[(size_t)i] : 0.0;
    }
    std::map<int, double> norm_prior = apply_min_prior_threshold(valid_prior_by_idx, min_p);

    nodes[(size_t)node_idx].nn_priors_after_threshold.assign(priors.size(), 0.0);
    for (const auto& kv : norm_prior) {
        if (kv.first >= 0 && (size_t)kv.first < nodes[(size_t)node_idx].nn_priors_after_threshold.size()) {
            nodes[(size_t)node_idx].nn_priors_after_threshold[(size_t)kv.first] = kv.second;
        }
    }
    if (perf != nullptr) perf->expand_threshold_sec += (now_sec() - t_thr0);

    nodes[(size_t)node_idx].action_alias_to_canonical.clear();
    nodes[(size_t)node_idx].canonical_to_action_aliases.clear();

    std::vector<int> canonical_indices;
    std::map<int, double> canonical_prior;

    const double t_dedup0 = now_sec();
    if (is_controller_player) {
        std::map<std::string, int> sig_to_canon;
        if (perf != nullptr) perf->expand_controller_actions_total += (long long)am.valid.size();
        for (int idx : am.valid) {
            if ((size_t)idx >= am.controller_actions.size()) continue;
            if (perf != nullptr) {
                perf->expand_controller_alloc_pairs_total +=
                    (long long)am.controller_actions[(size_t)idx].token_allocations.size();
            }
            const std::string sig = controller_action_signature(am.controller_actions[(size_t)idx]);
            auto it = sig_to_canon.find(sig);
            int canon = idx;
            if (it == sig_to_canon.end()) {
                sig_to_canon[sig] = canon;
                nodes[(size_t)node_idx].canonical_to_action_aliases[canon] = {idx};
            } else {
                canon = it->second;
                nodes[(size_t)node_idx].canonical_to_action_aliases[canon].push_back(idx);
            }
            nodes[(size_t)node_idx].action_alias_to_canonical[idx] = canon;
        }
        for (const auto& kv : nodes[(size_t)node_idx].canonical_to_action_aliases) {
            canonical_indices.push_back(kv.first);
            double psum = 0.0;
            for (int aidx : kv.second) {
                auto itp = norm_prior.find(aidx);
                if (itp != norm_prior.end()) psum += itp->second;
            }
            canonical_prior[kv.first] = psum;
        }
    } else {
        canonical_indices = am.valid;
        for (int idx : canonical_indices) {
            auto itp = norm_prior.find(idx);
            canonical_prior[idx] = (itp != norm_prior.end()) ? itp->second : 0.0;
        }
    }
    if (perf != nullptr) perf->expand_dedup_sec += (now_sec() - t_dedup0);

    ensure_node_capacity(nodes, canonical_indices.size() + 4, perf);

    const double t_child0 = now_sec();
    for (int idx : canonical_indices) {
        if (nodes[(size_t)node_idx].children.find(idx) != nodes[(size_t)node_idx].children.end()) continue;
        if (is_controller_player) {
            if ((size_t)idx >= am.controller_actions.size()) continue;
            create_child_native(
                nodes,
                node_idx,
                idx,
                &am.controller_actions[(size_t)idx],
                nullptr,
                canonical_prior[idx],
                next_node_id
            );
        } else {
            if ((size_t)idx >= am.adversary_actions.size()) continue;
            create_child_native(
                nodes,
                node_idx,
                idx,
                nullptr,
                &am.adversary_actions[(size_t)idx],
                canonical_prior[idx],
                next_node_id
            );
        }
    }
    if (perf != nullptr) perf->expand_child_create_sec += (now_sec() - t_child0);

    return std::make_tuple(model_value, true, (int)am.valid.size());
}

template <typename InferRuntimeT>
static py::dict search_mcts_dnn_torchscript_impl(
    py::object env,
    py::object root_state,
    std::string root_player,
    int iterations,
    InferRuntimeT& infer_runtime,
    int model_version,
    int max_branching,
    double controller_min_prior_threshold,
    double adversary_min_prior_threshold,
    bool root_dirichlet_noise_enabled,
    double root_dirichlet_alpha,
    double root_dirichlet_epsilon,
    double pb_c_base,
    double pb_c_init,
    double discount_factor,
    double prefill_step_time,
    double reward_knee,
    double reward_max_penalty,
    double reward_tail_alpha,
    int seed,
    int root_node_id,
    int root_depth,
    int game_id,
    int root_id,
    std::string iter_log_path,
    bool iter_complete_log
) {
    const long long search_call = ++g_native_ts_search_calls;
    const double t_search0 = now_sec();
    NativeSearchPerf perf;
    if (native_trace_should_log(search_call)) {
        std::ostringstream oss;
        oss
            << "search_ts begin call=" << search_call
            << " root_player=" << root_player
            << " iterations=" << iterations
            << " model_version=" << model_version
            << " root_node_id=" << root_node_id
            << " root_depth=" << root_depth;
        native_trace(oss.str());
    }

    NativeSearchCfg cfg;
    cfg.max_branching = max_branching;
    cfg.controller_min_prior_threshold = controller_min_prior_threshold;
    cfg.adversary_min_prior_threshold = adversary_min_prior_threshold;
    cfg.root_dirichlet_noise_enabled = root_dirichlet_noise_enabled;
    cfg.root_dirichlet_alpha = root_dirichlet_alpha;
    cfg.root_dirichlet_epsilon = root_dirichlet_epsilon;
    cfg.pb_c_base = pb_c_base;
    cfg.pb_c_init = pb_c_init;
    cfg.discount_factor = discount_factor;
    cfg.prefill_step_time = prefill_step_time;
    cfg.reward_knee = reward_knee;
    cfg.reward_max_penalty = reward_max_penalty;
    cfg.reward_tail_alpha = reward_tail_alpha;
    cfg.seed = seed;
    cfg.root_node_id = root_node_id;
    cfg.root_depth = root_depth;

    const double t_state0 = now_sec();
    NativeRuntimeConfig runtime_cfg = runtime_cfg_from_env(env);
    NativeSimState root_native_state = native_state_from_py(env, root_state);
    perf.state_build_sec += (now_sec() - t_state0);
    if (native_trace_should_log(search_call)) {
        std::ostringstream oss;
        oss
            << "search_ts state call=" << search_call
            << " sim_time=" << root_native_state.sim_time
            << " requests=" << root_native_state.requests.size()
            << " active_ids=" << root_native_state.stats.active_request_ids.size();
        native_trace(oss.str());
    }

    const double t_pred0 = now_sec();
    NativePredictor predictor;
    std::string predictor_csv_path;
    try {
        predictor_csv_path = py::cast<std::string>(env.attr("_native_predictor_table_path"));
    } catch (...) {
    }
    if (predictor_csv_path.empty()) {
        const char* p = std::getenv("VIDUR_NATIVE_BATCH_TIME_TABLE");
        if (p != nullptr) predictor_csv_path = p;
    }
    if (!predictor_csv_path.empty()) {
        predictor.load_csv(predictor_csv_path);
    }
    perf.predictor_load_sec += (now_sec() - t_pred0);
    if (native_trace_should_log(search_call)) {
        std::ostringstream oss;
        oss
            << "search_ts predictor call=" << search_call
            << " table=" << (predictor_csv_path.empty() ? std::string("<none>") : predictor_csv_path);
        native_trace(oss.str());
    }

    std::mt19937 rng((uint32_t)cfg.seed);
    NativeMinMaxStats minmax;

    std::vector<NativeTreeNode> nodes;
    {
        const size_t expected_nodes =
            (size_t)std::max(8, iterations + 8) * (size_t)std::max(2, max_branching);
        nodes.reserve(expected_nodes);
    }

    NativeTreeNode root;
    root.player = root_player;
    root.node_id = cfg.root_node_id;
    root.depth = cfg.root_depth;
    root.parent = -1;
    root.parent_action = py::none();
    root.state_cost = NativeSim::evaluate_objective_cost(root_native_state);
    root.sim_time = root_native_state.sim_time;
    root.has_state_snapshot = true;
    root.state_snapshot = root_native_state;
    nodes.push_back(std::move(root));

    int next_node_id = cfg.root_node_id + 1;
    NativeSimState root_work = root_native_state;
    const double t_root_expand0 = now_sec();
    auto root_expand = expand_node_full_native(
        nodes,
        0,
        root_work,
        infer_runtime,
        model_version,
        cfg,
        runtime_cfg,
        predictor,
        rng,
        next_node_id,
        &perf
    );
    perf.root_expand_sec += (now_sec() - t_root_expand0);
    maybe_add_root_dirichlet_noise(
        nodes,
        0,
        std::get<1>(root_expand),
        std::get<2>(root_expand),
        cfg,
        rng
    );
    NativeIterCsvLogger iter_logger(iter_log_path);
    const bool iter_log_enabled = iter_logger.enabled();
    if (native_trace_should_log(search_call)) {
        std::ostringstream oss;
        oss
            << "search_ts root_expanded call=" << search_call
            << " root_children=" << nodes[0].children.size()
            << " root_valid_actions=" << nodes[0].num_valid_actions;
        native_trace(oss.str());
    }

    for (int it = 0; it < iterations; ++it) {
        if (native_trace_should_log(search_call) && (it == 0 || ((it + 1) % native_trace_every()) == 0)) {
            std::ostringstream oss;
            oss
                << "search_ts iter call=" << search_call
                << " it=" << (it + 1)
                << "/" << iterations
                << " nodes=" << nodes.size();
            native_trace(oss.str());
        }
        int current = 0;
        std::vector<int> search_path;
        search_path.reserve(32);
        search_path.push_back(current);

        const double t_select0 = now_sec();
        while (!nodes[(size_t)current].children.empty()) {
            ensure_node_index(nodes, current, "search_loop.select.current");
            int child_idx = select_child(nodes, current, cfg, minmax, rng);
            if (child_idx < 0) break;
            ensure_node_index(nodes, child_idx, "search_loop.select.child");
            current = child_idx;
            search_path.push_back(current);
            perf.selection_steps += 1;
        }
        perf.selection_sec += (now_sec() - t_select0);

        int restore_missing_nodes = 0;
        const double t_restore0 = now_sec();
        NativeSimState state = restore_state_for_node_native(
            nodes,
            current,
            root_native_state,
            cfg,
            runtime_cfg,
            predictor,
            &restore_missing_nodes
        );
        perf.restore_sec += (now_sec() - t_restore0);
        perf.restore_missing_nodes_total += (long long)restore_missing_nodes;
        if (native_trace_should_log(search_call) && restore_missing_nodes > 0 && (it == 0 || ((it + 1) % native_trace_every()) == 0)) {
            std::ostringstream oss;
            oss
                << "search_ts restore call=" << search_call
                << " it=" << (it + 1)
                << " missing_nodes=" << restore_missing_nodes
                << " target_node_idx=" << current;
            native_trace(oss.str());
        }

        const double t_forced0 = now_sec();
        while (true) {
            ensure_node_index(nodes, current, "search_loop.forced.current");
            const double t_forced_am0 = now_sec();
            auto am = actions_and_mask_full_native(state, nodes[(size_t)current].player, runtime_cfg);
            const double forced_am_dt = (now_sec() - t_forced_am0);
            perf.forced_actions_mask_sec += forced_am_dt;
            perf.actions_mask_total_sec += forced_am_dt;
            nodes[(size_t)current].num_valid_actions = (int)am.valid.size();
            nodes[(size_t)current].nn_valid_mask = am.mask;
            if (am.valid.size() != 1) break;

            if (iter_log_enabled && iter_complete_log) {
                const int parent_idx = nodes[(size_t)current].parent;
                const bool parent_multi =
                    (parent_idx < 0) || (nodes[(size_t)parent_idx].children.size() > 1);
                write_native_iter_row(
                    iter_logger,
                    nodes,
                    current,
                    state,
                    game_id,
                    root_id,
                    it,
                    cfg,
                    root_player,
                    std::string("forced_step:") + iter_phase_label(parent_multi, 1),
                    false,
                    1,
                    (int)nodes[(size_t)current].children.size()
                );
            }

            const int only_idx = am.valid[0];
            auto it_child = nodes[(size_t)current].children.find(only_idx);
            int child_idx = -1;
            if (it_child == nodes[(size_t)current].children.end()) {
                if (nodes[(size_t)current].player == "controller") {
                    if (only_idx < 0 || (size_t)only_idx >= am.controller_actions.size()) {
                        std::ostringstream oss;
                        oss
                            << "forced chain controller index out of bounds idx=" << only_idx
                            << " actions_size=" << am.controller_actions.size();
                        throw std::runtime_error(oss.str());
                    }
                    child_idx = create_child_native(
                        nodes,
                        current,
                        only_idx,
                        &am.controller_actions[(size_t)only_idx],
                        nullptr,
                        1.0,
                        next_node_id
                    );
                } else {
                    if (only_idx < 0 || (size_t)only_idx >= am.adversary_actions.size()) {
                        std::ostringstream oss;
                        oss
                            << "forced chain adversary index out of bounds idx=" << only_idx
                            << " actions_size=" << am.adversary_actions.size();
                        throw std::runtime_error(oss.str());
                    }
                    child_idx = create_child_native(
                        nodes,
                        current,
                        only_idx,
                        nullptr,
                        &am.adversary_actions[(size_t)only_idx],
                        1.0,
                        next_node_id
                    );
                }
            } else {
                child_idx = it_child->second;
            }
            ensure_node_index(nodes, child_idx, "search_loop.forced.child");
            const double t_forced_apply0 = now_sec();
            apply_edge_transition_native(nodes, current, child_idx, state, cfg, runtime_cfg, predictor);
            perf.forced_apply_sec += (now_sec() - t_forced_apply0);
            if (!nodes[(size_t)child_idx].has_state_snapshot) {
                nodes[(size_t)child_idx].state_snapshot = state;
                nodes[(size_t)child_idx].has_state_snapshot = true;
            }
            current = child_idx;
            search_path.push_back(current);
            perf.forced_steps += 1;
        }
        perf.forced_chain_sec += (now_sec() - t_forced0);

        const double t_leaf_expand0 = now_sec();
        auto leaf_expand = expand_node_full_native(
            nodes,
            current,
            state,
            infer_runtime,
            model_version,
            cfg,
            runtime_cfg,
            predictor,
            rng,
            next_node_id,
            &perf
        );
        perf.leaf_expand_sec += (now_sec() - t_leaf_expand0);
        double value = std::get<0>(leaf_expand);

        if (iter_log_enabled) {
            ensure_node_index(nodes, current, "iter_logger.leaf");
            const auto& leaf = nodes[(size_t)current];
            const int parent_idx = leaf.parent;
            const bool parent_multi =
                (parent_idx < 0) || (nodes[(size_t)parent_idx].children.size() > 1);
            write_native_iter_row(
                iter_logger,
                nodes,
                current,
                state,
                game_id,
                root_id,
                it,
                cfg,
                root_player,
                iter_phase_label(parent_multi, leaf.num_valid_actions),
                (leaf.num_valid_actions > 1),
                leaf.num_valid_actions,
                (int)leaf.children.size()
            );
        }

        const double t_backprop0 = now_sec();
        for (int i = (int)search_path.size() - 1; i >= 0; --i) {
            const int node_idx = search_path[(size_t)i];
            auto& node = nodes[(size_t)node_idx];
            node.value_sum += value;
            node.visits += 1;

            if (node.parent < 0) continue;
            auto& parent = nodes[(size_t)node.parent];
            const bool parent_is_branching =
                (parent.num_valid_actions > 1) || (parent.children.size() > 1);
            const double disc =
                parent_is_branching ? time_discount(cfg, node.sim_time, parent.sim_time) : 1.0;
            const double reward_used = parent_is_branching ? node.reward : 0.0;

            if (parent_is_branching && node.visits > 0) {
                const double q = reward_used + disc * (node.value_sum / (double)node.visits);
                minmax.update(q);
            }
            value = reward_used + disc * value;
        }
        perf.backprop_sec += (now_sec() - t_backprop0);
    }

    perf.total_sec = (now_sec() - t_search0);
    const auto& root_final = nodes[0];
    py::dict out;
    out["root_node_id"] = py::int_(root_final.node_id);
    out["root_depth"] = py::int_(root_final.depth);
    out["root_player"] = py::str(root_final.player);
    out["root_visits"] = py::int_(root_final.visits);
    out["root_value_sum"] = py::float_(root_final.value_sum);
    out["root_state_cost"] = py::float_(root_final.state_cost);
    out["root_sim_time"] = py::float_(root_final.sim_time);
    out["root_num_valid_actions"] = py::int_(root_final.num_valid_actions);
    out["root_nn_value_controller"] = root_final.has_nn_value
        ? py::object(py::float_(root_final.nn_value_controller))
        : py::object(py::none());
    out["root_nn_priors"] = py::cast(root_final.nn_priors);
    out["root_nn_priors_after_threshold"] = py::cast(root_final.nn_priors_after_threshold);
    out["root_nn_valid_mask"] = py::cast(root_final.nn_valid_mask);
    out["action_alias_to_canonical"] = py::cast(root_final.action_alias_to_canonical);
    out["canonical_to_action_aliases"] = py::cast(root_final.canonical_to_action_aliases);

    py::list children;
    for (const auto& kv : root_final.children) {
        const int action_idx = kv.first;
        const auto& child = nodes[(size_t)kv.second];
        py::dict c;
        c["index"] = py::int_(action_idx);
        c["node_id"] = py::int_(child.node_id);
        c["depth"] = py::int_(child.depth);
        c["player"] = py::str(child.player);
        c["prior"] = py::float_(child.prior);
        c["reward"] = py::float_(child.reward);
        c["visits"] = py::int_(child.visits);
        c["value_sum"] = py::float_(child.value_sum);
        c["state_cost"] = py::float_(child.state_cost);
        c["sim_time"] = py::float_(child.sim_time);
        c["num_valid_actions"] = py::int_(child.num_valid_actions);
        try {
            if (child.parent_action_is_controller) c["parent_action"] = controller_action_to_py(child.parent_controller_action);
            else c["parent_action"] = adversary_action_to_py(child.parent_adversary_action);
        } catch (...) {
            c["parent_action"] = py::none();
        }
        c["parent_action_index"] = py::int_(child.parent_action_index);
        children.append(c);
    }
    out["children"] = children;
    out["node_count"] = py::int_(nodes.size());
    py::dict perf_out;
    perf_out["total_sec"] = py::float_(perf.total_sec);
    perf_out["state_build_sec"] = py::float_(perf.state_build_sec);
    perf_out["predictor_load_sec"] = py::float_(perf.predictor_load_sec);
    perf_out["root_expand_sec"] = py::float_(perf.root_expand_sec);
    perf_out["selection_sec"] = py::float_(perf.selection_sec);
    perf_out["restore_sec"] = py::float_(perf.restore_sec);
    perf_out["forced_chain_sec"] = py::float_(perf.forced_chain_sec);
    perf_out["forced_actions_mask_sec"] = py::float_(perf.forced_actions_mask_sec);
    perf_out["forced_apply_sec"] = py::float_(perf.forced_apply_sec);
    perf_out["leaf_expand_sec"] = py::float_(perf.leaf_expand_sec);
    perf_out["backprop_sec"] = py::float_(perf.backprop_sec);
    perf_out["actions_mask_total_sec"] = py::float_(perf.actions_mask_total_sec);
    perf_out["infer_total_sec"] = py::float_(perf.infer_total_sec);
    perf_out["infer_build_sec"] = py::float_(perf.infer_build_sec);
    perf_out["infer_forward_sec"] = py::float_(perf.infer_forward_sec);
    perf_out["expand_threshold_sec"] = py::float_(perf.expand_threshold_sec);
    perf_out["expand_dedup_sec"] = py::float_(perf.expand_dedup_sec);
    perf_out["expand_child_create_sec"] = py::float_(perf.expand_child_create_sec);
    perf_out["infer_calls"] = py::int_(perf.infer_calls);
    perf_out["expand_calls"] = py::int_(perf.expand_calls);
    perf_out["selection_steps"] = py::int_(perf.selection_steps);
    perf_out["forced_steps"] = py::int_(perf.forced_steps);
    perf_out["restore_missing_nodes_total"] = py::int_(perf.restore_missing_nodes_total);
    perf_out["expand_controller_actions_total"] = py::int_(perf.expand_controller_actions_total);
    perf_out["expand_controller_alloc_pairs_total"] = py::int_(perf.expand_controller_alloc_pairs_total);
    perf_out["nodes_capacity_grows"] = py::int_(perf.nodes_capacity_grows);
    out["perf"] = perf_out;
    if (native_trace_should_log(search_call)) {
        std::ostringstream oss;
        oss
            << "search_ts end call=" << search_call
            << " root_visits=" << root_final.visits
            << " root_children=" << root_final.children.size()
            << " node_count=" << nodes.size()
            << " total_sec=" << perf.total_sec
            << " infer_sec=" << perf.infer_total_sec
            << " actions_mask_sec=" << perf.actions_mask_total_sec
            << " restore_sec=" << perf.restore_sec
            << " leaf_expand_sec=" << perf.leaf_expand_sec;
        native_trace(oss.str());
    }
    return out;
}

static py::dict search_mcts_dnn_torchscript(
    py::object env,
    py::object root_state,
    std::string root_player,
    int iterations,
    NativeTorchScriptInferRuntime& infer_runtime,
    int model_version,
    int max_branching,
    double controller_min_prior_threshold,
    double adversary_min_prior_threshold,
    bool root_dirichlet_noise_enabled,
    double root_dirichlet_alpha,
    double root_dirichlet_epsilon,
    double pb_c_base,
    double pb_c_init,
    double discount_factor,
    double prefill_step_time,
    double reward_knee,
    double reward_max_penalty,
    double reward_tail_alpha,
    int seed,
    int root_node_id,
    int root_depth,
    int game_id,
    int root_id,
    std::string iter_log_path,
    bool iter_complete_log
) {
    return search_mcts_dnn_torchscript_impl(
        std::move(env),
        std::move(root_state),
        std::move(root_player),
        iterations,
        infer_runtime,
        model_version,
        max_branching,
        controller_min_prior_threshold,
        adversary_min_prior_threshold,
        root_dirichlet_noise_enabled,
        root_dirichlet_alpha,
        root_dirichlet_epsilon,
        pb_c_base,
        pb_c_init,
        discount_factor,
        prefill_step_time,
        reward_knee,
        reward_max_penalty,
        reward_tail_alpha,
        seed,
        root_node_id,
        root_depth,
        game_id,
        root_id,
        std::move(iter_log_path),
        iter_complete_log
    );
}

static py::dict search_mcts_dnn_torchscript_service(
    py::object env,
    py::object root_state,
    std::string root_player,
    int iterations,
    NativeInferServiceRuntime& infer_runtime,
    int model_version,
    int max_branching,
    double controller_min_prior_threshold,
    double adversary_min_prior_threshold,
    bool root_dirichlet_noise_enabled,
    double root_dirichlet_alpha,
    double root_dirichlet_epsilon,
    double pb_c_base,
    double pb_c_init,
    double discount_factor,
    double prefill_step_time,
    double reward_knee,
    double reward_max_penalty,
    double reward_tail_alpha,
    int seed,
    int root_node_id,
    int root_depth,
    int game_id,
    int root_id,
    std::string iter_log_path,
    bool iter_complete_log
) {
    return search_mcts_dnn_torchscript_impl(
        std::move(env),
        std::move(root_state),
        std::move(root_player),
        iterations,
        infer_runtime,
        model_version,
        max_branching,
        controller_min_prior_threshold,
        adversary_min_prior_threshold,
        root_dirichlet_noise_enabled,
        root_dirichlet_alpha,
        root_dirichlet_epsilon,
        pb_c_base,
        pb_c_init,
        discount_factor,
        prefill_step_time,
        reward_knee,
        reward_max_penalty,
        reward_tail_alpha,
        seed,
        root_node_id,
        root_depth,
        game_id,
        root_id,
        std::move(iter_log_path),
        iter_complete_log
    );
}

PYBIND11_MODULE(mcts_native, m) {
    m.doc() = "Native MCTS runtime scaffolding and fast controller sampler";

    py::class_<AllocationEntry>(m, "AllocationEntry")
        .def(py::init<>())
        .def_readwrite("request_id", &AllocationEntry::request_id)
        .def_readwrite("tokens", &AllocationEntry::tokens);

    py::class_<ControllerRequestStateNative>(m, "ControllerRequestStateNative")
        .def(py::init<>())
        .def_readwrite("request_id", &ControllerRequestStateNative::request_id)
        .def_readwrite("prefill_done", &ControllerRequestStateNative::prefill_done)
        .def_readwrite("remaining_prefill", &ControllerRequestStateNative::remaining_prefill)
        .def_readwrite("remaining_decode", &ControllerRequestStateNative::remaining_decode)
        .def_readwrite("arrived_at", &ControllerRequestStateNative::arrived_at)
        .def_readwrite("prefill_slo", &ControllerRequestStateNative::prefill_slo)
        .def_readwrite("num_processed_tokens", &ControllerRequestStateNative::num_processed_tokens);

    py::class_<ControllerActionSpecNative>(m, "ControllerActionSpecNative")
        .def(py::init<>())
        .def_readwrite("token_budget", &ControllerActionSpecNative::token_budget)
        .def_readwrite("selected_request_ids", &ControllerActionSpecNative::selected_request_ids)
        .def_readwrite("token_allocations", &ControllerActionSpecNative::token_allocations)
        .def_readwrite("prefill_allocations", &ControllerActionSpecNative::prefill_allocations)
        .def_readwrite("decode_allocations", &ControllerActionSpecNative::decode_allocations)
        .def_readwrite("heuristic", &ControllerActionSpecNative::heuristic)
        .def_readwrite("strategy", &ControllerActionSpecNative::strategy)
        .def_readwrite("valid", &ControllerActionSpecNative::valid);

    py::class_<AdversaryRequestSpecNative>(m, "AdversaryRequestSpecNative")
        .def(py::init<>())
        .def_readwrite("prefill_tokens", &AdversaryRequestSpecNative::prefill_tokens)
        .def_readwrite("decode_tokens", &AdversaryRequestSpecNative::decode_tokens)
        .def_readwrite("prefill_slo", &AdversaryRequestSpecNative::prefill_slo)
        .def_readwrite("decode_slo", &AdversaryRequestSpecNative::decode_slo);

    py::class_<AdversaryActionSpecNative>(m, "AdversaryActionSpecNative")
        .def(py::init<>())
        .def_readwrite("requests", &AdversaryActionSpecNative::requests)
        .def_readwrite("stop_decode_ids", &AdversaryActionSpecNative::stop_decode_ids)
        .def_readwrite("valid", &AdversaryActionSpecNative::valid);

    py::class_<ControllerSampleOutput>(m, "ControllerSampleOutput")
        .def(py::init<>())
        .def_readwrite("actions", &ControllerSampleOutput::actions)
        .def_readwrite("mask", &ControllerSampleOutput::mask);

    py::class_<AdversarySampleOutput>(m, "AdversarySampleOutput")
        .def(py::init<>())
        .def_readwrite("actions", &AdversarySampleOutput::actions)
        .def_readwrite("mask", &AdversarySampleOutput::mask);

    py::class_<NativeRequestState>(m, "NativeRequestState")
        .def(py::init<>())
        .def_readwrite("request_id", &NativeRequestState::request_id)
        .def_readwrite("arrived_at", &NativeRequestState::arrived_at)
        .def_readwrite("queued_at", &NativeRequestState::queued_at)
        .def_readwrite("num_prefill_tokens", &NativeRequestState::num_prefill_tokens)
        .def_readwrite("num_processed_prefill_tokens", &NativeRequestState::num_processed_prefill_tokens)
        .def_readwrite("num_decode_tokens", &NativeRequestState::num_decode_tokens)
        .def_readwrite("num_processed_decode_tokens", &NativeRequestState::num_processed_decode_tokens)
        .def_readwrite("prefill_done", &NativeRequestState::prefill_done)
        .def_readwrite("completed", &NativeRequestState::completed)
        .def_readwrite("prefill_completed_at", &NativeRequestState::prefill_completed_at)
        .def_readwrite("prefill_slo", &NativeRequestState::prefill_slo)
        .def_readwrite("decode_slo", &NativeRequestState::decode_slo);

    py::class_<NativeGameStats>(m, "NativeGameStats")
        .def(py::init<>())
        .def_readwrite("requests_generated", &NativeGameStats::requests_generated)
        .def_readwrite("requests_completed", &NativeGameStats::requests_completed)
        .def_readwrite("slo_violations", &NativeGameStats::slo_violations)
        .def_readwrite("slo_lateness_sum", &NativeGameStats::slo_lateness_sum)
        .def_readwrite("maximum_qps", &NativeGameStats::maximum_qps)
        .def_readwrite("last_prefill_batch_time", &NativeGameStats::last_prefill_batch_time)
        .def_readwrite("recent_arrivals", &NativeGameStats::recent_arrivals)
        .def_readwrite("per_request_prefill_lateness", &NativeGameStats::per_request_prefill_lateness)
        .def_readwrite("per_request_decode_lateness", &NativeGameStats::per_request_decode_lateness)
        .def_readwrite("decode_tokens_counted", &NativeGameStats::decode_tokens_counted)
        .def_readwrite("decode_next_deadline_by_id", &NativeGameStats::decode_next_deadline_by_id)
        .def_readwrite("prefill_lateness_finalized", &NativeGameStats::prefill_lateness_finalized)
        .def_readwrite("violated_request_ids", &NativeGameStats::violated_request_ids)
        .def_readwrite("active_request_ids", &NativeGameStats::active_request_ids)
        .def_readwrite("completed_request_ids", &NativeGameStats::completed_request_ids);

    py::class_<NativeRuntimeConfig>(m, "NativeRuntimeConfig")
        .def(py::init<>())
        .def_readwrite("interval_request_size", &NativeRuntimeConfig::interval_request_size)
        .def_readwrite("max_request_tokens", &NativeRuntimeConfig::max_request_tokens)
        .def_readwrite("adversary_num_actions", &NativeRuntimeConfig::adversary_num_actions)
        .def_readwrite("adversary_fixed_decode_tokens", &NativeRuntimeConfig::adversary_fixed_decode_tokens)
        .def_readwrite("prefill_slowdown", &NativeRuntimeConfig::prefill_slowdown)
        .def_readwrite("default_decode_slo", &NativeRuntimeConfig::default_decode_slo)
        .def_readwrite("prefill_profile_tokens", &NativeRuntimeConfig::prefill_profile_tokens)
        .def_readwrite("prefill_profile_times", &NativeRuntimeConfig::prefill_profile_times);

    py::class_<NativeSimState>(m, "NativeSimState")
        .def(py::init<>())
        .def_readwrite("sim_time", &NativeSimState::sim_time)
        .def_readwrite("next_request_id", &NativeSimState::next_request_id)
        .def_readwrite("requests", &NativeSimState::requests)
        .def_readwrite("stats", &NativeSimState::stats);

    py::class_<NativeSim>(m, "NativeSim")
        .def(py::init<>())
        .def_static("apply_adversary_action", &NativeSim::apply_adversary_action,
            py::arg("in_state"),
            py::arg("action"),
            py::arg("cfg"))
        .def_static("apply_controller_action", &NativeSim::apply_controller_action,
            py::arg("in_state"),
            py::arg("action"),
            py::arg("cfg"),
            py::arg("predictor"))
        .def_static("evaluate_objective_cost", &NativeSim::evaluate_objective_cost,
            py::arg("state"))
        .def_static("controller_view_from_state", &NativeSim::controller_view_from_state,
            py::arg("state"))
        .def_static("snapshot", [](const NativeSimState& s) { return py::bytes(NativeSim::snapshot(s)); })
        .def_static("restore", [](py::bytes b) {
            std::string payload = b;
            return NativeSim::restore(payload);
        })
        .def_static("describe_state", [](const NativeSimState& s) {
            py::dict d;
            d["sim_time"] = s.sim_time;
            d["requests_in_system"] = py::int_(s.requests.size());
            d["slo_violations"] = py::int_(s.stats.slo_violations);
            d["total_lateness"] = s.stats.slo_lateness_sum;
            return d;
        });

    py::class_<NativePredictorKey>(m, "NativePredictorKey")
        .def(py::init<>())
        .def_readwrite("total_tokens_rounded", &NativePredictorKey::total_tokens_rounded)
        .def_readwrite("batch_size", &NativePredictorKey::batch_size)
        .def_readwrite("prefill_batch_size", &NativePredictorKey::prefill_batch_size)
        .def_readwrite("decode_batch_size", &NativePredictorKey::decode_batch_size)
        .def_readwrite("decode_avg_kv_cache_size", &NativePredictorKey::decode_avg_kv_cache_size)
        .def_readwrite("prefill_agg_kv_cache_size", &NativePredictorKey::prefill_agg_kv_cache_size)
        .def_readwrite("prefill_agg_chunk_size", &NativePredictorKey::prefill_agg_chunk_size);

    py::class_<NativePredictor>(m, "NativePredictor")
        .def(py::init<>())
        .def("load_csv", &NativePredictor::load_csv, py::arg("path"))
        .def("is_loaded", &NativePredictor::is_loaded)
        .def("lookup_batch_time", &NativePredictor::lookup_batch_time,
            py::arg("reqs"),
            py::arg("token_alloc"),
            py::arg("profile_tokens"),
            py::arg("profile_times"),
            py::arg("fallback_total") = 0.001,
            py::arg("fallback_model") = 0.0007)
        .def_static("build_key", &NativePredictor::build_key,
            py::arg("reqs"),
            py::arg("token_alloc"),
            py::arg("kv_granularity") = 64,
            py::arg("prefill_chunk_granularity") = 32);

    py::class_<NativeTorchScriptInferRuntime>(m, "NativeTorchScriptInferRuntime")
        .def(py::init<const std::string&, double, double>(),
            py::arg("device") = "cpu",
            py::arg("v_min") = -50.0,
            py::arg("v_step") = 0.125)
        .def("load_models", &NativeTorchScriptInferRuntime::load_models,
            py::arg("model_version"),
            py::arg("controller_path"),
            py::arg("adversary_path"))
        .def("infer_from_inputs", &NativeTorchScriptInferRuntime::infer_from_inputs,
            py::arg("inputs"),
            py::arg("player"),
            py::arg("model_version"));

    py::class_<NativeInferServiceRuntime>(m, "NativeInferServiceRuntime")
        .def(py::init<const std::string&, double, double, int, int>(),
            py::arg("addr"),
            py::arg("v_min") = -50.0,
            py::arg("v_step") = 0.125,
            py::arg("connect_timeout_ms") = 2000,
            py::arg("request_timeout_ms") = 30000)
        .def("ping", &NativeInferServiceRuntime::ping)
        .def("shutdown", &NativeInferServiceRuntime::shutdown)
        .def("load_models", &NativeInferServiceRuntime::load_models,
            py::arg("model_version"),
            py::arg("controller_path"),
            py::arg("adversary_path"))
        .def("infer_from_inputs", &NativeInferServiceRuntime::infer_from_inputs,
            py::arg("inputs"),
            py::arg("player"),
            py::arg("model_version"));

    py::class_<NativeMCTS>(m, "NativeMCTS")
        .def(py::init<>())
        .def_static("search", []() { NativeMCTS::search(); });

    py::class_<NativeActionSpace>(m, "NativeActionSpace")
        .def(py::init<>())
        .def_static("sample_controller_actions", &sample_controller_actions_pyready,
            py::arg("request_states"),
            py::arg("sim_time"),
            py::arg("budgets"),
            py::arg("profile_tokens"),
            py::arg("profile_times"),
            py::arg("num_actions"))
        .def_static("sample_adversary_actions", &sample_adversary_actions_pyready,
            py::arg("num_actions"));

    m.def(
        "sample_controller_actions",
        &sample_controller_actions_native,
        py::arg("request_states"),
        py::arg("sim_time"),
        py::arg("budgets"),
        py::arg("profile_tokens"),
        py::arg("profile_times")
    );
    m.def("sample_controller_actions_pyready",
      &sample_controller_actions_pyready,
      py::arg("request_states"),
      py::arg("sim_time"),
      py::arg("budgets"),
      py::arg("profile_tokens"),
      py::arg("profile_times"),
      py::arg("num_actions"));
    m.def(
      "search_mcts_dnn",
      &search_mcts_dnn,
      py::arg("env"),
      py::arg("root_state"),
      py::arg("root_player"),
      py::arg("iterations"),
      py::arg("infer_cb"),
      py::arg("max_branching"),
      py::arg("controller_min_prior_threshold"),
      py::arg("adversary_min_prior_threshold"),
      py::arg("root_dirichlet_noise_enabled"),
      py::arg("root_dirichlet_alpha"),
      py::arg("root_dirichlet_epsilon"),
      py::arg("pb_c_base"),
      py::arg("pb_c_init"),
      py::arg("discount_factor"),
      py::arg("prefill_step_time"),
      py::arg("reward_knee"),
      py::arg("reward_max_penalty"),
      py::arg("reward_tail_alpha"),
      py::arg("seed"),
      py::arg("root_node_id"),
      py::arg("root_depth"),
      py::arg("game_id") = 0,
      py::arg("root_id") = 0,
      py::arg("iter_log_path") = "",
      py::arg("iter_complete_log") = false);
    m.def(
      "search_mcts_dnn_torchscript",
      &search_mcts_dnn_torchscript,
      py::arg("env"),
      py::arg("root_state"),
      py::arg("root_player"),
      py::arg("iterations"),
      py::arg("infer_runtime"),
      py::arg("model_version"),
      py::arg("max_branching"),
      py::arg("controller_min_prior_threshold"),
      py::arg("adversary_min_prior_threshold"),
      py::arg("root_dirichlet_noise_enabled"),
      py::arg("root_dirichlet_alpha"),
      py::arg("root_dirichlet_epsilon"),
      py::arg("pb_c_base"),
      py::arg("pb_c_init"),
      py::arg("discount_factor"),
      py::arg("prefill_step_time"),
      py::arg("reward_knee"),
      py::arg("reward_max_penalty"),
      py::arg("reward_tail_alpha"),
      py::arg("seed"),
      py::arg("root_node_id"),
      py::arg("root_depth"),
      py::arg("game_id") = 0,
      py::arg("root_id") = 0,
      py::arg("iter_log_path") = "",
      py::arg("iter_complete_log") = false);
    m.def(
      "search_mcts_dnn_torchscript_service",
      &search_mcts_dnn_torchscript_service,
      py::arg("env"),
      py::arg("root_state"),
      py::arg("root_player"),
      py::arg("iterations"),
      py::arg("infer_runtime"),
      py::arg("model_version"),
      py::arg("max_branching"),
      py::arg("controller_min_prior_threshold"),
      py::arg("adversary_min_prior_threshold"),
      py::arg("root_dirichlet_noise_enabled"),
      py::arg("root_dirichlet_alpha"),
      py::arg("root_dirichlet_epsilon"),
      py::arg("pb_c_base"),
      py::arg("pb_c_init"),
      py::arg("discount_factor"),
      py::arg("prefill_step_time"),
      py::arg("reward_knee"),
      py::arg("reward_max_penalty"),
      py::arg("reward_tail_alpha"),
      py::arg("seed"),
      py::arg("root_node_id"),
      py::arg("root_depth"),
      py::arg("game_id") = 0,
      py::arg("root_id") = 0,
      py::arg("iter_log_path") = "",
      py::arg("iter_complete_log") = false);
}
