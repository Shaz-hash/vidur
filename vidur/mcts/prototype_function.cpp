#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct AllocationEntry {
    int request_id = -1;
    int tokens = 0;
};

struct ControllerRequestStateNative {
    int request_id = -1;
    bool prefill_done = false;
    int remaining_prefill = 0;
    int remaining_decode = 0;
    double arrived_at = 0.0;
    double prefill_slo = 0.0;
};

struct ControllerActionSpecNative {
    int token_budget = 0;
    std::vector<int> selected_request_ids;
    std::vector<AllocationEntry> token_allocations;
    std::vector<AllocationEntry> prefill_allocations;
    std::vector<AllocationEntry> decode_allocations;
    std::string heuristic;
    std::string strategy;
    bool valid = false;
};

struct ControllerSampleOutput {
    std::vector<ControllerActionSpecNative> actions;
    std::vector<int> mask; // 0/1
};

struct PrefillRecord {
    int rid = -1;
    int rem_pref = 0;
    double edf_key = 0.0;
    double lst_key = 0.0;
};

static inline int nonneg_i(int x) { return x < 0 ? 0 : x; }

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


PYBIND11_MODULE(prototype_function, m) {
    m.doc() = "Native controller action sampler";

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
        .def_readwrite("prefill_slo", &ControllerRequestStateNative::prefill_slo);

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

    py::class_<ControllerSampleOutput>(m, "ControllerSampleOutput")
        .def(py::init<>())
        .def_readwrite("actions", &ControllerSampleOutput::actions)
        .def_readwrite("mask", &ControllerSampleOutput::mask);

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

}
