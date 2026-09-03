#include "gv4/inference.hpp"
#include "gv4/mcts.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <utility>

namespace py = pybind11;

namespace gv4 {
namespace {

IterationObserver make_observer(const py::object& value) {
    if (value.is_none()) {
        return {};
    }
    py::function callback = value.cast<py::function>();
    return [callback](
               int iteration,
               const std::vector<const MCTSNode*>& path,
               const MCTSNode& root) {
        py::gil_scoped_acquire acquire;
        py::list path_view;
        for (const MCTSNode* node : path) {
            path_view.append(
                py::cast(node, py::return_value_policy::reference));
        }
        callback(
            py::arg("iteration_index") = iteration,
            py::arg("path") = std::move(path_view),
            py::arg("root") =
                py::cast(&root, py::return_value_policy::reference));
    };
}

void put_optional(
    py::dict& result,
    const char* name,
    const std::optional<double>& value) {
    result[name] = value.has_value() ? py::cast(*value) : py::none();
}

py::dict rollout_stats_to_dict(const RolloutStats& stats) {
    py::dict result;
    result["leaf_evaluations"] = stats.leaf_evaluations;
    result["trajectories"] = stats.trajectories;
    result["actions"] = stats.actions;
    result["terminal_trajectories"] = stats.terminal_trajectories;
    result["bootstrap_calls"] = stats.bootstrap_calls;
    result["cutoff_leaf_evaluations"] = stats.cutoff_leaf_evaluations;
    put_optional(result, "root_time", stats.root_time);
    put_optional(result, "deadline", stats.deadline);
    put_optional(result, "min_start_time", stats.min_start_time);
    put_optional(result, "max_start_time", stats.max_start_time);
    put_optional(result, "min_final_time", stats.min_final_time);
    put_optional(result, "max_final_time", stats.max_final_time);
    put_optional(result, "min_deadline", stats.min_deadline);
    put_optional(result, "max_deadline", stats.max_deadline);
    put_optional(
        result,
        "min_expansion_parent_time",
        stats.min_expansion_parent_time);
    put_optional(
        result,
        "max_expansion_parent_time",
        stats.max_expansion_parent_time);
    put_optional(
        result,
        "min_remaining_rollout_sec",
        stats.min_remaining_rollout_sec);
    put_optional(
        result,
        "max_remaining_rollout_sec",
        stats.max_remaining_rollout_sec);
    result["first_history_hash"] = stats.first_history_hash.has_value()
        ? py::cast(*stats.first_history_hash)
        : py::none();
    result["first_history_actions"] = stats.first_history_actions;
    return result;
}

py::dict search_result_to_dict(const MCTSSearchResult& result) {
    py::list root_stats;
    for (const RootActionStats& item : result.root_action_stats) {
        py::dict row;
        row["representative_raw_index"] = item.representative_raw_index;
        row["equivalent_raw_indices"] = item.equivalent_raw_indices;
        row["visits"] = item.visits;
        row["value_sum"] = item.value_sum;
        row["mean_value"] = item.mean_value;
        root_stats.append(std::move(row));
    }

    py::dict output;
    output["root_node_id"] = result.root_node_id;
    output["root_player"] = result.root_player;
    output["next_player"] = result.next_player;
    output["best_action_index"] = result.best_action_index;
    output["best_action_value"] = result.best_action_value;
    output["action_values"] = result.action_values;
    output["valid_mask"] = result.valid_mask;
    output["root_action_stats"] = std::move(root_stats);
    output["used_bootstrap"] = result.used_bootstrap;
    output["used_rollout"] = result.used_rollout;
    output["rollout_stats"] = rollout_stats_to_dict(result.rollout_stats);
    return output;
}

}  // namespace

void bind_rollout_mcts(py::module_& module) {
    module.def(
        "run_policy_rollout_mcts",
        [](const Environment& environment,
           const State& state,
           Player player,
           InferenceRuntime* inference,
           int iterations,
           double puct_c,
           double policy_prior_temperature,
           double prior_min_probability,
           int rollout_count,
           double rollout_horizon_sec,
           std::uint64_t rollout_seed,
           double rollout_policy_temperature,
           double rollout_probability_quantum,
           int rollout_max_actions,
           bool use_policy_prior,
           bool use_model_bootstrap,
           int root_node_id,
           int root_depth,
           py::object iteration_observer) {
            MCTSConfig config;
            config.iterations = iterations;
            config.puct_c = puct_c;
            config.policy_prior_temperature = policy_prior_temperature;
            config.prior_min_probability = prior_min_probability;
            config.use_policy_prior = use_policy_prior;
            config.rollout_count = rollout_count;
            config.rollout_horizon_sec = rollout_horizon_sec;
            config.rollout_seed = rollout_seed;
            config.rollout_policy_temperature = rollout_policy_temperature;
            config.rollout_probability_quantum = rollout_probability_quantum;
            config.rollout_max_actions = rollout_max_actions;
            config.use_model_bootstrap = use_model_bootstrap;

            UniformMCTS search(environment, config, inference);
            return search_result_to_dict(search.search(
                state,
                player,
                root_node_id,
                root_depth,
                make_observer(iteration_observer)));
        },
        py::arg("environment"),
        py::arg("root_state"),
        py::arg("root_player"),
        py::arg("inference") = nullptr,
        py::arg("iterations") = 1000,
        py::arg("puct_c") = 1.0,
        py::arg("policy_prior_temperature") = 1.0,
        py::arg("prior_min_probability") = 1e-8,
        py::arg("rollout_count") = 10,
        py::arg("rollout_horizon_sec") = 0.4,
        py::arg("rollout_seed") = 0,
        py::arg("rollout_policy_temperature") = 1.0,
        py::arg("rollout_probability_quantum") = 1e-6,
        py::arg("rollout_max_actions") = 4096,
        py::arg("use_policy_prior") = false,
        py::arg("use_model_bootstrap") = false,
        py::arg("root_node_id") = 0,
        py::arg("root_depth") = 0,
        py::arg("iteration_observer") = py::none());
}

}  // namespace gv4
