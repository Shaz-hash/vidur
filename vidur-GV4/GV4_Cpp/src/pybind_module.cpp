#include "gv4/actions.hpp"
#include "gv4/config.hpp"
#include "gv4/engine.hpp"
#include "gv4/mcts.hpp"
#include "gv4/state.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <utility>

namespace py = pybind11;

namespace gv4 {
void bind_feature_types(py::module_& module);
void bind_inference_types(py::module_& module);
void bind_rollout_mcts(py::module_& module);

namespace {

BatchTiming call_timing_provider(
    const std::shared_ptr<py::function>& callback,
    const State& state,
    const ResolvedControllerAction& action) {
    py::gil_scoped_acquire acquire;
    py::object result = (*callback)(state, action);
    if (py::isinstance<BatchTiming>(result)) {
        return result.cast<BatchTiming>();
    }
    py::tuple values = result.cast<py::tuple>();
    if (values.size() != 2) {
        throw std::invalid_argument(
            "batch timing callback must return (stage_service, pp_communication)");
    }
    return BatchTiming{
        values[0].cast<std::vector<double>>(),
        values[1].cast<std::vector<double>>()};
}

double call_prefill_estimator(
    const std::shared_ptr<py::function>& callback,
    int tokens) {
    py::gil_scoped_acquire acquire;
    return py::cast<double>((*callback)(tokens));
}

std::unique_ptr<Environment> make_environment(
    Config config,
    py::function batch_timing,
    py::function prefill_time) {
    auto timing = std::make_shared<py::function>(std::move(batch_timing));
    auto prefill = std::make_shared<py::function>(std::move(prefill_time));
    return std::make_unique<Environment>(
        std::move(config),
        [timing](const State& state, const ResolvedControllerAction& action) {
            return call_timing_provider(timing, state, action);
        },
        [prefill](int tokens) { return call_prefill_estimator(prefill, tokens); });
}

}  // namespace
}  // namespace gv4

PYBIND11_MODULE(gv4_native, module) {
    using namespace gv4;

    module.doc() = "Single-replica native GV4 engine and MCTS runtime";
    module.attr("NO_ID") = kNoId;
    module.attr("UNSET_TIME") = kUnsetTime;

    py::enum_<Player>(module, "Player")
        .value("ADVERSARY", Player::Adversary)
        .value("CONTROLLER", Player::Controller);
    py::enum_<RequestLifecycle>(module, "RequestLifecycle")
        .value("WAITING_PREFILL", RequestLifecycle::WaitingPrefill)
        .value("INFLIGHT_PREFILL", RequestLifecycle::InflightPrefill)
        .value("WAITING_DECODE", RequestLifecycle::WaitingDecode)
        .value("INFLIGHT_DECODE", RequestLifecycle::InflightDecode)
        .value("STOP_PENDING", RequestLifecycle::StopPending)
        .value("DROP_PENDING", RequestLifecycle::DropPending)
        .value("COMPLETED", RequestLifecycle::Completed)
        .value("STOPPED", RequestLifecycle::Stopped)
        .value("DROPPED", RequestLifecycle::Dropped);
    py::enum_<TerminalReason>(module, "TerminalReason")
        .value("NONE", TerminalReason::None)
        .value("NATURAL_COMPLETION", TerminalReason::NaturalCompletion)
        .value("ADVERSARY_STOP", TerminalReason::AdversaryStop)
        .value("CONTROLLER_EVICTION", TerminalReason::ControllerEviction)
        .value("AUTOMATIC_SLO_DROP", TerminalReason::AutomaticSloDrop)
        .value("DECODE_CREDIT_EXHAUSTED", TerminalReason::DecodeCreditExhausted);
    py::enum_<ControllerTransitionKind>(module, "ControllerTransitionKind")
        .value("WAIT", ControllerTransitionKind::Wait)
        .value("EVICT_ONLY", ControllerTransitionKind::EvictOnly)
        .value("BATCH", ControllerTransitionKind::Batch);

    py::class_<ControllerActionConfig>(module, "ControllerActionConfig")
        .def(py::init<>())
        .def_readwrite("eviction_rules", &ControllerActionConfig::eviction_rules)
        .def_readwrite("prefill_budgets", &ControllerActionConfig::prefill_budgets)
        .def_readwrite("ordering_heuristics", &ControllerActionConfig::ordering_heuristics)
        .def_property_readonly("raw_action_count", &ControllerActionConfig::raw_action_count)
        .def("components", &ControllerActionConfig::components);
    py::class_<AdversaryActionConfig>(module, "AdversaryActionConfig")
        .def(py::init<>())
        .def_readwrite("max_launch_count_per_tick", &AdversaryActionConfig::max_launch_count_per_tick)
        .def_readwrite("prefill_templates", &AdversaryActionConfig::prefill_templates)
        .def_readwrite("stop_rules", &AdversaryActionConfig::stop_rules)
        .def_property_readonly("raw_action_count", &AdversaryActionConfig::raw_action_count)
        .def("components", &AdversaryActionConfig::components);
    py::class_<Config>(module, "Config")
        .def(py::init<>())
        .def_readwrite("tensor_parallel_size", &Config::tensor_parallel_size)
        .def_readwrite("pipeline_parallel_size", &Config::pipeline_parallel_size)
        .def_readwrite("rank_ids", &Config::rank_ids)
        .def_readwrite("rank_kv_capacity_blocks", &Config::rank_kv_capacity_blocks)
        .def_readwrite("block_size_tokens", &Config::block_size_tokens)
        .def_readwrite("max_batch_tokens", &Config::max_batch_tokens)
        .def_readwrite("max_sequences", &Config::max_sequences)
        .def_readwrite("max_prefill_chunk_tokens", &Config::max_prefill_chunk_tokens)
        .def_readwrite("max_inflight_microbatches", &Config::max_inflight_microbatches)
        .def_readwrite("inter_stage_queue_capacity", &Config::inter_stage_queue_capacity)
        .def_readwrite("adversary_tick_sec", &Config::adversary_tick_sec)
        .def_readwrite("launch_window_sec", &Config::launch_window_sec)
        .def_readwrite("max_requests_per_launch_window", &Config::max_requests_per_launch_window)
        .def_readwrite("epsilon", &Config::epsilon)
        .def_readwrite("time_round_digits", &Config::time_round_digits)
        .def_readwrite("max_zero_time_transitions_per_boundary", &Config::max_zero_time_transitions_per_boundary)
        .def_readwrite("decode_credit_mint", &Config::decode_credit_mint)
        .def_readwrite("max_prefill_tokens_per_request", &Config::max_prefill_tokens_per_request)
        .def_readwrite("min_decode_tokens_per_request", &Config::min_decode_tokens_per_request)
        .def_readwrite("max_decode_tokens_per_request", &Config::max_decode_tokens_per_request)
        .def_readwrite("target_decode_tokens_average", &Config::target_decode_tokens_average)
        .def_readwrite("target_prefill_tokens_window_average", &Config::target_prefill_tokens_window_average)
        .def_readwrite("prefill_slowdown_factor", &Config::prefill_slowdown_factor)
        .def_readwrite("decode_token_slo_sec", &Config::decode_token_slo_sec)
        .def_readwrite("violation_base_cost", &Config::violation_base_cost)
        .def_readwrite("lateness_cap_sec", &Config::lateness_cap_sec)
        .def_readwrite("terminal_drop_cost", &Config::terminal_drop_cost)
        .def_readwrite("automatic_drop_lateness_sec", &Config::automatic_drop_lateness_sec)
        .def_readwrite("discount_factor", &Config::discount_factor)
        .def_readwrite("discount_reference_step_sec", &Config::discount_reference_step_sec)
        .def_readwrite("controller_actions", &Config::controller_actions)
        .def_readwrite("adversary_actions", &Config::adversary_actions)
        .def_readwrite("max_requests", &Config::max_requests)
        .def_readwrite("max_launch_history_entries", &Config::max_launch_history_entries)
        .def_readwrite("global_seed", &Config::global_seed)
        .def_readwrite("enable_debug_asserts", &Config::enable_debug_asserts)
        .def_readwrite("state_schema_version", &Config::state_schema_version)
        .def_readwrite("feature_schema_version", &Config::feature_schema_version)
        .def_readwrite("manifest_sha256", &Config::manifest_sha256)
        .def("validate", &Config::validate)
        .def_property_readonly("logical_kv_capacity_blocks", &Config::logical_kv_capacity_blocks)
        .def("discount_for_elapsed", &Config::discount_for_elapsed);

    py::class_<LaunchRecord>(module, "LaunchRecord")
        .def(py::init<>())
        .def_readwrite("launch_time", &LaunchRecord::launch_time)
        .def_readwrite("request_count", &LaunchRecord::request_count)
        .def_readwrite("prefill_tokens", &LaunchRecord::prefill_tokens);
    py::class_<BatchAllocation>(module, "BatchAllocation")
        .def(py::init<>())
        .def_readwrite("request_id", &BatchAllocation::request_id)
        .def_readwrite("prefill_tokens", &BatchAllocation::prefill_tokens)
        .def_readwrite("decode_tokens", &BatchAllocation::decode_tokens)
        .def_readwrite("new_kv_blocks", &BatchAllocation::new_kv_blocks)
        .def_property_readonly("total_tokens", &BatchAllocation::total_tokens);
    py::class_<InflightMicrobatch>(module, "InflightMicrobatch")
        .def(py::init<>())
        .def_readwrite("microbatch_id", &InflightMicrobatch::microbatch_id)
        .def_readwrite("replica_id", &InflightMicrobatch::replica_id)
        .def_readwrite("raw_action_index", &InflightMicrobatch::raw_action_index)
        .def_readwrite("canonical_action_index", &InflightMicrobatch::canonical_action_index)
        .def_readwrite("allocations", &InflightMicrobatch::allocations)
        .def_readwrite("stage_ready_times", &InflightMicrobatch::stage_ready_times)
        .def_readwrite("stage_start_times", &InflightMicrobatch::stage_start_times)
        .def_readwrite("stage_finish_times", &InflightMicrobatch::stage_finish_times)
        .def_readwrite("completion_applied", &InflightMicrobatch::completion_applied)
        .def_property_readonly("final_completion_time", &InflightMicrobatch::final_completion_time)
        .def_property_readonly("total_prefill_tokens", &InflightMicrobatch::total_prefill_tokens)
        .def_property_readonly("total_decode_tokens", &InflightMicrobatch::total_decode_tokens);
    py::class_<RequestState>(module, "RequestState")
        .def(py::init<>())
        .def_readwrite("request_id", &RequestState::request_id)
        .def_readwrite("owner_replica_id", &RequestState::owner_replica_id)
        .def_readwrite("lifecycle", &RequestState::lifecycle)
        .def_readwrite("arrival_time", &RequestState::arrival_time)
        .def_readwrite("prefill_deadline", &RequestState::prefill_deadline)
        .def_readwrite("decode_token_slo_sec", &RequestState::decode_token_slo_sec)
        .def_readwrite("original_prefill_tokens", &RequestState::original_prefill_tokens)
        .def_readwrite("original_decode_tokens", &RequestState::original_decode_tokens)
        .def_readwrite("decode_credit_minted", &RequestState::decode_credit_minted)
        .def_readwrite("committed_prefill_tokens", &RequestState::committed_prefill_tokens)
        .def_readwrite("reserved_prefill_tokens", &RequestState::reserved_prefill_tokens)
        .def_readwrite("committed_decode_tokens", &RequestState::committed_decode_tokens)
        .def_readwrite("reserved_decode_tokens", &RequestState::reserved_decode_tokens)
        .def_readwrite("committed_kv_blocks", &RequestState::committed_kv_blocks)
        .def_readwrite("reserved_kv_blocks", &RequestState::reserved_kv_blocks)
        .def_readwrite("inflight_microbatch_id", &RequestState::inflight_microbatch_id)
        .def_readwrite("next_decode_deadline", &RequestState::next_decode_deadline)
        .def_readwrite("prefill_lateness_sec", &RequestState::prefill_lateness_sec)
        .def_readwrite("decode_lateness_sec", &RequestState::decode_lateness_sec)
        .def_readwrite("violation_recorded", &RequestState::violation_recorded)
        .def_readwrite("terminal_reason", &RequestState::terminal_reason)
        .def_readwrite("terminal_requested_at", &RequestState::terminal_requested_at)
        .def_readwrite("terminal_time", &RequestState::terminal_time)
        .def_property_readonly("remaining_prefill_tokens", &RequestState::remaining_prefill_tokens)
        .def_property_readonly("remaining_decode_tokens", &RequestState::remaining_decode_tokens)
        .def_property_readonly("resident_tokens", &RequestState::resident_tokens)
        .def_property_readonly("has_inflight_work", &RequestState::has_inflight_work);
    py::class_<ObjectiveState>(module, "ObjectiveState")
        .def(py::init<>())
        .def_readwrite("requests_generated", &ObjectiveState::requests_generated)
        .def_readwrite("requests_completed", &ObjectiveState::requests_completed)
        .def_readwrite("requests_stopped", &ObjectiveState::requests_stopped)
        .def_readwrite("requests_dropped", &ObjectiveState::requests_dropped)
        .def_readwrite("slo_violations", &ObjectiveState::slo_violations)
        .def_readwrite("prefill_lateness_sec", &ObjectiveState::prefill_lateness_sec)
        .def_readwrite("decode_lateness_sec", &ObjectiveState::decode_lateness_sec)
        .def_readwrite("terminal_cost", &ObjectiveState::terminal_cost)
        .def_readwrite("total_cost", &ObjectiveState::total_cost);
    py::class_<ReplicaState>(module, "ReplicaState")
        .def(py::init<>())
        .def_readwrite("replica_id", &ReplicaState::replica_id)
        .def_readwrite("rank_ids", &ReplicaState::rank_ids)
        .def_readwrite("rank_kv_capacity_blocks", &ReplicaState::rank_kv_capacity_blocks)
        .def_readwrite("rank_kv_committed_blocks", &ReplicaState::rank_kv_committed_blocks)
        .def_readwrite("rank_kv_reserved_blocks", &ReplicaState::rank_kv_reserved_blocks)
        .def_readwrite("stage_tail_finish_times", &ReplicaState::stage_tail_finish_times)
        .def_readwrite("stage_last_microbatch_ids", &ReplicaState::stage_last_microbatch_ids)
        .def_readwrite("inflight_microbatches", &ReplicaState::inflight_microbatches)
        .def_property_readonly("inflight_count", &ReplicaState::inflight_count)
        .def_property_readonly("pipeline_parallel_size", &ReplicaState::pipeline_parallel_size);
    py::class_<State>(module, "State")
        .def(py::init<>())
        .def_readwrite("state_schema_version", &State::state_schema_version)
        .def_readwrite("config_manifest_sha256", &State::config_manifest_sha256)
        .def_readwrite("now", &State::now)
        .def_readwrite("next_player", &State::next_player)
        .def_readwrite("next_adversary_tick", &State::next_adversary_tick)
        .def_readwrite("next_request_id", &State::next_request_id)
        .def_readwrite("next_microbatch_id", &State::next_microbatch_id)
        .def_readwrite("tie_break_counter", &State::tie_break_counter)
        .def_readwrite("rng_seed", &State::rng_seed)
        .def_readwrite("rng_counter", &State::rng_counter)
        .def_readwrite("launch_history", &State::launch_history)
        .def_readwrite("decode_credits_available", &State::decode_credits_available)
        .def_readwrite("decode_credits_reserved", &State::decode_credits_reserved)
        .def_readwrite("decode_credits_minted_total", &State::decode_credits_minted_total)
        .def_readwrite("decode_tokens_committed_total", &State::decode_tokens_committed_total)
        .def_readwrite("requests", &State::requests)
        .def_readwrite("replica", &State::replica)
        .def_readwrite("objective", &State::objective)
        .def(
            "request",
            py::overload_cast<int>(&State::request),
            py::return_value_policy::reference_internal)
        .def("validate", &State::validate)
        .def("clone", [](const State& state) { return state; });

    py::class_<ResolvedControllerAction>(module, "ResolvedControllerAction")
        .def(py::init<>())
        .def_readwrite("raw_action_index", &ResolvedControllerAction::raw_action_index)
        .def_readwrite("replica_id", &ResolvedControllerAction::replica_id)
        .def_readwrite("eviction_rule", &ResolvedControllerAction::eviction_rule)
        .def_readwrite("prefill_budget", &ResolvedControllerAction::prefill_budget)
        .def_readwrite("ordering_heuristic", &ResolvedControllerAction::ordering_heuristic)
        .def_readwrite("transition_kind", &ResolvedControllerAction::transition_kind)
        .def_readwrite("evicted_request_ids", &ResolvedControllerAction::evicted_request_ids)
        .def_readwrite("allocations", &ResolvedControllerAction::allocations)
        .def_readwrite("released_kv_blocks", &ResolvedControllerAction::released_kv_blocks)
        .def_readwrite("reserved_kv_blocks", &ResolvedControllerAction::reserved_kv_blocks)
        .def_readwrite("rank_kv_delta", &ResolvedControllerAction::rank_kv_delta)
        .def_property_readonly("total_prefill_tokens", &ResolvedControllerAction::total_prefill_tokens)
        .def_property_readonly("total_decode_tokens", &ResolvedControllerAction::total_decode_tokens);
    py::class_<CanonicalControllerAction>(module, "CanonicalControllerAction")
        .def(py::init<>())
        .def_readwrite("canonical_action_index", &CanonicalControllerAction::canonical_action_index)
        .def_readwrite("action", &CanonicalControllerAction::action)
        .def_readwrite("equivalent_raw_indices", &CanonicalControllerAction::equivalent_raw_indices)
        .def_property_readonly("representative_raw_index", &CanonicalControllerAction::representative_raw_index);
    py::class_<ResolvedAdversaryAction>(module, "ResolvedAdversaryAction")
        .def(py::init<>())
        .def_readwrite("raw_action_index", &ResolvedAdversaryAction::raw_action_index)
        .def_readwrite("launch_count", &ResolvedAdversaryAction::launch_count)
        .def_readwrite("prefill_tokens", &ResolvedAdversaryAction::prefill_tokens)
        .def_readwrite("stop_rule", &ResolvedAdversaryAction::stop_rule)
        .def_readwrite("stop_request_ids", &ResolvedAdversaryAction::stop_request_ids);
    py::class_<CanonicalAdversaryAction>(module, "CanonicalAdversaryAction")
        .def(py::init<>())
        .def_readwrite("canonical_action_index", &CanonicalAdversaryAction::canonical_action_index)
        .def_readwrite("action", &CanonicalAdversaryAction::action)
        .def_readwrite("equivalent_raw_indices", &CanonicalAdversaryAction::equivalent_raw_indices)
        .def_property_readonly("representative_raw_index", &CanonicalAdversaryAction::representative_raw_index);
    py::class_<ControllerActionSpace>(module, "ControllerActionSpace")
        .def_readonly("raw_to_canonical", &ControllerActionSpace::raw_to_canonical)
        .def_readonly("canonical_actions", &ControllerActionSpace::canonical_actions);
    py::class_<AdversaryActionSpace>(module, "AdversaryActionSpace")
        .def_readonly("raw_to_canonical", &AdversaryActionSpace::raw_to_canonical)
        .def_readonly("canonical_actions", &AdversaryActionSpace::canonical_actions);
    py::class_<BatchTiming>(module, "BatchTiming")
        .def(py::init<>())
        .def_readwrite("stage_service_times", &BatchTiming::stage_service_times)
        .def_readwrite("pp_communication_times", &BatchTiming::pp_communication_times);
    py::class_<TransitionOutcome>(module, "TransitionOutcome")
        .def_readonly("transition_kind", &TransitionOutcome::transition_kind)
        .def_readonly("elapsed_sec", &TransitionOutcome::elapsed_sec)
        .def_readonly("objective_before", &TransitionOutcome::objective_before)
        .def_readonly("objective_after", &TransitionOutcome::objective_after)
        .def_readonly("edge_reward", &TransitionOutcome::edge_reward)
        .def_readonly("discount", &TransitionOutcome::discount);

    bind_feature_types(module);
    bind_inference_types(module);

    py::class_<Environment>(module, "Environment")
        .def(py::init(&make_environment),
             py::arg("config"), py::arg("batch_timing_provider"),
             py::arg("prefill_time_estimator"))
        .def_property_readonly("config", &Environment::config,
                               py::return_value_policy::reference_internal)
        .def("initial_state", &Environment::initial_state,
             py::arg("now") = 0.0, py::arg("next_player") = Player::Adversary)
        .def("sample_controller_actions", &Environment::sample_controller_actions)
        .def("sample_adversary_actions", &Environment::sample_adversary_actions)
        .def("apply_controller_action_only", &Environment::apply_controller_action_only,
             py::arg("state"), py::arg("action"), py::arg("fast_forward") = true)
        .def("apply_adversary_action_only", &Environment::apply_adversary_action_only);

    py::class_<MCTSActionEntry>(module, "MCTSActionEntry")
        .def_readonly("representative_raw_index", &MCTSActionEntry::representative_raw_index)
        .def_readonly("equivalent_raw_indices", &MCTSActionEntry::equivalent_raw_indices)
        .def_readonly("action", &MCTSActionEntry::action)
        .def_readonly("prior", &MCTSActionEntry::prior);
    py::class_<MCTSNode>(module, "MCTSNode")
        .def_readonly("player", &MCTSNode::player)
        .def_readonly("node_id", &MCTSNode::node_id)
        .def_readonly("depth", &MCTSNode::depth)
        .def_readonly("parent_action", &MCTSNode::parent_action)
        .def_readonly("parent_action_index", &MCTSNode::parent_action_index)
        .def_readonly("reward", &MCTSNode::reward)
        .def_readonly("visits", &MCTSNode::visits)
        .def_readonly("value_sum", &MCTSNode::value_sum)
        .def_readonly("state_cost", &MCTSNode::state_cost)
        .def_readonly("sim_time", &MCTSNode::sim_time)
        .def_readonly("edge_discount", &MCTSNode::edge_discount)
        .def_readonly("state", &MCTSNode::state)
        .def_readonly("expanded", &MCTSNode::expanded)
        .def_readonly("actions", &MCTSNode::actions)
        .def_readonly("valid_mask", &MCTSNode::valid_mask)
        .def_property_readonly("mean_value", &MCTSNode::mean_value);

    module.def(
        "run_uniform_mcts",
        [](const Environment& environment,
           const State& state,
           Player player,
           int iterations,
           double puct_c,
           int root_node_id,
           int root_depth,
           py::object iteration_observer) {
            MCTSConfig config;
            config.iterations = iterations;
            config.puct_c = puct_c;
            UniformMCTS search(environment, config);
            IterationObserver observer;
            if (!iteration_observer.is_none()) {
                py::function callback = iteration_observer.cast<py::function>();
                observer = [callback](
                               int iteration,
                               const std::vector<const MCTSNode*>& path,
                               const MCTSNode& root) {
                    py::gil_scoped_acquire acquire;
                    py::list path_view;
                    for (const MCTSNode* node : path) {
                        path_view.append(py::cast(
                            node, py::return_value_policy::reference));
                    }
                    callback(
                        py::arg("iteration_index") = iteration,
                        py::arg("path") = std::move(path_view),
                        py::arg("root") = py::cast(
                            &root, py::return_value_policy::reference));
                };
            }
            const MCTSSearchResult result = search.search(
                state, player, root_node_id, root_depth, observer);
            py::list stats;
            for (const RootActionStats& item : result.root_action_stats) {
                py::dict row;
                row["representative_raw_index"] = item.representative_raw_index;
                row["equivalent_raw_indices"] = item.equivalent_raw_indices;
                row["visits"] = item.visits;
                row["value_sum"] = item.value_sum;
                row["mean_value"] = item.mean_value;
                stats.append(std::move(row));
            }
            py::dict output;
            output["root_node_id"] = result.root_node_id;
            output["root_player"] = result.root_player;
            output["next_player"] = result.next_player;
            output["best_action_index"] = result.best_action_index;
            output["best_action_value"] = result.best_action_value;
            output["action_values"] = result.action_values;
            output["valid_mask"] = result.valid_mask;
            output["root_action_stats"] = std::move(stats);
            output["used_bootstrap"] = result.used_bootstrap;
            return output;
        },
        py::arg("environment"),
        py::arg("root_state"),
        py::arg("root_player"),
        py::arg("iterations") = 1000,
        py::arg("puct_c") = 1.0,
        py::arg("root_node_id") = 0,
        py::arg("root_depth") = 0,
        py::arg("iteration_observer") = py::none());

    bind_rollout_mcts(module);

    module.def("blocks_for_tokens", &blocks_for_tokens);
    module.def("additional_blocks_for_work", &additional_blocks_for_work);
    module.def("free_logical_blocks", &free_logical_blocks);
    module.def("next_pipeline_admission_time", &next_pipeline_admission_time);
    module.def("next_wait_boundary_time", &next_wait_boundary_time);
    module.def("next_internal_completion_time", &next_internal_completion_time);
}
