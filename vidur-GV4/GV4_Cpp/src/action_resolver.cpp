#include "gv4/engine.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace gv4 {
namespace {

double prefill_lateness(const RequestState& request, double now) {
    return std::max({request.prefill_lateness_sec,
                     now - request.prefill_deadline,
                     0.0});
}

double decode_lateness(const RequestState& request) {
    return std::max(0.0,
                    request.prefill_lateness_sec + request.decode_lateness_sec);
}

int prefill_class_tokens(const RequestState& request) {
    const int recompute = request.remaining_recompute_tokens();
    return recompute > 0 ? recompute : request.remaining_prefill_tokens();
}

double prefill_class_deadline(const RequestState& request) {
    if (request.remaining_recompute_tokens() > 0 && request.is_decode_phase()) {
        return request.next_decode_deadline;
    }
    return request.prefill_deadline;
}

struct PreemptionCandidate {
    const RequestState* request = nullptr;
    double release_time = 0.0;
    int released_blocks = 0;
    int recompute_tokens = 0;
    double recovery_deadline = 0.0;
    bool pending = false;
};

std::vector<PreemptionCandidate> preemption_candidates(const State& state) {
    std::vector<PreemptionCandidate> result;
    for (const auto& request : state.requests) {
        if (is_terminal(request.lifecycle) ||
            request.lifecycle == RequestLifecycle::StopPending ||
            request.lifecycle == RequestLifecycle::DropPending ||
            request.lifecycle == RequestLifecycle::PreemptPending) {
            continue;
        }

        PreemptionCandidate candidate;
        candidate.request = &request;
        if (request.has_inflight_work()) {
            const auto* batch = state.replica.find_microbatch(
                request.inflight_microbatch_id);
            if (batch == nullptr) {
                throw std::logic_error(
                    "in-flight preemption candidate has no microbatch");
            }
            const auto allocation = std::find_if(
                batch->allocations.begin(), batch->allocations.end(),
                [&](const BatchAllocation& item) {
                    return item.request_id == request.request_id;
                });
            if (allocation == batch->allocations.end()) {
                throw std::logic_error(
                    "in-flight preemption candidate has no allocation");
            }
            if (allocation->decode_tokens > 0 &&
                request.remaining_decode_tokens() == 0) {
                continue;
            }
            candidate.release_time = batch->final_completion_time();
            candidate.recompute_tokens = request.logical_context_tokens() +
                                         allocation->prefill_tokens +
                                         allocation->decode_tokens;
            candidate.released_blocks = request.committed_kv_blocks +
                                        request.reserved_kv_blocks;
            const int prefill_after = request.committed_prefill_tokens +
                                      allocation->prefill_tokens;
            if (prefill_after < request.original_prefill_tokens) {
                candidate.recovery_deadline = request.prefill_deadline;
            } else if (allocation->prefill_tokens > 0 ||
                       allocation->decode_tokens > 0) {
                candidate.recovery_deadline = candidate.release_time +
                                              request.decode_token_slo_sec;
            } else {
                candidate.recovery_deadline = request.next_decode_deadline;
            }
            candidate.pending = true;
        } else {
            candidate.release_time = state.now;
            candidate.recompute_tokens = request.logical_context_tokens();
            candidate.released_blocks = request.committed_kv_blocks;
            candidate.recovery_deadline = request.is_decode_phase()
                ? request.next_decode_deadline
                : request.prefill_deadline;
        }
        if (candidate.recompute_tokens > 0 && candidate.released_blocks > 0) {
            result.push_back(candidate);
        }
    }
    return result;
}

double recompute_time(
    const PreemptionCandidate& candidate,
    const PrefillTimeEstimator& estimator) {
    const double duration = estimator(candidate.recompute_tokens);
    if (!std::isfinite(duration) || duration < 0.0) {
        throw std::runtime_error("prefill estimator returned invalid duration");
    }
    return duration;
}

std::vector<int> preemption_targets(
    const State& state,
    const Config& config,
    const std::string& rule,
    const std::vector<int>& excluded_ids,
    const PrefillTimeEstimator& estimator) {
    if (rule == "preempt_none") return {};
    const std::unordered_set<int> excluded(
        excluded_ids.begin(), excluded_ids.end());
    auto candidates = preemption_candidates(state);
    candidates.erase(
        std::remove_if(
            candidates.begin(), candidates.end(),
            [&](const PreemptionCandidate& item) {
                return excluded.count(item.request->request_id) != 0;
            }),
        candidates.end());
    if (candidates.empty()) return {};

    const PreemptionCandidate* selected = &candidates.front();
    if (rule == "preempt_min_recompute") {
        for (const auto& item : candidates) {
            const auto key = std::tuple{
                item.recompute_tokens, -item.released_blocks,
                item.request->request_id};
            const auto selected_key = std::tuple{
                selected->recompute_tokens, -selected->released_blocks,
                selected->request->request_id};
            if (key < selected_key) selected = &item;
        }
    } else if (rule == "preempt_largest_kv") {
        for (const auto& item : candidates) {
            const auto key = std::tuple{
                -item.released_blocks, item.recompute_tokens,
                item.request->request_id};
            const auto selected_key = std::tuple{
                -selected->released_blocks, selected->recompute_tokens,
                selected->request->request_id};
            if (key < selected_key) selected = &item;
        }
    } else if (rule == "preempt_max_recovery_slack") {
        auto key = [&](const PreemptionCandidate& item) {
            return std::tuple{
                item.recovery_deadline - item.release_time -
                    recompute_time(item, estimator),
                item.released_blocks,
                -item.request->request_id};
        };
        for (const auto& item : candidates) {
            if (key(item) > key(*selected)) selected = &item;
        }
    } else if (rule == "preempt_best_relief_cost") {
        auto key = [&](const PreemptionCandidate& item) {
            const double duration = recompute_time(item, estimator);
            const double lateness = std::max(
                0.0, item.release_time + duration - item.recovery_deadline);
            const double violation_cost =
                lateness > config.epsilon && !item.request->violation_recorded
                ? config.violation_base_cost
                : 0.0;
            const double recovery_cost = duration + violation_cost +
                std::min(lateness, config.lateness_cap_sec);
            const double score = item.released_blocks /
                std::max(config.epsilon, recovery_cost);
            return std::tuple{
                score, item.released_blocks, -item.request->request_id};
        };
        for (const auto& item : candidates) {
            if (key(item) > key(*selected)) selected = &item;
        }
    } else {
        throw std::runtime_error("unsupported preemption rule");
    }
    return {selected->request->request_id};
}

std::vector<const RequestState*> waiting_prefills(const State& state) {
    std::vector<const RequestState*> result;
    for (const auto& request : state.requests) {
        if (request.owner_replica_id == 0 &&
            request.lifecycle == RequestLifecycle::WaitingPrefill) {
            result.push_back(&request);
        }
    }
    return result;
}

std::vector<const RequestState*> waiting_decodes(const State& state) {
    std::vector<const RequestState*> result;
    for (const auto& request : state.requests) {
        if (request.owner_replica_id == 0 &&
            request.lifecycle == RequestLifecycle::WaitingDecode) {
            result.push_back(&request);
        }
    }
    return result;
}

std::vector<int> eviction_targets(
    const State& state,
    const Config& config,
    const std::string& rule,
    const std::vector<const RequestState*>& prefills,
    const std::vector<const RequestState*>& decodes) {
    std::vector<const RequestState*> resident_prefills;
    std::vector<const RequestState*> resident_decodes;
    for (const RequestState* request : prefills) {
        if (request->committed_kv_blocks > 0) resident_prefills.push_back(request);
    }
    for (const RequestState* request : decodes) {
        if (request->committed_kv_blocks > 0) resident_decodes.push_back(request);
    }

    std::vector<int> result;
    if (rule == "evict_none") {
        return result;
    } else if (rule == "evict_largest_prefill" && !resident_prefills.empty()) {
        const auto* selected = *std::max_element(
            resident_prefills.begin(), resident_prefills.end(),
            [](const RequestState* left, const RequestState* right) {
                if (left->remaining_prefill_tokens() != right->remaining_prefill_tokens()) {
                    return left->remaining_prefill_tokens() < right->remaining_prefill_tokens();
                }
                return left->request_id > right->request_id;
            });
        result.push_back(selected->request_id);
    } else if (rule == "evict_earliest_prefill_deadline" &&
               !resident_prefills.empty()) {
        const auto* selected = *std::min_element(
            resident_prefills.begin(), resident_prefills.end(),
            [](const RequestState* left, const RequestState* right) {
                if (left->prefill_deadline != right->prefill_deadline) {
                    return left->prefill_deadline < right->prefill_deadline;
                }
                return left->request_id < right->request_id;
            });
        result.push_back(selected->request_id);
    } else if (rule == "evict_prefill_missed_deadline") {
        for (const auto* request : resident_prefills) {
            if (prefill_lateness(*request, state.now) > config.epsilon) {
                result.push_back(request->request_id);
            }
        }
    } else if (rule == "evict_prefill_lateness_over_0p5") {
        for (const auto* request : resident_prefills) {
            if (prefill_lateness(*request, state.now) > 0.5) {
                result.push_back(request->request_id);
            }
        }
    } else if (rule == "evict_longest_decode" && !resident_decodes.empty()) {
        const auto* selected = *std::max_element(
            resident_decodes.begin(), resident_decodes.end(),
            [](const RequestState* left, const RequestState* right) {
                if (left->committed_decode_tokens != right->committed_decode_tokens) {
                    return left->committed_decode_tokens < right->committed_decode_tokens;
                }
                return left->request_id > right->request_id;
            });
        result.push_back(selected->request_id);
    } else if (rule == "evict_decode_lateness_over_0p5") {
        for (const auto* request : resident_decodes) {
            if (decode_lateness(*request) > 0.5) result.push_back(request->request_id);
        }
    } else if (rule == "evict_prefill_highest_lateness" &&
               !resident_prefills.empty()) {
        const auto* selected = *std::max_element(
            resident_prefills.begin(), resident_prefills.end(),
            [&](const RequestState* left, const RequestState* right) {
                const double left_lateness = prefill_lateness(*left, state.now);
                const double right_lateness = prefill_lateness(*right, state.now);
                if (left_lateness != right_lateness) return left_lateness < right_lateness;
                return left->request_id > right->request_id;
            });
        if (prefill_lateness(*selected, state.now) > config.epsilon) {
            result.push_back(selected->request_id);
        }
    } else if (rule == "evict_decode_highest_lateness" &&
               !resident_decodes.empty()) {
        const auto* selected = *std::max_element(
            resident_decodes.begin(), resident_decodes.end(),
            [](const RequestState* left, const RequestState* right) {
                const double left_lateness = decode_lateness(*left);
                const double right_lateness = decode_lateness(*right);
                if (left_lateness != right_lateness) return left_lateness < right_lateness;
                return left->request_id > right->request_id;
            });
        if (decode_lateness(*selected) > config.epsilon) {
            result.push_back(selected->request_id);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<const RequestState*> order_prefill_class(
    std::vector<const RequestState*> requests,
    const std::string& heuristic,
    const State& state,
    const PrefillTimeEstimator& estimator) {
    if (heuristic == "SJF") {
        std::sort(requests.begin(), requests.end(),
                  [](const RequestState* left, const RequestState* right) {
                      if (prefill_class_tokens(*left) !=
                          prefill_class_tokens(*right)) {
                          return prefill_class_tokens(*left) <
                                 prefill_class_tokens(*right);
                      }
                      return left->request_id < right->request_id;
                  });
    } else if (heuristic == "EDF") {
        std::sort(requests.begin(), requests.end(),
                  [](const RequestState* left, const RequestState* right) {
                      if (prefill_class_deadline(*left) !=
                          prefill_class_deadline(*right)) {
                          return prefill_class_deadline(*left) <
                                 prefill_class_deadline(*right);
                      }
                      return left->request_id < right->request_id;
                  });
    } else if (heuristic == "LJF") {
        std::sort(requests.begin(), requests.end(),
                  [](const RequestState* left, const RequestState* right) {
                      if (prefill_class_tokens(*left) !=
                          prefill_class_tokens(*right)) {
                          return prefill_class_tokens(*left) >
                                 prefill_class_tokens(*right);
                      }
                      return left->request_id < right->request_id;
                  });
    } else if (heuristic == "LST") {
        std::unordered_map<int, double> slack;
        for (const RequestState* request : requests) {
            const double duration = estimator(prefill_class_tokens(*request));
            if (!std::isfinite(duration) || duration < 0.0) {
                throw std::runtime_error("prefill estimator returned invalid duration");
            }
            slack[request->request_id] =
                prefill_class_deadline(*request) - state.now - duration;
        }
        std::sort(requests.begin(), requests.end(),
                  [&](const RequestState* left, const RequestState* right) {
                      const double left_slack = slack.at(left->request_id);
                      const double right_slack = slack.at(right->request_id);
                      if (left_slack != right_slack) return left_slack < right_slack;
                      return left->request_id < right->request_id;
                  });
    } else {
        throw std::runtime_error("unsupported prefill ordering heuristic");
    }
    return requests;
}

std::tuple<int, int, bool> fit_prefill_class_to_kv(
    const RequestState& request,
    int desired_tokens,
    int free_blocks,
    int block_size_tokens) {
    const int owned_blocks = request.committed_kv_blocks + request.reserved_kv_blocks;
    const int maximum_resident = (owned_blocks + free_blocks) * block_size_tokens;
    const int tokens = std::min(
        desired_tokens,
        std::max(0, maximum_resident - request.resident_tokens()));
    if (tokens <= 0) return {0, 0, false};
    const bool is_recompute = request.remaining_recompute_tokens() > 0;
    const int blocks = is_recompute
        ? additional_blocks_for_work(
              request, 0, 0, tokens, block_size_tokens)
        : additional_blocks_for_work(
              request, tokens, 0, 0, block_size_tokens);
    return {tokens, blocks, is_recompute};
}

std::optional<ResolvedControllerAction> resolve_controller_raw(
    const State& state,
    const Config& config,
    int raw_index,
    const PrefillTimeEstimator& estimator) {
    auto [preemption_rule, eviction_rule, budget, heuristic] =
        config.controller_actions.components(raw_index);
    if (state.next_player != Player::Controller) return std::nullopt;
    if (!config.request_preemption_enabled) {
        preemption_rule = "preempt_none";
    }

    const auto prefills = waiting_prefills(state);
    const auto decodes = waiting_decodes(state);
    const bool pipeline_open = can_admit_microbatch(
        state.replica, state.now, config);
    if (budget == 0 && heuristic != config.controller_actions.ordering_heuristics.front()) {
        return std::nullopt;
    }

    const std::vector<int> evicted_ids =
        eviction_targets(state, config, eviction_rule, prefills, decodes);
    if (eviction_rule != "evict_none" && evicted_ids.empty()) return std::nullopt;
    const std::vector<int> preempted_ids = preemption_targets(
        state, config, preemption_rule, evicted_ids, estimator);
    if (preemption_rule != "preempt_none" && preempted_ids.empty()) {
        return std::nullopt;
    }

    std::vector<int> pending_preemption_ids;
    std::vector<int> immediate_preemption_ids;
    for (int request_id : preempted_ids) {
        (state.request(request_id).has_inflight_work()
             ? pending_preemption_ids
             : immediate_preemption_ids)
            .push_back(request_id);
    }
    std::vector<int> excluded_ids = evicted_ids;
    excluded_ids.insert(
        excluded_ids.end(), preempted_ids.begin(), preempted_ids.end());
    std::sort(excluded_ids.begin(), excluded_ids.end());
    const std::unordered_set<int> excluded(
        excluded_ids.begin(), excluded_ids.end());

    std::vector<const RequestState*> eligible_prefill_class;
    for (const RequestState* request : prefills) {
        if (!excluded.count(request->request_id) &&
            prefill_class_tokens(*request) > 0) {
            eligible_prefill_class.push_back(request);
        }
    }
    for (const RequestState* request : decodes) {
        if (!excluded.count(request->request_id) &&
            request->remaining_recompute_tokens() > 0) {
            eligible_prefill_class.push_back(request);
        }
    }
    const auto ordered_prefill_work = order_prefill_class(
        std::move(eligible_prefill_class), heuristic, state, estimator);
    int total_prefill_class = 0;
    for (const auto* request : ordered_prefill_work) {
        total_prefill_class += prefill_class_tokens(*request);
    }

    if (!pipeline_open) {
        if (budget != 0 ||
            heuristic != config.controller_actions.ordering_heuristics.front()) {
            return std::nullopt;
        }
    } else if (budget > 0) {
        if (total_prefill_class == 0) return std::nullopt;
        const auto positive = std::find_if(
            config.controller_actions.prefill_budgets.begin(),
            config.controller_actions.prefill_budgets.end(),
            [](int value) { return value > 0; });
        if (positive == config.controller_actions.prefill_budgets.end()) {
            throw std::runtime_error("controller has no positive prefill budget");
        }
        if (budget > total_prefill_class &&
            !(total_prefill_class < *positive && budget == *positive)) {
            return std::nullopt;
        }
    }

    int evicted_blocks = 0;
    for (int request_id : evicted_ids) {
        evicted_blocks += state.request(request_id).committed_kv_blocks;
    }
    int preempted_blocks = 0;
    for (int request_id : immediate_preemption_ids) {
        preempted_blocks += state.request(request_id).committed_kv_blocks;
    }
    const int released_blocks = evicted_blocks + preempted_blocks;
    int free_blocks = free_logical_blocks(state.replica) + released_blocks;
    int tokens_left = config.max_batch_tokens;
    int sequences_left = config.max_sequences;
    int desired_left = std::min(budget, tokens_left);
    std::vector<BatchAllocation> allocations;

    if (pipeline_open) {
        for (const RequestState* request : ordered_prefill_work) {
            if (desired_left <= 0 || sequences_left <= 0) break;
            const int desired = std::min(
                prefill_class_tokens(*request), desired_left);
            const auto [tokens, blocks, is_recompute] =
                fit_prefill_class_to_kv(
                    *request, desired, free_blocks, config.block_size_tokens);
            if (tokens <= 0) continue;
            BatchAllocation allocation;
            allocation.request_id = request->request_id;
            allocation.prefill_tokens = is_recompute ? 0 : tokens;
            allocation.recompute_tokens = is_recompute ? tokens : 0;
            allocation.new_kv_blocks = blocks;
            allocations.push_back(allocation);
            desired_left -= tokens;
            tokens_left -= tokens;
            --sequences_left;
            free_blocks -= blocks;
        }

        std::vector<const RequestState*> zero_block_decodes;
        std::vector<const RequestState*> boundary_decodes;
        for (const RequestState* request : decodes) {
            if (excluded.count(request->request_id) ||
                request->remaining_recompute_tokens() > 0) continue;
            const int blocks = additional_blocks_for_work(
                *request, 0, 1, 0, config.block_size_tokens);
            (blocks == 0 ? zero_block_decodes : boundary_decodes)
                .push_back(request);
        }
        zero_block_decodes.insert(
            zero_block_decodes.end(),
            boundary_decodes.begin(), boundary_decodes.end());
        int funded_decode_slots = state.decode_credits_available;
        for (const RequestState* request : zero_block_decodes) {
            if (funded_decode_slots <= 0 || tokens_left <= 0 ||
                sequences_left <= 0) break;
            const int blocks = additional_blocks_for_work(
                *request, 0, 1, 0, config.block_size_tokens);
            if (blocks > free_blocks) continue;
            allocations.push_back({request->request_id, 0, 1, blocks});
            --funded_decode_slots;
            --tokens_left;
            --sequences_left;
            free_blocks -= blocks;
        }
    }
    std::sort(allocations.begin(), allocations.end(),
              [](const BatchAllocation& left, const BatchAllocation& right) {
                  return left.request_id < right.request_id;
              });
    int reserved_blocks = 0;
    for (const auto& allocation : allocations) reserved_blocks += allocation.new_kv_blocks;

    ControllerTransitionKind kind = ControllerTransitionKind::Wait;
    if (!allocations.empty()) kind = ControllerTransitionKind::Batch;
    else if (!evicted_ids.empty() && !preempted_ids.empty()) {
        kind = ControllerTransitionKind::EvictAndPreempt;
    }
    else if (!evicted_ids.empty()) kind = ControllerTransitionKind::EvictOnly;
    else if (!preempted_ids.empty()) kind = ControllerTransitionKind::PreemptOnly;
    else if (raw_index != 0) return std::nullopt;

    ResolvedControllerAction result;
    result.raw_action_index = raw_index;
    result.preemption_rule = preemption_rule;
    result.eviction_rule = eviction_rule;
    result.prefill_budget = budget;
    result.ordering_heuristic = heuristic;
    result.transition_kind = kind;
    result.evicted_request_ids = evicted_ids;
    result.preempted_request_ids = preempted_ids;
    result.pending_preemption_request_ids = pending_preemption_ids;
    result.allocations = std::move(allocations);
    result.released_kv_blocks = released_blocks;
    result.preempted_kv_blocks = preempted_blocks;
    result.reserved_kv_blocks = reserved_blocks;
    const int net_blocks = reserved_blocks - released_blocks;
    for (int rank_id : state.replica.rank_ids) {
        result.rank_kv_delta.emplace_back(rank_id, net_blocks);
    }
    return result;
}

std::vector<int> stop_ids(const State& state, const std::string& rule) {
    std::vector<const RequestState*> decodes;
    for (const auto& request : state.requests) {
        if (request.lifecycle == RequestLifecycle::WaitingDecode ||
            request.lifecycle == RequestLifecycle::InflightDecode ||
            (request.lifecycle == RequestLifecycle::InflightRecompute &&
             request.is_decode_phase()) ||
            (request.lifecycle == RequestLifecycle::PreemptPending &&
             (request.reserved_decode_tokens > 0 ||
              (request.reserved_recompute_tokens > 0 &&
               request.is_decode_phase())))) {
            decodes.push_back(&request);
        }
    }
    if (rule == "stop_none" || decodes.empty()) return {};
    if (rule == "stop_longest_decode") {
        const auto* selected = *std::max_element(
            decodes.begin(), decodes.end(),
            [](const RequestState* left, const RequestState* right) {
                if (left->committed_decode_tokens != right->committed_decode_tokens) {
                    return left->committed_decode_tokens < right->committed_decode_tokens;
                }
                return left->request_id > right->request_id;
            });
        return {selected->request_id};
    }
    if (rule == "stop_shortest_decode") {
        const auto* selected = *std::min_element(
            decodes.begin(), decodes.end(),
            [](const RequestState* left, const RequestState* right) {
                if (left->committed_decode_tokens != right->committed_decode_tokens) {
                    return left->committed_decode_tokens < right->committed_decode_tokens;
                }
                return left->request_id < right->request_id;
            });
        return {selected->request_id};
    }
    std::vector<int> result;
    const int threshold = rule == "stop_all_decodes_over_512" ? 512 : 216;
    for (const RequestState* request : decodes) {
        if (request->committed_decode_tokens > threshold) result.push_back(request->request_id);
    }
    return result;
}

std::optional<ResolvedAdversaryAction> resolve_adversary_raw(
    const State& state, const Config& config, int raw_index) {
    const auto [launch_count, prefill_tokens, stop_rule] =
        config.adversary_actions.components(raw_index);
    if (state.next_player != Player::Adversary) return std::nullopt;
    if (state.now + config.epsilon < state.next_adversary_tick) {
        if (raw_index != 0) return std::nullopt;
        return ResolvedAdversaryAction{0, 0, 0, "stop_none", {}};
    }

    const std::vector<int> stopped = stop_ids(state, stop_rule);
    if (stop_rule != "stop_none" && stopped.empty()) return std::nullopt;
    const double cutoff = state.now - config.launch_window_sec;
    int used_count = 0;
    int used_prefill = 0;
    for (const auto& record : state.launch_history) {
        if (record.launch_time > cutoff + config.epsilon) {
            used_count += record.request_count;
            used_prefill += record.prefill_tokens;
        }
    }
    if (launch_count > 0) {
        const int prefill_cap = config.target_prefill_tokens_window_average *
                                config.max_requests_per_launch_window;
        if (used_count + launch_count > config.max_requests_per_launch_window ||
            used_prefill + launch_count * prefill_tokens > prefill_cap ||
            static_cast<int>(state.requests.size()) + launch_count > config.max_requests) {
            return std::nullopt;
        }
    }
    return ResolvedAdversaryAction{
        raw_index, launch_count, prefill_tokens, stop_rule, stopped};
}

}  // namespace

const char* transition_kind_name(ControllerTransitionKind kind) {
    switch (kind) {
        case ControllerTransitionKind::Wait: return "WAIT";
        case ControllerTransitionKind::EvictOnly: return "EVICT_ONLY";
        case ControllerTransitionKind::Batch: return "BATCH";
        case ControllerTransitionKind::PreemptOnly: return "PREEMPT_ONLY";
        case ControllerTransitionKind::EvictAndPreempt:
            return "EVICT_AND_PREEMPT";
    }
    throw std::logic_error("unknown controller transition kind");
}

int ResolvedControllerAction::total_prefill_tokens() const {
    int total = 0;
    for (const auto& allocation : allocations) total += allocation.prefill_tokens;
    return total;
}

int ResolvedControllerAction::total_decode_tokens() const {
    int total = 0;
    for (const auto& allocation : allocations) total += allocation.decode_tokens;
    return total;
}

int ResolvedControllerAction::total_recompute_tokens() const {
    int total = 0;
    for (const auto& allocation : allocations) total += allocation.recompute_tokens;
    return total;
}

int ResolvedControllerAction::total_prefill_class_tokens() const {
    return total_prefill_tokens() + total_recompute_tokens();
}

bool ResolvedControllerAction::same_effect(
    const ResolvedControllerAction& other) const {
    return transition_kind == other.transition_kind &&
           evicted_request_ids == other.evicted_request_ids &&
           preempted_request_ids == other.preempted_request_ids &&
           pending_preemption_request_ids ==
               other.pending_preemption_request_ids &&
           allocations == other.allocations && rank_kv_delta == other.rank_kv_delta;
}

bool ResolvedAdversaryAction::same_effect(
    const ResolvedAdversaryAction& other) const {
    return launch_count == other.launch_count &&
           prefill_tokens == other.prefill_tokens &&
           stop_request_ids == other.stop_request_ids;
}

ControllerActionSpace resolve_controller_actions(
    const State& state,
    const Config& config,
    const PrefillTimeEstimator& prefill_time_estimator) {
    ControllerActionSpace space;
    space.raw_to_canonical.assign(config.controller_actions.raw_action_count(), -1);
    for (int raw_index = 0; raw_index < config.controller_actions.raw_action_count();
         ++raw_index) {
        auto resolved = resolve_controller_raw(
            state, config, raw_index, prefill_time_estimator);
        if (!resolved.has_value()) continue;
        int canonical_index = -1;
        for (int index = 0; index < static_cast<int>(space.canonical_actions.size()); ++index) {
            if (space.canonical_actions[index].action.same_effect(*resolved)) {
                canonical_index = index;
                break;
            }
        }
        if (canonical_index < 0) {
            canonical_index = static_cast<int>(space.canonical_actions.size());
            space.canonical_actions.push_back(
                {canonical_index, *resolved, {raw_index}});
        } else {
            space.canonical_actions[canonical_index].equivalent_raw_indices.push_back(raw_index);
        }
        space.raw_to_canonical[raw_index] = canonical_index;
    }
    return space;
}

AdversaryActionSpace resolve_adversary_actions(
    const State& state, const Config& config) {
    AdversaryActionSpace space;
    space.raw_to_canonical.assign(config.adversary_actions.raw_action_count(), -1);
    for (int raw_index = 0; raw_index < config.adversary_actions.raw_action_count();
         ++raw_index) {
        auto resolved = resolve_adversary_raw(state, config, raw_index);
        if (!resolved.has_value()) continue;
        int canonical_index = -1;
        for (int index = 0; index < static_cast<int>(space.canonical_actions.size()); ++index) {
            if (space.canonical_actions[index].action.same_effect(*resolved)) {
                canonical_index = index;
                break;
            }
        }
        if (canonical_index < 0) {
            canonical_index = static_cast<int>(space.canonical_actions.size());
            space.canonical_actions.push_back(
                {canonical_index, *resolved, {raw_index}});
        } else {
            space.canonical_actions[canonical_index].equivalent_raw_indices.push_back(raw_index);
        }
        space.raw_to_canonical[raw_index] = canonical_index;
    }
    return space;
}

}  // namespace gv4
