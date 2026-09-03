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

std::vector<const RequestState*> order_prefills(
    std::vector<const RequestState*> requests,
    const std::string& heuristic,
    const State& state,
    const PrefillTimeEstimator& estimator) {
    if (heuristic == "SJF") {
        std::sort(requests.begin(), requests.end(),
                  [](const RequestState* left, const RequestState* right) {
                      if (left->remaining_prefill_tokens() !=
                          right->remaining_prefill_tokens()) {
                          return left->remaining_prefill_tokens() <
                                 right->remaining_prefill_tokens();
                      }
                      return left->request_id < right->request_id;
                  });
    } else if (heuristic == "EDF") {
        std::sort(requests.begin(), requests.end(),
                  [](const RequestState* left, const RequestState* right) {
                      if (left->prefill_deadline != right->prefill_deadline) {
                          return left->prefill_deadline < right->prefill_deadline;
                      }
                      return left->request_id < right->request_id;
                  });
    } else if (heuristic == "LJF") {
        std::sort(requests.begin(), requests.end(),
                  [](const RequestState* left, const RequestState* right) {
                      if (left->remaining_prefill_tokens() !=
                          right->remaining_prefill_tokens()) {
                          return left->remaining_prefill_tokens() >
                                 right->remaining_prefill_tokens();
                      }
                      return left->request_id < right->request_id;
                  });
    } else if (heuristic == "LST") {
        std::unordered_map<int, double> slack;
        for (const RequestState* request : requests) {
            const double duration = estimator(request->remaining_prefill_tokens());
            if (!std::isfinite(duration) || duration < 0.0) {
                throw std::runtime_error("prefill estimator returned invalid duration");
            }
            slack[request->request_id] =
                request->prefill_deadline - state.now - duration;
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

std::pair<int, int> fit_prefill_to_kv(
    const RequestState& request,
    int desired_tokens,
    int free_blocks,
    int block_size_tokens) {
    const int owned_blocks = request.committed_kv_blocks + request.reserved_kv_blocks;
    const int maximum_resident = (owned_blocks + free_blocks) * block_size_tokens;
    const int tokens = std::min(
        desired_tokens,
        std::max(0, maximum_resident - request.resident_tokens()));
    if (tokens <= 0) return {0, 0};
    return {tokens, additional_blocks_for_work(
                        request, tokens, 0, block_size_tokens)};
}

std::optional<ResolvedControllerAction> resolve_controller_raw(
    const State& state,
    const Config& config,
    int raw_index,
    const PrefillTimeEstimator& estimator) {
    const auto [eviction_rule, budget, heuristic] =
        config.controller_actions.components(raw_index);
    if (state.next_player != Player::Controller) return std::nullopt;

    if (!can_admit_microbatch(state.replica, state.now, config)) {
        if (raw_index != 0) return std::nullopt;
        ResolvedControllerAction action;
        action.raw_action_index = 0;
        action.eviction_rule = eviction_rule;
        action.prefill_budget = budget;
        action.ordering_heuristic = heuristic;
        action.rank_kv_delta.reserve(state.replica.rank_ids.size());
        for (int rank_id : state.replica.rank_ids) action.rank_kv_delta.emplace_back(rank_id, 0);
        return action;
    }

    const auto prefills = waiting_prefills(state);
    const auto decodes = waiting_decodes(state);
    if (prefills.empty() && decodes.empty() && raw_index != 0) return std::nullopt;
    if (budget == 0 && heuristic != config.controller_actions.ordering_heuristics.front()) {
        return std::nullopt;
    }

    const std::vector<int> evicted_ids =
        eviction_targets(state, config, eviction_rule, prefills, decodes);
    if (eviction_rule != "evict_none" && evicted_ids.empty()) return std::nullopt;
    const std::unordered_set<int> evicted(evicted_ids.begin(), evicted_ids.end());

    std::vector<const RequestState*> eligible_prefills;
    for (const RequestState* request : prefills) {
        if (!evicted.count(request->request_id)) eligible_prefills.push_back(request);
    }
    const auto ordered_prefills =
        order_prefills(std::move(eligible_prefills), heuristic, state, estimator);
    int total_prefill = 0;
    for (const auto* request : ordered_prefills) {
        total_prefill += request->remaining_prefill_tokens();
    }
    if (budget > 0) {
        if (total_prefill == 0) return std::nullopt;
        const auto positive = std::find_if(
            config.controller_actions.prefill_budgets.begin(),
            config.controller_actions.prefill_budgets.end(),
            [](int value) { return value > 0; });
        if (positive == config.controller_actions.prefill_budgets.end()) {
            throw std::runtime_error("controller has no positive prefill budget");
        }
        if (budget > total_prefill && !(total_prefill < *positive && budget == *positive)) {
            return std::nullopt;
        }
    }

    int released_blocks = 0;
    for (int request_id : evicted_ids) {
        released_blocks += state.request(request_id).committed_kv_blocks;
    }
    int free_blocks = free_logical_blocks(state.replica) + released_blocks;
    int tokens_left = config.max_batch_tokens;
    int sequences_left = config.max_sequences;
    int desired_left = std::min(budget, tokens_left);
    std::vector<BatchAllocation> allocations;

    for (const RequestState* request : ordered_prefills) {
        if (desired_left <= 0 || sequences_left <= 0) break;
        const int desired = std::min(request->remaining_prefill_tokens(), desired_left);
        const auto [tokens, blocks] = fit_prefill_to_kv(
            *request, desired, free_blocks, config.block_size_tokens);
        if (tokens <= 0) continue;
        allocations.push_back({request->request_id, tokens, 0, blocks});
        desired_left -= tokens;
        tokens_left -= tokens;
        --sequences_left;
        free_blocks -= blocks;
    }

    std::vector<const RequestState*> zero_block_decodes;
    std::vector<const RequestState*> boundary_decodes;
    for (const RequestState* request : decodes) {
        if (evicted.count(request->request_id)) continue;
        const int blocks = additional_blocks_for_work(
            *request, 0, 1, config.block_size_tokens);
        (blocks == 0 ? zero_block_decodes : boundary_decodes).push_back(request);
    }
    zero_block_decodes.insert(zero_block_decodes.end(),
                              boundary_decodes.begin(), boundary_decodes.end());
    int funded_decode_slots = state.decode_credits_available;
    for (const RequestState* request : zero_block_decodes) {
        if (funded_decode_slots <= 0 || tokens_left <= 0 || sequences_left <= 0) break;
        const int blocks = additional_blocks_for_work(
            *request, 0, 1, config.block_size_tokens);
        if (blocks > free_blocks) continue;
        allocations.push_back({request->request_id, 0, 1, blocks});
        --funded_decode_slots;
        --tokens_left;
        --sequences_left;
        free_blocks -= blocks;
    }
    std::sort(allocations.begin(), allocations.end(),
              [](const BatchAllocation& left, const BatchAllocation& right) {
                  return left.request_id < right.request_id;
              });
    int reserved_blocks = 0;
    for (const auto& allocation : allocations) reserved_blocks += allocation.new_kv_blocks;

    ControllerTransitionKind kind = ControllerTransitionKind::Wait;
    if (!allocations.empty()) kind = ControllerTransitionKind::Batch;
    else if (!evicted_ids.empty()) kind = ControllerTransitionKind::EvictOnly;
    else if (raw_index != 0) return std::nullopt;

    ResolvedControllerAction result;
    result.raw_action_index = raw_index;
    result.eviction_rule = eviction_rule;
    result.prefill_budget = budget;
    result.ordering_heuristic = heuristic;
    result.transition_kind = kind;
    result.evicted_request_ids = evicted_ids;
    result.allocations = std::move(allocations);
    result.released_kv_blocks = released_blocks;
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
            request.lifecycle == RequestLifecycle::InflightDecode) {
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

bool ResolvedControllerAction::same_effect(
    const ResolvedControllerAction& other) const {
    return transition_kind == other.transition_kind &&
           evicted_request_ids == other.evicted_request_ids &&
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
