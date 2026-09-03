#pragma once

#include "gv4/config.hpp"
#include "gv4/state.hpp"

#include <functional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace gv4 {

using PrefillTimeEstimator = std::function<double(int)>;

enum class ControllerTransitionKind : int { Wait = 0, EvictOnly = 1, Batch = 2 };

[[nodiscard]] const char* transition_kind_name(ControllerTransitionKind kind);

struct ResolvedControllerAction {
    int raw_action_index = 0;
    int replica_id = 0;
    std::string eviction_rule;
    int prefill_budget = 0;
    std::string ordering_heuristic;
    ControllerTransitionKind transition_kind = ControllerTransitionKind::Wait;
    std::vector<int> evicted_request_ids;
    std::vector<BatchAllocation> allocations;
    int released_kv_blocks = 0;
    int reserved_kv_blocks = 0;
    std::vector<std::pair<int, int>> rank_kv_delta;

    [[nodiscard]] int total_prefill_tokens() const;
    [[nodiscard]] int total_decode_tokens() const;
    [[nodiscard]] bool same_effect(const ResolvedControllerAction& other) const;
};

struct CanonicalControllerAction {
    int canonical_action_index = 0;
    ResolvedControllerAction action;
    std::vector<int> equivalent_raw_indices;

    [[nodiscard]] int representative_raw_index() const { return action.raw_action_index; }
};

struct ResolvedAdversaryAction {
    int raw_action_index = 0;
    int launch_count = 0;
    int prefill_tokens = 0;  // Zero means no launch/template.
    std::string stop_rule;
    std::vector<int> stop_request_ids;

    [[nodiscard]] bool same_effect(const ResolvedAdversaryAction& other) const;
};

struct CanonicalAdversaryAction {
    int canonical_action_index = 0;
    ResolvedAdversaryAction action;
    std::vector<int> equivalent_raw_indices;

    [[nodiscard]] int representative_raw_index() const { return action.raw_action_index; }
};

using CanonicalAction = std::variant<CanonicalControllerAction, CanonicalAdversaryAction>;

struct ControllerActionSpace {
    std::vector<int> raw_to_canonical;  // -1 denotes a masked raw action.
    std::vector<CanonicalControllerAction> canonical_actions;
};

struct AdversaryActionSpace {
    std::vector<int> raw_to_canonical;
    std::vector<CanonicalAdversaryAction> canonical_actions;
};

[[nodiscard]] ControllerActionSpace resolve_controller_actions(
    const State& state,
    const Config& config,
    const PrefillTimeEstimator& prefill_time_estimator);

[[nodiscard]] AdversaryActionSpace resolve_adversary_actions(
    const State& state,
    const Config& config);

}  // namespace gv4
