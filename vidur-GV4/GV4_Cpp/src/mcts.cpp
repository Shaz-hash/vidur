#include "gv4/mcts.hpp"

#include "gv4/inference.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace gv4 {
namespace {

int representative(const CanonicalAction& action) {
    return std::visit(
        [](const auto& edge) { return edge.representative_raw_index(); }, action);
}

std::vector<int> aliases(const CanonicalAction& action) {
    return std::visit(
        [](const auto& edge) { return edge.equivalent_raw_indices; }, action);
}

State apply(
    const Environment& environment,
    const State& state,
    const CanonicalAction& action) {
    if (const auto* controller = std::get_if<CanonicalControllerAction>(&action)) {
        return environment.apply_controller_action_only(state, *controller, true);
    }
    return environment.apply_adversary_action_only(
        state, std::get<CanonicalAdversaryAction>(action));
}

}  // namespace

double MCTSNode::mean_value() const {
    return visits == 0 ? 0.0 : value_sum / static_cast<double>(visits);
}

const MCTSActionEntry& MCTSNode::action_entry(int representative) const {
    const auto found = std::find_if(
        actions.begin(), actions.end(), [representative](const MCTSActionEntry& item) {
            return item.representative_raw_index == representative;
        });
    if (found == actions.end()) {
        throw std::out_of_range("MCTS representative action is missing");
    }
    return *found;
}

UniformMCTS::UniformMCTS(
    Environment environment,
    MCTSConfig config,
    const InferenceRuntime* inference)
    : environment_(std::move(environment)), config_(config), inference_(inference) {
    if (config_.iterations <= 0) {
        throw std::invalid_argument("MCTS iterations must be positive");
    }
    if (!std::isfinite(config_.puct_c) || config_.puct_c <= 0.0) {
        throw std::invalid_argument("PUCT constant must be positive and finite");
    }
    if (!std::isfinite(config_.policy_prior_temperature) ||
        config_.policy_prior_temperature <= 0.0) {
        throw std::invalid_argument("policy temperature must be positive and finite");
    }
    if (!std::isfinite(config_.prior_min_probability) ||
        config_.prior_min_probability < 0.0) {
        throw std::invalid_argument("minimum prior probability must be finite and nonnegative");
    }
    if (config_.rollout_count < 0) {
        throw std::invalid_argument("rollout count cannot be negative");
    }
    if (!std::isfinite(config_.rollout_horizon_sec) ||
        config_.rollout_horizon_sec < 0.0) {
        throw std::invalid_argument("rollout horizon must be finite and nonnegative");
    }
    if (!std::isfinite(config_.rollout_policy_temperature) ||
        config_.rollout_policy_temperature <= 0.0) {
        throw std::invalid_argument("rollout policy temperature must be positive and finite");
    }
    if (!std::isfinite(config_.rollout_probability_quantum) ||
        config_.rollout_probability_quantum < 0.0) {
        throw std::invalid_argument("rollout probability quantum must be finite and nonnegative");
    }
    if (config_.rollout_max_actions <= 0) {
        throw std::invalid_argument("rollout maximum actions must be positive");
    }
    if ((config_.use_policy_prior || config_.use_model_bootstrap) &&
        inference_ == nullptr) {
        throw std::invalid_argument("policy priors and model bootstrap require inference");
    }
    if (inference_ != nullptr &&
        inference_->config().manifest_sha256 !=
            environment_.config().manifest_sha256) {
        throw std::invalid_argument("MCTS and inference manifests do not match");
    }
}

void UniformMCTS::ensure_expanded(MCTSNode& node) {
    if (node.expanded) {
        return;
    }

    if (node.player == Player::Controller) {
        ControllerActionSpace space = environment_.sample_controller_actions(node.state);
        node.valid_mask.reserve(space.raw_to_canonical.size());
        for (const int value : space.raw_to_canonical) {
            node.valid_mask.push_back(value >= 0);
        }
        node.actions.reserve(space.canonical_actions.size());
        for (auto& edge : space.canonical_actions) {
            CanonicalAction action{std::move(edge)};
            node.actions.push_back(MCTSActionEntry{
                representative(action), aliases(action), std::move(action), 0.0});
        }
    } else {
        AdversaryActionSpace space = environment_.sample_adversary_actions(node.state);
        node.valid_mask.reserve(space.raw_to_canonical.size());
        for (const int value : space.raw_to_canonical) {
            node.valid_mask.push_back(value >= 0);
        }
        node.actions.reserve(space.canonical_actions.size());
        for (auto& edge : space.canonical_actions) {
            CanonicalAction action{std::move(edge)};
            node.actions.push_back(MCTSActionEntry{
                representative(action), aliases(action), std::move(action), 0.0});
        }
    }

    std::sort(
        node.actions.begin(), node.actions.end(),
        [](const MCTSActionEntry& left, const MCTSActionEntry& right) {
            return left.representative_raw_index < right.representative_raw_index;
        });
    assign_priors(
        node.state,
        node.actions,
        config_.policy_prior_temperature,
        config_.use_policy_prior);
    node.untried_action_indices.reserve(node.actions.size());
    for (const MCTSActionEntry& entry : node.actions) {
        node.untried_action_indices.push_back(entry.representative_raw_index);
    }
    node.expanded = true;
}

double UniformMCTS::normalized_exploitation(
    const MCTSNode& parent,
    const MCTSNode& child) const {
    if (!std::isfinite(parent.min_value) ||
        !std::isfinite(parent.max_value) ||
        parent.max_value <= parent.min_value + 1e-12) {
        return 0.5;
    }
    double result = (child.mean_value() - parent.min_value) /
                    (parent.max_value - parent.min_value);
    result = std::clamp(result, 0.0, 1.0);
    if (parent.player == Player::Adversary) {
        result = 1.0 - result;
    }
    return result;
}

double UniformMCTS::explore(
    const MCTSNode& parent,
    int representative,
    int child_visits) const {
    const double prior = std::clamp(
        parent.action_entry(representative).prior, 0.0, 1.0);
    return prior * std::sqrt(static_cast<double>(std::max(1, parent.visits))) /
           (1.0 + static_cast<double>(std::max(0, child_visits)));
}

std::tuple<bool, int, MCTSNode*> UniformMCTS::select(MCTSNode& node) const {
    bool found = false;
    bool selected_child = false;
    int selected_index = kNoId;
    MCTSNode* selected_node = nullptr;
    std::pair<double, int> best{};

    auto consider = [&](double score, int index, MCTSNode* child) {
        const std::pair<double, int> key{score, -index};
        if (!found || key > best) {
            found = true;
            best = key;
            selected_child = child != nullptr;
            selected_index = index;
            selected_node = child;
        }
    };

    for (const auto& [index, child] : node.children) {
        const double score = normalized_exploitation(node, *child) +
                             config_.puct_c * explore(node, index, child->visits);
        consider(score, index, child.get());
    }
    for (const int index : node.untried_action_indices) {
        const double score = 0.5 + config_.puct_c * explore(node, index, 0);
        consider(score, index, nullptr);
    }
    if (!found) {
        throw std::logic_error("PUCT selection has no candidates");
    }
    return {selected_child, selected_index, selected_node};
}

MCTSNode& UniformMCTS::expand(MCTSNode& parent, int representative) {
    const auto untried = std::find(
        parent.untried_action_indices.begin(),
        parent.untried_action_indices.end(),
        representative);
    if (untried == parent.untried_action_indices.end()) {
        throw std::logic_error("MCTS action is not untried");
    }
    parent.untried_action_indices.erase(untried);

    const MCTSActionEntry& entry = parent.action_entry(representative);
    State child_state = apply(environment_, parent.state, entry.action);
    auto child = std::make_unique<MCTSNode>();
    child->player = child_state.next_player;
    child->node_id = ++node_id_counter_;
    child->depth = parent.depth + 1;
    child->parent = &parent;
    child->parent_action = entry.action;
    child->parent_action_index = representative;
    child->state_cost = child_state.objective.total_cost;
    child->sim_time = child_state.now;
    child->reward = parent.state_cost - child->state_cost;
    child->edge_discount = environment_.config().discount_for_elapsed(
        std::max(0.0, child->sim_time - parent.sim_time));
    child->state = std::move(child_state);

    MCTSNode* result = child.get();
    parent.children.emplace(representative, std::move(child));
    return *result;
}

void UniformMCTS::backpropagate(
    const std::vector<MCTSNode*>& path,
    double leaf_value) {
    double value = leaf_value;
    if (value > 0.0) {
        throw std::logic_error("controller-valued bootstrap must not be positive");
    }
    for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
        MCTSNode& node = **iterator;
        if (node.parent != nullptr) {
            value = node.reward + node.edge_discount * value;
        }
        ++node.visits;
        node.value_sum += value;
        if (node.value_sum > 1e-12) {
            throw std::logic_error("controller-valued MCTS sum became positive");
        }
        if (node.parent != nullptr) {
            node.parent->min_value = std::min(node.parent->min_value, value);
            node.parent->max_value = std::max(node.parent->max_value, value);
        }
    }
}

int UniformMCTS::best_root_action() const {
    if (root_ == nullptr || root_->children.empty()) {
        return kNoId;
    }
    const double sign = root_->player == Player::Controller ? 1.0 : -1.0;
    int best_index = kNoId;
    std::tuple<int, double, int> best{};
    bool found = false;
    for (const auto& [index, child] : root_->children) {
        const std::tuple<int, double, int> key{
            child->visits, sign * child->mean_value(), -index};
        if (!found || key > best) {
            found = true;
            best = key;
            best_index = index;
        }
    }
    return best_index;
}

MCTSSearchResult UniformMCTS::search(
    const State& root_state,
    Player root_player,
    int root_node_id,
    int root_depth,
    const IterationObserver& observer) {
    root_state.validate(environment_.config());
    if (root_state.next_player != root_player) {
        throw std::invalid_argument("MCTS root player does not match state turn");
    }

    root_ = std::make_unique<MCTSNode>();
    root_->player = root_player;
    root_->node_id = root_node_id;
    root_->depth = root_depth;
    root_->state = root_state;
    root_->state_cost = root_state.objective.total_cost;
    root_->sim_time = root_state.now;
    node_id_counter_ = std::max(node_id_counter_, root_node_id + 1);
    rollout_stats_ = RolloutStats{};
    rollout_stats_.root_time = root_state.now;
    ensure_expanded(*root_);

    const bool rollout_enabled =
        config_.rollout_count > 0 && config_.rollout_horizon_sec > 0.0;
    for (int iteration = 1; iteration <= config_.iterations; ++iteration) {
        MCTSNode* node = root_.get();
        std::vector<MCTSNode*> path{node};
        bool expanded_leaf = false;
        double expansion_parent_time = node->sim_time;
        while (true) {
            ensure_expanded(*node);
            if (node->children.empty() && node->untried_action_indices.empty()) {
                break;
            }
            const auto [is_child, index, selected] = select(*node);
            if (is_child) {
                node = selected;
                path.push_back(node);
                continue;
            }
            expansion_parent_time = node->sim_time;
            node = &expand(*node, index);
            path.push_back(node);
            expanded_leaf = true;
            break;
        }

        double leaf_value = 0.0;
        if (rollout_enabled) {
            const double target_time = expanded_leaf
                ? expansion_parent_time + config_.rollout_horizon_sec
                : node->sim_time;
            leaf_value = rollout_value(
                node->state, node->player, expansion_parent_time, target_time);
        } else {
            leaf_value = bootstrap_value(node->state, node->player);
        }
        backpropagate(path, leaf_value);
        if (observer) {
            std::vector<const MCTSNode*> view(path.begin(), path.end());
            observer(iteration, view, *root_);
        }
    }

    MCTSSearchResult result;
    result.root_node_id = root_->node_id;
    result.root_player = root_->player;
    result.next_player = root_->player;
    result.best_action_index = best_root_action();
    result.valid_mask = root_->valid_mask;
    result.used_bootstrap = config_.use_model_bootstrap;
    result.used_rollout = rollout_enabled;
    result.rollout_stats = rollout_stats_;
    result.action_values.assign(
        root_->valid_mask.size(), -std::numeric_limits<double>::infinity());
    for (const auto& [index, child] : root_->children) {
        const MCTSActionEntry& entry = root_->action_entry(index);
        for (const int alias : entry.equivalent_raw_indices) {
            result.action_values.at(static_cast<std::size_t>(alias)) = child->mean_value();
        }
        result.root_action_stats.push_back(RootActionStats{
            index,
            entry.equivalent_raw_indices,
            child->visits,
            child->value_sum,
            child->mean_value()});
    }
    if (result.best_action_index != kNoId) {
        const MCTSNode& child = *root_->children.at(result.best_action_index);
        result.best_action_value = child.mean_value();
        result.next_player = child.player;
    }
    return result;
}

}  // namespace gv4
