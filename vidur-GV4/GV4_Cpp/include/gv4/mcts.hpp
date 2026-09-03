#pragma once

#include "gv4/engine.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace gv4 {

class InferenceRuntime;

struct MCTSConfig {
    int iterations = 1000;
    double puct_c = 1.0;
    bool use_policy_prior = false;
    double policy_prior_temperature = 1.0;
    double prior_min_probability = 1e-8;

    // A zero count keeps the proven zero-bootstrap uniform-MCTS path unchanged.
    int rollout_count = 0;
    double rollout_horizon_sec = 0.4;
    std::uint64_t rollout_seed = 0;
    double rollout_policy_temperature = 1.0;
    double rollout_probability_quantum = 1e-6;
    int rollout_max_actions = 4096;
    bool use_model_bootstrap = false;
};

struct RolloutStats {
    int leaf_evaluations = 0;
    int trajectories = 0;
    int actions = 0;
    int terminal_trajectories = 0;
    int bootstrap_calls = 0;
    int cutoff_leaf_evaluations = 0;
    std::optional<double> root_time;
    std::optional<double> deadline;
    std::optional<double> min_start_time;
    std::optional<double> max_start_time;
    std::optional<double> min_final_time;
    std::optional<double> max_final_time;
    std::optional<double> min_deadline;
    std::optional<double> max_deadline;
    std::optional<double> min_expansion_parent_time;
    std::optional<double> max_expansion_parent_time;
    std::optional<double> min_remaining_rollout_sec;
    std::optional<double> max_remaining_rollout_sec;
    std::optional<std::uint32_t> first_history_hash;
    int first_history_actions = 0;
};

struct MCTSActionEntry {
    int representative_raw_index = 0;
    std::vector<int> equivalent_raw_indices;
    CanonicalAction action;
    double prior = 0.0;
};

struct MCTSNode {
    MCTSNode() = default;
    MCTSNode(const MCTSNode&) = delete;
    MCTSNode& operator=(const MCTSNode&) = delete;
    MCTSNode(MCTSNode&&) = default;
    MCTSNode& operator=(MCTSNode&&) = default;

    Player player = Player::Adversary;
    int node_id = 0;
    int depth = 0;
    MCTSNode* parent = nullptr;
    std::optional<CanonicalAction> parent_action;
    int parent_action_index = kNoId;
    double reward = 0.0;
    int visits = 0;
    double value_sum = 0.0;
    double state_cost = 0.0;
    double sim_time = 0.0;
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    double edge_discount = 1.0;
    State state;
    bool expanded = false;
    std::vector<MCTSActionEntry> actions;
    std::vector<bool> valid_mask;
    std::vector<int> untried_action_indices;
    std::map<int, std::unique_ptr<MCTSNode>> children;

    [[nodiscard]] double mean_value() const;
    [[nodiscard]] const MCTSActionEntry& action_entry(int representative) const;
};

struct RootActionStats {
    int representative_raw_index = 0;
    std::vector<int> equivalent_raw_indices;
    int visits = 0;
    double value_sum = 0.0;
    double mean_value = 0.0;
};

struct MCTSSearchResult {
    int root_node_id = 0;
    Player root_player = Player::Adversary;
    Player next_player = Player::Adversary;
    int best_action_index = kNoId;
    double best_action_value = 0.0;
    std::vector<double> action_values;
    std::vector<bool> valid_mask;
    std::vector<RootActionStats> root_action_stats;
    bool used_bootstrap = false;
    bool used_rollout = false;
    RolloutStats rollout_stats;
};

using IterationObserver = std::function<void(
    int iteration_index,
    const std::vector<const MCTSNode*>& path,
    const MCTSNode& root)>;

class UniformMCTS {
public:
    UniformMCTS(
        Environment environment,
        MCTSConfig config = {},
        const InferenceRuntime* inference = nullptr);

    [[nodiscard]] MCTSSearchResult search(
        const State& root_state,
        Player root_player,
        int root_node_id = 0,
        int root_depth = 0,
        const IterationObserver& observer = {});
    [[nodiscard]] const MCTSNode* root() const { return root_.get(); }

private:
    void ensure_expanded(MCTSNode& node);
    [[nodiscard]] std::vector<MCTSActionEntry> legal_actions(
        const State& state) const;
    void assign_priors(
        const State& state,
        std::vector<MCTSActionEntry>& actions,
        double temperature,
        bool use_policy) const;
    [[nodiscard]] double rollout_value(
        const State& leaf_state,
        Player leaf_player,
        double expansion_parent_time,
        double target_time);
    [[nodiscard]] double one_rollout(
        const State& leaf_state,
        Player leaf_player,
        double target_time,
        std::uint64_t seed,
        bool capture_history);
    [[nodiscard]] double bootstrap_value(
        const State& state,
        Player player) const;
    [[nodiscard]] std::tuple<bool, int, MCTSNode*> select(MCTSNode& node) const;
    [[nodiscard]] MCTSNode& expand(MCTSNode& parent, int representative);
    void backpropagate(const std::vector<MCTSNode*>& path, double leaf_value);
    [[nodiscard]] double normalized_exploitation(
        const MCTSNode& parent,
        const MCTSNode& child) const;
    [[nodiscard]] double explore(
        const MCTSNode& parent,
        int representative,
        int child_visits) const;
    [[nodiscard]] int best_root_action() const;

    Environment environment_;
    MCTSConfig config_;
    const InferenceRuntime* inference_ = nullptr;
    std::unique_ptr<MCTSNode> root_;
    int node_id_counter_ = 0;
    RolloutStats rollout_stats_;
};

}  // namespace gv4
