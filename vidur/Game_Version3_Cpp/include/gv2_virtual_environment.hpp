#pragma once

#include "gv2_credits.hpp"
#include "gv2_player_sample_actions.hpp"
#include "gv2_types.hpp"
#include "virtual_simulator.hpp"

#include <unordered_set>
#include <string>
#include <utility>
#include <vector>

namespace mcts_native_gv2 {

struct GV2EnvConfig {
    double adversary_tick_sec = 0.2;
    double launch_window_sec = 1.0;
    int max_requests_per_launch_window = 7;
    int prefill_window_cap_tokens = 7 * 1024;

    int max_prefill_tokens_per_request = 4096;
    int max_decode_tokens_per_request = 864;
    int min_decode_tokens_per_request = 1;

    double decode_slo_time_default = 0.05;
    double auto_drop_lateness_sec = 2.0;
    double drop_cost = 3.0;

    double eps = 1e-9;
    int time_round_digits = 10;

    bool controller_noop_prefill_only_jump_to_next_adv_tick = true;
    bool enforce_nonnegative_decode_credits = true;
    int decode_credit_mint_per_prefill_complete = 216;

    AdversarySamplerConfig adversary_sampler;
    ControllerSamplerConfig controller_sampler;
};

class GV2VirtualEnvironment {
public:
    explicit GV2VirtualEnvironment(GV2EnvConfig cfg);
    GV2VirtualEnvironment(GV2EnvConfig cfg, VirtualSimulatorConfig sim_cfg);

    const GV2EnvConfig& cfg() const;

    VirtualSimulatorGV2& virtual_simulator();
    const VirtualSimulatorGV2& virtual_simulator() const;

    bool load_predictor_csv(const std::string& path);
    void set_prefill_profile(std::vector<int> tokens, std::vector<double> times);

    SampledActionSet<AdversaryAction> sample_adversary_actions(
        const SimState& state,
        const std::unordered_set<int>& forbidden_stop_ids = {},
        bool compact_valid_only = false,
        bool compact_request_materialization = false) const;
    SampledActionSet<ControllerAction> sample_controller_actions(
        const SimState& state,
        bool compact_valid_only = false,
        bool canonical_compact_only = false) const;
    SampledActionSet<RolloutControllerAction> sample_controller_rollout_actions(
        const SimState& state) const;

    void apply_adversary_action_inplace(SimState& state, const AdversaryAction& action) const;
    void apply_controller_action_inplace(
        SimState& state,
        const ControllerAction& action,
        bool fast_forward = true) const;

    static std::pair<int, double> evaluate_objective(const SimState& state);

private:
    GV2EnvConfig cfg_;
    mutable VirtualSimulatorGV2 virtual_sim_;

    void refresh_sampler_configs();
    double round_time(double t) const;
    double quantize_down(double t) const;
    void init_clock_if_needed(SimState& state) const;
    double current_adv_tick(SimState& state) const;
    double next_adv_tick_state(SimState& state) const;
    bool has_pending_adv_tick(SimState& state) const;

    static bool id_in_sorted(const std::vector<int>& sorted_ids, int rid);
    static void add_sorted_unique(std::vector<int>* sorted_ids, int rid);
    static void remove_if_present(std::vector<int>* sorted_ids, int rid);

    static bool has_active_prefill(const SimState& s);
    static bool has_active_decode(const SimState& s);
    static bool is_controller_strict_noop(const ControllerAction& a);
    static double next_adv_tick(double sim_time, double tick_sec, double eps);

    static RequestState* find_request(SimState& s, int request_id);
    static const RequestState* find_request_const(const SimState& s, int request_id);

    void prune_recent_launches(SimState& state, double anchor_time) const;
    std::pair<int, int> window_usage(const SimState& state, double anchor_time) const;
    void append_launch_event(SimState& state, double ts, int count, int prefill_tokens) const;

    void apply_controller_eviction_rule(SimState& state, const ControllerAction& action, DecodeCreditLedger* ledger) const;
    void drop_request(SimState& state, int rid, DecodeCreditLedger* ledger) const;
    void enforce_decode_caps(SimState& state) const;
    void finalize_decodes_to_credit_budget(SimState& state, DecodeCreditLedger* ledger) const;

    void apply_batch_progress(
        SimState& state,
        const ControllerBatchPlan& plan,
        double batch_start,
        double batch_end,
        DecodeCreditLedger* ledger) const;

    void refresh_request_and_stats_post_step(
        SimState& state,
        DecodeCreditLedger* ledger) const;

    bool maybe_fast_forward_decode_only_to_next_adv_tick(
        SimState& state,
        DecodeCreditLedger* ledger) const;

    static void rebuild_active_completed_ids(SimState& s);
};

}  // namespace mcts_native_gv2
