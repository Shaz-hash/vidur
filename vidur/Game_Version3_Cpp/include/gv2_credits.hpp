#pragma once

#include "gv2_types.hpp"

#include <unordered_map>

namespace mcts_native_gv2 {

class DecodeCreditLedger {
public:
    explicit DecodeCreditLedger(int initial_balance = 0);
    explicit DecodeCreditLedger(GameStats* borrowed_stats);

    void load_from_stats(const GameStats& stats);
    void write_back_stats(GameStats* stats, bool enforce_nonnegative) const;

    void set_balance(int v);
    int raw_balance() const;
    int available_balance(bool enforce_nonnegative) const;

    bool has_decode_entry(int request_id) const;
    void ensure_decode_entry(int request_id);

    // Mint once at prefill->decode transition (caller controls transition predicate).
    bool mint_on_prefill_complete_once(int request_id, int minted_tokens);

    // Consume decode budget and record counted decode tokens.
    int consume_decode(int request_id, int requested_tokens, bool enforce_nonnegative);

    // Record decode tokens without touching credit balance (used when nonnegative enforcement is off).
    void record_decode_without_spend(int request_id, int processed_tokens);

    // Reclaim remaining minted budget when dropped/evicted.
    int reclaim_on_drop(int request_id, int mint_per_request);

    void erase_request(int request_id);

    const std::unordered_map<int, int>& decode_tokens_counted_by_id() const;

private:
    std::unordered_map<int, int>& mutable_decode_tokens_counted_by_id();
    const std::unordered_map<int, int>& current_decode_tokens_counted_by_id() const;

    int balance_;
    GameStats* borrowed_stats_ = nullptr;
    std::unordered_map<int, int> decode_tokens_counted_by_id_;
};

}  // namespace mcts_native_gv2
