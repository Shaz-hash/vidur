#include "gv2_credits.hpp"

#include <algorithm>

namespace mcts_native_gv2 {

DecodeCreditLedger::DecodeCreditLedger(int initial_balance)
    : balance_(initial_balance) {}

DecodeCreditLedger::DecodeCreditLedger(GameStats* borrowed_stats)
    : balance_(borrowed_stats != nullptr
          ? static_cast<int>(borrowed_stats->decode_credit_balance)
          : 0),
      borrowed_stats_(borrowed_stats) {}

std::unordered_map<int, int>&
DecodeCreditLedger::mutable_decode_tokens_counted_by_id() {
    return borrowed_stats_ != nullptr
        ? borrowed_stats_->decode_tokens_counted_by_id
        : decode_tokens_counted_by_id_;
}

const std::unordered_map<int, int>&
DecodeCreditLedger::current_decode_tokens_counted_by_id() const {
    return borrowed_stats_ != nullptr
        ? borrowed_stats_->decode_tokens_counted_by_id
        : decode_tokens_counted_by_id_;
}

void DecodeCreditLedger::load_from_stats(const GameStats& stats) {
    borrowed_stats_ = nullptr;
    balance_ = int(stats.decode_credit_balance);
    decode_tokens_counted_by_id_ = stats.decode_tokens_counted_by_id;
}

void DecodeCreditLedger::write_back_stats(
    GameStats* stats,
    bool enforce_nonnegative) const {
    if (stats == nullptr) return;
    stats->decode_credit_balance = balance_;
    stats->decode_credit_available = available_balance(enforce_nonnegative);
    if (borrowed_stats_ != stats) {
        stats->decode_tokens_counted_by_id =
            current_decode_tokens_counted_by_id();
    }
}

void DecodeCreditLedger::set_balance(int v) { balance_ = v; }

int DecodeCreditLedger::raw_balance() const { return balance_; }

int DecodeCreditLedger::available_balance(bool enforce_nonnegative) const {
    if (!enforce_nonnegative) return balance_;
    return std::max(0, balance_);
}

bool DecodeCreditLedger::has_decode_entry(int request_id) const {
    const auto& counted = current_decode_tokens_counted_by_id();
    return counted.find(int(request_id)) != counted.end();
}

void DecodeCreditLedger::ensure_decode_entry(int request_id) {
    (void)mutable_decode_tokens_counted_by_id()[int(request_id)];
}

bool DecodeCreditLedger::mint_on_prefill_complete_once(int request_id, int minted_tokens) {
    if (minted_tokens <= 0) {
        ensure_decode_entry(request_id);
        return !has_decode_entry(request_id);
    }
    const int rid = int(request_id);
    if (has_decode_entry(rid)) return false;
    mutable_decode_tokens_counted_by_id()[rid] = 0;
    balance_ += int(minted_tokens);
    return true;
}

int DecodeCreditLedger::consume_decode(int request_id, int requested_tokens, bool enforce_nonnegative) {
    const int rid = int(request_id);
    const int req = std::max(0, requested_tokens);
    if (req <= 0) return 0;

    int allowed = req;
    if (enforce_nonnegative) {
        allowed = std::min(req, std::max(0, balance_));
    }
    if (allowed <= 0) return 0;

    if (enforce_nonnegative) {
        balance_ -= allowed;
    }
    auto& counted = mutable_decode_tokens_counted_by_id();
    counted[rid] = int(counted[rid]) + allowed;
    return allowed;
}

void DecodeCreditLedger::record_decode_without_spend(int request_id, int processed_tokens) {
    const int rid = int(request_id);
    const int n = std::max(0, processed_tokens);
    if (n <= 0) return;
    auto& counted = mutable_decode_tokens_counted_by_id();
    counted[rid] = int(counted[rid]) + n;
}

int DecodeCreditLedger::reclaim_on_drop(int request_id, int mint_per_request) {
    const int rid = int(request_id);
    auto& counted_by_id = mutable_decode_tokens_counted_by_id();
    const auto it = counted_by_id.find(rid);
    if (it == counted_by_id.end()) {
        return 0;
    }

    const int counted = std::max(0, int(it->second));
    const int reclaim = std::max(0, int(mint_per_request) - counted);
    if (reclaim > 0) {
        balance_ -= reclaim;
    }
    counted_by_id.erase(it);
    return reclaim;
}

void DecodeCreditLedger::erase_request(int request_id) {
    mutable_decode_tokens_counted_by_id().erase(int(request_id));
}

const std::unordered_map<int, int>& DecodeCreditLedger::decode_tokens_counted_by_id() const {
    return current_decode_tokens_counted_by_id();
}

}  // namespace mcts_native_gv2
