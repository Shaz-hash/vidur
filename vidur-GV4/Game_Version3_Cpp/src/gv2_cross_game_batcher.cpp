#include "gv2_cross_game_batcher.hpp"

#include "gv2_mcts_dnn.hpp"
#include "gv2_virtual_environment.hpp"
#include "new_features_226_inference.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace mcts_native_gv2 {
namespace {

using Clock = std::chrono::steady_clock;

struct PolicyRequest {
    const NativeHGBModelRuntime* runtime = nullptr;
    std::string player;
    std::vector<MarkovValueFeatures> states;
    std::vector<float> actions;
    int num_rows = 0;
    std::vector<int> offsets;
    std::promise<std::vector<double>> promise;
};

struct ValueRequest {
    const NewFeatures226HGBRuntime* runtime = nullptr;
    std::vector<SimState> states;
    std::vector<const VirtualSimulatorGV2*> simulators;
    std::promise<std::vector<double>> promise;
};

class CrossGameInferenceBatcherImpl final : public CrossGameInferenceBatcher {
public:
    CrossGameInferenceBatcherImpl(
        int inference_threads,
        int max_batch_requests,
        int max_batch_wait_us)
        : inference_threads_(std::max(1, inference_threads)),
          max_batch_requests_(std::max(1, max_batch_requests)),
          max_batch_wait_us_(std::max(0, max_batch_wait_us)),
          server_(&CrossGameInferenceBatcherImpl::run, this) {}

    ~CrossGameInferenceBatcherImpl() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        if (server_.joinable()) server_.join();
    }

    std::vector<double> predict_markov_policy(
        const NativeHGBModelRuntime& runtime,
        const std::string& player,
        const std::vector<MarkovValueFeatures>& markov_states,
        const std::vector<float>& flat_actions,
        int num_rows,
        const std::vector<int>& group_offsets) override {
        auto request = std::make_shared<PolicyRequest>();
        request->runtime = &runtime;
        request->player = player;
        request->states = markov_states;
        request->actions = flat_actions;
        request->num_rows = num_rows;
        request->offsets = group_offsets;
        std::future<std::vector<double>> future = request->promise.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                throw std::runtime_error("cross-game inference batcher is stopping");
            }
            policy_queue_.push_back(std::move(request));
        }
        cv_.notify_one();
        return future.get();
    }

    std::vector<double> infer_values(
        const NewFeatures226HGBRuntime& runtime,
        const std::vector<SimState>& states,
        const std::vector<const VirtualSimulatorGV2*>& simulators) override {
        if (states.size() != simulators.size()) {
            throw std::invalid_argument(
                "cross-game value states/simulators size mismatch");
        }
        auto request = std::make_shared<ValueRequest>();
        request->runtime = &runtime;
        request->states = states;
        request->simulators = simulators;
        std::future<std::vector<double>> future = request->promise.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                throw std::runtime_error("cross-game inference batcher is stopping");
            }
            value_queue_.push_back(std::move(request));
        }
        cv_.notify_one();
        return future.get();
    }

    CrossGameBatchStats stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }

private:
    void run() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] {
                return stopping_ || !policy_queue_.empty() ||
                    !value_queue_.empty();
            });
            if (stopping_ && policy_queue_.empty() && value_queue_.empty()) {
                return;
            }

            if (max_batch_wait_us_ > 0) {
                const auto deadline = Clock::now() +
                    std::chrono::microseconds(max_batch_wait_us_);
                cv_.wait_until(lock, deadline, [&] {
                    return stopping_ ||
                        static_cast<int>(
                            policy_queue_.size() + value_queue_.size()) >=
                            max_batch_requests_;
                });
            }

            if (!policy_queue_.empty()) {
                const NativeHGBModelRuntime* runtime =
                    policy_queue_.front()->runtime;
                const std::string player = policy_queue_.front()->player;
                std::vector<std::shared_ptr<PolicyRequest>> batch;
                for (auto it = policy_queue_.begin();
                     it != policy_queue_.end() &&
                     static_cast<int>(batch.size()) < max_batch_requests_;) {
                    if ((*it)->runtime == runtime && (*it)->player == player) {
                        batch.push_back(*it);
                        it = policy_queue_.erase(it);
                    } else {
                        ++it;
                    }
                }
                lock.unlock();
                process_policy_batch(batch);
                continue;
            }

            const NewFeatures226HGBRuntime* runtime =
                value_queue_.front()->runtime;
            std::vector<std::shared_ptr<ValueRequest>> batch;
            for (auto it = value_queue_.begin();
                 it != value_queue_.end() &&
                 static_cast<int>(batch.size()) < max_batch_requests_;) {
                if ((*it)->runtime == runtime) {
                    batch.push_back(*it);
                    it = value_queue_.erase(it);
                } else {
                    ++it;
                }
            }
            lock.unlock();
            process_value_batch(batch);
        }
    }

    void process_policy_batch(
        const std::vector<std::shared_ptr<PolicyRequest>>& batch) {
        if (batch.empty()) return;
        try {
            std::vector<MarkovValueFeatures> states;
            std::vector<float> actions;
            std::vector<int> offsets = {0};
            std::vector<int> row_boundaries = {0};
            int total_rows = 0;
            for (const auto& request : batch) {
                if (request->runtime == nullptr ||
                    request->offsets.empty() ||
                    request->offsets.front() != 0 ||
                    request->offsets.back() != request->num_rows ||
                    request->states.size() + 1u != request->offsets.size()) {
                    throw std::runtime_error(
                        "invalid cross-game Markov policy request");
                }
                states.insert(
                    states.end(),
                    request->states.begin(),
                    request->states.end());
                actions.insert(
                    actions.end(),
                    request->actions.begin(),
                    request->actions.end());
                for (std::size_t index = 1;
                     index < request->offsets.size();
                     ++index) {
                    offsets.push_back(
                        total_rows + request->offsets[index]);
                }
                total_rows += request->num_rows;
                row_boundaries.push_back(total_rows);
            }

            const auto started = Clock::now();
            const std::vector<double> scores =
                batch.front()->runtime->predict_markov_policy_grouped_batch(
                    states,
                    actions,
                    total_rows,
                    offsets,
                    inference_threads_);
            const double elapsed = std::chrono::duration_cast<
                std::chrono::duration<double>>(Clock::now() - started).count();
            if (scores.size() != static_cast<std::size_t>(total_rows)) {
                throw std::runtime_error(
                    "cross-game policy batch returned wrong row count");
            }

            for (std::size_t index = 0; index < batch.size(); ++index) {
                const int begin = row_boundaries[index];
                const int end = row_boundaries[index + 1];
                batch[index]->promise.set_value(std::vector<double>(
                    scores.begin() + begin, scores.begin() + end));
            }
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.policy_requests +=
                static_cast<std::int64_t>(batch.size());
            ++stats_.policy_batches;
            stats_.policy_action_rows += total_rows;
            stats_.max_policy_batch_requests = std::max(
                stats_.max_policy_batch_requests, batch.size());
            stats_.policy_inference_sec += elapsed;
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            for (const auto& request : batch) {
                request->promise.set_exception(error);
            }
        }
    }

    void process_value_batch(
        const std::vector<std::shared_ptr<ValueRequest>>& batch) {
        if (batch.empty()) return;
        try {
            std::vector<SimState> states;
            std::vector<const VirtualSimulatorGV2*> simulators;
            std::vector<std::size_t> boundaries = {0};
            for (const auto& request : batch) {
                if (request->runtime == nullptr ||
                    request->states.size() != request->simulators.size()) {
                    throw std::runtime_error(
                        "invalid cross-game value request");
                }
                states.insert(
                    states.end(),
                    request->states.begin(),
                    request->states.end());
                simulators.insert(
                    simulators.end(),
                    request->simulators.begin(),
                    request->simulators.end());
                boundaries.push_back(states.size());
            }

            std::vector<double> values(states.size(), 0.0);
            const auto started = Clock::now();
            const int state_count = static_cast<int>(states.size());
            #pragma omp parallel for if(state_count > 1) \
                num_threads(inference_threads_) schedule(static)
            for (int index = 0; index < state_count; ++index) {
                values[static_cast<std::size_t>(index)] =
                    batch.front()->runtime->infer_value(
                        states[static_cast<std::size_t>(index)],
                        simulators[static_cast<std::size_t>(index)],
                        -1);
            }
            const double elapsed = std::chrono::duration_cast<
                std::chrono::duration<double>>(Clock::now() - started).count();

            for (std::size_t index = 0; index < batch.size(); ++index) {
                batch[index]->promise.set_value(std::vector<double>(
                    values.begin() + static_cast<std::ptrdiff_t>(boundaries[index]),
                    values.begin() + static_cast<std::ptrdiff_t>(boundaries[index + 1])));
            }
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.value_requests +=
                static_cast<std::int64_t>(batch.size());
            ++stats_.value_batches;
            stats_.value_states +=
                static_cast<std::int64_t>(states.size());
            stats_.max_value_batch_requests = std::max(
                stats_.max_value_batch_requests, batch.size());
            stats_.value_inference_sec += elapsed;
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            for (const auto& request : batch) {
                request->promise.set_exception(error);
            }
        }
    }

    int inference_threads_;
    int max_batch_requests_;
    int max_batch_wait_us_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
    std::deque<std::shared_ptr<PolicyRequest>> policy_queue_;
    std::deque<std::shared_ptr<ValueRequest>> value_queue_;
    std::thread server_;
    mutable std::mutex stats_mutex_;
    CrossGameBatchStats stats_;
};

}  // namespace

CrossGameBatchResult run_search_hgb226_value_prior_cross_game_batch(
    const std::vector<SearchInput>& inputs,
    const GV2VirtualEnvironment& env,
    NewFeatures226HGBRuntime& infer_runtime,
    NativeHGBModelRuntime& controller_prior_runtime,
    NativeHGBModelRuntime& adversary_prior_runtime,
    int worker_threads,
    int inference_threads,
    int max_batch_requests,
    int max_batch_wait_us) {
    CrossGameBatchResult result;
    result.outputs.resize(inputs.size());
    if (inputs.empty()) return result;

    CrossGameInferenceBatcherImpl batcher(
        inference_threads, max_batch_requests, max_batch_wait_us);
    const int workers = std::max(
        1, std::min(worker_threads, static_cast<int>(inputs.size())));
    std::atomic<std::size_t> next_index{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));

    const auto started = Clock::now();
    for (int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&] {
            while (!failed.load(std::memory_order_relaxed)) {
                const std::size_t index =
                    next_index.fetch_add(1, std::memory_order_relaxed);
                if (index >= inputs.size()) return;
                try {
                    SearchInput input = inputs[index];
                    input.cross_game_inference_batcher = &batcher;
                    result.outputs[index] =
                        run_search_hgb226_value_prior_with_env(
                            input,
                            const_cast<GV2VirtualEnvironment&>(env),
                            infer_runtime,
                            controller_prior_runtime,
                            adversary_prior_runtime);
                } catch (...) {
                    failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (first_error == nullptr) {
                        first_error = std::current_exception();
                    }
                    return;
                }
            }
        });
    }
    for (std::thread& thread : threads) thread.join();
    result.elapsed_sec = std::chrono::duration_cast<
        std::chrono::duration<double>>(Clock::now() - started).count();
    result.stats = batcher.stats();
    if (first_error != nullptr) std::rethrow_exception(first_error);
    return result;
}

CrossGameBatchResult run_search_hgb226_value_prior_parallel_baseline(
    const std::vector<SearchInput>& inputs,
    const GV2VirtualEnvironment& env,
    NewFeatures226HGBRuntime& infer_runtime,
    NativeHGBModelRuntime& controller_prior_runtime,
    NativeHGBModelRuntime& adversary_prior_runtime,
    int worker_threads) {
    CrossGameBatchResult result;
    result.outputs.resize(inputs.size());
    if (inputs.empty()) return result;

    const int workers = std::max(
        1, std::min(worker_threads, static_cast<int>(inputs.size())));
    std::atomic<std::size_t> next_index{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers));

    const auto started = Clock::now();
    for (int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&] {
            while (!failed.load(std::memory_order_relaxed)) {
                const std::size_t index =
                    next_index.fetch_add(1, std::memory_order_relaxed);
                if (index >= inputs.size()) return;
                try {
                    SearchInput input = inputs[index];
                    input.cross_game_inference_batcher = nullptr;
                    result.outputs[index] =
                        run_search_hgb226_value_prior_with_env(
                            input,
                            const_cast<GV2VirtualEnvironment&>(env),
                            infer_runtime,
                            controller_prior_runtime,
                            adversary_prior_runtime);
                } catch (...) {
                    failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (first_error == nullptr) {
                        first_error = std::current_exception();
                    }
                    return;
                }
            }
        });
    }
    for (std::thread& thread : threads) thread.join();
    result.elapsed_sec = std::chrono::duration_cast<
        std::chrono::duration<double>>(Clock::now() - started).count();
    if (first_error != nullptr) std::rethrow_exception(first_error);
    return result;
}

}  // namespace mcts_native_gv2
