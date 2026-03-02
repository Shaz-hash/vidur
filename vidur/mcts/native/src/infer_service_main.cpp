#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <deque>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include "infer_shared.hpp"

namespace {

constexpr uint8_t CMD_PING = 1;
constexpr uint8_t CMD_SHUTDOWN = 2;
constexpr uint8_t CMD_LOAD_MODELS = 3;
constexpr uint8_t CMD_INFER = 4;

constexpr float kValueRealMin = -50.0f;
constexpr float kValueRealMax = 0.0f;
constexpr float kValueLinearRealMin = -48.0f;
constexpr float kValueNormMin = -1.0f;
constexpr float kValueNormMax = 0.0f;
constexpr float kValueLinearNormMin = -0.98f;
constexpr float kValueTailPower = 2.0f;

torch::Tensor denormalize_value_model(torch::Tensor value_norm) {
    value_norm = torch::clamp(value_norm, kValueNormMin, kValueNormMax);

    const float linear_scale = std::abs(kValueLinearNormMin) / std::abs(kValueLinearRealMin);
    torch::Tensor x_linear = value_norm / linear_scale;

    const float tail_real_span = kValueLinearRealMin - kValueRealMin;  // 2.0
    const float tail_norm_span = kValueLinearNormMin - kValueNormMin;  // 0.02
    torch::Tensor t = torch::clamp((kValueLinearNormMin - value_norm) / tail_norm_span, 0.0f, 1.0f);
    torch::Tensor x_tail = kValueLinearRealMin
        - tail_real_span * torch::pow(t, 1.0f / kValueTailPower);

    torch::Tensor x = torch::where(value_norm >= kValueLinearNormMin, x_linear, x_tail);
    return torch::clamp(x, kValueRealMin, kValueRealMax);
}

torch::Tensor value_real_from_output(torch::Tensor value_raw) {
    torch::Tensor value_norm = -torch::sigmoid(value_raw);
    return denormalize_value_model(value_norm);
}

struct Args {
    std::string addr = "127.0.0.1:50201";
    std::string device = "cuda:0";
    int max_batch = 256;
    int max_wait_us = 2000;
};

struct InferResult {
    bool ok = false;
    std::string error;
    float value = 0.0f;
    std::vector<float> priors;
};

struct InferTask {
    int model_version = 0;
    int player = 0;  // 0 controller, 1 adversary
    uint16_t action_len = 0;
    std::array<float, 60> req_features{};   // 20*3
    std::array<float, 9> global_features{}; // 9
    std::array<uint8_t, 20> req_mask{};
    std::vector<uint8_t> action_mask;
    std::promise<InferResult> promise;
};

struct ShmTask {
    uint32_t slot_idx = 0;
    uint32_t request_id = 0;
    int model_version = 0;
    int player = 0;
    uint16_t action_len = 0;
    std::array<float, 60> req_features{};
    std::array<float, 9> global_features{};
    std::array<uint8_t, 20> req_mask{};
    std::array<uint8_t, infer_shared::kActionMax> action_mask{};
    InferResult result;
};

struct ModelPair {
    torch::jit::script::Module controller;
    torch::jit::script::Module adversary;
};

class SharedInferRegion {
public:
    bool init_server(const std::string& addr) {
        name_ = infer_shared::shm_name_from_addr(addr);
        ::shm_unlink(name_.c_str());

        fd_ = ::shm_open(name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd_ < 0) {
            std::cerr << "shm_open failed: " << std::strerror(errno) << "\n";
            return false;
        }
        if (::ftruncate(fd_, static_cast<off_t>(sizeof(infer_shared::Region))) != 0) {
            std::cerr << "ftruncate failed: " << std::strerror(errno) << "\n";
            return false;
        }
        void* p = ::mmap(
            nullptr,
            sizeof(infer_shared::Region),
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd_,
            0
        );
        if (p == MAP_FAILED) {
            std::cerr << "mmap failed: " << std::strerror(errno) << "\n";
            return false;
        }
        reg_ = reinterpret_cast<infer_shared::Region*>(p);
        creator_ = true;
        init_region_unsafe();
        return true;
    }

    void close() {
        if (reg_ != nullptr) {
            ::munmap(reg_, sizeof(infer_shared::Region));
            reg_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        if (creator_ && !name_.empty()) {
            ::shm_unlink(name_.c_str());
        }
        creator_ = false;
    }

    infer_shared::Region* region() const { return reg_; }

    void signal_shutdown() {
        if (reg_ == nullptr) return;
        pthread_mutex_lock(&reg_->mu);
        reg_->shutdown = 1;
        pthread_cond_broadcast(&reg_->cv_ready);
        pthread_cond_broadcast(&reg_->cv_done);
        pthread_mutex_unlock(&reg_->mu);
    }

    bool pop_ready_batch(int max_batch, int max_wait_us, std::vector<ShmTask>& out) {
        out.clear();
        if (reg_ == nullptr) return false;

        pthread_mutex_lock(&reg_->mu);
        while (reg_->ready_count == 0 && reg_->shutdown == 0) {
            const int rc = pthread_cond_wait(&reg_->cv_ready, &reg_->mu);
            if (rc != 0) break;
        }

        if (reg_->ready_count == 0) {
            const bool keep_running = (reg_->shutdown == 0);
            pthread_mutex_unlock(&reg_->mu);
            return keep_running;
        }

        // Once at least one request is available, keep the service hot for a
        // bounded window so more workers can join the same batch. We actively
        // poll here (yielding, not sleeping) to reduce launch latency while
        // still respecting the configured maximum batching delay.
        if (max_wait_us > 0 && max_batch > 1) {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::microseconds(max_wait_us);
            const uint32_t target = std::min<uint32_t>(
                static_cast<uint32_t>(std::max(1, max_batch)),
                infer_shared::kSlots
            );
            while (
                reg_->shutdown == 0 &&
                reg_->ready_count < target &&
                std::chrono::steady_clock::now() < deadline
            ) {
                pthread_mutex_unlock(&reg_->mu);
                std::this_thread::yield();
                pthread_mutex_lock(&reg_->mu);
            }
        }

        const int take = std::min<int>(max_batch, static_cast<int>(reg_->ready_count));
        out.reserve(static_cast<size_t>(take));
        for (int i = 0; i < take; ++i) {
            const uint32_t idx = reg_->ready_ring[reg_->ready_head];
            reg_->ready_head = (reg_->ready_head + 1U) % infer_shared::kSlots;
            reg_->ready_count -= 1U;
            if (idx >= infer_shared::kSlots) continue;

            auto& s = reg_->slots[idx];
            if (s.state != infer_shared::SLOT_READY) continue;
            s.state = infer_shared::SLOT_RUNNING;

            ShmTask t;
            t.slot_idx = idx;
            t.request_id = s.request_id;
            t.model_version = s.model_version;
            t.player = static_cast<int>(s.player);
            t.action_len = s.action_len;
            std::memcpy(t.req_features.data(), s.req_features, sizeof(s.req_features));
            std::memcpy(t.global_features.data(), s.global_features, sizeof(s.global_features));
            std::memcpy(t.req_mask.data(), s.req_mask, sizeof(s.req_mask));
            std::memcpy(t.action_mask.data(), s.action_mask, sizeof(s.action_mask));
            out.push_back(std::move(t));
        }
        pthread_mutex_unlock(&reg_->mu);
        return true;
    }

    void complete_batch(const std::vector<ShmTask>& tasks) {
        if (reg_ == nullptr || tasks.empty()) return;
        pthread_mutex_lock(&reg_->mu);
        for (const auto& t : tasks) {
            if (t.slot_idx >= infer_shared::kSlots) continue;
            auto& s = reg_->slots[t.slot_idx];
            if (s.request_id != t.request_id) continue;

            s.error_code = t.result.ok ? 0 : 1;
            std::memset(s.error_msg, 0, sizeof(s.error_msg));
            if (!t.result.ok) {
                const std::string em = t.result.error;
                std::memcpy(
                    s.error_msg,
                    em.data(),
                    std::min<size_t>(em.size(), sizeof(s.error_msg) - 1U)
                );
            } else {
                s.value = t.result.value;
                std::memset(s.priors, 0, sizeof(s.priors));
                const size_t n = std::min<size_t>(t.result.priors.size(), infer_shared::kActionMax);
                if (n > 0) {
                    std::memcpy(s.priors, t.result.priors.data(), n * sizeof(float));
                }
            }
            s.state = infer_shared::SLOT_DONE;
        }
        pthread_cond_broadcast(&reg_->cv_done);
        pthread_mutex_unlock(&reg_->mu);
    }

private:
    void init_region_unsafe() {
        std::memset(reg_, 0, sizeof(infer_shared::Region));
        reg_->magic = infer_shared::kMagic;
        reg_->version = infer_shared::kVersion;
        reg_->slot_count = infer_shared::kSlots;

        pthread_mutexattr_t ma;
        pthread_mutexattr_init(&ma);
        pthread_mutexattr_setpshared(&ma, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&reg_->mu, &ma);
        pthread_mutexattr_destroy(&ma);

        pthread_condattr_t ca;
        pthread_condattr_init(&ca);
        pthread_condattr_setpshared(&ca, PTHREAD_PROCESS_SHARED);
        pthread_cond_init(&reg_->cv_ready, &ca);
        pthread_cond_init(&reg_->cv_done, &ca);
        pthread_condattr_destroy(&ca);

        reg_->free_head = 0;
        reg_->free_tail = 0;
        reg_->free_count = infer_shared::kSlots;
        reg_->ready_head = 0;
        reg_->ready_tail = 0;
        reg_->ready_count = 0;
        reg_->request_seq = 1;
        reg_->shutdown = 0;
        for (uint32_t i = 0; i < infer_shared::kSlots; ++i) {
            reg_->free_ring[i] = i;
            reg_->slots[i].state = infer_shared::SLOT_FREE;
            reg_->slots[i].action_len = 0;
            reg_->slots[i].request_id = 0;
            reg_->slots[i].error_code = 0;
            std::memset(reg_->slots[i].error_msg, 0, sizeof(reg_->slots[i].error_msg));
        }
    }

    std::string name_;
    int fd_ = -1;
    infer_shared::Region* reg_ = nullptr;
    bool creator_ = false;
};

static bool read_exact(int fd, void* dst, size_t n) {
    uint8_t* p = static_cast<uint8_t*>(dst);
    size_t off = 0;
    while (off < n) {
        ssize_t r = ::recv(fd, p + off, n - off, 0);
        if (r == 0) return false;
        if (r < 0) return false;
        off += static_cast<size_t>(r);
    }
    return true;
}

static bool write_exact(int fd, const void* src, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(src);
    size_t off = 0;
    while (off < n) {
        ssize_t w = ::send(fd, p + off, n - off, 0);
        if (w <= 0) return false;
        off += static_cast<size_t>(w);
    }
    return true;
}

static void append_u16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

static void append_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

static void append_i32(std::vector<uint8_t>& out, int32_t v) {
    append_u32(out, static_cast<uint32_t>(v));
}

static void append_f32(std::vector<uint8_t>& out, float f) {
    static_assert(sizeof(float) == 4, "float must be 4 bytes");
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(float));
    append_u32(out, bits);
}

static bool write_frame(int fd, const std::vector<uint8_t>& payload) {
    const uint32_t n = static_cast<uint32_t>(payload.size());
    uint8_t hdr[4] = {
        static_cast<uint8_t>(n & 0xFF),
        static_cast<uint8_t>((n >> 8) & 0xFF),
        static_cast<uint8_t>((n >> 16) & 0xFF),
        static_cast<uint8_t>((n >> 24) & 0xFF),
    };
    return write_exact(fd, hdr, sizeof(hdr)) &&
           (payload.empty() || write_exact(fd, payload.data(), payload.size()));
}

static bool read_frame(int fd, std::vector<uint8_t>& payload) {
    uint8_t hdr[4];
    if (!read_exact(fd, hdr, sizeof(hdr))) return false;
    const uint32_t n = static_cast<uint32_t>(hdr[0]) |
                       (static_cast<uint32_t>(hdr[1]) << 8) |
                       (static_cast<uint32_t>(hdr[2]) << 16) |
                       (static_cast<uint32_t>(hdr[3]) << 24);
    payload.resize(static_cast<size_t>(n));
    if (n == 0) return true;
    return read_exact(fd, payload.data(), payload.size());
}

class Reader {
public:
    explicit Reader(const std::vector<uint8_t>& buf) : b_(buf) {}

    bool read_u8(uint8_t& out) {
        if (off_ + 1 > b_.size()) return false;
        out = b_[off_++];
        return true;
    }

    bool read_u16(uint16_t& out) {
        if (off_ + 2 > b_.size()) return false;
        out = static_cast<uint16_t>(b_[off_]) |
              (static_cast<uint16_t>(b_[off_ + 1]) << 8);
        off_ += 2;
        return true;
    }

    bool read_u32(uint32_t& out) {
        if (off_ + 4 > b_.size()) return false;
        out = static_cast<uint32_t>(b_[off_]) |
              (static_cast<uint32_t>(b_[off_ + 1]) << 8) |
              (static_cast<uint32_t>(b_[off_ + 2]) << 16) |
              (static_cast<uint32_t>(b_[off_ + 3]) << 24);
        off_ += 4;
        return true;
    }

    bool read_i32(int32_t& out) {
        uint32_t u = 0;
        if (!read_u32(u)) return false;
        out = static_cast<int32_t>(u);
        return true;
    }

    bool read_f32(float& out) {
        uint32_t u = 0;
        if (!read_u32(u)) return false;
        std::memcpy(&out, &u, sizeof(float));
        return true;
    }

    bool read_bytes(uint8_t* out, size_t n) {
        if (off_ + n > b_.size()) return false;
        std::memcpy(out, b_.data() + off_, n);
        off_ += n;
        return true;
    }

    bool read_string(std::string& out) {
        uint32_t n = 0;
        if (!read_u32(n)) return false;
        if (off_ + n > b_.size()) return false;
        out.assign(reinterpret_cast<const char*>(b_.data() + off_), static_cast<size_t>(n));
        off_ += static_cast<size_t>(n);
        return true;
    }

private:
    const std::vector<uint8_t>& b_;
    size_t off_ = 0;
};

static std::vector<uint8_t> make_ok_empty() {
    return std::vector<uint8_t>{0};
}

static std::vector<uint8_t> make_err(const std::string& err) {
    std::vector<uint8_t> out;
    out.reserve(1 + 4 + err.size());
    out.push_back(1);
    append_u32(out, static_cast<uint32_t>(err.size()));
    out.insert(out.end(), err.begin(), err.end());
    return out;
}

static std::vector<uint8_t> make_ok_infer(float value, const std::vector<float>& priors) {
    std::vector<uint8_t> out;
    out.reserve(1 + 4 + 2 + priors.size() * 4);
    out.push_back(0);
    append_f32(out, value);
    append_u16(out, static_cast<uint16_t>(priors.size()));
    for (float p : priors) append_f32(out, p);
    return out;
}

static std::pair<std::string, int> parse_addr(const std::string& s) {
    auto pos = s.rfind(':');
    if (pos == std::string::npos) {
        throw std::runtime_error("addr must be host:port");
    }
    std::string host = s.substr(0, pos);
    int port = std::stoi(s.substr(pos + 1));
    return {host, port};
}

static torch::Device parse_device(const std::string& s) {
    try {
        torch::Device d(s);
        if (d.is_cuda() && !torch::cuda::is_available()) {
            throw std::runtime_error("CUDA requested but torch::cuda::is_available() is false");
        }
        return d;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("invalid device: ") + e.what());
    }
}

class InferService {
public:
    explicit InferService(const Args& args)
        : args_(args), device_(parse_device(args.device)),
          max_batch_(std::max(1, args.max_batch)),
          max_wait_us_(std::max(0, args.max_wait_us)) {}

    int run() {
        auto [host, port] = parse_addr(args_.addr);
        if (!shared_region_.init_server(args_.addr)) {
            std::cerr << "failed to initialize shared memory inference region\n";
            return 2;
        }
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) {
            std::cerr << "failed to create socket\n";
            shared_region_.close();
            return 2;
        }
        int one = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
            std::cerr << "invalid host: " << host << "\n";
            shared_region_.close();
            return 2;
        }
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            std::cerr << "bind failed\n";
            shared_region_.close();
            return 2;
        }
        if (::listen(listen_fd_, 256) != 0) {
            std::cerr << "listen failed\n";
            shared_region_.close();
            return 2;
        }

        std::cerr << "[cpp_infer_service] listening addr=" << args_.addr
                  << " device=" << args_.device
                  << " max_batch=" << max_batch_
                  << " max_wait_us=" << max_wait_us_
                  << " shm=" << infer_shared::shm_name_from_addr(args_.addr) << "\n";

        batch_thread_ = std::thread([this]() { this->batch_loop(); });

        while (!stopping_.load()) {
            int cfd = ::accept(listen_fd_, nullptr, nullptr);
            if (cfd < 0) {
                if (stopping_.load()) break;
                continue;
            }
            std::lock_guard<std::mutex> lk(conn_mu_);
            conn_threads_.emplace_back([this, cfd]() { this->handle_conn(cfd); });
        }

        stopping_.store(true);
        shared_region_.signal_shutdown();
        if (batch_thread_.joinable()) batch_thread_.join();

        {
            std::lock_guard<std::mutex> lk(conn_mu_);
            for (auto& t : conn_threads_) {
                if (t.joinable()) t.join();
            }
        }
        if (listen_fd_ >= 0) {
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
        shared_region_.close();
        return 0;
    }

private:
    void request_stop() {
        bool expected = false;
        if (!stopping_.compare_exchange_strong(expected, true)) return;
        if (listen_fd_ >= 0) {
            ::shutdown(listen_fd_, SHUT_RDWR);
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
        shared_region_.signal_shutdown();
    }

    void handle_conn(int fd) {
        std::vector<uint8_t> in;
        while (!stopping_.load()) {
            if (!read_frame(fd, in)) break;
            if (in.empty()) break;
            Reader r(in);
            uint8_t cmd = 0;
            if (!r.read_u8(cmd)) {
                write_frame(fd, make_err("bad frame"));
                continue;
            }

            try {
                if (cmd == CMD_PING) {
                    if (!write_frame(fd, make_ok_empty())) break;
                    continue;
                }
                if (cmd == CMD_SHUTDOWN) {
                    if (!write_frame(fd, make_ok_empty())) break;
                    request_stop();
                    break;
                }
                if (cmd == CMD_LOAD_MODELS) {
                    int32_t version = 0;
                    std::string cpath;
                    std::string apath;
                    if (!r.read_i32(version) || !r.read_string(cpath) || !r.read_string(apath)) {
                        if (!write_frame(fd, make_err("invalid load_models payload"))) break;
                        continue;
                    }
                    load_models(version, cpath, apath);
                    if (!write_frame(fd, make_ok_empty())) break;
                    continue;
                }
                if (cmd == CMD_INFER) {
                    auto task = std::make_shared<InferTask>();
                    int32_t version = 0;
                    uint8_t player = 0;
                    uint16_t action_len = 0;
                    if (!r.read_i32(version) || !r.read_u8(player) || !r.read_u16(action_len)) {
                        if (!write_frame(fd, make_err("invalid infer header"))) break;
                        continue;
                    }
                    task->model_version = version;
                    task->player = static_cast<int>(player);
                    task->action_len = action_len;
                    if (task->action_len == 0 || task->action_len > 24) {
                        if (!write_frame(fd, make_err("invalid action_len"))) break;
                        continue;
                    }
                    for (size_t i = 0; i < task->req_features.size(); ++i) {
                        float v = 0.0f;
                        if (!r.read_f32(v)) {
                            if (!write_frame(fd, make_err("invalid req_features"))) goto next_conn_iter;
                            goto next_conn_iter;
                        }
                        task->req_features[i] = v;
                    }
                    for (size_t i = 0; i < task->global_features.size(); ++i) {
                        float v = 0.0f;
                        if (!r.read_f32(v)) {
                            if (!write_frame(fd, make_err("invalid global_features"))) goto next_conn_iter;
                            goto next_conn_iter;
                        }
                        task->global_features[i] = v;
                    }
                    if (!r.read_bytes(task->req_mask.data(), task->req_mask.size())) {
                        if (!write_frame(fd, make_err("invalid req_mask"))) break;
                        continue;
                    }
                    task->action_mask.resize(task->action_len);
                    if (!r.read_bytes(task->action_mask.data(), task->action_mask.size())) {
                        if (!write_frame(fd, make_err("invalid action_mask"))) break;
                        continue;
                    }
                    auto fut = task->promise.get_future();
                    std::vector<std::shared_ptr<InferTask>> one;
                    one.push_back(task);
                    run_group(task->model_version, task->player, task->action_len, one);
                    InferResult res = fut.get();
                    if (!res.ok) {
                        if (!write_frame(fd, make_err(res.error))) break;
                    } else {
                        if (!write_frame(fd, make_ok_infer(res.value, res.priors))) break;
                    }
                    continue;
                }

                if (!write_frame(fd, make_err("unknown command"))) break;
            } catch (const std::exception& e) {
                if (!write_frame(fd, make_err(e.what()))) break;
            }
        next_conn_iter:
            continue;
        }
        ::close(fd);
    }

    void load_models(int version, const std::string& controller_path, const std::string& adversary_path) {
        std::lock_guard<std::mutex> lk(models_mu_);
        if (models_.count(version) > 0) return;
        auto pair = std::make_shared<ModelPair>(ModelPair{
            torch::jit::load(controller_path, device_),
            torch::jit::load(adversary_path, device_),
        });
        pair->controller.eval();
        pair->adversary.eval();
        models_[version] = std::move(pair);
    }

    std::shared_ptr<ModelPair> get_model_pair(int version) {
        std::lock_guard<std::mutex> lk(models_mu_);
        auto it = models_.find(version);
        if (it == models_.end()) return nullptr;
        return it->second;
    }

    void run_group(int version, int player, uint16_t action_len, std::vector<std::shared_ptr<InferTask>>& tasks) {
        auto pair = get_model_pair(version);
        if (!pair) {
            for (auto& t : tasks) {
                t->promise.set_value(InferResult{false, "model version not loaded", 0.0f, {}});
            }
            return;
        }

        auto& mod = (player == 0) ? pair->controller : pair->adversary;
        const int64_t B = static_cast<int64_t>(tasks.size());
        const int64_t A = static_cast<int64_t>(action_len);

        try {
            torch::Tensor req = torch::empty({B, 20, 3}, torch::TensorOptions().dtype(torch::kFloat32));
            torch::Tensor glb = torch::empty({B, 9}, torch::TensorOptions().dtype(torch::kFloat32));
            torch::Tensor req_mask = torch::empty({B, 20}, torch::TensorOptions().dtype(torch::kBool));
            torch::Tensor action_mask = torch::empty({B, A}, torch::TensorOptions().dtype(torch::kBool));

            auto req_ptr = req.data_ptr<float>();
            auto glb_ptr = glb.data_ptr<float>();
            auto req_mask_ptr = req_mask.data_ptr<bool>();
            auto action_mask_ptr = action_mask.data_ptr<bool>();

            for (int64_t i = 0; i < B; ++i) {
                const auto& t = tasks[static_cast<size_t>(i)];
                std::memcpy(req_ptr + i * 60, t->req_features.data(), 60 * sizeof(float));
                std::memcpy(glb_ptr + i * 9, t->global_features.data(), 9 * sizeof(float));
                for (int j = 0; j < 20; ++j) {
                    req_mask_ptr[i * 20 + j] = (t->req_mask[static_cast<size_t>(j)] != 0);
                }
                for (int j = 0; j < static_cast<int>(A); ++j) {
                    action_mask_ptr[i * A + j] = (t->action_mask[static_cast<size_t>(j)] != 0);
                }
            }

            c10::InferenceMode guard(true);
            std::vector<torch::jit::IValue> in;
            in.reserve(4);
            in.emplace_back(req.to(device_));
            in.emplace_back(glb.to(device_));
            in.emplace_back(req_mask.to(device_));
            in.emplace_back(action_mask.to(device_));

            auto out_iv = mod.forward(in);
            auto out_t = out_iv.toTuple();
            torch::Tensor policy_logits = out_t->elements()[0].toTensor();
            torch::Tensor value_raw = out_t->elements()[1].toTensor();

            torch::Tensor priors_t = torch::softmax(policy_logits, -1).to(torch::kCPU).contiguous();
            torch::Tensor values_t = value_real_from_output(value_raw).reshape({B}).to(torch::kCPU).contiguous();

            const auto* pri_ptr = priors_t.data_ptr<float>();
            const auto* val_ptr = values_t.data_ptr<float>();
            for (int64_t i = 0; i < B; ++i) {
                InferResult res;
                res.ok = true;
                res.value = val_ptr[i];
                res.priors.resize(static_cast<size_t>(A));
                std::memcpy(res.priors.data(), pri_ptr + i * A, static_cast<size_t>(A) * sizeof(float));
                tasks[static_cast<size_t>(i)]->promise.set_value(std::move(res));
            }
        } catch (const std::exception& e) {
            for (auto& t : tasks) {
                t->promise.set_value(InferResult{false, e.what(), 0.0f, {}});
            }
        }
    }

    void run_group_shm(int version, int player, uint16_t action_len, std::vector<ShmTask*>& tasks) {
        auto pair = get_model_pair(version);
        if (!pair) {
            for (auto* t : tasks) {
                t->result = InferResult{false, "model version not loaded", 0.0f, {}};
            }
            return;
        }

        auto& mod = (player == 0) ? pair->controller : pair->adversary;
        const int64_t B = static_cast<int64_t>(tasks.size());
        const int64_t A = static_cast<int64_t>(action_len);

        try {
            torch::Tensor req = torch::empty({B, 20, 3}, torch::TensorOptions().dtype(torch::kFloat32));
            torch::Tensor glb = torch::empty({B, 9}, torch::TensorOptions().dtype(torch::kFloat32));
            torch::Tensor req_mask = torch::empty({B, 20}, torch::TensorOptions().dtype(torch::kBool));
            torch::Tensor action_mask = torch::empty({B, A}, torch::TensorOptions().dtype(torch::kBool));

            auto req_ptr = req.data_ptr<float>();
            auto glb_ptr = glb.data_ptr<float>();
            auto req_mask_ptr = req_mask.data_ptr<bool>();
            auto action_mask_ptr = action_mask.data_ptr<bool>();

            for (int64_t i = 0; i < B; ++i) {
                const auto& t = *tasks[static_cast<size_t>(i)];
                std::memcpy(req_ptr + i * 60, t.req_features.data(), 60 * sizeof(float));
                std::memcpy(glb_ptr + i * 9, t.global_features.data(), 9 * sizeof(float));
                for (int j = 0; j < 20; ++j) {
                    req_mask_ptr[i * 20 + j] = (t.req_mask[static_cast<size_t>(j)] != 0);
                }
                for (int j = 0; j < static_cast<int>(A); ++j) {
                    action_mask_ptr[i * A + j] = (t.action_mask[static_cast<size_t>(j)] != 0);
                }
            }

            c10::InferenceMode guard(true);
            std::vector<torch::jit::IValue> in;
            in.reserve(4);
            in.emplace_back(req.to(device_));
            in.emplace_back(glb.to(device_));
            in.emplace_back(req_mask.to(device_));
            in.emplace_back(action_mask.to(device_));

            auto out_iv = mod.forward(in);
            auto out_t = out_iv.toTuple();
            torch::Tensor policy_logits = out_t->elements()[0].toTensor();
            torch::Tensor value_raw = out_t->elements()[1].toTensor();

            torch::Tensor priors_t = torch::softmax(policy_logits, -1).to(torch::kCPU).contiguous();
            torch::Tensor values_t = value_real_from_output(value_raw).reshape({B}).to(torch::kCPU).contiguous();

            const auto* pri_ptr = priors_t.data_ptr<float>();
            const auto* val_ptr = values_t.data_ptr<float>();
            for (int64_t i = 0; i < B; ++i) {
                auto& t = *tasks[static_cast<size_t>(i)];
                t.result.ok = true;
                t.result.error.clear();
                t.result.value = val_ptr[i];
                t.result.priors.resize(static_cast<size_t>(A));
                std::memcpy(
                    t.result.priors.data(),
                    pri_ptr + i * A,
                    static_cast<size_t>(A) * sizeof(float)
                );
            }
        } catch (const std::exception& e) {
            for (auto* t : tasks) {
                t->result = InferResult{false, e.what(), 0.0f, {}};
            }
        }
    }

    void batch_loop() {
        while (!stopping_.load()) {
            std::vector<ShmTask> batch;
            const bool keep_running = shared_region_.pop_ready_batch(max_batch_, max_wait_us_, batch);
            if (!keep_running && batch.empty()) break;
            if (batch.empty()) continue;

            std::map<std::tuple<int, int, uint16_t>, std::vector<ShmTask*>> groups;
            for (auto& t : batch) {
                groups[std::make_tuple(t.model_version, t.player, t.action_len)].push_back(&t);
            }
            for (auto& kv : groups) {
                auto version = std::get<0>(kv.first);
                auto player = std::get<1>(kv.first);
                auto action_len = std::get<2>(kv.first);
                run_group_shm(version, player, action_len, kv.second);
            }
            shared_region_.complete_batch(batch);
        }
    }

    Args args_;
    torch::Device device_;
    int max_batch_ = 256;
    int max_wait_us_ = 2000;

    std::atomic<bool> stopping_{false};
    int listen_fd_ = -1;
    std::thread batch_thread_;

    std::mutex models_mu_;
    std::unordered_map<int, std::shared_ptr<ModelPair>> models_;

    std::mutex conn_mu_;
    std::vector<std::thread> conn_threads_;
    SharedInferRegion shared_region_;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k(argv[i]);
        auto get_next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::string(argv[++i]);
        };
        if (k == "--addr") a.addr = get_next("--addr");
        else if (k == "--device") a.device = get_next("--device");
        else if (k == "--max-batch") a.max_batch = std::stoi(get_next("--max-batch"));
        else if (k == "--max-wait-us") a.max_wait_us = std::stoi(get_next("--max-wait-us"));
        else if (k == "--help" || k == "-h") {
            std::cout << "Usage: infer_service_main --addr host:port --device cuda:0 "
                         "--max-batch 256 --max-wait-us 2000\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    return a;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        InferService svc(args);
        return svc.run();
    } catch (const std::exception& e) {
        std::cerr << "infer_service_main error: " << e.what() << "\n";
        return 1;
    }
}
