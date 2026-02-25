#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <pthread.h>

namespace infer_shared {

constexpr uint32_t kMagic = 0x56494452U;  // "VIDR"
constexpr uint32_t kVersion = 1U;
constexpr uint32_t kSlots = 4096U;
constexpr uint32_t kActionMax = 24U;

enum SlotState : uint8_t {
    SLOT_FREE = 0,
    SLOT_READY = 1,
    SLOT_RUNNING = 2,
    SLOT_DONE = 3,
};

struct alignas(64) Slot {
    uint8_t state = SLOT_FREE;
    uint8_t player = 0;  // 0 controller, 1 adversary
    uint16_t action_len = 0;
    int32_t model_version = 0;
    uint32_t request_id = 0;
    float req_features[60];
    float global_features[9];
    uint8_t req_mask[20];
    uint8_t action_mask[kActionMax];
    float value = 0.0f;
    float priors[kActionMax];
    int32_t error_code = 0;
    char error_msg[128];
};

struct alignas(64) Region {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t slot_count = 0;
    uint32_t reserved0 = 0;

    pthread_mutex_t mu;
    pthread_cond_t cv_ready;
    pthread_cond_t cv_done;

    uint8_t shutdown = 0;
    uint8_t reserved1[3] = {0, 0, 0};
    uint32_t request_seq = 0;

    uint32_t free_head = 0;
    uint32_t free_tail = 0;
    uint32_t free_count = 0;

    uint32_t ready_head = 0;
    uint32_t ready_tail = 0;
    uint32_t ready_count = 0;

    uint32_t free_ring[kSlots];
    uint32_t ready_ring[kSlots];
    Slot slots[kSlots];
};

inline std::string shm_name_from_addr(const std::string& addr) {
    auto pos = addr.rfind(':');
    if (pos == std::string::npos || pos + 1 >= addr.size()) {
        throw std::runtime_error("addr must be host:port for shared memory naming");
    }
    int port = std::stoi(addr.substr(pos + 1));
    if (port <= 0) {
        throw std::runtime_error("invalid port in addr for shared memory naming");
    }
    return std::string("/vidur_infer_shm_") + std::to_string(port);
}

}  // namespace infer_shared

