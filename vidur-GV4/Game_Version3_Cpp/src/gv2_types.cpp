#include "gv2_types.hpp"

#include <algorithm>
#include <numeric>

namespace mcts_native_gv2 {

std::vector<double> normalize_masked(
    const std::vector<double>& raw,
    const std::vector<uint8_t>& mask) {
    const std::size_t n = std::min(raw.size(), mask.size());
    std::vector<double> out(n, 0.0);
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (mask[i]) {
            const double v = raw[i] > 0.0 ? raw[i] : 0.0;
            out[i] = v;
            s += v;
        }
    }
    if (s <= 0.0) {
        int k = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (mask[i]) ++k;
        }
        if (k == 0) return out;
        const double p = 1.0 / static_cast<double>(k);
        for (std::size_t i = 0; i < n; ++i) {
            if (mask[i]) out[i] = p;
        }
        return out;
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (mask[i]) out[i] /= s;
    }
    return out;
}

int argmax_masked(const std::vector<double>& values, const std::vector<uint8_t>& mask) {
    const std::size_t n = std::min(values.size(), mask.size());
    int best = -1;
    double best_v = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!mask[i]) continue;
        if (best < 0 || values[i] > best_v) {
            best = static_cast<int>(i);
            best_v = values[i];
        }
    }
    return best;
}

}  // namespace mcts_native_gv2
