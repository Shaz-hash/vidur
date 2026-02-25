#pragma once

#include <stdexcept>

namespace mcts_native {

class NativeMCTS {
public:
    static void search() {
        throw std::runtime_error("NativeMCTS.search is not implemented yet in this build");
    }
};

} // namespace mcts_native
