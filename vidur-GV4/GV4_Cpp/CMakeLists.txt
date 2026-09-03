cmake_minimum_required(VERSION 3.18)
project(gv4_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(gv4_native
    src/pybind_module.cpp
    src/feature_bindings.cpp
    src/inference_bindings.cpp
    src/mcts_rollout_bindings.cpp
    src/config.cpp
    src/state.cpp
    src/action_resolver.cpp
    src/engine.cpp
    src/features.cpp
    src/inference.cpp
    src/mcts.cpp
    src/rollout.cpp
)

target_include_directories(gv4_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gv4_native PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
else()
    target_compile_options(gv4_native PRIVATE -O3)
endif()

set_target_properties(gv4_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
