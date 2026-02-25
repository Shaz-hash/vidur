cmake_minimum_required(VERSION 3.16)
project(mcts_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Query Torch include dirs from the active Python env instead of find_package(Torch),
# which can fail on CUDA-enabled wheels when toolkit discovery is unavailable.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch, torch.utils.cpp_extension as ce; print('\\n'.join(ce.include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_PATHS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_INCLUDE_PATHS_RAW STREQUAL "")
    message(FATAL_ERROR "Unable to discover torch include paths from Python")
endif()
string(REPLACE "\n" ";" TORCH_INCLUDE_PATHS "${TORCH_INCLUDE_PATHS_RAW}")

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch, torch.utils.cpp_extension as ce; print('\\n'.join(ce.library_paths()))"
    OUTPUT_VARIABLE TORCH_LIBRARY_PATHS_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_LIBRARY_PATHS_RAW STREQUAL "")
    message(FATAL_ERROR "Unable to discover torch library paths from Python")
endif()
string(REPLACE "\n" ";" TORCH_LIBRARY_PATHS "${TORCH_LIBRARY_PATHS_RAW}")

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_CXX11_ABI STREQUAL "")
    set(TORCH_CXX11_ABI 1)
endif()

pybind11_add_module(mcts_native
    src/pybind_module.cpp
    src/native_sim.cpp
    src/native_predictor.cpp
    src/native_mcts.cpp
)

target_include_directories(mcts_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_PATHS}
)
target_link_directories(mcts_native PRIVATE ${TORCH_LIBRARY_PATHS})

target_compile_options(mcts_native PRIVATE -O3)
target_compile_definitions(mcts_native PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_link_options(mcts_native PRIVATE -Wl,--no-as-needed)
target_link_libraries(mcts_native PRIVATE torch_python torch torch_cpu c10)
if(EXISTS "${TORCH_LIBRARY_PATHS}/libtorch_cuda.so")
    target_link_libraries(mcts_native PRIVATE torch_cuda c10_cuda)
endif()

set_target_properties(mcts_native PROPERTIES
    BUILD_RPATH "${TORCH_LIBRARY_PATHS}"
    INSTALL_RPATH "${TORCH_LIBRARY_PATHS}"
)

add_executable(infer_service_main
    src/infer_service_main.cpp
)

target_include_directories(infer_service_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_PATHS}
)
target_link_directories(infer_service_main PRIVATE ${TORCH_LIBRARY_PATHS})
target_compile_options(infer_service_main PRIVATE -O3)
target_compile_definitions(infer_service_main PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_link_options(infer_service_main PRIVATE -Wl,--no-as-needed)
target_link_libraries(infer_service_main PRIVATE torch torch_cpu c10)
if(EXISTS "${TORCH_LIBRARY_PATHS}/libtorch_cuda.so")
    target_link_libraries(infer_service_main PRIVATE torch_cuda c10_cuda)
endif()

set_target_properties(infer_service_main PROPERTIES
    BUILD_RPATH "${TORCH_LIBRARY_PATHS}"
    INSTALL_RPATH "${TORCH_LIBRARY_PATHS}"
)
