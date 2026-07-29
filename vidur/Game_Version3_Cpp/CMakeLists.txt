cmake_minimum_required(VERSION 3.16)
project(mcts_native_gv2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

# Query torch paths from active Python env.
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
    COMMAND ${Python_EXECUTABLE} -c "import pathlib,numpy; p=pathlib.Path(numpy.__file__).resolve().parent.parent/'numpy.libs'; print(next((str(x) for x in p.glob('libopenblas64*.so')), ''))"
    OUTPUT_VARIABLE NUMPY_OPENBLAS_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT NUMPY_OPENBLAS_LIBRARY STREQUAL "")
    get_filename_component(NUMPY_OPENBLAS_DIR "${NUMPY_OPENBLAS_LIBRARY}" DIRECTORY)
else()
    set(NUMPY_OPENBLAS_DIR "")
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_CXX11_ABI STREQUAL "")
    set(TORCH_CXX11_ABI 1)
endif()

pybind11_add_module(mcts_native_gv2
    src/pybind_module.cpp
    src/gv2_types.cpp
    src/gv2_credits.cpp
    src/gv2_player_sample_actions.cpp
    src/gv2_virtual_environment.cpp
    src/gv2_mcts_dnn.cpp
    src/gv2_mcts_value_prior.cpp
    src/gv3_native_selfplay.cpp
    src/gv2_infer_runtime.cpp
    src/new_features_226_inference.cpp
    src/agz_dense_dnn.cpp
    src/markov_value_features.cpp
    src/gv2_logger.cpp
    src/virtual_simulator.cpp
)

target_include_directories(mcts_native_gv2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(mcts_native_gv2 SYSTEM PRIVATE
    ${TORCH_INCLUDE_PATHS}
)
target_link_directories(mcts_native_gv2 PRIVATE ${TORCH_LIBRARY_PATHS})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        mcts_native_gv2
        PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
    # The AGZ dense kernels only consume finite replay features. Permit SIMD
    # reassociation for their small GEMV reductions without changing MCTS code.
    set_source_files_properties(
        src/agz_dense_dnn.cpp
        PROPERTIES COMPILE_OPTIONS "-O3;-ffast-math;-funroll-loops;-march=native"
    )
else()
    target_compile_options(mcts_native_gv2 PRIVATE -O3)
endif()

target_compile_definitions(mcts_native_gv2 PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_link_options(mcts_native_gv2 PRIVATE -Wl,--no-as-needed)

target_link_libraries(mcts_native_gv2 PRIVATE torch_python torch torch_cpu c10)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mcts_native_gv2 PRIVATE OpenMP::OpenMP_CXX)
endif()
if(NOT NUMPY_OPENBLAS_LIBRARY STREQUAL "")
    target_link_libraries(mcts_native_gv2 PRIVATE "${NUMPY_OPENBLAS_LIBRARY}")
    target_compile_definitions(mcts_native_gv2 PRIVATE GV2_HAVE_OPENBLAS_ILP64=1)
endif()

find_library(TORCH_CUDA_LIB torch_cuda PATHS ${TORCH_LIBRARY_PATHS} NO_DEFAULT_PATH)
find_library(C10_CUDA_LIB c10_cuda PATHS ${TORCH_LIBRARY_PATHS} NO_DEFAULT_PATH)
if(TORCH_CUDA_LIB AND C10_CUDA_LIB)
    target_link_libraries(mcts_native_gv2 PRIVATE ${TORCH_CUDA_LIB} ${C10_CUDA_LIB})
endif()

set_target_properties(mcts_native_gv2 PROPERTIES
    BUILD_RPATH "${TORCH_LIBRARY_PATHS};${NUMPY_OPENBLAS_DIR}"
    INSTALL_RPATH "${TORCH_LIBRARY_PATHS};${NUMPY_OPENBLAS_DIR}"
)

get_filename_component(MCTS_PY_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

set_target_properties(mcts_native_gv2 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${MCTS_PY_PACKAGE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${MCTS_PY_PACKAGE_DIR}"
    BUILD_RPATH "${TORCH_LIBRARY_PATHS};${NUMPY_OPENBLAS_DIR}"
    INSTALL_RPATH "${TORCH_LIBRARY_PATHS};${NUMPY_OPENBLAS_DIR}"
)
