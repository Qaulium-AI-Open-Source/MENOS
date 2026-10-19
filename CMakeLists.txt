cmake_minimum_required(VERSION 3.25)
project(menos CXX)

# OFF for the dev/release presets (scripts/verify.sh's C++-only stages),
# so pure C++ development never needs Python/nanobind installed. Turned ON
# by pyproject.toml's [tool.scikit-build.cmake.define] when building the
# menos._core extension module.
option(MENOS_BUILD_PYTHON_BINDINGS "Build the menos._core nanobind extension module" OFF)

# OFF by default so ordinary dev/release ctest runs (and scripts/verify.sh's
# main path) never pay Google Benchmark's fetch/build cost. Phase 3's
# benchmark programs (docs/BUILD_PLAN.md 3.9) are built only when
# explicitly requested: cmake --preset dev -DMENOS_BUILD_BENCHMARKS=ON.
option(MENOS_BUILD_BENCHMARKS "Build the Google Benchmark programs under benchmarks/" OFF)

# OFF by default so a machine with no CUDA toolkit builds and passes the gate
# byte-for-byte unchanged (docs/adr/0014). Turned ON by the cuda-dev/cuda-release
# presets, which also pin CMAKE_CUDA_COMPILER -- see those presets for why the
# pin is required rather than optional.
option(MENOS_ENABLE_CUDA "Build the CUDA dense backend" OFF)

# Multi-threaded CPU gate application (Phase 15, docs/adr/0019 section 3.1)
# runs on MENOS's own thread pool, built on the platform thread library.
find_package(Threads REQUIRED)

# Windows is supported with MSVC only (docs/adr/0022). A MinGW build of the
# Python extension was observed to configure, build and then misbehave at run
# time, so any other compiler is refused here rather than half-working.
if(WIN32 AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "menos: on Windows only MSVC is supported (found "
                        "${CMAKE_CXX_COMPILER_ID} at ${CMAKE_CXX_COMPILER}); build from a "
                        "Visual Studio x64 developer environment (docs/adr/0022 section 3)")
endif()

# MSVC 19.51 (toolset 14.51) miscompiles operator_schmidt_rank in
# src/analysis/schmidt_rank_bound.cpp: the optimised code reads a loop bound
# from a stack slot it never initialised, so a run crashes, hangs or silently
# computes a wrong rank bound (measured, docs/adr/0022 section 3.1). MSVC 19.50
# compiles the same source correctly. Refused rather than half-working.
if(MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 19.51
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.52)
    message(FATAL_ERROR "menos: MSVC ${CMAKE_CXX_COMPILER_VERSION} (toolset 14.51) miscompiles "
                        "MENOS (docs/adr/0022 section 3.1); use the 14.50 toolset, e.g. "
                        "Enter-VsDevShell -DevCmdArguments \"-arch=x64 -host_arch=x64 "
                        "-vcvars_ver=14.50\"")
endif()

# Windows is supported for the CPU build only (docs/adr/0022). Refuse a CUDA
# request outright rather than silently building without CUDA.
if(MSVC AND MENOS_ENABLE_CUDA)
    message(FATAL_ERROR "menos: MENOS_ENABLE_CUDA is not supported with MSVC. "
                        "CUDA on Windows is a recorded Phase 18 deferral "
                        "(docs/adr/0022 section 7); configure with MENOS_ENABLE_CUDA=OFF")
endif()

if(MENOS_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # CUDA 12.0 is the floor because nvcc accepts -std=c++20 only from 12.0
    # on, and dense_kernels.cu is compiled as C++20 (docs/adr/0014 section 17).
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0)
        message(FATAL_ERROR "menos: CUDA ${CMAKE_CUDA_COMPILER_VERSION} at "
                            "${CMAKE_CUDA_COMPILER} is below the supported floor of 12.0")
    endif()
    # A compiler, headers and runtime from different CUDA major versions build
    # and link silently and then misread struct fields (docs/adr/0014 section
    # 8, trap 2). Refuse that configuration here rather than at run time.
    string(REGEX MATCH "^[0-9]+" menos_nvcc_major "${CMAKE_CUDA_COMPILER_VERSION}")
    if(NOT CUDAToolkit_VERSION_MAJOR STREQUAL menos_nvcc_major)
        message(FATAL_ERROR "menos: nvcc ${CMAKE_CUDA_COMPILER_VERSION} does not match the "
                            "CUDA toolkit ${CUDAToolkit_VERSION} found at "
                            "${CUDAToolkit_INCLUDE_DIRS}; set CUDAToolkit_ROOT")
    endif()
    get_filename_component(menos_cudart_real "${CUDA_cudart_LIBRARY}" REALPATH)
    if(menos_cudart_real MATCHES "libcudart\\.so\\.([0-9]+)"
       AND NOT CMAKE_MATCH_1 STREQUAL CUDAToolkit_VERSION_MAJOR)
        message(FATAL_ERROR "menos: libcudart ${menos_cudart_real} is from CUDA "
                            "${CMAKE_MATCH_1}, but the headers are CUDA ${CUDAToolkit_VERSION}; "
                            "set CUDAToolkit_ROOT")
    endif()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        # Explicit, never a default: this project targets the compute
        # capability it measured (8.6 on the development GPU).
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    message(STATUS "menos: CUDA enabled -- nvcc ${CMAKE_CUDA_COMPILER_VERSION} "
                   "at ${CMAKE_CUDA_COMPILER}, runtime ${menos_cudart_real}, "
                   "architectures ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# All menos_* static libraries must be position-independent so they can be
# linked into the menos._core shared object (src/python) when
# MENOS_BUILD_PYTHON_BINDINGS=ON. Applying it globally (rather than only
# under that option) keeps every target's compile flags identical between
# the dev/release presets and the Python build -- no separate ABI to
# reason about -- at negligible cost for the plain C++ executables.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warning set from CLAUDE.md, applied to menos targets only (not to
# FetchContent'd third-party code, which is not held to this bar). MSVC gets
# its own equivalent (docs/adr/0022 section 3).
if(MSVC)
    set(MENOS_WARNING_FLAGS
        /W4
        /WX
        /permissive-
        /Zc:__cplusplus
        /utf-8
    )
else()
    set(MENOS_WARNING_FLAGS
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Wold-style-cast
        -Wnon-virtual-dtor
        -Woverloaded-virtual
        -Werror
    )
endif()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 063de7e9578f82b369302001269680b4b1553359 # v1.18.0, see docs/adr/0001-toolchain-and-dependencies.md
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# GoogleTest is test-only: without this its install rules put gtest headers
# and libraries into every Python wheel (seen in the Phase 18 wheel builds).
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test-only: the tests/ir fixture loader parses Qiskit-generated JSON.
# Never linked into a production (menos_*) target. See
# docs/adr/0002-json-fixture-dependency.md. Shallow-cloned by tag (commit
# 55f93686c01528224f448c19128836e7df245f72, confirmed via the GitHub API)
# -- nlohmann/json's full history is large enough that a non-shallow clone
# was impractically slow in this environment.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# Benchmark-only dependency (docs/BUILD_PLAN.md Phase 3, 3.9), fetched only
# when MENOS_BUILD_BENCHMARKS=ON. Pin verified live against GitHub's
# release API this session (latest stable release at the time), not
# guessed, matching this repo's existing pin discipline
# (docs/adr/0001-toolchain-and-dependencies.md).
if(MENOS_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        google_benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG 192ef10025eb2c4cdd392bc502f0c852196baa48 # v1.9.5
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(google_benchmark)
endif()

enable_testing()

add_subdirectory(src/common)
add_subdirectory(src/ir)
add_subdirectory(src/qasm2)
add_subdirectory(src/qasm3)
add_subdirectory(src/kernels)
add_subdirectory(src/engines/dense)
add_subdirectory(src/engines/stabilizer)
add_subdirectory(src/engines/pauli)
add_subdirectory(src/engines/mps)
add_subdirectory(src/engines/tensor_network)
add_subdirectory(src/engines/noise)
add_subdirectory(src/engines/density_matrix)
add_subdirectory(src/engines/trajectory)
if(MENOS_ENABLE_CUDA)
    add_subdirectory(src/engines/dense_cuda)
endif()
add_subdirectory(src/analysis)
add_subdirectory(src/cost)
add_subdirectory(src/plan)
add_subdirectory(src/runtime)
add_subdirectory(src/mitigation)
add_subdirectory(src/api)
add_subdirectory(cli)
add_subdirectory(tests/common)
add_subdirectory(tests/ir)
add_subdirectory(tests/qasm2)
add_subdirectory(tests/qasm3)
add_subdirectory(tests/cli)
add_subdirectory(tests/engines/dense)
add_subdirectory(tests/engines/stabilizer)
add_subdirectory(tests/engines/pauli)
add_subdirectory(tests/engines/mps)
add_subdirectory(tests/engines/tensor_network)
add_subdirectory(tests/engines/density_matrix)
add_subdirectory(tests/engines/trajectory)
add_subdirectory(tests/analysis)
add_subdirectory(tests/cost)
add_subdirectory(tests/plan)
add_subdirectory(tests/runtime)
add_subdirectory(tests/api)
add_subdirectory(tests/mitigation)
add_subdirectory(tests/memory)

if(MENOS_BUILD_PYTHON_BINDINGS)
    add_subdirectory(src/python)
endif()

if(MENOS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Windows has no rpath: an executable finds a DLL in its own directory. Copy
# the OpenBLAS DLL (src/common) next to every menos executable, at configure
# time so that gtest_discover_tests can already run them when they are first
# built (docs/adr/0022 section 4). A single-config generator (the windows-*
# presets use Ninja) puts each executable in its directory's binary dir; a
# multi-config one (Visual Studio, scikit-build-core's default) in a
# per-configuration subdirectory of it.
if(WIN32)
    function(menos_copy_openblas_dll directory)
        get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(target IN LISTS targets)
            get_target_property(type ${target} TYPE)
            if(type STREQUAL "EXECUTABLE")
                get_target_property(binary_dir ${target} BINARY_DIR)
                if(CMAKE_CONFIGURATION_TYPES)
                    foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
                        file(COPY "${MENOS_OPENBLAS_DLL}" DESTINATION "${binary_dir}/${config}")
                    endforeach()
                else()
                    file(COPY "${MENOS_OPENBLAS_DLL}" DESTINATION "${binary_dir}")
                endif()
            endif()
        endforeach()
        get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
        foreach(subdirectory IN LISTS subdirectories)
            menos_copy_openblas_dll("${subdirectory}")
        endforeach()
    endfunction()
    menos_copy_openblas_dll("${CMAKE_SOURCE_DIR}")
endif()
