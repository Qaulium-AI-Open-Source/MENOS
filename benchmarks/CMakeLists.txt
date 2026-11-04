# Built only when MENOS_BUILD_BENCHMARKS=ON (root CMakeLists.txt).
# Benchmarks are not tests (CLAUDE.md) -- none of these are registered
# with ctest.

add_library(menos_benchmark_environment STATIC
    menos_benchmark_environment.cpp
)
target_include_directories(menos_benchmark_environment PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(menos_benchmark_environment PUBLIC menos_plan)
target_compile_options(menos_benchmark_environment PRIVATE ${MENOS_WARNING_FLAGS})

# Phase 21 step 1 (docs/adr/0025 section 5): plan_job latency and heap
# traffic over the fixed planner corpus shared with the golden-plan test, and
# a cold-calibration probe run once per fresh process by bench/run_calibration.sh.
add_executable(menos_planner_benchmark
    planner_benchmark.cpp
    counting_allocator.cpp
    menos_benchmark_main.cpp
)
target_include_directories(menos_planner_benchmark PRIVATE ${PROJECT_SOURCE_DIR}/tests/plan)
target_link_libraries(menos_planner_benchmark PRIVATE
    menos_plan menos_benchmark_environment benchmark::benchmark
)
target_compile_options(menos_planner_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

# Phase 21 step 2 pass 2 (docs/adr/0025 section 12): each planner data
# structure timed alone.
add_executable(menos_planner_structures_benchmark
    planner_structures_benchmark.cpp
    menos_benchmark_main.cpp
)
target_include_directories(menos_planner_structures_benchmark PRIVATE ${PROJECT_SOURCE_DIR}/tests/plan)
target_link_libraries(menos_planner_structures_benchmark PRIVATE
    menos_plan menos_benchmark_environment benchmark::benchmark
)
target_compile_options(menos_planner_structures_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

add_executable(menos_calibration_probe calibration_probe.cpp)
target_link_libraries(menos_calibration_probe PRIVATE menos_plan)
target_compile_options(menos_calibration_probe PRIVATE ${MENOS_WARNING_FLAGS})

# Phase 21 owner question 4: does calibration noise change the chosen plan?
add_executable(menos_calibration_sensitivity calibration_sensitivity.cpp)
target_include_directories(menos_calibration_sensitivity PRIVATE ${PROJECT_SOURCE_DIR}/tests/plan)
target_link_libraries(menos_calibration_sensitivity PRIVATE menos_plan)
target_compile_options(menos_calibration_sensitivity PRIVATE ${MENOS_WARNING_FLAGS})

add_executable(menos_dense_gate_application_benchmark
    dense_gate_application_benchmark.cpp
    menos_benchmark_main.cpp
)
target_link_libraries(menos_dense_gate_application_benchmark PRIVATE
    menos_engines_dense menos_common menos_benchmark_environment benchmark::benchmark
)
target_compile_options(menos_dense_gate_application_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

add_executable(menos_factorized_vs_direct_benchmark
    factorized_vs_direct_benchmark.cpp
    menos_benchmark_main.cpp
)
target_link_libraries(menos_factorized_vs_direct_benchmark PRIVATE
    menos_api menos_engines_dense menos_benchmark_environment benchmark::benchmark
)
target_compile_options(menos_factorized_vs_direct_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

add_executable(menos_cost_model_correlation_benchmark
    cost_model_correlation_benchmark.cpp
)
target_link_libraries(menos_cost_model_correlation_benchmark PRIVATE
    menos_api menos_cost menos_engines_dense menos_benchmark_environment
)
target_compile_options(menos_cost_model_correlation_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

add_executable(menos_stabilizer_cost_correlation_benchmark
    stabilizer_cost_correlation_benchmark.cpp
)
target_link_libraries(menos_stabilizer_cost_correlation_benchmark PRIVATE
    menos_runtime menos_plan menos_engines_dense menos_engines_stabilizer menos_benchmark_environment
)
target_compile_options(menos_stabilizer_cost_correlation_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

# Phase 12 (docs/adr/0016 section 5): whether the calibrated dense work-unit
# rate predicts the density-matrix engine's runtime.
add_executable(menos_density_matrix_cost_correlation_benchmark
    density_matrix_cost_correlation_benchmark.cpp
)
target_link_libraries(menos_density_matrix_cost_correlation_benchmark PRIVATE
    menos_runtime menos_plan menos_engines_density_matrix menos_benchmark_environment
)
target_compile_options(menos_density_matrix_cost_correlation_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

# Phase 13 (docs/adr/0017 section 8): whether the calibrated dense work-unit
# rate predicts the trajectory engine's runtime, and how conservative N_eps is.
add_executable(menos_trajectory_cost_correlation_benchmark
    trajectory_cost_correlation_benchmark.cpp
)
target_link_libraries(menos_trajectory_cost_correlation_benchmark PRIVATE
    menos_runtime menos_plan menos_cost menos_engines_trajectory menos_benchmark_environment
)
target_compile_options(menos_trajectory_cost_correlation_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

add_executable(menos_tensor_network_cost_correlation_benchmark
    tensor_network_cost_correlation_benchmark.cpp
)
target_link_libraries(menos_tensor_network_cost_correlation_benchmark PRIVATE
    menos_plan menos_engines_tensor_network menos_benchmark_environment
)
target_compile_options(menos_tensor_network_cost_correlation_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

# Phase 9 AC-0 precondition (docs/adr/0013 section 10.1). Needs no cutting
# engine: it assembles a cut plan's cost from the existing engines' own
# estimates on the fragment circuits.
add_executable(menos_cut_ac0_benchmark
    cut_ac0_benchmark.cpp
)
target_link_libraries(menos_cut_ac0_benchmark PRIVATE
    menos_plan menos_cost menos_ir menos_common menos_engines_dense menos_benchmark_environment
)
target_compile_options(menos_cut_ac0_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

# Phase 11: what the execution-monitor seam costs (docs/adr/0015).
add_executable(menos_monitor_overhead_benchmark
    monitor_overhead_benchmark.cpp
    menos_benchmark_main.cpp
)
target_link_libraries(menos_monitor_overhead_benchmark PRIVATE
    menos_engines_dense menos_engines_stabilizer menos_engines_mps
    menos_benchmark_environment benchmark::benchmark
)
target_compile_options(menos_monitor_overhead_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

if(MENOS_ENABLE_CUDA)
    # The CPU/CUDA crossover benchmark (docs/BUILD_PLAN.md Phase 10).
    add_executable(menos_cuda_crossover_benchmark
        cuda_crossover_benchmark.cpp
        menos_benchmark_main.cpp
    )
    target_link_libraries(menos_cuda_crossover_benchmark PRIVATE
        menos_engines_dense menos_engines_dense_cuda menos_plan
        menos_benchmark_environment benchmark::benchmark
        nlohmann_json::nlohmann_json  # MENOS_CROSSOVER_JSON output
    )
    target_compile_options(menos_cuda_crossover_benchmark PRIVATE ${MENOS_WARNING_FLAGS})

    # The Phase 16 step-1 kernel benchmark (docs/adr/0020 section 2): the
    # re-created H/CX probe against the generic kernel, a per-gate cost
    # breakdown and device bandwidth. cuda_probe_kernels.cu is benchmark-only
    # device code, linked into no library.
    add_executable(menos_cuda_kernel_benchmark
        cuda_kernel_benchmark.cpp
        cuda_probe_kernels.cu
        menos_benchmark_main.cpp
    )
    target_link_libraries(menos_cuda_kernel_benchmark PRIVATE
        menos_kernels_cuda menos_ir menos_plan
        menos_benchmark_environment benchmark::benchmark
        nlohmann_json::nlohmann_json  # MENOS_CUDA_KERNEL_JSON output
    )
    set_target_properties(menos_cuda_kernel_benchmark PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_compile_options(menos_cuda_kernel_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${MENOS_WARNING_FLAGS}>)
    # The same nvcc flags as src/kernels/cuda/CMakeLists.txt, for the same
    # reasons given there: the host warning set minus -Wpedantic and
    # -Wold-style-cast, and -fmad=false so the probe's arithmetic is compiled
    # as the generic kernel's is.
    target_compile_options(menos_cuda_kernel_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wconversion,-Wsign-conversion,-Wshadow,-Wnon-virtual-dtor,-Woverloaded-virtual,-Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)

    # Phase 17 step 4a (docs/adr/0021 section 5): every CPU/CUDA assignment of
    # a Factorized plan's regions, and one fragment-sized evaluation on each
    # backend. Benchmark-only; the planner is unchanged by it.
    add_executable(menos_cuda_factorized_backend_benchmark
        cuda_factorized_backend_benchmark.cpp
        menos_benchmark_main.cpp
    )
    target_link_libraries(menos_cuda_factorized_backend_benchmark PRIVATE
        menos_api menos_kernels_cuda menos_benchmark_environment benchmark::benchmark
        nlohmann_json::nlohmann_json  # MENOS_FACTORIZED_BACKEND_JSON output
    )
    target_compile_options(menos_cuda_factorized_backend_benchmark PRIVATE ${MENOS_WARNING_FLAGS})
endif()
