add_executable(menos_api_tests
    run_test.cpp
    noise_run_test.cpp
    trajectory_run_test.cpp
    analytic_circuits_test.cpp
    explain_test.cpp
    wall_time_report_test.cpp
    determinism_test.cpp
    fixtures_test.cpp
    factorized_differential_test.cpp
    stabilizer_scale_test.cpp
    light_cone_scale_test.cpp
    accuracy_policy_test.cpp
    mps_scale_test.cpp
    mps_rejection_test.cpp
    tensor_network_scale_test.cpp
    cut_differential_test.cpp
    execution_monitor_test.cpp
    replan_test.cpp
    batch_test.cpp
    readout_mitigation_run_test.cpp
    max_threads_test.cpp
    backend_request_test.cpp
    fuse_diagonal_run_test.cpp
)

if(MENOS_ENABLE_CUDA)
    # The CPU/CUDA differential suite. Skipped at runtime (loudly) if no device
    # is present; it never silently passes without one.
    target_sources(menos_api_tests PRIVATE cpu_cuda_differential_test.cpp
                                           cuda_execution_monitor_test.cpp
                                           cuda_backend_request_test.cpp)
    target_link_libraries(menos_api_tests PRIVATE menos_engines_dense_cuda)
endif()

# execution_monitor_test.cpp exercises the ExecutionMonitor seam against each
# concrete engine directly (Phase 11), so it names them rather than reaching
# them through the planner.
target_link_libraries(menos_api_tests PRIVATE menos_api menos_test_fixture_loader
                                              menos_engines_dense menos_engines_stabilizer
                                              menos_engines_pauli menos_engines_mps
                                              menos_engines_tensor_network
                                              GTest::gtest_main)
target_compile_options(menos_api_tests PRIVATE ${MENOS_WARNING_FLAGS})
target_compile_definitions(menos_api_tests PRIVATE MENOS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

include(GoogleTest)
gtest_discover_tests(menos_api_tests)
