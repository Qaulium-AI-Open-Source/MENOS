add_executable(menos_engines_mps_tests
    mps_state_test.cpp
    apply_gate_test.cpp
    sampling_test.cpp
    expectation_test.cpp
    mps_engine_test.cpp
    schmidt_rank_bound_consistency_test.cpp
)

target_link_libraries(menos_engines_mps_tests PRIVATE
    menos_engines_mps menos_engines_dense menos_ir menos_analysis GTest::gtest_main
)
target_include_directories(menos_engines_mps_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(menos_engines_mps_tests PRIVATE ${MENOS_WARNING_FLAGS})

include(GoogleTest)
gtest_discover_tests(menos_engines_mps_tests)
