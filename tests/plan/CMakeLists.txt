add_executable(menos_plan_tests
    planner_test.cpp
    hardware_profile_test.cpp
    region_builder_test.cpp
    engine_noise_refusal_test.cpp
    plan_cache_test.cpp
    planner_golden_test.cpp
    execution_engines_test.cpp
    disk_tier_selection_test.cpp
)

if(MENOS_ENABLE_CUDA)
    target_sources(menos_plan_tests PRIVATE cuda_backend_selection_test.cpp
                                            cuda_engine_noise_refusal_test.cpp)
    target_link_libraries(menos_plan_tests PRIVATE menos_engines_dense_cuda)
endif()

target_link_libraries(menos_plan_tests PRIVATE menos_plan GTest::gtest_main)
# Golden plans for the Phase 24 planner corpus (docs/adr/0028 section 6).
target_compile_definitions(menos_plan_tests PRIVATE
    MENOS_PLANNER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden/planner_corpus")
target_compile_options(menos_plan_tests PRIVATE ${MENOS_WARNING_FLAGS})

include(GoogleTest)
gtest_discover_tests(menos_plan_tests)

