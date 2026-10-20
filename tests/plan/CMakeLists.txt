add_executable(menos_plan_tests
    planner_test.cpp
    hardware_profile_test.cpp
    region_builder_test.cpp
    engine_noise_refusal_test.cpp
    plan_cache_test.cpp
)

if(MENOS_ENABLE_CUDA)
    target_sources(menos_plan_tests PRIVATE cuda_backend_selection_test.cpp
                                            cuda_engine_noise_refusal_test.cpp)
    target_link_libraries(menos_plan_tests PRIVATE menos_engines_dense_cuda)
endif()

target_link_libraries(menos_plan_tests PRIVATE menos_plan GTest::gtest_main)
target_compile_options(menos_plan_tests PRIVATE ${MENOS_WARNING_FLAGS})

include(GoogleTest)
gtest_discover_tests(menos_plan_tests)
