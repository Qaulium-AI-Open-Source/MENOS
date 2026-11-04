add_executable(menos_engines_dense_tiered_tests
    pass_plan_test.cpp
    file_backing_store_test.cpp
)

target_link_libraries(menos_engines_dense_tiered_tests PRIVATE menos_engines_dense_tiered menos_ir GTest::gtest_main)
# benchmarks/crossover_circuit.h: the crossover family whose pass counts step 1
# measured (docs/adr/0027 section 10.1) are pinned here as goldens.
target_include_directories(menos_engines_dense_tiered_tests PRIVATE ${CMAKE_SOURCE_DIR}/benchmarks)
target_compile_options(menos_engines_dense_tiered_tests PRIVATE ${MENOS_WARNING_FLAGS})

include(GoogleTest)
gtest_discover_tests(menos_engines_dense_tiered_tests)
