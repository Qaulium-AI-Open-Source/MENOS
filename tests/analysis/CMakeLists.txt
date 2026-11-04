add_executable(menos_analysis_tests
    interaction_graph_test.cpp
    light_cone_test.cpp
    schmidt_rank_bound_test.cpp
    treewidth_test.cpp
    cut_test.cpp
    bit_adjacency_test.cpp
    min_fill_queue_test.cpp
    schmidt_rank_cache_test.cpp
)

target_link_libraries(menos_analysis_tests PRIVATE menos_analysis GTest::gtest_main)
target_compile_options(menos_analysis_tests PRIVATE ${MENOS_WARNING_FLAGS})

include(GoogleTest)
gtest_discover_tests(menos_analysis_tests)
