# Phase 21 memory accounting (docs/adr/0025 section 3.1).
#
# Two executables share the fixture set. menos_memory_tests replaces the global
# operator new with the heap tracker (heap_tracker.cpp), so it must stay the
# only test binary that does; menos_memory_probe runs one fixture per process
# for the peak-RSS figure and is built with the tracker disabled, so it measures
# the real allocator.

set(MENOS_MEMORY_FIXTURE_LIBS
    menos_api menos_engines_dense menos_engines_density_matrix menos_engines_trajectory
    menos_engines_stabilizer menos_engines_pauli menos_engines_mps
    menos_engines_tensor_network nlohmann_json::nlohmann_json)
if(MENOS_ENABLE_CUDA)
    list(APPEND MENOS_MEMORY_FIXTURE_LIBS menos_engines_dense_cuda)
endif()

add_executable(menos_memory_probe
    memory_probe.cpp
    memory_fixtures.cpp
    heap_tracker.cpp
)
target_link_libraries(menos_memory_probe PRIVATE ${MENOS_MEMORY_FIXTURE_LIBS})
target_compile_definitions(menos_memory_probe PRIVATE MENOS_HEAP_TRACKER_DISABLED)
if(WIN32)
    target_link_libraries(menos_memory_probe PRIVATE psapi)
endif()
target_compile_options(menos_memory_probe PRIVATE ${MENOS_WARNING_FLAGS})

add_executable(menos_memory_tests
    memory_accounting_test.cpp
    memory_fixtures.cpp
    heap_tracker.cpp
)
target_link_libraries(menos_memory_tests PRIVATE ${MENOS_MEMORY_FIXTURE_LIBS} GTest::gtest_main)
target_compile_options(menos_memory_tests PRIVATE ${MENOS_WARNING_FLAGS})
target_compile_definitions(menos_memory_tests PRIVATE
    MENOS_MEMORY_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    MENOS_MEMORY_PROBE="$<TARGET_FILE:menos_memory_probe>")
add_dependencies(menos_memory_tests menos_memory_probe)

if(MENOS_ENABLE_CUDA)
    target_compile_definitions(menos_memory_probe PRIVATE MENOS_MEMORY_WITH_CUDA)
    target_compile_definitions(menos_memory_tests PRIVATE MENOS_MEMORY_WITH_CUDA)
endif()

include(GoogleTest)
gtest_discover_tests(menos_memory_tests DISCOVERY_TIMEOUT 60)
