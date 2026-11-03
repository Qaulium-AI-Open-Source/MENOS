add_executable(menos_ir_tests
    wire_cut_basis_test.cpp
    append_batch_test.cpp
    circuit_test.cpp
    gate_table_test.cpp
    validate_test.cpp
    channel_validate_test.cpp
    noise_test.cpp
    fuse_diagonal_test.cpp
)

target_link_libraries(menos_ir_tests PRIVATE menos_ir GTest::gtest_main)
target_compile_options(menos_ir_tests PRIVATE ${MENOS_WARNING_FLAGS})

include(GoogleTest)
gtest_discover_tests(menos_ir_tests)

# Test-support only (never linked into a menos_* production target): loads
# tools/gen_fixtures.py's JSON output into a Circuit + expected statevector.
# Lives here because it constructs Circuit (an ir concept); consumed by
# tests/api's end-to-end fixture-differential test, which needs the full
# run() pipeline.
add_library(menos_test_fixture_loader STATIC
    fixture_loader.cpp
)
target_include_directories(menos_test_fixture_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(menos_test_fixture_loader PUBLIC menos_ir PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(menos_test_fixture_loader PRIVATE ${MENOS_WARNING_FLAGS})
