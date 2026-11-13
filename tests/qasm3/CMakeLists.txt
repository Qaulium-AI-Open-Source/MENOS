add_executable(menos_qasm3_tests
    emitter_test.cpp
    equivalence_test.cpp
    modifiers_test.cpp
    parser_test.cpp
    refusal_test.cpp
)

target_link_libraries(menos_qasm3_tests PRIVATE menos_qasm3 menos_qasm2 menos_api GTest::gtest_main)
target_compile_options(menos_qasm3_tests PRIVATE ${MENOS_WARNING_FLAGS})

include(GoogleTest)
gtest_discover_tests(menos_qasm3_tests)
