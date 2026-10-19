add_executable(menos_common_tests
    error_test.cpp
    checked_math_test.cpp
    bitstring_test.cpp
    splitmix64_test.cpp
    rng_test.cpp
    pauli_row_test.cpp
    svd_test.cpp
    svd_lapacke_reference_test.cpp
)

target_link_libraries(menos_common_tests PRIVATE menos_common GTest::gtest_main)
target_compile_options(menos_common_tests PRIVATE ${MENOS_WARNING_FLAGS})
# svd_lapacke_reference_test.cpp calls LAPACKE_zgesdd as the reference that
# common::svd's direct zgesdd call must match bit for bit (docs/adr/0025).
target_include_directories(menos_common_tests PRIVATE ${LAPACKE_INCLUDE_DIR})

include(GoogleTest)
gtest_discover_tests(menos_common_tests)
