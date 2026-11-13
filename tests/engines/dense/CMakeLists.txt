add_executable(menos_engines_dense_tests
    state_vector_test.cpp
    apply_matrix_test.cpp
    specialised_kernels_test.cpp
    simd_kernels_test.cpp
    threaded_kernels_test.cpp
    thread_pool_test.cpp
    sample_test.cpp
    chunked_kernels_test.cpp
    dense_engine_test.cpp
    fused_block_test.cpp
)

target_link_libraries(menos_engines_dense_tests PRIVATE menos_engines_dense menos_ir GTest::gtest_main)
target_compile_options(menos_engines_dense_tests PRIVATE ${MENOS_WARNING_FLAGS})

if(MENOS_ENABLE_CUDA)
    # The CUDA specialised kernels against the generic CUDA kernel, bitwise.
    # Skipped at runtime (loudly) if no device is present.
    target_sources(menos_engines_dense_tests PRIVATE cuda_specialised_kernels_test.cpp
                                                     cuda_device_pool_test.cpp)
    target_link_libraries(menos_engines_dense_tests PRIVATE menos_kernels_cuda)
endif()

include(GoogleTest)
gtest_discover_tests(menos_engines_dense_tests)
