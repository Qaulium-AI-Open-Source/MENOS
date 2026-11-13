# The dense-statevector primitive API (docs/adr/0014). One target per backend;
# the shared contract lives in dense_kernels.h, which both implement.
add_library(menos_kernels_cpu STATIC
    cpu/dense_kernels.cpp
    cpu/thread_pool.cpp
)
target_include_directories(menos_kernels_cpu PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(menos_kernels_cpu PUBLIC menos_common Threads::Threads)
target_compile_features(menos_kernels_cpu PUBLIC cxx_std_20)
# The CPU twin of the CUDA kernels' -fmad=false (docs/adr/0014 section 10):
# no a*b+c is ever fused, so the specialised kernels stay bitwise identical
# to the generic one and to CUDA even if an FMA-capable -march is added
# (docs/adr/0019 section 4). A no-op on baseline x86-64, which has no FMA.
# MSVC's /fp:precise does not contract either; /fp:contract and /fp:fast must
# never be used (docs/adr/0022 section 3).
if(MSVC)
    set(menos_no_fp_contract /fp:precise)
else()
    set(menos_no_fp_contract -ffp-contract=off)
endif()
target_compile_options(menos_kernels_cpu PRIVATE ${MENOS_WARNING_FLAGS} ${menos_no_fp_contract})

# The AVX2 table (Phase 24 item 8, docs/adr/0028 section 19): one translation
# unit compiled for AVX2 and chosen at run time by cpuid, so the library still
# runs on any x86-64. -mavx2 does not enable FMA (that is -mfma, never passed);
# MSVC's /arch:AVX2 allows FMA instructions but /fp:precise above forbids
# contracting a*b+c into one, and the kernels write every multiply and add as
# its own intrinsic. tests/engines/dense/simd_kernels_test.cpp pins both tables
# to the generic kernel bitwise.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
    target_sources(menos_kernels_cpu PRIVATE cpu/dense_kernels_avx2.cpp)
    target_compile_definitions(menos_kernels_cpu PRIVATE MENOS_HAVE_AVX2_KERNELS=1)
    if(MSVC)
        set_source_files_properties(cpu/dense_kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(cpu/dense_kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
    endif()
endif()

if(MENOS_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
