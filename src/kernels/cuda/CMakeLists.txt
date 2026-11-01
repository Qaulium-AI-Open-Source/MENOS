# The CUDA half of the dense primitive API (docs/adr/0014).
#
# device_buffer.cpp is ordinary C++ compiled by the host compiler (cuda_runtime.h
# is host-compilable), so it keeps the full MENOS warning set. Only the actual
# device kernels go through nvcc.
add_library(menos_kernels_cuda STATIC
    device_buffer.cpp
    device_info.cpp
    device_pool.cpp
    device_benchmarks.cpp
    dense_kernels.cu
)

target_include_directories(menos_kernels_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# CUDA headers are included with -isystem so their own diagnostics (they use
# C-style casts throughout) are not held to the MENOS warning bar, exactly as
# FetchContent'd third-party code is not.
target_include_directories(menos_kernels_cuda SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(menos_kernels_cuda PUBLIC menos_common menos_ir CUDA::cudart)
target_compile_features(menos_kernels_cuda PUBLIC cxx_std_20)
target_compile_definitions(menos_kernels_cuda PUBLIC MENOS_ENABLE_CUDA)

set_target_properties(menos_kernels_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
)

# Full warning set on the C++ TUs.
target_compile_options(menos_kernels_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${MENOS_WARNING_FLAGS}>)

# On .cu TUs the same flags are forwarded to the host compiler via -Xcompiler,
# minus two that cannot apply. Both exclusions were established by compiling
# with one flag at a time, and neither is caused by MENOS code:
#   -Wpedantic       fires on nvcc's own generated intermediate (its "# line"
#                    directives are a GCC extension), not on this source.
#   -Wold-style-cast fires inside CUDA's own channel_descriptor.h.
# -Wconversion is deliberately KEPT: it fires on real index arithmetic in our
# kernels, which is exactly where it earns its place.
target_compile_options(menos_kernels_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wconversion,-Wsign-conversion,-Wshadow,-Wnon-virtual-dtor,-Woverloaded-virtual,-Werror>)

# -fmad=false disables nvcc's default fusing of a*b+c into FMA. This is a
# correctness requirement here, not a style preference: the device
# apply_matrix sums its 2^k products in exactly the same order as the CPU
# primitive, so with contraction off the two produce BITWISE IDENTICAL
# amplitudes rather than merely close ones. That matters because counts are
# sampled from the returned statevector, so bitwise identity is what makes
# CPU and CUDA counts identical by construction for a given seed, rather than
# identical-in-practice-and-hopefully-never-not. Measured cost on the
# crossover benchmark is small next to the margin (docs/adr/0014).
#
# This is NOT -ffast-math, which CLAUDE.md forbids; it is the opposite
# direction, removing a fast-math-style transform nvcc applies by default.
target_compile_options(menos_kernels_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
