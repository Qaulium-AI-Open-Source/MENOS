# The CUDA dense engine. Links menos_engines_dense because it deliberately
# REUSES that engine's host-side sampler (sample/CumulativeDistribution)
# rather than reimplementing it: that reuse is what makes counts
# identical between backends for a given seed (docs/adr/0014), so it is shared
# code, not a layering shortcut.
add_library(menos_engines_dense_cuda STATIC
    cuda_dense_engine.cpp
)

target_include_directories(menos_engines_dense_cuda PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(menos_engines_dense_cuda PUBLIC
    menos_common menos_ir menos_kernels_cuda menos_engines_dense)
target_compile_features(menos_engines_dense_cuda PUBLIC cxx_std_20)
target_compile_options(menos_engines_dense_cuda PRIVATE ${MENOS_WARNING_FLAGS})
