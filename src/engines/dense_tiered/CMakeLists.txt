add_library(menos_engines_dense_tiered STATIC
    pass_plan.cpp
    file_backing_store.cpp
)

target_include_directories(menos_engines_dense_tiered PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# menos_engines_dense: the chunked executor applies gates through the dense
# engine's apply_matrix forwarders and samples through its sort-then-walk
# sampler, so counts and expectations are bitwise the in-memory engine's
# (Phase 23, docs/adr/0027 section 3).
target_link_libraries(menos_engines_dense_tiered PUBLIC menos_engines_dense menos_kernels_cpu menos_common menos_ir)
target_compile_features(menos_engines_dense_tiered PUBLIC cxx_std_20)
target_compile_options(menos_engines_dense_tiered PRIVATE ${MENOS_WARNING_FLAGS})
