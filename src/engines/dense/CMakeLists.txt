add_library(menos_engines_dense STATIC
    state_vector.cpp
    apply_matrix.cpp
    sample.cpp
    cumulative_distribution.cpp
    dense_engine.cpp
    fused_block.cpp
)

target_include_directories(menos_engines_dense PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# menos_ir: DenseEngine::execute() dispatches GateOp opcodes to their
# matrices via ir::generate_matrix (the gate table's single source of
# truth), the same call runtime.cpp used to make directly before Phase 4.
target_link_libraries(menos_engines_dense PUBLIC menos_common menos_ir menos_kernels_cpu)
target_compile_features(menos_engines_dense PUBLIC cxx_std_20)
target_compile_options(menos_engines_dense PRIVATE ${MENOS_WARNING_FLAGS})
