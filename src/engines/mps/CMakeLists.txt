add_library(menos_engines_mps STATIC
    mps_state.cpp
    apply_gate.cpp
    sampling.cpp
    expectation.cpp
    memory_model.cpp
    mps_engine.cpp
)

target_include_directories(menos_engines_mps PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# menos_common: common::svd, the LAPACK wrapper mps_state.cpp uses to
# move the orthogonality center. menos_ir: gate matrices via
# ir::generate_matrix, the same source every other engine uses. Since Phase 21
# the estimate replays bond dimensions (memory_model.cpp) instead of using
# menos_analysis's Schmidt-rank bound (docs/adr/0025 Q7).
target_link_libraries(menos_engines_mps PUBLIC menos_common menos_ir)
target_compile_features(menos_engines_mps PUBLIC cxx_std_20)
target_compile_options(menos_engines_mps PRIVATE ${MENOS_WARNING_FLAGS})
