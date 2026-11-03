add_library(menos_plan STATIC
    hardware_profile.cpp
    planner.cpp
    plan_cache.cpp
    region_builder.cpp
    available_engines.cpp
    execution_engines.cpp
    plan_impl.cpp
)

target_include_directories(menos_plan PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
# menos_engines_dense/menos_engines_stabilizer/menos_engines_pauli/
# menos_engines_mps/menos_engines_tensor_network: isolated to
# available_engines.cpp, the one file CLAUDE.md's architecture rule
# permits to name every concrete engine (docs/BUILD_PLAN.md Phase 4,
# extended Phase 5, 6 and 8).
target_link_libraries(menos_plan PUBLIC menos_analysis menos_cost menos_ir menos_common
                                        menos_engines_dense menos_engines_stabilizer
                                        menos_engines_pauli menos_engines_mps
                                        menos_engines_tensor_network
                                        menos_engines_density_matrix
                                        menos_engines_trajectory)
if(MENOS_ENABLE_CUDA)
    # execution_engines.cpp is the one file that names the CUDA engine, the
    # same containment available_engines.cpp gives the CPU engines.
    target_sources(menos_plan PRIVATE cuda_calibration.cpp)
    target_link_libraries(menos_plan PUBLIC menos_engines_dense_cuda)
    target_compile_definitions(menos_plan PUBLIC MENOS_ENABLE_CUDA)
endif()

target_compile_features(menos_plan PUBLIC cxx_std_20)
target_compile_options(menos_plan PRIVATE ${MENOS_WARNING_FLAGS})
