add_library(menos_analysis STATIC
    interaction_graph.cpp
    light_cone.cpp
    schmidt_rank_bound.cpp
    cut.cpp
    treewidth.cpp
)

target_include_directories(menos_analysis PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(menos_analysis PUBLIC menos_ir menos_common)
target_compile_features(menos_analysis PUBLIC cxx_std_20)
target_compile_options(menos_analysis PRIVATE ${MENOS_WARNING_FLAGS})
if(MENOS_PLANNER_COUNTERS)
    target_compile_definitions(menos_analysis PUBLIC MENOS_PLANNER_COUNTERS)
endif()
