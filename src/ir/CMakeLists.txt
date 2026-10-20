add_library(menos_ir STATIC
    wire_cut_basis.cpp
    circuit.cpp
    depth.cpp
    gate_table.cpp
    validate.cpp
    final_writes.cpp
    noise.cpp
    fuse_diagonal.cpp
)

target_include_directories(menos_ir PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(menos_ir PUBLIC menos_common)
target_compile_features(menos_ir PUBLIC cxx_std_20)
target_compile_options(menos_ir PRIVATE ${MENOS_WARNING_FLAGS})
