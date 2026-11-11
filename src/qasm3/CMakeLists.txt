add_library(menos_qasm3 STATIC
    lexer.cpp
    modifiers.cpp
    emitter.cpp
    parser.cpp
    stdgates.cpp
)

target_include_directories(menos_qasm3 PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(menos_qasm3 PUBLIC menos_ir menos_common)
target_compile_features(menos_qasm3 PUBLIC cxx_std_20)
target_compile_options(menos_qasm3 PRIVATE ${MENOS_WARNING_FLAGS})
