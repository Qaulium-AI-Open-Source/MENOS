# Phase 6: svd.cpp wraps LAPACKE_zgesdd. CMake's own FindLAPACK module
# does not know about LAPACKE (checked the installed
# /usr/share/cmake-*/Modules/FindLAPACK.cmake -- no mention of it), so
# the header and library are located manually, matching the working
# approach verified in a scratch probe before this file was written.
#
# On Windows, LAPACKE comes from the official OpenBLAS x64 (LP64) binary
# release, which scripts/fetch-openblas-windows.ps1 downloads and checks
# (docs/adr/0022 section 4). Its DLL exports LAPACKE itself, so one imported
# library provides both.
if(WIN32)
    # Defaults to where the fetch script puts it, or $ENV{MENOS_OPENBLAS_ROOT},
    # so that a pip/uv build needs no extra CMake arguments.
    if(DEFINED ENV{MENOS_OPENBLAS_ROOT})
        set(menos_openblas_default "$ENV{MENOS_OPENBLAS_ROOT}")
    else()
        set(menos_openblas_default "${CMAKE_SOURCE_DIR}/build/_deps/openblas-0.3.34")
    endif()
    set(MENOS_OPENBLAS_ROOT "${menos_openblas_default}" CACHE PATH
        "OpenBLAS x64 release directory (scripts/fetch-openblas-windows.ps1)")
    find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h HINTS "${MENOS_OPENBLAS_ROOT}/include" REQUIRED)
    find_library(MENOS_OPENBLAS_IMPLIB NAMES libopenblas openblas
                 HINTS "${MENOS_OPENBLAS_ROOT}/lib" REQUIRED)
    find_file(MENOS_OPENBLAS_DLL NAMES libopenblas.dll HINTS "${MENOS_OPENBLAS_ROOT}/bin" REQUIRED)
    # svd.cpp casts sizes to a 32-bit lapack_int; an ILP64 build would
    # misread every integer argument.
    file(STRINGS "${LAPACKE_INCLUDE_DIR}/openblas_config.h" menos_openblas_ilp64
         REGEX "^#define[ \t]+OPENBLAS_USE64BITINT")
    if(menos_openblas_ilp64)
        message(FATAL_ERROR "menos: ${LAPACKE_INCLUDE_DIR} is an ILP64 OpenBLAS; "
                            "use the LP64 (OpenBLAS-<version>-x64.zip) release")
    endif()
    add_library(menos_openblas SHARED IMPORTED GLOBAL)
    set_target_properties(menos_openblas PROPERTIES
        IMPORTED_IMPLIB "${MENOS_OPENBLAS_IMPLIB}"
        IMPORTED_LOCATION "${MENOS_OPENBLAS_DLL}")
    set(menos_lapack_libraries menos_openblas)
else()
    find_package(LAPACK REQUIRED)

    find_path(
        LAPACKE_INCLUDE_DIR
        NAMES lapacke.h
        PATHS
            /usr/local/include
            /usr/include
            /usr/include/lapacke
        REQUIRED
    )
    find_library(LAPACKE_LIBRARY NAMES lapacke REQUIRED)
    set(menos_lapack_libraries LAPACK::LAPACK ${LAPACKE_LIBRARY})
endif()

add_library(menos_common STATIC
    bitstring.cpp
    error.cpp
    rng.cpp
    pauli_row.cpp
    observable_restrict.cpp
    svd.cpp
    memory_ledger.cpp
)

target_include_directories(menos_common PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
                                         PRIVATE ${LAPACKE_INCLUDE_DIR})
target_compile_features(menos_common PUBLIC cxx_std_20)
target_compile_options(menos_common PRIVATE ${MENOS_WARNING_FLAGS})
target_link_libraries(menos_common PUBLIC ${menos_lapack_libraries})
