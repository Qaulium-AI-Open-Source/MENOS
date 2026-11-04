# The frozen pre-Phase-21 planner, for the differential test (owner decision,
# 2026-11-03, docs/adr/0025 section 9 Q2): a test-only copy of the planner at
# d89dde3, in its own namespace, never part of the shipped library.
#
# The copy is not checked in. It is extracted at configure time from this
# repository's own history with `git archive`, so it is exactly the d89dde3
# tree and cannot drift. It holds every library plan_job reaches (common, ir,
# analysis, cost, the CPU kernels, the engines, plan) and the public headers,
# without CUDA (the frozen side always plans for a machine with no GPU).
#
# Every translation unit is compiled with `menos` defined as
# `menos_frozen_d89dde3`. The preprocessor renames the namespace everywhere,
# so the frozen symbols cannot collide with the live ones in one test binary.
# String literals and header names are untouched, so explain() text is the
# d89dde3 text.
#
# It is extracted under _deps/ so that .clang-tidy's ExcludeHeaderFilterRegex
# leaves the frozen headers alone, as it does every fetched dependency.

set(menos_frozen_commit d89dde3c7e66f84ae1c9ff99d92550ad439928e8)
set(menos_frozen_root "${CMAKE_BINARY_DIR}/_deps/menos_frozen_d89dde3")
set(menos_frozen_marker "${menos_frozen_root}/.extracted-${menos_frozen_commit}")

if(NOT EXISTS "${menos_frozen_marker}")
    find_package(Git REQUIRED)
    file(REMOVE_RECURSE "${menos_frozen_root}")
    file(MAKE_DIRECTORY "${menos_frozen_root}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_SOURCE_DIR}" archive --format=tar
                "--output=${menos_frozen_root}/frozen.tar" ${menos_frozen_commit} --
                include src/common src/ir src/analysis src/cost src/kernels src/engines src/plan
        RESULT_VARIABLE menos_frozen_rc
        ERROR_VARIABLE menos_frozen_err)
    if(NOT menos_frozen_rc EQUAL 0)
        message(FATAL_ERROR "menos: MENOS_BUILD_FROZEN_PLANNER needs commit "
                            "${menos_frozen_commit} in this repository's history "
                            "(git archive failed: ${menos_frozen_err}); fetch it, or "
                            "configure with -DMENOS_BUILD_FROZEN_PLANNER=OFF")
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xf frozen.tar
                    WORKING_DIRECTORY "${menos_frozen_root}"
                    RESULT_VARIABLE menos_frozen_rc)
    if(NOT menos_frozen_rc EQUAL 0)
        message(FATAL_ERROR "menos: extracting the frozen d89dde3 planner failed")
    endif()
    file(REMOVE "${menos_frozen_root}/frozen.tar")
    file(TOUCH "${menos_frozen_marker}")
endif()

file(GLOB_RECURSE menos_frozen_sources CONFIGURE_DEPENDS "${menos_frozen_root}/src/*.cpp")
list(FILTER menos_frozen_sources EXCLUDE
     REGEX "/src/(kernels/cuda|engines/dense_cuda)/|/src/plan/cuda_calibration\\.cpp$")

add_library(menos_frozen_d89dde3 STATIC
    ${menos_frozen_sources}
    frozen_planner.cpp
)
target_include_directories(menos_frozen_d89dde3 PRIVATE
    "${menos_frozen_root}/include"
    "${menos_frozen_root}/src"
    ${LAPACKE_INCLUDE_DIR})
target_compile_definitions(menos_frozen_d89dde3 PRIVATE menos=menos_frozen_d89dde3)
target_compile_features(menos_frozen_d89dde3 PRIVATE cxx_std_20)
target_compile_options(menos_frozen_d89dde3 PRIVATE ${MENOS_WARNING_FLAGS})
# The LAPACK libraries menos_common links (src/common/CMakeLists.txt), without
# menos_common itself: linking it would put the live headers on the frozen
# include path. LAPACK::LAPACK is an imported target local to the directory
# that found it, so it is found again here (on Windows menos_openblas is
# already GLOBAL).
if(NOT WIN32)
    find_package(LAPACK REQUIRED)
endif()
get_target_property(menos_frozen_lapack menos_common INTERFACE_LINK_LIBRARIES)
target_link_libraries(menos_frozen_d89dde3 PRIVATE ${menos_frozen_lapack} Threads::Threads)
