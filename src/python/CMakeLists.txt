# Only configured when MENOS_BUILD_PYTHON_BINDINGS=ON (root CMakeLists.txt).
# The dev/release presets used by scripts/verify.sh's C++-only stages leave
# this option OFF, so pure C++ development never needs Python or nanobind
# installed.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(menos_core_ext bindings.cpp)
target_link_libraries(menos_core_ext PRIVATE menos_api menos_qasm3)
set_target_properties(menos_core_ext PROPERTIES OUTPUT_NAME "_core")

install(TARGETS menos_core_ext LIBRARY DESTINATION menos)

# Windows: Python searches an extension module's own directory for the DLLs
# it imports, so OpenBLAS ships beside _core.pyd, with its licence
# (docs/adr/0022 section 5), and in a CUDA build so does the CUDA runtime, with
# NVIDIA's (docs/adr/0030 section 4). The GPU driver's nvcuda.dll is loaded by
# the runtime only when a device is asked for, so the module still imports on
# a machine with no NVIDIA driver.
#
# That is right for an editable install and a plain `uv build`. The published
# wheels turn MENOS_INSTALL_RUNTIME_DLLS off and let delvewheel vendor both
# DLLs under hashed names instead: Windows binds an import to any already
# loaded DLL of the same file name, so a plainly named libopenblas.dll loses
# to another package's (qiskit-aer ships one) when that package is imported
# first (docs/adr/0030 sections 5 and 10.4). The licences are always installed.
option(MENOS_INSTALL_RUNTIME_DLLS "Install the OpenBLAS and CUDA runtime DLLs beside _core.pyd" ON)
if(WIN32)
    if(MENOS_INSTALL_RUNTIME_DLLS)
        install(FILES "${MENOS_OPENBLAS_DLL}" DESTINATION menos)
    endif()
    install(FILES "${MENOS_OPENBLAS_ROOT}/LICENSE" DESTINATION menos RENAME OPENBLAS_LICENSE)
    if(MENOS_ENABLE_CUDA)
        if(MENOS_INSTALL_RUNTIME_DLLS)
            install(FILES "${MENOS_CUDART_DLL}" DESTINATION menos)
        endif()
        install(FILES "${CUDAToolkit_BIN_DIR}/../LICENSE" DESTINATION menos RENAME CUDA_LICENSE)
    endif()
endif()
